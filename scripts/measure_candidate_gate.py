"""Price the candidate gate of the deep junk tier: benefit against population, per threshold.

The gate (#14/V1) sends a frame to the VLM when its product-CLIP score reaches
`naming.product_candidate_min` (0.4). On the live run of 2026-07-28 that opened for
7 896 frames of 24 196 (32.6%), took 95 minutes and produced 2 592 changed verdicts.
Nobody has ever seen the shape of that distribution. If the benefit sits at the top of
the score, then half the population can be dropped for almost nothing — and half the
candidates is half the time, which is a bigger lever than anything else touched so far
(`use_fast` gave x1.15, the overlap x1.05, the resolution was rejected).

**The measurement costs no VLM call at all.** Everything needed is already on disk:

    verdict BEFORE the tier   a snapshot of media_class taken before the run (--before)
    verdict AFTER the tier    the current media_class (source='vlm' — what the model
                              answered)
    product-score             recomputed with the fast CLIP — minutes, not hours

A frame is useful if the model changed its verdict (`after != before`); the curve is
then arithmetic over per-frame aggregates.

Two things this script is careful about:

* The product score is NOT reimplemented here. `junk._product_score`, the prompt classes
  and the CLIP classifier of the pipeline are imported and called — a private copy would
  measure itself instead of the gate.
* The gate is not just the product score. A frame is a candidate ALSO when the fast tier
  already said `document`, or when its document-CLIP sits in the suspicious zone
  (`naming.text_rescue_docscore_min`). Those frames stay candidates at every threshold,
  so counting them as lost would invent a loss that raising the knob never causes. The
  condition of `junk.classify` is replayed here in full (see `Frame.forced`).

Privacy: nothing here identifies a frame. No path, no basename, no file id is printed or
stored — losses are reported as counts per label pair, which is all that is needed to go
and look, and never a list of where somebody's documents are (the rule of
measure_ocr_gate.py / measure_vlm_speed.py / measure_vlm_resolution.py before it).

The database is opened `mode=ro`: a measurement writes nothing.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_candidate_gate.py --before report_output/media_class_before_vlm.json
    python scripts/measure_candidate_gate.py --before before.json --thresholds 0.3,0.4,0.5
    python scripts/measure_candidate_gate.py --before before.json --sample 500   # debug
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.landmarks import batched, clip_classifier  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The grid. The current default (0.4) sits inside it on purpose — a row you can read the
# status quo off is what makes the other rows comparable. Below it the curve can only
# show how the population grows: no frame under today's gate was ever shown to the model,
# so there are no verdicts down there to count as benefit (the report says so).
DEFAULT_GRID = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)

# Seconds per candidate frame, measured on the live run of 2026-07-28 (95 minutes over
# 7 896 candidates at 1.38 frames/s in the pipelined path). The time column is this
# number times the population — the deep tier has no per-run fixed cost worth naming
# next to a 95-minute pass.
SEC_PER_FRAME = 0.78

# --- Pre-registered acceptance criteria (F106) -------------------------------
#
# Written down before the first run, so that the table cannot talk anybody into a number
# afterwards. The threshold is recommended for a change only if some value BOTH
#
# 1. keeps >= 95% of the changed verdicts, AND
# 2. cuts the population by >= 25%.
#
# If no value does, the outcome is C — the gate is already set well, the curve goes into
# the backlog and the subject is closed. That is a useful result: it means 0.4 was chosen
# correctly rather than by accident.
MIN_BENEFIT_KEPT = 0.95
MIN_POPULATION_CUT = 0.25

# The verdict whose loss is not "slightly worse filing" but somebody's papers in a city
# folder. Reported separately, with the threshold at which the first one goes.
DOCUMENT_VERDICT = "document"


@dataclass(frozen=True)
class Frame:
    """The per-frame aggregate the sweep needs — and nothing that identifies a frame.

    `product_score` is None for a frame the gate never looks at (it has faces, or the
    fast tier called it screenshot/meme): it counts in the collection, but it is a
    candidate at no threshold.
    `forced` — a candidate at EVERY threshold: the fast tier said `document`, or the
    document-CLIP is in the rescue zone. Raising `product_candidate_min` cannot drop it.
    `before` is the fast verdict from the snapshot, None — the frame is not in it (it was
    indexed after the snapshot was taken), so whether the tier changed anything for it is
    unknown and is counted apart instead of being guessed.
    `after` is the verdict now.
    """
    product_score: float | None
    forced: bool
    before: str | None
    after: str

    def is_candidate(self, threshold: float) -> bool:
        """The gate of `junk.classify`, with `product_candidate_min` set to `threshold`."""
        return self.forced or (self.product_score is not None
                               and self.product_score >= threshold)

    @property
    def changed(self) -> bool:
        """The tier was useful here: it moved the verdict off the fast one."""
        return self.before is not None and self.before != self.after


@dataclass(frozen=True)
class CurveRow:
    """One row of the curve: what a threshold gates, what it keeps and what it gives up."""
    threshold: float
    candidates: int
    kept: int                  # changed verdicts that survive this threshold
    lost: int                  # changed verdicts given up
    unknown: int               # candidates with no baseline — benefit not knowable
    lost_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def changed_total(self) -> int:
        return self.kept + self.lost

    @property
    def kept_frac(self) -> float:
        """Share of the changed verdicts that survive (1.0 — an empty benefit loses nothing)."""
        return self.kept / self.changed_total if self.changed_total else 1.0

    @property
    def seconds(self) -> float:
        return self.candidates * SEC_PER_FRAME

    @property
    def documents_lost(self) -> int:
        """Lost changes whose ANSWER was `document` — a document the tier found and that
        this threshold hands back to the city folders. The opposite direction
        (`document -> photo`) is a milder mistake: it leaves a document a document."""
        return sum(n for (_was, now), n in self.lost_pairs.items() if now == DOCUMENT_VERDICT)


def load_before(path: Path) -> dict[int, str]:
    """The pre-tier snapshot -> {file_id: verdict}.

    The snapshot is produced outside this repo (the orchestrator dumps media_class before
    a deep run), so the shape is accepted in the forms such a dump comes in: a list of
    rows (dicts with file_id/verdict, or [file_id, verdict] pairs), a mapping of id to
    verdict, or either of those under a "rows" key. Anything else is an error rather than
    a silently empty baseline — a curve with no `before` column would report the entire
    benefit as unknown and still print a table.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    out: dict[int, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            verdict = value.get("verdict") if isinstance(value, dict) else value
            out[int(key)] = str(verdict)
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                out[int(row["file_id"])] = str(row["verdict"])
            else:
                file_id, verdict = row[0], row[1]
                out[int(file_id)] = str(verdict)
    else:
        raise SystemExit(f"{path}: не похоже на снимок media_class "
                         f"(ожидается список строк или отображение id -> вердикт)")
    if not out:
        raise SystemExit(f"{path}: снимок пуст — сравнивать не с чем")
    return out


def sweep(frames: list[Frame], thresholds: list[float]) -> list[CurveRow]:
    """The curve itself: replay the gate at every threshold over the same frames.

    Only `product_candidate_min` moves. The verdicts, the face veto, the document branch
    and the rescue zone stay exactly as the run had them — the question is what the gate
    costs, not what a different classifier would decide.
    """
    rows: list[CurveRow] = []
    for threshold in thresholds:
        candidates = kept = lost = unknown = 0
        lost_pairs: Counter[tuple[str, str]] = Counter()
        for f in frames:
            gated = f.is_candidate(threshold)
            if gated:
                candidates += 1
                if f.before is None:
                    unknown += 1
            if not f.changed:
                continue
            if gated:
                kept += 1
            else:
                lost += 1
                lost_pairs[(str(f.before), f.after)] += 1
        rows.append(CurveRow(threshold, candidates, kept, lost, unknown, lost_pairs))
    return rows


def baseline_row(rows: list[CurveRow], current: float) -> CurveRow:
    """The row of the threshold in force today — everything is measured against it.

    The nearest row of the grid, so a grid that does not contain the configured value
    still produces a comparison instead of a crash.
    """
    return min(rows, key=lambda r: abs(r.threshold - current))


def population_cut(base: CurveRow, row: CurveRow) -> float:
    """Share of the candidates this threshold removes (0.0 — a base that gated nobody)."""
    if not base.candidates:
        return 0.0
    return (base.candidates - row.candidates) / base.candidates


def benefit_kept(base: CurveRow, row: CurveRow) -> float:
    """Share of the baseline's changed verdicts this threshold still gets."""
    return row.kept / base.changed_total if base.changed_total else 1.0


def first_document_loss(rows: list[CurveRow], base: CurveRow) -> CurveRow | None:
    """The lowest threshold above the baseline that gives up a found `document`.

    Printed on its own, whatever the outcome: an average over 2 592 changed verdicts
    hides the twenty that are personal papers, and those are the ones that end up laid
    out by city.
    """
    for row in sorted(rows, key=lambda r: r.threshold):
        if row.threshold > base.threshold and row.documents_lost:
            return row
    return None


def recommend(rows: list[CurveRow], current: float) -> tuple[str, str]:
    """(the outcome letter, one line saying why) — decided by the pre-registered criteria.

    A — a threshold meets both criteria: raise `naming.product_candidate_min` to it (in a
        separate commit, after a human has read the table).
    C — none does: the gate is already set well, the subject is closed with numbers.
    """
    base = baseline_row(rows, current)
    higher = [r for r in rows if r.threshold > base.threshold]
    if not higher:
        return "C", (f"выше текущего порога {base.threshold:.2f} в сетке ничего нет — "
                     f"сравнивать не с чем")
    good = [r for r in higher
            if benefit_kept(base, r) >= MIN_BENEFIT_KEPT
            and population_cut(base, r) >= MIN_POPULATION_CUT]
    if not good:
        best = max(higher, key=lambda r: (population_cut(base, r) >= MIN_POPULATION_CUT,
                                          benefit_kept(base, r)))
        return "C", (
            f"ни один порог не берёт оба критерия (лучший — {best.threshold:.2f}: "
            f"пользы сохраняется {benefit_kept(base, best):.1%} при "
            f"{MIN_BENEFIT_KEPT:.0%}, популяция сокращается "
            f"{population_cut(base, best):.1%} при {MIN_POPULATION_CUT:.0%}): "
            f"гейт настроен хорошо, порог {base.threshold:.2f} остаётся, тему закрываем")
    pick = max(good, key=lambda r: population_cut(base, r))
    saved_min = (base.seconds - pick.seconds) / 60.0
    return "A", (
        f"порог {pick.threshold:.2f} берёт оба критерия (пользы сохраняется "
        f"{benefit_kept(base, pick):.1%} при {MIN_BENEFIT_KEPT:.0%}, популяция "
        f"сокращается {population_cut(base, pick):.1%} при {MIN_POPULATION_CUT:.0%}, "
        f"экономия ~{saved_min:.0f} мин): рекомендуем naming.product_candidate_min "
        f"{base.threshold:.2f} -> {pick.threshold:.2f}")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_table(rows: list[CurveRow], total: int, current: float) -> str:
    """The curve. Aggregates only — no path, no file id, no basename."""
    base = baseline_row(rows, current)
    out = [
        "=" * 104,
        f"ГЕЙТ КАНДИДАТОВ VLM: порог -> популяция -> польза -> время "
        f"({total} кадров в коллекции)",
        f"{'порог':>6} {'кандидатов':>19} {'популяция':>10} {'польза':>17} "
        f"{'потеряно':>9} {'из них док':>11} {'время яруса':>14}",
    ]
    for r in sorted(rows, key=lambda x: x.threshold):
        mark = "*" if r is base else " "
        cut = population_cut(base, r)
        out.append(
            f"{r.threshold:>5.2f}{mark}"
            f"{r.candidates:>11d} ({_pct(r.candidates, total):>5}) "
            f"{cut:>+9.1%} "
            f"{r.kept:>9d} ({benefit_kept(base, r):>6.1%}) "
            f"{r.lost:>9d} {r.documents_lost:>11d} "
            f"{r.seconds / 60.0:>11.0f} мин"
        )
    out.append("=" * 104)
    out.append(f"* — порог из конфига (naming.product_candidate_min), {SEC_PER_FRAME:.2f} "
               f"с/кадр по замеру 2026-07-28")
    out.append("«популяция» и «польза» — против строки с *; ниже неё польза недосчитана: "
               "кадры под\nтекущим порогом модель никогда не видела, и менять их вердикт "
               "было некому.")
    unknown = max((r.unknown for r in rows), default=0)
    if unknown:
        out.append(f"нет базы для сравнения: {unknown} кандидат(ов) отсутствуют в снимке "
                   f"«до» — в пользу не считаются")
    return "\n".join(out)


def format_losses(rows: list[CurveRow], current: float) -> str:
    """What exactly is given up, per label pair.

    Losing 200 `photo -> product` finds and losing 200 `document -> photo` finds are
    different losses: the first puts goods back among the cities, the second leaves a
    document a document.
    """
    base = baseline_row(rows, current)
    lines = [f"ЧТО ТЕРЯЕТСЯ (изменения вердикта, которые порог отдаёт; база — "
             f"{base.threshold:.2f}, изменений {base.changed_total}):"]
    for r in sorted(rows, key=lambda x: x.threshold):
        if r.threshold <= base.threshold or not r.lost:
            continue
        lines.append(f"  {r.threshold:.2f}: потеряно {r.lost} из {base.changed_total}")
        for (was, now), count in sorted(r.lost_pairs.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"{'':>8}{was} -> {now}: {count}")
    if len(lines) == 1:
        lines.append("  (выше базового порога ничего не теряется)")
    hit = first_document_loss(rows, base)
    lines.append(
        f"первый потерянный документ: порог {hit.threshold:.2f} "
        f"({hit.documents_lost} шт.) — это личные бумаги в папке города, не «чуть хуже "
        f"разложено»" if hit else
        "первый потерянный документ: ни на одном пороге сетки документы не теряются")
    return "\n".join(lines)


def format_outcome(rows: list[CurveRow], current: float) -> str:
    letter, why = recommend(rows, current)
    return f"ИСХОД {letter}: {why}"


def sample_rows(db_path: str, n: int | None, seed: int) -> list[sqlite3.Row]:
    """The classified collection, read-only: id, path, faces and the verdict now.

    `mode=ro` is the contract of the brief — a measurement writes nothing. Files with no
    media_class row are outside the question: the tier never saw them.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT f.id, f.path, mc.verdict AS verdict, mc.source AS source,
                      EXISTS(SELECT 1 FROM faces fa
                             WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces
               FROM files f JOIN media_class mc ON mc.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()
    if n is None:
        return rows
    picked = list(rows)
    random.Random(seed).shuffle(picked)
    return picked[:n]


def measure(cfg: Config, rows: list[sqlite3.Row],
            before: dict[int, str]) -> list[Frame]:  # pragma: no cover — ML
    """CLIP over the collection -> the per-frame aggregates of the sweep.

    Both prompt groups of the gate are run, with the pipeline's own classifier and its
    own scoring functions: the product score decides the threshold, the document score
    decides whether the threshold matters for this frame at all (`forced`). Frames the
    gate never looks at — faces, screenshot, meme — skip CLIP entirely, which is what
    keeps this a matter of minutes.
    """
    s = naming_settings(cfg)
    g = junk.gate_settings(cfg)
    classifier = clip_classifier(s)
    doc_prompts = [prompt for _cls, prompt in junk._DOCUMENT_CLASSES]
    prod_prompts = [prompt for _cls, prompt in junk._PRODUCT_CLASSES]

    frames: list[Frame] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        looked_at = [i for i, r in enumerate(chunk) if gate_looks_at(r, before)]
        doc_score: dict[int, float] = {}
        prod_score: dict[int, float] = {}
        if looked_at:
            paths = [chunk[i]["path"] for i in looked_at]
            doc_probs = classifier(paths, doc_prompts)
            prod_probs = classifier(paths, prod_prompts)
            for k, i in enumerate(looked_at):
                doc_score[i] = junk._document_score(doc_probs[k])
                prod_score[i] = junk._product_score(prod_probs[k])
        for i, r in enumerate(chunk):
            fast = before.get(int(r["id"]))
            frames.append(Frame(
                product_score=prod_score.get(i),
                forced=(i in doc_score
                        and (fast == DOCUMENT_VERDICT
                             or doc_score[i] >= g.text_rescue_docscore_min)),
                before=fast,
                after=str(r["verdict"]),
            ))
        done += len(chunk)
        print(f"  CLIP {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return frames


def gate_looks_at(row: sqlite3.Row, before: dict[int, str]) -> bool:
    """Can this frame be a candidate at all? (junk.classify: no faces, not screenshot/meme)

    The verdict that decides is the FAST one — the snapshot. Without it the current
    verdict is the best available stand-in: for a frame the tier skipped they are the
    same, and a frame the tier answered was a candidate by definition.
    """
    if row["has_faces"]:
        return False
    verdict = before.get(int(row["id"]), str(row["verdict"]))
    return verdict not in ("screenshot", "meme")


def parse_thresholds(text: str) -> list[float]:
    """"0.3,0.4,0.5" -> [0.3, 0.4, 0.5]. Sorted, deduplicated, never empty."""
    values = sorted({float(part) for part in text.replace(",", " ").split()})
    if not values:
        raise SystemExit("--thresholds: пустая сетка")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--before", required=True,
                    help="JSON snapshot of media_class taken BEFORE the deep run")
    ap.add_argument("--thresholds", default=",".join(f"{t:g}" for t in DEFAULT_GRID),
                    help=f"the product_score grid (default {DEFAULT_GRID[0]:g}..."
                         f"{DEFAULT_GRID[-1]:g})")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="measure N random frames instead of the whole collection (debug)")
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    cfg = load_config(args.config)
    current = float(getattr(cfg.naming, "product_candidate_min",
                            junk._DEFAULT_PRODUCT_CANDIDATE_MIN))
    before = load_before(Path(args.before))
    rows = sample_rows(str(cfg.database), args.sample, args.seed)
    if not rows:
        raise SystemExit("в индексе нет классифицированных кадров — нечего мерить")
    answered = sum(1 for r in rows if r["source"] == "vlm")
    print(f"коллекция: {len(rows)} кадров, снимок «до»: {len(before)} строк, "
          f"ответов модели в БД (source='vlm'): {answered}")
    print(f"сетка: {', '.join(f'{t:g}' for t in thresholds)}, "
          f"текущий порог: {current:g}")

    frames = measure(cfg, rows, before)
    rows_out = sweep(frames, thresholds)
    print()
    print(format_table(rows_out, len(frames), current))
    print(format_losses(rows_out, current))
    print()
    print(format_outcome(rows_out, current))
    return 0


if __name__ == "__main__":
    sys.exit(main())
