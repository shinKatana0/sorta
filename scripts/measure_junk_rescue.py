"""Price the rescue gate of F140: what every threshold would select, before one is chosen.

The finding this exists for: a search by words (F134) put memes and screenshots at the top
of its results, and the table it searches is clean — all 19 753 rows carry the verdict
`photo`. So the junk stage is wrong about a few percent of what it calls a photograph, and
those frames are laid out by city like any other picture. They can be found with a
zero-shot query over the vectors F128 already stores:

    junk_score = max(similarity to screenshot/meme/text/receipt)
               - max(similarity to a photograph)

**The measurement costs no pass over any image.** The vectors are on disk; the prompts are
five short strings through the text tower. What it prints is the distribution of the score
and, per threshold, how many frames the gate would select and what the deep tier would then
cost — and it prints them BEFORE `features.junk_rescue_threshold` is chosen, which is the
whole reason the script exists (brief requirement 4).

Nothing is reimplemented here: `junk.junk_rescue_prompts`, `junk.junk_rescue_score`,
`junk.embedding_model` and `junk.read_clip_embeddings` are the pipeline's own, driven off
`junk.quality_settings`. A private copy of the arithmetic would price a gate the stage does
not have.

What the script does NOT do is tell anybody whether a frame is junk. The score is a
resemblance, right about 85% of the time near the useful threshold, and the decision it
feeds is "show this one to the model", not "delete it" — see the F130 measurement, where a
signal of that accuracy applied directly made a better baseline worse. The precision column
of this table is therefore a review by eye, in bands, and the script says so instead of
inventing one.

Privacy: nothing printed identifies a frame. No path, no basename, no file id — counts and
aggregates only (the rule of measure_ocr_gate.py / measure_frame_quality.py before it).

The database is opened `mode=ro`: a measurement writes nothing.

Usage (from the repo root, with the venv python):
    python scripts/measure_junk_rescue.py
    python scripts/measure_junk_rescue.py --thresholds 0,0.01,0.02,0.05
    python scripts/measure_junk_rescue.py --stored     # after a run, no model at all
    python scripts/measure_junk_rescue.py --write-bands bands.json --per-band 20
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The grid. The measured rows of the brief (0.00, +0.02, +0.05) sit inside it on purpose —
# a table you can find the reviewed bands in is what makes the other rows readable.
DEFAULT_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)

# Seconds per candidate frame in the deep tier, measured on the live run of 2026-07-28
# (95 minutes over 7 896 candidates at 1.38 frames/s in the pipelined path).
VLM_SECONDS_PER_FRAME = 0.78

# How the distribution is read: the score has no natural units, so what matters is where
# the mass sits relative to itself.
PERCENTILES = (5, 25, 50, 75, 90, 95, 99)

# The bands a review by eye is stratified over. Uniform in score and not in count: the
# population is enormously bottom-heavy (28.8% of it sits below zero), so a uniform sample
# would spend every look on frames no threshold will ever select.
SCORE_BANDS = (-1.0, -0.05, 0.0, 0.01, 0.02, 0.05, 1.0)


@dataclass(frozen=True)
class GateRow:
    """One row of the table: what a threshold selects, and what that costs."""
    threshold: float
    candidates: int
    total: int

    @property
    def share(self) -> float:
        return self.candidates / self.total if self.total else 0.0

    @property
    def minutes(self) -> float:
        return self.candidates * VLM_SECONDS_PER_FRAME / 60.0


def sweep(scores: list[float], thresholds: list[float]) -> list[GateRow]:
    """How many frames each threshold would send to the model.

    The rule replayed here is `_JunkRescuePass._score`'s own — a candidate is a frame whose
    score REACHES the threshold — so the table cannot disagree with what the stage does.
    """
    return [GateRow(t, sum(1 for s in scores if s >= t), len(scores))
            for t in thresholds]


def percentiles(values: list[float], points: tuple[int, ...] = PERCENTILES
                ) -> list[tuple[int, float]]:
    """(percentile, value) pairs by nearest rank — no interpolation.

    Nearest rank because the number is going to be copied into a config file as a
    threshold, and a value no frame actually has is a worse answer than one some frame does.
    """
    if not values:
        return []
    ordered = sorted(values)
    out: list[tuple[int, float]] = []
    for p in points:
        rank = max(0, min(len(ordered) - 1, round(p / 100.0 * len(ordered)) - 1))
        out.append((p, ordered[rank]))
    return out


def band_counts(scores: list[float]) -> list[tuple[float, float, int]]:
    """(low, high, frames) per score band — the shape the percentiles do not show."""
    return [(low, high, sum(1 for s in scores if low <= s < high))
            for low, high in zip(SCORE_BANDS, SCORE_BANDS[1:])]


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_distribution(scores: list[float], missing: int) -> str:
    """The distribution the threshold has to be read off. Aggregates only."""
    out = [
        "=" * 92,
        f"ОЦЕНКА «СКРИНШОТ, А НЕ ФОТОГРАФИЯ» ({len(scores)} фотографий с вектором)",
        "  " + "  ".join(f"p{p}={v:+.3f}" for p, v in percentiles(scores)),
        f"{'полоса':>16} {'кадров':>10} {'доля':>9}",
    ]
    for low, high, count in band_counts(scores):
        out.append(f"{low:>+8.2f}..{high:>+6.2f} {count:>10d} "
                   f"{_pct(count, len(scores)):>9}")
    if missing:
        out.append(f"  без вектора: {missing} — счёт NULL, это не «0»: у таких кадров "
                   f"фича просто не работает")
    out.append("=" * 92)
    out.append("Счёт — разность косинусных близостей, не вероятность: он упорядочивает "
               "кадры\nотносительно друг друга. Полоса, в которой стоит смотреть глазами, "
               "начинается около нуля.")
    return "\n".join(out)


def format_thresholds(rows: list[GateRow], current: float) -> str:
    """The table the threshold is chosen from: population and price, per row."""
    out = [
        "=" * 92,
        "КАНДИДАТЫ НА ПРОВЕРКУ VLM: порог -> популяция -> время глубокого яруса",
        f"{'порог':>7} {'кандидатов':>12} {'доля':>9} {'время яруса':>16}",
    ]
    for r in sorted(rows, key=lambda x: x.threshold):
        mark = "*" if abs(r.threshold - current) < 1e-9 else " "
        out.append(f"{r.threshold:>+6.2f}{mark}{r.candidates:>12d} "
                   f"{_pct(r.candidates, r.total):>9} {r.minutes:>12.1f} мин")
    out.append("=" * 92)
    out.append(f"* — порог из конфига (features.junk_rescue_threshold). Цена кадра — "
               f"{VLM_SECONDS_PER_FRAME} с, замер 2026-07-28.")
    out.append("ГЛАЗАМИ: откройте по десятку кадров из полос выше и ниже выбранного "
               "порога.\nЭта таблица показывает охват и цену, но не точность — её "
               "здесь никто не мерил.")
    return "\n".join(out)


def write_band_template(path: Path, scored: dict[int, float], per_band: int,
                        seed: int) -> int:
    """A worksheet to review by eye: `{file_id: null}`, stratified over the score bands.

    File ids and nothing else — the same privacy rule the tables follow. Whoever fills it
    in opens those frames in the web app and replaces each null with true (this really is a
    screenshot / a screen / a receipt) or false. Returns how many frames were written.
    """
    rng = random.Random(seed)
    picked: list[int] = []
    for low, high in zip(SCORE_BANDS, SCORE_BANDS[1:]):
        band = [fid for fid, score in scored.items() if low <= score < high]
        rng.shuffle(band)
        picked.extend(sorted(band[:per_band]))
    path.write_text(json.dumps({str(fid): None for fid in picked}, indent=1),
                    encoding="utf-8")
    return len(picked)


def open_ro(db_path: str) -> sqlite3.Connection:
    """The index, read-only — a measurement writes nothing."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def photo_ids(conn: sqlite3.Connection) -> list[int]:
    """The population: canonical photographs the stage calls `photo`.

    The same population `frame_quality` and `clip_embeddings` have (F120) — a frame already
    classified as a screenshot is not what this feature is looking for, it is what the
    feature is looking for frames LIKE.
    """
    return [int(r["id"]) for r in conn.execute(
        """SELECT f.id FROM files f JOIN media_class mc ON mc.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
             AND mc.verdict = ? ORDER BY f.id""", (junk.QUALITY_VERDICT,))]


def stored_scores(conn: sqlite3.Connection, ids: list[int]) -> dict[int, float]:
    """`frame_quality.junk_score` as a live run left it — `--stored`, no model needed."""
    keep = set(ids)
    return {int(r["file_id"]): float(r["junk_score"]) for r in conn.execute(
        "SELECT file_id, junk_score FROM frame_quality WHERE junk_score IS NOT NULL")
        if int(r["file_id"]) in keep}


def computed_scores(conn: sqlite3.Connection, model: str, ids: list[int],
                    features: np.ndarray) -> dict[int, float]:
    """The score of every frame that has a vector OF THIS MODEL, computed here.

    The model filter is not optional and is not applied here either: it lives in
    `junk.read_clip_embeddings`, the one function that reads that table, because a vector
    from another model is not comparable with these prompts and mixing the two would
    produce a plausible distribution that nothing in the output marks as wrong.
    """
    vectors = junk.read_clip_embeddings(conn, model, ids)
    scored: dict[int, float] = {}
    for file_id in ids:
        vec = vectors.get(file_id)
        score = None if vec is None else junk.junk_rescue_score(vec, features)
        if score is not None:
            scored[file_id] = score
    return scored


def text_features(cfg: Config) -> np.ndarray:  # pragma: no cover — ML
    """The prompts of the stage through the project's own text tower, as unit rows."""
    encoder = junk.clip_text_encoder(naming_settings(cfg))
    return junk.unit_rows(np.asarray(encoder(junk.junk_rescue_prompts()),
                                     dtype=np.float32))


def parse_thresholds(text: str) -> list[float]:
    """"0,0.02,0.05" -> [0.0, 0.02, 0.05]. Sorted, deduplicated, never empty."""
    values = sorted({float(part) for part in text.replace(",", " ").split()})
    if not values:
        raise SystemExit("--thresholds: пустая сетка")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--thresholds", default=",".join(f"{t:g}" for t in DEFAULT_GRID),
                    help=f"the score grid (default {DEFAULT_GRID[0]:g}..."
                         f"{DEFAULT_GRID[-1]:g})")
    ap.add_argument("--stored", action="store_true",
                    help="read frame_quality.junk_score written by a run instead of "
                         "computing it (needs no model at all)")
    ap.add_argument("--write-bands", help="write a stratified worksheet to review by eye "
                                          "(file ids only) and exit")
    ap.add_argument("--per-band", type=int, default=20,
                    help="frames per score band in the worksheet (default 20)")
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    cfg = load_config(args.config)
    q = junk.quality_settings(cfg)
    model = junk.embedding_model(naming_settings(cfg))

    conn = open_ro(str(cfg.database))
    try:
        ids = photo_ids(conn)
        if not ids:
            raise SystemExit("в индексе нет кадров с вердиктом photo — нечего мерить")
        if args.stored:
            scored = stored_scores(conn, ids)
            source = "frame_quality.junk_score"
        else:
            scored = computed_scores(conn, model, ids, text_features(cfg))
            source = f"clip_embeddings ({model})"
    finally:
        conn.close()

    if not scored:
        raise SystemExit(
            "нет сохранённых CLIP-векторов для этих кадров — сначала нужен прогон "
            "junk с features.store_embeddings (или --stored после прогона с "
            "features.junk_rescue)")

    if args.write_bands:
        written = write_band_template(
            Path(args.write_bands), scored, args.per_band, args.seed)
        print(f"смотреть глазами: {written} кадров записано в {args.write_bands} "
              f"(только file_id; замените null на true/false)")
        return 0

    scores = list(scored.values())
    print(f"коллекция: {len(ids)} фотографий, оценок посчитано {len(scores)}, "
          f"источник: {source}")
    print()
    print(format_distribution(scores, len(ids) - len(scores)))
    print(format_thresholds(sweep(scores, thresholds), q.junk_rescue_threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
