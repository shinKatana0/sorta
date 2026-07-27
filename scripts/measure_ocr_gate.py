"""Price the OCR gate of the junk stage: coverage, benefit and time per threshold.

The gate (F38) sends a frame to OCR when its document-CLIP score reaches
`naming.text_rescue_docscore_min`. That number was set by eye and never measured; on
the validation collection it opens for 28% of the frames and changes 2% of the
verdicts — 14 OCR calls per verdict, and OCR is ~40% of the junk stage.

Raising it would be cheap and might be wrong: the gate exists to catch a real
document that CLIP scored low, and a document that lands in the city folders instead
of _Documents is the expensive error here. So this script decides nothing. It prints
the trade-off — for each threshold, how many frames pay for OCR, how many verdicts
that actually buys, and what it costs in wall-clock time — and the threshold in the
config is then a decision for a human looking at that table.

The verdict and the gate are NOT reimplemented here: `junk.clip_verdict`,
`junk.ocr_gate_open` and `junk.apply_text_frac` are the same functions the pipeline
runs, called with the same thresholds. A private copy of those branches would drift
and price a gate that no longer exists.

Privacy: nothing here identifies a frame. No path is printed, and the cache stores
file ids only — a table about documents must not become a list of where the documents
are (see the document-verdict rules).

Usage (from the repo root, with the venv python):
    python scripts/measure_ocr_gate.py                       # 2000 frames, grid 0.2..0.6
    python scripts/measure_ocr_gate.py --sample 4000 --cache ocr_gate.json
    python scripts/measure_ocr_gate.py --cache ocr_gate.json  # replay, no CLIP/OCR run
    python scripts/measure_ocr_gate.py --probe-below 200      # what the gate never sees

A full run costs a CLIP pass over the sample plus an OCR pass over everything the
LOWEST threshold of the grid gates — minutes, not seconds. `--cache` writes those
per-frame aggregates out, and a later run replays the whole sweep from them
instantly; that is also how a different grid is tried without paying for the models
again.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402
from sorta.landmarks import batched, clip_classifier  # noqa: E402
from sorta.naming import naming_settings  # noqa: E402

# The grid the brief asks for. 0.3 (the current default) sits inside it on purpose —
# a row you can read the status quo off is what makes the others comparable.
DEFAULT_GRID = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)

# Building 4 easyocr Readers + CLIP costs ~35 s per junk run (measured). It is 4% of a
# full run and 100% of an incremental catch-up on five new photos, so the time column
# carries it — but only for thresholds that gate at least one frame, because a gate
# that opens for nobody builds no Reader at all (the pool is lazy, F73).
DEFAULT_STARTUP_SEC = 35.0

CACHE_VERSION = 1


@dataclass(frozen=True)
class Frame:
    """The per-frame aggregate the sweep needs — and nothing that identifies a frame.

    `verdict` is the fast-tier verdict BEFORE OCR, `text_frac` the OCR signal (None —
    OCR was not run for this frame, so it carries no signal at any threshold).
    """
    file_id: int
    has_faces: bool
    verdict: str
    doc_score: float
    text_frac: float | None


@dataclass(frozen=True)
class GateRow:
    """One row of the table: what a threshold gates, buys and costs."""
    threshold: float
    gated: int
    rescued: int    # photo -> document (the FN catch the gate exists for)
    fp_fixed: int   # document -> photo (the FP gate; not threshold-dependent)
    no_signal: int  # gated, but OCR returned nothing — benefit unknown, cost paid
    seconds: float

    @property
    def changed(self) -> int:
        return self.rescued + self.fp_fixed


def sample_rows(db_path: str, n: int, seed: int) -> list[sqlite3.Row]:
    """`n` canonical photos that still exist on disk, deterministic for a given seed.

    The row shape is the one `junk.classify` selects, so the same helpers can be fed
    with it unchanged (`junk._is_real_photo` reads these columns).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT f.id, f.path, f.width, f.height, f.camera_make, f.camera_model,
                      f.gps_lat,
                      EXISTS(SELECT 1 FROM faces fa
                             WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces
               FROM files f
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()
    random.Random(seed).shuffle(rows)
    return [r for r in rows if Path(r["path"]).exists()][:n]


def sweep(frames: list[Frame], thresholds: list[float], g: junk.GateSettings,
          ocr_ms: float, startup_sec: float) -> list[GateRow]:
    """The table itself: replay the gate and the OCR signal at every threshold.

    Only `text_rescue_docscore_min` moves — the verdict, the face veto and the
    text_frac thresholds stay exactly as the config has them, because the question is
    what the gate costs, not what a different classifier would do.
    """
    rows: list[GateRow] = []
    for threshold in thresholds:
        gated = rescued = fp_fixed = no_signal = 0
        for f in frames:
            if not junk.ocr_gate_open(f.has_faces, f.verdict, f.doc_score, threshold):
                continue
            gated += 1
            if f.text_frac is None:
                no_signal += 1
                continue
            verdict, _score, source = junk.apply_text_frac(
                f.verdict, 0.0, f.text_frac, g)
            if source != "ocr":
                continue
            if verdict == "document":
                rescued += 1
            else:
                fp_fixed += 1
        seconds = gated * ocr_ms / 1000.0 + (startup_sec if gated else 0.0)
        rows.append(GateRow(threshold, gated, rescued, fp_fixed, no_signal, seconds))
    return rows


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "—"


def format_table(rows: list[GateRow], total: int, current: float | None = None) -> str:
    """The report block. Aggregates only — no path, no file id, no basename."""
    out = [
        "=" * 96,
        f"ГЕЙТ OCR: порог -> охват -> польза -> время ({total} кадров в выборке)",
        f"{'порог':>6} {'кадров в OCR':>18} {'вердиктов изменено':>20} "
        f"{'спасено':>8} {'FP-фикс':>8} {'б/сигн':>7} {'цена':>7} {'время шага':>18}",
    ]
    for r in rows:
        mark = "*" if current is not None and abs(r.threshold - current) < 1e-9 else " "
        price = f"{r.gated / r.changed:.0f}:1" if r.changed else "—"
        per_file = f"{1000.0 * r.seconds / total:.0f}" if total else "—"
        out.append(
            f"{r.threshold:>5.2f}{mark}"
            f"{r.gated:>10d} ({_pct(r.gated, total):>6}) "
            f"{r.changed:>10d} ({_pct(r.changed, total):>6}) "
            f"{r.rescued:>8d} {r.fp_fixed:>8d} {r.no_signal:>7d} {price:>7} "
            f"{r.seconds:>8.0f} с ({per_file:>4} мс/файл)"
        )
    out.append("=" * 96)
    if current is not None:
        out.append("* — порог из конфига (naming.text_rescue_docscore_min)")
    out.append(
        "спасено — photo -> document (FN, ради которых гейт и существует); "
        "FP-фикс — document -> photo,\nот порога не зависит (гейт для вердикта "
        "'document' не ограничен). Порог менять только руками пользователя."
    )
    return "\n".join(out)


def probe_summary(frames: list[Frame], floor: float, g: junk.GateSettings) -> str:
    """What the gate never sees: OCR on a sample of frames BELOW the grid.

    Every row of the table above is measured on frames the lowest threshold already
    gates, so the table can only say how much benefit a HIGHER threshold gives up. It
    cannot say how much is already being given up today. This does: a random sample of
    clear-scene frames gets an OCR call it would never get in production, and the
    documents found among them are documents the gate misses at any threshold in the
    grid.
    """
    below = [f for f in frames
             if not f.has_faces and f.verdict == "photo" and f.doc_score < floor]
    probed = [f for f in below if f.text_frac is not None]
    if not probed:
        return ""
    hits = sum(1 for f in probed
               if junk.apply_text_frac(f.verdict, 0.0, f.text_frac, g)[2] == "ocr")
    rate = hits / len(probed)
    return (
        f"проба под сеткой (doc_score < {floor:.2f}): OCR на {len(probed)} кадрах из "
        f"{len(below)}, документов найдено {hits} ({_pct(hits, len(probed))})\n"
        f"  -> оценка ~{rate * len(below):.0f} документов, которые гейт не видит ни "
        f"при одном пороге сетки"
    )


def save_cache(path: Path, frames: list[Frame], ocr_ms: float, floor: float) -> None:
    """Per-frame aggregates for a later replay. File ids only — never paths."""
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "ocr_ms": ocr_ms,
        "floor": floor,
        "frames": [[f.file_id, int(f.has_faces), f.verdict, f.doc_score, f.text_frac]
                   for f in frames],
    }), encoding="utf-8")


def load_cache(path: Path) -> tuple[list[Frame], float, float]:
    """-> (frames, ocr_ms, floor). A cache of another version is an error, not a
    silently wrong table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise SystemExit(f"{path}: кэш версии {data.get('version')}, ожидается "
                         f"{CACHE_VERSION} — перемерить с --refresh")
    frames = [Frame(int(fid), bool(faces), str(verdict), float(doc), text_frac)
              for fid, faces, verdict, doc, text_frac in data["frames"]]
    return frames, float(data["ocr_ms"]), float(data["floor"])


def measure(cfg: Config, rows: list[sqlite3.Row], g: junk.GateSettings, floor: float,
            probe_below: int, seed: int) -> tuple[list[Frame], float]:  # pragma: no cover — ML
    """CLIP over the sample + OCR over what `floor` gates -> (frames, ms per OCR call).

    The OCR phase goes through the pipeline's own `_OcrPool` with the configured
    `naming.ocr_workers`, so the measured per-frame cost is the one the stage really
    pays (the workers contend — F73 measured ~2x from 4 threads, not 4x), not a
    single-threaded number that would make every row of the table look cheap.
    """
    s = naming_settings(cfg)
    classifier = clip_classifier(s)
    prompts = [prompt for _cls, prompt in junk._CLIP_CLASSES]
    doc_prompts = [prompt for _cls, prompt in junk._DOCUMENT_CLASSES]

    frames: list[Frame] = []
    jobs: list[junk.OcrJob] = []
    done = 0
    for chunk in batched(rows, s.clip_batch_size):
        paths = [r["path"] for r in chunk]
        probs = classifier(paths, prompts)
        noface_idx = [i for i, r in enumerate(chunk) if not r["has_faces"]]
        doc_score: dict[int, float] = {}
        if noface_idx:
            doc_probs = classifier([paths[i] for i in noface_idx], doc_prompts)
            for k, i in enumerate(noface_idx):
                doc_score[i] = junk._document_score(doc_probs[k])
        for i, (r, p) in enumerate(zip(chunk, probs)):
            best = int(np.argmax(p))
            heuristic = junk.heuristic_verdict(
                r["path"], r["width"], r["height"], r["camera_make"], r["camera_model"])
            verdict, _score = junk.clip_verdict(
                junk._CLIP_CLASSES[best][0], float(p[best]), heuristic,
                doc_score.get(i), junk._is_real_photo(r), g)
            score = doc_score.get(i, 0.0)
            frames.append(Frame(r["id"], bool(r["has_faces"]), verdict, score, None))
            if junk.ocr_gate_open(bool(r["has_faces"]), verdict, score, floor):
                jobs.append((r["id"], r["path"], r["width"], r["height"]))
        done += len(chunk)
        print(f"  CLIP {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    workers = junk.resolve_ocr_workers(cfg.raw)
    pool = junk._OcrPool(junk._resolve_detector_factory(cfg, None), workers)
    try:
        print(f"OCR: {len(jobs)} кадров в {workers} поток(а/ов)...")
        started = time.perf_counter()
        fracs = pool.text_frac(jobs)
        elapsed = time.perf_counter() - started
        ocr_ms = 1000.0 * elapsed / len(jobs) if jobs else 0.0
        print(f"OCR: {elapsed:.0f} с, {ocr_ms:.0f} мс/кадр (с учётом конкуренции потоков)")
        if probe_below:
            # The probe deliberately runs OCR where production never would — see
            # probe_summary. It is measured separately so it cannot skew ocr_ms.
            candidates = [f for f in frames
                          if not f.has_faces and f.verdict == "photo"
                          and f.doc_score < floor and f.file_id not in fracs]
            picked = random.Random(seed).sample(
                candidates, min(probe_below, len(candidates)))
            picked_ids = {f.file_id for f in picked}
            probe_jobs = [(r["id"], r["path"], r["width"], r["height"])
                          for r in rows if r["id"] in picked_ids]
            print(f"проба: OCR ещё на {len(probe_jobs)} кадрах под порогом {floor:.2f}...")
            fracs.update(pool.text_frac(probe_jobs))
    finally:
        pool.close()
    return ([Frame(f.file_id, f.has_faces, f.verdict, f.doc_score, fracs.get(f.file_id))
             for f in frames], ocr_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=2000,
                    help="frames to measure (default 2000)")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_GRID),
                    help="the doc_score grid (default 0.2 ... 0.6)")
    ap.add_argument("--cache", help="JSON with the per-frame aggregates: written after "
                                    "a measurement, replayed instead of one")
    ap.add_argument("--refresh", action="store_true",
                    help="measure again even if the cache exists")
    ap.add_argument("--probe-below", type=int, default=0, metavar="N",
                    help="also OCR N frames BELOW the grid — an estimate of the "
                         "documents the gate never sees (costs extra time)")
    ap.add_argument("--ocr-ms", type=float,
                    help="ms per OCR call for the time column (default: measured)")
    ap.add_argument("--startup-sec", type=float, default=DEFAULT_STARTUP_SEC,
                    help=f"model start-up added to a non-empty gate "
                         f"(default {DEFAULT_STARTUP_SEC:.0f} s)")
    args = ap.parse_args()

    thresholds = sorted(args.thresholds)  # argparse (nargs='+') guarantees at least one
    cfg = load_config(args.config)
    g = junk.gate_settings(cfg)
    floor = thresholds[0]
    cache = Path(args.cache) if args.cache else None

    if cache and cache.exists() and not args.refresh:
        frames, ocr_ms, cached_floor = load_cache(cache)
        print(f"кэш: {len(frames)} кадров из {cache}, OCR {ocr_ms:.0f} мс/кадр")
        if cached_floor > floor:
            print(f"ВНИМАНИЕ: кэш мерился от порога {cached_floor:.2f} — строки ниже "
                  f"него недосчитывают пользу (столбец 'б/сигн')")
    else:
        rows = sample_rows(str(cfg.database), args.sample, args.seed)
        if not rows:
            raise SystemExit("нет подходящих файлов в индексе — нечего мерить")
        print(f"выборка: {len(rows)} кадров, сетка {floor:.2f}..{thresholds[-1]:.2f}")
        frames, ocr_ms = measure(cfg, rows, g, floor, args.probe_below, args.seed)
        if cache:
            save_cache(cache, frames, ocr_ms, floor)
            print(f"кэш записан: {cache} (только file_id и агрегаты, без путей)")

    if args.ocr_ms is not None:
        ocr_ms = args.ocr_ms
    print()
    print(format_table(sweep(frames, thresholds, g, ocr_ms, args.startup_sec),
                       len(frames), current=g.text_rescue_docscore_min))
    probe = probe_summary(frames, floor, g)
    if probe:
        print(probe)


if __name__ == "__main__":
    main()
