"""Price the OCR thread ceiling of the junk stage: 1 / 2 / 4 / 6 / 8 workers.

`_DEFAULT_OCR_WORKERS_CAP` is 4 and was never measured. F73 set it "so a weak card is
not knocked over" — every worker builds its OWN easyocr Reader, i.e. its own copy of
the detector in VRAM — and the number stayed while the phase became the most expensive
one in the stage (F147, 2026-08-03: `junk_ocr` 614.6 s over 6 793 frames, 90.5 ms per
frame, 15% of the whole run).

The F73 measurement that does exist says the ceiling is probably too low: 1 -> 4
workers gave x3.7 on the test phase, which is nearly linear, and a lever that is still
linear where it was capped was capped by taste. So this sweeps past it.

The table has three columns that decide the question, and they decide it together:

    ms/frame     what a frame costs at this thread count
    speedup      against ONE worker — the shape of the curve, not a single point
    detectors    how many Readers actually built. A pool that could not build its
                 N-th Reader silently shrinks (see _OcrPool) and its row is then a
                 measurement of a SMALLER pool wearing a bigger label — which is
                 exactly the failure the conservative default was protecting against,
                 so it has to be visible rather than averaged away.

The pool, the detector factory and the gate are the stage's own (`junk._OcrPool`,
`junk._resolve_detector_factory`): a private re-implementation would price a pool
nobody runs, which is the lesson measure_ocr_gate.py opens with.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — the same rule every measurement in this project follows.

Usage (from the repo root, with the venv python; on the GPU profile for the number
that decides the default — the ceiling protects VRAM, and a CPU run measures a
different machine):
    python scripts/measure_ocr_workers.py                        # 200 frames, 1..8
    python scripts/measure_ocr_workers.py --sample 500 --workers 4 8 12
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk  # noqa: E402
from sorta.config import Config, load_config  # noqa: E402

# The grid. 1 is the baseline every speedup is measured against, 4 is the current
# default (a row you can read the status quo off), 6 and 8 are what the brief asks for.
DEFAULT_GRID = (1, 2, 4, 6, 8)

# What a bigger pool has to buy before the shipped default moves. The same x1.15 the
# VLM measurements pre-register: below that the extra Readers in VRAM are not paid for.
MIN_SPEEDUP = 1.15


@dataclass(frozen=True)
class WorkerRow:
    """One thread count over the same frames: what it cost and what it really was."""
    workers: int
    frames: int
    seconds: float
    detectors: int          # Readers actually built (< workers -> the pool shrank)
    answered: int           # frames that came back with a text_frac
    peak_vram_mb: float | None

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.seconds / self.frames if self.frames else 0.0

    @property
    def shrank(self) -> bool:
        return self.detectors < self.workers


def sample_jobs(db_path: str, n: int, seed: int) -> list[junk.OcrJob]:
    """`n` OCR jobs from the index — the population the gate draws from.

    Canonical photographs without a detected face: faces are the gate's unconditional
    veto (F15), so a frame with one never reaches OCR and would price nothing.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT f.id, f.path, f.width, f.height FROM files f
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
                 AND NOT EXISTS(SELECT 1 FROM faces fa
                                WHERE fa.file_id = f.id AND fa.bbox != '[]')
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()
    random.Random(seed).shuffle(rows)
    return [(r["id"], r["path"], r["width"], r["height"])
            for r in rows if Path(r["path"]).exists()][:n]


def _vram_peak_mb(reset: bool = False) -> float | None:  # pragma: no cover — needs CUDA
    """Peak VRAM reserved by torch since the last reset, MB (None — no CUDA).

    The resource the ceiling exists to protect: N Readers are N copies of the detector
    on the card, and a thread count that does not fit is not a speedup at any price.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
    except Exception:  # noqa: BLE001 — a measurement of a missing card is not an error
        return None


def measure_workers(cfg: Config, jobs: list[junk.OcrJob],
                    workers: int) -> WorkerRow:  # pragma: no cover — ML
    """One pass over the sample with `workers` threads, through the stage's own pool.

    A pool of its own per row, and closed before the next one: the Readers of one
    thread count must not still be on the card while the next count builds its own —
    that would price a VRAM peak nobody ever reaches in a run.
    """
    _vram_peak_mb(reset=True)
    pool = junk._OcrPool(junk._resolve_detector_factory(cfg, None), workers)
    try:
        started = time.perf_counter()
        fracs = pool.text_frac(jobs)
        seconds = time.perf_counter() - started
        detectors = pool.detectors_built
    finally:
        pool.close()
    return WorkerRow(workers=workers, frames=len(jobs), seconds=seconds,
                     detectors=detectors,
                     answered=sum(1 for v in fracs.values() if v is not None),
                     peak_vram_mb=_vram_peak_mb())


def format_table(rows: list[WorkerRow], default: int) -> str:
    """The sweep, with the current default marked — the table the brief asks for."""
    base = rows[0]
    out = [
        "=" * 96,
        f"ПОТОКИ OCR: {base.frames} кадров на строку, база — {base.workers} поток(а/ов)",
        f"{'потоков':>8} {'секунд':>9} {'мс/кадр':>10} {'ускорение':>11} "
        f"{'детекторов':>12} {'с сигналом':>12} {'пик VRAM':>11}",
    ]
    for r in rows:
        gain = (f"x{base.ms_per_frame / r.ms_per_frame:.2f}"
                if r.ms_per_frame and base.ms_per_frame else "—")
        vram = f"{r.peak_vram_mb:.0f} МБ" if r.peak_vram_mb is not None else "—"
        built = f"{r.detectors}{' !' if r.shrank else ''}"
        mark = "*" if r.workers == default else " "
        out.append(f"{r.workers:>7d}{mark} {r.seconds:>9.1f} {r.ms_per_frame:>10.1f} "
                   f"{gain:>11} {built:>12} {r.answered:>12d} {vram:>11}")
    out.append("=" * 96)
    out.append("* — текущий дефолт (naming.ocr_workers); ! — пул не смог построить "
               "столько детекторов\n  и ужался: строка измеряет пул поменьше, чем "
               "написано в её первой колонке")
    return "\n".join(out)


def outcome(rows: list[WorkerRow], default: int) -> str:
    """Does the ceiling move? — the pre-registered answer, decided by the numbers.

    Raising it needs BOTH: a real win over the current default, and every Reader of the
    winning row actually built. A row whose pool shrank is a row about fewer threads.
    """
    current = next((r for r in rows if r.workers == default), None)
    above = [r for r in rows if r.workers > default and not r.shrank]
    if current is None or not above or not current.ms_per_frame:
        return (f"сравнивать не с чем — в сетке нет исправной строки выше текущего "
                f"дефолта ({default})")
    best = min(above, key=lambda r: r.ms_per_frame)
    gain = current.ms_per_frame / best.ms_per_frame if best.ms_per_frame else 0.0
    if gain >= MIN_SPEEDUP:
        return (f"потолок поднимать: {best.workers} потоков дают x{gain:.2f} против "
                f"{default} ({best.ms_per_frame:.0f} мс/кадр против "
                f"{current.ms_per_frame:.0f}), все {best.detectors} детекторов "
                f"построились")
    return (f"потолок оставить {default}: лучшее выше него — {best.workers} потоков, "
            f"x{gain:.2f} при пороге x{MIN_SPEEDUP:.2f}; лишние Reader'ы в VRAM за "
            f"такую прибавку не платятся")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=200,
                    help="frames per thread count (default 200)")
    ap.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_GRID),
                    help="the thread counts to compare, the baseline first")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    if any(w < 1 for w in args.workers):
        raise SystemExit("--workers: потоков не меньше одного")

    cfg = load_config(args.config)
    jobs = sample_jobs(str(cfg.database), args.sample, args.seed)
    if not jobs:
        raise SystemExit("нет подходящих кадров в индексе — нечего мерить")
    grid = sorted(dict.fromkeys(args.workers))
    default = junk.resolve_ocr_workers(cfg.raw)
    print(f"выборка: {len(jobs)} кадров (канонические фото без лиц)")
    print(f"сетка: {', '.join(str(w) for w in grid)}; дефолт сейчас {default}, "
          f"ядер в системе {os.cpu_count()}")

    rows: list[WorkerRow] = []
    for workers in grid:
        print(f"{workers} поток(а/ов): {len(jobs)} кадров...")
        rows.append(measure_workers(cfg, jobs, workers))
    print()
    print(format_table(rows, default))
    print(f"ИТОГ: {outcome(rows, default)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
