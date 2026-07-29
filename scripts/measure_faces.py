"""Throughput of the faces stage, per configuration — without touching the DB.

Why a separate script: `faces` is incremental, so once a collection is processed a
re-run does nothing and there is nothing left to time. This takes a fixed sample of
already-indexed files and runs detection over it as the pipeline would, reporting
img/s. Nothing is written: the DB is opened read-only and the hits are counted, not
stored.

Two shapes are measured, because they are not the same pipeline:

  pipeline   — the real `faces._detect_parallel`, whatever shape it is in right now.
               Before F87 decode and inference were coupled in one thread (while a
               worker read a 40 MB RAW its session idled, and `faces.decode_workers`
               was not consulted at all on that path); F87 split them.
  decoupled  — a prototype of the F64 shape that CLIP already uses: a pool of D
               decode threads feeding N inference sessions. Same sessions, same
               decode, only the coupling removed. If this is faster, the fix is in
               the pipeline, not in the config.

Usage (from the repo root, with the venv python):
    python scripts/measure_faces.py                       # 500 frames, 4 vs 8 sessions
    python scripts/measure_faces.py --sample 800 --infer-workers 4 8 12
    python scripts/measure_faces.py --skip-decoupled      # current shape only

Run it on an otherwise idle machine: a concurrent test suite or an open UI decodes
frames on the same cores and quietly doubles the numbers.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from sorta import faces
from sorta.config import load_config


def sample_rows(db_path: str, n: int, seed: int) -> list[sqlite3.Row]:
    """`n` canonical photos that still exist on disk, deterministic for a given seed."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, path, orientation FROM files "
            "WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo' "
            "ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    random.Random(seed).shuffle(rows)
    picked = [r for r in rows if Path(r["path"]).exists()]
    return picked[:n]


def warm_os_cache(rows: list[sqlite3.Row]) -> float:
    """Read every sampled file once, so no configuration pays for cold disk.

    Without this the first measured configuration reads from the platters and every
    later one from RAM — which looks exactly like the later one being faster.
    """
    started = time.perf_counter()
    for r in rows:
        try:
            with open(r["path"], "rb") as fh:
                while fh.read(1 << 20):
                    pass
        except OSError:
            pass
    return time.perf_counter() - started


def run_pipeline(rows: list[sqlite3.Row], factory, workers: int,
                 decode_workers: int) -> int:
    """The real `faces._detect_parallel` — whatever shape it currently is.

    Before F87 it decoded inside its inference workers and took no decode_workers
    argument. Measuring the merged pipeline (not the prototype below) is what makes
    this an acceptance check; to compare against the old shape, check the pre-F87
    files out into the tree and run this script from that same session — the spread
    BETWEEN sessions (±14%) is larger than the effect, so only within-session
    comparisons mean anything.
    """
    found = 0

    def on_result(_row: sqlite3.Row, hits) -> None:
        nonlocal found
        if hits:
            found += len(hits)

    faces._detect_parallel(rows, faces._decode_for_faces, factory, workers,
                           decode_workers, on_result)
    return found


def run_decoupled(rows: list[sqlite3.Row], factory, workers: int,
                  decode_workers: int) -> int:
    """Prototype: a decode pool feeds N inference sessions (the F64 shape).

    The in-flight window is bounded on both sides — faces._prefetch_decode keeps
    ~2x decode_workers frames, and no more than 2x workers wait for inference — so
    full-resolution frames cannot pile up in memory.
    """
    found = 0
    local = threading.local()

    def infer_one(img):
        infer = getattr(local, "infer", None)
        if infer is None:
            infer = local.infer = factory()
        return infer(img)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set = set()
        for _row, img, err in faces._prefetch_decode(
                rows, faces._decode_for_faces, decode_workers):
            if err is not None or img is None:
                continue
            pending.add(pool.submit(infer_one, img))
            if len(pending) >= workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    hits = future.result()
                    found += len(hits) if hits else 0
        for future in pending:
            hits = future.result()
            found += len(hits) if hits else 0
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=500,
                        help="frames per measurement (default 500)")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--infer-workers", type=int, nargs="+", default=[4, 8],
                        help="session counts to compare (default: 4 8)")
    parser.add_argument("--decode-workers", type=int, default=16,
                        help="decode pool for the decoupled shape (default 16)")
    parser.add_argument("--skip-decoupled", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = str(cfg.raw.get("database"))
    rows = sample_rows(db_path, args.sample, args.seed)
    if not rows:
        raise SystemExit("нет подходящих файлов в индексе — нечего мерить")

    settings = faces._settings(cfg)

    def factory():
        return faces._insightface_infer(settings)

    print(f"выборка: {len(rows)} кадров из {db_path}")
    print(f"прогрев файлового кэша: {warm_os_cache(rows):.1f} с")

    # insightface prints a dozen stdout lines for every session it brings up, so a
    # line-per-measurement output drowns in the noise — the final table is collected
    # and printed as ONE block at the end.
    results: list[tuple[str, int, float, float, int]] = []

    def measure(shape: str, workers: int, run) -> None:
        started = time.perf_counter()
        found = run()
        elapsed = time.perf_counter() - started
        results.append((shape, workers, elapsed, len(rows) / elapsed, found))

    for workers in args.infer_workers:
        measure("pipeline", workers,
                lambda w=workers: run_pipeline(rows, factory, w, args.decode_workers))
        if not args.skip_decoupled:
            measure("decoupled", workers,
                    lambda w=workers: run_decoupled(rows, factory, w, args.decode_workers))

    print("\n" + "=" * 52)
    print(f"РЕЗУЛЬТАТ ({len(rows)} кадров, decode_workers={args.decode_workers})")
    print(f"{'режим':11s} {'сессий':>7s} {'время, с':>10s} {'img/s':>8s} {'лиц':>7s}")
    for shape, workers, elapsed, rate, found in results:
        print(f"{shape:11s} {workers:7d} {elapsed:10.1f} {rate:8.2f} {found:7d}")
    print("=" * 52)
    print("Каждая строка платит загрузку своих сессий (~секунды на сессию) —"
          " на выборке в сотни кадров это единицы процентов, но на маленькой"
          " выборке результат этим и определится.")


if __name__ == "__main__":
    main()
