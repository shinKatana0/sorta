"""Price `vlm.workers`: 4 / 6 / 8 / 12 preparation threads for the deep junk tier.

`default_vlm_workers()` returns min(4, cores). On the machine this was measured on
there are 24 cores, and the F101 profile says why four is not obviously enough:
preparing a frame costs ~0.6 s of CPU, the model call ~0.19 s of GPU, and the two
alternate strictly. One tick of the card needs more than three ticks of the processor,
so four threads cannot keep it fed — and the live run of 2026-08-03 agrees: the card sat
at 51% during `junk_vlm` (26% before the pipeline existed). Half the time it waits.

The sweep therefore measures ONE thing — how many threads it takes to feed the card —
and reports it as the number the profile is about: the share of the wall clock the GPU
half is actually running.

Two ways to run it, and the difference matters when the numbers are read:

    --full        the real thing: the weights are loaded and the sweep runs the stage's
                  own pass (`junk._vlm_labels`) at every thread count, with the GPU load
                  read off the driver and the labels compared across counts. Needs the
                  card free — the runtime peaks at 20.5 GB, and a second copy does not
                  fit next to somebody else's.
    (default)     the CPU half is REAL — the same decode and the same processor, on the
                  same frames, through the same pipeline — and the GPU half is replaced
                  by a sleep of `--gen-ms` (0.19 s, the F101 measurement). It needs no
                  weights and no VRAM, and it answers the question the ceiling is about,
                  because the bottleneck being priced is the CPU half and its contention.
                  What it cannot answer is anything about the card itself.

The invariant of F101 is checked in both: the labels of every thread count must come
back in the candidate order and be identical to the labels of the baseline. A thread
count that changes an answer is a defect, not a speed, and the script exits non-zero.

Memory is a column and not a footnote (the brief's third requirement): the frames in
flight are ~2 per worker and they are CPU tensors, so more threads mean more RAM. The
process peak is printed for every row.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — the same rule every measurement in this project follows.

Usage (from the repo root, with the venv python):
    python scripts/measure_vlm_workers.py                     # 120 frames, 1/4/6/8/12
    python scripts/measure_vlm_workers.py --sample 300 --workers 4 8
    python scripts/measure_vlm_workers.py --full              # needs a free GPU + [vlm]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import random
import sqlite3
import statistics
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk, naming  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.junk import SplitVlmClassifier  # noqa: E402

# The grid: 1 (no pipeline at all — the pre-F101 shape), 4 (the current default, the row
# the status quo is read off) and the counts the brief asks for.
DEFAULT_GRID = (1, 4, 6, 8, 12)
# The GPU half of one frame, measured in F101 and confirmed by the live run: 0.19 s of
# generate() per frame. In the default mode this is what the card's turn is replaced by.
DEFAULT_GEN_MS = 190.0
# What a bigger pool has to buy before the shipped default moves — the same x1.15 the
# other VLM measurements of this project pre-register.
MIN_SPEEDUP = 1.15
# How often the GPU-load sampler asks the driver (--full only).
GPU_POLL_SEC = 0.1
# The labels the stub answers with. The set of the deep tier, so a row of the table
# looks like a row of a run; WHICH one a frame gets is a function of the path (see
# StubbedFrame) and means nothing beyond making the order invariant checkable.
_JUNK_LABELS = ("photo", "document", "product", "screenshot")


@dataclass(frozen=True)
class WorkerRow:
    """One thread count over the same frames: the cost, the card, the memory."""
    workers: int
    labels: tuple[str, ...]
    wall_sec: float
    gen_sec: float                # seconds spent inside the model half
    cpu_cores: float              # process CPU seconds per wall second
    peak_rss_mb: float | None
    gpu_util_pct: float | None = None   # --full only: read off the driver

    @property
    def frames(self) -> int:
        return len(self.labels)

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.wall_sec / self.frames if self.frames else 0.0

    @property
    def frames_per_sec(self) -> float:
        return self.frames / self.wall_sec if self.wall_sec else 0.0

    @property
    def gpu_busy_pct(self) -> float:
        """The share of the wall clock the model half was running — the F101 number.

        Computed from the seconds, not from the driver: it is the same quantity the
        profile is about ("half the time the card waits for data") and it exists in both
        modes, which is what makes the two comparable at all.
        """
        return 100.0 * self.gen_sec / self.wall_sec if self.wall_sec else 0.0


def sample_paths(db_path: str, n: int, seed: int) -> tuple[list[str], str]:
    """`n` frames of the deep tier's own kind -> (paths, where they came from).

    The candidates of a previous deep run (`media_class.source='vlm'`) ARE the population
    the gate produces, so a measurement over them is a measurement of the stage. Without
    such a run the fallback is canonical photographs without faces — what the candidates
    are drawn from. (The same choice, and for the same reason, as measure_vlm_speed.py.)
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT f.path, mc.source AS source FROM files f
               LEFT JOIN media_class mc ON mc.file_id = f.id
               WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
                 AND NOT EXISTS(SELECT 1 FROM faces fa
                                WHERE fa.file_id = f.id AND fa.bbox != '[]')
               ORDER BY f.id"""
        ).fetchall()
    finally:
        conn.close()
    candidates = [r for r in rows if r["source"] == "vlm"]
    origin = "кандидаты прошлого deep-прогона (source='vlm')"
    if not candidates:
        candidates, origin = rows, "канонические фото без лиц (deep-прогона ещё не было)"
    random.Random(seed).shuffle(candidates)
    return [r["path"] for r in candidates if Path(r["path"]).exists()][:n], origin


# --- What the machine did ------------------------------------------------------

def peak_rss_mb() -> float | None:
    """Peak resident memory of THIS PROCESS, MB (None where the OS will not say).

    The process peak is a high-water mark and cannot be reset, so it only ever grows
    across the sweep — read the DELTA between rows, which is what the brief's third
    requirement asks for (does a bigger pool inflate the queue of prepared frames?).
    """
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        try:
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(
                ctypes.c_void_p(kernel32.GetCurrentProcess()),
                ctypes.byref(counters), counters.cb)
        except OSError:  # pragma: no cover — a Windows without psapi
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024) if ok else None
    try:  # pragma: no cover — POSIX
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return peak / 1024.0 if sys.platform.startswith("linux") else peak / (1024 * 1024)
    except Exception:  # noqa: BLE001 — a measurement of memory is not worth a crash
        return None


class GpuSampler:
    """Mean GPU load over one row, polled in the background; silent where there is none.

    Only `--full` has a card to watch. The seconds-based `gpu_busy_pct` above is the
    column that exists in both modes; this one is the driver's own opinion next to it.
    """

    def __init__(self, poll_sec: float = GPU_POLL_SEC) -> None:
        self._poll = poll_sec
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        if self._read() is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def mean_pct(self) -> float | None:
        return statistics.fmean(self._samples) if self._samples else None

    def _read(self) -> float | None:  # pragma: no cover — needs a CUDA driver
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.utilization())
        except Exception:  # noqa: BLE001 — no driver: report nothing, measure on
            return None

    def _run(self) -> None:  # pragma: no cover — needs a CUDA driver
        while not self._stop.wait(self._poll):
            value = self._read()
            if value is not None:
                self._samples.append(value)


# --- The two classifiers -------------------------------------------------------

class GenClock:
    """Adds up the seconds spent in the model half, across the threads that spend them.

    The GPU half runs on the consumer's thread alone (that is the shape of the
    pipeline), but the clock is locked anyway: it is also used by the `--full` path,
    where a future runtime is free to change that, and a silently wrong total is worse
    than a lock nobody contends for.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.seconds = 0.0

    def add(self, elapsed: float) -> None:
        with self._lock:
            self.seconds += elapsed


@dataclass(frozen=True)
class StubbedFrame:
    """A prepared frame plus the path it came from — the stub's label is derived from it.

    The real `PreparedFrame` carries no path (the model does not need one), and the
    stub's label has to be a FUNCTION OF THE FRAME rather than of the arrival order, or
    the invariant check below would compare noise with noise and always pass.
    """
    path: str
    frame: object


def stub_classifier(cfg_max_edge: int, gen_ms: float,
                    clock: GenClock) -> SplitVlmClassifier:
    """The real CPU half, and a sleep where the card would be.

    `prepare` is the stage's own — `junk.vlm_classifier_from` over a runtime built from
    a REAL processor — so the decode, the chat template and the image preprocessing all
    happen exactly as they do in a run, on real frames, with the real contention between
    threads. Only the model is missing, and the frame's turn on the card is replaced by
    a sleep of the measured length.

    A sleep is an honest stand-in for `generate()` in the one respect this measurement
    depends on: both release the GIL for their whole duration, so the preparation threads
    keep running while the frame is "on the card". It is not a stand-in for anything
    about the card itself — the label it returns is derived from the path, so the order
    invariant is still checked, but no model ever sees a frame.
    """
    model_name = _MODEL_NAME[0]
    processor = naming.qwen_processor(model_name, use_fast=True)
    # device is never used: `generate` is what would move the tensors, and this runtime's
    # generate is not called (the classifier below replaces the whole GPU half).
    runtime = naming.qwen_runtime(None, processor, "cpu",
                                  lambda: naming.qwen_processor(model_name, use_fast=True))
    real = junk.vlm_classifier_from(runtime, max_edge=cfg_max_edge)
    assert isinstance(real, SplitVlmClassifier)

    def prepare(path: str) -> StubbedFrame:
        return StubbedFrame(path=path, frame=real.prepare(path))

    def classify_prepared(prepared: StubbedFrame) -> str:
        started = time.perf_counter()
        time.sleep(gen_ms / 1000.0)
        clock.add(time.perf_counter() - started)
        return _JUNK_LABELS[zlib.crc32(prepared.path.encode("utf-8", "replace"))
                            % len(_JUNK_LABELS)]

    return SplitVlmClassifier(prepare=prepare, classify_prepared=classify_prepared)


def full_classifier(model_name: str, max_edge: int,
                    clock: GenClock) -> SplitVlmClassifier:  # pragma: no cover — ML
    """The stage's own classifier over the loaded weights, with the model half timed."""
    runtime = naming.shared_vlm(model_name)
    real = junk.vlm_classifier_from(runtime, max_edge=max_edge)
    if not isinstance(real, SplitVlmClassifier):
        raise SystemExit("рантайм не отдал половины (SplitVlm) — мерить конвейер нечем")

    def classify_prepared(prepared: object) -> str:
        started = time.perf_counter()
        try:
            return real.classify_prepared(prepared)
        finally:
            clock.add(time.perf_counter() - started)

    return SplitVlmClassifier(prepare=real.prepare, classify_prepared=classify_prepared)


# Set once by main(); the stub classifier needs the processor of the configured model
# and building it per row would time a load instead of a pass.
_MODEL_NAME = [""]


def measure_workers(classifier: SplitVlmClassifier, paths: list[str], workers: int,
                    clock: GenClock) -> WorkerRow:
    """One pass over the sample with `workers` preparation threads.

    Through `junk._vlm_labels` — the stage's own pipeline, not a copy of it — so the
    window, the FIFO of futures and the single consumer thread are the ones that run in
    production. `closing()` for the same reason the stage uses it: the threads must be
    gone before the next row starts, or a row would be measured against the leftovers of
    the previous one.
    """
    from contextlib import closing

    clock.seconds = 0.0
    labels: list[str] = []
    with GpuSampler() as gpu:
        cpu0, wall0 = time.process_time(), time.perf_counter()
        stream = junk._vlm_labels(classifier, paths, workers)
        with closing(stream):
            for item in stream:
                labels.append("ERROR" if isinstance(item, BaseException) else item)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
    return WorkerRow(workers=workers, labels=tuple(labels), wall_sec=wall,
                     gen_sec=clock.seconds, cpu_cores=cpu / wall if wall else 0.0,
                     peak_rss_mb=peak_rss_mb(), gpu_util_pct=gpu.mean_pct)


# --- The report ----------------------------------------------------------------

def format_table(rows: list[WorkerRow], default: int, mode: str) -> str:
    """The sweep: threads -> seconds -> the card's share -> memory."""
    base = rows[0]
    out = [
        "=" * 108,
        f"ПОТОКИ ПОДГОТОВКИ VLM ({mode}): {base.frames} кадров на строку, "
        f"база — {base.workers} поток(а/ов)",
        f"{'потоков':>8} {'секунд':>9} {'мс/кадр':>10} {'кадр/с':>8} {'ускорение':>11} "
        f"{'занятость GPU':>14} {'GPU (драйвер)':>14} {'ядер':>6} {'пик RSS':>10}",
    ]
    for r in rows:
        gain = (f"x{r.frames_per_sec / base.frames_per_sec:.2f}"
                if base.frames_per_sec else "—")
        driver = f"{r.gpu_util_pct:.0f}%" if r.gpu_util_pct is not None else "—"
        rss = f"{r.peak_rss_mb:.0f} МБ" if r.peak_rss_mb is not None else "—"
        mark = "*" if r.workers == default else " "
        out.append(f"{r.workers:>7d}{mark} {r.wall_sec:>9.1f} {r.ms_per_frame:>10.0f} "
                   f"{r.frames_per_sec:>8.2f} {gain:>11} {r.gpu_busy_pct:>13.0f}% "
                   f"{driver:>14} {r.cpu_cores:>6.2f} {rss:>10}")
    out.append("=" * 108)
    out.append("занятость GPU — доля стены, которую половина модели реально считает "
               "(из секунд).\nпик RSS — high-water mark процесса: он только растёт, "
               "смотреть на РАЗНИЦУ между строками.")
    return "\n".join(out)


def format_invariant(rows: list[WorkerRow]) -> tuple[str, bool]:
    """(the F101 order-invariant block, did every row agree?) — the acceptance criterion.

    Nothing about the seconds counts if this says no: the labels are applied in the
    candidate order whatever the thread count is, and a row that disagrees with the
    baseline has broken the one property that makes this a perf change.
    """
    base = rows[0]
    lines = [f"ИНВАРИАНТ F101 (порядок и содержание меток против {base.workers} "
             f"поток(а/ов), кадров {base.frames}):"]
    ok = True
    for r in rows[1:]:
        if r.labels == base.labels:
            lines.append(f"  {r.workers:>3d} поток(а/ов): совпадение полное")
            continue
        ok = False
        diff = sum(1 for a, b in zip(base.labels, r.labels) if a != b)
        diff += abs(len(base.labels) - len(r.labels))
        lines.append(f"  {r.workers:>3d} поток(а/ов): РАСХОЖДЕНИЙ {diff} — СТОП, "
                     f"разбираться")
    if len(rows) == 1:
        lines.append("  (сравнивать не с чем — запрошено одно число потоков)")
    return "\n".join(lines), ok


def outcome(rows: list[WorkerRow], default: int) -> str:
    """Does the default move? — the pre-registered answer, decided by the numbers."""
    current = next((r for r in rows if r.workers == default), None)
    above = [r for r in rows if r.workers > default]
    if current is None or not above or not current.frames_per_sec:
        return (f"сравнивать не с чем — в сетке нет строки выше текущего дефолта "
                f"({default})")
    best = max(above, key=lambda r: r.frames_per_sec)
    gain = best.frames_per_sec / current.frames_per_sec
    if gain < MIN_SPEEDUP:
        return (f"дефолт оставить {default}: лучшее выше него — {best.workers} потоков, "
                f"x{gain:.2f} при пороге x{MIN_SPEEDUP:.2f}")
    # The knee, not the maximum: the smallest count that gets within 5% of the best is
    # what a default should be — the rest buys noise and spends RAM on it.
    knee = min((r for r in above if r.frames_per_sec >= 0.95 * best.frames_per_sec),
               key=lambda r: r.workers)
    return (f"дефолт поднимать до {knee.workers}: x"
            f"{knee.frames_per_sec / current.frames_per_sec:.2f} против {default} "
            f"({knee.frames_per_sec:.2f} кадр/с против {current.frames_per_sec:.2f}), "
            f"занятость GPU {current.gpu_busy_pct:.0f}% -> {knee.gpu_busy_pct:.0f}%; "
            f"выше — {best.workers} потоков дают лишь x"
            f"{best.frames_per_sec / knee.frames_per_sec:.2f} сверх этого")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=120,
                    help="frames per thread count (default 120)")
    ap.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_GRID),
                    help="the thread counts to compare, the baseline first")
    ap.add_argument("--gen-ms", type=float, default=DEFAULT_GEN_MS,
                    help=f"the model half of one frame, ms (default {DEFAULT_GEN_MS:.0f} "
                         f"— the F101 measurement); ignored with --full")
    ap.add_argument("--full", action="store_true",
                    help="load the weights and measure the real pass (needs a free GPU)")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    if any(w < 1 for w in args.workers):
        raise SystemExit("--workers: потоков не меньше одного")

    cfg = load_config(args.config)
    paths, origin = sample_paths(str(cfg.database), args.sample, args.seed)
    if not paths:
        raise SystemExit("нет подходящих кадров в индексе — нечего мерить")
    grid = sorted(dict.fromkeys(args.workers))
    _MODEL_NAME[0] = cfg.vlm.model
    mode = ("реальная модель" if args.full
            else f"половина модели заменена сном {args.gen_ms:.0f} мс")
    print(f"выборка: {len(paths)} кадров — {origin}")
    print(f"модель: {cfg.vlm.model}, {cfg.vlm.max_edge}px; режим: {mode}")
    print(f"сетка: {', '.join(str(w) for w in grid)}; дефолт сейчас {cfg.vlm.workers}, "
          f"ядер в системе {os.cpu_count()}")

    clock = GenClock()
    classifier = (full_classifier(cfg.vlm.model, cfg.vlm.max_edge, clock) if args.full
                  else stub_classifier(cfg.vlm.max_edge, args.gen_ms, clock))
    rows: list[WorkerRow] = []
    for workers in grid:
        print(f"{workers} поток(а/ов): {len(paths)} кадров...")
        rows.append(measure_workers(classifier, paths, workers, clock))
    print()
    print(format_table(rows, cfg.vlm.workers, mode))
    report, ok = format_invariant(rows)
    print(report)
    print()
    print(f"ИТОГ: {outcome(rows, cfg.vlm.workers)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
