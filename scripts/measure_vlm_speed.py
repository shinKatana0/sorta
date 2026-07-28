"""Price the F101 speedup of the deep junk tier — and prove it changed no verdict.

The deep tier is worth keeping: on the live run of 2026-07-28 it changed 2 592 of
24 196 verdicts (10.7%), 2 202 of them into `product`, a class the fast tier never
produces. It is also, at 1.38 frames/s, a 95-minute stage. The profile said why — not
heavy, SEQUENTIAL: ~0.6 s of CPU (decode + the processor's image preprocessing) then
~0.19 s of GPU per frame, with no overlap, 0.84 cores busy out of 24 and the card at
~26%. F101 pulls on the two levers that follow from that: the fast image processor
(`use_fast=True`, one line) and a pool of preparation threads overlapping the CPU half
with the GPU half.

This script measures both, on ONE loaded model, over the same frames:

    baseline   slow processor, no overlap  — the code as it ran on 2026-07-28
    fast       fast processor, no overlap  — lever 1 alone
    pipelined  fast processor + N threads  — levers 1 and 2, i.e. the shipped path

and prints, per mode: median and p90 ms per frame, mean GPU load, peak VRAM, how many
cores the process actually kept busy — and the comparison the whole feature stands on,
LABEL BY LABEL against the baseline. A speedup that moves a single verdict is not a
speedup, it is a regression with a stopwatch; the report says so in as many words and
`main` exits non-zero.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — mismatches are reported as counts per label pair, which is all that is
needed to go and look, and never a list of where the documents are (the same rule
measure_ocr_gate.py and measure_streetclip.py follow).

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_vlm_speed.py                    # 100 frames, all three modes
    python scripts/measure_vlm_speed.py --sample 300 --workers 6
    python scripts/measure_vlm_speed.py --modes baseline pipelined
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import junk, naming  # noqa: E402
from sorta.config import load_config  # noqa: E402

# name -> (fast image processor?, overlap the halves?)
MODES: dict[str, tuple[bool, bool]] = {
    "baseline": (False, False),
    "fast": (True, False),
    "pipelined": (True, True),
}
DEFAULT_MODES = ("baseline", "fast", "pipelined")
# How often the GPU-load sampler asks the driver. Fine enough to catch the gaps
# between generate() calls (~0.6 s wide in the baseline), cheap enough to ignore.
GPU_POLL_SEC = 0.1


@dataclass(frozen=True)
class ModeResult:
    """What one pass over the sample cost, and what it decided."""
    name: str
    workers: int
    fast_processor: bool
    labels: tuple[str, ...]
    frame_ms: tuple[float, ...]   # wall time between consecutive labels
    wall_sec: float
    cpu_cores: float              # process CPU seconds per wall second
    gpu_util_pct: float | None    # None — no CUDA / the driver did not answer
    peak_vram_mb: float | None

    @property
    def median_ms(self) -> float:
        return statistics.median(self.frame_ms) if self.frame_ms else 0.0

    @property
    def p90_ms(self) -> float:
        return percentile(self.frame_ms, 0.9)

    @property
    def frames_per_sec(self) -> float:
        return len(self.labels) / self.wall_sec if self.wall_sec else 0.0


def percentile(values: tuple[float, ...] | list[float], q: float) -> float:
    """The q-quantile by nearest rank — no interpolation, no numpy, empty -> 0.0.

    Nearest rank on purpose: with 100 frames the p90 should be a frame that really
    took that long, not the average of two that did not.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return ordered[rank - 1]


def sample_paths(db_path: str, n: int, seed: int) -> tuple[list[str], str]:
    """`n` frames of the deep tier's own kind -> (paths, where they came from).

    First choice is the real thing: files a previous deep run already classified
    (media_class.source='vlm') ARE the candidates the gate produces, so a measurement
    over them is a measurement of the stage. Without such a run there is nothing to
    read, and the fallback is canonical photos without faces — the population the
    candidates are drawn from, which is close enough to price a frame.
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


class GpuSampler:
    """Mean GPU load over a mode, polled in the background; silent where there is none.

    The number this feature exists to move: 26% average on the live run, against a card
    that is free the rest of the time. It is sampled rather than computed because the
    only honest source is the driver.
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
        except Exception:  # noqa: BLE001 — no driver / no pynvml: report nothing, measure on
            return None

    def _run(self) -> None:  # pragma: no cover — needs a CUDA driver
        while not self._stop.wait(self._poll):
            value = self._read()
            if value is not None:
                self._samples.append(value)


def _vram_peak_mb(reset: bool = False) -> float | None:  # pragma: no cover — needs CUDA
    """Peak VRAM reserved by torch since the last reset, MB (None — no CUDA).

    Reserved, not allocated: the allocator's blocks are what actually occupy the card,
    and 23.1 GB of 24.4 is where this run already sits — the acceptance criterion is
    that this number does NOT grow.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return torch.cuda.max_memory_reserved() / (1024 * 1024)
    except Exception:  # noqa: BLE001 — a measurement of a missing card is not an error
        return None


def run_mode(name: str, classifier: junk.VlmClassifyFn, paths: list[str],
             workers: int, fast_processor: bool) -> ModeResult:
    """One pass over `paths` through the pipeline's own `_vlm_labels`.

    The stage's function, not a copy of it: what is being measured has to be what runs,
    or the table prices something nobody executes (the lesson `measure_ocr_gate.py`
    starts with). `workers=1` is the serial pass, i.e. the code before F101.
    """
    _vram_peak_mb(reset=True)
    frame_ms: list[float] = []
    labels: list[str] = []
    with GpuSampler() as gpu:
        cpu0, wall0 = time.process_time(), time.perf_counter()
        previous = wall0
        for item in junk._vlm_labels(classifier, paths, workers):
            now = time.perf_counter()
            frame_ms.append((now - previous) * 1000.0)
            previous = now
            labels.append("ERROR" if isinstance(item, BaseException) else item)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
    return ModeResult(
        name=name, workers=workers, fast_processor=fast_processor,
        labels=tuple(labels), frame_ms=tuple(frame_ms), wall_sec=wall,
        cpu_cores=cpu / wall if wall else 0.0,
        gpu_util_pct=gpu.mean_pct, peak_vram_mb=_vram_peak_mb(),
    )


def label_mismatches(baseline: ModeResult,
                     other: ModeResult) -> dict[tuple[str, str], int]:
    """{(baseline label, other label): count} — empty means byte-identical verdicts.

    Counts, never file ids: a mismatch table is a table about how the two modes differ,
    not a list of somebody's documents. A different NUMBER of labels is itself a
    mismatch and is reported as one.
    """
    out: dict[tuple[str, str], int] = {}
    for a, b in zip(baseline.labels, other.labels):
        if a != b:
            out[(a, b)] = out.get((a, b), 0) + 1
    missing = abs(len(baseline.labels) - len(other.labels))
    if missing:
        out[("<нет кадра>", "<нет кадра>")] = missing
    return out


def format_table(results: list[ModeResult]) -> str:
    """The speed table: one row per mode, the speedup measured against the first."""
    base = results[0]
    out = [
        "=" * 96,
        f"VLM-ПРОХОД: {len(base.labels)} кадров, база — режим '{base.name}'",
        f"{'режим':>10} {'потоков':>8} {'процессор':>10} {'медиана':>9} {'p90':>9} "
        f"{'кадр/с':>8} {'ускорение':>10} {'GPU':>6} {'ядер':>6} {'пик VRAM':>10}",
    ]
    for r in results:
        speedup = f"x{r.frames_per_sec / base.frames_per_sec:.2f}" \
            if base.frames_per_sec else "—"
        gpu = f"{r.gpu_util_pct:.0f}%" if r.gpu_util_pct is not None else "—"
        vram = f"{r.peak_vram_mb:.0f} МБ" if r.peak_vram_mb is not None else "—"
        out.append(
            f"{r.name:>10} {r.workers:>8d} "
            f"{'быстрый' if r.fast_processor else 'медленный':>10} "
            f"{r.median_ms:>8.0f}м {r.p90_ms:>8.0f}м {r.frames_per_sec:>8.2f} "
            f"{speedup:>10} {gpu:>6} {r.cpu_cores:>6.2f} {vram:>10}"
        )
    out.append("=" * 96)
    return "\n".join(out)


def format_verdicts(results: list[ModeResult]) -> tuple[str, bool]:
    """(the verdict-comparison block, everything matched?) — the acceptance criterion.

    Nothing about the numbers above counts if this says no: the whole feature is a
    perf change, and a perf change that moves a label has broken the tier it was
    speeding up.
    """
    base = results[0]
    lines = [f"ВЕРДИКТЫ (сверка с '{base.name}', кадров {len(base.labels)}):"]
    ok = True
    for r in results[1:]:
        diff = label_mismatches(base, r)
        if not diff:
            lines.append(f"  {r.name:>10}: совпадение полное")
            continue
        ok = False
        total = sum(diff.values())
        lines.append(f"  {r.name:>10}: РАСХОЖДЕНИЙ {total} — СТОП, разбираться")
        for (was, now), count in sorted(diff.items(), key=lambda kv: -kv[1]):
            lines.append(f"{'':>14}{was} -> {now}: {count}")
    if len(results) == 1:
        lines.append("  (сравнивать не с чем — запрошен один режим)")
    return "\n".join(lines), ok


def measure(model_name: str, paths: list[str],
            modes: list[str], workers: int) -> list[ModeResult]:  # pragma: no cover — ML
    """Every requested mode on ONE load of the weights (20.5 GB — a second does not fit).

    The processor is the only thing rebuilt between modes: it costs milliseconds, and
    building it twice is what lets the slow and the fast preprocessing be compared
    without reloading the model or restarting the script.
    """
    model, processor, device = naming.load_qwen(model_name, use_fast=True)
    processors = {True: processor,
                  False: naming.qwen_processor(model_name, use_fast=False)}
    if not naming.processor_is_fast(processor):
        print("ВНИМАНИЕ: transformers не отдал быстрый процессор для этой модели — "
              "строки 'быстрый' и 'медленный' измеряют одно и то же")
    results: list[ModeResult] = []
    for name in modes:
        fast, overlap = MODES[name]
        n_workers = workers if overlap else 1
        # ...and every preparation thread its own processor, exactly as the pipeline
        # builds them in production — sharing one is not safe to preprocess with, and a
        # measurement of a setup nobody runs is worth nothing.
        classifier = junk.vlm_classifier_from(naming.qwen_runtime(
            model, processors[fast], device,
            lambda fast=fast: naming.qwen_processor(model_name, use_fast=fast)))
        print(f"режим '{name}': {len(paths)} кадров, потоков {n_workers}, "
              f"процессор {'быстрый' if fast else 'медленный'}...")
        results.append(run_mode(name, classifier, paths, n_workers, fast))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=100,
                    help="frames per mode (default 100)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES),
                    choices=list(MODES), help="which modes to run, base first")
    ap.add_argument("--workers", type=int,
                    help="preparation threads for the pipelined mode "
                         "(default: naming.vlm_workers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, origin = sample_paths(str(cfg.database), args.sample, args.seed)
    if not paths:
        raise SystemExit("нет подходящих кадров в индексе — нечего мерить")
    workers = args.workers or junk.resolve_vlm_workers(cfg.raw)
    model_name = str(getattr(cfg.naming, "classify_vlm_model", junk.DEFAULT_VLM_MODEL))
    print(f"выборка: {len(paths)} кадров — {origin}")
    print(f"модель: {model_name}, потоков подготовки: {workers}")

    results = measure(model_name, paths, args.modes, workers)
    print()
    print(format_table(results))
    report, ok = format_verdicts(results)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
