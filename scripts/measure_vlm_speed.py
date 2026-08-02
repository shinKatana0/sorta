"""Price the two remaining levers of the deep junk tier — and prove they moved no verdict.

The deep tier is worth keeping: on the live run of 2026-07-28 it changed 2 592 of
24 196 verdicts (10.7%), 2 202 of them into `product`, a class the fast tier never
produces. F101 removed the CPU half of the pass (the fast image processor plus a pool of
preparation threads overlapping the halves) and left it at 782 ms and 1.20 frames/s per
frame. F102 priced the input resolution and closed it with outcome B: 672px is x1.48 but
loses 7.5% of the documents, so the default stayed at 896.

F105 measures what is left, and both levers are THE SAME MATHEMATICS DONE DIFFERENTLY:

    --attn      the attention kernel. Nobody had looked at what transformers chose:
                the language half is dispatched to `sdpa`, the visual tower to `eager` —
                and the tower is the half that dominates, because at 896px a frame is
                over a thousand visual tokens. Another kernel of the same attention.
    --batch N   N frames through ONE generate() instead of N. The same frames, counted
                together.

Neither changes what the model is asked, so the bar is stricter than F102's, not looser:
the verdicts must match EXACTLY. A moved label here is a bug (a batch padded on the
wrong side, answers read off in the wrong order), not the price of speed — and this
script exits non-zero when it sees one.

Both levers are OFF in the product and are turned on here by arguments only. What
happens to the default is decided by the numbers, per the pre-registered criteria below.

Privacy: nothing here identifies a frame. No path, no file id and no basename is
printed — mismatches are reported as counts per label pair, which is all that is needed
to go and look, and never a list of where the documents are (the same rule
measure_ocr_gate.py, measure_streetclip.py and measure_vlm_resolution.py follow).

F144 reopens the batch half of that, and only the batch half — see the section marked
F144 below for why a closed verdict is worth reopening and what `--per-call` measures
instead. It shares this file because it is the same question asked again under a
condition that changed; it shares nothing else, and it writes no product code.

Usage (from the repo root, with a GPU venv — `uv sync --extra gpu --extra vlm`):
    python scripts/measure_vlm_speed.py                       # 300 frames, default+sdpa x 1/2/4/8
    python scripts/measure_vlm_speed.py --attn default sdpa --batch 1 4 8 16
    python scripts/measure_vlm_speed.py --attn default vision-sdpa --batch 1
    python scripts/measure_vlm_speed.py --per-call            # F144: 1/2/4/8 images per call
    python scripts/measure_vlm_speed.py --per-call --batch 1 8 --sample 120
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sqlite3
import statistics
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import imaging, junk, naming  # noqa: E402
from sorta.config import load_config  # noqa: E402

# The prompt, the answer parsing, the conservative label of a frame that would not
# decode and the token budget belong to the STAGE, and the measurement has to use the
# stage's own, private or not: a table that prices a slightly different prompt prices
# something nobody runs (the lesson measure_ocr_gate.py starts with).
_PROMPT = junk._VLM_PROMPT
_MAX_NEW_TOKENS = junk._VLM_MAX_NEW_TOKENS
_FALLBACK_LABEL = junk._VLM_FALLBACK_LABEL

# --- The modes ---------------------------------------------------------------
#
# `default` is the baseline and comes first everywhere: it passes NO attention
# implementation, which is the load that ships. `eager` and `sdpa` name one kernel for
# the whole model; `vision-sdpa` moves the visual tower alone and leaves the language
# half to whatever transformers picks — the targeted form of the lever, and the reason
# naming.attn_implementation understands the per-sub-config dict at all.
ATTN_SPECS: dict[str, naming.AttnImplementation] = {
    "default": None,
    "eager": naming.attn_implementation("eager"),
    "sdpa": naming.attn_implementation("sdpa"),
    "vision-sdpa": naming.attn_implementation(vision="sdpa"),
}
DEFAULT_ATTN = ("default", "sdpa")
# F144 added the 2: the one hint that reopened the question was measured at two images
# (see the F144 section), so the grid has to contain the size it was measured at.
DEFAULT_BATCHES = (1, 2, 4, 8)

# How often the GPU-load sampler asks the driver. Fine enough to catch the gaps
# between generate() calls (~0.6 s wide in the baseline), cheap enough to ignore.
GPU_POLL_SEC = 0.1

# --- Pre-registered acceptance criteria (F105) -------------------------------
#
# 1. The verdicts match COMPLETELY on at least 300 frames. Not "98%" as in F102: there
#    the input changed, here it does not, so a disagreement of any size is a defect to
#    go and find. (In fp16 different kernels do give slightly different logits; on a
#    four-class label under greedy decoding that must not be visible.)
# 2. A speedup of at least x1.15. Less is not worth the complication: batching adds
#    padding, masks and an order, i.e. three new ways to spoil a verdict quietly.
# 3. A peak of at most 16 GB of VRAM. The pass sits at 7 890 MB of 24 463 today, and a
#    batch multiplies the activations; on the user's machine CLIP may be resident at the
#    same moment. A mode that needs more is unusable whatever its speed.
MIN_SAMPLE = 300
MIN_SPEEDUP = 1.15
MAX_VRAM_MB = 16 * 1024


@dataclass(frozen=True)
class ModeSpec:
    """One row of the table: an attention implementation and a batch size."""
    attn: str
    batch: int

    @property
    def name(self) -> str:
        return f"{self.attn}/b{self.batch}"


def requested_kernels(attn: str) -> dict[str, str]:
    """What `--attn X` asks of each half; {} — nothing is asked and nothing is passed."""
    spec = ATTN_SPECS[attn]
    if spec is None:
        return {}
    if isinstance(spec, str):
        return {"language": spec, "vision": spec}
    asked = {}
    if "" in spec:
        asked["language"] = spec[""]
    if naming.VISION_SUBCONFIG in spec:
        asked["vision"] = spec[naming.VISION_SUBCONFIG]
    return asked


def unmet_request(attn: str, kernels: dict[str, str]) -> dict[str, tuple[str, str]]:
    """{half: (asked for, got)} for the halves that did NOT get what was requested.

    transformers downgrades a kernel it cannot provide without saying much, and the
    whole point of the run is which kernel ran: a table that prices `sdpa` while `eager`
    was executing is worse than no table at all.
    """
    return {half: (want, kernels.get(half, "?"))
            for half, want in requested_kernels(attn).items()
            if kernels.get(half) != want}


def mode_specs(attns: Sequence[str], batches: Sequence[int]) -> list[ModeSpec]:
    """The grid, ATTENTION-MAJOR — the order the modes are measured in.

    Attention-major because the kernel is baked into the layers when the model is built:
    every batch size of one implementation runs on one load of the weights, and only a
    change of implementation costs a reload (20.5 GB — the previous copy has to go
    first).
    """
    return [ModeSpec(attn=attn, batch=batch) for attn in attns for batch in batches]


@dataclass(frozen=True)
class ModeResult:
    """What one pass over the sample cost, and what it decided."""
    name: str
    workers: int
    labels: tuple[str, ...]
    frame_ms: tuple[float, ...]   # wall time per frame (a batch shares its time evenly)
    wall_sec: float
    cpu_cores: float              # process CPU seconds per wall second
    gpu_util_pct: float | None    # None — no CUDA / the driver did not answer
    peak_vram_mb: float | None
    attn: str = "default"         # what was ASKED for
    kernels: str = ""             # ...and what the loaded model really uses
    batch: int = 1

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


def spread_over_batch(frame_ms: list[float], batch: int) -> list[float]:
    """Give every frame of a batch its share of the batch's time.

    Timing between consecutive labels is exact while frames are answered one by one. A
    batch answers all of its frames at the same instant, so the raw timings would read
    "one frame took 3 s, the next three took 0 ms" — a median of zero and a meaningless
    p90. The batch is the unit that has a duration; the frame's cost is that duration
    divided by the frames in it.
    """
    if batch < 2:
        return frame_ms
    out: list[float] = []
    for start in range(0, len(frame_ms), batch):
        chunk = frame_ms[start:start + batch]
        out.extend([statistics.fmean(chunk)] * len(chunk))
    return out


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

    Reserved, not allocated: the allocator's blocks are what actually occupy the card.
    Printed for EVERY mode, because a batch multiplies the activations — this is the
    number that decides whether a fast mode is usable at all (MAX_VRAM_MB).
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


def _free_vram() -> None:  # pragma: no cover — needs CUDA
    """Give the card back before the next attention implementation is loaded."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — nothing to free is not an error
        pass


# --- The batched pass --------------------------------------------------------

@dataclass
class PreparedChunk:
    """`batch` frames decoded and preprocessed TOGETHER, ready for one generate().

    `groups` holds only the frames that decoded, `kept` says where each of them sat in
    the chunk, and `size` is the whole chunk — a frame that vanished or would not decode
    never reaches the model and takes the tier's conservative label, exactly as in
    junk.vlm_classifier_from. `prepared` is None when the processor refused the batch;
    the consumer then gives every frame its own chance instead of losing all of them.
    """
    size: int
    kept: list[int] = field(default_factory=list)
    groups: list[list[Image.Image]] = field(default_factory=list)
    prepared: Any = None


def decode_frame(path: str, max_edge: int) -> Image.Image | None:
    """One frame for the model — the decode junk.vlm_classifier_from does per frame.

    Through the shared preview cache (Unicode/HEIC-safe, F38/F67), downscaled to
    max_edge. None means the tier answers without asking the model.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=max_edge)


def prepare_chunk(runtime: naming.BatchVlm, paths: Sequence[str],
                  max_edge: int) -> PreparedChunk:
    """The CPU half of one batch: decode every frame, then ONE processor call.

    Runs on a worker thread (see `batched_labels`), so it must not raise: a chunk whose
    preparation failed is handed on with `prepared=None` and retried frame by frame.
    """
    chunk = PreparedChunk(size=len(paths))
    for i, path in enumerate(paths):
        image = decode_frame(path, max_edge)
        if image is not None:
            chunk.kept.append(i)
            chunk.groups.append([image])
    if chunk.groups:
        try:
            chunk.prepared = runtime.prepare_batch(chunk.groups, _PROMPT)
        except Exception:  # noqa: BLE001 — one bad frame must not cost the batch
            chunk.prepared = None
    return chunk


def chunk_labels(runtime: naming.BatchVlm,
                 chunk: PreparedChunk) -> list[str | BaseException]:
    """The GPU half of one batch: labels for the whole chunk, in the chunk's order.

    The answers are scattered back by position, never by arrival: an answer given to the
    wrong file is worse than a slow pass. Anything that goes wrong with the batch — a
    failed preparation, an out-of-memory generate, a model that answered a different
    number of times than it was asked — falls through to naming.batched_describe, which
    gives each frame its own call rather than guessing at the alignment.
    """
    out: list[str | BaseException] = [_FALLBACK_LABEL] * chunk.size
    if not chunk.groups:
        return out
    answers: list[str | BaseException] | None = None
    if chunk.prepared is not None:
        try:
            batched = list(runtime.generate_batch(chunk.prepared, _MAX_NEW_TOKENS))
            if len(batched) == len(chunk.groups):
                answers = list(batched)
        except Exception:  # noqa: BLE001 — retried below, one frame at a time
            pass
    if answers is None:
        answers = naming.batched_describe(runtime, chunk.groups, _PROMPT,
                                          _MAX_NEW_TOKENS)
    for position, answer in zip(chunk.kept, answers):
        out[position] = (answer if isinstance(answer, BaseException)
                         else junk._vlm_label(answer))
    return out


def batched_labels(runtime: naming.BatchVlm, paths: list[str], batch: int,
                   workers: int, max_edge: int) -> Iterator[str | BaseException]:
    """Labels for `paths` IN INPUT ORDER, `batch` frames per generate() (F105).

    The shape of junk._vlm_labels_pipelined one level up: a FIFO of futures, so the
    chunk at the head is the next one yielded no matter how the preparations interleave,
    with the workers decoding and preprocessing the chunks that follow while this thread
    runs the model on the one that is ready. The GPU half stays on ONE thread — several
    would only queue up inside the driver and cost VRAM, which is the resource a batch
    is already spending.

    The window is one chunk per worker (at least two): that many prepared batches exist
    at once, and they are CPU tensors — naming.qwen_runtime keeps them off the card — so
    the VRAM peak stays the peak of ONE generate, which is what the table reports.
    """
    chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    remaining = iter(chunks)
    window = max(2, workers)
    with ThreadPoolExecutor(max_workers=max(1, workers),
                            thread_name_prefix="sorta-vlm") as pool:
        pending: deque[tuple[list[str], Future[PreparedChunk]]] = deque()

        def fill() -> None:
            """Top the preparation queue up (this thread only — it owns the iterator)."""
            while len(pending) < window:
                chunk = next(remaining, None)
                if chunk is None:
                    return
                pending.append((chunk, pool.submit(prepare_chunk, runtime, chunk,
                                                   max_edge)))

        fill()
        while pending:
            paths_of_chunk, future = pending.popleft()
            labels: list[str | BaseException]
            try:
                labels = chunk_labels(runtime, future.result())
            except Exception as exc:  # noqa: BLE001 — one bad chunk must not end the pass
                labels = [exc] * len(paths_of_chunk)
            fill()  # refill BEFORE yielding: the workers keep going while the caller writes
            yield from labels


def mode_items(runtime: naming.BatchVlm, paths: list[str], spec: ModeSpec,
               workers: int, max_edge: int) -> Iterable[str | BaseException]:
    """The label stream of one mode — the stage's own pass unless a batch was asked for.

    Batch 1 goes through junk._vlm_labels with the stage's own classifier, i.e. the code
    that runs in production today: the baseline row of the table has to be the thing
    being compared against, not a re-implementation of it.
    """
    if spec.batch < 2:
        classifier = junk.vlm_classifier_from(runtime, max_edge=max_edge)
        return junk._vlm_labels(classifier, paths, workers)
    return batched_labels(runtime, paths, spec.batch, workers, max_edge)


def run_mode(spec: ModeSpec, items: Iterable[str | BaseException], workers: int,
             kernels: str = "") -> ModeResult:
    """One pass over the sample: time it, watch the card, keep every label."""
    _vram_peak_mb(reset=True)
    frame_ms: list[float] = []
    labels: list[str] = []
    with GpuSampler() as gpu:
        cpu0, wall0 = time.process_time(), time.perf_counter()
        previous = wall0
        for item in items:
            now = time.perf_counter()
            frame_ms.append((now - previous) * 1000.0)
            previous = now
            labels.append("ERROR" if isinstance(item, BaseException) else item)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
    return ModeResult(
        name=spec.name, workers=workers, attn=spec.attn, kernels=kernels,
        batch=spec.batch, labels=tuple(labels),
        frame_ms=tuple(spread_over_batch(frame_ms, spec.batch)), wall_sec=wall,
        cpu_cores=cpu / wall if wall else 0.0,
        gpu_util_pct=gpu.mean_pct, peak_vram_mb=_vram_peak_mb(),
    )


# --- The report --------------------------------------------------------------

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


def speedup(base: ModeResult, other: ModeResult) -> float:
    """How many times faster `other` is than the baseline (0.0 — nothing to divide by)."""
    return other.frames_per_sec / base.frames_per_sec if base.frames_per_sec else 0.0


@dataclass(frozen=True)
class Assessment:
    """One mode measured against the pre-registered criteria."""
    spec: ModeSpec
    speedup: float
    mismatches: int
    peak_vram_mb: float | None
    sample: int

    @property
    def verdicts_match(self) -> bool:
        return self.mismatches == 0

    @property
    def fast_enough(self) -> bool:
        return self.speedup >= MIN_SPEEDUP

    @property
    def fits_in_vram(self) -> bool:
        return self.peak_vram_mb is None or self.peak_vram_mb <= MAX_VRAM_MB

    @property
    def sample_enough(self) -> bool:
        return self.sample >= MIN_SAMPLE

    @property
    def accepted(self) -> bool:
        return (self.verdicts_match and self.fast_enough and self.fits_in_vram
                and self.sample_enough)


def assess(base: ModeResult, other: ModeResult) -> Assessment:
    return Assessment(
        spec=ModeSpec(attn=other.attn, batch=other.batch),
        speedup=speedup(base, other),
        mismatches=sum(label_mismatches(base, other).values()),
        peak_vram_mb=other.peak_vram_mb, sample=len(base.labels))


def outcome(results: list[ModeResult]) -> tuple[str, str]:
    """(A | B | C, one line saying why) — the verdict of the brief, decided by the data.

    A — a batched mode passed both criteria: the orchestrator changes the default in a
        commit of its own (the config section belongs to another feature right now).
    B — only the attention kernel passed: take `sdpa` and leave the batch alone.
    C — nothing passed: the question of speeding the VLM up is closed with numbers, and
        is not reopened without a new idea.
    """
    base = results[0]
    if len(results) < 2:
        return "C", "сравнивать не с чем — запрошен один режим"
    checks = [assess(base, r) for r in results[1:]]
    accepted = [c for c in checks if c.accepted]
    batched = [c for c in accepted if c.spec.batch > 1]
    kernel_only = [c for c in accepted if c.spec.batch == 1]
    if batched:
        pick = max(batched, key=lambda c: c.speedup)
        return "A", (f"режим {pick.spec.name} взял оба критерия "
                     f"(x{pick.speedup:.2f}, вердикты совпали полностью, пик "
                     f"{pick.peak_vram_mb or 0:.0f} МБ): батчинг и ядро внимания "
                     f"переносим в дефолт отдельным коммитом")
    if kernel_only:
        pick = max(kernel_only, key=lambda c: c.speedup)
        return "B", (f"ядро внимания прошло ({pick.spec.name}, x{pick.speedup:.2f}, "
                     f"вердикты совпали полностью), батчинг — нет: берём только attn")
    best = max(checks, key=lambda c: c.speedup)
    return "C", (f"ни один режим не прошёл (лучшее — {best.spec.name}, "
                 f"x{best.speedup:.2f} при пороге x{MIN_SPEEDUP:.2f}, расхождений "
                 f"{best.mismatches}): тему ускорения VLM закрываем с цифрами")


def format_table(results: list[ModeResult]) -> str:
    """The speed table: one row per mode, the speedup measured against the first."""
    base = results[0]
    out = [
        "=" * 108,
        f"VLM-ПРОХОД: {len(base.labels)} кадров, база — режим '{base.name}'",
        f"{'режим':>16} {'ядра внимания':>20} {'потоков':>8} {'медиана':>9} {'p90':>9} "
        f"{'кадр/с':>8} {'ускорение':>10} {'GPU':>6} {'ядер':>6} {'пик VRAM':>10}",
    ]
    for r in results:
        gain = f"x{speedup(base, r):.2f}" if base.frames_per_sec else "—"
        gpu = f"{r.gpu_util_pct:.0f}%" if r.gpu_util_pct is not None else "—"
        vram = f"{r.peak_vram_mb:.0f} МБ" if r.peak_vram_mb is not None else "—"
        if r.peak_vram_mb is not None and r.peak_vram_mb > MAX_VRAM_MB:
            vram += " !"
        out.append(
            f"{r.name:>16} {r.kernels or '—':>20} {r.workers:>8d} "
            f"{r.median_ms:>8.0f}м {r.p90_ms:>8.0f}м {r.frames_per_sec:>8.2f} "
            f"{gain:>10} {gpu:>6} {r.cpu_cores:>6.2f} {vram:>10}"
        )
    out.append("=" * 108)
    return "\n".join(out)


def format_verdicts(results: list[ModeResult]) -> tuple[str, bool]:
    """(the verdict-comparison block, everything matched?) — the acceptance criterion.

    Nothing about the numbers above counts if this says no: both levers are the same
    arithmetic done differently, so a label that moved is a defect in the plumbing —
    left padding, a hand-built mask, an answer read off the wrong position.
    """
    base = results[0]
    lines = [f"ВЕРДИКТЫ (сверка с '{base.name}', кадров {len(base.labels)}):"]
    ok = True
    for r in results[1:]:
        diff = label_mismatches(base, r)
        if not diff:
            lines.append(f"  {r.name:>16}: совпадение полное")
            continue
        ok = False
        total = sum(diff.values())
        lines.append(f"  {r.name:>16}: РАСХОЖДЕНИЙ {total} — СТОП, разбираться")
        for (was, now), count in sorted(diff.items(), key=lambda kv: -kv[1]):
            lines.append(f"{'':>20}{was} -> {now}: {count}")
    if len(results) == 1:
        lines.append("  (сравнивать не с чем — запрошен один режим)")
    return "\n".join(lines), ok


def format_criteria(results: list[ModeResult]) -> str:
    """Every mode against every pre-registered threshold, in one place."""
    base = results[0]
    lines = [f"КРИТЕРИИ (вердикты в точности, ускорение >= x{MIN_SPEEDUP:.2f}, "
             f"пик VRAM <= {MAX_VRAM_MB} МБ, выборка >= {MIN_SAMPLE}):"]
    for r in results[1:]:
        check = assess(base, r)
        vram = (f"{check.peak_vram_mb:.0f} МБ" if check.peak_vram_mb is not None
                else "—")
        lines.append(
            f"  {r.name:>16}: {'ПРИНЯТ' if check.accepted else 'отклонён':>8} — "
            f"x{check.speedup:.2f}{'' if check.fast_enough else ' (медленно)'}, "
            f"расхождений {check.mismatches}, VRAM {vram}"
            f"{'' if check.fits_in_vram else ' (не влезает)'}"
            f"{'' if check.sample_enough else ', выборка мала'}")
    return "\n".join(lines)


def format_outcome(results: list[ModeResult]) -> str:
    letter, why = outcome(results)
    return f"ИСХОД {letter}: {why}"


# --- F144: what a call costs by the number of images in it -------------------
#
# Batching was rejected once already and the verdict is in the CHANGELOG: the bottleneck
# was the PROCESSOR — ~0.6 s of preparation against ~0.19 s on the card, 0.84 cores of
# 24 busy, the card at 26% — and a starving GPU has no use for bigger portions.
#
# THE CONDITION THAT VERDICT WAS MEASURED UNDER WAS REMOVED BY THE SAME FEATURE. F105
# fixed exactly that cause: the processor is built with `use_fast=True` (the slow one was
# pure Python over PIL and ate most of the 0.6 s) and the runtime was split into its two
# halves so that `vlm.workers` (`naming.vlm_workers` when the verdict was written) drives
# the preparation apart from the inference. A verdict whose premise is gone is not a
# verdict any more, only a memory of one.
#
# A second hint arrived from somewhere else entirely: F132 timed the keeper question at
# 1.32 s with TWO images against 0.78 s with one — 0.66 s per image, 15% cheaper. That is
# NOT a controlled comparison (another prompt, another answer length, another code path),
# which is precisely why this is a measurement and not a conclusion.
#
# So: one question, one set of frames, and the only thing that moves is how many images
# go through a SINGLE generate() — 1, 2, 4, 8. Every image carries the same question in
# its own sequence, so nothing about what is asked depends on the size of the call; only
# how many answers come back out of it.
#
# The question below is this measurement's OWN, and that reverses the rule the F105 half
# of this file opens with (`_PROMPT = junk._VLM_PROMPT` — price the stage's own prompt or
# price something nobody runs). On purpose: F105 priced a STAGE and had to compare its
# verdicts, this prices a CALL and compares seconds. A question of the same shape — one
# image in, one short word back — times the same work without turning the table into a
# statement about what the product decides, which is what would happen if the stage's
# prompt were re-timed here while another feature is free to reword it.
#
# Nothing here writes anywhere: the index is opened read-only by `sample_paths`, the
# answers are counted and dropped, and no config key, stage or prompt of the product is
# touched by this file.

_CALL_PROMPT = (
    "Look at this image and answer with exactly one word: indoor, outdoor, or unclear.\n"
    "indoor = the picture was taken inside a building.\n"
    "outdoor = the picture was taken outside.\n"
    "unclear = too little of the scene is visible to tell.\n"
    "Answer with exactly one word: indoor, outdoor, or unclear."
)
# One short word back, the same budget the deep tier gives its own label: a longer one
# would time the model writing prose rather than the call.
_CALL_MAX_NEW_TOKENS = 8
_CALL_ANSWERS: tuple[str, ...] = ("indoor", "outdoor", "unclear")

# How much of the answered share a bigger call may lose before its speed stops counting.
# One point rather than zero: the baseline's own share is a measurement too, and a single
# frame out of 300 is 0.33% — a threshold of exactly zero would turn noise into a verdict.
PARSE_TOLERANCE = 0.01


def parse_call_answer(answer: str) -> str | None:
    """The measurement's own answer -> one of its words, or None when nothing read.

    Lenient in the way every answer in this project is read (the F96 lesson: asked for a
    strict format the model answers in prose anyway) — the word is looked for anywhere in
    the line. None is deliberately NOT turned into a fallback label the way the deep tier
    turns it: an unreadable answer is the thing being counted here, and a fallback is how
    a batch that started producing rubbish would come out of the table looking fast.
    """
    lowered = (answer or "").lower()
    for word in _CALL_ANSWERS:
        if word in lowered:
            return word
    return None


@dataclass(frozen=True)
class CallStats:
    """What ONE call size cost: an entry per generate(), plus what the machine did.

    `images` is what really reached the model. A frame that would not decode is skipped,
    so a call can be narrower than the size it was asked for — and seconds per image has
    to divide by what was actually shown, or a pass full of unreadable files would look
    like a bargain.
    """
    batch: int
    seconds: tuple[float, ...] = ()   # one per call, in call order
    images: tuple[int, ...] = ()      # images in that call
    asked: int = 0                    # answers expected — the images that went in
    parsed: int = 0                   # ...of them, the answers that read as one word
    failed: int = 0                   # calls that raised or came back the wrong length
    skipped: int = 0                  # frames that would not decode, never called about
    cpu_cores: float = 0.0            # process CPU seconds per wall second
    gpu_util_pct: float | None = None
    peak_vram_mb: float | None = None

    @property
    def calls(self) -> int:
        return len(self.seconds)

    @property
    def total_images(self) -> int:
        return sum(self.images)

    @property
    def total_sec(self) -> float:
        return sum(self.seconds)

    @property
    def median_sec(self) -> float:
        return statistics.median(self.seconds) if self.seconds else 0.0

    @property
    def mean_sec(self) -> float:
        return statistics.fmean(self.seconds) if self.seconds else 0.0

    @property
    def min_sec(self) -> float:
        return min(self.seconds) if self.seconds else 0.0

    @property
    def max_sec(self) -> float:
        return max(self.seconds) if self.seconds else 0.0

    @property
    def sec_per_image(self) -> float:
        """The answer of the whole measurement: the seconds spent over the images shown."""
        return self.total_sec / self.total_images if self.total_images else 0.0

    @property
    def parsed_share(self) -> float:
        return self.parsed / self.asked if self.asked else 0.0


def call_groups(paths: Sequence[str],
                max_edge: int) -> tuple[list[list[Image.Image]], int]:
    """The frames of one call, each in a group of its own; and how many would not decode.

    One image per group is what keeps the sizes comparable: every image carries the same
    question whether the call holds one of them or eight, so the only variable left is
    how many answers a single generate() produces. A frame that did not decode is DROPPED
    rather than replaced by a conservative label — it has no answer to give, and a call
    about a frame that is not there measures nothing — and it is counted, so the report
    can say how many the pass lost.
    """
    groups: list[list[Image.Image]] = []
    skipped = 0
    for path in paths:
        image = decode_frame(path, max_edge)
        if image is None:
            skipped += 1
            continue
        groups.append([image])
    return groups, skipped


def one_call(runtime: naming.BatchVlm, groups: list[list[Image.Image]], prompt: str,
             clock: Callable[[], float] = time.perf_counter,
             ) -> tuple[float, list[str] | None]:
    """(seconds, answers) for ONE generate over `groups`; None — the call did not answer.

    Deliberately NOT retried frame by frame the way the F105 pass retries a batch. There
    the point was to lose no verdict; here the point is what a call of this size costs,
    and a silent retry would time N + 1 calls and print the total as one. A call that
    raises, or that comes back a different length than it was asked (nothing can then be
    said about which answer belongs to which image), is reported as a failure with its
    seconds intact — a batch size that dies is a result of the measurement, not a stop.
    """
    started = clock()
    try:
        prepared = runtime.prepare_batch(groups, prompt)
        answers = [str(a) for a in runtime.generate_batch(prepared,
                                                          _CALL_MAX_NEW_TOKENS)]
    except Exception:  # noqa: BLE001 — a batch that dies is a row of the table
        return clock() - started, None
    elapsed = clock() - started
    return elapsed, answers if len(answers) == len(groups) else None


def measure_calls(runtime: naming.BatchVlm, paths: Sequence[str], batch: int,
                  max_edge: int, prompt: str = _CALL_PROMPT,
                  clock: Callable[[], float] = time.perf_counter) -> CallStats:
    """One pass over `paths` with `batch` images per generate() — timed and counted.

    The decode happens on this thread, right before the call it belongs to, and is NOT
    inside the timed span: the preparation is a pipeline in the product and would only
    add the noise of a cold preview cache to a number about the card. The CPU and GPU
    load, on the other hand, are read over the WHOLE pass, decode included — the verdict
    being reopened was made of exactly those two numbers, so they have to cover the same
    work they covered then.
    """
    size = max(1, batch)
    seconds: list[float] = []
    images: list[int] = []
    asked = parsed = failed = skipped = 0
    _vram_peak_mb(reset=True)
    with GpuSampler() as gpu:
        cpu0, wall0 = time.process_time(), clock()
        for start in range(0, len(paths), size):
            groups, lost = call_groups(paths[start:start + size], max_edge)
            skipped += lost
            if not groups:
                continue
            elapsed, answers = one_call(runtime, groups, prompt, clock)
            seconds.append(elapsed)
            images.append(len(groups))
            asked += len(groups)
            if answers is None:
                failed += 1
                continue
            parsed += sum(1 for answer in answers
                          if parse_call_answer(answer) is not None)
        wall = clock() - wall0
        cpu = time.process_time() - cpu0
    return CallStats(
        batch=batch, seconds=tuple(seconds), images=tuple(images), asked=asked,
        parsed=parsed, failed=failed, skipped=skipped,
        cpu_cores=cpu / wall if wall else 0.0, gpu_util_pct=gpu.mean_pct,
        peak_vram_mb=_vram_peak_mb())


def call_speedup(base: CallStats, other: CallStats) -> float:
    """How many times cheaper ONE IMAGE is in `other` than in `base` (0.0 — no rate)."""
    if not base.sec_per_image or not other.sec_per_image:
        return 0.0
    return base.sec_per_image / other.sec_per_image


def format_call_table(stats: list[CallStats]) -> str:
    """The table the brief asks for: call size -> s/call -> s/image -> answers read."""
    base = stats[0]
    frames = base.total_images + base.skipped
    out = [
        "=" * 112,
        f"ЦЕНА ВЫЗОВА ПО ЧИСЛУ ИЗОБРАЖЕНИЙ: {frames} кадров, вопрос замера "
        f"(не промпт стадии), база — {base.batch} на вызов",
        f"{'картинок':>9} {'вызовов':>8} {'медиана':>9} {'среднее':>9} {'мин':>8} "
        f"{'макс':>8} {'с/изобр':>9} {'ускорение':>10} {'разобрано':>12} {'GPU':>6} "
        f"{'ядер':>6} {'пик VRAM':>10}",
    ]
    for s in stats:
        gain = f"x{call_speedup(base, s):.2f}" if call_speedup(base, s) else "—"
        gpu = f"{s.gpu_util_pct:.0f}%" if s.gpu_util_pct is not None else "—"
        vram = f"{s.peak_vram_mb:.0f} МБ" if s.peak_vram_mb is not None else "—"
        read = f"{s.parsed}/{s.asked}"
        out.append(
            f"{s.batch:>9d} {s.calls:>8d} {s.median_sec:>8.2f}с {s.mean_sec:>8.2f}с "
            f"{s.min_sec:>7.2f}с {s.max_sec:>7.2f}с {s.sec_per_image:>9.3f} "
            f"{gain:>10} {read:>12} {gpu:>6} {s.cpu_cores:>6.2f} {vram:>10}")
    out.append("=" * 112)
    lost = sum(s.skipped for s in stats)
    failed = sum(s.failed for s in stats)
    if lost:
        out.append(f"кадров не декодировалось (пропущены, модель их не видела): {lost}")
    if failed:
        out.append(f"вызовов не ответило (упали или вернули не столько ответов, "
                   f"сколько спросили): {failed}")
    return "\n".join(out)


def call_outcome(stats: list[CallStats]) -> tuple[bool, str]:
    """(does a bigger call pay?, one line saying why) — what the verdict is written from.

    Two conditions, both straight from the brief. Cheaper per image by at least
    MIN_SPEEDUP: below that the complication buys nothing worth the padding, the masks
    and the order it adds. And no worse at ANSWERING — a call that starts producing
    unreadable answers when it grows is not a speedup, it is the same work done twice,
    and that is the whole reason the share is counted next to the seconds.
    """
    base = stats[0]
    bigger = [s for s in stats[1:] if s.batch > base.batch]
    if not bigger:
        return False, "сравнивать не с чем — запрошен один размер вызова"
    passed = [s for s in bigger
              if call_speedup(base, s) >= MIN_SPEEDUP
              and s.parsed_share >= base.parsed_share - PARSE_TOLERANCE]
    if passed:
        pick = max(passed, key=lambda s: call_speedup(base, s))
        return True, (f"выигрыш есть: {pick.batch} изображений в вызове — "
                      f"x{call_speedup(base, pick):.2f} на изображение "
                      f"({pick.sec_per_image:.3f} с против {base.sec_per_image:.3f} с), "
                      f"разобрано {pick.parsed}/{pick.asked} против "
                      f"{base.parsed}/{base.asked}; продуктовую фичу можно заводить, "
                      f"зная цифры")
    best = max(bigger, key=lambda s: call_speedup(base, s))
    return False, (f"выигрыша нет: лучший — {best.batch} изображений в вызове, "
                   f"x{call_speedup(base, best):.2f} при пороге x{MIN_SPEEDUP:.2f}, "
                   f"разобрано {best.parsed}/{best.asked} против "
                   f"{base.parsed}/{base.asked}; вердикт F105 подтверждён на новых "
                   f"условиях")


def format_call_outcome(stats: list[CallStats]) -> str:
    _win, why = call_outcome(stats)
    return f"ИТОГ: {why}"


# --- Wiring ------------------------------------------------------------------

def measure(model_name: str, paths: list[str], specs: list[ModeSpec], workers: int,
            max_edge: int) -> list[ModeResult]:  # pragma: no cover — ML
    """Every requested mode, one load of the weights per attention implementation.

    The kernel is chosen when the layers are BUILT, so it cannot be switched on a loaded
    model — every batch size of one implementation therefore shares a load, and a change
    of implementation frees the previous copy first (20.5 GB of 24.4: two do not fit).
    What the model actually dispatches to is read off the loaded config and printed: a
    request for a kernel that is unavailable is downgraded quietly, and a table that
    prices `sdpa` while `eager` ran would be worse than no table.
    """
    results: list[ModeResult] = []
    if any(spec.batch > 1 for spec in specs):
        # The processor that pads a batch is the preparing thread's own, so the side is
        # verified on one built exactly like it — the argument may be ignored silently,
        # and a right-padded batch answers plausibly and WRONG.
        probe = naming.pad_generation_left(
            naming.qwen_processor(model_name, use_fast=True))
        if not naming.processor_pads_left(probe):
            print("ВНИМАНИЕ: паддинг этого процессора не встаёт налево — батчевые "
                  "режимы недостоверны, их вердикты сравнивать нельзя")
    for attn in dict.fromkeys(spec.attn for spec in specs):
        model, processor, device = naming.load_qwen(model_name, use_fast=True,
                                                    attn=ATTN_SPECS[attn])
        kernels = naming.attention_kernels(model)
        line = f"{kernels['language']}/{kernels['vision']}"
        print(f"загружено с attn={attn}: язык {kernels['language']}, "
              f"визуальная башня {kernels['vision']}")
        for half, (asked, got) in unmet_request(attn, kernels).items():
            print(f"ВНИМАНИЕ: просили {asked} для половины '{half}', а работает {got} — "
                  f"строки этого режима измеряют не то, что написано в их названии")
        runtime = naming.qwen_runtime(
            model, processor, device,
            lambda: naming.qwen_processor(model_name, use_fast=True))
        if not naming.processor_is_fast(processor):
            print("ВНИМАНИЕ: transformers не отдал быстрый процессор для этой модели")
        for spec in [s for s in specs if s.attn == attn]:
            print(f"режим '{spec.name}': {len(paths)} кадров, потоков {workers}, "
                  f"батч {spec.batch}...")
            results.append(run_mode(
                spec, mode_items(runtime, paths, spec, workers, max_edge), workers,
                kernels=line))
        del runtime, model, processor
        _free_vram()
    return results


def measure_per_call(model_name: str, paths: list[str], batches: Sequence[int],
                     max_edge: int) -> list[CallStats]:  # pragma: no cover — ML
    """F144: every call size on ONE load of the weights, loaded as the product loads it.

    No attention argument and no pipeline of preparation threads: this measurement moves
    ONE thing, and the model it moves it on has to be the one that ships. The padding
    side is checked before anything is timed, because a right-padded batch does not fail
    — it answers plausibly and wrong, and its share of readable answers would then be a
    lie in the one column that exists to catch exactly that.
    """
    model, processor, device = naming.load_qwen(model_name, use_fast=True)
    runtime = naming.qwen_runtime(
        model, processor, device,
        lambda: naming.qwen_processor(model_name, use_fast=True))
    if not naming.processor_is_fast(processor):
        print("ВНИМАНИЕ: transformers не отдал быстрый процессор для этой модели — "
              "условие, ради которого вердикт переоткрыт, не выполнено")
    probe = naming.pad_generation_left(naming.qwen_processor(model_name, use_fast=True))
    if not naming.processor_pads_left(probe):
        print("ВНИМАНИЕ: паддинг этого процессора не встаёт налево — ответы вызовов "
              "больше чем на одно изображение недостоверны")
    out: list[CallStats] = []
    for batch in batches:
        print(f"{batch} изображений в вызове: {len(paths)} кадров...")
        out.append(measure_calls(runtime, paths, batch, max_edge))
    del runtime, model, processor
    _free_vram()
    return out


def run_per_call(cfg: Any, batches: Sequence[int], paths: list[str],
                 origin: str) -> int:
    """The `--per-call` report end to end; non-zero only when nothing could be measured."""
    sizes = sorted(dict.fromkeys(batches))
    print(f"выборка: {len(paths)} кадров — {origin}")
    print(f"модель: {cfg.vlm.model}, {cfg.vlm.max_edge}px; вопрос — собственный вопрос "
          f"замера, промпты стадий не трогаются")
    print(f"размеры вызова: {', '.join(str(size) for size in sizes)}; "
          f"ядер в системе: {os.cpu_count()}")

    stats = measure_per_call(cfg.vlm.model, paths, sizes, cfg.vlm.max_edge)
    if not stats or not stats[0].calls:
        print("замер не состоялся: ни одного вызова не удалось сделать")
        return 1
    print()
    print(format_call_table(stats))
    print()
    print(format_call_outcome(stats))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", type=int, default=MIN_SAMPLE,
                    help=f"frames per mode (default {MIN_SAMPLE} — the pre-registered "
                         f"minimum)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--attn", nargs="+", default=list(DEFAULT_ATTN),
                    choices=list(ATTN_SPECS),
                    help="attention implementations to compare, the baseline first")
    ap.add_argument("--batch", nargs="+", type=int, default=list(DEFAULT_BATCHES),
                    help="batch sizes to compare, the baseline (1) first")
    ap.add_argument("--workers", type=int,
                    help="preparation threads (default: vlm.workers)")
    ap.add_argument("--per-call", action="store_true",
                    help="F144: price ONE generate() by the number of images in it — "
                         "the measurement's own question, no attention lever, no "
                         "verdict comparison (see the F144 section)")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if any(batch < 1 for batch in args.batch):
        raise SystemExit("--batch: размер батча не меньше 1")

    cfg = load_config(args.config)
    paths, origin = sample_paths(str(cfg.database), args.sample, args.seed)
    if not paths:
        raise SystemExit("нет подходящих кадров в индексе — нечего мерить")
    if args.per_call:
        return run_per_call(cfg, args.batch, paths, origin)

    workers = args.workers or cfg.vlm.workers
    specs = mode_specs(args.attn, args.batch)
    print(f"выборка: {len(paths)} кадров — {origin}")
    print(f"модель: {cfg.vlm.model}, {cfg.vlm.max_edge}px, "
          f"потоков подготовки: {workers}")
    print(f"режимов: {len(specs)} — {', '.join(spec.name for spec in specs)}")
    if len(paths) < MIN_SAMPLE:
        print(f"ВНИМАНИЕ: кадров меньше {MIN_SAMPLE} — совпадение вердиктов на такой "
              f"выборке не считается доказанным (исходы A и B недоступны)")

    results = measure(cfg.vlm.model, paths, specs, workers, cfg.vlm.max_edge)
    print()
    print(format_table(results))
    report, ok = format_verdicts(results)
    print(report)
    print(format_criteria(results))
    print()
    print(format_outcome(results))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
