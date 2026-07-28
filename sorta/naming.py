"""F6 (Phase 5): event names.

Contract: reads events/event_files/files, updates ONLY events.name
(and only rows with name_is_manual = 0 — manual names are untouchable).

Providers behind a common EventNamer interface (switching — one line
naming.provider in config.yaml):
- template  — a local template "YYYY-MM-DD <City>" (events.py is not imported here — modules talk only through the DB);
- vlm       — the local Qwen2.5-VL through transformers, in this process, 3–5 sample
  frames of the event in ONE call (F95);
- local_vlm — the ollama HTTP API, 3–5 sample frames of the event;
- claude    — the Anthropic Messages API, key from env; a network call is possible
  ONLY if provider='claude' is explicitly chosen in the config.

F95: the template name ("2025-04-24..05-06 Тайланд") carries exactly the information
the folder path already shows — a year later a trip is looked for by "Пхукет с
детьми", not by a date range. The `vlm` provider adds the CONTENT of the event to
that base and nothing else: places and dates are known exactly from geo/EXIF, and a
model asked for them invents them. The model itself is the one the junk stage
already loads — see the shared runtime below, a second copy of the weights does not
fit in VRAM.

Settings — the typed config.yaml `naming:` section (cfg.naming, referred to further
in the code under the familiar name NamingSettings).
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import threading
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from PIL import Image

from . import imaging
from .config import DEFAULT_VLM_MAX_EDGE, DEFAULT_VLM_MODEL, Config, VlmConfig
from .config import NamingConfig as NamingSettings  # flat phase-5 settings

_MAX_NAME_LEN = 80
# The Anthropic API supports only these image types; HEIC/RAW are skipped
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

_DESCRIBE_PROMPT = (
    "Это несколько фотографий с одного события из семейного фотоархива. "
    "Придумай короткое название события (2-4 слова, по-русски, без дат и без "
    "кавычек), например: Свадьба, Поход в горы, День рождения. "
    "Ответь ТОЛЬКО названием, без пояснений."
)
# 2-4 words fit well inside this; the sanitizer keeps only the first line anyway, so
# the budget exists to let the model finish the phrase, not to allow a story.
_VLM_MAX_NEW_TOKENS = 32


def naming_settings(cfg: Config) -> NamingSettings:
    """Phase-5 settings (an alias for cfg.naming — module signature compatibility)."""
    return cfg.naming


# --- Provider interface -----------------------------------------------------

@dataclass(frozen=True)
class EventContext:
    """Everything the provider knows about an event (without DB access)."""
    started_at: str
    ended_at: str
    city: str | None
    sample_paths: tuple[str, ...] = ()


class EventNamer(Protocol):
    def name(self, ctx: EventContext) -> str | None:
        """Event name or None (keep the current name)."""
        ...  # pragma: no cover — protocol signature


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return None


def _date_base(ctx: EventContext) -> str | None:
    """The date part of the name per the F4 template: YYYY-MM-DD, multi-day — ..MM-DD."""
    start = _parse_date(ctx.started_at)
    if start is None:
        return None
    end = _parse_date(ctx.ended_at)
    if end is None or end == start:
        return start.isoformat()
    if end.year == start.year:
        return f"{start.isoformat()}..{end:%m-%d}"
    return f"{start.isoformat()}..{end.isoformat()}"


def _sanitize(text: str) -> str | None:
    """Model response → a safe piece of a folder name (one line, no quotes)."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.strip().strip("\"'«»").rstrip(".")
    line = re.sub(r'[\\/:*?"<>|]', " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line[:_MAX_NAME_LEN] or None


class TemplateNamer:
    """Template names without ML or network: YYYY-MM-DD <City> (brief F4, item 3)."""

    def name(self, ctx: EventContext) -> str | None:
        base = _date_base(ctx)
        if base is None:
            return None
        return f"{base} {ctx.city}" if ctx.city else base


def _http_post_json(url: str, payload: dict[str, Any],
                    headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"неожиданный ответ {url}: не JSON-объект")
    return result


def _evenly_picked(paths: list[str], max_n: int) -> list[str]:
    """At most max_n frames spread evenly over the event (the first and last included)."""
    if len(paths) <= max_n:
        return paths
    if max_n <= 1:
        return paths[:max_n]
    step = (len(paths) - 1) / (max_n - 1)
    return [paths[round(i * step)] for i in range(max_n)]


def _encode_images(paths: tuple[str, ...], max_n: int) -> list[tuple[str, str]]:
    """Up to max_n evenly picked frames → [(media_type, base64), ...]."""
    usable = _evenly_picked(
        [p for p in paths if Path(p).suffix.lower() in _IMAGE_MEDIA_TYPES], max_n)
    out: list[tuple[str, str]] = []
    for p in usable:
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        out.append((_IMAGE_MEDIA_TYPES[Path(p).suffix.lower()],
                    base64.standard_b64encode(data).decode("ascii")))
    return out


# --- The shared local VLM runtime (F95) -------------------------------------
# Qwen2.5-VL is loaded ONCE per process and shared by every stage that needs it: the
# deep junk tier (junk.qwen_vlm_classifier, F37-B) and the `vlm` naming provider
# below. The loader lives HERE, and not in junk.py where it was born, only because
# of the direction of the imports: junk.py already imports this module
# (naming_settings/utcnow_iso), the reverse edge would be a cycle, and modules do not
# import each other otherwise. What must not happen is two copies of the weights —
# the peak is 20.5 GB of VRAM, a second instance does not fit.
#
# describe(frames, prompt, max_new_tokens) is deliberately the whole interface: the
# prompt, the decode and the parsing of the answer stay with the stage that owns
# them, the runtime only knows how to run the model.
#
# F101: `describe` is now assembled from the two halves it always consisted of —
# `prepare` (the processor: chat template + image preprocessing, pure CPU) and
# `generate` (the model itself, the GPU). Profiling the first live deep run showed the
# pass is not heavy but SEQUENTIAL: ~0.6 s of CPU then ~0.19 s of GPU per frame, with
# no overlap — one core busy out of 24, the card idle three quarters of the time. A
# caller that has many frames to classify (junk) can now run the halves in a pipeline;
# a caller with one call per event (VlmNamer) keeps using `describe` and notices
# nothing. The split changes no tensor and no token: SplitVlm.__call__ is literally
# generate(prepare(...)), which is the body the single function had.
#
# F105 adds the two levers that are the SAME MATHEMATICS done differently — the
# attention kernel (`sdpa` instead of `eager`, another kernel of the same attention) and
# the batch (the same frames counted together). Both are parameters here and both are
# off by default: the product path is the one that shipped until the measurement
# (`scripts/measure_vlm_speed.py`) says otherwise. Because neither changes what the
# model is asked, the bar is that the verdicts match EXACTLY — a moved label is a bug,
# not the price of speed.

VlmDescribeFn = Callable[[Sequence[Image.Image], str, int], str]
# The CPU half: frames + prompt -> whatever the GPU half needs (for Qwen — the
# processor's BatchFeature, deliberately left on the CPU, see qwen_runtime).
VlmPrepareFn = Callable[[Sequence[Image.Image], str], Any]
VlmGenerateFn = Callable[[Any, int], str]
# F105, the same two halves for SEVERAL frame groups at once: one prepared batch in,
# one answer per group out, IN THE ORDER THE GROUPS CAME IN.
VlmPrepareBatchFn = Callable[[Sequence[Sequence[Image.Image]], str], Any]
VlmGenerateBatchFn = Callable[[Any, int], list[str]]

# What transformers takes as `attn_implementation`: one name for the whole model, a
# name per sub-config, or None — "whatever transformers picks", which is what shipped.
AttnImplementation = str | dict[str, str] | None
# The sub-config of the visual tower in Qwen2.5-VL. F105 measured (transformers 4.51.3)
# that the language half is dispatched to `sdpa` and the tower to `eager` — and the
# tower is the half that dominates: at 896px a frame is over a thousand visual tokens.
VISION_SUBCONFIG = "vision_config"

# F102: both of these were defined here, and both describe the shared runtime rather
# than this module — so the values moved to the `vlm:` config section (config.VlmConfig)
# and the historical names stay bound to its defaults for the importers (junk.py, the
# measurement scripts). What is actually in use comes from cfg.vlm, not from here.
VLM_MAX_EDGE = DEFAULT_VLM_MAX_EDGE


@dataclass(frozen=True)
class SplitVlm:
    """A runtime that hands out both halves of `describe`, not only the whole (F101).

    It IS a VlmDescribeFn — calling it runs prepare then generate, exactly as the one
    function did — so every existing caller keeps working unchanged. A caller that
    wants overlap asks for the halves instead (`isinstance(runtime, SplitVlm)`), runs
    `prepare` on worker threads and `generate` on its own: one thread per GPU is the
    point, several threads generating would only serialize inside the driver.

    A runtime that is NOT a SplitVlm (a test double, a future provider) is not an
    error anywhere — the caller falls back to the serial path.
    """
    prepare: VlmPrepareFn
    generate: VlmGenerateFn

    def __call__(self, frames: Sequence[Image.Image], prompt: str,
                 max_new_tokens: int) -> str:
        return self.generate(self.prepare(frames, prompt), max_new_tokens)


@dataclass(frozen=True)
class BatchVlm(SplitVlm):
    """A SplitVlm that can also answer for SEVERAL frame groups in one generate (F105).

    It IS a SplitVlm and IS a VlmDescribeFn — every existing caller (the namer, the deep
    junk tier, an injected test double) sees exactly what it saw before, and the batched
    halves are extra. A caller that wants them asks for the type, as the junk pipeline
    asks for SplitVlm.

    The whole risk of batching lives in the two halves and is answered there: the
    padding for generation has to be on the LEFT, the attention mask has to come from
    the processor, and answer i has to belong to group i. `batched_describe` below is
    the safe way to use them — it keeps the positions straight and never lets one bad
    frame cost the batch.
    """
    prepare_batch: VlmPrepareBatchFn
    generate_batch: VlmGenerateBatchFn


def _one_answer(runtime: VlmDescribeFn, frames: Sequence[Image.Image], prompt: str,
                max_new_tokens: int) -> str | BaseException:
    """One group through the plain (unbatched) path; its exception, if it raised."""
    try:
        return runtime(frames, prompt, max_new_tokens)
    except Exception as exc:  # noqa: BLE001 — one bad frame is the caller's business
        return exc


def _batch_answers(runtime: VlmDescribeFn, groups: list[list[Image.Image]], prompt: str,
                   max_new_tokens: int) -> list[str | BaseException]:
    """Answers for non-empty `groups`, batched when the runtime can and serially if not.

    A batch that fails for ANY reason — a frame the processor chokes on, no VRAM for the
    activations of N frames at once, a model that answered a different number of times
    than it was asked — is retried one group at a time. That is the brief's rule that a
    single bad frame must not take the batch with it, and it is also what keeps the
    answers aligned: if the count does not match, nothing can be said about which answer
    belongs to which frame, so the batch is thrown away rather than guessed at.
    """
    if isinstance(runtime, BatchVlm):
        try:
            answers = list(runtime.generate_batch(
                runtime.prepare_batch(groups, prompt), max_new_tokens))
            if len(answers) != len(groups):
                raise ValueError(
                    f"модель вернула {len(answers)} ответов на {len(groups)} кадров")
            batched: list[str | BaseException] = [str(a) for a in answers]
            return batched
        except Exception:  # noqa: BLE001 — fall back to the path that cannot misalign
            pass
    return [_one_answer(runtime, group, prompt, max_new_tokens) for group in groups]


def batched_describe(runtime: VlmDescribeFn,
                     groups: Sequence[Sequence[Image.Image]], prompt: str,
                     max_new_tokens: int) -> list[str | BaseException]:
    """One answer per frame group, IN INPUT ORDER; the exception in place of a failure.

    The order is the whole contract: a shuffled batch gives the verdicts of one file to
    another, which is worse than being slow. Groups that hold no frames never reach the
    model (there would be nothing to look at) and come back as an error, exactly as a
    frame that did not decode does everywhere else in the project — the caller decides
    what a frame without an answer is worth.

    A runtime without the batched halves is not an error: the groups go through the
    plain path one at a time, which is what the caller would have done anyway.
    """
    items = [list(group) for group in groups]
    out: list[str | BaseException] = [
        ValueError("нет кадров — модель не спрашивается") for _ in items]
    kept = [i for i, group in enumerate(items) if group]
    if not kept:
        return out
    answers = _batch_answers(runtime, [items[i] for i in kept], prompt, max_new_tokens)
    for i, answer in zip(kept, answers):
        out[i] = answer
    return out


_VLM_RUNTIMES: dict[str, VlmDescribeFn] = {}


def processor_is_fast(processor: Any) -> bool:
    """Did transformers really give us the fast (torchvision) image processor?

    F101: `use_fast=True` is a request, not a promise — for a model whose repo ships
    only the slow processor config transformers quietly hands back the slow one, and
    the slow one is pure Python over PIL, i.e. a good part of the 0.6 s of CPU per
    frame this feature exists to remove. So the answer is read off the built processor
    rather than assumed, and `scripts/measure_vlm_speed.py` prints it: "we asked for
    fast" and "we got fast" must not be the same line in a report.
    """
    return type(getattr(processor, "image_processor", processor)).__name__.endswith("Fast")


def processor_pads_left(processor: Any) -> bool:
    """Is the processor's tokenizer padding on the LEFT — the side generation needs?

    F105: right padding does not fail a batch, it answers WRONG. The short sequences get
    their pad tokens at the END, generation continues from padding instead of from the
    prompt, and the answers of those positions are decoded out of nothing. Asked off the
    built processor for the same reason `processor_is_fast` is: "we asked for left" and
    "we got left" must not be the same line in a report.
    """
    return str(getattr(getattr(processor, "tokenizer", processor),
                       "padding_side", "right")) == "left"


def pad_generation_left(processor: Any) -> Any:
    """Set the processor's tokenizer to pad on the LEFT; returns the same processor.

    The processor call is also given `padding_side="left"`, but a processor that does
    not know the argument ignores it SILENTLY — and a silently right-padded batch is the
    one failure mode of this feature that produces plausible wrong verdicts. So the
    tokenizer is set as well, and `processor_pads_left` is what the report prints.

    Mutating the tokenizer is safe here for the reason ThreadLocalProcessors exists: the
    processor belongs to the thread that prepares with it. For a single sequence the
    side is a no-op anyway — nothing is padded.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    return processor


def attn_implementation(overall: str | None = None,
                        vision: str | None = None) -> AttnImplementation:
    """The `attn_implementation` argument for from_pretrained — None means "as before".

    `overall` alone is one name for the whole model (transformers hands a plain string
    down to every sub-config). `vision` is the visual tower ALONE, which needs the
    per-sub-config dict form: the tower is where the slow kernel was found, and it has to
    be switchable without touching the language half that was already on `sdpa`.
    """
    if vision is None:
        return overall
    spec = {VISION_SUBCONFIG: vision}
    if overall is not None:
        spec[""] = overall
    return spec


def attention_kernels(model: Any) -> dict[str, str]:
    """What the LOADED model really dispatches to: {"language": ..., "vision": ...}.

    Read off the config rather than assumed, because nobody had looked: the request is
    one thing (and a request for a kernel that is unavailable is quietly downgraded),
    the kernel that runs is another. "?" — this model does not have that half.
    """
    config = getattr(model, "config", None)
    return {
        "language": str(getattr(config, "_attn_implementation", "?")),
        "vision": str(getattr(getattr(config, VISION_SUBCONFIG, None),
                              "_attn_implementation", "?")),
    }


class ThreadLocalProcessors:
    """One processor per thread that prepares frames — built lazily, kept for the run.

    F101: the weights are shared (they are read-only and 20.5 GB), but a processor is
    NOT read-only. Its tokenizer sets truncation/padding on itself at every call, so
    two threads preprocessing at once share mutable state — the same reason every OCR
    worker gets its own easyocr Reader (F73) and every faces worker its own session
    (F12.1). A processor costs milliseconds and no VRAM, so the fix is the same one:
    per thread, once, and never per frame.

    Without a `factory` (a single-threaded caller, a measurement, a test) the shared
    instance is handed out unchanged — there is nothing to protect it from.
    """

    def __init__(self, shared: Any, factory: Callable[[], Any] | None = None) -> None:
        self._shared = shared
        self._factory = factory
        self._local = threading.local()
        self._built = 0
        self._lock = threading.Lock()

    @property
    def built(self) -> int:
        """Processors created besides the shared one — one per preparation thread."""
        return self._built

    def get(self) -> Any:
        if self._factory is None:
            return self._shared
        own: Any = getattr(self._local, "processor", None)
        if own is None:
            own = self._local.processor = self._factory()
            with self._lock:
                self._built += 1
        return own


def qwen_processor(model_name: str, use_fast: bool = True) -> Any:
    """The Qwen processor, fast by default (F101).

    Separate from the model on purpose: a processor costs milliseconds to build while
    the weights cost seconds and 20.5 GB of VRAM, so the measurement script can compare
    the slow and the fast preprocessing on ONE loaded model — and so every preparation
    thread can have its own (see ThreadLocalProcessors).
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name, use_fast=use_fast)


def load_qwen(model_name: str, use_fast: bool = True,
              attn: AttnImplementation = None) -> tuple[Any, Any, str]:
    """(model, processor, device) — the parts `qwen_vlm` assembles a runtime from.

    Lazy-import: the module loads without transformers installed (as junk did with
    easyocr/transformers before) — the build fails ONLY here, and every caller wraps
    that in a graceful fallback (junk → the fast CLIP tier, the namer → the template).

    F105: `attn` is the attention implementation (see `attn_implementation`), and it is
    passed to transformers ONLY when it is given. Not passing the argument at all and
    passing None are not the same thing for a library that decides by "did the user set
    it", so the default here is literally the call that shipped.
    """
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    requested: dict[str, Any] = {} if attn is None else {"attn_implementation": attn}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, **requested)
    model.eval()
    return model, qwen_processor(model_name, use_fast=use_fast), device


def qwen_runtime(model: Any, processor: Any, device: str,
                 processor_factory: Callable[[], Any] | None = None) -> BatchVlm:
    """A loaded Qwen2.5-VL as its CPU and GPU halves (F101), batched or not (F105).

    `prepare` leaves the tensors ON THE CPU. That is the VRAM contract of the pipeline
    above it: with the halves overlapped several frames are in flight at once, and
    moving each to the card at preprocessing time would multiply the peak by the window
    size. The card sees exactly one frame's inputs at a time — `generate` does the
    `.to(device)` itself, the same call the single function made.

    `processor_factory` gives each preparation thread its own processor (see
    ThreadLocalProcessors on why a shared one is not safe to preprocess with). The
    `generate` side keeps using `processor` for decoding, and it runs on one thread
    only, so no processor is ever touched by two threads.

    F105: the halves take SEVERAL frame groups as easily as one — `prepare` is
    `prepare_batch` of a single group and `generate` is the first answer of
    `generate_batch`, so the unbatched path is not a different path, it is the batch of
    one. Two things are true only of a real batch and are handled where they arise:
    the sequences have different lengths, so they are padded (by the processor, which
    also produces the attention mask — a hand-built mask is how a batch silently
    becomes wrong), and the padding for GENERATION goes on the left.
    """
    import torch

    processors = ThreadLocalProcessors(processor, processor_factory)

    def prepare_batch(groups: Sequence[Sequence[Image.Image]], prompt: str) -> Any:
        own = processors.get()
        texts: list[str] = []
        images: list[Image.Image] = []
        for frames in groups:
            group = list(frames)
            content: list[dict[str, Any]] = [
                {"type": "image", "image": im} for im in group]
            content.append({"type": "text", "text": prompt})
            texts.append(own.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True))
            images.extend(group)
        if len(texts) < 2:
            # One sequence has nothing to pad, and this is the call that runs in
            # production — it reaches the processor exactly as it did before F105.
            return own(text=texts, images=images, return_tensors="pt")
        return pad_generation_left(own)(
            text=texts, images=images, padding=True, padding_side="left",
            return_tensors="pt")

    def prepare(frames: Sequence[Image.Image], prompt: str) -> Any:
        return prepare_batch([frames], prompt)

    def generate_batch(prepared: Any, max_new_tokens: int) -> list[str]:
        inputs = prepared.to(device)
        with torch.no_grad():
            # #30 (V1): greedy, NOT sampling. Qwen's default generation_config is
            # do_sample=True: on some frames fp16 logits go to NaN/inf -> softmax
            # gives a zero distribution -> torch.multinomial triggers a CUDA
            # device-side assert that POISONS the context (all subsequent frames
            # also fail). Neither a label nor a short caption needs sampling.
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        # One offset for the whole batch is correct BECAUSE the padding is on the left:
        # every row starts generating at the same index. With right padding this line
        # would quietly slice somebody else's tokens.
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        answers = processor.batch_decode(gen_ids, skip_special_tokens=True)
        return [str(answer).strip() for answer in answers]

    def generate(prepared: Any, max_new_tokens: int) -> str:
        return generate_batch(prepared, max_new_tokens)[0]

    return BatchVlm(prepare=prepare, generate=generate,
                    prepare_batch=prepare_batch, generate_batch=generate_batch)


def qwen_vlm(model_name: str, use_fast: bool = True,
             attn: AttnImplementation = None,
             ) -> VlmDescribeFn:  # pragma: no cover — ML, smoke test
    """Load Qwen2.5-VL through transformers → describe(frames, prompt, max_new_tokens).

    `attn` is off by default (F105): the product path loads the weights exactly as it
    did until the measurement decides otherwise.
    """
    model, processor, device = load_qwen(model_name, use_fast=use_fast, attn=attn)
    return qwen_runtime(model, processor, device,
                        lambda: qwen_processor(model_name, use_fast=use_fast))


def shared_vlm(model_name: str,
               loader: Callable[[str], VlmDescribeFn] | None = None) -> VlmDescribeFn:
    """The single process-wide runtime of `model_name` (built on first use).

    Cached by model name, so the junk stage and the naming stage of one run share one
    copy of the weights. A build failure is NOT cached — it propagates to the caller,
    which decides how to degrade.
    """
    runtime = _VLM_RUNTIMES.get(model_name)
    if runtime is None:
        runtime = (loader or qwen_vlm)(model_name)
        _VLM_RUNTIMES[model_name] = runtime
    return runtime


def reset_shared_vlm() -> None:
    """Forget the loaded runtimes (tests; a caller that wants the weights released)."""
    _VLM_RUNTIMES.clear()


class LocalVLMNamer:
    """A local VLM via the ollama HTTP API (no external network)."""

    def __init__(self, settings: NamingSettings) -> None:
        self._s = settings

    def name(self, ctx: EventContext) -> str | None:
        base = _date_base(ctx)
        images = _encode_images(ctx.sample_paths, self._s.max_samples)
        if base is None or not images:
            return TemplateNamer().name(ctx)
        try:
            resp = _http_post_json(
                f"{self._s.vlm_base_url}/api/generate",
                {
                    "model": self._s.vlm_model,
                    "prompt": _DESCRIBE_PROMPT,
                    "images": [b64 for _mt, b64 in images],
                    "stream": False,
                },
                headers={}, timeout=self._s.vlm_timeout,
            )
            described = _sanitize(str(resp["response"]))
        except (OSError, ValueError, KeyError):
            return None  # network/model unavailable — leave the name untouched
        return f"{base} {described}" if described else None


def _sample_frames(paths: tuple[str, ...], max_n: int,
                   max_edge: int = VLM_MAX_EDGE) -> list[Image.Image]:
    """Up to max_n evenly picked frames of the event, decoded for a local model.

    Decoding goes through the shared preview cache — Unicode/HEIC-safe (the F38
    lesson) and free of format restrictions, unlike _encode_images, which may only
    send an HTTP API the handful of formats that API accepts. An unreadable frame is
    skipped, not fatal: the event is described by the ones that decoded.

    F102: `max_edge` is `vlm.max_edge` — the same size the deep junk tier feeds the
    model, because it is the same model. The default keeps a caller that has no config
    section on the historical 896.
    """
    frames: list[Image.Image] = []
    for p in _evenly_picked(list(paths), max_n):
        try:
            st = os.stat(p)
        except OSError:
            continue
        img = imaging.decode_rgb_preview(p, st.st_mtime, st.st_size,
                                         max_edge=max_edge)
        if img is not None:
            frames.append(img)
    return frames


class VlmNamer:
    """The local Qwen2.5-VL through transformers — the model the junk stage loads (F95).

    ONE call per event (3–5 frames at once), not per file: on a live collection that
    is 473 calls, minutes rather than hours. The name is the template base — the
    dates and the place, both known exactly — plus what the model saw: `2025-04-24..
    05-06 Тайланд пляжный отдых с детьми`. The model is never asked for a place or a
    date; it would invent them.

    Opt-in (naming.provider: vlm), because the model is heavy and the tool must keep
    working on a laptop. Every failure degrades to the template name instead of
    breaking the naming stage: transformers missing, the model not loading, no VRAM,
    no readable frames, a garbage answer. Nothing here logs — an event name is built
    out of what is in the frames, and neither the answer nor a hint of it belongs in
    a run log (see the module contract).
    """

    def __init__(self, settings: NamingSettings,
                 loader: Callable[[str], VlmDescribeFn] | None = None,
                 vlm: VlmConfig | None = None) -> None:
        self._s = settings
        # F102: the model and the size its frames are decoded to come from the `vlm:`
        # section. Without one — a caller holding only NamingSettings — the legacy
        # `naming.classify_vlm_model` still answers, which is the very key load_config
        # resolves that section from.
        self._vlm = vlm if vlm is not None else VlmConfig(
            model=str(getattr(settings, "classify_vlm_model", DEFAULT_VLM_MODEL)))
        self._loader = loader
        self._describe: VlmDescribeFn | None = None
        self._unavailable = False

    def _runtime(self) -> VlmDescribeFn | None:
        """The shared runtime, built on first use; None — degrade to the template.

        A failed build is remembered: 473 events must not each retry loading a model
        that is not there (the retry is the slow part, not the failure).
        """
        if self._describe is None and not self._unavailable:
            try:
                self._describe = shared_vlm(self._vlm.model, self._loader)
            except Exception:  # noqa: BLE001 — an optional tier must not break naming
                self._unavailable = True
        return self._describe

    def name(self, ctx: EventContext) -> str | None:
        template = TemplateNamer().name(ctx)
        if template is None:
            return None  # no usable dates — there is no base to attach a description to
        describe = self._runtime()
        if describe is None:
            return template
        frames = _sample_frames(ctx.sample_paths, self._s.max_samples,
                                self._vlm.max_edge)
        if not frames:
            return template
        try:
            answer = describe(frames, _DESCRIBE_PROMPT, _VLM_MAX_NEW_TOKENS)
        except Exception:  # noqa: BLE001 — one bad event must not break the stage
            return template
        described = _sanitize(answer)
        return f"{template} {described}" if described else template


class ClaudeNamer:
    """The Anthropic Messages API. Called ONLY when naming.provider='claude'."""

    def __init__(self, settings: NamingSettings) -> None:
        self._s = settings
        self._api_key = os.environ.get(settings.claude_api_key_env, "")
        if not self._api_key:
            raise RuntimeError(
                f"naming.provider=claude требует API-ключ в переменной окружения "
                f"{settings.claude_api_key_env}"
            )

    def name(self, ctx: EventContext) -> str | None:
        base = _date_base(ctx)
        images = _encode_images(ctx.sample_paths, self._s.max_samples)
        if base is None or not images:
            return TemplateNamer().name(ctx)
        content: list[dict[str, Any]] = [
            {"type": "image",
             "source": {"type": "base64", "media_type": mt, "data": b64}}
            for mt, b64 in images
        ]
        content.append({"type": "text", "text": _DESCRIBE_PROMPT})
        try:
            resp = _http_post_json(
                _ANTHROPIC_URL,
                {
                    "model": self._s.claude_model,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": content}],
                },
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                },
                timeout=self._s.claude_timeout,
            )
            blocks = resp.get("content") or []
            text = next(b["text"] for b in blocks if b.get("type") == "text")
            described = _sanitize(str(text))
        except (OSError, ValueError, KeyError, StopIteration):
            return None  # network/response invalid — leave the name untouched
        return f"{base} {described}" if described else None


def make_namer(settings: NamingSettings, vlm: VlmConfig | None = None) -> EventNamer:
    """Pick the provider from config (naming.provider).

    `vlm` is the `vlm:` section (F102) — only the local `vlm` provider has any use for
    it, and only for the model and its input size; omitting it falls back to the legacy
    `naming.*` keys on `settings`.
    """
    if settings.provider == "template":
        return TemplateNamer()
    if settings.provider == "vlm":
        return VlmNamer(settings, vlm=vlm)
    if settings.provider == "local_vlm":
        return LocalVLMNamer(settings)
    if settings.provider == "claude":
        return ClaudeNamer(settings)
    raise ValueError(
        f"naming.provider={settings.provider!r}: "
        f"ожидается template | vlm | local_vlm | claude"
    )


# --- Applying to events -----------------------------------------------------

@dataclass
class NamingStats:
    total: int = 0            # auto events on input (name_is_manual = 0)
    renamed: int = 0
    unchanged: int = 0        # the provider returned the same name
    failed: int = 0           # the provider returned None (name kept)
    manual_kept: int = 0      # events with a manual name — not touched


def _sample_paths(conn: sqlite3.Connection, event_id: int, max_n: int) -> tuple[str, ...]:
    """Frames of the event a provider may look at: canonical photos, junk excluded.

    F95: the media_class filter sits HERE, before the provider is chosen, and not
    inside the local VLM namer. Two reasons, in this order:

    * `claude` reaches the CLOUD through this very function — a filter inside the
      local provider would mean that switching the provider sends documents over the
      network;
    * whatever the frames show becomes the name of a physical folder ("2024-05-01..
      05-03 медицинская справка"), which then travels into backups, reports and
      screenshots. The photo of the passport leaks nowhere; the fact that it is a
      passport does. See the confidential-documents rule.

    An empty media_class does NOT block naming: junk is the sixth pipeline stage and
    events the fifth, so on the first run there is nothing to filter by yet. The
    price of a miss is one unfortunate folder name, corrected on the next run — that
    is not worth reordering the pipeline for.
    """
    rows = conn.execute(
        """SELECT f.path FROM event_files ef JOIN files f ON f.id = ef.file_id
           LEFT JOIN media_class mc ON mc.file_id = f.id
           WHERE ef.event_id = ? AND f.dup_of IS NULL AND f.error IS NULL
             AND f.media_type = 'photo'
             AND (mc.verdict IS NULL OR mc.verdict = 'photo')
           ORDER BY f.taken_at""",
        (event_id,),
    ).fetchall()
    # with headroom: the provider itself takes up to max_samples of suitable formats
    return tuple(r["path"] for r in rows[: max_n * 4])


def name_events(
    cfg: Config, conn: sqlite3.Connection,
    namer: EventNamer | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> NamingStats:
    """Name auto events with the chosen provider; does not touch name_is_manual=1."""
    s = naming_settings(cfg)
    if namer is None:
        namer = make_namer(s, cfg.vlm)

    stats = NamingStats()
    (stats.manual_kept,) = conn.execute(
        "SELECT COUNT(*) FROM events WHERE name_is_manual = 1"
    ).fetchone()
    rows = conn.execute(
        """SELECT id, started_at, ended_at, place_city, name FROM events
           WHERE name_is_manual = 0 ORDER BY started_at"""
    ).fetchall()
    stats.total = len(rows)
    with conn:
        for i, r in enumerate(rows, 1):
            ctx = EventContext(
                started_at=r["started_at"], ended_at=r["ended_at"],
                city=r["place_city"],
                sample_paths=_sample_paths(conn, r["id"], s.max_samples),
            )
            new = namer.name(ctx)
            if new is None:
                stats.failed += 1
            elif new == r["name"]:
                stats.unchanged += 1
            else:
                # safety predicate: never overwrite a manual name, even under a race
                conn.execute(
                    "UPDATE events SET name = ? WHERE id = ? AND name_is_manual = 0",
                    (new, r["id"]),
                )
                stats.renamed += 1
            if progress:
                progress(i, len(rows))
    return stats


def utcnow_iso() -> str:
    """A single updated_at format for the F6 modules."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
