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
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from PIL import Image

from . import imaging
from .config import Config
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

VlmDescribeFn = Callable[[Sequence[Image.Image], str, int], str]

DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# The VLM input is not for fine details like OCR, a large frame is not needed; saves
# VRAM/generation time.
VLM_MAX_EDGE = 896

_VLM_RUNTIMES: dict[str, VlmDescribeFn] = {}


def qwen_vlm(model_name: str) -> VlmDescribeFn:  # pragma: no cover — ML, smoke test
    """Load Qwen2.5-VL through transformers → describe(frames, prompt, max_new_tokens).

    Lazy-import: the module loads without transformers installed (as junk did with
    easyocr/transformers before) — the build fails ONLY here, and every caller wraps
    that in a graceful fallback (junk → the fast CLIP tier, the namer → the template).
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    def describe(frames: Sequence[Image.Image], prompt: str, max_new_tokens: int) -> str:
        images = list(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            # #30 (V1): greedy, NOT sampling. Qwen's default generation_config is
            # do_sample=True: on some frames fp16 logits go to NaN/inf -> softmax
            # gives a zero distribution -> torch.multinomial triggers a CUDA
            # device-side assert that POISONS the context (all subsequent frames
            # also fail). Neither a label nor a short caption needs sampling.
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return str(answer).strip()

    return describe


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


def _sample_frames(paths: tuple[str, ...], max_n: int) -> list[Image.Image]:
    """Up to max_n evenly picked frames of the event, decoded for a local model.

    Decoding goes through the shared preview cache — Unicode/HEIC-safe (the F38
    lesson) and free of format restrictions, unlike _encode_images, which may only
    send an HTTP API the handful of formats that API accepts. An unreadable frame is
    skipped, not fatal: the event is described by the ones that decoded.
    """
    frames: list[Image.Image] = []
    for p in _evenly_picked(list(paths), max_n):
        try:
            st = os.stat(p)
        except OSError:
            continue
        img = imaging.decode_rgb_preview(p, st.st_mtime, st.st_size,
                                         max_edge=VLM_MAX_EDGE)
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
                 loader: Callable[[str], VlmDescribeFn] | None = None) -> None:
        self._s = settings
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
                self._describe = shared_vlm(
                    str(getattr(self._s, "classify_vlm_model", DEFAULT_VLM_MODEL)),
                    self._loader)
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
        frames = _sample_frames(ctx.sample_paths, self._s.max_samples)
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


def make_namer(settings: NamingSettings) -> EventNamer:
    """Pick the provider from config (naming.provider)."""
    if settings.provider == "template":
        return TemplateNamer()
    if settings.provider == "vlm":
        return VlmNamer(settings)
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
        namer = make_namer(s)

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
