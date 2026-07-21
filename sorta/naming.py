"""F6 (Phase 5): event names.

Contract: reads events/event_files/files, updates ONLY events.name
(and only rows with name_is_manual = 0 — manual names are untouchable).

Providers behind a common EventNamer interface (switching — one line
naming.provider in config.yaml):
- template  — a local template "YYYY-MM-DD <City>" (events.py is not imported here — modules talk only through the DB);
- local_vlm — the ollama HTTP API, 3–5 sample frames of the event;
- claude    — the Anthropic Messages API, key from env; a network call is possible
  ONLY if provider='claude' is explicitly chosen in the config.

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
from typing import Any, Callable, Protocol

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


def _encode_images(paths: tuple[str, ...], max_n: int) -> list[tuple[str, str]]:
    """Up to max_n evenly picked frames → [(media_type, base64), ...]."""
    usable = [p for p in paths if Path(p).suffix.lower() in _IMAGE_MEDIA_TYPES]
    if len(usable) > max_n:
        step = (len(usable) - 1) / (max_n - 1)
        usable = [usable[round(i * step)] for i in range(max_n)]
    out: list[tuple[str, str]] = []
    for p in usable:
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        out.append((_IMAGE_MEDIA_TYPES[Path(p).suffix.lower()],
                    base64.standard_b64encode(data).decode("ascii")))
    return out


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
    if settings.provider == "local_vlm":
        return LocalVLMNamer(settings)
    if settings.provider == "claude":
        return ClaudeNamer(settings)
    raise ValueError(
        f"naming.provider={settings.provider!r}: ожидается template | local_vlm | claude"
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
    rows = conn.execute(
        """SELECT f.path FROM event_files ef JOIN files f ON f.id = ef.file_id
           WHERE ef.event_id = ? AND f.dup_of IS NULL AND f.error IS NULL
             AND f.media_type = 'photo'
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
