"""Load config.yaml into a typed configuration. stdlib + PyYAML only."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from . import i18n

_log = logging.getLogger(__name__)

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(level: str) -> None:
    """Configure the root `sorta` logger (level + StreamHandler).

    Idempotent: a repeated call does not add duplicate handlers (a marker on the
    handler object). An invalid `level` -> WARNING + a warning (does not crash).
    """
    logger = logging.getLogger("sorta")
    lvl_name = str(level).upper()
    invalid = lvl_name not in _VALID_LOG_LEVELS
    if invalid:
        lvl_name = "WARNING"
    console_level = getattr(logging, lvl_name)
    logger.setLevel(console_level)
    if not any(getattr(h, "_sorta_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._sorta_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    # F69: every command already funnels through here, so this is the one place that
    # gives all of them a run log without touching twenty call sites. The console
    # handler above is untouched — the file is added, not substituted. A failure to
    # open it must never take a command down, hence the broad guard.
    try:
        from .runlog import file_log_level, setup_file_logging

        setup_file_logging()
        # A record is dropped by the level of the LOGGER it was emitted on, before any
        # handler is consulted. With log_level=WARNING (the default) that swallowed the
        # INFO `stage=... elapsed=...` lines the run log exists for — the file stayed
        # silent about exactly the thing it was added to measure. Lower the logger to
        # whatever the file wants, and give the console its own level back so its
        # output does not become chattier.
        for existing in logger.handlers:
            if getattr(existing, "_sorta_handler", False):
                existing.setLevel(console_level)
        logger.setLevel(min(console_level, file_log_level()))
    except Exception as exc:  # noqa: BLE001 — logging is never worth crashing over
        logger.warning("config: не удалось включить файловый лог: %s", exc)
    if invalid:
        logger.warning("config: неверный log_level=%r, используется WARNING", level)


@dataclass
class IndexConfig:
    extensions: dict[str, list[str]] = field(default_factory=lambda: {
        "photo": ["jpg", "jpeg", "png", "heic", "heif", "webp", "tif", "tiff", "bmp"],
        "raw": ["cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "raf"],
        "video": ["mp4", "mov", "avi", "mts", "m2ts", "3gp", "mkv"],
    })
    min_file_size_kb: int = 5
    compute_phash: bool = True
    phash_max_distance: int = 5  # Hamming threshold for the near-duplicate report
    skip_dirs: list[str] = field(default_factory=lambda: [
        ".thumbnails", "@eaDir", "$RECYCLE.BIN", "System Volume Information",
    ])

    def media_type_of(self, ext: str) -> str | None:
        e = ext.lower().lstrip(".")
        for mtype, exts in self.extensions.items():
            if e in exts:
                return mtype
        return None


@dataclass
class DatesConfig:
    min_year: int = 1990
    max_year: int = 2035


@dataclass
class DedupConfig:
    canonical_strategy: str = "prefer_exif_then_largest"


@dataclass
class GeoConfig:
    session_gap_hours: float = 6  # a gap larger than this starts a new session
    provider: str = "offline"     # offline (geodata, default) | online (Nominatim, G2b)
    # online provider (Nominatim/OSM, G2b) — used only when provider=online
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "sorta-photo-organizer"  # required by the OSM policy
    nominatim_timeout: float = 10.0
    # F93: coordinate rounding for the GRID FALLBACK key of the online-geo cache —
    # used only for coordinates the bundled base cannot place (everything else is keyed
    # by city+district geonameid, which is both faster and exact). 3 digits ≈ 110 m.
    # Fewer digits = faster, but coarser (may confuse districts).
    cache_coord_digits: int = 3
    # F93: how long a cached provider answer stays fresh, in days. City and district
    # borders move rarely, but not never — 180 days ≈ half a year, an expired row is
    # re-asked once and rewritten. 0 turns the expiry off entirely.
    cache_max_age_days: int = 180


@dataclass
class EventsConfig:
    gap_hours: float = 6         # a larger gap — a new session
    merge_gap_hours: float = 18  # DEPRECATED (F30: replaced by trip_merge_gap_hours); kept for compatibility
    trip_merge_gap_hours: float = 48  # F30: adjacent sessions of the same city merge into a trip on a smaller gap
    min_event_size: int = 5      # F30: smaller groups are not an event (files go down the no_event branch → Year/month)
    trip_merge_max_km: float = 120  # F44/#19: adjacent sessions merge into a trip even across DIFFERENT cities if in the same country and closer than this (Bali across villages → one trip); 0 — only on city/region equality


@dataclass
class SortConfig:
    # photos with multiple people: primary (largest face) | shared_folder (_Совместные)
    multi_person: str = "primary"
    # directories of the already-manually-sorted part of the collection — files in
    # them (and subfolders) are not sorted (they stay in the index). Combined with --exclude (F16).
    exclude_dirs: list[str] = field(default_factory=list)
    # threads for parallel report-thumbnail generation (--thumbnails); 0 = auto
    # (min(8, cores)). Decoding is the heavy step, the GIL is released in the C decode (F18).
    thumbnail_workers: int = 0
    # F35: album root (sorta album / "Collect into folder" buttons); None →
    # ui/cli fall back to "_Albums" next to the DB
    album_dir: str | None = None
    # F56: directory for sort plan reports (CSV/HTML/thumbs). None → report_output/
    # next to the DB. Isolates one-off reports (real place names/paths) from the
    # DB/repo directory and keeps them gitignored (report_output/).
    report_dir: str | None = None
    # F49 (#4-B): drop the district level in the city layout when the district name
    # is not localized in the config language (foreign transliteration Wichit/Tuban ->
    # Country/City/Year path). RU and localized foreign districts (Ubud/Kuta) stay;
    # an online district from Nominatim (already localized) is not affected. False —
    # previous behaviour (transliterated district in the path).
    drop_unlocalized_district: bool = True


@dataclass
class FacesConfig:
    min_face_px: int = 40        # smaller — not embedded (quality filter)
    det_threshold: float = 0.7   # detector threshold (Immich default)
    min_cluster_size: int = 5    # HDBSCAN; smaller — noise
    max_distance: float = 0.5    # cosine similarity threshold (Immich default)


# F102: the local VLM used to be configured out of the `naming:` section — the toggle
# (naming.vlm_enabled), the model (naming.classify_vlm_model) and the preparation
# threads (naming.vlm_workers) all sat under "naming" because there was no other
# address, even though the first of them switches JUNK CLASSIFICATION on. The one knob
# that decides what the pass costs — the input resolution — was not in the config at all
# but a constant in the code, against the project rule that thresholds live in
# config.yaml. `vlm:` is that address. The dividing line: what describes the shared
# model RUNTIME lives here, what belongs to a consumer stays with the consumer
# (naming.provider chooses who names events; naming.product_candidate_min is the junk
# gate deciding which frame is worth a VLM call at all — a property of the gate, not of
# the model).
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# The VLM input is not for fine details like OCR, a large frame is not needed; it saves
# VRAM and generation time. Since F102 this is only the DEFAULT — the value in use comes
# from vlm.max_edge, and scripts/measure_vlm_resolution.py prices what lowering it buys
# and what it costs in verdicts.
DEFAULT_VLM_MAX_EDGE = 896
# The workers cap is deliberately modest and NOT "all your cores": the machine that runs
# this may well have two, the GPU half is a single consumer that only needs to be kept
# fed, and every worker in flight holds a preprocessed frame in RAM (F101).
_VLM_WORKERS_CAP = 4


def default_vlm_workers() -> int:
    """Preparation threads when the config does not say: min(4, cores), always >= 1."""
    return min(_VLM_WORKERS_CAP, os.cpu_count() or 1)


@dataclass(frozen=True)
class VlmConfig:
    """`vlm:` — the shared runtime of the local VLM (F102).

    Both consumers — the deep junk tier and the `vlm` event namer — run the SAME
    weights, one copy per process (the peak is 20.5 GB of VRAM, a second instance does
    not fit), so the model, its input size and the preparation pool describe that
    runtime rather than either stage.

    `enabled` is mirrored onto `NamingConfig.vlm_enabled` by load_config, and that is
    deliberate: `--deep` and the "Deep analysis (VLM)" checkbox force the tier for ONE
    run by replacing `cfg.naming.vlm_enabled` on their own copy of the config, so that
    field stays the effective per-run toggle — this section is where the config FILE
    states the default it starts from.
    """
    enabled: bool = False
    model: str = DEFAULT_VLM_MODEL
    workers: int = field(default_factory=default_vlm_workers)
    max_edge: int = DEFAULT_VLM_MAX_EDGE


# F102: the pre-`vlm:` address of each knob. A live config.yaml holds
# `naming.vlm_enabled: false`, and silently ignoring that would switch a 20 GB tier ON
# on somebody else's machine — so the old keys keep working. The new key wins when it is
# given; otherwise the old one is used and says so ONCE per run (this is read once at
# startup anyway, and a warning per frame is how a log becomes unreadable).
#
#     vlm.enabled  <- naming.vlm_enabled
#     vlm.model    <- naming.classify_vlm_model
#     vlm.workers  <- naming.vlm_workers
#
# vlm.max_edge has no old address: it was a constant in the code.
#
# Process-wide on purpose — "once per run", not once per load_config call (the web app
# reloads the config on every request). Tests clear it.
_ALIAS_WARNED: set[str] = set()


def _mapping(value: Any) -> dict:
    """A config section as a dict — `vlm:` left empty, or filled with junk, is not a crash."""
    return value if isinstance(value, dict) else {}


def _vlm_value(new: dict, old: dict, new_key: str, old_key: str) -> Any:
    """The raw value of one knob: `vlm.<new_key>`, else the legacy `naming.<old_key>`."""
    if new_key in new:
        return new[new_key]
    if old_key not in old:
        return None
    if old_key not in _ALIAS_WARNED:
        _ALIAS_WARNED.add(old_key)
        _log.warning(
            "config: ключ naming.%s устарел — переименуйте его в vlm.%s "
            "(пока читается по-старому)", old_key, new_key)
    return old[old_key]


def _as_bool(value: Any, default: bool) -> bool:
    """YAML truth for a toggle; anything unrecognizable -> `default`, never a crash.

    Strings are parsed instead of being handed to bool(): a quoted "false" is truthy in
    Python, and a config that says false must never switch a heavy tier on.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


def _as_positive_int(value: Any, default: int) -> int:
    """A positive whole number; absent / 0 / negative / garbage -> `default`.

    A bad number in a config file is a typo, not a reason to refuse to start — and a
    silent 0 threads or a 0-pixel frame would be worse than the default either way.
    """
    if isinstance(value, bool):  # bool is an int in Python; `max_edge: true` is garbage
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def resolve_vlm_workers(raw: dict | None) -> int:
    """Threads preparing frames for the VLM — `vlm.workers` (was `naming.vlm_workers`).

    Default min(4, cpu_count). 1 means the serial pass, which is what the deep tier did
    before F101 and what a runtime without split halves does anyway. Absent / 0 /
    negative / garbage -> the default; the result is always >= 1.

    Takes the raw YAML rather than a Config so the measurement scripts can ask the same
    question of a config they only parsed.
    """
    data = _mapping(raw)
    value = _vlm_value(_mapping(data.get("vlm")), _mapping(data.get("naming")),
                       "workers", "vlm_workers")
    return _as_positive_int(value, default_vlm_workers())


def _vlm_from(data: dict) -> VlmConfig:
    """The `vlm:` section of the whole YAML, with the legacy `naming.*` keys honoured."""
    new = _mapping(data.get("vlm"))
    old = _mapping(data.get("naming"))
    d = VlmConfig()
    model = _vlm_value(new, old, "model", "classify_vlm_model")
    return VlmConfig(
        enabled=_as_bool(_vlm_value(new, old, "enabled", "vlm_enabled"), d.enabled),
        model=model.strip() if isinstance(model, str) and model.strip() else d.model,
        workers=resolve_vlm_workers(data),
        max_edge=_as_positive_int(new.get("max_edge"), d.max_edge),
    )


@dataclass(frozen=True)
class NamingConfig:
    """Phase 5 (F6): places without GPS, event names, junk. A flat view of the
    nested naming section of config.yaml (clip.*/local_vlm.*/claude.* — see load_config)."""
    provider: str = "template"           # template | vlm | local_vlm | claude
    #                                      F95: `vlm` describes the event with the local
    #                                      Qwen2.5-VL of classify_vlm_model (the junk model,
    #                                      one copy per run); opt-in, template stays the default
    landmark_threshold: float = 0.85     # CLIP threshold for places — conservative: 0.35
    #                                      gave false matches (cafe→Istanbul), and a wrong
    #                                      city is worse than unknown. Proper fix — a geo model (backlog #11)
    # F75: a single CLIP score does not separate a right city from a wrong one — on the
    # live collection the wrong ones scored 0.980 against 0.991 for the right one — so a
    # match is corroborated by its neighbours: within one directory, a city held by less
    # than `dominance` of at least `min` matches is dropped back to unknown (one card
    # dump is one trip; you cannot be in Prague and Berlin at once). Raising `min` or
    # `dominance` makes the rule fire less often.
    landmark_group_min: int = 5
    landmark_group_dominance: float = 0.6
    junk_threshold: float = 0.85         # CLIP threshold for junk classes (high: CLIP
    #                                      zero-shot readily mislabels real photos)
    document_threshold: float = 0.9      # CLIP threshold for the "documents" category (F15,
    #                                      above junk: a photographed document → _Documents, not junk)
    text_frac_min: float = 0.08          # F37: document + text_frac below → scene (FP gate, beach→city)
    text_frac_document: float = 0.15     # F38: photo + text_frac above → document (FN rescue); lowered
    #                                      0.35→0.15 by validation (a document at an angle gave 0.247, scenes 0.0)
    text_rescue_docscore_min: float = 0.3  # F38: FN rescue runs OCR only if doc_score ≥ this
    #                                        (clear scenes doc_score≈0 spend no OCR — perf)
    text_frac_downscale_px: int = 1280   # F38: downscale the frame to this before easyocr.detect (×3–10 speed)
    vlm_enabled: bool = False            # F37-B: deep tier — VLM 3-way (memory/document/product).
    #                                      opt-in (needs the [vlm] extra); default OFF, graceful fallback to CLIP.
    #                                      F102: the config key moved to `vlm.enabled` and load_config keeps this
    #                                      field equal to it — but the field itself stays, because --deep and the
    #                                      UI checkbox force the tier for one run through it (see _legacy_naming_view)
    classify_vlm_model: str = DEFAULT_VLM_MODEL  # F37-B: classifier VLM (NOT vlm_model — that is for
    #                                      event-naming/llava; a separate field to avoid a collision).
    #                                      F102: superseded by `vlm.model`, kept in sync by load_config
    product_candidate_min: float = 0.4   # #14/V1: product-CLIP above this → the file goes to the VLM (candidate gate, not final)
    landmarks_file: str = "data/landmarks.yaml"
    clip_model: str = "ViT-L-14-quickgelu"  # the quickgelu variant for the openai weights (without it — a mismatch)
    clip_pretrained: str = "openai"
    clip_batch_size: int = 16
    clip_decode_workers: int = 0         # F64: CLIP decode-pool threads; 0 = auto min(cpu, 16)
    max_samples: int = 4                 # sample frames of an event for the VLM (3–5)
    vlm_base_url: str = "http://localhost:11434"
    vlm_model: str = "llava"
    vlm_timeout: float = 120.0
    claude_model: str = "claude-opus-4-8"
    claude_api_key_env: str = "ANTHROPIC_API_KEY"
    claude_timeout: float = 60.0


def _naming_from(raw: dict) -> NamingConfig:
    clip = raw.get("clip") or {}
    vlm = raw.get("local_vlm") or {}
    claude = raw.get("claude") or {}
    d = NamingConfig()
    return NamingConfig(
        provider=str(raw.get("provider", d.provider)),
        landmark_threshold=float(raw.get("landmark_threshold", d.landmark_threshold)),
        landmark_group_min=int(raw.get("landmark_group_min", d.landmark_group_min)),
        landmark_group_dominance=float(
            raw.get("landmark_group_dominance", d.landmark_group_dominance)),
        junk_threshold=float(raw.get("junk_threshold", d.junk_threshold)),
        document_threshold=float(raw.get("document_threshold", d.document_threshold)),
        text_frac_min=float(raw.get("text_frac_min", d.text_frac_min)),
        text_frac_document=float(raw.get("text_frac_document", d.text_frac_document)),
        text_rescue_docscore_min=float(
            raw.get("text_rescue_docscore_min", d.text_rescue_docscore_min)),
        text_frac_downscale_px=int(raw.get("text_frac_downscale_px", d.text_frac_downscale_px)),
        vlm_enabled=bool(raw.get("vlm_enabled", d.vlm_enabled)),
        classify_vlm_model=str(raw.get("classify_vlm_model", d.classify_vlm_model)),
        product_candidate_min=float(raw.get("product_candidate_min", d.product_candidate_min)),
        landmarks_file=str(raw.get("landmarks_file", d.landmarks_file)),
        clip_model=str(clip.get("model", d.clip_model)),
        clip_pretrained=str(clip.get("pretrained", d.clip_pretrained)),
        clip_batch_size=int(clip.get("batch_size", d.clip_batch_size)),
        clip_decode_workers=int(clip.get("decode_workers", d.clip_decode_workers)),
        max_samples=int(raw.get("max_samples", d.max_samples)),
        vlm_base_url=str(vlm.get("base_url", d.vlm_base_url)).rstrip("/"),
        vlm_model=str(vlm.get("model", d.vlm_model)),
        vlm_timeout=float(vlm.get("timeout", d.vlm_timeout)),
        claude_model=str(claude.get("model", d.claude_model)),
        claude_api_key_env=str(claude.get("api_key_env", d.claude_api_key_env)),
        claude_timeout=float(claude.get("timeout", d.claude_timeout)),
    )


def _legacy_naming_view(naming: NamingConfig, vlm: VlmConfig) -> NamingConfig:
    """`naming.vlm_enabled`/`classify_vlm_model` held equal to the resolved `vlm:` (F102).

    The two fields stay on NamingConfig instead of being deleted, and not only for the
    sake of old configs: `--deep` (cli) and the "Deep analysis (VLM)" checkbox force the
    tier for one run by replacing `cfg.naming.vlm_enabled` on their own copy of the
    config, so that field is the effective toggle the junk stage reads. This makes it
    agree with the section the value now comes from — whichever of the two addresses the
    file happened to use.
    """
    return replace(naming, vlm_enabled=vlm.enabled, classify_vlm_model=vlm.model)


@dataclass
class Config:
    sources: list[Path] = field(default_factory=list)
    database: Path = Path("sorta.db")
    index: IndexConfig = field(default_factory=IndexConfig)
    dates: DatesConfig = field(default_factory=DatesConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    sort: SortConfig = field(default_factory=SortConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)  # F102: the shared VLM runtime
    language: str = "en"  # folder/name language (ru|en|ja), normalized in load_config (F25/F27)
    log_level: str = "WARNING"  # DEBUG|INFO|WARNING|ERROR; validated in configure_logging (F52)
    raw: dict = field(default_factory=dict)  # the full YAML for future-phase sections


def _known(cls, raw: dict) -> dict:
    """Keep from raw only the declared fields of the dataclass cls. Config sections
    may carry "raw" keys that a module reads directly from cfg.raw (e.g.
    faces.decode_workers in faces.py) or future-phase keys — they are kept in
    Config.raw but must not break the section constructor."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def load_config(path: str | Path = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    idx = data.get("index", {})
    vlm = _vlm_from(data)
    cfg = Config(
        sources=[Path(p) for p in data.get("sources", [])],
        database=Path(data.get("database", "sorta.db")),
        index=IndexConfig(
            extensions={**IndexConfig().extensions, **idx.get("extensions", {})},
            min_file_size_kb=idx.get("min_file_size_kb", 5),
            compute_phash=idx.get("compute_phash", True),
            phash_max_distance=idx.get("phash_max_distance", 5),
            skip_dirs=idx.get("skip_dirs", IndexConfig().skip_dirs),
        ),
        dates=DatesConfig(**_known(DatesConfig, data.get("dates") or {})),
        dedup=DedupConfig(**_known(DedupConfig, data.get("dedup") or {})),
        geo=GeoConfig(**_known(GeoConfig, data.get("geo") or {})),
        faces=FacesConfig(**_known(FacesConfig, data.get("faces") or {})),
        events=EventsConfig(**_known(EventsConfig, data.get("events") or {})),
        sort=SortConfig(**_known(SortConfig, data.get("sort") or {})),
        naming=_legacy_naming_view(_naming_from(data.get("naming") or {}), vlm),
        vlm=vlm,
        language=i18n.normalize_lang(data.get("language")),
        log_level=str(data.get("log_level", "WARNING")),
        raw=data,
    )
    _apply_imaging_config(data.get("imaging") or {})
    # sources may be empty: the source is given positionally (sorta index <dir>).
    # The non-empty requirement is at the point of use (index / in-place sort).
    return cfg


# F67 follow-up: the preview cache is configured through env vars, because imaging.py
# is a leaf module with no access to Config (decode_rgb_preview is called from pool
# workers that only carry a path). Rather than thread settings through every caller,
# the config file seeds those env vars — and only when they are NOT already set, so a
# variable exported in the shell still wins, which is the documented contract.
_IMAGING_ENV = {
    "preview_cache": "SORTA_PREVIEW_CACHE",
    "preview_dir": "SORTA_PREVIEW_DIR",
    "preview_max_edge": "SORTA_PREVIEW_MAX_EDGE",
    "preview_quality": "SORTA_PREVIEW_QUALITY",
    # F74/F80: video tiles and the lightbox filmstrip sit in the same leaf module and
    # were env-only until now, so config.example.yaml documented keys that nothing
    # read. They seed their vars exactly like the preview ones — env still wins.
    "video_previews": "SORTA_VIDEO_PREVIEWS",
    "video_workers": "SORTA_VIDEO_WORKERS",
    "video_frames": "SORTA_VIDEO_FRAMES",
}


def _apply_imaging_config(raw: dict) -> None:
    for key, env_name in _IMAGING_ENV.items():
        if key not in raw or os.environ.get(env_name):
            continue
        value = raw[key]
        if isinstance(value, bool):  # YAML `false` -> the "0" imaging expects
            value = "1" if value else "0"
        os.environ[env_name] = str(value)


# F104: words YAML reads as something other than a plain string. A model name will
# never be one of them, but a saver that emits `model: no` and reads back `False` is a
# trap waiting for the one value that hits it — quoting them costs nothing.
_YAML_RESERVED = frozenset({
    "y", "n", "yes", "no", "true", "false", "on", "off", "null", "none", "~",
})
# Characters that carry no YAML meaning outside quotes: a model id, a path fragment,
# a language code. Anything else (a colon, a hash, a leading dash, spaces) is quoted.
_YAML_PLAIN_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+-]*$")


def _yaml_scalar(value: bool | int | float | str) -> str:
    """One value as the text of a YAML scalar — bools lower-case, strings safe."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if _YAML_PLAIN_RE.match(text) and text.lower() not in _YAML_RESERVED:
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _set_top_level(text: str, key: str, scalar: str) -> str:
    """Replace the value of a top-level `key:` line, or append the line."""
    pattern = re.compile(rf"(?m)^{re.escape(key)}:.*$")
    if pattern.search(text):
        # A lambda replacement, not a template: a quoted value may contain a backslash,
        # which re.sub would read as a group reference and mangle.
        return pattern.sub(lambda _m: f"{key}: {scalar}", text, count=1)
    head = text.rstrip("\n") + "\n" if text.strip() else ""
    return f"{head}{key}: {scalar}\n"


def _set_in_section(text: str, section: str, key: str, scalar: str) -> str:
    """Replace (or add) `key:` inside a top-level `section:` block of the YAML text.

    Line-level, like `_set_top_level`: the block is everything indented under the
    header, and only the ONE line of that block is rewritten. A missing key is appended
    to the end of the block (with the block's own indentation), a missing section is
    appended to the end of the file — so a config.yaml that never mentioned `vlm:`
    grows the section instead of the value going nowhere.
    """
    lines = text.split("\n")
    header = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        head = text.rstrip("\n") + "\n" if text.strip() else ""
        return f"{head}{section}:\n  {key}: {scalar}\n"
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line[:1].isspace():
            break
        end += 1
    entry = re.compile(rf"^(\s+){re.escape(key)}:.*$")
    indent = "  "
    last_filled = start
    for i in range(start + 1, end):
        match = entry.match(lines[i])
        if match is not None:
            lines[i] = f"{match.group(1)}{key}: {scalar}"
            return "\n".join(lines)
        if lines[i].strip():
            last_filled = i
            if not lines[i].lstrip().startswith("#"):
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(last_filled + 1, f"{indent}{key}: {scalar}")
    return "\n".join(lines)


def save_setting(path: str | Path, key: str, value: bool | int | float | str) -> None:
    """Persist one `key: value` into config.yaml, preserving the rest of the file.

    `key` is either a top-level name (`language`) or one level of nesting
    (`vlm.enabled`) — the two shapes the settings column of the web app writes (F104).

    A line-level edit, not a YAML round-trip: the file belongs to the user and is full
    of their comments, ordering and blank lines, all of which `yaml.safe_dump` would
    silently throw away on the first change of a checkbox. A missing file is created.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    section, _dot, leaf = key.rpartition(".")
    scalar = _yaml_scalar(value)
    updated = (_set_in_section(text, section, leaf, scalar) if section
               else _set_top_level(text, leaf, scalar))
    p.write_text(updated, encoding="utf-8")


def save_language(path: str | Path, lang: str) -> None:
    """Persist `language: <lang>` into config.yaml, preserving the rest of the file.

    `lang` is normalized to a supported code (ru|en|ja) — an invalid value falls back
    to the i18n default. The writing itself is `save_setting` (F104), which generalized
    what this function used to do inline.
    """
    save_setting(path, "language", i18n.normalize_lang(lang))
