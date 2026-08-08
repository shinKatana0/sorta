"""Load config.yaml into a typed configuration. stdlib + PyYAML only."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import i18n

_log = logging.getLogger(__name__)

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(level: str) -> None:
    """Configure the root `sorta` logger (level + StreamHandler).

    Idempotent: a repeated call does not add duplicate handlers (a marker on the handler
    object). An invalid `level` is WARNING plus a warning, never a crash.
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
    # F69: the file log is added here because every command funnels through this
    # function. Failing to open it must never take a command down — hence the guard.
    try:
        from .runlog import file_log_level, setup_file_logging

        setup_file_logging()
        # TRAP: a record is dropped by the level of the LOGGER, before any handler is
        # consulted, so log_level=WARNING swallowed the INFO `stage=... elapsed=...` lines
        # the run log exists for. Lower the logger, give the console its level back.
        for existing in logger.handlers:
            if getattr(existing, "_sorta_handler", False):
                existing.setLevel(console_level)
        logger.setLevel(min(console_level, file_log_level()))
    except Exception as exc:  # noqa: BLE001 — logging is never worth crashing over
        logger.warning("config: не удалось включить файловый лог: %s", exc)
    if invalid:
        logger.warning("config: неверный log_level=%r, используется WARNING", level)


@dataclass
class LoggingConfig:
    """F166: the `logging:` section — how often the run log repeats a running stage.

    Separate from the top-level `log_level`, which decides WHAT gets written.
    """

    # 0 switches the periodic lines off; start and summary lines are not affected.
    progress_interval_sec: float = 60.0


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
    # F186 retired `keeper_vlm`, the comparative "which frame of the group to keep"
    # question: measured 2026-08-04 over 111 groups labelled blind by the owner, the model
    # agreed with the person on 32% against 30.4% for a random pick, for 451 s of GPU. F194
    # retired what its answers were used for — on the same 111 groups every signal sits at
    # the level of a coin (sharpness 27%), so `group_keeper` is read as the ORDER the
    # frames are shown in, not as the group's answer.
    #
    # The two sizes below outlive that question: they describe the POPULATION of
    # near-duplicate groups (`dedup.keeper_groups`). `keeper_max_frames` is the best N by
    # sharpness of a group — the live collection holds one of 38 near-duplicates.
    keeper_max_frames: int = 5
    # The smallest group worth asking about. MEASURED 2026-08-02 and moved 2 -> 3: 85% of
    # the groups on a live collection are PAIRS (676 of 791), and in 40 of the 73 pairs
    # asked (55%) the model differed from sharpness only because the two frames are
    # INDISTINGUISHABLE — 1.44 s of VLM for a question with no answer. At 3 the 115 groups
    # whose frames actually differ are asked: 791 calls -> 115, ~17 min -> ~2.5. Pairs
    # fall back to sharpness (F120: a fair comparison inside a group, one scene one scale).
    keeper_min_group_size: int = 3


def _dedup_from(raw: dict) -> DedupConfig:
    """The `dedup:` section — every value tolerant of garbage, like `features:` below.

    `canonical_strategy` keeps its plain str() reading (dedup._canonical falls back to
    `largest` for anything it does not know).
    """
    d = DedupConfig()
    return DedupConfig(
        canonical_strategy=str(raw.get("canonical_strategy", d.canonical_strategy)),
        keeper_max_frames=_as_positive_int(
            raw.get("keeper_max_frames"), d.keeper_max_frames),
        keeper_min_group_size=_as_positive_int(
            raw.get("keeper_min_group_size"), d.keeper_min_group_size),
    )


@dataclass
class EstimateConfig:
    """F159: what the run screen prices a run with when it has nothing measured here.

    The rest of the budget comes from the run log (F147: how fast every stage ran ON THIS
    MACHINE).
    """

    # How long a timing from the run log is worth trusting. Mirrors
    # `runlog.DEFAULT_MEASUREMENT_MAX_AGE_DAYS`; 0 or less switches the expiry off. The
    # real guard is the build check inside `read_measurements`; this one only catches the
    # same version running months later on a machine that has moved on since.
    measurement_max_age_days: float = 90.0


def _estimate_from(raw: dict) -> EstimateConfig:
    """The `estimate:` section — garbage-tolerant like every section around it."""
    d = EstimateConfig()
    return EstimateConfig(
        measurement_max_age_days=_as_float(
            raw.get("measurement_max_age_days"), d.measurement_max_age_days),
    )


@dataclass
class GeoConfig:
    session_gap_hours: float = 6  # a gap larger than this starts a new session
    provider: str = "offline"     # offline (geodata, default) | online (Nominatim, G2b)
    # online provider (Nominatim/OSM, G2b) — used only when provider=online
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "sorta-photo-organizer"  # required by the OSM policy
    nominatim_timeout: float = 10.0
    # F93: rounding for the GRID FALLBACK key of the online-geo cache, used only for
    # coordinates the bundled base cannot place (the rest is keyed by city+district
    # geonameid — faster and exact). 3 digits ≈ 110 m; fewer may confuse districts.
    cache_coord_digits: int = 3
    # F93: how long a cached provider answer stays fresh, in days — borders move rarely
    # but not never. An expired row is re-asked once and rewritten; 0 turns expiry off.
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
    # photos with multiple people: primary (largest face) | shared_folder (the folder
    # is named «_Совместные» in a ru layout — the product spells it that way)
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
    # next to the DB, which keeps real place names and paths out of the repo directory
    # and gitignored.
    report_dir: str | None = None
    # F49 (#4-B): drop the district level in the city layout when the district name is
    # not localized in the config language (Wichit/Tuban -> Country/City/Year). RU and
    # localized foreign districts (Ubud/Kuta) stay; an online Nominatim district is
    # already localized and unaffected. False — the transliterated district in the path.
    drop_unlocalized_district: bool = True


@dataclass
class FacesConfig:
    min_face_px: int = 40        # smaller — not embedded (quality filter)
    det_threshold: float = 0.7   # detector threshold (Immich default)
    min_cluster_size: int = 5    # HDBSCAN; smaller — noise
    max_distance: float = 0.5    # cosine similarity threshold (Immich default)


# F102 gave the local VLM its own `vlm:` section. The dividing line: what describes the
# shared model RUNTIME lives there, what belongs to a consumer stays with the consumer
# (naming.provider chooses who names events; naming.product_candidate_min is the junk
# gate deciding which frame is worth a call at all — a property of the gate).
DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# The VLM is not asked for fine detail like OCR, so a large frame buys nothing and costs
# VRAM and generation time. Only the DEFAULT since F102 (`vlm.max_edge` is what runs);
# scripts/measure_vlm_resolution.py prices what lowering it buys and costs in verdicts.
DEFAULT_VLM_MAX_EDGE = 896
# F164 was sent to RAISE this cap and measured that it must not be raised. The case for
# more threads was the F101 profile — ~0.6 s of CPU per frame against ~0.19 s of GPU — but
# that premise expired with F105's fast image processor: the 0.6 s of one core became
# ~0.12 s of about seven, so four threads on a 24-core machine already ask for more cores
# than exist. Measured with scripts/measure_vlm_workers.py (120 frames, real decode and
# processor, the model's turn stubbed by a sleep of the measured 0.19 s):
#
#     threads   ms/frame   frames/s   vs 1 thread   model half busy   peak RSS
#           1        290       3,45         x1,00               66%      551 MB
#           2        230       4,35         x1,26               83%      954 MB
#           4        249       4,01         x1,16               79%     1299 MB
#           6        274       3,66         x1,06               70%     1645 MB
#           8        295       3,39         x0,98               66%     2061 MB
#          12        334       2,99         x0,87               57%     2599 MB
#
# Monotonically slower from two on (reproducible to within 2%): ONE preparation already
# saturates the machine — 7.6 cores busy with a single worker — so the next thread takes
# cores from the previous one. The 51% card utilization of the 2026-08-03 live run has
# the same cause, not a starving card. The pool costs ~170 MB per thread.
#
# The default stays min(4, cores), inside 8% of the best row, because that peak was
# measured against a STUBBED model half: a sleep releases the interpreter lock for its
# whole duration and a real `generate()` does not, so moving it to two needs
# `scripts/measure_vlm_workers.py --full` on a free card. Only the DEFAULT is capped —
# `vlm.workers` is never clamped to it.
_VLM_WORKERS_CAP = 4


def default_vlm_workers() -> int:
    """Preparation threads when the config does not say: min(4, cores), always >= 1."""
    return min(_VLM_WORKERS_CAP, os.cpu_count() or 1)


# F153: how the two indexes answer one query — `clip_embeddings` (ViT-L-14) and
# `search_embeddings` (the multilingual model), which are equally accurate at the top and
# WRONG IN DIFFERENT PLACES. That is what makes fusion worth anything, and it is why both
# modes work on RANKS:
#
#   off     the search index alone
#   rank    reciprocal rank fusion: a frame is weighted by its POSITION in each list,
#           so agreement between the models outranks a single model's favourite
#   union   the two lists merged as sets: a frame keeps its best position, so a frame
#           found by one model only is not pushed out by one both models found
#
# No mode ADDS THE SCORES: a cosine of ViT-L-14 and a cosine of xlm-roberta-base-ViT-B-32
# live in different spaces, look comparable and print alike. The fusion functions in
# `search.py` are handed file ids and no scores at all, so that rule is mechanical.
SEARCH_FUSION_OFF = "off"
SEARCH_FUSION_RANK = "rank"
SEARCH_FUSION_UNION = "union"
SEARCH_FUSION_MODES = (SEARCH_FUSION_OFF, SEARCH_FUSION_RANK, SEARCH_FUSION_UNION)

# F120: the media classes a frame can be excluded by before any VLM sees it. These are
# the `media_class.verdict` values; `photo` is deliberately absent — excluding personal
# photographs would leave the tier with nothing to do.
VLM_EXCLUDABLE_CLASSES = ("document", "product", "screenshot", "meme")
# Documents by default: the bucket holds passports, medical forms and bank papers, which
# the project already refuses to DECODE for display. The cost of excluding them is that
# the deep tier is what CORRECTS a wrong `document` verdict, so an excluded class keeps
# whatever the fast tier decided.
DEFAULT_VLM_EXCLUDE_CLASSES = ("document",)


@dataclass(frozen=True)
class VlmConfig:
    """`vlm:` — the shared runtime of the local VLM (F102).

    Both consumers — the deep junk tier and the `vlm` event namer — run the SAME weights,
    one copy per process (the peak is 20.5 GB of VRAM; a second instance does not fit).

    `enabled` is mirrored onto `NamingConfig.vlm_enabled` by load_config: `--deep` and the
    "Deep analysis (VLM)" checkbox force the tier for ONE run by replacing that field on
    their own copy of the config, so it stays the effective per-run toggle.
    """
    enabled: bool = False
    # F161: the deep junk tier, which `enabled` used to switch on by itself. It is the
    # only producer of the `product` class — on the live run of 2026-07-28 it moved 2 202
    # of its 2 592 verdicts into exactly that class — so the effect gets a key of its own.
    #
    # Default TRUE, unlike every other subordinate key here: before this key existed
    # `vlm.enabled: true` meant the tier, and such a file must run what it ran yesterday.
    # Still subordinate — see `products_allowed`.
    products: bool = True
    model: str = DEFAULT_VLM_MODEL
    workers: int = field(default_factory=default_vlm_workers)
    max_edge: int = DEFAULT_VLM_MAX_EDGE
    # F186 retired `quality` and `quality_scope`: the frame-quality question was down to
    # "are the eyes open", and F179 answered it off eyelid geometry the faces stage
    # already produces — 62% precision at 48% recall against the model's 60% at 9%, for
    # no call at all.
    # F120: privacy — media classes no VLM is ever shown, by verdict of the fast tier.
    # Empty tuple = send everything, which is what happened before this key existed.
    exclude_classes: tuple[str, ...] = DEFAULT_VLM_EXCLUDE_CLASSES


# Retired key names already warned about. F102 moved three knobs out of `naming:`, and a
# live config.yaml holding `naming.vlm_enabled: false` must not have a 20 GB tier switched
# ON by that move — so the old keys keep working, the new one winning when both are given:
#
#     vlm.enabled  <- naming.vlm_enabled
#     vlm.model    <- naming.classify_vlm_model
#     vlm.workers  <- naming.vlm_workers
#
# F173 added a rename inside ONE section (`features.search_page` <- `features.search_limit`,
# see `_renamed_value`), which is why an alias is registered by its full `section.key`.
#
# Process-wide on purpose — once per RUN, not once per load_config call (the web app
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


def _renamed_value(raw: dict, section: str, new_key: str, old_key: str) -> Any:
    """The raw value of a knob that was renamed INSIDE one section: new spelling wins.

    The `_vlm_value` rule with one section instead of two — a key that moved must not take
    somebody's setting with it. Warns once per process.
    """
    if new_key in raw:
        return raw[new_key]
    if old_key not in raw:
        return None
    alias = f"{section}.{old_key}"
    if alias not in _ALIAS_WARNED:
        _ALIAS_WARNED.add(alias)
        _log.warning(
            "config: ключ %s.%s устарел — переименуйте его в %s.%s "
            "(пока читается по-старому)", section, old_key, section, new_key)
    return raw[old_key]


def _as_bool(value: Any, default: bool) -> bool:
    """YAML truth for a toggle; anything unrecognizable -> `default`, never a crash.

    TRAP: strings are parsed rather than handed to bool(), because a quoted "false" is
    truthy in Python and a config that says false must never switch a heavy tier on.
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
    """A positive whole number; absent / 0 / negative / garbage -> `default`."""
    if isinstance(value, bool):  # bool is an int in Python; `max_edge: true` is garbage
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _as_float(value: Any, default: float) -> float:
    """A finite number; absent / garbage -> `default` (a typo must not stop a run).

    Booleans are rejected like in `_as_positive_int`: YAML `true` is a 1 nobody meant.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default


# F170: who may name an event — every one of them runs on this machine. The fourth name
# here uploaded an event's sample frames to a vendor API; it was DELETED rather than
# defaulted off, so "no code sends images anywhere" is a property of the sources.
NAMING_PROVIDERS = ("template", "vlm", "local_vlm")
# Values that named a REAL feature until it was removed — not typos, so they earn a
# sentence and a fallback rather than a crash on the first line of a run.
REMOVED_NAMING_PROVIDERS = ("claude",)


def removed_provider_message(provider: str) -> str:
    """What a run says about a `naming.provider` that no longer exists."""
    return (
        f"config: naming.provider={provider!r} удалён — этот провайдер отправлял "
        f"фотографии в облако, теперь в продукте нет кода, отправляющего изображения "
        f"наружу. Доступны {' | '.join(NAMING_PROVIDERS)}; использую 'template'"
    )


def _as_provider(value: Any, default: str) -> str:
    """`naming.provider`, with a removed provider answered rather than obeyed.

    Anything else, typos included, passes through untouched: `make_namer` is the one place
    that decides what a provider name means.
    """
    name = value.strip().lower() if isinstance(value, str) else value
    if name in REMOVED_NAMING_PROVIDERS:
        _log.warning("%s", removed_provider_message(str(name)))
        return default
    return str(value)


def _as_repo_id(key: str, value: Any, default: str) -> str:
    """A huggingface repo id (`owner/name`); anything else -> the default, with a warning.

    Deliberately weaker than `_as_model_name`: the string goes to `from_pretrained`, which
    fails loudly by itself, so all this stops is a download attempt for an empty name.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        _log.warning("config: %s=%r — ожидался идентификатор модели, использую %r",
                     key, value, default)
    return default


def _as_fusion(value: Any, default: str) -> str:
    """F153: one of SEARCH_FUSION_MODES; anything else -> the default, with a warning.

    Announced rather than silent: which of the three runs decides what the ranking IS.
    """
    if isinstance(value, str) and value.strip().lower() in SEARCH_FUSION_MODES:
        return value.strip().lower()
    if value is not None:
        _log.warning("config: features.search_fusion=%r не из %s — использую %r",
                     value, "/".join(SEARCH_FUSION_MODES), default)
    return default


def _as_model_name(value: Any, default: str) -> str:
    """F141: `<open_clip architecture>/<weights>`; anything else -> the default.

    Both halves are required and that is the whole check: open_clip happily builds a model
    with no weights, filling the search index with vectors of an untrained tower.
    """
    text = value.strip() if isinstance(value, str) else ""
    architecture, sep, weights = text.partition("/")
    if sep and architecture.strip() and weights.strip():
        return text
    if value is not None:
        _log.warning("config: features.search_model=%r — ожидалось "
                     "'<модель open_clip>/<веса>', использую %r", value, default)
    return default


def resolve_vlm_workers(raw: dict | None) -> int:
    """Threads preparing frames for the VLM — `vlm.workers` (was `naming.vlm_workers`).

    Default min(4, cpu_count); 1 means the serial pass, and the result is always >= 1.
    Takes the raw YAML rather than a Config so the measurement scripts can ask the same
    question of a config they only parsed.
    """
    data = _mapping(raw)
    value = _vlm_value(_mapping(data.get("vlm")), _mapping(data.get("naming")),
                       "workers", "vlm_workers")
    return _as_positive_int(value, default_vlm_workers())


def _as_exclude_classes(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """F120: `vlm.exclude_classes` — a list of media classes, or `[]` to send everything.

    An EMPTY list is a real answer, so only `None`/a non-list falls back to the default.
    An unknown class name is dropped with a warning: a typo there would silently send the
    very bucket the user meant to protect.
    """
    if value is None or isinstance(value, (str, bytes)) or not isinstance(value, list):
        if value is not None:
            _log.warning("config: vlm.exclude_classes=%r — ожидался список, "
                         "использую %r", value, list(default))
        return default
    out: list[str] = []
    for item in value:
        name = str(item).strip().lower()
        if name in VLM_EXCLUDABLE_CLASSES:
            out.append(name)
        elif name:
            _log.warning("config: vlm.exclude_classes — неизвестный класс %r, "
                         "допустимы %s", name, "/".join(VLM_EXCLUDABLE_CLASSES))
    if value and not out:
        # Every name was a typo. Falling through to "nothing excluded" would turn the
        # protection off for somebody who was writing it down: an empty list stays empty,
        # a list of typos does not become one.
        _log.warning("config: vlm.exclude_classes — ни одно имя не распознано, "
                     "использую %r", list(default))
        return default
    return tuple(dict.fromkeys(out))


def _vlm_from(data: dict) -> VlmConfig:
    """The `vlm:` section of the whole YAML, with the legacy `naming.*` keys honoured."""
    new = _mapping(data.get("vlm"))
    old = _mapping(data.get("naming"))
    d = VlmConfig()
    model = _vlm_value(new, old, "model", "classify_vlm_model")
    return VlmConfig(
        enabled=_as_bool(_vlm_value(new, old, "enabled", "vlm_enabled"), d.enabled),
        products=_as_bool(new.get("products"), d.products),
        model=model.strip() if isinstance(model, str) and model.strip() else d.model,
        workers=resolve_vlm_workers(data),
        max_edge=_as_positive_int(new.get("max_edge"), d.max_edge),
        exclude_classes=_as_exclude_classes(new.get("exclude_classes"),
                                            d.exclude_classes),
    )


# F154: torchvision's own COCO checkpoint, resolved by name — no new dependency
# (torchvision 0.28 is already installed for the CLIP side), only weights are downloaded.
# The mobilenet backbone and not a ResNet one because the feature rests on the detector
# being cheap: 83.8 ms per frame is 30.8 min over 22 096 photographs as a pass, and
# ~3 min over the ~2 000 candidates a query selects.
DEFAULT_DETECT_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"


@dataclass(frozen=True)
class DetectConfig:
    """`detect:` — the runtime of the object detector (F154), and its master switch.

    A section of its own for the reason `vlm:` is one: the RUNTIME lives here, the
    QUESTION stays with the consumer (`features.detector*`). `enabled` is the F145 rule
    applied to a second kind of model, and deliberately NOT `vlm.enabled`: somebody who
    cleared the deep tier did not thereby ask for a detector.
    """
    enabled: bool = False
    model: str = DEFAULT_DETECT_MODEL


def _detect_from(data: dict) -> DetectConfig:
    """The `detect:` section of the whole YAML — garbage-tolerant like `vlm:` above."""
    raw = _mapping(data.get("detect"))
    d = DetectConfig()
    model = raw.get("model")
    return DetectConfig(
        enabled=_as_bool(raw.get("enabled"), d.enabled),
        model=model.strip() if isinstance(model, str) and model.strip() else d.model,
    )


@dataclass(frozen=True)
class SavedSlice:
    """F151: one pinned query — the name of a slice and the phrases it is ranked by.

    A slice, not a filter with a threshold: both halves are DATA, so a slice is added,
    retuned or dropped by editing `config.yaml` and never by editing code.

    The phrases are ENGLISH whatever `language:` says — they are input to a CLIP text
    tower, the measured pairs (children 61%/89%, products 65%/95%) came from English
    wording, and translating them is unmeasured.
    """

    name: str
    queries: tuple[str, ...]


# Why these three (measured 2026-08-02 on 200 hand-labelled frames out of 22 096):
#
#     concept    today's filter          the same vectors, asked
#                recall  precision       recall@N  recall@2N  precision@N
#     children   no filter at all          61%        89%         61%
#     products      0%        —             65%        95%         65%
#     animals      33%       71%            60%        87%         60%
#
# The animal slice does NOT replace the `pet` label: at 71% precision the label answers
# "is this confidently an animal", a different question, so the two coexist.
#
# Blurred is deliberately NOT here: the sharpness filter is 100% precise on that sample
# against the query's 36%, so folding them together would sink an exact signal into an
# approximate one.
#
# Three phrases each is not what makes it work — on the same sample one, three and six
# phrases differ by less than the noise floor (animals 67/60/73, children 68/61/68,
# products 65/65/60, one labelled frame being worth ~6.7 points).
DEFAULT_SAVED_SLICES: tuple[SavedSlice, ...] = (
    SavedSlice("children", ("a photo of a child",
                            "children playing",
                            "a photo of a kid at a party")),
    SavedSlice("products", ("a photo of a product",
                            "a product photo of an item for sale",
                            "a catalogue photo of goods")),
    SavedSlice("animals", ("a photo of an animal",
                           "a pet at home",
                           "a photo of a dog or a cat")),
)


def _as_saved_slices(value: object,
                     default: tuple[SavedSlice, ...]) -> tuple[SavedSlice, ...]:
    """F151: `features.saved_slices` — a mapping of slice name -> list of phrases.

    An EMPTY mapping is a real answer ("pin nothing") and survives, the
    `_as_exclude_classes` rule. A mapping in which nothing survived falls back to the
    default: a file full of typos is not a request to remove every slice. The ORDER is
    kept — it is the order of the pins.
    """
    if not isinstance(value, dict):
        if value is not None:
            _log.warning("config: features.saved_slices=%r — ожидалось отображение "
                         "«имя среза: список формулировок», использую значение "
                         "по умолчанию", value)
        return default
    out: list[SavedSlice] = []
    for raw_name, raw_queries in value.items():
        name = str(raw_name).strip()
        # A single phrase written as a plain string is a list of one, not a mistake.
        items = [raw_queries] if isinstance(raw_queries, str) else raw_queries
        phrases = items if isinstance(items, (list, tuple)) else []
        queries = tuple(dict.fromkeys(
            q.strip() for q in phrases if isinstance(q, str) and q.strip()))
        if name and queries:
            out.append(SavedSlice(name, queries))
        else:
            _log.warning("config: features.saved_slices[%r] — ожидался непустой список "
                         "формулировок, срез пропущен", raw_name)
    if value and not out:
        return default
    return tuple(out)


@dataclass(frozen=True)
class FeaturesConfig:
    """`features:` — the F113 frame-quality cascade: one toggle, then its thresholds.

    The rule the section exists for: every signal is taken with the cheapest tool that can
    answer it, and a new function gets a new toggle. Sharpness has none — it is a laplacian
    over a preview every other stage has already paid for — while pets, which cost a prompt
    group inside the junk stage's CLIP call, do.

    The thresholds are here and not in the code because none can be guessed:
    `scripts/measure_frame_quality.py` prints the distributions to choose them from, on
    the collection they will be applied to.
    """
    # F222: the landmark stage, which until then had no switch — it ran on every run and
    # downloaded 1.6 GB of CLIP weights the first time. MEASURED on the owner's collection
    # (26 137 frames, 2026-08-07):
    #
    #     places by source        landmarks contributed
    #       exact_gps      14 254     Prague     121
    #       unknown         7 622     Moscow      16
    #       session_inferred 3 171    Paris        5
    #       trip_inferred     881     New York     1
    #       visual            143  <- this stage
    #       path_inferred      66
    #
    # 143 of 26 137 is 0.55%, practically one trip. 45.3% of the frames have no GPS, and
    # what rescues them is `session_inferred` + `trip_inferred` (4 052), which need no
    # model at all. Off does NOT mean gone: `visual` places already in a database survive
    # the stage being skipped.
    landmarks: bool = False
    pets: bool = False
    # F130: check every candidate with the local VLM before the label is written — one
    # question, three answers (a live animal / a picture of one / no animal). Needs `pets`
    # as well: it verifies what the CLIP group found, it does not look by itself.
    pets_verify: bool = False
    # F122: MEASURED, on 320 hand-labelled frames stratified by score and weighted back
    # to the collection:
    #
    #     cutoff   marked   correct   precision   recall
    #      0.85       665       615         92%      45%
    #      0.70       805       738         92%      54%
    #      0.60       895       801         89%      58%   (the old default)
    #      0.50       993       842         85%      61%
    #      0.30      1331       905         68%      66%
    #
    # 0.70 dominates 0.85 outright — the same precision for nine more points of recall —
    # and buys three points of precision over 0.60 for four of recall. With ~40 frames a
    # band the interval is about ±8 points, so this is a preference, not a proven optimum.
    pet_threshold: float = 0.7
    # F130: the OTHER threshold — who is shown to the model when `pets_verify` is on. Far
    # below the one above, because 0.70 is high only for want of anything checking CLIP.
    # Counted on the stored `pet_score` of the live collection: 0.70 selects 805 frames
    # (10.5 min at 0.78 s/frame), 0.50 993 (12.9), 0.30 1 331 (17.3), 0.20 1 679 (21.8),
    # everything 19 757 (4.3 h).
    #
    # F158 replaced the 0.50 F130 shipped, which had been read off a REPLAY against the
    # F122 labels — a set stratified by score band, so its low band rested on a handful of
    # labels. Re-measured on 500 RANDOM hand-labelled frames (36 animals), scored by
    # `pet_label`:
    #
    #     way                             marked  correct  precision  recall
    #     threshold 0.70 (before F130)        18       17        94%     47%
    #     gate 0.30, no check                 34       23        68%     64%
    #     cascade 0.30 + VLM                  28       23        82%     64%
    #     gate 0.50, no check                 21       18        86%     50%
    #     cascade 0.50 + VLM (F130 shipped)   20       18        90%     50%
    #
    # At 0.50 the cascade buys three points of recall over the bare 0.70 threshold for 21
    # model calls. At 0.30 it buys seventeen (47% -> 64%) for twelve of precision (94% ->
    # 82%), removing 6 of the 11 false marks and losing no correct one. The price on the
    # live collection is ~1 500 frames against ~930 — ~19 min against ~12 at 0.77 s/frame.
    pet_candidate_threshold: float = 0.3
    # F140: score every photograph on how much it looks like a screenshot, a photographed
    # screen or a receipt, and show the ones above `junk_rescue_threshold` to the VLM. The
    # score costs no model pass (a matmul over the vectors `store_embeddings` keeps), each
    # frame it selects ~0.78 s in the deep tier. With the deep tier off the score is
    # written and the candidates marked, but no verdict changes.
    junk_rescue: bool = False
    # Who is shown to the model. MEASURED by eye on the live collection (19 753 stored
    # vectors, every one classified `photo` — these are the classifier's own misses, not
    # junk that leaked into the index):
    #
    #     threshold   frames   share of the photographs   what the review found
    #      +0.05          93            0.5%              junk outright
    #      +0.02         955            4.8%              ~17% real photographs in
    #                                                     the band +0.02..+0.05
    #       0.00        5688           28.8%              junk in single figures
    #
    # A selection threshold, NOT a verdict: at ~85% precision, reclassifying by it directly
    # would throw ~150 living photographs out of the city layout — the mistake F130
    # measured for animals. 955 frames is ~12 minutes of the deep tier.
    junk_rescue_threshold: float = 0.02
    # F154: a THIRD tier for the animal label — an object detector over the candidates a
    # query picks out of the stored vectors. Needs `detect.enabled` as well (this key says
    # the cascade wants a detector, that one says a detector may be loaded at all).
    #
    # ANIMALS ONLY, and the table that drew that boundary — animals against people against
    # food, 200 hand-labelled frames, 2026-08-02 — is in the `detect` module docstring,
    # beside the class list it decided. The animal row re-measured on 500 frames
    # (2026-08-03) reads 78% / 69% at confidence 0.5; see `detector_threshold` below.
    detector: bool = False
    # How deep into the query ranking the candidate list goes — the ONE number that decides
    # what this feature costs, since the detector sees nothing else. MEASURED (2026-08-03,
    # 500 hand-labelled frames, 36 animals); the last column is the share of the known
    # animals the query put in front of the detector at all:
    #
    #     depth   candidates     time    recall ceiling
    #       500          500   0.7 min        25%
    #      1000         1000   1.4 min        50%
    #      2000         2000   2.8 min        83%   <- the F154 default
    #      4000         4000   5.6 min       100%   <- chosen
    #     10000        10000  14.0 min       100%
    #
    # THE CEILING BOUNDS EVERY RECALL BELOW IT: a frame the query never showed is not found
    # at any threshold, so at 2 000 candidates 17% of the animals were unreachable in
    # principle. 2.8 minutes more bought the whole remaining ceiling, against the ~19 min
    # the animal stage already spends on the VLM. 10 000 buys nothing.
    detector_candidates: int = 4000
    # The confidence at which a detected box counts. CHOSEN FROM A TABLE, not in advance —
    # `python scripts/measure_detector.py` prints it over a labelled sample. On the same
    # 500 frames:
    #
    #                            precision   recall
    #     the CLIP label today       94%       47%
    #     detector at 0.50           78%       69%   <- the F154 default
    #     detector at 0.60           86%       69%   <- chosen
    #     detector at 0.70           86%       67%
    #
    # 0.60 dominates 0.50 with nothing traded away: the same recall, 25 correct marks out
    # of 29 instead of 25 out of 32. F154 shipped 62% / 87% here, read off 200 frames where
    # fifteen animals made each one worth 6.7 points of recall; both figures moved by two
    # dozen points on the larger sample. The boxes are stored with their scores, so
    # re-choosing needs no new pass.
    detector_threshold: float = 0.6
    # F131: the same cascade for places — CLIP proposes a landmark, the local VLM is asked
    # what place the frame shows, and only a proposal the model names itself goes on to F75
    # corroboration. Off, `naming.landmark_threshold` alone selects.
    #
    # The cascade was NOT assumed to work here: F75 measured the landmark failure as one of
    # DISCRIMINATING KNOWLEDGE (wrong cities scored 0.980 against 0.991 — no threshold
    # splits them), and a 3B model could easily share it. The phase-0 probe asked 104
    # frames with a known answer, 24 of them hard negatives, and got ZERO false
    # confirmations at 92% accuracy. The mechanism is silence, not knowledge: 71 of the 104
    # answers named nothing at all.
    landmarks_verify: bool = False
    # Who is shown to the model when `landmarks_verify` is on — the second threshold, far
    # below `naming.landmark_threshold` (0.85), which is high only because nothing was
    # checking CLIP's proposal. MEASURED by the same probe over the 7 619 place-less frames
    # of the live collection, against what F75 corroboration would do with them:
    #
    #     threshold  proposals  F75 keeps  F75 drops
    #       0.85            10          8          2   <- the CLIP-only gate today
    #       0.70            66         52         14
    #       0.50           151        127         24
    #
    # 151 questions is a couple of minutes of VLM, an order of magnitude cheaper than the
    # animal cascade's 1 331, so the band is taken whole. Never ABOVE
    # `naming.landmark_threshold`: that would narrow the population the stage already
    # finds, and the check exists to widen it.
    landmark_candidate_threshold: float = 0.5
    # The longer side the frame is scaled to before the laplacian. FIXED on purpose: the
    # variance of the laplacian is scale-dependent, so two frames measured at different
    # resolutions are not comparable and no threshold over them means anything.
    sharpness_max_edge: int = 512
    # The band where sharpness decides nothing — clearly blurred is below it, clearly
    # sharp above — and one of the two ways into the VLM population.
    sharpness_band_min: float = 30.0
    sharpness_band_max: float = 300.0
    # The other way in: the junk-group CLIP probability of "a photograph" below this is
    # CLIP saying it does not know what it is looking at.
    subject_score_min: float = 0.9
    # F126/F157: how far down the blur review list opens by default — the DEPTH OF THE
    # FIRST PAGE of a ranking ordered by ascending sharpness, not the edge of "blurred".
    #
    # F157 raised it from 90. On 300 frames labelled by the STRICT criterion the user chose
    # ("visibly smeared", not "a little soft"; 17 blurred), a cutoff buys recall far faster
    # than it loses precision:
    #
    #     threshold  flagged  right  precision  recall
    #        90          7      2       29%       12%    <- the old default
    #       200         23      5       22%       29%
    #       300         47      9       19%       53%
    #       450         73     10       14%       59%
    #       700        120     14       12%       82%
    #
    # The same signal read as an ordering, with no threshold: the top 5% of the list holds
    # 24% of the blurred frames, the top 10% 41%, the top 20% 53%, the top 30% 65%.
    #
    # How long the list is on a real archive (2026-08-04, 19 211 photographs carrying a
    # sharpness; `measure_frame_quality.py --features sharpness` sweeps any collection):
    #
    #     threshold  first page  of the photographs
    #        90            523         2.7%
    #       200          1 663         8.7%
    #       300          2 968        15.4%
    #       450          4 988        26.0%
    #       700          7 859        40.9%
    #
    # 300 is where the two tables meet: half the blurred frames inside a first page of ~15%
    # of the collection, against the 41% that 700 opens and the 12% recall 90 cut it at.
    #
    # NOT a verdict at any value — four of five frames on that page are not blurred.
    blur_review_max: float = 300.0
    # F150: the ceiling of the "low resolution" slice, in MEGAPIXELS. The one number here
    # that measures nothing: width and height are a FACT the indexer wrote down, so there
    # is no precision or recall to quote.
    #
    # MEASURED (2026-08-02, 22 095 photographs) — the distribution the default comes from:
    #
    #     megapixels   frames   share   of them inside the blur window
    #     < 0.2            94    0.4%              5%
    #     0.2 - 0.5       133    0.6%             10%
    #     0.5 - 1         479    2.2%              1%
    #     1 - 2           586    2.7%              2%
    #     5 - 12        2 942   13.3%              6%
    #     > 12         17 493   79.2%              2%
    #
    # 1.0 selects 706 frames, and the shape is what makes the round number right: phones do
    # not take pictures of that size, so what lies under a megapixel arrived through a
    # messenger or a download. No other slice sees them — 682 of the 706 are formally
    # sharp, a 3% intersection with the blur window.
    #
    # NOT a verdict: a small frame can be the only surviving photograph of somebody.
    low_resolution_mp: float = 1.0
    # F155: the same laplacian measured INSIDE the face boxes, and the number below which
    # such a frame is a blur candidate. A separate setting from `blur_review_max` because
    # the two are not on one scale — a variance over a whole preview against one over a
    # 100-200 px crop — and no factor converts them.
    #
    # MEASURED (2026-08-02, the 68 frames of a 200-frame hand-checked sample that have a
    # face; 13 of them blurred):
    #
    #     measured over    threshold  flagged  right  precision  recall
    #     the whole frame        300       10      2       20%      15%
    #     the face crop          100       17      5       29%      38%
    #     the face crop          200       33      8       24%      62%
    #     the face crop          400       44     10       23%      77%
    #
    # 200 quadruples recall for a comparable number of frames flagged, and it is a
    # direction rather than a figure: 13 blurred frames make each one worth ~8 points of
    # recall. NOT A VERDICT: precision is ~25% at every row above.
    face_sharpness_max: float = 200.0
    # F179: how open the eyes of the largest face are (`frame_quality.eye_openness` — the
    # height of the eye opening over its width, off the 106-point contour), and the number
    # BELOW which the frame joins the "closed eyes" slice. The only threshold here a
    # smaller number passes: a closed eye is a thin slit.
    #
    # MEASURED (F178, 2026-08-03, the same 249 hand-labelled frames the retired VLM
    # question was judged on, weighted back to the collection):
    #
    #     way in                threshold  precision  recall
    #     the VLM (retired)             —      60%       9%
    #     eyelid geometry            0.16      68%      34%
    #     eyelid geometry            0.18      62%      48%   <- this line
    #     eyelid geometry            0.22      56%      61%
    #     a classifier over the crop  0.9      46%      57%
    #     CLIP over the crop          0.8      58%      49%
    #
    # 0.18 was chosen by a rule fixed before the run, not by eye; the neighbouring rows
    # price moving it. The population is ~948 frames, 15.6% of everything with a face.
    # NOT A VERDICT: at 62% precision one frame in three of the slice has its eyes open, so
    # this ORDERS the list (most closed first).
    eye_openness_max: float = 0.18
    # F128: keep the CLIP vector the junk stage already computes instead of discarding it
    # (table `clip_embeddings`). ON by default: ~60 MB per 20 000 photos, written inside a
    # pass that is already running, against an off state where search by words, an album
    # from a query and "frames like this one" each start with a full CLIP pass. The switch
    # is for very large collections — 300 000 photos mean ~920 MB.
    store_embeddings: bool = True
    # F129: how many frames a search by words takes at a time. NOT a similarity threshold,
    # and there will not be one, for the reason sharpness has none: a CLIP score orders
    # frames against each other and means nothing in absolute terms.
    #
    # F173 renamed it from `search_limit`, and the name is the fix: a CEILING cuts the
    # result off, a PAGE only decides how much arrives first. Depth is the single measured
    # lever of completeness (2026-08-02/03 — doubling the list adds ~25 points on average,
    # and the query «дети» goes from 61% to 89%). The old spelling keeps working.
    search_page: int = 200
    # F151: the pinned queries — name -> the phrases that slice is ranked by. A SAVED QUERY
    # and not a filter of its own: the engine (F129), the index (F141) and the paging
    # (F173) already exist, so "children" costs a config entry rather than a threshold, a
    # calibration and a code path. See `DEFAULT_SAVED_SLICES` for how these three arose.
    saved_slices: tuple[SavedSlice, ...] = DEFAULT_SAVED_SLICES
    # F156: how many pins the interface will let a person add. NOT a resource bound — a pin
    # costs a config entry and one matmul when opened — but the F133 bound on the screen: a
    # row of forty pins is a remote control again. A file edited BY HAND past it keeps
    # every slice: this governs what the interface adds, not what the config may say.
    max_pinned_slices: int = 12
    # F152: the two numbers the face slices need. Everything else about them is a FACT of
    # the `faces` table, so neither is a confidence threshold — both are GEOMETRIC.
    #
    # How many real faces make a photograph a GROUP photograph. Three: two people are a
    # couple or a passer-by, and 2026-08-02 moved the keeper VLM to the same number.
    group_photo_faces: int = 3
    # A portrait is ONE face taking a noticeable share of the frame: bbox area over
    # `files.width * files.height`. 0.08 is geometry rather than measurement — 8% of the
    # area is roughly 28% of each side, head-and-shoulders rather than a person in a
    # landscape — and has NOT been calibrated on the live collection.
    portrait_face_share: float = 0.08
    # F141: the SEARCH index — a second CLIP vector per photograph, computed by a
    # multilingual model and read by search alone (`search_embeddings`). OFF by default
    # because unlike `store_embeddings` it is not free: a SECOND CLIP pass over the
    # collection, 19 753 frames in 635 s (~10.5 min) on the machine it was measured on,
    # plus ~40 MB per 20 000 photographs.
    #
    # What it buys, on 217 hand-labelled judgements over 8 concepts: Russian queries go
    # from 22% to 98% precision at top-5, and four of the eight (cake, food, mountains,
    # children) go from returning NOTHING to working. English does not regress (95%
    # against 98% — three points on forty judgements).
    #
    # Off, search says it has nothing to rank (F134). The classification vectors are NOT a
    # fallback: a search silently answered by the wrong model is the one outcome nobody
    # can see.
    search_index: bool = False
    # The model of the search side, `<open_clip architecture>/<weights>` — a key of its own
    # rather than `naming.clip.*`, which is the entire point of the feature. `naming.clip.*`
    # stays ViT-L-14 because the landmark threshold (F75), the animal threshold (F122), the
    # cascade selection (F130) and the junk classification are calibrated on ITS numbers.
    search_model: str = "xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k"
    # F149: the model behind "try to improve" — a processed COPY of ONE frame a person
    # opened and chose. No toggle: nothing loads until the button is pressed, and pressing
    # it is the opt-in. The price is ~400 MB of weights, downloaded once, and ~1 s per
    # frame on the card this was measured on. `realworld` and not `classical` for the
    # measured reason the `restore` module docstring gives.
    restore_model: str = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
    # F169: the longer side the frame is scaled to BEFORE that model, which is x4 and works
    # at the UPSCALED resolution — a full 12 Mpx frame would become 16 000 px and fit on no
    # card here. A frame ABOVE the ceiling is reduced first and the x4 brings it back to
    # roughly its own size: real detail dropped, plausible detail drawn in its place, and
    # the interface says so instead of calling it an improvement.
    # `scripts/measure_restore.py` prices raising this on the three populations separately.
    restore_max_edge: int = 1024
    # F153: how a query uses the two indexes at once — `off` | `rank` | `union` (see
    # SEARCH_FUSION_MODES above). The price is one extra matmul over a stored table,
    # ~0.9 ms per index, at query time and no pass over any image.
    #
    # OFF until the measurement exists. The observation behind the feature is that the two
    # models return DIFFERENT frames at the same precision (88/96/98% at 1/3/5 for both),
    # so the expected gain is in RECALL — which has never been measured for either.
    # `scripts/measure_search.py --fusion --labels ...` prints precision AND recall at each
    # depth for all four variants; until it has been run there is no number to choose a
    # default by.
    search_fusion: str = SEARCH_FUSION_OFF


def _features_from(raw: dict) -> FeaturesConfig:
    """The `features:` section — every value tolerant of garbage, like `vlm:` above."""
    d = FeaturesConfig()
    return FeaturesConfig(
        landmarks=_as_bool(raw.get("landmarks"), d.landmarks),
        pets=_as_bool(raw.get("pets"), d.pets),
        pets_verify=_as_bool(raw.get("pets_verify"), d.pets_verify),
        pet_threshold=_as_float(raw.get("pet_threshold"), d.pet_threshold),
        pet_candidate_threshold=_as_float(
            raw.get("pet_candidate_threshold"), d.pet_candidate_threshold),
        junk_rescue=_as_bool(raw.get("junk_rescue"), d.junk_rescue),
        junk_rescue_threshold=_as_float(
            raw.get("junk_rescue_threshold"), d.junk_rescue_threshold),
        detector=_as_bool(raw.get("detector"), d.detector),
        detector_candidates=_as_positive_int(
            raw.get("detector_candidates"), d.detector_candidates),
        detector_threshold=_as_float(
            raw.get("detector_threshold"), d.detector_threshold),
        landmarks_verify=_as_bool(raw.get("landmarks_verify"), d.landmarks_verify),
        landmark_candidate_threshold=_as_float(
            raw.get("landmark_candidate_threshold"), d.landmark_candidate_threshold),
        sharpness_max_edge=_as_positive_int(
            raw.get("sharpness_max_edge"), d.sharpness_max_edge),
        sharpness_band_min=_as_float(raw.get("sharpness_band_min"), d.sharpness_band_min),
        sharpness_band_max=_as_float(raw.get("sharpness_band_max"), d.sharpness_band_max),
        subject_score_min=_as_float(raw.get("subject_score_min"), d.subject_score_min),
        blur_review_max=_as_float(raw.get("blur_review_max"), d.blur_review_max),
        low_resolution_mp=_as_float(
            raw.get("low_resolution_mp"), d.low_resolution_mp),
        face_sharpness_max=_as_float(
            raw.get("face_sharpness_max"), d.face_sharpness_max),
        eye_openness_max=_as_float(
            raw.get("eye_openness_max"), d.eye_openness_max),
        store_embeddings=_as_bool(raw.get("store_embeddings"), d.store_embeddings),
        search_page=_as_positive_int(
            _renamed_value(raw, "features", "search_page", "search_limit"), d.search_page),
        saved_slices=_as_saved_slices(raw.get("saved_slices"), d.saved_slices),
        max_pinned_slices=_as_positive_int(
            raw.get("max_pinned_slices"), d.max_pinned_slices),
        group_photo_faces=_as_positive_int(
            raw.get("group_photo_faces"), d.group_photo_faces),
        portrait_face_share=_as_float(
            raw.get("portrait_face_share"), d.portrait_face_share),
        search_index=_as_bool(raw.get("search_index"), d.search_index),
        search_model=_as_model_name(raw.get("search_model"), d.search_model),
        restore_model=_as_repo_id(
            "features.restore_model", raw.get("restore_model"), d.restore_model),
        restore_max_edge=_as_positive_int(
            raw.get("restore_max_edge"), d.restore_max_edge),
        search_fusion=_as_fusion(raw.get("search_fusion"), d.search_fusion),
    )


@dataclass(frozen=True)
class NamingConfig:
    """Phase 5 (F6): places without GPS, event names, junk. A flat view of the
    nested naming section of config.yaml (clip.*/local_vlm.* — see load_config)."""
    provider: str = "template"           # template | vlm | local_vlm (NAMING_PROVIDERS).
    #                                      F95: `vlm` describes the event with the local
    #                                      Qwen2.5-VL of classify_vlm_model, one copy per run
    landmark_threshold: float = 0.85     # CLIP threshold for places — conservative: 0.35
    #                                      gave false matches (cafe→Istanbul), and a wrong
    #                                      city is worse than unknown (backlog #11: a geo model)
    # F75: a single CLIP score does not separate a right city from a wrong one — on the
    # live collection the wrong ones scored 0.980 against 0.991 — so a match is
    # corroborated by its neighbours: within one directory, a city held by less than
    # `dominance` of at least `min` matches drops back to unknown (one card dump is one
    # trip). Raising either makes the rule fire less often.
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
    vlm_enabled: bool = False            # F37-B: deep tier — VLM 3-way (memory/document/product),
    #                                      needs the [vlm] extra, falls back to CLIP. The config key
    #                                      is `vlm.enabled`; this field stays because --deep and the
    #                                      UI checkbox force the tier through it (_legacy_naming_view)
    classify_vlm_model: str = DEFAULT_VLM_MODEL  # F37-B: classifier VLM. NOT vlm_model, which is
    #                                      event-naming/llava. Kept in sync with `vlm.model`
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


def _naming_from(raw: dict) -> NamingConfig:
    clip = raw.get("clip") or {}
    vlm = raw.get("local_vlm") or {}
    d = NamingConfig()
    return NamingConfig(
        provider=_as_provider(raw.get("provider", d.provider), d.provider),
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
    )


def _legacy_naming_view(naming: NamingConfig, vlm: VlmConfig) -> NamingConfig:
    """`naming.vlm_enabled`/`classify_vlm_model` held equal to the resolved `vlm:` (F102).

    The two fields stay on NamingConfig because `--deep` and the "Deep analysis (VLM)"
    checkbox force the tier for one run by replacing `cfg.naming.vlm_enabled` on their own
    copy of the config — that field, not `vlm.enabled`, is what the junk stage reads.
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
    detect: DetectConfig = field(default_factory=DetectConfig)  # F154: the detector runtime
    features: FeaturesConfig = field(default_factory=FeaturesConfig)  # F113: frame quality
    language: str = "en"  # folder/name language (ru|en|ja), normalized in load_config (F25/F27)
    log_level: str = "WARNING"  # DEBUG|INFO|WARNING|ERROR; validated in configure_logging (F52)
    logging: LoggingConfig = field(default_factory=LoggingConfig)  # F166: run-log pacing
    estimate: EstimateConfig = field(default_factory=EstimateConfig)  # F159: run budget
    raw: dict = field(default_factory=dict)  # the full YAML for future-phase sections


# --- F222: a stage that is skipped while its settings sit in the file ------------------
#
# The case is one real config.yaml holding
#
#     naming:    landmark_threshold: 0.50      # lowered from 0.85 for a measurement
#     features:  landmarks_verify: true        # switched on 2026-08-02 for the cascade
#
# and no `features.landmarks`, because that key did not exist when those lines were
# written. After F222 the stage stops running for that file, and its owner would find out
# from a missing result rather than from the program.
#
# The settings are deliberately NOT read as an intention: a program that infers consent
# from a leftover number is worse than one that asks. It says so out loud instead, once.
# What counts as a setting somebody CHOSE is a value that differs from the default: until
# 2026-08-08 the test was "is the key in the file", and the installer writes config.yaml
# from an example that spells every key out, so every fresh install got the note on every
# run. Keyed by stage rather than written as `if landmarks` because it is about any stage
# whose default changes under a file that already configures it.


@dataclass(frozen=True)
class StageSettings:
    """One stage, the config keys that belong to it, and how it is switched on.

    `keys` are `section.key` paths, read against the raw YAML and compared with the
    defaults.
    """

    stage: str
    keys: tuple[str, ...]
    # Catalog key of the sentence naming every way to switch the stage back on. Per stage,
    # because a flag, a config key and a checkbox are three different things.
    enable_key: str


STAGE_SETTINGS: tuple[StageSettings, ...] = (
    StageSettings(
        stage="landmarks",
        keys=("naming.landmark_threshold", "naming.landmark_group_min",
              "naming.landmark_group_dominance", "naming.landmarks_file",
              "features.landmarks_verify", "features.landmark_candidate_threshold"),
        enable_key="cli.run.enable_landmarks",
    ),
)

STAGE_SETTINGS_BY_STAGE: dict[str, StageSettings] = {
    entry.stage: entry for entry in STAGE_SETTINGS}

# A config nobody edited, built once: the yardstick for "did somebody choose this value".
_DEFAULT_CONFIG = Config()
# A key with no default at all (a section this Config does not carry) counts as chosen —
# the safe direction: a note too many is read, a note missing is not.
_NO_DEFAULT = object()


def _default_setting(path: str):
    """What that `section.key` is worth in a config nobody edited."""
    section, _, key = path.partition(".")
    return getattr(getattr(_DEFAULT_CONFIG, section, None), key, _NO_DEFAULT)


def configured_settings_of(cfg: Config, stage: str) -> tuple[str, ...]:
    """Which of that stage's settings this person actually chose. Usually none.

    A value equal to the default is documentation the product shipped, not a decision
    somebody made — see the F222 note above.
    """
    entry = STAGE_SETTINGS_BY_STAGE.get(stage)
    if entry is None:
        return ()
    raw = getattr(cfg, "raw", None)
    if not isinstance(raw, dict):
        return ()
    found: list[str] = []
    for path in entry.keys:
        section, _, key = path.partition(".")
        values = raw.get(section)
        if not isinstance(values, dict) or key not in values:
            continue
        default = _default_setting(path)
        if default is _NO_DEFAULT or values[key] != default:
            found.append(path)
    return tuple(found)


def skipped_stage_notes(cfg: Config, skipped: Sequence[str],
                        lang: i18n.Lang) -> list[str]:
    """One line per skipped stage this config still configures. Usually none.

    Both entry points call this — `sorta run` prints the lines, the web app carries them
    into the status of the run — so the two cannot say different things about one file.
    """
    notes: list[str] = []
    for stage in skipped:
        keys = configured_settings_of(cfg, stage)
        if not keys:
            continue
        entry = STAGE_SETTINGS_BY_STAGE[stage]
        notes.append(i18n.cli_text(
            "cli.run.stage_skipped_configured", lang,
            stage=i18n.stage_label(stage, lang), keys=", ".join(keys),
            how=i18n.cli_text(entry.enable_key, lang)))
    return notes


def vlm_allowed(cfg: Config) -> bool:
    """F145: may this run load the VLM at all? — `vlm.enabled`, the master switch.

    Every question the model is asked has a toggle of its own (`vlm.products`,
    `features.pets_verify`, `features.landmarks_verify`, `features.junk_rescue`), and until
    F145 each could raise the weights by itself: a run started WITHOUT deep analysis still
    loaded 20 GB because one subordinate key was true in config.yaml. Those keys decide
    WHAT to ask; this one decides whether there is anybody to ask.

    Read off `cfg.naming.vlm_enabled` and not `cfg.vlm.enabled`: the two agree after
    load_config, and that field is the effective per-run toggle.
    """
    return bool(getattr(getattr(cfg, "naming", None), "vlm_enabled", False))


def products_allowed(cfg: Config) -> bool:
    """F161: may this run ask the model what a frame IS? — the deep junk tier.

    The F145 rule with `vlm.products` as the subordinate key, and the master keeping the
    veto. Absent (an old config, or the settings object a measurement script builds by
    hand) is TRUE: the tier is what a master switch alone used to mean.
    """
    return vlm_allowed(cfg) and bool(
        getattr(getattr(cfg, "vlm", None), "products", True))


def detector_allowed(cfg: Config) -> bool:
    """F154: may this run load the object detector at all? — `detect.enabled`.

    The F145 hierarchy applied to the second kind of model, with a switch of its own and
    not `vlm_allowed`: a detector costs 83.8 ms and no VRAM to speak of against 0.78 s and
    20 GB, so the two decisions belong to the user separately.
    """
    return bool(getattr(getattr(cfg, "detect", None), "enabled", False))


def _apply_logging_config(cfg: LoggingConfig) -> None:
    """Hand the `logging:` section to the run log (F166).

    Pushed like `_apply_imaging_config` pushes the imaging keys: `runlog` is a leaf module
    everything else imports, so it cannot read the config back. Never fatal.
    """
    try:
        from .runlog import set_progress_interval

        set_progress_interval(cfg.progress_interval_sec)
    except Exception as exc:  # noqa: BLE001 — logging is never worth crashing over
        _log.warning("config: не удалось применить logging: %s", exc)


def _known(cls, raw: dict) -> dict:
    """Keep from raw only the declared fields of the dataclass cls.

    A section may carry keys a module reads straight off cfg.raw (faces.decode_workers) or
    keys of a future phase: they stay in Config.raw and must not break the constructor.
    """
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
        dedup=_dedup_from(_mapping(data.get("dedup"))),
        geo=GeoConfig(**_known(GeoConfig, data.get("geo") or {})),
        faces=FacesConfig(**_known(FacesConfig, data.get("faces") or {})),
        events=EventsConfig(**_known(EventsConfig, data.get("events") or {})),
        sort=SortConfig(**_known(SortConfig, data.get("sort") or {})),
        naming=_legacy_naming_view(_naming_from(data.get("naming") or {}), vlm),
        vlm=vlm,
        detect=_detect_from(data),
        features=_features_from(_mapping(data.get("features"))),
        language=i18n.normalize_lang(data.get("language")),
        log_level=str(data.get("log_level", "WARNING")),
        logging=LoggingConfig(**_known(LoggingConfig, data.get("logging") or {})),
        estimate=_estimate_from(_mapping(data.get("estimate"))),
        raw=data,
    )
    _apply_imaging_config(data.get("imaging") or {})
    _apply_logging_config(cfg.logging)
    # sources may be empty: the source is given positionally (sorta index <dir>).
    # The non-empty requirement is at the point of use (index / in-place sort).
    return cfg


# F67: imaging.py is a leaf module with no access to Config (decode_rgb_preview is called
# from pool workers that carry only a path), so the config file seeds its env vars — and
# only when they are NOT already set, so a variable exported in the shell still wins. That
# is the documented contract.
_IMAGING_ENV = {
    "preview_cache": "SORTA_PREVIEW_CACHE",
    "preview_dir": "SORTA_PREVIEW_DIR",
    "preview_max_edge": "SORTA_PREVIEW_MAX_EDGE",
    "preview_quality": "SORTA_PREVIEW_QUALITY",
    # F117: a ceiling in GB, 0 = unbounded (the behaviour since F67). The answer to a full
    # disk is a bound, not switching off a cache that pays for itself on every full run.
    "preview_cache_max_gb": "SORTA_PREVIEW_MAX_GB",
    # F74/F80: video tiles and the lightbox filmstrip, same leaf module, same rule.
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


# F104: words YAML reads as something other than a plain string. A saver that emits
# `model: no` and reads back `False` is a trap waiting for the one value that hits it.
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

    `key` is a top-level name (`language`) or one level of nesting (`vlm.enabled`) — the
    two shapes the settings column of the web app writes (F104). A line-level edit, not a
    YAML round-trip: `yaml.safe_dump` would throw the user's comments, ordering and blank
    lines away on the first change of a checkbox. A missing file is created.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    section, _dot, leaf = key.rpartition(".")
    scalar = _yaml_scalar(value)
    updated = (_set_in_section(text, section, leaf, scalar) if section
               else _set_top_level(text, leaf, scalar))
    p.write_text(updated, encoding="utf-8")


def _set_block_in_section(text: str, section: str, key: str,
                          block: Sequence[str]) -> str:
    """`_set_in_section` for a value that is a BLOCK rather than a scalar (F156).

    `block` is written relative to the key — its first line is `key:` with no indentation
    of its own — and is prefixed here with the indentation the file already uses. The old
    block is everything under its `key:` line that is blank or indented deeper, and that is
    the only thing removed: whatever stood ABOVE the key (thirty lines of comments, in
    `config.example.yaml`) is untouched.
    """
    lines = text.split("\n")
    header = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        head = text.rstrip("\n") + "\n" if text.strip() else ""
        fresh = "\n".join(f"  {line}" if line else "" for line in block)
        return f"{head}{section}:\n{fresh}\n"
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
            indent = match.group(1)
            stop = i + 1
            while stop < end and (not lines[stop].strip()
                                  or lines[stop].startswith(indent + " ")):
                stop += 1
            body = [f"{indent}{line}" if line else "" for line in block]
            return "\n".join(lines[:i] + body + lines[stop:])
        if lines[i].strip():
            last_filled = i
            if not lines[i].lstrip().startswith("#"):
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    body = [f"{indent}{line}" if line else "" for line in block]
    return "\n".join(lines[:last_filled + 1] + body + lines[last_filled + 1:])


def _saved_slices_block(slices: Sequence[SavedSlice]) -> list[str]:
    """The `saved_slices:` mapping as YAML lines, relative to its own key.

    An empty list is written as `saved_slices: {}` and not as an absent key: "pin
    nothing" is a real wish (`_as_saved_slices`), and an omitted key would come back as
    the three slices that ship — unpinning the last pin would silently restore them.
    """
    if not slices:
        return ["saved_slices: {}"]
    lines = ["saved_slices:"]
    for slice_ in slices:
        lines.append(f"  {_yaml_scalar(slice_.name)}:")
        lines.extend(f"    - {_yaml_scalar(phrase)}" for phrase in slice_.queries)
    return lines


def save_saved_slices(path: str | Path, slices: Sequence[SavedSlice]) -> None:
    """Persist `features.saved_slices` into config.yaml, preserving the rest of the file.

    F156: the pins live HERE and not in the index, which `reset` and any re-processing
    rebuild — a slice somebody named must not be one re-index away from gone. The phrases
    are written back exactly as typed, quoting being the only transformation: they go to a
    CLIP text tower, so a name like «горы» or a phrase with a colon in it has to survive
    the round trip unchanged.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    updated = _set_block_in_section(text, "features", "saved_slices",
                                    _saved_slices_block(slices))
    p.write_text(updated, encoding="utf-8")


def save_language(path: str | Path, lang: str) -> None:
    """Persist `language: <lang>` into config.yaml, normalized to ru|en|ja."""
    save_setting(path, "language", i18n.normalize_lang(lang))
