"""Load config.yaml into a typed configuration. stdlib + PyYAML only."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from . import i18n

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
    logger.setLevel(lvl_name)
    if not any(getattr(h, "_sorta_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._sorta_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
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
    # coordinate rounding for the in-memory online-geo cache: neighbouring photos in
    # the same ~cell = ONE request to Nominatim. 3 digits ≈ 110 m (default: balance of
    # speed and accuracy — on 4547 GPS photos ~721 requests ≈ ~12 min @1req/sec vs
    # ~26 min at 4 digits/~11 m). Fewer digits = faster, but coarser (may confuse districts).
    cache_coord_digits: int = 3


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


@dataclass(frozen=True)
class NamingConfig:
    """Phase 5 (F6): places without GPS, event names, junk. A flat view of the
    nested naming section of config.yaml (clip.*/local_vlm.*/claude.* — see load_config)."""
    provider: str = "template"           # template | local_vlm | claude
    landmark_threshold: float = 0.85     # CLIP threshold for places — conservative: 0.35
    #                                      gave false matches (cafe→Istanbul), and a wrong
    #                                      city is worse than unknown. Proper fix — a geo model (backlog #11)
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
    #                                      opt-in (needs the [vlm] extra); default OFF, graceful fallback to CLIP
    classify_vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # F37-B: classifier VLM (NOT vlm_model —
    #                                      that is for event-naming/llava; a separate field to avoid a collision)
    product_candidate_min: float = 0.4   # #14/V1: product-CLIP above this → the file goes to the VLM (candidate gate, not final)
    landmarks_file: str = "data/landmarks.yaml"
    clip_model: str = "ViT-L-14-quickgelu"  # the quickgelu variant for the openai weights (without it — a mismatch)
    clip_pretrained: str = "openai"
    clip_batch_size: int = 16
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
        max_samples=int(raw.get("max_samples", d.max_samples)),
        vlm_base_url=str(vlm.get("base_url", d.vlm_base_url)).rstrip("/"),
        vlm_model=str(vlm.get("model", d.vlm_model)),
        vlm_timeout=float(vlm.get("timeout", d.vlm_timeout)),
        claude_model=str(claude.get("model", d.claude_model)),
        claude_api_key_env=str(claude.get("api_key_env", d.claude_api_key_env)),
        claude_timeout=float(claude.get("timeout", d.claude_timeout)),
    )


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
        naming=_naming_from(data.get("naming") or {}),
        language=i18n.normalize_lang(data.get("language")),
        log_level=str(data.get("log_level", "WARNING")),
        raw=data,
    )
    # sources may be empty: the source is given positionally (sorta index <dir>).
    # The non-empty requirement is at the point of use (index / in-place sort).
    return cfg


def save_language(path: str | Path, lang: str) -> None:
    """Persist `language: <lang>` into config.yaml, preserving the rest of the file.

    A line-level replace (not a YAML round-trip) so user comments and formatting
    survive: replace the value of an existing top-level `language:` line, otherwise
    append the line; create the file if it does not exist. `lang` is normalized to a
    supported code (ru|en|ja) — an invalid value falls back to the i18n default.
    """
    lang = i18n.normalize_lang(lang)
    p = Path(path)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        pattern = re.compile(r"(?m)^language:.*$")
        if pattern.search(text):
            text = pattern.sub(f"language: {lang}", text, count=1)
        else:
            text = text.rstrip("\n") + f"\nlanguage: {lang}\n"
    else:
        text = f"language: {lang}\n"
    p.write_text(text, encoding="utf-8")
