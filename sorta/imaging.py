"""F18: a shared image-decode layer + a bounded in-process cache.

Consolidates what used to be spread across four copy-pastes
(faces._decode_for_faces, landmarks.clip_classifier._load, dedup._phash_one,
sorter._make_thumbnail): lazy HEIF-opener registration + JPEG draft downscale
+ convert + "any error -> None". Faces (full-resolution decode for the
ArcFace crop) is deliberately NOT moved onto this module — it has its own branch
(faces._decode_for_faces). The other consumers use imaging.decode_rgb[_cached].

decode_rgb_cached caches the decode result (a small, max_edge-bounded image) —
it is the decode that is expensive, not storing the original on disk.

F74 lets that same preview cache serve VIDEO files: decode_rgb_preview extracts one
frame through PyAV and stores it as an ordinary preview, so the UI gets tiles for
clips too. It stays inside the preview layer on purpose — no pipeline stage decodes
video (they all filter media_type = 'photo' in SQL) and none should start.

F80 widens that one frame into a filmstrip (video_filmstrip / video_frame): one tile
answers "what is this", six frames answer "is this mine, and was it shot there" —
which is the question that matters on a collection where whole countries are almost
only video. Same cache, same JPEG format, same key plus the frame index, and frame 0
is still the very frame F74 wrote, so nothing already cached is invalidated.

F67 adds a second, DISK-level layer on top of the same decode: decode_rgb_preview.
The same frame used to be decoded 3-5 times per run (CLIP in landmarks, CLIP/OCR/VLM
in junk, pHash in dedup) because decode_rgb_cached is bounded and per-process, and
the stages run one after another. The preview cache decodes once, writes a 1536px
JPEG next to nothing (a shared cache dir) and every later stage reads that instead
of the original (HEIC full decode 473 ms -> pHash from a preview 1.4 ms, measured
2026-07-25). It is lazy: no separate pass or command, whichever stage needs the
frame first creates the preview.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageOps

_log = logging.getLogger(__name__)

# LRU limit of the in-process decode_rgb_cached cache. Could be moved into config
# (imaging.cache_max_items) when consumers are wired up.
CACHE_MAX_ITEMS = 512

# JPEG draft decodes directly at a reduced scale (DCT scaling), but only down to
# the nearest power of two; we request with headroom so that after draft the exact
# thumbnail() almost always only shrinks rather than upscales.
_DRAFT_FACTOR = 2

# F48: the _DRAFT_FACTOR=2× headroom is a quality trade-off (draft is asked for
# larger than the final size so the exact thumbnail() can still polish with LANCZOS),
# but it can also FULLY negate the draft win at large max_edge. draft() picks the
# nearest power of two NOT SMALLER than the requested size: for a typical camera
# frame (~4000px) a request of max_edge*2=2560 does not pass the first halving
# threshold (4000/2=2000 < 2560) -> draft stays silent, the FULL frame is decoded
# (see the F48 profile — 315 ms/frame on the OCR path at max_edge=1280).
# A margin=1.0 request (no headroom) for the same frame passes the first halving
# (2000 >= 1280) -> ~4× fewer pixels decoded (F48 measurement: ~45 ms ->
# ~17 ms on a synthetic 4032x3024 JPEG). The parameter default is NOT changed (=
# _DRAFT_FACTOR) — existing consumers (thumbs in ui.py/sorter.py, VLM decode)
# behave identically; the aggressive margin is opt-in for consumers that do not
# care about sub-pixel downscale sharpness (OCR text_frac — only needs the text-box
# area, not the text itself).
_DRAFT_MARGIN_AGGRESSIVE = 1.0

_heif_lock = threading.Lock()
_heif_registered = False


def _ensure_heif_registered() -> None:
    """Register the pillow_heif opener once (lazily, thread-safe).

    Without the pillow_heif package, HEIC/HEIF stay unrecognized by Pillow —
    decode_rgb returns None on them, as before in all consumers.
    """
    global _heif_registered
    if _heif_registered:
        return
    with _heif_lock:
        if _heif_registered:
            return
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        _heif_registered = True


def decode_rgb(
    path: str | Path,
    max_edge: int | None = None,
    *,
    grayscale: bool = False,
    apply_orientation: bool = False,
    draft_margin: float = _DRAFT_FACTOR,
) -> Image.Image | None:
    """Decode path into a PIL Image (RGB or L), or None on any error.

    max_edge given -> the JPEG is decoded directly at a reduced scale
    (im.draft), then finished if needed with an exact thumbnail() down to
    max_edge on the longer side; max_edge=None -> full size.
    grayscale=True -> mode "L" (for phash), otherwise "RGB".
    apply_orientation=True -> the EXIF orientation is applied (exif_transpose).
    draft_margin (F48) — the draft() request multiplier relative to max_edge; the
    default preserves the previous behaviour for ALL existing callers (thumbs in
    ui.py/sorter.py, VLM decode in junk.py). A smaller value (down to 1.0, see
    _DRAFT_MARGIN_AGGRESSIVE) gives a more aggressive JPEG draft for consumers that
    do not need sub-pixel downscale sharpness — the final size is still driven
    exactly to max_edge by thumbnail(), and draft() is guaranteed never to return a
    frame SMALLER than requested.
    A decode error (corrupt/unrecognized file, missing path, HEIC without
    pillow-heif) does not raise — the contract of all current consumers.
    """
    _ensure_heif_registered()
    mode = "L" if grayscale else "RGB"
    try:
        with Image.open(path) as im:
            if max_edge is not None:
                try:
                    draft_edge = int(max_edge * draft_margin)
                    im.draft(mode, (draft_edge, draft_edge))
                except Exception:
                    pass
            # load() before any further operations — otherwise a repeated implicit
            # load() inside convert()/thumbnail() may fail on an already-closed fp
            # (the same trick as in dedup._phash_one).
            im.load()
            transposed: Image.Image = im
            if apply_orientation:
                transposed = ImageOps.exif_transpose(im)
            out = transposed.convert(mode)
            if max_edge is not None and max(out.size) > max_edge:
                out.thumbnail((max_edge, max_edge))
            return out
    except Exception:
        return None


_CacheKey = tuple[str, float, int | None, bool, bool]

_cache: OrderedDict[_CacheKey, Image.Image] = OrderedDict()
_cache_lock = threading.Lock()


def decode_rgb_cached(
    path: str | Path,
    mtime: float,
    max_edge: int | None = None,
    *,
    grayscale: bool = False,
    apply_orientation: bool = False,
) -> Image.Image | None:
    """decode_rgb with a bounded in-process LRU cache.

    The key is (path, mtime, max_edge, grayscale, apply_orientation): a change of
    mtime (file reindexed/modified) naturally invalidates the entry, since it yields
    a different key. The cache is bounded to CACHE_MAX_ITEMS entries — on overflow
    the least-recently-used one is evicted. None results (corrupt files) are NOT
    cached: the decode error itself is cheap, and holding a "forever None" in the
    cache for a file that mutates without changing mtime is risky.

    Thread-safety: the cache is under a Lock for reads (+move-to-end) and writes
    (+eviction); decode_rgb itself is called without holding the lock, so parallel
    calls with different paths do not block each other on decode, and the only
    possible race is "both missed and both decoded the same key" — harmless (last
    writer wins), see the thread-safety tests.
    """
    key: _CacheKey = (str(path), mtime, max_edge, grayscale, apply_orientation)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    result = decode_rgb(path, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if result is None:
        return None

    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ITEMS:
            _cache.popitem(last=False)
    return result


def cache_clear() -> None:
    """Clear the in-process decode_rgb_cached cache (tests, between CLI commands)."""
    with _cache_lock:
        _cache.clear()


# --- F67: the lazy disk preview cache ---------------------------------------
#
# Configured through env vars only, on purpose: config.py is outside this feature's
# ownership, the `imaging:` config section (preview_cache / preview_dir /
# preview_max_edge / preview_quality) comes later and keeps env as an override.
ENV_PREVIEW_CACHE = "SORTA_PREVIEW_CACHE"
ENV_PREVIEW_DIR = "SORTA_PREVIEW_DIR"
ENV_PREVIEW_MAX_EDGE = "SORTA_PREVIEW_MAX_EDGE"
ENV_PREVIEW_QUALITY = "SORTA_PREVIEW_QUALITY"

# 1536 on the long edge covers every consumer with headroom (OCR 1280, VLM 896,
# CLIP 448, pHash 96); q88 keeps a frame at ~150 KB and is visually lossless at
# those sizes (measured pHash drift vs a full decode: <= 2 bits at 512px already,
# against a near-duplicate threshold of 5).
PREVIEW_MAX_EDGE = 1536
PREVIEW_QUALITY = 88

_FALSE_VALUES = {"0", "false", "no", "off"}
_EXIF_ORIENTATION = 274


def _env_int(name: str, default: int) -> int:
    """Positive int from an env var; garbage/non-positive -> the default."""
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def preview_cache_enabled() -> bool:
    """SORTA_PREVIEW_CACHE=0 -> decode_rgb_preview degrades to plain decode_rgb."""
    return os.environ.get(ENV_PREVIEW_CACHE, "").strip().lower() not in _FALSE_VALUES


def preview_dir() -> Path:
    """Where the preview JPEGs live (SORTA_PREVIEW_DIR overrides).

    Default: %LOCALAPPDATA%\\sorta\\previews on Windows, ~/.cache/sorta/previews
    elsewhere — a user-level cache, never inside the sorted collection.
    """
    override = os.environ.get(ENV_PREVIEW_DIR, "").strip()
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "sorta" / "previews"
    return Path.home() / ".cache" / "sorta" / "previews"


def preview_max_edge() -> int:
    return _env_int(ENV_PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE)


def preview_quality() -> int:
    return _env_int(ENV_PREVIEW_QUALITY, PREVIEW_QUALITY)


def preview_key(path: str | Path, mtime: float, size: int, frame: int = 0) -> str:
    """Stable cache key for (file, mtime, size) — and, for video, a frame index.

    A changed file yields a changed key, so invalidation is free — the same
    principle as the in-process decode_rgb_cached key. Stale entries of the old key
    are simply never read again (preview_cache_clear removes them).

    F80: `frame` addresses one frame of a video filmstrip. Frame 0 keeps the key
    string EXACTLY as it was, so every preview already on disk (including every F74
    video tile) still hits — a suffix on frame 0 would silently obsolete the whole
    cache the day this feature landed.
    """
    raw = f"{Path(os.path.abspath(path)).as_posix()}|{mtime}|{size}"
    if frame:
        raw = f"{raw}|frame={frame}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _preview_path(key: str) -> Path:
    # Sharded by the first 2 hex chars: 37k files in a single NTFS directory
    # degrade noticeably on lookup.
    return preview_dir() / key[:2] / f"{key}.jpg"


def _peek(path: str | Path) -> tuple[tuple[int, int], int] | None:
    """(size, exif orientation) from the header alone, without decoding pixels."""
    _ensure_heif_registered()
    try:
        with Image.open(path) as im:
            size = im.size
            try:
                orientation = int(im.getexif().get(_EXIF_ORIENTATION) or 1)
            except Exception:
                orientation = 1
        return size, orientation
    except Exception:
        return None


def _render(
    img: Image.Image, max_edge: int | None, orientation: int,
    *, grayscale: bool, apply_orientation: bool,
) -> Image.Image:
    """Render a decoded frame the way decode_rgb would have returned it.

    Used only when the cache is unusable (nothing was stored to read back), so the
    call still returns the right frame instead of failing. The orientation is passed
    explicitly: a freshly decoded frame carries no exif of its own we can rely on,
    while the preview ON DISK does (see _write_preview).
    """
    out = img
    if apply_orientation and orientation != 1:
        out.getexif()[_EXIF_ORIENTATION] = orientation
        out = ImageOps.exif_transpose(out)
    out = out.convert("L" if grayscale else "RGB")  # convert() always returns a copy
    if max_edge is not None and max(out.size) > max_edge:
        out.thumbnail((max_edge, max_edge))
    return out


def _read_preview(
    dest: Path, max_edge: int | None, *, grayscale: bool, apply_orientation: bool,
) -> Image.Image | None:
    if not dest.exists():
        return None
    img = decode_rgb(dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if img is None:
        # A corrupt cache entry (bad sector, a write interrupted by a crash) must
        # not poison the file forever — drop it, the caller falls back to the source.
        # A spurious drop (a concurrent os.replace on Windows can make one open fail)
        # costs nothing but one regeneration.
        try:
            dest.unlink()
        except OSError:
            pass
    return img


def _write_preview(img: Image.Image, dest: Path, orientation: int) -> None:
    """Store img as the preview for dest. Any failure is silently ignored.

    Written to a temp file next to the target + os.replace: the pool has ~20 threads
    and readers of other stages run concurrently — nobody may ever observe a half
    file.

    Two workers on the same path at once is fine, but "last one wins" does NOT hold on
    Windows: os.replace onto a destination another thread has OPEN fails with
    PermissionError (WinError 5), and readers open the preview immediately after it
    appears. Swallowing that silently meant a hot path could end up with no cached
    file at all. The content is a pure function of the key, so an existing file is
    always as good as ours: skip up front, and treat a lost race as success.
    """
    if dest.exists():
        return
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        params: dict[str, Image.Exif] = {}
        if orientation != 1:
            # The preview is stored UNROTATED (consumers do not apply orientation),
            # but the tag is kept so a later apply_orientation=True read still works.
            exif = Image.Exif()
            exif[_EXIF_ORIENTATION] = orientation
            params["exif"] = exif
        img.save(tmp, "JPEG", quality=preview_quality(), **params)
        try:
            os.replace(tmp, dest)
        except OSError:
            if not dest.exists():
                raise  # a real failure (no space, no permission on the directory)
            tmp.unlink(missing_ok=True)  # someone else won the race — their file stands
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# --- F74: one extracted frame as the preview of a video ----------------------
#
# Same env-only configuration as the F67 block above (the `imaging:` config section
# comes later and keeps env as an override).
ENV_VIDEO_PREVIEWS = "SORTA_VIDEO_PREVIEWS"
ENV_VIDEO_WORKERS = "SORTA_VIDEO_WORKERS"
# F80: how many frames the lightbox filmstrip is made of. 1 is the documented way to
# switch the feature off — the UI then shows exactly the single F74 frame.
ENV_VIDEO_FRAMES = "SORTA_VIDEO_FRAMES"

# Kept deliberately next to the decode layer instead of read from Config: imaging is a
# leaf module with no access to Config (see the comment on ENV_PREVIEW_CACHE), and a
# guess-by-content probe on every non-photo would cost an open per file. Mirrors
# IndexConfig.extensions["video"].
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mts", ".m2ts", ".3gp", ".mkv")

# PyAV spawns its own decoder threads and the thumb pool already runs up to 8 decodes
# at once — together that oversubscribes the CPU. A 4K frame is also ~24 MB in RAM, so
# 8 parallel extractions would be ~200 MB of transient frames. 4 is the compromise.
VIDEO_WORKERS = 4

# The first frame is black surprisingly often (fade-in, an intro card), which makes a
# useless tile. ~1 s in is recognizable on virtually any clip; for a short one take 10%
# of the duration instead, so a 2-second clip is not seeked past its own end. No
# brightness analysis on purpose — the goal is a recognizable tile, not the best frame.
VIDEO_FRAME_SECONDS = 1.0
VIDEO_FRAME_FRACTION = 0.1

# F80: six frames is what it took, on a sample of the collection, to answer "is this
# Cuba" about as reliably as watching the clip — and it is still one screenful of dots
# in the lightbox. The last one stops short of the very end: clips fade out, and a
# truncated file breaks on its last packet more often than anywhere else.
VIDEO_FRAMES = 6
VIDEO_LAST_FRACTION = 0.95

# A seek lands on the keyframe at or before the target, and we decode forward from
# there. The cap is a safety belt for files with minutes between keyframes or with
# broken timestamps: an earlier frame beats decoding the whole clip for a thumbnail.
_VIDEO_MAX_DECODED_FRAMES = 300

_av_lock = threading.Lock()
_av_warned = False

_video_gate_lock = threading.Lock()
_video_semaphore: threading.Semaphore | None = None
_video_semaphore_slots = 0


def video_previews_enabled() -> bool:
    """SORTA_VIDEO_PREVIEWS=0 -> decode_rgb_preview returns None on video, as before."""
    return os.environ.get(ENV_VIDEO_PREVIEWS, "").strip().lower() not in _FALSE_VALUES


def video_workers() -> int:
    return _env_int(ENV_VIDEO_WORKERS, VIDEO_WORKERS)


def video_frames() -> int:
    """SORTA_VIDEO_FRAMES — frames per filmstrip; 1 degrades to the F74 single frame."""
    return _env_int(ENV_VIDEO_FRAMES, VIDEO_FRAMES)


def is_video_path(path: str | Path) -> bool:
    """True for the extensions we are willing to hand to PyAV (case-insensitive)."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _video_gate() -> threading.Semaphore:
    """The limit on CONCURRENT frame extractions.

    Deliberately around the extraction only, not around decode_rgb_preview as a whole:
    the photo path must not queue behind video decodes. Rebuilt when the configured
    number changes — in a run that never happens, but tests set the env per case.
    """
    global _video_semaphore, _video_semaphore_slots
    slots = video_workers()
    with _video_gate_lock:
        if _video_semaphore is None or _video_semaphore_slots != slots:
            _video_semaphore = threading.Semaphore(slots)
            _video_semaphore_slots = slots
        return _video_semaphore


def _import_av() -> ModuleType | None:
    """The PyAV module, or None (with a single warning) when it is not installed.

    Lazy, inside the call: importing av loads the FFmpeg libraries, and no command
    except the UI ever touches a video — `import sorta.imaging` must not pay for it.
    A missing package degrades to "no video previews", the same way HEIC degrades
    without pillow-heif, instead of breaking the caller.
    """
    global _av_warned
    try:
        import av
    except ImportError:
        with _av_lock:
            if not _av_warned:
                _av_warned = True
                _log.warning(
                    "imaging: пакет av не установлен — превью для видео недоступны")
        return None
    return av


def _duration_seconds(container: object, stream: object) -> float | None:
    """Length of the clip in seconds — from the stream, else the container, else None."""
    stream_duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if stream_duration is not None and time_base:
        return float(stream_duration * time_base)
    container_duration = getattr(container, "duration", None)
    if container_duration is not None:
        return float(container_duration) / 1_000_000  # av.time_base units
    return None


def _target_seconds(container: object, stream: object) -> float:
    """Where to look for the preview frame, in seconds from the start."""
    duration = _duration_seconds(container, stream)
    if duration is None or duration <= 0:
        return VIDEO_FRAME_SECONDS
    return min(VIDEO_FRAME_SECONDS, duration * VIDEO_FRAME_FRACTION)


def _frame_rotation(frame: object, stream: object) -> int:
    """Counter-clockwise degrees to apply so the clip stands the way a player shows it.

    Phone clips keep their rotation in the container display matrix, not in the pixels:
    without applying it every portrait video would show up lying on its side.
    """
    try:
        rotation = int(getattr(frame, "rotation", 0) or 0)
    except (TypeError, ValueError):
        rotation = 0
    if rotation:
        return rotation
    # Older containers carry the angle as a `rotate` metadata tag, and there the
    # convention is CLOCKWISE (rotate=90 -> a player turns the frame 90° CW).
    try:
        metadata = getattr(stream, "metadata", None) or {}
        return -int(float(metadata.get("rotate", 0)))
    except (TypeError, ValueError, AttributeError):
        return 0


def _rotate_frame(img: Image.Image, rotation: int) -> Image.Image:
    """Apply the container rotation. Pillow rotates counter-clockwise, as does PyAV."""
    normalized = rotation % 360
    if normalized == 0:
        return img
    return img.rotate(normalized, expand=True)


def _decode_at(container: Any, stream: Any, target: float) -> Any | None:
    """Seek to `target` seconds and decode forward to the first frame at/after it.

    Split out of _grab_frame for F80: a filmstrip is this loop run once per target
    inside ONE open container, and both paths must land on the same frame for the
    same second (frame 0 of the strip IS the F74 preview).
    """
    try:
        container.seek(int(target / stream.time_base), stream=stream)
    except Exception:
        # A container that cannot seek (fragmented, streamed, broken index) still
        # decodes from the start — we simply fall back to the first frame we get.
        pass
    frame = None
    for index, decoded in enumerate(container.decode(stream)):
        frame = decoded
        when = getattr(decoded, "time", None)
        if when is None or when >= target or index + 1 >= _VIDEO_MAX_DECODED_FRAMES:
            break
    return frame


def _grab_frame(av: ModuleType, path: str | Path) -> Image.Image | None:
    """Decode one representative frame of path, already rotated.

    The container is always closed (`with av.open`) — on a collection of thousands of
    clips a leaked descriptor per call would exhaust the process. Any AV failure
    propagates to _extract_video_frame, which turns it into None.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]  # IndexError on an audio-only file
        stream.thread_type = "AUTO"
        frame = _decode_at(container, stream, _target_seconds(container, stream))
        if frame is None:
            return None
        return _rotate_frame(frame.to_image(), _frame_rotation(frame, stream))


def _extract_video_frame(path: str | Path) -> Image.Image | None:
    """A preview-worthy frame of a video file, or None. Never raises."""
    av = _import_av()
    if av is None:
        return None
    with _video_gate():
        try:
            return _grab_frame(av, path)
        except Exception:
            # Corrupt / truncated / unsupported file, no video stream, any PyAV error:
            # the contract of decode_rgb_preview is None, never an exception.
            return None


def _video_preview(
    path: str | Path,
    mtime: float,
    size: int,
    max_edge: int | None,
    *,
    grayscale: bool,
    apply_orientation: bool,
) -> Image.Image | None:
    """decode_rgb_preview for a video: one extracted frame, in the very same cache.

    Same key (path+mtime+size), same directory, same JPEG format as for photos — the
    consumer must not need to know whether the tile came from a photo or from a clip.
    Unlike a photo, the frame is stored ALREADY rotated (orientation=1): the rotation
    comes from the container, so there is no exif on the source to defer it to.
    """
    if not video_previews_enabled():
        return None
    if not preview_cache_enabled():
        # Nothing may be written while the cache is off, but the frame is still worth
        # returning — the alternative is a video tile that is simply missing.
        frame = _extract_video_frame(path)
        if frame is None:
            return None
        return _render(
            frame, max_edge, 1, grayscale=grayscale, apply_orientation=apply_orientation)

    dest = _preview_path(preview_key(path, mtime, size))
    cached = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if cached is not None:
        return cached

    frame = _extract_video_frame(path)
    if frame is None:
        return None
    edge = preview_max_edge()
    if max(frame.size) > edge:
        frame.thumbnail((edge, edge))
    _write_preview(frame, dest, 1)
    # Read back what was written, for the same reason as on the photo path: a cold and
    # a warm call must return the same pixels.
    stored = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if stored is not None:
        return stored
    # The cache is unusable (read-only dir, full disk) — render from what we have.
    return _render(
        frame, max_edge, 1, grayscale=grayscale, apply_orientation=apply_orientation)


def decode_rgb_preview(
    path: str | Path,
    mtime: float,
    size: int,
    max_edge: int | None = None,
    *,
    grayscale: bool = False,
    apply_orientation: bool = False,
) -> Image.Image | None:
    """decode_rgb backed by a lazy disk cache of 1536px previews.

    The first stage that needs the frame decodes the original once and writes the
    preview; every later stage (pHash 96, CLIP 448, VLM 896, OCR 1280) decodes the
    small JPEG instead. Contract is that of decode_rgb — same size/mode for the same
    max_edge/grayscale/apply_orientation, None on an undecodable source.

    The cache is skipped (a direct decode_rgb, nothing written) when it is disabled,
    when the source is already no larger than preview_max_edge (a preview would be a
    copy, not a saving — PNG screenshots), and on any cache read/write failure
    (read-only dir, full disk): the call must degrade, never fail.

    max_edge=None on a cache hit returns the PREVIEW-sized frame, not the original
    resolution — this layer is for the small-frame consumers (all of them pass
    max_edge). Full resolution (faces) still goes through decode_rgb.

    F74: a video path is served by one frame extracted through PyAV (_video_preview),
    stored in the same cache under the same key. decode_rgb is NOT touched — it stays
    image-only, so the sorter thumbnails behave exactly as before.
    """
    if is_video_path(path):
        return _video_preview(
            path, mtime, size, max_edge,
            grayscale=grayscale, apply_orientation=apply_orientation)

    if not preview_cache_enabled():
        return decode_rgb(path, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)

    dest = _preview_path(preview_key(path, mtime, size))
    cached = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if cached is not None:
        return cached

    peeked = _peek(path)
    if peeked is None:  # unreadable source — decode_rgb gives the same None
        return decode_rgb(path, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    src_size, orientation = peeked
    if max(src_size) <= preview_max_edge():
        return decode_rgb(path, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)

    # draft_margin: the F48 trap, one level up. The default 2x margin asks draft() for
    # 2*1536=3072, which a typical ~4000px camera frame cannot satisfy by halving
    # (4000/2=2000 < 3072), so draft stays silent and the FULL frame is decoded. At
    # margin=1.0 the request is 1536, the first halving qualifies, and ~4x fewer pixels
    # are decoded — measured 363 -> 259 ms per 15.7 MP JPEG for a byte-identical
    # 1536x1157 result, since thumbnail() lands on the exact size either way.
    full = decode_rgb(path, preview_max_edge(),  # RGB, unrotated — as stored
                      draft_margin=_DRAFT_MARGIN_AGGRESSIVE)
    if full is None:
        return None
    _write_preview(full, dest, orientation)
    # Read back what we have just written instead of rendering `full` in memory: a
    # cold and a warm call must return the SAME pixels, otherwise a pHash would
    # depend on whether the cache happened to be warm. The extra decode is of a
    # 1536px JPEG — cheap next to the full-size decode just paid for.
    stored = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if stored is not None:
        return stored
    # The cache is unusable (read-only dir, full disk) — render from what we have.
    return _render(
        full, max_edge, orientation, grayscale=grayscale, apply_orientation=apply_orientation)


# --- F80: several frames of one clip, so it can be judged without playing it -
#
# Why not playback: 68% of the collection is HEVC, which Chrome/Firefox do not decode
# by default — a <video> tag would show a black rectangle on two clips out of three.
# Frames work on 100% of the files, because PyAV decodes what the browser will not.
#
# Cost on a synthetic 3840x2160 h264 clip, 10 s at 30 fps (measured 2026-07-26):
# a cold strip of 6 frames — 3.6 s in ONE container open (six opens would re-parse the
# index six times); the warm strip — 43 ms, and the single frame the lightbox actually
# asks for — 7 ms. Nothing of this is paid until a lightbox is opened.


def _filmstrip_targets(container: object, stream: object, count: int) -> list[float]:
    """The seconds to grab, ascending — targets[0] is EXACTLY the F74 frame.

    The strip is deliberately "the F74 frame plus count-1 positions spread over the
    clip" rather than an even split of the whole duration: frame 0 has to keep landing
    on the frame already in the cache, otherwise every tile in the UI is redrawn and
    the cache of a 227 GB collection is thrown away for cosmetics. The rest is
    count-1 evenly spaced fractions ending on VIDEO_LAST_FRACTION — at the default
    count that is exactly 20/40/60/80/95% of the duration.

    An unknown duration gives a single target: there is nothing to spread over, and
    decoding a clip to its end just to measure it costs more than the strip is worth.
    """
    first = _target_seconds(container, stream)
    duration = _duration_seconds(container, stream)
    if count <= 1 or duration is None or duration <= 0:
        return [first]
    fractions = [index / (count - 1) for index in range(1, count - 1)]
    fractions.append(VIDEO_LAST_FRACTION)
    return [first] + [duration * fraction for fraction in fractions]


def _grab_filmstrip(av: ModuleType, path: str | Path, count: int) -> list[Image.Image]:
    """Up to `count` frames of path, already rotated, in ONE container open.

    Opening a 4K clip six times means parsing its index six times; the seeks run
    inside a single `with av.open` instead. A target that lands on a frame already
    taken is skipped rather than returned twice — that is what turns "the clip is
    shorter than the strip" into fewer frames instead of six copies of its last one.

    A decode that blows up mid-strip keeps the frames collected so far: on a real
    collection a truncated tail is common, and half a strip still answers the question.
    """
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]  # IndexError on an audio-only file
        stream.thread_type = "AUTO"
        seen: set[float] = set()
        for target in _filmstrip_targets(container, stream, count):
            try:
                decoded = _decode_at(container, stream, target)
            except Exception:
                break
            if decoded is None:
                break
            when = getattr(decoded, "time", None)
            if when is not None:
                if when in seen:
                    continue
                seen.add(when)
            frames.append(_rotate_frame(decoded.to_image(), _frame_rotation(decoded, stream)))
    return frames


def _extract_filmstrip(path: str | Path, count: int) -> list[Image.Image]:
    """The frames of a filmstrip, or []. Never raises.

    The SAME semaphore as F74, held for the whole strip: six 4K frames are ~150 MB of
    transient pixels, so the number of clips decoded at once must not grow just
    because each of them now costs more.
    """
    av = _import_av()
    if av is None:
        return []
    with _video_gate():
        try:
            return _grab_filmstrip(av, path, count)
        except Exception:
            # Corrupt / truncated / unsupported file, no video stream, any PyAV error:
            # the contract of video_filmstrip is [], never an exception.
            return []


def _frame_path(path: str | Path, mtime: float, size: int, index: int) -> Path:
    return _preview_path(preview_key(path, mtime, size, index))


def _read_filmstrip(
    path: str | Path, mtime: float, size: int, count: int, max_edge: int | None,
    *, grayscale: bool, apply_orientation: bool,
) -> list[Image.Image]:
    """Frames 0..count-1 from the preview cache, stopping at the first gap."""
    frames: list[Image.Image] = []
    for index in range(count):
        img = _read_preview(
            _frame_path(path, mtime, size, index), max_edge,
            grayscale=grayscale, apply_orientation=apply_orientation)
        if img is None:
            break
        frames.append(img)
    return frames


def video_filmstrip(
    path: str | Path,
    mtime: float,
    size: int,
    count: int | None = None,
    max_edge: int | None = None,
    *,
    grayscale: bool = False,
    apply_orientation: bool = False,
) -> list[Image.Image]:
    """Several frames of a video, ascending in time. Never raises.

    Contract:
      - returns ready PIL images (as decode_rgb_preview does, not paths), oldest
        first; [] for anything that cannot be decoded — a corrupt or truncated file,
        no video stream, a missing path, a photo, no PyAV installed, videos switched
        off. An empty list is a normal answer, never an exception;
      - element 0 is byte-for-byte the frame F74 already serves, under the very same
        cache key, so this feature invalidates nothing;
      - `count` defaults to SORTA_VIDEO_FRAMES (6); <= 1 returns exactly the F74 list
        of one, which is the supported way to switch the filmstrip off;
      - a clip with fewer distinct frames than asked for returns fewer, down to [];
      - every frame is stored in the F67 preview cache as an ordinary JPEG, under the
        same key plus the frame index — a second call decodes nothing and does not
        open the container again.
    """
    if not is_video_path(path) or not video_previews_enabled():
        return []
    wanted = video_frames() if count is None else count
    if wanted <= 1:
        single = _video_preview(
            path, mtime, size, max_edge,
            grayscale=grayscale, apply_orientation=apply_orientation)
        return [single] if single is not None else []

    def rendered(frames: list[Image.Image]) -> list[Image.Image]:
        # Unlike a photo, a frame comes out of the container already rotated, so it
        # carries no orientation of its own to defer (orientation=1), exactly as F74.
        return [
            _render(frame, max_edge, 1,
                    grayscale=grayscale, apply_orientation=apply_orientation)
            for frame in frames
        ]

    if not preview_cache_enabled():
        # Nothing may be written while the cache is off, but the frames are still
        # worth returning — the same trade-off as on the F74 path.
        return rendered(_extract_filmstrip(path, wanted))

    cached = _read_filmstrip(
        path, mtime, size, wanted, max_edge,
        grayscale=grayscale, apply_orientation=apply_orientation)
    # ONE cached frame is not evidence of a built strip: F74 leaves exactly that
    # behind for every clip whose tile the grid has drawn, which is all of them. Two
    # or more can only have come from here, so they are the strip — including a short
    # clip that yielded fewer frames than asked for.
    if len(cached) >= 2:
        return cached

    extracted = _extract_filmstrip(path, wanted)
    if not extracted:
        return []
    edge = preview_max_edge()
    for index, frame in enumerate(extracted):
        if max(frame.size) > edge:
            frame.thumbnail((edge, edge))
        _write_preview(frame, _frame_path(path, mtime, size, index), 1)
    # Read back what was written, for the same reason as on the photo path: a cold and
    # a warm call must return the same pixels.
    stored = _read_filmstrip(
        path, mtime, size, wanted, max_edge,
        grayscale=grayscale, apply_orientation=apply_orientation)
    if len(stored) == len(extracted):
        return stored
    # The cache is unusable (read-only dir, full disk) — render from what we have.
    return rendered(extracted)


def video_frame(
    path: str | Path,
    mtime: float,
    size: int,
    index: int,
    max_edge: int | None = None,
    *,
    grayscale: bool = False,
    apply_orientation: bool = False,
) -> Image.Image | None:
    """Frame `index` of the filmstrip, or None when the clip has no such frame.

    This is what the UI lightbox calls, and it is why the strip is lazy: a frame is
    decoded only once it is actually looked at, so a grid of thousands of tiles never
    pays for anything beyond frame 0. A cached frame is read straight back; a miss
    builds the WHOLE strip once (one container open) and then reads it.

    index 0 is the F74 frame for a clip and the ordinary preview for a photo — the
    photo lightbox goes through this call completely unchanged.
    """
    if index < 0:
        return None
    if index == 0:
        return decode_rgb_preview(
            path, mtime, size, max_edge,
            grayscale=grayscale, apply_orientation=apply_orientation)
    if not is_video_path(path):
        return None  # a photo has exactly one frame
    if preview_cache_enabled() and video_previews_enabled():
        cached = _read_preview(
            _frame_path(path, mtime, size, index), max_edge,
            grayscale=grayscale, apply_orientation=apply_orientation)
        if cached is not None:
            return cached
    frames = video_filmstrip(
        path, mtime, size, max_edge=max_edge,
        grayscale=grayscale, apply_orientation=apply_orientation)
    return frames[index] if index < len(frames) else None


def preview_cache_clear() -> None:
    """Remove the preview cache directory (tests, a future `sorta cache clear`)."""
    shutil.rmtree(preview_dir(), ignore_errors=True)
