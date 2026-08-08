"""F18: a shared image-decode layer + a bounded in-process cache.

Lazy HEIF-opener registration, JPEG draft downscale, convert, "any error -> None".
Faces is deliberately NOT on this module — the full-resolution decode for the ArcFace
crop has its own branch (faces._decode_for_faces). Three layers, cheapest first:
`decode_rgb_cached`, a bounded per-process LRU; `decode_rgb_preview` (F67), a lazy DISK
cache of 1536px JPEGs; `video_filmstrip`/`video_frame` (F74, F80), the same cache
serving clips one frame per key+index through PyAV. Video stays inside this layer on
purpose: no pipeline stage decodes video (they all filter media_type = 'photo' in SQL).

F67 is the one that pays: a frame used to be decoded 3-5 times per run (CLIP in
landmarks, CLIP/OCR/VLM in junk, pHash in dedup), and now the first stage that needs it
writes a preview the rest read — HEIC full decode 473 ms -> pHash off a preview 1.4 ms,
measured 2026-07-25.
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

CACHE_MAX_ITEMS = 512  # LRU limit of the in-process decode_rgb_cached cache

# JPEG draft decodes at a reduced scale (DCT), but only down to the nearest power of two;
# the headroom keeps the exact thumbnail() afterwards a shrink, never an upscale.
_DRAFT_FACTOR = 2

# F48: that headroom can FULLY negate the draft win at a large max_edge. For a ~4000px
# frame a request of max_edge*2=2560 misses the first halving (2000 < 2560), draft stays
# silent and the FULL frame is decoded — 315 ms/frame on the OCR path at max_edge=1280.
# At margin=1.0 it passes: ~4× fewer pixels, ~45 ms -> ~17 ms on a 4032x3024 JPEG.
_DRAFT_MARGIN_AGGRESSIVE = 1.0

_heif_lock = threading.Lock()
_heif_registered = False


def _ensure_heif_registered() -> None:
    """Register the pillow_heif opener once; without it HEIC decodes to None."""
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

    With `max_edge` the JPEG is decoded at a reduced scale (im.draft) and finished with
    an exact thumbnail(); `draft_margin` (F48) is that request multiplier, and draft()
    never returns a frame SMALLER than asked for.
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
            # load() first: a repeated implicit load() inside convert()/thumbnail()
            # may fail on an already-closed fp (the trick dedup._phash_one used).
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

    The key carries mtime, so a modified file invalidates itself. None is NOT cached: a
    "forever None" for a file that mutates without changing mtime is a trap. decode_rgb
    runs OUTSIDE the lock, so the only race is two decodes of one key — last writer wins.
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


# --- F67: the lazy disk preview cache (env vars only: imaging cannot see Config) ---
ENV_PREVIEW_CACHE = "SORTA_PREVIEW_CACHE"
ENV_PREVIEW_DIR = "SORTA_PREVIEW_DIR"
ENV_PREVIEW_MAX_EDGE = "SORTA_PREVIEW_MAX_EDGE"
ENV_PREVIEW_QUALITY = "SORTA_PREVIEW_QUALITY"
ENV_PREVIEW_MAX_GB = "SORTA_PREVIEW_MAX_GB"

# 1536 on the long edge covers every consumer with headroom (OCR 1280, VLM 896, CLIP
# 448, pHash 96); q88 keeps a frame at ~150 KB, with a measured pHash drift against a
# full decode of <= 2 bits at 512px already, against a near-duplicate threshold of 5.
PREVIEW_MAX_EDGE = 1536
PREVIEW_QUALITY = 88

# 0 = no ceiling, as the cache behaved from F67 on. The answer to "the disk filled up"
# is a bound rather than switching the cache off: 38 485 files took 12 GB at the
# measured ~150 KB each, which extrapolates to ~45 GB at 300k and ~75 GB at 500k.
PREVIEW_MAX_GB = 0.0

# Checking the ceiling costs a walk of the whole directory, so it cannot run per write.
# Every 512th is ~75 MB between checks at the measured 150 KB each.
_EVICT_EVERY_N_WRITES = 512

_FALSE_VALUES = {"0", "false", "no", "off"}
_EXIF_ORIENTATION = 274

_writes_since_evict = 0
_evict_lock = threading.Lock()


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
    """Where the preview JPEGs live — a user-level cache, never inside the collection."""
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


def preview_cache_max_gb() -> float:
    """Ceiling in GB, 0 = unbounded. Garbage falls back: a typo may not fail a run."""
    try:
        value = float(os.environ.get(ENV_PREVIEW_MAX_GB, "").strip())
    except ValueError:
        return PREVIEW_MAX_GB
    return value if value > 0 else PREVIEW_MAX_GB


def preview_cache_size() -> tuple[int, int]:
    """(files, bytes) — one walker for the CLI, the UI and eviction; missing dir = (0, 0)."""
    directory = preview_dir()
    if not directory.exists():
        return 0, 0
    files = 0
    total = 0
    for entry in directory.rglob("*.jpg"):
        try:
            total += entry.stat().st_size
        except OSError:
            continue  # evicted or replaced under us — it is not in the cache any more
        files += 1
    return files, total


def preview_cache_evict(max_bytes: int | None = None) -> tuple[int, int]:
    """Delete the least recently USED previews until the cache fits. Returns (files, bytes).

    Least recently used, not oldest: the read path opens these files, so atime tracks
    what is in play. Windows updates atime lazily, so mtime is the fallback. Purging by
    AGE is deliberately not what happens — the ceiling is about disk, nothing else.
    """
    if max_bytes is None:
        max_bytes = int(preview_cache_max_gb() * 1e9)
    if max_bytes <= 0:
        return 0, 0
    directory = preview_dir()
    if not directory.exists():
        return 0, 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for entry in directory.rglob("*.jpg"):
        try:
            st = entry.stat()
        except OSError:
            continue
        used = max(st.st_atime, st.st_mtime)
        entries.append((used, st.st_size, entry))
        total += st.st_size
    if total <= max_bytes:
        return 0, 0
    removed_files = 0
    removed_bytes = 0
    for _used, size, entry in sorted(entries):  # oldest use first
        if total <= max_bytes:
            break
        try:
            entry.unlink()
        except OSError:
            continue  # a reader holds it open (Windows); the next pass will get it
        total -= size
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


def _note_preview_write() -> None:
    """Count a stored preview; every Nth evicts OUTSIDE the lock, never serializing."""
    global _writes_since_evict
    if preview_cache_max_gb() <= 0:
        return
    with _evict_lock:
        _writes_since_evict += 1
        if _writes_since_evict < _EVICT_EVERY_N_WRITES:
            return
        _writes_since_evict = 0
    preview_cache_evict()


def preview_key(path: str | Path, mtime: float, size: int, frame: int = 0) -> str:
    """Stable cache key for (file, mtime, size) — and, for video, a frame index.

    F80: frame 0 keeps the key EXACTLY as it was; a suffix would have obsoleted the cache.
    """
    raw = f"{Path(os.path.abspath(path)).as_posix()}|{mtime}|{size}"
    if frame:
        raw = f"{raw}|frame={frame}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _preview_path(key: str) -> Path:
    # Sharded by 2 hex chars: 37k files in one NTFS directory degrade on lookup.
    return preview_dir() / key[:2] / f"{key}.jpg"


# F210: this is a user-level directory of decoded photographs, one of which may be a
# passport, and the umask default would make it 0755 on Linux — readable by every other
# local account. Windows ignores the mode.
_PREVIEW_DIR_MODE = 0o700


def _make_preview_dir(directory: Path) -> None:
    """Create a preview directory and the cache root above it, private to this user.

    Separately, because `Path.mkdir(parents=True)` gives the PARENTS the default mode and
    a 0700 shard inside a 0755 root protects nothing.
    """
    root = preview_dir()
    root.mkdir(parents=True, exist_ok=True, mode=_PREVIEW_DIR_MODE)
    if directory != root:
        directory.mkdir(parents=True, exist_ok=True, mode=_PREVIEW_DIR_MODE)


# Frames are written from 0 up and contiguously, so a hole normally means the end of the
# strip — normally, because eviction removes single files by last use, and stopping at
# the first hole would leave the tail of a reel on disk. So the walk checks the whole
# configured strip and goes on while frames keep being found; this cap bounds it.
_PREVIEW_DELETE_MAX_FRAMES = 1024


def _unlink_preview(dest: Path) -> bool:
    """Remove one preview file; True when this call removed it. Never raises."""
    try:
        dest.unlink()
        return True
    except OSError as exc:
        _log.debug("imaging: превью %s не удалено: %s", dest, exc)
        return False


def preview_delete(path: str | Path, mtime: float, size: int) -> int:
    """Delete every cached preview of one file. Returns how many were removed.

    F210 — the derivative does not outlive the original. The key hashes (path, mtime,
    size), so once the file is gone none of the three is readable: the caller has to ask
    while the `files` row still stands. Every FRAME, not frame 0. Never raises.
    """
    # At least the DEFAULT number of frames even when fewer are configured now: a strip
    # written before somebody lowered SORTA_VIDEO_FRAMES must still be removed whole.
    floor = max(video_frames(), VIDEO_FRAMES) if is_video_path(path) else 1
    removed = 0
    index = 0
    while index < _PREVIEW_DELETE_MAX_FRAMES:
        dest = _preview_path(preview_key(path, mtime, size, index))
        found = dest.exists()
        if found and _unlink_preview(dest):
            removed += 1
        index += 1
        if not found and index >= floor:
            break
    return removed


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
    """decode_rgb's output shape, for when the cache is unusable.

    The orientation is passed in: a decoded frame has no exif to read it off.
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
        # A corrupt entry must not poison the file forever — drop it and let the caller
        # fall back to the source. A spurious drop costs one regeneration.
        try:
            dest.unlink()
        except OSError:
            pass
    return img


def _write_preview(img: Image.Image, dest: Path, orientation: int) -> None:
    """Store img as the preview for dest. Any failure is silently ignored.

    Temp file + os.replace: the pool has ~20 threads and readers of other stages run
    concurrently, so nobody may observe a half file. "Last one wins" does NOT hold on
    Windows — os.replace onto a destination another thread has OPEN fails with
    PermissionError (WinError 5) — so skip up front and treat a lost race as success.
    """
    if dest.exists():
        return
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        _make_preview_dir(dest.parent)
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
        _note_preview_write()
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# --- F74: one extracted frame as the preview of a video, same env-only config ---
ENV_VIDEO_PREVIEWS = "SORTA_VIDEO_PREVIEWS"
ENV_VIDEO_WORKERS = "SORTA_VIDEO_WORKERS"
# F80: frames per lightbox filmstrip; 1 is the documented way to switch it off.
ENV_VIDEO_FRAMES = "SORTA_VIDEO_FRAMES"

# By extension, not content — a probe costs an open per file. Mirrors IndexConfig.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mts", ".m2ts", ".3gp", ".mkv")

# PyAV spawns its own decoder threads and the thumb pool already runs up to 8 decodes at
# once; a 4K frame is ~24 MB, so 8 parallel extractions would be ~200 MB of transients.
VIDEO_WORKERS = 4

# The first frame is black surprisingly often (fade-in, an intro card). ~1 s in is
# recognizable on almost any clip; a short one takes 10% of its duration, so a 2-second
# clip is not seeked past its end. No brightness analysis: a recognizable tile, not the
# best frame.
VIDEO_FRAME_SECONDS = 1.0
VIDEO_FRAME_FRACTION = 0.1

# F80: six frames answered "is this Cuba" about as reliably as watching the clip, on a
# sample of the collection. The last stops short of the very end: clips fade out, and a
# truncated file breaks on its last packet more often than anywhere else.
VIDEO_FRAMES = 6
VIDEO_LAST_FRACTION = 0.95

# A seek lands on the keyframe at or before the target and decoding runs forward. The cap
# covers minutes between keyframes or broken timestamps: an earlier frame beats decoding
# a whole clip for a thumbnail.
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
    """Concurrency limit for the extraction alone — the photo path must not queue."""
    global _video_semaphore, _video_semaphore_slots
    slots = video_workers()
    with _video_gate_lock:
        if _video_semaphore is None or _video_semaphore_slots != slots:
            _video_semaphore = threading.Semaphore(slots)
            _video_semaphore_slots = slots
        return _video_semaphore


def _import_av() -> ModuleType | None:
    """The PyAV module, or None (with a single warning) when it is not installed.

    Lazy: importing av loads FFmpeg and only the UI touches a video. A missing package
    degrades to "no video previews", as HEIC does without pillow-heif.
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
    """CCW degrees so the clip stands up: phone rotation lives in the display matrix."""
    try:
        rotation = int(getattr(frame, "rotation", 0) or 0)
    except (TypeError, ValueError):
        rotation = 0
    if rotation:
        return rotation
    # Older containers carry the angle as a `rotate` tag, where the convention is
    # CLOCKWISE (rotate=90 -> a player turns the frame 90° CW).
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
    """Seek to `target` seconds, decode forward to the first frame at/after it.

    Shared with the filmstrip, whose frame 0 IS the F74 preview.
    """
    try:
        container.seek(int(target / stream.time_base), stream=stream)
    except Exception:
        # A container that cannot seek still decodes from the start.
        pass
    frame = None
    for index, decoded in enumerate(container.decode(stream)):
        frame = decoded
        when = getattr(decoded, "time", None)
        if when is None or when >= target or index + 1 >= _VIDEO_MAX_DECODED_FRAMES:
            break
    return frame


def _grab_frame(av: ModuleType, path: str | Path) -> Image.Image | None:
    """One representative frame, rotated; AV failures propagate to _extract_video_frame."""
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
            # Corrupt, truncated, unsupported, no video stream, any PyAV error: the
            # contract of decode_rgb_preview is None, never an exception.
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

    Same key, directory and format as a photo, but stored ALREADY rotated: the rotation
    comes from the container, not from an exif tag.
    """
    if not video_previews_enabled():
        return None
    if not preview_cache_enabled():
        # Nothing may be written while the cache is off, but a tile is still worth it.
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
    stored = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if stored is not None:
        return stored
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
    preview; every later stage (pHash 96, CLIP 448, VLM 896, OCR 1280) reads the small
    JPEG. The cache is skipped — a direct decode_rgb, nothing written — when it is off,
    when the source is no larger than preview_max_edge, and on any read/write failure.
    max_edge=None on a cache HIT returns the PREVIEW-sized frame, not the original.
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

    # The F48 trap one level up: the default 2x margin asks draft() for 2*1536=3072, which
    # a ~4000px frame cannot satisfy by halving. At margin=1.0 it qualifies — measured
    # 363 -> 259 ms per 15.7 MP JPEG for a byte-identical 1536x1157 result.
    full = decode_rgb(path, preview_max_edge(),  # RGB, unrotated — as stored
                      draft_margin=_DRAFT_MARGIN_AGGRESSIVE)
    if full is None:
        return None
    _write_preview(full, dest, orientation)
    # Read back instead of rendering `full` in memory: a cold and a warm call must return
    # the SAME pixels, or a pHash would depend on whether the cache happened to be warm.
    stored = _read_preview(
        dest, max_edge, grayscale=grayscale, apply_orientation=apply_orientation)
    if stored is not None:
        return stored
    # The cache is unusable (read-only dir, full disk) — render from what we have.
    return _render(
        full, max_edge, orientation, grayscale=grayscale, apply_orientation=apply_orientation)


# --- F80: several frames of one clip, so it can be judged without playing it -
# Not playback: 68% of the collection is HEVC, which Chrome/Firefox do not decode by
# default, so a <video> tag would show a black rectangle on two clips out of three.
# Cost on a synthetic 3840x2160 h264 clip, 10 s at 30 fps (measured 2026-07-26): a cold
# strip of 6 frames — 3.6 s in ONE container open (six opens would re-parse the index six
# times); the warm strip — 43 ms; the single frame the lightbox asks for — 7 ms.


def _filmstrip_targets(container: object, stream: object, count: int) -> list[float]:
    """The seconds to grab, ascending — targets[0] is EXACTLY the F74 frame.

    Not an even split of the duration: frame 0 has to keep landing on the frame already
    cached, or the cache of a 227 GB collection is thrown away for cosmetics. The rest are
    evenly spaced fractions ending on VIDEO_LAST_FRACTION — by default 20/40/60/80/95%.
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

    Opening a 4K clip six times means parsing its index six times. A target landing on a
    frame already taken is skipped: "shorter than the strip" gives fewer frames, not six
    copies of the last one. A mid-strip failure keeps what it has.
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

    The SAME F74 semaphore, held for the whole strip: six 4K frames are ~150 MB.
    """
    av = _import_av()
    if av is None:
        return []
    with _video_gate():
        try:
            return _grab_filmstrip(av, path, count)
        except Exception:
            # Corrupt, truncated, unsupported, no video stream, any PyAV error: the
            # contract of video_filmstrip is [], never an exception.
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

    Ready PIL images, oldest first; [] for anything undecodable is a normal answer, never
    an exception. Element 0 is byte-for-byte the frame F74 already serves under the same
    key. `count` defaults to SORTA_VIDEO_FRAMES (6); <= 1 switches the strip off.
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
        # A frame leaves the container already rotated — nothing to defer (as F74).
        return [
            _render(frame, max_edge, 1,
                    grayscale=grayscale, apply_orientation=apply_orientation)
            for frame in frames
        ]

    if not preview_cache_enabled():
        return rendered(_extract_filmstrip(path, wanted))  # nothing written, still shown

    cached = _read_filmstrip(
        path, mtime, size, wanted, max_edge,
        grayscale=grayscale, apply_orientation=apply_orientation)
    # ONE cached frame is not evidence of a built strip: F74 leaves exactly that behind
    # for every clip whose tile the grid has drawn. Two or more can only be from here.
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
    stored = _read_filmstrip(
        path, mtime, size, wanted, max_edge,
        grayscale=grayscale, apply_orientation=apply_orientation)
    if len(stored) == len(extracted):
        return stored
    return rendered(extracted)  # the cache is unusable — render from what we have


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

    Why the strip is lazy: a grid of thousands of tiles never pays beyond frame 0, and a
    miss builds the WHOLE strip once, in one container open.
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
