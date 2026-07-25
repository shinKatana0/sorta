"""F18: a shared image-decode layer + a bounded in-process cache.

Consolidates what used to be spread across four copy-pastes
(faces._decode_for_faces, landmarks.clip_classifier._load, dedup._phash_one,
sorter._make_thumbnail): lazy HEIF-opener registration + JPEG draft downscale
+ convert + "any error -> None". Faces (full-resolution decode for the
ArcFace crop) is deliberately NOT moved onto this module — it has its own branch
(faces._decode_for_faces). The other consumers use imaging.decode_rgb[_cached].

decode_rgb_cached caches the decode result (a small, max_edge-bounded image) —
it is the decode that is expensive, not storing the original on disk.

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
import os
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageOps

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


def preview_key(path: str | Path, mtime: float, size: int) -> str:
    """Stable cache key for (file, mtime, size).

    A changed file yields a changed key, so invalidation is free — the same
    principle as the in-process decode_rgb_cached key. Stale entries of the old key
    are simply never read again (preview_cache_clear removes them).
    """
    raw = f"{Path(os.path.abspath(path)).as_posix()}|{mtime}|{size}"
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
    file. Two workers on the same path at the same time is fine (last one wins), a
    per-key lock is not needed.
    """
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
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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
    """
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


def preview_cache_clear() -> None:
    """Remove the preview cache directory (tests, a future `sorta cache clear`)."""
    shutil.rmtree(preview_dir(), ignore_errors=True)
