"""F18: a shared image-decode layer + a bounded in-process cache.

Consolidates what used to be spread across four copy-pastes
(faces._decode_for_faces, landmarks.clip_classifier._load, dedup._phash_one,
sorter._make_thumbnail): lazy HEIF-opener registration + JPEG draft downscale
+ convert + "any error -> None". Faces (full-resolution decode for the
ArcFace crop) is deliberately NOT moved onto this module — it has its own branch
(faces._decode_for_faces). The other consumers use imaging.decode_rgb[_cached].

decode_rgb_cached caches the decode result (a small, max_edge-bounded image) —
it is the decode that is expensive, not storing the original on disk.
"""
from __future__ import annotations

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
