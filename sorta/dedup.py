"""FR-1: exact-hash deduplication + near-duplicate report (pHash).

Files are not deleted: exact duplicates are marked with dup_of, near-duplicates
are only grouped into a report (near_duplicate_groups) without writing to the DB.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

from .hashing import resolve_workers

if TYPE_CHECKING:
    from .config import Config

try:
    import imagehash  # type: ignore
    from PIL import Image
    _PHASH = True
except ImportError:  # pragma: no cover
    _PHASH = False

try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover — HEIC is silently skipped without the package
    pass

_PHASH_BATCH = 200
# imagehash.phash (hash_size=8, highfreq_factor=4) itself shrinks the image to
# 32×32 before the DCT — decoding 12 MP for that is wasteful. 96 — headroom above
# 32 so the downscale does not lose sharpness before imagehash's final resample.
_PHASH_DECODE = 96


def _phash_one(path: str) -> str | None:
    if not _PHASH:
        return None
    try:
        with Image.open(path) as img:
            img.draft("L", (_PHASH_DECODE, _PHASH_DECODE))  # JPEG DCT scaling during decode
            img.load()  # required before thumbnail(), else a repeated load() fails on fp=None
            img.thumbnail((_PHASH_DECODE, _PHASH_DECODE))
            return str(imagehash.phash(img))
    except Exception:
        return None


def compute_phashes(
    cfg: "Config", conn: sqlite3.Connection,
    progress: Callable[[int, int | None], None] | None = None,
) -> int:
    """Compute pHash for files without one (incremental). Returns the number computed.

    Moved out of the hot `index()` path (F11): the pHash decode is done at a reduced
    resolution (Image.draft + thumbnail before imagehash.phash), in parallel (the
    same ThreadPoolExecutor as in indexer.index — Pillow decoding releases the GIL).
    HEIC — if pillow-heif is installed, otherwise such files are silently skipped
    (phash stays NULL, as before).
    """
    if not _PHASH:
        return 0
    rows = conn.execute(
        """SELECT id, path FROM files
           WHERE phash IS NULL AND error IS NULL AND media_type = 'photo'"""
    ).fetchall()
    total = len(rows)
    if total == 0:
        return 0
    computed = 0
    processed = 0
    workers = resolve_workers(cfg.raw)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, total, _PHASH_BATCH):
            batch = rows[start:start + _PHASH_BATCH]
            results = list(pool.map(_phash_one, [r["path"] for r in batch]))
            with conn:  # one transaction per batch — as in index()
                for r, ph in zip(batch, results):
                    if ph is not None:
                        conn.execute("UPDATE files SET phash = ? WHERE id = ?", (ph, r["id"]))
                        computed += 1
            processed += len(batch)
            if progress:
                progress(processed, total)
    return computed


def _canonical(rows: list[sqlite3.Row], strategy: str) -> sqlite3.Row:
    if strategy == "prefer_exif_then_largest":
        return sorted(
            rows,
            key=lambda r: (r["taken_at_source"] != "exif", -(r["size"] or 0), r["id"]),
        )[0]
    # largest — fallback strategy
    return sorted(rows, key=lambda r: (-(r["size"] or 0), r["id"]))[0]


def assign_duplicates(conn: sqlite3.Connection, strategy: str = "prefer_exif_then_largest") -> int:
    """Returns the number of files marked as duplicates."""
    marked = 0
    groups = conn.execute(
        """SELECT hash FROM files
           WHERE hash IS NOT NULL AND error IS NULL
           GROUP BY hash HAVING COUNT(*) > 1"""
    ).fetchall()
    with conn:
        for (h,) in [(g["hash"],) for g in groups]:
            rows = conn.execute(
                "SELECT id, size, taken_at_source FROM files WHERE hash = ? AND error IS NULL",
                (h,),
            ).fetchall()
            canon = _canonical(rows, strategy)
            for r in rows:
                is_dup = r["id"] != canon["id"]
                conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                             (canon["id"] if is_dup else None, r["id"]))
                marked += is_dup
    return marked


def hamming(a: str, b: str) -> int:
    """Bitwise Hamming distance between hex pHash strings of equal length."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _band_ranges(bits: int, bands: int) -> list[tuple[int, int]]:
    """Split `bits` bit positions into `bands` disjoint contiguous ranges.

    Returns (offset, width) pairs that cover [0, bits) exactly once, with widths
    differing by at most 1 — 64 bits over 6 bands gives 11,11,11,11,10,10.

    F66: the previous split was done on hex characters with a ceil step, so 16
    characters over 6 bands gave five bands of 3 characters and a last one of a
    single character. That tiny band put ~1/16 of the collection into each of its
    buckets and produced almost all of the candidate pairs.
    """
    base, rem = divmod(bits, bands)
    ranges: list[tuple[int, int]] = []
    offset = 0
    for i in range(bands):
        width = base + (1 if i < rem else 0)
        ranges.append((offset, width))
        offset += width
    return ranges


def near_duplicate_groups(
    conn: sqlite3.Connection, max_distance: int = 5,
) -> list[list[sqlite3.Row]]:
    """Near-duplicate groups among canonical files by pHash.

    Pairs with a Hamming distance <= max_distance are merged into groups
    (union-find, i.e. a group is transitive: A~B and B~C put A and C together,
    even if dist(A, C) > the threshold — for a report this is expected).

    Candidates are found via band buckets (pigeonhole): the hash is cut into
    max_distance+1 parts of near-equal BIT width (_band_ranges); a pair within the
    threshold shares at least one part — a full O(n^2) scan is not needed. Hashes of
    different bit length are never compared (the length is part of the bucket key).

    F66 — three things keep this cheap on tens of thousands of files: each pHash is
    parsed into an int once per file (not once per pair), identical pHashes are
    collapsed into the group up front so the buckets hold one representative per
    distinct value instead of a whole clique, and the distance itself is
    (a ^ b).bit_count() over those ints instead of re-parsing hex strings.

    Exact duplicates (dup_of IS NOT NULL) and errored files are excluded.
    Writes nothing to the DB. Groups are sorted by the path of the first file,
    and within a group — by descending size.
    """
    rows = conn.execute(
        """SELECT id, path, size, phash FROM files
           WHERE phash IS NOT NULL AND error IS NULL AND dup_of IS NULL"""
    ).fetchall()

    parent = {r["id"]: r["id"] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Files sharing the exact same pHash are always one group — union them right away
    # and keep a single representative per distinct (bit length, value).
    by_value: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        h = r["phash"]
        by_value.setdefault((len(h) * 4, int(h, 16)), []).append(r["id"])
    for ids in by_value.values():
        root = ids[0]
        for other in ids[1:]:
            parent[other] = root

    bands = max(1, max_distance + 1)
    ranges_by_len: dict[int, list[tuple[int, int]]] = {}
    buckets: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for (bit_len, value), ids in by_value.items():
        ranges = ranges_by_len.get(bit_len)
        if ranges is None:
            ranges = ranges_by_len[bit_len] = _band_ranges(bit_len, bands)
        rep = ids[0]
        for bi, (offset, width) in enumerate(ranges):
            part = (value >> offset) & ((1 << width) - 1)
            buckets.setdefault((bit_len, bi, part), []).append((value, rep))

    for items in buckets.values():
        n = len(items)
        if n < 2:
            continue
        for i in range(n - 1):
            va, ia = items[i]
            ra = find(ia)  # stays the root: unions below only re-point other roots at it
            for j in range(i + 1, n):
                vb, ib = items[j]
                rb = find(ib)
                if ra != rb and (va ^ vb).bit_count() <= max_distance:
                    parent[rb] = ra

    grouped: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        grouped.setdefault(find(r["id"]), []).append(r)
    result = [sorted(g, key=lambda r: (-(r["size"] or 0), r["path"]))
              for g in grouped.values() if len(g) > 1]
    return sorted(result, key=lambda g: g[0]["path"])
