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
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def near_duplicate_groups(
    conn: sqlite3.Connection, max_distance: int = 5,
) -> list[list[sqlite3.Row]]:
    """Near-duplicate groups among canonical files by pHash.

    Pairs with a Hamming distance <= max_distance are merged into groups
    (union-find, i.e. a group is transitive: A~B and B~C put A and C together,
    even if dist(A, C) > the threshold — for a report this is expected).

    Candidates are found via band buckets (pigeonhole): the hash is cut into
    max_distance+1 parts; a pair within the threshold shares at least one part —
    a full O(n^2) scan is not needed.

    Exact duplicates (dup_of IS NOT NULL) and errored files are excluded.
    Writes nothing to the DB. Groups are sorted by the path of the first file,
    and within a group — by descending size.
    """
    rows = conn.execute(
        """SELECT id, path, size, phash FROM files
           WHERE phash IS NOT NULL AND error IS NULL AND dup_of IS NULL"""
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    parent = {r["id"]: r["id"] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bands = max_distance + 1
    buckets: dict[tuple[int, int, str], list[int]] = {}
    for r in rows:
        h = r["phash"].lower()
        step = max(1, -(-len(h) // bands))  # ceil; length in the key — do not compare different pHashes
        for bi in range(bands):
            part = h[bi * step:(bi + 1) * step]
            buckets.setdefault((len(h), bi, part), []).append(r["id"])

    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = by_id[ids[i]], by_id[ids[j]]
                ra, rb = find(a["id"]), find(b["id"])
                if ra == rb:
                    continue
                if hamming(a["phash"], b["phash"]) <= max_distance:
                    parent[rb] = ra

    grouped: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        grouped.setdefault(find(r["id"]), []).append(r)
    result = [sorted(g, key=lambda r: (-(r["size"] or 0), r["path"]))
              for g in grouped.values() if len(g) > 1]
    return sorted(result, key=lambda g: g[0]["path"])
