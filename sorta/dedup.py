"""FR-1: exact-hash deduplication + near-duplicate report (pHash).

Files are not deleted: exact duplicates are marked with dup_of, near-duplicates
are only grouped into a report (near_duplicate_groups) without writing to the DB.

F132: the one thing this module now DOES write is the keeper RECOMMENDATION of a
near-duplicate group (`group_keeper`) — which frame of a burst is worth keeping, and
whether that was decided by sharpness or by the model. It is still not a decision about a
file: `dedup_choice` (the table the sorter and the trash button read) is written by the
user's hand alone, and nothing here touches it. See the schema comment on `group_keeper`
for why a group is addressed by a hash of its members instead of by an id.
"""
from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

from . import imaging
from .hashing import resolve_workers

if TYPE_CHECKING:
    from .config import Config

try:
    import imagehash  # type: ignore
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


def _phash_one(path: str, mtime: float, size: int) -> str | None:
    """pHash of one file, decoded through the shared preview cache (F67).

    The frame comes from imaging.decode_rgb_preview instead of a local
    Image.open/draft/thumbnail: draft is a no-op on HEIC (425 ms/frame — this stage
    was the slowest one on the live collection at 7-10 img/s), while a pHash off an
    existing preview costs ~1.4 ms. Already-stored pHashes are NOT invalidated: on
    live files a preview-based pHash differs from the full-decode one by at most
    2 bits (avg 0.3) against the near-duplicate threshold of 5, so the
    `phash IS NULL` incrementality stays as it was.
    """
    if not _PHASH:
        return None
    img = imaging.decode_rgb_preview(
        path, mtime, size, max_edge=_PHASH_DECODE, grayscale=True)
    if img is None:
        return None
    try:
        return str(imagehash.phash(img))
    except Exception:
        return None


def compute_phashes(
    cfg: "Config", conn: sqlite3.Connection,
    progress: Callable[[int, int | None], None] | None = None,
) -> int:
    """Compute pHash for files without one (incremental). Returns the number computed.

    Moved out of the hot `index()` path (F11): the pHash decode is done at a reduced
    resolution (F67: via the shared preview cache, see _phash_one), in parallel (the
    same ThreadPoolExecutor as in indexer.index — Pillow decoding releases the GIL).
    HEIC — if pillow-heif is installed, otherwise such files are silently skipped
    (phash stays NULL, as before).

    Exact duplicates are skipped: `near_duplicate_groups` — the only consumer of
    phash — selects `dup_of IS NULL`, so a hash computed for a duplicate is never
    read by anything. On the validation collection that was 12 799 of 37 301 files
    (34%), and most of them also paid for building a preview that nothing else
    wanted. Self-healing if roles change later: a file that becomes canonical still
    has `phash IS NULL` and is picked up by the next run.
    """
    if not _PHASH:
        return 0
    rows = conn.execute(
        """SELECT id, path, mtime, size FROM files
           WHERE phash IS NULL AND error IS NULL AND media_type = 'photo'
             AND dup_of IS NULL"""
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
            results = list(pool.map(
                _phash_one,
                [r["path"] for r in batch],
                [r["mtime"] for r in batch],
                [r["size"] for r in batch],
            ))
            with conn:  # one transaction per batch — as in index()
                for r, ph in zip(batch, results):
                    if ph is not None:
                        conn.execute("UPDATE files SET phash = ? WHERE id = ?", (ph, r["id"]))
                        computed += 1
            processed += len(batch)
            if progress:
                progress(processed, total)
    return computed


# --- F194: the three tiers of sameness ------------------------------------------------
# One word, "duplicate", has been covering three populations whose cost of a mistake
# differs by orders of magnitude. Counted on the live collection 2026-08-04:
#
#   exact       the same BYTES (blake3): 12 350 files over 7 631 originals — HALF the
#               archive. Losing the wrong one of two identical files loses nothing, the
#               file is the same file, so the tier is a NUMBER and never a list: there is
#               no judgement to make and nothing to look at.
#   same_image  the same picture in different files (one pHash, different bytes): 299
#               groups, 652 frames. A mistake costs a better or worse copy of one
#               picture, and the rule that avoids it is CHECKABLE — resolution and weight
#               are facts. So this tier gets a recommendation, and a person may overrule it.
#   similar     frames that merely look alike, within the pHash threshold: 791 groups. A
#               mistake here loses a photograph for good, and nobody can make the call:
#               111 groups labelled blind by the owner (2026-08-04) gave sharpness 27%,
#               arithmetic 28%, cascade 28%, the model 32% — against 30.4% for choosing at
#               random. So nothing recommends here; every frame stays until a hand says
#               otherwise.
#
# The tier is READ OFF the data (see `group_tier`), never stored: it follows from the
# hashes, and a stored copy of it would be a second answer to drift from the first.
TIER_EXACT = "exact"
TIER_SAME_IMAGE = "same_image"
TIER_SIMILAR = "similar"


@dataclass(frozen=True)
class ExactDuplicates:
    """The first tier, as the two numbers it is worth: how many copies, over how many
    originals."""
    copies: int
    originals: int


def exact_duplicate_summary(conn: sqlite3.Connection) -> ExactDuplicates:
    """How many byte-identical copies the index holds and how many files they fold onto.

    A copy is a row carrying `dup_of` — written by `assign_duplicates`, which picks one
    canonical file per blake3 hash and points the rest at it. Counting instead of listing
    is the entire tier: the bytes are the same bytes, so "choose which to keep" is a
    question about nothing, and twelve thousand rows of it would bury the two tiers where
    there IS something to decide.

    Nothing is deleted by this or by anything reading it. Collapsed means shown as one
    line instead of twelve thousand; the files stay on disk, and removing them stays a
    separate deliberate act of a person.
    """
    row = conn.execute(
        """SELECT COUNT(*) AS copies, COUNT(DISTINCT dup_of) AS originals
           FROM files WHERE dup_of IS NOT NULL AND error IS NULL""").fetchone()
    return ExactDuplicates(int(row["copies"]), int(row["originals"]))


def group_tier(phashes: Iterable[str]) -> str:
    """`TIER_SAME_IMAGE` or `TIER_SIMILAR` for a near-duplicate group, by its pHashes.

    One distinct pHash across the group means one picture stored more than once — the
    frames are byte-different by construction (`near_duplicate_groups` excludes `dup_of`),
    so what differs is the encoding, the scale or the metadata, and "keep the largest" is
    a statement about facts. More than one pHash means frames that merely resemble each
    other, which is the tier where the measurement found no rule at all.
    """
    return TIER_SAME_IMAGE if len(set(phashes)) == 1 else TIER_SIMILAR


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

    F149 excludes one more population: DERIVED files, i.e. the model-processed copies
    listed in `restored_files`. Such a copy is a near-duplicate of its source by
    construction, so without this it would come back as a new pair to sort out on every
    run — a person forever deciding about pairs they made themselves. Leaving it out of
    the groups is the exclusion half of that decision; the alternative (put it in a group
    with the answer already filled in) would mean writing `dedup_choice` from here, and
    that table is the user's own hand alone. The cost is that a restored frame is never
    offered as a duplicate of some THIRD file either, which is the same statement read
    the other way: it is not a frame to decide about, the decision was taken when it was
    made.
    """
    rows = conn.execute(
        """SELECT id, path, size, phash FROM files
           WHERE phash IS NOT NULL AND error IS NULL AND dup_of IS NULL
             AND id NOT IN (SELECT file_id FROM restored_files)"""
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


# --- F132: the keeper of a group ------------------------------------------------------

# Who chose the keeper. `sharpness` is the ranking the Duplicates tab has always used and
# the answer a group falls back to whenever the model does not produce one; the other
# value is `vlm#<prompt fingerprint>`, written by junk.py — the fingerprint lives with the
# prompt it hashes, not here.
KEEPER_SOURCE_SHARPNESS = "sharpness"

_ID_CHUNK = 500  # SQLite has a ceiling on bound parameters; a photo library reaches it


@dataclass(frozen=True)
class GroupFrame:
    """One frame of a near-duplicate group, with everything the keeper rule reads."""
    file_id: int
    path: str
    sharpness: float | None = None
    pixels: int = 0   # width * height, 0 when the index has no dimensions for the file
    size: int = 0


@dataclass(frozen=True)
class GroupKeeper:
    """One stored recommendation: which frame of this group, and who chose it."""
    group_key: str
    keeper_id: int
    source: str


def group_key(file_ids: Iterable[int]) -> str:
    """The identity of a GROUP — sha1 over its sorted file ids.

    Groups have no id of their own and cannot have one: they are recomputed by union-find
    on every call, so nothing in the database owns a group. Hashing the membership gives
    the missing identifier AND its invalidation in one stroke — a frame added to the burst
    or deleted from it produces a different key, the stored answer is simply not found,
    and the question is asked again about the group that exists now.

    Sorted, so the key does not depend on the order the caller happened to have the frames
    in; the ids are decimal and comma-separated, which is enough to keep "1,23" and
    "12,3" apart.
    """
    return hashlib.sha1(
        ",".join(str(int(i)) for i in sorted(file_ids)).encode("ascii")).hexdigest()


def rank_frames(frames: Sequence[GroupFrame]) -> list[GroupFrame]:
    """The group, best frame first — the ranking the recommendation has always used.

    Sharpness leads only when EVERY frame of the group has it. A partial comparison would
    quietly prefer whichever frames happened to be measured, and since F120 only personal
    photographs are measured at all, a mixed group is a real case rather than a corner one.
    Resolution, then file size, then id break the ties — id last so the order is total and
    two runs over an unchanged group produce the same answer.

    Inside a group is the one place the laplacian answers the question it was measured for:
    the frames are the same picture at the same scale, so the number compares focus rather
    than content (across a collection it does not — a screenshot averages 2854 against a
    photograph's 1253).
    """
    by_sharpness = all(f.sharpness is not None for f in frames)
    return sorted(frames, key=lambda f: (
        -(f.sharpness or 0.0) if by_sharpness else 0.0,
        -f.pixels, -f.size, f.file_id))


def keeper_groups(conn: sqlite3.Connection, max_distance: int = 5,
                  min_size: int = 2) -> list[list[GroupFrame]]:
    """Near-duplicate groups of at least `min_size` frames, each ranked best-first.

    The signals the ranking needs are read here, in one query per chunk of ids, instead of
    being carried by `near_duplicate_groups` — that function is a report over pHashes and
    has three other callers who want nothing of the sort.

    `min_size` is `dedup.keeper_min_group_size`: a group smaller than it is not returned at
    all, so it is never shown to a model and never gets a row. Its recommendation is the
    interface's own sharpness ranking, which is exactly what it was before this feature.
    """
    groups = [g for g in near_duplicate_groups(conn, max_distance) if len(g) >= min_size]
    if not groups:
        return []
    ids = [r["id"] for g in groups for r in g]
    extra: dict[int, tuple[float | None, int]] = {}
    for start in range(0, len(ids), _ID_CHUNK):
        part = ids[start:start + _ID_CHUNK]
        placeholders = ",".join("?" * len(part))
        for row in conn.execute(
            f"""SELECT f.id, f.width, f.height, fq.sharpness
                FROM files f LEFT JOIN frame_quality fq ON fq.file_id = f.id
                WHERE f.id IN ({placeholders})""", part):
            extra[int(row["id"])] = (
                None if row["sharpness"] is None else float(row["sharpness"]),
                int((row["width"] or 0) * (row["height"] or 0)))
    out = []
    for group in groups:
        frames = []
        for r in group:
            sharpness, pixels = extra.get(int(r["id"]), (None, 0))
            frames.append(GroupFrame(file_id=int(r["id"]), path=str(r["path"]),
                                     sharpness=sharpness, pixels=pixels,
                                     size=int(r["size"] or 0)))
        out.append(rank_frames(frames))
    return out


def read_group_keepers(conn: sqlite3.Connection,
                       keys: Sequence[str] | None = None) -> dict[str, GroupKeeper]:
    """Stored recommendations by group key; `keys=None` — the whole table.

    A key that is not in the result means the group has no recommendation of its own —
    either it has never been asked about, or its membership has changed since it was (see
    `group_key`). Both readings lead to the same behaviour in every consumer: fall back to
    the sharpness ranking, which is always available.
    """
    sql = "SELECT group_key, keeper_id, source FROM group_keeper"

    def rows(cursor: sqlite3.Cursor) -> dict[str, GroupKeeper]:
        return {str(r["group_key"]): GroupKeeper(str(r["group_key"]),
                                                 int(r["keeper_id"]), str(r["source"]))
                for r in cursor}

    if keys is None:
        return rows(conn.execute(sql))
    out: dict[str, GroupKeeper] = {}
    keys = list(keys)
    for start in range(0, len(keys), _ID_CHUNK):
        part = keys[start:start + _ID_CHUNK]
        out.update(rows(conn.execute(
            f"{sql} WHERE group_key IN ({','.join('?' * len(part))})", tuple(part))))
    return out


_KEEPER_UPSERT = """INSERT INTO group_keeper (group_key, keeper_id, source, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(group_key) DO UPDATE SET
                        keeper_id = excluded.keeper_id, source = excluded.source,
                        updated_at = excluded.updated_at"""


def store_group_keeper(conn: sqlite3.Connection, key: str, keeper_id: int,
                       source: str, updated_at: str) -> None:
    """Write one recommendation. Writes NOTHING else — `dedup_choice` is not touched.

    The caller owns the transaction, as everywhere in the stage this is called from.
    """
    conn.execute(_KEEPER_UPSERT, (key, int(keeper_id), source, updated_at))
