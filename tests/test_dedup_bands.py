"""F66: near_duplicate_groups — bit bands, fast hamming, identical-pHash collapse.

The centerpiece is an equivalence test against a brute-force O(n^2) reference:
the band buckets are only an optimisation, so on the same input the two must
produce byte-identical group lists. Everything here works on synthetic `files`
rows — no real images are decoded.
"""
from __future__ import annotations

import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect
from sorta.dedup import _band_ranges, hamming, near_duplicate_groups


def _brute_force_groups(
    conn: sqlite3.Connection, max_distance: int,
) -> list[list[sqlite3.Row]]:
    """Reference implementation: all pairs + union-find + the documented sorting."""
    rows = conn.execute(
        """SELECT id, path, size, phash FROM files
           WHERE phash IS NOT NULL AND error IS NULL AND dup_of IS NULL"""
    ).fetchall()
    parent = {r["id"]: r["id"] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if len(a["phash"]) != len(b["phash"]):
                continue  # different hash lengths are never compared
            if hamming(a["phash"], b["phash"]) <= max_distance:
                ra, rb = find(a["id"]), find(b["id"])
                if ra != rb:
                    parent[rb] = ra

    grouped: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        grouped.setdefault(find(r["id"]), []).append(r)
    result = [sorted(g, key=lambda r: (-(r["size"] or 0), r["path"]))
              for g in grouped.values() if len(g) > 1]
    return sorted(result, key=lambda g: g[0]["path"])


class TestBandRanges(unittest.TestCase):
    def test_covers_64_bits_over_6_bands_without_overlap(self):
        ranges = _band_ranges(64, 6)
        self.assertEqual(len(ranges), 6)
        widths = [w for _, w in ranges]
        self.assertEqual(widths, [11, 11, 11, 11, 10, 10])
        self.assertEqual(sum(widths), 64)
        self.assertLessEqual(max(widths) - min(widths), 1)
        covered = []
        for offset, width in ranges:
            covered.extend(range(offset, offset + width))
        self.assertEqual(sorted(covered), list(range(64)))  # exact cover, no overlap

    def test_arbitrary_sizes_stay_balanced_and_exact(self):
        for bits in (8, 32, 64, 128, 144):
            for bands in range(1, 12):
                with self.subTest(bits=bits, bands=bands):
                    ranges = _band_ranges(bits, bands)
                    self.assertEqual(len(ranges), bands)
                    widths = [w for _, w in ranges]
                    self.assertEqual(sum(widths), bits)
                    self.assertLessEqual(max(widths) - min(widths), 1)
                    offset = 0
                    for start, width in ranges:
                        self.assertEqual(start, offset)
                        offset += width


class NearDupTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _add(self, path: str, phash: str | None, size: int = 100,
             dup_of: int | None = None, error: str | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, phash,
                                  dup_of, error, indexed_at)
               VALUES (?,?,0,'jpg','photo',?,?,?,'2026-01-01')""",
            (path, size, phash, dup_of, error))
        self.conn.commit()
        return int(cur.lastrowid)


class TestBruteForceEquivalence(NearDupTestBase):
    """The main acceptance test: banding must not lose or invent a single pair."""

    def _populate(self) -> None:
        rnd = random.Random(20260725)
        base_pool = [rnd.getrandbits(64) for _ in range(60)]
        values: list[int] = []
        for base in base_pool:
            values.append(base)
            values.append(base)  # exact copy — must collapse into the group
            for flips in (1, 2, 3, 5):
                v = base
                for bit in rnd.sample(range(64), flips):
                    v ^= 1 << bit
                values.append(v)
        values.extend(rnd.getrandbits(64) for _ in range(60))  # pure noise
        rnd.shuffle(values)
        self.assertGreaterEqual(len(values), 300)
        for i, v in enumerate(values):
            self._add(f"/p{i:04d}.jpg", f"{v:016x}", size=rnd.randrange(1000, 9000))

    def test_matches_brute_force_for_every_threshold(self):
        self._populate()
        for max_distance in (0, 1, 3, 5, 8):
            with self.subTest(max_distance=max_distance):
                fast = near_duplicate_groups(self.conn, max_distance=max_distance)
                slow = _brute_force_groups(self.conn, max_distance)
                self.assertEqual(
                    [[r["path"] for r in g] for g in fast],
                    [[r["path"] for r in g] for g in slow],
                )


class TestNearDuplicateSemantics(NearDupTestBase):
    def test_different_phash_lengths_are_not_mixed(self):
        # "00000000" (32 bit) and "0000000000000000" (64 bit) are distance 0 as ints,
        # but hashes of different lengths must never be compared.
        self._add("/short_a.jpg", "0" * 8)
        self._add("/short_b.jpg", "0" * 8)
        self._add("/long_a.jpg", "0" * 16)
        self._add("/long_b.jpg", "0" * 16)
        groups = near_duplicate_groups(self.conn, max_distance=5)
        self.assertEqual(
            [[r["path"] for r in g] for g in groups],
            [["/long_a.jpg", "/long_b.jpg"], ["/short_a.jpg", "/short_b.jpg"]],
        )

    def test_excludes_exact_dups_and_errors(self):
        keeper = self._add("/a.jpg", "0" * 16)
        self._add("/exact.jpg", "0" * 16, dup_of=keeper)
        self._add("/broken.jpg", "0" * 16, error="Boom")
        self._add("/no_phash.jpg", None)
        self.assertEqual(near_duplicate_groups(self.conn, max_distance=5), [])

    def test_identical_phashes_collapse_at_distance_zero(self):
        self._add("/a.jpg", "abcdef0123456789", size=300)
        self._add("/b.jpg", "abcdef0123456789", size=900)
        self._add("/c.jpg", "abcdef012345678a", size=100)  # 1 bit away
        groups = near_duplicate_groups(self.conn, max_distance=0)
        self.assertEqual([[r["path"] for r in g] for g in groups], [["/b.jpg", "/a.jpg"]])

    def test_uppercase_phash_matches_lowercase(self):
        self._add("/a.jpg", "ABCDEF0123456789")
        self._add("/b.jpg", "abcdef0123456789")
        groups = near_duplicate_groups(self.conn, max_distance=0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_hamming_contract_unchanged(self):
        self.assertEqual(hamming("00", "ff"), 8)
        self.assertEqual(hamming("0" * 16, "0" * 16), 0)
        self.assertEqual(hamming("0" * 16, "0" * 15 + "1"), 1)
        self.assertEqual(hamming("0" * 16, "f" * 16), 64)


if __name__ == "__main__":
    unittest.main()
