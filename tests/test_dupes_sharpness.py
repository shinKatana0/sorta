"""F120: sharpness decides which frame of a near-duplicate group is recommended.

Across the collection sharpness is not comparable — a screenshot averages 2854 against a
photograph's 1253, because hard edges and text are what the laplacian measures. Inside a
near-duplicate group the frames are the same picture, and that is the one place the
number answers the question it was taken for: which of these is in focus.

The case that matters most here is the MIXED group: after F120 only personal photographs
are measured, so a group where some frames have no sharpness is ordinary, and a partial
comparison would silently prefer whichever frames happened to be measured.
"""
from __future__ import annotations

import json

from tests.test_ui_dupes import DupesTestBase


class TestSharpnessDecidesTheRecommendation(DupesTestBase):
    def make_pair(self) -> tuple[int, int]:
        """One near-duplicate group: a big frame and a smaller one, same pHash."""
        big = self.add_dupe("a.jpg", phash="f" * 16, width=4000, height=3000, size=900)
        small = self.add_dupe("b.jpg", phash="f" * 16, width=1000, height=750, size=100)
        return big, small

    def groups(self) -> list[dict]:
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        return json.loads(body)

    def set_sharpness(self, file_id: int, value: float | None) -> None:
        self.conn.execute(
            "INSERT INTO frame_quality (file_id, sharpness, source, updated_at)"
            " VALUES (?, ?, 'classic', '2026-08-01T00:00:00')"
            " ON CONFLICT(file_id) DO UPDATE SET sharpness = excluded.sharpness",
            (file_id, value))
        self.conn.commit()

    def test_the_sharper_frame_wins_over_the_bigger_one(self):
        """The whole point: a larger blurred frame is not the one to keep."""
        big, small = self.make_pair()
        self.set_sharpness(big, 40.0)     # bigger, out of focus
        self.set_sharpness(small, 900.0)  # smaller, sharp
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["recommended_by"], "sharpness")
        by_id = {f["file_id"]: f for f in group["frames"]}
        self.assertTrue(by_id[small]["recommended"])
        self.assertFalse(by_id[big]["recommended"])

    def test_a_group_without_sharpness_falls_back_to_resolution(self):
        big, small = self.make_pair()
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["recommended_by"], "resolution")
        by_id = {f["file_id"]: f for f in group["frames"]}
        self.assertTrue(by_id[big]["recommended"])
        self.assertIsNone(by_id[big]["sharpness"])

    def test_a_partly_measured_group_does_not_rank_by_sharpness(self):
        """A mixed group is ordinary after F120, and half a comparison is not one:
        ranking here would prefer the measured frame for having been measured."""
        big, small = self.make_pair()
        self.set_sharpness(small, 5000.0)  # the only frame with a number
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["recommended_by"], "resolution")
        by_id = {f["file_id"]: f for f in group["frames"]}
        self.assertTrue(by_id[big]["recommended"])
        self.assertEqual(by_id[small]["sharpness"], 5000.0)

    def test_the_number_travels_to_the_client(self):
        """The tab shows WHY a frame is starred; a star alone asks to be trusted."""
        big, small = self.make_pair()
        self.set_sharpness(big, 111.0)
        self.set_sharpness(small, 222.0)
        self.start_server()
        frames = {f["file_id"]: f for f in self.groups()[0]["frames"]}
        self.assertEqual(frames[big]["sharpness"], 111.0)
        self.assertEqual(frames[small]["sharpness"], 222.0)
