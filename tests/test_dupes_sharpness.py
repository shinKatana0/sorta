"""F120: sharpness is comparable INSIDE a near-duplicate group, and nowhere else.

Across the collection it is not — a screenshot averages 2854 against a photograph's 1253,
because hard edges and text are what the laplacian measures. Inside a group the frames are
the same scene at the same scale, and that is the one place the number answers the question
it was taken for: which of these is in focus.

F194 kept that finding and took away what had been built on it. Sharpness decided which
frame of a burst was RECOMMENDED, and 111 groups labelled blind by the owner (2026-08-04)
put it at 27% against 30.4% for choosing at random — below a coin, while the interface
highlighted it as the answer. So it stays as the ORDER of the third tier (which is what a
ranking is) and stops being the recommendation. What is checked here is the part that
survived: the number is comparable inside a group, a partly measured group is not ranked
by it, and it reaches the client.

The pairs below carry DIFFERENT pHashes on purpose: identical ones make the second tier,
where the checkable size rule decides and sharpness has no say at all
(`test_three_tiers_of_sameness`).
"""
from __future__ import annotations

import json

from tests.test_ui_dupes import DupesTestBase


class TestSharpnessOrdersAGroupAndDecidesNothing(DupesTestBase):
    def make_pair(self) -> tuple[int, int]:
        """One near-duplicate group of the third tier: two similar frames, one bigger."""
        big = self.add_dupe("a.jpg", phash="f" * 16, width=4000, height=3000, size=900)
        small = self.add_dupe("b.jpg", phash="f" * 15 + "e", width=1000, height=750,
                              size=100)
        return big, small

    def groups(self) -> list[dict]:
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        return json.loads(body)["groups"]

    def set_sharpness(self, file_id: int, value: float | None) -> None:
        self.conn.execute(
            "INSERT INTO frame_quality (file_id, sharpness, source, updated_at)"
            " VALUES (?, ?, 'classic', '2026-08-01T00:00:00')"
            " ON CONFLICT(file_id) DO UPDATE SET sharpness = excluded.sharpness",
            (file_id, value))
        self.conn.commit()

    def test_the_sharper_frame_is_shown_first_and_is_not_chosen(self):
        """The order a person reads in, and nothing more: a larger blurred frame is not
        the one to look at first, and neither frame is the one to keep."""
        big, small = self.make_pair()
        self.set_sharpness(big, 40.0)     # bigger, out of focus
        self.set_sharpness(small, 900.0)  # smaller, sharp
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["order"], "sharpness")
        self.assertEqual([f["file_id"] for f in group["frames"]], [small, big])
        self.assertIsNone(group["recommended_by"])
        self.assertEqual([f["file_id"] for f in group["frames"] if f["recommended"]], [])

    def test_a_group_without_sharpness_falls_back_to_size(self):
        big, small = self.make_pair()
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["order"], "size")
        self.assertEqual([f["file_id"] for f in group["frames"]], [big, small])
        by_id = {f["file_id"]: f for f in group["frames"]}
        self.assertIsNone(by_id[big]["sharpness"])

    def test_a_partly_measured_group_is_not_ordered_by_sharpness(self):
        """A mixed group is ordinary after F120, and half a comparison is not one:
        ranking here would put the measured frame first for having been measured."""
        big, small = self.make_pair()
        self.set_sharpness(small, 5000.0)  # the only frame with a number
        self.start_server()
        group = self.groups()[0]
        self.assertEqual(group["order"], "size")
        by_id = {f["file_id"]: f for f in group["frames"]}
        self.assertEqual(by_id[small]["sharpness"], 5000.0)

    def test_the_number_travels_to_the_client(self):
        """The screen says what ordered the list; a bare order asks to be trusted."""
        big, small = self.make_pair()
        self.set_sharpness(big, 111.0)
        self.set_sharpness(small, 222.0)
        self.start_server()
        frames = {f["file_id"]: f for f in self.groups()[0]["frames"]}
        self.assertEqual(frames[big]["sharpness"], 111.0)
        self.assertEqual(frames[small]["sharpness"], 222.0)
