"""F157: the blurred slice is a list in order, not a window with a threshold.

The morning of 2026-08-02 measured the completeness of the blur filter at 6% and spent
half a day calling that a catastrophe. The evening measurement cancelled the number — not
because the filter turned out better, but because the criterion had moved between the two
labelling sessions (five of six features agreed to within 0.77-0.93; blur diverged
threefold). Under the strict criterion the user chose — "visibly smeared", not "a little
soft" — a cutoff at 90 catches 12% of the blurred frames and one at 700 catches 82%, with
precision drifting 29% → 12% on the way. Precision falls slowly, recall climbs quickly:
the shape where a threshold is the wrong instrument.

So what is pinned here is a RANKING and the four ways one gets quietly turned back into a
verdict:

* the ORDER is the feature. The softest frame first, and `features.blur_review_max` is
  only how far the first page reaches — a larger value makes the list longer without
  reordering a thing, and the ordering of the second page continues the first;
* the counter is a LENGTH. The chip, the payload's `window_total` and the list all say one
  number, and it is the number of frames shown rather than a claim about how many
  photographs of the collection are blurred;
* NULL is "not measured". A frame with no sharpness is in no list and breaks no ordering;
* the F155 seam. Where `frame_quality.face_sharpness` exists, the frames that have one are
  ordered by it and BEFORE the rest — the face number finds 62% of the blurred frames
  against 15% for the whole-frame one, and the two scales never meet inside a comparison.
  Where the column does not exist (a database from before v25 — the merge order of F155
  and F157 was never fixed), the frame number orders everything and nothing raises.

No pixels are decoded anywhere here: every number in these fixtures is a value the quality
stage would have written, which is the whole population these rules run on.
"""
from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import ui
from sorta.config import FeaturesConfig
from sorta.sorter import plan_album

from tests.test_ui_review import ReviewTestBase


class BlurRankingCase(ReviewTestBase):
    """The slice over a ready `frame_quality`: no stage, no pixels — just the rule."""

    def blurred(self, *, offset: int = 0, limit: int = 50, beyond: bool = False,
                blur_max: float | None = None) -> dict:
        """`/api/review?slice=blurred`, without the HTTP round trip.

        The threshold comes off the CONFIG, exactly as the route reads it: a literal here
        would test the number this file happened to be written on.
        """
        features = self.cfg.features
        if blur_max is not None:
            features = dataclasses.replace(features, blur_review_max=blur_max)
        return ui._review_payload(
            self.cfg.database, "blurred", offset, limit, beyond=beyond,
            features=features, max_distance=self.cfg.index.phash_max_distance)

    def ids(self, payload: dict) -> list[int]:
        return [item["file_id"] for item in payload["items"]]

    def soft(self, rel: str, sharpness: float | None,
             face_sharpness: float | None = None) -> int:
        """A photograph the quality stage measured — optionally inside the face too."""
        file_id = self.add_reviewable(rel, sharpness=sharpness)
        if face_sharpness is not None:
            self.conn.execute(
                "UPDATE frame_quality SET face_sharpness = ? WHERE file_id = ?",
                (face_sharpness, file_id))
            self.conn.commit()
        return file_id

    def drop_face_sharpness(self) -> None:
        """The database as it was before F155 — the column simply is not there."""
        self.conn.execute("ALTER TABLE frame_quality DROP COLUMN face_sharpness")
        self.conn.commit()


class TestTheOrderIsTheFeature(BlurRankingCase):
    def test_the_softest_frame_comes_first(self):
        sharper = self.soft("sharper.jpg", 300.0)
        softest = self.soft("softest.jpg", 40.0)
        self.assertEqual(self.ids(self.blurred(blur_max=1000.0)), [softest, sharper])

    def test_frames_of_equal_sharpness_keep_one_order_on_every_page(self):
        # The tie-break is `f.id`. Without it two frames of the same sharpness could
        # swap places between two requests and the seam would drop one of them.
        same = [self.soft(f"a{i}.jpg", 50.0) for i in range(4)]
        first = self.ids(self.blurred(limit=2))
        second = self.ids(self.blurred(offset=2, limit=2))
        self.assertEqual(first + second, same)


class TestTheNumberIsTheDepthOfTheFirstPage(BlurRankingCase):
    """`features.blur_review_max` makes the list longer. It does not make it different."""

    def setUp(self):
        super().setUp()
        self.by_sharpness = [self.soft(f"f{i}.jpg", s)
                             for i, s in enumerate((40.0, 120.0, 280.0, 500.0, 900.0))]

    def test_a_larger_value_opens_a_longer_list_in_the_same_order(self):
        short = self.ids(self.blurred(blur_max=300.0))
        long = self.ids(self.blurred(blur_max=1000.0))
        self.assertEqual(short, self.by_sharpness[:3])
        self.assertEqual(long, self.by_sharpness)
        self.assertEqual(long[:len(short)], short)   # a prefix, not a re-sort

    def test_the_default_opens_the_page_the_measurement_chose(self):
        # F157 raised it from 90 (12% of the blurred frames on the labelled sample) to
        # 300 (53%). The value lives in one place and this is the test that says so.
        self.assertEqual(FeaturesConfig().blur_review_max, 300.0)
        self.assertEqual(self.blurred()["blur_max"], 300.0)
        self.assertEqual(self.ids(self.blurred()), self.by_sharpness[:3])

    def test_the_example_config_ships_the_same_number(self):
        example = (Path(__file__).resolve().parent.parent
                   / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("blur_review_max: 300.0", example)

    def test_show_more_walks_further_down_the_same_list(self):
        page = self.ids(self.blurred(blur_max=300.0))
        beyond = self.ids(self.blurred(blur_max=300.0, beyond=True, offset=len(page)))
        self.assertEqual(beyond, self.by_sharpness[len(page):])
        self.assertEqual(page + beyond, self.by_sharpness)     # no re-sort at the seam
        self.assertEqual(len(set(page + beyond)), len(self.by_sharpness))   # no repeats

    def test_the_counter_and_the_length_of_the_list_are_one_number(self):
        data = self.blurred(blur_max=300.0)
        counts = {row["slice"]: row["count"] for row in data["counts"]}
        self.assertEqual(counts["blurred"], 3)
        self.assertEqual(data["window_total"], 3)
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["items"]), 3)


class TestAFrameNobodyMeasured(BlurRankingCase):
    def test_a_null_sharpness_is_in_no_list_and_breaks_no_ordering(self):
        # NULL means "not measured" everywhere in this table, and a frame nobody looked
        # at must not be handed to a person as an answer — least of all as the softest
        # one, which is where a NULL sorts on its own in SQLite.
        self.soft("unmeasured.jpg", None)
        ordered = [self.soft("soft.jpg", 40.0), self.soft("mid.jpg", 200.0)]
        data = self.blurred(blur_max=1000.0)
        self.assertEqual(self.ids(data), ordered)
        self.assertEqual(data["total"], 2)


class TestTheSeamWithFaceSharpness(BlurRankingCase):
    """F155: the same laplacian measured INSIDE the face, and what the list does with it.

    The main test of the seam. The two features were written in parallel and could be
    merged in either order, so both states of the database have to work: with the column
    the frames that have a face number are ordered by it and come first, without it every
    frame is ordered by the whole-frame number and nothing raises.
    """

    def test_the_face_number_orders_the_frames_that_have_one(self):
        # The whole-frame numbers say the opposite of the face numbers here on purpose:
        # a detailed sharp street scores high while the smooth blurred face beside it
        # scores low, which is the failure this ordering exists for.
        blurred_face = self.soft("portrait.jpg", 260.0, face_sharpness=20.0)
        sharp_face = self.soft("group.jpg", 100.0, face_sharpness=180.0)
        no_face = self.soft("street.jpg", 40.0)
        data = self.blurred(blur_max=1000.0)
        self.assertEqual(data["blur_order"], "face_sharpness")
        self.assertEqual(self.ids(data), [blurred_face, sharp_face, no_face])

    def test_a_frame_whose_face_was_never_measured_sorts_by_the_frame(self):
        # NULL in `face_sharpness` is "not measured" and not "sharp face": such a frame
        # keeps its place in the list, after the measured ones, by its own number.
        with_face = self.soft("face.jpg", 900.0, face_sharpness=300.0)
        softer = self.soft("softer.jpg", 40.0)
        sharper = self.soft("sharper.jpg", 500.0)
        self.assertEqual(self.ids(self.blurred(blur_max=1000.0)),
                         [with_face, softer, sharper])

    def test_without_the_column_the_frame_number_orders_everything(self):
        ordered = [self.soft("a.jpg", 40.0), self.soft("b.jpg", 120.0),
                   self.soft("c.jpg", 280.0)]
        self.drop_face_sharpness()
        data = self.blurred(blur_max=1000.0)
        self.assertEqual(data["blur_order"], "sharpness")
        self.assertEqual(self.ids(data), ordered)
        self.assertEqual(data["total"], 3)


class TestTheAlbumGathersWhatWasShown(BlurRankingCase):
    """The button collects the first page, and the first page only."""

    def album_ids(self) -> list[int]:
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "blurred", "", self.root / "albums")
        return sorted(item.file_id for item in report.plan)

    def test_the_album_is_exactly_the_first_page(self):
        for sharpness in (40.0, 120.0, 280.0, 500.0, 900.0):
            self.soft(f"f{sharpness}.jpg", sharpness)
        shown = self.ids(self.blurred())
        self.assertEqual(len(shown), 3)
        self.assertEqual(self.album_ids(), sorted(shown))

    def test_a_longer_page_gathers_a_longer_album(self):
        for sharpness in (40.0, 500.0):
            self.soft(f"f{sharpness}.jpg", sharpness)
        self.cfg.features = dataclasses.replace(
            self.cfg.features, blur_review_max=1000.0)
        self.assertEqual(self.album_ids(), sorted(self.ids(self.blurred(
            blur_max=1000.0))))


class TestTheScreenSaysItIsAnOrder(unittest.TestCase):
    """What the caption owes: it is an order, and the signal is coarse.

    A caption that describes a window ("the list opens down to 90") reads as a verdict
    about the frames inside it, and this one is right about one frame in five. Both halves
    are checked in all three languages, because a sentence that exists only in Russian is
    a sentence two thirds of the users do not have.
    """

    LANGS = ("ru", "en", "ja")

    def test_the_caption_says_where_to_stop_reading(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                hint = ui._UI_STRINGS["review_hint_blurred"][lang]
                self.assertIn("{max}", hint)
                # The order, the direction it runs in, and the reader's own stop.
                for word in {
                    "ru": ("порядок", "приговор", "сверху", "остановит"),
                    "en": ("order", "verdict", "top", "stop"),
                    "ja": ("並び順", "判定", "上から", "止め"),
                }[lang]:
                    self.assertIn(word, hint)

    def test_the_caption_admits_what_the_number_cannot_tell_apart(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                hint = ui._UI_STRINGS["review_hint_blurred"][lang]
                for word in {
                    "ru": ("улица", "лицо"),
                    "en": ("street", "face"),
                    "ja": ("街並み", "顔"),
                }[lang]:
                    self.assertIn(word, hint)

    def test_the_face_ordering_is_explained_where_it_applies(self):
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                self.assertTrue(
                    ui._UI_STRINGS["review_hint_blurred_faces"][lang].strip())

    def test_the_counter_of_the_list_promises_no_population(self):
        # "Blurred: 2 210" is a claim the signal cannot make. The ranked counter states
        # the length of what is on screen and says the list goes on.
        for lang in self.LANGS:
            with self.subTest(lang=lang):
                label = ui._UI_STRINGS["review_shown_ranked"][lang]
                self.assertIn("{shown}", label)
                self.assertNotIn("{total}", label)


class TestTheButtonKeepsSayingShowMore(BlurRankingCase):
    """The blurred slice steps past its first page without changing its promise.

    Closed eyes keep "show past the window" (F179): there the window is the measured 62%
    and leaving it means something. Here the first page ends where a config key put it,
    so the next page is simply more of the same list.
    """

    def test_the_browser_asks_for_the_plain_label_on_this_slice(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('beyondNext && reviewSlice !== "blurred"', html)
        self.assertIn("I18N.review_shown_ranked", html)
        self.assertIn("I18N.review_hint_blurred_faces", html)


if __name__ == "__main__":
    unittest.main()
