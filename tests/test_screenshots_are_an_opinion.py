"""F171: the screenshot bucket says it is an opinion, not a fact.

On 350 hand-labelled frames the `screenshot` verdict is right about 59% of what it points
at (83% recall): every third frame in that bucket is an ordinary photograph. The live run
of 2026-08-04 reproduced the prediction made from that sample to within one frame — the
rescue added 441 frames to the bucket (1 782 against 1 341) and 41% of what it adds are
photographs, ~181 personal pictures leaving the city layout for a list a person reads as
"these are your screenshots" and does not look through.

Nothing about the verdict moves here — no threshold, no class, no file. What is pinned is
the three things the slice owes a reader who is about to press "delete":

* the caption names the MODEL as the author of the verdict and the measurement it was
  written from, and it names returning a frame as an ordinary step rather than as the
  repair of a rare mistake;
* a bucket is a LIST IN ORDER — `media_class.score` descending, the frames the classifier
  settled without a number keeping the path order behind them — and the page says whether
  that ordering actually happened (`ordered_by_score`), so the caption promises a ranking
  only where there is one. The "all" view is left alone: four classes are four separate
  softmaxes, and an order across them would be a comparison nobody measured;
* the way back is in sight before the grid is, and it is the action that already exists
  (`POST /api/overrides` with `photo`), not a second one.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from sorta import ui

from tests.test_ui_junk_buckets import JunkViewTestBase

_ROOT = Path(__file__).resolve().parent.parent
_GUIDES = {lang: _ROOT / "docs" / "guide" / f"user-guide.{lang}.md"
           for lang in ("en", "ru", "ja")}

LANGS = ("ru", "en", "ja")


class ScreenshotSliceCase(JunkViewTestBase):
    """The bucket over a ready `media_class`: no stage, no pixels — just the rule."""

    def add_scored(self, rel: str, verdict: str, score: float | None) -> int:
        """A classified frame plus the confidence the fast tier decided it by.

        `score` is NULL wherever no number decided the verdict — a heuristics-only run,
        or the deep tier rewriting one — and the ordering has to keep that meaning.
        """
        file_id = self.add_classified(rel, verdict, source="clip")
        self.conn.execute("UPDATE media_class SET score = ? WHERE file_id = ?",
                          (score, file_id))
        self.conn.commit()
        return file_id

    def ids(self, query: str = "") -> list[int]:
        return [item["file_id"] for item in self.junk(query)["items"]]


class TestTheBucketIsAListInOrder(ScreenshotSliceCase):
    def test_the_frame_the_model_is_surest_of_comes_first(self):
        # The paths run a.jpg, b.jpg, c.jpg — the old order — and the scores do not.
        low = self.add_scored("a.jpg", "screenshot", 0.41)
        high = self.add_scored("b.jpg", "screenshot", 0.93)
        middle = self.add_scored("c.jpg", "screenshot", 0.70)
        self.start_server()
        self.assertEqual(self.ids("?bucket=screenshot"), [high, middle, low])

    def test_a_frame_with_no_estimate_sits_behind_the_scored_ones(self):
        # NULL is "no estimate", never "unsure": such a frame keeps the path order at the
        # END of the list instead of sinking to a number it was never given.
        unscored = self.add_scored("a.jpg", "screenshot", None)
        scored = self.add_scored("b.jpg", "screenshot", 0.12)
        self.start_server()
        self.assertEqual(self.ids("?bucket=screenshot"), [scored, unscored])

    def test_frames_without_an_estimate_keep_the_order_they_always_had(self):
        by_path = [self.add_scored(f"p{i}.jpg", "screenshot", None) for i in range(3)]
        self.start_server()
        self.assertEqual(self.ids("?bucket=screenshot"), by_path)

    def test_the_order_survives_the_seam_between_two_pages(self):
        # `f.path` breaks every tie, so a card keeps its place and the paging neither
        # repeats a frame nor drops one.
        for i, score in enumerate((0.10, 0.90, 0.50, 0.70)):
            self.add_scored(f"f{i}.jpg", "screenshot", score)
        self.start_server()
        whole = self.ids("?bucket=screenshot")
        first = self.ids("?bucket=screenshot&limit=2")
        second = self.ids("?bucket=screenshot&limit=2&offset=2")
        self.assertEqual(first + second, whole)
        self.assertEqual(len(set(whole)), 4)

    def test_the_all_view_is_not_ordered_across_four_buckets(self):
        # F175's rule about the captions, applied to the order: a product's softmax and a
        # screenshot's are not one scale, so "all" stays in the path order it had.
        screenshot = self.add_scored("z.jpg", "screenshot", 0.93)
        product = self.add_scored("a.jpg", "product", 0.11)
        self.start_server()
        self.assertEqual(self.ids(), [product, screenshot])
        self.assertEqual(self.ids("?bucket=screenshot"), [screenshot])

    def test_every_bucket_is_ordered_the_same_way(self):
        # Nothing here is about screenshots in particular — the bucket that was measured
        # is simply the one that made it urgent.
        expected = {}
        for verdict in ("product", "document", "screenshot", "meme"):
            low = self.add_scored(f"a_{verdict}.jpg", verdict, 0.20)
            high = self.add_scored(f"b_{verdict}.jpg", verdict, 0.80)
            expected[verdict] = [high, low]
        self.start_server()
        for verdict, order in expected.items():
            with self.subTest(verdict=verdict):
                self.assertEqual(self.ids("?bucket=" + verdict), order)


class TestThePageSaysWhetherItIsARanking(ScreenshotSliceCase):
    def test_a_scored_bucket_reports_the_order_it_was_built_with(self):
        self.add_scored("a.jpg", "screenshot", 0.42)
        self.start_server()
        self.assertTrue(self.junk("?bucket=screenshot")["ordered_by_score"])

    def test_a_bucket_nobody_scored_promises_nothing(self):
        self.add_scored("a.jpg", "screenshot", None)
        self.start_server()
        self.assertFalse(self.junk("?bucket=screenshot")["ordered_by_score"])

    def test_the_all_view_promises_nothing_either(self):
        self.add_scored("a.jpg", "screenshot", 0.42)
        self.start_server()
        self.assertFalse(self.junk()["ordered_by_score"])

    def test_an_empty_bucket_is_still_a_well_formed_answer(self):
        self.add_scored("a.jpg", "product", 0.42)
        self.start_server()
        data = self.junk("?bucket=meme")
        self.assertEqual(data["items"], [])
        self.assertFalse(data["ordered_by_score"])

    def test_the_client_states_the_order_only_where_the_server_did(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('return data.ordered_by_score ? I18N.junk_order_hint : "";', html)
        self.assertIn(
            "accuracy.textContent = junkAccuracyText(data.bucket) + junkOrderText(data);",
            html)


class TestTheCaptionIsAnOpinionNotAVerdict(unittest.TestCase):
    """What the caption owes, in all three languages — a sentence that exists only in
    Russian is a sentence two thirds of the users do not have."""

    def test_it_states_the_measurement_it_was_written_from(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                caption = ui._UI_STRINGS["junk_accuracy_screenshot"][lang]
                for token in ("59%", "83%", "2026-08-03", "350"):
                    self.assertIn(token, caption)

    def test_it_names_the_model_as_the_author_of_the_verdict(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                caption = ui._UI_STRINGS["junk_accuracy_screenshot"][lang]
                for word in {
                    "ru": ("Модель", "оценка", "не факт"),
                    "en": ("model", "estimate", "not a fact"),
                    "ja": ("モデル", "推定", "事実ではなく"),
                }[lang]:
                    self.assertIn(word, caption)

    def test_it_calls_the_return_an_ordinary_step_of_the_work(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                caption = ui._UI_STRINGS["junk_accuracy_screenshot"][lang]
                for word in {
                    "ru": ("удалением", "верните", "обычный шаг"),
                    "en": ("deleting", "return", "ordinary step"),
                    "ja": ("削除する前", "戻して", "通常の作業"),
                }[lang]:
                    self.assertIn(word, caption)

    def test_the_order_is_explained_where_it_applies(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                hint = ui._UI_STRINGS["junk_order_hint"][lang]
                self.assertTrue(hint.strip())
                # It joins the caption of the open bucket, so it starts on a space.
                self.assertTrue(hint.startswith(" "), repr(hint))
                for word in {
                    "ru": ("уверена", "сверху"),
                    "en": ("sure", "top"),
                    "ja": ("確信", "上から"),
                }[lang]:
                    self.assertIn(word, hint)

    def test_the_caption_reaches_the_page_in_the_chosen_language(self):
        # It travels in the string catalog the client looks up, not in the markup.
        for lang in LANGS:
            with self.subTest(lang=lang):
                html = ui._render_index_html(lang)
                for key in ("junk_accuracy_screenshot", "junk_order_hint"):
                    self.assertIn(ui._UI_STRINGS[key][lang], html)


class TestTheWayBackIsInSight(ScreenshotSliceCase):
    """41% wrong makes the return an expected step, so it has to be readable as one:
    above the grid, next to the number that explains why it is there."""

    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_button_stands_between_the_caption_and_the_grid(self):
        accuracy = self.html.index('id="junk-accuracy"')
        button = self.html.index('id="junk-restore-btn"')
        grid = self.html.index('id="junk-grid"')
        self.assertLess(accuracy, button)
        self.assertLess(button, grid)

    def test_the_button_is_named_after_its_destination_in_every_language(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                html = ui._render_index_html(lang)
                controls = html.split('id="junk-restore-btn"', 1)[1]
                self.assertIn(ui._UI_STRINGS["slice_return_button"][lang],
                              controls.split("</button>", 1)[0])

    def test_the_card_offers_the_same_movement_by_name(self):
        self.assertIn("label.appendChild(document.createTextNode(I18N.slice_return_button));",
                      self.html)

    def test_returning_a_screenshot_puts_it_back_into_the_city_layout(self):
        # The action is the one that already exists — no verdict is rewritten, and the
        # plan lays the frame out by city again.
        file_id = self.add_scored("wrong.jpg", "screenshot", 0.51)
        self.post("/api/overrides", {"file_ids": [file_id], "action": "photo"})
        self.assertEqual(self.verdicts()[file_id], "screenshot")
        self.assertEqual(self.plan_targets()[file_id], "Russia/Moscow/2022/wrong.jpg")


class TestTheGuidesSayTheSame(unittest.TestCase):
    """The guides described the bucket as a classification. All three now carry the
    measurement, the run that confirmed it, and the advice to look the list over."""

    def read(self, lang: str) -> str:
        return _GUIDES[lang].read_text(encoding="utf-8")

    def test_every_guide_carries_the_run_that_confirmed_the_sample(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                text = self.read(lang)
                for token in ("59%", "441", "181"):
                    self.assertIn(token, text)

    def test_every_guide_says_the_list_is_worth_looking_through(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                text = self.read(lang)
                for word in {
                    "ru": ("просмотреть", "верните"),
                    "en": ("look the list over", "return"),
                    "ja": ("目を通", "戻す"),
                }[lang]:
                    self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
