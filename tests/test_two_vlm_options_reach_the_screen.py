"""F204: the two model questions nobody was ever shown a switch for.

The run screen offered five decisions about the deep tier — the master switch, products,
online geo, animals and the animal check — and the pipeline asked the model two more:

* `features.junk_rescue` (F140), which shows the model every photograph whose junk score
  clears a threshold and moves the ones it calls screen captures out of the city layout.
  On the run of 2026-08-04 it added 441 frames to the screenshots slice and 41% of them
  were ordinary photographs of the owner (the F171 measurement over 350 frames) — ~181
  personal pictures leaving the layout for a bucket read as "these are your screenshots";
* `features.landmarks_verify` (F131), the corroboration of a CLIP landmark proposal by
  the model.

Neither could be switched on, switched off, or found out about from the interface — and
the estimate on that same screen was already READING one of them to price the run. This
is the mirror of F193: there, a missing button forbade nothing and the refusal moved into
the route with a reason; here a missing checkbox switched nothing off and the option ran
in silence. Both are about a product knowing the grounds of its own behaviour and saying
them out loud.

What is pinned below, in the order of the brief:

1. both options are on the screen and both are subordinate to "Deep analysis (VLM)" —
   checked by BEHAVIOUR, at the two stages that would raise the weights, not by markup;
2. the rescue's caption carries its measured share of false finds;
3. the value starts from the config, the checkbox moves it in BOTH directions, and
   config.yaml is not rewritten (the F127 one-run override);
4. the budget prices both lines, and the sum on screen moves with the rescue's box;
5. the captions exist in all three languages.

The thresholds are untouched on purpose: F140 measured that raising
`junk_rescue_threshold` LOWERS precision (41% false at 0.05 against 59% at 0.02), so the
missing thing was never a knob — it was the choice to run the pass at all.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unittest

from sorta import ui
from sorta.ui import process
from sorta.ui.process import _RunOptions, _run_cfg

from tests.test_junk_rescue import Asker, RescueCase
from tests.test_landmark_corroboration import PRAGUE
from tests.test_landmarks_verify import VerifyCase
from tests.test_ui_process import _poll_until
from tests.test_ui_run_costs import RunCostsTestBase


def screen_cfg(cfg, **boxes):
    """The config ONE run gets from the run screen — the ticks applied over the file.

    Every case that talks about "the checkbox" goes through here rather than editing
    `cfg.features` by hand: what is being tested is the screen's override, and an
    assertion made on a config nobody assembled the way the server assembles it would
    pass just as happily with the override missing.
    """
    return _run_cfg(cfg, None, _RunOptions(**boxes))


class TestTheRescueIsSubordinateInFact(RescueCase):
    """Requirement 1 at the stage that asks: the rescue raises no model on its own.

    `RescueCase` gives one frame above the selection threshold and injects everything —
    the classifier, the text encoder, the asker — so no weights exist anywhere in these
    cases; what is asserted is whether the stage WOULD have used them.
    """

    def rescue_run(self, deep: bool, junk_rescue: bool):
        """One classify() under the ticks a person leaves on the screen."""
        self.cfg = screen_cfg(self.cfg, deep=deep, junk_rescue=junk_rescue)
        asker = Asker({"shot.jpg": "screenshot"})
        built: list[str] = []
        self.run_stage({"shot.jpg": 0.5}, asker=None,
                       junk_rescue_vlm_factory=lambda name: (
                           built.append(name) or asker))
        return asker, built

    def test_the_box_alone_asks_nothing_and_builds_nothing(self):
        """The F145 rule for the new line: permission is not granted by a subordinate.

        The factory is counted rather than the questions, because "the model answered
        nothing" and "the model was never built" differ by several gigabytes.
        """
        asker, built = self.rescue_run(deep=False, junk_rescue=True)
        self.assertEqual(asker.asked, [])
        self.assertEqual(built, [])
        self.assertEqual(self.verdict_of("shot.jpg"), "photo")

    def test_with_the_master_on_the_box_is_what_decides(self):
        asker, built = self.rescue_run(deep=True, junk_rescue=True)
        self.assertEqual(asker.asked, ["shot.jpg"])
        self.assertEqual(built, [self.cfg.vlm.model])
        self.assertEqual(self.verdict_of("shot.jpg"), "screenshot")

    def test_an_unticked_box_switches_off_what_the_file_asks_for(self):
        """The override has to work downwards too, or the checkbox is a decoration on
        a config file that has already decided."""
        self.features(junk_rescue=True, junk_rescue_threshold=0.02)
        asker, built = self.rescue_run(deep=True, junk_rescue=False)
        self.assertEqual(asker.asked, [])
        self.assertEqual(built, [])
        self.assertIsNone(self.score_of("shot.jpg"))  # not even scored

    def test_the_ticked_box_switches_on_what_the_file_leaves_off(self):
        self.features(junk_rescue=False, junk_rescue_threshold=0.02)
        asker, _built = self.rescue_run(deep=True, junk_rescue=True)
        self.assertEqual(asker.asked, ["shot.jpg"])

    def test_the_threshold_is_not_something_the_screen_moves(self):
        """A boundary of the brief: F140 closed the question of where the gate stands
        (raising it LOWERS precision), so the screen carries the switch and not the
        number."""
        cfg = screen_cfg(self.cfg, deep=True, junk_rescue=True)
        self.assertEqual(cfg.features.junk_rescue_threshold,
                         self.cfg.features.junk_rescue_threshold)


class TestTheLandmarkCheckIsSubordinateInFact(VerifyCase):
    """Requirement 1 at the other stage, on the F75/F131 fixture.

    The file says one thing and the ticks say another in every case, because that is the
    whole shape of the feature: the config is where the value lives, one run is what the
    screen decides.
    """

    verify = False   # the file leaves the check off; the box is what turns it on
    deep = False

    def verify_run(self, deep: bool, landmarks_verify: bool):
        self.cfg = screen_cfg(self.cfg, deep=deep, landmarks_verify=landmarks_verify)
        built: list[str] = []
        self.run_stage(asker=None, asker_factory=lambda name: (
            built.append(name) or self.ask))
        return built

    def test_the_box_alone_asks_nothing_and_builds_nothing(self):
        self.add("/photos/DCIM", PRAGUE, prob=0.60)
        built = self.verify_run(deep=False, landmarks_verify=True)
        self.assertEqual(self.asked, [])
        self.assertEqual(built, [])

    def test_the_gate_stays_where_a_run_without_the_check_leaves_it(self):
        """Not merely "no questions": with the check dead the stage is the one that ran
        before F131 existed, down to which proposals it collects at all."""
        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        self.verify_run(deep=False, landmarks_verify=True)
        self.assertEqual(self.place_of(band)[3], "unknown")

    def test_with_the_master_on_the_box_is_what_decides(self):
        self.says(self.add("/photos/DCIM", PRAGUE, prob=0.60), "Charles Bridge, Prague")
        built = self.verify_run(deep=True, landmarks_verify=True)
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(len(built), 1)

    def test_an_unticked_box_switches_off_what_the_file_asks_for(self):
        self.cfg = dataclasses.replace(
            self.cfg, features=dataclasses.replace(self.cfg.features,
                                                   landmarks_verify=True))
        self.add("/photos/DCIM", PRAGUE, prob=0.60)
        self.verify_run(deep=True, landmarks_verify=False)
        self.assertEqual(self.asked, [])


class RunScreenCase(RunCostsTestBase):
    """A real server with a real config.yaml, plus the cfg the two stages were given."""

    def run_stages(self, body: dict) -> tuple[object, object]:
        """Start a run with `body`; return the config `junk` and `landmarks` saw.

        Two stages instead of `run_once`'s one: the rescue is a setting of the back half
        of the junk stage, the landmark check is a setting of the landmarks stage, and
        "the checkbox reached the run" is a statement about the config each of them got.

        F222 made the landmark stage opt-in, so it is asked for here — a config that
        never reaches a stage says nothing about whether the checkbox reached it.
        """
        junk_cfg: list[object] = []
        landmark_cfg: list[object] = []

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                      verdicts_only=False, progress=None):
            self.calls.append("classify" if verdicts_only else "junk")
            if not verdicts_only:
                junk_cfg.append(cfg)

        def fake_landmarks(cfg, conn, classifier=None, progress=None):
            self.calls.append("landmarks")
            landmark_cfg.append(cfg)

        self.patch_fast_stages()
        self._patch("classify_junk", fake_junk)
        self._patch("detect_landmarks", fake_landmarks)
        self.start_server()
        status, resp = self.post("/api/process",
                                 {"source_dir": str(self.src_dir), "landmarks": True,
                                  **body})
        self.assertEqual(status, 200, resp)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual((len(junk_cfg), len(landmark_cfg)), (1, 1))
        return junk_cfg[0], landmark_cfg[0]

    def file_says(self, **features) -> None:
        self.cfg.features = dataclasses.replace(self.cfg.features, **features)


class TestTheValueComesFromTheConfig(RunScreenCase):
    """Requirement 3, first half: the screen opens describing the run the file describes."""

    def test_the_defaults_route_answers_from_the_file(self):
        self.file_says(junk_rescue=True, landmarks_verify=True)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertIs(data["junk_rescue"], True)
        self.assertIs(data["landmarks_verify"], True)

    def test_an_untouched_file_answers_the_way_it_behaves(self):
        """Both keys default to off, so an untouched config opens with both boxes clear
        — which is the run it has always described."""
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertIs(data["junk_rescue"], False)
        self.assertIs(data["landmarks_verify"], False)

    def test_the_script_sets_both_boxes_from_that_answer(self):
        html = ui._render_index_html("en")
        for field, control in (("junk_rescue", "process-junk-rescue-checkbox"),
                               ("landmarks_verify",
                                "process-landmarks-verify-checkbox")):
            with self.subTest(field=field):
                self.assertIn(f'document.getElementById("{control}").checked =', html)
                self.assertIn(f"!!data.{field};", html)


class TestTheBoxMovesTheRunBothWays(RunScreenCase):
    """Requirement 3, second half: one run, both directions, and the file left alone."""

    def test_a_ticked_box_switches_both_on_for_this_run(self):
        junk_cfg, landmark_cfg = self.run_stages(
            {"deep": True, "junk_rescue": True, "landmarks_verify": True})
        self.assertIs(junk_cfg.features.junk_rescue, True)
        self.assertIs(landmark_cfg.features.landmarks_verify, True)

    def test_an_unticked_box_switches_both_off_for_this_run(self):
        """The F57 rule: a cleared box forces OFF what config.yaml switched on, rather
        than quietly deferring to the file."""
        self.file_says(junk_rescue=True, landmarks_verify=True)
        junk_cfg, landmark_cfg = self.run_stages(
            {"deep": True, "junk_rescue": False, "landmarks_verify": False})
        self.assertIs(junk_cfg.features.junk_rescue, False)
        self.assertIs(landmark_cfg.features.landmarks_verify, False)

    def test_a_body_without_them_leaves_the_file_deciding(self):
        """`/api/process/rerun-optional` and every caller outside the browser send
        neither key, and must keep running what the config asks for."""
        self.file_says(junk_rescue=True, landmarks_verify=True)
        junk_cfg, landmark_cfg = self.run_stages({"deep": True})
        self.assertIs(junk_cfg.features.junk_rescue, True)
        self.assertIs(landmark_cfg.features.landmarks_verify, True)

    def test_the_run_does_not_rewrite_the_config(self):
        """The value has ONE home. A run that wrote the ticks back would make every
        checkbox on this screen a permanent setting nobody asked to change."""
        before = self.config_path.read_text(encoding="utf-8")
        self.run_stages({"deep": True, "junk_rescue": True, "landmarks_verify": True})
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)
        self.assertNotIn("junk_rescue", self.saved().get("features", {}))

    def test_the_master_is_not_touched_by_either_of_them(self):
        """Permission is a separate decision — a subordinate box must not grant it."""
        junk_cfg, _landmarks = self.run_stages(
            {"deep": False, "junk_rescue": True, "landmarks_verify": True})
        self.assertIs(junk_cfg.naming.vlm_enabled, False)

    def test_a_flag_that_is_not_a_boolean_is_refused(self):
        self.patch_fast_stages()
        self.start_server()
        for body in ({"junk_rescue": "yes"}, {"landmarks_verify": 1}):
            with self.subTest(body=body):
                status, _resp = self.post(
                    "/api/process", {"source_dir": str(self.src_dir), **body})
                self.assertEqual(status, 400)


class TestTheBudgetPricesThem(RunScreenCase):
    """Requirement 4: the price is on the screen, and it is computed, not written down."""

    def estimate(self) -> dict:
        status, body, _ctype = self.get("/api/process/estimate")
        self.assertEqual(status, 200)
        return json.loads(body)

    def scored(self, values: list[float]) -> None:
        """`frame_quality.junk_score` for one photo per value — the rescue's population."""
        for value in values:
            file_id, _p, _c = self.add_photo_file(f"s{self._n}.jpg")
            self.conn.execute(
                "INSERT INTO frame_quality (file_id, junk_score, source, updated_at)"
                " VALUES (?, ?, 'clip', '2026-01-01')", (file_id, value))
        self.conn.commit()

    def asked_about(self, count: int) -> None:
        """Rows in `landmark_checks` — proposals the check was shown on a previous run."""
        for i in range(count):
            file_id, _p, _c = self.add_photo_file(f"lm{i}.jpg")
            self.conn.execute(
                "INSERT INTO landmark_checks (file_id, landmark, score, verdict, model,"
                " updated_at) VALUES (?, 'Charles Bridge', 0.6, 'confirmed', 'q#1', ?)",
                (file_id, "2026-01-01"))
        self.conn.commit()

    def test_the_rescue_is_priced_over_the_band_it_selects(self):
        self.scored([0.5, 0.03, -0.1])
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["junk_rescue"], 2)
        self.assertAlmostEqual(data["seconds"]["junk_rescue"],
                               round(2 * ui._SEC_PER_VLM_FRAME, 1))

    def test_it_is_priced_before_it_is_ticked(self):
        """The whole point of a line: the price has to be readable while a person is
        still deciding, not after the run has been started with it on."""
        self.scored([0.5, 0.5])
        self.assertIs(self.cfg.features.junk_rescue, False)
        self.start_server()
        self.assertGreater(self.estimate()["seconds"]["junk_rescue"], 0)

    def test_a_collection_nobody_has_scored_yet_says_so(self):
        """A dash, not a zero. The score is written by the rescue itself, so before the
        first run with it on this index genuinely cannot tell — and "free" is the one
        answer that would be a lie."""
        self.add_photo_file("plain.jpg")
        self.start_server()
        data = self.estimate()
        self.assertIsNone(data["counts"]["junk_rescue"])
        self.assertIsNone(data["seconds"]["junk_rescue"])

    def test_the_landmark_check_is_priced_off_what_it_asked_last_time(self):
        self.asked_about(3)
        self.start_server()
        data = self.estimate()
        self.assertEqual(data["counts"]["landmarks_verify"], 3)
        self.assertAlmostEqual(data["seconds"]["landmarks_verify"],
                               round(3 * ui._SEC_PER_VLM_FRAME, 1))

    def test_the_stages_own_scan_rows_are_not_questions(self):
        """`landmark_checks` also holds one reserved row per file (F136) recording what
        CLIP found. Counting those would price a model pass over frames the model was
        never shown."""
        self.asked_about(2)
        file_id, _p, _c = self.add_photo_file("scan.jpg")
        self.conn.execute(
            "INSERT INTO landmark_checks (file_id, landmark, score, verdict, model,"
            " updated_at) VALUES (?, ?, NULL, '#none', 'marker', '2026-01-01')",
            (file_id, process._LANDMARK_SCAN_KEY))
        self.conn.commit()
        self.start_server()
        self.assertEqual(self.estimate()["counts"]["landmarks_verify"], 2)

    def test_a_check_that_never_ran_is_a_dash(self):
        self.add_photo_file("plain.jpg")
        self.start_server()
        self.assertIsNone(self.estimate()["counts"]["landmarks_verify"])

    def test_the_price_follows_the_index_and_not_a_stale_answer(self):
        """The estimate is cached; a band that grows has to re-price, or the screen
        serves the cost of a collection that no longer exists."""
        self.scored([0.5])
        self.start_server()
        first = self.estimate()["counts"]["junk_rescue"]
        self.scored([0.5, 0.5])
        self.assertEqual(first, 1)
        self.assertEqual(self.estimate()["counts"]["junk_rescue"], 3)


class TestTheSumMovesWithTheBox(unittest.TestCase):
    """Requirement 4 in the browser: the total is a sum over the lines that are ticked.

    There is no engine here to run the script, so what is pinned is the shape that IS
    the behaviour — the same way F145 and F161 pin theirs: the line is in `COST_ROWS`
    with the checkbox that enables it, and that checkbox recomputes the sum.
    """

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("en")

    def rows(self) -> str:
        rows = self.html[self.html.index("var COST_ROWS = ["):]
        return rows[:rows.index("];")]

    def test_both_lines_are_summed_from_their_own_checkbox(self):
        for key, control in (("junk_rescue", "process-junk-rescue-checkbox"),
                             ("landmarks_verify",
                              "process-landmarks-verify-checkbox")):
            with self.subTest(key=key):
                entry = re.search(r'\{ key: "%s"(.*?)\}' % key, self.rows(), re.S)
                self.assertIsNotNone(entry)
                self.assertIn(f'id: "{control}"', entry.group(1))
                # `vlm: true` is what zeroes the line when the master is clear, and
                # `costRowEnabled` is what adds it to the total when the box is ticked.
                self.assertIn("vlm: true", entry.group(1))

    def test_moving_the_box_recomputes_the_total(self):
        listeners = self.html.split(
            '"process-landmarks-verify-checkbox"].forEach', 1)[1]
        self.assertIn('addEventListener("change", renderCosts)',
                      listeners.split("});", 1)[0])

    def test_both_boxes_are_sent_with_the_run(self):
        for key, control in (("junk_rescue", "process-junk-rescue-checkbox"),
                             ("landmarks_verify",
                              "process-landmarks-verify-checkbox")):
            with self.subTest(key=key):
                sent = self.html.split(f"{key}:", 1)[1][:120]
                self.assertIn(f'getElementById("{control}").checked', sent)


class TestBothLinesAreOnTheScreen(unittest.TestCase):
    """Requirement 1's other half, and the one thing markup can answer: they are THERE.

    A dead option stays visible for the F145 reason — a vanished one reads as "there is
    no such feature", and one of these two is the reason 181 photographs left the layout.
    """

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("en")

    def options(self) -> str:
        return self.html.split('id="step-options"', 1)[1].split('id="step-actions"', 1)[0]

    def test_both_stand_under_the_line_they_belong_to_with_a_price_of_their_own(self):
        """F222 moved the landmark check from under the master switch to under the
        landmark stage it is a question about. It is still subordinate to the master —
        that is behaviour, and `TestTheLandmarkCheckIsSubordinateInFact` above asserts it
        at the stage — but a check of a stage that is not in the run is a control that
        does nothing, and putting it inside that stage's row is how the screen says so.
        """
        for parent, control, key in (
                ("process-deep-checkbox", "process-junk-rescue-checkbox",
                 "junk_rescue"),
                ("process-landmarks-checkbox", "process-landmarks-verify-checkbox",
                 "landmarks_verify")):
            with self.subTest(control=control):
                block = self.options().split(f'id="{parent}"', 1)[1]
                self.assertIn(f'id="{control}"', block)
                self.assertIn(f'data-cost="{key}"', block)

    def test_neither_starts_hidden(self):
        for row in ("process-junk-rescue-row", "process-landmarks-verify-row"):
            with self.subTest(row=row):
                self.assertNotIn("display:none",
                                 self.options().split(f'id="{row}"', 1)[1][:60])

    def test_each_has_exactly_one_control_in_the_page(self):
        for control in ("process-junk-rescue-checkbox",
                        "process-landmarks-verify-checkbox"):
            with self.subTest(control=control):
                self.assertEqual(self.html.count(f'id="{control}"'), 1)

    def test_each_says_which_switch_turns_it_back_on(self):
        """The F145 caption, inside the row and not somewhere after it: a line that is
        dead with the master clear has to name the switch that revives it."""
        for row in ("process-junk-rescue-row", "process-landmarks-verify-row"):
            with self.subTest(row=row):
                block = self.options().split(f'id="{row}"', 1)[1].split("</span>\n<span "
                                                                       'class="cost-child"',
                                                                       1)[0]
                self.assertIn("vlm-off-hint", block)
                self.assertIn(ui._t("process_needs_deep_hint", "en"), block)


class TestTheCaptionsSayThePrice(unittest.TestCase):
    """Requirements 2 and 5: what the two lines tell a person before the run."""

    def test_the_rescue_names_its_measured_share_of_false_finds(self):
        """Requirement 2. The number is not hidden: switching this on is a trade, and
        the side that costs photographs is the side a person cannot see afterwards."""
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                hint = ui._t("process_junk_rescue_hint", lang)
                self.assertIn("41%", hint)
                self.assertIn("441", hint)

    def test_the_rescue_says_what_the_false_finds_ARE(self):
        """A percentage of nothing named is not information: the frames it takes are
        ordinary photographs, and they leave the city layout with the screenshots."""
        for lang, word in (("ru", "фотографии"), ("en", "photographs"),
                           ("ja", "普通の写真")):
            with self.subTest(lang=lang):
                self.assertIn(word, ui._t("process_junk_rescue_hint", lang))

    def test_the_landmark_check_says_what_it_does_and_why(self):
        """Requirement 3 of the brief: the corroboration, and the F75 rule behind it —
        a wrong city is worse than no city."""
        for lang, (what, why) in (("ru", ("CLIP", "хуже")),
                                  ("en", ("CLIP", "worse")),
                                  ("ja", ("CLIP", "悪い"))):
            with self.subTest(lang=lang):
                hint = ui._t("process_landmarks_verify_hint", lang)
                self.assertIn(what, hint)
                self.assertIn(why, hint)

    def test_every_new_caption_exists_in_all_three_languages(self):
        for key in ("process_junk_rescue_label", "process_junk_rescue_hint",
                    "process_landmarks_verify_label", "process_landmarks_verify_hint"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_captions_reach_the_page_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                html = ui._render_index_html(lang)
                for key in ("process_junk_rescue_label",
                            "process_landmarks_verify_label"):
                    self.assertIn(ui._t(key, lang), html)


if __name__ == "__main__":
    unittest.main()
