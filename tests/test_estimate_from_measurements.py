"""F159: the run estimate is computed from measurements, not from baked-in constants.

The run screen (F138) promised a time for every stage, and one of those numbers was
wrong in a way no test could catch, because the number was the test: the comparative
keeper question was priced at a flat 1.32 s per group whatever the group held. Measured
2026-08-03, the real price is linear in the frames the prompt carries — 0.45 s plus
1.03 s each — and on the live collection that is 1.9 minutes against an estimated 0.5, a
3.7x understatement that grows with the archive.

The fix is not a better constant. After F147 the run log holds the true rate of every
stage ON THIS MACHINE, so the estimate reads it from there and says that it did; a
constant is what it falls back to, and the screen calls that a default in as many words.

What is pinned here is the whole of that:

* a group of five costs more than a group of three, and the bill is the SUM over the
  actual groups, not an average times a count (§1, test 1);
* an empty log falls back to the defaults AND says so (§2, test 2);
* a log with timings in it is believed over the constants (§3, test 3 — the main one);
* a timing of a stage that has changed since is not used (§5, test 4);
* the estimate and the run it describes agree within 10% on staged data (test 5);
* no caption claims the group question is the cheap way to ask (§4, test 6).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta

from sorta import __version__, ui
from sorta.config import EstimateConfig, load_config

from tests.test_ui_run_costs import RunCostsTestBase

# Rates that no default could be mistaken for: the shipped ones are hundredths of a
# second per frame, these are whole seconds. A test that passes with either is a test
# about nothing.
MEASURED = {
    "stage=index": 0.10,
    "stage=geo": 0.20,
    "stage=landmarks": 0.30,
    "stage=phash": 0.40,
    "stage=faces": 2.00,
    "stage=events": 0.50,
    # F165 split the stage that asks the model in two, and both halves call their VLM
    # phase `junk_vlm`. The rates differ here on purpose: a line priced off the wrong
    # half would be charged the rate of a different population, and these numbers are
    # what makes that visible instead of a coincidence.
    "stage=classify phase=junk_vlm": 3.00,
    "stage=junk phase=junk_vlm": 5.00,
}
BASE_RATE = sum(MEASURED[f"stage={s}"] for s in ("index", "geo", "landmarks", "phash"))


def log_line(at: datetime, message: str) -> str:
    return f"{at:%Y-%m-%dT%H:%M:%S}.000 INFO     sorta.runlog [MainThread] {message}"


class EstimateTestBase(RunCostsTestBase):
    """A collection plus, when a case wants one, a run log with known timings in it."""

    def add_photo(self, rel: str, *, phash: str | None = None) -> int:
        file_id, _p, _c = self.add_photo_file(rel)
        if phash is not None:
            self.conn.execute("UPDATE files SET phash = ? WHERE id = ?",
                              (phash, file_id))
            self.conn.commit()
        return file_id

    def add_group(self, name: str, size: int, phash: str) -> None:
        """`size` near-duplicates of one scene — one group of `size` frames."""
        for i in range(size):
            self.add_photo(f"{name}{i}.jpg", phash=phash)

    def write_run_log(self, *, build: str = __version__, at: datetime | None = None,
                      rates: dict[str, float] | None = None, units: int = 100) -> None:
        """A finished run of `build`, in which every unit ran at the given rate."""
        moment = at or datetime.now() - timedelta(hours=1)
        lines = [log_line(moment, "environment:"), f"  sorta: {build}",
                 "  python: 3.10.0"]
        for unit, rate in (MEASURED if rates is None else rates).items():
            lines.append(log_line(
                moment, f"{unit} elapsed={rate * units:.3f} processed={units}"
                        f" rate={1 / rate:.1f}/s"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def estimate(self) -> dict:
        status, body, ctype = self.get("/api/process/estimate")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)


class TestTheGroupQuestionIsPricedByTheGroup(EstimateTestBase):
    """Test 1 and §1. The old flat rate per group was the price of a PAIR applied to
    every size, which is the arithmetic that turns 1.9 minutes into 0.5."""

    def price(self, frames: int) -> float:
        cfg = self.cfg.estimate
        return cfg.keeper_call_sec + cfg.keeper_frame_sec * frames

    def test_a_group_of_five_costs_more_than_a_group_of_three(self):
        three = ui._keeper_seconds(self.cfg, [[1, 2, 3]])
        five = ui._keeper_seconds(self.cfg, [[1, 2, 3, 4, 5]])
        self.assertGreater(five, three)
        self.assertAlmostEqual(three, self.price(3))
        self.assertAlmostEqual(five, self.price(5))

    def test_the_bill_is_the_sum_over_the_groups_not_an_average_times_a_count(self):
        """Two groups cost exactly what they cost apart — and two collections with the
        SAME number of groups get different bills when the groups differ in size. That
        is the property the old flat rate per group did not have, and the reason it
        priced an archive of nines, tens and elevens as an archive of pairs."""
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=3,
                                             keeper_max_frames=5)
        mixed = ui._keeper_seconds(self.cfg, [list(range(3)), list(range(5))])
        self.assertAlmostEqual(mixed, self.price(3) + self.price(5))
        self.assertNotAlmostEqual(
            mixed, ui._keeper_seconds(self.cfg, [list(range(3)), list(range(3))]))
        # Where the price stops being a straight line — the cap on the frames a prompt
        # may hold — the sum and the average part company outright: two groups of 3 and
        # 11 are not two groups of 7.
        capped = ui._keeper_seconds(self.cfg, [list(range(3)), list(range(11))])
        self.assertAlmostEqual(capped, self.price(3) + self.price(5))
        self.assertNotAlmostEqual(capped, 2 * self.price(5))

    def test_only_the_frames_that_are_actually_sent_are_paid_for(self):
        """`dedup.keeper_max_frames` caps what one question holds — the rest of a group
        is never shown to the model, so pricing it would invent work."""
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_max_frames=5)
        eleven = ui._keeper_seconds(self.cfg, [list(range(11))])
        self.assertAlmostEqual(eleven, self.price(5))

    def test_a_group_below_the_configured_size_is_not_paid_for(self):
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=3)
        self.assertAlmostEqual(ui._keeper_seconds(self.cfg, [[1, 2]]), 0.0)

    def test_the_endpoint_sums_the_real_groups_of_this_index(self):
        self.add_group("three", 3, "f" * 16)
        self.add_group("five", 5, "0" * 16)
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=3,
                                             keeper_max_frames=5)
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["counts"]["keeper"], 2)
        self.assertAlmostEqual(data["seconds"]["keeper"],
                               round(self.price(3) + self.price(5), 1))

    def test_the_two_prices_are_settings_rather_than_constants_in_the_page(self):
        """The last set of constants sat in `ui.py` for two features being wrong. These
        move with the config, so a slower machine can be told about it."""
        self.cfg.estimate = EstimateConfig(keeper_call_sec=10.0, keeper_frame_sec=1.0)
        self.assertAlmostEqual(ui._keeper_seconds(self.cfg, [[1, 2, 3]]), 13.0)

    def test_the_config_file_carries_both_of_them(self):
        self.config_path.write_text(
            "estimate:\n  keeper_call_sec: 0.9\n  keeper_frame_sec: 2.5\n"
            "  measurement_max_age_days: 7\n", encoding="utf-8")
        loaded = load_config(self.config_path)
        self.assertAlmostEqual(loaded.estimate.keeper_call_sec, 0.9)
        self.assertAlmostEqual(loaded.estimate.keeper_frame_sec, 2.5)
        self.assertAlmostEqual(loaded.estimate.measurement_max_age_days, 7)

    def test_a_garbled_price_falls_back_instead_of_stopping_the_app(self):
        self.config_path.write_text(
            "estimate:\n  keeper_call_sec: yes\n  keeper_frame_sec: -3\n",
            encoding="utf-8")
        loaded = load_config(self.config_path)
        self.assertEqual(loaded.estimate, EstimateConfig())


class TestAnEmptyLogSaysItIsGuessing(EstimateTestBase):
    """Test 2 and §2. A default is honest; a default presented as a measurement is not."""

    def test_the_shipped_defaults_are_used(self):
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        self.start_server()
        data = self.estimate()

        self.assertAlmostEqual(data["seconds"]["faces"],
                               round(4 * ui._SEC_PER_FACES_FRAME, 1))
        self.assertAlmostEqual(data["seconds"]["base"],
                               round(4 * ui._SEC_PER_BASE_FRAME, 1))

    def test_the_answer_says_the_numbers_are_defaults(self):
        self.add_photo("a.jpg")
        self.start_server()
        data = self.estimate()

        self.assertIsNone(data["measured_at"])
        for key in ("base", "faces", "events", "products", "keeper", "quality_all"):
            with self.subTest(key=key):
                self.assertEqual(data["sources"][key], "default")

    def test_the_animal_line_is_neither_measured_nor_guessed(self):
        """It costs 0 because the prompts ride inside a CLIP call that runs anyway — a
        structural zero has no pedigree to state.

        F161 gave the master switch the same answer for the same reason: with the deep
        tier moved out into `products`, "Deep analysis (VLM)" runs nothing at all.
        """
        self.add_photo("a.jpg")
        self.start_server()
        sources = self.estimate()["sources"]
        self.assertEqual(sources["pets"], "fixed")
        self.assertEqual(sources["deep"], "fixed")

    def test_the_screen_has_somewhere_to_say_it(self):
        html = ui._render_index_html("en")
        block = html.split('id="process-costs"', 1)[1].split("</div>\n</div>", 1)[0]
        self.assertIn('id="process-costs-source"', block)
        self.assertIn("costs_source_default", html)
        self.assertIn("costs_source_measured", html)
        self.assertIn("costs_source_mixed", html)

    def test_the_note_is_written_from_the_sources_of_the_lines_that_are_on(self):
        """It describes the total standing above the button; a caveat about a stage
        nobody asked for would be a caveat about nothing."""
        html = ui._render_index_html("en")
        self.assertIn("costSources = (data && data.sources) || null;", html)
        self.assertIn("costMeasuredAt = (data && data.measured_at) || null;", html)
        self.assertIn("renderCostSource(measured, byDefault);", html)
        self.assertIn('if (source === "measured") measured = true;', html)

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in ("costs_source_measured", "costs_source_default",
                    "costs_source_mixed"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")
        for key in ("costs_source_measured", "costs_source_mixed"):
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    self.assertIn("{date}", ui._t(key, lang))


class TestTheLogIsBelievedOverTheConstants(EstimateTestBase):
    """Test 3 — the main one. A machine that has run a stage knows what that stage
    costs HERE, and no number shipped in a wheel beats that."""

    def test_the_rate_comes_from_the_run_log(self):
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertAlmostEqual(data["seconds"]["faces"],
                               round(4 * MEASURED["stage=faces"], 1))
        self.assertNotAlmostEqual(data["seconds"]["faces"],
                                  round(4 * ui._SEC_PER_FACES_FRAME, 1))
        self.assertEqual(data["sources"]["faces"], "measured")

    def test_the_base_line_adds_up_the_stages_that_always_run(self):
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertAlmostEqual(data["seconds"]["base"], round(4 * BASE_RATE, 1))
        self.assertEqual(data["sources"]["base"], "measured")

    def test_a_half_measured_line_stays_a_default(self):
        """Three measured stages plus a guessed fourth is a guess wearing a
        measurement's clothes — the base line covers four."""
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        partial = {unit: rate for unit, rate in MEASURED.items()
                   if unit != "stage=phash"}
        self.write_run_log(rates=partial)
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["sources"]["base"], "default")
        self.assertAlmostEqual(data["seconds"]["base"],
                               round(4 * ui._SEC_PER_BASE_FRAME, 1))
        # The lines whose every unit WAS measured are unaffected by the missing one.
        self.assertEqual(data["sources"]["faces"], "measured")

    def test_the_model_questions_are_priced_by_the_vlm_phase_that_asks_them(self):
        """F165 runs the deep tier ahead of faces, in `classify`, and leaves the quality
        and animal questions behind it, in `junk`. Both phases are named `junk_vlm`, so a
        line priced off the wrong one is charged the rate of a different population — and
        nothing but this case would notice."""
        ids = [self.add_photo(f"p{i}.jpg") for i in range(3)]
        for file_id in ids[:2]:
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', 'vlm', 'vlm', '2026-01-01')", (file_id,))
        self.conn.commit()
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["counts"]["products"], 2)
        self.assertAlmostEqual(data["seconds"]["products"],
                               round(2 * MEASURED["stage=classify phase=junk_vlm"], 1))
        self.assertAlmostEqual(data["seconds"]["quality_all"],
                               round(3 * MEASURED["stage=junk phase=junk_vlm"], 1))
        for key in ("products", "quality_all", "pets_verify"):
            with self.subTest(key=key):
                self.assertEqual(data["sources"][key], "measured")

    def test_the_date_of_the_measurement_travels_with_it(self):
        """"So it was for you last time" is only worth saying if "last time" is named."""
        when = datetime.now() - timedelta(days=2)
        self.add_photo("a.jpg")
        self.write_run_log(at=when)
        self.start_server()
        self.assertEqual(self.estimate()["measured_at"], when.date().isoformat())

    def test_a_run_that_writes_new_timings_re_prices_the_screen(self):
        """The payload is cached on everything it reads, and after F159 that includes
        the log: the moment a run has measured this machine is the moment the old
        prices stop being the right answer."""
        self.add_photo("a.jpg")
        self.start_server()
        self.assertEqual(self.estimate()["sources"]["faces"], "default")

        self.write_run_log()
        self.assertEqual(self.estimate()["sources"]["faces"], "measured")

    def test_a_mixture_is_reported_as_a_mixture(self):
        """The keeper line can never be measured — its question shares a log phase with
        the per-frame ones — so a run with it switched on is a mixture by construction,
        and the screen has a string for exactly that."""
        self.add_group("dup", 3, "f" * 16)
        self.write_run_log()
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["sources"]["keeper"], "default")
        self.assertEqual(data["sources"]["base"], "measured")


class TestAStaleMeasurementIsNotAMeasurement(EstimateTestBase):
    """Test 4 and §5. The same device `frame_quality.source` uses: a stored answer is
    kept only while the question behind it is still the same one."""

    def test_a_timing_from_another_version_of_the_stage_is_not_used(self):
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        self.write_run_log(build="0.0.1-before-the-rewrite")
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["sources"]["faces"], "default")
        self.assertAlmostEqual(data["seconds"]["faces"],
                               round(4 * ui._SEC_PER_FACES_FRAME, 1))
        self.assertIsNone(data["measured_at"])

    def test_a_timing_older_than_the_window_is_not_used(self):
        self.add_photo("a.jpg")
        self.write_run_log(at=datetime.now() - timedelta(days=400))
        self.start_server()
        self.assertEqual(self.estimate()["sources"]["faces"], "default")

    def test_the_window_is_a_setting(self):
        self.add_photo("a.jpg")
        self.write_run_log(at=datetime.now() - timedelta(days=400))
        self.cfg.estimate = dataclasses.replace(self.cfg.estimate,
                                                measurement_max_age_days=0)
        self.start_server()
        self.assertEqual(self.estimate()["sources"]["faces"], "measured")


class TestTheEstimateAndTheRunAgree(EstimateTestBase):
    """Test 5: on staged data the budget and the run it describes land within 10%.

    The point is not the arithmetic of one line but that the lines ADD UP to the run:
    the rates the log holds, over the populations this index holds, over the stages this
    set of checkboxes would actually execute.
    """

    def elapsed_of_a_run(self, photos: int, deep: int, groups: list[int]) -> float:
        """What a run over this collection would take at the rates in the log.

        Written out stage by stage rather than reusing the payload — the two agreeing
        because they are the same expression would prove nothing.
        """
        total = photos * BASE_RATE                    # index/geo/landmarks/phash
        total += photos * MEASURED["stage=faces"]     # faces, switched on
        total += photos * MEASURED["stage=events"]    # events, switched on
        total += deep * MEASURED["stage=classify phase=junk_vlm"]   # the deep tier
        for size in groups:                           # the keeper question
            if size >= int(self.cfg.dedup.keeper_min_group_size):
                total += (self.cfg.estimate.keeper_call_sec
                          + self.cfg.estimate.keeper_frame_sec
                          * min(size, int(self.cfg.dedup.keeper_max_frames)))
        return total

    def test_the_budget_lands_within_a_tenth_of_the_run_it_describes(self):
        sizes = [3, 5, 9]
        for index, size in enumerate(sizes):
            self.add_group(f"g{index}_", size, f"{index}" * 16)
        photos = sum(sizes)
        deep_ids = [row[0] for row in self.conn.execute(
            "SELECT id FROM files ORDER BY id LIMIT 6")]
        for file_id in deep_ids:
            self.conn.execute(
                "INSERT INTO media_class (file_id, verdict, source, tier, updated_at)"
                " VALUES (?, 'photo', 'vlm', 'vlm', '2026-01-01')", (file_id,))
        self.conn.commit()
        self.cfg.dedup = dataclasses.replace(self.cfg.dedup, keeper_min_group_size=3,
                                             keeper_max_frames=5)
        self.write_run_log()
        self.start_server()
        data = self.estimate()["seconds"]

        budget = sum(data[key] for key in
                     ("base", "faces", "events", "pets", "products", "keeper"))
        actual = self.elapsed_of_a_run(photos, len(deep_ids), sizes)
        self.assertLessEqual(abs(budget - actual), 0.1 * actual,
                             f"budget {budget:.1f}s vs run {actual:.1f}s")


class TestNoCaptionClaimsTheGroupQuestionSaves(EstimateTestBase):
    """Test 6 and §4. The same measurement that fixed the number retired the premise:
    from three frames up, one question over a group is not cheaper than asking about the
    frames one at a time — and the pairs it IS cheaper for are the ones
    `keeper_min_group_size: 3` stopped asking about.
    """

    # Claims of thrift, in the three languages the captions are written in.
    THRIFT = ("дешевл", "экономи", "выгодн", "быстрее, чем",
              "cheaper", "saves", "saving", "less time than", "faster than",
              "節約", "お得", "安上が")

    def keeper_strings(self) -> dict[str, str]:
        return {f"{key}/{lang}": value
                for key in ("process_keeper_label", "process_keeper_hint",
                            "costs_estimate_note", "costs_title")
                for lang, value in ui._UI_STRINGS[key].items()}

    def test_no_keeper_caption_promises_a_saving(self):
        for name, value in self.keeper_strings().items():
            for claim in self.THRIFT:
                with self.subTest(string=name, claim=claim):
                    self.assertNotIn(claim, value.lower())

    def test_the_caption_says_the_price_grows_with_the_group(self):
        """Silence about the cost would leave "one question per group" reading as the
        saving it is not, so the replacement states the shape of the price."""
        for lang, expected in (("ru", "растёт с размером группы"),
                               ("en", "grows with the group"),
                               ("ja", "グループの大きさに比例")):
            with self.subTest(lang=lang):
                self.assertIn(expected, ui._t("process_keeper_hint", lang))

    def test_the_caption_still_says_what_the_question_is_for(self):
        """The stage is not being talked out of: it answers something separate questions
        do not, which is now the only thing said for it."""
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertTrue(ui._t("process_keeper_hint", lang).strip())
        self.assertIn("which frame is the best one",
                      ui._t("process_keeper_hint", "en"))
