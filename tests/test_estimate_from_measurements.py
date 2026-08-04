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

F186 then retired the question this feature was found on. The two prices it was given
(`estimate.keeper_call_sec`/`keeper_frame_sec`) went with it, and so did the cases that
priced a group and the ones that policed the captions of that line — a budget must not
quote a price for a stage that no longer runs. What the feature IS survives it intact:
the log is read, believed, aged out, and named as the source of every number.

What is pinned here is the whole of that:

* an empty log falls back to the defaults AND says so (§2, test 2);
* a log with timings in it is believed over the constants (§3, test 3 — the main one);
* a timing of a stage that has changed since is not used (§5, test 4);
* the estimate and the run it describes agree within 10% on staged data (test 5).
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


class TestTheConfigSectionSurvivesTheRetiredPrices(EstimateTestBase):
    """`estimate:` outlived the two numbers it was created for (F186).

    They priced the comparative keeper question, which was asked inside the junk stage's
    VLM phase and so could never be told from the per-frame questions by the log. The
    section still holds the one key that is read by everything here — how long a timing
    from the log stays trustworthy — and a config.yaml that still carries the retired
    prices has to load exactly as it did.
    """

    def test_the_key_that_stayed_is_read(self):
        self.config_path.write_text(
            "estimate:\n  measurement_max_age_days: 7\n", encoding="utf-8")
        loaded = load_config(self.config_path)
        self.assertAlmostEqual(loaded.estimate.measurement_max_age_days, 7)

    def test_a_file_that_still_carries_the_retired_prices_loads(self):
        self.config_path.write_text(
            "estimate:\n  keeper_call_sec: 0.9\n  keeper_frame_sec: 2.5\n"
            "  measurement_max_age_days: 7\n", encoding="utf-8")
        loaded = load_config(self.config_path)
        self.assertAlmostEqual(loaded.estimate.measurement_max_age_days, 7)
        self.assertFalse(hasattr(loaded.estimate, "keeper_call_sec"))
        self.assertFalse(hasattr(loaded.estimate, "keeper_frame_sec"))

    def test_a_garbled_value_falls_back_instead_of_stopping_the_app(self):
        self.config_path.write_text(
            "estimate:\n  measurement_max_age_days: yes\n", encoding="utf-8")
        self.assertEqual(load_config(self.config_path).estimate, EstimateConfig())


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
        for key in ("base", "faces", "events", "products"):
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
        for key in ("products", "pets_verify"):
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
        """A log this machine has only partly filled prices some lines and not others,
        and the screen has a string for exactly that state.

        Until F186 the mixture was guaranteed by the keeper line, which could never be
        measured — its question shared a log phase with the per-frame ones. With that
        line gone the mixture has to be staged, which is what the partial log below is.
        """
        for i in range(4):
            self.add_photo(f"p{i}.jpg")
        partial = {unit: rate for unit, rate in MEASURED.items()
                   if unit != "stage=events"}
        self.write_run_log(rates=partial)
        self.start_server()
        data = self.estimate()

        self.assertEqual(data["sources"]["events"], "default")
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

    def elapsed_of_a_run(self, photos: int, deep: int) -> float:
        """What a run over this collection would take at the rates in the log.

        Written out stage by stage rather than reusing the payload — the two agreeing
        because they are the same expression would prove nothing. The keeper term left
        this sum with F186: the stage it priced is not run, so a budget that still
        carried it would describe a longer run than the one about to happen.
        """
        total = photos * BASE_RATE                    # index/geo/landmarks/phash
        total += photos * MEASURED["stage=faces"]     # faces, switched on
        total += photos * MEASURED["stage=events"]    # events, switched on
        total += deep * MEASURED["stage=classify phase=junk_vlm"]   # the deep tier
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
        self.write_run_log()
        self.start_server()
        data = self.estimate()["seconds"]

        budget = sum(data[key] for key in
                     ("base", "faces", "events", "pets", "products"))
        actual = self.elapsed_of_a_run(photos, len(deep_ids))
        self.assertLessEqual(abs(budget - actual), 0.1 * actual,
                             f"budget {budget:.1f}s vs run {actual:.1f}s")
