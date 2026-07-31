"""F96: the scene-text probe — everything about it that is not the model itself.

The script decides whether a place-by-reading feature is worth writing at all, so the
arithmetic behind that decision has to be trustworthy without a GPU: the sampling, the
parsing of the reply, the matching against the geo base, the two accuracy levels and the
pre-registered criteria are pure functions and are tested here with a fake reader — no
transformers, no photo.

Three of these tests are about the brief rather than about code, and they stay even if
they look pedantic:

* the bars (city >= 85%, country >= 95%, answers >= 15%) were written down BEFORE the
  measurement — `TestVerdict` pins them so a disappointing table cannot be met by
  quietly lowering one;
* abstention is the point of the method, not an inconvenience — `TestParseAnswer` pins
  that "unknown" in any of its usual spellings is an abstention and not a country;
* nothing the script prints or caches may carry the text the model read off a personal
  frame, or identify a file (the rule of the document verdict, and of
  measure_streetclip.py before this).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_scene_text.py"


def _load_script():
    """Import scripts/measure_scene_text.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_scene_text", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_script()

# The bundled base is loaded once for the whole file: every GeoResolver() instance reads
# the TSVs and builds its own KD-tree.
_RESOLVER: list = []


def resolver():
    if not _RESOLVER:
        _RESOLVER.append(probe.GeoResolver())
    return _RESOLVER[0]


MOSCOW = 524901   # PPLC, RU
KHIMKI = 550280   # PPLA2, RU — ~20 km from the Moscow anchor
BANGKOK = 1609350  # PPLC, TH


def answer(true_cc="RU", pred_cc="RU", said_country=True, said_city=True,
           city_hit=True, city_near=False, has_text=True, labelled=True,
           seconds=0.8, file_id=1):
    return probe.Answer(file_id, true_cc, pred_cc, said_country, said_city, city_hit,
                        city_near, has_text, labelled, seconds)


def row(file_id: int, country="RU", city="Moscow", gid=MOSCOW, path="x.jpg"):
    """A stand-in for the sqlite3.Row the query returns (same subscript access)."""
    return {"id": file_id, "path": path, "country": country, "city": city,
            "city_geonameid": gid}


def reply(text="yes", country="Thailand", city="Bangkok") -> str:
    return f"TEXT: {text}\nCOUNTRY: {country}\nCITY: {city}"


class TestParseAnswer(unittest.TestCase):
    """The reply is the only thing the model gives us — parsing it is the measurement."""

    def test_the_requested_three_line_format(self):
        parsed = probe.parse_answer(reply())
        self.assertEqual((parsed.has_text, parsed.country, parsed.city),
                         (True, "Thailand", "Bangkok"))
        self.assertTrue(parsed.labelled)

    def test_unknown_is_an_abstention_not_a_place(self):
        for word in ("unknown", "Unknown.", "N/A", "none", "-", "not sure", "неизвестно"):
            parsed = probe.parse_answer(reply(text="no", country=word, city=word))
            self.assertEqual((parsed.country, parsed.city), ("", ""), word)

    def test_one_level_may_abstain_while_the_other_answers(self):
        parsed = probe.parse_answer(reply(country="Japan", city="unknown"))
        self.assertEqual((parsed.country, parsed.city), ("Japan", ""))

    def test_tolerant_to_case_dashes_decoration_and_a_preamble(self):
        parsed = probe.parse_answer(
            "Sure, here is my answer.\ntext - Yes\n**country**: *Turkey*\n"
            "  cITy : \"Istanbul\".\n")
        self.assertEqual((parsed.has_text, parsed.country, parsed.city),
                         (True, "Turkey", "Istanbul"))

    def test_a_city_written_with_its_country_keeps_the_city(self):
        parsed = probe.parse_answer(reply(city="Bangkok, Thailand"))
        self.assertEqual(parsed.city, "Bangkok")

    def test_a_country_written_after_a_city_keeps_the_country(self):
        parsed = probe.parse_answer(reply(country="Bangkok, Thailand"))
        self.assertEqual(parsed.country, "Thailand")

    def test_the_first_occurrence_of_a_line_wins(self):
        parsed = probe.parse_answer(
            "COUNTRY: Thailand\nCOUNTRY: Malaysia\nCITY: unknown\nTEXT: no")
        self.assertEqual(parsed.country, "Thailand")

    def test_text_flag_is_no_unless_the_model_said_yes(self):
        self.assertFalse(probe.parse_answer(reply(text="no")).has_text)
        self.assertFalse(probe.parse_answer("COUNTRY: unknown").has_text)
        self.assertTrue(probe.parse_answer(reply(text="Yes, a shop sign")).has_text)

    def test_a_reply_in_no_format_at_all_is_an_unlabelled_abstention(self):
        parsed = probe.parse_answer("I cannot determine the location of this photo.")
        self.assertFalse(parsed.labelled)
        self.assertEqual((parsed.country, parsed.city), ("", ""))

    def test_an_empty_reply_does_not_crash(self):
        parsed = probe.parse_answer("")
        self.assertEqual((parsed.has_text, parsed.country, parsed.city), (False, "", ""))


class TestPositionalFallback(unittest.TestCase):
    """The shape the model actually produces: three bare lines, no labels.

    Measured on the first real run — 239 replies out of 300 came back like this, and
    scoring them as abstentions measured the parser instead of the hypothesis. The
    labelled path stays primary; this one only runs when there is no label anywhere.
    """

    def test_three_bare_lines_are_read_by_position(self):
        # the exact reply from the run: the model read the sign and named the city
        parsed = probe.parse_answer("Yes.\nUnknown.\nBangkok.")
        self.assertEqual((parsed.has_text, parsed.country, parsed.city),
                         (True, "", "Bangkok"))
        self.assertFalse(parsed.labelled)  # still reported as off-format

    def test_a_positional_country_and_city_both_arrive(self):
        parsed = probe.parse_answer("Yes\nThailand\nBangkok")
        self.assertEqual((parsed.country, parsed.city), ("Thailand", "Bangkok"))

    def test_all_unknown_stays_a_full_abstention(self):
        # the asymmetric cost: an invented city is worse than an empty field
        parsed = probe.parse_answer("Yes.\nUnknown.\nUnknown.")
        self.assertEqual((parsed.country, parsed.city), ("", ""))

    def test_a_preamble_does_not_shift_the_positions(self):
        parsed = probe.parse_answer(
            "Sure, here is what I can read.\nYes\nTurkey\nIstanbul")
        self.assertEqual((parsed.country, parsed.city), ("Turkey", "Istanbul"))

    def test_trailing_chatter_does_not_shift_them_either(self):
        parsed = probe.parse_answer("Yes\nJapan\nKyoto\nI hope this helps.")
        self.assertEqual((parsed.country, parsed.city), ("Japan", "Kyoto"))

    def test_two_lines_are_an_abstention_not_a_guessed_mapping(self):
        # which line is the country and which the city would be a coin flip
        parsed = probe.parse_answer("Yes\nThailand")
        self.assertEqual((parsed.country, parsed.city), ("", ""))
        self.assertFalse(parsed.labelled)

    def test_a_partly_labelled_reply_goes_by_labels_alone(self):
        parsed = probe.parse_answer("Yes\nCOUNTRY: Thailand\nBangkok")
        self.assertTrue(parsed.labelled)
        self.assertEqual((parsed.country, parsed.city), ("Thailand", ""))
        self.assertFalse(parsed.has_text)  # no TEXT label — nothing is invented for it

    def test_the_positional_values_are_cleaned_like_the_labelled_ones(self):
        parsed = probe.parse_answer("**Yes**\n*Indonesia.*\n\"Denpasar\".")
        self.assertEqual((parsed.has_text, parsed.country, parsed.city),
                         (True, "Indonesia", "Denpasar"))

    def test_a_no_on_the_first_line_is_still_read(self):
        parsed = probe.parse_answer("No.\nUnknown.\nUnknown.")
        self.assertEqual((parsed.has_text, parsed.country, parsed.city),
                         (False, "", ""))


class TestMatchCountry(unittest.TestCase):
    def test_the_plain_english_name_resolves(self):
        self.assertEqual(probe.match_country("Thailand", resolver()), "TH")
        self.assertEqual(probe.match_country("  indonesia ", resolver()), "ID")

    def test_informal_spellings_the_base_does_not_carry(self):
        self.assertEqual(probe.match_country("USA", resolver()), "US")
        self.assertEqual(probe.match_country("UK", resolver()), "GB")
        self.assertEqual(probe.match_country("Türkiye", resolver()), "TR")

    def test_an_unrecognized_name_is_empty_not_a_crash(self):
        # counted as a wrong answer downstream, and reported apart so a parser problem
        # is not mistaken for a model problem
        self.assertEqual(probe.match_country("Middle-earth", resolver()), "")
        self.assertEqual(probe.match_country("", resolver()), "")


class TestMatchCity(unittest.TestCase):
    def test_the_truth_city_by_name(self):
        self.assertEqual(probe.match_city("Moscow", "Moscow", MOSCOW, resolver()),
                         (True, False))

    def test_a_different_spelling_of_the_same_geonameid_is_a_hit(self):
        # the base holds the localized names; the layout would use the same folder
        self.assertEqual(probe.match_city("Москва", "Moscow", MOSCOW, resolver()),
                         (True, False))

    def test_a_neighbouring_town_is_a_near_miss_not_a_hit(self):
        hit, near = probe.match_city("Khimki", "Moscow", MOSCOW, resolver())
        self.assertFalse(hit)   # still the wrong folder
        self.assertTrue(near)   # but the model knows the region

    def test_another_continent_is_neither(self):
        self.assertEqual(probe.match_city("Bangkok", "Moscow", MOSCOW, resolver()),
                         (False, False))

    def test_an_unknown_city_name_is_neither(self):
        self.assertEqual(probe.match_city("Nowhereville", "Moscow", MOSCOW, resolver()),
                         (False, False))

    def test_no_answer_is_neither(self):
        self.assertEqual(probe.match_city("", "Moscow", MOSCOW, resolver()), (False, False))

    def test_without_a_truth_geonameid_only_the_name_can_match(self):
        self.assertEqual(probe.match_city("Moscow", "Moscow", None, resolver()),
                         (True, False))
        self.assertEqual(probe.match_city("Khimki", "Moscow", None, resolver()),
                         (False, False))

    def test_the_near_radius_is_a_parameter_and_excludes_by_distance(self):
        self.assertEqual(probe.match_city("Khimki", "Moscow", MOSCOW, resolver(),
                                          near_km=1.0), (False, False))
        self.assertEqual(probe.match_city("Bangkok", "Moscow", BANGKOK, resolver()),
                         (True, False))


class TestRandomSample(unittest.TestCase):
    """The collection is grouped into trips: the first 300 rows are one city."""

    def _rows(self):
        return [row(i, country="RU" if i < 90 else "TH") for i in range(100)]

    def test_the_size_is_respected_and_the_rows_are_not_the_first_ones(self):
        picked = probe.random_sample(self._rows(), 10, seed=1)
        self.assertEqual(len(picked), 10)
        self.assertNotEqual([r["id"] for r in picked], list(range(10)))

    def test_deterministic_for_a_seed_and_sensitive_to_it(self):
        ids = [r["id"] for r in probe.random_sample(self._rows(), 20, seed=7)]
        self.assertEqual(ids, [r["id"] for r in probe.random_sample(self._rows(), 20, 7)])
        self.assertNotEqual(ids, [r["id"] for r in probe.random_sample(self._rows(), 20, 8)])

    def test_no_duplicates_and_never_more_than_there_is(self):
        picked = probe.random_sample(self._rows(), 500, seed=3)
        self.assertEqual(len(picked), 100)
        self.assertEqual(len({r["id"] for r in picked}), 100)

    def test_empty_input(self):
        self.assertEqual(probe.random_sample([], 10, seed=1), [])


class TestLevels(unittest.TestCase):
    """Accuracy is counted over the answers, the answer rate over the whole sample."""

    def _answers(self):
        return [
            answer(true_cc="TH", pred_cc="TH", city_hit=True),
            answer(true_cc="TH", pred_cc="ID", city_hit=False, city_near=False),
            answer(true_cc="RU", pred_cc="", said_country=False, said_city=False,
                   city_hit=False),
            answer(true_cc="RU", pred_cc="", said_country=True, said_city=False,
                   city_hit=False),  # named a country the base does not know
        ]

    def test_the_two_levels_have_their_own_denominators(self):
        country, city = probe.levels(self._answers())
        self.assertEqual((country.answered, country.correct, country.total), (3, 1, 4))
        self.assertAlmostEqual(country.accuracy, 1 / 3)
        self.assertAlmostEqual(country.answer_rate, 0.75)
        self.assertEqual((city.answered, city.correct), (2, 1))
        self.assertAlmostEqual(city.accuracy, 0.5)
        self.assertAlmostEqual(city.answer_rate, 0.5)

    def test_an_unresolved_country_counts_as_an_answer_and_as_a_miss(self):
        # the model DID speak — hiding it in the abstention rate would flatter accuracy
        [only] = [a for a in self._answers() if a.unresolved_country]
        self.assertTrue(only.said_country)
        self.assertFalse(only.country_correct)

    def test_a_silent_model_does_not_divide_by_zero(self):
        country, city = probe.levels([answer(said_country=False, said_city=False,
                                             city_hit=False, pred_cc="")])
        self.assertEqual((country.accuracy, city.accuracy), (0.0, 0.0))
        self.assertFalse(country.passes)

    def test_no_answers_at_all(self):
        country, city = probe.levels([])
        self.assertEqual((country.total, country.answer_rate), (0, 0.0))
        self.assertFalse(city.passes)


class TestVerdict(unittest.TestCase):
    """The criteria from the brief, pinned so the table cannot be met halfway."""

    def _level(self, name, answered, correct, total, bar):
        return probe.Level(name, answered, correct, total, bar)

    def _country(self, answered=50, correct=49, total=100):
        return self._level("страна", answered, correct, total, probe.MIN_COUNTRY_ACCURACY)

    def _city(self, answered=50, correct=45, total=100):
        return self._level("город", answered, correct, total, probe.MIN_CITY_ACCURACY)

    def test_the_bars_are_the_ones_written_in_the_brief(self):
        self.assertEqual((probe.MIN_CITY_ACCURACY, probe.MIN_COUNTRY_ACCURACY,
                          probe.MIN_ANSWER_RATE), (0.85, 0.95, 0.15))

    def test_all_three_cleared_is_outcome_a(self):
        letter, line = probe.verdict(self._country(), self._city())
        self.assertEqual(letter, "A")
        self.assertIn("ИСХОД A", line)

    def test_country_without_city_is_outcome_b(self):
        letter, line = probe.verdict(self._country(), self._city(correct=40))
        self.assertEqual(letter, "B")
        self.assertIn("только страну", line)

    def test_low_country_accuracy_is_outcome_c(self):
        letter, _line = probe.verdict(self._country(correct=44), self._city())
        self.assertEqual(letter, "C")

    def test_accuracy_without_answers_is_outcome_c(self):
        # perfect on 10 frames out of 300 is not a feature, it is a rounding error
        letter, line = probe.verdict(self._country(answered=10, correct=10, total=300),
                                     self._city(answered=10, correct=10, total=300))
        self.assertEqual(letter, "C")
        self.assertIn("ИСХОД C", line)

    def test_exactly_on_the_bars_qualifies(self):
        # 15% answers, 95% country, 85% city — the criterion is inclusive
        letter, _line = probe.verdict(
            self._level("страна", 20, 19, 133, probe.MIN_COUNTRY_ACCURACY),
            self._level("город", 20, 17, 133, probe.MIN_CITY_ACCURACY))
        self.assertEqual(letter, "A")


class TestTiming(unittest.TestCase):
    def test_median_p90_and_the_forecast(self):
        answers = [answer(seconds=s) for s in (0.5, 0.7, 0.9, 1.1, 9.0)]
        median_ms, p90_ms, total_s = probe.timing(answers)
        self.assertAlmostEqual(median_ms, 900.0)      # the warm-up frame does not move it
        self.assertAlmostEqual(p90_ms, 9000.0)
        self.assertAlmostEqual(total_s, 12.2)
        # 900 ms over 5 092 files -> ~76 minutes
        self.assertAlmostEqual(probe.forecast_minutes(median_ms), 76.38, places=2)

    def test_the_forecast_is_over_the_place_less_population(self):
        self.assertEqual(probe.CANDIDATE_FILES, 5092)

    def test_no_frames_is_zero_not_a_crash(self):
        self.assertEqual(probe.timing([]), (0.0, 0.0, 0.0))


class TestScore(unittest.TestCase):
    """The seam between the model and the arithmetic."""

    def test_a_reply_becomes_outcomes(self):
        rows = [row(1, country="TH", city="Bangkok", gid=BANGKOK),
                row(2, country="RU", city="Moscow", gid=MOSCOW)]
        replies = [reply(), reply(country="Thailand", city="Bangkok")]
        answers = probe.score(lambda _p: replies.pop(0), rows, resolver())
        self.assertEqual([(a.file_id, a.pred_cc, a.city_hit) for a in answers],
                         [(1, "TH", True), (2, "TH", False)])
        self.assertTrue(all(a.seconds >= 0.0 for a in answers))

    def test_an_unlabelled_reply_is_scored_not_thrown_away(self):
        # the regression the first run exposed: a correct answer in the wrong format
        answers = probe.score(lambda _p: "Yes.\nRussia.\nMoscow.", [row(1)], resolver())
        self.assertEqual((answers[0].pred_cc, answers[0].city_hit), ("RU", True))
        self.assertFalse(answers[0].labelled)

    def test_an_abstention_is_recorded_as_one(self):
        answers = probe.score(lambda _p: reply(text="no", country="unknown",
                                               city="unknown"), [row(1)], resolver())
        self.assertEqual((answers[0].said_country, answers[0].said_city,
                          answers[0].has_text), (False, False, False))
        self.assertEqual(answers[0].pred_cc, "")

    def test_undecodable_frames_are_dropped_not_counted_as_abstentions(self):
        # a broken file left in would depress the answer rate the criterion is read from
        rows = [row(1), row(2), row(3)]
        replies = {1: reply(country="Russia", city="Moscow"), 2: None,
                   3: reply(country="Russia", city="Moscow")}
        answers = probe.score(lambda p: replies[int(p)],
                              [dict(r, path=str(r["id"])) for r in rows], resolver())
        self.assertEqual([a.file_id for a in answers], [1, 3])

    def test_progress_output_carries_neither_a_path_nor_the_recognized_text(self):
        rows = [dict(row(1), path=r"D:\SORT\2019 Bali\IMG_0001.jpg")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            answers = probe.score(
                lambda _p: reply(country="Indonesia", city="Denpasar")
                + "\nsign: Apteka No 5, Lenina 12",
                rows, resolver())
        printed = buf.getvalue()
        for forbidden in ("IMG_0001", "SORT", "Apteka", "Lenina", "Denpasar"):
            self.assertNotIn(forbidden, printed)
        self.assertNotIn("Denpasar", repr(answers))


class TestCache(unittest.TestCase):
    def test_round_trip(self):
        answers = [answer(true_cc="TH", pred_cc="ID", city_hit=False, city_near=True,
                          seconds=1.2345, file_id=11)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            probe.save_cache(path, answers, {"device": "cuda", "seed": 1})
            back, meta = probe.load_cache(path)
        self.assertEqual((back[0].file_id, back[0].true_cc, back[0].pred_cc),
                         (11, "TH", "ID"))
        self.assertEqual((back[0].city_hit, back[0].city_near), (False, True))
        self.assertAlmostEqual(back[0].seconds, 1.2345)
        self.assertEqual(meta["device"], "cuda")

    def test_a_foreign_version_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"version": 99, "answers": []}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                probe.load_cache(path)

    def test_the_cache_holds_no_paths_and_no_city_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            probe.save_cache(path, [answer(file_id=5)], {"model": probe.MODEL_ID})
            text = path.read_text(encoding="utf-8")
        for forbidden in (".jpg", "Moscow", "Bangkok"):
            self.assertNotIn(forbidden, text)


class TestReport(unittest.TestCase):
    """A table about places must not become a list of where the photos were taken."""

    def _answers(self):
        return [answer(true_cc="TH", pred_cc="TH", file_id=1),
                answer(true_cc="TH", pred_cc="ID", city_hit=False, city_near=True,
                       file_id=2),
                answer(true_cc="RU", pred_cc="", said_country=False, said_city=False,
                       city_hit=False, has_text=False, file_id=3),
                answer(true_cc="RU", pred_cc="", said_country=False, said_city=False,
                       city_hit=False, has_text=True, labelled=False, file_id=4)]

    def test_the_blocks_render_from_aggregates_only(self):
        text = "\n".join([
            probe.format_report(self._answers(), {"seed": 1, "model": probe.MODEL_ID,
                                                  "device": "cuda", "max_edge": 896}),
            probe.format_country_table(probe.country_table(self._answers())),
        ])
        self.assertIn("TH", text)
        self.assertIn("прогноз", text)
        for forbidden in (".jpg", "\\", "Moscow"):
            self.assertNotIn(forbidden, text)

    def test_the_report_states_abstention_format_and_text_counts(self):
        text = probe.format_report(self._answers(), {"seed": 1})
        self.assertIn("воздержалась полностью: 2", text)
        self.assertIn("ответов не по формату: 1", text)
        self.assertIn("текст в кадре (по мнению модели): 3", text)
        # a frame with text where the model still said nothing — the interesting case
        self.assertIn("из них с текстом в кадре: 1", text)

    def test_the_country_table_counts_frames_answers_and_hits(self):
        rows = {r.cc: r for r in probe.country_table(self._answers())}
        self.assertEqual((rows["TH"].n, rows["TH"].answered, rows["TH"].correct),
                         (2, 2, 1))
        self.assertEqual((rows["RU"].n, rows["RU"].answered, rows["RU"].correct),
                         (2, 0, 0))


class TestQueries(unittest.TestCase):
    """The SQL runs against the real schema, so a column rename cannot go unnoticed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "test.db"
        conn = connect(self.db)
        rows = [
            # path, cc, city, confidence, dup_of, error, verdict
            ("a.jpg", "RU", "Moscow", "exact_gps", None, None, "photo"),
            ("b.jpg", "TH", "Bangkok", "exact_gps", None, None, None),
            ("c.jpg", None, None, "unknown", None, None, None),          # no place
            ("d.jpg", "RU", "Moscow", "exact_gps", 1, None, None),       # a duplicate
            ("e.jpg", "RU", "Moscow", "exact_gps", None, "boom", None),  # unreadable
            ("f.jpg", "RU", None, "exact_gps", None, None, None),        # no city
            ("g.jpg", "RU", "Moscow", "session_inferred", None, None, None),  # not truth
            ("h.jpg", "RU", "Moscow", "exact_gps", None, None, "document"),
            ("i.jpg", "RU", "Moscow", "exact_gps", None, None, "screenshot"),
        ]
        with conn:
            for i, (path, cc, city, conf, dup, err, verdict) in enumerate(rows, start=1):
                conn.execute(
                    """INSERT INTO files (id, path, size, mtime, ext, media_type,
                           dup_of, error, indexed_at)
                       VALUES (?, ?, 1, 0, 'jpg', 'photo', ?, ?, '2026-01-01')""",
                    (i, path, dup, err))
                conn.execute(
                    """INSERT INTO places (file_id, country, city, city_geonameid,
                           confidence, updated_at)
                       VALUES (?, ?, ?, 1, ?, '2026-01-01')""", (i, cc, city, conf))
                if verdict:
                    conn.execute(
                        """INSERT INTO media_class (file_id, verdict, source, updated_at)
                           VALUES (?, ?, 'clip', '2026-01-01')""", (i, verdict))
        conn.close()

    def test_only_gps_truth_with_a_city_gets_in(self):
        self.assertEqual([r["id"] for r in probe.truth_rows(str(self.db))], [1, 2])

    def test_documents_and_screenshots_never_reach_the_model(self):
        # documents are not opened at all, and a screenshot of a map would produce a
        # flattering number that real photos would never reproduce
        ids = [r["id"] for r in probe.truth_rows(str(self.db))]
        self.assertNotIn(8, ids)
        self.assertNotIn(9, ids)

    def test_the_truth_carries_the_city_and_its_geonameid(self):
        first = probe.truth_rows(str(self.db))[0]
        self.assertEqual((first["country"], first["city"], first["city_geonameid"]),
                         ("RU", "Moscow", 1))

    def test_the_database_is_opened_read_only(self):
        before = self.db.read_bytes()
        probe.truth_rows(str(self.db))
        self.assertEqual(self.db.read_bytes(), before)

    def test_existing_drops_files_that_are_no_longer_on_disk(self):
        here = str(Path(__file__).resolve())
        rows = [row(1, path=here), row(2, path=here + ".missing")]
        self.assertEqual([r["id"] for r in probe.existing(rows)], [1])


class TestPrompt(unittest.TestCase):
    """The prompt is the method: it must allow the model to say nothing."""

    def test_it_asks_for_the_three_lines_the_parser_reads(self):
        for label in ("TEXT:", "COUNTRY:", "CITY:"):
            self.assertIn(label, probe.PROMPT)

    def test_it_permits_and_demands_abstention(self):
        self.assertIn("unknown", probe.PROMPT)
        self.assertIn("Do NOT guess", probe.PROMPT)


if __name__ == "__main__":
    unittest.main()
