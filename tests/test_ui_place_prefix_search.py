"""F201: the place picker answers while the name is typed, not only when it is finished.

The field promised a combobox and delivered an exact lookup: «Моск» found nothing,
«Москв» found nothing, and on the way «Мо» found a town in Norway — so the only thing
the user saw for a whole word was «no such place, check the spelling», a message that
blames the spelling of a word that is simply not typed out yet.

What the tests below pin is the shape of an answer that is useful mid-word:
* a prefix of the name finds the city, and so does a word inside a composite name
  (TestPrefixFindsTheCity) — while a substring that starts nowhere does not, or every
  «Рим» in the base would arrive with a «Дурим» attached;
* the answer is ordered, because a prefix always matches more than one place: the exact
  name first, then the bigger city (TestOrderOfTheAnswer);
* it is short — a two-letter prefix matches thousands, and the country half never
  crowds the cities out (TestTheAnswerIsShort);
* «not found» and «not asked yet» are different answers and the payload says which
  (TestSearchedTellsTheTwoEmptyAnswersApart);
* and none of it reaches the network (TestNothingLeavesTheMachine).
"""
from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta import ui
from sorta.geodata import GeoResolver
from sorta.ui.layout import (
    _PLACE_COUNTRY_LIMIT, _PLACE_SEARCH_LIMIT, _PLACE_SEARCH_MIN_QUERY, _places_search,
    _places_search_payload,
)

# A tiny world built for the failure that started F201. It holds, on purpose:
# «Мо» — a real town in Norway, the accidental hit on the way to «Москва»; Москва with
# a population and Мосальск without one; a composite «Нижний Новгород»; «Дурим», which
# contains «Рим» without starting with it; and enough «Мост...» to overflow a short list.
_MOSCOW, _MOSALSK, _MO, _NIZHNY, _ROSTOV = 524901, 526401, 3144873, 520555, 501175
_ROME, _DURIM, _MAGADAN = 3169070, 900001, 2123628
_FILLER_BASE = 910000
_FILLER_COUNT = 20

_PLACES = [
    (_MOSCOW, 55.7522, 37.6156, "PPLC", "RU", "48", "", "Moscow", 10381222),
    (_MOSALSK, 54.4867, 34.9764, "PPLA3", "RU", "30", "", "Mosalsk", 4300),
    (_MO, 66.3167, 14.1667, "PPLA3", "NO", "18", "", "Mo", 1000),
    (_NIZHNY, 56.3287, 44.0020, "PPLA", "RU", "51", "", "Nizhniy Novgorod", 1284164),
    (_ROSTOV, 47.2313, 39.7233, "PPLA", "RU", "61", "", "Rostov-na-Donu", 1074482),
    (_ROME, 41.8919, 12.5113, "PPLC", "IT", "07", "", "Rome", 2318895),
    (_DURIM, 40.0, 20.0, "PPLA3", "AL", "", "", "Durim", 500),
    (_MAGADAN, 59.5638, 150.8035, "PPLA", "RU", "44", "", "Magadan", 95982),
] + [
    # Same prefix as Москва, no population — the tail a short list has to cut off.
    (_FILLER_BASE + n, 55.0 + n / 100, 37.0, "PPLA3", "RU", "48", "", f"Mostograd{n:02d}", 0)
    for n in range(_FILLER_COUNT)
]
_ADMIN1 = [("RU", "48", 400, "Moscow"), ("RU", "51", 401, "Nizhny Novgorod Oblast")]
# Six countries starting with «Ма»/"Ma" — one prefix matching more countries than a
# 12-line list can afford to spend on them.
_COUNTRIES = [("RU", 600, "Russia"), ("NO", 601, "Norway"), ("IT", 602, "Italy"),
              ("AL", 603, "Albania"), ("MG", 610, "Madagascar"), ("MW", 611, "Malawi"),
              ("MY", 612, "Malaysia"), ("MV", 613, "Maldives"), ("ML", 614, "Mali"),
              ("MT", 615, "Malta")]
_NAMES = [
    (_MOSCOW, "ru", "Москва"), (_MOSCOW, "en", "Moscow"),
    (_MOSALSK, "ru", "Мосальск"), (_MOSALSK, "en", "Mosalsk"),
    (_MO, "ru", "Мо"), (_MO, "en", "Mo"),
    (_NIZHNY, "ru", "Нижний Новгород"), (_NIZHNY, "en", "Nizhny Novgorod"),
    (_ROSTOV, "ru", "Ростов-на-Дону"), (_ROSTOV, "en", "Rostov-on-Don"),
    (_ROME, "ru", "Рим"), (_ROME, "en", "Rome"),
    (_DURIM, "ru", "Дурим"), (_DURIM, "en", "Durim"),
    (_MAGADAN, "ru", "Магадан"), (_MAGADAN, "en", "Magadan"),
    (600, "ru", "Россия"), (600, "en", "Russia"),
    (601, "ru", "Норвегия"), (601, "en", "Norway"),
    (602, "ru", "Италия"), (602, "en", "Italy"),
    (603, "ru", "Албания"), (603, "en", "Albania"),
    (610, "ru", "Мадагаскар"), (611, "ru", "Малави"), (612, "ru", "Малайзия"),
    (613, "ru", "Мальдивы"), (614, "ru", "Мали"), (615, "ru", "Мальта"),
] + [(_FILLER_BASE + n, "ru", f"Мостоград{n:02d}") for n in range(_FILLER_COUNT)]


def _write_fixture(data_dir: Path) -> None:
    """The mini bundled base — the real GeoResolver over a handful of rows."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "places.tsv").write_text(
        "".join("\t".join(str(v) for v in row) + "\n" for row in _PLACES),
        encoding="utf-8")
    (data_dir / "names.tsv").write_text(
        "".join(f"{gid}\t{lang}\t{name}\n" for gid, lang, name in _NAMES),
        encoding="utf-8")
    (data_dir / "admin1.tsv").write_text(
        "".join(f"{cc}\t{a1}\t{gid}\t{en}\n" for cc, a1, gid, en in _ADMIN1),
        encoding="utf-8")
    (data_dir / "countries.tsv").write_text(
        "".join(f"{cc}\t{gid}\t{en}\n" for cc, gid, en in _COUNTRIES),
        encoding="utf-8")


class PrefixSearchTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        data_dir = Path(self.tmp.name) / "geo"
        _write_fixture(data_dir)
        self.resolver = GeoResolver(data_dir=data_dir)
        patcher = patch("sorta.ui.layout._geo_resolver", return_value=self.resolver)
        patcher.start()
        self.addCleanup(patcher.stop)

    def search(self, query: str, lang: str = "ru") -> list[dict]:
        return _places_search(query, lang)  # type: ignore[arg-type]

    def city_ids(self, query: str, lang: str = "ru") -> list[int]:
        return [r["city_geonameid"] for r in self.search(query, lang)
                if r["kind"] == "city"]


class TestPrefixFindsTheCity(PrefixSearchTestBase):
    def test_the_beginning_of_a_name_finds_the_city(self):
        # The report that started the feature: «Моск» used to answer nothing at all.
        self.assertIn(_MOSCOW, self.city_ids("Моск"))

    def test_every_step_of_typing_keeps_finding_it(self):
        for typed in ("Мо", "Мос", "Моск", "Москв", "Москва"):
            with self.subTest(typed=typed):
                self.assertIn(_MOSCOW, self.city_ids(typed))

    def test_a_word_inside_a_composite_name_is_a_beginning_too(self):
        self.assertIn(_NIZHNY, self.city_ids("Новг"))

    def test_a_substring_that_starts_nowhere_is_not_a_match(self):
        # «Рим» sits inside «Дурим», and a substring search would drag it in — along
        # with every other name that merely contains the letters.
        found = self.city_ids("Рим")
        self.assertIn(_ROME, found)
        self.assertNotIn(_DURIM, found)

    def test_the_english_name_is_found_by_prefix_as_well(self):
        # A place is looked up by the name the user knows it under, in any of the three
        # languages; `lang` only decides the labels.
        self.assertIn(_MOSCOW, self.city_ids("Mosc", lang="en"))

    def test_case_and_edge_spaces_still_do_not_matter(self):
        self.assertEqual(self.city_ids("  МОСКВ  "), self.city_ids("москв"))

    def test_a_full_name_still_works_as_it_did(self):
        self.assertEqual(self.city_ids("Москва")[0], _MOSCOW)

    def test_a_name_that_is_in_no_language_finds_nothing(self):
        self.assertEqual(self.search("Шмиргород"), [])


class TestOrderOfTheAnswer(PrefixSearchTestBase):
    def test_an_exact_name_comes_before_everything_it_is_a_prefix_of(self):
        # «Мо» IS a town in Norway. Москва is ten thousand times bigger and still comes
        # second: the user who typed the whole name typed the whole name.
        self.assertEqual(self.city_ids("Мо")[0], _MO)

    def test_the_bigger_city_comes_before_the_hamlet(self):
        found = self.city_ids("Мос")
        self.assertEqual(found[0], _MOSCOW)
        self.assertLess(found.index(_MOSCOW), found.index(_MOSALSK))

    def test_places_without_a_population_come_last_and_in_name_order(self):
        found = self.city_ids("Мост")
        labels = [r["label"] for r in self.search("Мост")]
        self.assertEqual(len(found), len(labels))
        self.assertEqual(labels, sorted(labels))

    def test_the_country_is_the_first_line_of_the_answer(self):
        # A wrong country is visible in the plan at a glance, a wrong city is not.
        results = self.search("Рос")
        self.assertEqual(results[0]["kind"], "country")
        self.assertEqual(results[0]["country"], "RU")
        self.assertIn(_ROSTOV, self.city_ids("Рос"))

    def test_a_city_only_prefix_answers_with_cities_only(self):
        self.assertEqual({r["kind"] for r in self.search("Моск")}, {"city"})

    def test_the_canonical_english_anchor_still_travels_with_a_city(self):
        # `places.city` is the en/asciiname anchor everywhere in the program.
        city = next(r for r in self.search("Моск") if r["city_geonameid"] == _MOSCOW)
        self.assertEqual(city["city"], "Moscow")

    def test_the_label_tells_same_prefixed_cities_apart(self):
        city = next(r for r in self.search("Нижн") if r["city_geonameid"] == _NIZHNY)
        self.assertIn("Нижний Новгород", city["label"])
        self.assertIn("Россия", city["label"])


class TestTheAnswerIsShort(PrefixSearchTestBase):
    def test_a_broad_prefix_is_cut_to_the_limit(self):
        self.assertGreater(_FILLER_COUNT, _PLACE_SEARCH_LIMIT)
        self.assertEqual(len(self.search("Мос")), _PLACE_SEARCH_LIMIT)

    def test_the_countries_never_take_the_whole_list(self):
        # Six countries start with «Ма» in the fixture; Магадан has to survive them.
        results = self.search("Ма")
        countries = [r for r in results if r["kind"] == "country"]
        self.assertEqual(len(countries), _PLACE_COUNTRY_LIMIT)
        self.assertIn(_MAGADAN, self.city_ids("Ма"))


class TestSearchedTellsTheTwoEmptyAnswersApart(PrefixSearchTestBase):
    def test_one_letter_is_not_a_search(self):
        payload = _places_search_payload("М", "ru")
        self.assertEqual(payload["results"], [])
        self.assertFalse(payload["searched"])

    def test_the_threshold_is_low_enough_for_a_short_name(self):
        payload = _places_search_payload("Мо", "ru")
        self.assertTrue(payload["searched"])
        self.assertTrue(payload["results"])

    def test_an_empty_query_is_not_a_search_either(self):
        self.assertFalse(_places_search_payload("   ", "ru")["searched"])

    def test_a_long_enough_query_with_no_answer_did_search(self):
        # Only THIS is what «no such place» may be said about.
        payload = _places_search_payload("Шмиргород", "ru")
        self.assertEqual(payload["results"], [])
        self.assertTrue(payload["searched"])

    def test_the_payload_echoes_the_trimmed_query(self):
        self.assertEqual(_places_search_payload("  Москва  ", "ru")["query"], "Москва")

    def test_the_threshold_is_short_enough_to_be_reachable(self):
        self.assertLessEqual(_PLACE_SEARCH_MIN_QUERY, 3)


class TestNothingLeavesTheMachine(PrefixSearchTestBase):
    def test_a_search_opens_no_socket(self):
        # The picker reads the bundled base and nothing else — the privacy invariant of
        # the whole geo half, and worth more than any convenience a lookup could add.
        def explode(*_args, **_kwargs):
            raise AssertionError("the place picker must not touch the network")

        with patch.object(socket, "socket", explode), \
                patch.object(socket, "create_connection", explode):
            self.assertIn(_MOSCOW, self.city_ids("Моск"))
            self.assertTrue(_places_search_payload("Рос", "ru")["results"])


class TestTheResolverIndex(unittest.TestCase):
    """The prefix lookups themselves, without the UI on top."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        data_dir = Path(self.tmp.name) / "geo"
        _write_fixture(data_dir)
        self.resolver = GeoResolver(data_dir=data_dir)

    def test_an_empty_prefix_is_not_a_query(self):
        self.assertEqual(self.resolver.city_ids_by_prefix("  ", "ru"), [])
        self.assertEqual(self.resolver.country_ccs_by_prefix("", "ru"), [])

    def test_a_city_is_listed_once_however_many_words_match(self):
        found = self.resolver.city_ids_by_prefix("н", "ru")
        self.assertEqual(found.count(_NIZHNY), 1)

    def test_the_index_is_built_once_and_reused(self):
        first = self.resolver.city_ids_by_prefix("моск", "ru")
        self.assertEqual(first, self.resolver.city_ids_by_prefix("моск", "ru"))

    def test_a_country_is_found_by_the_start_of_its_name(self):
        self.assertEqual(self.resolver.country_ccs_by_prefix("Норв", "ru"), ["NO"])

    def test_population_ranks_but_never_crashes(self):
        self.assertEqual(self.resolver.population_of(_MOSCOW), 10381222)
        self.assertEqual(self.resolver.population_of(_FILLER_BASE), 0)
        self.assertEqual(self.resolver.population_of(-1), 0)


class TestTheMessageStopsBlamingTheSpelling(unittest.TestCase):
    def test_both_answers_have_a_string_in_all_three_languages(self):
        for key in ("place_not_found", "place_keep_typing"):
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_not_found_message_no_longer_says_the_spelling_is_wrong(self):
        # It used to say «проверьте написание» to a half-typed word — sending the user
        # to fix a word that was never misspelled.
        self.assertNotIn("написание", ui._UI_STRINGS["place_not_found"]["ru"])
        self.assertNotIn("spelling", ui._UI_STRINGS["place_not_found"]["en"])


class TestThePageUsesTheDifference(unittest.TestCase):
    """The two empty answers have to reach the screen as two different things."""

    def setUp(self):
        self.js = (Path(__file__).resolve().parents[1] / "sorta" / "web" / "app"
                   / "app.js").read_text(encoding="utf-8")

    def test_the_picker_reads_the_searched_flag(self):
        self.assertIn("data.searched", self.js)

    def test_the_picker_can_ask_for_more_letters(self):
        self.assertIn("I18N.place_keep_typing", self.js)
        self.assertIn("place-hint", self.js)

    def test_the_error_is_no_longer_shown_just_because_something_was_typed(self):
        self.assertNotIn("picker.typed() ? I18N.place_not_found", self.js)


if __name__ == "__main__":
    unittest.main()
