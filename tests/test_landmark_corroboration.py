"""F75: a CLIP match is corroborated by its folder before it reaches the DB.

The measured problem (live collection, 2026-07-26): 17 files landed in Germany and 1
in the USA, in places the user has never been, while Prague (182 files, a real 2016
trip) was correct — and the score does not separate the two (median 0.980 against
0.991). The three rules under test here do: anti-classes drag non-photographs down,
agreement inside a folder discards the odd city out, and a country named in the path
confirms or refutes a match outright.

CLIP is mocked throughout; the geo base is a tiny fixture (the same shape as
tests/test_geodata.py) so the localized names are real data, not stubs.
"""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path, PurePath

import numpy as np

from sorta.config import Config, NamingConfig, _naming_from
from sorta.db import connect
from sorta.geodata import GeoResolver
from sorta.landmarks import (
    _ANTI_PROMPTS,
    _FolderHint,
    _folder_hint,
    _folder_tokens,
    _parent_dir,
    detect_landmarks,
    landmark_prompts,
    load_landmarks,
)

# The landmarks the fake classifier picks from; the geonameids match the fixture geo
# base below, exactly as the bundled list matches the bundled base.
LANDMARKS_YAML = """\
landmarks:
  - prompt: "a photo of the Charles Bridge in Prague"
    name: "Карлов мост"
    country: CZ
    city: Prague
    geonameid: 3067696
  - prompt: "a photo of the Brandenburg Gate in Berlin"
    name: "Бранденбургские ворота"
    country: DE
    city: Berlin
    geonameid: 2950159
  - prompt: "a photo of the Eiffel Tower in Paris"
    name: "Эйфелева башня"
    country: FR
    city: Paris
    geonameid: 2988507
  - prompt: "a photo of Red Square in Moscow"
    name: "Красная площадь"
    country: RU
    city: Moscow
    geonameid: 524901
"""
PRAGUE, BERLIN, PARIS, MOSCOW = 0, 1, 2, 3

# geonameid, lat, lon, fcode, cc, admin1, admin2, name_en, population
PLACES = [
    (3067696, 50.088, 14.421, "PPLC", "CZ", "52", "", "Prague", "1300000"),
    (2950159, 52.524, 13.411, "PPLC", "DE", "16", "", "Berlin", "3400000"),
    (2988507, 48.853, 2.349, "PPLC", "FR", "11", "", "Paris", "2100000"),
    (658225, 60.170, 24.938, "PPLC", "FI", "01", "", "Helsinki", "600000"),
    (524901, 55.752, 37.616, "PPLC", "RU", "48", "", "Moscow", "12000000"),
]
ADMIN1 = [("CZ", "52", 400, "Praha"), ("DE", "16", 401, "Berlin"),
          ("FR", "11", 402, "Ile-de-France"), ("FI", "01", 403, "Uusimaa")]
# Chad is in the fixture for one reason: its Russian name is three letters long, which
# is exactly the case the minimum component length must refuse to look up.
COUNTRIES = [("CZ", 600, "Czechia"), ("DE", 601, "Germany"), ("FR", 602, "France"),
             ("FI", 603, "Finland"), ("TD", 604, "Chad")]
NAMES = [
    (3067696, "ru", "Прага"), (3067696, "en", "Prague"),
    (2950159, "ru", "Берлин"), (2950159, "en", "Berlin"),
    (2988507, "ru", "Париж"), (2988507, "en", "Paris"), (2988507, "ja", "パリ"),
    (658225, "ru", "Хельсинки"), (658225, "en", "Helsinki"),
    (524901, "ru", "Москва"), (524901, "en", "Moscow"),
    (600, "ru", "Чехия"), (600, "en", "Czechia"),
    (601, "ru", "Германия"), (601, "en", "Germany"),
    (602, "ru", "Франция"), (602, "en", "France"), (602, "ja", "フランス"),
    (603, "ru", "Финляндия"), (603, "en", "Finland"),
    (604, "ru", "Чад"), (604, "en", "Chad"),
]


def _write_geo_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "places.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for row in PLACES:
            f.write("\t".join(str(v) for v in row) + "\n")
    with (data_dir / "names.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for gid, lang, name in NAMES:
            f.write(f"{gid}\t{lang}\t{name}\n")
    with (data_dir / "admin1.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for cc, a1, gid, name_en in ADMIN1:
            f.write(f"{cc}\t{a1}\t{gid}\t{name_en}\n")
    with (data_dir / "countries.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for cc, gid, name_en in COUNTRIES:
            f.write(f"{cc}\t{gid}\t{name_en}\n")


class PathClassifier:
    """Mock CLIP: the verdict is looked up by the full file path.

    A negative index addresses the distractor columns from the end (-1 is the last
    anti-class), which is how the "an anti-class took the mass" case is expressed.
    """

    def __init__(self, scores: dict[str, tuple[int, float]]) -> None:
        self.scores = scores

    def __call__(self, image_paths: list[str], prompts: list[str]) -> np.ndarray:
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, p in enumerate(image_paths):
            idx, prob = self.scores.get(p, (0, 0.0))
            out[i, idx] = prob
        return out


@dataclasses.dataclass(frozen=True)
class NamingWithGroupSettings(NamingConfig):
    """The settings object as it will look once the orchestrator adds the fields.

    landmarks.py reads them through getattr, so both shapes have to work: the plain
    NamingConfig (the measured defaults) and this one.
    """

    landmark_group_min: int = 3
    landmark_group_dominance: float = 0.5


class CorroborationCase(unittest.TestCase):
    """A DB with place-less photos + a fake classifier + the fixture geo base."""

    language = "ru"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        yaml_path = tmp / "landmarks.yaml"
        yaml_path.write_text(LANDMARKS_YAML, encoding="utf-8")
        _write_geo_fixture(tmp / "geo")
        self.resolver = GeoResolver(data_dir=tmp / "geo")
        self.cfg = Config(
            sources=[tmp], database=tmp / "test.db",
            naming=_naming_from({"landmarks_file": str(yaml_path),
                                 "landmark_threshold": 0.3}),
            language=self.language,
        )
        self.conn = connect(self.cfg.database)
        self.scores: dict[str, tuple[int, float]] = {}
        self.ids: dict[str, int] = {}
        self._n = 0

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def add(self, folder: str, landmark: int, prob: float = 0.9, n: int = 1) -> str:
        """Register n place-less photos in `folder` that CLIP matches to `landmark`."""
        path = ""
        for _ in range(n):
            self._n += 1
            path = f"{folder}/photo{self._n}.jpg"
            cur = self.conn.execute(
                """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
                   VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""", (path,))
            fid = cur.lastrowid
            self.conn.execute(
                """INSERT INTO places (file_id, country, region, city, confidence,
                       updated_at) VALUES (?, NULL, NULL, NULL, 'unknown', '2026-01-01')""",
                (fid,))
            self.ids[path] = int(fid or 0)
            self.scores[path] = (landmark, prob)
        self.conn.commit()
        return path

    def run_stage(self):
        return detect_landmarks(self.cfg, self.conn,
                                classifier=PathClassifier(self.scores),
                                resolver=self.resolver)

    def place_of(self, path: str) -> tuple:
        row = self.conn.execute(
            "SELECT country, city, city_geonameid, confidence FROM places WHERE file_id = ?",
            (self.ids[path],)).fetchone()
        return tuple(row)

    def cities(self) -> dict[str, int]:
        """city -> how many rows carry it, over the whole DB."""
        return {r["city"]: r["n"] for r in self.conn.execute(
            """SELECT city, COUNT(*) AS n FROM places WHERE confidence = 'visual'
               GROUP BY city""")}


class TestAntiPrompts(CorroborationCase):
    """A: the anti-classes only drain probability mass, they never win."""

    def test_prompts_carry_the_anti_classes_after_the_landmarks(self) -> None:
        landmarks = load_landmarks(Path(self.tmp.name) / "landmarks.yaml")
        prompts = landmark_prompts(landmarks)
        self.assertEqual(prompts[:len(landmarks)], [lm.prompt for lm in landmarks])
        for anti in _ANTI_PROMPTS:
            self.assertIn(anti, prompts)

    def test_anti_class_taking_the_mass_leaves_the_row_unknown(self) -> None:
        """The "video game -> New York" file: the anti-class wins, the landmark cannot.

        argmax still points at some landmark — it is taken over the landmark columns
        only — but with the mass drained its probability no longer clears the
        threshold, which is the entire mechanism the anti-classes buy.
        """
        wallpaper = self.add("/photos/DCIM", -1, prob=0.95)  # the last anti-class column
        stats = self.run_stage()
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(wallpaper)[3], "unknown")


class TestFolderAgreement(CorroborationCase):
    """C: the strongest signal — one card dump is one trip."""

    def test_minority_city_in_a_folder_is_dropped(self) -> None:
        self.add("/photos/DCIM/100D3300", PRAGUE, n=8)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN, n=2)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 8)
        self.assertEqual(stats.dropped_by_group, 2)
        self.assertEqual(self.cities(), {"Prague": 8})
        self.assertEqual(self.place_of(berlin), (None, None, None, "unknown"))

    def test_group_below_min_is_left_alone(self) -> None:
        self.add("/photos/DCIM/100D3300", PRAGUE, n=3)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN, n=1)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_group, 0)
        self.assertEqual(stats.matched, 4)
        self.assertEqual(self.place_of(berlin), ("DE", "Berlin", 2950159, "visual"))

    def test_no_dominant_city_leaves_everything(self) -> None:
        self.add("/photos/DCIM/100D3300", PRAGUE, n=5)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN, n=5)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_group, 0)
        self.assertEqual(stats.matched, 10)
        self.assertEqual(self.place_of(berlin), ("DE", "Berlin", 2950159, "visual"))

    def test_minority_is_dropped_not_reassigned(self) -> None:
        """We do not know these frames were shot in Prague — unknown is honest."""
        self.add("/photos/DCIM/100D3300", PRAGUE, n=9)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN, n=1)
        self.run_stage()
        country, city, gid, confidence = self.place_of(berlin)
        self.assertEqual(confidence, "unknown")
        self.assertIsNone(city)
        self.assertIsNone(country)
        self.assertIsNone(gid)

    def test_groups_are_per_directory(self) -> None:
        """A dominant city in one folder says nothing about the folder next to it."""
        self.add("/photos/DCIM/100D3300", PRAGUE, n=9)
        berlin = self.add("/photos/DCIM/101D3300", BERLIN, n=1)
        self.run_stage()
        self.assertEqual(self.place_of(berlin)[3], "visual")

    def test_thresholds_come_from_settings(self) -> None:
        """min_group/dominance are read off the settings object, not hardcoded."""
        base = self.cfg.naming
        self.cfg.naming = NamingWithGroupSettings(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
        self.add("/photos/DCIM/100D3300", PRAGUE, n=2)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN, n=1)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_group, 1)
        self.assertEqual(self.place_of(berlin)[3], "unknown")


class TestFolderName(CorroborationCase):
    """B: a country named in the path is a human statement about the place."""

    def test_country_in_path_refutes_a_foreign_match(self) -> None:
        berlin = self.add("/photos/Финляндия/Хельсинки", BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_folder_name, 1)
        self.assertEqual(stats.matched, 0)
        self.assertEqual(self.place_of(berlin), (None, None, None, "unknown"))

    def test_country_in_path_confirms_a_minority_match(self) -> None:
        """An explicit label outranks the statistics of the neighbouring files.

        Paris is one file against nine Pragues — a textbook minority — and it survives
        because the path says France. The nine Pragues go the other way for the same
        reason, which is the whole point of letting the folder name outrank the count.
        """
        self.add("/photos/Франция/Париж", PRAGUE, n=9)
        paris = self.add("/photos/Франция/Париж", PARIS, n=1)
        stats = self.run_stage()
        self.assertEqual(stats.confirmed_by_folder_name, 1)
        self.assertEqual(stats.dropped_by_folder_name, 9)
        self.assertEqual(stats.dropped_by_group, 0)  # the rescue happens first
        self.assertEqual(stats.matched, 1)
        self.assertEqual(self.place_of(paris), ("FR", "Paris", 2988507, "visual"))

    def test_compound_folder_name_is_split(self) -> None:
        """«чехия-австрия» must confirm Prague, not read as one unknown word."""
        prague = self.add("/photos/SORT/foto/чехия-австрия", PRAGUE)
        stats = self.run_stage()
        self.assertEqual(stats.confirmed_by_folder_name, 1)
        self.assertEqual(self.place_of(prague)[3], "visual")

    def test_city_alone_never_refutes(self) -> None:
        """A city name is far too easy to hit by accident to be allowed a veto."""
        berlin = self.add("/photos/Хельсинки", BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_folder_name, 0)
        self.assertEqual(self.place_of(berlin)[3], "visual")

    def test_city_alone_still_confirms(self) -> None:
        self.add("/photos/Париж", PRAGUE, n=9)
        paris = self.add("/photos/Париж", PARIS, n=1)
        stats = self.run_stage()
        self.assertEqual(stats.confirmed_by_folder_name, 1)
        self.assertEqual(self.place_of(paris)[3], "visual")

    def test_technical_components_recognize_nothing(self) -> None:
        """No blocklist needed: DCIM and 100D3300 are simply not places."""
        for folder in ("/photos/DCIM/100D3300", "/photos/Camera", "/photos/SORT/foto",
                       "/photos/foto-sort/DCIM/101D3300"):
            with self.subTest(folder):
                self.assertEqual(_folder_hint(str(PurePath(folder)), self.resolver, "ru"),
                                 _FolderHint())

    def test_technical_components_drop_nothing_end_to_end(self) -> None:
        berlin = self.add("/photos/foto-sort/DCIM/100D3300", BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_folder_name, 0)
        self.assertEqual(self.place_of(berlin)[3], "visual")

    def test_short_component_is_not_looked_up(self) -> None:
        """«Чад» is a country in the geo base and three letters long — too short to trust."""
        self.assertNotIn("Чад", _folder_tokens(str(PurePath("/photos/Чад"))))
        berlin = self.add("/photos/Чад", BERLIN)
        stats = self.run_stage()
        self.assertEqual(stats.dropped_by_folder_name, 0)
        self.assertEqual(self.place_of(berlin)[3], "visual")

    def test_missing_geo_data_disables_the_rule(self) -> None:
        """No bundled base -> the folder rule stops firing, the stage keeps working."""
        empty = GeoResolver(data_dir=Path(self.tmp.name) / "nowhere")
        berlin = self.add("/photos/Финляндия/Хельсинки", BERLIN)
        stats = detect_landmarks(self.cfg, self.conn,
                                 classifier=PathClassifier(self.scores), resolver=empty)
        self.assertEqual(stats.dropped_by_folder_name, 0)
        self.assertEqual(self.place_of(berlin)[3], "visual")


class TestFolderNameLocalization(unittest.TestCase):
    """The same country in three languages resolves through names.tsv, not a word list."""

    def test_country_name_recognized_in_the_config_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_geo_fixture(Path(tmp) / "geo")
            resolver = GeoResolver(data_dir=Path(tmp) / "geo")
            for lang, folder in (("en", "France"), ("ru", "Франция"), ("ja", "フランス")):
                with self.subTest(lang):
                    hint = _folder_hint(str(PurePath(f"/photos/{folder}")), resolver, lang)
                    self.assertEqual(hint.countries, frozenset({"FR"}))

    def test_folder_in_another_language_is_not_recognized(self) -> None:
        """Deliberate: an index per language, no cross-language guessing."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_geo_fixture(Path(tmp) / "geo")
            resolver = GeoResolver(data_dir=Path(tmp) / "geo")
            hint = _folder_hint(str(PurePath("/photos/Франция")), resolver, "en")
            self.assertEqual(hint.countries, frozenset())


class TestCounters(CorroborationCase):
    """Without these numbers the effect of the feature cannot be measured."""

    def test_every_bucket_is_counted_once(self) -> None:
        self.add("/photos/DCIM/100D3300", PRAGUE, n=6)
        self.add("/photos/DCIM/100D3300", BERLIN, n=2)       # minority -> group
        self.add("/photos/Германия", PRAGUE, n=1)            # refuted by the path
        self.add("/photos/Франция/Париж", PARIS, n=1)        # confirmed by the path
        stats = self.run_stage()
        self.assertEqual(stats.scanned, 10)
        self.assertEqual(stats.dropped_by_group, 2)
        self.assertEqual(stats.dropped_by_folder_name, 1)
        self.assertEqual(stats.confirmed_by_folder_name, 1)
        self.assertEqual(stats.matched, 7)
        self.assertEqual(stats.by_landmark, {"Карлов мост": 6, "Эйфелева башня": 1})
        self.assertEqual(sum((stats.matched, stats.dropped_by_group,
                              stats.dropped_by_folder_name)), 10)

    def test_below_threshold_files_are_not_counted_as_dropped(self) -> None:
        self.add("/photos/DCIM/100D3300", BERLIN, prob=0.1, n=6)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 0)
        self.assertEqual(stats.dropped_by_group, 0)
        self.assertEqual(stats.dropped_by_folder_name, 0)
        self.assertEqual(stats.confirmed_by_folder_name, 0)


class TestLiveScenario(CorroborationCase):
    """The scenario the user found by eye, reproduced at its measured proportions."""

    def test_one_card_dump_keeps_prague_and_loses_berlin(self) -> None:
        """The measured folders, at their measured counts (brief §C)."""
        self.add("/photos/foto-sort/DCIM/100D3300", PRAGUE, n=146)
        self.add("/photos/foto-sort/DCIM/100D3300", BERLIN, n=16)
        self.add("/photos/SORT/foto/всякое и хельсинки", MOSCOW, n=8)
        self.add("/photos/SORT/foto/всякое и хельсинки", BERLIN, n=1)
        self.add("/photos/SORT/foto/чехия-австрия", PRAGUE, n=6)
        stats = self.run_stage()
        self.assertEqual(self.cities(), {"Prague": 152, "Moscow": 8})  # no Berlin left
        self.assertEqual(stats.dropped_by_group, 17)
        self.assertEqual(stats.confirmed_by_folder_name, 6)  # «чехия-австрия»
        self.assertEqual(stats.matched, 160)

    def test_untouched_data_behaves_exactly_as_before(self) -> None:
        """Regression: where no rule applies, the stage is the old stage."""
        prague = self.add("/photos/DCIM/100D3300", PRAGUE)
        berlin = self.add("/photos/DCIM/100D3300", BERLIN)
        paris = self.add("/photos/Camera", PARIS)
        stats = self.run_stage()
        self.assertEqual(stats.matched, 3)
        self.assertEqual((stats.dropped_by_group, stats.dropped_by_folder_name,
                          stats.confirmed_by_folder_name), (0, 0, 0))
        self.assertEqual(self.place_of(prague), ("CZ", "Prague", 3067696, "visual"))
        self.assertEqual(self.place_of(berlin), ("DE", "Berlin", 2950159, "visual"))
        self.assertEqual(self.place_of(paris), ("FR", "Paris", 2988507, "visual"))


class TestFolderTokens(unittest.TestCase):
    def test_compound_names_are_split_on_non_letters(self) -> None:
        tokens = _folder_tokens(str(PurePath("/foto/чехия-австрия")))
        self.assertIn("чехия", tokens)
        self.assertIn("австрия", tokens)

    def test_digits_do_not_form_a_token(self) -> None:
        """A camera folder yields itself and nothing else — "D" is below the minimum."""
        self.assertEqual(_folder_tokens(str(PurePath("/DCIM/100D3300"))),
                         ["DCIM", "100D3300"])

    def test_the_file_name_itself_is_not_a_component(self) -> None:
        self.assertNotIn("Франция", _folder_tokens(
            _parent_dir(str(PurePath("/photos/Франция.jpg")))))

    def test_tokens_are_unique_and_include_the_whole_component(self) -> None:
        tokens = _folder_tokens(str(PurePath("/photos/Франция Париж/Франция")))
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertIn("Франция Париж", tokens)


if __name__ == "__main__":
    unittest.main()
