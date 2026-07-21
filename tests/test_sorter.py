"""F5: sorting by moving files — plan, apply, undo, --where, the CSV diagnosis.

All FS operations — on tmp_path only (self.root from TemporaryDirectory).
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, SortConfig
from sorta.db import connect
from sorta.geodata import GeoResolver
from sorta.hashing import file_hash
from sorta.sorter import (
    TransferError,
    _resolve_dst,
    _sanitize,
    _transfer,
    parse_where,
    plan_album,
    plan_and_sort,
    undo,
)


# F46: a tiny bundled geo fixture for a localized --where. 200/250 —
# "Moscow" homonyms in en (RU and US, like the real Moscow/Moscow, Idaho in
# GeoNames); 250 has its own ru name «Москоу» (a transliteration, as in the real
# bundled data) — it does not match the ru name of 200, so the en fallback of 250
# does not leak into the ru index under the key "moscow".
_GEO_MOSCOW_RU = (200, 55.7558, 37.6173, "PPLC", "RU", "48", "", "Moscow", "12000000")
_GEO_MOSCOW_US = (250, 46.7324, -117.0002, "PPLA2", "US", "16", "", "Moscow", "25000")
_GEO_PLACES = [_GEO_MOSCOW_RU, _GEO_MOSCOW_US]
_GEO_COUNTRIES = [("RU", 600, "Russia")]
_GEO_NAMES = [
    (200, "ru", "Москва"), (200, "en", "Moscow"),
    (250, "ru", "Москоу"), (250, "en", "Moscow"),
    (600, "ru", "Россия"), (600, "en", "Russia"),
]


def _write_geo_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "places.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for row in _GEO_PLACES:
            f.write("\t".join(str(v) for v in row) + "\n")
    with (data_dir / "countries.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for cc, gid, name_en in _GEO_COUNTRIES:
            f.write(f"{cc}\t{gid}\t{name_en}\n")
    with (data_dir / "names.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for gid, lang, name in _GEO_NAMES:
            f.write(f"{gid}\t{lang}\t{name}\n")


class SorterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "src"
        self.dest = self.root / "dest"
        self.src_dir.mkdir()
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                         raw={"language": "en"})
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    # --- fixtures ------------------------------------------------------

    def write_file(self, rel: str, content: bytes = b"data") -> Path:
        p = self.src_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def add_file(self, rel: str, content: bytes = b"data", taken_at: str | None = "2022-05-01T10:00:00",
                confidence: str | None = "high", country: str | None = None, city: str | None = None,
                place_confidence: str | None = None, junk_verdict: str | None = None,
                junk_source: str | None = None, gps_lat: float | None = None,
                gps_lon: float | None = None, city_geonameid: int | None = None,
                district_geonameid: int | None = None, country_name: str | None = None) -> int:
        self._n += 1
        p = self.write_file(rel, content)
        digest, algo = file_hash(p)
        path = str(p.resolve())
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   taken_at, taken_at_source, taken_at_confidence, gps_lat, gps_lon,
                   indexed_at)
               VALUES (?, ?, 0, 'jpg', 'photo', ?, ?, ?, 'exif', ?, ?, ?, '2026-01-01')""",
            (path, len(content), digest, algo, taken_at, confidence, gps_lat, gps_lon),
        )
        file_id = cur.lastrowid
        if country is not None or city is not None or place_confidence is not None:
            self.conn.execute(
                """INSERT INTO places (file_id, country, country_name, region, city,
                       confidence, city_geonameid, district_geonameid, updated_at)
                   VALUES (?, ?, ?, NULL, ?, ?, ?, ?, '2026-01-01')""",
                (file_id, country, country_name, city, place_confidence or "exact_gps",
                 city_geonameid, district_geonameid),
            )
        if junk_verdict is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, updated_at)
                   VALUES (?, ?, ?, '2026-01-01')""",
                (file_id, junk_verdict, junk_source or "heuristic"),
            )
        self.conn.commit()
        return file_id

    def add_person(self, file_id: int, label: str, bbox: str = "[0,0,10,10]") -> int:
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (?, NULL)", (label,))
        cluster_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (file_id, bbox, b"\x00" * 4, cluster_id))
        self.conn.commit()
        return cluster_id

    def add_event(self, file_id: int, name: str, started_at: str = "2022-05-01T09:00:00",
                  ended_at: str = "2022-05-01T20:00:00") -> int:
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name, name_is_manual, origin)
               VALUES (?, ?, ?, 0, 'auto')""",
            (started_at, ended_at, name))
        event_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)", (event_id, file_id))
        self.conn.commit()
        return event_id

    def path_of(self, file_id: int) -> str:
        return self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()["path"]

    def move_status(self, batch_id: int, file_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM moves WHERE batch_id = ? AND file_id = ?",
            (batch_id, file_id)).fetchone()
        return row["status"] if row else None

    def read_csv(self, csv_path: Path) -> list[dict]:
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh, delimiter=";"))


class TestParseWhere(unittest.TestCase):
    def test_string_equality_case_insensitive(self):
        cond, params = parse_where(["city=Paris"])
        self.assertIn("casefold(p.city) = casefold(?)", cond)
        self.assertEqual(params, ["Paris"])

    def test_year_operators(self):
        for op in ("=", "!=", ">=", "<=", ">", "<"):
            cond, params = parse_where([f"year{op}2020"])
            self.assertIn(op, cond)
            self.assertEqual(params, [2020])

    def test_and_combination(self):
        cond, params = parse_where(["country=France", "year>=2020"])
        self.assertIn(" AND ", cond)
        self.assertEqual(params, ["France", 2020])

    def test_person_field(self):
        cond, params = parse_where(["person=Мама"])
        self.assertIn("_person_files", cond)
        self.assertEqual(params, ["Мама"])

    def test_event_field(self):
        cond, params = parse_where(["event=Новый год"])
        self.assertIn("event_files", cond)
        self.assertEqual(params, ["Новый год"])

    def test_unknown_field_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_where(["planet=Mars"])
        self.assertIn("planet", str(ctx.exception))

    def test_bad_operator_on_string_field_raises(self):
        with self.assertRaises(ValueError):
            parse_where(["city>Paris"])

    def test_malformed_expr_raises(self):
        with self.assertRaises(ValueError):
            parse_where(["city"])

    def test_empty_is_always_true(self):
        cond, params = parse_where([])
        self.assertEqual((cond, params), ("1", []))


class TestParseWhereLocalized(unittest.TestCase):
    """F46: localized (by lang) country/city via GeoResolver, with a fallback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        geo_dir = Path(self.tmp.name) / "geo"
        _write_geo_fixture(geo_dir)
        self.resolver = GeoResolver(data_dir=geo_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_country_localized_resolves_to_cc(self):
        cond, params = parse_where(["country=Россия"], "ru", self.resolver)
        self.assertIn("casefold(p.country) = casefold(?)", cond)
        self.assertEqual(params, ["RU"])

    def test_country_canonical_still_works_with_resolver(self):
        cond, params = parse_where(["country=RU"], "ru", self.resolver)
        self.assertIn("casefold(p.country) = casefold(?)", cond)
        self.assertEqual(params, ["RU"])

    def test_country_unresolved_falls_back_to_raw_value(self):
        cond, params = parse_where(["country=Marsland"], "ru", self.resolver)
        self.assertIn("casefold(p.country) = casefold(?)", cond)
        self.assertEqual(params, ["Marsland"])

    def test_city_localized_resolves_to_geonameid_in(self):
        # a geonameid filter OR a string match (does not lose online-geocoded records
        # without a city_geonameid) — see parse_where.
        cond, params = parse_where(["city=Москва"], "ru", self.resolver)
        self.assertIn("p.city_geonameid IN (?)", cond)
        self.assertIn("casefold(p.city) = casefold(?)", cond)
        self.assertEqual(params, [200, "Москва"])

    def test_city_canonical_still_works_with_resolver(self):
        # the canonical asciiname "Moscow" under lang="ru" is not in the ru index (200
        # and 250 have their own distinct ru names) -> fallback to a p.city string match.
        cond, params = parse_where(["city=Moscow"], "ru", self.resolver)
        self.assertIn("casefold(p.city) = casefold(?)", cond)
        self.assertEqual(params, ["Moscow"])

    def test_city_unresolved_falls_back_to_string_match(self):
        cond, params = parse_where(["city=Atlantis"], "ru", self.resolver)
        self.assertIn("casefold(p.city) = casefold(?)", cond)
        self.assertEqual(params, ["Atlantis"])

    def test_city_homonyms_match_all_geonameids(self):
        cond, params = parse_where(["city=Moscow"], "en", self.resolver)
        self.assertIn("p.city_geonameid IN (?,?)", cond)
        self.assertEqual(sorted(params[:-1]), [200, 250])
        self.assertEqual(params[-1], "Moscow")

    def test_no_resolver_falls_back_like_before(self):
        # resolver=None (default) — the same behaviour as before F46.
        cond, params = parse_where(["country=Россия", "city=Москва"])
        self.assertIn("casefold(p.country) = casefold(?)", cond)
        self.assertIn("casefold(p.city) = casefold(?)", cond)
        self.assertEqual(params, ["Россия", "Москва"])


class TestSanitize(unittest.TestCase):
    def test_forbidden_chars_replaced(self):
        self.assertEqual(_sanitize('a<b>c:d"e/f\\g|h?i*j'), "a_b_c_d_e_f_g_h_i_j")

    def test_reserved_windows_name(self):
        self.assertEqual(_sanitize("CON"), "_CON")
        self.assertEqual(_sanitize("con.txt"), "_con.txt")

    def test_trailing_dot_space_stripped(self):
        self.assertEqual(_sanitize("Paris. "), "Paris")

    def test_empty_becomes_placeholder(self):
        self.assertEqual(_sanitize(""), "_")


class TestTransfer(SorterTestBase):
    def test_rename_fast_path(self):
        src = self.write_file("a.jpg", b"hello")
        dst = self.dest / "a.jpg"
        _transfer(src, dst)
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"hello")

    def test_cross_device_copy_verify_delete(self):
        src = self.write_file("b.jpg", b"cross-device")
        dst = self.dest / "b.jpg"
        with patch("sorta.sorter.os.rename", side_effect=OSError("cross-device")):
            _transfer(src, dst)
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"cross-device")

    def test_cross_device_hash_mismatch_cleans_up(self):
        src = self.write_file("c.jpg", b"content")
        dst = self.dest / "c.jpg"
        with patch("sorta.sorter.os.rename", side_effect=OSError("cross-device")), \
             patch("sorta.sorter.file_hash", return_value=("deadbeef", "blake3")):
            with self.assertRaises(TransferError):
                _transfer(src, dst, src_hash="cafebabe")
        self.assertFalse(dst.exists())
        self.assertTrue(src.exists())  # src untouched on a verification failure

    def test_never_overwrites_existing_dst(self):
        src = self.write_file("d.jpg", b"new")
        dst = self.dest / "d.jpg"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"existing")
        with self.assertRaises(TransferError):
            _transfer(src, dst)
        self.assertEqual(dst.read_bytes(), b"existing")
        self.assertTrue(src.exists())

    def test_resolve_dst_suffix_on_conflict(self):
        claimed: set[str] = set()
        self.dest.mkdir()
        (self.dest / "x.jpg").write_bytes(b"occupied")
        src = self.write_file("sub/x.jpg", b"incoming")
        dst, in_place = _resolve_dst(self.dest, src, claimed)
        self.assertEqual(dst.name, "x_1.jpg")
        self.assertFalse(in_place)


class TestPlanDryRun(SorterTestBase):
    def test_dry_run_touches_no_fs_and_writes_csv(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        before = self.path_of(fid)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertTrue(Path(before).exists())
        self.assertFalse(self.dest.exists())
        self.assertTrue(report.csv_path.exists())
        self.assertEqual(self.path_of(fid), before)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        rows = self.read_csv(report.csv_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], "France/Paris/2022/img1.jpg")
        self.assertEqual(rows[0]["reason"], "city")

    def test_online_country_name_used_over_iso_cc(self):
        # G6: an online places row stores the full country name from Nominatim — it
        # takes priority over localizing the ISO cc via the curated dict i18n.country.
        self.add_file("img1.jpg", country="FR", city="Paris",
                      country_name="Французская Республика")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel,
                         "Французская Республика/Paris/2022/img1.jpg")
        self.assertEqual(report.plan[0].reason, "city")

    def test_csv_columns_and_reasons_per_branch(self):
        self.add_file("ok.jpg", country="RU", city="Moskva")
        self.add_file("noplace.jpg")  # no places row
        self.add_file("junk.jpg", junk_verdict="screenshot", junk_source="heuristic")
        self.add_file("nodate.jpg", taken_at=None)
        self.add_file("lowconf.jpg", confidence="low")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        rows = {r["path"].split(os.sep)[-1].split("/")[-1]: r for r in self.read_csv(report.csv_path)}
        self.assertEqual(rows["ok.jpg"]["reason"], "city")
        self.assertEqual(rows["noplace.jpg"]["reason"], "no_place")
        self.assertEqual(rows["junk.jpg"]["reason"], "junk")
        self.assertEqual(rows["junk.jpg"]["target"], "_Unsorted/junk/screenshot/junk.jpg")
        self.assertEqual(rows["nodate.jpg"]["reason"], "low_date")
        self.assertEqual(rows["lowconf.jpg"]["reason"], "low_date")
        expected_cols = ["path", "taken_at", "taken_at_confidence", "country", "city",
                         "place_confidence", "persons", "event", "junk_verdict",
                         "junk_source", "target", "reason"]
        with open(report.csv_path, encoding="utf-8-sig") as fh:
            header = fh.readline().strip().split(";")
        self.assertEqual(header, expected_cols)

    def test_junk_overrides_any_mode(self):
        self.add_file("meme.jpg", junk_verdict="meme")
        for mode in ("city", "person", "event"):
            report = plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False)
            self.assertEqual(report.plan[0].reason, "junk")
            self.assertEqual(report.plan[0].target_rel, "_Unsorted/junk/meme/meme.jpg")

    def test_document_verdict_routes_to_documents_not_junk(self):
        # F15: a photographed document — a separate review folder _Documents/, not
        # junk, regardless of the sort mode.
        self.add_file("receipt.jpg", junk_verdict="document", junk_source="clip")
        for mode in ("city", "person", "event"):
            report = plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False)
            self.assertEqual(report.plan[0].reason, "document")
            self.assertEqual(report.plan[0].target_rel, "_Documents/receipt.jpg")

    def test_product_verdict_routes_to_products_not_junk(self):
        # F37-B (deep VLM tier): an item for sale — a separate review folder
        # _Products/, not junk, regardless of the mode.
        self.add_file("item.jpg", junk_verdict="product", junk_source="vlm")
        for mode in ("city", "person", "event"):
            report = plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False)
            self.assertEqual(report.plan[0].reason, "product")
            self.assertEqual(report.plan[0].target_rel, "_Products/item.jpg")

    def test_not_personal_routes_to_unsorted_not_personal(self):
        # F17: non-personal media marked at indexing (movie/series) goes to
        # _Unsorted/not_personal/, past the city/date/people layout.
        fid = self.add_file("Movie.2021.1080p.mkv", country="RU", city="Moskva")
        self.conn.execute("UPDATE files SET not_personal = 1 WHERE id = ?", (fid,))
        self.conn.commit()
        for mode in ("city", "person", "event"):
            report = plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False)
            self.assertEqual(report.plan[0].reason, "not_personal")
            self.assertEqual(report.plan[0].target_rel,
                             "_Unsorted/not_personal/Movie.2021.1080p.mkv")

    def test_empty_media_class_table_does_not_trigger_junk(self):
        self.add_file("plain.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "city")

    def test_person_mode_no_faces(self):
        self.add_file("alone.jpg")
        report = plan_and_sort(self.cfg, self.conn, "person", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "no_faces")
        self.assertEqual(report.plan[0].target_rel, "_Unsorted/no_faces/alone.jpg")

    def test_person_mode_single_person(self):
        fid = self.add_file("mama.jpg")
        self.add_person(fid, "Мама")
        report = plan_and_sort(self.cfg, self.conn, "person", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "person")
        self.assertEqual(report.plan[0].target_rel, "Мама/2022/mama.jpg")

    def test_person_mode_multi_primary_by_bbox_area(self):
        fid = self.add_file("two.jpg")
        self.add_person(fid, "Small", bbox="[0,0,5,5]")
        self.add_person(fid, "Big", bbox="[0,0,100,100]")
        report = plan_and_sort(self.cfg, self.conn, "person", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "person_primary")
        self.assertEqual(report.plan[0].target_rel, "Big/2022/two.jpg")
        self.assertEqual(report.plan[0].persons[0], "Big")

    def test_person_mode_multi_shared_folder_strategy(self):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "en"}, sort=SortConfig(multi_person="shared_folder"))
        fid = self.add_file("two.jpg")
        self.add_person(fid, "A")
        self.add_person(fid, "B")
        report = plan_and_sort(cfg, self.conn, "person", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "person_shared")
        self.assertEqual(report.plan[0].target_rel, "_Shared/2022/two.jpg")

    def test_event_mode(self):
        fid = self.add_file("party.jpg")
        self.add_event(fid, "День рождения")
        report = plan_and_sort(self.cfg, self.conn, "event", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "event")
        self.assertEqual(report.plan[0].target_rel, "2022/День рождения/party.jpg")

    def test_event_mode_no_event(self):
        # F30: a file without an event is laid out by date Year/month (not a flat
        # service folder), the reason stays no_event
        self.add_file("solo.jpg")  # taken_at defaults to 2022-05-01
        report = plan_and_sort(self.cfg, self.conn, "event", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "no_event")
        self.assertEqual(report.plan[0].target_rel, "2022/05/solo.jpg")

    def test_event_mode_low_confidence_file_uses_event_year(self):
        # F5.1: low-confidence must not send a file to low_date if it is included in
        # an event — the year is taken from events.started_at.
        fid = self.add_file("msg.jpg", confidence="low")
        self.add_event(fid, "Конференция", started_at="2021-01-01T09:00:00")
        report = plan_and_sort(self.cfg, self.conn, "event", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "event")
        self.assertEqual(report.plan[0].target_rel, "2021/Конференция/msg.jpg")

    def test_event_mode_no_date_file_uses_event_year(self):
        fid = self.add_file("undated.jpg", taken_at=None, confidence=None)
        self.add_event(fid, "Конференция", started_at="2021-01-01T09:00:00")
        report = plan_and_sort(self.cfg, self.conn, "event", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "event")
        self.assertEqual(report.plan[0].target_rel, "2021/Конференция/undated.jpg")

    def test_event_mode_uses_event_year_even_over_confident_file_date(self):
        # The year is always from the event, not the file — even when the file has a
        # reliable date from another year (a manual event can capture files broadly).
        fid = self.add_file("high_conf.jpg", taken_at="2022-05-01T10:00:00", confidence="high")
        self.add_event(fid, "Конференция", started_at="2021-01-01T09:00:00")
        report = plan_and_sort(self.cfg, self.conn, "event", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "2021/Конференция/high_conf.jpg")

    def test_person_mode_low_confidence_regression_low_date(self):
        # Regression: only event mode uses the event year; person still goes to
        # low_date with a low-confidence file date.
        fid = self.add_file("lowconf.jpg", confidence="low")
        self.add_person(fid, "Мама")
        report = plan_and_sort(self.cfg, self.conn, "person", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "low_date")
        self.assertEqual(report.plan[0].target_rel, "_Unsorted/low_date/lowconf.jpg")

    def test_city_mode_low_confidence_regression_low_date(self):
        self.add_file("lowconf.jpg", confidence="low", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "low_date")
        self.assertEqual(report.plan[0].target_rel, "_Unsorted/low_date/lowconf.jpg")

    def test_duplicates_excluded(self):
        canon = self.add_file("orig.jpg", content=b"same")
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, dup_of, indexed_at)
               VALUES (?, 4, 0, 'jpg', 'photo', ?, '2026-01-01')""",
            (str(self.write_file("dup.jpg", b"same").resolve()), canon))
        self.conn.commit()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(len(report.plan), 1)

    def test_where_filters_selection(self):
        self.add_file("paris.jpg", country="France", city="Paris")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               where=["city=paris"])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.plan[0].city, "Paris")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            plan_and_sort(self.cfg, self.conn, "planet", self.dest, apply=False)

    def test_name_conflict_gets_suffix(self):
        self.add_file("a/img.jpg", content=b"one", country="RU", city="Moskva")
        self.add_file("b/img.jpg", content=b"two", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        names = sorted(it.dst.name for it in report.plan)
        self.assertEqual(names, ["img.jpg", "img_1.jpg"])

    def test_cyrillic_and_spaces_in_paths(self):
        fid = self.add_file("папка с пробелами/фото № 1.jpg", country="Россия", city="Москва")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertIn("фото № 1.jpg", report.plan[0].target_rel)
        self.assertEqual(self.path_of(fid), str(report.plan[0].src))


class TestApplyAndJournal(SorterTestBase):
    def test_apply_moves_and_journals(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 1)
        new_path = self.dest / "France" / "Paris" / "2022" / "img1.jpg"
        self.assertTrue(new_path.exists())
        self.assertEqual(self.path_of(fid), str(new_path))
        self.assertEqual(self.move_status(report.batch_id, fid), "done")
        batch = self.conn.execute(
            "SELECT mode, dest_root, finished_at FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["mode"], "city")
        self.assertIsNotNone(batch["finished_at"])

    def test_journal_row_committed_before_transfer(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        seen_status_at_transfer_time = {}
        real_transfer = _transfer

        def spy_transfer(src, dst, src_hash=None, copy=False):
            row = self.conn.execute(
                "SELECT status FROM moves WHERE file_id = ?", (fid,)).fetchone()
            seen_status_at_transfer_time["status"] = row["status"] if row else None
            return real_transfer(src, dst, src_hash, copy=copy)

        with patch("sorta.sorter._transfer", side_effect=spy_transfer):
            plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(seen_status_at_transfer_time["status"], "planned")

    def test_junk_verdict_moves_under_unsorted(self):
        fid = self.add_file("meme.jpg", junk_verdict="meme")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertTrue((self.dest / "_Unsorted" / "junk" / "meme" / "meme.jpg").exists())
        self.assertNotIn("src", self.path_of(fid))

    def test_apply_does_not_overwrite_existing_file(self):
        self.add_file("a/img.jpg", content=b"one", country="RU", city="Moskva")
        self.add_file("b/img.jpg", content=b"two", country="RU", city="Moskva")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        target_dir = self.dest / "Russia" / "Moskva" / "2022"
        self.assertTrue((target_dir / "img.jpg").exists())
        self.assertTrue((target_dir / "img_1.jpg").exists())
        contents = {(target_dir / "img.jpg").read_bytes(), (target_dir / "img_1.jpg").read_bytes()}
        self.assertEqual(contents, {b"one", b"two"})

    def test_stale_hash_marks_failed_and_skips_move(self):
        fid = self.add_file("img1.jpg", content=b"indexed-content",
                            country="France", city="Paris")
        orig = Path(self.path_of(fid))
        orig.write_bytes(b"changed-after-indexing")  # the file changed after index

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 0)
        self.assertEqual(report.failed, 1)
        self.assertTrue(orig.exists())  # untouched
        self.assertEqual(self.move_status(report.batch_id, fid), "failed")
        self.assertEqual(self.path_of(fid), str(orig))  # files.path not updated

    def test_missing_source_marks_failed_and_continues(self):
        fid1 = self.add_file("gone.jpg", country="France", city="Paris")
        fid2 = self.add_file("ok.jpg", country="France", city="Paris")
        Path(self.path_of(fid1)).unlink()

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.moved, 1)
        self.assertEqual(self.move_status(report.batch_id, fid1), "failed")
        self.assertEqual(self.move_status(report.batch_id, fid2), "done")

    def test_progress_callback_invoked(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        self.add_file("img2.jpg", country="RU", city="Moskva")
        calls = []
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                     progress=lambda i, n: calls.append((i, n)))
        self.assertEqual(calls, [(1, 2), (2, 2)])

    def test_transfer_error_marks_failed_and_continues(self):
        fid1 = self.add_file("img1.jpg", country="France", city="Paris")
        fid2 = self.add_file("img2.jpg", country="RU", city="Moskva")
        with patch("sorta.sorter._transfer", side_effect=TransferError("boom")):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.failed, 2)
        self.assertEqual(report.moved, 0)
        self.assertEqual(self.move_status(report.batch_id, fid1), "failed")
        self.assertEqual(self.move_status(report.batch_id, fid2), "failed")
        # the files stayed at their original locations, files.path not updated
        self.assertTrue((self.src_dir / "img1.jpg").exists())
        self.assertTrue((self.src_dir / "img2.jpg").exists())


class TestUndoRoundTrip(SorterTestBase):
    def test_apply_then_undo_restores_bit_identical(self):
        fid1 = self.add_file("img1.jpg", content=b"aaa", country="France", city="Paris")
        fid2 = self.add_file("img2.jpg", content=b"bbb", country="RU", city="Moskva")
        orig1, orig2 = self.path_of(fid1), self.path_of(fid2)
        h1_before = file_hash(Path(orig1))[0]
        h2_before = file_hash(Path(orig2))[0]

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertFalse(Path(orig1).exists())

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 2)
        self.assertEqual(stats.missing, 0)
        self.assertTrue(Path(orig1).exists())
        self.assertTrue(Path(orig2).exists())
        self.assertEqual(file_hash(Path(orig1))[0], h1_before)
        self.assertEqual(file_hash(Path(orig2))[0], h2_before)
        self.assertEqual(self.path_of(fid1), orig1)
        self.assertEqual(self.path_of(fid2), orig2)
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM moves WHERE batch_id = ? AND status = 'undone'",
                (report.batch_id,)).fetchone()[0], 2)

    def test_undo_missing_dst_logs_and_continues(self):
        fid1 = self.add_file("img1.jpg", country="France", city="Paris")
        fid2 = self.add_file("img2.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        Path(self.path_of(fid1)).unlink()  # dst for fid1 vanished before undo

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.missing, 1)
        self.assertEqual(stats.undone, 1)
        self.assertEqual(self.move_status(report.batch_id, fid1), "done")
        self.assertEqual(self.move_status(report.batch_id, fid2), "undone")

    def test_undo_restores_with_suffix_if_src_occupied(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        orig = Path(self.path_of(fid))
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        orig.parent.mkdir(parents=True, exist_ok=True)
        orig.write_bytes(b"someone put a new file here")

        undo(self.conn, batch_id=report.batch_id)
        self.assertTrue(orig.exists())
        self.assertEqual(orig.read_bytes(), b"someone put a new file here")
        restored = orig.with_name("img1_1.jpg")
        self.assertTrue(restored.exists())

    def test_undo_default_picks_last_batch(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        stats = undo(self.conn)
        self.assertEqual(stats.undone, 1)
        self.assertEqual(self.path_of(fid), str(self.src_dir / "img1.jpg"))

    def test_undo_no_batches_raises(self):
        with self.assertRaises(ValueError):
            undo(self.conn)

    def test_undo_progress_callback_invoked(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        self.add_file("img2.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        calls = []
        undo(self.conn, batch_id=report.batch_id,
            progress=lambda i, n: calls.append((i, n)))
        self.assertEqual(calls, [(1, 2), (2, 2)])

    def test_undo_transfer_error_marks_failed_and_continues(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        with patch("sorta.sorter._transfer", side_effect=TransferError("boom")):
            stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.undone, 0)
        self.assertEqual(self.move_status(report.batch_id, fid), "done")  # not rolled back


class TestInterruptedApply(SorterTestBase):
    def test_interrupted_apply_resumes_and_undo_reverts_all(self):
        ids = []
        for i in range(5):
            ids.append(self.add_file(f"img{i}.jpg", content=f"data{i}".encode(),
                                     country="RU", city="Moskva"))
        originals = {fid: self.path_of(fid) for fid in ids}
        original_hashes = {fid: file_hash(Path(p))[0] for fid, p in originals.items()}

        from sorta import sorter as sorter_mod
        real_transfer = sorter_mod._transfer
        call_count = {"n": 0}

        def flaky(src, dst, src_hash=None, copy=False):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("simulated crash mid-apply")
            return real_transfer(src, dst, src_hash, copy=copy)

        with patch("sorta.sorter._transfer", side_effect=flaky):
            with self.assertRaises(RuntimeError):
                plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)

        moved_after_crash = sum(1 for fid in ids if Path(originals[fid]).exists() is False)
        self.assertEqual(moved_after_crash, 2)

        # a repeated apply finishes what was started (already-moved files are in_place)
        report2 = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertGreaterEqual(report2.moved, 3)
        for fid in ids:
            self.assertFalse(Path(originals[fid]).exists())
            self.assertTrue(Path(self.path_of(fid)).exists())

        # undo: may need several batches (a stack)
        for _ in range(10):
            remaining = self.conn.execute(
                "SELECT COUNT(*) FROM moves WHERE status = 'done'").fetchone()[0]
            if remaining == 0:
                break
            undo(self.conn)

        for fid in ids:
            restored = Path(originals[fid])
            self.assertTrue(restored.exists(), f"{restored} must be restored")
            self.assertEqual(file_hash(restored)[0], original_hashes[fid])
            self.assertEqual(self.path_of(fid), originals[fid])


class TestFullCycleAcceptance(SorterTestBase):
    def test_100_files_apply_undo_bit_identical(self):
        ids = []
        cities = [("RU", "Moskva"), ("FR", "Paris"), (None, None)]
        for i in range(100):
            country, city = cities[i % 3]
            content = os.urandom(37) + str(i).encode()
            rel = f"batch{i % 5}/photo_{i:03d}.jpg"
            fid = self.add_file(rel, content=content, country=country, city=city,
                                place_confidence=None if country else "unknown")
            ids.append(fid)
        originals = {fid: self.path_of(fid) for fid in ids}
        original_hashes = {fid: file_hash(Path(p))[0] for fid, p in originals.items()}

        dry = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(len(dry.plan), 100)
        self.assertFalse(self.dest.exists())

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 100)
        for fid in ids:
            self.assertFalse(Path(originals[fid]).exists())

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 100)
        self.assertEqual(stats.missing, 0)
        self.assertEqual(stats.failed, 0)

        for fid in ids:
            restored = Path(originals[fid])
            self.assertTrue(restored.exists())
            self.assertEqual(file_hash(restored)[0], original_hashes[fid])
            self.assertEqual(self.path_of(fid), originals[fid])

        leftover_files = [p for p in self.dest.rglob("*") if p.is_file()]
        self.assertEqual(leftover_files, [])


class TestReportDir(SorterTestBase):
    """F56: plan reports are written to report_output/ next to the DB (isolated from
    the DB/repo directory, gitignored), or to cfg.sort.report_dir if set."""

    def test_default_report_output_dir_next_to_db(self):
        self.add_file("paris.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        expected = self.cfg.database.resolve().parent / "report_output"
        self.assertEqual(report.csv_path.parent, expected)
        self.assertEqual(report.html_path.parent, expected)
        self.assertTrue(report.csv_path.exists())
        self.assertTrue(report.html_path.exists())

    def test_report_dir_config_override(self):
        self.add_file("paris.jpg", country="France", city="Paris")
        custom = self.root / "my_reports"
        self.cfg.sort.report_dir = str(custom)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.csv_path.parent, custom)
        self.assertTrue(report.csv_path.exists())
        self.assertTrue(custom.is_dir())


class TestHtmlReport(SorterTestBase):
    def test_html_created_next_to_csv_with_source_links_and_groups(self):
        fid1 = self.add_file("paris.jpg", country="France", city="Paris")
        fid2 = self.add_file("moskva.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.html_path.parent, report.csv_path.parent)
        self.assertEqual(report.html_path.suffix, ".html")
        self.assertTrue(report.html_path.exists())
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn(Path(self.path_of(fid1)).as_uri(), html)
        self.assertIn(Path(self.path_of(fid2)).as_uri(), html)
        # the category tree: Country/City/Year nodes, each with its own <summary>
        self.assertIn("<summary>France <small>(1)</small></summary>", html)
        self.assertIn("<summary>Paris <small>(1)</small></summary>", html)
        self.assertIn("<summary>Russia <small>(1)</small></summary>", html)
        self.assertIn("<summary>Moskva <small>(1)</small></summary>", html)
        self.assertIn("<summary>2022 <small>(1)</small></summary>", html)
        # nesting: a city is a child of a country, so it comes after it in the markup
        self.assertLess(html.index("<summary>France"), html.index("<summary>Paris"))
        self.assertLess(html.index("<summary>Russia"), html.index("<summary>Moskva"))
        self.assertIn("paris.jpg", html)
        self.assertIn("moskva.jpg", html)

    def test_tree_nesting_and_counts(self):
        self.add_file("a.jpg", taken_at="2023-01-01T00:00:00", country="RU", city="Moscow")
        self.add_file("b.jpg", taken_at="2023-02-01T00:00:00", country="RU", city="Moscow")
        self.add_file("c.jpg", taken_at="2024-01-01T00:00:00", country="RU", city="Moscow")
        self.add_file("d.jpg", taken_at="2024-01-01T00:00:00", country="TH", city="Bangkok")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        # counters — the sum of subtree files on each node
        self.assertIn("<summary>Russia <small>(3)</small></summary>", html)
        self.assertIn("<summary>Moscow <small>(3)</small></summary>", html)
        self.assertIn("<summary>2023 <small>(2)</small></summary>", html)
        self.assertIn("<summary>2024 <small>(1)</small></summary>", html)
        self.assertIn("<summary>Thailand <small>(1)</small></summary>", html)
        self.assertIn("<summary>Bangkok <small>(1)</small></summary>", html)
        # the top level is expanded, deeper — collapsed
        self.assertIn("<details open><summary>Russia ", html)
        self.assertIn("<details open><summary>Thailand ", html)
        self.assertNotIn("<details open><summary>Moscow", html)
        self.assertNotIn("<details open><summary>2023", html)
        self.assertNotIn("<details open><summary>2024", html)

    def test_mixed_depths_in_one_plan(self):
        # _Documents — 1 level, Russia/Moscow/2022 — 3 levels, in one plan
        self.add_file("doc.jpg", junk_verdict="document")
        self.add_file("y.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("<summary>_Documents <small>(1)</small></summary>", html)
        self.assertIn("<summary>Russia <small>(1)</small></summary>", html)
        self.assertIn("<summary>Moscow <small>(1)</small></summary>", html)
        self.assertIn("<summary>2022 <small>(1)</small></summary>", html)
        self.assertIn("doc.jpg", html)
        self.assertIn("y.jpg", html)

    def test_no_external_resources_inline_script_only(self):
        # F23: an inline script (expand/collapse buttons) — a deliberate relaxation of
        # F21 from "no JS" to "no EXTERNAL resources"; CDN/fetch/external src/href — none.
        self.add_file("z.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("<script>", html)
        self.assertNotIn("<script src", html.lower())
        self.assertNotIn("fetch(", html)
        for scheme in ("http://", "https://", "//"):
            self.assertNotIn(f'src="{scheme}', html)
            self.assertNotIn(f'href="{scheme}', html)

    def test_expand_collapse_all_buttons_present(self):
        # F23: buttons above the tree, acting on all <details> via querySelectorAll.
        self.add_file("a.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("Развернуть всё", html)
        self.assertIn("Свернуть всё", html)
        self.assertIn("querySelectorAll('details')", html)
        # the tree stays functional without the buttons — native <details open>
        self.assertIn("<details open>", html)

    def test_scroll_to_top_button_present(self):
        # The floating "Top" button: fixed position + an inline scrollTo handler.
        self.add_file("a.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn('id="sorta-top"', html)
        self.assertIn("window.scrollTo(", html)
        self.assertIn("position: fixed", html)

    def test_escapes_special_chars_in_name_and_diagnosis(self):
        # '<', '"', '|' etc. are forbidden in Windows filenames — we check name
        # escaping via a legal '&', and the diagnosis (city/person) via '<'/'"'.
        # F23: the inline <script> of the expand/collapse buttons is legitimate (see
        # test_no_external_resources_inline_script_only); here we check that
        # UNESCAPED user input ('Мама<script>') does not leak through as-is.
        fid = self.add_file("a&b.jpg", country='O<Fallon"', city="Town")
        self.add_person(fid, 'Мама<script>')
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertNotIn("Мама<script>", html)
        self.assertIn("Мама&lt;script&gt;", html)
        self.assertIn("a&amp;b.jpg", html)
        self.assertIn("O&lt;Fallon&quot;", html)

    def test_no_thumbnails_flag_creates_no_cache_dir(self):
        self.add_file("plain.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        thumbs_dir = report.html_path.parent / f"{report.html_path.stem}_thumbs"
        self.assertFalse(thumbs_dir.exists())

    def test_thumbnails_flag_creates_cache_dir_with_jpeg_previews(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, "JPEG")
        fid = self.add_file("photo.jpg", content=buf.getvalue(),
                            country="RU", city="Moskva")

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        thumbs_dir = report.html_path.parent / f"{report.html_path.stem}_thumbs"
        self.assertTrue(thumbs_dir.is_dir())
        thumb_file = thumbs_dir / f"{fid}.jpg"
        self.assertTrue(thumb_file.exists())
        with Image.open(thumb_file) as im:
            self.assertLessEqual(im.width, 200)
            self.assertLessEqual(im.height, 200)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn(f"{thumbs_dir.name}/{fid}.jpg", html)

    def test_thumbnails_parallel_partial_failure(self):
        # F18: parallel generation of several thumbnails; a corrupt file drops out
        # without a preview and without a crash, valid ones get an <img>.
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (48, 48), (5, 6, 7)).save(buf, "JPEG")
        good = buf.getvalue()
        ok_ids = [self.add_file(f"ok{i}.jpg", content=good,
                                country="RU", city="Moskva") for i in range(4)]
        bad_id = self.add_file("bad.jpg", content=b"\xff\xd8 not jpeg",
                               country="RU", city="Moskva")

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        thumbs_dir = report.html_path.parent / f"{report.html_path.stem}_thumbs"
        for fid in ok_ids:
            self.assertTrue((thumbs_dir / f"{fid}.jpg").exists())
        self.assertFalse((thumbs_dir / f"{bad_id}.jpg").exists())
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn(f"{thumbs_dir.name}/{ok_ids[0]}.jpg", html)
        self.assertNotIn(f"{thumbs_dir.name}/{bad_id}.jpg", html)

    def test_thumbnails_heic_via_pillow_fallback_or_no_crash(self):
        import io

        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif not installed")
        from PIL import Image
        pillow_heif.register_heif_opener()

        buf = io.BytesIO()
        Image.new("RGB", (48, 32), (10, 220, 30)).save(buf, "HEIF")
        fid = self.add_file("iphone.heic", content=buf.getvalue())

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        thumbs_dir = report.html_path.parent / f"{report.html_path.stem}_thumbs"
        thumb_file = thumbs_dir / f"{fid}.jpg"
        self.assertTrue(thumb_file.exists())

    def test_leaf_table_has_date_geo_category_columns(self):
        # F23: leaf columns — File/Date·time/Geo/People·Event/Category.
        fid = self.add_file("paris.jpg", taken_at="2022-05-01T10:00:00",
                            country="France", city="Paris",
                            place_confidence="exact_gps", gps_lat=48.8566, gps_lon=2.3522)
        self.add_person(fid, "Alice")
        self.add_event(fid, "Отпуск")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        # F24: leaf headers are clickable (data-sort-type + onclick), but the column
        # text is still visible inside <th>.
        self.assertIn('data-sort-type="date"', html)
        self.assertIn("Дата/время", html)
        self.assertIn("Гео", html)
        self.assertIn("Люди/Событие", html)
        self.assertIn("Категория", html)
        self.assertIn("2022-05-01", html)
        self.assertIn("France/Paris", html)
        self.assertIn("48.8566, 2.3522", html)
        self.assertIn("Alice", html)
        self.assertIn("Отпуск", html)
        self.assertIn(">city<", html)

    def test_leaf_table_geo_cell_empty_without_place_or_coords(self):
        self.add_file("nowhere.jpg", taken_at="2022-05-01T10:00:00")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("nowhere.jpg", html)  # the plan builds, without crashing
        self.assertNotIn("None", html)  # gps/place None does not leak into the markup

    def test_leaf_table_marks_low_confidence_date(self):
        self.add_file("mtime_only.jpg", taken_at="2019-01-01T00:00:00",
                      confidence="low", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("низкая точность", html)

    def test_leaf_table_category_shows_junk_and_document_verdict(self):
        self.add_file("meme.jpg", junk_verdict="meme")
        self.add_file("doc.jpg", junk_verdict="document")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("junk · meme", html)
        self.assertIn("document · document", html)

    def test_thumbnails_undecodable_file_no_crash_no_preview(self):
        fid = self.add_file("broken.jpg", content=b"not-an-image")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        thumbs_dir = report.html_path.parent / f"{report.html_path.stem}_thumbs"
        thumb_file = thumbs_dir / f"{fid}.jpg"
        self.assertFalse(thumb_file.exists())
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("broken.jpg", html)  # the row is there, just without a preview

    def test_leaf_headers_are_clickable_sort_controls(self):
        # F24: leaf headers are clickable — a sort attribute + a handler on <th>.
        self.add_file("a.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn('data-sort-type="text"', html)
        self.assertIn('data-sort-type="date"', html)
        self.assertIn('onclick="sortaSort(this)"', html)

    def test_date_cell_carries_iso_data_sort(self):
        # F24: the Date/time cell carries data-sort with the full ISO taken_at
        # (ISO lexicographic = chronological order).
        self.add_file("a.jpg", taken_at="2022-05-01T10:30:00", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn('data-sort="2022-05-01T10:30:00"', html)

    def test_date_cell_without_taken_at_has_empty_data_sort(self):
        # F24: without a date — a valid empty key (sortaSort pushes it to the end).
        self.add_file("nodate.jpg", taken_at=None, country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn('<td data-sort="">без даты</td>', html)

    def test_sort_operates_on_own_table_not_whole_document(self):
        # F24: sortaSort finds the table from the clicked <th> (closest), NOT via a
        # global document.querySelectorAll('tbody tr') over the whole report —
        # so sorting one leaf does not touch the other tree tables.
        self.add_file("a.jpg", taken_at="2023-01-01T00:00:00", country="RU", city="Moscow")
        self.add_file("b.jpg", taken_at="2024-01-01T00:00:00", country="TH", city="Bangkok")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("function sortaSort(th)", html)
        self.assertIn("th.closest('table')", html)
        self.assertNotIn("document.querySelectorAll('tbody tr')", html)
        self.assertNotIn('document.querySelectorAll("tbody tr")', html)

    def test_sort_indicator_span_present_and_empty_cells_pushed_last_in_js(self):
        # F24: the direction indicator (▲/▼) is updated in a span in the header;
        # a sort key with an empty value always returns "to the end" regardless of
        # direction (asc/desc) — we check this branch is present in the generated JS.
        self.add_file("a.jpg", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn('class="sorta-sort-ind"', html)
        self.assertIn("ea || eb", html)  # empty key -> to the end, regardless of dir

    def test_sort_columns_do_not_break_tree_buttons_thumbnails(self):
        # F24: does not break the tree (F21), the expand/collapse buttons (F23), thumbnails (F18).
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (48, 48), (1, 2, 3)).save(buf, "JPEG")
        self.add_file("p.jpg", content=buf.getvalue(),
                      taken_at="2022-05-01T10:00:00", country="RU", city="Moscow")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, thumbnails=True)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("<details open>", html)
        self.assertIn("Развернуть всё", html)
        self.assertIn("Свернуть всё", html)
        self.assertIn("<img src=", html)
        for scheme in ("http://", "https://", "//"):
            self.assertNotIn(f'src="{scheme}', html)
            self.assertNotIn(f'href="{scheme}', html)


class TestI18nWiring(SorterTestBase):
    """F27: the language from cfg.raw['language'] localizes the service folders and
    country in the layout; the CSV reason codes and city (transliterated) do not change."""

    def test_ru_language_localizes_service_folders_and_country(self):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        self.add_file("ok.jpg", country="RU", city="Moskva")
        self.add_file("noplace.jpg")
        self.add_file("junk.jpg", junk_verdict="screenshot")
        self.add_file("doc.jpg", junk_verdict="document")
        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        by_name = {it.src.name: it for it in report.plan}
        self.assertEqual(by_name["ok.jpg"].target_rel, "Россия/Moskva/2022/ok.jpg")
        self.assertEqual(by_name["ok.jpg"].reason, "city")
        self.assertEqual(by_name["noplace.jpg"].target_rel,
                         "_Неразобрано/без_места/noplace.jpg")
        self.assertEqual(by_name["noplace.jpg"].reason, "no_place")
        self.assertEqual(by_name["junk.jpg"].target_rel,
                         "_Неразобрано/мусор/screenshot/junk.jpg")
        self.assertEqual(by_name["junk.jpg"].reason, "junk")
        self.assertEqual(by_name["doc.jpg"].target_rel, "_Документы/doc.jpg")
        self.assertEqual(by_name["doc.jpg"].reason, "document")
        # the CSV reason codes — stable English strings, localization does not touch them
        rows = {r["path"].split(os.sep)[-1].split("/")[-1]: r["reason"]
               for r in self.read_csv(report.csv_path)}
        self.assertEqual(rows["ok.jpg"], "city")
        self.assertEqual(rows["noplace.jpg"], "no_place")
        self.assertEqual(rows["junk.jpg"], "junk")
        self.assertEqual(rows["doc.jpg"], "document")

    def test_en_language_country_localized_even_though_shared_ascii(self):
        # en also localizes the country (RU -> Russia), not only the service folders.
        self.add_file("ok.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Russia/Moskva/2022/ok.jpg")

    def test_unknown_country_code_falls_back_to_itself(self):
        self.add_file("xx.jpg", country="XX", city="Nowhere")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "XX/Nowhere/2022/xx.jpg")

    def test_unknown_language_in_config_falls_back_to_en(self):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "xx"})
        self.add_file("noplace.jpg")
        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "_Unsorted/no_place/noplace.jpg")

    def test_missing_language_key_defaults_to_en(self):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db")
        self.add_file("noplace.jpg")
        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "_Unsorted/no_place/noplace.jpg")


# fixture geonameids (see tests/test_geo.py) — arbitrary, just stable
_GID_SPB = 498817
_GID_MOSCOW = 524901
_GID_SOCHI = 542420
_GID_AKADEM = 1487117  # a district near SPb
_GID_WICHIT = 1609350  # F49: a foreign district without a localized (ru/ja) name — transliteration only

_CITY_NAMES = {
    _GID_SPB: {"ru": "Санкт-Петербург", "en": "Saint Petersburg", "ja": "サンクトペテルブルク"},
    _GID_MOSCOW: {"ru": "Москва", "en": "Moscow", "ja": "モスクワ"},
    _GID_SOCHI: {"ru": "Сочи", "en": "Sochi", "ja": "ソチ"},
}
_DISTRICT_NAMES = {
    _GID_AKADEM: {"ru": "Академическое", "en": "Akademicheskoe", "ja": "アカデミーチェスコエ"},
    _GID_WICHIT: {"en": "Wichit"},  # F49: en only -> transliteration, no ru/ja
}


class _FakeGeoResolver:
    """A mini resolver instead of geodata.GeoResolver — without the real bundled data
    (the same trick as in tests/test_geo.py)."""

    def name(self, geonameid, lang):
        names = _CITY_NAMES.get(geonameid) or _DISTRICT_NAMES.get(geonameid)
        if names is None:
            return str(geonameid)  # G1: fallback for an unknown geonameid — the id as a string
        return names.get(lang, names["en"])

    def has_localized_name(self, geonameid, lang):
        # F49: like geodata.GeoResolver.has_localized_name — True only if the geonameid
        # is in the dict SPECIFICALLY in lang (without an en fallback).
        names = _CITY_NAMES.get(geonameid) or _DISTRICT_NAMES.get(geonameid)
        return names is not None and lang in names

    # F46: reverse lookups for a localized --where (country/city).
    _COUNTRY_CC_BY_NAME = {"россия": "RU", "russia": "RU"}
    _CITY_IDS_BY_NAME = {
        "москва": [_GID_MOSCOW], "moscow": [_GID_MOSCOW],
        "санкт-петербург": [_GID_SPB], "saint petersburg": [_GID_SPB],
    }

    def country_cc_by_name(self, name, lang):
        return self._COUNTRY_CC_BY_NAME.get(name.strip().casefold())

    def city_ids_by_name(self, name, lang):
        return list(self._CITY_IDS_BY_NAME.get(name.strip().casefold(), []))


class TestCityGeoLocalization(SorterTestBase):
    """G3: city mode — Country/City/Year/District with localized names via
    geodata.GeoResolver (city_geonameid/district_geonameid from places, G2)."""

    def test_city_with_district_ru(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel,
                         "Россия/Санкт-Петербург/2022/Академическое/spb.jpg")
        self.assertEqual(report.plan[0].reason, "city")

    def test_city_without_district_stays_three_levels(self):
        self.add_file("sochi.jpg", country="RU", city="Sochi", city_geonameid=_GID_SOCHI)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Россия/Сочи/2022/sochi.jpg")

    def test_online_district_name_used_when_no_geonameid(self):
        # G2b online: city/district — names from Nominatim (geonameids NULL);
        # city is taken from the text row["city"], the district — from p.district_name.
        fid = self.add_file("ist.jpg", country="TR", city="Стамбул")
        self.conn.execute("UPDATE places SET district_name = ? WHERE file_id = ?",
                          ("Бешикташ", fid))
        self.conn.commit()
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Турция/Стамбул/2022/Бешикташ/ist.jpg")

    def test_landmark_without_geonameid_uses_place_text(self):
        # G1/G2: a landmark without a gps->geonameid resolve (the visual/landmark
        # branch) — city_geonameid NULL, the text row["city"] is used as-is.
        self.add_file("tower.jpg", country="FR", city="Eiffel Tower")
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Франция/Eiffel Tower/2022/tower.jpg")

    def test_en_language_localizes_city_and_district(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel,
                         "Russia/Saint Petersburg/2022/Akademicheskoe/spb.jpg")

    def test_ja_language_localizes_city_and_district(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ja"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel,
                         "ロシア/サンクトペテルブルク/2022/アカデミーチェスコエ/spb.jpg")

    def test_unknown_geonameid_falls_back_to_id_string(self):
        # G1: a geonameid absent from the resolver (e.g. a stale bundled
        # snapshot) does not break the plan — the resolver degrades to the id as a string.
        self.add_file("mystery.jpg", country="RU", city="Mystery Town",
                      city_geonameid=999999)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Россия/999999/2022/mystery.jpg")

    def test_geo_resolver_constructed_once_per_plan(self):
        # G3: the resolver is created ONCE in plan_and_sort, not per file.
        self.add_file("a.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        self.add_file("b.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        with patch("sorta.sorter.GeoResolver") as mock_cls:
            mock_cls.return_value = _FakeGeoResolver()
            plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        mock_cls.assert_called_once()


class TestCityDropUnlocalizedDistrict(SorterTestBase):
    """F49 (#4-B): a foreign transliterated district (no ru name) is dropped from the
    city path (sort.drop_unlocalized_district, default True); RU and localized foreign
    districts, as well as an online district_name (Nominatim), are unaffected."""

    def test_unlocalized_district_dropped_by_default(self):
        self.add_file("phuket.jpg", country="RU", city="Sochi",
                      city_geonameid=_GID_SOCHI, district_geonameid=_GID_WICHIT)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Россия/Сочи/2022/phuket.jpg")

    def test_unlocalized_district_included_when_flag_false(self):
        self.add_file("phuket.jpg", country="RU", city="Sochi",
                      city_geonameid=_GID_SOCHI, district_geonameid=_GID_WICHIT)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        cfg.sort.drop_unlocalized_district = False
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Россия/Сочи/2022/Wichit/phuket.jpg")

    def test_localized_district_kept_regardless_of_flag(self):
        # a RU district (has_localized_name=True) stays in the path even when the flag is True.
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel,
                         "Россия/Санкт-Петербург/2022/Академическое/spb.jpg")

    def test_online_district_name_included_even_when_flag_false(self):
        # G2b online (Nominatim) is already localized — the flag does not affect it.
        fid = self.add_file("ist.jpg", country="TR", city="Стамбул")
        self.conn.execute("UPDATE places SET district_name = ? WHERE file_id = ?",
                          ("Бешикташ", fid))
        self.conn.commit()
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        cfg.sort.drop_unlocalized_district = False
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].target_rel, "Турция/Стамбул/2022/Бешикташ/ist.jpg")


class TestLocalizedWhereFilter(SorterTestBase):
    """F46: --where country=Россия/city=Москва (lang=ru) selects the same files
    as the canonical --where country=RU/city=Moscow."""

    def _report_where(self, where):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        with patch("sorta.sorter.GeoResolver", return_value=_FakeGeoResolver()):
            return plan_and_sort(cfg, self.conn, "city", self.dest, apply=False,
                                 where=where)

    def test_localized_city_matches_canonical(self):
        moscow = self.add_file("moscow.jpg", country="RU", city="Moscow",
                               city_geonameid=_GID_MOSCOW)
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        localized = self._report_where(["city=Москва"])
        canonical = self._report_where(["city=Moscow"])
        self.assertEqual({it.file_id for it in localized.plan}, {moscow})
        self.assertEqual({it.file_id for it in canonical.plan}, {moscow})

    def test_localized_country_matches_canonical(self):
        moscow = self.add_file("moscow.jpg", country="RU", city="Moscow",
                               city_geonameid=_GID_MOSCOW)
        self.add_file("paris.jpg", country="FR", city="Paris")
        localized = self._report_where(["country=Россия"])
        canonical = self._report_where(["country=RU"])
        self.assertEqual({it.file_id for it in localized.plan}, {moscow})
        self.assertEqual({it.file_id for it in canonical.plan}, {moscow})

    def test_unknown_localized_city_falls_back_to_string_match(self):
        # an unknown localized city name does NOT silently yield an empty result —
        # it falls back to a string match on p.city (like canonical input).
        self.add_file("moscow.jpg", country="RU", city="Moscow",
                      city_geonameid=_GID_MOSCOW)
        fid = self.add_file("atlantis.jpg", country="RU", city="Atlantis")
        report = self._report_where(["city=Atlantis"])
        self.assertEqual({it.file_id for it in report.plan}, {fid})


class TestInPlaceSort(SorterTestBase):
    """F28: dest=None — in-place layout into the root of the single source."""

    def test_dest_none_single_source_plans_within_source(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", None, apply=False)
        self.assertEqual(report.dest, self.src_dir.resolve())
        self.assertTrue(report.in_place)
        dst = report.plan[0].dst
        self.assertTrue(dst.is_relative_to(self.src_dir.resolve()))
        self.assertEqual(report.plan[0].target_rel, "France/Paris/2022/img1.jpg")

    def test_explicit_dest_is_not_in_place(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertFalse(report.in_place)
        self.assertEqual(report.dest, self.dest.resolve())

    def test_zero_sources_raises(self):
        cfg = Config(sources=[], database=self.root / "test.db", raw={"language": "en"})
        with self.assertRaises(ValueError) as ctx:
            plan_and_sort(cfg, self.conn, "city", None, apply=False)
        self.assertIn("единственного источника", str(ctx.exception))

    def test_two_sources_raises(self):
        other = self.root / "other_src"
        other.mkdir()
        cfg = Config(sources=[self.src_dir, other], database=self.root / "test.db",
                    raw={"language": "en"})
        with self.assertRaises(ValueError):
            plan_and_sort(cfg, self.conn, "city", None, apply=False)

    def test_apply_prints_in_place_warning(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan_and_sort(self.cfg, self.conn, "city", None, apply=True)
        self.assertIn("ИСХОДНОЕ дерево", buf.getvalue())

    def test_dry_run_does_not_print_in_place_warning(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan_and_sort(self.cfg, self.conn, "city", None, apply=False)
        self.assertNotIn("ИСХОДНОЕ дерево", buf.getvalue())

    def test_second_apply_is_idempotent_moves_nothing(self):
        ids = [self.add_file(f"img{i}.jpg", content=f"data{i}".encode(),
                             country="France", city="Paris") for i in range(3)]
        report1 = plan_and_sort(self.cfg, self.conn, "city", None, apply=True)
        self.assertEqual(report1.moved, 3)
        self.assertEqual(report1.failed, 0)
        for fid in ids:
            self.assertTrue(Path(self.path_of(fid)).is_relative_to(self.src_dir.resolve()))

        report2 = plan_and_sort(self.cfg, self.conn, "city", None, apply=True)
        self.assertEqual(report2.moved, 0)
        self.assertEqual(report2.skipped_in_place, 3)
        for fid in ids:
            self.assertTrue(Path(self.path_of(fid)).exists())

    def test_copy_true_dest_none_copies_within_source_originals_intact(self):
        fid = self.add_file("img1.jpg", content=b"hello", country="France", city="Paris")
        before = self.path_of(fid)
        report = plan_and_sort(self.cfg, self.conn, "city", None, apply=True, copy=True)
        self.assertEqual(report.moved, 1)
        self.assertTrue(Path(before).exists())  # original untouched
        self.assertEqual(self.path_of(fid), before)  # files.path is not updated in copy mode
        copied = self.src_dir / "France" / "Paris" / "2022" / "img1.jpg"
        self.assertTrue(copied.exists())
        self.assertEqual(copied.read_bytes(), b"hello")


# --- F34: album engine -------------------------------------------------------

class TestTransferLink(SorterTestBase):
    def test_link_creates_hardlink_original_intact(self):
        src = self.write_file("a.jpg", b"hello")
        dst = self.dest / "a.jpg"
        _transfer(src, dst, link=True)
        self.assertTrue(dst.exists())
        self.assertTrue(src.exists())
        self.assertEqual(dst.read_bytes(), b"hello")
        self.assertGreaterEqual(dst.stat().st_nlink, 2)
        self.assertEqual(dst.stat().st_ino, src.stat().st_ino)

    def test_link_fallback_to_copy_on_oserror(self):
        src = self.write_file("b.jpg", b"world")
        dst = self.dest / "b.jpg"
        with patch("sorta.sorter.os.link", side_effect=OSError("cross-device")):
            _transfer(src, dst, link=True)
        self.assertTrue(dst.exists())
        self.assertTrue(src.exists())
        self.assertEqual(dst.read_bytes(), b"world")


class TestPlanAlbumSelection(SorterTestBase):
    def add_merged_face(self, file_id: int, root_cluster_id: int,
                        bbox: str = "[0,0,10,10]") -> int:
        """A face on file_id in a NEW cluster merged (merged_into) into root_cluster_id (F31)."""
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (NULL, ?)", (root_cluster_id,))
        cluster_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (file_id, bbox, b"\x00" * 4, cluster_id))
        self.conn.commit()
        return cluster_id

    def test_person_album_excludes_other_people(self):
        fid1 = self.add_file("a.jpg")
        self.add_person(fid1, "Мама")
        fid2 = self.add_file("b.jpg")
        self.add_person(fid2, "Папа")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest, apply=False)
        self.assertEqual([it.file_id for it in report.plan], [fid1])
        self.assertEqual(report.album_name, "Мама")

    def test_person_album_includes_merged_cluster_files(self):
        fid1 = self.add_file("a.jpg")
        root = self.add_person(fid1, "Мама")
        fid2 = self.add_file("b.jpg")
        self.add_merged_face(fid2, root)
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest, apply=False)
        self.assertEqual({it.file_id for it in report.plan}, {fid1, fid2})

    def test_person_album_where_narrows(self):
        fid1 = self.add_file("a.jpg", country="France", city="Paris")
        self.add_person(fid1, "Мама")
        fid2 = self.add_file("b.jpg", country="Russia", city="Moskva")
        self.add_person(fid2, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            where=["city=Paris"], apply=False)
        self.assertEqual([it.file_id for it in report.plan], [fid1])

    def test_person_album_excludes_dup_files(self):
        fid1 = self.add_file("a.jpg")
        fid2 = self.add_file("b.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (fid1, fid2))
        self.conn.commit()
        self.add_person(fid1, "Мама")
        self.add_person(fid2, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest, apply=False)
        self.assertEqual([it.file_id for it in report.plan], [fid1])

    def test_event_album_by_name(self):
        fid = self.add_file("a.jpg")
        self.add_event(fid, "Свадьба")
        report = plan_album(self.cfg, self.conn, "event", "Свадьба", self.dest, apply=False)
        self.assertEqual(report.album_name, "Свадьба")
        self.assertEqual([it.file_id for it in report.plan], [fid])

    def test_event_album_by_id(self):
        fid = self.add_file("a.jpg")
        eid = self.add_event(fid, "NYE")
        report = plan_album(self.cfg, self.conn, "event", str(eid), self.dest, apply=False)
        self.assertEqual(report.album_name, "NYE")
        self.assertEqual([it.file_id for it in report.plan], [fid])

    def test_event_album_name_override(self):
        fid = self.add_file("a.jpg")
        self.add_event(fid, "IEEE Conference on Whatever 2022")
        report = plan_album(self.cfg, self.conn, "event", "IEEE Conference on Whatever 2022",
                            self.dest, album_name="IEEE", apply=False)
        self.assertEqual(report.album_name, "IEEE")

    def test_empty_person_selection_no_crash(self):
        report = plan_album(self.cfg, self.conn, "person", "Ghost", self.dest, apply=True)
        self.assertEqual(report.plan, [])
        self.assertIsNone(report.batch_id)
        self.assertFalse(self.dest.exists())

    def test_empty_event_selection_no_crash(self):
        report = plan_album(self.cfg, self.conn, "event", "NoSuchEvent", self.dest, apply=True)
        self.assertEqual(report.plan, [])
        self.assertIsNone(report.batch_id)

    def test_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            plan_album(self.cfg, self.conn, "city", "x", self.dest)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            plan_album(self.cfg, self.conn, "person", "Мама", self.dest, mode="teleport")


class TestPlanAlbumApply(SorterTestBase):
    def test_dry_run_touches_no_fs_or_journal(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="link", apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_apply_link_flat_layout_and_undo(self):
        fid = self.add_file("sub/deep/a.jpg", content=b"hello")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / "Мама" / "a.jpg"
        self.assertTrue(dst.exists())
        src = Path(self.path_of(fid))
        self.assertTrue(src.exists())  # link/copy: the original stays in the canon
        self.assertEqual(src, Path(self.src_dir / "sub" / "deep" / "a.jpg").resolve())
        batch = self.conn.execute(
            "SELECT operation FROM move_batches WHERE id = ?", (report.batch_id,)).fetchone()
        self.assertEqual(batch["operation"], "link")

        stats = undo(self.conn, report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertFalse(dst.exists())
        self.assertTrue(src.exists())

    def test_apply_link_fallback_to_copy(self):
        fid = self.add_file("a.jpg", content=b"hello")
        self.add_person(fid, "Мама")
        with patch("sorta.sorter.os.link", side_effect=OSError("cross-device")):
            report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                                mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / "Мама" / "a.jpg"
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_bytes(), b"hello")

    def test_apply_copy_original_intact(self):
        fid = self.add_file("a.jpg", content=b"hello")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        src = Path(self.path_of(fid))
        self.assertTrue(src.exists())
        self.assertTrue((self.dest / "Мама" / "a.jpg").exists())

    def test_event_album_apply_flat_layout(self):
        fid = self.add_file("sub/a.jpg")
        self.add_event(fid, "Свадьба")
        report = plan_album(self.cfg, self.conn, "event", "Свадьба", self.dest,
                            mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertTrue((self.dest / "Свадьба" / "a.jpg").exists())

    def test_move_prints_warning_dry_run_and_apply(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                      mode="move", apply=False)
        self.assertIn("ВНИМАНИЕ", buf.getvalue())

    def test_move_transfers_single_owner_file_and_updates_path(self):
        fid = self.add_file("a.jpg", content=b"hello")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertEqual(report.blocked_multi, 0)
        dst = self.dest / "Мама" / "a.jpg"
        self.assertTrue(dst.exists())
        self.assertEqual(self.path_of(fid), str(dst))

    def test_move_blocks_multi_person_files(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        self.add_person(fid, "Папа", bbox="[20,20,30,30]")
        before = self.path_of(fid)
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="move", apply=True)
        self.assertEqual(report.blocked_multi, 1)
        self.assertEqual(report.transferred, 0)
        self.assertTrue(Path(before).exists())
        self.assertFalse((self.dest / "Мама" / "a.jpg").exists())
        self.assertEqual(self.path_of(fid), before)

    def test_link_and_copy_transfer_multi_person_files_without_restriction(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        self.add_person(fid, "Папа", bbox="[20,20,30,30]")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertEqual(report.blocked_multi, 0)

    # --- F61: dest_root — the albums root, not the album folder (for the UI tree) ----

    def test_person_album_dest_root_is_dest_not_album_dir(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="link", apply=True)
        dest_root = self.conn.execute(
            "SELECT dest_root FROM move_batches WHERE id = ?", (report.batch_id,)
        ).fetchone()["dest_root"]
        self.assertEqual(dest_root, str(Path(self.dest).resolve()))
        self.assertNotEqual(dest_root, str(report.dest))  # NOT the album folder

    def test_event_album_dest_root_is_dest_not_album_dir(self):
        fid = self.add_file("a.jpg")
        self.add_event(fid, "Свадьба")
        report = plan_album(self.cfg, self.conn, "event", "Свадьба", self.dest,
                            mode="copy", apply=True)
        dest_root = self.conn.execute(
            "SELECT dest_root FROM move_batches WHERE id = ?", (report.batch_id,)
        ).fetchone()["dest_root"]
        self.assertEqual(dest_root, str(Path(self.dest).resolve()))

    def test_album_dst_not_flat_under_dest_root(self):
        fid = self.add_file("a.jpg")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="link", apply=True)
        dest_root = self.conn.execute(
            "SELECT dest_root FROM move_batches WHERE id = ?", (report.batch_id,)
        ).fetchone()["dest_root"]
        dst = self.conn.execute(
            "SELECT dst FROM moves WHERE batch_id = ?", (report.batch_id,)
        ).fetchone()["dst"]
        rel = Path(dst).relative_to(Path(dest_root))
        self.assertEqual(rel.parts[0], "Мама")  # album name, not file name — not flat

    def test_album_undo_copy_mode_unaffected_by_dest_root_change(self):
        fid = self.add_file("a.jpg", content=b"hello")
        self.add_event(fid, "Свадьба")
        report = plan_album(self.cfg, self.conn, "event", "Свадьба", self.dest,
                            mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / "Свадьба" / "a.jpg"
        self.assertTrue(dst.exists())
        src = Path(self.path_of(fid))
        stats = undo(self.conn, report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertFalse(dst.exists())
        self.assertTrue(src.exists())

    def test_album_undo_move_mode_unaffected_by_dest_root_change(self):
        fid = self.add_file("a.jpg", content=b"hello")
        self.add_person(fid, "Мама")
        report = plan_album(self.cfg, self.conn, "person", "Мама", self.dest,
                            mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / "Мама" / "a.jpg"
        self.assertTrue(dst.exists())
        stats = undo(self.conn, report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertFalse(dst.exists())
        self.assertTrue(Path(self.path_of(fid)).exists())


if __name__ == "__main__":
    unittest.main()
