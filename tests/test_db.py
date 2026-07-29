"""SQLite schema and migrations."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect


class TestMigrations(unittest.TestCase):
    def test_fresh_db_has_current_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "new.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
            self.assertIn("orientation", cols)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("media_class", tables)
            ev_cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
            self.assertIn("origin", ev_cols)
            self.assertIn("not_personal", cols)
            pl_cols = {r["name"] for r in conn.execute("PRAGMA table_info(places)")}
            self.assertIn("city_geonameid", pl_cols)
            self.assertIn("district_geonameid", pl_cols)
            self.assertIn("district_name", pl_cols)
            self.assertIn("country_name", pl_cols)
            tbls = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("dedup_choice", tbls)
            self.assertIn("geo_cache", tbls)  # v13 (F93)
            self.assertIn("manual_places", tbls)  # v14 (F85c)
            self.assertIn("frame_quality", tbls)  # v15 (F113)
            (v,) = conn.execute("PRAGMA user_version").fetchone()
            self.assertEqual(v, 15)
            conn.close()

    def test_v1_db_migrates_to_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            raw = sqlite3.connect(db)
            # a minimal v1 files table without orientation
            raw.executescript(
                """PRAGMA user_version = 1;
                   CREATE TABLE files (
                       id INTEGER PRIMARY KEY,
                       path TEXT NOT NULL UNIQUE, size INTEGER NOT NULL,
                       mtime REAL NOT NULL, ext TEXT NOT NULL, media_type TEXT NOT NULL,
                       hash TEXT, hash_algo TEXT, phash TEXT,
                       taken_at TEXT, taken_at_source TEXT, taken_at_confidence TEXT,
                       gps_lat REAL, gps_lon REAL, camera_make TEXT, camera_model TEXT,
                       width INTEGER, height INTEGER,
                       dup_of INTEGER REFERENCES files(id), error TEXT,
                       indexed_at TEXT NOT NULL
                   );
                   CREATE TABLE events (
                       id INTEGER PRIMARY KEY,
                       started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
                       place_city TEXT, name TEXT NOT NULL,
                       name_is_manual INTEGER NOT NULL DEFAULT 0
                   );
                   CREATE TABLE places (
                       file_id INTEGER PRIMARY KEY REFERENCES files(id),
                       country TEXT, region TEXT, city TEXT,
                       confidence TEXT NOT NULL, updated_at TEXT NOT NULL
                   );
                   CREATE TABLE move_batches (
                       id INTEGER PRIMARY KEY, mode TEXT NOT NULL,
                       dest_root TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT
                   );"""
            )
            raw.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')"
            )
            raw.commit()
            raw.close()

            conn = connect(db)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
            self.assertIn("orientation", cols)
            ev_cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
            self.assertIn("origin", ev_cols)
            self.assertIn("not_personal", cols)
            pl_cols = {r["name"] for r in conn.execute("PRAGMA table_info(places)")}
            self.assertIn("city_geonameid", pl_cols)  # added by the v6 migration
            self.assertIn("country_name", pl_cols)     # added by the v10 migration
            (v,) = conn.execute("PRAGMA user_version").fetchone()
            self.assertEqual(v, 15)
            row = conn.execute("SELECT * FROM files").fetchone()
            self.assertEqual(row["path"], "/a.jpg")
            self.assertIsNone(row["orientation"])
            self.assertEqual(row["not_personal"], 0)
            conn.close()


    def test_v10_db_gets_media_class_tier_backfilled(self):
        """v11: media_class.tier is added and backfilled from source.

        'ocr' is a verdict of the fast (clip) tier, not a tier of its own — mapping
        it to 'clip' is what keeps the upgrade from reclassifying the collection.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v10.db"
            raw = sqlite3.connect(db)
            raw.executescript(
                """PRAGMA user_version = 10;
                   CREATE TABLE media_class (
                       file_id INTEGER PRIMARY KEY,
                       verdict TEXT NOT NULL, source TEXT NOT NULL,
                       score REAL, updated_at TEXT NOT NULL
                   );"""
            )
            for fid, source in enumerate(("clip", "ocr", "vlm", "heuristic"), start=1):
                raw.execute(
                    "INSERT INTO media_class (file_id, verdict, source, updated_at) "
                    "VALUES (?, 'photo', ?, 'x')", (fid, source))
            raw.commit()
            raw.close()

            conn = connect(db)
            tiers = {r["source"]: r["tier"]
                     for r in conn.execute("SELECT source, tier FROM media_class")}
            self.assertEqual(tiers, {"clip": "clip", "ocr": "clip",
                                     "vlm": "vlm", "heuristic": "heuristic"})
            conn.close()

    def test_v1_db_without_media_class_migrates(self):
        """Regression: media_class only appeared in v3 — the v11 ALTER must not run
        on a v1/v2 DB, where executescript creates the table from scratch instead."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v1.db"
            raw = sqlite3.connect(db)
            raw.executescript("PRAGMA user_version = 1; CREATE TABLE files ("
                              "id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,"
                              "size INTEGER NOT NULL, mtime REAL NOT NULL,"
                              "ext TEXT NOT NULL, media_type TEXT NOT NULL,"
                              "hash TEXT, hash_algo TEXT, phash TEXT,"
                              "taken_at TEXT, taken_at_source TEXT,"
                              "taken_at_confidence TEXT, gps_lat REAL, gps_lon REAL,"
                              "camera_make TEXT, camera_model TEXT,"
                              "width INTEGER, height INTEGER,"
                              "dup_of INTEGER REFERENCES files(id), error TEXT,"
                              "indexed_at TEXT NOT NULL);"
                              "CREATE TABLE events (id INTEGER PRIMARY KEY,"
                              "started_at TEXT NOT NULL, ended_at TEXT NOT NULL,"
                              "place_city TEXT, name TEXT NOT NULL,"
                              "name_is_manual INTEGER NOT NULL DEFAULT 0);"
                              "CREATE TABLE places (file_id INTEGER PRIMARY KEY,"
                              "country TEXT, region TEXT, city TEXT,"
                              "confidence TEXT NOT NULL, updated_at TEXT NOT NULL);"
                              "CREATE TABLE move_batches (id INTEGER PRIMARY KEY,"
                              "mode TEXT NOT NULL, dest_root TEXT NOT NULL,"
                              "started_at TEXT NOT NULL, finished_at TEXT);")
            raw.commit()
            raw.close()

            conn = connect(db)  # must not raise "no such table: media_class"
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(media_class)")}
            self.assertIn("tier", cols)
            conn.close()


    def test_manual_overrides_table_exists_and_is_wiped_by_reset(self):
        """v12 (F77): manual corrections live by the same rule as every other manual
        decision — a from-scratch reindex starts clean."""
        from sorta.db import reset_index

        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "m.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(manual_overrides)")}
            self.assertEqual(cols, {"file_id", "action", "target", "updated_at"})
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/x.jpg', 1, 0.0, 'jpg', 'photo', 'now')")
            conn.execute(
                "INSERT INTO manual_overrides (file_id, action, target, updated_at) "
                "VALUES (1, 'exclude', NULL, 'now')")
            conn.commit()
            reset_index(conn)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM manual_overrides").fetchone()[0], 0)
            conn.close()

    def test_manual_places_table_exists_and_is_wiped_by_reset(self):
        """v14 (F85c): a place the user assigned to a whole group. It lives outside
        `places` because geo recomputes that table from scratch — and it goes away on a
        reset like every other manual decision."""
        from sorta.db import reset_index

        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "p.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(manual_places)")}
            self.assertEqual(cols, {"file_id", "country", "city", "city_geonameid",
                                    "updated_at"})
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/x.jpg', 1, 0.0, 'jpg', 'photo', 'now')")
            conn.execute(
                "INSERT INTO manual_places (file_id, country, city, city_geonameid, "
                "updated_at) VALUES (1, 'GR', 'Athens', 264371, 'now')")
            conn.commit()
            reset_index(conn)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM manual_places").fetchone()[0], 0)
            conn.close()


class TestReset(unittest.TestCase):
    def test_reset_index_clears_data_keeps_schema(self):
        from sorta.db import reset_index
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "r.db")
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/x.jpg', 1, 0.0, 'jpg', 'photo', 'now')")
            conn.commit()
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 1)
            reset_index(conn)
            # data wiped, schema alive (tables + user_version)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0], 0)
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 15)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("media_class", tables)
            self.assertIn("move_batches", tables)
            conn.close()


if __name__ == "__main__":
    unittest.main()
