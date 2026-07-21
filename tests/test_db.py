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
            (v,) = conn.execute("PRAGMA user_version").fetchone()
            self.assertEqual(v, 10)
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
            self.assertEqual(v, 10)
            row = conn.execute("SELECT * FROM files").fetchone()
            self.assertEqual(row["path"], "/a.jpg")
            self.assertIsNone(row["orientation"])
            self.assertEqual(row["not_personal"], 0)
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
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 10)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("media_class", tables)
            self.assertIn("move_batches", tables)
            conn.close()


if __name__ == "__main__":
    unittest.main()
