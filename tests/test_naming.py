"""F6: naming settings, the template provider, provider selection, name_events."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, _naming_from
from sorta.db import connect
from sorta.naming import (
    ClaudeNamer,
    EventContext,
    LocalVLMNamer,
    TemplateNamer,
    make_namer,
    name_events,
    naming_settings,
)


def cfg_with(naming: dict | None = None, tmp: str = ".") -> Config:
    return Config(sources=[Path(tmp)], database=Path(tmp) / "test.db",
                  naming=_naming_from(naming or {}))


class TestSettings(unittest.TestCase):
    def test_defaults(self):
        s = naming_settings(cfg_with())
        self.assertEqual(s.provider, "template")
        self.assertAlmostEqual(s.landmark_threshold, 0.85)
        self.assertAlmostEqual(s.junk_threshold, 0.85)
        self.assertEqual(s.landmarks_file, "data/landmarks.yaml")

    def test_overrides_from_raw(self):
        s = naming_settings(cfg_with({
            "provider": "claude",
            "landmark_threshold": 0.5,
            "junk_threshold": 0.8,
            "clip": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
            "local_vlm": {"base_url": "http://gpu:11434/", "model": "qwen2.5vl"},
            "claude": {"model": "claude-haiku-4-5", "api_key_env": "MY_KEY"},
        }))
        self.assertEqual(s.provider, "claude")
        self.assertAlmostEqual(s.landmark_threshold, 0.5)
        self.assertAlmostEqual(s.junk_threshold, 0.8)
        self.assertEqual(s.clip_model, "ViT-B-32")
        self.assertEqual(s.vlm_base_url, "http://gpu:11434")  # without a trailing /
        self.assertEqual(s.vlm_model, "qwen2.5vl")
        self.assertEqual(s.claude_model, "claude-haiku-4-5")
        self.assertEqual(s.claude_api_key_env, "MY_KEY")


class TestTemplateNamer(unittest.TestCase):
    def name(self, started, ended, city=None):
        return TemplateNamer().name(
            EventContext(started_at=started, ended_at=ended, city=city))

    def test_single_day_with_city(self):
        self.assertEqual(
            self.name("2023-05-01T10:00:00", "2023-05-01T18:00:00", "Paris"),
            "2023-05-01 Paris")

    def test_single_day_without_city(self):
        self.assertEqual(
            self.name("2023-05-01T10:00:00", "2023-05-01T18:00:00"), "2023-05-01")

    def test_multi_day_same_year(self):
        self.assertEqual(
            self.name("2023-05-01T10:00:00", "2023-05-03T18:00:00", "Praha"),
            "2023-05-01..05-03 Praha")

    def test_multi_day_cross_year(self):
        self.assertEqual(
            self.name("2023-12-31T22:00:00", "2024-01-01T04:00:00"),
            "2023-12-31..2024-01-01")

    def test_invalid_date_returns_none(self):
        self.assertIsNone(self.name("мусор", "2023-05-01T18:00:00"))


class TestMakeNamer(unittest.TestCase):
    def test_template_is_default(self):
        namer = make_namer(naming_settings(cfg_with()))
        self.assertIsInstance(namer, TemplateNamer)

    def test_local_vlm(self):
        namer = make_namer(naming_settings(cfg_with({"provider": "local_vlm"})))
        self.assertIsInstance(namer, LocalVLMNamer)

    def test_claude_requires_api_key(self):
        s = naming_settings(cfg_with({"provider": "claude"}))
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                make_namer(s)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            self.assertIsInstance(make_namer(s), ClaudeNamer)

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            make_namer(naming_settings(cfg_with({"provider": "gpt"})))


class RecordingNamer:
    """Records contexts; answers from a prepared name dictionary."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.contexts = []

    def name(self, ctx: EventContext):
        self.contexts.append(ctx)
        return self.answers.get(ctx.started_at, "renamed")


class TestNameEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = cfg_with(tmp=self.tmp.name)
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at, media_type="photo", dup_of=None, error=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, dup_of, error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', ?, ?, 'exif', 'high', ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", media_type, taken_at, dup_of, error))
        self.conn.commit()
        return cur.lastrowid

    def add_event(self, started, ended, city=None, name="старое",
                  manual=0, file_ids=()):
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, place_city, name,
                   name_is_manual) VALUES (?,?,?,?,?)""",
            (started, ended, city, name, manual))
        for fid in file_ids:
            self.conn.execute(
                "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                (cur.lastrowid, fid))
        self.conn.commit()
        return cur.lastrowid

    def event_name(self, event_id):
        return self.conn.execute(
            "SELECT name FROM events WHERE id = ?", (event_id,)).fetchone()["name"]

    def test_template_provider_renames_auto_events(self):
        eid = self.add_event("2023-05-01T10:00:00", "2023-05-01T18:00:00", "Paris")
        stats = name_events(self.cfg, self.conn)  # default provider: template
        self.assertEqual(stats.renamed, 1)
        self.assertEqual(self.event_name(eid), "2023-05-01 Paris")

    def test_manual_event_untouched(self):
        eid = self.add_event("2023-05-01T10:00:00", "2023-05-01T18:00:00", "Paris",
                             name="Свадьба Ани", manual=1)
        namer = RecordingNamer()
        stats = name_events(self.cfg, self.conn, namer=namer)
        self.assertEqual(stats.manual_kept, 1)
        self.assertEqual(stats.total, 0)
        self.assertEqual(namer.contexts, [])  # a manual event is not shown to the provider
        self.assertEqual(self.event_name(eid), "Свадьба Ани")

    def test_none_keeps_current_name(self):
        eid = self.add_event("2023-05-01T10:00:00", "2023-05-01T18:00:00",
                             name="как было")
        namer = RecordingNamer(answers={"2023-05-01T10:00:00": None})
        stats = name_events(self.cfg, self.conn, namer=namer)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.renamed, 0)
        self.assertEqual(self.event_name(eid), "как было")

    def test_unchanged_counted(self):
        self.add_event("2023-05-01T10:00:00", "2023-05-01T18:00:00", "Paris",
                       name="2023-05-01 Paris")
        stats = name_events(self.cfg, self.conn)
        self.assertEqual(stats.unchanged, 1)
        self.assertEqual(stats.renamed, 0)

    def test_sample_paths_canonical_photos_only(self):
        ok1 = self.add_file("2023-05-01T10:00:00")
        canon = self.add_file("2023-05-01T11:00:00")
        dup = self.add_file("2023-05-01T12:00:00", dup_of=canon)
        broken = self.add_file("2023-05-01T13:00:00", error="boom")
        video = self.add_file("2023-05-01T14:00:00", media_type="video")
        self.add_event("2023-05-01T10:00:00", "2023-05-01T18:00:00",
                       file_ids=[ok1, canon, dup, broken, video])
        namer = RecordingNamer()
        name_events(self.cfg, self.conn, namer=namer)
        paths = namer.contexts[0].sample_paths
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.endswith(("img_1.jpg", "img_2.jpg")) for p in paths))


if __name__ == "__main__":
    unittest.main()
