"""F20: stage summaries (_summarize_*) and printing per-step results in `sorta run`."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sorta import cli


class TestSummaries(unittest.TestCase):
    def test_index(self):
        s = SimpleNamespace(added=2, updated=1, skipped=3, errors=1)
        out = cli._summarize_index(s, 4)
        self.assertIn("+2 новых", out)
        self.assertIn("~1 обновлено", out)
        self.assertIn("1 ошибок", out)
        self.assertIn("4 дубликатов", out)

    def test_geo(self):
        s = SimpleNamespace(total=10, exact_gps=4, session_inferred=2, unknown=4)
        out = cli._summarize_geo(s)
        self.assertIn("10 файлов", out)
        self.assertIn("exact_gps 4", out)
        self.assertIn("unknown 4", out)

    def test_landmarks_with_breakdown(self):
        s = SimpleNamespace(scanned=5, matched=2, by_landmark={"Айя-София": 1, "Колизей": 1})
        out = cli._summarize_landmarks(s)
        self.assertIn("просмотрено 5, определено 2", out)
        self.assertIn("  Айя-София: 1", out)
        self.assertIn("  Колизей: 1", out)

    def test_faces_with_malformed(self):
        fs = SimpleNamespace(files_processed=8, faces_found=12, no_face_files=2, errors=1)
        cs = SimpleNamespace(clusters=3, faces=12, noise=2, labels_kept=1, malformed=4)
        out = cli._summarize_faces(fs, cs)
        self.assertIn("8 файлов, 12 лиц", out)
        self.assertIn("лиц в кластерах: 10", out)  # faces - noise
        self.assertIn("повреждённых эмбеддингов пропущено: 4", out)

    def test_faces_without_malformed_omits_warning(self):
        fs = SimpleNamespace(files_processed=1, faces_found=1, no_face_files=0, errors=0)
        cs = SimpleNamespace(clusters=1, faces=1, noise=0, labels_kept=0, malformed=0)
        out = cli._summarize_faces(fs, cs)
        self.assertNotIn("повреждённых", out)

    def test_events(self):
        s = SimpleNamespace(auto_events=3, auto_files=20, names_preserved=1,
                            manual_events=1, manual_files=5)
        out = cli._summarize_events(s)
        self.assertIn("3 авто (20 файлов", out)
        self.assertIn("1 ручных (5 файлов)", out)

    def test_junk_verdict_breakdown(self):
        s = SimpleNamespace(total=100, processed=40, by_verdict={"photo": 30, "screenshot": 10})
        out = cli._summarize_junk(s)
        self.assertIn("40/100 обработано", out)
        self.assertIn("photo: 30", out)
        self.assertIn("screenshot: 10", out)


class TestRunPrintsSummaries(unittest.TestCase):
    """`sorta run` prints a header AND each stage's summary (indented)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{(root / "test.db").as_posix()}"\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_prints_indented_multiline_summaries(self):
        # Replace the steps with fakes (no ML): we check specifically the summary printing.
        fake_steps = [
            ("alpha", lambda cfg, conn, cb: "Готово: одна строка"),
            ("beta", lambda cfg, conn, cb: "строка1\n  вложенная2"),
        ]
        original = cli._pipeline_steps
        cli._pipeline_steps = lambda: fake_steps  # type: ignore[assignment]
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._cmd_run(str(self.cfg_path))
            out = buf.getvalue()
        finally:
            cli._pipeline_steps = original  # type: ignore[assignment]

        self.assertIn("[этап 1/2] alpha", out)
        self.assertIn("  Готово: одна строка", out)
        self.assertIn("[этап 2/2] beta", out)
        self.assertIn("  строка1", out)
        self.assertIn("    вложенная2", out)  # the original indent + the print indent


class TestRunDeepGeoOverride(unittest.TestCase):
    """F50/#34: `run --deep`/`--geo online` build a per-run cfg via
    dataclasses.replace, without touching config.yaml on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{(root / "test.db").as_posix()}"\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _run_and_capture_cfg(self, **kwargs) -> object:
        captured = {}

        def fake_step(cfg, conn, cb):
            captured["cfg"] = cfg
            return "ok"

        original = cli._pipeline_steps
        cli._pipeline_steps = lambda: [("alpha", fake_step)]  # type: ignore[assignment]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._cmd_run(str(self.cfg_path), **kwargs)
        finally:
            cli._pipeline_steps = original  # type: ignore[assignment]
        return captured["cfg"]

    def test_no_flags_cfg_matches_config_defaults(self):
        cfg = self._run_and_capture_cfg()
        self.assertFalse(cfg.naming.vlm_enabled)
        self.assertEqual(cfg.geo.provider, "offline")

    def test_deep_true_overrides_vlm_enabled(self):
        cfg = self._run_and_capture_cfg(deep=True)
        self.assertTrue(cfg.naming.vlm_enabled)

    def test_deep_false_overrides_vlm_enabled_off(self):
        cfg = self._run_and_capture_cfg(deep=False)
        self.assertFalse(cfg.naming.vlm_enabled)

    def test_geo_online_overrides_provider(self):
        cfg = self._run_and_capture_cfg(geo="online")
        self.assertEqual(cfg.geo.provider, "online")

    def test_geo_offline_overrides_provider(self):
        cfg = self._run_and_capture_cfg(geo="offline")
        self.assertEqual(cfg.geo.provider, "offline")


class TestRunOptionalStages(unittest.TestCase):
    """F53/#39: `--faces`/`--events` — opt-in steps, default off. The base run builds
    only index/geo/landmarks/junk/phash; faces/events are added independently of each
    other by flags."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{(root / "test.db").as_posix()}"\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _run_and_capture_calls(self, **kwargs) -> tuple[list[str], str]:
        calls: list[str] = []

        def fake_step(name):
            def _fn(cfg, conn, cb):
                calls.append(name)
                return "ok"
            return _fn

        fake_steps = [(name, fake_step(name)) for name in
                      ("index", "geo", "landmarks", "faces", "events", "junk", "phash")]
        original = cli._pipeline_steps
        cli._pipeline_steps = lambda: fake_steps  # type: ignore[assignment]
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._cmd_run(str(self.cfg_path), **kwargs)
            out = buf.getvalue()
        finally:
            cli._pipeline_steps = original  # type: ignore[assignment]
        return calls, out

    def test_no_flags_skips_faces_and_events(self):
        calls, out = self._run_and_capture_calls()
        self.assertEqual(calls, ["index", "geo", "landmarks", "junk", "phash"])
        self.assertIn("[этап 1/5] index", out)
        self.assertIn("[этап 5/5] phash", out)

    def test_faces_true_adds_faces_only(self):
        calls, _out = self._run_and_capture_calls(faces=True)
        self.assertEqual(calls, ["index", "geo", "landmarks", "faces", "junk", "phash"])

    def test_events_true_adds_events_only(self):
        calls, _out = self._run_and_capture_calls(events=True)
        self.assertEqual(calls, ["index", "geo", "landmarks", "events", "junk", "phash"])

    def test_src_overrides_config_sources(self):
        # F59: --src overrides config sources for this run.
        captured: dict = {}

        def capture_step(cfg, conn, cb):
            captured.setdefault("sources", list(cfg.sources))
            return "ok"

        fake_steps = [(name, capture_step) for name in
                      ("index", "geo", "landmarks", "junk", "phash")]
        other = Path(self.tmp.name) / "explicit_src"
        other.mkdir()
        original = cli._pipeline_steps
        cli._pipeline_steps = lambda: fake_steps  # type: ignore[assignment]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._cmd_run(str(self.cfg_path), src=str(other))
        finally:
            cli._pipeline_steps = original  # type: ignore[assignment]
        self.assertEqual(captured["sources"], [other.resolve()])

    def test_faces_and_events_true_adds_both(self):
        calls, out = self._run_and_capture_calls(faces=True, events=True)
        self.assertEqual(
            calls, ["index", "geo", "landmarks", "faces", "events", "junk", "phash"])
        self.assertIn("[этап 1/7] index", out)
        self.assertIn("[этап 7/7] phash", out)


class TestLazySharedClassifier(unittest.TestCase):
    """F19: the shared CLIP classifier is built lazily and reused."""

    def test_builds_once_on_first_call_and_reuses(self):
        builds = []

        def fake_real(paths, prompts):
            return np.zeros((len(paths), len(prompts)), dtype=np.float32)

        def factory():
            builds.append(1)
            return fake_real

        clf = cli._LazySharedClassifier(factory)
        self.assertEqual(builds, [])  # lazy: the model is not built before the call
        r1 = clf(["/a.jpg"], ["p"])
        r2 = clf(["/b.jpg"], ["p", "q"])
        self.assertEqual(len(builds), 1)  # the factory is called exactly once
        self.assertEqual(r1.shape, (1, 1))
        self.assertEqual(r2.shape, (1, 2))

    def test_never_called_never_builds(self):
        builds = []
        cli._LazySharedClassifier(lambda: builds.append(1) or (lambda p, q: None))
        self.assertEqual(builds, [])  # never called — never built


class TestPipelineSharesClassifier(unittest.TestCase):
    """F19: landmarks and junk in the pipeline get ONE classifier."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_landmarks_and_junk_get_same_lazy_classifier(self):
        from sorta.config import Config
        from sorta.db import connect

        cfg = Config(sources=[self.root / "src"], database=self.root / "t.db")
        conn = connect(cfg.database)
        captured = {}

        def fake_landmarks(cfg, conn, classifier=None, progress=None):
            captured["landmarks"] = classifier
            return SimpleNamespace(scanned=0, matched=0, by_landmark={})

        def fake_junk(cfg, conn, classifier=None, progress=None):
            captured["junk"] = classifier
            return SimpleNamespace(total=0, processed=0, by_verdict={})

        orig_l, orig_j = cli.detect_landmarks, cli.classify_junk
        cli.detect_landmarks, cli.classify_junk = fake_landmarks, fake_junk
        try:
            steps = dict(cli._pipeline_steps())
            steps["landmarks"](cfg, conn, lambda a, b: None)  # type: ignore[operator]
            steps["junk"](cfg, conn, lambda a, b: None)  # type: ignore[operator]
        finally:
            cli.detect_landmarks, cli.classify_junk = orig_l, orig_j
            conn.close()

        self.assertIsInstance(captured["landmarks"], cli._LazySharedClassifier)
        self.assertIs(captured["landmarks"], captured["junk"])  # one instance


class TestIndexSourceArg(unittest.TestCase):
    """F28 (#16 pt.1): a positional source + config tolerates empty sources."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.cfg_path = self.root / "config.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_cfg(self, sources_line: str = "") -> None:
        self.cfg_path.write_text(
            sources_line + f'database: "{(self.root / "t.db").as_posix()}"\n',
            encoding="utf-8")

    def test_load_config_tolerates_empty_sources(self):
        from sorta.config import load_config
        self._write_cfg()  # no sources section
        cfg = load_config(str(self.cfg_path))
        self.assertEqual(cfg.sources, [])

    def test_index_without_source_raises(self):
        self._write_cfg()  # neither sources in config nor positional
        with self.assertRaises(ValueError) as ctx:
            cli._cmd_index(str(self.cfg_path))
        self.assertIn("источник", str(ctx.exception))

    def test_index_positional_overrides_config_sources(self):
        photos = self.root / "photos"
        photos.mkdir()
        self._write_cfg(f'sources: ["{(self.root / "other").as_posix()}"]\n')
        captured = {}

        def fake_index(cfg, conn, progress=None):
            captured["sources"] = list(cfg.sources)
            return SimpleNamespace(added=0, updated=0, skipped=0, errors=0)

        orig_i, orig_d = cli.run_index, cli.assign_duplicates
        cli.run_index = fake_index  # type: ignore[assignment]
        cli.assign_duplicates = lambda conn, strat: 0  # type: ignore[assignment]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._cmd_index(str(self.cfg_path), src=str(photos))
        finally:
            cli.run_index, cli.assign_duplicates = orig_i, orig_d  # type: ignore[assignment]
        self.assertEqual(captured["sources"], [photos.resolve()])


class TestConfigureLoggingCalledAfterLoadConfig(unittest.TestCase):
    """F52: commands configure logging by cfg.log_level after load_config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'database: "{(root / "test.db").as_posix()}"\n'
            'log_level: DEBUG\n',
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cmd_stats_configures_logging_from_config(self):
        with patch("sorta.cli.configure_logging") as mock_configure:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._cmd_stats(str(self.cfg_path))
        mock_configure.assert_called_once_with("DEBUG")


if __name__ == "__main__":
    unittest.main()
