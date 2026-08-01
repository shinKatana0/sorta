"""F127: everything F113-F125 added is reachable from the terminal, by flag.

The hole this closes is one the web app already had closed (F119): animals, the
quality cascade and the preview-cache ceiling arrived in the product and none of them
grew a flag, so the only way to switch them on for a single run was to edit
config.yaml — and then remember to edit it back.

Two properties carry most of the cases below, because they are what makes a flag an
override rather than a second source of truth:

* the value is taken from the CONFIG when the flag is absent, and the flag can move it
  in BOTH directions (`--no-pets` has to switch off what `features.pets: true`
  switched on — an option defaulting to False could only ever add);
* nothing is written to config.yaml. The override lives for one run.

The third one is the dangerous case, and it is about the album: `animal` gains an
optional selector, and a missing selector for `person`/`event` has to be an error
rather than an album quietly gathered from the whole collection.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sorta import cli, imaging
from sorta.config import VLM_QUALITY_SCOPES
from sorta.junk import quality_settings


class _FlagCase(unittest.TestCase):
    """A temp project and the built application, invoked the way a user does."""

    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback
            self.skipTest("typer is not installed")
        # F117 keys are APPLIED by setting an environment variable, so a command run
        # here changes the process for every case that follows — the preview cache then
        # runs with a ceiling nobody in that test asked for, and the failure surfaces in
        # a completely different file (test_imaging_preview). Snapshot the environment
        # for EVERY case in this file, not only the ones that obviously write it: the
        # leak is invisible where it is caused. The same guard exists in
        # test_ui_settings.SettingsTestBase, for the same reason.
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        from typer.testing import CliRunner
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        self.cfg_path = self.root / "config.yaml"
        self.write_config()

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, extra: str = "") -> None:
        self.cfg_path.write_text(
            f'sources: ["{(self.root / "src").as_posix()}"]\n'
            f'database: "{(self.root / "t.db").as_posix()}"\n'
            "language: en\n" + extra, encoding="utf-8")

    def invoke(self, argv: list[str]):
        return self.runner.invoke(cli.build_app("en"),
                                  [*argv, "--config", str(self.cfg_path)])


class _CapturesTheRunConfig(_FlagCase):
    """The config a stage actually receives — for `junk` the real command body, for
    `run` the pipeline with its steps replaced (no ML in the suite)."""

    def junk_cfg(self, argv: list[str]):
        captured = {}

        def fake_classify(cfg, conn, classifier=None, progress=None):
            captured["cfg"] = cfg
            return SimpleNamespace(total=0, processed=0, by_verdict={})

        with patch.object(cli, "classify_junk", fake_classify):
            result = self.invoke(["junk", *argv])
        self.assertEqual(result.exit_code, 0, result.output)
        return captured["cfg"]

    def run_cfg(self, argv: list[str]):
        captured = {}

        def fake_step(cfg, conn, cb):
            captured["cfg"] = cfg
            return "ok"

        with patch.object(cli, "_pipeline_steps", lambda: [("junk", fake_step)]):
            result = self.invoke(["run", *argv])
        self.assertEqual(result.exit_code, 0, result.output)
        return captured["cfg"]


class TestPets(_CapturesTheRunConfig):
    """Requirement 1: `--pets`/`--no-pets` on `junk` and on `run` — exactly like
    `--faces`/`--events`, an override of `features.pets` for this run."""

    def test_the_flag_switches_animals_on_for_this_run(self):
        for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
            with self.subTest(command=command):
                self.assertTrue(cfg_of(["--pets"]).features.pets)

    def test_no_pets_switches_off_what_the_config_switched_on(self):
        self.write_config("features:\n  pets: true\n")
        for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
            with self.subTest(command=command):
                self.assertFalse(cfg_of(["--no-pets"]).features.pets)

    def test_without_the_flag_the_config_decides(self):
        for expected, section in ((True, "features:\n  pets: true\n"),
                                  (False, "features:\n  pets: false\n")):
            self.write_config(section)
            for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
                with self.subTest(command=command, expected=expected):
                    self.assertIs(cfg_of([]).features.pets, expected)

    def test_the_flag_reaches_the_settings_the_stage_reads(self):
        # features.pets is not consumed as a field but through junk.quality_settings —
        # the override has to be visible THERE, not merely on the dataclass.
        self.assertTrue(quality_settings(self.junk_cfg(["--pets"])).pets)


class TestQuality(_CapturesTheRunConfig):
    """Requirement 2.1: `--quality`/`--no-quality` overrides `vlm.quality`."""

    def test_the_flag_switches_the_quality_questions_on(self):
        for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
            with self.subTest(command=command):
                self.assertTrue(quality_settings(cfg_of(["--quality"])).vlm_quality)

    def test_no_quality_switches_off_what_the_config_switched_on(self):
        self.write_config("vlm:\n  quality: true\n")
        for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
            with self.subTest(command=command):
                self.assertFalse(
                    quality_settings(cfg_of(["--no-quality"])).vlm_quality)

    def test_without_the_flag_the_config_decides(self):
        self.write_config("vlm:\n  quality: true\n")
        self.assertTrue(quality_settings(self.junk_cfg([])).vlm_quality)
        self.write_config("vlm:\n  quality: false\n")
        self.assertFalse(quality_settings(self.junk_cfg([])).vlm_quality)

    def test_the_quality_flag_does_not_touch_the_deep_junk_tier(self):
        """Two consumers of one runtime (F113): `--quality` is not `--deep`."""
        cfg = self.junk_cfg(["--quality"])
        self.assertFalse(cfg.naming.vlm_enabled)
        self.assertFalse(cfg.vlm.enabled)


class TestQualityScope(_CapturesTheRunConfig):
    """Requirement 2.2/2.3: a closed list of scopes, and a typo says so."""

    def test_faces_reaches_the_quality_settings(self):
        # F125's value in particular: the point of the dependency this feature waited on.
        for command, cfg_of in (("junk", self.junk_cfg), ("run", self.run_cfg)):
            with self.subTest(command=command):
                self.assertEqual(
                    quality_settings(cfg_of(["--quality-scope", "faces"])).vlm_scope,
                    "faces")

    def test_every_scope_the_config_accepts_is_accepted_here(self):
        for scope in VLM_QUALITY_SCOPES:
            with self.subTest(scope=scope):
                self.assertEqual(
                    quality_settings(
                        self.junk_cfg(["--quality-scope", scope])).vlm_scope, scope)

    def test_without_the_flag_the_config_decides(self):
        self.write_config("vlm:\n  quality_scope: events\n")
        self.assertEqual(quality_settings(self.junk_cfg([])).vlm_scope, "events")

    def test_a_typo_is_an_error_that_lists_the_values(self):
        for command in ("junk", "run"):
            with self.subTest(command=command):
                result = self.invoke([command, "--quality-scope", "facse"])
                self.assertNotEqual(result.exit_code, 0)
                printed = " ".join(result.output.split())
                for scope in VLM_QUALITY_SCOPES:
                    self.assertIn(scope, printed)

    def test_a_typo_does_not_quietly_fall_back_to_the_default(self):
        """The whole reason the list is closed: a misspelt scope must not run the
        wrong population for four hours."""
        with patch.object(cli, "_cmd_junk") as cmd:
            result = self.invoke(["junk", "--quality-scope", "all-of-them"])
        self.assertNotEqual(result.exit_code, 0)
        cmd.assert_not_called()


class TestAlbumSelectorIsOptionalOnlyForAnimals(_FlagCase):
    """Requirement 3. The animal slice has nothing to select inside it, so
    `sorta album animal --dest ...` is the whole command; for a person and an event the
    selector IS the subject and its absence has to be refused out loud."""

    def album_call(self, argv: list[str]):
        captured = {}

        def fake_plan(cfg, conn, kind, selector, dest, **kwargs):
            captured.update(kind=kind, selector=selector, dest=dest)
            return SimpleNamespace(album_name="A", transferred=0, failed=0,
                                   blocked_multi=0)

        with patch.object(cli, "plan_album", fake_plan):
            result = self.invoke(argv)
        return result, captured

    def test_the_animal_album_needs_no_selector(self):
        dest = self.root / "albums"
        result, called = self.album_call(["album", "animal", "--dest", str(dest)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(called["kind"], "animal")
        self.assertEqual(called["selector"], "")  # what plan_album documents for animal
        self.assertEqual(called["dest"], dest)

    def test_the_animal_album_still_tolerates_the_old_empty_selector(self):
        """`sorta album animal "" --dest ...` was the only spelling until now, and a
        command line in someone's notes has to keep working."""
        result, called = self.album_call(
            ["album", "animal", "", "--dest", str(self.root / "albums")])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(called["selector"], "")

    def test_a_person_album_without_a_selector_is_refused(self):
        for kind in ("person", "event"):
            with self.subTest(kind=kind):
                result, called = self.album_call(
                    ["album", kind, "--dest", str(self.root / "albums")])
                self.assertNotEqual(result.exit_code, 0)
                self.assertEqual(called, {})  # nothing was gathered
                self.assertIn("selector", " ".join(result.output.split()))

    def test_a_blank_selector_is_refused_the_same_way(self):
        """An empty string is an absent selector spelled differently — it used to give
        an empty album with no explanation."""
        result, called = self.album_call(
            ["album", "person", "   ", "--dest", str(self.root / "albums")])
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(called, {})

    def test_a_selector_still_reaches_the_album(self):
        result, called = self.album_call(
            ["album", "person", "Mum", "--dest", str(self.root / "albums")])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual((called["kind"], called["selector"]), ("person", "Mum"))


class TestPreviewCacheCeiling(_FlagCase):
    """Requirement 4: `--preview-max-gb` sets the ceiling without editing the config."""

    def setUp(self):
        super().setUp()
        # The preview cache is a user-level directory: point it at the temp project so
        # the case neither reads nor reports the developer's real cache.
        env = patch.dict(os.environ, {imaging.ENV_PREVIEW_DIR:
                                      str(self.root / "previews")})
        env.start()
        self.addCleanup(env.stop)
        for name in (imaging.ENV_PREVIEW_MAX_GB,):
            os.environ.pop(name, None)

    def cache(self, argv: list[str]) -> tuple[str, float]:
        """The printed report and the ceiling in force while it was printed."""
        seen = {}
        original = imaging.preview_cache_max_gb

        def spy() -> float:
            seen["limit"] = original()
            return seen["limit"]

        with patch.object(imaging, "preview_cache_max_gb", spy):
            result = self.invoke(["cache", *argv])
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output, seen["limit"]

    def test_the_flag_sets_the_ceiling(self):
        _out, limit = self.cache(["--preview-max-gb", "40"])
        self.assertEqual(limit, 40.0)

    def test_zero_means_no_ceiling_even_when_the_config_sets_one(self):
        self.write_config("imaging:\n  preview_cache_max_gb: 40\n")
        output, limit = self.cache(["--preview-max-gb", "0"])
        self.assertEqual(limit, 0.0)
        self.assertIn("ceiling: none set", output)

    def test_without_the_flag_the_config_decides(self):
        self.write_config("imaging:\n  preview_cache_max_gb: 40\n")
        output, limit = self.cache([])
        self.assertEqual(limit, 40.0)
        self.assertIn("ceiling: 40.00 GB", output)

    def test_the_flag_wins_over_the_config(self):
        self.write_config("imaging:\n  preview_cache_max_gb: 40\n")
        _out, limit = self.cache(["--preview-max-gb", "7"])
        self.assertEqual(limit, 7.0)

    def test_a_negative_ceiling_is_refused(self):
        result = self.invoke(["cache", "--preview-max-gb", "-3"])
        self.assertNotEqual(result.exit_code, 0)


class TestNoFlagWritesTheConfig(_CapturesTheRunConfig):
    """Requirement of the brief's boundaries, and the reason every flag above is an
    override: config.yaml on disk is the user's file and no run may edit it."""

    def test_the_config_file_is_untouched_by_every_new_flag(self):
        self.write_config("features:\n  pets: true\nvlm:\n  quality: true\n"
                          "imaging:\n  preview_cache_max_gb: 40\n")
        before = self.cfg_path.read_bytes()
        self.junk_cfg(["--no-pets", "--no-quality", "--quality-scope", "faces"])
        self.run_cfg(["--pets", "--quality", "--quality-scope", "all"])
        with patch.dict(os.environ, {imaging.ENV_PREVIEW_DIR:
                                     str(self.root / "previews")}):
            os.environ.pop(imaging.ENV_PREVIEW_MAX_GB, None)
            self.invoke(["cache", "--preview-max-gb", "0"])
        self.assertEqual(self.cfg_path.read_bytes(), before)

    def test_a_second_run_reads_the_config_again_and_not_the_override(self):
        """The override lives for one run: the next one starts from the file."""
        self.write_config("features:\n  pets: true\n")
        self.assertFalse(self.junk_cfg(["--no-pets"]).features.pets)
        self.assertTrue(self.junk_cfg([]).features.pets)


if __name__ == "__main__":
    unittest.main()
