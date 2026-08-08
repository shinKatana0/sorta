"""F225: a download is carried to the end, on screen, and a half-finished one is not one.

Four defects of one line, found by the owner in a clean virtual machine on 2026-08-08 —
the third attempt to check the installer:

1. a run started from the shortcut died on the first line of progress any library tried
   to print. `pythonw.exe` leaves `sys.stdout` and `sys.stderr` as None, the run happens
   on a thread of that process, and huggingface_hub's progress bar ended the 1.6 GB
   download with `'NoneType' object has no attribute 'write'`. Fixed at the ENTRY POINT,
   because the next library to print a line arrives with the next version of
   transformers — hence nothing here knows the name `tqdm`;
2. the gigabytes travelled with nothing moving on the run screen, which reads as a hang
   for as long as it lasts;
3. the wizard asked which tiers to download and downloaded one of the four — the other
   three printed "it will be fetched some time later" and closed;
4. `doctor` and the wizard answered the same question two ways inside ONE output, and
   the cause underneath it is worse than the disagreement: an interrupted download left a
   cache directory behind, the probe read the directory as a downloaded model, and that
   machine would have been told "everything is in place" for ever while the stage failed
   on every run.

Everything here fakes the state of the disk — an empty cache, a `.incomplete` blob,
`sys.stdout is None`. That is the point: half of this cannot be reproduced on a machine
that already has the weights and runs its programs from a console, and "it works for me"
is precisely how the whole class of defect survived three releases.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import cli, i18n, tiers, tray, ui, weights, wizard
from sorta.ui import strings as ui_strings
from tests.test_ui_process import ProcessTestBase

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent
_APP_JS = _ROOT / "sorta" / "web" / "app" / "app.js"


class _NoisyLibrary:
    """Somebody else's code, drawing its progress into whatever stream it finds.

    Deliberately not tqdm and deliberately not named after it: the fix under test is that
    a windowed process HAS streams, and a test that knew the name of one library would
    say nothing about the next one.
    """

    def __init__(self, lines: int = 3) -> None:
        self.lines = lines
        self.built = 0

    def __call__(self, *args, **kwargs):
        self.built += 1
        for step in range(self.lines):
            print(f"\r{step * 33}%", end="", file=sys.stderr)
            print(f"fetching part {step}", file=sys.stdout)
        sys.stderr.flush()
        return lambda paths, prompts: [[0.0] * len(prompts) for _ in paths]


class _NoStreams:
    """A windowed interpreter: `pythonw.exe` starts with both streams set to None."""

    def __enter__(self):
        self._saved = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]
        sys.__stdout__ = None  # type: ignore[assignment]
        sys.__stderr__ = None  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__ = self._saved
        return False


# --- defect 1: the streams a windowed launcher does not give --------------------------


class TestAWindowedProcessHasSomewhereToPrint(unittest.TestCase):
    """The entry point, not the call that raised."""

    def test_both_streams_exist_after_the_entry_point_has_prepared_the_process(self):
        with _NoStreams():
            replaced = tray.ensure_streams()
            self.assertEqual(set(replaced), {"stdout", "stderr"})
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            # A library that reaches past the current streams finds them too.
            self.assertIsNotNone(sys.__stdout__)
            self.assertIsNotNone(sys.__stderr__)
            sys.stdout.write("still here\n")
            sys.stderr.write("and here\n")

    def test_a_console_that_is_there_is_left_exactly_as_it_was(self):
        out, err = sys.stdout, sys.stderr
        self.assertEqual(tray.ensure_streams(), ())
        self.assertIs(sys.stdout, out)
        self.assertIs(sys.stderr, err)

    def test_the_line_that_cannot_be_shown_goes_to_the_run_log(self):
        """A percentage nobody can read is still the line somebody reads afterwards."""
        with _NoStreams():
            tray.ensure_streams()
            with mock.patch.object(tray, "_LOG") as log:
                sys.stderr.write("Downloading open_clip_pytorch_model.bin: 42%\n")
                sys.stdout.write("resolving\n")
        logged = " ".join(str(call) for call in log.info.call_args_list)
        self.assertIn("42%", logged)
        self.assertIn("resolving", logged)

    def test_a_progress_bar_redrawing_one_line_is_not_a_log_record_per_redraw(self):
        with _NoStreams():
            tray.ensure_streams()
            with mock.patch.object(tray, "_LOG") as log:
                for percent in (10, 20, 30):
                    sys.stderr.write(f"\r{percent}%")
                sys.stderr.flush()
        self.assertEqual(log.info.call_count, 3)

    def test_nothing_here_pretends_to_be_a_terminal(self):
        """A bar that believes it has one redraws a line nobody will ever see."""
        with _NoStreams():
            tray.ensure_streams()
            self.assertFalse(sys.stderr.isatty())
            self.assertTrue(sys.stderr.writable())

    def test_the_entry_point_prepares_them_before_it_reads_anything(self):
        """`sorta-tray` is the launcher the shortcut runs, and everything it touches
        afterwards — the config, the logging, the whole pipeline — may print."""
        seen: dict[str, object] = {}

        def remember(*_args, **_kwargs):
            seen["stdout"] = sys.stdout
            seen["stderr"] = sys.stderr
            return 0

        with _NoStreams():
            with mock.patch.object(tray, "load_config", side_effect=remember):
                with self.assertRaises(Exception):
                    tray.main(["--no-browser"])
        self.assertIsNotNone(seen["stdout"])
        self.assertIsNotNone(seen["stderr"])

    def test_a_stream_that_is_written_bytes_by_mistake_does_not_take_the_run_down(self):
        with _NoStreams():
            tray.ensure_streams()
            sys.stderr.write(b"100%\n")  # type: ignore[arg-type]


class TestTheDownloadSurvivesAWindowWithNoConsole(ProcessTestBase):
    """The owner's crash, reproduced and then fixed, through the real download path."""

    def _run(self) -> ui.process._ProcessState:
        self.library = _NoisyLibrary()
        state = ui.process._ProcessState()
        state.try_start(str(self.src_dir))
        cache = ui.process.PlanCache(self.cfg, self.conn, self.root / "dest")
        options = ui.process._RunOptions(landmarks=True)
        self.patch_fast_stages()
        # The stage asks its classifier for something, which is what makes the factory
        # build one — and the factory is where the download happens.
        self._patch("detect_landmarks",
                    lambda cfg, conn, classifier=None, progress=None:
                    classifier(["a.jpg"], ["a photo"]))
        self._patch("clip_classifier", self.library)
        self._patch("stage_downloads", lambda stage, states=None: ("ViT-L-14",))
        ui.process._run_pipeline(self.cfg.database, self.cfg, str(self.src_dir),
                                 state, cache, options)
        return state

    def test_without_the_streams_the_run_dies_the_way_it_did_on_the_clean_machine(self):
        with _NoStreams():
            state = self._run()
        snapshot = state.snapshot()
        self.assertIsNotNone(snapshot["error"])
        self.assertEqual(snapshot["error_stage"], "landmarks")

    def test_with_them_the_download_is_carried_to_the_end(self):
        with _NoStreams():
            tray.ensure_streams()
            state = self._run()
        snapshot = state.snapshot()
        self.assertIsNone(snapshot["error"])
        self.assertTrue(snapshot["finished"])
        self.assertEqual(self.library.built, 1)  # the download itself went through
        # ...and the line about the download is gone once it is over.
        self.assertIsNone(snapshot["download"])


# --- defect 2: how much has arrived, on both screens ----------------------------------


class TestTheMeasurementIsSharedByBothScreens(unittest.TestCase):
    """One measurement, two callers — the wizard's console and the run screen."""

    def test_the_bytes_are_counted_on_the_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            blobs = Path(tmp) / "models--timm--vit" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "part.incomplete").write_bytes(b"x" * 4096)
            self.assertEqual(tiers.downloaded_bytes(Path(tmp)), 4096)

    def test_the_progress_is_reported_while_the_work_lasts(self):
        arrived = iter([0, 300_000_000, 700_000_000])
        seen: list[int] = []
        with mock.patch.object(tiers.threading, "Thread", _StepThread):
            failure = tiers.watch_download(lambda: None, seen.append,
                                           measure=lambda: next(arrived), tick=0.0)
        self.assertIsNone(failure)
        self.assertEqual(seen, [300_000_000])

    def test_a_refusal_comes_back_as_a_value_and_not_as_a_traceback(self):
        error = OSError("SSL: CERTIFICATE_VERIFY_FAILED")

        def work() -> None:
            raise error

        failure = tiers.watch_download(work, lambda _done: None,
                                       measure=lambda: 0, tick=0.01)
        self.assertIs(failure, error)

    def test_the_report_never_goes_backwards_when_a_cache_shrinks(self):
        """Another program clearing its own model out of the shared cache must not
        produce a negative "downloaded so far"."""
        arrived = iter([500_000_000, 0, 0])
        seen: list[int] = []
        with mock.patch.object(tiers.threading, "Thread", _StepThread):
            tiers.watch_download(lambda: None, seen.append,
                                 measure=lambda: next(arrived), tick=0.0)
        self.assertEqual(seen, [0])

    def test_the_wizards_console_says_how_much_of_how_many(self):
        said: list[str] = []
        arrived = iter([0, 400 * 1_000_000, 900 * 1_000_000])
        with mock.patch.object(tiers.threading, "Thread", _StepThread):
            ok = wizard.download_weights(wizard.TIERS_BY_KEY["vision"], "en",
                                         say=said.append,
                                         fetch=lambda tier, config_path: None,
                                         measure=lambda: next(arrived), tick=0.0)
        self.assertTrue(ok)
        self.assertIn(i18n.cli_text("cli.setup.weights_progress", "en",
                                    done=wizard.human_size(400, "en"),
                                    size=wizard.human_size(1600, "en")), said)


class TestTheRunScreenShowsItMoving(ProcessTestBase):
    """§2: the model was named and then nothing changed for twenty minutes."""

    def test_the_status_carries_what_has_arrived(self):
        state = ui.process._ProcessState()
        state.set_download("classify", ("ViT-L-14",), 512 * 1_000_000)
        download = state.snapshot()["download"]
        self.assertEqual(download["mb"], 1600)
        self.assertEqual(download["done_mb"], 512)

    def test_the_line_goes_away_when_the_download_is_over(self):
        state = ui.process._ProcessState()
        state.set_download("classify", ("ViT-L-14",), 512 * 1_000_000)
        state.set_download(None)
        self.assertIsNone(state.snapshot()["download"])

    def test_the_factory_reports_the_progress_while_it_builds(self):
        """The number reaches the screen from the same place the sentence does."""
        seen: list[tuple] = []
        steps = dict(ui.process._pipeline_steps(
            lambda stage, weights=(), done=0: seen.append((stage, weights, done))))
        self.patch_fast_stages()
        self._patch("detect_landmarks",
                    lambda cfg, conn, classifier=None, progress=None:
                    classifier(["a.jpg"], ["a photo"]))
        self._patch("clip_classifier", lambda settings: (lambda paths, prompts: []))
        self._patch("stage_downloads", lambda stage, states=None: ("ViT-L-14",))
        with mock.patch.object(tiers.threading, "Thread", _StepThread):
            with mock.patch.object(tiers, "downloaded_bytes",
                                   side_effect=[0, 250_000_000]):
                steps["landmarks"](self.cfg, self.conn, lambda *_a, **_k: None)
        self.assertEqual(seen[0], ("landmarks", ("ViT-L-14",), 0))
        self.assertEqual(seen[1], ("landmarks", ("ViT-L-14",), 250_000_000))
        self.assertEqual(seen[-1], (None, (), 0))

    def test_a_refusal_is_still_the_sentence_naming_the_stage_and_the_model(self):
        def refuse(_settings):
            raise OSError("SSL: CERTIFICATE_VERIFY_FAILED")

        steps = dict(ui.process._pipeline_steps(lambda *_a, **_k: None))
        self.patch_fast_stages()
        self._patch("detect_landmarks",
                    lambda cfg, conn, classifier=None, progress=None:
                    classifier(["a.jpg"], ["a photo"]))
        self._patch("clip_classifier", refuse)
        self._patch("stage_downloads", lambda stage, states=None: ("ViT-L-14",))
        with self.assertRaises(ui.process._DownloadRefused) as raised:
            steps["landmarks"](self.cfg, self.conn, lambda *_a, **_k: None)
        message = str(raised.exception)
        self.assertIn("ViT-L-14", message)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", message)

    def test_a_run_from_a_terminal_says_it_too(self):
        """The third screen a download can happen on: `sorta run` in a console."""
        printed: list[str] = []
        arrived = iter([0, 800_000_000])
        cfg = ui.process.Config()
        with mock.patch.object(cli, "clip_classifier", lambda settings: None):
            with mock.patch.object(cli.tiers, "stage_downloads",
                                   return_value=("ViT-L-14",)):
                with mock.patch.object(tiers.threading, "Thread", _StepThread):
                    with mock.patch.object(tiers, "downloaded_bytes",
                                           side_effect=list(arrived)):
                        with mock.patch("builtins.print", side_effect=printed.append):
                            cli._build_clip(cfg, "classify")
        self.assertIn(tiers.download_progress(("ViT-L-14",), 800_000_000, "en"), printed)

    def test_the_console_progress_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = i18n.cli_text("cli.download.progress", lang, done="1", size="2")
                self.assertIn("1", text)
                self.assertIn("2", text)

    def test_the_page_says_how_much_of_it_has_arrived(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn("I18N.download_progress", source)
        self.assertIn("info.done_mb", source)

    def test_the_sentence_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = ui_strings._UI_STRINGS["download_progress"][lang]
                self.assertIn("{done}", text)
                self.assertIn("{size}", text)


# --- defect 3: the wizard downloads what it was told to download ----------------------


class TestEveryTierThatWasSaidYesToIsFetched(unittest.TestCase):
    """"I choose what to download — and where is the download itself?" (2026-08-08)."""

    def _wizard(self, chosen, **kwargs):
        said: list[str] = []
        fetched: list[str] = []

        def download(tier, lang, config_path="config.yaml", *, say) -> bool:
            fetched.append(tier.key)
            return kwargs.get("download_ok", True)

        states = [tiers.TierState(tier.key, missing_weights=tier.weights)
                  for tier in wizard.TIERS]
        code = wizard.run_setup("en", manifest={}, chosen=chosen, states=states,
                                say=said.append, ask=lambda q, d=False: False,
                                doctor=lambda path, states=None: None,
                                install=lambda command: 0, download=download)
        return code, fetched, "\n".join(said)

    def test_a_weights_only_tier_is_downloaded_at_the_screen(self):
        code, fetched, _said = self._wizard(("faces",))
        self.assertEqual(code, 0)
        self.assertEqual(fetched, ["faces"])

    def test_every_optional_tier_of_the_catalog_would_be(self):
        """The narrow version of this — one tier — is the defect itself."""
        for tier in wizard.OPTIONAL_TIERS:
            if not tier.weights:
                continue
            with self.subTest(tier=tier.key):
                _code, fetched, _said = self._wizard((tier.key,))
                self.assertIn(tier.key, fetched)

    def test_nothing_promises_a_download_for_later_any_more(self):
        """The sentence three tiers of four used to print instead of downloading."""
        self.assertNotIn("cli.setup.weights_later", i18n._CLI_STRINGS)
        _code, _fetched, said = self._wizard(("faces",))
        self.assertIn(i18n.cli_text("cli.setup.added", "en",
                                    names=wizard.TIERS_BY_KEY["faces"].name("en")), said)

    def test_a_network_that_refuses_leaves_the_install_whole(self):
        code, fetched, said = self._wizard(("faces",), download_ok=False)
        faces = wizard.TIERS_BY_KEY["faces"].name("en")
        self.assertEqual(code, 0)          # a refused download is not a failed install
        self.assertEqual(fetched, ["faces"])
        self.assertNotIn(i18n.cli_text("cli.setup.added", "en", names=faces), said)
        skipped = [line for line in said.splitlines()
                   if line.startswith(i18n.cli_text("cli.setup.skipped", "en",
                                                    names="").rstrip())]
        self.assertTrue(skipped and faces in skipped[0], said)

    def test_the_longest_download_of_the_catalog_is_announced_before_it_starts(self):
        """7.0 GB is half an hour of a quiet window, and that is read as a hang unless
        somebody has been told what to expect."""
        said: list[str] = []
        deep = wizard.TIERS_BY_KEY["deep"]
        wizard.download_weights(deep, "en", say=said.append,
                                fetch=lambda tier, config_path: None,
                                measure=lambda: 0, tick=0.01)
        self.assertIn(i18n.cli_text("cli.setup.weights_slow", "en",
                                    size=wizard.human_size(deep.download_mb, "en")),
                      said)

    def test_an_ordinary_download_is_not_announced_as_a_long_one(self):
        said: list[str] = []
        wizard.download_weights(wizard.TIERS_BY_KEY["faces"], "en", say=said.append,
                                fetch=lambda tier, config_path: None,
                                measure=lambda: 0, tick=0.01)
        self.assertNotIn(i18n.cli_text("cli.setup.weights_slow", "en",
                                       size=wizard.human_size(400, "en")), said)

    def test_the_long_line_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = i18n.cli_text("cli.setup.weights_slow", lang, size="7.0 GB")
                self.assertIn("7.0 GB", text)

    def test_the_model_names_come_from_the_config_and_not_from_this_code(self):
        """The same rule `clip_weight_names` follows: a machine whose owner chose another
        model must not have a different one downloaded into its cache."""
        from sorta.config import Config, FeaturesConfig, VlmConfig

        configured = Config(features=FeaturesConfig(search_model="xlm-tiny/laion-mini"),
                            vlm=VlmConfig(model="Qwen/Qwen2.5-VL-72B-Instruct"))
        with mock.patch("sorta.config.load_config", return_value=configured):
            self.assertEqual(wizard.search_weight_names("config.yaml"),
                             ("xlm-tiny", "laion-mini"))
            self.assertEqual(wizard.vlm_model_name("config.yaml"),
                             "Qwen/Qwen2.5-VL-72B-Instruct")

    def test_a_machine_with_no_config_yet_gets_the_defaults(self):
        self.assertEqual(wizard.search_weight_names("nowhere/config.yaml"),
                         ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"))
        self.assertEqual(wizard.vlm_model_name("nowhere/config.yaml"),
                         "Qwen/Qwen2.5-VL-3B-Instruct")


# --- defect 4: a directory is not a downloaded model ----------------------------------


class TestAnInterruptedDownloadIsNotADownload(unittest.TestCase):
    """The hypothesis of the brief, reproduced first and only then fixed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.insightface = self.root / "insightface"
        self.hub = self.root / "hub"
        self.insightface.mkdir()
        self.hub.mkdir()
        self.addCleanup(self.tmp.cleanup)

    # The cache exactly as the crash of 2026-08-08 left it: the directory created, the
    # revision registered, the snapshot empty and 1.2 GB sitting in a `.incomplete` blob.
    def _aborted_clip(self) -> Path:
        entry = self.hub / "models--timm--vit_large_patch14_clip_224.openai"
        (entry / "snapshots" / "0123456789abcdef").mkdir(parents=True)
        (entry / "blobs").mkdir()
        (entry / "blobs" / "a6344aac8c09.incomplete").write_bytes(b"x" * 4096)
        return entry

    def _finished_clip(self) -> Path:
        entry = self.hub / "models--timm--vit_large_patch14_clip_224.openai"
        snapshot = entry / "snapshots" / "0123456789abcdef"
        snapshot.mkdir(parents=True)
        (entry / "blobs").mkdir(exist_ok=True)
        (entry / "blobs" / "a6344aac8c09").write_bytes(b"x" * 4096)
        (snapshot / "open_clip_pytorch_model.bin").write_bytes(b"x" * 4096)
        return entry

    def _cached(self, name: str) -> bool:
        return tiers._weights_cached(name, insightface=self.insightface, hub=self.hub)

    def test_the_wreck_of_a_download_reads_as_nothing_downloaded(self):
        self._aborted_clip()
        self.assertFalse(self._cached("ViT-L-14"))

    def test_a_finished_one_reads_as_downloaded(self):
        self._finished_clip()
        self.assertTrue(self._cached("ViT-L-14"))

    def test_a_revision_with_an_empty_snapshot_is_not_a_model_either(self):
        """The half of the wreck that carries no `.incomplete` at all: the directory and
        the revision are there, and not one byte of the weights is."""
        entry = self.hub / "models--timm--vit_large_patch14_clip_224.openai"
        (entry / "snapshots" / "0123456789abcdef").mkdir(parents=True)
        self.assertFalse(self._cached("ViT-L-14"))

    def test_a_partly_written_snapshot_is_not_a_model_either(self):
        """The config lands first and weighs a kilobyte; the 1.6 GB beside it is still
        arriving. A rule that looked only for "some file" would call that downloaded."""
        entry = self._finished_clip()
        (entry / "blobs" / "b1122334455.incomplete").write_bytes(b"x" * 4096)
        self.assertFalse(self._cached("ViT-L-14"))

    def test_the_insightface_side_answers_the_same_way(self):
        (self.insightface / "buffalo_l").mkdir()
        self.assertFalse(self._cached("buffalo_l"))
        (self.insightface / "buffalo_l" / "det_10g.onnx.part").write_bytes(b"x")
        self.assertFalse(self._cached("buffalo_l"))
        (self.insightface / "buffalo_l" / "det_10g.onnx").write_bytes(b"x")
        self.assertFalse(self._cached("buffalo_l"))  # the part file is still there
        (self.insightface / "buffalo_l" / "det_10g.onnx.part").unlink()
        self.assertTrue(self._cached("buffalo_l"))

    def test_the_uninstaller_reads_the_same_rule(self):
        """`weights.downloaded()` shares the marker table with the probe, so it has to
        share the presence rule too — otherwise `sorta cache` reports a model this
        machine has and `sorta doctor`, one command later, reports it as missing."""
        self._aborted_clip()
        found = weights.downloaded(insightface=self.insightface, hub=self.hub)
        entry = [item for item in found if item.weight == "ViT-L-14"]
        self.assertEqual(len(entry), 1)
        # Listed — the bytes are real and somebody asked to get them back...
        self.assertGreater(entry[0].size, 0)
        # ...and not called a model this machine has.
        self.assertFalse(entry[0].complete)
        self.assertFalse(self._cached("ViT-L-14"))

    def test_a_finished_download_is_complete_on_both_sides(self):
        self._finished_clip()
        found = weights.downloaded(insightface=self.insightface, hub=self.hub)
        entry = [item for item in found if item.weight == "ViT-L-14"]
        self.assertTrue(entry[0].complete)
        self.assertTrue(self._cached("ViT-L-14"))


class TestTheDoctorAndTheWizardCannotDisagree(unittest.TestCase):
    """Requirement 3 of defect 4: both sides, one state of the disk, one answer.

    The owner's output — ONE run of `python -m sorta.wizard --tiers faces` — said

        doctor:  ... the models (ViT-L-14, 1.6 GB) download on the first run of the stage
        wizard:  ... already in place — nothing to download

    about the same tier, four lines apart. Two probes of a disk that anything may be
    writing to can answer twice; there is one probe now, and it is passed to both.
    """

    def _both(self, states):
        said: list[str] = []
        doctor_lines: list[str] = []

        def doctor(config_path, probed=None) -> None:
            doctor_lines.extend(cli._doctor_tier_lines("en", list(probed)))

        wizard.run_setup("en", manifest={}, chosen=(), states=states,
                         say=said.append, ask=lambda q, d=False: False, doctor=doctor,
                         install=lambda command: 0,
                         download=lambda *a, **k: True)
        return "\n".join(doctor_lines), "\n".join(said)

    def test_one_state_of_the_disk_produces_one_answer(self):
        vision = wizard.TIERS_BY_KEY["vision"]
        missing = [tiers.TierState(tier.key, missing_weights=tier.weights)
                   for tier in wizard.TIERS]
        doctor_said, wizard_said = self._both(missing)
        self.assertIn(i18n.cli_text("cli.doctor.tier_weights", "en",
                                    name=vision.name("en"), weights="ViT-L-14",
                                    size=wizard.human_size(1600, "en")), doctor_said)
        self.assertNotIn(i18n.cli_text("cli.setup.in_place", "en",
                                       name=vision.name("en")), wizard_said)

    def test_and_the_other_state_produces_the_other_one(self):
        vision = wizard.TIERS_BY_KEY["vision"]
        present = [tiers.TierState(tier.key) for tier in wizard.TIERS]
        doctor_said, wizard_said = self._both(present)
        self.assertIn(i18n.cli_text("cli.doctor.tier_ready", "en",
                                    name=vision.name("en")), doctor_said)
        self.assertIn(i18n.cli_text("cli.setup.in_place", "en",
                                    name=vision.name("en")), wizard_said)

    def test_the_disk_is_read_once_and_shown_to_both(self):
        probes: list[str] = []
        seen: list[object] = []

        def probe():
            probes.append("probe")
            return [tiers.TierState(tier.key) for tier in wizard.TIERS]

        with mock.patch.object(wizard, "probe_tiers", side_effect=probe):
            wizard.run_setup("en", manifest={}, chosen=(), say=lambda text: None,
                             ask=lambda q, d=False: False,
                             doctor=lambda path, states=None: seen.append(states),
                             install=lambda command: 0,
                             download=lambda *a, **k: True)
        self.assertEqual(len(probes), 1)
        self.assertEqual([state.key for state in seen[0]],
                         [tier.key for tier in wizard.TIERS])

    def test_the_doctor_command_still_probes_for_itself_when_nobody_hands_it_one(self):
        with mock.patch.object(cli, "tier_states", return_value=[]) as probe:
            with mock.patch("builtins.print"):
                with mock.patch.object(cli, "gpu_health"), \
                        mock.patch.object(cli, "geo_data_health"):
                    cli._cmd_doctor("config.yaml")
        probe.assert_called_once_with()


class _StepThread:
    """A thread that is alive for exactly one report, then done.

    The real one is `threading.Thread`; what a test needs is a deterministic number of
    turns through the loop, not a race with a sleep.
    """

    def __init__(self, target=None, daemon=None) -> None:
        self._target = target
        self._turns = 0

    def start(self) -> None:
        if self._target is not None:
            self._target()

    def join(self, timeout=None) -> None:
        self._turns += 1

    def is_alive(self) -> bool:
        return self._turns < 2


if __name__ == "__main__":
    unittest.main()
