"""F211: the first-run wizard — the tiers offered honestly, and a no that is a real no.

What the brief asks to pin, in its own order:

* the wizard CALLS `sorta doctor` instead of growing a second check screen — proven by
  substitution, which is also the only way to prove it;
* refusing every tier leaves a WORKING product: no install command runs, the exit code is
  zero, and the person is told so in words;
* accepting one builds the install command out of the package metadata (so the versions
  are `pyproject.toml`'s and nobody's copy of them), and the CUDA profile keeps its own
  index;
* the exiftool situation is SAID — bundled, found on PATH, or absent with what that costs;
* an install that fails leaves the program as it was and reports which tier failed;
* every string comes from the i18n catalog, in all three languages.

Nothing here installs anything: `install` is injected, and what is checked is the command
that would run.
"""
from __future__ import annotations

import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import i18n, install, tiers, wizard

_LANGS = ("ru", "en", "ja")


def nothing_cached() -> list[tiers.TierState]:
    """A machine with none of it: every package and every weight still missing.

    Passed to every `run_setup` below on purpose. Without it the wizard probes THIS
    machine (F223 — a tier already on the disk is not offered again), and a suite whose
    answers depend on what the developer happens to have cached is a suite that says
    different things on two machines.
    """
    return [tiers.TierState(tier.key,
                            missing_packages=tuple(f"extra:{e}" for e in tier.extras),
                            missing_weights=tier.weights)
            for tier in wizard.TIERS]


class Screen:
    """The console the wizard talks to: what it said, and what it was answered."""

    def __init__(self, answers: bool | None = False) -> None:
        self.lines: list[str] = []
        self.questions: list[str] = []
        # None — answer whatever the tier's own default is, i.e. press Enter every time.
        self.answers = answers
        self.commands: list[list[str]] = []
        self.doctor_calls: list[str] = []
        self.doctor_states: list[object] = []
        self.install_code = 0
        self.downloaded: list[str] = []
        self.download_ok = True

    def say(self, text: str) -> None:
        self.lines.append(text)

    def ask(self, question: str, default: bool = False) -> bool:
        self.questions.append(question)
        return default if self.answers is None else self.answers

    def doctor(self, config_path: str, states=None) -> None:
        # F225: the wizard hands over the probe it has already taken — one reading of the
        # disk for both screens of one window.
        self.doctor_calls.append(config_path)
        self.doctor_states.append(states)

    def install(self, command) -> int:
        self.commands.append(list(command))
        return self.install_code

    def download(self, tier, lang, config_path="config.yaml", *, say) -> bool:
        self.downloaded.append(tier.key)
        return self.download_ok

    @property
    def said(self) -> str:
        return "\n".join(self.lines)


def _requirements(*names: str):
    """Stand in for the installed metadata — the tests are about the command, not pip."""
    return mock.patch.object(wizard, "tier_requirements", lambda tier: tuple(names))


class TestTheCheckScreenIsDoctor(unittest.TestCase):
    """Requirement 3 of the brief: the wizard asks `sorta doctor`, it does not re-answer."""

    def test_the_wizard_calls_the_doctor_command_itself(self):
        with mock.patch("sorta.cli._cmd_doctor") as doctor:
            wizard.show_doctor("some/config.yaml")
        doctor.assert_called_once_with("some/config.yaml", states=None)

    def test_the_real_doctor_is_the_default_the_wizard_runs(self):
        """The tests inject a stand-in; the program must not."""
        self.assertIs(inspect.signature(wizard.run_setup).parameters["doctor"].default,
                      wizard.show_doctor)

    def test_the_check_screen_runs_before_a_single_question(self):
        screen = Screen()
        wizard.run_setup("en", manifest={}, chosen=(), states=nothing_cached(),
                         say=screen.say, ask=screen.ask, doctor=screen.doctor,
                         install=screen.install, download=screen.download)
        self.assertEqual(screen.doctor_calls, ["config.yaml"])


class TestRefusingEverything(unittest.TestCase):
    """A no to all of it is a normal path, not a dead end."""

    def setUp(self):
        self.screen = Screen(answers=False)
        self.code = wizard.run_setup("en", manifest={"exiftool": "exiftool.exe"},
                                     states=nothing_cached(),
                                     say=self.screen.say, ask=self.screen.ask,
                                     doctor=self.screen.doctor,
                                     install=self.screen.install,
                                     download=self.screen.download)

    def test_every_optional_tier_was_offered_once(self):
        self.assertEqual(len(self.screen.questions), len(wizard.OPTIONAL_TIERS))
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertIn(tier.name("en"), self.screen.said)
                self.assertIn(tier.benefit("en"), self.screen.said)

    def test_nothing_was_installed_and_the_exit_code_is_zero(self):
        self.assertEqual(self.screen.commands, [])
        self.assertEqual(self.code, 0)

    def test_the_person_is_told_the_product_works(self):
        self.assertIn(i18n.cli_text("cli.setup.works_anyway", "en"), self.screen.said)
        self.assertIn(i18n.cli_text(wizard.rerun_key(), "en"), self.screen.said)

    def test_each_refusal_says_what_stays_unavailable(self):
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertIn(tier.without("en"), self.screen.said)

    def test_a_line_a_legacy_console_cannot_encode_still_appears(self):
        """The catalog has em dashes and Japanese; a cp866 console has neither. Losing
        the sentence about exiftool to an encoding is not an option, so it degrades to
        replacement characters instead of to silence."""
        printed: list[str] = []

        def print_once(text: str) -> None:
            if not printed:
                printed.append(text)
                raise UnicodeEncodeError("cp866", "—", 0, 1, "no room")
            printed.append(text)

        with mock.patch("builtins.print", side_effect=print_once):
            with mock.patch.object(wizard.sys, "stdout", mock.Mock(encoding="cp866")):
                wizard.say_console("dates — from HEIC")
        self.assertEqual(len(printed), 2)
        self.assertIn("dates", printed[1])

    def test_a_stream_that_is_not_there_at_all_is_survived(self):
        with mock.patch("builtins.print", side_effect=OSError("closed")):
            wizard.say_console("anything")

    def test_a_question_nobody_answers_is_a_no(self):
        """Enter, EOF, a closed stdin: the wizard may not start a 7 GB download on any
        of them."""
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(wizard.ask_console("Add it?"))
        with mock.patch("builtins.input", return_value=""):
            self.assertFalse(wizard.ask_console("Add it?"))
        for yes in ("y", "Yes", "да", "はい"):
            with self.subTest(answer=yes):
                with mock.patch("builtins.input", return_value=yes):
                    self.assertTrue(wizard.ask_console("Add it?"))


class TestAcceptingATier(unittest.TestCase):
    """What a yes actually does — one uv command, built from the package metadata."""

    def test_the_deep_tier_installs_its_extra_and_names_the_weights(self):
        screen = Screen()
        with _requirements("transformers>=4.49,<4.52", "accelerate>=0.34"):
            code = wizard.run_setup("en", manifest={"uv": "C:/app/uv.exe",
                                                    "python": "C:/app/env/python.exe"},
                                    chosen=("deep",), states=nothing_cached(),
                                    say=screen.say, ask=screen.ask,
                                    doctor=screen.doctor, install=screen.install,
                                    download=screen.download)
        self.assertEqual(code, 0)
        self.assertEqual(screen.commands, [[
            "C:/app/uv.exe", "pip", "install", "--python", "C:/app/env/python.exe",
            "transformers>=4.49,<4.52", "accelerate>=0.34"]])
        # F225: the 7 GB of model weights are fetched HERE, at the screen. They used to be
        # announced as arriving "on the first run of the stage", which is what the owner
        # met as "I choose what to download — and where is the download?".
        self.assertEqual(screen.downloaded, ["deep"])
        self.assertIn(i18n.cli_text("cli.setup.added", "en",
                                    names=wizard.TIERS_BY_KEY["deep"].name("en")),
                      screen.said)

    def test_a_weights_only_tier_installs_no_package_and_says_so(self):
        """`faces` and `search` cost weights, not packages: onnxruntime and the CLIP
        code are already in the base tier. Accepting one must not pretend to install
        anything."""
        screen = Screen()
        wizard.run_setup("en", manifest={}, chosen=("faces",),
                         states=nothing_cached(), say=screen.say, ask=screen.ask,
                         doctor=screen.doctor, install=screen.install,
                         download=screen.download)
        self.assertEqual(screen.commands, [])
        # ...and what it costs is downloaded on the spot instead of being promised (F225).
        self.assertEqual(screen.downloaded, ["faces"])

    def test_the_cuda_profile_keeps_its_own_package_index(self):
        tier = wizard.TIERS_BY_KEY["gpu"]
        command = wizard.install_command(tier, ["torch>=2.10.0"], uv="uv",
                                         python="python.exe")
        self.assertEqual(command[-2:], ["--index", wizard.PYTORCH_CU130_INDEX])

    def test_the_index_is_the_one_the_project_resolves_with(self):
        """A second URL written here would resolve a different torch than `uv sync`
        does — the boundary the brief draws around not building a second mechanism."""
        pyproject = (Path(wizard.__file__).resolve().parent.parent
                     / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'url = "{wizard.PYTORCH_CU130_INDEX}"', pyproject)

    def test_the_cuda_profile_replaces_the_cpu_one_instead_of_adding_to_it(self):
        """Without `--reinstall` this tier is a no-op: `torch>=2.10.0` is already
        satisfied by the CPU wheel the installer carried, so nothing would be fetched and
        the card would stay idle with the wizard reporting success."""
        command = wizard.install_command(wizard.TIERS_BY_KEY["gpu"], ["torch>=2.10.0"],
                                         uv="uv", python="python.exe")
        self.assertIn("--reinstall", command)
        self.assertLess(command.index("--reinstall"), command.index("torch>=2.10.0"))

    def test_a_tier_that_only_adds_packages_is_not_reinstalled(self):
        self.assertFalse(wizard.TIERS_BY_KEY["deep"].reinstall)
        self.assertNotIn("--reinstall",
                         wizard.install_command(wizard.TIERS_BY_KEY["deep"],
                                                ["transformers"], uv="uv",
                                                python="python.exe"))

    def test_a_tier_without_an_index_gets_no_flag(self):
        command = wizard.install_command(wizard.TIERS_BY_KEY["deep"], ["transformers"],
                                         uv="uv", python="python.exe")
        self.assertNotIn("--index", command)


class TestWhenAnInstallFails(unittest.TestCase):
    """The program is left exactly as it was, and the failure is named."""

    def test_a_failed_tier_is_reported_and_the_exit_code_is_not_zero(self):
        screen = Screen()
        screen.install_code = 3
        with _requirements("transformers>=4.49,<4.52"):
            code = wizard.run_setup("en", manifest={}, chosen=("deep",),
                                    states=nothing_cached(), say=screen.say,
                                    ask=screen.ask, doctor=screen.doctor,
                                    install=screen.install, download=screen.download)
        self.assertEqual(code, 1)
        deep = wizard.TIERS_BY_KEY["deep"].name("en")
        self.assertIn(i18n.cli_text("cli.setup.install_failed", "en", name=deep,
                                    status=3), screen.said)
        self.assertNotIn(i18n.cli_text("cli.setup.added", "en", names=deep), screen.said)

    def test_an_extra_with_no_metadata_behind_it_is_not_silently_skipped(self):
        screen = Screen()
        with _requirements():
            code = wizard.run_setup("en", manifest={}, chosen=("deep",),
                                    states=nothing_cached(), say=screen.say,
                                    ask=screen.ask, doctor=screen.doctor,
                                    install=screen.install, download=screen.download)
        self.assertEqual(code, 1)
        self.assertEqual(screen.commands, [])
        self.assertIn(i18n.cli_text("cli.setup.no_metadata", "en",
                                    name=wizard.TIERS_BY_KEY["deep"].name("en")),
                      screen.said)

    # F228: the install runs through `sorta.launch`, which is the only place in the
    # package that touches `subprocess` — so that a wizard started from the shortcut opens
    # no console window. What is hidden and when is tested in
    # tests/test_no_console_nobody_asked_for.py; these two are about the exit code.
    def test_a_command_that_cannot_be_run_is_an_exit_code_and_not_a_traceback(self):
        with mock.patch.object(wizard.launch, "run", side_effect=OSError("no uv")):
            self.assertEqual(wizard.run_install(["uv", "pip", "install", "x"]), 1)

    def test_a_command_that_runs_returns_its_own_code(self):
        completed = mock.Mock(returncode=7)
        with mock.patch.object(wizard.launch, "run", return_value=completed):
            self.assertEqual(wizard.run_install(["uv", "pip", "install", "x"]), 7)


class TestWhatIsSaidAboutExiftool(unittest.TestCase):
    """The brief's condition on the packaging decision: whatever was chosen, say it.

    The installer bundles the binary (see packaging/windows/README.md), so the first
    answer is the shipped one — but a Sorta started from a checkout has to get the truth
    about its own machine too, and the fallback sentence has to name what it costs.
    """

    def test_the_bundled_binary_is_reported_as_bundled(self):
        self.assertEqual(wizard.exiftool_state({"exiftool": "exiftool.exe"}),
                         wizard.EXIFTOOL_BUNDLED)

    def test_a_binary_on_path_is_recognised(self):
        self.assertEqual(wizard.exiftool_state({}, which=lambda name: "C:/bin/exiftool"),
                         wizard.EXIFTOOL_ON_PATH)

    def test_no_binary_at_all_is_said_out_loud_with_the_formats_it_costs(self):
        self.assertEqual(wizard.exiftool_state({}, which=lambda name: None),
                         wizard.EXIFTOOL_ABSENT)
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = i18n.cli_text("cli.setup.exiftool_absent", lang)
                for token in ("exiftool", "HEIC", "RAW", "Pillow"):
                    self.assertIn(token, text)

    def test_the_sentence_reaches_the_screen(self):
        screen = Screen()
        with mock.patch.object(wizard.shutil, "which", return_value=None):
            wizard.run_setup("en", manifest={}, chosen=(), states=nothing_cached(),
                             say=screen.say, ask=screen.ask, doctor=screen.doctor,
                             install=screen.install, download=screen.download)
        self.assertIn(i18n.cli_text("cli.setup.exiftool_absent", "en"), screen.said)


class TestTheManifestTheInstallerLeft(unittest.TestCase):
    """Where the wizard learns what was shipped — and what happens without one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write(self, payload: str) -> Path:
        path = self.root / wizard.MANIFEST_NAME
        path.write_text(payload, encoding="utf-8")
        return path

    def test_an_explicit_path_is_read(self):
        path = self._write(json.dumps({"uv": "C:/app/uv.exe"}))
        self.assertEqual(wizard.load_manifest(path)["uv"], "C:/app/uv.exe")

    def test_the_environment_names_it_for_the_wizard_the_installer_starts(self):
        path = self._write(json.dumps({"exiftool": "exiftool.exe"}))
        with mock.patch.dict("os.environ", {wizard.ENV_MANIFEST: str(path)}):
            manifest = wizard.load_manifest()
        self.assertEqual(manifest["exiftool"], "exiftool.exe")
        # Where it was read from, so the relative paths in it can be resolved.
        self.assertEqual(manifest[wizard.MANIFEST_ROOT], str(self.root))

    def test_a_broken_manifest_is_an_empty_one_and_not_a_crash(self):
        path = self._write("{not json")
        self.assertEqual(wizard.load_manifest(path), {})

    def test_a_missing_manifest_is_an_empty_one(self):
        self.assertEqual(wizard.load_manifest(self.root / "nowhere.json"), {})
        with mock.patch.dict("os.environ", {wizard.ENV_MANIFEST: "C:/nowhere.json"}):
            self.assertIsNone(wizard.manifest_path())

    def test_without_a_manifest_the_tools_of_this_machine_are_used(self):
        """A checkout is a legitimate place to run the wizard from: it then talks about
        the interpreter it is running in, not about an install that does not exist."""
        with mock.patch.object(wizard.shutil, "which", return_value="C:/tools/uv.exe"):
            self.assertEqual(wizard.uv_binary({}), "C:/tools/uv.exe")
        self.assertEqual(wizard.python_binary({}), wizard.sys.executable)
        self.assertEqual(wizard.uv_binary({"uv": "C:/app/uv.exe"}), "C:/app/uv.exe")


class TestTheEntryPoint(unittest.TestCase):
    """`sorta-setup` itself: the flags, and the language it answers in."""

    def test_the_entry_point_is_declared_in_the_project(self):
        pyproject = (Path(wizard.__file__).resolve().parent.parent
                     / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('sorta-setup = "sorta.wizard:main"', pyproject)

    def test_none_asks_nothing_and_installs_nothing(self):
        with mock.patch("sorta.cli._cmd_doctor") as doctor:
            with mock.patch("builtins.input", side_effect=AssertionError("asked")):
                with mock.patch.object(wizard, "run_install") as install:
                    code = wizard.main(["--tiers", "none", "--lang", "ja"])
        self.assertEqual(code, 0)
        doctor.assert_called_once()
        install.assert_not_called()

    def test_all_selects_every_optional_tier(self):
        self.assertEqual(wizard.selected_tiers("all"),
                         tuple(tier.key for tier in wizard.OPTIONAL_TIERS))
        self.assertEqual(wizard.selected_tiers("none"), ())
        self.assertEqual(wizard.selected_tiers("faces, deep"), ("faces", "deep"))
        self.assertIsNone(wizard.selected_tiers(None))

    def test_a_tier_that_does_not_exist_is_an_error_and_not_a_silent_skip(self):
        with self.assertRaises(ValueError):
            wizard.selected_tiers("faces,telepathy")
        with mock.patch("sorta.cli._cmd_doctor"):
            self.assertEqual(wizard.main(["--tiers", "telepathy"]), 2)

    def test_the_language_comes_from_the_config_and_the_flag_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("language: ja\n", encoding="utf-8")
            self.assertEqual(wizard.language(str(path)), "ja")
            self.assertEqual(wizard.language(str(path), "ru"), "ru")
        # A fresh install may have no config.yaml yet — the wizard still speaks.
        self.assertEqual(wizard.language("nowhere/config.yaml"), "en")


class TestTheTierCatalogReadsAsWords(unittest.TestCase):
    """Every tier is described in three languages, and the sizes are readable."""

    def test_every_tier_has_a_name_and_a_benefit_in_all_three_languages(self):
        for tier in wizard.TIERS:
            for suffix in ("name", "benefit"):
                key = f"cli.setup.tier.{tier.key}.{suffix}"
                with self.subTest(key=key):
                    entry = i18n._CLI_STRINGS[key]
                    self.assertEqual(set(entry), set(_LANGS))
                    self.assertEqual(len(set(entry.values())), 3, key)

    def test_every_optional_tier_says_what_a_no_costs(self):
        for tier in wizard.OPTIONAL_TIERS:
            key = f"cli.setup.tier.{tier.key}.without"
            with self.subTest(key=key):
                entry = i18n._CLI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang in _LANGS:
                    self.assertTrue(entry[lang].strip())

    def test_sizes_are_printed_the_way_a_person_reads_them(self):
        self.assertEqual(wizard.human_size(400, "en"), "400 MB")
        self.assertEqual(wizard.human_size(3000, "en"), "3.0 GB")
        self.assertEqual(wizard.human_size(7000, "ru"), "7.0 ГБ")

    def test_every_optional_tier_states_a_size(self):
        """A tier offered without its price is the thing this feature exists against."""
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertGreater(tier.download_mb, 0)

    def test_the_catalog_is_asked_for_exactly_the_keys_that_exist(self):
        source = Path(wizard.__file__).read_text(encoding="utf-8")
        prefixes = set(re.findall(r'f"\{_SETUP_PREFIX\}([a-z0-9_.]+)"', source))
        used = {f"cli.setup.{name}" for name in prefixes
                if not name.startswith("tier.") and not name.startswith("exiftool_")}
        used |= {f"cli.setup.exiftool_{state}" for state in
                 (wizard.EXIFTOOL_BUNDLED, wizard.EXIFTOOL_ON_PATH,
                  wizard.EXIFTOOL_ABSENT)}
        self.assertGreaterEqual(len(used), 15)  # the catalog is actually wired up
        for key in sorted(used):
            with self.subTest(key=key):
                # F230: two of these are BASES of an install-kind family (`rerun`,
                # `cpu_back`) — the sentence differs between a checkout and an installed
                # copy, and `install.advice_key` picks. Then all three have to exist.
                if key in install.INSTALL_ADVICE:
                    for variant in install.advice_keys(key):
                        self.assertIn(variant, i18n._CLI_STRINGS, variant)
                    continue
                self.assertIn(key, i18n._CLI_STRINGS, key)


if __name__ == "__main__":
    unittest.main()
