"""F213: on Linux the install IS one line, so `doctor` is the rest of the installer.

The owner closed the fork on 2026-08-06: no AppImage, no deb, no rpm — `uv tool install`
and nothing else. That decision moves the whole burden onto two things, and both are what
these cases pin:

* **`sorta doctor` answering each way that line fails, in words.** A person on a clean
  machine meets `sorta: command not found` (uv writes into `~/.local/bin`, which a
  default shell profile does not read), a collection with no dates (`exiftool` is not a
  python package and no install command brings it), a CPU torch on a GPU box (already
  answered by `gpu_health`) and a preview cache every other local account can read. The
  first, the second and the fourth had nowhere to be said until now.
* **The guides describing that exact path.** Checked against the code and against each
  other rather than by eye, the way `test_docs_guides` already checks the command line:
  a Linux block that drifts in one language, or names a command this project does not
  install, fails here.

Two requirements of the brief are covered by cases that already exist and are not
duplicated here — a machine with no tray keeps serving (`test_tray_icon`,
`TestAMachineWithoutATray`) and the cache is created 0700 (F210,
`test_derivative_does_not_outlive_original`). What is new is that `doctor` now SAYS so
when the second one is not true of a directory that predates that rule.
"""
from __future__ import annotations

import io
import os
import re
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sorta import cli, i18n
from tests.test_docs_guides import GUIDES, read

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent


def _install_lines(lang: i18n.Lang = "en", *, command: str | None = "/usr/bin/sorta",
                   scripts: str = "/home/me/.local/bin",
                   exiftool: str | None = "/usr/bin/exiftool",
                   hint: str | None = None) -> list[str]:
    return cli._doctor_install_lines(lang, command=command, scripts=Path(scripts),
                                     exiftool=exiftool, hint_key=hint)


class TestTheCommandThatIsNotOnPath(unittest.TestCase):
    """`uv tool install` succeeds and `sorta` is still not a word the shell knows.

    This is the first thing that happens on a clean Debian, it happens to everybody, and
    uv's own warning about it scrolls past inside two hundred lines of resolver output.
    """

    def test_a_command_on_path_is_reported_with_where_it_is(self):
        lines = _install_lines()
        self.assertIn("/usr/bin/sorta", lines[0])
        self.assertEqual(len(lines), 2)  # the command and exiftool, no hints

    def test_a_command_that_is_not_on_path_names_the_directory_it_is_in(self):
        lines = _install_lines(command=None)
        self.assertIn(str(Path("/home/me/.local/bin")), lines[0])

    def test_and_the_command_that_puts_it_there(self):
        """A diagnosis without a fix is a diagnosis a person takes to a search engine."""
        self.assertIn("uv tool update-shell", _install_lines(command=None)[1])

    def test_the_directory_named_is_the_one_this_install_writes_into(self):
        """`sysconfig`, not the directory of `sys.executable`: on Windows a system python
        keeps its scripts one level down, and a path in an error message has to be one a
        person can open."""
        import sysconfig

        self.assertEqual(cli._scripts_dir(), Path(sysconfig.get_path("scripts")))


class TestTheMetadataReaderThatIsNotAPythonPackage(unittest.TestCase):
    """No install command brings `exiftool`, and without it a phone collection has no
    dates, no GPS and no orientation — silently, because the reader falls back to Pillow
    and Pillow simply returns nothing for HEIC/RAW/video."""

    def test_a_present_exiftool_is_reported_with_its_path(self):
        self.assertIn("/usr/bin/exiftool", _install_lines()[-1])

    def test_a_missing_exiftool_says_what_stops_working(self):
        line = _install_lines(exiftool=None)[-2]
        self.assertIn("exiftool", line)
        for word in ("HEIC/RAW", "GPS"):
            self.assertIn(word, line)

    def test_a_missing_exiftool_is_followed_by_a_pasteable_install_command(self):
        self.assertIn("apt install libimage-exiftool-perl",
                      _install_lines(exiftool=None, hint="cli.doctor.exiftool_linux")[-1])

    def test_the_install_command_is_the_one_this_platform_has(self):
        """A person on Debian has no use for `winget`, which is why this is a choice and
        not a list of three."""
        for platform, expected in (("linux", "cli.doctor.exiftool_linux"),
                                   ("freebsd14", "cli.doctor.exiftool_linux"),
                                   ("win32", "cli.doctor.exiftool_windows"),
                                   ("darwin", "cli.doctor.exiftool_macos")):
            with self.subTest(platform=platform):
                self.assertEqual(cli._exiftool_hint_key(platform), expected)
        for key, command in (("cli.doctor.exiftool_windows", "winget install"),
                             ("cli.doctor.exiftool_macos", "brew install exiftool")):
            with self.subTest(key=key):
                self.assertIn(command, i18n.cli_text(key, "en"))


class TestEveryLineExistsInThreeLanguages(unittest.TestCase):
    """The parity rule of the catalog, on the states this feature added."""

    def test_both_halves_of_both_answers_are_translated(self):
        for command, exiftool in ((None, None), ("/usr/bin/sorta", "/usr/bin/exiftool")):
            rendered = {lang: _install_lines(lang, command=command, exiftool=exiftool)
                        for lang in _LANGS}
            for index in range(len(rendered["en"])):
                with self.subTest(command=command, line=index):
                    texts = {rendered[lang][index] for lang in _LANGS}
                    self.assertEqual(len(texts), 3, texts)
                    self.assertTrue(all(text.strip() for text in texts))

    def test_the_way_out_of_a_missing_tier_is_the_one_this_machine_has(self):
        """F216 named the Start menu, which exists only where the installer put it."""
        self.assertEqual(cli._tier_hint_key("nt"), "cli.doctor.tier_hint")
        self.assertEqual(cli._tier_hint_key("posix"), "cli.doctor.tier_hint_posix")
        posix = i18n.cli_text("cli.doctor.tier_hint_posix", "en")
        self.assertIn("sorta-setup", posix)
        self.assertNotIn("Start menu", posix)

    def test_the_hint_the_tier_block_prints_follows_this_machine(self):
        states = [cli.TierState("faces", missing_weights=("buffalo_l",))]
        with patch.object(cli, "_tier_hint_key", lambda: "cli.doctor.tier_hint_posix"):
            self.assertEqual(cli._doctor_tier_lines("en", states)[-1],
                             i18n.cli_text("cli.doctor.tier_hint_posix", "en"))


class TestThePreviewCacheIsPrivateToItsOwner(unittest.TestCase):
    """Requirement 3 of the brief. F210 creates the directory 0700 and deliberately does
    not repair one that already exists; a cache made before that rule is still 0755, and
    `doctor` is the only place its owner can be told."""

    def test_a_private_cache_says_nothing(self):
        self.assertEqual(cli._doctor_cache_lines("en", Path("/c"), 0o700), [])

    def test_a_cache_others_can_read_is_a_warning_with_the_fix_in_it(self):
        cache = Path("/home/me/.cache/sorta/previews")
        line = cli._doctor_cache_lines("en", cache, 0o755)[0]
        self.assertIn("755", line)
        self.assertIn("chmod 700", line)
        self.assertIn(str(cache), line)

    def test_a_group_that_can_read_counts_as_others(self):
        """A shared group is how a family machine is set up, so 0750 is not private."""
        for mode in (0o750, 0o705, 0o701, 0o770):
            with self.subTest(mode=oct(mode)):
                self.assertEqual(len(cli._doctor_cache_lines("en", Path("/c"), mode)), 1)

    def test_a_question_that_does_not_apply_is_not_answered(self):
        """Windows (the bits mean nothing to NTFS) and a cache nobody has written yet."""
        self.assertEqual(cli._doctor_cache_lines("en", Path("/c"), None), [])

    def test_a_directory_that_does_not_exist_has_no_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cli._directory_mode(Path(tmp) / "never-created"))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits; NTFS inherits the ACL")
    def test_the_bits_are_read_off_a_real_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "previews"
            directory.mkdir(mode=0o700)
            self.assertEqual(cli._directory_mode(directory), 0o700)
            self.assertEqual(cli._doctor_cache_lines("en", directory,
                                                     cli._directory_mode(directory)), [])
            directory.chmod(0o755)
            self.assertEqual(len(cli._doctor_cache_lines("en", directory,
                                                         cli._directory_mode(directory))),
                             1)

    @unittest.skipIf(os.name != "nt", "the Windows half of the same question")
    def test_on_windows_the_bits_are_not_a_question_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cli._directory_mode(Path(tmp)))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits; NTFS inherits the ACL")
    def test_the_cache_the_product_actually_creates_is_private(self):
        """The pairing: F210 writes the mode, and this is the one place that reads it
        back through the same helper `doctor` uses."""
        from sorta import imaging

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"SORTA_PREVIEW_DIR": str(Path(tmp) / "cache")}):
                directory = imaging.preview_dir()
                imaging._make_preview_dir(directory / "ab")
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(cli._doctor_cache_lines("en", directory,
                                                     cli._directory_mode(directory)), [])


class TestDoctorPrintsAllOfIt(unittest.TestCase):
    """The wiring. Every helper above is pure, so without this case the command could
    print none of them and the suite would stay green."""

    def _doctor(self, **which) -> list[str]:
        health = SimpleNamespace(summary="health", available=True)
        found = {"sorta": "/home/me/.local/bin/sorta", "exiftool": "/usr/bin/exiftool",
                 "uv": "/usr/bin/uv", **which}
        buffer = io.StringIO()
        with patch.object(cli.shutil, "which", lambda name: found.get(name)), \
                patch.object(cli, "gpu_health", lambda: health), \
                patch.object(cli, "geo_data_health", lambda: health), \
                patch.object(cli, "tier_states", lambda: []), \
                patch.object(cli, "default_log_path", lambda: "run.log"), \
                patch.object(cli, "_directory_mode", lambda _path: 0o755), \
                redirect_stdout(buffer):
            cli._cmd_doctor("no-such-config.yaml")
        return buffer.getvalue().splitlines()

    def test_a_healthy_machine_gets_both_statements(self):
        printed = self._doctor()
        self.assertTrue(any("/home/me/.local/bin/sorta" in line for line in printed),
                        printed)
        self.assertTrue(any("/usr/bin/exiftool" in line for line in printed), printed)

    def test_the_environment_is_named_before_the_health_lines(self):
        """F211's rule: a shadowed PATH turns every later line into a statement about
        somebody else's installation, so it is stated first."""
        printed = self._doctor()
        exiftool = next(i for i, line in enumerate(printed) if "exiftool" in line)
        self.assertLess(exiftool, printed.index("health"))

    def test_a_missing_command_and_a_missing_exiftool_both_reach_the_screen(self):
        printed = self._doctor(sorta=None, exiftool=None)
        self.assertTrue(any("uv tool update-shell" in line for line in printed), printed)
        self.assertTrue(any("libimage-exiftool-perl" in line
                            or "winget" in line or "brew" in line
                            for line in printed), printed)

    def test_an_open_cache_is_warned_about_next_to_its_path(self):
        printed = self._doctor()
        self.assertTrue(any("chmod 700" in line for line in printed), printed)
        cache = next(i for i, line in enumerate(printed) if "chmod 700" in line)
        self.assertIn(str(cli.imaging.preview_dir()), printed[cache - 1])


# --- the guides ---------------------------------------------------------------
# The install section is now the whole of the Linux experience, so it is checked the way
# `test_docs_guides` checks the command line: against what the project actually installs,
# and against the other two languages. A block that drifts is a block that lies, and this
# feature's acceptance criterion is that the instruction does not lie in one line.

_FENCE = re.compile(r"(?ms)^```(?:bash|sh)?\n(.*?)^```")
# What makes a fenced block THE Linux install block, in any language: it is the one that
# installs uv the way a Linux machine installs it.
_UV_INSTALLER = "astral.sh/uv/install.sh"


def install_block(text: str) -> list[str]:
    """The command lines of the Linux install block — comments and blanks dropped."""
    for body in _FENCE.findall(text):
        if _UV_INSTALLER not in body:
            continue
        return [line.strip() for line in body.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    return []


class TestTheGuidesDescribeTheInstallThatExists(unittest.TestCase):
    """Requirement 4 of the brief: the section is compared with the commands, not read."""

    def setUp(self):
        self.blocks = {lang: install_block(read(path)) for lang, path in GUIDES.items()}

    def test_every_guide_has_the_linux_block(self):
        for lang, commands in self.blocks.items():
            with self.subTest(lang=lang):
                self.assertTrue(commands, f"{lang}: no Linux install block")

    def test_the_three_languages_install_the_same_way(self):
        """Prose is translated; commands are not, and a `ru` block a session behind the
        `en` one is how a reader ends up with the wrong profile."""
        self.assertEqual(self.blocks["ru"], self.blocks["en"])
        self.assertEqual(self.blocks["ja"], self.blocks["en"])

    def test_the_block_installs_exiftool_uv_and_sorta_and_then_checks_it(self):
        commands = self.blocks["en"]
        self.assertTrue(any("exiftool" in line for line in commands), commands)
        self.assertTrue(any(line.startswith("uv tool install") for line in commands),
                        commands)
        self.assertIn("sorta doctor", commands)

    def test_the_profile_is_chosen_in_the_package_spec(self):
        """The F79 trap, in the one block a Linux reader copies: `uv tool install` has no
        `--extra`, and without the extra inside the spec it quietly builds the CPU
        profile on a machine with a card."""
        installs = [line for line in self.blocks["en"]
                    if line.startswith("uv tool install")]
        self.assertTrue(installs)
        for line in installs:
            with self.subTest(command=line):
                self.assertNotIn("--extra", line)
                self.assertRegex(line, r"\[(cpu|gpu)[\w,]*\]")

    def test_every_sorta_command_of_the_block_is_one_this_project_installs(self):
        """The entry points are read out of `pyproject.toml`: a guide naming
        `sorta-gui` would pass any prose check ever written."""
        pyproject = read(_ROOT / "pyproject.toml")
        entry_points = set(re.findall(r'(?m)^([\w-]+) = "sorta\.', pyproject))
        self.assertIn("sorta", entry_points)  # the file was parsed, not guessed at
        for lang, commands in self.blocks.items():
            for line in commands:
                word = line.split()[0]
                if not word.startswith("sorta"):
                    continue
                with self.subTest(lang=lang, command=line):
                    self.assertIn(word, entry_points)

    def test_the_fix_a_guide_prints_is_the_fix_doctor_prints(self):
        """The other half of the pairing. The section tells a Linux reader that `doctor`
        answers the PATH, the `exiftool` and the cache failures — so the command each of
        those answers ends in is read out of the catalog, in that reader's language, and
        required to be the one the guide teaches. Editing one of them now breaks the
        other until it follows."""
        for key, fix in (("cli.doctor.command_hint", "uv tool update-shell"),
                         ("cli.doctor.exiftool_linux", "libimage-exiftool-perl"),
                         ("cli.doctor.cache_open", "chmod 700")):
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, key=key):
                    self.assertIn(fix, i18n.cli_text(key, lang))
                    # `assertTrue`, as in `test_docs_guides`: a failure that prints the
                    # whole 2,000-line guide buries the one word it is about.
                    self.assertTrue(fix in read(path), f"{path.name}: {key} — {fix}")

    def test_the_section_states_that_no_package_is_coming(self):
        """The owner's decision of 2026-08-06 is a fact a reader needs: somebody looking
        for a `.deb` has to find the answer here instead of an open question."""
        for lang, path in GUIDES.items():
            text = read(path)
            for token in ("AppImage", "deb/rpm"):
                with self.subTest(lang=lang, token=token):
                    self.assertTrue(token in text, f"{path.name}: {token} unmentioned")


if __name__ == "__main__":
    unittest.main()
