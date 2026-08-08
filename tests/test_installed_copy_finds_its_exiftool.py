"""F226: the installed copy uses the exiftool it brought with it.

The installer carried a 25 MB exiftool, the manifest named it — and the reader looked for
the binary by one name on PATH, which nothing in the install ever puts there. So every
installed copy fell through to Pillow and read no HEIC/RAW/video dates, no GPS and no
orientation, while the wizard's screen promised that "exiftool ships with the program".
Nobody noticed because the developer's machine and the CI runner both have exiftool on
PATH: "does it work here" was always answering about somebody else's machine.

That is what these cases are built around. The fixture is not a hand-written layout that
describes the day it was typed — it is assembled from `scripts/build_installer.py`'s own
constants and its own `build_manifest`, so a build that moves the binary or renames the
key fails here rather than on somebody's clean virtual machine. (`flatten_python_install`
had a test and caught nothing for exactly the opposite reason.)

The one thing the fixture cannot be faithful about is the binary itself: Windows will not
run a script named `.exe`, so the stand-in is a launcher `.cmd` there and a shell script
everywhere else. Everything around it — the `exiftool\\` directory, the `exiftool_files\\`
beside it, the manifest, the relative path inside it — is what the build writes.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sorta import cli, exif, i18n, install, wizard
from tests.test_build_installer import builder

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")

# The install directory of the owner's clean virtual machine — `{autopf}\Sorta` under
# `lowest`, which is `%LOCALAPPDATA%\Programs`. Written out rather than invented because a
# fixture that describes a convenient layout is a fixture that catches nothing.
_TARGET_INSTALL_DIR = Path("Users") / "vboxuser" / "AppData" / "Local" / "Programs" / "Sorta"
# ...and the same `{autopf}` when the install is elevated, which is where the spaces come
# from. (They also come from an ordinary account whose user name has one in it — the
# per-user path above has none only because that virtual machine's account is `vboxuser`.)
_SPACED_INSTALL_DIR = Path("Program Files") / "Sorta"

# What the stand-in binary reports, and what the fake claims to be. The version is the one
# the manifest of the shipped build carries, so a reader of this file can match the two.
_FAKE_VERSION = "13.59"


def _launcher(target: Path, body: str) -> Path:
    """A runnable file at `target` that prints `body` for any arguments.

    Windows cannot make an executable out of a script, so there the launcher is a `.cmd`
    next to where the build puts `exiftool.exe`; on everything else the script IS the
    name the build uses. Either way what the resolver receives is an absolute path to a
    file that starts, which is the property under test.
    """
    if sys.platform == "win32":
        target = target.with_suffix(".cmd")
        target.write_text(f"@echo off\r\necho {body}\r\n", encoding="utf-8")
    else:
        target.write_text(f'#!/bin/sh\necho "{body}"\n', encoding="utf-8")
        target.chmod(0o755)
    return target


def _dead_launcher(target: Path) -> Path:
    """The failure this feature must not mistake for success: an exiftool.exe with no
    `exiftool_files\\` beside it. The real one prints `Could not find ...perl5*.dll` on
    stderr and exits 1, which is what this does."""
    if sys.platform == "win32":
        target = target.with_suffix(".cmd")
        target.write_text("@echo off\r\nexit /b 1\r\n", encoding="utf-8")
    else:
        target.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        target.chmod(0o755)
    return target


def install_layout(root: Path, *, exiftool: bool = True, runnable: bool = True) -> Path:
    """An installed copy under `root`, laid out the way the BUILD lays it out.

    Returns the manifest path. The paths inside it come from `build_manifest`, so they
    are relative exactly as the build writes them and resolving them is the product's own
    job — the same code path an installed copy runs.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest = builder.build_manifest("1.2.3", exiftool=exiftool,
                                      tool_version=_FAKE_VERSION if exiftool else None)
    if exiftool:
        binary = root / builder.PAYLOAD_EXIFTOOL
        binary.parent.mkdir(parents=True, exist_ok=True)
        # The directory without which the real .exe does not start — part of the layout,
        # not decoration: the resolver is required to answer by RUNNING the binary, and
        # the fixture is what proves it does not answer by looking for a name.
        (root / builder.PAYLOAD_EXIFTOOL_FILES).mkdir(parents=True, exist_ok=True)
        written = (_launcher(binary, _FAKE_VERSION) if runnable
                   else _dead_launcher(binary))
        manifest["exiftool"] = str(written.relative_to(root))
    python = root / builder.PAYLOAD_PYTHON_EXE
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    path = root / install.MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class TheInstalledCopy(unittest.TestCase):
    """A temporary directory with an install in it, and no exiftool on the PATH of it."""

    exiftool = True
    runnable = True
    directory = _TARGET_INSTALL_DIR

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / self.directory
        self.manifest_path = install_layout(self.root, exiftool=self.exiftool,
                                            runnable=self.runnable)
        self.manifest = install.load_manifest(self.manifest_path)

    def tearDown(self):
        self.tmp.cleanup()

    def bundled(self) -> str | None:
        return install.tool_path(self.manifest, "exiftool")

    def resolve(self, *, on_path: str | None = None) -> str | None:
        return exif.resolve_exiftool(which=lambda _name: on_path,
                                     manifest=self.manifest)


class TestTheManifestIsReadRatherThanCopied(TheInstalledCopy):
    """Requirement 2: a small module that finds the manifest and resolves a name in it.

    It is deliberately the wizard's logic, so the two are pinned to each other here —
    moving `wizard.py` onto this module is the cleanup that follows, and until it happens
    a copy that drifts is a copy that answers differently about the same install.
    """

    def test_the_file_it_looks_for_is_the_one_the_installer_writes(self):
        self.assertEqual(install.MANIFEST_NAME, wizard.MANIFEST_NAME)
        self.assertEqual(install.ENV_MANIFEST, wizard.ENV_MANIFEST)
        self.assertEqual(install.MANIFEST_ROOT, wizard.MANIFEST_ROOT)

    def test_it_reads_the_same_manifest_the_wizard_reads(self):
        self.assertEqual(install.load_manifest(self.manifest_path),
                         wizard.load_manifest(self.manifest_path))

    def test_a_name_in_it_becomes_an_absolute_path_under_the_install(self):
        """The build writes `exiftool\\exiftool.exe`; where that is depends on where the
        person installed the program, which is the whole point of the relative form."""
        bundled = self.bundled()
        self.assertIsNotNone(bundled)
        self.assertTrue(Path(bundled).is_absolute(), bundled)
        self.assertEqual(Path(bundled).parent.parent, self.root)
        self.assertEqual(install.tool_path(self.manifest, "python"),
                         str(self.root / builder.PAYLOAD_PYTHON_EXE))

    def test_an_absolute_path_in_the_manifest_is_taken_as_written(self):
        self.assertEqual(install.tool_path({"exiftool": str(Path.cwd() / "e.exe")},
                                           "exiftool"),
                         str(Path.cwd() / "e.exe"))

    def test_a_key_the_manifest_does_not_name_is_no_path_at_all(self):
        """A build made with `--no-exiftool` is not a broken build — it records the
        decision, and every reader of it has a fallback behind this None."""
        fallback = builder.build_manifest("1.2.3", exiftool=False)
        self.assertIsNone(install.tool_path(fallback, "exiftool"))
        self.assertIsNone(install.tool_path({}, "exiftool"))

    def test_the_environment_can_name_it_and_a_checkout_has_none(self):
        with patch.dict(os.environ, {install.ENV_MANIFEST: str(self.manifest_path)}):
            self.assertEqual(install.manifest_path(), self.manifest_path)
            self.assertEqual(install.load_manifest()["exiftool_version"], _FAKE_VERSION)
        with patch.dict(os.environ, {install.ENV_MANIFEST: str(self.root / "nope.json")}):
            self.assertIsNone(install.manifest_path())
            self.assertEqual(install.load_manifest(), {})

    def test_a_manifest_that_cannot_be_parsed_is_an_empty_one(self):
        """It may not stop the product: an install that answers "nothing was shipped"
        degrades exactly like a checkout, which is a state everything here handles."""
        self.manifest_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(install.load_manifest(self.manifest_path), {})
        self.manifest_path.write_text("[]", encoding="utf-8")
        self.assertEqual(install.load_manifest(self.manifest_path), {})

    def test_the_manifest_is_found_above_the_running_interpreter(self):
        """How an installed copy finds it with nobody passing anything: the layout is
        `{app}\\python\\python.exe` and the manifest sits at `{app}\\sorta-install.json`."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(install.ENV_MANIFEST, None)
            with patch.object(install.sys, "executable",
                              str(self.root / builder.PAYLOAD_PYTHON_EXE)):
                self.assertEqual(install.manifest_path(), self.manifest_path)


class TestTheOrderTheBinaryIsResolvedIn(TheInstalledCopy):
    """Requirement 1, one case per step: PATH first, then the manifest.

    There is no explicit-path step: `config.py` has no key for one and this feature does
    not invent settings, so the order is the two that exist.
    """

    def test_no_config_key_names_an_exiftool_so_none_is_consulted(self):
        """The brief said to check before adding one. Written down as a case so that the
        day somebody adds `index.exiftool` this fails and the resolver is taught it."""
        example = (Path(__file__).resolve().parent.parent / "config.example.yaml")
        for line in example.read_text(encoding="utf-8").splitlines():
            self.assertNotIn("exiftool_path", line)
            self.assertNotIn("exiftool:", line)

    def test_a_machine_with_exiftool_on_path_keeps_using_that_one(self):
        """The developer's machine and the runner, i.e. the no-regression case: a copy
        somebody installed on purpose is also the copy they can update."""
        self.assertEqual(self.resolve(on_path="/usr/bin/exiftool"), "/usr/bin/exiftool")

    def test_path_wins_over_the_bundled_one_without_even_probing_it(self):
        probed: list[str] = []
        found = exif.resolve_exiftool(which=lambda _n: "/usr/bin/exiftool",
                                      runs=lambda binary: probed.append(binary) or True,
                                      manifest=self.manifest)
        self.assertEqual(found, "/usr/bin/exiftool")
        self.assertEqual(probed, [])

    def test_with_nothing_on_path_the_shipped_binary_is_used(self):
        self.assertEqual(self.resolve(), self.bundled())

    def test_and_it_is_an_absolute_path_so_the_machines_path_is_left_alone(self):
        """The command is the full path; nothing this program does changes what the word
        `exiftool` means in somebody's shell."""
        found = self.resolve()
        self.assertIsNotNone(found)
        self.assertTrue(Path(found).is_absolute(), found)


class TestABinaryThatCannotStartIsNotAnAnswer(TheInstalledCopy):
    """The trap the brief names: `exiftool.exe` is half of the Windows build, and without
    `exiftool_files\\` beside it the file is there, is named right, and exits 1."""

    runnable = False

    def test_a_bundled_binary_that_does_not_run_is_refused(self):
        self.assertIsNone(self.resolve())

    def test_a_binary_that_is_not_there_at_all_is_refused_too(self):
        self.assertFalse(exif._starts(str(self.root / "no-such-exiftool")))

    def test_the_probe_asks_for_a_version_rather_than_for_a_file_name(self):
        """`Path.exists()` would say yes to the broken copy above — this is the whole
        difference between the check that catches it and the one that does not."""
        self.assertTrue(Path(self.bundled()).is_file())
        self.assertFalse(exif._starts(self.bundled()))


class TestTheShippedBinaryReallyRuns(TheInstalledCopy):
    """An elevated install lands in `C:\\Program Files\\Sorta`, and both ways into exiftool
    hand that path to the OS. A shell would need it quoted; a list of arguments does not —
    and this is the case that proves the resolved path survives the trip whole."""

    directory = _SPACED_INSTALL_DIR

    def test_the_probe_starts_the_binary_through_a_path_with_spaces(self):
        self.assertIn(" ", str(self.root))
        self.assertTrue(exif._starts(self.bundled()))

    def test_the_command_the_reader_builds_is_the_resolved_binary(self):
        with patch.object(exif, "_EXIFTOOL_CMD", None), \
                patch.object(exif, "_resolved", (self.bundled(),)):
            self.assertEqual(exif._exiftool_cmd(), [self.bundled()])
            self.assertTrue(exif.exiftool_available())

    def test_a_stay_open_session_starts_on_that_command(self):
        """Not a mock of subprocess: the session is what the indexer uses, and `-stay_open`
        through a path with spaces is the thing that has to work on the target machine."""
        proc = subprocess.Popen([self.bundled(), "-stay_open", "True", "-@", "-"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        try:
            self.assertIsNone(proc.poll())
            self.assertIn(_FAKE_VERSION, (proc.stdout.readline() or b"").decode())
        finally:
            proc.kill()
            proc.wait(timeout=10)


class TestAMachineWithNeither(TheInstalledCopy):
    """A checkout, or an install built with `--no-exiftool`: the fallback to Pillow has to
    stay, and it has to be SAID rather than left for somebody to work out from empty
    dates."""

    exiftool = False

    def test_nothing_is_resolved_and_that_is_the_honest_answer(self):
        self.assertIsNone(self.resolve())

    def test_the_reader_falls_back_to_pillow(self):
        with patch.object(exif, "_resolved", (None,)):
            self.assertFalse(exif.exiftool_available())
            with patch.object(exif, "read_one_pillow",
                              lambda _path: exif.ExifData(make="pillow")):
                out = exif.read_batch([self.root / "a.heic"])
        self.assertEqual(next(iter(out.values())).make, "pillow")

    def test_doctor_says_so_and_offers_the_install_command_of_this_platform(self):
        lines = cli._doctor_install_lines("en", command="/usr/bin/sorta",
                                          scripts=Path("/bin"), exiftool=None,
                                          hint_key="cli.doctor.exiftool_windows")
        self.assertIn("HEIC/RAW", lines[-2])
        self.assertIn("winget install", lines[-1])


class TestDoctorNamesTheExiftoolItWillActuallyUse(TheInstalledCopy):
    """Requirement 3. The screen said two opposite things at once: the wizard promised a
    bundled exiftool and `doctor` reported the consequence of the defect as normal, then
    advised installing a second copy of what was already on the disk."""

    def _lines(self, lang: i18n.Lang = "en", *, on_path: str | None = None) -> list[str]:
        found = {"exiftool": on_path}
        with patch.object(cli.shutil, "which", found.get), \
                patch.object(cli.install, "load_manifest", lambda: self.manifest):
            exiftool = cli.exif.resolve_exiftool(which=cli.shutil.which,
                                                 manifest=self.manifest)
            return cli._doctor_install_lines(
                lang, command=None, scripts=Path("/bin"), exiftool=exiftool,
                bundled=exiftool == self.bundled(),
                installed_python=install.tool_path(self.manifest, "python"))

    def test_it_names_the_path_of_the_bundled_binary(self):
        self.assertIn(self.bundled(), self._lines()[-1])

    def test_and_does_not_advise_installing_a_second_copy_of_it(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                printed = "\n".join(self._lines(lang))
                self.assertNotIn("winget", printed)
                self.assertNotIn("brew install", printed)
                self.assertNotIn("apt install", printed)

    def test_it_says_the_binary_came_with_the_program(self):
        """The wizard's sentence, made true and then repeated by the screen that checks
        it: a person reading `doctor` must not have to guess which of two exiftools the
        index will use."""
        line = self._lines()[-1]
        self.assertIn("ships with the program", line)
        self.assertIn("exiftool", line)

    def test_a_copy_on_path_is_still_reported_as_the_plain_one(self):
        line = self._lines(on_path="/usr/bin/exiftool")[-1]
        self.assertIn("/usr/bin/exiftool", line)
        self.assertNotIn("ships with the program", line)

    def test_an_installed_copy_is_not_told_to_run_uv_tool_update_shell(self):
        """That command fixes an install made by `uv tool install`. An installed copy was
        never on PATH and is not meant to be; what it needs is how it IS run."""
        printed = "\n".join(self._lines())
        self.assertNotIn("uv tool update-shell", printed)
        self.assertIn(str(self.root / builder.PAYLOAD_PYTHON_EXE), printed)
        self.assertIn("-m sorta.cli", printed)

    def test_a_checkout_keeps_the_advice_that_is_right_for_it(self):
        lines = cli._doctor_install_lines("en", command=None, scripts=Path("/bin"),
                                          exiftool="/usr/bin/exiftool")
        self.assertIn("uv tool update-shell", lines[1])

    def test_every_new_line_exists_in_three_languages(self):
        rendered = {lang: self._lines(lang) for lang in _LANGS}
        for index in range(len(rendered["en"])):
            with self.subTest(line=index):
                texts = {rendered[lang][index] for lang in _LANGS}
                self.assertEqual(len(texts), 3, texts)
                self.assertTrue(all(text.strip() for text in texts))

    def test_doctor_prints_it(self):
        """The wiring: every helper above is pure, so without this the command could
        print none of them and the suite would stay green."""
        health = SimpleNamespace(summary="health", available=True)
        buffer = io.StringIO()
        with patch.object(cli.shutil, "which", lambda _name: None), \
                patch.object(cli.install, "load_manifest", lambda: self.manifest), \
                patch.object(cli, "gpu_health", lambda: health), \
                patch.object(cli, "geo_data_health", lambda: health), \
                patch.object(cli, "tier_states", lambda: []), \
                patch.object(cli, "default_log_path", lambda: "run.log"), \
                patch.object(cli, "_directory_mode", lambda _path: 0o700), \
                redirect_stdout(buffer):
            cli._cmd_doctor("no-such-config.yaml")
        printed = buffer.getvalue()
        self.assertIn(self.bundled(), printed)
        self.assertNotIn("winget", printed)
        self.assertNotIn("uv tool update-shell", printed)


class TestTheBuildCarriesWhatMakesItStart(unittest.TestCase):
    """The half of the defect that is about FILES, which is what the brief says would have
    caught the whole thing before the first install: the payload copied `exiftool.exe` and
    nothing else, so the 25 MB that travelled could not have started even once it was
    found."""

    def test_the_directory_the_binary_needs_travels_with_it(self):
        plan = builder.payload_plan(Path("C:/tools/exiftool.exe"))
        self.assertIn((Path("C:/tools/exiftool.exe"), builder.PAYLOAD_EXIFTOOL), plan)
        self.assertIn((Path("C:/tools/exiftool_files"), builder.PAYLOAD_EXIFTOOL_FILES),
                      plan)

    def test_it_lands_beside_the_binary_inside_the_payload(self):
        self.assertEqual(builder.PAYLOAD_EXIFTOOL_FILES.parent,
                         builder.PAYLOAD_EXIFTOOL.parent)
        self.assertEqual(builder.PAYLOAD_EXIFTOOL_FILES.name, "exiftool_files")

    def test_a_build_without_exiftool_carries_neither(self):
        destinations = {destination for _source, destination in builder.payload_plan(None)}
        self.assertNotIn(builder.PAYLOAD_EXIFTOOL, destinations)
        self.assertNotIn(builder.PAYLOAD_EXIFTOOL_FILES, destinations)

    def test_the_manifest_names_the_binary_by_the_key_the_reader_reads(self):
        """The pairing the brief asks for, in one assertion: what the BUILD writes against
        what `sorta.exif` looks up. A renamed key fails here."""
        manifest = builder.build_manifest("1.2.3", exiftool=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / install.MANIFEST_NAME).write_text(json.dumps(manifest),
                                                      encoding="utf-8")
            read = install.load_manifest(root / install.MANIFEST_NAME)
            self.assertEqual(install.tool_path(read, "exiftool"),
                             str(root / builder.PAYLOAD_EXIFTOOL))


class TestTheResolutionIsRememberedRatherThanRepeated(unittest.TestCase):
    """`read_batch` asks per batch and the shipped branch of the answer costs a
    subprocess, so it is resolved once per process."""

    def setUp(self):
        self._orig = exif._resolved
        exif._resolved = None

    def tearDown(self):
        exif._resolved = self._orig

    def test_the_binary_is_looked_for_once(self):
        calls: list[str] = []

        def which(name: str) -> str | None:
            calls.append(name)
            return "/usr/bin/exiftool"

        with patch.object(exif.shutil, "which", which):
            self.assertEqual(exif.exiftool_binary(), "/usr/bin/exiftool")
            self.assertTrue(exif.exiftool_available())
            self.assertTrue(exif.exiftool_available())
        self.assertEqual(calls, ["exiftool"])

    def test_resolving_to_nothing_is_remembered_too(self):
        """"Not found" and "not asked yet" are different states — a machine with no
        exiftool must not pay for the lookup on every batch it reads."""
        calls: list[str] = []

        def which(name: str) -> str | None:
            calls.append(name)
            return None

        with patch.object(exif.shutil, "which", which), \
                patch.object(exif.install, "load_manifest", dict):
            self.assertIsNone(exif.exiftool_binary())
            self.assertFalse(exif.exiftool_available())
        self.assertEqual(calls, ["exiftool"])

    def test_it_can_be_asked_again_on_purpose(self):
        with patch.object(exif.shutil, "which", lambda _n: "/a/exiftool"):
            self.assertEqual(exif.exiftool_binary(), "/a/exiftool")
        with patch.object(exif.shutil, "which", lambda _n: "/b/exiftool"):
            self.assertEqual(exif.exiftool_binary(), "/a/exiftool")
            self.assertEqual(exif.exiftool_binary(refresh=True), "/b/exiftool")


if __name__ == "__main__":
    unittest.main()
