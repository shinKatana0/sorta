"""F211: the build script and the installer script — everything checkable without building.

An installer cannot be run through the gate, and pretending otherwise would produce a
test that proves nothing (the manual checklist lives in packaging/windows/README.md). What
IS checkable is every decision the build makes before a byte is downloaded:

* the commands — which interpreter is fetched, what the base tier is installed as, what
  Inno is handed, and the signing step that must stay OFF unless it is asked for;
* the payload — which files travel, and the manifest the wizard reads afterwards, in the
  relative form that survives being installed somewhere else;
* the .iss — the default profile put in place from `config.example.yaml` and never over
  an edited one, a shortcut that starts the tray without a console, and the two absences
  the brief asks for: no autostart, no signing on the mandatory path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sorta import wizard
from sorta.config import load_config

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_installer.py"
_ISS = _ROOT / "packaging" / "windows" / "sorta.iss"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_installer", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_script()


def _link_directory(target: Path, link: Path) -> bool:
    """Give `target` a second name at `link`, the way uv does. False if this machine won't.

    On Windows that means a JUNCTION and not a symlink: a symlink needs a privilege an
    ordinary account does not have (which would skip this test on the very platform the
    installer is for), while `mklink /J` needs none — and a junction is what uv actually
    leaves behind, so it is also the faithful fixture.
    """
    if sys.platform == "win32":
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                   capture_output=True, text=True)
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True
ISS_TEXT = _ISS.read_text(encoding="utf-8")


def iss_section(name: str) -> list[str]:
    """The non-comment lines of one `[Section]` of the Inno script."""
    lines: list[str] = []
    inside = False
    for line in ISS_TEXT.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped.lower() == f"[{name.lower()}]"
            continue
        if inside and stripped and not stripped.startswith(";"):
            lines.append(stripped)
    return lines


def iss_entry(section: str, needle: str) -> str:
    """The one line of `section` that mentions `needle` — the entry under test."""
    found = [line for line in iss_section(section) if needle in line]
    assert len(found) == 1, f"[{section}] x {needle!r}: {found}"
    return found[0]


class TestTheCommands(unittest.TestCase):
    """What the build would run, checked as data."""

    def test_the_interpreter_is_fetched_into_the_payload(self):
        command = builder.python_install_command("uv", Path("dist/payload/python"))
        self.assertEqual(command[:3], ["uv", "python", "install"])
        self.assertIn(builder.PYTHON_VERSION, command)
        self.assertIn("--install-dir", command)

    def test_the_build_machine_gets_no_shim_and_no_registry_entry(self):
        """A payload is staged, not installed: nothing here may point the build machine's
        PATH or registry into `dist/`."""
        command = builder.python_install_command("uv", Path("dist/payload/python"))
        self.assertIn("--no-bin", command)
        self.assertIn("--no-registry", command)

    def test_the_versioned_directory_uv_writes_is_lifted_to_one_fixed_path(self):
        """`python\\python.exe` is named by the manifest, the .pth and two shortcuts —
        a path with a patch version in it would have to be rewritten in all four."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "python"
            staged = root / "cpython-3.13.1-windows-x86_64-none"
            (staged / "Lib" / "site-packages").mkdir(parents=True)
            (staged / "python.exe").write_bytes(b"")
            self.assertEqual(builder.flatten_python_install(root), root / "python.exe")
            self.assertTrue((root / "python.exe").is_file())
            self.assertTrue((root / "Lib" / "site-packages").is_dir())
            self.assertFalse(staged.exists())
            # ...and doing it twice changes nothing.
            self.assertEqual(builder.flatten_python_install(root), root / "python.exe")

    def test_the_alias_uv_leaves_beside_the_interpreter_is_not_a_second_interpreter(self):
        """Caught by building for real, 2026-08-07: uv 0.11 writes the versioned
        directory AND a `cpython-3.13-...` junction pointing at it, so a minor version
        can be named without its patch. Both are directories holding a python.exe, and
        the build stopped with "found two installations" on a perfectly good download.

        The fixture above was hand-made and therefore never disagreed with uv about
        anything — the test pinned the layout of the day it was written."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "python"
            staged = root / "cpython-3.13.14-windows-x86_64-none"
            (staged / "Lib").mkdir(parents=True)
            (staged / "python.exe").write_bytes(b"")
            alias = root / "cpython-3.13-windows-x86_64-none"
            if not _link_directory(staged, alias):  # pragma: no cover — platform
                self.skipTest("this machine will not let the test create a link")
            # The distinction the fix turns on: a junction is NOT a symlink, and asking
            # `is_symlink()` about one gets a confident no. If this ever stops being a
            # junction on Windows, the test below stops covering the real case.
            if sys.platform == "win32":
                self.assertFalse(alias.is_symlink())

            self.assertEqual(builder.flatten_python_install(root), root / "python.exe")
            self.assertTrue((root / "python.exe").is_file())
            self.assertTrue((root / "Lib").is_dir())
            # Both names are gone: the alias too, because a link left pointing at a
            # directory that has just been emptied would ship inside the payload.
            self.assertFalse(staged.exists())
            self.assertFalse(alias.exists() or alias.is_symlink())

    def test_two_real_interpreters_are_still_an_error(self):
        """The alias case must not turn into "take whichever comes first": two genuine
        installations mean the payload would be a coin toss."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "python"
            for version in ("cpython-3.13.14-windows-x86_64-none",
                            "cpython-3.12.9-windows-x86_64-none"):
                (root / version).mkdir(parents=True)
                (root / version / "python.exe").write_bytes(b"")
            with self.assertRaises(SystemExit):
                builder.flatten_python_install(root)

    def test_an_install_directory_with_no_interpreter_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                builder.flatten_python_install(Path(tmp))

    def test_the_base_tier_is_installed_as_a_target_tree_with_its_extras(self):
        command = builder.base_install_command("uv", Path("py/python.exe"), Path("lib"))
        self.assertEqual(command[:4], ["uv", "pip", "install", "--python"])
        self.assertIn("--target", command)
        # The extras come from the tier catalog, so the payload and the wizard cannot
        # describe different base tiers.
        self.assertTrue(command[-1].endswith("[cpu,tray]"), command[-1])

    def test_a_target_tree_is_what_makes_the_payload_movable(self):
        """Not a virtualenv, and the .pth that finds it is relative for the same reason:
        the payload is built here and installed on somebody else's disk."""
        self.assertNotIn("venv", builder.base_install_command("uv", Path("p"), Path("l")))
        self.assertTrue(builder.PTH_LINE.startswith(".."), builder.PTH_LINE)
        self.assertTrue(builder.PTH_LINE.endswith("lib"), builder.PTH_LINE)

    def test_every_define_the_build_passes_is_one_the_script_expects(self):
        """The pairing between the two files, so a renamed define fails here and not in
        an installer that quietly compiled with a default."""
        command = builder.iscc_command("ISCC", "1.2.3")
        names = [part.split("=")[0][2:] for part in command if part.startswith("/D")]
        self.assertEqual(sorted(names), ["OutputDir", "PayloadDir", "Version"])
        for name in names:
            with self.subTest(define=name):
                self.assertIn(f"#ifndef {name}", ISS_TEXT)
        self.assertEqual(command[-1], str(_ISS))

    def test_the_name_the_script_writes_is_the_name_the_build_looks_for(self):
        self.assertIn("OutputBaseFilename=sorta-{#Version}-setup", ISS_TEXT)
        self.assertEqual(builder.installer_path("1.2.3", Path("out")).name,
                         "sorta-1.2.3-setup.exe")


class TestSigningIsOffUntilItIsAskedFor(unittest.TestCase):
    """The owner's decision of 2026-08-06: unsigned, with a place for a signature."""

    def test_nothing_is_signed_by_default(self):
        self.assertFalse(builder.signing_requested({}, flag=False))
        self.assertFalse(builder.signing_requested({builder.ENV_SIGN: "0"}))
        self.assertFalse(builder.signing_requested({builder.ENV_SIGN: ""}))

    def test_a_flag_or_a_variable_switches_it_on(self):
        self.assertTrue(builder.signing_requested({}, flag=True))
        self.assertTrue(builder.signing_requested({builder.ENV_SIGN: "1"}))

    def test_the_step_is_a_signtool_call_with_a_timestamp(self):
        command = builder.sign_command(Path("out/sorta-1.2.3-setup.exe"), {
            builder.ENV_SIGN_TOOL: "C:/kits/signtool.exe",
            builder.ENV_SIGN_CERT: "C:/certs/sorta.pfx",
            builder.ENV_SIGN_PASSWORD: "secret",
        })
        self.assertEqual(command[:4],
                         ["C:/kits/signtool.exe", "sign", "/f", "C:/certs/sorta.pfx"])
        self.assertIn("/p", command)
        self.assertIn(builder.DEFAULT_TIMESTAMP_URL, command)
        self.assertEqual(command[-1], str(Path("out/sorta-1.2.3-setup.exe")))

    def test_asking_for_a_signature_without_a_certificate_is_an_error(self):
        """It has to fail loudly: an unsigned file quietly produced by a build that was
        told to sign is the one thing worse than an unsigned release."""
        with self.assertRaises(ValueError):
            builder.sign_command(Path("out/x.exe"), {})

    def test_the_installer_script_carries_no_signing_of_its_own(self):
        self.assertNotIn("SignTool", ISS_TEXT)
        self.assertNotIn("SignedUninstaller", ISS_TEXT)


class TestThePayloadAndItsManifest(unittest.TestCase):
    """What travels, and what the wizard reads on the other side."""

    def test_the_static_files_include_the_licence_and_the_icon(self):
        plan = builder.payload_plan(None)
        destinations = {str(destination) for _source, destination in plan}
        self.assertEqual(destinations,
                         {"config.example.yaml", "LICENSE", "NOTICE", "favicon.ico"})
        for source, _destination in plan:
            with self.subTest(source=source.name):
                self.assertTrue(source.is_file(), source)

    def test_the_exiftool_binary_travels_when_there_is_one(self):
        plan = builder.payload_plan(Path("C:/tools/exiftool.exe"))
        self.assertIn((Path("C:/tools/exiftool.exe"), builder.PAYLOAD_EXIFTOOL), plan)

    def test_the_bundled_version_travels_so_the_update_debt_is_visible(self):
        """NOTICE §3 promises it: bundling a binary means owing an update to whoever
        installed it, and an obligation nobody can see the state of is not one that
        gets met."""
        manifest = builder.build_manifest("1.2.3", exiftool=True, tool_version="13.10")
        self.assertEqual(manifest["exiftool_version"], "13.10")
        self.assertIsNone(builder.exiftool_version(None))

    def test_the_manifest_records_the_exiftool_decision_either_way(self):
        bundled = builder.build_manifest("1.2.3", exiftool=True)
        self.assertEqual(bundled["exiftool"], str(builder.PAYLOAD_EXIFTOOL))
        self.assertEqual(wizard.exiftool_state(bundled), wizard.EXIFTOOL_BUNDLED)
        fallback = builder.build_manifest("1.2.3", exiftool=False)
        self.assertIsNone(fallback["exiftool"])
        self.assertEqual(wizard.exiftool_state(fallback, which=lambda _n: None),
                         wizard.EXIFTOOL_ABSENT)

    def test_the_manifest_lists_the_tiers_it_shipped(self):
        manifest = builder.build_manifest("1.2.3", exiftool=True)
        self.assertEqual([tier["key"] for tier in manifest["tiers"]],
                         [tier.key for tier in wizard.TIERS])
        self.assertEqual(manifest["version"], "1.2.3")
        self.assertEqual(manifest["python_version"], builder.PYTHON_VERSION)

    def test_the_paths_are_relative_and_the_wizard_resolves_them_where_it_finds_them(self):
        """The two halves paired: the build writes paths relative to the manifest, and
        the wizard reads them against wherever the person installed the program."""
        manifest = builder.build_manifest("1.2.3", exiftool=True)
        for key in ("python", "lib", "uv", "exiftool"):
            with self.subTest(key=key):
                self.assertFalse(Path(manifest[key]).is_absolute(), manifest[key])
        with tempfile.TemporaryDirectory() as tmp:
            installed = Path(tmp)
            (installed / wizard.MANIFEST_NAME).write_text(json.dumps(manifest),
                                                          encoding="utf-8")
            read = wizard.load_manifest(installed / wizard.MANIFEST_NAME)
            self.assertEqual(wizard.python_binary(read),
                             str(installed / builder.PAYLOAD_PYTHON_EXE))
            self.assertEqual(wizard.uv_binary(read), str(installed / builder.PAYLOAD_UV))
            self.assertEqual(wizard.lib_directory(read),
                             str(installed / builder.PAYLOAD_LIB))
            # ...and a tier added later lands in the same tree the base one is in.
            command = wizard.install_command(wizard.TIERS_BY_KEY["deep"], ["transformers"],
                                             uv=wizard.uv_binary(read),
                                             python=wizard.python_binary(read),
                                             target=wizard.lib_directory(read))
            self.assertIn(str(installed / builder.PAYLOAD_LIB), command)

    def test_the_checksum_is_the_one_a_person_can_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "setup.exe"
            path.write_bytes(b"not really an installer")
            self.assertEqual(builder.sha256(path),
                             hashlib.sha256(b"not really an installer").hexdigest())

    def test_a_dry_run_asks_for_nothing_and_writes_nothing(self):
        before = sorted(p.name for p in (_ROOT / "dist").glob("*")) \
            if (_ROOT / "dist").exists() else []
        code = builder.main(["--dry-run", "--skip-payload", "--no-exiftool"])
        self.assertEqual(code, 0)
        after = sorted(p.name for p in (_ROOT / "dist").glob("*")) \
            if (_ROOT / "dist").exists() else []
        self.assertEqual(before, after)


class TestTheDefaultProfileIsPutInPlace(unittest.TestCase):
    """Requirement 4 of the brief, and its test: the config the installer leaves IS
    `config.example.yaml` — not a second copy of it that will drift."""

    def test_the_installer_copies_the_example_and_renames_it(self):
        entry = iss_entry("Files", "config.example.yaml")
        self.assertIn('DestName: "config.yaml"', entry)
        self.assertIn("onlyifdoesntexist", entry)

    def test_there_is_no_second_copy_of_the_config_in_the_packaging_directory(self):
        """A profile written out by hand would be the one that drifts — the whole
        failure this pairing prevents."""
        copies = list((_ROOT / "packaging").rglob("config*.yaml"))
        self.assertEqual(copies, [])

    def test_what_gets_installed_carries_the_defaults_the_product_wants(self):
        cfg = load_config(str(_ROOT / "config.example.yaml"))
        # The deep tier OFF is the one the brief names: an installer that switched it on
        # would quadruple the first run of everybody who accepted the defaults.
        self.assertFalse(cfg.vlm.enabled)
        self.assertFalse(cfg.naming.vlm_enabled)
        self.assertEqual(cfg.language, "en")

    def test_an_edited_config_is_never_overwritten_or_removed(self):
        entry = iss_entry("Files", "config.example.yaml")
        self.assertIn("uninsneveruninstall", entry)


class TestTheShortcutsAndTheAbsences(unittest.TestCase):
    """What a person clicks — and the two things the installer deliberately does not do."""

    def test_the_shortcut_starts_the_tray_without_a_console(self):
        entry = iss_entry("Icons", "{group}\\{#AppName}\"")
        self.assertIn("pythonw.exe", entry)
        self.assertIn("-m sorta.tray", entry)
        # The working directory is where config.yaml and the index live: the program
        # reads `config.yaml` from the current directory.
        self.assertIn('WorkingDir: "{userappdata}\\sorta"', entry)

    def test_the_wizard_runs_once_at_the_end_of_the_installation(self):
        entry = iss_entry("Run", "sorta.wizard")
        self.assertIn("postinstall", entry)
        self.assertIn("python.exe", entry)

    def test_the_wizard_is_started_in_utf8(self):
        """Its catalog is Russian, English and Japanese, and a Windows console runs on a
        legacy code page unless it is told otherwise — `-X utf8` is that telling."""
        for section in ("Run", "Icons"):
            with self.subTest(section=section):
                self.assertIn("-X utf8 -m sorta.wizard", iss_entry(section, "sorta.wizard"))

    def test_nothing_is_registered_to_start_with_the_system(self):
        """A boundary of the brief: a program that starts with the machine is the
        owner's decision, and this installer does not take it."""
        self.assertEqual(iss_section("Registry"), [])
        for forbidden in ("{userstartup}", "{commonstartup}", "CurrentVersion\\Run"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, ISS_TEXT)

    def test_the_uninstaller_deletes_nothing_of_the_persons(self):
        """Uninstalling a program is not a request to delete data.

        This test used to say the opposite — that sweeping the preview cache was fine
        because "we wrote it". Two things make that wrong. The run log sits in the same
        directory and carries this machine's measurements, which cost a five-hour run to
        replace. And the capability already exists where it belongs: the interface clears
        the preview cache on the Process tab, with the size shown beside it — a
        destructive default would duplicate it silently, at the one moment nobody is
        watching.

        Offering it as a question during uninstall would be fine. What this pins is only
        that nothing goes without being asked.
        """
        self.assertEqual(iss_section("UninstallDelete"), [])

    def test_the_installer_needs_no_administrator(self):
        self.assertIn("PrivilegesRequired=lowest", ISS_TEXT)

    def test_the_three_interface_languages_are_offered(self):
        languages = iss_section("Languages")
        self.assertEqual(len(languages), 3)
        for name in ("Russian.isl", "Japanese.isl", "Default.isl"):
            with self.subTest(language=name):
                self.assertTrue(any(name in line for line in languages))

    def test_the_setup_icon_is_the_one_the_program_shows(self):
        """F207 again: three pictures of one program read as three programs."""
        self.assertIn("SetupIconFile=..\\..\\sorta\\web\\favicon.ico", ISS_TEXT)
        self.assertTrue((_ROOT / "sorta" / "web" / "favicon.ico").is_file())


class TestTheBuildIsDocumented(unittest.TestCase):
    """The parts of this feature that are prose, and are load-bearing anyway."""

    def setUp(self):
        self.readme = (_ROOT / "packaging" / "windows" / "README.md").read_text(
            encoding="utf-8")

    def test_the_exiftool_decision_is_written_down(self):
        self.assertIn("exiftool", self.readme)
        self.assertRegex(self.readme, r"(?i)bundle")

    def test_smartscreen_is_warned_about_before_the_download(self):
        """An unsigned installer meets a red screen; a person who was told about it
        beforehand reads it as `no certificate`, and one who was not reads it as
        `dangerous`."""
        self.assertIn("SmartScreen", self.readme)
        self.assertIn("sha256", self.readme.lower())
        for path in (_ROOT / "README.md", _ROOT / "docs" / "guide" / "user-guide.en.md"):
            with self.subTest(document=path.name):
                self.assertIn("SmartScreen", path.read_text(encoding="utf-8"))

    def test_the_manual_checklist_exists_because_it_cannot_be_automated(self):
        for item in ("clean machine", "shortcut", "refus", "tier"):
            with self.subTest(item=item):
                self.assertRegex(self.readme, rf"(?i){re.escape(item)}")


if __name__ == "__main__":
    unittest.main()
