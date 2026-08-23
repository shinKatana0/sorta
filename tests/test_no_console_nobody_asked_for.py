"""F228: a windowed Sorta opens no console nobody asked for.

The shortcut runs `pythonw.exe`. On Windows a console program started from a parent that
has no console gets a NEW WINDOW created for it, and nothing in the product passed
`CREATE_NO_WINDOW` — so the `nvidia-smi` probe, the `exiftool -ver` probe, up to eight
`exiftool -stay_open` sessions and `uv` in the wizard each flashed one. The owner read
the last of those as a black window that opened while the interface said a model was
downloading and in which nothing ever happened.

Two halves are tested here, and the first one is the point of the feature.

* **The watchdog.** `sorta/` is read with `ast` and every call that STARTS A PROCESS is
  listed. Each one has to come through `sorta/launch.py`, or be named in
  `_NOT_THROUGH_THE_HELPER` below with the reason it is not. Source-reading rather than
  a grep: `subprocess.run` written in a comment is not a launch, and a check that cannot
  tell the difference is a check people learn to ignore. Without it the feature closes
  four cases and not the class — which is exactly what happened before it, F226 having
  added the fourth call the day before the defect was reported.
* **The behaviour.** With a faked "there is no console" the helper passes the flag, with
  "there is a console" it does not, and off Windows it never does. That third one is not
  a formality: `sorta-setup` typed into a terminal shows the output of `uv` in that
  console, and hiding it there would throw away the install log somebody is reading.

A watchdog nobody has seen go red is not a watchdog (F182 and F216 both taught this
project that the expensive way), so `TestTheWatchdogGoesRed` feeds the scanner a module
with a bare `subprocess.run` in it and reads the complaint.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

from sorta import diagnostics, exif, launch, wizard

_PKG = Path(launch.__file__).resolve().parent

# What counts as starting a process. `subprocess` gives most of them; `os.system`,
# `os.popen` and the `os.spawn*` family are here because they open the same window and
# would be the obvious way around a check that only knew about `subprocess`.
_LAUNCHERS = frozenset({
    "subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.getoutput", "subprocess.getstatusoutput",
    "os.system", "os.popen", "os.startfile",
})
_LAUNCHER_MODULES = frozenset({"subprocess", "os"})

# The helper, as the call sites name it: `from . import launch` and then `launch.run`.
_HELPERS = frozenset({"launch.run", "launch.popen"})
_HELPER_MODULES = frozenset({"launch"})


def _is_launcher(name: str) -> bool:
    return name in _LAUNCHERS or name.startswith("os.spawn")


class _Calls(ast.NodeVisitor):
    """Every call to one of `wanted`, with the qualified name of what it sits inside.

    Aliases are followed (`import subprocess as sp`, `from subprocess import run`) —
    otherwise the check is one rename away from seeing nothing at all.
    """

    def __init__(self, modules: frozenset[str], wanted: Callable[[str], bool]) -> None:
        self._modules = modules
        self._wanted = wanted
        self._alias: dict[str, str] = {}
        self._bound: dict[str, str] = {}
        self._scope: list[str] = []
        self.found: list[tuple[str, int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in self._modules:
                self._alias[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None and node.level:
            # `from . import launch` — the name IS the module.
            for alias in node.names:
                if alias.name in self._modules:
                    self._alias[alias.asname or alias.name] = alias.name
        elif node.module in self._modules and not node.level:
            for alias in node.names:
                self._bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def _nested(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._nested(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._nested(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._nested(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._called(node.func)
        if name is not None and self._wanted(name):
            self.found.append((".".join(self._scope) or "<module>", node.lineno, name))
        self.generic_visit(node)

    def _called(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self._alias.get(func.value.id)
            return None if module is None else f"{module}.{func.attr}"
        if isinstance(func, ast.Name):
            return self._bound.get(func.id)
        return None


def calls_in(source: str, modules: frozenset[str],
             wanted: Callable[[str], bool]) -> list[tuple[str, int, str]]:
    visitor = _Calls(modules, wanted)
    visitor.visit(ast.parse(source))
    return visitor.found


def launch_sites(source: str) -> list[tuple[str, int, str]]:
    """Where this source starts a process, whatever it imported to do it."""
    return calls_in(source, _LAUNCHER_MODULES, _is_launcher)


def package_sources() -> list[Path]:
    return sorted(_PKG.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(_PKG).as_posix()


def direct_launches() -> dict[tuple[str, str], str]:
    """Every launch in the package that does NOT come through the helper.

    `launch.py` is skipped: it is the one file whose job is to call `subprocess`.
    """
    out: dict[tuple[str, str], str] = {}
    for path in package_sources():
        if _relative(path) == "launch.py":
            continue
        for where, line, name in launch_sites(path.read_text(encoding="utf-8")):
            out[(_relative(path), where)] = f"{name} at line {line}"
    return out


# The launches deliberately left alone, and why. An empty dict would be the goal; this
# entry is not an exemption from the rule but a statement of ownership, and it is pinned
# in both directions below so it cannot quietly become a habit.
_NOT_THROUGH_THE_HELPER: dict[tuple[str, str], str] = {
    ("ui/process.py", "_run_browse_dialog"):
        "F227 is being written inside `sorta/ui/` while this feature is, so F228 does not "
        "reach into that package. It is also the one launch of the five that costs "
        "nothing: it starts `sys.executable`, which under the shortcut is `pythonw.exe` "
        "— a windowed interpreter, which opens no console for anybody.",
}

# Where the helper HAS to be called from — the four places of the brief plus the one-shot
# exiftool fallback, which starts the same program by another route.
_THROUGH_THE_HELPER = {
    ("diagnostics.py", "_run_nvidia_smi"),
    # F246: the GPU probe moved out of the server process and became the sixth child.
    ("diagnostics.py", "_run_gpu_probe"),
    ("exif.py", "_starts"),
    ("exif.py", "ExifToolSession._ensure"),
    ("exif.py", "read_batch_exiftool"),
    ("wizard.py", "run_install"),
}


class TestEveryLaunchGoesThroughTheHelper(unittest.TestCase):
    """The sentinel this whole feature exists for: about FILES, not about a window."""

    def test_nothing_in_the_package_starts_a_process_on_its_own(self):
        unaccounted = {
            key: why for key, why in direct_launches().items()
            if key not in _NOT_THROUGH_THE_HELPER
        }
        self.assertEqual(unaccounted, {},
                         "these start a process without `sorta.launch`, so on a machine "
                         "with no console they open a window: route them through "
                         "`launch.run` / `launch.popen`")

    def test_the_list_of_places_left_alone_has_nothing_stale_on_it(self):
        """The other direction. A name that stops being a direct launch has to leave this
        list, or the list stops describing the package and starts excusing it."""
        found = direct_launches()
        for key, reason in _NOT_THROUGH_THE_HELPER.items():
            with self.subTest(place=key):
                self.assertIn(key, found,
                              f"{key} no longer starts a process directly — delete its "
                              f"entry from _NOT_THROUGH_THE_HELPER ({reason})")

    def test_every_reason_on_that_list_is_written_out(self):
        for key, reason in _NOT_THROUGH_THE_HELPER.items():
            with self.subTest(place=key):
                self.assertGreater(len(reason), 40, reason)

    def test_the_places_the_defect_named_call_the_helper(self):
        """The positive half: the check above would also pass on a module that simply
        stopped starting anything."""
        calling: set[tuple[str, str]] = set()
        for path in package_sources():
            source = path.read_text(encoding="utf-8")
            for where, _line, _name in calls_in(source, _HELPER_MODULES,
                                                _HELPERS.__contains__):
                calling.add((_relative(path), where))
        for place in _THROUGH_THE_HELPER:
            with self.subTest(place=place):
                self.assertIn(place, calling)


class TestTheWatchdogGoesRed(unittest.TestCase):
    """A check nobody has seen fail is not a check."""

    def test_a_new_launch_written_the_old_way_is_found(self):
        source = (
            "import subprocess\n"
            "def probe():\n"
            "    return subprocess.run(['ffprobe', '-version'])\n"
        )
        self.assertEqual(launch_sites(source), [("probe", 3, "subprocess.run")])

    def test_an_alias_does_not_hide_it(self):
        source = (
            "import subprocess as sp\n"
            "from subprocess import Popen as spawn\n"
            "from os import system\n"
            "class Session:\n"
            "    def start(self):\n"
            "        sp.Popen(['x'])\n"
            "        spawn(['y'])\n"
            "        system('z')\n"
        )
        self.assertEqual(
            [(where, name) for where, _line, name in launch_sites(source)],
            [("Session.start", "subprocess.Popen"),
             ("Session.start", "subprocess.Popen"),
             ("Session.start", "os.system")])

    def test_the_whole_os_spawn_family_counts(self):
        source = "import os\ndef go():\n    os.spawnv(os.P_NOWAIT, 'x', ['x'])\n"
        self.assertEqual([name for _where, _line, name in launch_sites(source)],
                         ["os.spawnv"])

    def test_a_launch_that_is_only_written_about_is_not_a_launch(self):
        """The reason this reads the syntax tree instead of the text: `subprocess.run` in
        a comment, in a docstring or in a string literal must not go red, or the check
        becomes noise and noise gets switched off."""
        source = (
            "import subprocess\n"
            "def helper():\n"
            "    '''Replaces subprocess.run(...) for every caller.'''\n"
            "    # subprocess.Popen would open a window here\n"
            "    hint = 'os.system is not what we do'\n"
            "    return hint, subprocess.PIPE, subprocess.SubprocessError\n"
        )
        self.assertEqual(launch_sites(source), [])

    def test_a_call_through_the_helper_is_not_reported(self):
        source = (
            "from . import launch\n"
            "def probe():\n"
            "    return launch.run(['ffprobe', '-version'])\n"
        )
        self.assertEqual(launch_sites(source), [])
        self.assertEqual(calls_in(source, _HELPER_MODULES, _HELPERS.__contains__),
                         [("probe", 3, "launch.run")])

    def test_the_real_package_is_what_is_being_read(self):
        """A scanner pointed at nothing finds nothing and looks exactly like a green
        gate — the failure mode of every check that walks a directory."""
        self.assertGreater(len(package_sources()), 20)
        self.assertIn("launch.py", [_relative(path) for path in package_sources()])


class TestWhenTheWindowIsHidden(unittest.TestCase):
    """Only when this process has no console of its own — see the module docstring."""

    def test_no_console_means_the_flag(self):
        self.assertEqual(launch.creation_flags(console=lambda: False),
                         launch.CREATE_NO_WINDOW)

    def test_a_console_means_no_flag(self):
        self.assertEqual(launch.creation_flags(console=lambda: True), 0)

    def test_off_windows_there_is_never_a_flag(self):
        """Linux and macOS answer yes without asking anything, so `creation_flags` is 0
        there and no caller needs a branch of its own. Asserted through the probe rather
        than by patching `os.name`: the platform is read once, as the default of
        `has_console`, and a patch of the module attribute would not reach it."""
        for os_name in ("posix", "java"):
            with self.subTest(os_name=os_name):
                self.assertTrue(launch.has_console(os_name))
                self.assertEqual(
                    launch.creation_flags(console=lambda: launch.has_console(os_name)), 0)

    def test_the_constant_is_the_one_windows_defines(self):
        """Written out because `subprocess.CREATE_NO_WINDOW` exists only on Windows —
        which is also why it has to be checked against it there."""
        self.assertEqual(launch.CREATE_NO_WINDOW, 0x08000000)
        if sys.platform == "win32":
            self.assertEqual(launch.CREATE_NO_WINDOW, subprocess.CREATE_NO_WINDOW)

    def test_a_process_with_no_console_is_recognised(self):
        """`GetConsoleWindow` returns NULL when the process is attached to no console,
        which is what `pythonw.exe` looks like."""
        kernel32 = mock.Mock()
        kernel32.GetConsoleWindow.return_value = 0
        with mock.patch.dict(sys.modules,
                             {"ctypes": mock.Mock(windll=mock.Mock(kernel32=kernel32))}):
            self.assertFalse(launch.has_console("nt"))
            kernel32.GetConsoleWindow.return_value = 0x1234
            self.assertTrue(launch.has_console("nt"))

    def test_a_question_that_cannot_be_asked_leaves_things_as_they_were(self):
        """A yes, so the worst case is the behaviour of before this feature. A no would
        hide the output of `uv` in somebody's terminal, which is strictly worse."""
        with mock.patch.dict(sys.modules, {"ctypes": mock.Mock(windll=None)}):
            self.assertTrue(launch.has_console("nt"))
        broken = mock.Mock()
        broken.windll.kernel32.GetConsoleWindow.side_effect = OSError("no kernel32")
        with mock.patch.dict(sys.modules, {"ctypes": broken}):
            self.assertTrue(launch.has_console("nt"))

    def test_the_real_probe_answers_without_raising(self):
        self.assertIn(launch.has_console(), (True, False))


class TestTheHelperPassesItOn(unittest.TestCase):
    """It is a wrapper, so it has to be transparent about everything except the flag."""

    def _run(self, *, console: bool, **kwargs):
        with mock.patch.object(launch, "has_console", return_value=console), \
                mock.patch.object(subprocess, "run") as run:
            launch.run(["exiftool", "-ver"], capture_output=True, **kwargs)
        return run.call_args

    def test_the_flag_is_added_when_there_is_no_console(self):
        args, kwargs = self._run(console=False)
        self.assertEqual(args, (["exiftool", "-ver"],))
        self.assertEqual(kwargs["creationflags"], launch.CREATE_NO_WINDOW)
        self.assertTrue(kwargs["capture_output"])

    def test_nothing_is_added_when_there_is_a_console(self):
        """Not `creationflags=0` either: on POSIX `subprocess` refuses anything but the
        absence of it, and this is the branch a Linux run takes."""
        _args, kwargs = self._run(console=True)
        self.assertNotIn("creationflags", kwargs)

    def test_a_caller_that_has_its_own_flags_keeps_them(self):
        """`creationflags` is a bit field: the no-window bit is ADDED, never assigned."""
        below_normal = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
        _args, kwargs = self._run(console=False, creationflags=below_normal)
        self.assertEqual(kwargs["creationflags"], below_normal | launch.CREATE_NO_WINDOW)

    def test_popen_is_wrapped_the_same_way(self):
        with mock.patch.object(launch, "has_console", return_value=False), \
                mock.patch.object(subprocess, "Popen") as popen:
            launch.popen(("exiftool", "-stay_open", "True"), stdin=subprocess.PIPE)
        args, kwargs = popen.call_args
        # Handed on as it arrived: `_NVIDIA_SMI_CMD` is a tuple and stays one.
        self.assertEqual(args, (("exiftool", "-stay_open", "True"),))
        self.assertEqual(kwargs["creationflags"], launch.CREATE_NO_WINDOW)
        self.assertEqual(kwargs["stdin"], subprocess.PIPE)

    def test_it_really_starts_a_process(self):
        """Every other test here mocks `subprocess`; one has to not."""
        done = launch.run([sys.executable, "-c", "print('sorta')"],
                          capture_output=True, text=True)
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stdout.strip(), "sorta")

    @unittest.skipUnless(sys.platform == "win32", "the flag only exists on Windows")
    def test_windows_accepts_the_flag_on_a_real_child(self):
        """The acceptance criterion, as close as a suite can get to it: the wrapper with
        the hidden branch forced still runs the program and still brings its output back.
        A flag Windows refused would raise here."""
        with mock.patch.object(launch, "has_console", return_value=False):
            done = launch.run([sys.executable, "-c", "print('hidden')"],
                              capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(), "hidden")


class TestTheCallSitesUseIt(unittest.TestCase):
    """What the wired-up modules actually do — the watchdog reads the source, this runs
    it."""

    def test_the_nvidia_probe_goes_through_the_helper(self):
        with mock.patch.object(launch, "run") as run:
            diagnostics._run_nvidia_smi()
        command, kwargs = run.call_args
        self.assertEqual(command[0], diagnostics._NVIDIA_SMI_CMD)
        self.assertEqual(kwargs["timeout"], diagnostics._NVIDIA_SMI_TIMEOUT_S)

    def test_the_exiftool_version_probe_goes_through_the_helper(self):
        with mock.patch.object(launch, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "13.10\n", "")
            self.assertTrue(exif._starts("exiftool.exe"))
        self.assertEqual(run.call_args[0][0], ["exiftool.exe", "-ver"])

    def test_the_stay_open_session_goes_through_the_helper(self):
        session = exif.ExifToolSession()
        with mock.patch.object(launch, "popen") as popen:
            popen.return_value = mock.Mock(poll=mock.Mock(return_value=None))
            session._ensure()
        self.assertIn("-stay_open", popen.call_args[0][0])

    def test_the_one_shot_fallback_goes_through_the_helper(self):
        with mock.patch.object(launch, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            self.assertEqual(exif.read_batch_exiftool([Path(os.getcwd()) / "a.jpg"]), {})
        self.assertTrue(run.called)

    def test_the_wizard_install_goes_through_the_helper(self):
        with mock.patch.object(launch, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 3)
            self.assertEqual(wizard.run_install(("uv", "pip", "install", "torch")), 3)
        self.assertEqual(run.call_args[0][0], ("uv", "pip", "install", "torch"))
        self.assertIs(run.call_args[1]["check"], False)

    def test_a_terminal_that_runs_the_wizard_still_sees_uv(self):
        """The acceptance criterion `sorta-setup` carries: our console is not hidden."""
        with mock.patch.object(launch, "has_console", return_value=True), \
                mock.patch.object(subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            wizard.run_install(["uv", "pip", "install", "torch"])
        self.assertNotIn("creationflags", run.call_args[1])

    def test_a_shortcut_that_runs_the_wizard_hides_the_window(self):
        with mock.patch.object(launch, "has_console", return_value=False), \
                mock.patch.object(subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            wizard.run_install(["uv", "pip", "install", "torch"])
        self.assertEqual(run.call_args[1]["creationflags"], launch.CREATE_NO_WINDOW)

    def test_an_uv_that_cannot_be_started_is_still_an_exit_code(self):
        """The wrapper must not change what `run_install` promises: never a raise."""
        with mock.patch.object(launch, "run", side_effect=OSError("no uv")):
            self.assertEqual(wizard.run_install(["uv"]), 1)


if __name__ == "__main__":
    unittest.main()
