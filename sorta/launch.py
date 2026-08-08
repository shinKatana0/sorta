"""F228: the one place that starts a subprocess, so no window opens that nobody asked for.

The shortcut runs `pythonw.exe`, a process with no console of its own. On Windows a
CONSOLE program has to have a console to write to, so when the parent has none the
loader CREATES ONE — a new black window per child. Nothing in the product passed
`CREATE_NO_WINDOW`, and four places start children: the `nvidia-smi` probe at start-up,
`exiftool -ver` on the first read of metadata, up to eight `exiftool -stay_open` sessions
on the index stage, and `uv` in the wizard.

The owner met that as a window that opened while the interface said a model was being
downloaded and in which nothing then happened. The window had nothing to do with the
download — but there is nowhere for a person to learn that: they were waiting for
progress and were shown an empty console.

Why a module and not the flag written four times: the fifth call arrives without it. That
is not a prediction, it is the history of this defect — F226 added the fourth one the day
before it was reported. `tests/test_no_console_nobody_asked_for.py` reads the package with
`ast` and fails on any launch that does not come through here.

**Hiding is a question about the parent, not about the platform.** `sorta-setup` typed
into a terminal shows the output of `uv` in THAT console, and hiding the window there
would throw away the install log somebody is watching. So what is asked below is "does
this process have a console at all" — with none there is nothing to lose and the child is
hidden; with one it is somebody's, and the child writes into it. That is deliberately NOT
`wizard.owns_console`, which answers the neighbouring question "would this console die
with this process" (one process attached, so the window was created for us): a terminal
that has our process alone would answer yes there, and hiding `uv` in it is exactly the
mistake this paragraph exists to prevent.

On Linux and macOS there is no such flag and no such window: `creation_flags()` is 0,
`run`/`popen` are `subprocess.run`/`subprocess.Popen` with nothing added, and no caller
carries a branch for it.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Sequence

# `subprocess.CREATE_NO_WINDOW` is defined only on Windows, so the value is written out
# rather than read off the module: an attribute lookup would make this file fail to
# import on Linux, which is the opposite of harmless.
CREATE_NO_WINDOW = 0x08000000


def has_console(os_name: str = os.name) -> bool:
    """Does this process have a console at all? Then a child of it opens no new window.

    `GetConsoleWindow` returns the window of the console this process is attached to, and
    NULL when it is attached to none — which is what a `pythonw.exe` start looks like.
    It is asked per call rather than once at import: a process can acquire a console
    later (`AllocConsole`), and the answer costs one call into kernel32.

    Anything that cannot be asked — a Python without `ctypes.windll`, a kernel32 that
    will not answer — is a yes, and that direction is the safe one: the worst it can do
    is leave today's behaviour in place, while a wrong no would silently swallow the
    output of `uv` in a terminal somebody is reading.
    """
    if os_name != "nt":
        return True
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return True
        return bool(windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 — no answer is "leave it as it was", never a crash
        return True


def creation_flags(*, console: Callable[[], bool] | None = None) -> int:
    """`creationflags` for a child of this process — 0 unless the window has to be hidden.

    `console` is injectable so the suite can ask the question for a machine other than
    this one; by default it is `has_console`, which also answers for the platform (on
    Linux and macOS it is always a yes and this is always 0).
    """
    probe = has_console if console is None else console
    return 0 if probe() else CREATE_NO_WINDOW


def _windowless(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The caller's keyword arguments with the no-window flag added to `creationflags`.

    ADDED and not assigned: `creationflags` is a bit field, and a caller that passes one
    of its own must keep it. When there is nothing to add the key is left out entirely —
    on POSIX `subprocess` refuses any non-zero value, and passing a zero it never asked
    for is noise in the one place that must stay boring.
    """
    flags = int(kwargs.pop("creationflags", 0)) | creation_flags()
    return {**kwargs, "creationflags": flags} if flags else kwargs


def run(command: Sequence[str], **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """`subprocess.run`, without a console window when this process has no console.

    The command is handed on exactly as it arrived — a list, a tuple, whatever the caller
    keeps it in. Normalising it here would be a second thing this wrapper does, and the
    only way a wrapper stays trustworthy is by doing one.
    """
    return subprocess.run(command, **_windowless(kwargs))


def popen(command: Sequence[str], **kwargs: Any) -> "subprocess.Popen[Any]":
    """`subprocess.Popen`, without a console window when this process has no console."""
    return subprocess.Popen(command, **_windowless(kwargs))
