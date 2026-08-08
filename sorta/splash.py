"""The "Sorta is starting" window, kept apart from everything heavy.

`sorta.launcher` opens it before importing the program: from `sorta.tray` it could not
appear until `sorta.ui` had been imported, which is 3.59 s on its own (payload
interpreter, warm cache) and far longer on a cold disk.
"""
from __future__ import annotations

import logging
import subprocess
import sys

from . import i18n, launch

_LOG = logging.getLogger(__name__)

# The name and the line the window shows. `_SPLASH_NAME` is the product, not a translated
# caption: it is the same word the tray tooltip, the page and the shortcut carry.
_SPLASH_NAME = "Sorta"
# How long to wait for the window to take the hint and close itself before ending it.
# Short on purpose — the tab is already open by then, and nothing is lost by killing a
# window that is about to be redundant.
_SPLASH_CLOSE_TIMEOUT_S = 3.0
# The window itself, in a process that has nothing else to do. Deliberately tiny and
# import-light: `tkinter` plus `ttk` for the bar, no sorta module at all, so it draws
# while THIS process is still importing whatever it imports.
#
# The bar is indeterminate because there is no honest percentage to show — see the note in
# `ui/common.py`. It closes on EOF of its own stdin, which is how it goes away if the
# program that opened it dies without asking.
_SPLASH_SCRIPT = (
    "import sys, threading, tkinter\n"
    "from tkinter import ttk\n"
    "root = tkinter.Tk()\n"
    "root.title(sys.argv[1])\n"
    "root.resizable(False, False)\n"
    "frame = ttk.Frame(root, padding=28)\n"
    "frame.pack()\n"
    "ttk.Label(frame, text=sys.argv[1], font=('', 18, 'bold')).pack()\n"
    "ttk.Label(frame, text=sys.argv[2]).pack(pady=(10, 14))\n"
    "bar = ttk.Progressbar(frame, mode='indeterminate', length=280)\n"
    "bar.pack()\n"
    "bar.start(12)\n"
    "def watch():\n"
    "    try:\n"
    "        sys.stdin.readline()\n"
    "    except Exception:\n"
    "        pass\n"
    "    root.after(0, root.destroy)\n"
    "threading.Thread(target=watch, daemon=True).start()\n"
    "try:\n"
    "    root.attributes('-topmost', True)\n"
    "    root.eval('tk::PlaceWindow . center')\n"
    "except Exception:\n"
    "    pass\n"
    "root.mainloop()\n"
)


class _Splash:
    """The "Sorta is starting" window — a handle on the process that draws it."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._closed = False

    def close(self) -> None:
        """Take the window away. Idempotent, and never raises at the caller.

        EOF on its stdin first, because that asks the window to destroy ITSELF, which is
        the only way tkinter likes to be shut down; `terminate` is the second line, for a
        child that cannot read its stdin (a launcher that left it closed) or is wedged.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=_SPLASH_CLOSE_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            pass
        except OSError:  # the child is already gone
            return
        try:
            self._process.terminate()
        except OSError:
            pass


def open_splash(lang: i18n.Lang | None = None) -> _Splash | None:
    """Put a window on the screen now, and return the handle that closes it.

    None means there is no window and the launch carries on exactly as it did before: a
    machine without tkinter, without a display or without a window manager is a machine
    where nobody can be shown anything, and that is never a reason not to start. tkinter
    is in the installer payload, so on the machine this feature is for there is one.

    `hide_window=True` and not the plain helper: F228 hides a child's console only when
    THIS process has none, which is right for `uv` in a terminal somebody is reading and
    wrong here — both streams of the splash go to DEVNULL, so its console could only ever
    be an empty rectangle beside the window it belongs to.
    """
    try:
        process = launch.popen(
            [sys.executable, "-c", _SPLASH_SCRIPT, _SPLASH_NAME,
             i18n.cli_text("cli.tray.starting", i18n.normalize_lang(lang))],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, hide_window=True)
    except Exception as exc:  # no interpreter to spawn, no permission, no tkinter
        _LOG.warning("splash: could not show the starting window (%s)", exc)
        return None
    return _Splash(process)
