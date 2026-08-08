"""F207: the tray icon — open the window, close the program, and nothing else.

Somebody who installed Sorta with an installer does not keep a terminal open. `sorta ui`
prints its address to the console and lives until Ctrl+C, which for an installed program
means the address is nowhere to be read and there is nothing to press to close it. This
module is the second entry point that closes exactly those holes — plus the third one a
shortcut brings with it: a second double-click must not meet a busy-port error.

What it is NOT: it starts no runs, shows no progress, sends no notifications and installs
no autostart. It opens a window and it closes the program.

Three things are borrowed rather than rewritten:

* the server. `ui.build_server` builds it and a thread of this process serves it. The
  `sorta ui` path is not touched at all — this is a SECOND entry point, not a
  replacement for the first one;
* the exit. Quitting goes through `POST /api/quit` (F209), the very request the "Quit"
  button of the page sends. That is what keeps the protection of a run ONE rule instead
  of two implementations of it: the server answers 409 `run_in_progress` while a run, a
  layout or a rollback is in flight, and only a second request carrying
  `{"confirm": true}` interrupts it. The menu item asks the question; it does not decide.
  Nothing here kills anything: `/api/quit` ends `serve_forever` the Ctrl+C way, and the
  `finally` below closes the server socket and the index connection;
* the picture. `sorta/web/favicon.ico` is what the page and the browser tab already show
  — three different images of one program read as three programs.

A machine with no tray (a server, an SSH session, a Linux desktop without an indicator)
keeps serving without an icon. Every step of building the icon is therefore allowed to
fail into `TrayUnavailable`, and none of them may take the server down with it.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Sequence

from . import i18n, ui
from .config import configure_logging, load_config
from .db import connect
from .diagnostics import warn_if_geo_data_missing, warn_if_gpu_mismatch
from .runlog import log_environment

_LOG = logging.getLogger(__name__)

DEFAULT_CONFIG = "config.yaml"
# The same file `sorta ui` serves to the browser tab, read from the package rather than
# from a copy: an icon that has to be kept in sync with another icon is two icons.
ICON_PATH = Path(__file__).resolve().parent / "web" / "favicon.ico"
# "Is the program on this port ours?" — a route of this server that reads nothing and
# needs no body. A foreign server may well answer 200 on it, so the shape of the answer
# is checked too: `/api/env` carries `gpu_profile`, which nothing else answers with.
# (F217 added two more fields to that route; the probe reads the one field, so a payload
# that grows stays recognisable.)
PROBE_ROUTE = "/api/env"
PROBE_FIELD = "gpu_profile"
PROBE_TIMEOUT = 2.0
QUIT_TIMEOUT = 15.0

# Who is on the port, before anything is bound to it. Asked rather than inferred from a
# failed bind, and that is not belt-and-braces: `http.server` sets `SO_REUSEADDR`, and on
# Windows that option lets a second bind to a LIVE port SUCCEED — the new server steals
# the port from the running one instead of raising. So "the port is busy" is a question a
# `connect()` answers on every platform, while a failed bind answers it only on POSIX.
PORT_FREE = "free"
PORT_OURS = "sorta"
PORT_STRANGER = "stranger"

# F209 names what can be in flight (`process`/`sort`/`undo`); this is the question the
# tray asks about each of them. The pairing is checked by the suite the same way the
# page's own questions are: a fourth long operation added tomorrow would otherwise come
# back in `running` with no question behind it, and the menu would quietly do nothing.
QUIT_QUESTION_KEYS: dict[str, str] = {
    name: f"cli.tray.quit_{name}" for name in ui.QUIT_RUNNING_NAMES
}
# What to ask about an operation this build has no question for. A generic question is
# still a question — the one thing that may not happen is interrupting without asking.
QUIT_QUESTION_FALLBACK = "cli.tray.quit_running"


class TrayUnavailable(RuntimeError):
    """This machine has no tray (or no library for it) — serve without an icon.

    Requirement 4 of the brief in one exception type: the absence of an indicator is a
    property of somebody's desktop, never a reason for the program not to start.
    """


def url_for(port: int) -> str:
    """The address the icon opens and shows in its tooltip."""
    return f"http://127.0.0.1:{port}/"


# --- F225: a windowed interpreter has no streams, and every library assumes it has ------
#
# The shortcut runs `pythonw.exe -m sorta.tray`, and a windowed interpreter starts with
# `sys.stdout` and `sys.stderr` set to None — there is no console for them to point at.
# `_say` below has known that since F207 and guards ITS OWN lines. Nothing guarded
# anybody else's, and the run happens inside THIS process (`ui/process.py` runs the
# pipeline on a thread of it), so the first library that prints takes the run down with
# it. On a clean VM on 2026-08-08 that was huggingface_hub drawing its progress bar:
#
#     Failed to download weights for tag 'openai' ...
#     Last error: 'NoneType' object has no attribute 'write'
#
# — a 1.6 GB download that failed with the network, the certificates and the disk all
# perfectly fine, because there was nowhere to print the percentage to.
#
# Hence the fix is HERE, at the entry point, and not at the call that raised: the next
# library to print a line comes with the next version of transformers, and it must not be
# a second report of this defect. Both streams are made to exist and both go to the run
# log (`%LOCALAPPDATA%\\sorta\\logs\\sorta.log`) — a line that cannot be shown to anybody
# is still the line somebody reads afterwards to find out what happened.


# How often a REDRAWN line reaches the log. A progress bar rewrites one line hundreds of
# times per gigabyte, and a log record per redraw would bury the run in its own progress;
# a line that ends properly is never held back by this.
_REDRAW_SECONDS = 5.0


class _LogStream(io.TextIOBase):
    """A text stream that has nowhere to print, so it writes the run log instead.

    Line-buffered by hand, and the two kinds of line are treated differently: a line
    ending in `\\n` is something somebody wrote and goes to the log as it is, while one
    ending in `\\r` is a progress bar overwriting itself and goes at most every
    `_REDRAW_SECONDS`. Anything left over is written when the writer flushes.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._pending = ""
        self._last_redraw = 0.0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        # Some libraries write bytes to a stream they believe is a console; refusing
        # would be the crash this class exists to prevent.
        data = text if isinstance(text, str) else bytes(text).decode("utf-8", "replace")
        self._pending += data
        while True:
            cuts = [at for at in (self._pending.find("\n"), self._pending.find("\r"))
                    if at >= 0]
            if not cuts:
                break
            at = min(cuts)
            line, terminator = self._pending[:at], self._pending[at]
            self._pending = self._pending[at + 1:]
            self._emit(line, redraw=terminator == "\r")
        return len(data)

    def flush(self) -> None:
        if self._pending:
            line, self._pending = self._pending, ""
            self._emit(line)

    def isatty(self) -> bool:
        """No. A progress bar that believes otherwise redraws a line nobody can see."""
        return False

    def _emit(self, line: str, redraw: bool = False) -> None:
        if not line.strip():
            return
        if redraw:
            now = time.monotonic()
            if now - self._last_redraw < _REDRAW_SECONDS:
                return
            self._last_redraw = now
        _LOG.info("%s: %s", self._name, line.rstrip())


def ensure_streams() -> tuple[str, ...]:
    """Give this process the two standard streams, if the launcher left it without them.

    Returns the names that had to be replaced — () on an ordinary console, where the real
    streams are left exactly as they are. Also fills `sys.__stdout__`/`sys.__stderr__`,
    which a library reaching past the current streams would otherwise find as None.
    """
    replaced: list[str] = []
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, _LogStream(name))
            replaced.append(name)
        if getattr(sys, f"__{name}__", None) is None:
            setattr(sys, f"__{name}__", getattr(sys, name))
    return tuple(replaced)


def _say(text: str, *, error: bool = False) -> None:
    """Print — on a machine that may have nowhere to print to.

    This entry point exists for a Sorta started from a shortcut, and a windowed launcher
    (`pythonw`, the `gui-scripts` wrapper) leaves the standard streams closed or absent.
    A line that cannot be shown is not a reason to fail, so it also goes to the log,
    which is where it can be read afterwards anyway.
    """
    stream = sys.stderr if error else sys.stdout
    try:
        if stream is not None:
            print(text, file=stream)
    except (OSError, ValueError):  # a closed or detached stream under pythonw
        pass
    if error:
        _LOG.error(text)
    else:
        _LOG.info(text)


def sorta_is_serving(port: int, *, timeout: float = PROBE_TIMEOUT) -> bool:
    """Is the program holding `port` OUR server?

    Requirement 3: clicking a shortcut twice is normal, and the second click must not
    show an error. Asked over HTTP because that is the only thing that tells our server
    apart from whatever else may be listening — a port number cannot.
    """
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{PROBE_ROUTE}", timeout=timeout) as resp:
            if resp.status != HTTPStatus.OK:
                return False
            payload = json.loads(resp.read())
    except (OSError, ValueError):  # urllib.error.*/timeouts subclass OSError; JSON, ValueError
        return False
    return isinstance(payload, dict) and PROBE_FIELD in payload


def port_holder(port: int, *, timeout: float = PROBE_TIMEOUT) -> str:
    """Who holds `port` — `PORT_FREE`, `PORT_OURS` or `PORT_STRANGER`.

    Two questions in the order they can be answered: is anything listening at all (a
    plain TCP connect), and if so, is it Sorta (`/api/env`). Port 0 is "let the OS
    choose" and is nobody's by definition.
    """
    if port <= 0:
        return PORT_FREE
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
    except OSError:
        return PORT_FREE
    return PORT_OURS if sorta_is_serving(port, timeout=timeout) else PORT_STRANGER


def request_quit(port: int, *, confirm: bool = False,
                 timeout: float = QUIT_TIMEOUT) -> tuple[int, dict]:
    """`POST /api/quit` exactly as the page sends it (F208: the content type is what
    gets it served). Returns (status, body) — including the 409 that protects a run."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/quit",
        data=json.dumps({"confirm": confirm}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return int(resp.status), _json_or_empty(resp.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), _json_or_empty(exc.read())


def _json_or_empty(raw: bytes) -> dict:
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def quit_question(running: object, lang: i18n.Lang) -> str:
    """The sentence for what is in flight, by the name `/api/quit` reported."""
    key = QUIT_QUESTION_KEYS.get(running if isinstance(running, str) else "",
                                 QUIT_QUESTION_FALLBACK)
    return i18n.cli_text(key, lang)


def ask_yes_no(title: str, question: str) -> bool:
    """The confirmation dialog, drawn with tkinter — stdlib, so no second dependency.

    Anything that goes wrong here answers NO. A machine where no dialog can be drawn is
    a machine where the person cannot be asked, and the rule of this feature is that a
    run is never interrupted without an answer. The window is created and destroyed
    inside the call: the tray has no window of its own to hang it on.
    """
    try:
        import tkinter
        from tkinter import messagebox
    except ImportError as exc:  # pragma: no cover — a python built without tkinter
        _LOG.warning("tray: no dialog toolkit, refusing to quit silently (%s)", exc)
        return False
    try:
        root = tkinter.Tk()
        try:
            root.withdraw()
            # The tray is behind everything by definition; a question nobody sees is a
            # question nobody answered.
            root.attributes("-topmost", True)
            return bool(messagebox.askyesno(title, question))
        finally:
            root.destroy()
    except Exception as exc:  # no display, no window manager, a headless session
        _LOG.warning("tray: could not ask for a confirmation (%s)", exc)
        return False


def quit_program(port: int, lang: i18n.Lang, *,
                 ask: Callable[[str, str], bool] = ask_yes_no) -> bool:
    """Ask the server to close. True — it agreed and is stopping.

    Requirement 2 of the brief lives here and nowhere else. A 409 whose reason is
    `run_in_progress` is turned into a QUESTION — not a warning followed by quitting
    anyway — and only a yes sends the second request, the one carrying the confirmation.
    A no leaves the run and the server exactly as they were: the refusal never touched
    the cancel flag.
    """
    # A menu callback runs on the tray library's own loop, so nothing here may throw at
    # it: a socket that refuses (the server is already gone, say) is an answer like any
    # other and is reported as one.
    try:
        status, payload = request_quit(port)
    except OSError as exc:
        _say(i18n.cli_text("cli.tray.quit_failed", lang, status=exc), error=True)
        return False
    if status == HTTPStatus.OK:
        return True
    if status != HTTPStatus.CONFLICT or payload.get("reason") != ui.QUIT_RUN_IN_PROGRESS:
        _say(i18n.cli_text("cli.tray.quit_failed", lang, status=status), error=True)
        return False
    if not ask(i18n.cli_text("cli.tray.quit_title", lang),
               quit_question(payload.get("running"), lang)):
        return False
    try:
        status, _payload = request_quit(port, confirm=True)
    except OSError as exc:
        _say(i18n.cli_text("cli.tray.quit_failed", lang, status=exc), error=True)
        return False
    if status != HTTPStatus.OK:
        _say(i18n.cli_text("cli.tray.quit_failed", lang, status=status), error=True)
        return False
    return True


def icon_image() -> Any:
    """The `favicon.ico` of the page, as an image pystray can show.

    `convert` forces the decode here rather than inside the tray library: a picture that
    cannot be read is one more way this machine has no icon, and it has to be reported
    as that instead of surfacing later as a crash of the menu.
    """
    from PIL import Image

    try:
        with Image.open(ICON_PATH) as image:
            return image.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise TrayUnavailable(f"cannot read {ICON_PATH.name}: {exc}") from exc


def build_icon(port: int, lang: i18n.Lang, *,
               on_open: Callable[[], None],
               on_quit: Callable[[], None]) -> Any:
    """The icon and its two-item menu, or `TrayUnavailable` on a machine without one.

    "Open" is the DEFAULT item, which is what makes a double-click on the icon open the
    window — the same action, reachable two ways, rather than two behaviours to keep in
    step. The tooltip carries the address so it can be read and copied.
    """
    try:
        import pystray
    except ImportError as exc:
        raise TrayUnavailable(f"pystray is not installed: {exc}") from exc
    image = icon_image()
    try:
        menu = pystray.Menu(
            pystray.MenuItem(i18n.cli_text("cli.tray.open", lang),
                             lambda _icon, _item: on_open(), default=True),
            pystray.MenuItem(i18n.cli_text("cli.tray.quit", lang),
                             lambda _icon, _item: on_quit()),
        )
        return pystray.Icon("sorta", icon=image,
                            title=i18n.cli_text("cli.tray.tooltip", lang,
                                                url=url_for(port)),
                            menu=menu)
    except Exception as exc:  # a backend that refuses to load (no indicator, no DISPLAY)
        raise TrayUnavailable(f"no tray on this system: {exc}") from exc


def _stop_icon_when_the_server_stops(serving: threading.Thread, icon: Any) -> None:
    """Take the icon away when the server is gone, whichever way it went.

    The menu is not the only way out: the page has its own "Quit" button (F209), and
    both end in the same `serve_forever` returning. An icon still sitting in the tray of
    a program that has closed is a shortcut to nothing.
    """
    serving.join()
    try:
        icon.stop()
    except Exception as exc:  # the icon may not have been shown yet
        _LOG.warning("tray: could not remove the icon (%s)", exc)


def start(cfg: Any, conn: Any, *, port: int = ui.DEFAULT_PORT,
          config_path: str | Path | None = None,
          open_browser: bool = True,
          ask: Callable[[str, str], bool] = ask_yes_no,
          icon_factory: Callable[..., Any] = build_icon) -> int:
    """Serve, with an icon in the tray if this machine has one. The exit code of `main`.

    `ask`/`icon_factory` are injected by the tests: what is worth pinning is that the
    question is asked and that a machine without a tray keeps serving, and neither of
    those is checkable through somebody else's desktop.
    """
    lang = i18n.normalize_lang(getattr(cfg, "language", None))
    try:
        holder = port_holder(port)
        if holder != PORT_FREE:
            return _busy_port(port, lang, holder, open_browser=open_browser)
        try:
            httpd = ui.build_server(cfg, conn, port=port, config_path=config_path)
        except OSError as exc:
            # Somebody took the port between the question above and this bind. Rare, and
            # answered by asking the same question again rather than by guessing.
            _LOG.warning("tray: could not bind port %s (%s)", port, exc)
            return _busy_port(port, lang, port_holder(port), open_browser=open_browser)
        log_environment()  # F69: one environment header per server start
        warn_if_geo_data_missing()  # F65: an unreadable geo base empties every place
        port = httpd.server_port  # port=0 (the tests) -> whatever the OS handed out
        url = url_for(port)
        serving = threading.Thread(target=httpd.serve_forever, daemon=True)
        serving.start()
        try:
            if open_browser:
                webbrowser.open(url)
            _serve_until_closed(port, lang, url, serving, ask=ask,
                                icon_factory=icon_factory)
        finally:
            httpd.server_close()
        return 0
    finally:
        conn.close()


def _serve_until_closed(port: int, lang: i18n.Lang, url: str, serving: threading.Thread,
                        *, ask: Callable[[str, str], bool],
                        icon_factory: Callable[..., Any]) -> None:
    """Block until the server has stopped — with an icon if there is one, without it
    otherwise. The `sorta ui` behaviour is the fallback, word for word: the address on
    the console and a wait that Ctrl+C ends."""
    try:
        icon = icon_factory(port, lang,
                            on_open=lambda: webbrowser.open(url),
                            on_quit=lambda: quit_program(port, lang, ask=ask))
    except TrayUnavailable as exc:
        _say(i18n.cli_text("cli.tray.no_icon", lang, reason=exc))
        _say(i18n.cli_text("cli.ui.serving", lang, url=url))
        serving.join()
        return
    _say(i18n.cli_text("cli.tray.serving", lang, url=url))
    threading.Thread(target=_stop_icon_when_the_server_stops,
                     args=(serving, icon), daemon=True).start()
    try:
        icon.run()
    except Exception as exc:
        # Not the same failure as the one above, and it has to be survived just as
        # thoroughly: a Linux backend that only finds out at run time that there is no
        # indicator to attach to raises HERE, with the server already serving. The
        # `join` below is what keeps that from becoming an exit — the program carries
        # on without the icon, which is requirement 4.
        _say(i18n.cli_text("cli.tray.no_icon", lang, reason=exc))
        _say(i18n.cli_text("cli.ui.serving", lang, url=url))
    serving.join()


def _busy_port(port: int, lang: i18n.Lang, holder: str, *, open_browser: bool) -> int:
    """The port is taken. Requirement 3: by WHOM decides what happens next.

    Ours — this is the second click on the shortcut, which is a normal thing to do, so
    the window opens and this process leaves quietly with a zero exit code. Anybody
    else's — a clear error, because a port held by a stranger is a thing the person has
    to be told about, together with what to do next.
    """
    if holder == PORT_OURS:
        _say(i18n.cli_text("cli.tray.already_running", lang, url=url_for(port)))
        if open_browser:
            webbrowser.open(url_for(port))
        return 0
    _say(i18n.cli_text("cli.tray.port_busy", lang, port=port), error=True)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """The options of `sorta-tray`, mirroring the ones `sorta ui` takes."""
    parser = argparse.ArgumentParser(
        prog="sorta-tray",
        description="Sorta in the tray: open the window, close the program.")
    parser.add_argument("--port", type=int, default=ui.DEFAULT_PORT,
                        help="TCP port on 127.0.0.1 (default: %(default)s)")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG,
                        help="path to config.yaml (default: %(default)s)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser at start-up")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The `sorta-tray` entry point. Everything `sorta ui` does at start-up, plus a
    picture in the tray.

    F225: the streams first, before a single line of this is read or a single module of
    the pipeline is imported. This is the entry point of the windowed launcher, and from
    here on the process is one where any library may print — see `ensure_streams`.
    """
    ensure_streams()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    warn_if_gpu_mismatch()  # F63: loud if torch is CPU-only while a GPU is expected
    conn = connect(cfg.database)
    return start(cfg, conn, port=args.port, config_path=args.config,
                 open_browser=not args.no_browser)


if __name__ == "__main__":  # pragma: no cover — the console-script wrapper calls main()
    sys.exit(main())
