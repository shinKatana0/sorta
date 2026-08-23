"""F207: the tray icon — open the window, close the program, and nothing else.

Somebody who installed Sorta with an installer keeps no terminal open, so the address
`sorta ui` prints is nowhere to be read and there is nothing to press to close it. This
is the second entry point that closes those holes, plus the one a shortcut adds: a second
double-click must not meet a busy-port error. It starts no runs, shows no progress, sends
no notifications and installs no autostart.

Three things are borrowed rather than rewritten: the SERVER (`ui.build_server`, served on
a thread of this process, with the `sorta ui` path untouched); the PICTURE
(`sorta/web/favicon.ico`, because three images of one program read as three programs);
and the EXIT. Quitting goes through `POST /api/quit` (F209), the request the page's own
"Quit" button sends, so the protection of a run is ONE rule: the server answers 409
`run_in_progress` while a run, a layout or a rollback is in flight, and only a second
request carrying `{"confirm": true}` interrupts it. The menu item asks the question, it
does not decide, and nothing here kills anything — `/api/quit` ends `serve_forever` the
Ctrl+C way and the `finally` below closes the socket and the index connection.

A machine with no tray (a server, an SSH session, a Linux desktop without an indicator)
keeps serving without an icon, so every step of building it may fail into
`TrayUnavailable` and none of them may take the server down.

F227 changed the ORDER below and only the order — see the measurement above
`_SPLASH_NAME`.
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
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import i18n, ui
from .splash import _Splash
from .config import configure_logging, load_config
from .db import connect
from .diagnostics import warn_if_geo_data_missing, warn_if_gpu_mismatch
from .faults import Fault
from .runlog import log_environment

_LOG = logging.getLogger(__name__)

DEFAULT_CONFIG = "config.yaml"
# The same file `sorta ui` serves to the browser tab, read from the package rather than
# from a copy: an icon that has to be kept in sync with another icon is two icons.
ICON_PATH = Path(__file__).resolve().parent / "web" / "favicon.ico"
# "Is the program on this port ours?" — a route that reads nothing and needs no body. A
# foreign server may answer 200, so the SHAPE is checked too: `/api/env` carries
# `gpu_profile`. The probe reads that one field, so a payload that grows stays
# recognisable.
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
# tray asks about each. The suite checks the pairing: a fourth long operation would
# otherwise come back in `running` with no question behind it and the menu would do
# nothing.
QUIT_QUESTION_KEYS: dict[str, str] = {
    name: f"cli.tray.quit_{name}" for name in ui.QUIT_RUNNING_NAMES
}
# What to ask about an operation this build has no question for. A generic question is
# still a question — the one thing that may not happen is interrupting without asking.
QUIT_QUESTION_FALLBACK = "cli.tray.quit_running"

# --- F227: the launch says it is launching ------------------------------------------
#
# Measured with the interpreter from the installer payload, on a fast machine:
#
#     import sorta.tray        1.53 s
#     warn_if_gpu_mismatch     3.76 s      the torch import
#     config + db connect      0.16 s
#     ui.build_server          0.20 s
#     total to a bound port    5.65 s
#
# On the owner's VM that is tens of seconds of the shortcut showing NOTHING — `pythonw`
# has no console, the icon is not in the tray and no tab is open — so the person clicks
# again. The "are we already running" question used to stand in `start()`, after the
# config, torch and the index: ten clicks were ten concurrent torch imports.
#
# Three answers, in the order the launch meets them: the port question is FIRST (`main`),
# before any heavy import, so a second click costs a TCP connect and opens a tab; a window
# appears while the rest happens (`open_splash`), in a process of its own because tkinter
# wants a main thread and this one is about to be taken by `icon.run()`; and the
# diagnostics move BEHIND the bind (`_finish_startup`) — 3.9 s of the 5.65, none of it
# needed to answer a request, reported through `ui.startup_state()` to the waiting tab.
#
# The log line of one launch step, in `runlog`'s shape (one line, INFO, key=value) so a
# launch is as greppable as a run. `startup step=` and NOT `stage=`:
# `runlog.read_measurements` reads `stage=<name> elapsed=` as a timing to price the next
# run with, and a launch is not a stage of the pipeline.
_STARTUP_LINE = "startup step=%s elapsed=%.3f"
_STARTUP_READY_LINE = "startup ready elapsed=%.3f"
# F246: a step that has not finished used to leave NO line at all, so fifteen minutes of
# silence in the log read exactly like a dead process — the owner had no way to tell them
# apart and restarted the program. The beginning is a line of its own; `elapsed=` is
# absent from it, which is also what keeps `runlog` from reading it as a timing.
_STARTUP_BEGIN_LINE = "startup step=%s started"


class TrayUnavailable(Fault, RuntimeError):
    """This machine has no tray (or no library for it) — serve without an icon. The
    absence of an indicator is a property of somebody's desktop, never a reason for the
    program not to start."""

    codes = ("tray_icon_unreadable", "tray_no_pystray", "tray_no_backend")


def url_for(port: int) -> str:
    """The address the icon opens and shows in its tooltip."""
    return f"http://127.0.0.1:{port}/"


# --- F225: a windowed interpreter has no streams, and every library assumes it has ------
#
# The shortcut runs `pythonw.exe -m sorta.tray`, and a windowed interpreter starts with
# `sys.stdout` and `sys.stderr` set to None. `_say` below guards ITS OWN lines; nothing
# guarded anybody else's, and the pipeline runs on a thread of THIS process, so the first
# library that printed took the run down with it. On a clean VM on 2026-08-08 that was
# huggingface_hub drawing its progress bar:
#
#     Failed to download weights for tag 'openai' ...
#     Last error: 'NoneType' object has no attribute 'write'
#
# — a 1.6 GB download that failed with the network, the certificates and the disk all
# fine, because there was nowhere to print the percentage to.
#
# So the fix is HERE, at the entry point, and not at the call that raised: the next
# library to print a line arrives with the next version of transformers. Both streams are
# made to exist and both go to the run log — a line nobody can be shown is still the line
# somebody reads afterwards.


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

    Returns the names that had to be replaced — () on an ordinary console. Also fills
    `sys.__stdout__`/`sys.__stderr__`, which a library reaching past the current streams
    would otherwise find as None.
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
    """Print — on a machine that may have nowhere to print to. A windowed launcher
    (`pythonw`, the `gui-scripts` wrapper) leaves the standard streams closed or absent,
    and a line that cannot be shown is not a reason to fail: it goes to the log."""
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


# --- F227 requirement 3: something on the screen before anything is measured ---------




# --- F227 requirements 1 and 5: the steps, in order, with their durations -------------


@contextmanager
def _startup_step(step: str) -> Iterator[None]:
    """Time one step of the launch — into the log, and into what the page reads. "Slow"
    was a guess about the owner's VM, and the next person to ask why should get the
    answer out of the file instead of measuring somebody else's machine by hand."""
    state = ui.startup_state()
    state.enter(step)
    _LOG.info(_STARTUP_BEGIN_LINE, step)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        state.leave(step, elapsed)
        _LOG.info(_STARTUP_LINE, step, elapsed)


def _finish_startup() -> None:
    """The diagnostics, with the port already answering.

    None of the three is needed to answer an HTTP request and together they were most of
    the silence, so they run here, on a thread of a program that is already serving —
    which is why nothing they do may escape: a probe that fails is a failed probe, not a
    failed launch. Ready is declared FIRST and means "the server can serve"; waiting for
    the probes held the page behind `log_environment`.

    F246: nothing here may import torch either. Declaring ready early gave the page back
    but not the PROCESS — the GPU probe went on holding the interpreter, and the tab that
    had just been shown the program could not fetch anything for as long as it took. The
    torch question now leaves for a child of its own (`diagnostics.current_gpu_health`),
    and `log_environment` is called without `probe_gpu`, which is what keeps it cheap.
    """
    state = ui.startup_state()
    state.ready()
    _LOG.info(_STARTUP_READY_LINE, state.elapsed())
    for step, probe in ((ui.STARTUP_ENVIRONMENT, log_environment),
                        (ui.STARTUP_GPU, warn_if_gpu_mismatch),
                        (ui.STARTUP_GEO, warn_if_geo_data_missing)):
        try:
            with _startup_step(step):
                probe()
        except Exception:
            _LOG.exception("tray: the %s check of the launch failed", step)


def sorta_is_serving(port: int, *, timeout: float = PROBE_TIMEOUT) -> bool:
    """Is the program holding `port` OUR server? Clicking a shortcut twice is normal and
    the second click must not show an error. Asked over HTTP because a port number cannot
    tell our server apart from whatever else may be listening."""
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

    Anything that goes wrong answers NO: where no dialog can be drawn, nobody can be
    asked, and a run is never interrupted without an answer. The window is created and
    destroyed inside the call, the tray having none of its own to hang it on.
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

    A 409 whose reason is `run_in_progress` becomes a QUESTION, not a warning followed by
    quitting anyway, and only a yes sends the second request with the confirmation. A no
    leaves the run and the server as they were — the refusal never touches the cancel flag.
    """
    # A menu callback runs on the tray library's own loop, so nothing here may throw at
    # it: a socket that refuses is an answer like any other.
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
    """The `favicon.ico` of the page, as an image pystray can show. `convert` forces the
    decode here rather than inside the tray library: an unreadable picture is one more
    way this machine has no icon, not a crash of the menu later on."""
    from PIL import Image

    try:
        with Image.open(ICON_PATH) as image:
            return image.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise TrayUnavailable(f"cannot read {ICON_PATH.name}: {exc}",
                              "tray_icon_unreadable", icon=ICON_PATH.name,
                              error=str(exc)) from exc


def build_icon(port: int, lang: i18n.Lang, *,
               on_open: Callable[[], None],
               on_quit: Callable[[], None]) -> Any:
    """The icon and its two-item menu, or `TrayUnavailable` on a machine without one.

    "Open" is the DEFAULT item, which is what makes a double-click on the icon open the
    window: one action reachable two ways, not two behaviours to keep in step. The
    tooltip carries the address so it can be read and copied.
    """
    try:
        import pystray
    except ImportError as exc:
        raise TrayUnavailable(f"pystray is not installed: {exc}", "tray_no_pystray",
                              error=str(exc)) from exc
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
        raise TrayUnavailable(f"no tray on this system: {exc}", "tray_no_backend",
                              error=str(exc)) from exc


def _stop_icon_when_the_server_stops(serving: threading.Thread, icon: Any) -> None:
    """Take the icon away when the server is gone, whichever way it went — the menu is
    not the only exit, the page has its own "Quit" button (F209). An icon still sitting
    in the tray of a program that has closed is a shortcut to nothing."""
    serving.join()
    try:
        icon.stop()
    except Exception as exc:  # the icon may not have been shown yet
        _LOG.warning("tray: could not remove the icon (%s)", exc)


def start(cfg: Any, conn: Any, *, port: int = ui.DEFAULT_PORT,
          config_path: str | Path | None = None,
          open_browser: bool = True,
          ask: Callable[[str, str], bool] = ask_yes_no,
          icon_factory: Callable[..., Any] = build_icon,
          splash: _Splash | None = None) -> int:
    """Serve, with an icon in the tray if this machine has one. The exit code of `main`.

    `ask`/`icon_factory` are injected by the tests: what is worth pinning is that the
    question is asked and that a machine without a tray keeps serving, and neither is
    checkable through somebody else's desktop.

    F227: the port question stays here even though `main` asks it first — it is a cheap
    TCP connect, it is the only guard for a caller that reaches `start` directly, and the
    bind below can still lose a race the earlier question won. `splash` is closed the
    moment the tab is opened, whichever way this returns.
    """
    lang = i18n.normalize_lang(getattr(cfg, "language", None))
    try:
        holder = port_holder(port)
        if holder != PORT_FREE:
            return _busy_port(port, lang, holder, open_browser=open_browser)
        try:
            with _startup_step(ui.STARTUP_SERVER):
                httpd = ui.build_server(cfg, conn, port=port, config_path=config_path)
        except OSError as exc:
            # Somebody took the port between the question above and this bind. Rare, and
            # answered by asking the same question again rather than by guessing.
            _LOG.warning("tray: could not bind port %s (%s)", port, exc)
            return _busy_port(port, lang, port_holder(port), open_browser=open_browser)
        port = httpd.server_port  # port=0 (the tests) -> whatever the OS handed out
        url = url_for(port)
        serving = threading.Thread(target=httpd.serve_forever, daemon=True)
        serving.start()
        try:
            # F227: the port answers from here on, so the diagnostics that used to stand
            # in front of it run beside it. F69 still gets one environment header per
            # server start and F65 its one geo warning, a few seconds later.
            threading.Thread(target=_finish_startup, name="sorta-startup",
                             daemon=True).start()
            if open_browser:
                webbrowser.open(url)
            # The window has done its job: from here the tab says what the launch is
            # still doing.
            if splash is not None:
                splash.close()
            _serve_until_closed(port, lang, url, serving, ask=ask,
                                icon_factory=icon_factory)
        finally:
            httpd.server_close()
        return 0
    finally:
        # A busy port, a failed bind or a raise must not leave a window nobody can close.
        if splash is not None:
            splash.close()
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
        # Not the failure above and just as survivable: a Linux backend that only finds
        # out at run time that there is no indicator raises HERE, with the server already
        # serving. The `join` below is what keeps that from becoming an exit.
        _say(i18n.cli_text("cli.tray.no_icon", lang, reason=exc))
        _say(i18n.cli_text("cli.ui.serving", lang, url=url))
    serving.join()


def _busy_port(port: int, lang: i18n.Lang, holder: str, *, open_browser: bool) -> int:
    """The port is taken, and by WHOM decides what happens next. Ours — a second click on
    the shortcut, so the window opens and this process leaves with a zero exit code.
    Anybody else's — an error, because that is something the person has to be told."""
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
    parser.add_argument("--no-splash", action="store_true",
                        help="do not show the starting window (F227)")
    return parser


def main(argv: Sequence[str] | None = None, splash: "_Splash | None" = None) -> int:
    """The `sorta-tray` entry point. Everything `sorta ui` does at start-up, plus a
    picture in the tray.

    F225: the streams first, before a module of the pipeline is imported — this is the
    windowed launcher's entry point, and from here on any library may print.

    F227: then the ORDER, which is the feature. The port is asked about before the index
    is opened and before anything heavy is imported, so a second click costs a TCP connect
    and gives back a tab; the window goes up before the work starts; and
    `warn_if_gpu_mismatch`, 3.76 s of torch, has moved behind the bind.
    """
    ensure_streams()
    args = build_parser().parse_args(argv)
    state = ui.startup_state()
    state.expect()
    with _startup_step(ui.STARTUP_CONFIG):
        cfg = load_config(args.config)
        configure_logging(cfg.log_level)
    lang = i18n.normalize_lang(cfg.language)
    # The FIRST question, not the fifth: F207's own machinery, asked here instead of
    # inside `start`, with nothing but the config read before it.
    with _startup_step(ui.STARTUP_PORT):
        holder = port_holder(args.port)
    if holder != PORT_FREE:
        state.ready()  # this process is not launching anything; the other one already did
        if splash is not None:
            splash.close()
        return _busy_port(args.port, lang, holder, open_browser=not args.no_browser)
    try:
        with _startup_step(ui.STARTUP_DATABASE):
            conn = connect(cfg.database)
        return start(cfg, conn, port=args.port, config_path=args.config,
                     open_browser=not args.no_browser, splash=splash)
    except BaseException:
        # `start` closes the window itself on every path it owns; this covers the one it
        # never reaches, an index that will not open.
        if splash is not None:
            splash.close()
        raise


if __name__ == "__main__":  # pragma: no cover — the console-script wrapper calls main()
    sys.exit(main())
