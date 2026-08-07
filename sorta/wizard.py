"""F211: the tiers of the installed program, and the wizard that offers them.

An installer carrying everything would be 12-15 GB of download — torch with the CUDA
wheels, Qwen2.5-VL-3B, ViT-L-14, XLM-RoBERTa, buffalo_l — and nobody installs that. The
product is already built in tiers ("heavy behind a flag", a stage that refuses in words
instead of a traceback), so the INSTALL lies on the same tiers: the installer carries the
base one whole and works with no network afterwards, and everything else is offered here,
once, with its real size and its real benefit.

Two things this module is careful about.

* **Refusing is a normal answer, never a dead end.** A person who says no to all of it
  keeps a working product — index, EXIF, geo, duplicates, sorting by city — and is told
  so in as many words. That is the difference between an honest install and a trimmed
  one: no button is left on screen that does nothing.
* **The check screen is `sorta doctor`.** It is written, it already answers what was
  found and what is missing, so the wizard CALLS it (`show_doctor` below) instead of
  growing a second one that will disagree with it by the next release.

The tier catalog is here rather than in `packaging/` because it is read at RUN time: the
wizard ships inside the wheel and the packaging directory does not. `scripts/build_installer.py`
reads the same tuple, so the installer and the wizard can never describe different tiers,
and a watchdog test pairs it with the extras of `pyproject.toml` — an extra added to the
project and forgotten here fails the gate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from . import i18n
from .offline import hf_cache_dir

if TYPE_CHECKING:  # `sorta.tiers` imports THIS module, so the probe is imported lazily
    from .tiers import TierState

# --- what the installer left behind -------------------------------------------------
# The build writes this file next to the program; the wizard reads it to know what was
# actually shipped (the exiftool decision, where `uv` and the environment's python are).
# Absent — this is a checkout rather than an install, and every answer below is probed
# instead of assumed.
MANIFEST_NAME = "sorta-install.json"
ENV_MANIFEST = "SORTA_INSTALL_MANIFEST"
# Paths inside the manifest are written RELATIVE to it, and this key is where the
# directory it was read from is remembered. The build cannot know where a person will
# install the program, and rewriting a JSON file from an installer script to teach it
# its own address is a step that can fail; a relative path cannot.
MANIFEST_ROOT = "root"
# How far up from the running interpreter to look for it: the install layout is
# `{app}\python\python.exe` and the manifest sits at `{app}\sorta-install.json`.
_MANIFEST_LEVELS = 4

# What the person is told about metadata, by what the machine actually has. The third
# answer is the one the brief insists on being SAID: without exiftool the reader falls
# back to Pillow, and HEIC/RAW/video metadata is simply not read — a person must not be
# left guessing why the dates of their phone photographs are missing.
EXIFTOOL_BUNDLED = "bundled"
EXIFTOOL_ON_PATH = "on_path"
EXIFTOOL_ABSENT = "absent"

_SETUP_PREFIX = "cli.setup."


@dataclass(frozen=True)
class Tier:
    """One step of the install: what it adds, what it weighs, what it buys.

    `extras` are extras of `pyproject.toml` — the packages `uv` installs, and the only
    part of a tier that is a command. `weights` are model files the stages download
    themselves on first use; they are named and priced here because they are most of
    what a tier costs, and a size nobody states is a size nobody agreed to.
    `index_url` is for the profile that needs a package index of its own (the CUDA
    wheels), and it is checked against `pyproject.toml` by the suite.
    """

    key: str
    extras: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    download_mb: int = 0
    optional: bool = True
    index_url: str | None = None
    # For a tier that REPLACES what the base one installed rather than adding to it.
    # Without it the CUDA profile is a no-op: `torch>=2.10.0` is already satisfied by the
    # CPU wheel sitting there, so a resolver has nothing to do and the card stays idle.
    reinstall: bool = False
    # F223: the tiers this one does not work WITHOUT. A field rather than an order in the
    # tuple, because order reads as "this one first, then that one" while what has to be
    # said is "without that one this does nothing": search by words encodes the pictures
    # with ViT-L-14 and the words with XLM-RoBERTa, so half of it lives in another tier.
    requires: tuple[str, ...] = ()
    # F223: what pressing Enter answers. Every tier before this one defaulted to no —
    # nothing multi-gigabyte should arrive because somebody wanted the screen gone — and
    # that stays true of every tier a person may simply not want. It is not true of a
    # tier the MAIN TASK needs: without the verdicts the screenshots, the documents and
    # the product shots ride into the city folders among the photographs.
    default_yes: bool = False
    # F223: fetch the weights HERE, during the install, instead of leaving them to the
    # first run of the stage. Worth the wait only where the wait is the point: a person
    # is at the screen, the progress is visible, and a refusal can be explained at once.
    preload: bool = False

    def name(self, lang: i18n.Lang) -> str:
        return i18n.cli_text(f"{_SETUP_PREFIX}tier.{self.key}.name", lang)

    def benefit(self, lang: i18n.Lang) -> str:
        return i18n.cli_text(f"{_SETUP_PREFIX}tier.{self.key}.benefit", lang)

    def without(self, lang: i18n.Lang) -> str:
        """What stays unavailable after a no — the other half of an honest question."""
        return i18n.cli_text(f"{_SETUP_PREFIX}tier.{self.key}.without", lang)


# The CUDA index of `pyproject.toml` ([[tool.uv.index]] name = "pytorch-cu130"). Repeated
# rather than parsed at run time — the installed program has no pyproject to read — and
# pinned to that file by a test, so the two cannot drift apart silently.
PYTORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"

# The tiers, in the order the wizard offers them: what the layout itself needs first,
# then the optional ones from cheapest to the 7 GB one. `base` is not offered at all — it
# is what the installer already put on the disk, and it is in the list so that the extras
# it carries are accounted for.
TIERS: tuple[Tier, ...] = (
    Tier("base", extras=("cpu", "tray"), optional=False),
    # F223: ViT-L-14 used to sit inside the tier below, named after ONE of the things it
    # buys — and a person who did not want to search by words switched off the
    # classification without being told. The two models are separated here and each is
    # named by what it DOES: 1 631 MB that every run needs against 1 397 MB that only
    # search does, which the single 3.0 GB line could not say.
    Tier("vision", weights=("ViT-L-14",), download_mb=1600, default_yes=True,
         preload=True),
    Tier("faces", weights=("buffalo_l",), download_mb=400),
    Tier("search", weights=("XLM-RoBERTa",), download_mb=1400, requires=("vision",)),
    # The one tier that replaces rather than adds: the CUDA builds of torch and
    # onnxruntime take the place of the CPU ones the installer carried. `onnxruntime` and
    # `onnxruntime-gpu` unpack into the SAME directory (the F76 trap), and here that works
    # in our favour — the GPU build is written last — but it is exactly why the wizard
    # ends by pointing at `sorta doctor`, which is the thing that can tell.
    Tier("gpu", extras=("gpu",), download_mb=2500, index_url=PYTORCH_CU130_INDEX,
         reinstall=True),
    Tier("deep", extras=("vlm",), weights=("Qwen2.5-VL-3B",), download_mb=7000),
)

BASE_TIER = TIERS[0]
OPTIONAL_TIERS: tuple[Tier, ...] = tuple(tier for tier in TIERS if tier.optional)
TIERS_BY_KEY: dict[str, Tier] = {tier.key: tier for tier in TIERS}

# The extras that are deliberately NOT part of any tier, and why. The watchdog test reads
# this together with the tiers above: every extra of `pyproject.toml` has to be in one
# list or the other, so an extra added to the project cannot quietly miss the installer.
NOT_SHIPPED: dict[str, str] = {
    "dev": "the quality-gate tools (ruff, mypy, pytest) — they are for a checkout of the "
           "sources, and an installed program has nothing to run them against.",
}

_YES_ANSWERS = frozenset({"y", "yes", "д", "да", "は", "はい", "h"})


def tier_keys() -> tuple[str, ...]:
    """Every tier key, including the base one — what `--tiers` accepts."""
    return tuple(tier.key for tier in TIERS)


def declared_extras() -> set[str]:
    """Every extra a tier installs, plus the ones deliberately left out."""
    return {extra for tier in TIERS for extra in tier.extras} | set(NOT_SHIPPED)


def in_catalog_order(keys: set[str]) -> tuple[Tier, ...]:
    """Those tiers, in the order the catalog states them."""
    return tuple(tier for tier in TIERS if tier.key in keys)


def with_requirements(accepted: Sequence[Tier], have: Sequence[str] = ()
                      ) -> tuple[tuple[Tier, ...], tuple[tuple[Tier, Tier], ...]]:
    """(what will be added, what pulled what in) — the tiers a yes implies.

    `have` is what this machine already has, so a requirement that is in place is
    satisfied silently: telling somebody that a tier they never asked about was added,
    when nothing will be downloaded for it, is noise in the one screen that must not have
    any. Everything else is SAID by the caller — a wizard that quietly turns a 1.4 GB
    answer into a 3.0 GB one is the defect this feature exists against, one level down.
    """
    satisfied = {tier.key for tier in accepted} | set(have)
    chosen = {tier.key for tier in accepted}
    pulled: list[tuple[Tier, Tier]] = []
    queue = list(accepted)
    while queue:
        tier = queue.pop(0)
        for key in tier.requires:
            required = TIERS_BY_KEY.get(key)
            if required is None or key in satisfied:
                continue
            satisfied.add(key)
            chosen.add(key)
            pulled.append((tier, required))
            queue.append(required)
    return in_catalog_order(chosen), tuple(pulled)


def human_size(mb: int, lang: i18n.Lang) -> str:
    """A download size the way a person reads it — 400 MB, 3.0 GB."""
    if mb < 1000:
        return i18n.cli_text(f"{_SETUP_PREFIX}size_mb", lang, mb=mb)
    return i18n.cli_text(f"{_SETUP_PREFIX}size_gb", lang, gb=f"{mb / 1000:.1f}")


# --- what the installer wrote --------------------------------------------------------


def manifest_path(explicit: str | Path | None = None) -> Path | None:
    """Where the install manifest is, or None on a machine that has no install.

    Three answers in the order they can be trusted: what the caller passed, what the
    environment names (the installer sets it for the wizard it launches), and finally the
    directories above the running interpreter.
    """
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None
    from_env = os.environ.get(ENV_MANIFEST)
    if from_env:
        candidate = Path(from_env)
        return candidate if candidate.is_file() else None
    here = Path(sys.executable).resolve().parent
    for parent in (here, *list(here.parents)[:_MANIFEST_LEVELS]):
        candidate = parent / MANIFEST_NAME
        if candidate.is_file():
            return candidate
    return None


def load_manifest(explicit: str | Path | None = None) -> dict:
    """The manifest as a dict; an empty one when there is none or it is unreadable.

    A broken manifest may not stop the wizard: everything read out of it has a probe
    behind it, and a person who ran the setup again is not the person to show a parse
    error to.
    """
    path = manifest_path(explicit)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    payload.setdefault(MANIFEST_ROOT, str(path.parent))
    return payload


def manifest_path_of(manifest: dict, key: str) -> str | None:
    """A path out of the manifest, resolved against the directory it was read from."""
    value = manifest.get(key)
    if not value:
        return None
    text = str(value)
    root = manifest.get(MANIFEST_ROOT)
    # An absolute path is returned WORD FOR WORD: it was written by somebody who knew
    # where the thing is, and normalising it here would only make it harder to recognise.
    if Path(text).is_absolute() or not root:
        return text
    return str(Path(str(root)) / text)


def exiftool_state(manifest: dict, *,
                   which: Callable[[str], str | None] | None = None) -> str:
    """Bundled with the program, found on PATH, or absent — the three honest answers.

    `which` is resolved at CALL time rather than bound as a default, so a caller (and
    the suite) can answer for a machine other than this one.
    """
    if manifest.get("exiftool"):
        return EXIFTOOL_BUNDLED
    finder = shutil.which if which is None else which
    return EXIFTOOL_ON_PATH if finder("exiftool") else EXIFTOOL_ABSENT


def uv_binary(manifest: dict) -> str:
    """The `uv` that installs the tiers — the bundled one, or whatever is on PATH.

    One mechanism and not two: `uv` already resolves our extras, our conflicting cpu/gpu
    profiles and our indexes, and a second resolver written for the installer would
    disagree with the first one inside a month (the boundary the brief draws).
    """
    return manifest_path_of(manifest, "uv") or shutil.which("uv") or "uv"


def python_binary(manifest: dict) -> str:
    """The interpreter of the installed environment — the one a tier is installed into."""
    return manifest_path_of(manifest, "python") or sys.executable


def lib_directory(manifest: dict) -> str | None:
    """Where the packages live: `{app}\\lib`, installed with `uv pip install --target`.

    A plain directory tree and not a virtualenv, and that is the whole reason the payload
    can simply be COPIED to wherever a person installs it: a venv records the absolute
    path of the interpreter it was made from, and a `--target` tree records nothing at
    all. `{app}\\python` finds it through one `.pth` file, and a tier added later has to
    land in the same place — hence this being part of the install command below.
    """
    return manifest_path_of(manifest, "lib")


def tier_requirements(tier: Tier, *, distribution: str = "sorta") -> tuple[str, ...]:
    """The requirement strings of the tier's extras, read off the installed metadata.

    The single source stays `pyproject.toml`: what is read here is what the build put
    into the wheel from it, so a version bound edited in the project reaches the wizard
    without anything being copied by hand. A distribution that cannot be found (running
    from a checkout that was never installed) yields nothing, and the caller says so
    rather than inventing a version.
    """
    if not tier.extras:
        return ()
    try:
        from importlib.metadata import requires as _requires

        declared = _requires(distribution) or []
    except Exception:  # noqa: BLE001 — no metadata is an answer, not a crash
        return ()
    found: list[str] = []
    for line in declared:
        requirement, _, marker = line.partition(";")
        for extra in tier.extras:
            if f'extra == "{extra}"' in marker or f"extra == '{extra}'" in marker:
                found.append(requirement.strip())
                break
    return tuple(found)


def install_command(tier: Tier, requirements: Sequence[str], *,
                    uv: str, python: str, target: str | None = None) -> list[str]:
    """The command that adds a tier: `uv pip install` into the installed environment."""
    command = [uv, "pip", "install", "--python", python]
    if target:
        command += ["--target", target]
    if tier.reinstall:
        command.append("--reinstall")
    command += list(requirements)
    if tier.index_url:
        # An ADDITIONAL index, the way `[[tool.uv.index]]` of pyproject.toml states it:
        # torch comes from the CUDA one, everything else stays on PyPI. (`--index` and
        # not the deprecated `--extra-index-url`.)
        command += ["--index", tier.index_url]
    return command


def run_install(command: Sequence[str]) -> int:
    """Run an install command. Anything that goes wrong is an exit code, never a raise.

    A tier that will not install must leave the program exactly as it was — the base one
    keeps working, and the wizard says which tier failed and that it can be tried again.
    """
    try:
        return int(subprocess.run(list(command), check=False).returncode)
    except OSError:  # no uv on this machine, no permission to run it
        return 1


# --- the screens ---------------------------------------------------------------------


def show_doctor(config_path: str) -> None:
    """The check screen, which IS `sorta doctor` (requirement 3 of the brief).

    Imported inside the call on purpose: `sorta.cli` pulls in the whole command line
    (typer, numpy, every stage module), and a wizard that only wanted to print two health
    lines has no business paying for that at import time.
    """
    from .cli import _cmd_doctor

    _cmd_doctor(config_path)


def say_console(text: str) -> None:
    """Print — on a console that may not encode every character of the catalog.

    The catalog is Russian, English and Japanese, and a Windows console still runs on a
    legacy code page unless it is told otherwise (the installer starts this with
    `-X utf8`, which is the real fix). Where it was not, a sentence with a character the
    code page has no room for must still APPEAR — losing the line about exiftool because
    of an em dash would be the worst way to fail.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            print(text.encode(encoding, "replace").decode(encoding, "replace"))
        except (OSError, ValueError):
            pass
    except (OSError, ValueError):  # a closed stream under a windowed launcher
        pass


def ask_console(question: str, default: bool = False) -> bool:
    """One yes/no question. An empty answer is `default`; EOF is always a no.

    A default of NO is the shape of almost the whole wizard: nothing multi-gigabyte is
    downloaded because somebody pressed Enter to make the screen go away. F223 adds the
    one exception and it is deliberate — the tier the LAYOUT needs, where the cost of a
    stray Enter falls the other way (screenshots and documents in the city folders).

    EOF keeps the old answer whatever the default is, and that is not an inconsistency:
    a stream that is closed means nobody is at this screen, and the point of downloading
    the weights here is precisely that somebody is. Unwatched, the stage fetches them on
    its first run and says so (F222).
    """
    try:
        answer = input(f"{question} ")
    except (EOFError, OSError):
        return False
    stripped = answer.strip().lower()
    if not stripped:
        return default
    return stripped in _YES_ANSWERS


# --- F223: the weights the wizard fetches while somebody is watching ------------------
#
# Until now the wizard installed PACKAGES and left every model file to the first run of
# the stage that needed it. That is still right for a tier somebody may never use — but
# not for the one the layout needs: there the download happens anyway, and doing it here
# is what buys the two things a run cannot give. A person is at the screen, so a refusal
# can be explained on the spot; and the 1.6 GB does not arrive in the middle of a run
# that looks, from outside, exactly like a hang.
#
# Progress is not optional. 1.6 GB with no line on screen is what cost the owner an hour
# on 2026-08-07. How much has arrived is huggingface_hub's to know and reaching into its
# progress bars from here would be a fragile way to learn it — so what is printed is the
# one number this side can measure honestly: how much bigger the hub cache has grown
# since the download started.

# How often the progress line is printed. Long enough not to fill the window of a slow
# download, short enough that the gap between two lines never reads as a stall.
PROGRESS_SECONDS = 5.0
_MB = 1_000_000


def downloaded_bytes(cache: Path | None = None) -> int:
    """How much the model cache holds right now — progress measured on the disk.

    Deliberately the whole cache rather than one model's directory: what a library names
    the folder it is filling (and whether it fills a `blobs/` file under a temporary
    name first) is its own business, and a measurement that depends on those names would
    quietly report zero the day one of them changes.
    """
    directory = hf_cache_dir() if cache is None else cache
    total = 0
    try:
        for item in directory.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:  # a file the downloader replaced between the two calls
                continue
    except OSError:  # no cache directory yet — nothing has been downloaded
        return 0
    return total


def clip_weight_names(config_path: str = "config.yaml") -> tuple[str, str]:
    """The CLIP the classification stage will load: `(architecture, weights)`.

    Read out of the config rather than repeated here, so a machine whose owner changed
    `naming.clip.model` preloads the model that machine will actually use. A config that
    cannot be read yet — the ordinary state of a fresh install — gives the defaults,
    which is what such a machine will load too.
    """
    from .config import NamingConfig, load_config

    naming = NamingConfig()
    try:
        naming = load_config(config_path).naming
    except Exception:  # noqa: BLE001 — an unreadable config still gets its weights
        pass
    return str(naming.clip_model), str(naming.clip_pretrained)


def fetch_weights(tier: Tier, config_path: str = "config.yaml") -> None:
    """Download this tier's models, the way the stage that needs them would.

    Through open_clip and not through a hub call written here: the stage asks for the
    weights by an open_clip name (`ViT-L-14-quickgelu` / `openai`), open_clip decides
    which repository and which file that is, and a second answer to that question would
    fill the cache with something the stage then downloads again.
    """
    for weight in tier.weights:
        fetcher = _FETCHERS.get(weight)
        if fetcher is None:
            raise LookupError(f"{weight}: the wizard has no downloader for it")
        fetcher(config_path)


def _fetch_clip(config_path: str) -> None:  # pragma: no cover — 1.6 GB over the network
    import open_clip

    model, pretrained = clip_weight_names(config_path)
    open_clip.create_model_and_transforms(model, pretrained=pretrained, device="cpu")


# Only the weights of a tier that PRELOADS need an entry, and a test pins the two lists
# together: a tier marked `preload` whose model nobody here can fetch would fail at
# install time, on the one screen where a failure is most expensive.
_FETCHERS: dict[str, Callable[[str], None]] = {"ViT-L-14": _fetch_clip}


def download_weights(tier: Tier, lang: i18n.Lang, config_path: str = "config.yaml", *,
                     say: Callable[[str], None] = say_console,
                     fetch: Callable[[Tier, str], None] = fetch_weights,
                     measure: Callable[[], int] = downloaded_bytes,
                     tick: float = PROGRESS_SECONDS) -> bool:
    """Fetch the tier's weights now, saying how it goes. True when they are on disk.

    A refusal by the network is not an error of the install: everything else stays as it
    was, the product works, and the stage that wanted the model fetches it on its first
    run — which F222 announces before that run starts. So this returns False and the
    caller says what it means, instead of raising into a window that is about to close.
    """
    say(i18n.cli_text(f"{_SETUP_PREFIX}weights_downloading", lang,
                      weights=", ".join(tier.weights),
                      size=human_size(tier.download_mb, lang)))
    failure: list[BaseException] = []

    def work() -> None:
        try:
            fetch(tier, config_path)
        except BaseException as exc:  # noqa: BLE001 — every refusal is a sentence below
            failure.append(exc)

    start = measure()
    # A thread and not a subprocess: the download has to land in the cache of THIS
    # machine's user, and the interpreter that would be started for a subprocess is the
    # question `python_binary` exists to answer for packages — one mechanism per job.
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while True:
        worker.join(tick)
        if not worker.is_alive():
            break
        say(i18n.cli_text(f"{_SETUP_PREFIX}weights_progress", lang,
                          done=human_size(max(0, measure() - start) // _MB, lang),
                          size=human_size(tier.download_mb, lang)))
    if failure:
        error = failure[0]
        say(i18n.cli_text(f"{_SETUP_PREFIX}weights_failed", lang,
                          weights=", ".join(tier.weights),
                          error=str(error).strip() or error.__class__.__name__))
        return False
    say(i18n.cli_text(f"{_SETUP_PREFIX}weights_ready", lang,
                      weights=", ".join(tier.weights),
                      size=human_size(tier.download_mb, lang)))
    return True


# --- F223: the window may not close over the answer it was opened for ------------------
#
# The owner started "Sorta setup" from the Start menu, chose the deep tier — and the
# window shut. Not on an error: on the END. The only `input()` in this module was the
# tier question, so after the last answer `run_setup` returned, `main` returned, the
# process ended and Windows destroyed the console it had created for it, taking the whole
# summary with it. A refusal disappeared exactly as fast as a success, which is why the
# certificate failure of F221 was invisible for as long as it was.
#
# Waiting is right only when the console is OURS. Typed into a terminal that was already
# open, `sorta-setup` must return to the prompt like any other command, and a pause there
# is one more keystroke for nothing. What tells the two apart on Windows is the number of
# processes attached to the console: a window created for this process has one.


def owns_console(os_name: str = os.name) -> bool:
    """Would this console die with this process? Then the last screen has to be held.

    Anything that cannot be asked is a no, and that is the safe direction: a wizard that
    pauses where it should not is a hang to whoever is waiting for the command to finish,
    while one that does not pause where it should have loses a screen a person can get
    back by running the setup again.
    """
    if os_name != "nt":
        return False
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        buffer = (ctypes.c_uint32 * 8)()
        return int(windll.kernel32.GetConsoleProcessList(buffer, 8)) == 1
    except Exception:  # noqa: BLE001 — no console, no kernel32, no answer: do not wait
        return False


def hold_console(lang: i18n.Lang, *, say: Callable[[str], None] = say_console,
                 wait: Callable[[str], str] = input,
                 owns: Callable[[], bool] = owns_console) -> None:
    """Keep the window open until somebody has read it — when the window is ours."""
    if not owns():
        return
    say(i18n.cli_text(f"{_SETUP_PREFIX}press_enter", lang))
    try:
        wait("")
    except (EOFError, OSError, KeyboardInterrupt):
        pass


@dataclass
class Outcome:
    """What the wizard did, for the summary and for the exit code."""

    # `chosen` and not `installed`: a weights-only tier says yes to a download that
    # happens on the first run of its stage, and nothing is installed at this moment.
    chosen: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    # F223: what this machine already had. Neither chosen nor skipped — a person who
    # reinstalls over a full model cache is offered nothing to download twice.
    present: list[str] = field(default_factory=list)


def probe_tiers() -> list[TierState]:
    """What this machine has of each tier — the probe `sorta doctor` reads.

    Imported inside the call because `sorta.tiers` imports this module: one probe and not
    two is the F216/F217 rule, and the direction of the import is the price of keeping the
    catalog where the wizard is.
    """
    from .tiers import tier_states

    return tier_states()


def run_setup(lang: i18n.Lang, *,
              manifest: dict | None = None,
              config_path: str = "config.yaml",
              chosen: Sequence[str] | None = None,
              tiers: Sequence[Tier] = OPTIONAL_TIERS,
              states: Sequence[TierState] | None = None,
              say: Callable[[str], None] = say_console,
              ask: Callable[[str, bool], bool] = ask_console,
              doctor: Callable[[str], None] = show_doctor,
              install: Callable[[Sequence[str]], int] = run_install,
              download: Callable[..., bool] = download_weights) -> int:
    """The first-run wizard. Returns the exit code: 0 unless an install failed.

    `chosen` is the non-interactive form — the tier keys to add, `()` for none at all.
    With it None every optional tier is offered one by one, and the answer defaults to
    what the tier itself states (no, except for the one the layout needs).
    """
    manifest = load_manifest() if manifest is None else manifest
    say(i18n.cli_text(f"{_SETUP_PREFIX}title", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}checking", lang))
    doctor(config_path)
    say(i18n.cli_text(f"{_SETUP_PREFIX}exiftool_{exiftool_state(manifest)}", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}base_ready", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}tiers_intro", lang))

    machine = {state.key: state for state in
               (probe_tiers() if states is None else states)}
    outcome = Outcome()
    accepted: list[Tier] = []
    in_place: list[str] = []
    for tier in tiers:
        state = machine.get(tier.key)
        if state is not None and state.ready:
            # Everything this tier consists of is already on the disk. Offering it would
            # be offering a download of nothing, and a person who reinstalled over a full
            # cache would take it (the acceptance criterion this feature carries).
            in_place.append(tier.key)
            outcome.present.append(tier.name(lang))
            say(i18n.cli_text(f"{_SETUP_PREFIX}in_place", lang, name=tier.name(lang)))
            continue
        say(i18n.cli_text(f"{_SETUP_PREFIX}offer", lang, name=tier.name(lang),
                          size=human_size(tier.download_mb, lang),
                          benefit=tier.benefit(lang)))
        if chosen is None:
            question = "question_yes" if tier.default_yes else "question"
            wanted = ask(i18n.cli_text(f"{_SETUP_PREFIX}{question}", lang),
                         tier.default_yes)
        else:
            wanted = tier.key in chosen
        if wanted:
            accepted.append(tier)
        else:
            outcome.skipped.append(tier.name(lang))
            say(i18n.cli_text(f"{_SETUP_PREFIX}refused", lang, without=tier.without(lang)))

    # F223: a yes to a tier is a yes to what it does not work without — said out loud,
    # because a wizard that silently turns one answer into two downloads is the same
    # defect as a tier named after its contents, one screen further in.
    resolved, pulled = with_requirements(accepted, in_place)
    for dependent, required in pulled:
        say(i18n.cli_text(f"{_SETUP_PREFIX}requires", lang, name=dependent.name(lang),
                          required=required.name(lang)))
        if required.name(lang) in outcome.skipped:
            outcome.skipped.remove(required.name(lang))

    _add_tiers(resolved, lang, manifest, outcome, config_path=config_path, say=say,
               install=install, download=download)
    _summary(outcome, lang, say=say)
    return 1 if outcome.failed else 0


def _add_tiers(accepted: Sequence[Tier], lang: i18n.Lang, manifest: dict,
               outcome: Outcome, *, config_path: str = "config.yaml",
               say: Callable[[str], None],
               install: Callable[[Sequence[str]], int],
               download: Callable[..., bool] = download_weights) -> None:
    """Install what was said yes to — packages now, model weights on first use.

    The two halves are reported apart because they cost the person different things: a
    package is downloaded here and now, while weights arrive inside the first run of the
    stage that needs them. Saying "installed" about the second would be a lie the person
    only discovers when a stage starts a 7 GB download.
    """
    uv = uv_binary(manifest)
    python = python_binary(manifest)
    target = lib_directory(manifest)
    for tier in accepted:
        requirements = tier_requirements(tier)
        if tier.extras and not requirements:
            outcome.failed.append(tier.name(lang))
            say(i18n.cli_text(f"{_SETUP_PREFIX}no_metadata", lang, name=tier.name(lang)))
            continue
        if requirements:
            say(i18n.cli_text(f"{_SETUP_PREFIX}installing", lang, name=tier.name(lang),
                              packages=" ".join(requirements)))
            code = install(install_command(tier, requirements, uv=uv, python=python,
                                           target=target))
            if code != 0:
                outcome.failed.append(tier.name(lang))
                say(i18n.cli_text(f"{_SETUP_PREFIX}install_failed", lang,
                                  name=tier.name(lang), status=code))
                continue
        if tier.weights and tier.preload:
            # F223: the download happens HERE, at the screen, with progress — and a
            # refusal by the network leaves the install as it was rather than failing it.
            if not download(tier, lang, config_path, say=say):
                outcome.skipped.append(tier.name(lang))
                continue
        elif tier.weights:
            say(i18n.cli_text(f"{_SETUP_PREFIX}weights_later", lang,
                              size=human_size(tier.download_mb, lang),
                              weights=", ".join(tier.weights)))
        outcome.chosen.append(tier.name(lang))


def _summary(outcome: Outcome, lang: i18n.Lang, *, say: Callable[[str], None]) -> None:
    """The last screen: what happened, and that a no is not a dead end."""
    if outcome.chosen:
        say(i18n.cli_text(f"{_SETUP_PREFIX}added", lang, names=", ".join(outcome.chosen)))
    if outcome.present:
        say(i18n.cli_text(f"{_SETUP_PREFIX}already", lang,
                          names=", ".join(outcome.present)))
    if outcome.skipped:
        say(i18n.cli_text(f"{_SETUP_PREFIX}skipped", lang,
                          names=", ".join(outcome.skipped)))
    if not outcome.chosen and not outcome.present:
        # The whole point of the tiers, said out loud to the person who took none of
        # them: what is on this machine is a working product and not a stub.
        say(i18n.cli_text(f"{_SETUP_PREFIX}works_anyway", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}rerun", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}doctor_hint", lang))


# --- the entry point -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The options of `sorta-setup`. Interactive by default; scriptable when asked."""
    parser = argparse.ArgumentParser(
        prog="sorta-setup",
        description="Sorta first-run setup: what is installed, and which tiers to add.")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="path to config.yaml (default: %(default)s) — the language "
                             "of this wizard is read from it")
    parser.add_argument("--lang", choices=("ru", "en", "ja"), default=None,
                        help="answer in this language instead of the configured one")
    parser.add_argument("--tiers", default=None,
                        help="add these tiers without asking: a comma-separated list of "
                             f"{', '.join(tier.key for tier in OPTIONAL_TIERS)}, "
                             "`all`, or `none`")
    parser.add_argument("--manifest", default=None,
                        help="path to " + MANIFEST_NAME + " (default: the one the "
                             "installer left next to the program)")
    return parser


def selected_tiers(spec: str | None) -> tuple[str, ...] | None:
    """`--tiers` as a list of keys; None means "ask about each one".

    An unknown name is an error rather than a silent skip: a script that asked for a
    tier and got none would look exactly like a successful run.
    """
    if spec is None:
        return None
    wanted = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if wanted == ["none"]:
        return ()
    if wanted == ["all"]:
        return tuple(tier.key for tier in OPTIONAL_TIERS)
    unknown = [name for name in wanted if name not in TIERS_BY_KEY]
    if unknown:
        raise ValueError(f"unknown tier(s): {', '.join(unknown)}")
    return tuple(wanted)


def language(config_path: str, override: str | None = None) -> i18n.Lang:
    """The language of the wizard: the flag, then config.yaml, then the default.

    Reading the config may fail in every way a fresh install can — no file yet, a file
    somebody is still editing — and none of those is a reason to say nothing at all.
    """
    if override:
        return i18n.normalize_lang(override)
    try:
        from .config import load_config

        return i18n.normalize_lang(getattr(load_config(config_path), "language", None))
    except Exception:  # noqa: BLE001 — an unreadable config still gets a wizard
        return i18n.normalize_lang(None)


def main(argv: Sequence[str] | None = None) -> int:
    """The `sorta-setup` entry point — what the installer runs when it has finished.

    The `finally` is the whole of the second defect F223 fixes: an error has to be read
    exactly as much as a summary, and until this the two disappeared at the same speed —
    with the window.
    """
    args = build_parser().parse_args(argv)
    lang = language(args.config, args.lang)
    try:
        try:
            chosen = selected_tiers(args.tiers)
        except ValueError as exc:
            say_console(str(exc))
            return 2
        return run_setup(lang, manifest=load_manifest(args.manifest),
                         config_path=args.config, chosen=chosen)
    finally:
        hold_console(lang)


if __name__ == "__main__":  # pragma: no cover — the console-script wrapper calls main()
    sys.exit(main())
