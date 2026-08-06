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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import i18n

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
# `{app}\env\Scripts\python.exe` and the manifest sits at `{app}\sorta-install.json`.
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

# The tiers, in the order the wizard offers them: cheapest and most useful first, the
# 7 GB one last. `base` is not offered at all — it is what the installer already put on
# the disk, and it is in the list so that the extras it carries are accounted for.
TIERS: tuple[Tier, ...] = (
    Tier("base", extras=("cpu", "tray"), optional=False),
    Tier("faces", weights=("buffalo_l",), download_mb=400),
    Tier("search", weights=("ViT-L-14", "XLM-RoBERTa"), download_mb=3000),
    Tier("gpu", extras=("gpu",), download_mb=2500, index_url=PYTORCH_CU130_INDEX),
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
    command += list(requirements)
    if tier.index_url:
        # An EXTRA index: torch comes from the CUDA one, everything else stays on PyPI.
        command += ["--extra-index-url", tier.index_url]
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
    """Print — on a console that may not encode every character of the catalog."""
    try:
        print(text)
    except (OSError, ValueError):  # a closed stream under a windowed launcher
        pass


def ask_console(question: str) -> bool:
    """One yes/no question. Anything that is not a yes — including EOF — is a no.

    A default of NO is the whole shape of this wizard: nothing multi-gigabyte is
    downloaded because somebody pressed Enter to make the screen go away.
    """
    try:
        answer = input(f"{question} ")
    except (EOFError, OSError):
        return False
    return answer.strip().lower() in _YES_ANSWERS


@dataclass
class Outcome:
    """What the wizard did, for the summary and for the exit code."""

    added: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def run_setup(lang: i18n.Lang, *,
              manifest: dict | None = None,
              config_path: str = "config.yaml",
              chosen: Sequence[str] | None = None,
              tiers: Sequence[Tier] = OPTIONAL_TIERS,
              say: Callable[[str], None] = say_console,
              ask: Callable[[str], bool] = ask_console,
              doctor: Callable[[str], None] = show_doctor,
              install: Callable[[Sequence[str]], int] = run_install) -> int:
    """The first-run wizard. Returns the exit code: 0 unless an install failed.

    `chosen` is the non-interactive form — the tier keys to add, `()` for none at all.
    With it None every optional tier is offered one by one, and the answer defaults to no.
    """
    manifest = load_manifest() if manifest is None else manifest
    say(i18n.cli_text(f"{_SETUP_PREFIX}title", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}checking", lang))
    doctor(config_path)
    say(i18n.cli_text(f"{_SETUP_PREFIX}exiftool_{exiftool_state(manifest)}", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}base_ready", lang))
    say(i18n.cli_text(f"{_SETUP_PREFIX}tiers_intro", lang))

    outcome = Outcome()
    accepted: list[Tier] = []
    for tier in tiers:
        say(i18n.cli_text(f"{_SETUP_PREFIX}offer", lang, name=tier.name(lang),
                          size=human_size(tier.download_mb, lang),
                          benefit=tier.benefit(lang)))
        if chosen is None:
            wanted = ask(i18n.cli_text(f"{_SETUP_PREFIX}question", lang))
        else:
            wanted = tier.key in chosen
        if wanted:
            accepted.append(tier)
        else:
            outcome.skipped.append(tier.name(lang))
            say(i18n.cli_text(f"{_SETUP_PREFIX}refused", lang, without=tier.without(lang)))

    _add_tiers(accepted, lang, manifest, outcome, say=say, install=install)
    _summary(outcome, lang, say=say)
    return 1 if outcome.failed else 0


def _add_tiers(accepted: Sequence[Tier], lang: i18n.Lang, manifest: dict,
               outcome: Outcome, *, say: Callable[[str], None],
               install: Callable[[Sequence[str]], int]) -> None:
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
        if tier.weights:
            say(i18n.cli_text(f"{_SETUP_PREFIX}weights_later", lang,
                              size=human_size(tier.download_mb, lang),
                              weights=", ".join(tier.weights)))
        outcome.added.append(tier.name(lang))


def _summary(outcome: Outcome, lang: i18n.Lang, *, say: Callable[[str], None]) -> None:
    """The last screen: what happened, and that a no is not a dead end."""
    if outcome.added:
        say(i18n.cli_text(f"{_SETUP_PREFIX}added", lang, names=", ".join(outcome.added)))
    if outcome.skipped:
        say(i18n.cli_text(f"{_SETUP_PREFIX}skipped", lang,
                          names=", ".join(outcome.skipped)))
    if not outcome.added:
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
    """The `sorta-setup` entry point — what the installer runs when it has finished."""
    args = build_parser().parse_args(argv)
    lang = language(args.config, args.lang)
    try:
        chosen = selected_tiers(args.tiers)
    except ValueError as exc:
        say_console(str(exc))
        return 2
    return run_setup(lang, manifest=load_manifest(args.manifest),
                     config_path=args.config, chosen=chosen)


if __name__ == "__main__":  # pragma: no cover — the console-script wrapper calls main()
    sys.exit(main())
