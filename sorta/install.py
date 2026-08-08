"""What the installer left next to the program, for anything that needs to find it (F226).

The Windows build writes `sorta-install.json` beside the installed copy and names in it
everything the payload carries — the interpreter, `lib\\`, `uv.exe`, and the bundled
`exiftool\\exiftool.exe`. `sorta.wizard` has read that file since F211, and for a while it
was the only reader, so the reading lived there.

That stopped being the right address the moment the metadata reader needed the same
answer. `sorta.exif` asking `sorta.wizard` where exiftool is would import the whole tier
catalog — six tiers, their sizes, their translated names, a screen — to resolve one path,
and it would make the module that reads EXIF depend on the module that installs CUDA.

So the lookup is here: find the manifest, read it, turn a name inside it into an absolute
path. No state, no catalog, nothing that has to be imported for it.

The constants and the two readers are deliberately identical to the wizard's, and a test
pins them to each other so the copies cannot drift. Moving `wizard.py` onto this module is
the cleanup that follows; it is not part of this feature, because that file is being
changed by another one right now.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The file itself, and the environment variable the installer sets for the process it
# launches. Same names as `sorta.wizard` uses — there is exactly one such file.
MANIFEST_NAME = "sorta-install.json"
ENV_MANIFEST = "SORTA_INSTALL_MANIFEST"
# Paths inside the manifest are written RELATIVE to it; this key is where the directory it
# was read from is remembered, so a relative name can be resolved afterwards. The build
# cannot know where a person will install the program, and a relative path needs no
# install-time rewriting to survive being copied somewhere else.
MANIFEST_ROOT = "root"
# How far up from the running interpreter to look: the install layout is
# `{app}\python\python.exe` and the manifest sits at `{app}\sorta-install.json`.
_MANIFEST_LEVELS = 4


def manifest_path(explicit: str | Path | None = None) -> Path | None:
    """Where the install manifest is, or None on a machine that has no install.

    Three answers in the order they can be trusted: what the caller passed, what the
    environment names, and finally the directories above the running interpreter. A
    checkout has none of the three and gets None, which is the honest answer — there is
    no install there to have shipped anything.
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

    A broken manifest may not break the product: every caller here has a fallback behind
    it (the reader falls back to Pillow, the wizard probes), and an install that answers
    "nothing was shipped" degrades exactly like a checkout, which is a state the whole
    program already handles.
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


def tool_path(manifest: dict, key: str) -> str | None:
    """The absolute path of a tool the manifest names, or None if it names none.

    Whether the file is THERE, and whether it runs, is the caller's question: this only
    turns a name written by the build into an address on this machine.
    """
    value = manifest.get(key)
    if not value:
        return None
    text = str(value)
    root = manifest.get(MANIFEST_ROOT)
    # An absolute path is returned word for word: it was written by somebody who knew
    # where the thing is, and normalising it here would only make it harder to recognise.
    if Path(text).is_absolute() or not root:
        return text
    return str(Path(str(root)) / text)


# --- F230: which install is reading this, and therefore which command is true for it ---
#
# Every piece of advice this product gives about ITS OWN installation is only true on one
# of the paths below, and until this feature each such line was written for whichever path
# its author happened to be on. Three were found by the owner in a virtual machine over
# two days — `uv sync --extra gpu --extra dev` printed by `doctor` to a copy that has no
# project directory, `uv sync --extra vlm` in the help of `--deep`, and a tier hint that
# offered the Start menu to a developer because it was keyed BY OPERATING SYSTEM.
#
# That last one is the whole diagnosis: the question is not "which OS is this" but "which
# INSTALL is this", and the answer already existed here (F226 reads the manifest the
# Windows build leaves next to the program). So it is answered ONCE, here, and everything
# that names a command asks this instead of guessing.
#
# The OS stays a separate question and is not folded in: `winget`/`brew`/`apt` is about
# which package manager a machine has, which is true of a checkout and an installed copy
# alike (`cli._exiftool_hint_key`). What the OS may NOT decide is which install this is.

# A copy the Windows installer put on the disk: a manifest sits next to the program, the
# interpreter and `uv` came with it, and `sorta-setup` (the Start menu item) is how tiers
# are added. There is no project directory and no `dev` extra here.
KIND_INSTALLED = "installed"
# A checkout of the sources: `pyproject.toml` above the package, a developer, `uv sync`.
# The owner's preferred path, and the one that must not be broken to fix the others.
KIND_CHECKOUT = "checkout"
# A wheel installed into an environment of its own — `uv tool install`, `pip install`.
# No sources to sync and no Start menu to look in, which is what F213 found on Linux.
KIND_TOOL = "tool"

KINDS: tuple[str, ...] = (KIND_CHECKOUT, KIND_INSTALLED, KIND_TOOL)

# What tells a checkout from a wheel: the project file the developer's commands need.
# `uv sync --extra gpu` is a statement about a directory that has this in it, so the
# presence of that file IS the precondition of the advice — not a proxy for it.
PYPROJECT_NAME = "pyproject.toml"

# The advice that has to be chosen by install kind, by the base of its key. Each name
# here stands for three keys of the catalog — `<base>.checkout`, `<base>.installed`,
# `<base>.tool` — and `advice_key` is the only way any of them is reached.
#
# This tuple is the registry the watchdog reads (tests/test_each_install_is_told_the_
# truth.py): a string that names a command and is NOT part of one of these families has
# to be justified there by hand. That is the point of the feature — the three cases above
# were caught one at a time by a person, and the fourth is caught by the suite.
INSTALL_ADVICE: tuple[str, ...] = (
    # `sorta doctor`: how to add a tier this machine is missing.
    "cli.doctor.tier_hint",
    # `sorta doctor`: how to run the command when it is not on PATH.
    "cli.doctor.command_hint",
    # `sorta-setup`: how to come back for a tier that was refused.
    "cli.setup.rerun",
    # `sorta-setup`: how to undo the acceleration tier and get the CPU profile back.
    "cli.setup.cpu_back",
    # `sorta run --help`: how to get the deep tier the `--deep` flag needs.
    "cli.help.run.deep_how",
)


def install_kind(manifest: dict | None = None, *,
                 package: str | Path | None = None) -> str:
    """Which of the three paths this copy is on — the one answer, asked by everybody.

    `manifest` is what `load_manifest()` returned when the caller already has it (the
    doctor does), so the file is not read twice in one command; passing an empty dict
    means "this machine has no install manifest" and is how a test states a checkout.
    `package` is the directory of the package itself, injectable for the same reason.

    A manifest wins over everything else: it is written by the build and it names the
    interpreter the copy runs on, so there is nothing to weigh it against. Below it, the
    presence of `pyproject.toml` above the package is what makes `uv sync` a real command
    rather than a sentence about somebody else's machine.
    """
    shipped = bool(manifest) if manifest is not None else manifest_path() is not None
    if shipped:
        return KIND_INSTALLED
    here = Path(__file__).resolve().parent if package is None else Path(package)
    return KIND_CHECKOUT if (here.parent / PYPROJECT_NAME).is_file() else KIND_TOOL


def advice_key(base: str, kind: str | None = None) -> str:
    """The key of the one variant of `base` that is true for this install.

    `kind` is passed by a caller that already knows (the doctor probes once for a whole
    command); None asks this machine. The key is composed rather than looked up in a table
    so that a family cannot be half-registered: `advice_key` returns a name, and the
    catalog either has all three or fails the parity test in tests/test_i18n.py.
    """
    return f"{base}.{install_kind() if kind is None else kind}"


def advice_keys(base: str) -> tuple[str, ...]:
    """All three variants of `base`, in the order `KINDS` states them — for the tests."""
    return tuple(f"{base}.{kind}" for kind in KINDS)
