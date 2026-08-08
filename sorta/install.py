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
