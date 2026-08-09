"""Keep model loading off the network once the weights are already on disk.

Project principle #3 is local-by-default, but every start of a stage that touches
open_clip / easyocr / transformers made huggingface_hub call out to huggingface.co
to check a revision — visible as "You are sending unauthenticated requests to the
HF Hub" in the console. The weights were already cached; the call bought nothing and
made a personal photo organiser talk to a server it did not need.

Forcing offline unconditionally is not an option: a fresh machine has to download the
CLIP weights once. So the switch is conditional on the cache actually holding
something — a machine that has already been set up never reaches for the network,
and a fresh one still completes its first download.

The variables are read by huggingface_hub/transformers AT IMPORT TIME, so this has to
run before anything pulls those modules in — hence the call sits at the CLI entry
point, not next to the model code.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_ALLOW_DOWNLOAD = "SORTA_ALLOW_MODEL_DOWNLOAD"
_HF_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def hf_cache_dir() -> Path:
    """Where huggingface_hub keeps its models, honouring its own overrides."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(var, "").strip()
        if value:
            return Path(value)
    home = os.environ.get("HF_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def hf_xet_cache_dir() -> Path:
    """Where hf_xet stages the chunks it is downloading.

    huggingface_hub 1.27 fetches through hf_xet, and hf_xet writes into `<HF_HOME>/xet`,
    materialising the file in `hub/` only when the last chunk has arrived. A progress line
    watching `hub/` therefore reports 0 for the whole download and the full size at the
    end — which is what a 1.6 GB fetch looked like on 2026-08-08.
    """
    value = os.environ.get("HF_XET_CACHE", "").strip()
    if value:
        return Path(value)
    home = os.environ.get("HF_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "xet"


# F225: what an ABORTED download leaves behind, by the names the two libraries give it.
# huggingface_hub writes a blob to `<sha>.incomplete` and renames it when the last byte
# has arrived, so a directory carrying one of these is a download that stopped halfway.
_UNFINISHED_SUFFIXES = (".incomplete", ".part", ".tmp")


def download_complete(path: Path) -> bool:
    """Is the model behind this cache entry whole, or is it half a download?

    F225: the run of 2026-08-08 left `models--timm--vit_large_patch14_clip_224.openai`
    with an empty snapshot and a `.incomplete` blob, and calling that "downloaded" makes
    the stage fail on every run for as long as the machine lives. So BOTH have to hold:
    nothing inside is still being written, and there is a finished file where the loader
    reads them (`snapshots/<revision>/...` for the hub, the directory for insightface).
    """
    if path.is_file():
        # insightface downloads `<model>.zip` next to the directory it unpacks it into;
        # the archive is one file and is whole or is not there.
        return not path.name.lower().endswith(_UNFINISHED_SUFFIXES)
    root = path / "snapshots" if (path / "snapshots").is_dir() else path
    finished = False
    try:
        for item in path.rglob("*"):
            if item.name.lower().endswith(_UNFINISHED_SUFFIXES):
                return False
            if not finished and (root == path or root in item.parents):
                finished = item.is_file()
    except OSError:  # unreadable entry — nothing can be claimed about it
        return False
    return finished


def models_are_cached(cache_dir: Path | None = None) -> bool:
    """True if the hub cache holds at least one downloaded model.

    Coarse on purpose — it decides whether the network is likely to be needed, not what
    is in the cache — but a HALF download does not count. One interrupted fetch used to
    leave `models--…` behind, and this function then declared the machine offline, so the
    next attempt met "cannot find the requested files in the local cache" with the network
    working perfectly. Met on Linux 2026-08-09; the same rule the tier probe uses.
    """
    directory = cache_dir if cache_dir is not None else hf_cache_dir()
    try:
        children = list(directory.iterdir())
    except OSError:
        return False
    return any(child.name.startswith("models--") and download_complete(child)
               for child in children)


# Set by `configure_model_offline` and read by whoever has to explain a failed download:
# the difference between "no network" and "we told the library there is none".
_OURS = False


def offline_by_us() -> bool:
    """Did THIS process switch the loaders offline?"""
    return _OURS


def configure_model_offline(cache_dir: Path | None = None) -> bool:
    """Switch huggingface off the network when its cache is already populated.

    Returns True if offline mode was turned on. Never overrides variables the user set
    themselves, and `SORTA_ALLOW_MODEL_DOWNLOAD=1` disables the whole thing — that is
    the escape hatch when a new model has to be fetched on a machine that already has
    others cached.
    """
    if os.environ.get(ENV_ALLOW_DOWNLOAD, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if any(os.environ.get(var, "").strip() for var in _HF_VARS):
        return False  # explicitly configured — leave it alone
    if not models_are_cached(cache_dir):
        return False  # nothing cached yet: the first download must be able to happen
    global _OURS
    for var in _HF_VARS:
        os.environ[var] = "1"
    _OURS = True
    return True
