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


def models_are_cached(cache_dir: Path | None = None) -> bool:
    """True if the hub cache holds at least one downloaded model.

    Deliberately coarse: it is a decision about whether the network is likely to be
    needed, not a manifest check. A cache with the wrong models still degrades
    gracefully — the loader raises, and the message names the opt-out.
    """
    directory = cache_dir if cache_dir is not None else hf_cache_dir()
    try:
        return any(child.name.startswith("models--") for child in directory.iterdir())
    except OSError:
        return False


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
    for var in _HF_VARS:
        os.environ[var] = "1"
    return True
