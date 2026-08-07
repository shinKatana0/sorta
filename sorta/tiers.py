"""F216/F217: which install tiers this machine actually has — one probe, two screens.

The probe was written for `sorta doctor` (F216) and lived inside `sorta/cli.py`. F217
gives the web app the same answer next to the checkbox it belongs to, and the web app
cannot import the command line: `sorta.cli` pulls in typer, numpy and every stage
module, and `sorta/ui/` deliberately calls the leaf functions instead (the cli<->ui
cycle noted at the top of `ui/process.py`).

So the probe moved DOWN here rather than being written a second time. That is the whole
point of the move and not a tidying: two check screens that answer the same question
from two implementations disagree within a release — the precedent is F211, where the
wizard calls `sorta doctor` instead of growing a check screen of its own.

Nothing about the answer changed in the move. A tier is still two halves that fail
differently — the PACKAGES `uv` put into `{app}\\lib`, and the model WEIGHTS the stage
downloads on the first run that needs them — and they are still reported apart, because
a tier whose packages are in place and whose 400 MB are not is neither installed nor
missing, and there is a sentence for exactly that state.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Callable

from . import wizard
from .offline import hf_cache_dir

_INSIGHTFACE_MODELS = Path.home() / ".insightface" / "models"

# What a weight is CALLED once it is on disk. The catalog names them the way a person
# reads them (`ViT-L-14`); what lands in a cache is whatever the loader asked the hub
# for, and the two are not the same string — open_clip fetches the openai weights of
# ViT-L-14 as `timm/vit_large_patch14_clip_224.openai`. Deliberately a substring match
# on the cache entries and not a manifest: this answers "has this been downloaded",
# which is a question about disk, and a wrong revision still degrades the way it did
# before (the loader raises and names the opt-out). A weight named by `wizard.TIERS`
# and missing from here fails the suite, the way the extras do.
_WEIGHT_MARKERS: dict[str, tuple[str, ...]] = {
    "buffalo_l": ("buffalo_l",),
    "ViT-L-14": ("vit-l-14", "vit_large_patch14_clip_224"),
    "XLM-RoBERTa": ("xlm-roberta",),
    "Qwen2.5-VL-3B": ("qwen2.5-vl-3b",),
}


def _tier_hint_key(os_name: str = os.name) -> str:
    """How to add a tier, named the way it exists on this machine.

    The Start menu belongs to the Windows installer and to nothing else: a machine that
    got Sorta from `uv tool install` has no such entry, and sending its owner to look for
    one is exactly the kind of line this feature exists to remove.

    F217: the web app names the same way out, so the choice lives next to the probe
    rather than in `doctor` — a page that offered the Start menu on Linux would be the
    F213 defect again, one screen further out.
    """
    return "cli.doctor.tier_hint" if os_name == "nt" else "cli.doctor.tier_hint_posix"


@dataclasses.dataclass(frozen=True)
class TierState:
    """One tier, as this machine has it: what is missing, by name."""

    key: str
    missing_packages: tuple[str, ...] = ()
    missing_weights: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing_packages and not self.missing_weights


def _distribution_name(requirement: str) -> str:
    """`onnxruntime>=1.27.0` -> `onnxruntime` — the name a package is installed under."""
    name = requirement.strip()
    for separator in " <>=!~;[(":
        name = name.split(separator)[0]
    return name.strip()


def _package_present(name: str) -> bool:
    """Is that distribution installed? Metadata, not an import.

    Not `importlib.import_module`: the module a distribution provides is frequently not
    its name (`onnxruntime-gpu` imports as `onnxruntime`, the CUDA runtime packages as
    nothing at all), and importing torch to find out whether torch is there costs 4.5 s
    on a command whose whole job is to answer quickly.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return bool(version(name))
    except PackageNotFoundError:
        return False
    except Exception:  # noqa: BLE001 — unreadable metadata is an answer, not a crash
        return False


def _normalized(text: str) -> str:
    return text.lower().replace("_", "-")


def _weights_cached(name: str, *, insightface: Path | None = None,
                    hub: Path | None = None) -> bool:
    """Are that model's files already on this disk?

    Two caches, because the two families of weights are downloaded by different
    libraries: insightface keeps buffalo_l in `~/.insightface/models/<name>`, and
    everything else in the catalog comes through huggingface_hub, which names a model
    `models--<org>--<repo>`.
    """
    models = _INSIGHTFACE_MODELS if insightface is None else insightface
    folder = models / name
    try:
        if folder.is_dir() and any(folder.iterdir()):
            return True
    except OSError:
        pass
    markers = tuple(_normalized(marker) for marker in _WEIGHT_MARKERS.get(name, (name,)))
    try:
        entries = [_normalized(child.name)
                   for child in (hf_cache_dir() if hub is None else hub).iterdir()]
    except OSError:  # no cache directory at all — nothing has been downloaded yet
        return False
    return any(marker in entry for entry in entries for marker in markers)


def tier_states(*, package_present: Callable[[str], bool] = _package_present,
                weights_cached: Callable[[str], bool] = _weights_cached
                ) -> list[TierState]:
    """Every tier of the catalog, with what this machine is missing of it.

    The requirements come from `wizard.tier_requirements`, i.e. off the metadata the
    build put into the wheel from `pyproject.toml` — so a version bound edited in the
    project reaches this without anything being copied by hand. When there is no
    metadata to read at all (a source directory that was never installed), a tier with
    extras is reported as missing them by extra name rather than as present: nothing
    was verified, and saying "in place" about that would be the failure this whole
    feature is against.
    """
    states: list[TierState] = []
    for tier in wizard.TIERS:
        requirements = wizard.tier_requirements(tier)
        if tier.extras and not requirements:
            missing_packages = tuple(f"extra:{extra}" for extra in tier.extras)
        else:
            missing_packages = tuple(
                name for name in map(_distribution_name, requirements)
                if not package_present(name))
        states.append(TierState(
            key=tier.key,
            missing_packages=missing_packages,
            missing_weights=tuple(name for name in tier.weights
                                  if not weights_cached(name)),
        ))
    return states
