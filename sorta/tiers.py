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
from typing import Callable, Sequence

from . import i18n, wizard
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

# F222: what ONE model weighs, so a run can say what IT will download rather than what
# the tier carrying it would. The catalog prices a tier (`wizard.Tier.download_mb`) and
# that is the right unit for the wizard, which installs tiers — but the run screen is
# asked a different question: the classification stage fetches ViT-L-14 and nothing else,
# and quoting the 3.0 GB of the whole "Search by words" tier for it would overstate the
# download by almost half. `wizard.TIERS` is deliberately NOT touched for this (the F222
# boundary): the numbers live here, next to the markers that answer "is it on disk", and
# a test pins their sum to the catalog so the two cannot drift.
#
# ViT-L-14 is the 1.6 GB the owner's run of 2026-08-07 measured; XLM-RoBERTa is the rest
# of the 3.0 GB the catalog states for the tier that carries both.
_WEIGHT_MB: dict[str, int] = {
    "buffalo_l": 400,
    "ViT-L-14": 1600,
    "XLM-RoBERTa": 1400,
    "Qwen2.5-VL-3B": 7000,
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


# --- F222: which weights a run raises, and what that costs before it starts -----------
#
# F217 hung a note about a tier on an OPTION, and the pairing was done by eye: the two
# options that got one are the two whose checkbox happens to be named after a tier
# ("Faces", "Deep analysis"). Animals, landmarks and the classification all load the same
# CLIP ViT-L-14, which lives in a tier called "Search by words", so nobody connected them
# and all three went to the network without a word.
#
# So the pairing is derived rather than guessed: a part of the run names the WEIGHTS it
# can raise, and which tier that is follows from `wizard.TIERS`. A part whose weights
# nobody carries fails the suite, as does a checkbox of the run screen that is missing
# from this table — the guard exists because the next option added without a note would
# again be found by a person and not by a test.


@dataclasses.dataclass(frozen=True)
class RunPart:
    """One priced line of the run screen — or a stage that has no line and runs anyway.

    `optional` is False for the work nobody is asked about. Those are the ones this
    feature is really for: the classification stage pulls 1.6 GB on a fresh machine and
    there has never been a checkbox, a note or a number in front of it.
    """

    key: str
    weights: tuple[str, ...] = ()
    optional: bool = True


# The whole run, line by line. Keys are the ones the run screen already uses for its
# prices and its checkbox ids (`process-<key with dashes>-checkbox`), so the guard can
# read the markup and pair the two without a translation table.
RUN_PARTS: tuple[RunPart, ...] = (
    # index -> geo -> phash: file system, EXIF and arithmetic. No model at all, which is
    # why a machine that refused every tier still sorts a collection by city.
    RunPart("base", optional=False),
    # The verdicts. No checkbox by decision (without them screenshots, documents and
    # product shots ride into the city folders among the photographs), so the ONLY way
    # its 1.6 GB can be stated before the run is a line like this one.
    RunPart("classify", ("ViT-L-14",), optional=False),
    RunPart("geo_online"),
    # F222: a stage with a checkbox for the first time — 0.55% of the owner's places for
    # 1.6 GB and minutes of every run, so it is off unless somebody asks for it.
    RunPart("landmarks", ("ViT-L-14",)),
    RunPart("faces", ("buffalo_l",)),
    RunPart("events"),
    RunPart("pets", ("ViT-L-14",)),
    RunPart("pets_verify", ("Qwen2.5-VL-3B",)),
    RunPart("deep", ("Qwen2.5-VL-3B",)),
    RunPart("products", ("Qwen2.5-VL-3B",)),
    RunPart("junk_rescue", ("Qwen2.5-VL-3B",)),
    RunPart("landmarks_verify", ("Qwen2.5-VL-3B",)),
)

RUN_PARTS_BY_KEY: dict[str, RunPart] = {part.key: part for part in RUN_PARTS}

# Which stage of the pipeline raises which weights — the run-time half of the table
# above, used to say "the model is being downloaded" while it happens and to name the
# model when the download fails. A stage that is absent from here loads nothing.
STAGE_WEIGHTS: dict[str, tuple[str, ...]] = {
    "landmarks": ("ViT-L-14",),
    "classify": ("ViT-L-14",),
    "junk": ("ViT-L-14",),
    "faces": ("buffalo_l",),
}


def weight_tier(weight: str) -> str | None:
    """Which tier of the catalog carries that model — the derivation F222 turns on."""
    for tier in wizard.TIERS:
        if weight in tier.weights:
            return tier.key
    return None


def part_tiers(key: str) -> tuple[str, ...]:
    """The tiers one line of the run screen needs, in catalog order."""
    part = RUN_PARTS_BY_KEY.get(key)
    if part is None:
        return ()
    found = [weight_tier(weight) for weight in part.weights]
    order = [tier.key for tier in wizard.TIERS]
    return tuple(sorted({name for name in found if name}, key=order.index))


def weights_size_mb(weights: Sequence[str]) -> int:
    """What these models weigh together, in the catalog's own megabytes."""
    return sum(_WEIGHT_MB.get(name, 0) for name in dict.fromkeys(weights))


@dataclasses.dataclass(frozen=True)
class PartState:
    """One line of the run screen as this machine has it.

    `missing` is what would be DOWNLOADED if the line runs — the answer the run summary
    is built from. `available` is the other half and it is not the same question: weights
    arrive by themselves on first use, while packages do not, so a line whose packages
    are absent cannot run at all however long one waits.
    """

    key: str
    tiers: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    optional: bool = True
    available: bool = True

    @property
    def download_mb(self) -> int:
        return weights_size_mb(self.missing)


def run_parts(states: list[TierState] | None = None) -> list[PartState]:
    """Every line of the run, against ONE reading of what is installed.

    The states come from `tier_states()` — the same call `sorta doctor` and the F217
    notes make — and nothing here asks the disk a second time: which weights are absent
    is already in `TierState.missing_weights`.
    """
    by_key = {state.key: state for state in
              (tier_states() if states is None else states)}
    absent_weights = {weight for state in by_key.values()
                      for weight in state.missing_weights}
    result: list[PartState] = []
    for part in RUN_PARTS:
        needed = part_tiers(part.key)
        known = [by_key[name] for name in needed if name in by_key]
        result.append(PartState(
            key=part.key,
            tiers=needed,
            weights=part.weights,
            missing=tuple(name for name in part.weights if name in absent_weights),
            optional=part.optional,
            # A tier nobody probed is not called broken: an unknown state is the state of
            # a caller that passed a partial list, and refusing an option over it would
            # be the F216 mistake with the sign flipped.
            available=all(not state.missing_packages for state in known),
        ))
    return result


def stage_downloads(stage: str, states: list[TierState] | None = None) -> tuple[str, ...]:
    """The models `stage` is about to fetch, or () when it will not go to the network."""
    weights = STAGE_WEIGHTS.get(stage, ())
    if not weights:
        return ()
    by_key = {state.key: state for state in
              (tier_states() if states is None else states)}
    absent = {weight for state in by_key.values() for weight in state.missing_weights}
    return tuple(name for name in weights if name in absent)


# --- F222: the two sentences a download owes a person ---------------------------------
#
# 1.6 GB with nothing on screen is indistinguishable from a hang, and that is not a
# guess: the owner's report of 2026-08-07 says "it hung on landmarks". How MUCH has
# arrived is known to huggingface_hub and not to us, and reaching into its progress bars
# from three call sites would be a fragile way to learn it — so the honest floor is what
# is written here: which model, for which stage, how big, and that it happens once.
#
# The failure is the same sentence with the reason on the end. What a person gets today
# is `<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>` with no hint that a model was
# being fetched at all, let alone which one or what for; the traceback keeps going to the
# log, where it belongs.


def download_notice(stage: str, weights: Sequence[str], lang: i18n.Lang) -> str:
    """«Downloading X for stage Y, ~N GB — this happens once»."""
    return i18n.cli_text("cli.download.started", lang,
                         stage=i18n.stage_label(stage, lang),
                         weights=", ".join(weights),
                         size=wizard.human_size(weights_size_mb(weights), lang))


def download_failure(stage: str, weights: Sequence[str], lang: i18n.Lang,
                     error: object) -> str:
    """The refusal in words: the stage, the model, the size and the way out."""
    return i18n.cli_text(
        "cli.download.failed", lang, stage=i18n.stage_label(stage, lang),
        weights=", ".join(weights) or "-",
        size=wizard.human_size(weights_size_mb(weights), lang),
        error=str(error).strip() or error.__class__.__name__)
