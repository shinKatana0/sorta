"""F216/F217: which install tiers this machine actually has — one probe, two screens.

The probe lives here and not in `sorta/cli.py` because the web app needs the same answer
and cannot import the command line (the cli<->ui cycle at the top of `ui/process.py`).
Two screens answering one question from two implementations disagree within a release.

A tier is two halves that fail differently: the PACKAGES `uv` put into `{app}\\lib` and
the model WEIGHTS a stage downloads on first use. They are reported apart — a tier whose
packages are in place and whose 400 MB are not is neither installed nor missing.
"""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Callable, Sequence

from . import i18n, install, wizard
from .offline import hf_cache_dir

_INSIGHTFACE_MODELS = Path.home() / ".insightface" / "models"

# What a weight is CALLED once it is on disk: the catalog names them the way a person
# reads them (`ViT-L-14`), a cache holds whatever the loader asked the hub for
# (`timm/vit_large_patch14_clip_224.openai`). A substring match and not a manifest — the
# question is about disk, and a wrong revision still degrades as it did (the loader raises
# and names the opt-out). A weight of `wizard.TIERS` missing from here fails the suite.
_WEIGHT_MARKERS: dict[str, tuple[str, ...]] = {
    "buffalo_l": ("buffalo_l",),
    "ViT-L-14": ("vit-l-14", "vit_large_patch14_clip_224"),
    "XLM-RoBERTa": ("xlm-roberta",),
    "Qwen2.5-VL-3B": ("qwen2.5-vl-3b",),
}

# F222: what ONE model weighs, so a run can price what IT will download instead of the
# whole tier carrying it (`wizard.Tier.download_mb` is the wizard's unit, which installs
# tiers). A test pins the sum here to the catalog so the two cannot drift.
#
# Measured 2026-08-07, then inside one 3.0 GB catalog line — and the gap is why F223 split
# that line: 1 631 MB every run needs against 1 397 MB only a search by words does.
_WEIGHT_MB: dict[str, int] = {
    "buffalo_l": 400,
    "ViT-L-14": 1600,
    "XLM-RoBERTa": 1400,
    "Qwen2.5-VL-3B": 7000,
}


def _tier_hint_key(kind: str | None = None) -> str:
    """How to add a tier, named the way it works on THIS INSTALL.

    F230: the question is which INSTALL this is, not which OS. `os.name` sent a checkout
    on Windows to a Start menu item it does not have, and one on Linux to `sorta-setup`,
    which would install into an environment the next `uv sync` rewrites.
    """
    return install.advice_key("cli.doctor.tier_hint", kind)


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
    """Is that distribution installed? Metadata, not an import: the module a distribution
    provides is often not its name (`onnxruntime-gpu` imports as `onnxruntime`), and
    importing torch to learn whether torch is there costs 4.5 s."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return bool(version(name))
    except PackageNotFoundError:
        return False
    except Exception:  # noqa: BLE001 — unreadable metadata is an answer, not a crash
        return False


def _normalized(text: str) -> str:
    return text.lower().replace("_", "-")


def entry_holds(weight: str, name: str) -> bool:
    """Does that cache entry (a directory name) hold this model of the catalog?

    The marker table above as a function, so everything answering "is it on disk" asks it
    the same way — `sorta/weights.py` lists the same directories for the uninstaller.
    """
    entry = _normalized(name)
    return any(_normalized(marker) in entry
               for marker in _WEIGHT_MARKERS.get(weight, (weight,)))


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


def _weights_cached(name: str, *, insightface: Path | None = None,
                    hub: Path | None = None) -> bool:
    """Are that model's files already on this disk, and all of them?

    Two caches: insightface keeps buffalo_l in `~/.insightface/models/<name>`, everything
    else comes through huggingface_hub as `models--<org>--<repo>`. F225: a directory is
    not an answer (`download_complete`) — a partial download reports as no download.
    """
    models = _INSIGHTFACE_MODELS if insightface is None else insightface
    if download_complete(models / name):
        return True
    try:
        entries = list((hf_cache_dir() if hub is None else hub).iterdir())
    except OSError:  # no cache directory at all — nothing has been downloaded yet
        return False
    return any(entry_holds(name, child.name) and download_complete(child)
               for child in entries)


def tier_states(*, package_present: Callable[[str], bool] = _package_present,
                weights_cached: Callable[[str], bool] = _weights_cached
                ) -> list[TierState]:
    """Every tier of the catalog, with what this machine is missing of it.

    The requirements come off the wheel metadata (`wizard.tier_requirements`), so a bound
    edited in `pyproject.toml` arrives here uncopied. With no metadata to read at all (a
    source directory nobody installed) a tier with extras is reported as missing them by
    name: nothing was verified, and "in place" is the answer this feature exists against.
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
# The pairing is DERIVED and not guessed: a part names the WEIGHTS it can raise and the
# tier follows from `wizard.TIERS`. F217 paired by eye, so only the two checkboxes named
# after a tier got a note while animals, landmarks and classification went to the network
# in silence. A part with unknown weights, or a checkbox missing from this table, fails
# the suite — the next option added without a note has to be found by a test.


@dataclasses.dataclass(frozen=True)
class RunPart:
    """One priced line of the run screen — or a stage that has no line and runs anyway.

    `optional` is False for the work nobody is asked about, which is what this feature is
    for: classification pulls 1.6 GB on a fresh machine behind no checkbox at all.
    """

    key: str
    weights: tuple[str, ...] = ()
    optional: bool = True


# The whole run, line by line. The keys are the ones the run screen uses for its prices
# and its checkbox ids (`process-<key with dashes>-checkbox`), so the guard can pair the
# markup with this table without a translation layer.
RUN_PARTS: tuple[RunPart, ...] = (
    # index -> geo -> phash: no model at all, which is why a machine that refused every
    # tier still sorts a collection by city.
    RunPart("base", optional=False),
    # The verdicts. No checkbox by decision — without them screenshots, documents and
    # product shots ride into the city folders — so a line like this is the only place
    # its 1.6 GB can be stated before the run.
    RunPart("classify", ("ViT-L-14",), optional=False),
    RunPart("geo_online"),
    # F222: 0.55% of the owner's places for 1.6 GB and minutes of every run, so it is off
    # unless somebody asks for it.
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

# Which stage raises which weights — the run-time half of the table above, used to say
# what is downloading and to name the model when it fails. Absent here means loads
# nothing.
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

    `missing` is what would be DOWNLOADED if the line runs; `available` is a different
    question — weights arrive by themselves on first use and packages do not, so a line
    whose packages are absent cannot run however long one waits.
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
    """Every line of the run, against ONE reading of what is installed — `tier_states()`,
    the same call `sorta doctor` and the F217 notes make. Nothing here asks the disk
    again: the absent weights are already in `TierState.missing_weights`."""
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
            # A tier nobody probed is not called broken — that is the state of a caller
            # that passed a partial list, and refusing an option over it would be the
            # F216 mistake with the sign flipped.
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


# --- F222/F225: the sentences a download owes a person, and the number under them ------
#
# 1.6 GB with nothing on screen is indistinguishable from a hang — the owner's reports of
# 2026-08-07 and 2026-08-08 both say "it hung on landmarks". A failure gets the same
# sentence with the reason on the end, instead of a bare SSL error naming no model.
#
# The progress is a question about FILES, not about somebody else's progress bar:
# huggingface_hub has one, insightface draws none, and reading a library's bar would
# report zero for half the catalog and break on the next release of the other half.

# Long enough not to fill the window of a slow download, short enough that the gap
# between two lines never reads as a stall.
PROGRESS_SECONDS = 5.0
_MB = 1_000_000


def megabytes(size: int) -> int:
    """Bytes as the megabytes every screen of this project prices a download in."""
    return size // _MB


def downloaded_bytes(cache: Path | None = None) -> int:
    """How much the model caches hold right now — progress measured on the disk.

    Whole caches rather than one model's directory: what a library names the folder it is
    filling is its own business, and a measurement that depends on those names reports
    zero the day one changes. BOTH of them, too — insightface never touches the hub, so
    watching one reported `0 MB of 400 MB` for a whole download.
    """
    if cache is not None:
        return _bytes_under(cache)
    return _bytes_under(hf_cache_dir()) + _bytes_under(_INSIGHTFACE_MODELS)


def _bytes_under(directory: Path) -> int:
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


def watch_download(work: Callable[[], None], report: Callable[[int], None], *,
                   measure: Callable[[], int] | None = None,
                   tick: float = PROGRESS_SECONDS) -> BaseException | None:
    """Run `work`, telling `report` how many bytes have arrived while it runs.

    Returns whatever `work` raised, or None: a refusal by the network is a sentence the
    caller words for its own screen, never a traceback out of here. A thread and not a
    subprocess — the download has to land in the cache of THIS user.
    """
    measured = downloaded_bytes if measure is None else measure
    failure: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as exc:  # noqa: BLE001 — handed back, not swallowed
            failure.append(exc)

    start = measured()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while True:
        worker.join(tick)
        if not worker.is_alive():
            break
        report(max(0, measured() - start))
    return failure[0] if failure else None


def download_notice(stage: str, weights: Sequence[str], lang: i18n.Lang) -> str:
    """«Downloading X for stage Y, ~N GB — this happens once»."""
    return i18n.cli_text("cli.download.started", lang,
                         stage=i18n.stage_label(stage, lang),
                         weights=", ".join(weights),
                         size=wizard.human_size(weights_size_mb(weights), lang))


def download_progress(weights: Sequence[str], done: int, lang: i18n.Lang) -> str:
    """«X of Y so far» — the same measurement the run screen draws, said in a console."""
    return i18n.cli_text("cli.download.progress", lang,
                         done=wizard.human_size(megabytes(done), lang),
                         size=wizard.human_size(weights_size_mb(weights), lang))


def download_failure(stage: str, weights: Sequence[str], lang: i18n.Lang,
                     error: object) -> str:
    """The refusal in words: the stage, the model, the size and the way out."""
    return i18n.cli_text(
        "cli.download.failed", lang, stage=i18n.stage_label(stage, lang),
        weights=", ".join(weights) or "-",
        size=wizard.human_size(weights_size_mb(weights), lang),
        error=str(error).strip() or error.__class__.__name__)
