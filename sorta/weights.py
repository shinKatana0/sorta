"""F224: the gigabytes an uninstall leaves behind — named, measured and removable.

Removing Sorta removes the program and nothing else. That is the right default and the
wrong only option: the owner cleaning a virtual machine on 2026-08-07 found that after
`unins000.exe` the disk still carried

    %APPDATA%\\sorta          config.yaml, sorta.db     - their own work
    %LOCALAPPDATA%\\sorta     logs, the preview cache
    ~/.cache/huggingface/hub  the CLIP weights, 1.6 GB and up
    ~/.insightface/models     buffalo_l, 0.3 GB

and not one of those directories is named after Sorta. Sorta's OWN data already has its
commands (`sorta cache --clear`, `sorta cache --clear-geo`, `sorta reset`), so exactly
one thing was missing and this module is that one thing: the model weights. It is a new
answer to a question nothing could answer, not a third way to erase things — the
weights are reached through `sorta cache`, because a cache is what they are: derived
files that come back by themselves on the next run that needs them, costing time and
bandwidth and no human judgement at all. What `reset` erases is the opposite of that
(the names of people, the decisions about duplicates), and it cannot be re-downloaded.

Why this is not one line of `Remove-Item`
-----------------------------------------

* `~/.cache/huggingface` and `~/.insightface` are SHARED directories that belong to
  nobody in particular. Any program on the same huggingface_hub or insightface keeps
  its models there, and deleting a cache whole takes a neighbour's download with it.
  So only the directories of the models the TIER CATALOG names are touched, one by
  one, and everything else found in those caches is left exactly as it was.
* Those directories can be LINKS, and on the owner's machine one is: `~/.insightface`
  is a junction to `C:\\AI\\buffalo`. A recursive delete that walks through it destroys
  weights living somewhere else entirely — which is what `shutil.rmtree` does on
  Windows, because `os.path.islink` is False for a junction and rmtree walks straight
  in. Hence the hand-written walk below: a link is removed AS a link, and anything
  found BEHIND one is reported and left alone, because what is behind it is not ours
  to delete.

The list of models is READ from the tier catalog (`wizard.TIERS`) rather than copied.
A second list of model names drifts from the first one the day a tier changes, and the
catalog is being edited in parallel by F223 as this is written.
"""
from __future__ import annotations

import dataclasses
import os
import stat
from pathlib import Path
from typing import Sequence

from . import tiers, wizard
from .offline import hf_cache_dir


def catalog_weights() -> tuple[str, ...]:
    """Every model the catalog names, in catalog order and without repeats."""
    return tuple(dict.fromkeys(name for tier in wizard.TIERS for name in tier.weights))


# --- links, and how not to walk through one -------------------------------------------


def is_link(path: Path) -> bool:
    """A symlink or a Windows junction — something whose content is somewhere else.

    `Path.is_symlink()` answers False for a junction (it is a mount point, not a
    symbolic link), and `Path.is_junction()` exists only from 3.12 while this project
    supports 3.11 — so the reparse bit is read off the stat structure, which is
    available on every version and answers for both kinds at once.
    """
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def link_behind(path: Path) -> Path | None:
    """The nearest ancestor of `path` that is a link, if there is one.

    This is the `~/.insightface -> C:\\AI\\buffalo` case. The model directory found
    inside such an ancestor is a real directory in somebody else's store, and deleting
    it would be exactly the destruction the junction was supposed to make visible.
    """
    for parent in path.parents:
        if is_link(parent):
            return parent
    return None


def _present(path: Path) -> bool:
    """Is there an entry at that path? Asked without following a link, so that a
    dangling junction still counts as something a person can be shown."""
    try:
        path.lstat()
    except OSError:
        return False
    return True


def _size(path: Path) -> int:
    """What this entry occupies on THIS disk.

    A link is zero: its content is somewhere else, is not ours, and removing the link
    frees nothing. That is what keeps the number shown before the deletion equal to the
    number of bytes the deletion actually returns.
    """
    if is_link(path):
        return 0
    try:
        info = path.lstat()
    except OSError:
        return 0
    if not stat.S_ISDIR(info.st_mode):
        return int(info.st_size)
    try:
        children = list(path.iterdir())
    except OSError:  # unreadable directory — measured as nothing rather than crashing
        return 0
    return sum(_size(child) for child in children)


# --- what is on the disk ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Downloaded:
    """One model of the catalog, as this machine happens to hold it.

    `size` is what removing it frees, so it is 0 for a link and 0 for anything behind
    one — neither returns space here. `behind` is the ancestor link a directory was
    found through and it is the reason an entry is shown and NOT removed.
    """

    weight: str
    path: Path
    size: int = 0
    link: bool = False
    behind: Path | None = None

    @property
    def removable(self) -> bool:
        return self.behind is None


def _entry(weight: str, path: Path) -> Downloaded:
    behind = link_behind(path)
    link = is_link(path)
    return Downloaded(weight=weight, path=path,
                      size=0 if link or behind is not None else _size(path),
                      link=link, behind=behind)


def _hub_entries(cache: Path) -> list[Path]:
    try:
        return sorted(cache.iterdir())
    except OSError:  # no cache directory at all — nothing has ever been downloaded
        return []


def _matches(weight: str, name: str) -> bool:
    """Does that cache entry hold this model?

    The same substring rule `tiers._weights_cached` answers "is it downloaded" with, and
    deliberately the same table rather than a second one: what a weight is CALLED in a
    hub cache is not what the catalog calls it (`ViT-L-14` arrives as
    `models--timm--vit_large_patch14_clip_224.openai`), and two tables of that would
    disagree the first time a loader changed its mind.
    """
    markers = tiers._WEIGHT_MARKERS.get(weight, (weight,))
    entry = tiers._normalized(name)
    return any(tiers._normalized(marker) in entry for marker in markers)


def downloaded(*, insightface: Path | None = None,
               hub: Path | None = None) -> list[Downloaded]:
    """Every model of the catalog this disk holds, in catalog order.

    Two caches, because two libraries download the weights: insightface keeps buffalo_l
    in `~/.insightface/models/<name>`, everything else comes through huggingface_hub,
    which names a model `models--<org>--<repo>`.
    """
    models = tiers._INSIGHTFACE_MODELS if insightface is None else insightface
    cache = hf_cache_dir() if hub is None else hub
    found: list[Downloaded] = []
    seen: set[Path] = set()
    for weight in catalog_weights():
        candidates = [models / weight, models / f"{weight}.zip"]
        candidates += [child for child in _hub_entries(cache)
                       if _matches(weight, child.name)]
        for path in candidates:
            if path in seen or not _present(path):
                continue
            seen.add(path)
            found.append(_entry(weight, path))
    return found


def total_bytes(entries: Sequence[Downloaded]) -> int:
    """What removing these would free — links and their targets excluded."""
    return sum(entry.size for entry in entries)


# --- removal ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Removal:
    """What a removal did, in the three ways it can end.

    `kept` is not a failure: it is the model that lives behind somebody else's link and
    was deliberately left where it is, which a person has to be told rather than have
    silently done for them either way.
    """

    removed: tuple[Path, ...] = ()
    freed: int = 0
    kept: tuple[Downloaded, ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()


def _unlink(path: Path) -> None:
    """Remove a link AS a link, leaving whatever it points at alone.

    `os.rmdir` never follows one and is what removes a junction or a directory symlink
    on Windows; a file symlink is not a directory anywhere, and `unlink` takes it.
    """
    try:
        os.rmdir(path)
    except OSError:
        path.unlink()


def _delete(path: Path) -> None:
    """Delete a tree without ever walking through a link.

    Not `shutil.rmtree`: on Windows it recurses INTO a junction, because the check it
    makes is `os.path.islink`, which a junction answers False to. That is the difference
    between removing 400 MB of cached weights and emptying the directory they are a
    junction to.
    """
    if is_link(path):
        _unlink(path)
        return
    if path.is_dir():
        for child in path.iterdir():
            _delete(child)
        path.rmdir()
        return
    path.unlink()


def remove(entries: Sequence[Downloaded]) -> Removal:
    """Remove exactly these model directories — nothing around them, nothing behind a
    link, and nothing that was not asked for."""
    removed: list[Path] = []
    kept: list[Downloaded] = []
    failed: list[tuple[Path, str]] = []
    freed = 0
    for entry in entries:
        if not entry.removable:
            kept.append(entry)
            continue
        try:
            _delete(entry.path)
        except OSError as exc:  # a file held open by another process, a read-only tree
            failed.append((entry.path, str(exc)))
            continue
        removed.append(entry.path)
        freed += entry.size
    return Removal(tuple(removed), freed, tuple(kept), tuple(failed))


# --- the two numbers the Windows uninstaller states before it asks ---------------------


def data_dirs() -> tuple[Path, ...]:
    """This machine's own Sorta directories — the index, config.yaml, logs, previews.

    Named here because the uninstaller offers to delete them and has to say what they
    weigh first. It does the deleting itself: those directories are the ones its own
    `[Dirs]` section created, they are ours by name, and no shared cache is inside them.
    """
    from . import imaging
    from .runlog import default_log_path

    roots = {imaging.preview_dir().parent, default_log_path().parent.parent}
    appdata = os.environ.get("APPDATA", "").strip()
    if os.name == "nt" and appdata:
        roots.add(Path(appdata) / "sorta")
    return tuple(sorted(roots))


def report() -> str:
    """Two lines of bytes for the uninstaller page: `models N` and `data N`.

    Deliberately machine-readable and deliberately NOT translated — it is read by an
    Inno Setup script, which then shows the number in the language of the install. The
    uninstaller calls this instead of measuring the caches itself for the same reason it
    calls `sorta cache --clear-models` instead of deleting them itself: one rule, in one
    place, covered by ordinary tests (F211's precedent — the wizard calls `sorta doctor`
    rather than growing a check screen of its own).
    """
    models = total_bytes(downloaded())
    data = sum(_size(path) for path in data_dirs() if _present(path))
    return f"models {models}\ndata {data}"
