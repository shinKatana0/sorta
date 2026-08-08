"""F224: the model weights an uninstall leaves behind — named, measured and removable.

After `unins000.exe` the disk still carried (owner's virtual machine, 2026-08-07):

    %APPDATA%\\sorta          config.yaml, sorta.db     - their own work
    %LOCALAPPDATA%\\sorta     logs, the preview cache
    ~/.cache/huggingface/hub  the CLIP weights, 1.6 GB and up
    ~/.insightface/models     buffalo_l, 0.3 GB

Sorta's own data already had its commands (`sorta cache --clear`, `--clear-geo`,
`sorta reset`); the weights had none. They belong to `sorta cache` because they are a
cache: derived files that come back on the next run that needs them.

Two reasons this is not a `Remove-Item`. The huggingface and insightface caches are
SHARED with any other program on those libraries, so only the directories of the models
the TIER CATALOG names are touched. And they can be LINKS — on the owner's machine
`~/.insightface` is a junction to `C:\\AI\\buffalo`, and `shutil.rmtree` walks INTO a
junction on Windows (`os.path.islink` is False for one), emptying the store it points at.
Hence the hand-written walk below.

The models are READ from the tier catalog (`wizard.TIERS`), never copied: a second list
drifts from the first the day a tier changes.
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
    """A symlink or a Windows junction. `is_symlink()` is False for a junction and
    `Path.is_junction()` needs 3.12, so the reparse bit answers for both on 3.11."""
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def link_behind(path: Path) -> Path | None:
    """The nearest ancestor of `path` that is a link (`~/.insightface -> C:\\AI\\buffalo`):
    what sits inside one is somebody else's store and not ours to delete."""
    for parent in path.parents:
        if is_link(parent):
            return parent
    return None


def _present(path: Path) -> bool:
    """Is there an entry there? Not followed through a link — a dangling junction counts."""
    try:
        path.lstat()
    except OSError:
        return False
    return True


def _size(path: Path) -> int:
    """What this entry occupies on THIS disk. A link is zero — removing it frees nothing,
    which keeps the number shown before a deletion equal to the bytes it returns."""
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

    `size` is what removing it frees: 0 for a link and for anything behind one. `behind`
    is the ancestor link it was found through — the reason an entry is shown, NOT removed.
    F225: `complete` comes from the probe's own rule (`tiers.download_complete`), so this
    module and `sorta doctor` cannot describe one directory two ways; an interrupted
    download is still LISTED, since 800 MB of wreckage is 800 MB to get back.
    """

    weight: str
    path: Path
    size: int = 0
    link: bool = False
    behind: Path | None = None
    complete: bool = True

    @property
    def removable(self) -> bool:
        return self.behind is None


def _entry(weight: str, path: Path) -> Downloaded:
    behind = link_behind(path)
    link = is_link(path)
    return Downloaded(weight=weight, path=path,
                      size=0 if link or behind is not None else _size(path),
                      link=link, behind=behind,
                      # A link is not walked to answer this: what is behind it is not ours.
                      complete=True if link else tiers.download_complete(path))


def _hub_entries(cache: Path) -> list[Path]:
    try:
        return sorted(cache.iterdir())
    except OSError:  # no cache directory at all — nothing has ever been downloaded
        return []


def _matches(weight: str, name: str) -> bool:
    """Does that cache entry hold this model? The FUNCTION `tiers._weights_cached` uses,
    not a second copy of the rule: a hub cache calls `ViT-L-14`
    `models--timm--vit_large_patch14_clip_224.openai`."""
    return tiers.entry_holds(weight, name)


def downloaded(*, insightface: Path | None = None,
               hub: Path | None = None) -> list[Downloaded]:
    """Every model of the catalog this disk holds, in catalog order. Two caches:
    insightface keeps buffalo_l in `~/.insightface/models/<name>`, everything else comes
    through huggingface_hub as `models--<org>--<repo>`."""
    models = tiers._INSIGHTFACE_MODELS if insightface is None else insightface
    entries = _hub_entries(hf_cache_dir() if hub is None else hub)
    found: list[Downloaded] = []
    seen: set[Path] = set()
    for weight in catalog_weights():
        # `<name>.zip` is the archive insightface unpacks next to itself, named after the
        # model exactly — so it is ours whenever it is there.
        candidates = [models / weight, models / f"{weight}.zip"]
        candidates += [child for child in entries if _matches(weight, child.name)]
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
    """What a removal did. `kept` is not a failure: it is the model behind somebody
    else's link, left there on purpose, which a person has to be told about."""

    removed: tuple[Path, ...] = ()
    freed: int = 0
    kept: tuple[Downloaded, ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()


def _unlink(path: Path) -> None:
    """Remove a link AS a link. `os.rmdir` never follows one and is what takes a junction
    or a directory symlink on Windows; a file symlink is not a directory anywhere."""
    try:
        os.rmdir(path)
    except OSError:
        path.unlink()


def _delete(path: Path) -> None:
    """Delete a tree without ever walking through a link. Not `shutil.rmtree`: on Windows
    it checks `os.path.islink`, which a junction answers False to, and recurses INTO it."""
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
    """Remove exactly these model directories — nothing around them, nothing behind a link."""
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
    weigh first; the deleting it does itself, they are its own `[Dirs]` and hold no
    shared cache.
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

    Machine-readable and NOT translated on purpose — an Inno Setup script reads it and
    shows the number in the language of the install. The uninstaller asks instead of
    measuring, for the same reason it calls `sorta cache --clear-models` instead of
    deleting: one rule, in one place, covered by ordinary tests (F211's precedent).
    """
    models = total_bytes(downloaded())
    data = sum(_size(path) for path in data_dirs() if _present(path))
    return f"models {models}\ndata {data}"
