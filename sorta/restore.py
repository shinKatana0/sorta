"""F149: one frame, one button — a model-processed COPY beside the original.

Two measurements decided this feature, on eight and then six frames from the 10-90
sharpness band (blurred but not hopeless — below 10 there is nothing left to restore).
The first run compared the original, an unsharp mask and `swin2SR-classical-sr-x2` and
the mask won; that run was set up wrong, and the backlog says why — "classical" is
trained on clean bicubic downscaling, a degradation this archive does not contain. The
second run compared the original, the mask, `swin2SR-realworld-sr-x4` and model+mask,
and the real-world model won outright. The price is ~400 MB of weights and ~1 second per
frame on the card this was measured on: nothing for an action a person asks for, on one
frame they are looking at.

THE MODEL DOES NOT BRING BACK WHAT WAS LOST — IT DRAWS SOMETHING PLAUSIBLE. For an
archive that is more dangerous than the blur: a smeared frame is honestly smeared, while
a redrawn face looks real and is not. Everything about the shape of this module follows
from that:

* the original is never touched (principle #5) — the result is a NEW file;
* the copy is marked in its NAME (`…_restored.jpg`) and in the interface. Not "your
  photograph got better" but "here is a processed copy";
* there is no bulk anything. The button acts on ONE frame a person opened and chose;
  there is no stage, no CLI command and no route that takes a list. The addressees are a
  handful of frames, and batch processing would turn an archive into a collection of
  plausible forgeries.

The weights are loaded ON FIRST USE and kept for the life of the process
(`shared_upscaler`, the same arrangement as `naming.shared_vlm`) — a 400 MB download at
server start, for a button most sessions never press, is not a trade anybody asked for.
A load that fails is not cached: it propagates to the caller, which turns it into a
reason a person can read. Offline is an ordinary state for this product (the weights come
from the network), so "the model is not there" has to be an answer, never an empty result.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import imaging
from .hashing import file_hash as hash_file

_log = logging.getLogger(__name__)

# The default of `features.restore_model`, and the model the second measurement chose.
DEFAULT_RESTORE_MODEL = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

# The longer side the frame is scaled to BEFORE the model. A COMPROMISE, not an optimum:
# the model is x4, so a full 4000 px frame would come out at 16000 px and eat memory on
# the way there (the transformer works on the upscaled resolution). 1024 -> 4096 is a
# frame nobody's screen is short of, computed in about a second. Raise it if you have the
# VRAM and want the full sensor back; nothing downstream depends on the number.
MAX_INPUT_EDGE = 1024

# What the copy is called and how it is written. JPEG regardless of what the original was
# (the model's output is RGB pixels, and a HEIC/RAW source has nothing left to preserve
# by the time it gets here) — the name says `.jpg` so what a person sees and what is on
# disk agree. 95 is visually lossless at this size and keeps the copy comparable to the
# original in the same viewer.
RESTORED_SUFFIX = "_restored"
JPEG_QUALITY = 95

# The reasons, as codes rather than sentences: the interface translates them (three
# languages), and a caller that has to parse prose to tell "no weights" from "unreadable
# frame" would get it wrong the first time the wording is edited.
ERROR_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_DECODE_FAILED = "decode_failed"
ERROR_WRITE_FAILED = "write_failed"

UpscaleFn = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class RestoreResult:
    """Either the copy that was written, or the reason there is none. Never both."""
    path: Path | None = None
    error: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


# --- the model ----------------------------------------------------------------------

_UPSCALERS: dict[str, UpscaleFn] = {}


def load_swin2sr(model_name: str) -> UpscaleFn:  # pragma: no cover — ML, smoke test
    """Load Swin2SR through transformers -> upscale(image) -> image.

    Lazy-import, like every other model in this project (`naming.load_qwen`): the module
    imports without transformers installed, and the failure happens HERE, where the
    caller is already wrapping it into a reason. `transformers` lives in the `vlm` extra,
    so on a base install this raises ImportError and the interface says so.
    """
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_name).to(device)
    model.eval()

    def upscale(image: Image.Image) -> Image.Image:
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        array: Any = output.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        array = np.moveaxis(array, source=0, destination=-1)
        return Image.fromarray((array * 255.0).round().astype(np.uint8))

    return upscale


def shared_upscaler(model_name: str,
                    loader: Callable[[str], UpscaleFn] | None = None) -> UpscaleFn:
    """The single process-wide upscaler of `model_name`, built on first use.

    Cached by model name so the second press of the button costs no load. A build FAILURE
    is deliberately not cached — it propagates to the caller, which decides how to
    degrade, and a machine that has just been given the weights must not have to restart
    the server to use them.
    """
    upscale = _UPSCALERS.get(model_name)
    if upscale is None:
        upscale = (loader or load_swin2sr)(model_name)
        _UPSCALERS[model_name] = upscale
    return upscale


def loaded_models() -> tuple[str, ...]:
    """Which models this process has actually loaded — empty until the first press."""
    return tuple(_UPSCALERS)


def reset_upscalers() -> None:
    """Forget the loaded weights (tests; a caller that wants the memory back)."""
    _UPSCALERS.clear()


# --- one frame ----------------------------------------------------------------------


def restored_path(src: Path) -> Path:
    """Where the copy of `src` goes — beside it, and never over an existing file.

    Beside the original rather than in a folder of restored frames, and that choice is
    the one `config.example.yaml` records: the copy is an ordinary member of the
    collection (it is indexed, it goes into the layout, it can be gathered into an
    album), so a service folder would be a second place to look for a photograph. The
    `_1`, `_2` fallback is the sorter's rule (`sorter._resolve_dst`) for the same reason
    it has one — nothing this program writes may land on top of something that is
    already there.
    """
    base = src.with_name(f"{src.stem}{RESTORED_SUFFIX}.jpg")
    candidate = base
    n = 0
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.stem}_{n}{base.suffix}")
    return candidate


def restore_frame(src: Path, model_name: str, *,
                  max_edge: int = MAX_INPUT_EDGE,
                  loader: Callable[[str], UpscaleFn] | None = None) -> RestoreResult:
    """Process ONE frame and write the copy; the original is never opened for writing.

    The decode comes FIRST on purpose: a frame that will not read costs no 400 MB model
    load, and a person who pointed at a broken file gets the honest answer instead of a
    minute of waiting followed by the same one. Everything that can fail — a missing
    package, weights that are not on disk with no network to fetch them from, a decode,
    the write itself — becomes a `RestoreResult.error` rather than an exception: this is
    called from a request handler, and a stack trace is not a reason a person can act on.
    """
    image = imaging.decode_rgb(src, max_edge, apply_orientation=True)
    if image is None:
        return RestoreResult(error=ERROR_DECODE_FAILED, detail=str(src))
    try:
        upscale = shared_upscaler(model_name, loader)
    except Exception as exc:  # noqa: BLE001 — no transformers, no weights, no network
        _log.warning("restore: model %s did not load (%s)", model_name, exc)
        return RestoreResult(error=ERROR_MODEL_UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}")
    try:
        processed = upscale(image)
    except Exception as exc:  # noqa: BLE001 — out of memory, a broken runtime
        _log.warning("restore: model %s failed on %s (%s)", model_name, src, exc)
        return RestoreResult(error=ERROR_MODEL_UNAVAILABLE, detail=f"{type(exc).__name__}: {exc}")
    dest = restored_path(src)
    try:
        processed.convert("RGB").save(dest, "JPEG", quality=JPEG_QUALITY)
    except OSError as exc:
        _log.warning("restore: copy %s was not written (%s)", dest, exc)
        return RestoreResult(error=ERROR_WRITE_FAILED, detail=f"{type(exc).__name__}: {exc}")
    return RestoreResult(path=dest)


# --- the copy as a member of the collection -----------------------------------------
# The question this feature had to answer: once the copy "appears right there and can be
# kept", it stops being an export and becomes a candidate for the archive. Decided by the
# user on 2026-08-02, and every consequence of that is closed here rather than "later":
#
# 1. THE COPY IS INDEXED, like any other file — so it goes into the layout, into the
#    slices, into albums. "Kept" that meant only "the file was not deleted" would leave a
#    person looking for a photograph in the album where they put it and not finding it.
# 2. THE LINK TO THE ORIGINAL IS STORED (`restored_files`), not guessed from the name: a
#    name can change, the link cannot.
# 3. The pair "an original and its copy" is not a near-duplicate — see `dedup`, which
#    leaves derived files out of the groups. Without that, the next `phash` run would put
#    every restored frame back in front of the person who made it, forever.
#
# The row is written HERE and not by the indexer, and the reason is the capture date. The
# copy is not SCANNED, it is DERIVED: it carries the same photograph as its source, and
# reading its metadata off a re-encoded JPEG (which has none) would date it by mtime,
# i.e. today — and file it under this year's folder instead of the year in the picture.
# So the copy inherits the facts of the frame from the row of its source, and every
# downstream stage (`geo`, `junk`, `faces`) computes its own tables for it on the next
# run, exactly as it would for any other file.

_CLONE_COLUMNS = (
    "taken_at", "taken_at_source", "taken_at_confidence", "gps_lat", "gps_lon",
    "camera_make", "camera_model", "not_personal",
)


def _dimensions(path: Path) -> tuple[int, int]:
    """(width, height) of the written copy; (0, 0) if it will not open.

    Read off the file rather than off the PIL image in memory, for the same reason the
    hash is: what the index records has to be what is on disk.
    """
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # pragma: no cover — the file was written by Pillow a moment ago
        return 0, 0


def record_restored(conn: sqlite3.Connection, source_id: int, dest: Path, *,
                    model: str, now: str | None = None) -> int:
    """Index the copy and record where it came from. Returns the new `files.id`.

    One transaction: a file row without its `restored_files` row would be a derived frame
    nothing knows is derived, which is exactly the near-duplicate pair this feature exists
    not to create. `orientation` is NULL because the decode already applied it, and
    `phash` is NULL because the next run computes it (and never groups it — see `dedup`).

    The hash IS computed here, on a file of a few megabytes: it is what the exact-duplicate
    pass and the sorter's copy verification both read, and a row without one is a file
    those two quietly skip.
    """
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    stat = dest.stat()
    size, mtime = stat.st_size, stat.st_mtime
    try:
        file_hash, hash_algo = hash_file(dest)
    except OSError:  # pragma: no cover — the file was written a moment ago
        file_hash, hash_algo = None, None
    width, height = _dimensions(dest)
    clone = ", ".join(_CLONE_COLUMNS)
    with conn:
        cur = conn.execute(
            f"""INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                    phash, orientation, dup_of, error, indexed_at, {clone})
                SELECT ?, ?, ?, 'jpg', 'photo', ?, ?, NULL, NULL, NULL, NULL, ?, {clone}
                FROM files WHERE id = ?""",
            (str(Path(dest).resolve()), int(size), float(mtime), file_hash, hash_algo,
             stamp, int(source_id)))
        file_id = int(cur.lastrowid or 0)
        conn.execute("UPDATE files SET width = ?, height = ? WHERE id = ?",
                     (int(width), int(height), file_id))
        conn.execute(
            """INSERT INTO restored_files (file_id, source_file_id, model, created_at)
               VALUES (?, ?, ?, ?)""",
            (file_id, int(source_id), model, stamp))
    return file_id


def existing_copy(conn: sqlite3.Connection, source_id: int,
                  model: str) -> tuple[int, str] | None:
    """(file_id, path) of the copy this source already has from this model, or None.

    Pressing the button twice must not make a second copy — repeating the same work on
    the same frame with the same model has the same answer, and two identical redraws
    beside one photograph is the mess this looks up to avoid.
    """
    row = conn.execute(
        """SELECT f.id, f.path FROM restored_files r JOIN files f ON f.id = r.file_id
           WHERE r.source_file_id = ? AND r.model = ?""",
        (int(source_id), model)).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def forget_copy(conn: sqlite3.Connection, file_id: int) -> None:
    """Drop the copy's rows — used when the file it names is no longer on disk.

    A person may delete the copy in their file manager; the index must not then keep
    answering "you already have one" and showing a card for a file that is gone.
    """
    with conn:
        conn.execute("DELETE FROM restored_files WHERE file_id = ?", (int(file_id),))
        conn.execute("DELETE FROM files WHERE id = ?", (int(file_id),))
