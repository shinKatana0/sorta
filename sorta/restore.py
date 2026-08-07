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

F169: THE CEILING ON THE WAY IN, AND WHY IT IS SAID OUT LOUD
------------------------------------------------------------
The model is x4 and the transformer computes at the UPSCALED resolution, so a full 4000
px frame would be 16 000 px on the way through and fit on no card here. The frame is
therefore scaled to `features.restore_max_edge` first, and for a big frame that is a
trade nobody was told about:

    4032 x 3024 (12 Mpx)  ->  1024 x 768  ->  4096 x 3072

The same size out — through a quarter and back. For a SMALL frame (a downloaded picture,
an old scan) the ceiling never fires and the gain is pure, which is the case F149 was
built for. For a full-sized one the true detail of the original is dropped and the model
draws something plausible in its place, and the copy can look sharper while holding less
of what was there.

Two things follow, and they are the whole of this module's F169 change:

* the ceiling is a SETTING (`features.restore_max_edge`) and not a constant in the code,
  because it is the single number that decides what a person gets back;
* every answer states what the model was actually shown (`source_edge` / `input_edge`,
  and `rebuilt` from the two). A copy rebuilt from a reduced frame is not a silent
  outcome: the interface says so beside the frame, in the same breath as "done".

What to DO about a frame above the ceiling — tile it in native resolution, supersample
back down, or refuse the action altogether — is not decided here. It is decided by the
measurement `scripts/measure_restore.py` prints, on the three populations separately,
with a human looking at blind pairs. Guessing that was the mistake F149's first probe
already made once.

F185: THE FILE APPEARS AFTER THE ROW, NOT BEFORE IT
---------------------------------------------------
The copy used to be written under its final name and the row inserted by a SEPARATE
call, so an insert that failed left a file the index had never heard of. That is not a
cosmetic leak: the next `index` run reads such a file as a NEW photograph, and the
collection gains a near-duplicate nobody made. It happened for real — 81 `_restored`
files on the owner's archive, none of them in the index.

So the copy is now written to a staging name beside its destination, the row is written
while the file is still called that, and only then is it renamed into place (a rename
inside one directory is atomic). Every way out of that sequence that is not "the row is
in" takes the staging file with it. The other order — write, insert, delete on failure —
would also work; this one is preferred because it never deletes anything that could
already have been seen, and because the failure it guards against is the one that leaves
rubbish rather than the one that leaves nothing.

The failure itself is a CODE like all the others (`ERROR_DATABASE_BUSY`). SQLite lets one
writer in at a time and an index stage can be running from the terminal, so a busy index
is an ordinary state of this program, not a defect — and the caller was getting it as a
stack trace out of a request handler.

A caller that KEEPS the copy has one entry point for all of that: `restore_and_record`.
`restore_frame` on its own still writes a file and says nothing about the index, which is
what the measurement scripts want — and is exactly how the 80 orphans before it got there.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import accel, imaging
from .hashing import file_hash as hash_file

_log = logging.getLogger(__name__)

# The default of `features.restore_model`, and the model the second measurement chose.
DEFAULT_RESTORE_MODEL = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

# The default of `features.restore_max_edge` — the longer side the frame is scaled to
# BEFORE the model. A COMPROMISE, not an optimum: the model is x4, so a full 4000 px frame
# would come out at 16000 px and eat memory on the way there (the transformer works on the
# upscaled resolution). 1024 -> 4096 is a frame nobody's screen is short of, computed in
# about a second.
#
# It lives HERE only as the default of the setting (the shape `DEFAULT_RESTORE_MODEL` has
# above), because a threshold in the code is a threshold nobody can change: this one
# decides, alone, whether a person gets their own detail back or a plausible redrawing of
# it. Raising it costs memory as the SQUARE of the number, on the x4 output — see the
# table `scripts/measure_restore.py` prints before touching it.
DEFAULT_RESTORE_MAX_EDGE = 1024

# What the copy is called and how it is written. JPEG regardless of what the original was
# (the model's output is RGB pixels, and a HEIC/RAW source has nothing left to preserve
# by the time it gets here) — the name says `.jpg` so what a person sees and what is on
# disk agree. 95 is visually lossless at this size and keeps the copy comparable to the
# original in the same viewer.
RESTORED_SUFFIX = "_restored"
JPEG_QUALITY = 95

# F185: what the copy is called while it is being written and before the index knows
# about it. It ends in something that is plainly not a photograph, so a file left behind
# by a killed process reads as debris rather than as a frame — and, more concretely, so
# `restored_path`'s "never over an existing file" scan cannot mistake one for a copy that
# is already there.
STAGING_SUFFIX = ".part"

# The reasons, as codes rather than sentences: the interface translates them (three
# languages), and a caller that has to parse prose to tell "no weights" from "unreadable
# frame" would get it wrong the first time the wording is edited.
ERROR_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_DECODE_FAILED = "decode_failed"
ERROR_WRITE_FAILED = "write_failed"
# F185. Deliberately `_BUSY` and not `_FAILED`: this one is TEMPORARY. Nothing is broken —
# an index stage holds the single writer SQLite allows and it will let go, so the same
# press works a minute later. The interface reads the difference off the name to decide
# whether offering "try again" is honest, which a shared `write_failed` would hide.
ERROR_DATABASE_BUSY = "database_busy"

# SQLite's primary result codes for "somebody else is writing" (SQLITE_BUSY) and "the
# writer is in this very process" (SQLITE_LOCKED). Matched on the CODE rather than on the
# message, because the message is SQLite's to reword and none of our business.
_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6

UpscaleFn = Callable[[Image.Image], Image.Image]

# (final path, the staging file the bytes are in) -> the new `files.id`. Called by
# `restore_frame` while the copy is still staged; see `restore_and_record`, which is the
# implementation of it this module ships.
RecordFn = Callable[[Path, Path], int]


@dataclass(frozen=True)
class RestoreResult:
    """Either the copy that was written, or the reason there is none. Never both."""
    path: Path | None = None
    error: str | None = None
    detail: str | None = None
    # F169: what the model was actually shown. `source_edge` is the longer side of the
    # frame as it lies on disk, `input_edge` the longer side of what went into the model —
    # equal whenever the ceiling did not fire, which is the ordinary case this action was
    # built for. Both are carried on the result rather than logged, because the caller has
    # to be able to SAY it: a copy silently rebuilt from a reduced frame is exactly the
    # trade a person did not agree to.
    source_edge: int = 0
    input_edge: int = 0
    # F185: the `files.id` of the row the copy was indexed under, when the caller asked
    # for the copy to be indexed at all (`restore_and_record`). 0 otherwise — a plain
    # `restore_frame` writes a file and makes no claim about the index.
    file_id: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None

    @property
    def rebuilt(self) -> bool:
        """True when the copy came out of a REDUCED frame rather than the original one.

        Not "the copy is worse" — nobody has measured that yet, and the measurement is a
        person looking at blind pairs (`scripts/measure_restore.py`). It is the narrower
        fact the interface owes: the detail of the original was dropped on the way in and
        what replaced it was drawn.
        """
        return self.input_edge > 0 and self.source_edge > self.input_edge


# --- the model ----------------------------------------------------------------------

_UPSCALERS: dict[str, UpscaleFn] = {}


def load_swin2sr(model_name: str) -> UpscaleFn:  # pragma: no cover — ML, smoke test
    """Load Swin2SR through transformers -> upscale(image) -> image.

    Lazy-import, like every other model in this project (`naming.load_qwen`): the module
    imports without transformers installed, and the failure happens HERE, where the
    caller is already wrapping it into a reason. `transformers` lives in the `vlm` extra,
    so on a base install this raises ImportError and the interface says so.

    F220: the device comes from `accel` (CUDA -> MPS -> CPU) and is NOT wrapped in
    `accel.CpuFallback`, unlike the batch stages. This is one frame per press of a button,
    with a person waiting on it, and a Swin2SR pass over a full-size frame on the CPU is
    minutes rather than the seconds a card takes. A refusal here means the model has no
    Metal kernel for this work — that is an answer the caller already turns into a reason
    a person can read (nothing is cached on failure, the next press retries), and it is a
    better one than a button that silently starts taking four minutes.
    """
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    device = accel.torch_device(torch)  # F220: CUDA -> MPS -> CPU, chosen in one place
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


def _staging_path(dest: Path) -> Path:
    """A unique, obviously-temporary neighbour of `dest` for the copy to be written into.

    In the SAME directory, and that is the whole point: a rename within one directory is
    a single atomic operation, so the copy never exists under its real name in a
    half-written state and never has to be copied across a device boundary to get there.
    """
    fd, name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.stem}.",
                                suffix=STAGING_SUFFIX)
    os.close(fd)
    return Path(name)


def _discard(staged: Path) -> None:
    """Remove the staging file if it is still there — the exit route of every failure.

    Silent about its own failure on purpose: it is called while another problem is being
    reported, and a file that could not be removed must not replace the reason a person
    is waiting for. It is a `.part` next to the frame either way, not a photograph.
    """
    try:
        staged.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover — a file we created a moment ago
        _log.warning("restore: the staging file %s stayed behind (%s)", staged, exc)


def _is_database_busy(exc: sqlite3.OperationalError) -> bool:
    """Is this "somebody else is writing" rather than "the query is wrong"?

    The result code first, because that is SQLite's actual contract; the text only as a
    fallback for a driver that did not attach one. A `no such table` must not come back
    to a person as "the index is busy, try again" — it never stops being true.
    """
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (_SQLITE_BUSY, _SQLITE_LOCKED)
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def source_edge(src: Path) -> int:
    """The longer side of the frame AS IT LIES ON DISK; 0 if the file will not open.

    Read off the header (`Image.open` does not decode the pixels), because the decode
    below is already scaled to the ceiling and so cannot say what was given up on the way
    in. Orientation is not consulted on purpose: a rotation swaps the two sides, it does
    not change which of them is longer.
    """
    try:
        with Image.open(src) as im:
            return int(max(im.size))
    except Exception:  # noqa: BLE001 — a missing/corrupt file is answered by the decode
        return 0


def restore_frame(src: Path, model_name: str, *,
                  max_edge: int = DEFAULT_RESTORE_MAX_EDGE,
                  loader: Callable[[str], UpscaleFn] | None = None,
                  record: RecordFn | None = None) -> RestoreResult:
    """Process ONE frame and write the copy; the original is never opened for writing.

    The decode comes FIRST on purpose: a frame that will not read costs no 400 MB model
    load, and a person who pointed at a broken file gets the honest answer instead of a
    minute of waiting followed by the same one. Everything that can fail — a missing
    package, weights that are not on disk with no network to fetch them from, a decode,
    the write itself — becomes a `RestoreResult.error` rather than an exception: this is
    called from a request handler, and a stack trace is not a reason a person can act on.

    `max_edge` is `features.restore_max_edge` and arrives from the caller (F169). A frame
    at or below it is handed to the model UNTOUCHED — that is the case where this action
    is a pure gain and nothing here narrows it. A frame above it is still processed, and
    the result says it was rebuilt from a reduced copy of itself; what else should happen
    to such a frame is the measurement's decision, not this function's.

    F185: `record` indexes the copy, and is called while the copy is still under its
    STAGING name — the file takes its real name only once that call has returned. If it
    raises, the staging file goes and nothing is left on disk for the next `index` run to
    read as a photograph of its own. A busy index is answered like every other
    foreseeable state, with a code (`ERROR_DATABASE_BUSY`); anything else the recorder
    raises is a genuine defect and propagates, cleaned up but not swallowed. Without
    `record` this writes a file and says nothing about the index — which is what the
    measurement scripts want and NOT what a caller keeping the copy should do.
    """
    original_edge = source_edge(src)
    image = imaging.decode_rgb(src, max_edge, apply_orientation=True)
    if image is None:
        return RestoreResult(error=ERROR_DECODE_FAILED, detail=str(src))
    input_edge = int(max(image.size))
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
        staged = _staging_path(dest)
    except OSError as exc:
        _log.warning("restore: nothing could be written beside %s (%s)", dest, exc)
        return RestoreResult(error=ERROR_WRITE_FAILED, detail=f"{type(exc).__name__}: {exc}")
    file_id = 0
    try:
        try:
            processed.convert("RGB").save(staged, "JPEG", quality=JPEG_QUALITY)
        except OSError as exc:
            _log.warning("restore: copy %s was not written (%s)", dest, exc)
            return RestoreResult(error=ERROR_WRITE_FAILED,
                                 detail=f"{type(exc).__name__}: {exc}")
        if record is not None:
            try:
                file_id = int(record(dest, staged))
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc):
                    raise
                _log.warning("restore: the index is busy, %s was not kept (%s)", dest, exc)
                return RestoreResult(error=ERROR_DATABASE_BUSY,
                                     detail=f"{type(exc).__name__}: {exc}")
        try:
            os.replace(staged, dest)
        except OSError as exc:
            _log.warning("restore: copy %s could not be put in place (%s)", dest, exc)
            return RestoreResult(error=ERROR_WRITE_FAILED,
                                 detail=f"{type(exc).__name__}: {exc}")
    finally:
        # Every way out of the block above other than the rename — a reason returned, an
        # exception on its way to the caller — leaves the archive as it was found. Once
        # the rename has happened there is nothing here to remove.
        _discard(staged)
    if original_edge > input_edge:
        _log.info("restore: %s is %d px and the ceiling is %d — the copy is rebuilt from "
                  "a reduced frame, not sharpened from the original",
                  src, original_edge, input_edge)
    return RestoreResult(path=dest, source_edge=original_edge, input_edge=input_edge,
                         file_id=file_id)


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
                    model: str, now: str | None = None,
                    measured_from: Path | None = None) -> int:
    """Index the copy and record where it came from. Returns the new `files.id`.

    One transaction: a file row without its `restored_files` row would be a derived frame
    nothing knows is derived, which is exactly the near-duplicate pair this feature exists
    not to create. `orientation` is NULL because the decode already applied it, and
    `phash` is NULL because the next run computes it (and never groups it — see `dedup`).

    The hash IS computed here, on a file of a few megabytes: it is what the exact-duplicate
    pass and the sorter's copy verification both read, and a row without one is a file
    those two quietly skip.

    F185: `measured_from` is where the bytes are RIGHT NOW — the staging file the copy is
    written to before the index has heard of it. `dest` is what the row says either way:
    that is the name the file will carry, and the rename that puts it there preserves its
    size, its mtime and every byte the hash was taken over.
    """
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = Path(measured_from) if measured_from is not None else Path(dest)
    stat = written.stat()
    size, mtime = stat.st_size, stat.st_mtime
    try:
        file_hash, hash_algo = hash_file(written)
    except OSError:  # pragma: no cover — the file was written a moment ago
        file_hash, hash_algo = None, None
    width, height = _dimensions(written)
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


def restore_and_record(conn: sqlite3.Connection, source_id: int, src: Path,
                       model_name: str, *,
                       max_edge: int = DEFAULT_RESTORE_MAX_EDGE,
                       loader: Callable[[str], UpscaleFn] | None = None,
                       now: str | None = None) -> RestoreResult:
    """F185: the whole action — process the frame, index the copy, put it in place.

    THE ONE ENTRY POINT FOR A CALLER THAT KEEPS THE COPY. The two halves used to be two
    calls, and a caller that made the first and lost the second left a file in somebody's
    archive that the index had never heard of; the next run then read it as a new
    photograph. Joined here so the order cannot be got wrong from outside: the row goes in
    while the copy is still staged, and the copy takes its name only afterwards.

    Retries are NOT done here, on purpose. Waiting for the index to free up inside a
    request handler means holding a connection and a thread for as long as an index stage
    takes; whether to ask again — and whether to say so first — belongs to whoever pressed
    the button.
    """
    def record(dest: Path, staged: Path) -> int:
        return record_restored(conn, source_id, dest, model=model_name, now=now,
                               measured_from=staged)

    return restore_frame(src, model_name, max_edge=max_edge, loader=loader, record=record)


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
