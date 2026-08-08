"""F3 (Phase 3): faces.

Contract: reads files (path, dup_of IS NULL), writes ONLY into faces, face_clusters and
cluster_state (F212 — the one row saying what the current clusters are an answer to).
- embedding: a BLOB of 512 float32 little-endian (ArcFace), see docs/ARCHITECTURE.md §3.
- A faces row with bbox='[]' and an empty embedding is the marker "file processed, no faces"
  (incrementality without a schema change).
- Re-clustering preserves labels: a new cluster inherits the label of the old
  cluster with the largest intersection by face.id, if it is > 50%. After a rescan
  (F89) the face ids are all new, so the intersection is taken over file ids
  instead — see ClusterSnapshot.

Thresholds come from the config.yaml `faces:` section (typed, cfg.faces);
the defaults are the tuned Immich values.

F165: the detector is not asked about frames the classifier has already ruled out — the
`classify` stage runs before this one, and skipping them saves 4 300 frames of 24 195 on
the reference collection at 77 ms each. The rule is `media_class.verdict = 'photo'` OR no
row at all: NULL means nobody has classified this frame, so a collection whose owner runs
`sorta faces` alone is detected in full as before. A frame that becomes a photograph later
(the deep tier reclassifies 2 592 of 24 196) has no `faces` row, so the ordinary
incrementality of this stage picks it up — the economy must not turn into lost faces.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import numpy as np

from . import accel, imaging
from .config import Config
from .progress import PhaseCB, ProgressCB

_NO_FACES_BBOX = "[]"  # the "processed, no faces" marker
EMBED_DIM = 512

# (bbox [x1,y1,x2,y2], detector confidence, embedding of length EMBED_DIM)
FaceHit = tuple[list[float], float, np.ndarray]
# analyzer(path, exif_orientation) -> found faces; replaced in tests
Analyzer = Callable[[str, int | None], list[FaceHit]]
# infer(bgr image) -> found faces; a factory builds one session per worker thread
Infer = Callable[[np.ndarray], list[FaceHit]]
InferFactory = Callable[[], Infer]
# decode(path, exif_orientation) -> the frame the pool hands to a session
Decode = Callable[[str, int | None], np.ndarray]
# on_result(row, hits) — hits=None means the frame failed; called on the main thread
OnResult = Callable[["sqlite3.Row", "list[FaceHit] | None"], None]
# One pass over a batch of rows with a given decode; results on the calling thread
Pipeline = Callable[["list[sqlite3.Row]", Decode, OnResult], None]


# F88: the detector's input side, in px — buffalo_l's det_10g is trained at 640, and it
# costs 16.5 ms/frame against 13.4 at 512, for the small faces 512 loses (−9%). Lowering
# it trades recall for weak hardware; it is not a speed knob.
DET_SIZE_DEFAULT = 640


@dataclass(frozen=True)
class FacesSettings:
    """Phase-3 thresholds; the defaults are Immich's."""
    min_face_px: int = 40        # smaller — not embedded (quality filter)
    det_threshold: float = 0.7   # detector confidence threshold
    min_cluster_size: int = 5    # HDBSCAN; smaller — noise
    max_distance: float = 0.5    # cosine face-similarity threshold
    det_size: int = DET_SIZE_DEFAULT  # F88: detector input side, pinned (see _det_size)


def _det_size(cfg: Config) -> int:
    """`faces.det_size` — the detector input side; DET_SIZE_DEFAULT when unset or bad.

    A garbage value falls back with a warning rather than raising inside a worker thread:
    it must not take down a run that has already spent an hour on the collection.
    """
    raw = (cfg.raw.get("faces") or {}).get("det_size")
    if raw is None:
        return DET_SIZE_DEFAULT
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        logging.warning(
            "faces.det_size: некорректное значение %r — используется %d",
            raw, DET_SIZE_DEFAULT,
        )
        return DET_SIZE_DEFAULT
    return n


def _settings(cfg: Config) -> FacesSettings:
    f = cfg.faces
    return FacesSettings(
        min_face_px=int(f.min_face_px),
        det_threshold=float(f.det_threshold),
        min_cluster_size=int(f.min_cluster_size),
        max_distance=float(f.max_distance),
        det_size=_det_size(cfg),
    )


@dataclass
class FaceStats:
    files_total: int = 0      # new (unprocessed) files on input
    files_processed: int = 0
    faces_found: int = 0
    no_face_files: int = 0
    errors: int = 0           # files with a read/decode error — will be retried


@dataclass
class ClusterStats:
    faces: int = 0
    clusters: int = 0
    noise: int = 0
    labels_kept: int = 0      # clusters that inherited a label on recomputation
    malformed: int = 0        # embeddings of the wrong length — excluded, cluster_id=NULL
    # F212: nothing that decides the clusters had moved, so they were left alone and the
    # numbers above describe what is ALREADY in the base — see _stored_cluster_stats.
    skipped: bool = False


# --- Detection + embeddings ------------------------------------------------

def _apply_orientation(img: np.ndarray, orientation: int | None) -> np.ndarray:
    """EXIF 274: cv2 does not rotate by itself, and the detector is orientation-sensitive.

    Mirror variants (2,4,5,7) are rare and do not affect detection — we ignore them.
    """
    if orientation == 3:
        return np.ascontiguousarray(np.rot90(img, 2))
    if orientation == 6:  # needs a 90° clockwise rotation
        return np.ascontiguousarray(np.rot90(img, 3))
    if orientation == 8:  # needs a 90° counter-clockwise rotation
        return np.ascontiguousarray(np.rot90(img, 1))
    return img


def _enable_cuda_dll_dirs() -> None:  # pragma: no cover — Windows-specific
    """CUDA/cuDNN come as pip wheels (the nvidia-* packages), not a system Toolkit.

    onnxruntime resolves provider-DLL dependencies through the classic PATH search, and
    its preload_dlls() (1.27) does not know the nvidia/cu13 layout — without these
    directories on PATH, ORT silently falls back to CPUExecutionProvider.
    """
    if sys.platform != "win32":
        return
    import site
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for dll_dir in {p.parent for p in nv.rglob("*.dll")}:
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def _read_image_bgr(path: str) -> np.ndarray:
    """Decode an image into a BGR array for insightface. ValueError if nothing can.

    cv2.imdecode returns None on HEIC/HEIF (the typical iPhone format), hence the Pillow +
    pillow-heif fallback. The bytes are read here rather than by cv2.imread, which does
    not take non-ASCII paths on Windows.
    """
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is not None:
        return img
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        with Image.open(path) as pil:
            rgb = np.asarray(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError(f"не удалось декодировать изображение: {path} ({exc})") from None


def _decode_for_faces(path: str, orientation: int | None) -> np.ndarray:
    """Full-resolution decode + rotation — the unit of work of the prefetch-decode pool.

    No downscale: the ArcFace embedding crops the face from the original, and
    shrinking the input would hurt clustering accuracy. Since F91 this is paid only
    for frames the gate found a face on — see _decode_preview_for_faces.
    """
    return _apply_orientation(_read_image_bgr(path), orientation)


# F91: the two passes, and why the second one detects AGAIN instead of reusing the boxes
# of the first.
#
# The step was decode-bound (16.6 frames/s with the GPU at 3-10%): every frame was
# decompressed at full resolution and 69% of them had no face at all. Detection does not
# need those pixels — insightface downscales its input to det_size=640 whatever it is
# given, so a 4000 px original and a 1536 px preview reach the network as the same 640 px
# frame. The crop DOES need them: ArcFace embeds the face out of the original.
#
# REJECTED: scaling the preview's boxes into original coordinates and handing them to
# recognition. It saves one detection pass (16.5 ms on the 31% of frames that have a face)
# at the price of splitting `app.get` into `det_model.detect` plus a hand-built alignment —
# and an alignment that ever takes coordinates from the wrong space makes the embeddings
# drift silently and the clusters rot weeks later. There is no insightface/GPU in this
# environment to prove equivalence on. So the preview is strictly a GATE ("is there
# anything here to crop?"), a frame that passes it goes through the unchanged
# `app.get(original)`, and the saved decode is untouched: the 69% never reach a full one.
#
# Unverified on real data: the gate's recall. Only frames where the preview sees nothing
# and the original would have seen something are lost; `sorta faces --rescan` before and
# after answers it.

def _decode_preview_for_faces(path: str, orientation: int | None) -> np.ndarray | None:
    """F91: a ~1536 px BGR frame for the detection GATE, or None if there is no cheap one.

    The frame comes from the shared preview cache (F67): warm a read of a small JPEG, cold
    a draft decode of the original (a DCT downscale, ~46 ms against ~1000 ms for the full
    frame on a 13 MP camera JPEG) that also fills the cache for other stages. With the
    cache off this still decodes SMALL and merely writes nothing — that is where the win is.

    None means "no cheap frame here" (an unreadable source, or a frame no smaller than the
    preview) and the caller goes the old way silently. mtime and size come from a local
    stat, which keeps the (path, orientation) signature the decode pool works with.
    """
    edge = imaging.preview_max_edge()
    try:
        st = os.stat(path)
    except OSError:
        return None
    img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=edge)
    if img is None or max(img.size) < edge:
        return None
    # PIL gives RGB, insightface wants BGR and a contiguous buffer (the reversed view is
    # neither). The preview is stored unrotated, as the source is, so orientation is
    # applied from the INDEX exactly as on the full path.
    return _apply_orientation(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), orientation)


class _GateDecoder:
    """The decode of the gate pass: a preview when there is one, the original otherwise.

    Which of the two a frame got decides how its hits are read, and that answer is needed
    on the main thread while the decode runs in the pool. Hence a set of paths rather than
    a second return value: the decode callable is (path, orientation) -> frame, one
    signature shared with the full-resolution path, and files.path is UNIQUE.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previewed: set[str] = set()

    def __call__(self, path: str, orientation: int | None) -> np.ndarray:
        try:
            img = _decode_preview_for_faces(path, orientation)
        except Exception:  # the gate must not fail a frame the old path could read
            img = None
        if img is None:  # the fallback: quiet, cheap, and exactly the old path
            return _decode_for_faces(path, orientation)
        with self._lock:
            self._previewed.add(path)
        return img

    def previewed(self, path: str) -> bool:
        """Was this frame decoded as a preview (so its hits are only a gate answer)?"""
        with self._lock:
            return path in self._previewed


def _decode_workers(cfg: Config) -> int:
    """Threads that decode frames — the same knob on both paths since F87.

    Before F87 the parallel path decoded inside its inference workers and never read this
    setting, so tuning it did nothing on a GPU machine.
    """
    n = (cfg.raw.get("faces") or {}).get("decode_workers")
    if n:
        return max(1, int(n))
    return min(8, os.cpu_count() or 4)


def _infer_workers(cfg: Config) -> int:
    """How many independent inference sessions run in parallel (F12.1).

    faces is inference-bound, not decode-bound: `app.get` takes ~256 ms/frame at ~6% GPU
    load while the decode pool feeds it ~9× faster. Measured ×3.17 on 4 sessions, ~0.6 GB
    VRAM each. CPU-only and CoreML machines keep the single-session default — there the
    sessions merely oversubscribe the cores, and how many Metal ones a Mac wants is a
    measurement nobody has made.
    """
    n = (cfg.raw.get("faces") or {}).get("infer_workers")
    if n:
        return max(1, int(n))
    return 4 if accel.cuda_provider_available() else 1


def _detect_parallel(
    rows: list[sqlite3.Row],
    decode: Decode,
    infer_factory: InferFactory,
    workers: int,
    decode_workers: int,
    on_result: OnResult,
) -> None:
    """Decode pool feeding `workers` inference sessions; results on the caller's thread.

    F87: decode and inference used to be one unit of work per thread — while a worker read
    a 40 MB RAW its own GPU session sat idle, and the card stayed at 2-5% on a real run.
    Decoupled here into the shape F64 gave CLIP, measured ×1.57 on 4 sessions (8.14 ->
    12.78 img/s, 500 real frames, RTX 5090 Laptop), same 300 faces, no extra VRAM.

    Every worker builds its OWN session on first use: an onnxruntime session is not
    thread-safe. The in-flight window is bounded on BOTH sides (~2×decode_workers decoded,
    ~2×workers waiting here), because an unbounded decode pool of full-res frames would
    read the whole collection into memory.

    `on_result(row, hits)` is called strictly from this thread, which keeps SQLite
    single-writer. hits=None means the frame failed and does not stop the rest; input order
    is not preserved.
    """
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    workers = max(1, workers)
    window = workers * 2  # bounded in-flight window: full-res frames are heavy
    local = threading.local()
    frames = _prefetch_decode(rows, decode, decode_workers)

    def process(img: np.ndarray) -> list[FaceHit]:
        infer: Infer | None = getattr(local, "infer", None)
        if infer is None:
            infer = local.infer = infer_factory()
        return infer(img)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future, sqlite3.Row] = {}

        def _fill() -> None:
            """Top the inference queue up from the decode pool (this thread only)."""
            while len(pending) < window:
                nxt = next(frames, None)
                if nxt is None:
                    return
                r, img, err = nxt
                if img is None or err is not None:  # undecodable frame — report, go on
                    on_result(r, None)
                    continue
                pending[pool.submit(process, img)] = r

        _fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                r = pending.pop(fut)
                try:
                    hits: list[FaceHit] | None = fut.result()
                except Exception:  # a broken frame — do not crash the pipeline
                    hits = None
                on_result(r, hits)
            _fill()


def _prefetch_decode(
    rows: list[sqlite3.Row],
    decode: Decode,
    max_workers: int,
) -> Iterator[tuple[sqlite3.Row, np.ndarray | None, Exception | None]]:
    """Decode frames in a thread pool with a bounded window (~2×max_workers in flight).

    Yields (row, image, error) as decoding completes; input order is not preserved. No
    inference happens here — the caller decides where the frame is inferred, on its own
    thread or in the session pool of `_detect_parallel` (F87).
    """
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    workers = max(1, max_workers)
    window = workers * 2
    it = iter(rows)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future, sqlite3.Row] = {}

        def _fill() -> None:
            while len(pending) < window:
                r = next(it, None)
                if r is None:
                    return
                pending[pool.submit(decode, r["path"], r["orientation"])] = r

        _fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                r = pending.pop(fut)
                try:
                    yield r, fut.result(), None
                except Exception as exc:  # an undecodable frame — do not crash the pipeline
                    yield r, None, exc
            _fill()


def _detect_serial(
    rows: list[sqlite3.Row],
    decode: Decode,
    infer: Infer,
    decode_workers: int,
    on_result: OnResult,
) -> None:
    """One session on THIS thread, fed by the decode pool — the CPU profile's pipeline.

    Extracted from `detect_faces` unchanged (F91): the gate runs the pipeline twice,
    and both passes must reuse the one session — loading buffalo_l costs seconds.
    """
    for r, img, err in _prefetch_decode(rows, decode, decode_workers):
        hits: list[FaceHit] | None = None
        if err is None:
            assert img is not None
            try:
                hits = infer(img)
            except Exception:  # a broken frame — do not crash the pipeline
                hits = None
        on_result(r, hits)


def _pipeline(factory: InferFactory, workers: int, decode_workers: int) -> Pipeline:
    """The runner both passes of a run share: rows + a decode -> results on this thread.

    `workers == 1` (the CPU profile) builds its single session HERE, once, so the second
    pass of the F91 gate does not load the model again. On the parallel path every pass
    builds its own thread-local sessions instead — a few seconds of model load per run
    against the minutes the gate saves, where sharing would mean keeping one executor alive
    across both passes.
    """
    if workers > 1:
        def parallel(rows: list[sqlite3.Row], decode: Decode, on_result: OnResult) -> None:
            _detect_parallel(rows, decode, factory, workers, decode_workers, on_result)

        return parallel

    infer = factory()

    def serial(rows: list[sqlite3.Row], decode: Decode, on_result: OnResult) -> None:
        _detect_serial(rows, decode, infer, decode_workers, on_result)

    return serial


def _split_for_gate(
    rows: list[sqlite3.Row],
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """(frames worth gating, frames that take the old path straight away) — F91.

    The gate only pays off when the original is bigger than the preview, which the indexer's
    width/height answer without touching the file; a frame whose dimensions are missing
    keeps the old path. One that slips through anyway is caught by
    _decode_preview_for_faces returning None.
    """
    edge = imaging.preview_max_edge()
    gated: list[sqlite3.Row] = []
    direct: list[sqlite3.Row] = []
    for r in rows:
        width, height = r["width"], r["height"]
        big = bool(width) and bool(height) and max(int(width), int(height)) > edge
        (gated if big else direct).append(r)
    return gated, direct


# buffalo_l loads 5 sub-models by default, and the pipeline uses only FaceHit —
# landmark_2d_106/landmark_3d_68/genderage would be computed per face and discarded.
# Recognition aligns by the 5 kps of det_10g, not by these, so disabling them does not
# change the embeddings (verified by the F47 smoke comparison).
_ALLOWED_MODULES = ["detection", "recognition"]


def _insightface_infer(s: FacesSettings) -> Infer:  # pragma: no cover — ML, smoke test
    """insightface buffalo_l: GPU (CUDA) with a CPU fallback. One session per worker.

    F88: without an explicit `det_size` insightface 1.0.1 leaves the detector in its
    two-pass mode (`set det-size: [(128, 128), (640, 640)]`) and runs the network TWICE per
    frame; both passes cost the same ~78 ms even though the first has 25× less to compute,
    because the price is in switching the input shape. Pinning one shape: 165.1 -> 16.5
    ms/frame on 100 real frames, 57 faces against 56. The first call stays expensive
    (plans/kernels warm up).
    """
    from insightface.app import FaceAnalysis

    _enable_cuda_dll_dirs()

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=_ALLOWED_MODULES,
        providers=accel.onnx_providers(),  # F214: CUDA -> CoreML -> CPU, in one place
    )
    app.prepare(ctx_id=0, det_thresh=float(s.det_threshold),
                det_size=(int(s.det_size), int(s.det_size)))

    def infer(img: np.ndarray) -> list[FaceHit]:
        return [
            (list(map(float, f.bbox)), float(f.det_score), f.embedding)
            for f in app.get(img)
        ]

    return infer


def _insightface_analyzer(s: FacesSettings) -> Analyzer:  # pragma: no cover — ML, smoke test
    """A serial analyzer (decode + inference in one call) — used in the smoke test."""
    infer = _insightface_infer(s)

    def analyze(path: str, orientation: int | None) -> list[FaceHit]:
        return infer(_decode_for_faces(path, orientation))

    return analyze


def _write_hits(
    conn: sqlite3.Connection, s: FacesSettings, stats: FaceStats,
    r: sqlite3.Row, hits: list[FaceHit], replace: bool = False,
) -> None:
    """Write the faces of one file; `replace` drops its previous rows first (F89).

    The delete lives INSIDE the per-file transaction so that a file always has either its
    old faces or its new ones. Wiping the whole table up front would be simpler, but a
    Ctrl+C halfway through a rescan of 24k frames would leave a collection with no faces.
    """
    kept = [
        (bbox, score, emb) for bbox, score, emb in hits
        if score >= s.det_threshold
        and min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= s.min_face_px
    ]
    with conn:  # one transaction per file: Ctrl+C-safe
        if replace:
            conn.execute("DELETE FROM faces WHERE file_id = ?", (r["id"],))
        if kept:
            conn.executemany(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
                [
                    (r["id"],
                     json.dumps([round(float(v), 1) for v in bbox]),
                     np.asarray(emb, dtype="<f4").tobytes())
                    for bbox, _score, emb in kept
                ],
            )
            stats.faces_found += len(kept)
        else:
            conn.execute(
                "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, ?, ?)",
                (r["id"], _NO_FACES_BBOX, b""),
            )
            stats.no_face_files += 1
    stats.files_processed += 1


# The canonical photos phase 3 works on: originals only (no duplicates), readable,
# and stills — a video has no faces row of its own.
_CANONICAL = "dup_of IS NULL AND error IS NULL AND media_type = 'photo'"

# F165: the same string `junk.QUALITY_VERDICT` holds, spelled out rather than imported so
# this module does not pull the whole classification stage in for one value. A test pins
# the two to each other.
_PHOTO_VERDICT = "photo"

# NOT EXISTS rather than a join or `verdict = ?`, because the frames with NO row have to
# pass: NULL means "nobody has classified this", the state of every collection whose owner
# runs `sorta faces` on its own.
_CLASSIFIED_AS_PHOTO = (
    "NOT EXISTS (SELECT 1 FROM media_class mc"
    f" WHERE mc.file_id = files.id AND mc.verdict != '{_PHOTO_VERDICT}')"
)

# width/height (F91) decide whether a frame is worth gating — see _split_for_gate.
_DETECT_COLUMNS = "id, path, orientation, width, height"


def _files_to_detect(
    conn: sqlite3.Connection, rescan: bool, limit: int | None,
) -> list[sqlite3.Row]:
    """The files this run detects on — all three narrowed by the F165 classification.

    Default (F68): only files with no faces row at all, the "no faces" marker counting as
    processed. `rescan` (F89) takes every canonical photo again, and with `limit` only N of
    them, picked at RANDOM rather than the first N by id: the head of a collection is one
    folder from one camera and answers a different question than "what does this cost on my
    photos".
    """
    if not rescan:
        return conn.execute(
            f"""SELECT {_DETECT_COLUMNS} FROM files
                WHERE {_CANONICAL} AND {_CLASSIFIED_AS_PHOTO}
                  AND id NOT IN (SELECT file_id FROM faces)
                ORDER BY id"""
        ).fetchall()
    if limit is None:
        return conn.execute(
            f"SELECT {_DETECT_COLUMNS} FROM files "
            f"WHERE {_CANONICAL} AND {_CLASSIFIED_AS_PHOTO} ORDER BY id"
        ).fetchall()
    return conn.execute(
        f"SELECT {_DETECT_COLUMNS} FROM files "
        f"WHERE {_CANONICAL} AND {_CLASSIFIED_AS_PHOTO} "
        f"ORDER BY RANDOM() LIMIT ?", (limit,)
    ).fetchall()


def detect_faces(
    cfg: Config, conn: sqlite3.Connection,
    progress: ProgressCB | None = None,
    analyzer: Analyzer | None = None,
    infer_factory: InferFactory | None = None,
    rescan: bool = False,
    limit: int | None = None,
) -> FaceStats:
    """Find faces in new canonical photos and write embeddings into faces.

    Incrementality: files that already have rows in faces (including the "no faces"
    marker) are skipped. A file with a read error gets no row and is retried next run.

    `rescan` (F89) detects the selected files again and replaces their old rows, one file
    per transaction; `limit` narrows it to N random files. A rescan gives every face a NEW
    id, so cluster labels can no longer be matched back by face — take
    `snapshot_clusters(conn)` BEFORE calling this and hand it to
    `cluster_faces(inherit_from=...)`, as `detect_and_cluster(rescan=True)` does, or every
    name the user typed is lost.

    The mock path (an `analyzer` passed, as in tests) is strictly serial. The real path
    runs `_infer_workers(cfg)` sessions in parallel (F12.1) fed by `_decode_workers(cfg)`
    decoding threads (F87), and in TWO passes — a preview gate, then the originals it
    promoted (see the F91 block comment above `_decode_preview_for_faces`).

    SQLite is written only from this thread in every case, one transaction per file.
    """
    if limit is not None and not rescan:
        raise ValueError("limit имеет смысл только вместе с rescan")
    s = _settings(cfg)
    rows = _files_to_detect(conn, rescan, limit)
    stats = FaceStats(files_total=len(rows))
    if not rows:
        return stats

    if analyzer is not None:
        for i, r in enumerate(rows, 1):
            try:
                hits = analyzer(r["path"], r["orientation"])
            except Exception:
                stats.errors += 1
                continue
            _write_hits(conn, s, stats, r, hits, replace=rescan)
            if progress:
                progress(i, len(rows))
        return stats

    factory: InferFactory = infer_factory or (lambda: _insightface_infer(s))
    done = 0

    def on_result(r: sqlite3.Row, hits: list[FaceHit] | None) -> None:
        """Called only from this thread — the single writer to SQLite."""
        nonlocal done
        if hits is None:
            stats.errors += 1
        else:
            _write_hits(conn, s, stats, r, hits, replace=rescan)
        done += 1
        if progress:
            progress(done, len(rows))

    run = _pipeline(factory, _infer_workers(cfg), _decode_workers(cfg))
    gated, direct = _split_for_gate(rows)

    if gated:
        decoder = _GateDecoder()
        promoted: list[sqlite3.Row] = []

        def on_gate(r: sqlite3.Row, hits: list[FaceHit] | None) -> None:
            """The gate asks one thing: is there anything on this frame to crop?"""
            if hits and decoder.previewed(r["path"]):
                promoted.append(r)  # a face is there — now the original is worth it
            else:
                # Nothing found (69% of a real collection), an undecodable frame, or one
                # that fell back to a full decode — there those hits ARE the answer.
                on_result(r, hits)

        run(gated, decoder, on_gate)
        direct += promoted
    if direct:
        run(direct, _decode_for_faces, on_result)
    return stats


# --- Clustering ------------------------------------------------------------

# F84: the phases `cluster_faces` reports. Stable identifiers, not captions — the served
# UI localizes them (ui._UI_STRINGS), the CLI labels them (cli._CLUSTER_PHASE_LABELS).
# CLUSTER is the only unmeasurable one: HDBSCAN is a single blocking call, and a percent
# guessed from elapsed time would be a lie on any collection but the calibrated one.
CLUSTER_PHASE_READ = "cluster_read"
CLUSTER_PHASE_CLUSTER = "cluster_hdbscan"
CLUSTER_PHASE_INHERIT = "cluster_inherit"
CLUSTER_PHASE_WRITE = "cluster_write"

# Rows between progress ticks on the measurable phases: on the reference run (13 237
# faces) ~66 updates — enough for a moving bar, not enough to spam the callback's lock.
_PROGRESS_EVERY = 200


class _PhaseProgress:
    """Phase + `(done, total)` reporting for `cluster_faces` (F84).

    The phase channel is optional and duck-typed: a callback that can show a caption
    exposes `phase(name)`, a bare `(done, total)` function gets no phases, and without a
    callback every method is a no-op.
    """

    def __init__(self, progress: ProgressCB | None) -> None:
        self._progress = progress
        phase = getattr(progress, "phase", None)
        self._phase: PhaseCB | None = phase if callable(phase) else None
        self._total: int | None = None

    def start(self, name: str, total: int | None) -> None:
        """Enter a phase: `total=None` marks it as unmeasurable (indeterminate bar)."""
        self._total = total
        if self._phase is not None:
            self._phase(name)
        self.step(0)

    def step(self, done: int) -> None:
        if self._progress is not None:
            self._progress(done, self._total)


def _read_face_rows(conn: sqlite3.Connection, report: _PhaseProgress) -> list[sqlite3.Row]:
    """The faces to cluster (embeddings + their current cluster), as a measurable phase.

    The count is asked for separately so the bar has a total from the first tick; rows are
    then pulled off the cursor instead of `fetchall()`, which would be one silent call.
    """
    total = conn.execute(
        "SELECT COUNT(*) FROM faces WHERE bbox != ?", (_NO_FACES_BBOX,)
    ).fetchone()[0]
    report.start(CLUSTER_PHASE_READ, int(total))
    rows: list[sqlite3.Row] = []
    cursor = conn.execute(
        "SELECT id, file_id, cluster_id, embedding FROM faces WHERE bbox != ? ORDER BY id",
        (_NO_FACES_BBOX,),
    )
    for row in cursor:
        rows.append(row)
        if len(rows) % _PROGRESS_EVERY == 0:
            report.step(len(rows))
    report.step(len(rows))
    return rows


def _hdbscan_labels(x: np.ndarray, s: FacesSettings) -> np.ndarray:
    """HDBSCAN over normalized vectors, with max_distance converted to epsilon.

    Euclidean and not cosine, which hdbscan cannot do directly: on the unit sphere the two
    are monotonic (d_e = sqrt(2*d_cos)). On a SMALL collection hdbscan defaults min_samples
    to min_cluster_size and its kd-tree asks for k = min_samples + 1 neighbours — with
    fewer points than k it raises ValueError and takes the whole faces phase down, hence
    the cap at n - 1 below (inactive for n > min_cluster_size).
    """
    import hdbscan

    n = int(x.shape[0])
    if n < 2:
        return np.full(n, -1, dtype=np.intp)  # everything is noise -> cluster_id NULL
    min_samples = max(1, min(s.min_cluster_size, n - 1))

    labels = hdbscan.HDBSCAN(
        min_cluster_size=s.min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=math.sqrt(2.0 * s.max_distance),
    ).fit_predict(x)
    if not (labels >= 0).any():
        # HDBSCAN does not return a single root cluster (e.g. all faces are one person),
        # so retry with allow_single_cluster. Only as a fallback: it glues different
        # people together, and it cannot be combined with cluster_selection_epsilon
        # (that pair returns an empty result on any data).
        labels = hdbscan.HDBSCAN(
            min_cluster_size=s.min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            allow_single_cluster=True,
        ).fit_predict(x)
    return labels


def _root_of(merged_into: dict[int, int | None], cid: int) -> int:
    """The root of the merged_into chain (the effective cluster)."""
    seen = set()
    while merged_into.get(cid) is not None and cid not in seen:
        seen.add(cid)
        cid = merged_into[cid]  # type: ignore[assignment]
    return cid


@dataclass(frozen=True)
class ClusterSnapshot:
    """Which FILES each effective cluster held, and under what name (F89).

    Labels normally survive a recomputation because a new cluster and its old counterpart
    share face ids; a rescan makes every id new, so the names the user typed would
    disappear without a word. Files are the identity that survives. Chains are already
    resolved — `files` is keyed by the ROOT of the merged_into chain, so a manually merged
    pair (F3) is one entry under the name the user gave the pair.
    """
    labels: dict[int, str | None]   # cluster id -> its label (looked up by root)
    files: dict[int, set[int]]      # root cluster id -> the files its faces sat on


def snapshot_clusters(conn: sqlite3.Connection) -> ClusterSnapshot:
    """Record the current clusters in file terms — call BEFORE a rescan deletes faces."""
    old = {
        r["id"]: (r["label"], r["merged_into"])
        for r in conn.execute("SELECT id, label, merged_into FROM face_clusters")
    }
    merged_into = {cid: m for cid, (_lbl, m) in old.items()}
    files: dict[int, set[int]] = defaultdict(set)
    for r in conn.execute(
        "SELECT file_id, cluster_id FROM faces WHERE cluster_id IS NOT NULL"
    ):
        files[_root_of(merged_into, r["cluster_id"])].add(r["file_id"])
    return ClusterSnapshot(
        labels={cid: lbl for cid, (lbl, _m) in old.items()}, files=dict(files),
    )


def _inherit_labels(
    groups: dict[int, list[int]],
    keys_of_group: Callable[[list[int]], set[int]],
    old_roots_of_key: dict[int, list[int]],
    labels: dict[int, str | None],
    report: _PhaseProgress,
) -> dict[int, str | None]:
    """New cluster -> the label it inherits: biggest overlap with an old one, share > 50%.

    The overlap is counted over identities both sides can name: face ids normally, file ids
    after a rescan (F89). A key can belong to several old clusters (one photo, two people),
    so it votes for each — at most once per cluster, which keeps the share below 1.
    """
    inherited: dict[int, str | None] = {}
    for done, (lab, face_ids) in enumerate(groups.items(), 1):
        report.step(done)
        keys = keys_of_group(face_ids)
        overlap = Counter(
            root for k in keys for root in old_roots_of_key.get(k, ())
        )
        if not overlap:
            continue
        best_root, best_n = overlap.most_common(1)[0]
        if best_n * 2 > len(keys):
            inherited[lab] = labels.get(best_root)
    return inherited


# --- F212: not clustering when there is nothing to cluster -----------------
#
# `cluster_faces` used to run HDBSCAN over every embedding in the base on every call: 24 477
# faces and 171.9 s on the reference collection, 67% of a repeat run in which detection had
# four new frames to look at.
#
# The device is the project's own (`frame_quality.source`, `landmark_checks.model`,
# `group_keeper.source`): the answer is stored next to a digest of the QUESTION, so a
# changed question invalidates it by itself and there is no cache-clearing step anybody can
# forget. This does not speed up a FIRST run, where the clustering is needed.

# Raised BY HAND when the code that decides the partition changes meaning — the one part of
# the question no digest can read off the database. Bumping it re-clusters every collection
# once, which is the point.
CLUSTER_ALGO_VERSION = 1

# The prefix of the stored marker, `<algorithm>#<digest>`, spelled out so a stored row can
# be read by eye.
_CLUSTER_ALGO = "hdbscan"


def _face_set_digest(conn: sqlite3.Connection) -> str:
    """A digest of the SET OF FACES the clustering would read — a hash of their sorted ids.

    REJECTED: `COUNT(*)` alone, which deleting one face and adding another leaves where it
    was; and the (`COUNT(*)`, `MAX(id)`, `SUM(id)`) triple, which survives that swap and
    costs the same (either way SQLite scans the table) but needs an argument about which
    combinations can agree by accident.

    WHAT IT DOES NOT CATCH is the CONTENT of an embedding, and that state is reachable:
    `_write_hits` deletes a file's rows and inserts new ones, and SQLite hands a deleted
    rowid straight back when it sat at the end of the table (`tests/test_faces_rescan.py`
    documents the effect). Hence a `--rescan` FORCES the recomputation.

    The population is `_read_face_rows`': real faces, without the "no faces" markers.
    """
    digest = hashlib.sha1()
    for row in conn.execute(
        "SELECT id FROM faces WHERE bbox != ? ORDER BY id", (_NO_FACES_BBOX,)
    ):
        digest.update(f"{row['id']}\n".encode("utf-8"))
    return digest.hexdigest()


def _cluster_fingerprint(conn: sqlite3.Connection, s: FacesSettings) -> str:
    """Everything that decides the clusters, in one string compared for equality.

    Three things do: the set of faces; the thresholds the splitting reads
    (`min_cluster_size`, `max_distance` — somebody who changes one and sees the same
    clusters would conclude the setting does nothing, which is worse than a slow run); and
    `CLUSTER_ALGO_VERSION`, for the splitting code, which no digest can see.

    `det_threshold` and `min_face_px` are deliberately OUT: they decide which faces get
    WRITTEN, and reach clustering as a changed set of faces. One string and one comparison,
    by the F120 rule — a partial match is not a match, and a mismatch means RECOMPUTE.
    """
    payload = "\n".join([
        f"algo={CLUSTER_ALGO_VERSION}",
        f"min_cluster_size={s.min_cluster_size}",
        f"max_distance={s.max_distance!r}",
        f"faces={_face_set_digest(conn)}",
    ])
    return f"{_CLUSTER_ALGO}#{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _stored_fingerprint(conn: sqlite3.Connection) -> str | None:
    """What the last full clustering was an answer to, or None if there has not been one.

    None is also what a database from before this table says (the migration adds it empty),
    and it means CLUSTER — never "up to date".
    """
    row = conn.execute("SELECT fingerprint FROM cluster_state WHERE id = 1").fetchone()
    return None if row is None else str(row["fingerprint"])


def _remember_clustering(conn: sqlite3.Connection, fingerprint: str) -> None:
    """Record what the clusters just written answer. Called INSIDE the write transaction.

    Inside it so the marker and the clusters become visible together: a Ctrl+C between them
    would leave a base whose fingerprint promises clusters that were never written.
    """
    conn.execute(
        "INSERT INTO cluster_state (id, fingerprint, updated_at) VALUES (1, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint,"
        " updated_at = excluded.updated_at",
        (fingerprint, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def _stored_cluster_stats(conn: sqlite3.Connection) -> ClusterStats:
    """The numbers of a skipped run — read off the clusters that are already in the base.

    Every field is the quantity the recomputing path would have produced for this base:
    `clusters` counts the groups faces actually sit in (a manual merge sets `merged_into`
    and moves no face), `labels_kept` counts named clusters that are nobody's merge source,
    and `noise` excludes the malformed rows, which are counted separately.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS faces,"
        " SUM(CASE WHEN LENGTH(embedding) != ? THEN 1 ELSE 0 END) AS malformed,"
        " SUM(CASE WHEN cluster_id IS NULL THEN 1 ELSE 0 END) AS unclustered"
        " FROM faces WHERE bbox != ?",
        (EMBED_DIM * 4, _NO_FACES_BBOX),
    ).fetchone()
    clusters = conn.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    labels = conn.execute(
        "SELECT COUNT(*) FROM face_clusters WHERE label IS NOT NULL AND merged_into IS NULL"
    ).fetchone()[0]
    malformed = int(row["malformed"] or 0)
    return ClusterStats(
        faces=int(row["faces"] or 0),
        clusters=int(clusters or 0),
        noise=int(row["unclustered"] or 0) - malformed,
        labels_kept=int(labels or 0),
        malformed=malformed,
        skipped=True,
    )


def cluster_faces(cfg: Config, conn: sqlite3.Connection,
                  progress: ProgressCB | None = None,
                  inherit_from: ClusterSnapshot | None = None,
                  force: bool = False) -> ClusterStats:
    """Full recomputation of clusters over all embeddings, preserving labels.

    F84: the phases are read -> HDBSCAN -> inheritance -> write, HDBSCAN reporting
    `total=None` because it is one blocking call.

    F89: `inherit_from` is a `ClusterSnapshot` taken before a rescan. Without it labels are
    inherited by face id; with it by FILE id, because a rescan gave every face a new id.

    F212: nothing is recomputed when `_cluster_fingerprint` still matches the stored one,
    and `ClusterStats.skipped` says so. `force`, a mismatch and a missing fingerprint all
    recompute — and so does `inherit_from`, for a reason of its own: a rescan deletes and
    re-inserts faces rows, and SQLite can hand back the same ids over DIFFERENT vectors.
    """
    s = _settings(cfg)
    fingerprint = _cluster_fingerprint(conn, s)
    if not force and inherit_from is None and _stored_fingerprint(conn) == fingerprint:
        return _stored_cluster_stats(conn)

    report = _PhaseProgress(progress)
    rows = _read_face_rows(conn, report)
    stats = ClusterStats(faces=len(rows))
    if not rows:
        with conn:
            conn.execute("DELETE FROM face_clusters")
            _remember_clustering(conn, fingerprint)
        return stats

    expected_len = EMBED_DIM * 4
    malformed_ids = [r["id"] for r in rows if len(r["embedding"]) != expected_len]
    if malformed_ids:
        logging.warning(
            "cluster_faces: пропущено %d строк faces с эмбеддингом неверной длины "
            "(ids=%s)", len(malformed_ids), malformed_ids,
        )
        stats.malformed = len(malformed_ids)
        rows = [r for r in rows if len(r["embedding"]) == expected_len]

    if not rows:
        with conn:
            conn.execute("UPDATE faces SET cluster_id = NULL")
            conn.execute("DELETE FROM face_clusters")
            _remember_clustering(conn, fingerprint)
        return stats

    report.start(CLUSTER_PHASE_CLUSTER, None)  # unmeasurable: one blocking call
    x = np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in rows]).astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    labels = _hdbscan_labels(x / norms, s)
    report.step(len(rows))

    groups: dict[int, list[int]] = defaultdict(list)  # new label -> [face_id]
    for r, lab in zip(rows, labels):
        if lab >= 0:
            groups[int(lab)].append(r["id"])

    report.start(CLUSTER_PHASE_INHERIT, len(groups))
    if inherit_from is not None:
        # F89: the faces rows were just replaced — match by file, the stable identity
        file_of_face = {r["id"]: r["file_id"] for r in rows}
        old_roots_of_key: dict[int, list[int]] = defaultdict(list)
        for root, file_ids in inherit_from.files.items():
            for file_id in file_ids:
                old_roots_of_key[file_id].append(root)
        old_labels = inherit_from.labels

        def keys_of_group(face_ids: list[int]) -> set[int]:
            return {file_of_face[fid] for fid in face_ids}
    else:
        # old state: face.id -> root old cluster, and its label
        old_clusters = {
            r["id"]: (r["label"], r["merged_into"])
            for r in conn.execute("SELECT id, label, merged_into FROM face_clusters")
        }
        merged_into = {cid: m for cid, (_lbl, m) in old_clusters.items()}
        old_roots_of_key = defaultdict(list)
        for r in rows:
            if r["cluster_id"] is not None:
                old_roots_of_key[r["id"]].append(_root_of(merged_into, r["cluster_id"]))
        old_labels = {cid: lbl for cid, (lbl, _m) in old_clusters.items()}

        def keys_of_group(face_ids: list[int]) -> set[int]:
            return set(face_ids)

    inherited = _inherit_labels(groups, keys_of_group, old_roots_of_key, old_labels, report)

    report.start(CLUSTER_PHASE_WRITE, len(groups))
    with conn:
        conn.execute("UPDATE faces SET cluster_id = NULL")
        conn.execute("DELETE FROM face_clusters")
        for written, lab in enumerate(sorted(groups), 1):
            label = inherited.get(lab)
            cur = conn.execute(
                "INSERT INTO face_clusters (label, merged_into) VALUES (?, NULL)", (label,)
            )
            conn.executemany(
                "UPDATE faces SET cluster_id = ? WHERE id = ?",
                [(cur.lastrowid, fid) for fid in groups[lab]],
            )
            if label is not None:
                stats.labels_kept += 1
            report.step(written)
        _remember_clustering(conn, fingerprint)
    stats.clusters = len(groups)
    stats.noise = int((labels < 0).sum())
    return stats


def detect_and_cluster(
    cfg: Config, conn: sqlite3.Connection,
    progress: ProgressCB | None = None,
    analyzer: Analyzer | None = None,
    rescan: bool = False,
    limit: int | None = None,
) -> tuple[FaceStats, ClusterStats]:
    """Full phase-3 pass: detection of new files + cluster recomputation.

    The same callback drives both halves, so the bar keeps moving across the boundary
    instead of freezing on `24196/24196` for the rest of the step.

    F89: `rescan` recomputes files that already have faces (all of them, or `limit` random
    ones). This is the ONLY place that pairs a rescan with the snapshot the labels are
    carried across on — the halves called separately with rescan=True would drop every name.

    F212: the clustering half may do nothing on a repeat run. `rescan` still recomputes
    unconditionally: the vectors under the faces are new even when their ids are not.
    """
    snapshot = snapshot_clusters(conn) if rescan else None
    face_stats = detect_faces(cfg, conn, progress=progress, analyzer=analyzer,
                              rescan=rescan, limit=limit)
    return face_stats, cluster_faces(cfg, conn, progress=progress, inherit_from=snapshot,
                                     force=rescan)


# --- Manual operations on clusters -----------------------------------------

def resolve_root(conn: sqlite3.Connection, cluster_id: int) -> int:
    """The effective cluster = the root of the merged_into chain."""
    cid = cluster_id
    seen: set[int] = set()
    while True:
        row = conn.execute(
            "SELECT merged_into FROM face_clusters WHERE id = ?", (cid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"кластер {cid} не найден")
        if row["merged_into"] is None or cid in seen:
            return cid
        seen.add(cid)
        cid = row["merged_into"]


def label_cluster(conn: sqlite3.Connection, cluster_id: int, label: str) -> int:
    """Name the effective cluster (the root of the merge chain). Returns its id."""
    root = resolve_root(conn, cluster_id)
    with conn:
        conn.execute("UPDATE face_clusters SET label = ? WHERE id = ?", (label, root))
    return root


def merge(conn: sqlite3.Connection, src_id: int, dst_id: int) -> int:
    """Merge src into dst via merged_into; returns the resulting root.

    If the destination root has no label but the source does — the label is carried
    over so the person's name is not lost on merge.
    """
    src_root = resolve_root(conn, src_id)
    dst_root = resolve_root(conn, dst_id)
    if src_root == dst_root:
        return dst_root
    src_label = conn.execute(
        "SELECT label FROM face_clusters WHERE id = ?", (src_root,)
    ).fetchone()["label"]
    with conn:
        conn.execute(
            "UPDATE face_clusters SET merged_into = ? WHERE id = ?", (dst_root, src_root)
        )
        if src_label is not None:
            conn.execute(
                "UPDATE face_clusters SET label = ? WHERE id = ? AND label IS NULL",
                (src_label, dst_root),
            )
    return dst_root


# --- Contact sheet ---------------------------------------------------------

def _file_uri(path: str) -> str:
    try:
        return Path(path).as_uri()
    except ValueError:  # a POSIX path without a Windows drive
        return "file://" + quote(path)


def export_contact_sheet(conn: sqlite3.Connection, cluster_id: int, out_html: str | Path) -> int:
    """An HTML grid of a cluster's thumbnails (including those merged into it) for identification.

    Returns the number of faces in the sheet.
    """
    root = resolve_root(conn, cluster_id)
    merged_into = {
        r["id"]: r["merged_into"]
        for r in conn.execute("SELECT id, merged_into FROM face_clusters")
    }
    member_ids = [cid for cid in merged_into if _root_of(merged_into, cid) == root]
    placeholders = ",".join("?" * len(member_ids))
    rows = conn.execute(
        f"""SELECT fa.id, fa.bbox, fl.path FROM faces fa
            JOIN files fl ON fl.id = fa.file_id
            WHERE fa.cluster_id IN ({placeholders})
            ORDER BY fl.path""",
        member_ids,
    ).fetchall()
    label = conn.execute(
        "SELECT label FROM face_clusters WHERE id = ?", (root,)
    ).fetchone()["label"]

    title = escape(label or f"Кластер {root}")
    cells = "\n".join(
        f'<figure><img src="{escape(_file_uri(r["path"]))}" loading="lazy" alt="">'
        f"<figcaption>{escape(Path(r['path']).name)}</figcaption></figure>"
        for r in rows
    )
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>{title} — {len(rows)} лиц</title>
<style>
body {{ font-family: sans-serif; margin: 1rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: 160px; object-fit: cover; border-radius: 4px; }}
figcaption {{ font-size: 11px; overflow-wrap: anywhere; }}
</style></head><body>
<h1>{title} <small>({len(rows)} лиц, кластер {root})</small></h1>
<div class="grid">
{cells}
</div></body></html>
"""
    Path(out_html).write_text(html, encoding="utf-8")
    return len(rows)


# --- F47: allowed_modules smoke comparison (manual GPU run) -----------------

@dataclass
class SmokeReport:
    faces_compared: int
    mismatched_face_counts: list[str]  # paths where the face count diverged between modes
    cosines: list[tuple[str, int, float]]  # (path, face index, cosine)
    elapsed_full: float
    elapsed_limited: float


def compare_allowed_modules_embeddings(paths: list[str]) -> SmokeReport:  # pragma: no cover — manual GPU smoke
    """F47: buffalo_l embeddings (all modules) vs allowed_modules=[detection, recognition].

    Checks that disabling the unused sub-models leaves the embeddings (cosine ≈ 1.0) and
    therefore the clusters alone. Needs REAL frames with faces — mocks verify nothing here.

        uv run python -m sorta.faces <img1> <img2> ...
    """
    import time
    from insightface.app import FaceAnalysis

    _enable_cuda_dll_dirs()
    providers = accel.onnx_providers()
    s = FacesSettings()  # F88: the same pinned det_size the pipeline runs with

    def run(allowed_modules: list[str] | None) -> tuple[list[list[np.ndarray]], float]:
        app = FaceAnalysis(name="buffalo_l", allowed_modules=allowed_modules, providers=providers)
        app.prepare(ctx_id=0, det_thresh=s.det_threshold, det_size=(s.det_size, s.det_size))
        t0 = time.perf_counter()
        per_image = [
            [np.asarray(f.embedding, dtype=np.float64) for f in app.get(_decode_for_faces(p, None))]
            for p in paths
        ]
        return per_image, time.perf_counter() - t0

    full, elapsed_full = run(None)
    limited, elapsed_limited = run(_ALLOWED_MODULES)

    mismatched = [p for p, ef, el in zip(paths, full, limited) if len(ef) != len(el)]
    cosines = [
        (p, i, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        for p, ef, el in zip(paths, full, limited)
        for i, (a, b) in enumerate(zip(ef, el))
    ]
    return SmokeReport(
        faces_compared=len(cosines),
        mismatched_face_counts=mismatched,
        cosines=cosines,
        elapsed_full=elapsed_full,
        elapsed_limited=elapsed_limited,
    )


def _print_smoke_report(paths: list[str], report: SmokeReport) -> None:  # pragma: no cover — manual GPU smoke
    speedup = report.elapsed_full / report.elapsed_limited if report.elapsed_limited else float("inf")
    print(f"кадров: {len(paths)}, лиц сопоставлено: {report.faces_compared}")
    print(f"полный набор модулей:              {report.elapsed_full:.2f}s")
    print(f"allowed_modules={_ALLOWED_MODULES}: {report.elapsed_limited:.2f}s (ускорение {speedup:.2f}x)")
    if report.mismatched_face_counts:
        print(f"РАСХОЖДЕНИЕ числа лиц в кадрах: {report.mismatched_face_counts}")
    for path, idx, cos in report.cosines:
        flag = "" if cos >= 0.999 else "  <-- ПОДОЗРИТЕЛЬНО"
        print(f"[{path}] лицо {idx}: cosine={cos:.6f}{flag}")


if __name__ == "__main__":  # pragma: no cover — manual GPU smoke, see F47
    _paths = sys.argv[1:]
    if not _paths:
        print("Использование: uv run python -m sorta.faces <img1> [img2 ...]")
        raise SystemExit(1)
    _print_smoke_report(_paths, compare_allowed_modules_embeddings(_paths))
