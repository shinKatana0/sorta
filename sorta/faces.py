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

F165: and it does not look for faces where the classifier has already said there is
nothing to look at. The `classify` stage runs before this one now (see junk.classify's
`verdicts_only`), so a screenshot, a document, a meme or a product carries its verdict
before the detector is asked about it — 4 300 frames of 24 195 on the reference
collection, at 77 ms each. The rule is `media_class.verdict = 'photo'` OR no row at all:
NULL means nobody has classified this frame, not "not a photograph", so a collection whose
owner runs `sorta faces` alone is detected in full exactly as before. A frame that becomes
a photograph later (the deep tier reclassifies 2 592 of 24 196 on the reference run) has no
`faces` row, so the ordinary incrementality of this stage picks it up on the next run — the
economy must not be able to turn into lost faces.
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


# F88: the detector's input side, in px. buffalo_l's det_10g is trained at 640, and
# that is the only value worth running: it is the native size and it costs 16.5 ms/frame
# against 13.4 at 512 — 3 ms that buy back the small faces 512 loses (−9% on the
# measurement). Lowering it is a "trade recall for weak hardware" knob, not a speed knob.
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

    A garbage value must not take down a run that has already spent an hour on the
    collection, so anything that is not a positive number falls back to the default
    with a warning instead of raising inside a worker thread.
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
    # F212: the clusters were left as they were, because nothing that decides them had
    # moved. The numbers above then describe the clusters ALREADY in the base rather than
    # ones this run produced — see _stored_cluster_stats.
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
    """CUDA/cuDNN are installed as pip wheels (the nvidia-* packages), not a system Toolkit.

    onnxruntime resolves provider-DLL dependencies via the classic PATH search, and
    its preload_dlls() (1.27) does not know the new nvidia/cu13 layout — so we add
    the DLL directories to the process PATH ourselves. Without them ORT silently
    falls back to CPUExecutionProvider.
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
    """Decode an image into a BGR array for insightface.

    cv2.imdecode cannot handle HEIC/HEIF (the typical iPhone format) — on such files
    it returns None; then a fallback to Pillow + pillow-heif (the plugin is
    registered globally). cv2.imread does not take non-ASCII paths on Windows, so we
    read the bytes ourselves. ValueError if nothing could decode it.
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


# F91: the two passes, and why the second one detects AGAIN instead of reusing the
# boxes of the first.
#
# The step was decode-bound (16.6 frames/s with the GPU at 3-10%): every frame was
# decompressed at full resolution, and 69% of them had no face at all. Detection does
# not need those pixels — insightface downscales its input to det_size=640 whatever it
# is given, so a 4000 px original and a 1536 px preview reach the network as the very
# same 640 px frame. The crop DOES need them: ArcFace embeds the face out of the
# original, and that is what must not change.
#
# The brief proposed scaling the preview's boxes into original coordinates and feeding
# them to recognition directly. That saves one detection pass (16.5 ms on the 31% of
# frames that have a face, on a card sitting at 3-10%) at the price of splitting
# `app.get` into `det_model.detect` + a hand-built alignment — and if that alignment
# ever takes coordinates from the wrong space, the embeddings drift silently and the
# clusters rot weeks later. The brief's own acceptance criterion is equivalence, not
# speed, and there is no insightface/GPU in this environment to prove it on. So the
# preview is used strictly as a GATE: "is there anything here to crop?". A frame that
# passes it goes through the unchanged `app.get(original)`, which makes the written
# embeddings identical to the previous behaviour by construction rather than by
# measurement. The saved decode — the whole point — is untouched: the 69% still never
# reach a full decode.
#
# What is left to verify on real data is the gate's recall (the brief's "no more than
# 2% difference in the number of faces found"): only frames where the preview sees
# nothing and the original would have seen something are lost. `sorta faces --rescan`
# before and after answers it.

def _decode_preview_for_faces(path: str, orientation: int | None) -> np.ndarray | None:
    """F91: a ~1536 px BGR frame for the detection GATE, or None if there is no cheap one.

    The frame comes from the shared preview cache (F67): warm it is a read of a small
    JPEG, cold it is a draft decode of the original (a DCT downscale, ~46 ms against
    ~1000 ms for the full frame on a 13 MP camera JPEG) that also fills the cache for
    the other stages. With the cache switched off `decode_rgb_preview` still decodes
    to the requested size and merely writes nothing — the win is in decoding SMALL,
    the cache only saves repeated touches.

    None means "no cheap frame here" and the caller must go the old way, silently:
    an unreadable/undecodable source, or a frame that came back no smaller than the
    preview size (a picture below 1536 px — a downscale that saves nothing). mtime and
    size for the cache key come from a local stat, microseconds against the decode, so
    that this keeps the (path, orientation) signature the decode pool works with.
    """
    edge = imaging.preview_max_edge()
    try:
        st = os.stat(path)
    except OSError:
        return None
    img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=edge)
    if img is None or max(img.size) < edge:
        return None
    # PIL gives RGB, insightface wants BGR and a contiguous buffer (the reversed
    # view is neither). Orientation is applied from the INDEX, exactly as on the full
    # path — the preview is stored unrotated, as the source is.
    return _apply_orientation(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), orientation)


class _GateDecoder:
    """The decode of the gate pass: a preview when there is one, the original otherwise.

    Which of the two a frame got decides how its hits are read — a preview only
    answers "is there anything to crop here", an original answers with the faces
    themselves — and that answer is needed on the main thread while the decode runs in
    the pool. Hence the set of paths rather than a second return value: the decode
    callable of `_prefetch_decode` is (path, orientation) -> frame, one signature
    shared with the plain full-resolution path, and files.path is UNIQUE, so it
    identifies the row.
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

    Until F87 the parallel path decoded inside its inference workers and never read
    this setting, so tuning it did nothing on a GPU machine; now it sizes the decode
    pool that feeds the sessions as well.
    """
    n = (cfg.raw.get("faces") or {}).get("decode_workers")
    if n:
        return max(1, int(n))
    return min(8, os.cpu_count() or 4)


def _infer_workers(cfg: Config) -> int:
    """How many independent inference sessions run in parallel (F12.1).

    faces is inference-bound, not decode-bound: `app.get` takes ~256 ms/frame at ~6%
    GPU load, while the decode pool feeds it ~9× faster. The lever is several
    independent FaceAnalysis sessions — measured ×3.17 on 4 sessions, ~0.6 GB VRAM
    each. On a CPU-only profile parallel sessions merely oversubscribe the cores,
    so the auto default there is 1 (and the pipeline keeps the prefetch-decode pool).

    F214 moved the "is CUDA there" question into `accel` with the rest of the device
    choice. A CoreML machine keeps the single-session default: how many Metal sessions
    a Mac wants is a measurement nobody has made, and this feature does not guess.
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

    F87: decode and inference used to be one unit of work per thread — while a worker
    read a 40 MB RAW its own GPU session sat idle, and on a real run the card stayed
    at 2-5% load. They are decoupled here into the shape F64 gave CLIP:
    `decode_workers` threads decode (the ready `_prefetch_decode` pool, so there is
    one such pool in the file, not two) and `workers` sessions do nothing but infer.
    Measured ×1.57 on 4 sessions (8.14 → 12.78 img/s, 500 real frames, RTX 5090
    Laptop) with the same 300 faces found and no extra VRAM.

    Every inference worker builds its OWN session on first use (thread-local): the
    onnxruntime session is not thread-safe, independent sessions share no state and
    run in parallel safely.

    The in-flight window is bounded on BOTH sides — full-res frames are heavy, and an
    unbounded decode pool would simply read the whole collection into memory:
    `_prefetch_decode` keeps at most ~2×decode_workers frames decoded, and at most
    ~2×workers of them wait for a session here.

    `on_result(row, hits)` is called strictly from this (the main) thread as frames
    complete — hence writes to SQLite stay single-writer. hits=None means the frame
    failed to decode or infer; it does not stop the rest. Input order is not
    preserved (faces rows are independent).
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

    Yields (row, image, error) as decoding completes — input order is not preserved
    (faces rows are independent, which is fine). No inference happens here: the
    caller decides where the frame is inferred — on its own thread (the 1-session
    path) or in the session pool of `_detect_parallel` (F87).
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

    `workers > 1` — the F87/F12.1 scheme (a decode pool feeding N independent
    sessions); `workers == 1` (the CPU profile) — the same decode pool feeding a
    single session on the calling thread, built here once so that the second pass of
    the F91 gate does not load the model again.

    On the parallel path every pass builds its own sessions: they are thread-local and
    the pool's threads end with the pass. That is a few seconds of model load per run,
    against the minutes the gate saves — sharing them would mean keeping one executor
    alive across the passes, i.e. more machinery in `_detect_parallel` than the win
    justifies.
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

    The gate only pays off when the original is bigger than the preview: for a picture
    that is already smaller the "preview" IS the frame, and the pass would be pure
    overhead. The indexer's width/height answer that without touching the file; when
    they are missing (an exotic format, a file indexed before those columns were
    filled) the answer is unknown and the frame keeps the old path — the brief's third
    fallback. A frame that slips through anyway (dimensions in the index disagreeing
    with the file) is caught by _decode_preview_for_faces, which then returns None.
    """
    edge = imaging.preview_max_edge()
    gated: list[sqlite3.Row] = []
    direct: list[sqlite3.Row] = []
    for r in rows:
        width, height = r["width"], r["height"]
        big = bool(width) and bool(height) and max(int(width), int(height)) > edge
        (gated if big else direct).append(r)
    return gated, direct


# The pipeline uses only (bbox, det_score, embedding) — FaceHit — from a face.
# buffalo_l loads 5 sub-models by default; landmark_2d_106/landmark_3d_68/
# genderage would be computed on every face and immediately discarded. Recognition
# aligns the input by the 5 kps from detection (det_10g), not by these models —
# disabling them does not change the embeddings (see the smoke comparison, F47).
_ALLOWED_MODULES = ["detection", "recognition"]


def _insightface_infer(s: FacesSettings) -> Infer:  # pragma: no cover — ML, smoke test
    """insightface buffalo_l: GPU (CUDA) with a CPU fallback.

    An onnxruntime session is not thread-safe, so this is called once per inference
    worker (F12.1): every thread gets its own FaceAnalysis, and nothing is shared
    between them. The contract of the returned infer (bbox/score/embedding) is fixed.

    F88: `det_size` is passed explicitly. Without it insightface 1.0.1 leaves the
    detector in its two-pass mode (`set det-size: [(128, 128), (640, 640)]`) and runs
    the network TWICE per frame; both passes cost the same ~78 ms even though the first
    has 25× less to compute, because the price is in switching the input shape, not in
    the arithmetic. Pinning one shape: 165.1 -> 16.5 ms/frame on 100 real frames, 57
    faces against 56. The first call stays expensive (plans/kernels warm up) — expected.
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

    The delete lives INSIDE the per-file transaction on purpose: at every moment a
    file either has its old faces or its new ones, never neither. Wiping the whole
    table up front would be simpler, but a Ctrl+C halfway through a rescan of 24k
    frames would then leave a collection with no faces at all.
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

# F165: the verdict a personal photograph carries in `media_class` — the same string
# `junk.QUALITY_VERDICT` holds, spelled here so that this module keeps its three imports
# and does not pull the whole classification stage in for one stored value. The two are
# pinned to each other by a test rather than by an import.
_PHOTO_VERDICT = "photo"

# F165: ...and the frames it lets through. NOT EXISTS rather than a join or `verdict = ?`,
# because the frames with NO row have to pass: NULL means "nobody has classified this",
# which is the state of every collection whose owner runs `sorta faces` on its own.
_CLASSIFIED_AS_PHOTO = (
    "NOT EXISTS (SELECT 1 FROM media_class mc"
    f" WHERE mc.file_id = files.id AND mc.verdict != '{_PHOTO_VERDICT}')"
)

# width/height (F91) decide whether a frame is worth gating — see _split_for_gate.
_DETECT_COLUMNS = "id, path, orientation, width, height"


def _files_to_detect(
    conn: sqlite3.Connection, rescan: bool, limit: int | None,
) -> list[sqlite3.Row]:
    """The files this run detects on.

    Default (F68): only files with no faces row at all — the "no faces" marker counts
    as processed. `rescan` (F89) takes every canonical photo again, and with `limit`
    only N of them, picked at random: a measurement run recomputes 500 frames and
    leaves the other 24 thousand alone. Random rather than the first N by id, because
    the head of the collection is one folder from one camera and would answer a
    different question than "what does this step cost on my photos".

    F165: every one of the three is narrowed by the classification (see the module
    docstring), the rescan included — "re-detect everything" means the population of this
    stage, and a screenshot has not been part of it since the `classify` stage started
    running first.
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
    marker) are skipped. Files with a row-read error do not get one and will be
    retried on the next run.

    `rescan` (F89) turns that off: the selected files are detected again and their
    old faces rows are replaced, one file per transaction. `limit` narrows a rescan
    to N random files. A rescan gives every face a new id, so the cluster labels can
    no longer be matched back by face — take `snapshot_clusters(conn)` BEFORE calling
    this and hand it to `cluster_faces(inherit_from=...)`, which is exactly what
    `detect_and_cluster(rescan=True)` does. Otherwise every name the user typed is
    lost.

    The mock path (an `analyzer` passed, as in tests) is strictly serial, decode and
    inference in one call, behaviour unchanged. The real path (analyzer=None) runs
    `_infer_workers(cfg)` inference sessions in parallel, one per thread (F12.1), fed
    by a pool of `_decode_workers(cfg)` decoding threads (F87); with a single worker
    (the CPU profile) it keeps the previous pipeline — the same decode pool feeding
    one session on this thread. `infer_factory` builds a session; in production it is
    `_insightface_infer`, tests inject a fake one.

    F91: that real path runs in TWO passes. The first one looks for faces on a ~1536 px
    preview (`_decode_preview_for_faces`) — the detector downscales its input to
    det_size=640 anyway, so a full decode buys detection nothing — and 69% of a real
    collection ends there, with the "no faces" marker written and the original never
    read. Only the frames a face was found on are decoded in full, in the second pass,
    and it is that ORIGINAL the faces written come from. Frames not worth gating
    (`_split_for_gate`) and frames with no cheap preview join the second pass directly,
    on exactly the old code path.

    SQLite is written only from this thread in every case (single-writer), one
    transaction per file; the order of faces rows does not matter.
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
                # nothing found (69% of a real collection), an undecodable frame, or
                # a frame that fell back to a full decode — those hits ARE the answer
                on_result(r, hits)

        run(gated, decoder, on_gate)
        direct += promoted
    if direct:
        run(direct, _decode_for_faces, on_result)
    return stats


# --- Clustering ------------------------------------------------------------

# F84: the phases `cluster_faces` reports. Stable identifiers, not captions — the
# served UI localizes them (ui._UI_STRINGS), the CLI labels them for the rich bar
# (cli._CLUSTER_PHASE_LABELS). CLUSTER is the only unmeasurable one: HDBSCAN is a
# single blocking call, and a percent guessed from elapsed time would be a lie on any
# collection that is not the one it was calibrated on.
CLUSTER_PHASE_READ = "cluster_read"
CLUSTER_PHASE_CLUSTER = "cluster_hdbscan"
CLUSTER_PHASE_INHERIT = "cluster_inherit"
CLUSTER_PHASE_WRITE = "cluster_write"

# Rows between progress ticks on the measurable phases: on the reference run
# (13 237 faces) that is ~66 updates — enough for a moving bar, not enough to spam
# the lock behind the callback.
_PROGRESS_EVERY = 200


class _PhaseProgress:
    """Phase + `(done, total)` reporting for `cluster_faces` (F84).

    `progress` is the ordinary stage callback; the phase channel is optional and
    duck-typed — a callback that can show a caption exposes `phase(name)`
    (progress.TaskProgress, ui._StageProgress), a bare `(done, total)` function simply
    gets no phases. Without a callback at all every method is a no-op, so
    `cluster_faces(cfg, conn)` behaves exactly as it did before.
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

    The count is asked for separately so the bar has a total from the first tick;
    rows are then pulled off the cursor instead of `fetchall()` — otherwise the whole
    read is one silent call again.
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
    """HDBSCAN over normalized vectors: euclidean on the unit sphere is monotonic
    with cosine distance (d_e = sqrt(2*d_cos)) — hdbscan cannot do cosine directly.
    The max_distance threshold is converted to epsilon on the same scale.

    Small collections: hdbscan defaults min_samples to min_cluster_size and its
    kd-tree then asks for k = min_samples + 1 neighbours; with fewer points than k
    it raises ValueError and takes the whole faces phase down. Hence the guard
    below — a single face cannot form a cluster anyway (min_cluster_size >= 2),
    and for n >= 2 min_samples is capped at n - 1 so that k <= n. For a normal
    collection (n > min_cluster_size) the cap is inactive and min_samples stays
    equal to min_cluster_size, i.e. exactly the previous implicit default.
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
        # Degenerate case: HDBSCAN does not return a single root cluster
        # (e.g. all faces are one person), so we try again with
        # allow_single_cluster. It cannot be combined with cluster_selection_epsilon
        # (gives an empty result on any data), and in the general case it is
        # dangerous — it glues different people together, so only as a fallback.
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

    Labels normally survive a recomputation because a new cluster and its old
    counterpart share face ids. A rescan deletes those rows and detects again, so
    every id is new, every intersection is empty, and the names the user typed by
    hand would disappear without a word. Files are the one identity that survives:
    the snapshot is taken BEFORE the delete and inheritance is then computed in file
    terms (`cluster_faces(inherit_from=...)`).

    Chains are already resolved: `files` is keyed by the ROOT of the merged_into
    chain, so a manually merged pair (F3) is one entry and inherits under the name
    the user gave the pair.
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

    The overlap is counted over identities that both sides can name: face ids on an
    ordinary recomputation, file ids after a rescan (F89), where the face ids are all
    new. A key can belong to several old clusters (one photo, two people), so a key
    votes for each of them — at most once per cluster, which keeps the share below 1.
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
# `detect_and_cluster` always called `cluster_faces`, and `cluster_faces` always ran
# HDBSCAN over every embedding in the base. On the reference collection that is 24 477
# faces and 171.9 s — 67% of a repeat run in which detection had exactly four new frames to
# look at. Two thirds of the run went on recomputing an answer that could not have changed.
#
# The device is the project's own and already carries three features: an answer stored next
# to a digest of the QUESTION (`frame_quality.source`, `landmark_checks.model`,
# `group_keeper.source`). A changed question no longer matches the stored digest, so it
# invalidates the answer by itself — there is no "clear the cache" step anybody can forget,
# and no stored state that can look fresh while what produced it has moved.
#
# This is not a speed-up of the first run: there the clustering is needed and honestly
# costs its seconds. It is the second run that stops paying for the first one's work.

# Raised BY HAND when the code that decides the partition changes meaning — the same rule
# the prompt fingerprints follow, and the one part of the question no digest can read off
# the database. Bumping it re-clusters every collection once, which is the point: whoever
# changes how faces are split must not have to explain why nobody's clusters moved.
CLUSTER_ALGO_VERSION = 1

# The prefix of the stored marker, `<algorithm>#<digest>` — which splitting the clusters
# came out of, kept spelled out so a stored row can be read by eye.
_CLUSTER_ALGO = "hdbscan"


def _face_set_digest(conn: sqlite3.Connection) -> str:
    """A digest of the SET OF FACES the clustering would read — a hash of their sorted ids.

    `COUNT(*)` alone would not do: deleting one face and adding another leaves the counter
    where it was, and the clusters would silently keep describing a collection that no
    longer exists. The brief's other candidate — `COUNT(*)`, `MAX(id)` and `SUM(id)`
    together — survives that swap as well and costs the same, because either way SQLite
    scans the table (`bbox` is not indexed and `id` is the rowid). The hash is taken
    because it needs no argument about which combinations of three aggregates can be made
    to agree by accident: it changes if and only if the set of ids changes.

    WHAT IT DOES NOT CATCH is the CONTENT of an embedding. A row that keeps its id and gets
    a different vector is invisible here, and that is a reachable state rather than a
    theoretical one: `_write_hits` deletes a file's rows and inserts new ones, and SQLite
    hands a deleted rowid straight back when it sat at the end of the table (the effect
    `tests/test_faces_rescan.py` already documents). A `--rescan` can therefore rebuild the
    very same id set over different vectors — which is why a rescan FORCES the
    recomputation instead of asking this question at all.

    The population is the one `_read_face_rows` reads: real faces, without the
    "processed, no faces" markers, which take no part in clustering.
    """
    digest = hashlib.sha1()
    for row in conn.execute(
        "SELECT id FROM faces WHERE bbox != ? ORDER BY id", (_NO_FACES_BBOX,)
    ):
        digest.update(f"{row['id']}\n".encode("utf-8"))
    return digest.hexdigest()


def _cluster_fingerprint(conn: sqlite3.Connection, s: FacesSettings) -> str:
    """Everything that decides the clusters, in one string compared for equality.

    Three things decide them and all three are in here:

    * the set of faces (`_face_set_digest`);
    * the thresholds the splitting reads — `min_cluster_size` and `max_distance`, which
      `_hdbscan_labels` turns into min_samples and the selection epsilon. Someone who
      changes a threshold in the config and sees the same clusters would conclude the
      setting does nothing, and that is a worse outcome than a slow run;
    * `CLUSTER_ALGO_VERSION`, for what no digest can see — the splitting code itself.

    `det_threshold` and `min_face_px` are deliberately OUT. They decide which faces get
    WRITTEN, not how the written ones are split, and their effect reaches clustering as a
    changed set of faces — after the rescan that is the only thing which applies them.

    One string and one comparison, because the question is "is everything that decides
    these clusters still what it was": a marker that matches in part is not a match at all
    (the F120 rule — a mismatch means RECOMPUTE, never use).
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

    Inside it on purpose: the marker and the clusters it describes must become visible
    together, or a Ctrl+C between them would leave a base whose fingerprint promises
    clusters that were never written.
    """
    conn.execute(
        "INSERT INTO cluster_state (id, fingerprint, updated_at) VALUES (1, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint,"
        " updated_at = excluded.updated_at",
        (fingerprint, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def _stored_cluster_stats(conn: sqlite3.Connection) -> ClusterStats:
    """The numbers of a skipped run — read off the clusters that are already in the base.

    A skipped run still has to report something, and the honest report is the state of the
    clusters it left alone. Every field is the same quantity the recomputing path would
    have produced for this base: `clusters` counts the groups faces actually sit in (a
    manual merge sets `merged_into` and moves no face, so it does not change the count a
    recomputation would report), `labels_kept` counts named clusters that are nobody's
    merge source, and `noise` excludes the malformed rows, which the recomputing path
    counts separately rather than as noise.
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

    F84: `progress` is the same `(done, total|None)` callback the other stages take —
    the step used to go silent here for as long as clustering ran, which from the
    outside is indistinguishable from a hang. The phases are read → HDBSCAN →
    inheritance of labels → write; HDBSCAN reports `total=None` (indeterminate), the
    rest are measurable. Called without a callback the function works exactly as
    before.

    F89: `inherit_from` is a `ClusterSnapshot` taken before a rescan. Without it
    labels are inherited by face id, as always; with it — by file id, because a
    rescan gave every face a new id and the face-wise intersection would be empty.

    F212: and it does not recompute at all when nothing that decides the clusters has
    moved. `_cluster_fingerprint` is the question the clusters in the base are an answer
    to — the set of faces, the thresholds, the version of the splitting code — and when it
    still matches the stored one the clusters are left exactly where they are and
    `ClusterStats.skipped` says so. Recomputation is unconditional in all three cases where
    it has to be: `force` (`sorta faces --rescan`), a fingerprint that does not match, and
    no fingerprint at all (a first clustering, or a database from before this existed).

    `inherit_from` forces it too, and for a reason of its own: a snapshot is taken only
    before a rescan, and a rescan deletes and re-inserts the faces rows — SQLite can hand
    back the same ids over DIFFERENT vectors, which the fingerprint cannot see. The one
    thing a caller must not do is take a snapshot and then expect this to skip.
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

    The same callback drives both halves — detection counts frames, clustering
    reports its phases (F84), so the bar keeps moving across the boundary instead of
    freezing on `24196/24196` for the rest of the step.

    F89: `rescan` recomputes files that already have faces (all of them, or `limit`
    random ones). This is the only place that pairs the rescan with the snapshot the
    labels are carried across on, so it is the entry point to use — the halves called
    separately with rescan=True would drop every name.

    F212: the clustering half is now allowed to do nothing. It recomputes when the set of
    faces or the clustering settings have moved, and on a repeat run over a collection with
    no new frames it leaves the clusters — and the names on them — exactly where they are.
    `rescan` still recomputes unconditionally: the vectors under the faces are new even
    when their ids are not.
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

    Confirms the brief's requirement: recognition aligns the input by the 5 kps from
    detection, not by landmark_2d_106/landmark_3d_68/genderage — so disabling them
    should not change the embeddings (cosine ≈ 1.0) and therefore the clusters.
    Real frames with faces are needed — synthetic/mocks do not verify this.

    Manual run (a smoke over a sample of the real collection):
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
