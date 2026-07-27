"""F3 (Phase 3): faces.

Contract: reads files (path, dup_of IS NULL), writes ONLY into faces and face_clusters.
- embedding: a BLOB of 512 float32 little-endian (ArcFace), see docs/ARCHITECTURE.md §3.
- A faces row with bbox='[]' and an empty embedding is the marker "file processed, no faces"
  (incrementality without a schema change).
- Re-clustering preserves labels: a new cluster inherits the label of the old
  cluster with the largest intersection by face.id, if it is > 50%. After a rescan
  (F89) the face ids are all new, so the intersection is taken over file ids
  instead — see ClusterSnapshot.

Thresholds come from the config.yaml `faces:` section (typed, cfg.faces);
the defaults are the tuned Immich values.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import numpy as np

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
    shrinking the input would hurt clustering accuracy.
    """
    return _apply_orientation(_read_image_bgr(path), orientation)


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


def _cuda_provider_available() -> bool:
    """Is onnxruntime built with CUDA available in this environment?

    The GPU profile installs onnxruntime-gpu, the CPU one plain onnxruntime; an
    absent/broken onnxruntime means CPU semantics.
    """
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def _infer_workers(cfg: Config) -> int:
    """How many independent inference sessions run in parallel (F12.1).

    faces is inference-bound, not decode-bound: `app.get` takes ~256 ms/frame at ~6%
    GPU load, while the decode pool feeds it ~9× faster. The lever is several
    independent FaceAnalysis sessions — measured ×3.17 on 4 sessions, ~0.6 GB VRAM
    each. On a CPU-only profile parallel sessions merely oversubscribe the cores,
    so the auto default there is 1 (and the pipeline keeps the prefetch-decode pool).
    """
    n = (cfg.raw.get("faces") or {}).get("infer_workers")
    if n:
        return max(1, int(n))
    return 4 if _cuda_provider_available() else 1


def _detect_parallel(
    rows: list[sqlite3.Row],
    decode: Callable[[str, int | None], np.ndarray],
    infer_factory: InferFactory,
    workers: int,
    decode_workers: int,
    on_result: Callable[[sqlite3.Row, list[FaceHit] | None], None],
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
    decode: Callable[[str, int | None], np.ndarray],
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
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
    """
    if not rescan:
        return conn.execute(
            f"""SELECT id, path, orientation FROM files
                WHERE {_CANONICAL} AND id NOT IN (SELECT file_id FROM faces)
                ORDER BY id"""
        ).fetchall()
    if limit is None:
        return conn.execute(
            f"SELECT id, path, orientation FROM files WHERE {_CANONICAL} ORDER BY id"
        ).fetchall()
    return conn.execute(
        f"SELECT id, path, orientation FROM files WHERE {_CANONICAL} "
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

    SQLite is written only from this thread in both cases (single-writer), one
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

    workers = _infer_workers(cfg)
    decode_workers = _decode_workers(cfg)
    if workers > 1:
        _detect_parallel(
            rows, _decode_for_faces, factory, workers, decode_workers, on_result
        )
        return stats

    infer = factory()
    for r, img, err in _prefetch_decode(rows, _decode_for_faces, decode_workers):
        frame_hits: list[FaceHit] | None = None
        if err is None:
            assert img is not None
            try:
                frame_hits = infer(img)
            except Exception:
                frame_hits = None
        on_result(r, frame_hits)
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


def cluster_faces(cfg: Config, conn: sqlite3.Connection,
                  progress: ProgressCB | None = None,
                  inherit_from: ClusterSnapshot | None = None) -> ClusterStats:
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
    """
    s = _settings(cfg)
    report = _PhaseProgress(progress)
    rows = _read_face_rows(conn, report)
    stats = ClusterStats(faces=len(rows))
    if not rows:
        with conn:
            conn.execute("DELETE FROM face_clusters")
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
    """
    snapshot = snapshot_clusters(conn) if rescan else None
    face_stats = detect_faces(cfg, conn, progress=progress, analyzer=analyzer,
                              rescan=rescan, limit=limit)
    return face_stats, cluster_faces(cfg, conn, progress=progress, inherit_from=snapshot)


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
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
