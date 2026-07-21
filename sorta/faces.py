"""F3 (Phase 3): faces.

Contract: reads files (path, dup_of IS NULL), writes ONLY into faces and face_clusters.
- embedding: a BLOB of 512 float32 little-endian (ArcFace), see docs/ARCHITECTURE.md §3.
- A faces row with bbox='[]' and an empty embedding is the marker "file processed, no faces"
  (incrementality without a schema change).
- Re-clustering preserves labels: a new cluster inherits the label of the old
  cluster with the largest intersection by face.id, if it is > 50%.

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import numpy as np

from .config import Config

_NO_FACES_BBOX = "[]"  # the "processed, no faces" marker
EMBED_DIM = 512

# (bbox [x1,y1,x2,y2], detector confidence, embedding of length EMBED_DIM)
FaceHit = tuple[list[float], float, np.ndarray]
# analyzer(path, exif_orientation) -> found faces; replaced in tests
Analyzer = Callable[[str, int | None], list[FaceHit]]


@dataclass(frozen=True)
class FacesSettings:
    """Phase-3 thresholds; the defaults are Immich's."""
    min_face_px: int = 40        # smaller — not embedded (quality filter)
    det_threshold: float = 0.7   # detector confidence threshold
    min_cluster_size: int = 5    # HDBSCAN; smaller — noise
    max_distance: float = 0.5    # cosine face-similarity threshold


def _settings(cfg: Config) -> FacesSettings:
    f = cfg.faces
    return FacesSettings(
        min_face_px=int(f.min_face_px),
        det_threshold=float(f.det_threshold),
        min_cluster_size=int(f.min_cluster_size),
        max_distance=float(f.max_distance),
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
    n = (cfg.raw.get("faces") or {}).get("decode_workers")
    if n:
        return max(1, int(n))
    return min(8, os.cpu_count() or 4)


def _prefetch_decode(
    rows: list[sqlite3.Row],
    decode: Callable[[str, int | None], np.ndarray],
    max_workers: int,
) -> Iterator[tuple[sqlite3.Row, np.ndarray | None, Exception | None]]:
    """Decode frames in a thread pool with a bounded window (~2×max_workers in flight).

    Yields (row, image, error) as decoding completes — input order is not preserved
    (faces rows are independent, which is fine). GPU inference stays entirely on the
    caller's side (the main thread) — only decoding happens here.
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


def _insightface_infer(s: FacesSettings) -> Callable[[np.ndarray], list[FaceHit]]:  # pragma: no cover — ML, smoke test
    """insightface buffalo_l: GPU (CUDA) with a CPU fallback.

    The onnxruntime session is not thread-safe for parallel inference — `app.get`
    is called only from the main thread, frame decoding is not part of this.
    """
    from insightface.app import FaceAnalysis

    _enable_cuda_dll_dirs()

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=_ALLOWED_MODULES,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_thresh=float(s.det_threshold))

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
    r: sqlite3.Row, hits: list[FaceHit],
) -> None:
    kept = [
        (bbox, score, emb) for bbox, score, emb in hits
        if score >= s.det_threshold
        and min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= s.min_face_px
    ]
    with conn:  # one transaction per file: Ctrl+C-safe
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


def detect_faces(
    cfg: Config, conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
    analyzer: Analyzer | None = None,
) -> FaceStats:
    """Find faces in new canonical photos and write embeddings into faces.

    Incrementality: files that already have rows in faces (including the "no faces"
    marker) are skipped. Files with a row-read error do not get one and will be
    retried on the next run.

    The mock path (an `analyzer` passed, as in tests) is strictly serial, decode and
    inference in one call, behaviour unchanged. The real path (analyzer=None,
    insightface) decodes frames ahead in a thread pool (`_prefetch_decode`), while
    `app.get` on the GPU stays strictly sequential on the main thread.
    """
    s = _settings(cfg)
    rows = conn.execute(
        """SELECT id, path, orientation FROM files
           WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'
             AND id NOT IN (SELECT file_id FROM faces)
           ORDER BY id"""
    ).fetchall()
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
            _write_hits(conn, s, stats, r, hits)
            if progress:
                progress(i, len(rows))
        return stats

    infer = _insightface_infer(s)  # pragma: no cover — ML, smoke test
    decoded = _prefetch_decode(rows, _decode_for_faces, _decode_workers(cfg))
    for i, (r, img, err) in enumerate(decoded, 1):  # pragma: no cover — ML, smoke test
        decoded_hits: list[FaceHit] | None = None
        if err is not None:
            stats.errors += 1
        else:
            assert img is not None
            try:
                decoded_hits = infer(img)
            except Exception:
                stats.errors += 1
        if decoded_hits is not None:
            _write_hits(conn, s, stats, r, decoded_hits)
        if progress:
            progress(i, len(rows))
    return stats


# --- Clustering ------------------------------------------------------------

def _hdbscan_labels(x: np.ndarray, s: FacesSettings) -> np.ndarray:
    """HDBSCAN over normalized vectors: euclidean on the unit sphere is monotonic
    with cosine distance (d_e = sqrt(2*d_cos)) — hdbscan cannot do cosine directly.
    The max_distance threshold is converted to epsilon on the same scale.
    """
    import hdbscan

    labels = hdbscan.HDBSCAN(
        min_cluster_size=s.min_cluster_size,
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


def cluster_faces(cfg: Config, conn: sqlite3.Connection) -> ClusterStats:
    """Full recomputation of clusters over all embeddings, preserving labels."""
    s = _settings(cfg)
    rows = conn.execute(
        "SELECT id, cluster_id, embedding FROM faces WHERE bbox != ? ORDER BY id",
        (_NO_FACES_BBOX,),
    ).fetchall()
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

    x = np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in rows]).astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    labels = _hdbscan_labels(x / norms, s)

    # old state: face.id -> root old cluster, and its label
    old_clusters = {
        r["id"]: (r["label"], r["merged_into"])
        for r in conn.execute("SELECT id, label, merged_into FROM face_clusters")
    }
    merged_into = {cid: m for cid, (_lbl, m) in old_clusters.items()}
    old_root_of_face = {
        r["id"]: _root_of(merged_into, r["cluster_id"])
        for r in rows if r["cluster_id"] is not None
    }

    groups: dict[int, list[int]] = defaultdict(list)  # new label -> [face_id]
    for r, lab in zip(rows, labels):
        if lab >= 0:
            groups[int(lab)].append(r["id"])

    # label inheritance: the old cluster with the largest intersection and share > 50%
    inherited: dict[int, str | None] = {}
    for lab, face_ids in groups.items():
        overlap = Counter(
            old_root_of_face[fid] for fid in face_ids if fid in old_root_of_face
        )
        if not overlap:
            continue
        best_root, best_n = overlap.most_common(1)[0]
        if best_n * 2 > len(face_ids):
            inherited[lab] = old_clusters[best_root][0] if best_root in old_clusters else None

    with conn:
        conn.execute("UPDATE faces SET cluster_id = NULL")
        conn.execute("DELETE FROM face_clusters")
        for lab in sorted(groups):
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
    stats.clusters = len(groups)
    stats.noise = int((labels < 0).sum())
    return stats


def detect_and_cluster(
    cfg: Config, conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
    analyzer: Analyzer | None = None,
) -> tuple[FaceStats, ClusterStats]:
    """Full phase-3 pass: detection of new files + cluster recomputation."""
    face_stats = detect_faces(cfg, conn, progress=progress, analyzer=analyzer)
    return face_stats, cluster_faces(cfg, conn)


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
    det_thresh = FacesSettings().det_threshold

    def run(allowed_modules: list[str] | None) -> tuple[list[list[np.ndarray]], float]:
        app = FaceAnalysis(name="buffalo_l", allowed_modules=allowed_modules, providers=providers)
        app.prepare(ctx_id=0, det_thresh=det_thresh)
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
