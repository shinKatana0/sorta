"""F6 (Phase 5): places without GPS — CLIP zero-shot over a curated landmark list.

Contract: reads files and places, writes ONLY into places and STRICTLY into rows
with confidence='unknown' (exact_gps / session_inferred / visual are not overwritten;
run order: geo always before landmarks).

The CLIP model (open_clip, the same as in junk.py) is mocked in tests via the
classifier parameter; the real load happens only in clip_classifier().
GPU: torch is installed as a CPU wheel (the project's CUDA wheels are only for
onnxruntime) — we run on the CPU, correctness over speed; the GPU variant will be
finished in Phase 6.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence, TypeVar

import numpy as np
import yaml

from .config import Config
from .naming import NamingSettings, naming_settings, utcnow_iso

# (image paths, text prompts) -> softmax probabilities (n_img, n_prompt);
# an unreadable image — a row of zeros. Replaced in tests.
Classifier = Callable[[list[str], list[str]], np.ndarray]

# path -> image features (normalized encoder vector) per path, the same order as
# in the input list; None at a position — could not decode/encode.
FeatureEncoder = Callable[[list[str]], list[np.ndarray | None]]
# stacked image features (of valid paths only) + prompts -> softmax probabilities.
FeatureScorer = Callable[[np.ndarray, list[str]], np.ndarray]

# Negative classes: they take probability mass away from ordinary photos so that a
# softmax over landmark prompts alone does not produce false positives.
_NEGATIVE_PROMPTS = (
    "a photo",
    "an indoor photo of people",
    "a snapshot of everyday life",
)

_T = TypeVar("_T")


def batched(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


@dataclass(frozen=True)
class Landmark:
    prompt: str    # English description for CLIP
    name: str      # place name for reports
    country: str   # ISO code (reverse_geocoder format, the cc field)
    city: str


def load_landmarks(path: str | Path) -> list[Landmark]:
    """Read data/landmarks.yaml (format: prompt/name/country/city)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("landmarks") or []
    result: list[Landmark] = []
    for i, e in enumerate(entries):
        missing = [k for k in ("prompt", "name", "country", "city") if not e.get(k)]
        if missing:
            raise ValueError(f"{path}: запись #{i + 1} без полей {missing}")
        result.append(Landmark(prompt=str(e["prompt"]), name=str(e["name"]),
                               country=str(e["country"]), city=str(e["city"])))
    return result


@dataclass
class CachingFeatureClassifier:
    """A caching wrapper over `Classifier`: CLIP image features do not depend on the
    text prompts, so each path is encoded (decode+encode_image) at most ONCE over
    the object's lifetime; a repeated call with the same path but a different set of
    prompts — only the cheap `score` (matmul + softmax), without re-decoding
    (previously one photo could be decoded up to three times per `sorta run` —
    landmarks, junk classes, the document pass).

    From the outside the object is a plain `Classifier` (`__call__` with the same
    signature `(paths, prompts) -> probs`), so the landmarks/junk test mocking
    infrastructure does not change.

    encode(paths) -> a list of features in the same order as paths; None at a
    position — the file did not decode/encode. Such paths are NOT cached (no
    "forever zero"): a repeated call with the same path will try to encode again —
    as before in `clip_classifier`, a decode error is cheaper than the risk of a
    stuck None on a file that is actually readable.

    score(features, prompts) -> softmax probabilities (n, len(prompts));
    receives the already-stacked features of ONLY the valid (successfully encoded)
    paths of the current call.

    Cache bounds: a plain dict without eviction — the object lives within a single
    CLI command (`sorta run`), not a long-lived process; features are small
    (~768 floats ≈ 3 KB per photo) — not a problem for a realistic collection size
    (tens of thousands of photos). If a long-lived process is needed — add an LRU
    modelled on `imaging.decode_rgb_cached`.
    """

    encode: FeatureEncoder
    score: FeatureScorer
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __call__(self, paths: list[str], prompts: list[str]) -> np.ndarray:
        missing = [p for p in paths if p not in self._cache]
        if missing:
            for p, feat in zip(missing, self.encode(missing)):
                if feat is not None:
                    self._cache[p] = feat
        zero = np.zeros(len(prompts), dtype=np.float32)
        valid_idx = [i for i, p in enumerate(paths) if p in self._cache]
        rows: list[np.ndarray] = [zero] * len(paths)
        if valid_idx:
            feats = np.stack([self._cache[paths[i]] for i in valid_idx])
            probs = self.score(feats, prompts)
            for j, i in enumerate(valid_idx):
                rows[i] = probs[j]
        return np.stack(rows)


def clip_classifier(s: NamingSettings) -> Classifier:  # pragma: no cover — ML, smoke test
    """The real open_clip zero-shot classifier (shared by landmarks and junk).

    Optimizations against the CPU-decode-bound bottleneck (Phase 6):
    - decode images in a batch IN PARALLEL (ThreadPoolExecutor; Pillow releases the
      GIL in the C decode);
    - inference in ONE batch on the GPU (encode_image over the whole batch, not one
      by one);
    - decode at a reduced resolution (Image.draft) — CLIP resizes to the model input
      anyway, so decoding a full-size HEIC/JPEG is pointless;
    - prompt text embeddings are cached (identical between batches);
    - image features are cached by path (F19, `CachingFeatureClassifier`) — one
      decode+encode_image per path over the classifier's lifetime, not per
      `classify()` call.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import open_clip
    import pillow_heif
    import torch
    from PIL import Image

    pillow_heif.register_heif_opener()  # so CLIP reads HEIC/HEIF (iPhone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        s.clip_model, pretrained=s.clip_pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(s.clip_model)
    model.eval()

    try:  # model input size → target for draft (with ×2 headroom for quality)
        _sz = preprocess.transforms[0].size
        _in = _sz[0] if isinstance(_sz, (list, tuple)) else int(_sz)
    except Exception:
        _in = 224
    _draft = (_in * 2, _in * 2)
    _pool = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))
    _text_cache: dict[tuple[str, ...], object] = {}

    def _load(path: str):
        try:
            with Image.open(path) as im:
                try:
                    im.draft("RGB", _draft)  # JPEG: decode at a reduced scale
                except Exception:
                    pass
                return preprocess(im.convert("RGB"))
        except Exception:
            return None  # corrupt/undecodable file → a zero row

    def _text_features(prompts: list[str]):
        key = tuple(prompts)
        cached = _text_cache.get(key)
        if cached is None:
            with torch.no_grad():
                tf = model.encode_text(tokenizer(list(prompts)).to(device))
                tf /= tf.norm(dim=-1, keepdim=True)
            _text_cache[key] = cached = tf
        return cached

    def encode(image_paths: list[str]) -> list[np.ndarray | None]:
        tensors = list(_pool.map(_load, image_paths))  # parallel decode
        results: list[np.ndarray | None] = [None] * len(image_paths)
        valid = [i for i, t in enumerate(tensors) if t is not None]
        if valid:
            batch = torch.stack([tensors[i] for i in valid]).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)  # the whole batch in one call
                feats /= feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()
            for j, i in enumerate(valid):
                results[i] = feats_np[j]
        return results

    def score(image_feats: np.ndarray, prompts: list[str]) -> np.ndarray:
        text_feat = _text_features(prompts)
        with torch.no_grad():
            feats_t = torch.from_numpy(image_feats).to(device)
            probs = (100.0 * feats_t @ text_feat.T).softmax(dim=-1).cpu().numpy()
        return probs

    return CachingFeatureClassifier(encode=encode, score=score)


@dataclass
class LandmarkStats:
    scanned: int = 0                  # files with places.confidence='unknown'
    matched: int = 0                  # got confidence='visual'
    by_landmark: dict[str, int] = field(default_factory=dict)


def detect_landmarks(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> LandmarkStats:
    """CLIP zero-shot over the landmark list for files without a resolved place.

    Incrementality for free: matched files get confidence='visual' and do not enter
    the next run (the selection is only for 'unknown').
    """
    s = naming_settings(cfg)
    landmarks = load_landmarks(s.landmarks_file)
    rows = conn.execute(
        """SELECT f.id, f.path FROM files f JOIN places p ON p.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
             AND p.confidence = 'unknown'
           ORDER BY f.id"""
    ).fetchall()
    stats = LandmarkStats(scanned=len(rows))
    if not rows or not landmarks:
        return stats
    if classifier is None:
        classifier = clip_classifier(s)  # pragma: no cover — ML, smoke test

    prompts = [lm.prompt for lm in landmarks] + list(_NEGATIVE_PROMPTS)
    now = utcnow_iso()
    done = 0
    if progress:
        progress(0, len(rows))  # total right away, even if the stage is small/fast (#37)
    with conn:
        for chunk in batched(rows, s.clip_batch_size):
            probs = classifier([r["path"] for r in chunk], prompts)
            for r, p in zip(chunk, probs):
                best = int(np.argmax(p[: len(landmarks)]))
                if float(p[best]) < s.landmark_threshold:
                    continue
                lm = landmarks[best]
                cur = conn.execute(
                    """UPDATE places SET country = ?, city = ?, confidence = 'visual',
                           updated_at = ?
                       WHERE file_id = ? AND confidence = 'unknown'""",
                    (lm.country, lm.city, now, r["id"]),
                )
                if cur.rowcount:
                    stats.matched += 1
                    stats.by_landmark[lm.name] = stats.by_landmark.get(lm.name, 0) + 1
            done += len(chunk)
            if progress:
                progress(done, len(rows))
    return stats
