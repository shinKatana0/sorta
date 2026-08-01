"""F6 (Phase 5): places without GPS — CLIP zero-shot over a curated landmark list.

Contract: reads files and places, writes ONLY into places and STRICTLY into rows
with confidence='unknown' (exact_gps / session_inferred / trip_inferred / visual are
not overwritten;
run order: geo always before landmarks).

F75: a single CLIP score does not separate a real Charles Bridge from a nice photo of
some other European street — measured on the live collection, the wrong cities scored
0.980 against 0.991 for the right one, so no threshold splits them. Three independent
signals do the job instead (see `_ANTI_PROMPTS`, `_group_minority`, `_folder_hint`):
the score only ever proposes a match, and the two corroboration rules run between the
proposal and the DB write.

The CLIP model (open_clip, the same as in junk.py) is mocked in tests via the
classifier parameter; the real load happens only in clip_classifier().
GPU: torch is installed as a CPU wheel (the project's CUDA wheels are only for
onnxruntime) — we run on the CPU, correctness over speed; the GPU variant will be
finished in Phase 6.
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Callable, Iterator, Sequence, TypeVar

import numpy as np
import yaml

from . import i18n, imaging
from .config import Config
from .geodata import GeoResolver
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

# F75 anti-classes. DO NOT DELETE AS "REDUNDANT" — each line here was measured on the
# live collection and each one buys something the plain negatives above do not:
# * the render/wallpaper/poster/figurine lines catch pictures OF a landmark that were
#   never taken at it (a video-game skyline scored 0.924 for Times Square; with these
#   it drops to 0.631);
# * the two generic-European lines catch the opposite error — a real photo of real
#   architecture that simply has no entry in our ten-landmark list, so CLIP hands it to
#   the nearest one it does know (17 Berlin fires went from a median 0.980 to 0.433).
# They cost the true positives some score too (Prague 0.991 -> 0.894), which is why the
# threshold is a config value and not a constant: adding or removing a line here shifts
# the whole distribution and `naming.landmark_threshold` has to be re-measured with
# `scripts/measure_landmarks.py`.
_ANTI_PROMPTS = (
    "a screenshot from a video game",
    "a desktop wallpaper",
    "a souvenir figurine of a famous landmark",
    "a poster or a painting of a landmark",
    "an ordinary building in a European city",
    "a street in an old European town",
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
    # GeoNames id of the city, so a landmark match lands in the SAME folder as a GPS
    # photo of the same place. Without it the sorter cannot localize the name — the
    # translation is looked up by geonameid — and "Paris" appeared next to "Париж" as
    # a second folder for one city. The ids are the ones a GPS photo taken AT the
    # landmark resolves to, which is what makes the two paths agree.
    # Optional: a user-supplied list without the field still works, just unlocalized.
    geonameid: int | None = None


# F65 follow-up: the same packaging trap the geo database fell into. The historical
# default is a path relative to the CURRENT DIRECTORY, so it only ever resolved when
# sorta was run from the repository root — an installed CLI found nothing. The
# curated list now ships as package data; _LEGACY_LANDMARKS_FILE keeps working trees
# checked out before the move alive.
DEFAULT_LANDMARKS_FILE = "data/landmarks.yaml"
_PACKAGE_LANDMARKS_FILE = Path(__file__).resolve().parent / "data" / "landmarks.yaml"
_LEGACY_LANDMARKS_FILE = Path(__file__).resolve().parent.parent / "data" / "landmarks.yaml"


def resolve_landmarks_file(configured: str | Path | None) -> Path:
    """Configured value -> an existing landmarks file.

    An existing path wins, so a user-supplied list is always respected. Only the
    historical default (or an empty value) falls back to the bundled file — a custom
    path that does not exist raises instead of silently swapping in our list, which
    would be indistinguishable from the config having been applied.
    """
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate
        if str(configured) != DEFAULT_LANDMARKS_FILE:
            raise FileNotFoundError(
                f"naming.landmarks_file: {candidate} not found. Point it at an "
                f"existing file or drop the setting to use the bundled list."
            )
    for fallback in (_PACKAGE_LANDMARKS_FILE, _LEGACY_LANDMARKS_FILE):
        if fallback.exists():
            return fallback
    raise FileNotFoundError(
        f"bundled landmark list not found at {_PACKAGE_LANDMARKS_FILE} — reinstall "
        f"sorta or set naming.landmarks_file to your own list."
    )


def load_landmarks(path: str | Path) -> list[Landmark]:
    """Read the landmark list (format: prompt/name/country/city)."""
    path = resolve_landmarks_file(path)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("landmarks") or []
    result: list[Landmark] = []
    for i, e in enumerate(entries):
        missing = [k for k in ("prompt", "name", "country", "city") if not e.get(k)]
        if missing:
            raise ValueError(f"{path}: запись #{i + 1} без полей {missing}")
        raw_id = e.get("geonameid")
        try:
            geonameid = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            raise ValueError(
                f"{path}: запись #{i + 1} ({e['name']}): geonameid должен быть числом, "
                f"получено {raw_id!r}"
            ) from None
        result.append(Landmark(prompt=str(e["prompt"]), name=str(e["name"]),
                               country=str(e["country"]), city=str(e["city"]),
                               geonameid=geonameid))
    return result


def landmark_prompts(landmarks: Sequence[Landmark]) -> list[str]:
    """The prompt list for one CLIP pass: landmarks first, distractors after them.

    The order is part of the contract — argmax is taken over the first len(landmarks)
    entries only, so the negative and anti classes can drain probability mass but can
    never win. `scripts/measure_landmarks.py` builds its prompts through here as well,
    otherwise it would measure a threshold for a prompt set the pipeline does not use.
    """
    return [lm.prompt for lm in landmarks] + list(_NEGATIVE_PROMPTS) + list(_ANTI_PROMPTS)


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

    def features(self, paths: list[str]) -> list[np.ndarray | None]:
        """The features ALREADY computed for `paths` — the cache, never a new encode.

        F128: the junk stage stores the vector it has just paid for, and the whole point
        of storing it is that no extra pass is run for it — so this hands back what the
        preceding `__call__` put in the cache and nothing else. A path that is not there
        (it did not decode, or nobody has scored it yet) is None, the same "no signal" a
        zero row means on the scoring side.
        """
        return [self._cache.get(p) for p in paths]

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


def _decode_pool_size(s: NamingSettings) -> int:
    """How many threads decode the frames of a CLIP batch (F64).

    The CLIP stages (junk/landmarks) are decode-bound, not GPU-bound: growing the
    batch barely moves the needle, while the decode pool does — on a 24-core
    machine 16 workers give ~1.6× the throughput of 8. So the auto default is one
    worker per core capped at 16 (past that the curve flattens out, and every
    in-flight decode costs memory).

    `clip.decode_workers` > 0 is a user override and wins over the auto default;
    the field is read via getattr — it is optional in the settings object.
    """
    override = int(getattr(s, "clip_decode_workers", 0) or 0)
    if override > 0:
        return override
    return max(1, min(os.cpu_count() or 4, 16))


def clip_classifier(s: NamingSettings) -> Classifier:  # pragma: no cover — ML, smoke test
    """The real open_clip zero-shot classifier (shared by landmarks and junk).

    Optimizations against the CPU-decode-bound bottleneck (Phase 6):
    - decode images in a batch IN PARALLEL (ThreadPoolExecutor; Pillow releases the
      GIL in the C decode);
    - inference in ONE batch on the GPU (encode_image over the whole batch, not one
      by one);
    - decode at a reduced resolution (F67: through the shared preview cache) — CLIP
      resizes to the model input anyway, so decoding a full-size HEIC/JPEG is
      pointless, and the preview is shared with the pHash/OCR/VLM stages;
    - prompt text embeddings are cached (identical between batches);
    - image features are cached by path (F19, `CachingFeatureClassifier`) — one
      decode+encode_image per path over the classifier's lifetime, not per
      `classify()` call.
    """
    from concurrent.futures import ThreadPoolExecutor

    import open_clip
    import pillow_heif
    import torch

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
    _pool = ThreadPoolExecutor(max_workers=_decode_pool_size(s))
    _text_cache: dict[tuple[str, ...], object] = {}

    def _load(path: str):
        # F67: the frame comes from the shared preview cache — the same file is also
        # decoded by junk (CLIP/OCR/VLM) and dedup (pHash), and the preview turns
        # every run after the first into a small-JPEG decode. mtime/size for the key
        # come from a local stat (microseconds against hundreds of ms of decode) so
        # that the Classifier/FeatureEncoder signatures stay untouched.
        try:
            st = os.stat(path)
            im = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=_draft[0])
            if im is None:
                return None  # corrupt/undecodable file → a zero row
            return preprocess(im)
        except Exception:
            return None

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
    # F75 corroboration: without these the feature cannot be measured, and it has to be
    # re-measured every time the prompts or the thresholds move. Each match above the
    # threshold falls into exactly one of these buckets or into `matched`.
    dropped_by_group: int = 0            # a minority city inside its own directory
    dropped_by_folder_name: int = 0      # the path names a different country
    confirmed_by_folder_name: int = 0    # the path names this country/city (kept)


# --- F75: corroboration of a CLIP match by its place in the tree -----------------

# A folder name may be compound ("чехия-австрия", "Франция Париж"), so besides the
# component itself every run of LETTERS in it is tried as well. Digits never belong to
# a place name here, so "100D3300" contributes only itself and the useless "D".
_WORD_RE = re.compile(r"[^\W\d_]+")

# Shorter components are not looked up at all: three letters collide with some city in
# a world-wide geo base far too easily ("Сад", "Море", "DCI"), and a false country would
# silently discard correct matches.
_MIN_COMPONENT_LEN = 4


@dataclass(frozen=True)
class _Match:
    """A landmark proposed by CLIP for a file, before corroboration.

    `folder` — the directory the file lies in; both rules are about the neighbours,
    so it is the grouping key and the source of the name hints.
    """

    file_id: int
    folder: str
    landmark: Landmark


@dataclass(frozen=True)
class _FolderHint:
    """What the directories above a file claim about where it was taken.

    Countries and cities are kept apart on purpose: a country name in the path is a
    deliberate human statement and may REFUTE a match, while a city name is only
    allowed to confirm one — "York", "Nice" or "Split" turn up inside perfectly
    innocent folder names, and a refutation on that basis would throw away good data.
    """

    countries: frozenset[str] = frozenset()       # ISO cc — confirm and refute
    city_countries: frozenset[str] = frozenset()  # ISO cc of a named city — confirm only
    city_ids: frozenset[int] = frozenset()        # geonameid of a named city — confirm only


def _parent_dir(path: str) -> str:
    return str(PurePath(path).parent)


def _folder_tokens(directory: str) -> list[str]:
    """Directory path -> candidate place names, in order, without repeats."""
    tokens: list[str] = []
    for part in PurePath(directory).parts:
        for token in (part, *_WORD_RE.findall(part)):
            token = token.strip()
            if len(token) >= _MIN_COMPONENT_LEN:
                tokens.append(token)
    return list(dict.fromkeys(tokens))


def _folder_hint(directory: str, resolver: GeoResolver, lang: i18n.Lang) -> _FolderHint:
    """Recognize countries/cities in a directory path via the bundled geo base.

    The geo base is the whole point: `names.tsv` holds the localized names, so
    "Франция", "France" and "フランス" resolve identically for the matching
    `cfg.language`, and technical components (DCIM, 100D3300, Camera, SORT) resolve to
    nothing simply because they are not places — no blocklist to maintain.
    """
    countries: set[str] = set()
    city_countries: set[str] = set()
    city_ids: set[int] = set()
    for token in _folder_tokens(directory):
        cc = resolver.country_cc_by_name(token, lang)
        if cc:
            countries.add(cc)
            continue
        for gid in resolver.city_ids_by_name(token, lang):
            city_ids.add(gid)
            region = resolver.region_key_of(gid)
            if region:
                city_countries.add(region[0])
    return _FolderHint(frozenset(countries), frozenset(city_countries), frozenset(city_ids))


_shared_geo: GeoResolver | None = None


def _default_resolver() -> GeoResolver:
    """One resolver per process, built on first use.

    The bundled base is read-only and costs ~12 MB plus a parse to load, while the
    stage can run several times inside one `sorta ui` session — a fresh resolver per
    call would pay for it every time.
    """
    global _shared_geo
    if _shared_geo is None:
        _shared_geo = GeoResolver()
    return _shared_geo


def _folder_hints(matches: Sequence[_Match], resolver: GeoResolver | None,
                  lang: i18n.Lang) -> list[_FolderHint]:
    """A hint per match, computed once per directory.

    Without the bundled geo data (`places.tsv` missing) every hint is empty and the
    folder-name rule simply does not fire — the stage degrades to the group rule
    instead of failing.
    """
    if resolver is None or not resolver.data_available():
        return [_FolderHint()] * len(matches)
    cache: dict[str, _FolderHint] = {}
    hints: list[_FolderHint] = []
    for m in matches:
        hint = cache.get(m.folder)
        if hint is None:
            hint = cache[m.folder] = _folder_hint(m.folder, resolver, lang)
        hints.append(hint)
    return hints


def _group_minority(matches: Sequence[_Match], min_group: int,
                    dominance: float) -> set[int]:
    """Indices of matches whose city is a minority inside its own directory.

    One card dump, one trip: you cannot physically have been in Prague and in Berlin
    within a single folder, so where a directory agrees on one city strongly enough,
    the odd ones out are the classifier being wrong — this alone removed 16 of the 17
    false Berlins that neither the threshold nor the anti-classes could touch.

    Deliberately NOT a reassignment to the dominant city: a folder called
    "чехия-австрия" makes Prague likely but Vienna just as possible, and inventing a
    place is worse than admitting we do not know one. Groups below `min_group` or
    without a clear majority are left completely alone — a two-photo folder says
    nothing.
    """
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(matches):
        groups.setdefault(m.folder, []).append(i)
    minority: set[int] = set()
    for idxs in groups.values():
        if len(idxs) < min_group:
            continue
        counts = Counter((matches[i].landmark.country, matches[i].landmark.city)
                         for i in idxs)
        top, n = counts.most_common(1)[0]
        if n < dominance * len(idxs):
            continue
        minority.update(i for i in idxs
                        if (matches[i].landmark.country, matches[i].landmark.city) != top)
    return minority


def _corroborate(matches: Sequence[_Match], hints: Sequence[_FolderHint],
                 min_group: int, dominance: float,
                 stats: LandmarkStats) -> list[_Match]:
    """Matches above the threshold -> the ones that survive both corroboration rules.

    Order (brief §4): the group rule decides first, the folder name is applied on top
    of it — an explicit human label on the path outranks the statistics of the
    neighbouring files and may bring back a match the group rule had discarded. A
    country in the path that contradicts the match wins over both: that is the case
    the user spotted by eye, and no amount of local agreement makes it right.
    """
    minority = _group_minority(matches, min_group, dominance)
    kept: list[_Match] = []
    for i, m in enumerate(matches):
        hint = hints[i]
        lm = m.landmark
        confirmed = (
            lm.country in hint.countries
            or lm.country in hint.city_countries
            or (lm.geonameid is not None and lm.geonameid in hint.city_ids)
        )
        if hint.countries and not confirmed:
            stats.dropped_by_folder_name += 1
            continue
        if confirmed:
            stats.confirmed_by_folder_name += 1
            kept.append(m)
            continue
        if i in minority:
            stats.dropped_by_group += 1
            continue
        kept.append(m)
    return kept


def detect_landmarks(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    progress: Callable[[int, int], None] | None = None,
    resolver: GeoResolver | None = None,
) -> LandmarkStats:
    """CLIP zero-shot over the landmark list for files without a resolved place.

    Incrementality for free: matched files get confidence='visual' and do not enter
    the next run (the selection is only for 'unknown').

    F75: the classifier only proposes. Every match above the threshold is collected
    first, corroborated against its folder (see `_corroborate`) and only then written —
    a discarded match leaves the row 'unknown', it is never moved to another city.
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

    prompts = landmark_prompts(landmarks)
    matches: list[_Match] = []
    done = 0
    if progress:
        progress(0, len(rows))  # total right away, even if the stage is small/fast (#37)
    for chunk in batched(rows, s.clip_batch_size):
        probs = classifier([r["path"] for r in chunk], prompts)
        for r, p in zip(chunk, probs):
            best = int(np.argmax(p[: len(landmarks)]))
            if float(p[best]) < s.landmark_threshold:
                continue
            matches.append(_Match(file_id=r["id"], folder=_parent_dir(r["path"]),
                                  landmark=landmarks[best]))
        done += len(chunk)
        if progress:
            progress(done, len(rows))
    if not matches:
        return stats

    # The thresholds are read through getattr: the fields live in config.py, which this
    # module does not own (the F30/F37/F64 pattern) — an older settings object keeps
    # working on the measured defaults.
    min_group = int(getattr(s, "landmark_group_min", 5))
    dominance = float(getattr(s, "landmark_group_dominance", 0.6))
    if resolver is None:
        resolver = _default_resolver()
    lang = i18n.normalize_lang(cfg.language)
    kept = _corroborate(matches, _folder_hints(matches, resolver, lang),
                        min_group, dominance, stats)

    now = utcnow_iso()
    with conn:
        for m in kept:
            lm = m.landmark
            cur = conn.execute(
                """UPDATE places SET country = ?, city = ?, city_geonameid = ?,
                       confidence = 'visual', updated_at = ?
                   WHERE file_id = ? AND confidence = 'unknown'""",
                (lm.country, lm.city, lm.geonameid, now, m.file_id),
            )
            if cur.rowcount:
                stats.matched += 1
                stats.by_landmark[lm.name] = stats.by_landmark.get(lm.name, 0) + 1
    return stats
