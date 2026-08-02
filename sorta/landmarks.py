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

F131: the proposal can additionally be put to the local VLM before corroboration sees
it (`features.landmarks_verify`, default off) — CLIP proposes widely, the model says
what place it is looking at, corroboration still decides. The order matters and is the
whole safety argument: the model is a filter placed BEFORE F75, never a way around it.
F145: that toggle is subordinate to `vlm.enabled` — with deep analysis off this stage
raises no weights at all, whatever `landmarks_verify` says.

F136: what CLIP found for a frame is remembered (see `_SCAN_KEY`), so a later run only
looks at the frames whose answer could have changed — a new or edited file, a new
landmark list, a moved threshold, a reworded prompt. Corroboration is NOT cached: it runs
over the whole selection every time, with the proposals of the skipped files raised back
out of the DB, because a group verdict is about the company a match keeps and would
change under a thinned-out set (F75).

The CLIP model (open_clip, the same as in junk.py) is mocked in tests via the
classifier parameter; the real load happens only in clip_classifier(). The VLM is
mocked the same way through the `asker` parameter — qwen_vlm_landmark() is the only
place weights are loaded.
GPU: torch is installed as a CPU wheel (the project's CUDA wheels are only for
onnxruntime) — we run on the CPU, correctness over speed; the GPU variant will be
finished in Phase 6.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Callable, Iterator, Sequence, TypeVar

import numpy as np
import yaml
from PIL import Image

from . import i18n, imaging
from .config import Config, vlm_allowed
from .geodata import GeoResolver
from .naming import (
    DEFAULT_VLM_MODEL,
    VLM_MAX_EDGE,
    NamingSettings,
    naming_settings,
    shared_vlm,
    utcnow_iso,
)

_log = logging.getLogger(__name__)

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
    # F131: the check, and the same rule applies — the addition it buys has to be
    # measurable rather than assumed. `proposals` is the population the widened gate
    # collects, `checked` prices the pass (one question each), and the gap between
    # `confirmed_by_model` and `proposals` is what the check removed before corroboration
    # ever saw it. All zero when the toggle is off: nothing was asked.
    proposals: int = 0                   # matches above the gate, before any check
    # F136: files of the selection the CLIP pass did not have to look at, because nothing
    # that decides their answer had moved since the run that did look (see `_SCAN_KEY`).
    # They stay part of `scanned` and their proposals still go through corroboration — the
    # counter is here so that "the stage did nothing this time" is visible in the numbers
    # instead of being inferred from a suspiciously short run.
    skipped: int = 0
    checked: int = 0                     # proposals a question was actually asked about
    checks_reused: int = 0               # answers taken from landmark_checks, not re-asked
    confirmed_by_model: int = 0          # the model named this landmark (asked or reused)
    rejected_by_model: int = 0           # it named another place, or none at all
    checks_failed: int = 0               # the model raised -> the CLIP-only rule decided


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

    `path` and `score` carry what the F131 check needs and corroboration does not: the
    frame to show the model, and the number that decides the fallback when the model
    cannot be asked. Both default, because `scripts/measure_landmarks.py` builds matches
    for corroboration alone and has no use for either.
    """

    file_id: int
    folder: str
    landmark: Landmark
    path: str = ""
    score: float = 0.0


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


# --- F131: the check between the proposal and corroboration -----------------------
#
# path -> the model's raw answer about the place in one frame, read back by
# `match_named_landmark`. The same shape as the askers of the junk stage (F113/F130/F132)
# on purpose: one prompt over one frame, injected by the caller, so no test loads weights.
LandmarkAskFn = Callable[[str], str]

# THE question — and its FORM is a measured result here, not a wording preference. The
# phase-0 probe put both forms of the brief to the same frames, three runs:
#
#     form     backed a RIGHT proposal    backed a WRONG one
#     verify      20% / 42% / 42%             0 / 0 / 0
#     naming      80% / 80% / 80%             0 / 0 / 0
#
# "Was this taken at X?" invites a cautious "no" — it threw away more than half of what
# the stage finds today, in every run. The open question makes the model look instead of
# agree, and it is also why the false confirmations came out at zero: a model that does
# not know the place answers nothing (71 of the 104 answers) rather than guessing.
#
# The proposed landmark is deliberately NOT named in the prompt. Naming it is what turns
# a check into an agreement, and two models agreeing on the same wrong city is the exact
# failure this feature was measured against ("a false city is worse than no city", F75).
LANDMARK_NAMING_PROMPT = (
    "Look at the photo. If it shows a famous landmark or a well-known "
    "place, answer with the name of that place and nothing else. "
    "If it does not, answer with exactly one word: none."
)
# A place name, not an essay. Wider than the junk stage's eight tokens because a name is
# several words ("the Charles Bridge in Prague"), and no wider, because past that the only
# thing a larger budget buys is room for the model to explain itself.
LANDMARK_MAX_NEW_TOKENS = 16

# The list writes `name` in the interface language, so the English wording of a place
# comes from the CLIP prompt instead: asking a 3B model about «Карлов мост» would measure
# its Russian rather than its geography.
_PHOTO_PREFIX_RE = re.compile(r"^\s*(?:a|an|the)\s+photo\s+of\s+", re.IGNORECASE)
_NON_LETTER_RE = re.compile(r"[^a-z]+")
# Words that carry no place: articles, prepositions and the wording of the prompt itself.
# The KIND of a place ("bridge", "tower") is deliberately NOT here — two landmarks sharing
# one is exactly what the uniqueness rule in `match_named_landmark` is for.
_STOPWORDS = frozenset({"a", "an", "the", "of", "in", "and", "at", "photo", "s"})


def landmark_phrase(landmark: Landmark) -> str:
    """The English wording of a landmark ("the Charles Bridge in Prague")."""
    return _PHOTO_PREFIX_RE.sub("", landmark.prompt).strip() or landmark.name


def _answer_words(text: str) -> str:
    """Lowercased text with every non-letter run turned into an underscore, and padded.

    So a word is looked for as `_word_` and cannot match inside another one — "york" must
    not be found in "yorkshire", and a landmark must not be named by half of a longer word.
    """
    return "_" + _NON_LETTER_RE.sub("_", (text or "").lower()).strip("_") + "_"


def _content_words(landmark: Landmark) -> tuple[str, ...]:
    """The words of a landmark's English wording that could name a place."""
    return tuple(dict.fromkeys(
        word for word in _answer_words(landmark_phrase(landmark)).strip("_").split("_")
        if word and word not in _STOPWORDS))


def match_named_landmark(answer: str, landmarks: Sequence[Landmark]) -> int | None:
    """A free-form answer -> the index of the landmark it names, or None.

    A landmark counts as named when the answer carries its city, or a word that belongs to
    it ALONE in this list ("eiffel", "colosseum"), or at least two of its words. The
    uniqueness is computed against the list in hand rather than from a private word list:
    "bridge" is evidence in a list with one bridge in it and none at all in a list with
    two, and hard-coding either answer would be wrong on the day somebody adds a landmark.
    Ties go to the landmark that matched more words, then to list order.

    None — the answer named nothing we know, which covers both the model's "none" and its
    silence. That is a rejection, not a parse failure; see `_verify_matches`.
    """
    text = _answer_words(answer)
    words = [_content_words(lm) for lm in landmarks]
    seen = Counter(word for group in words for word in set(group))
    best: int | None = None
    best_score = 0
    for i, landmark in enumerate(landmarks):
        hits = [word for word in words[i] if f"_{word}_" in text]
        city = _answer_words(landmark.city)
        city_hit = len(city) > 2 and city in text
        if not (city_hit or len(hits) >= 2 or any(seen[word] == 1 for word in hits)):
            continue
        score = len(hits) + (1 if city_hit else 0)
        if score > best_score:
            best, best_score = i, score
    return best


def vlm_landmark_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                       max_edge: int) -> LandmarkAskFn:
    """The landmark question over an ALREADY LOADED runtime (naming.shared_vlm).

    The decode goes through the shared preview cache — Unicode/HEIC-safe, the same frame
    every other stage sees. A frame that will not decode gets an empty answer, and an
    empty answer names nothing, which is a rejection here: the proposal is dropped and
    the row stays `unknown`, exactly as if the model had said "none".
    """
    def ask(path: str) -> str:
        try:
            st = os.stat(path)
        except OSError:
            return ""
        img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=max_edge)
        if img is None:
            return ""
        return describe([img], LANDMARK_NAMING_PROMPT, LANDMARK_MAX_NEW_TOKENS)

    return ask


def qwen_vlm_landmark(model_name: str = DEFAULT_VLM_MODEL,
                      max_edge: int = VLM_MAX_EDGE,
                      ) -> LandmarkAskFn:  # pragma: no cover — ML, smoke test
    """The real asker — the SAME weights as every other question (F95): one per run."""
    return vlm_landmark_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_landmark_factory(max_edge: int) -> Callable[[str], LandmarkAskFn]:
    """The default asker factory of detect_landmarks, carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_landmark(model_name, max_edge=max_edge)


# What `landmark_checks.verdict` holds. Two values and no third: "the model was never
# asked" is the ABSENCE of a row, not a verdict, and giving it a name here would invite a
# reader to treat silence as data.
CHECK_CONFIRMED = "confirmed"
CHECK_REJECTED = "rejected"

_CHECK_UPSERT = """INSERT INTO landmark_checks (file_id, landmark, score, verdict,
                       model, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_id, landmark) DO UPDATE SET
                       score = excluded.score, verdict = excluded.verdict,
                       model = excluded.model, updated_at = excluded.updated_at"""


def landmark_check_model(model: str) -> str:
    """The `model` column: the runtime AND the fingerprint of the question it was asked.

    The group_keeper/frame_quality device (F120/F130/F132). A marker that names only the
    tier leaves every stored answer looking fresh after a prompt edit, and nobody should
    have to remember to empty a table by hand after rewording a question.

    Only the question is hashed. The CLIP prompts and the thresholds decide WHO is asked,
    not what comes back, so moving a threshold must not throw away answers that are still
    true — which is the whole point of storing them.
    """
    fingerprint = hashlib.sha1(LANDMARK_NAMING_PROMPT.encode("utf-8")).hexdigest()[:8]
    return f"{model}#{fingerprint}"


def _stored_checks(conn: sqlite3.Connection, file_ids: Sequence[int],
                   model: str) -> dict[tuple[int, str], str]:
    """(file_id, landmark name) -> verdict, for the proposals already asked about.

    Keyed by the PAIR because that is what was asked: CLIP proposing a different landmark
    for the same frame next run is a new question, not a stale answer. Chunked over the
    ids like every other collection-sized lookup here — SQLite has a ceiling on bound
    parameters that a photo library reaches easily.
    """
    out: dict[tuple[int, str], str] = {}
    for part in batched(list(file_ids), 500):
        rows = conn.execute(
            "SELECT file_id, landmark, verdict FROM landmark_checks WHERE model = ?"
            f" AND file_id IN ({','.join('?' * len(part))})", (model, *part))
        out.update(((int(r["file_id"]), str(r["landmark"])), str(r["verdict"]))
                   for r in rows)
    return out


def _verify_matches(matches: Sequence[_Match], landmarks: Sequence[Landmark],
                    ask: LandmarkAskFn, conn: sqlite3.Connection, model: str,
                    threshold: float, stats: LandmarkStats) -> list[_Match]:
    """Proposals -> the ones the model backed, with every answer remembered.

    Three outcomes, and they are deliberately NOT the same thing:

    * the model names the proposed landmark — confirmed, and the proposal goes on to
      corroboration, which still has the last word over it;
    * it names something else, or nothing at all — rejected. Silence is the COMMON case
      (71 of the 104 probe answers) and it is not a parse failure: it is the entire reason
      the false confirmations came out at zero, so treating it as "could not read the
      answer" and falling back to the CLIP rule would throw the feature away. A rejected
      frame keeps `unknown` — it is never moved to the place the model did name (F75);
    * the model RAISES — nothing was learned, so the proposal falls back to the rule of a
      run without the check (`score >= naming.landmark_threshold`) and nothing is stored.
      An expensive tier that is unavailable must not lose what the cheap one already
      finds; this is the rule the whole cascade is built on (F130).

    Answers are written as they come in, inside one transaction with the reads: a question
    already paid for should not have to be paid for twice, and the stored answer is what
    keeps a rejected frame — which stays `unknown` and so comes back into the stage on
    every run — from being asked about again and again.
    """
    known = _stored_checks(conn, [m.file_id for m in matches], model)
    now = utcnow_iso()
    kept: list[_Match] = []
    with conn:
        for m in matches:
            verdict = known.get((m.file_id, m.landmark.name))
            if verdict is None:
                try:
                    answer = ask(m.path)
                except Exception as exc:  # noqa: BLE001 — the cheap tier must survive it
                    _log.warning(
                        "landmarks: VLM-проверка не ответила по file_id=%s (%s) — "
                        "решает порог CLIP, как без проверки", m.file_id, exc)
                    stats.checks_failed += 1
                    if m.score >= threshold:
                        kept.append(m)
                    continue
                named = match_named_landmark(answer, landmarks)
                verdict = (CHECK_CONFIRMED
                           if named is not None and landmarks[named] == m.landmark
                           else CHECK_REJECTED)
                conn.execute(_CHECK_UPSERT, (m.file_id, m.landmark.name, m.score,
                                             verdict, model, now))
                stats.checked += 1
            else:
                stats.checks_reused += 1
            if verdict == CHECK_CONFIRMED:
                stats.confirmed_by_model += 1
                kept.append(m)
            else:
                stats.rejected_by_model += 1
    return kept


# --- F136: the scan marker — what a later run is allowed not to look at again -------
#
# The stage used to be incremental in its SELECTION alone: a match becomes
# `confidence='visual'` and leaves, everything else keeps `'unknown'` and was handed to
# CLIP again on every single run — 7 619 frames and 138 s of the 176 s a full run costs on
# the live collection, for an answer that could not have come out differently. The
# interface used to work around it with a button ("do not run this stage again"), which is
# a shortcoming of the stage delegated to a person; F135 removed the button.
#
# WHERE THE MARKER LIVES, and why not somewhere more obvious:
#
# * not in `places` — `geo` recomputes that table from scratch (`DELETE FROM places`) and
#   always runs BEFORE landmarks, so a marker written there is gone before it is read;
# * not in a file next to the DB — that is a second store to keep in step with
#   `reset_index`, with `sorta ui`'s "Start over", and with a database that gets copied or
#   moved; a stale sidecar would silently hand back answers for files it no longer
#   describes, which is the failure this whole marker exists to prevent;
# * `landmark_checks` is this stage's own table (F131): wiped with the index, keyed by
#   (file_id, landmark), already carrying a fingerprint in `model`. One RESERVED key per
#   file is all that is needed, and the schema does not move — which the F136 brief
#   forbids outright.
_SCAN_KEY = "#scan"
# What a scan row's `verdict` holds: the name of the landmark CLIP proposed for the file,
# or this — nothing reached the gate. Both are results of the pass, and neither can be
# confused with an answer of the model: those rows are keyed by a real landmark name (a
# place name out of the list, never "#..."), and `_stored_checks` filters on `model` on top
# of that.
_SCAN_NONE = "#none"


def _stage_fingerprint(landmarks: Sequence[Landmark], prompts: Sequence[str],
                       threshold: float, gate: float, min_group: int,
                       dominance: float) -> str:
    """A short digest of everything that decides this stage's answer (the F120 device).

    Deliberately WIDER than what decides a single CLIP score. The prompts and the gate are
    what produce a proposal, but an edited `data/landmarks.yaml` also moves the country,
    the city and the geonameid a match is written with, and the group thresholds decide
    which proposals survive — the brief names all of them, and every one of them is a way
    for a stored answer to become a lie while still looking fresh. The cost of being wide
    is a recompute nobody strictly needed; the cost of being narrow is old verdicts
    silently outliving the settings that produced them.
    """
    payload = "\n".join([
        *prompts,
        *(f"{lm.name}|{lm.country}|{lm.city}|{lm.geonameid}" for lm in landmarks),
        f"threshold={threshold!r}", f"gate={gate!r}",
        f"group_min={min_group}", f"dominance={dominance!r}",
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _scan_marker(fingerprint: str, row: sqlite3.Row) -> str:
    """The `model` of a scan row: the stage's fingerprint AND the file's, in one string.

    One string compared for equality, because the question the row answers is "is
    everything that decides this file's result still what it was" — and a marker that
    matches in part is not a match at all (the F120 rule: a mismatch means RECOMPUTE,
    never use). The file's half is path + mtime + size, taken from the index rather than
    from a fresh stat: the indexer is the one stage that decides what a file currently is,
    and asking the disk here would make the stage disagree with it.
    """
    identity = f"{row['path']}\0{row['mtime']!r}\0{row['size']!r}"
    return (f"scan#{fingerprint}#"
            f"{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}")


def _stored_scans(conn: sqlite3.Connection,
                  file_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
    """file_id -> its scan row, for the files of the current selection."""
    out: dict[int, sqlite3.Row] = {}
    for part in batched(list(file_ids), 500):
        rows = conn.execute(
            "SELECT file_id, score, verdict, model FROM landmark_checks WHERE landmark = ?"
            f" AND file_id IN ({','.join('?' * len(part))})", (_SCAN_KEY, *part))
        out.update((int(r["file_id"]), r) for r in rows)
    return out


def _reuse_scan(row: sqlite3.Row, scan: sqlite3.Row | None, fingerprint: str,
                by_name: dict[str, Landmark]) -> tuple[bool, _Match | None]:
    """(may the CLIP pass skip this file, what it proposed the last time it did not).

    A landmark the current list no longer knows is treated as no marker at all rather than
    as a missing proposal — it cannot happen while the fingerprint covers the list, and if
    it ever does, paying for one more CLIP pass is the cheap way to be wrong.
    """
    if scan is None or str(scan["model"]) != _scan_marker(fingerprint, row):
        return False, None
    name = str(scan["verdict"])
    if name == _SCAN_NONE:
        return True, None
    landmark = by_name.get(name)
    if landmark is None:
        return False, None
    return True, _Match(file_id=int(row["id"]), folder=_parent_dir(str(row["path"])),
                        landmark=landmark, path=str(row["path"]),
                        score=float(scan["score"] or 0.0))


def _remember_scans(conn: sqlite3.Connection,
                    scans: Sequence[tuple[int, str, _Match | None]]) -> None:
    """Store what the CLIP pass found for each frame it looked at.

    Written right after the pass and BEFORE corroboration, because the row records the
    PROPOSAL and not where the file ended up. The files this feature is actually for are
    the ones that never match: they keep `unknown`, come back into the selection on every
    run, and are 7 619 of the 7 619 frames the live collection rescans. A row written only
    for the survivors would miss all of them — and the survivors are the one group that
    already skipped itself, by leaving the selection.
    """
    now = utcnow_iso()
    with conn:
        for file_id, marker, match in scans:
            conn.execute(_CHECK_UPSERT,
                         (file_id, _SCAN_KEY, match.score if match else None,
                          match.landmark.name if match else _SCAN_NONE, marker, now))


def _vlm_model_name(cfg: Config) -> str:
    """The runtime the check would use — `vlm.model`, the shared one (F95)."""
    return str(getattr(getattr(cfg, "vlm", None), "model", DEFAULT_VLM_MODEL))


def _resolve_asker(cfg: Config, asker: LandmarkAskFn | None,
                   factory: Callable[[str], LandmarkAskFn] | None) -> LandmarkAskFn | None:
    """The check, or None when the model will not build.

    The graceful fallback every expensive tier in this project has (F37/F113/F130/F132):
    weights that fail to load cost the cheap tier nothing. Here they cost one thing more,
    and the caller pays it — the widening of the candidate gate goes back with them.
    """
    if asker is not None:
        return asker
    max_edge = int(getattr(getattr(cfg, "vlm", None), "max_edge", VLM_MAX_EDGE))
    build = factory or qwen_vlm_landmark_factory(max_edge)
    try:
        return build(_vlm_model_name(cfg))
    except Exception as exc:  # noqa: BLE001 — the check is optional, must not crash
        _log.warning("landmarks: VLM-проверка недоступна (%s) — "
                     "остаётся порог CLIP, как без неё", exc)
        return None


def _candidate_gate(cfg: Config, s: NamingSettings) -> float:
    """The score a proposal needs to reach the check.

    Never above `naming.landmark_threshold`: this gate exists to WIDEN the population the
    stage looks at, and a value above today's threshold would narrow it instead — losing
    finds that need no check at all.
    """
    candidate = float(getattr(getattr(cfg, "features", None),
                              "landmark_candidate_threshold", 0.5))
    return min(candidate, float(s.landmark_threshold))


def detect_landmarks(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    progress: Callable[[int, int], None] | None = None,
    resolver: GeoResolver | None = None,
    asker: LandmarkAskFn | None = None,
    asker_factory: Callable[[str], LandmarkAskFn] | None = None,
) -> LandmarkStats:
    """CLIP zero-shot over the landmark list for files without a resolved place.

    Incrementality has two halves. Matched files get confidence='visual' and do not enter
    the next run at all (the selection is only for 'unknown'); everything else is selected
    again, and F136 keeps the CLIP pass off the frames whose answer cannot have changed —
    their proposal is raised out of `landmark_checks` instead. Corroboration then sees the
    same set of matches a full run would have built, which is the point: the group rule
    reads the company a match keeps, so a thinned-out set is not a saving but a different
    verdict.

    F75: the classifier only proposes. Every match above the threshold is collected
    first, corroborated against its folder (see `_corroborate`) and only then written —
    a discarded match leaves the row 'unknown', it is never moved to another city.

    F131: with `features.landmarks_verify` on, the local VLM is asked what place each
    proposal shows and only the ones it names itself go on. The order is the safety
    argument and it does not bend: CLIP proposes -> the model checks -> corroboration
    decides. Agreement between two models never overrules a country named in the path.
    With the toggle off — or with a model that will not load — this function does exactly
    what it did before, down to the gate the proposals are collected at. F145: `vlm.enabled`
    is the same kind of "off" and stands above the toggle, so a run without deep analysis
    resolves the same set of places this stage resolved before F131 existed.
    """
    s = naming_settings(cfg)
    landmarks = load_landmarks(s.landmarks_file)
    rows = conn.execute(
        """SELECT f.id, f.path, f.mtime, f.size FROM files f
             JOIN places p ON p.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
             AND p.confidence = 'unknown'
           ORDER BY f.id"""
    ).fetchall()
    stats = LandmarkStats(scanned=len(rows))
    if not rows or not landmarks:
        return stats
    if classifier is None:
        classifier = clip_classifier(s)  # pragma: no cover — ML, smoke test

    # F145: `features.landmarks_verify` says WHAT to ask, `vlm.enabled` says whether a
    # model may be raised at all — and this stage was the one with no such condition
    # whatsoever, so a base run without deep analysis called the model here. The master
    # switch is folded into `verify` itself rather than into the asker below, because
    # `verify` also widens the candidate gate and that widening is part of the
    # fingerprint: with it off the stage is the one it was before F131, down to which
    # frames it rescans.
    verify = (vlm_allowed(cfg)
              and bool(getattr(getattr(cfg, "features", None), "landmarks_verify", False)))
    gate = _candidate_gate(cfg, s) if verify else s.landmark_threshold
    # The thresholds are read through getattr: the fields live in config.py, which this
    # module does not own (the F30/F37/F64 pattern) — an older settings object keeps
    # working on the measured defaults. Read here rather than by the corroboration call
    # below because the scan marker fingerprints them (F136).
    min_group = int(getattr(s, "landmark_group_min", 5))
    dominance = float(getattr(s, "landmark_group_dominance", 0.6))

    prompts = landmark_prompts(landmarks)
    fingerprint = _stage_fingerprint(landmarks, prompts, float(s.landmark_threshold),
                                     float(gate), min_group, dominance)
    stored = _stored_scans(conn, [int(r["id"]) for r in rows])
    by_name = {lm.name: lm for lm in landmarks}
    # Keyed by file so the match list can be rebuilt in SELECTION order below, whichever
    # half it came from. The order is not cosmetic: `_group_minority` breaks a tie between
    # two equally frequent cities by the order it met them, so a run that appended the
    # raised proposals after the fresh ones could decide a folder differently from the full
    # run it is supposed to be indistinguishable from.
    proposals: dict[int, _Match] = {}
    todo: list[sqlite3.Row] = []
    for r in rows:
        reusable, raised = _reuse_scan(r, stored.get(int(r["id"])), fingerprint, by_name)
        if not reusable:
            todo.append(r)
            continue
        stats.skipped += 1
        if raised is not None:
            proposals[int(r["id"])] = raised

    done = len(rows) - len(todo)
    if progress:
        progress(0, len(rows))  # total right away, even if the stage is small/fast (#37)
        if done:
            progress(done, len(rows))  # everything the marker spared, in one jump
    fresh: list[tuple[int, str, _Match | None]] = []
    for chunk in batched(todo, s.clip_batch_size):
        probs = classifier([r["path"] for r in chunk], prompts)
        for r, p in zip(chunk, probs):
            best = int(np.argmax(p[: len(landmarks)]))
            proposal: _Match | None = None
            if float(p[best]) >= gate:
                proposal = _Match(file_id=int(r["id"]), folder=_parent_dir(str(r["path"])),
                                  landmark=landmarks[best], path=str(r["path"]),
                                  score=float(p[best]))
                proposals[int(r["id"])] = proposal
            fresh.append((int(r["id"]), _scan_marker(fingerprint, r), proposal))
        done += len(chunk)
        if progress:
            progress(done, len(rows))
    if fresh:
        _remember_scans(conn, fresh)
    if stats.skipped:
        _log.info("landmarks: %s из %s кадров не изменились с прошлого прогона — "
                  "CLIP по ним не гонялся", stats.skipped, len(rows))

    matches = [proposals[int(r["id"])] for r in rows if int(r["id"]) in proposals]
    if not matches:
        return stats

    stats.proposals = len(matches)
    if verify:
        # The weights are asked for HERE and not before the CLIP pass: a run whose band is
        # empty has nothing to ask and should not pay 20 GB to find that out. A model that
        # will not build takes the widening back with it — the band is filtered to the
        # threshold a run without the check would have used, because a wide band with
        # nothing checking it is the one outcome worse than not having the feature.
        #
        # No progress reporting over the loop on purpose: the callback is a single
        # (done, total) pair owned by the CLIP pass above, and a second phase pushed
        # through it would make the bar run twice. The population is a couple of hundred
        # frames — the measured band at 0.50 is 151 — not a collection-sized pass.
        ask = _resolve_asker(cfg, asker, asker_factory)
        if ask is None:
            matches = [m for m in matches if m.score >= s.landmark_threshold]
        else:
            matches = _verify_matches(matches, landmarks, ask, conn,
                                      landmark_check_model(_vlm_model_name(cfg)),
                                      s.landmark_threshold, stats)
        if not matches:
            return stats

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
