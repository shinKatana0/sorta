"""Closed eyes without a VLM: three ways to see them, priced on the labelling that exists.

F178, phase 0. The question "are the eyes open" was asked of a local VLM and the answer
cost more than it was worth: 60% precision, 9% recall over the collection, 92 minutes of
the stage. The question is gone (F167 -> F177) and the population is not — the same
labelling estimates ~948 frames with closed eyes, 15.6% of everything that has a face in
it. This script measures whether anything cheaper sees them, and it chooses nothing: it
prints a table and a verdict computed by criteria fixed in the code before the first run.

The three candidates of the brief, none preferred in advance (F130: the value proposed in
a brief turned out to be the worst row of its own table):

    ear    eyelid geometry — the eye contour of insightface `2d106det`, openness measured
           as the height of the eye opening over its width;
    crop   a small classifier over the eye crop — logistic regression on a 16x16
           grayscale patch, answered out-of-fold so the number is not its own memory;
    clip   the CLIP the pipeline already loads, asked about the CROP rather than the frame
           ("a closed eye" against "an open eye").

Two things the brief assumed that the index does not have
--------------------------------------------------------
`faces` stores `bbox` and `embedding` and nothing else — there is **no `kps` column**, so
the five detector points the brief builds on are not on disk. The detector computes them
and the stage throws them away. That changes what each variant costs rather than whether
it can be measured: this pass re-runs detection over the same previews, and the price
table below separates the part a phase 1 would pay (the 106-point model, 4.8 MB, already
inside the `buffalo_l` set) from the part it would not (detection, which the faces stage
runs anyway, and whose `kps` a phase 1 would store instead of recomputing).

`faces.bbox` is not usable as a crop either, and that is worth a brief of its own. On a
frame with EXIF orientation 6 the stage decodes through `cv2.imdecode`, which ALREADY
applied the rotation, and then rotates a second time (`faces._apply_orientation`), so the
box is written in a sideways frame — while the gate pass of the same stage
(`_decode_preview_for_faces`, PIL) and the sharpness crop of the junk stage (F155) work in
the upright one. This run reports how often a stored box lands on a face it re-detects
(`легли на бокс`), and takes its own boxes rather than trusting them: a phase 1 that crops
an eye out of `faces.bbox` would crop empty sky on every rotated photograph. Nothing here
touches that stage — the brief forbids it, and the finding belongs to whoever owns it.

The second one is the sample. The 249 labelled frames are NOT a random slice of the
collection: they are stratified by what the VLM answered, and every frame it called closed
is in them (50 of 135). Read flat, the labelling says 24% of frames have closed eyes and
the VLM found half of them. Weighted back to the population — 135 frames in the
`said_closed` stratum, 5 948 in the `said_open` one, from `frame_quality.eyes_open` — it
says 948 closed frames and 9% recall, which is exactly the number the brief was written
from. Every share printed below is weighted this way, and the VLM row is printed first
because without it any figure reads as an improvement.

What the numbers here cannot be
-------------------------------
Optimistic on the threshold: it is picked on the same 249 frames it is scored on, by the
rule in `best_row` and not by eye, but a threshold chosen and measured on one sample is an
upper bound and a phase 1 owes a re-measurement on fresh labels. Not optimistic on the
classifier: `crop` answers each frame from a model trained without it (`folds`).

"Cannot tell" is a third label, 17% of the sample, and it is counted the way the VLM
baseline was counted: a fire on a frame the owner could not read is a false positive, not
a free pass. The lenient column (`точн.+`) prints the other reading next to it, and the
share of the population those frames represent is the ceiling on any recall here.

Privacy: counts only. No path, no basename and no file id reaches the output — the rule of
measure_ocr_gate.py and measure_landmarks.py before it. The database is opened read-only.

Usage (from the repo root, with the GPU venv):
    python scripts/measure_eye_state.py --labels eyes_labels.json
    python scripts/measure_eye_state.py --labels eyes_labels.json --limit 40
    python scripts/measure_eye_state.py --labels eyes_labels.json --aggregate largest
    python scripts/measure_eye_state.py --labels eyes_labels.json --max-edge 512

The labelling file is `{file_id: "open" | "closed" | "cannot"}` — the worksheet the owner
filled in on 2026-08-03. The VLM answers are read from the database (`frame_quality`),
never from a second file: they are what the stage really stored.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import imaging, junk  # noqa: E402
from sorta.config import load_config  # noqa: E402
from sorta.landmarks import batched  # noqa: E402

# --- the labelling ----------------------------------------------------------------

LABEL_OPEN = "open"
LABEL_CLOSED = "closed"
LABEL_CANNOT = "cannot"      # the owner could not read the frame either
LABELS = (LABEL_CLOSED, LABEL_OPEN, LABEL_CANNOT)

# The two strata the sample was drawn from — what the VLM answered about the frame. Their
# sizes in the population come from the database, so the weights below describe the
# collection this ran on rather than a constant that would go stale after a re-index.
STRATUM_SAID_CLOSED = "said_closed"
STRATUM_SAID_OPEN = "said_open"
STRATA = (STRATUM_SAID_CLOSED, STRATUM_SAID_OPEN)

# --- the variants -----------------------------------------------------------------

VARIANT_EAR = "ear"
VARIANT_CROP = "crop"
VARIANT_CLIP = "clip"
VARIANTS = (VARIANT_EAR, VARIANT_CROP, VARIANT_CLIP)

# The grids each variant is swept over. `ear` fires BELOW its threshold (an eye is closed
# when the opening is small) and the other two fire above theirs, which is why the rule is
# carried per variant instead of being assumed once — see `fires`.
EAR_GRID = (0.10, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35)
PROBABILITY_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
GRIDS: Mapping[str, tuple[float, ...]] = {
    VARIANT_EAR: EAR_GRID,
    VARIANT_CROP: PROBABILITY_GRID,
    VARIANT_CLIP: PROBABILITY_GRID,
}
FIRES_BELOW = frozenset({VARIANT_EAR})

# The 106-point map of `2d106det`, verified against the detector's own eye centres on real
# frames (and re-verified on every run by `eye_rings_agree`): each eye is a ring of eight
# contour points plus a centre point that the model repeats twice. Only the SET matters —
# `eye_openness` finds the corners itself, so a reordering inside the ring cannot change a
# number, and a model that moves the ring elsewhere is caught by the check rather than
# quietly measured.
EYE_RINGS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (33, 35, 36, 37, 39, 40, 41, 42),
    (87, 89, 90, 91, 93, 94, 95, 96),
)
EYE_CENTRES: tuple[int, int] = (34, 88)
# How far a ring's own centre of mass may sit from the detector's eye point, in face
# widths, before the ring is not an eye at all.
EYE_RING_TOLERANCE = 0.08

# The crop both `crop` and `clip` are shown: a square around the eye centre, sized as a
# fraction of the face width, so the two variants differ in the classifier and in nothing
# else. 0.30 holds the eye with a little of the lid and the brow — the context a human
# needs for the same call.
CROP_FACE_FRACTION = 0.30
# Below this the crop is a dozen pixels of noise: the frame gets no answer rather than a
# guess (the rule of junk.FACE_CROP_MIN_PX, in preview pixels for the same reason).
MIN_CROP_PX = 12
# The classifier's input. 16x16 is 256 features against ~350 training eyes — larger learns
# the sample, smaller stops being a picture of an eye.
CROP_GRID_PX = 16

CLIP_PROMPTS = ("a close-up photo of a human eye, closed, the eyelid down",
                "a close-up photo of a human eye, open, the iris visible")

# The probe: fixed here, never tuned against the score it is judged by (measure_clip_probe).
PROBE_C = 1.0
PROBE_MAX_ITER = 2000
PROBE_FOLDS = 5

# How a frame with several faces answers. Both are printed: "any" is what a slice means
# (someone in this shot blinked) and "largest" is what a portrait means, and the
# difference between them on a collection with crowds in it is not small.
RULE_ANY = "any"
RULE_LARGEST = "largest"
RULES = (RULE_ANY, RULE_LARGEST)

# --- pre-registered acceptance criteria (F178) ------------------------------------
#
# 1. At least 50 frames labelled `closed` — below that the recall column is noise with a
#    decimal point. The sample has 59.
# 2. Precision no worse than the VLM's on the same frames. The baseline is not a constant
#    here: it is computed from the same labelling in the same run, so the comparison
#    survives a re-labelling.
# 3. Recall at least three times the VLM's. "Заметно выше 9%" is the brief's bar; three
#    times is the reading of it fixed before the run, and it is the same order as the gap
#    between asking every frame (which all three variants do) and asking the uncertain
#    band only (which is why the VLM's 51% on the frames it saw is 9% over the collection).
MIN_LABELLED_CLOSED = 50
MIN_RECALL_FACTOR = 3.0

VERDICT_GO = "ИДТИ В ФАЗУ 1"
VERDICT_CLOSE = "ЗАКРЫТЬ ТЕМУ"
VERDICT_UNCLEAR = "ВЕРДИКТА НЕТ"


def read_labels(path: str) -> dict[int, str]:
    """The worksheet `{file_id: label}` -> the same thing with ints and checked labels."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"{path}: файл разметки не читается ({exc})") from None
    except ValueError as exc:
        raise SystemExit(f"{path}: это не JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: ожидается отображение id -> метка")
    out: dict[int, str] = {}
    for key, value in raw.items():
        if str(value) not in LABELS:
            raise SystemExit(f"{path}: метка «{value}» не из набора {', '.join(LABELS)}")
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            raise SystemExit(f"{path}: ключ «{key}» не похож на file_id") from None
    if not out:
        raise SystemExit(f"{path}: разметка пуста — мерить не на чем")
    return out


@dataclass(frozen=True)
class Stratum:
    """One layer of the sample: how many frames were labelled, and how many exist."""
    name: str
    sampled: int
    population: int

    @property
    def weight(self) -> float:
        """Frames of the collection one labelled frame stands for."""
        return self.population / self.sampled if self.sampled else 0.0


def strata(sampled: Mapping[str, int], population: Mapping[str, int]) -> list[Stratum]:
    """The layers in a fixed order, so two runs print the same table."""
    return [Stratum(name=name, sampled=int(sampled.get(name, 0)),
                    population=int(population.get(name, 0)))
            for name in STRATA]


def weights(layers: Sequence[Stratum]) -> dict[str, float]:
    return {layer.name: layer.weight for layer in layers}


@dataclass(frozen=True)
class Frame:
    """One labelled frame after the models have run. Nothing here identifies it.

    `scores` holds one number per variant, or None where the variant could not answer —
    no face the detector agreed with, or an eye too small to crop. None never fires: a
    variant that says nothing has not found a closed eye, and that is a loss of recall
    rather than a silent exclusion from the denominator.
    """
    label: str
    stratum: str
    scores: Mapping[str, float | None] = field(default_factory=dict)


def fires(variant: str, score: float | None, threshold: float) -> bool:
    """Does `variant` call this frame closed at `threshold`?"""
    if score is None:
        return False
    return score <= threshold if variant in FIRES_BELOW else score >= threshold


@dataclass(frozen=True)
class Row:
    """One threshold of one variant, in frames of the COLLECTION (weighted).

    `unsure` is part of `fired`: a fire on a frame the owner could not read counts against
    precision, which is the convention the 60% of the baseline was computed under.
    """
    threshold: float
    fired: float
    hits: float
    unsure: float
    closed: float

    @property
    def precision(self) -> float:
        return self.hits / self.fired if self.fired else 0.0

    @property
    def lenient_precision(self) -> float:
        """The other reading: «не разглядеть» excused rather than counted as a miss."""
        judged = self.fired - self.unsure
        return self.hits / judged if judged > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.closed if self.closed else 0.0


def closed_population(frames: Sequence[Frame], w: Mapping[str, float]) -> float:
    return sum(w.get(f.stratum, 0.0) for f in frames if f.label == LABEL_CLOSED)


def score_rule(frames: Sequence[Frame], w: Mapping[str, float],
               threshold: float, fired: Callable[[Frame], bool]) -> Row:
    """One row from an arbitrary firing rule — the variants and the VLM share it."""
    hit = sum(w.get(f.stratum, 0.0) for f in frames if fired(f) and f.label == LABEL_CLOSED)
    unsure = sum(w.get(f.stratum, 0.0) for f in frames if fired(f) and f.label == LABEL_CANNOT)
    return Row(threshold=threshold,
               fired=sum(w.get(f.stratum, 0.0) for f in frames if fired(f)),
               hits=hit, unsure=unsure, closed=closed_population(frames, w))


def sweep(frames: Sequence[Frame], variant: str, w: Mapping[str, float],
          grid: Iterable[float] | None = None) -> list[Row]:
    """The variant over its whole grid — the table a threshold would be read off."""
    def rule(threshold: float) -> Callable[[Frame], bool]:
        return lambda frame: fires(variant, frame.scores.get(variant), threshold)

    return [score_rule(frames, w, threshold, rule(threshold))
            for threshold in (grid if grid is not None else GRIDS[variant])]


def baseline(frames: Sequence[Frame], w: Mapping[str, float]) -> Row:
    """The VLM, from the answers it stored: it fires exactly on the `said_closed` layer."""
    return score_rule(frames, w, threshold=0.0,
                      fired=lambda f: f.stratum == STRATUM_SAID_CLOSED)


def best_row(rows: Sequence[Row], precision_floor: float) -> Row | None:
    """The most recall a variant reaches without going below the precision floor.

    The rule is the whole point of writing it down: a threshold picked by looking at the
    table is not a measurement of anything (F131). Ties go to the higher precision, and a
    row that fires on nothing is not a candidate — its precision is 0/0.
    """
    usable = [r for r in rows if r.fired > 0 and r.precision >= precision_floor]
    if not usable:
        return None
    return max(usable, key=lambda r: (r.recall, r.precision))


@dataclass(frozen=True)
class Candidate:
    """A variant's best row under the criteria, kept next to the variant's name."""
    variant: str
    row: Row


def candidates(frames: Sequence[Frame], w: Mapping[str, float],
               floor: float) -> list[Candidate]:
    out: list[Candidate] = []
    for variant in VARIANTS:
        row = best_row(sweep(frames, variant, w), floor)
        if row is not None:
            out.append(Candidate(variant=variant, row=row))
    return out


def decide(found: Sequence[Candidate], base: Row,
           labelled_closed: int) -> tuple[str, str]:
    """The pre-registered verdict, from the criteria above and from nothing else."""
    if labelled_closed < MIN_LABELLED_CLOSED:
        return VERDICT_UNCLEAR, (f"кадров с меткой «закрыты» {labelled_closed} < "
                                 f"{MIN_LABELLED_CLOSED} — полнота не измеряется")
    if base.closed <= 0:
        return VERDICT_UNCLEAR, "в выборке нет закрытых глаз — сравнивать не с чем"
    bar = base.recall * MIN_RECALL_FACTOR
    passing = [c for c in found if c.row.recall >= bar]
    if passing:
        best = max(passing, key=lambda c: c.row.recall)
        return VERDICT_GO, (f"вариант «{best.variant}» на пороге {best.row.threshold:g}: "
                            f"полнота {best.row.recall:.0%} (>= {bar:.0%}) при точности "
                            f"{best.row.precision:.0%} (>= {base.precision:.0%} у VLM)")
    if not found:
        return VERDICT_CLOSE, (f"ни один вариант не удержал точность VLM "
                               f"({base.precision:.0%}) ни на одном пороге")
    best = max(found, key=lambda c: c.row.recall)
    return VERDICT_CLOSE, (f"лучший вариант «{best.variant}»: полнота "
                           f"{best.row.recall:.0%} < {bar:.0%} "
                           f"(порог — тройная полнота VLM, {base.recall:.0%})")


# --- geometry ---------------------------------------------------------------------


def eye_openness(points: np.ndarray) -> float | None:
    """The eye opening over the eye width, from a ring of contour points.

    The two points furthest apart are the corners: they define the eye's own axis, and the
    opening is the spread of the ring ACROSS that axis. Written this way the number does
    not care how the head is tilted, nor in which order the model happens to list the ring
    — the alternative, naming four indices as "upper lid" and "lower lid", is a promise
    about a model file that a new release can quietly break.

    None when the ring is degenerate (fewer than three points, or all of them in one
    place): "not measured", never a small number that would sort to the top of a list of
    closed eyes it was never measured for.
    """
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None
    diffs = points[:, None, :] - points[None, :, :]
    distances = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    width = float(distances[i, j])
    if width <= 0.0:
        return None
    axis = (points[j] - points[i]) / width
    normal = np.array([-axis[1], axis[0]])
    across = (points - points[i]) @ normal
    return float(across.max() - across.min()) / width


def eye_rings_agree(landmarks: np.ndarray, eye_points: np.ndarray,
                    face_width: float, tolerance: float = EYE_RING_TOLERANCE) -> bool:
    """Do `EYE_RINGS` still sit on the eyes the detector found?

    The index map is a property of one model file, not of the code, so it is checked
    against a second opinion on every run: the detector's own eye points, which come from
    a different network. A `2d106det` that ever ships a different order fails this loudly
    instead of producing an openness number about somebody's eyebrow.
    """
    if face_width <= 0 or landmarks.shape[0] <= max(max(r) for r in EYE_RINGS):
        return False
    centres = [landmarks[list(ring)].mean(axis=0) for ring in EYE_RINGS]
    for centre in centres:
        near = min(float(np.hypot(*(centre - point))) for point in eye_points)
        if near / face_width > tolerance:
            return False
    return True


def crop_box(centre: tuple[float, float], side: float,
             size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A square crop around the eye, clamped to the frame; None if it lands too small.

    Clamped rather than dropped for the same reason junk.face_crop_boxes clamps: a face at
    the border is an ordinary photograph, and half a crop of it still shows an eye. Too
    small after clamping is another matter — see MIN_CROP_PX.
    """
    width, height = size
    half = side / 2.0
    left = max(0, int(round(centre[0] - half)))
    top = max(0, int(round(centre[1] - half)))
    right = min(width, int(round(centre[0] + half)))
    bottom = min(height, int(round(centre[1] + half)))
    if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
        return None
    return left, top, right, bottom


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap of two boxes — how a detection is matched to a face already in the index."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- the frame's answer out of its faces -------------------------------------------


@dataclass(frozen=True)
class Eye:
    """One eye of one face: what each variant said about it, and how big the face was."""
    face_area: float
    openness: float | None = None
    crop_closed: float | None = None
    clip_closed: float | None = None

    def score(self, variant: str) -> float | None:
        return {VARIANT_EAR: self.openness, VARIANT_CROP: self.crop_closed,
                VARIANT_CLIP: self.clip_closed}[variant]


def frame_score(eyes: Sequence[Eye], variant: str, rule: str) -> float | None:
    """The frame's number from its eyes: the MOST CLOSED one, over the chosen faces.

    "Most closed" and not the average, and it is the mirror image of the choice F155 made
    for sharpness: a frame is blurred when every face in it is, and a frame belongs in the
    closed-eyes slice when one face in it has them shut. Which faces count is `rule` — all
    of them, or only the largest, which is the same frame seen as a portrait.
    """
    chosen = list(eyes)
    if rule == RULE_LARGEST and chosen:
        top = max(e.face_area for e in chosen)
        chosen = [e for e in chosen if e.face_area >= top]
    values = [score for score in (e.score(variant) for e in chosen) if score is not None]
    if not values:
        return None
    return min(values) if variant in FIRES_BELOW else max(values)


# --- the crop classifier ------------------------------------------------------------


def folds(labels: Sequence[str], count: int, seed: int) -> list[int]:
    """Fold number per frame, stratified by label and reproducible from the seed.

    Stratified because `closed` is a quarter of the sample: an unlucky shuffle would leave
    a fold with none of them in training, and the probe would then report a failure of the
    crop rather than of the split.
    """
    if count < 2:
        raise SystemExit(f"--folds: нужно хотя бы 2 части, получено {count}")
    rng = random.Random(seed)
    out = [0] * len(labels)
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            out[index] = position % count
    return out


def probe_predictions(features: np.ndarray, labels: Sequence[int],
                      groups: Sequence[int], fold: Sequence[int],
                      seed: int) -> list[float | None]:
    """Out-of-fold p(closed) per eye: every eye answered by a model that never saw it.

    Trained on eyes only, so `groups` (the frame each eye belongs to) is what carries the
    fold across them — the two eyes of one face are the same picture twice, and splitting
    them between training and test would measure that and nothing else.

    An eye whose frame carries no usable per-eye truth (`labels` is -1: a frame with
    several faces, where the frame's label says nothing about which face closed its eyes)
    is never trained on, only answered.
    """
    from sklearn.linear_model import LogisticRegression

    out: list[float | None] = [None] * len(labels)
    for part in sorted(set(fold)):
        train = [i for i in range(len(labels))
                 if fold[groups[i]] != part and labels[i] >= 0]
        test = [i for i in range(len(labels)) if fold[groups[i]] == part]
        if not test or len({labels[i] for i in train}) < 2:
            continue
        model = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=seed)
        model.fit(features[train], [labels[i] for i in train])
        closed = list(model.classes_).index(1)
        answers = model.predict_proba(features[test])
        for position, index in enumerate(test):
            out[index] = float(answers[position][closed])
    return out


# --- prices --------------------------------------------------------------------------


@dataclass
class Timing:
    """Seconds spent per stage, and how many things each stage was given."""
    decode_s: float = 0.0
    frames: int = 0
    detect_s: float = 0.0
    landmark_s: float = 0.0
    faces: int = 0
    crop_s: float = 0.0
    clip_s: float = 0.0
    eyes: int = 0

    def per(self, seconds: float, count: int) -> float:
        """Milliseconds per item; 0 when the stage never ran."""
        return seconds * 1000.0 / count if count else 0.0


@dataclass(frozen=True)
class Price:
    """What one variant costs per frame, and what it would have to download."""
    variant: str
    extra_ms: float          # milliseconds a phase 1 would ADD to a run, per frame
    weights_mb: float        # new weights to ship; 0 — already on disk
    note: str


def prices(timing: Timing) -> list[Price]:
    """The price table, per FRAME of the collection that has a face in it.

    What counts as "extra" is the whole judgement in this function. The decode is not:
    the junk stage already decodes this preview for the laplacian (F155), and the faces
    stage decodes it for its own gate. Detection is not either — the faces stage runs it,
    and a phase 1 that stored `kps` would not run it again. What is left is the model the
    project does not run today (the 106 points) and the arithmetic over the crop.
    """
    per_frame_faces = timing.faces / timing.frames if timing.frames else 0.0
    per_frame_eyes = timing.eyes / timing.frames if timing.frames else 0.0
    landmark = timing.per(timing.landmark_s, timing.faces) * per_frame_faces
    crop = timing.per(timing.crop_s, timing.eyes) * per_frame_eyes
    clip = timing.per(timing.clip_s, timing.eyes) * per_frame_eyes
    return [
        Price(VARIANT_EAR, landmark, 0.0,
              "2d106det уже лежит в наборе buffalo_l (4,8 МБ), детекция — не новая"),
        Price(VARIANT_CROP, crop, 0.0,
              "нужны координаты глаза: kps детектора (колонка в faces + пересканирование) "
              "или те же 106 точек"),
        Price(VARIANT_CLIP, clip, 0.0,
              "модель CLIP уже поднята стадией junk; вырезка кодируется отдельно от кадра"),
    ]


# --- the report -----------------------------------------------------------------------


def format_sample(layers: Sequence[Stratum], counts: Mapping[str, int]) -> str:
    lines = ["\nВыборка и вес слоёв (разметка стратифицирована по ответу VLM):",
             f"{'слой':>12} | {'размечено':>10} | {'в коллекции':>12} | {'вес':>7}",
             "-" * 50]
    for layer in layers:
        lines.append(f"{layer.name:>12} | {layer.sampled:>10} | {layer.population:>12} | "
                     f"{layer.weight:>7.1f}")
    labelled = ", ".join(f"{label} {counts.get(label, 0)}" for label in LABELS)
    total = sum(counts.values())
    share = counts.get(LABEL_CANNOT, 0) / total if total else 0.0
    lines.append(f"\nМетки: {labelled} (всего {total})")
    lines.append(f"«не разглядеть» — {share:.0%} выборки; это потолок: кадр, который не "
                 f"прочёл владелец,\n  не может быть уверенно показан срезом")
    return "\n".join(lines)


def format_population(frames: Sequence[Frame], w: Mapping[str, float]) -> str:
    closed = closed_population(frames, w)
    unsure = sum(w.get(f.stratum, 0.0) for f in frames if f.label == LABEL_CANNOT)
    total = sum(w.get(f.stratum, 0.0) for f in frames)
    return (f"\nЧто это значит для коллекции: из {total:.0f} кадров с лицами "
            f"{closed:.0f} с закрытыми глазами\n  ({closed / total:.1%} — популяция, ради "
            f"которой всё), и ещё {unsure:.0f} не читаются глазом.")


def format_table(title: str, rows: Sequence[Row], base: Row) -> str:
    lines = [f"\n{title}",
             f"{'порог':>7} | {'сработает':>10} | {'верно':>8} | {'точность':>9} | "
             f"{'точн.+':>7} | {'полнота':>8}",
             "-" * 62]
    for row in rows:
        lines.append(f"{row.threshold:>7g} | {row.fired:>10.0f} | {row.hits:>8.0f} | "
                     f"{row.precision:>9.0%} | {row.lenient_precision:>7.0%} | "
                     f"{row.recall:>8.0%}")
    lines.append(f"{'VLM':>7} | {base.fired:>10.0f} | {base.hits:>8.0f} | "
                 f"{base.precision:>9.0%} | {base.lenient_precision:>7.0%} | "
                 f"{base.recall:>8.0%}   <- базовая линия")
    return "\n".join(lines)


def format_prices(rows: Sequence[Price]) -> str:
    lines = ["\nЦена (миллисекунды НА КАДР сверх того, что прогон уже платит):",
             f"{'вариант':>8} | {'мс/кадр':>8} | {'новые веса':>11} | что именно",
             "-" * 84]
    for row in rows:
        lines.append(f"{row.variant:>8} | {row.extra_ms:>8.1f} | "
                     f"{row.weights_mb:>10.1f}Мб | {row.note}")
    return "\n".join(lines)


def format_coverage(frames: Sequence[Frame]) -> str:
    """How often each variant said nothing at all — the ceiling under every recall above."""
    lines = ["\nМолчание вариантов (кадр, о котором вариант не сказал ничего):"]
    for variant in VARIANTS:
        silent = sum(1 for f in frames if f.scores.get(variant) is None)
        share = f" ({silent / len(frames):.0%})" if frames else ""
        lines.append(f"  {variant:<6} {silent:>4} из {len(frames):>4}{share}")
    return "\n".join(lines)


def format_verdict(found: Sequence[Candidate], base: Row, labelled_closed: int) -> str:
    verdict, why = decide(found, base, labelled_closed)
    return (f"\nВЕРДИКТ ФАЗЫ 0: {verdict}\n  {why}\n"
            f"  (критерии зафиксированы до прогона: точность не ниже VLM, полнота "
            f"не ниже {MIN_RECALL_FACTOR:g}x полноты VLM,\n   кадров «закрыты» "
            f">= {MIN_LABELLED_CLOSED}; порог выбирается правилом best_row, не глазом)")


def report(frames: Sequence[Frame], layers: Sequence[Stratum], w: Mapping[str, float],
           counts: Mapping[str, int], timing: Timing, rule: str) -> str:
    base = baseline(frames, w)
    found = candidates(frames, w, base.precision)
    labelled_closed = counts.get(LABEL_CLOSED, 0)
    parts = [format_sample(layers, counts), format_population(frames, w),
             format_prices(prices(timing)), format_coverage(frames)]
    for variant in VARIANTS:
        parts.append(format_table(f"«{variant}» (кадр по правилу «{rule}»)",
                                  sweep(frames, variant, w), base))
    parts.append(format_verdict(found, base, labelled_closed))
    return "\n".join(parts)


# --- reading the collection -----------------------------------------------------------


def population_strata(conn: sqlite3.Connection) -> dict[str, int]:
    """The size of each layer in the collection, from the answers the stage stored.

    `eyes_open IS NOT NULL` is the population the sample was drawn from — the frames the
    question was ever asked about. NULL means "not asked" (the schema's rule) and those
    frames are outside every number here, which is exactly why the VLM's recall over the
    collection is 9% and not the 51% it scores on the frames it saw.
    """
    asked = conn.execute(
        "SELECT COUNT(*) FROM frame_quality WHERE eyes_open IS NOT NULL").fetchone()[0]
    said_closed = conn.execute(
        "SELECT COUNT(*) FROM frame_quality WHERE eyes_open = 0").fetchone()[0]
    return {STRATUM_SAID_CLOSED: int(said_closed),
            STRATUM_SAID_OPEN: int(asked) - int(said_closed)}


def sample_strata(conn: sqlite3.Connection, file_ids: Sequence[int]) -> dict[int, str]:
    """Which layer each labelled frame belongs to — the stored VLM answer, per frame."""
    out: dict[int, str] = {}
    for part in batched(list(file_ids), 500):
        rows = conn.execute(
            f"""SELECT file_id, eyes_open FROM frame_quality
                WHERE eyes_open IS NOT NULL
                  AND file_id IN ({','.join('?' * len(part))})""", tuple(part))
        for file_id, eyes_open in rows:
            out[int(file_id)] = (STRATUM_SAID_OPEN if int(eyes_open)
                                 else STRATUM_SAID_CLOSED)
    return out


def run(args: argparse.Namespace) -> int:  # pragma: no cover — ML, needs the GPU
    """The measurement end to end: one decode per frame, three answers out of it."""
    from sorta import faces as faces_mod

    labels = read_labels(args.labels)
    cfg = load_config(args.config)
    conn = sqlite3.connect(f"file:{cfg.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    known = sample_strata(conn, sorted(labels))
    missing = [file_id for file_id in labels if file_id not in known]
    if missing:
        print(f"без ответа VLM в базе: {len(missing)} кадров — они выпадают из замера")
    chosen = sorted(set(labels) & set(known))
    if args.limit:
        chosen = chosen[:args.limit]
    rows = conn.execute(
        f"""SELECT id, path, orientation FROM files
            WHERE id IN ({','.join('?' * len(chosen))}) ORDER BY id""", tuple(chosen)
    ).fetchall()
    boxes = junk.read_face_boxes(conn, chosen)

    settings = faces_mod._settings(cfg)
    print(f"поднимаем buffalo_l (детекция + 106 точек), {len(rows)} кадров...")
    detect, landmark = _face_models(settings)
    clip_score = _clip_crop_scorer(cfg)

    timing = Timing()
    eyes_by_frame: dict[int, list[Eye]] = {}
    crops: list[Image.Image] = []
    crop_owner: list[tuple[int, int]] = []   # (file_id, index of the eye in its frame)
    losses: Counter[str] = Counter()
    faces_found = agreed = small_crops = ring_failed = 0

    for done, row in enumerate(rows, 1):
        file_id = int(row["id"])
        eyes_by_frame[file_id] = []
        frame = _decode(row["path"], args.max_edge, timing)
        if frame is None:
            losses["кадр не декодировался"] += 1
            continue
        timing.frames += 1
        array = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
        start = time.perf_counter()
        found = detect(array)
        timing.detect_s += time.perf_counter() - start
        if not found:
            losses["детектор не нашёл лица на превью"] += 1
            continue
        stored = junk.face_crop_boxes(boxes.get(file_id, junk.NO_FACES), frame.size,
                                      min_px=0)
        eyes: list[Eye] = []
        for bbox, kps in found:
            faces_found += 1
            if stored and max(box_iou(bbox, box) for box in stored) >= args.min_iou:
                agreed += 1
            width = float(bbox[2] - bbox[0])
            area = width * float(bbox[3] - bbox[1])
            start = time.perf_counter()
            landmarks = landmark(array, bbox, kps)
            timing.landmark_s += time.perf_counter() - start
            if not eye_rings_agree(landmarks, kps[:2], width):
                ring_failed += 1
                continue
            for ring, centre_index in zip(EYE_RINGS, EYE_CENTRES):
                openness = eye_openness(landmarks[list(ring)])
                centre = (float(landmarks[centre_index][0]),
                          float(landmarks[centre_index][1]))
                box = crop_box(centre, width * CROP_FACE_FRACTION, frame.size)
                eyes.append(Eye(face_area=area, openness=openness))
                if box is None:
                    small_crops += 1
                    continue
                crops.append(frame.crop(box))
                crop_owner.append((file_id, len(eyes) - 1))
        eyes_by_frame[file_id] = eyes
        if not eyes:
            losses["у кадра не осталось ни одного измеримого глаза"] += 1
        print(f"  {done}/{len(rows)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    timing.faces = faces_found
    timing.eyes = len(crops)
    print(f"лиц на превью: {faces_found}; глаз с вырезкой: {len(crops)}, "
          f"вырезка меньше {MIN_CROP_PX} px: {small_crops}")
    print(f"легли на бокс из faces.bbox (IoU >= {args.min_iou:g}): {agreed} из "
          f"{faces_found} — см. примечание об ориентации в шапке файла")
    if ring_failed:
        print(f"кольцо 106 точек не легло на глаза детектора: {ring_failed} лиц из "
              f"{faces_found} — эти лица не измерялись ни одним вариантом")
    for reason, count in losses.most_common():
        print(f"  потеряно кадров — {reason}: {count}")

    if crops:
        start = time.perf_counter()
        features = np.stack([_crop_features(crop) for crop in crops])
        timing.crop_s += time.perf_counter() - start
        truth, groups, order = _probe_truth(crop_owner, labels, eyes_by_frame)
        fold = folds([labels[file_id] for file_id in order], PROBE_FOLDS, args.seed)
        start = time.perf_counter()
        probabilities = probe_predictions(features, truth, groups, fold, args.seed)
        timing.crop_s += time.perf_counter() - start
        start = time.perf_counter()
        clip_closed = clip_score(crops, args.clip_batch_size)
        timing.clip_s += time.perf_counter() - start
        for (file_id, index), probability, closed in zip(crop_owner, probabilities,
                                                         clip_closed):
            eye = eyes_by_frame[file_id][index]
            eyes_by_frame[file_id][index] = Eye(
                face_area=eye.face_area, openness=eye.openness,
                crop_closed=probability, clip_closed=closed)

    layers = strata(Counter(known[file_id] for file_id in chosen),
                    population_strata(conn))
    w = weights(layers)
    counts = Counter(labels[file_id] for file_id in chosen)
    for rule in (RULES if args.aggregate == "both" else (args.aggregate,)):
        frames = [Frame(label=labels[file_id], stratum=known[file_id],
                        scores={variant: frame_score(eyes_by_frame.get(file_id, ()),
                                                     variant, rule)
                                for variant in VARIANTS})
                  for file_id in chosen]
        print(report(frames, layers, w, counts, timing, rule))
    conn.close()
    return 0


def _decode(path: str, max_edge: int, timing: Timing):  # pragma: no cover — I/O
    """The frame every variant is measured on: the shared preview, oriented (F155)."""
    start = time.perf_counter()
    try:
        st = os.stat(path)
        image: Image.Image | None = imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge, apply_orientation=True)
    except OSError:
        image = None
    timing.decode_s += time.perf_counter() - start
    return image


def _face_models(settings):  # pragma: no cover — ML
    """(detect, landmark) out of buffalo_l — the two halves called and TIMED apart.

    `FaceAnalysis.get` runs both in one call, which is how the pipeline would use them and
    exactly what a price table must not do: the faces stage already pays for detection, so
    a number that bundles the two would price a phase 1 at four times what it costs. The
    two calls below are the ones `get` makes internally, with the same pinned `det_size`
    (F88) and the same threshold the stage runs with.
    """
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face

    from sorta import faces as faces_mod

    faces_mod._enable_cuda_dll_dirs()
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_2d_106"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=float(settings.det_threshold),
                det_size=(int(settings.det_size), int(settings.det_size)))
    model = app.models["landmark_2d_106"]

    def detect(array: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        found, points = app.det_model.detect(array, max_num=0, metric="default")
        return [(np.asarray(found[i][:4], dtype=np.float64),
                 np.asarray(points[i], dtype=np.float64)) for i in range(len(found))]

    def landmark(array: np.ndarray, bbox: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return np.asarray(model.get(array, Face(bbox=bbox, kps=kps, det_score=1.0)),
                          dtype=np.float64)

    return detect, landmark


def _crop_features(crop) -> np.ndarray:  # pragma: no cover — image arithmetic
    """One eye crop -> the classifier's input: a small grayscale square, contrast-free.

    Per-crop normalization is the point: an eye in the shade and an eye in the sun differ
    by an exposure the classifier must not learn, because the collection is full of both.
    """
    small = crop.convert("L").resize((CROP_GRID_PX, CROP_GRID_PX),
                                     Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.float64).reshape(-1) / 255.0
    spread = values.std()
    return (values - values.mean()) / spread if spread > 1e-6 else values - values.mean()


def _probe_truth(owner: Sequence[tuple[int, int]], labels: Mapping[int, str],
                 eyes: Mapping[int, list[Eye]]) -> tuple[list[int], list[int], list[int]]:
    """Per-eye truth, the frame each eye belongs to, and the frames in fold order.

    Only a frame with ONE face carries a truth an eye can be trained on: the label says
    "somebody in this shot has their eyes closed", and in a group photo that somebody is
    not identified. Such eyes get -1 — answered, never learned from — and so do the frames
    the owner could not read.
    """
    order = sorted({file_id for file_id, _ in owner})
    position = {file_id: i for i, file_id in enumerate(order)}
    single = {file_id for file_id in order if len(eyes.get(file_id, ())) <= 2}
    truth: list[int] = []
    groups: list[int] = []
    for file_id, _index in owner:
        label = labels[file_id]
        usable = file_id in single and label in (LABEL_OPEN, LABEL_CLOSED)
        truth.append((1 if label == LABEL_CLOSED else 0) if usable else -1)
        groups.append(position[file_id])
    return truth, groups, order


def _clip_crop_scorer(cfg):  # pragma: no cover — ML
    """p(closed) for a list of crops, from the CLIP the pipeline already runs.

    Built here rather than taken from `landmarks.clip_classifier` for one reason: that one
    is a classifier over PATHS, and the whole question is what CLIP sees on a crop that
    exists only in memory. The model, the weights and the softmax are the pipeline's own —
    only the source of the pixels differs.
    """
    import open_clip
    import torch

    from sorta.naming import naming_settings

    settings = naming_settings(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model, pretrained=settings.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(settings.clip_model)
    model.eval()
    with torch.no_grad():
        text = model.encode_text(tokenizer(list(CLIP_PROMPTS)).to(device))
        text /= text.norm(dim=-1, keepdim=True)

    def score(crops: Sequence, batch_size: int) -> list[float]:
        out: list[float] = []
        for chunk in batched(list(crops), batch_size):
            batch = torch.stack([preprocess(crop) for crop in chunk]).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
                probs = (100.0 * feats @ text.T).softmax(dim=-1)
            out.extend(float(p) for p in probs[:, 0].cpu().numpy())
        return out

    return score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="eyes_labels.json",
                    help='the worksheet {file_id: "open"|"closed"|"cannot"}')
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N frames")
    ap.add_argument("--max-edge", type=int, default=imaging.preview_max_edge(),
                    help="the preview side every variant is measured on")
    ap.add_argument("--min-iou", type=float, default=0.3,
                    help="overlap at which a stored faces.bbox counts as the same face "
                         "(a diagnostic — the measurement uses its own boxes)")
    ap.add_argument("--aggregate", choices=(*RULES, "both"), default="both",
                    help="how a frame with several faces answers")
    ap.add_argument("--clip-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
