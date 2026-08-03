"""F154: an object detector as a SECOND TIER OVER A QUERY, not a pass of its own.

The measurement this module exists because of (200 hand-labelled frames, 2026-08-02, the
same sample F122/F152 were scored on, detector confidence 0.5):

                         detector                  what it is up against
    ANIMALS     62% precision, 87% recall      the CLIP label: 71% / 33%
                                               a query:        67% / 67%@N, 80%@2N
    PEOPLE      42% precision, 96% recall      faces (F152): ~100% precision
    FOOD        20% precision, 15% recall      no filter exists; a query (F151) does it

    the price of a pass: 83.8 ms/frame -> 30.8 minutes over 22 096 photographs

**The detector earns its keep on exactly one slice of three**, and both halves of that
sentence are the feature:

* animals are where it wins, and clearly. 87% recall against the 33% the CLIP label
  reaches, while marking 21 frames out of 200 — this is what a detector is taken for. It
  sees the cat in the corner of the frame; CLIP compares the picture to a text AS A WHOLE
  and a cat that is not the subject of the shot is not what the picture is "of".
* people are where it loses outright. 42% precision against ~100% from faces: it finds a
  "person" where a person would not — backs in a crowd, figures in the distance, a hand.
  The people slice stays on the face boxes (F152) and this module has no person class.
* food is a failure of the label set rather than of the model. COCO has no `food`: it has
  a banana, a sandwich, a pizza. "A meal on a table" is not any of them, and 20% precision
  at 15% recall is what asking anyway produces. Food stays a query (F151).

Nobody may add the other two "while we are here" — the classes below are the boundary the
measurement drew, and `ANIMAL_CLASSES` says so again where the boundary lives.

WHY A CASCADE AND NOT A STAGE. A full pass is 31 minutes for a signal that helps one
slice. The candidates come from a query over the vectors the junk stage already stores
(F128) — the same shape as the animal cascade (F130) and the junk rescue (F140), both of
which have paid off:

    query over the embeddings   ->  ~2 000 candidates (free, 0.9 ms)
    the detector over those     ->  ~3 minutes instead of 31

Recall is then bounded by the query's recall at that depth (87% on the sample above), and
precision rises from the query's own (43%) to the detector's (62%).

This module is a leaf: the model, the classes, the query, the ranking and the cascade rule.
The pipeline half — which frames are candidates in a live index, what is written where and
what a repeated run skips — is `junk._DetectorPass`, next to every other pass of that
stage. The direction of the import follows: `junk` imports this, never the other way round.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .config import Config, DetectConfig, FeaturesConfig, detector_allowed

# The animal classes of COCO, by the 91-entry indexing torchvision's detection models use
# (ids 16-25, contiguous). THIS LIST IS THE BOUNDARY OF THE FEATURE, and it is short for a
# measured reason rather than for lack of time — see the module docstring for the three
# rows of the table. `person` (id 1) is not here: the detector reaches 42% precision on it
# against ~100% from the face boxes F152 already stores. `banana`, `sandwich`, `pizza` and
# the rest of COCO's food (ids 52-61) are not here either: they are objects, and the slice
# people ask for is a MEAL, which those labels miss at 20% precision.
#
# Adding either back is not a small edit — it is a claim that the measurement changed, and
# it needs its own table from `scripts/measure_detector.py` first.
ANIMAL_CLASSES: dict[int, str] = {
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
}
# The same boundary as a set of names — what a STORED box is checked against, since the
# table holds the class names and not COCO's ids.
ANIMAL_LABELS = frozenset(ANIMAL_CLASSES.values())

# Boxes weaker than this are not stored at all. It is a floor and NOT the decision
# threshold: `features.detector_threshold` decides what counts as an animal, and it is
# chosen from a table (scripts/measure_detector.py). Storing well below it is what makes
# re-choosing free — the same reason `frame_quality.pet_score` is written whether or not it
# reached its threshold. Below 0.05 a detector's output is noise by the hundred boxes.
STORE_FLOOR = 0.05

# The query that selects the candidates, over the CLASSIFICATION vectors (`clip_embeddings`
# — the space `naming.clip.*` produced, which is what those rows hold). Several short
# prompts and the best of them per frame, not one sentence: CLIP does single subjects well
# and compound phrases badly (F129), and a frame is a candidate if it looks like ANY of
# these. The list is deliberately about the animals COCO can then confirm.
ANIMAL_QUERY_PROMPTS: tuple[str, ...] = (
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a pet animal",
    "a photo of a wild animal",
    "a photo of a bird",
    "a photo of a horse",
)


@dataclass(frozen=True)
class Detection:
    """One box the detector returned: what it is, how sure it is, and where.

    The coordinates are the detector's own — pixels of the frame as it was given to the
    model, `(x1, y1, x2, y2)`. They are stored rather than dropped because there is
    nowhere else they could come from later, and a slice that wants to crop to the animal
    (or to say why a frame is in it) has to read them from somewhere.
    """

    label: str
    score: float
    box: tuple[float, float, float, float]


# path -> the boxes found on that frame. The real one is `torchvision_detector` below;
# every test injects its own, which is what keeps the suite free of model downloads (the
# rule the whole junk stage follows for CLIP, OCR and the VLM).
DetectFn = Callable[[str], Sequence[Detection]]


@dataclass(frozen=True)
class DetectorSettings:
    """Everything the cascade reads out of the config, resolved once per run.

    Same shape and same reason as `junk.GateSettings` / `junk.QualitySettings`: the
    measurement script drives the pipeline's own functions off this object, so a table it
    prints cannot describe a cascade the stage does not have.

    `enabled` is BOTH switches already ANDed together — `detect.enabled` (may a detector be
    loaded at all, the F145 rule) and `features.detector` (is the cascade wanted). A caller
    that reads one and forgets the other is exactly the failure F145 was written about, so
    there is one field and no way to ask for half of it.
    """

    enabled: bool
    model: str
    candidates: int
    threshold: float


def detector_settings(cfg: Config) -> DetectorSettings:
    """`features.detector*` + the `detect:` section of a config (or of a measurement)."""
    f = getattr(cfg, "features", None) or FeaturesConfig()
    d = getattr(cfg, "detect", None) or DetectConfig()
    return DetectorSettings(
        enabled=detector_allowed(cfg) and bool(getattr(f, "detector", False)),
        model=str(getattr(d, "model", DetectConfig.model)),
        candidates=int(getattr(f, "detector_candidates",
                               FeaturesConfig.detector_candidates)),
        threshold=float(getattr(f, "detector_threshold",
                                FeaturesConfig.detector_threshold)),
    )


def query_scores(vectors: dict[int, np.ndarray],
                 features: np.ndarray) -> dict[int, float]:
    """file_id -> how much this frame looks like ANY of the animal prompts.

    A dot product, because both sides are unit vectors (`junk.pack_embedding` normalizes
    what it stores and `junk.unit_rows` the prompt rows), so the number IS a cosine and
    nothing is normalized per frame. A vector of another width is not scored at all rather
    than scored across two spaces: a number computed that way looks exactly like a real
    one, which is the single thing a selection signal must never do.
    """
    if features.ndim != 2 or not features.size:
        return {}
    width = int(features.shape[1])
    return {file_id: float(np.max(features @ vec))
            for file_id, vec in vectors.items() if int(vec.size) == width}


def rank_candidates(vectors: dict[int, np.ndarray], features: np.ndarray,
                    depth: int) -> list[int]:
    """The `depth` frames the query likes best, best first — the detector's whole population.

    A RANKING AND NOT A THRESHOLD, and that is what `features.detector_candidates` means:
    the score orders frames against each other and says nothing in absolute terms (the
    reason `features.search_limit` is a sample size and not a cutoff, F129). Depth is what
    the feature costs — 2 000 frames at 83.8 ms is ~3 minutes — and what bounds its recall.

    Ties are broken by file_id, so a repeated run selects the same frames: a candidate list
    that reshuffles between runs cannot be measured, and measuring it is a condition of
    this feature.
    """
    scored = query_scores(vectors, features)
    if depth <= 0 or not scored:
        return []
    ids = sorted(scored)
    order = np.argsort(-np.asarray([scored[i] for i in ids], dtype=np.float32),
                       kind="stable")[:depth]
    return [ids[i] for i in order]


def animal_boxes(found: Sequence[Detection], threshold: float) -> list[Detection]:
    """The animal boxes at or above `threshold`, best first — people and food never here.

    The class filter is applied by the detector itself (see `torchvision_detector`), and
    repeated here as the last word on it: this is the one function the label is read off,
    and a caller handing it a `person` box must not be able to turn it into an animal.
    """
    return sorted((d for d in found
                   if d.label in ANIMAL_LABELS and d.score >= threshold),
                  key=lambda d: -d.score)


def best_animal(found: Sequence[Detection], threshold: float) -> Detection | None:
    """The highest-scoring animal box of a frame, or None if it holds none."""
    boxes = animal_boxes(found, threshold)
    return boxes[0] if boxes else None


def cascade_label(found: Detection | None, examined: bool, verified: bool,
                  previous: str | None, animal: str) -> str | None:
    """The animal label of one frame once the detector has had its say — the whole rule.

    The order of precedence, and each step is a measurement rather than a preference:

    * a frame the detector never examined (below the candidate depth, the toggle off, the
      model unavailable, an error on that one frame) keeps `previous` — the F130 cascade's
      own answer. A refusal is never read as "no animal": the fallback tier surviving the
      failure of the expensive one is the rule this whole stage is built on;
    * a frame the VLM has already answered about (`verified`) keeps `previous` too. That
      answer is to a question this detector cannot be asked — "is the animal alive, or is
      it a drawing, a plush toy, a print on a shirt" — and a box detector says `cat` to
      every one of those, which is the exact error F130 exists to remove;
    * otherwise the detector OVERRIDES THE CLIP LABEL, in both directions. An animal found
      where the score was below `features.pet_threshold` is labelled (this is the 87%
      recall against 33%), and a frame CLIP called an animal with nothing detected on it
      loses the label (this is the precision half).

    `animal` is the label value itself, passed in rather than repeated here: it belongs to
    `junk.PET_CLASS`, the column's one meaning (F122), and two spellings of one fact is how
    a consumer ends up with a slice that misses half its frames.

    This is the rule the STAGE writes with. The readers of the animal slice derive their
    own since F137 (`sorter.animal_auto_sql`, over `pet_score`/`pet_vlm`, so that a
    threshold moved in the config moves the slice without a run), and F160 wrote this tier
    into it — with the same order, the boxes re-read at the threshold in force now, and a
    case table run through both spellings so they cannot drift apart again. See the junk
    module docstring.
    """
    if not examined or verified:
        return previous
    return animal if found is not None else None


def pack_boxes(found: Sequence[Detection]) -> str:
    """The boxes of one frame as the stored JSON: `[[label, score, x1, y1, x2, y2], ...]`.

    A list of lists and not a list of objects: this column is read by a slice query and by
    a human debugging one, and the short form keeps the row small (a frame with a crowd of
    animals is a few hundred bytes). Rounded to what a detector's numbers actually mean —
    three decimals of a score and one of a pixel — so the same frame produces the same text.
    """
    return json.dumps([[d.label, round(float(d.score), 4),
                        *(round(float(v), 1) for v in d.box)]
                       for d in found], separators=(",", ":"))


def unpack_boxes(text: str | None) -> list[Detection]:
    """The stored JSON back into boxes; anything unreadable is no boxes at all.

    Lenient on purpose. This column is the incrementality marker's payload, and a row that
    cannot be parsed must cost the frame its stored answer (it is examined again next run),
    never the stage — the same "a broken row is not a reason to fail" rule
    `search.search` applies to a truncated vector.
    """
    try:
        rows = json.loads(text) if text else []
    except (TypeError, ValueError):
        return []
    out: list[Detection] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, list) or len(row) != 6:
            continue
        label, score, *box = row
        if not isinstance(label, str):
            continue
        try:
            out.append(Detection(label, float(score),
                                 (float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3]))))
        except (TypeError, ValueError):
            continue
    return out


def torchvision_detector(model_name: str,
                         floor: float = STORE_FLOOR) -> DetectFn:  # pragma: no cover — ML
    """The real detector: a torchvision COCO model, one frame per call, animals only.

    No new dependency — torchvision is installed for the CLIP side already, and only the
    COCO weights are downloaded. The model is resolved BY NAME through
    `torchvision.models.detection`, so `detect.model` can point at another checkpoint of
    the same family without a code change, and the weights come from the matching
    `*_Weights.DEFAULT` enum rather than from a hard-coded URL.

    The frame is taken from the shared preview cache (`imaging.decode_rgb_preview`), like
    every other model in this pipeline: a 1536px preview is what the detector sees, decoded
    once for the whole run and shared with the CLIP, OCR and VLM passes (F67).

    Boxes below `floor` are dropped here rather than stored: at 0.05 a detector is already
    returning noise by the hundred, and the threshold that DECIDES is applied later, over
    the stored scores (see `animal_boxes`).
    """
    import torch
    from torchvision.models import detection as tv_detection

    from . import imaging

    builder = getattr(tv_detection, model_name, None)
    if builder is None:
        raise ValueError(f"detect: неизвестная модель детектора {model_name!r}")
    weights_enum = getattr(tv_detection, f"{_weights_enum_name(model_name)}", None)
    weights = getattr(weights_enum, "DEFAULT", None) if weights_enum is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = builder(weights=weights).to(device)
    model.eval()

    def detect(path: str) -> list[Detection]:
        try:
            st = os.stat(path)
        except OSError:
            return []  # gone or unreadable: no boxes, the same "no signal" the rest gives
        image = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size)
        if image is None:
            return []
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
        with torch.no_grad():
            (result,) = model([tensor])
        return _animals_from(result, floor)

    return detect


def _weights_enum_name(model_name: str) -> str:  # pragma: no cover — ML
    """`fasterrcnn_mobilenet_v3_large_fpn` -> `FasterRCNN_MobileNet_V3_Large_FPN_Weights`.

    torchvision spells its weight enums in a case its model functions do not, and there is
    no lookup from one to the other — so the mapping is written out for the models this
    feature was measured with, and anything else falls back to pretrained=None (which is a
    detector that finds nothing, loudly, rather than a wrong checkpoint quietly).
    """
    known = {
        "fasterrcnn_mobilenet_v3_large_fpn":
            "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
        "fasterrcnn_mobilenet_v3_large_320_fpn":
            "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
        "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
        "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    }
    return known.get(model_name, "")


def _animals_from(result: Any, floor: float) -> list[Detection]:  # pragma: no cover — ML
    """One torchvision prediction dict -> the animal boxes of that frame.

    The class filter is applied HERE, at the model's edge, and not only at the label rule:
    a `person` box that never enters the process cannot be stored, cannot be counted and
    cannot be turned into an animal by a later reader. That is the boundary the measurement
    drew, and it is cheaper to keep at one point than to remember at three.
    """
    labels = result["labels"].tolist()
    scores = result["scores"].tolist()
    boxes = result["boxes"].tolist()
    found = []
    for label_id, score, box in zip(labels, scores, boxes):
        name = ANIMAL_CLASSES.get(int(label_id))
        if name is None or float(score) < floor:
            continue
        found.append(Detection(name, float(score),
                               (float(box[0]), float(box[1]),
                                float(box[2]), float(box[3]))))
    return sorted(found, key=lambda d: -d.score)
