"""F6 (Phase 5, FR-7): junk classification of canonical photos.

Contract: reads `files` (+`faces` as a signal), writes `media_class`, `frame_quality`,
`clip_embeddings`, `search_embeddings` and `detections`. Deletes and moves NOTHING — the
layout into _Unsorted/junk is the sorter's job, off this table.

THE FAST TIER, in the order the branches run (`clip_verdict`): an explicit
Screenshot_/«снимок экрана» name (F13, source='heuristic'), then a high document score
(F15 — BEFORE the camera/GPS veto, because a photographed document carries camera EXIF,
and never for a frame with faces), then the veto itself (camera EXIF, GPS or a detected
face: messengers strip EXIF, so a face is the third condition), then the three junk classes
over `naming.junk_threshold`. Conservative throughout: a real photo in the trash costs more
than junk left among the city folders.

F37-A hangs a text-density signal (easyocr, the fraction of the frame under text boxes) on
the document<->photo pair alone, and only for frames without faces: below `text_frac_min` a
document goes back to being a photo, above `text_frac_document` a photo CLIP scored low
becomes a document. source='ocr' means, and only means, that OCR moved the verdict.

F37-B adds the DEEP TIER: one VLM (Qwen2.5-VL) over the candidates the fast tier doubts,
answering personal_photo/document/product, opt-in and off by default. Screenshots stay the
fast tier's job — the deep tier has no such answer. Every model in this file is optional in
the strong sense: a factory that will not build is caught around the BUILD, logged, and the
run goes on with the verdicts the tier below it gave.

F68: incrementality runs on `media_class.tier`, not on `source`. `source` is WHAT decided
(heuristic | clip | ocr | vlm — user-facing, read by sorter.py), `tier` is WHICH TIER
walked the row (heuristic | clip | vlm). They do not coincide — the OCR gate rewrites
source, and the deep tier deliberately leaves clear photographs on 'clip' — and under the
old source-based marker both kinds of row were reclassified on EVERY run.

F48/F67: every decode here comes from the shared disk preview cache
(`imaging.decode_rgb_preview`), never from the original. Before F67 the second decode
inside OCR was 315 ms a frame, ~80% of the whole stage.

THE THREE MODEL PASSES ARE PIPELINED, all of them through `_vlm_labels`: `vlm.workers`
threads prepare frames (decode + the processor's preprocessing) while this thread generates
and writes. F101 built it for the deep tier off a profile that ruled batching out — ~0.6 s
of CPU then ~0.19 s of GPU per frame, strictly alternating, 0.84 cores of 24 busy, the card
at ~26% — and F206 moved the other two questions onto it after the run of 2026-08-05 priced
them apart:

    stage=classify phase=junk_vlm         7 951 frames    5 503 s   1.4 frames/s
    stage=junk     phase=junk_pets_vlm  }
    stage=junk     phase=junk_rescue_vlm }  4 281 frames  ~10 200 s  0.42

Same model, same one frame per call, three times the price — 116 minutes a run. NO VERDICT
MAY MOVE FOR IT: answers come back in the CANDIDATE order (a FIFO of futures, not whatever
finishes first), the model still sees one frame per call with the same prompt and the same
greedy decode, the writes still happen on this thread alone, and a frame whose preparation
fails keeps its cheap-tier answer and still steps the progress bar.
`tests/test_junk_asker_pipeline.py` prices the animal phase AGAINST the deep tier's phase
of the same run: seconds per frame are a statement about a machine, a ratio between two
phases asking one model one question is a statement about this stage.

The OCR half has a pool of its own (`_OcrPool`, F73): K threads, each with its OWN easyocr
Reader — a Reader holds a torch model and its buffers, the same reason F12.1 gives every
faces worker its own FaceAnalysis. Serially it ran at 4.27 files/s with every decode worker
idle. Only that middle phase leaves the caller's thread, so SQLite stays single-writer.

F95: the VLM weights are loaded by `naming.shared_vlm` and not by this module — the naming
stage runs the same model and two copies do not fit in VRAM (peak 20.5 GB).

F90: OCR runs on 28% of the frames and changes 2% of the verdicts (14:1), and
`text_rescue_docscore_min` is the number that decides that ratio. It is a decision for a
user in front of the table `scripts/measure_ocr_gate.py` prints, not something a worker
raises quietly — so the verdict and gate branches live in functions of their own
(`clip_verdict`, `ocr_gate_open`, `apply_text_frac`, over `gate_settings`) and the script
drives exactly those, or it would price a gate the pipeline does not have.

F100/F205: the stage names the phase it is in (CLASSIFY_PHASE_* below) through the optional
`progress.phase(name)` channel, and the three model passes have three names. A phase name
IS the unit a measurement is filed under (runlog.measurement_unit), so one name over three
prices that differ threefold prices none of them.

F113: the stage also fills `frame_quality`, and every signal there is taken with the
CHEAPEST TOOL THAT CAN ANSWER IT — a laplacian over the preview for sharpness, a prompt
group inside the CLIP call this stage already makes for animals, eyelid geometry for the
eyes (F179; the VLM that used to answer them is retired). The next tier up is paid for only
where the cheap one is not sure. This half keeps its OWN incrementality marker
(`frame_quality.source`), because the two go stale independently: switching `features.pets`
on changes no junk verdict, and a collection classified before the feature existed has no
quality rows at all.

F128/F141: the CLIP vector of every frame this stage looks at is KEPT (`clip_embeddings`)
where it used to be read for three scores and dropped, and behind `features.search_index` a
SECOND vector from a multilingual model is computed beside it (`search_embeddings`). The
first costs no model call at all — the vector comes out of the caching classifier that has
just scored the chunk. Both tables write the model into every row, a mismatch means
recompute rather than use, and both hold the F120 population.

F130/F154: the animal label is a CASCADE — CLIP selects widely, the VLM answers whether the
animal is alive, an object detector answers whether there is one at all. Each tier
overrides the one below it and falls back to it, never to "no animal". WHERE THE ANSWER IS
READ IS NOT HERE: since F137 the album, the "Animals" tab and the Overview counter derive
the label as they read, through `sorter.animal_auto_sql`. F154 shipped without that branch
and the gap was the worst kind — the stage ran, the boxes were in the database, and nothing
a user looks at moved — so `pet_label` here and `animal_auto_sql` there are now run through
one case table (`tests/test_detector_reaches_the_screen.py`).

F140: the search by words put memes and screenshots at the top of its results while all
19 753 rows it searches carry the verdict `photo` — this stage being wrong about ~4% of
what it calls a photograph, which nothing made visible until a query did. A zero-shot
margin over the vectors F128 already stores SELECTS those frames and does not judge them;
only the model's answer moves `media_class`. The same device as the F38 OCR gate, and
cheaper: a matmul over a table that already exists.

F186 RETIRED THE COMPARATIVE QUESTION (F132) — which frame of a near-duplicate group is the
one to keep. The owner labelled 111 groups BLIND on 2026-08-04, the frames shuffled and the
model's answer hidden:

    way in                agreement with the person    calls    seconds
    sharpness                       27%                  0         0
    arithmetic                      28%                  0         0
    a cascade                       28%                  0         0
    the model                       32%                115       451

    PICKING AT RANDOM               30.4%   (20 000 shuffles: 30.3%)

Nothing was bought to replace it, because a question no rule answers is not a question with
a cheaper answer somewhere. The MECHANISM stays in dedup.py (`group_keeper`,
`dedup.group_key`, `dedup.keeper_groups` and the sharpness ranking the Duplicates tab
shows), and `dedup_choice` is what it has always been: the user's own decision, which no
path of this stage has ever written.

F164: the two thread ceilings of the stage keep their values for two different reasons.
`vlm.workers` was measured and 4 is past the knee already — one frame's preparation uses
about seven cores since F105, so 6, 8 and 12 threads came back SLOWER on the live
collection (config.default_vlm_workers holds that table). `_DEFAULT_OCR_WORKERS_CAP` is
unmeasured on purpose: what it protects is VRAM, only a free card can price it, so the tool
ships and the number waits for the run that earns it.

F165: THE STAGE RUNS IN TWO HALVES, and `verdicts_only` is which one. The faces stage is
46% of a full run and used to walk 4 300 frames of 24 195 that this stage already knew were
screenshots, documents, memes or products — so the verdicts move ahead of it:

    index -> geo -> landmarks -> classify -> faces -> events -> junk -> phash

The split is by dependency and nothing else: `verdicts_only=True` runs everything that does
not read `frame_quality` (the fast pass, the deep tier, the stored vectors) and leaves the
rest behind faces. Swapping the two stages instead of splitting them would have switched
`face_sharpness` off silently on every first run. Both halves are the SAME function under
the same incrementality, so the second call reclassifies nothing and `sorta junk` alone
still does the whole thing.

THE ONE THING THAT DOES CHANGE WITH IT, written here because no test can show it and no log
line will mention it: the fast tier reads `has_faces` in four places (the F13 veto, the F15
document pass, the F38 OCR gate, the #14 VLM gate), and before the faces stage has ever run
there is nothing to read. On a FIRST run with `--faces` a frame CLIP calls a meme is no
longer vetoed by the face in it — which is exactly what a default run has always done
(faces are opt-in, F53) — and a frame this stage calls junk is a frame `faces` now skips,
so the veto cannot reach it afterwards. The brief accepted that trade for the 18% (F165,
«Оговорки», 2 and 3).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Generator, Sequence

import numpy as np
from PIL import Image

from . import accel, imaging
from .config import Config, FeaturesConfig, products_allowed, vlm_allowed
from .detect import (
    ANIMAL_QUERY_PROMPTS,
    STORE_FLOOR,
    Detection,
    DetectFn,
    DetectorSettings,
    animal_boxes,
    best_animal,
    cascade_label,
    detector_settings,
    pack_boxes,
    rank_candidates,
    torchvision_detector,
    unpack_boxes,
)

# F102 moved this resolver to config along with the knob it reads, but the measurement
# scripts import it from here, so the name stays re-exported.
from .config import resolve_vlm_workers  # noqa: F401
from .landmarks import CachingFeatureClassifier, Classifier, batched, clip_classifier
from .naming import (
    DEFAULT_VLM_MODEL,
    VLM_MAX_EDGE,
    NamingSettings,
    SplitVlm,
    naming_settings,
    shared_vlm,
    utcnow_iso,
)
from .progress import PhaseCB, ProgressCB
from .runlog import track_phases

_log = logging.getLogger(__name__)

# Classes in a fixed order; the prompts are curated. «document» was removed
# (brief F13): it fired on portraits/interiors — the main FP source.
_CLIP_CLASSES: tuple[tuple[str, str], ...] = (
    ("photo", "a photograph"),
    ("screenshot", "a screenshot of a phone or computer screen"),
    ("meme", "a meme image with text"),
)

_SCREENSHOT_NAME_RE = re.compile(
    r"^(screen[ _-]?shot|снимок[ _]экрана)", re.IGNORECASE)

# F29: a "floor" for verdict='photo' — a file in a Screenshots directory cannot stay an
# ordinary photo (see the override in classify).
_SCREENSHOT_DIRS = {"screenshots", "screenshot"}


def _in_screenshots_dir(path: str) -> bool:
    """True if any path segment is screenshots|screenshot (case-insensitive).

    Split on both separators: paths in the DB come with `\\` and with `/`, depending on
    the platform that indexed them.
    """
    return any(
        seg.lower() in _SCREENSHOT_DIRS for seg in re.split(r"[\\/]", path)
    )

# F15/F22: a separate CLIP run for documents, with a softmax group of its own. The
# anti-classes drain the mass that otherwise made travel photos of buildings with signs
# come out as receipt/paper/scan (F22); the score is the max over the DOCUMENT
# subclasses alone.
_DOC_ANTI_CLASSES: tuple[tuple[str, str], ...] = (
    ("photo", "a regular photograph of people, places or things"),
    ("building", "a photo of a building or house"),
    ("street", "an outdoor street scene"),
    ("storefront", "a storefront or shop sign"),
    ("street_signs", "a city street with signs"),
)
_DOC_POS_CLASSES: tuple[tuple[str, str], ...] = (
    ("receipt", "a photo of a receipt"),
    ("paper", "a photo of a paper document"),
    ("meter", "a photo of a utility meter or counter display"),
    ("scan", "a scanned document"),
)
_DOCUMENT_CLASSES: tuple[tuple[str, str], ...] = _DOC_ANTI_CLASSES + _DOC_POS_CLASSES
_N_DOC_ANTI = len(_DOC_ANTI_CLASSES)

# #14/V1: the same trick for "productness", and it is a CANDIDATE GATE and never a
# verdict — a high product_score buys a frame a VLM call, which is what keeps the deep
# tier off the whole collection.
_PROD_ANTI_CLASSES: tuple[tuple[str, str], ...] = (
    ("photo", "a personal photograph of people, places or pets"),
    ("scene", "an everyday life scene or travel photo"),
)
_PROD_POS_CLASSES: tuple[tuple[str, str], ...] = (
    ("product", "a product photo on a plain background"),
    ("catalog", "an e-commerce or online marketplace listing photo"),
    ("object", "an isolated single object photographed for sale"),
)
_PRODUCT_CLASSES: tuple[tuple[str, str], ...] = _PROD_ANTI_CLASSES + _PROD_POS_CLASSES
_N_PROD_ANTI = len(_PROD_ANTI_CLASSES)
# the "product" zone threshold for VLM candidates (>= -> the file goes to the VLM). Tuned on a run.
_DEFAULT_PRODUCT_CANDIDATE_MIN = 0.4

# F113: "is there a cat in this frame" is a question about an OBJECT, which is what CLIP
# does well — the CLIP failure measured in F110 (a beach 0.95 against a medical form 0.79)
# was about the PURPOSE of a frame, a different question. The arithmetic decides the rest:
# the same coverage through the VLM is 19 757 x 0.78 s = 4.3 hours, while these prompts
# ride along on frames CLIP is looking at anyway.
#
# APPENDED to the junk prompt list, never a pass of their own, and `_group_probs` protects
# the junk verdict from them: a renormalized slice of a softmax IS the softmax over that
# slice, so `naming.junk_threshold` does not move under prompts it was not measured with.
_PET_POS_CLASSES: tuple[tuple[str, str], ...] = (
    ("cat", "a photo of a cat"),
    ("dog", "a photo of a dog"),
    # F121: was "a photo of a pet animal at home", and a review of all 649 of its frames
    # found people and children in it — "at home" describes a SCENE. Naming the animals
    # keeps the class for what it is for: the pets that are neither a cat nor a dog.
    ("pet", "a photo of a rabbit, a hamster, a bird, a horse or another animal"),
)
# Anti-classes: without somewhere for the mass of a pet-less frame to go, every photo comes
# out as the most cat-like of the three prompts above. F120 measured the contamination the
# first two alone left — 45% of `dog` and 15% of `cat` were not photographs of an animal —
# and each line below answers one observed failure. Anti-classes and not a higher threshold
# on purpose: a drawn cat is a CONFIDENT cat to CLIP, so no threshold separates it.
_PET_ANTI_CLASSES: tuple[tuple[str, str], ...] = (
    # F121: the review found children in the general class, so the prompt names them.
    ("people", "a photo of a person, a child or a group of people, with no animal"),
    ("scene", "a photo of a place, a building or an object, with no animal in it"),
    # drawn cats in `cat`
    ("drawing", "a drawing, painting, cartoon or illustration of an animal"),
    # F121: the drawing prompt does not catch these and should not — a wallpaper of a cat
    # IS a photograph of a cat. The distinction wanted is "mine or somebody else's".
    ("stock", "a wallpaper, a stock photograph, a poster or a magazine picture"),
    # F121: the "puppies" frame — CLIP reads lettering and believes it over the picture.
    ("text", "a picture with large text, a caption or lettering written on it"),
    # F121: two plush dogs got through the previous wording; the toy has to be the SUBJECT
    # of the prompt rather than an adjective in it.
    ("toy", "a photo of a stuffed plush toy, a soft toy or a figurine of an animal"),
    # the hotdog in `dog`
    ("food", "a photo of food or a dish on a plate"),
    # game pictures and product shots near the threshold
    ("screen", "a screenshot, a user interface or a picture of a screen"),
    # people in fur coats coming out as `pet`
    ("clothing", "a photo of clothing, fur or a fabric texture"),
)
_PET_CLASSES: tuple[tuple[str, str], ...] = _PET_POS_CLASSES + _PET_ANTI_CLASSES
_N_PET_POS = len(_PET_POS_CLASSES)

# F122: what `frame_quality.pet` holds when the group fires. One value, because the
# measurement says the binary question is worth publishing and the three-way one is not.
PET_CLASS = "animal"

# F130: what `frame_quality.pet_vlm` holds. The errors left at 92% precision are drawn
# cats, plush toys, fur coats and a hotdog — CLIP compares a picture to a text as a whole
# and cannot tell a cat from a picture of a cat, while "alive, or a rendering?" is a
# question about the meaning of the scene.
#
# NULL is not one of these three: it means the question was not asked (below the candidate
# threshold, the check off, no model, an unreadable answer), and `pet_label` treats it as
# such rather than as a "no".
PET_VLM_REAL = "real"
PET_VLM_DEPICTION = "depiction"
PET_VLM_NONE = "none"

# Where each group sits in the prompt list of the single call (start, stop).
_JUNK_GROUP = (0, len(_CLIP_CLASSES))
_PET_GROUP = (len(_CLIP_CLASSES), len(_CLIP_CLASSES) + len(_PET_CLASSES))

# --- F140: the screenshots and receipts this stage took for photographs ----------------
#
# Deliberately NOT appended to the call above: the score is read off the STORED vector
# (F128), which is what makes it free and what lets it answer for a collection whose junk
# classification is already current. The junk classes are reused rather than re-worded
# (`_CLIP_PROMPT`) — a second wording of "a screenshot" would be a second definition of
# one — and what is added is what the review of the live collection found among the
# misclassified frames: a photographed screen, a page of text, a receipt.
_CLIP_PROMPT = dict(_CLIP_CLASSES)
_JUNK_RESCUE_POS_PROMPTS: tuple[str, ...] = (
    _CLIP_PROMPT["screenshot"],
    _CLIP_PROMPT["meme"],
    "a photo of a computer monitor or a phone screen",
    "a picture that is mostly text",
    "a photo of a receipt, a bill or a ticket",
)
# The other side of the subtraction. One prompt, and the same one the junk group calls
# `photo`: the score is a MARGIN over that class, so adding prompts here would move every
# number and the threshold measured against them with it.
_JUNK_RESCUE_NEG_PROMPTS: tuple[str, ...] = (_CLIP_PROMPT["photo"],)


def junk_rescue_prompts() -> list[str]:
    """The prompt list of the rescue score, POSITIVES FIRST — the order is the contract.

    One list because it is one text-encoder call, split back at
    `len(_JUNK_RESCUE_POS_PROMPTS)`. The measurement script builds its prompts through
    here too, or it would price a score the stage does not compute.
    """
    return list(_JUNK_RESCUE_POS_PROMPTS) + list(_JUNK_RESCUE_NEG_PROMPTS)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows of a text-feature matrix, L2-normalized — so a dot product is a cosine.

    Done again although the encoder normalizes, for the reason `pack_embedding` repeats
    it: a norm over a few rows is worth more than the trust. A zero row has no direction
    to preserve and is left as it is rather than divided by zero.
    """
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=m.copy(), where=norms > 0)


def junk_rescue_score(vec: np.ndarray, text_features: np.ndarray) -> float | None:
    """max(junk prompts) - max(photograph prompts) over one stored vector.

    A margin and not a probability: softmax would compress exactly the region the
    threshold lives in (the useful band is 0.02 wide), and these frames are the ones the
    softmax of the main call already called a photograph.

    None — the widths do not match (a vector of another model, a truncated blob). A number
    computed across two spaces looks exactly like a real score, which is the one thing a
    selection signal must not do; `search.search` drops such rows for the same reason.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    if text_features.ndim != 2 or v.size != text_features.shape[1]:
        return None
    sims = text_features @ v
    split = len(_JUNK_RESCUE_POS_PROMPTS)
    return float(sims[:split].max() - sims[split:].max())


def clip_prompts(pets: bool) -> list[str]:
    """The prompts of the ONE main CLIP call of the stage; pets appended when asked for.

    With pets off the list is byte-for-byte what it always was — same prompts, same order,
    same text-embedding cache key.
    """
    prompts = [prompt for _cls, prompt in _CLIP_CLASSES]
    if pets:
        prompts.extend(prompt for _cls, prompt in _PET_CLASSES)
    return prompts


def _group_probs(probs_row: np.ndarray, group: tuple[int, int]) -> np.ndarray:
    """The softmax over ONE prompt group, recovered from the row of the shared call.

    Renormalizing the slice gives exactly the probabilities a separate CLIP call over
    those prompts alone would produce, which is what lets one call serve two independent
    questions. A row of zeros (a frame that did not decode) has no mass to renormalize and
    comes back as it is: score 0, the "no signal" it has always meant.
    """
    part = probs_row[group[0]:group[1]]
    if len(part) == len(probs_row):
        return part  # nothing else in this softmax — already normalized, do not touch it
    total = float(part.sum())
    return part / total if total > 0 else part


def pet_verdict(probs_row: np.ndarray, threshold: float) -> tuple[str | None, float]:
    """(class, score) of the pet group -> the class is None below `threshold`.

    The score is stored either way: a threshold chosen from a distribution has to be
    re-choosable from the stored scores, without a new pass over the collection.

    F122: ONE class is stored, whichever positive prompt won. A labelled sample of 320
    frames priced the two halves apart — "is there an animal here" is right 92% of the
    time at 0.70, WHICH animal is what the review kept finding wrong (people landing in
    `dog`, a concert photo in the general class) — so the ensemble stays, being what the
    92% was measured on, and only its unreliable half stops being published.

    The three prompts are deliberately NOT collapsed into one: merging them would move the
    probability mass into a single class, raise every score and invalidate that threshold.
    """
    group = _group_probs(probs_row, _PET_GROUP)
    if not len(group):
        return None, 0.0  # pets are off — this call had no pet prompts in it
    positives = group[:_N_PET_POS]
    score = float(positives[int(np.argmax(positives))])
    return (PET_CLASS if score >= threshold else None), score


def pet_label(pet_vlm: str | None, pet_score: float | None, threshold: float, *,
              candidate_threshold: float | None = None,
              detected: bool | None = None) -> str | None:
    """The animal label of one frame — the whole cascade, in one place.

    The model OUTRANKS the score, which is why the cascade exists: a frame scored 0.95 and
    answered `depiction` is a plush toy, and no threshold over a CLIP score separates one
    from a dog (F120: a drawn cat is a confident cat). `pet_vlm IS NULL` — not asked, not
    understood, no model — falls back to the rule that ran before the check existed, never
    to a guess in either direction (brief item 3.2).

    THIS FUNCTION AND `sorter.animal_auto_sql` ARE ONE RULE IN TWO SPELLINGS — this one
    labels a single frame, that one answers "which files" over a whole index — so every
    new source of the label has to reach both. F160 found the detector reaching only one,
    and `tests/test_detector_reaches_the_screen.py` now runs both over one case table.

    The two keyword arguments are what a READER knows and the stage does not:

    `detected` is the F154 tier — True: examined and an animal found at or above
    `features.detector_threshold`; False: examined and none found; None: never examined
    (below the candidate depth, off, no model, an error, or boxes from another detector).
    None falls through to the score, never to "no animal", and the VLM answer stays above
    it: a box detector cannot be asked whether the cat it sees is alive.

    `candidate_threshold` is the F137 gate on a STORED answer — it counts only for a frame
    the current `features.pet_candidate_threshold` would still show the model. None: the
    caller has just asked, and there is nothing to re-gate.
    """
    if pet_vlm is not None and (candidate_threshold is None
                                or (pet_score is not None
                                    and pet_score >= candidate_threshold)):
        return PET_CLASS if pet_vlm == PET_VLM_REAL else None
    if detected is not None:
        return PET_CLASS if detected else None
    if pet_score is None:
        return None
    return PET_CLASS if pet_score >= threshold else None

# F37 (Phase A): defaults for naming.text_frac_min/text_frac_document, which are not typed
# in NamingConfig (the getattr pattern this file uses for them).
# F38 lowered text_frac_document 0.35 -> 0.15 on real data: a document at an angle gave
# text_frac=0.247, scenes 0.0-0.002 — a large margin either side of it.
_DEFAULT_TEXT_FRAC_MIN = 0.08
_DEFAULT_TEXT_FRAC_DOCUMENT = 0.15

# F38: the OCR rescue (verdict='photo' -> 'document') runs only where the document-CLIP
# already doubts (doc_score in 0.3..document_threshold) — clear scenes never pay for OCR.
#
# F164 WAS ASKED TO SWEEP THIS AND DID NOT, which is worth as much as a table would be. A
# sweep needs BOTH columns: how many frames a threshold cuts, and how many documents go
# into the city folders with them. The first is a CLIP pass, the second an OCR pass over
# everything the lowest threshold gates — and with the card occupied (see
# _DEFAULT_OCR_WORKERS_CAP) only the first was affordable. A coverage column with no
# benefit column is exactly the table that raises a threshold for the wrong reason: F38
# raised this one because a document among the holiday photographs is the expensive error.
# `scripts/measure_ocr_gate.py` prints both columns plus `--probe-below`, and the rule is
# the module docstring's: this constant is a user's decision in front of that output.
_DEFAULT_TEXT_RESCUE_DOCSCORE_MIN = 0.3

# F38: the frame is shrunk before reader.detect() — a full-size decode is 1.2-3.2 s/frame
# on large photos, ~1280px gives a x3-10 speedup.
_DEFAULT_TEXT_FRAC_DOWNSCALE_PX = 1280

TextFracDetector = Callable[[str, int | None, int | None], float | None]
# F73: builds the detector of ONE worker thread — an easyocr Reader is not thread-safe, so
# the pool takes a factory rather than a ready detector.
TextFracDetectorFactory = Callable[[], TextFracDetector]
# (file_id, path, width, height) — one OCR job. file_id keys the result back to the row:
# the pool does not preserve the input order.
OcrJob = tuple[int, str, int | None, int | None]

# F73: the default ceiling for naming.ocr_workers, and it is UNMEASURED ON PURPOSE. Each
# worker keeps its own Reader, i.e. its own copy of the detector on the card, so what this
# number protects is VRAM and only a run on the GPU profile can price it — a CPU run
# prices a different machine, and a pool that cannot build its N-th Reader quietly shrinks
# to the ones it built (see _OcrPool), so such a row measures a smaller pool than its label
# says. The card was occupied throughout F164, which is when the question was asked.
#
# What is known says the ceiling is low rather than right: `junk_ocr` is the most expensive
# phase of the stage (614,6 s over 6 793 frames on the live run of 2026-08-03 — 90,5 ms a
# frame, 15% of the run), and F73 measured x3,7 from 1 to 4 workers, still almost linear
# where it was capped. So the tool ships and the number waits for a free card:
#
#     python scripts/measure_ocr_workers.py --sample 500 --workers 1 4 6 8
#
# It prints ms/frame, the speedup, the Readers actually built and the VRAM peak, and says
# in one line whether the ceiling moves: at least x1,15 with every Reader built. Lowering
# stays a value in `naming.ocr_workers`, which nothing here caps.
_DEFAULT_OCR_WORKERS_CAP = 4


def resolve_ocr_workers(raw: dict | None) -> int:
    """How many OCR threads run in parallel — `naming.ocr_workers` in config.yaml.

    Read straight out of `cfg.raw`, the way hashing.resolve_workers reads `index.workers`.
    Absent / 0 / negative / garbage -> min(4, cpu_count); see _DEFAULT_OCR_WORKERS_CAP on
    why that is low.
    """
    default = min(_DEFAULT_OCR_WORKERS_CAP, os.cpu_count() or 1)
    workers = ((raw or {}).get("naming") or {}).get("ocr_workers")
    if workers is None:
        return default
    try:
        n = int(workers)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


class _OcrPool:
    """text_frac over a pool of worker threads, one own detector per thread (F73).

    Workers outlive a chunk and pull from a shared queue. Each builds its OWN detector on
    its first job — lazily, thread-local, reused for every later chunk, because loading an
    easyocr Reader costs far more than the detection it then does.

    VRAM degradation: a worker that cannot build its detector (typically no memory for the
    second and further Readers) does not kill the stage — the pool shrinks to the detectors
    actually created, the job goes back to a surviving worker, and the reason is LOGGED
    rather than swallowed (the F37-B lesson: a silent refusal is a refusal nobody can
    price). With not one detector buildable, text_frac() re-raises the build error.

    Results come back on the caller's thread; nothing here touches SQLite.
    """

    def __init__(self, factory: TextFracDetectorFactory, workers: int) -> None:
        self._factory = factory
        self._workers = max(1, workers)
        self._local = threading.local()
        self._lock = threading.Lock()   # guards the counters and the batch state
        self._built = 0                 # detectors created (reserved slots included)
        self._limit = self._workers     # shrunk on a build failure, never below 1
        self._error: BaseException | None = None   # the last build failure
        self._jobs: Queue[OcrJob | None] = Queue()
        self._threads: list[threading.Thread] = []
        self._live = 0                  # worker threads still in the pool
        self._results: dict[int, float | None] = {}
        self._left = 0                  # jobs of the current batch not finished yet
        self._batch_done = threading.Event()

    @property
    def detectors_built(self) -> int:
        """Detectors created over the whole run (<= workers) — for the stage log."""
        return self._built

    def text_frac(self, jobs: list[OcrJob]) -> dict[int, float | None]:
        """text_frac for `jobs`, keyed by file_id; returns on the caller's thread.

        A detector error on one frame is None for that file_id — "no signal", the verdict
        is left alone — and does not touch its neighbours. A missing file_id means the
        same.
        """
        if not jobs:
            return {}
        results = self._serial(jobs) if self._workers == 1 else self._parallel(jobs)
        if len(results) < len(jobs) and self._error is not None:
            # Not one detector alive: OCR is unavailable for the whole stage. Fail
            # loudly instead of quietly dropping the signal for every frame.
            raise self._error
        return results

    def close(self) -> None:
        """Stop the workers (their detectors die with the threads)."""
        for _ in self._threads:
            self._jobs.put(None)
        for t in self._threads:
            t.join()
        self._threads.clear()
        self._live = 0
        while not self._jobs.empty():  # sentinels of workers that had already retired
            self._jobs.get_nowait()

    def _detector(self) -> TextFracDetector | None:
        """The calling thread's detector, built on first use; None -> leave the pool.

        The slot is reserved under the lock but the factory runs outside it: several
        Readers may load in parallel, and serializing that would only delay the stage.
        """
        det: TextFracDetector | None = getattr(self._local, "det", None)
        if det is not None:
            return det
        with self._lock:
            if self._built >= self._limit:
                return None  # the pool is already at its (shrunk) size
            self._built += 1
        try:
            det = self._factory()
        except Exception as exc:  # noqa: BLE001 — degrade the pool, do not kill the stage
            with self._lock:
                self._built -= 1
                self._error = exc
                self._limit = max(1, self._built)
                limit = self._limit
            _log.warning(
                "junk: OCR-детектор не построился (%s) — пул уменьшен до %d воркер(ов)",
                exc, limit)
            return None
        self._local.det = det
        return det

    def _one(self, det: TextFracDetector, job: OcrJob) -> float | None:
        _fid, path, width, height = job
        try:
            return det(path, width, height)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not break the stage
            _log.warning("junk: OCR не удался для %s: %s", path, exc)
            return None

    def _serial(self, jobs: list[OcrJob]) -> dict[int, float | None]:
        """workers == 1: everything on the caller's thread — the pre-F73 path."""
        det = self._detector()
        if det is None:
            return {}
        return {job[0]: self._one(det, job) for job in jobs}

    def _parallel(self, jobs: list[OcrJob]) -> dict[int, float | None]:
        self._start()
        with self._lock:
            self._results = {}
            self._left = len(jobs)
            self._batch_done.clear()
        for job in jobs:
            self._jobs.put(job)
        self._batch_done.wait()
        with self._lock:
            results, left = self._results, self._left
        if left:  # the pool died mid-batch — drop the leftovers, text_frac() reports it
            while not self._jobs.empty():
                self._jobs.get_nowait()
        return results

    def _start(self) -> None:
        """Bring the worker set up to the current (possibly shrunk) pool size."""
        with self._lock:
            missing = self._limit - self._live
            self._live += max(0, missing)
        for _ in range(max(0, missing)):
            t = threading.Thread(target=self._run, name="sorta-ocr", daemon=True)
            self._threads.append(t)
            t.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:  # close() — a shutdown sentinel per started thread
                with self._lock:
                    self._live -= 1
                return
            det = self._detector()
            if det is None:
                # No detector for this thread (VRAM): hand the job back and leave the
                # pool — the surviving workers finish the batch.
                self._jobs.put(job)
                self._retire()
                return
            frac = self._one(det, job)
            with self._lock:
                self._results[job[0]] = frac
                self._left -= 1
                if self._left == 0:
                    self._batch_done.set()

    def _retire(self) -> None:
        """This worker leaves the pool; the last one to go unblocks the batch."""
        with self._lock:
            self._live -= 1
            if self._live == 0:
                self._batch_done.set()


def _resolve_detector_factory(
    cfg: Config, text_detector: TextFracDetector | None,
) -> TextFracDetectorFactory:
    """The factory the OCR pool builds its per-thread detectors with (F73).

    An injected `text_detector` is handed to every worker as it is — how it copes with
    threads is then the caller's business. Otherwise each worker builds its own easyocr
    detector.
    """
    if text_detector is not None:
        return lambda: text_detector
    downscale_px = int(
        getattr(cfg.naming, "text_frac_downscale_px", _DEFAULT_TEXT_FRAC_DOWNSCALE_PX))
    return lambda: easyocr_text_frac_detector(downscale_px)  # pragma: no cover — ML


def _document_score(probs_row: np.ndarray) -> float:
    """Max probability among the document subclasses (without the anti-classes)."""
    return float(np.max(probs_row[_N_DOC_ANTI:]))


def _product_score(probs_row: np.ndarray) -> float:
    """Max probability among the product subclasses (without the personal-photo anti-classes)."""
    return float(np.max(probs_row[_N_PROD_ANTI:]))


def _polygon_area(points: list) -> float:
    """Polygon area by the shoelace formula — easyocr boxes of slanted text are not
    rectangles."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def easyocr_text_frac_detector(
    maxpx: int = _DEFAULT_TEXT_FRAC_DOWNSCALE_PX,
) -> TextFracDetector:  # pragma: no cover — ML, smoke test
    """easyocr (the CRAFT detector) — the fraction of frame area under text boxes.

    Lazy-import, and the Reader is built once for the whole classify() run.

    F38: decoded through imaging and NOT by `reader.detect(path)` — cv2 silently fails to
    read non-ASCII paths and HEIC, and those frames dropped out of the OCR signal
    altogether. The box area is relative to the downscaled frame.
    """
    import easyocr

    from .diagnostics import warn_if_gpu_mismatch

    # F63: easyocr(gpu=True) falls back to the CPU on a CPU-only torch build, and
    # verbose=False below hides easyocr's own notice about it — so it is surfaced here.
    warn_if_gpu_mismatch()
    # verbose=False: the model-download progress bar draws a █, which the Windows cp1251
    # console cannot encode (UnicodeEncodeError). The download itself is unaffected.
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def text_frac(path: str, width: int | None, height: int | None) -> float | None:
        # F67: the frame comes from the shared preview cache, which is why the F48
        # aggressive draft margin is gone from this path — a 1536px preview leaves draft
        # nothing to save. mtime/size come from a local stat, so the TextFracDetector
        # signature stays as it is.
        try:
            st = os.stat(path)
        except OSError:
            return None  # vanished/unreadable file — same contract as a decode error
        img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=maxpx)
        if img is None:
            return None  # could not decode (corrupt/unrecognized file)
        # detect() and not readtext(): the areas are all the density needs, the
        # recognition model is never loaded, and the recognition path fails on degenerate
        # crops (cv2.resize !ssize.empty).
        try:
            horizontal, free = reader.detect(np.asarray(img))
        except Exception as exc:  # noqa: BLE001 — one bad frame must not break the stage
            _log.warning("junk: детекция текста не удалась для %s: %s", path, exc)
            return None
        area = 0.0
        for box in (horizontal[0] if horizontal else []):
            if len(box) == 4:  # [x_min, x_max, y_min, y_max]
                x_min, x_max, y_min, y_max = box
                area += max(0.0, float(x_max - x_min)) * max(0.0, float(y_max - y_min))
        for poly in (free[0] if free else []):
            if len(poly) >= 3:  # the quadrilateral of slanted text
                area += _polygon_area(poly)
        img_w, img_h = img.size
        return min(1.0, area / (float(img_w) * float(img_h)))

    return text_frac


# F37 (Phase B): VLM 3-way classify_media(path) -> label, mapped to a verdict below. An
# unrecognized answer is 'personal_photo' — the rule everywhere in this file: letting a
# document through as a photo costs less than losing a real photo.
VlmClassifyFn = Callable[[str], str]

# F101: the label a frame gets without the model being asked at all — it is gone from
# disk, or it did not decode. Conservative by the same rule.
_VLM_FALLBACK_LABEL = "personal_photo"

_VLM_LABEL_TO_VERDICT: dict[str, str] = {
    "personal_photo": "photo",
    "document": "document",
    "product": "product",
}

# F95/F102: the model name and its input size describe the MODEL and live in the `vlm:`
# config section; these two are only the defaults for a caller with no config in hand.
_DEFAULT_VLM_MODEL = DEFAULT_VLM_MODEL
_DEFAULT_VLM_MAX_EDGE = VLM_MAX_EDGE

# One label is one short word — a longer budget only buys the model room to explain
# itself, which the parser below would then have to wade through.
_VLM_MAX_NEW_TOKENS = 8

# F196: the `product` line names an item HELD IN A HAND. The old wording narrowed itself —
# `isolated object, catalog shot` — and a hand holding the thing fell into `everyday life`.
# The owner labelled 20 misses against a list of reasons fixed BEFORE the frames were
# opened: `narrow` 17 (85%), `borderline` 3 (15%), `feature_missing` 0, `other` 0.
#
# What the wider question buys and costs, measured on 2026-08-05 over 733 frames the model
# had ALREADY been asked about (a prompt edit does not move the gate, so the selection
# layer was left out), counted by layer and weighted by population:
#
#     wording                precision   recall   marked
#     narrow (before)            78%       80%    ~2 107
#     this one (after)           75%       94%    ~2 604
#
# ~290 more products found for ~190 frames in the product folder that do not belong there.
# Whoever clarifies this wording next needs both halves of that: the previous clarification
# WAS measured, and three points of precision are what it spent. `personal_photo` and
# `document` are word for word what they were — the measurement priced ONE edit.
#
# TRAP: an edit here invalidates nothing by itself. The deep tier's marker is
# `media_class.tier = 'vlm'` (F68) and carries no fingerprint of the question, unlike
# `frame_quality.source` (F120), so a collection keeps the verdicts the old wording
# produced until a fast-tier run moves every marker to `clip` and a `--deep` run asks
# again: ~6 901 candidates ≈ 90 minutes, once.
_VLM_PROMPT = (
    "Classify this image into exactly one category: personal_photo, document, "
    "or product.\n"
    "personal_photo = a personal/casual photograph of people, places, pets or "
    "everyday life.\n"
    "document = a photographed or scanned document, receipt, ID card, form, or "
    "other text-heavy paper.\n"
    "product = an item photographed to show the item itself — for sale, for a "
    "listing, or to show someone. This includes an item held in a hand or held "
    "up to the camera: a hand in the frame does not make it a personal photo.\n"
    "Answer with exactly one word: personal_photo, document, or product."
)


@dataclass(frozen=True)
class PreparedFrame:
    """What the CPU half of the deep tier produces for one frame (F101).

    Either model `inputs` or a ready `label`: a frame that vanished or would not decode
    never reaches the model, and carrying its answer through the pipeline keeps the GPU
    half free of file-system branches.
    """
    inputs: Any = None
    label: str | None = None


@dataclass(frozen=True)
class SplitVlmClassifier:
    """classify_media(path) as its CPU half and its GPU half (F101).

    It IS a VlmClassifyFn — calling it runs both halves in turn, which is the serial
    classifier — so nothing that knows only the old interface changes. The deep tier
    checks for this type to decide whether a pass can be pipelined at all.
    """
    prepare: Callable[[str], PreparedFrame]
    classify_prepared: Callable[[PreparedFrame], str]

    def __call__(self, path: str) -> str:
        return self.classify_prepared(self.prepare(path))


def _vlm_label(answer: str) -> str:
    """The model's answer -> one of the three labels; anything else -> personal_photo."""
    lowered = answer.lower()
    for label in ("personal_photo", "document", "product"):
        if label in lowered:
            return label
    return _VLM_FALLBACK_LABEL


def vlm_classifier_from(describe: Callable[[Sequence[Image.Image], str, int], str],
                        max_edge: int = _DEFAULT_VLM_MAX_EDGE) -> VlmClassifyFn:
    """The stage's classifier over an ALREADY LOADED runtime — the halves included.

    Everything that belongs to this stage lives here: the prompt, the decode (through the
    shared preview cache, Unicode/HEIC-safe — the F38 lesson) and the parsing.

    F101: a runtime that offers its halves (naming.SplitVlm) gets a classifier that does
    too — `prepare` is the CPU part of a frame, which `_vlm_labels` moves off this thread,
    `classify_prepared` is the GPU part plus the parsing. Without them, the serial
    classifier, unchanged.
    """
    split = describe if isinstance(describe, SplitVlm) else None

    def decode(path: str) -> Image.Image | None:
        try:
            st = os.stat(path)
        except OSError:
            return None  # unreadable — the caller answers conservatively
        return imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge)

    if split is None:
        def classify_media(path: str) -> str:
            img = decode(path)
            if img is None:
                return _VLM_FALLBACK_LABEL
            return _vlm_label(describe([img], _VLM_PROMPT, _VLM_MAX_NEW_TOKENS))

        return classify_media

    def prepare(path: str) -> PreparedFrame:
        img = decode(path)
        if img is None:
            return PreparedFrame(label=_VLM_FALLBACK_LABEL)
        return PreparedFrame(inputs=split.prepare([img], _VLM_PROMPT))

    def classify_prepared(prepared: PreparedFrame) -> str:
        if prepared.label is not None:
            return prepared.label
        return _vlm_label(split.generate(prepared.inputs, _VLM_MAX_NEW_TOKENS))

    return SplitVlmClassifier(prepare=prepare, classify_prepared=classify_prepared)


def qwen_vlm_classifier(
    model_name: str = _DEFAULT_VLM_MODEL,
    max_edge: int = _DEFAULT_VLM_MAX_EDGE,
) -> VlmClassifyFn:  # pragma: no cover — ML, smoke test
    """The real VLM classifier (Qwen2.5-VL via transformers).

    F95: the weights come from naming.shared_vlm — ONE runtime per model name for the
    whole process. The load is lazy and still fails only when the classifier is actually
    built, which classify() wraps for its graceful fallback to the fast tier.
    """
    return vlm_classifier_from(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_classifier_factory(max_edge: int) -> Callable[[str], VlmClassifyFn]:
    """The default `vlm_classifier_factory` of classify(), carrying `vlm.max_edge` (F102).

    The interface stays (model_name) -> classifier and the input size travels in the
    closure: widening it would make every injected test factory carry a number it does
    not use.
    """
    return lambda model_name: qwen_vlm_classifier(model_name, max_edge=max_edge)


def _vlm_labels(vlm_fn: VlmClassifyFn, paths: list[str],
                workers: int) -> Generator[str | BaseException, None, None]:
    """Labels for `paths` IN INPUT ORDER, pipelined when that is possible (F101).

    One item per path: the label, or the exception the classifier raised on that frame
    (the caller logs it and keeps the fast verdict).

    The pipeline needs both halves from the runtime (SplitVlmClassifier) and more than one
    worker; anything else — an injected test classifier, a runtime without halves,
    vlm_workers=1 — takes the serial path, which is the pre-F101 loop verbatim.

    F206: all three questions of the stage come through here, because a second copy would
    be a second place for the order to go wrong. What an item MEANS is the caller's
    business: the tier reads a label, the askers read a raw answer.
    """
    split = vlm_fn if isinstance(vlm_fn, SplitVlmClassifier) else None
    if split is None or workers < 2:
        for path in paths:
            try:
                yield vlm_fn(path)
            except Exception as exc:  # noqa: BLE001 — one bad frame is the caller's business
                yield exc
    else:
        yield from _vlm_labels_pipelined(split, paths, workers)


def _vlm_labels_pipelined(split: SplitVlmClassifier, paths: list[str],
                          workers: int) -> Generator[str | BaseException, None, None]:
    """`workers` threads preparing frames while this thread runs the model on them.

    A FIFO of futures, not "first finished wins": the frame at the head is the next one
    yielded, so the output order is the input order however the preparations interleave.
    The GPU half runs HERE, on the consumer's thread — several streams of generate() would
    only queue inside the driver and cost VRAM.

    The window (2 per worker) is the RAM bound: at most that many preprocessed frames
    exist at once, and they are CPU tensors (naming.qwen_runtime), so the VRAM peak is one
    frame's inputs — what it was when the pass was serial.
    """
    from collections import deque
    from concurrent.futures import Future, ThreadPoolExecutor

    window = workers * 2
    remaining = iter(paths)
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="sorta-vlm") as pool:
        pending: deque[Future[PreparedFrame]] = deque()

        def fill() -> None:
            """Top the preparation queue up (this thread only — it owns the iterator)."""
            while len(pending) < window:
                path = next(remaining, None)
                if path is None:
                    return
                pending.append(pool.submit(split.prepare, path))

        fill()
        while pending:
            future = pending.popleft()
            result: str | BaseException
            try:
                result = split.classify_prepared(future.result())
            except Exception as exc:  # noqa: BLE001 — one bad frame must not end the pass
                result = exc
            fill()  # refill BEFORE yielding: the workers keep going while the caller writes
            yield result


def heuristic_verdict(
    path: str, width: int | None, height: int | None,
    camera_make: str | None, camera_model: str | None,
) -> str | None:
    """A screenshot candidate without ML; None — the heuristic is silent (= photo).

    The only signal (brief F13): an explicit Screenshot_/"снимок экрана" name. Screen
    ratio (3:4/4:3) and messenger-name→meme were REMOVED — they were the main FP source
    on real family photos.
    """
    if camera_make or camera_model:
        return None  # shot with a camera — not junk
    name = Path(path).name
    if _SCREENSHOT_NAME_RE.match(name):
        return "screenshot"
    return None


def _is_real_photo(row: sqlite3.Row) -> bool:
    """Camera EXIF/GPS or a detected face — a veto against a CLIP verdict.

    Messengers strip EXIF from forwarded photos, so camera/GPS alone do not protect real
    photos without metadata (brief F13); a face is an equally reliable "not a
    document/meme/screenshot", which is why it is the third condition.
    """
    return bool(
        row["camera_make"] or row["camera_model"]
        or row["gps_lat"] is not None or row["has_faces"]
    )


# F90: the fast-tier verdict and the OCR gate, lifted out of the classify() loop so that
# scripts/measure_ocr_gate.py can sweep the real decision. A second copy of these three
# branches in the script would drift from this one and price a gate the pipeline does not
# have.


@dataclass(frozen=True)
class GateSettings:
    """The thresholds the CLIP verdict and the OCR gate/rescue are built from.

    Read through getattr with the module defaults (see the F37/F38 constants above): the
    fields appeared in NamingConfig later than the code reading them.
    """
    junk_threshold: float
    document_threshold: float
    text_frac_min: float
    text_frac_document: float
    text_rescue_docscore_min: float


def gate_settings(cfg: Config) -> GateSettings:
    """The gate thresholds of a config, resolved once per run (or per measurement)."""
    s = naming_settings(cfg)
    return GateSettings(
        junk_threshold=float(s.junk_threshold),
        document_threshold=float(s.document_threshold),
        text_frac_min=float(getattr(s, "text_frac_min", _DEFAULT_TEXT_FRAC_MIN)),
        text_frac_document=float(
            getattr(s, "text_frac_document", _DEFAULT_TEXT_FRAC_DOCUMENT)),
        text_rescue_docscore_min=float(
            getattr(s, "text_rescue_docscore_min", _DEFAULT_TEXT_RESCUE_DOCSCORE_MIN)),
    )


def clip_verdict(best_class: str, best_score: float, heuristic: str | None,
                 doc_score: float | None, real_photo: bool,
                 g: GateSettings) -> tuple[str, float]:
    """The verdict of one frame BEFORE the OCR signal -> (verdict, score).

    The order of the branches is the contract, not a detail — see the module docstring
    for why each one sits where it does. `doc_score` is None for frames with faces: the
    document pass is not run for them at all.
    """
    if heuristic == "screenshot":
        return "screenshot", best_score
    if doc_score is not None and doc_score >= g.document_threshold:
        return "document", doc_score
    if real_photo:
        return "photo", best_score
    if best_score >= g.junk_threshold:
        return best_class, best_score
    return heuristic or "photo", best_score


def ocr_gate_open(has_faces: bool, verdict: str, doc_score: float,
                  rescue_docscore_min: float) -> bool:
    """Does this frame cost an OCR call? (F37 Phase A + the F38 doc-score gate.)

    `rescue_docscore_min` is a parameter and not a field of GateSettings because F90
    sweeps exactly this number over a grid. The FP gate (verdict=='document') is
    deliberately NOT limited by it: there are few documents anyway, and letting one
    through is the expensive error.
    """
    return not has_faces and (
        verdict == "document"
        or (verdict == "photo" and doc_score >= rescue_docscore_min)
    )


def apply_text_frac(verdict: str, score: float, text_frac: float | None,
                    g: GateSettings) -> tuple[str, float, str]:
    """The OCR signal on top of the CLIP verdict -> (verdict, score, source).

    text_frac None — "no signal" (the gate stayed shut, the frame did not decode, the
    detector failed on it): the verdict is left exactly as CLIP left it and source
    stays 'clip'. source == 'ocr' means, and only means, that OCR changed the verdict.
    """
    if text_frac is not None:
        if verdict == "document" and text_frac < g.text_frac_min:
            return "photo", text_frac, "ocr"        # a beach, not a document
        if verdict == "photo" and text_frac >= g.text_frac_document:
            return "document", text_frac, "ocr"     # dense text, whatever CLIP scored
    return verdict, score, "clip"


# --- F113: the frame-quality cascade ------------------------------------------------
#
# Three questions, three prices: sharpness is a laplacian over the preview every other
# stage has already paid for (no toggle, written always), pets are a prompt group inside
# the CLIP call above (`features.pets`), and what is left is a VLM at ~0.78 s a frame,
# asked only about frames the cheap tiers did not settle. The band that selects them is the
# F109 result put to use — the least confident 30% of frames kept 98.2% of the findings.

# F155: one box of the `faces` table, as written there — (x1, y1, x2, y2) in pixels of the
# FULL original frame, after its EXIF orientation has been applied (that stage detects on a
# rotated full-resolution decode, see faces._decode_for_faces).
FaceBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceBoxes:
    """The faces of one frame, plus the frame their coordinates are measured in.

    `long_edge` is what makes the boxes usable at all: they are written in pixels of the
    ORIGINAL, and the laplacian is taken over a preview that is a small copy of it, so
    without the size of the original there is no way back from one to the other. It is the
    longer side (max of `files.width/height`) rather than the pair, because a thumbnail
    scales both axes by one factor and the longer side is the axis that factor is set by —
    and because the longer side is the one quantity of the two that an EXIF rotation
    cannot swap.

    Empty is the ordinary case, not an error: two thirds of a collection have no face, and
    a frame the faces stage has never seen is empty here in exactly the same way.
    """
    boxes: tuple[FaceBox, ...] = ()
    long_edge: float = 0.0

    @property
    def usable(self) -> bool:
        """Are there boxes AND a scale to read them with?"""
        return bool(self.boxes) and self.long_edge > 0


NO_FACES = FaceBoxes()


@dataclass(frozen=True)
class Sharpness:
    """What ONE decode of a frame's preview yields (F155, F179).

    The three numbers are returned together because they are measured together: the face
    laplacian is the same variance taken over a crop of the very same array, the eye
    opening is fitted to a face box on it, and computing either in a pass of its own would
    decode every frame in the collection a second time for pixels that are already in
    memory.

    None means NOT MEASURED, the `frame_quality` rule: the frame did not decode (`frame`),
    or it has no face, no faces run behind it, or a crop too small to measure (`face`,
    `eyes` — the latter also when the 106-point model is not available at all).
    """
    frame: float | None = None
    face: float | None = None
    # F179: the eye opening over the eye width of the LARGEST face, small = closed. Named
    # apart from the two above because it is not a laplacian and is not on their scale;
    # it rides here because it rides on their decode.
    eyes: float | None = None


# (path, the faces of that frame) -> the numbers of one decode. The second argument is
# `NO_FACES` for every frame outside the face population, which is most of them.
SharpnessFn = Callable[[str, FaceBoxes], Sharpness]

# The tier that produced a frame_quality row, and with it the incrementality marker.
QUALITY_SOURCE_CLASSIC = "classic"   # sharpness only
QUALITY_SOURCE_CLIP = "clip"         # + pets
QUALITY_SOURCE_VLM = "vlm"           # + the model answers about a candidate list


def laplacian_variance(img: Image.Image) -> float | None:
    """Variance of the 4-neighbour laplacian of a grayscale frame — the blur signal.

    The classic measure and deliberately the plain one: a blurred frame has little
    high-frequency energy, so the second derivative barely moves and its variance
    collapses. Computed with numpy rather than cv2 (which this project does not import)
    over the interior pixels only — the border has no full neighbourhood.

    None for a frame too small to have an interior: nothing to measure, and 0.0 would be
    read as "completely flat", which is a different statement.
    """
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return None
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


# F155: the shortest side a crop may have before the laplacian over it stops meaning
# anything. In PREVIEW pixels, because that is the array the variance is taken over: at
# `sharpness_max_edge` = 512 a 24 px box is a face occupying ~5% of the frame's width, and
# below that the crop is a few hundred pixels of mostly sensor noise. Such a frame gets
# NULL rather than the number — "not measured", never a small value that would sort it to
# the top of a blur list it was never measured for.
FACE_CROP_MIN_PX = 24


def face_crop_boxes(faces: FaceBoxes, size: tuple[int, int],
                    min_px: int = FACE_CROP_MIN_PX) -> list[tuple[int, int, int, int]]:
    """Face boxes rescaled from the ORIGINAL frame into preview pixels, clamped to it.

    THE RESCALING IS THE FEATURE, not a detail of it. `faces.bbox` is written in
    coordinates of the full original — ArcFace embeds out of it — while the laplacian is
    taken over a preview of a few hundred pixels, so a box used as written falls outside
    the array it is supposed to index. That is not a hypothetical: the measurement this
    feature is built on made exactly that mistake, 39 of its 68 crops fell off the frame
    and were dropped, and the 29 that survived reported 100% recall instead of the real
    62%. A broken crop does not fail loudly — it flatters the result.

    Clamping is the second half of the same guard: a box may legitimately run a pixel or
    two past the edge (a face at the border, rounding in the scale), and a crop is taken
    of the part that is inside rather than not at all.

    Boxes below `min_px` on either side after scaling are left out — see FACE_CROP_MIN_PX.
    A `faces` with no scale to it yields nothing, for the same reason: a box in unknown
    units is not a box.
    """
    if not faces.usable:
        return []
    width, height = size
    scale = max(width, height) / faces.long_edge
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in faces.boxes:
        left = max(0, min(width, int(round(min(x1, x2) * scale))))
        top = max(0, min(height, int(round(min(y1, y2) * scale))))
        right = max(0, min(width, int(round(max(x1, x2) * scale))))
        bottom = max(0, min(height, int(round(max(y1, y2) * scale))))
        if right - left >= min_px and bottom - top >= min_px:
            out.append((left, top, right, bottom))
    return out


def face_crop_sharpness(img: Image.Image, faces: FaceBoxes) -> float | None:
    """The laplacian of the SHARPEST face on an already-decoded frame; None if none.

    The sharpest and not the average, because of what the number is asked for: "was this
    shot taken properly". One person in focus and another walking past out of it is a
    photograph that worked, and an average would call it half-blurred. If any face in the
    frame is sharp, the frame is.
    """
    best: float | None = None
    for box in face_crop_boxes(faces, img.size):
        value = laplacian_variance(img.crop(box))
        if value is not None and (best is None or value > best):
            best = value
    return best


# F179: the eye contour of insightface `2d106det` — each eye is a ring of eight points,
# and only the SET of indices matters, never their order: `eye_openness` below finds the
# corners itself, so a model that lists the ring the other way round gives the same number
# and a model that moves the ring elsewhere gives a wrong one either way. The map was
# verified against the detector's own eye points on real frames during the measurement
# (scripts/measure_eye_state.py, `eye_rings_agree`), which is where it belongs: here there
# is nothing to check it against, because the five detector points are not stored.
EYE_RINGS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (33, 35, 36, 37, 39, 40, 41, 42),
    (87, 89, 90, 91, 93, 94, 95, 96),
)

# The whole frame's preview and ONE face box in ITS pixels -> the 106 contour points, in
# the same pixels; None when the model has nothing to say (or is not there at all). The
# box is passed rather than a crop because the model aligns the face itself out of it.
EyeLandmarkFn = Callable[[Image.Image, tuple[int, int, int, int]], "np.ndarray | None"]


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

    Copied unchanged from the measurement this feature was decided by (F178,
    scripts/measure_eye_state.py) — the numbers in `features.eye_openness_max` are this
    function's numbers, and a paraphrase would silently invalidate them.
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


def largest_face_box(faces: FaceBoxes,
                     size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The biggest face of a frame, rescaled into preview pixels; None if there is none.

    THE BIGGEST AND ONLY THE BIGGEST — the `largest` rule the measurement was run under.
    A frame where a passer-by at the back has their eyes shut is not a portrait with closed
    eyes, and reading every face would put it in the slice; the largest face is the one the
    shot is about. (`face_sharpness` picks the sharpest face for the mirror-image reason:
    each takes the face its own question is about.)

    Rescaled through `face_crop_boxes`, so it inherits the guard that feature exists for —
    the boxes are in pixels of the ORIGINAL and the preview is a small copy of it — and its
    minimum size: a box too small there is not a face this can be fitted to.
    """
    boxes = face_crop_boxes(faces, size)
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def face_eye_openness(img: Image.Image, faces: FaceBoxes,
                      landmark: EyeLandmarkFn | None) -> float | None:
    """How open the eyes of the largest face on an already-decoded frame are; None if not.

    The MORE CLOSED of the two eyes answers for the face, because that is the question the
    slice asks — a frame where one eye is shut is a frame the person wants to look at — and
    because a half-blink averaged with an open eye is nothing at all.

    None all the way down, never a small number: no face, no model, a model that answered
    with something that is not 106 points, or a ring that came out degenerate. A wrong
    small value would sort itself to the very top of a list ordered by "most closed first".
    """
    if landmark is None:
        return None
    box = largest_face_box(faces, img.size)
    if box is None:
        return None
    points = landmark(img, box)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= max(max(ring) for ring in EYE_RINGS):
        return None
    values = [value for value in
              (eye_openness(points[list(ring)]) for ring in EYE_RINGS)
              if value is not None]
    return min(values) if values else None


def preview_sharpness_detector(max_edge: int,
                               landmark: EyeLandmarkFn | None = None) -> SharpnessFn:
    """The real detector: the shared preview cache, at a FIXED resolution.

    Fixed because the variance of the laplacian is scale-dependent — the same photo
    measured at 512 and at 1536 px gives two different numbers, and a threshold over a
    mixture of the two means nothing. `features.sharpness_max_edge` is that resolution;
    changing it invalidates every threshold chosen against the old one.

    Decoded grayscale straight away (the measure only looks at luma) and through
    `decode_rgb_preview`, so the cost on any stage after the first is a small-JPEG decode,
    not a decode of the original. A vanished or undecodable file is None — "no signal",
    the same contract the OCR detector gives.

    F155: ONE decode, two numbers — the whole frame and the sharpest face in it. The
    orientation is applied here where it used to be left alone, because the face boxes are
    written in the rotated space (faces._decode_for_faces) and the two have to be the same
    space for the crop to land on the face. The frame number does not move under that: the
    laplacian kernel is symmetric under every rotation and mirror an EXIF orientation can
    express, and so is the set of interior pixels its variance is taken over. What is not
    exactly symmetric is the RESAMPLE around it — scaling and then turning a frame is not
    pixel-identical to turning and then scaling — and that is worth ~0.01% on a rotated
    frame, against a band 270 units wide.

    F179: a THIRD number off that same decode, and no third decode for it — the eye
    openness of the largest face, fitted to the box the crop above is taken from. The one
    thing it changes is the COLOUR of the decode: the 106-point contour was measured on a
    colour preview (F178) and a grayscale one is not the input those numbers came from, so
    a run with the model present asks for RGB and `laplacian_variance` converts to luma
    itself. Both paths measure the luma of the SAME preview — libjpeg's grayscale output
    against a YCbCr->RGB->luma round trip — and both weight the channels the same way, so
    what is left is rounding: measured at 0.15% on noise and 0.12% on a photograph, against
    a band 270 units wide. The F155 argument about the orientation, one step on. Without a
    model (`landmark is None`) the decode is grayscale exactly as before.
    """
    def sharpness(path: str, faces: FaceBoxes = NO_FACES) -> Sharpness:
        try:
            st = os.stat(path)
        except OSError:
            return Sharpness()
        img = imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge,
            grayscale=landmark is None, apply_orientation=True)
        if img is None:
            return Sharpness()
        try:
            return Sharpness(frame=laplacian_variance(img),
                             face=face_crop_sharpness(img, faces),
                             eyes=face_eye_openness(img, faces, landmark))
        except Exception as exc:  # noqa: BLE001 — one bad frame must not break the stage
            _log.warning("junk: резкость не посчиталась для %s: %s", path, exc)
            return Sharpness()

    return sharpness


def lazy_eye_landmarks(build: Callable[[], EyeLandmarkFn]) -> EyeLandmarkFn:
    """The 106-point model, built on the FIRST face of a run and never before it.

    Two runs must not pay for it: one over a collection with no faces in it (two thirds of
    a typical archive have none, and a first run has no `faces` rows at all), and one on a
    machine where the model cannot be built — the [faces] extra missing, no weights on
    disk, a broken onnxruntime. The second is why a failure answers None for the rest of
    the run instead of raising: the eye number is one column of `frame_quality`, and an
    optional column must never take the stage that writes the other five down with it. The
    reason is logged ONCE, not once per frame.
    """
    state: dict[str, EyeLandmarkFn | None] = {}

    def landmark(img: Image.Image,
                 box: tuple[int, int, int, int]) -> np.ndarray | None:
        if "fn" not in state:
            try:
                state["fn"] = build()
            except Exception as exc:  # noqa: BLE001 — an optional column, not the stage
                _log.warning(
                    "junk: модель 106 точек не поднялась (%s) — колонка eye_openness "
                    "останется пустой, остальная часть стадии не затронута", exc)
                state["fn"] = None
        built = state["fn"]
        return None if built is None else built(img, box)

    return landmark


def insightface_eye_landmarks() -> EyeLandmarkFn:  # pragma: no cover — ML
    """`2d106det` out of the buffalo_l set the faces stage already downloads.

    NO NEW WEIGHTS: the 106-point model is 4.8 MB inside the set `sorta faces` has been
    fetching since phase 3, and it is disabled there (`faces._ALLOWED_MODULES`) only
    because that stage has nothing to do with it. The detector is loaded beside it because
    insightface's FaceAnalysis insists on one, and it is never CALLED: the boxes come from
    the `faces` table, which is the whole point of doing this inside the junk stage.

    The box arrives in preview pixels and the points come back in them, so nothing here
    knows about the original frame — `largest_face_box` has already done that conversion.
    """
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face

    from sorta import faces as faces_mod

    faces_mod._enable_cuda_dll_dirs()
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_2d_106"],
                       providers=accel.onnx_providers())  # F214: CUDA -> CoreML -> CPU
    edge = faces_mod.DET_SIZE_DEFAULT   # F88's pinned shape; nothing here detects anything
    app.prepare(ctx_id=0, det_size=(edge, edge))
    model = app.models["landmark_2d_106"]

    def landmark(img: Image.Image,
                 box: tuple[int, int, int, int]) -> np.ndarray | None:
        # BGR, the order every insightface model is trained on — the same conversion the
        # measurement made (scripts/measure_eye_state.py), and the same one `faces` makes.
        array = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
        found = model.get(array, Face(bbox=np.asarray(box, dtype=np.float32),
                                      det_score=1.0))
        return np.asarray(found, dtype=np.float64)

    return landmark


# F186: the frame-quality prompt is gone, and with it the last of the three questions it
# once carried. F122 measured the "accidental" one out (5% precision, and the frames the
# model called DELIBERATE held twice that rate), F177 dropped "is there a subject" by
# inspection (212 subjectless frames out of 6 111, all of them ordinary photographs), and
# the eyes were the one left. F179 answered them without a model at all — the spread of an
# eyelid contour the faces stage already fits — at 62% precision over 48% recall against
# the model's 60% over 9% on the same 249 hand labels. The replacement shipped and the
# question stayed behind it; this is where it is taken out.
#
# The three ANSWER COLUMNS stay and stay NULL. That is the same decision F122 and F177
# already made for the other two: NULL is exactly what "not asked" means in that table,
# and a documented empty column is cheaper than a rebuild of it.
_NON_WORD_RE = re.compile(r"[^a-z]+")


def _frame_question(describe: Callable[[Sequence[Image.Image], str, int], str],
                    max_edge: int, prompt: str,
                    max_new_tokens: int) -> Callable[[str], str]:
    """One prompt over one frame, over an ALREADY LOADED runtime (naming.shared_vlm).

    The decode goes through the shared preview cache, Unicode/HEIC-safe, exactly as
    everywhere else here; a frame that will not decode gets an empty answer, which parses
    to "not asked".

    Shared by the two questions this stage still asks a frame (the pet check, F130; the
    rescue, F140) because they differ in the prompt and the token budget and in nothing
    else — a second copy of the decode would be a second place for the cache key to go
    wrong.

    F206: and it comes back with its HALVES when the runtime has them, exactly like
    `vlm_classifier_from` above. This used to be "deliberately the plain, serial path",
    on the argument that these populations are a candidate list rather than a whole
    collection and the pipeline would cost more reading than it saves seconds. The
    argument was measured out of date by the run of 2026-08-05: 4 281 frames through
    these two questions at 0.42 frames/s against the deep tier's pipelined 1.4, i.e.
    116 minutes a run. The halves are what `_vlm_labels` needs to overlap the CPU half
    of one frame with the GPU half of the previous one, and it is the same machinery,
    the same window and the same order guarantee the deep tier has run on since F101 —
    a question whose prompt, token budget and input size are untouched by any of it.
    """
    def decode(path: str) -> Image.Image | None:
        try:
            st = os.stat(path)
        except OSError:
            return None  # unreadable — the caller answers with "not asked"
        return imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge)

    split = describe if isinstance(describe, SplitVlm) else None
    if split is None:
        def ask(path: str) -> str:
            img = decode(path)
            if img is None:
                return ""
            return describe([img], prompt, max_new_tokens)

        return ask

    def prepare(path: str) -> PreparedFrame:
        img = decode(path)
        # An empty answer IS the answer here (both parsers read it as "not asked"), so a
        # frame that never reaches the model carries it through the pipeline as a ready
        # label — the GPU half then stays free of file-system branches, as in the tier.
        if img is None:
            return PreparedFrame(label="")
        return PreparedFrame(inputs=split.prepare([img], prompt))

    def answer(prepared: PreparedFrame) -> str:
        if prepared.label is not None:
            return prepared.label
        return split.generate(prepared.inputs, max_new_tokens)

    return SplitVlmClassifier(prepare=prepare, classify_prepared=answer)


# --- F130: the pet check --------------------------------------------------------------
#
# path -> the model's raw answer about the animal in one frame (parsed by
# `parse_pet_answer`). The same shape as QualityAskFn on purpose: both are one prompt over
# one frame, and both are injected by the suite so no test loads a model.
PetAskFn = Callable[[str], str]

# ONE question with three outcomes, and the species is deliberately not among them. F122
# retired the species labels by measurement — the binary call was 92% right and the
# `cat`/`dog`/`pet` assignment on top of it was not — and bringing them back through a
# different model without a new measurement would be an unmeasured label that looks like
# data. If a consumer ever needs the species, that is a feature with its own measurement.
#
# The wording names what the collection actually got wrong (F120/F121: drawn cats, plush
# toys, a fur coat, a picture on a screen), because those are the frames the check exists
# to catch, and "a picture of a cat" is not a category a model volunteers unprompted.
_PET_VLM_PROMPT = (
    "Look at this photo and answer with exactly one word from this list:\n"
    "real — a living animal is actually present in the photo;\n"
    "depiction — the only animal is a picture of one: a drawing, a painting, a cartoon, "
    "a plush toy, a figurine, a statue, a print on clothing, an animal on a screen or "
    "on a poster;\n"
    "none — there is no animal in the photo at all.\n"
    "Answer with one word: real, depiction or none."
)
# One word, like the deep tier's label: a larger budget only buys the model room to
# explain itself, which the parser below would then have to wade through.
_PET_VLM_MAX_NEW_TOKENS = 8

# Keyword -> stored value, IN PRIORITY ORDER, and the order is a decision rather than an
# accident. `real` is the word a model reaches for while EXPLAINING one of the other two
# ("not a real animal", "no real animal here"), so a scan that met it first would read
# half the rejections as agreement — the same trap `_QUALITY_KEYWORDS` used to avoid by
# putting `no_subject` before `subject`, until F177 retired that question.
_PET_VLM_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("depiction", PET_VLM_DEPICTION),
    ("none", PET_VLM_NONE),
    ("real", PET_VLM_REAL),
)


def parse_pet_answer(answer: str) -> str | None:
    """The model's answer -> real | depiction | none; None when nothing was recognized.

    Read as leniently as `parse_quality_answer` reads its keywords, and for the same
    reason (the F96 lesson: asked for a composite format the model answers in prose
    anyway) — everything that is not a letter is a separator, and the word is looked for
    anywhere in the line rather than as the whole answer.

    None is NOT `none`. An answer nobody could read means the question was effectively not
    asked, so the frame falls back to the unverified rule; reading it as "no animal" would
    be guessing, and guessing here silently deletes a label the cheap tier was right about.
    """
    text = "_" + _NON_WORD_RE.sub("_", (answer or "").lower()) + "_"
    return next((value for keyword, value in _PET_VLM_KEYWORDS
                 if f"_{keyword}_" in text), None)


def vlm_pet_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                  max_edge: int) -> PetAskFn:
    """The pet question over a loaded runtime — see _frame_question."""
    return _frame_question(describe, max_edge, _PET_VLM_PROMPT, _PET_VLM_MAX_NEW_TOKENS)


def qwen_vlm_pet(model_name: str = _DEFAULT_VLM_MODEL,
                 max_edge: int = _DEFAULT_VLM_MAX_EDGE,
                 ) -> PetAskFn:  # pragma: no cover — ML, smoke test
    """The real pet asker — the SAME weights as everything else (F95): one per run."""
    return vlm_pet_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_pet_factory(max_edge: int) -> Callable[[str], PetAskFn]:
    """The default `pet_vlm_factory` of classify(), carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_pet(model_name, max_edge=max_edge)


# --- F140: the question asked of a rescue candidate ------------------------------------
#
# path -> the model's raw answer about one candidate (parsed by `parse_junk_rescue_answer`).
# The same shape as PetAskFn, and injected by the suite for the same reason: no test loads
# a model.
JunkAskFn = Callable[[str], str]

# Query strings -> one text feature per string, in the same order. The real one is
# `search.text_encoder` — the project's own open_clip, because the query has to land in the
# space the stored vectors live in — and it is resolved lazily, only for a run that has
# frames to score.
TextEncoder = Callable[[Sequence[str]], np.ndarray]

# The three answers, and they are the `media_class.verdict` values on purpose: this
# question exists to correct a verdict, so an answer that had to be translated into one
# would be a second vocabulary for the same fact.
JUNK_RESCUE_SCREENSHOT = "screenshot"
JUNK_RESCUE_DOCUMENT = "document"
JUNK_RESCUE_PHOTO = "photo"

# A question about the KIND of picture, not about its quality, and asked separately from
# the deep tier's own 3-way prompt (personal_photo/document/product) rather than by widening
# it: that prompt has no screenshot among its answers — the deep tier never had to detect
# one, it is the fast tier's job — and adding a fourth class there would move the verdicts
# of every frame it sees, including on runs where this feature is off.
_JUNK_RESCUE_PROMPT = (
    "Classify this image into exactly one category: screenshot, document, or photo.\n"
    "screenshot = a screen capture, or a photograph of a phone, monitor or TV screen, or "
    "a meme, or a picture that is mostly text.\n"
    "document = a receipt, a bill, a ticket, a form, an ID card or another text-heavy "
    "paper.\n"
    "photo = an ordinary photograph of people, places, pets or everyday life.\n"
    "Answer with exactly one word: screenshot, document, or photo."
)
# One word, like the other labels of this stage: a larger budget only buys the model room
# to explain itself, which the parser below would then have to wade through.
_JUNK_RESCUE_MAX_NEW_TOKENS = 8

# Keyword -> answer, IN PRIORITY ORDER, and the order is the same decision `_PET_VLM_KEYWORDS`
# documents: `photo` is the word a model reaches for while describing one of the other two
# ("a photo of a receipt", "a photo of a screen"), so a scan that met it first would read
# half the rejections as agreement.
_JUNK_RESCUE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("screenshot", JUNK_RESCUE_SCREENSHOT),
    ("screen", JUNK_RESCUE_SCREENSHOT),
    ("meme", JUNK_RESCUE_SCREENSHOT),
    ("document", JUNK_RESCUE_DOCUMENT),
    ("receipt", JUNK_RESCUE_DOCUMENT),
    ("photo", JUNK_RESCUE_PHOTO),
    ("photograph", JUNK_RESCUE_PHOTO),
)


def parse_junk_rescue_answer(answer: str) -> str | None:
    """The model's answer -> screenshot | document | photo; None when nothing was read.

    Lenient in the two ways every answer of this stage is read (the F96 lesson), and None
    is NOT `photo`: an unreadable answer means the question was effectively not asked, so
    the frame keeps the verdict the fast tier gave it. Reading it as a junk class would
    delete a photograph on the strength of a resemblance score — the one thing the whole
    feature is arranged to avoid.
    """
    text = "_" + _NON_WORD_RE.sub("_", (answer or "").lower()) + "_"
    return next((value for keyword, value in _JUNK_RESCUE_KEYWORDS
                 if f"_{keyword}_" in text), None)


def vlm_junk_rescue_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                          max_edge: int) -> JunkAskFn:
    """The rescue question over a loaded runtime — see _frame_question."""
    return _frame_question(describe, max_edge, _JUNK_RESCUE_PROMPT,
                           _JUNK_RESCUE_MAX_NEW_TOKENS)


def qwen_vlm_junk_rescue(model_name: str = _DEFAULT_VLM_MODEL,
                         max_edge: int = _DEFAULT_VLM_MAX_EDGE,
                         ) -> JunkAskFn:  # pragma: no cover — ML, smoke test
    """The real rescue asker — the SAME weights as everything else (F95): one per run."""
    return vlm_junk_rescue_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_junk_rescue_factory(max_edge: int) -> Callable[[str], JunkAskFn]:
    """The default `junk_rescue_vlm_factory` of classify(), carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_junk_rescue(model_name, max_edge=max_edge)


def clip_text_encoder(s: NamingSettings) -> TextEncoder:  # pragma: no cover — ML
    """The default text encoder of the rescue score — the search engine's own (F129).

    Imported here and not at module scope because `search` imports this module, and built
    only when a run actually has frames to score: it loads the model, and a stage that has
    nothing to ask must not pay for one (`_JunkRescuePass` calls this lazily, the same
    rule `_unused_classifier` states for the image side).
    """
    from .search import text_encoder

    return text_encoder(s)


# --- F132/F186: the keeper of a near-duplicate group, and why nothing asks about it -----
#
# The comparative question ("which of these five is the one to keep") is gone. It was
# measured on 2026-08-04 against 111 groups the owner labelled BLIND — the frames shuffled,
# the model's answer hidden — and it agreed with the person on 32% of them. Picking a frame
# at random agrees on 30.4% (20 000 shuffles say 30.3%), and sharpness, arithmetic and a
# cascade all land on 27-28%. Nothing here is a replacement, because there was nothing to
# buy: 451 seconds of GPU a run for the accuracy of a coin.
#
# What STAYS is the mechanism, in dedup.py: `group_keeper`, `KEEPER_SOURCE_SHARPNESS`,
# `store_group_keeper` and the ranking the Duplicates tab has always shown. Only the
# question to the model left, so the interface behaves exactly as it did.


@dataclass(frozen=True)
class QualitySettings:
    """Everything the cascade reads out of the config, resolved once per run.

    Same shape and same reason as GateSettings: the measurement script sweeps these
    numbers, and it can only price the real cascade if it drives the same functions off
    the same object the pipeline uses.
    """
    pets: bool
    pet_threshold: float
    sharpness_max_edge: int
    sharpness_band: tuple[float, float]
    subject_score_min: float
    # F120: media classes no VLM is shown (`vlm.exclude_classes`).
    exclude_classes: frozenset[str] = frozenset()
    # F130: the pet check — its own toggle, and the second, much lower threshold that
    # decides who is shown to the model. Defaulted (unlike the two fields above them in
    # the class) so a caller that built these settings by hand — the measurement script,
    # the suite — keeps working unchanged when the check is not what it is asking about.
    pets_verify: bool = False
    pet_candidate_threshold: float = 0.3
    # F140: the rescue score — its own toggle, and the threshold that decides who is shown
    # to the model. Defaulted for the same reason the two fields above are: a caller that
    # built these settings by hand keeps working when this is not what it is asking about.
    junk_rescue: bool = False
    junk_rescue_threshold: float = 0.02


def quality_settings(cfg: Config) -> QualitySettings:
    """`features:` + the `vlm:` keys the cascade reads (or those of a measurement)."""
    f = getattr(cfg, "features", None) or FeaturesConfig()
    vlm = cfg.vlm
    return QualitySettings(
        pets=bool(f.pets),
        pet_threshold=float(f.pet_threshold),
        sharpness_max_edge=int(f.sharpness_max_edge),
        sharpness_band=(float(f.sharpness_band_min), float(f.sharpness_band_max)),
        subject_score_min=float(f.subject_score_min),
        exclude_classes=frozenset(getattr(vlm, "exclude_classes", ()) or ()),
        pets_verify=bool(getattr(f, "pets_verify", False)),
        pet_candidate_threshold=float(getattr(f, "pet_candidate_threshold", 0.3)),
        junk_rescue=bool(getattr(f, "junk_rescue", False)),
        junk_rescue_threshold=float(getattr(f, "junk_rescue_threshold", 0.02)),
    )


# F120: the quality questions — is there a pet, are the eyes open, was this shot an
# accident — are questions about a PERSONAL PHOTOGRAPH (of the three, only the pet one is
# still put to a model — see F186 above). Asked of a screenshot or a product shot they
# produce an answer that means nothing, and the first live run showed
# what that costs: 45% of the `dog` class and 45% of the sharpest frames were not
# photographs at all. Screenshots are also structurally "sharper" than any photo (hard
# edges and text: mean laplacian 2854 against 1253), so a global sharpness ranking put
# them on top by construction.
QUALITY_VERDICT = "photo"

# The `faces` row that means "this file was processed and no face was found in it" (see
# the faces module docstring) — a bookkeeping marker, not a face. Every question about
# REAL faces has to exclude it: on the live collection 24 195 files out of 24 196 carry
# one, so a predicate that forgets it answers "all of them".
NO_FACES_BBOX = "[]"


def uncertain_band(sharpness: float | None, subject_score: float,
                   q: QualitySettings) -> bool:
    """Is this frame one the cheap tiers did NOT settle?

    Two independent ways in, either of them enough: sharpness inside the band where it
    decides nothing (clearly blurred is below it, clearly sharp above), or a junk-group
    CLIP probability of "a photograph" low enough that CLIP is saying it does not know
    what it is looking at. A frame that did not decode has no sharpness signal and is
    judged on the second condition alone.

    F186: this used to select the population of the quality VLM, and that question is
    retired. The band itself is kept because it is what `scripts/measure_frame_quality.py`
    prices the cascade over, and `features.sharpness_band_*` / `features.subject_score_min`
    are the thresholds it sweeps — the stage asks nothing of it.
    """
    low, high = q.sharpness_band
    if sharpness is not None and low <= sharpness <= high:
        return True
    return subject_score < q.subject_score_min


def faces_stage_ran(conn: sqlite3.Connection) -> bool:
    """Has the faces stage ever found a face in this index? One real row is enough.

    `bbox = '[]'` is not a face but the marker "this file was processed and had none"
    (faces module docstring), so it says nothing about the stage having run usefully and
    everything about it having run at all — which is why every question about REAL faces
    has to exclude it. On the live collection 24 195 files out of 24 196 carry such a row.
    """
    return bool(conn.execute(
        "SELECT EXISTS(SELECT 1 FROM faces WHERE bbox != ?)", (NO_FACES_BBOX,)
    ).fetchone()[0])


# F186 took `quality_scope_ready` and `quality_scope_ids` out with the question they
# served. They chose WHO the quality VLM was asked about — the frames of a near-duplicate
# group, of an event, with a face on them, or the whole collection — and there is no longer
# anybody to ask. Nothing replaced them: `faces_stage_ran` above is the one piece of that
# machinery with a second consumer (F121, whether an absent face is a fact or an absence of
# evidence), and it stays for that consumer alone.


@dataclass(frozen=True)
class FrameQuality:
    """One `frame_quality` row as Python types — None stays None, and is not a False."""
    file_id: int
    sharpness: float | None = None
    # F155: the same laplacian inside the sharpest face box of the frame, or None for "not
    # measured" — no face, no faces run, or a crop below FACE_CROP_MIN_PX. It ranks the
    # blur list and decides nothing: ~25% of what it flags is actually blurred.
    face_sharpness: float | None = None
    # F179: the eye opening over the eye width of the LARGEST face, or None for "not
    # measured" — no face, no faces run, a crop below FACE_CROP_MIN_PX, or no 106-point
    # model on this machine. Small means closed; it ranks the slice and decides nothing.
    eye_openness: float | None = None
    pet: str | None = None
    pet_score: float | None = None
    # F130: real | depiction | none, or None for "the model was not asked about this
    # frame" — which is what tells a rejected frame from one below the candidate threshold.
    pet_vlm: str | None = None
    eyes_open: bool | None = None
    # The two retired questions (F177, F122). The columns are read like every other one
    # so that a row is a row, but nothing asks them any more; `has_subject` was emptied
    # of the answers it did collect by the v26 migration, so both are NULL everywhere.
    has_subject: bool | None = None
    is_accidental: bool | None = None
    # F140: the zero-shot "screenshot rather than photograph" margin, or None for "not
    # computed" — the toggle is off, or this frame has no stored CLIP vector to read it off.
    junk_score: float | None = None
    source: str = QUALITY_SOURCE_CLASSIC


def _bool_or_none(value: object) -> bool | None:
    """SQLite 0/1/NULL -> False/True/None. The one place the distinction is decided."""
    return None if value is None else bool(value)


def _parse_bbox(raw: object) -> FaceBox | None:
    """One `faces.bbox` string -> (x1, y1, x2, y2); None for anything that is not one.

    Tolerant on purpose. The column is a JSON list written by another stage, and a row
    this cannot read has to cost that frame its face number and nothing else — the
    laplacian over the whole frame, the pets, the verdict all stand.
    """
    try:
        values = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def read_face_boxes(conn: sqlite3.Connection,
                    file_ids: Sequence[int]) -> dict[int, FaceBoxes]:
    """The real faces of `file_ids`, with the size of the frame they were found in (F155).

    `bbox = '[]'` is EXCLUDED, and that is not tidiness: the marker means "this file was
    processed and had no face in it", and on the live collection 24 195 of 24 196 files
    carry one (F125's trap). A predicate that forgets it answers "every file has a face"
    and then tries to crop a zero-length box out of every frame in the archive.

    A file with no width/height recorded is left out too — the boxes are in pixels of that
    frame, so without its size there is nothing to scale them by. Files missing from the
    result simply have no faces: the caller reads it with `NO_FACES` as the default.
    """
    out: dict[int, FaceBoxes] = {}
    for part in batched(list(file_ids), 500):
        rows = conn.execute(
            f"""SELECT fa.file_id, fa.bbox, f.width, f.height
                FROM faces fa JOIN files f ON f.id = fa.file_id
                WHERE fa.bbox != ? AND fa.file_id IN ({','.join('?' * len(part))})""",
            (NO_FACES_BBOX, *part))
        for r in rows:
            box = _parse_bbox(r["bbox"])
            if box is None or r["width"] is None or r["height"] is None:
                continue
            file_id = int(r["file_id"])
            known = out.get(file_id)
            long_edge = float(max(int(r["width"]), int(r["height"])))
            out[file_id] = FaceBoxes(
                boxes=(known.boxes if known else ()) + (box,), long_edge=long_edge)
    return out


def read_frame_quality(conn: sqlite3.Connection,
                       file_ids: Sequence[int] | None = None) -> dict[int, FrameQuality]:
    """`frame_quality` by file_id — the reading side of the "NULL is not False" rule.

    The consumers of this table (F114: the web app, the sorter, the events stage) must not
    each rebuild the 0/NULL distinction out of raw rows; one of them would get it wrong
    exactly once and quietly discard frames nobody had looked at.
    """
    sql = ("SELECT file_id, sharpness, face_sharpness, eye_openness, pet, pet_score,"
           " pet_vlm, eyes_open, has_subject, is_accidental, junk_score, source"
           " FROM frame_quality")

    def rows(cursor: sqlite3.Cursor) -> dict[int, FrameQuality]:
        return {
            int(r["file_id"]): FrameQuality(
                file_id=int(r["file_id"]),
                sharpness=None if r["sharpness"] is None else float(r["sharpness"]),
                face_sharpness=(None if r["face_sharpness"] is None
                                else float(r["face_sharpness"])),
                eye_openness=(None if r["eye_openness"] is None
                              else float(r["eye_openness"])),
                pet=r["pet"],
                pet_score=None if r["pet_score"] is None else float(r["pet_score"]),
                pet_vlm=r["pet_vlm"],
                eyes_open=_bool_or_none(r["eyes_open"]),
                has_subject=_bool_or_none(r["has_subject"]),
                is_accidental=_bool_or_none(r["is_accidental"]),
                junk_score=None if r["junk_score"] is None else float(r["junk_score"]),
                source=str(r["source"]),
            )
            for r in cursor
        }

    if file_ids is None:
        return rows(conn.execute(sql))
    ids = list(file_ids)
    out: dict[int, FrameQuality] = {}
    # In chunks because a caller asking about a whole collection is the expected case and
    # SQLite has a ceiling on bound parameters — one that a photo library reaches easily.
    for part in batched(ids, 500):
        out.update(rows(conn.execute(
            f"{sql} WHERE file_id IN ({','.join('?' * len(part))})", tuple(part))))
    return out


# The fast half of the cascade writes the row; the model halves update it in place. Every
# model column is reset to NULL by the fast half on purpose: this run has not asked yet,
# and a leftover answer from a previous run would describe a frame the current settings
# may never look at. F130 puts `pet_vlm` under the same rule — the fast half re-walks a
# frame only when its own marker went stale (a prompt edit among other things), and a
# stale prompt is exactly when a stored answer must not survive.
#
# F186: `eyes_open` joins `has_subject` and `is_accidental` as a column this statement
# only ever sets to NULL. All three questions are retired and nothing updates them
# afterwards any more — NULL is what "not asked" means here, and it is now the only value
# the three of them ever hold.
_QUALITY_UPSERT = """INSERT INTO frame_quality (file_id, sharpness, face_sharpness,
                         eye_openness, pet, pet_score, pet_vlm, eyes_open, has_subject,
                         is_accidental, junk_score, source, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)
                     ON CONFLICT(file_id) DO UPDATE SET
                         sharpness = excluded.sharpness,
                         face_sharpness = excluded.face_sharpness,
                         eye_openness = excluded.eye_openness, pet = excluded.pet,
                         pet_score = excluded.pet_score, pet_vlm = NULL, eyes_open = NULL,
                         has_subject = NULL, is_accidental = NULL, junk_score = NULL,
                         source = excluded.source, updated_at = excluded.updated_at"""
# F130: the answer AND the label it decides, written together — `pet` is a function of
# `pet_vlm` (see pet_label), so leaving the two to be reconciled by a later reader would
# be two sources of truth for one fact. `pet_score` is untouched: it is what a threshold
# is re-chosen from, and a rejected frame keeps it like every other one.
_PET_ANSWER_UPDATE = """UPDATE frame_quality
                        SET pet_vlm = ?, pet = ?, updated_at = ?
                        WHERE file_id = ?"""
# F140: the rescue score, written on its own after the fast pass — it is read off the
# vector `_EmbeddingPass` stores inside the same loop, so it cannot be part of the upsert
# that opens the row.
_JUNK_SCORE_UPDATE = """UPDATE frame_quality
                        SET junk_score = ?, updated_at = ?
                        WHERE file_id = ?"""
# F154: the animal label after the detector has spoken. `pet_vlm` is untouched — the two
# tiers answer different questions (which object is in the frame; whether the animal in it
# is alive) and overwriting one with the other would lose the fact that the model was
# asked. `pet_score` is untouched for the same reason it survives the F130 check: it is
# what a threshold is re-chosen from.
_PET_DETECTOR_UPDATE = """UPDATE frame_quality
                          SET pet = ?, updated_at = ?
                          WHERE file_id = ?"""
# F154: what the detector saw on one frame. The row IS the "already examined" marker, so
# it is written for a frame with NO animal on it too (`label` NULL, `boxes` an empty list)
# — otherwise every later run pays 83.8 ms again for each frame the detector turned down.
_DETECTIONS_UPSERT = """INSERT INTO detections (file_id, label, score, boxes, model,
                                                updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(file_id) DO UPDATE SET
                            label = excluded.label, score = excluded.score,
                            boxes = excluded.boxes, model = excluded.model,
                            updated_at = excluded.updated_at"""

# The verdict of one frame. F68: `tier` is written on every path and always equals the
# run's active tier — a row the active tier touched must never stay unmarked (or marked by
# an older tier), otherwise it is reclassified on every run. Lifted out of classify() by
# F140 for the same reason F90 lifted the gate functions: a second writer of this row
# (the rescue) must write it the same way, and a paraphrase would drift.
#
# HOW THIS ROW IS WRITTEN, AND WHY IT IS NOT BATCHED (F164). The F147 phase table read
# `junk_write 470,3 s / 24 196 frames / 19,4 ms` and the obvious suspect was a commit per
# row — 19,4 ms for one INSERT would be more than CLIP spends on a frame in the same
# stage (11,7 ms on the GPU). It is not what happens. Every write of the fast pass goes
# through ONE transaction: sqlite3 is left at its default isolation, so no statement
# autocommits, `with conn:` sits OUTSIDE the chunk loop in classify(), and the single
# COMMIT happens when that loop is done. The deep tier and every later pass have one
# `with conn:` each, for the same reason.
#
# Measured on this machine (scripts/measure_junk_write.py, 24 196 rows, this exact
# statement, a throwaway database on the same disk, three runs):
#
#     commit strategy            commits   ms/row, best run   ms/row, worst run
#     one transaction (today)          1              0,004               0,005
#     one commit per 16 rows       1 513              0,003               0,157
#     one commit per row          24 196              0,014               2,271
#
# The spread on the two lower rows is not noise to be averaged away: it is the operating
# system deciding whether a COMMIT really reaches the disk, and it is exactly what makes
# committing often a gamble — two of three runs never flushed and one paid 55 s for the
# same 24 196 rows. The row that matters has no spread at all: what the stage does today
# costs 0,004-0,005 ms whatever the disk feels like, i.e. 0,1 s of the 470,3 s the phase
# was billed for — 0,02%.
#
# So the lever the brief proposed was already pulled, and there is nothing left to batch.
# A batch size in the config would buy those 0,02% back at best, would be slower whenever
# the flush is real, and would break the property this shape gives for free: a run that
# dies mid-pass leaves the whole collection with its PREVIOUS verdicts, never half of
# today's and half of yesterday's.
#
# Where the phase's seconds actually are: `report.enter(CLASSIFY_PHASE_WRITE)` covers the
# per-frame loop that also measures the laplacian (`_QualityPass.measure`) and stores the
# CLIP vector (`_EmbeddingPass.store`). The same script prices those on real frames of the
# live collection: 26,9 ms for the sharpness of one frame, 0,06 ms for its vector, 0,005
# ms for its verdict. 79% of the frames of that run got a quality row (19 216 of 24 196),
# and 0,79 x 26,9 = 21,3 ms against the 19,4 ms the phase was billed — the laplacian IS
# the phase. Making it cheaper is a question about `features.sharpness_max_edge` and the
# preview cache, not about SQLite; whoever picks it up should start from that number and
# not from this statement.
_MEDIA_CLASS_UPSERT = """INSERT INTO media_class (file_id, verdict, source, score,
                                                  updated_at, tier)
                         VALUES (?, ?, ?, ?, ?, ?)
                         ON CONFLICT(file_id) DO UPDATE SET verdict = excluded.verdict,
                             source = excluded.source, score = excluded.score,
                             updated_at = excluded.updated_at, tier = excluded.tier"""


def _unused_classifier(paths: list[str], prompts: list[str]) -> np.ndarray:
    """The classifier of a run that asks CLIP nothing — being called is a bug, not a case.

    A backfill of sharpness alone (F113, both toggles off, junk already classified) needs
    no model, and building one would be the entire cost of such a run. This stands in for
    it so nothing downstream has to carry an optional classifier around.
    """
    raise AssertionError(  # pragma: no cover — unreachable by construction
        "junk: CLIP вызван в прогоне, где он не нужен")


def quality_prompt_fingerprint(pets: bool, *, verify_pets: bool = False,
                               rescue: bool = False,
                               rescue_vlm: bool = False) -> str:
    """Eight hex characters over the TEXT that decides what lands in `frame_quality`.

    F120: the marker used to name the tier and nothing else, so editing a prompt left
    every stored answer looking fresh. That is not hypothetical — F120 rewrote the pet
    group (five anti-classes for drawings, toys, food, screens and clothing) and the old
    labels would have survived it untouched, because a tier called `vlm` still equals a
    tier called `vlm`. Nobody should have to remember to empty a table by hand after
    editing a prompt.

    Only the text that actually reaches the stored columns is hashed: the CLIP prompt
    list (the pet group writes `pet`/`pet_score`, the junk group decides through the
    subject score who is asked at all) and, when the model runs, the question it is
    asked. Sharpness depends on neither, which is why the `classic` tier carries no
    fingerprint and a sharpness-only collection is not invalidated by prompt work.

    F186 dropped the `with_vlm` half of that — the frame-quality prompt is retired, so
    there is no longer a question of the model whose wording could reach `eyes_open`. The
    hash a collection already carries does not move: `with_vlm=False` appended nothing.
    """
    parts = list(clip_prompts(pets))
    if pets:
        # F122: what the stored value MEANS is part of what makes a row stale, not only
        # the text that produced it. Collapsing three class names into one changed the
        # meaning of `frame_quality.pet` without touching a prompt, and a marker blind to
        # that would have left every row saying `cat` and looking fresh.
        parts.append(PET_CLASS)
    if verify_pets:
        # F130: the check's question decides `pet_vlm` and, through it, `pet` — so an edit
        # to its wording has to invalidate the rows it produced, exactly as an edit to the
        # CLIP prompts does. Only when the check actually ran: a collection measured
        # without it must not be invalidated by a prompt nobody asked.
        parts.append(_PET_VLM_PROMPT)
    if rescue:
        # F140: these prompts decide `junk_score` and, through the threshold, which frames
        # are shown to the model at all — so an edit has to invalidate the stored scores
        # (brief requirement 1) and, with them, the candidate list a later run rebuilds.
        parts.extend(junk_rescue_prompts())
    if rescue_vlm:
        # And the question itself, only when it is asked: a collection scored without the
        # deep tier must not be invalidated by wording nobody used on it.
        parts.append(_JUNK_RESCUE_PROMPT)
    raw = "\x00".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def quality_tier(source: str) -> str:
    """The tier out of a stored marker — `vlm#1a2b3c4d` -> `vlm`.

    Consumers care which tier answered, not which revision of the prompts did; the
    fingerprint is for invalidation alone.
    """
    return source.split("#", 1)[0]


def _quality_source(use_clip: bool, pets: bool, pet_ask: PetAskFn | None = None,
                    rescue: bool = False,
                    rescue_ask: JunkAskFn | None = None) -> str:
    """The tier marker this run writes — and therefore what it considers up to date.

    F130: the pet check is a model too, so a run that only does that one still writes the
    `vlm` tier — the marker names WHICH TIER processed the row, and a row whose animal
    label came from the model did not come from the CLIP tier.

    F140: the rescue score is a CLIP signal (text vectors against a stored image vector),
    so switching it on moves the row to the `clip` tier at worst — and to `vlm` when the
    deep tier is there to answer for its candidates. Either way the fingerprint carries the
    prompts, which is what makes the score recomputable after an edit.

    F186 removed the third asker (the frame-quality question) from the list. A collection
    marked by a run that asked only the pet check or only the rescue keeps the marker it
    has: the fingerprint of those runs never carried the retired prompt.
    """
    if pet_ask is not None or rescue_ask is not None:
        fingerprint = quality_prompt_fingerprint(
            pets, verify_pets=pet_ask is not None,
            rescue=rescue, rescue_vlm=rescue_ask is not None)
        return f"{QUALITY_SOURCE_VLM}#{fingerprint}"
    if use_clip and (pets or rescue):
        return (f"{QUALITY_SOURCE_CLIP}#"
                f"{quality_prompt_fingerprint(pets, rescue=rescue)}")
    # Sharpness only: no prompt took part, so there is nothing for a prompt edit to
    # invalidate — and a bare marker keeps the cheap case cheap to read.
    return QUALITY_SOURCE_CLASSIC


# F100: the phases `classify` reports. Stable identifiers, not captions — the served
# UI localizes them (ui._UI_STRINGS), the CLI labels them for the rich bar
# (cli._CLUSTER_PHASE_LABELS is the precedent). The keys mirror the three parts F73
# split the per-chunk loop into, plus the deep tier.
#
# The difference from faces.CLUSTER_PHASE_* (F84) is the point of this feature: there
# the long phase (HDBSCAN) is ONE blocking call, so it can only show a stopwatch;
# here EVERY phase is measurable, the VLM one included — its candidate list is known
# before the loop starts, because the gate has already run. That is also why the VLM
# phase needs a caption more than the others: it is the only one that changes the
# denominator under the user (24 196 frames -> 7 896 candidates on the live run), and
# a bar that restarts from zero without a word reads as a bar that lost its place.
CLASSIFY_PHASE_CLIP = "junk_clip"
CLASSIFY_PHASE_OCR = "junk_ocr"
CLASSIFY_PHASE_VLM = "junk_vlm"
# F205: the model is asked in THREE places, and until this feature all three reported
# under the name above. The run of 2026-08-05 measured what that cost: the deep tier ran
# 7 951 frames at 1.4 frames/s (pipelined), the animal check 2 997 at 0.42 and the rescue
# 1 284 at 0.41-0.49 — one name over three prices that differ by a factor of three, so no
# reader of the log and no arithmetic over it could price any of them. A phase name IS the
# unit a measurement is filed under (runlog.measurement_unit), which is why three prices
# need three names.
#
# The deep tier KEEPS `junk_vlm`. It is the one of the three whose seconds a log written
# before this split can still be trusted for — it dominated that shared bucket — and
# renaming it would throw those measurements away and buy nothing. The two questions of
# the back half get the new names, and a log without them simply has no measurement for
# them, which the estimate already handles by falling back to its default.
CLASSIFY_PHASE_PETS_VLM = "junk_pets_vlm"
CLASSIFY_PHASE_RESCUE_VLM = "junk_rescue_vlm"
# F164: this phase is NOT the cost of writing. Its seconds are the per-frame loop the
# writes share with the laplacian and the stored vector, and the laplacian is ~99% of
# them — see the measured breakdown at _MEDIA_CLASS_UPSERT before reading a number under
# this name as a number about SQLite.
CLASSIFY_PHASE_WRITE = "junk_write"
# F141: the search index — a phase of its own and not part of `junk_clip`, because it is
# the one pass here that is a SECOND encode of the same frames rather than a use of the
# first. Its seconds are the price of `features.search_index` and nothing else, and that
# is exactly the number somebody deciding whether to switch it on needs to read.
CLASSIFY_PHASE_SEARCH = "junk_search"
# F154: the object detector over the candidates of the animal query — its own phase for
# the reason the search index has one. It is neither the fast CLIP pass (it is a second
# model over a short list) nor the VLM tier (it is not a VLM, and it costs 83.8 ms where
# that one costs 0.78 s), so its seconds price `features.detector` and nothing else.
CLASSIFY_PHASE_DETECT = "junk_detect"

# F147: the name this stage is timed under in the run log — the same one the pipeline
# calls it by (`cli._pipeline_steps`), because the phase lines are read next to the
# `stage=junk elapsed=...` line they break down, and a second spelling would break
# every grep that puts the two together.
CLASSIFY_STAGE = "junk"

# F165: and the name of the half that runs BEFORE faces — the verdicts alone. The phases
# keep their `junk_*` identifiers: they are the same passes, run by the same function, and
# renaming them per caller would break the captions (i18n `cli.phase.junk_*`), the UI
# labels and every grep over a run log written before this split. What the stage name
# decides is which `stage=` the phase lines are filed under, and that one has to match the
# `stage_timer` the caller opened or the F166 close-out would look for phases nobody
# registered.
VERDICTS_STAGE = "classify"


class _PhaseProgress:
    """Phase + `(done, total)` reporting for `classify` (F100).

    The phase channel is optional and duck-typed exactly as in faces (F84): a callback
    that can show a caption exposes `phase(name)` (progress.TaskProgress,
    ui._StageProgress), a bare `(done, total)` function simply gets no phases, and
    without a callback at all every method is a no-op — the CLI path, the quiet mode
    and most of the suite call classify() that way.

    Unlike clustering, the fast-tier phases interleave INSIDE the per-chunk loop
    (CLIP -> OCR -> write, F73) over one shared counter of frames, so `enter` only
    relabels and never touches the count — a bar that restarted three times per chunk
    would be worse than no phases at all. Repeating the current phase is not re-sent:
    the UI restarts the phase clock on every report. `start` is for the one place
    where the denominator really does change (the VLM tier counts its own candidates),
    and it changes the caption at the same moment, which is what makes the new numbers
    readable instead of a bar that silently slid backwards.

    F147 hangs the stopwatch on the very same object, and that is the whole design: the
    phases are timed under the names they are ANNOUNCED under, so the breakdown in the
    run log and the caption on the bar can never drift apart. It follows that timing
    happens with no callback at all — every method here already works that way.

    F205 is what that design cost once the passes it covered stopped being alike: the
    three model passes re-`start`ed ONE phase over three candidate lists, so one bucket
    held three prices that differ threefold. They now announce three names — still one
    name each for the caption and for the stopwatch, which is the part of the design that
    holds.

    F166 moved the stopwatch itself into `runlog.StagePhases` and kept that design
    intact: the same `enter`/`start` call still drives both the caption and the clock,
    one after the other. What changed is WHEN the clock is read out — as the stage
    goes rather than at its end — and that the object is registered under the stage
    name, so a run cut short in the middle still leaves the phases that finished.
    """

    def __init__(self, progress: ProgressCB | None,
                 stage: str = CLASSIFY_STAGE) -> None:
        self._progress = progress
        phase = getattr(progress, "phase", None)
        self._phase: PhaseCB | None = phase if callable(phase) else None
        self._current: str | None = None
        self._total: int | None = None
        self._log = track_phases(stage)

    def count(self, name: str, units: int) -> None:
        """Add `units` to what phase `name` has processed (F147).

        Separate from `enter` because the number is rarely known when the phase opens:
        the CLIP phase begins at the top of a chunk and only decides a few lines later
        how many of its frames actually need encoding. Only ever called for a phase the
        stage has entered — a counter alone must not conjure a line for work that did
        not happen.
        """
        self._log.count(name, units)

    def log_timings(self) -> None:
        """Write out whatever the phases still hold (F147/F166).

        Most of the breakdown has been written on the way already; what is left here is
        the pass that was running when the stage reached its end. Still called at the
        exits rather than from a `finally`, because the broken paths are not this
        object's business any more: `stage_timer` closes the phases it registered, and
        it is the one that knows whether the stage failed, was cancelled or finished.
        """
        self._log.close()

    def enter(self, name: str) -> None:
        """Relabel to phase `name`, keeping the counter as it is."""
        self._log.enter(name)
        if name == self._current:
            return
        self._current = name
        if self._phase is not None:
            self._phase(name)

    def start(self, name: str, total: int) -> None:
        """Enter a phase that counts its OWN items: caption and denominator together."""
        self._total = total
        self._current = None
        self._log.start(name, total)
        self.enter(name)
        self.step(0)

    def step(self, done: int) -> None:
        self._log.step(done, self._total)
        if self._progress is not None:
            self._progress(done, self._total)


# --- F128: the CLIP vector, kept ----------------------------------------------------

# The stored element type, and it is part of the file format rather than a local choice:
# a reader on another machine has to get back the numbers this machine wrote, so the byte
# order is spelled out instead of left to the platform.
#
# float32 AND NOT float16, which is what the brief proposed for the size. The brief also
# made the format conditional on a measurement — half precision "has to be checked, not
# taken on trust" — and the measurement (tests/test_clip_embeddings.py) says it does not
# hold: over 256 unit vectors of the real width, 18 of 20 queries come back in a different
# order in float16, and at 2 000 vectors all 20 do. The reordering is small in score
# (max |delta| 3e-5 of a cosine) and it is always a pair the format cannot tell apart, but
# the rule was pre-committed for exactly this outcome, so it is followed rather than argued
# with: float32 stores the encoder's own numbers and reproduces its ranking exactly.
# The price is the table, doubled — ~60 MB per 20 000 photos instead of ~30, ~920 MB at
# 300 000 instead of ~460. Same wire format as `faces.embedding`, which is little-endian
# float32 for the same reason.
_EMBEDDING_DTYPE = np.dtype("<f4")

# paths -> the image feature of each path, in the same order; None where the frame did not
# encode. The real source is `landmarks.CachingFeatureClassifier.features` — the cache the
# scoring call has just filled, NOT a second encode, which is the whole economy of this
# feature. A classifier without such a method (a plain function in a test) simply hands
# back nothing and no vector is stored.
FeatureSource = Callable[[list[str]], list[np.ndarray | None]]

_EMBEDDING_UPSERT = """INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           model = excluded.model, dim = excluded.dim,
                           vec = excluded.vec, updated_at = excluded.updated_at"""


def embedding_model(s: NamingSettings) -> str:
    """What produced a vector: the open_clip model AND its weights, as one name.

    The weights belong in the name as much as the architecture does — the same
    `ViT-L-14-quickgelu` loaded with `openai` and with a laion checkpoint produces two
    incomparable spaces, and a row that recorded only the architecture would let them mix
    silently. A mismatch means recompute; see `read_clip_embeddings` for the reading side.
    """
    return f"{s.clip_model}/{s.clip_pretrained}"


def pack_embedding(vec: np.ndarray) -> bytes:
    """A CLIP feature -> the stored blob: L2-normalized, float32, little-endian.

    Normalizing HERE and not in every consumer is the point of doing it at all: with unit
    vectors cosine similarity is a dot product, so a search does one matmul and no
    per-query normalization. The encoder already returns unit vectors, and this does it
    again anyway — the cost is a norm over 768 numbers and the guarantee is worth more than
    the trust. A zero vector (no direction to preserve) is stored as it is rather than
    divided by zero; it can only come from a caller that made one up.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.astype(_EMBEDDING_DTYPE).tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    """The stored blob -> a float32 vector, ready to be dotted with another one.

    A copy rather than the buffer itself: `np.frombuffer` gives a read-only view over
    memory that belongs to the sqlite row, and a consumer that stacks a few thousand of
    those has no reason to care which of them may be written to.
    """
    return np.frombuffer(blob, dtype=_EMBEDDING_DTYPE).astype(np.float32)


def read_clip_embeddings(conn: sqlite3.Connection, model: str,
                         file_ids: Sequence[int] | None = None,
                         ) -> dict[int, np.ndarray]:
    """Stored vectors OF THIS MODEL by file_id — the model filter is not optional.

    A consumer that read every row regardless of `model` would mix two incomparable
    spaces and return plausible nonsense that nothing in the output marks as wrong, so the
    filter lives here, in the one function that reads the table, rather than in each
    caller. Rows of another model are absent from the result exactly like frames that were
    never encoded — and the stage recomputes them on its next run.

    Chunked over `file_ids` for the reason `read_frame_quality` gives: asking about a whole
    collection is the expected case and SQLite has a ceiling on bound parameters.
    """
    sql = "SELECT file_id, vec FROM clip_embeddings WHERE model = ?"

    def rows(cursor: sqlite3.Cursor) -> dict[int, np.ndarray]:
        return {int(r["file_id"]): unpack_embedding(r["vec"]) for r in cursor}

    if file_ids is None:
        return rows(conn.execute(sql, (model,)))
    out: dict[int, np.ndarray] = {}
    for part in batched(list(file_ids), 500):
        out.update(rows(conn.execute(
            f"{sql} AND file_id IN ({','.join('?' * len(part))})",
            (model, *part))))
    return out


class _EmbeddingPass:
    """The F128 half of `classify`: keep the vector the stage has already paid for.

    Owns the same three things the quality half owns and nothing more: which frames want a
    vector this run (its own incrementality, on `clip_embeddings.model`), where the vector
    comes from (the classifier's cache, never a new encode), and the rule that only a
    personal photograph gets a row. It writes on the caller's thread inside the caller's
    transaction — SQLite stays single-writer, as everywhere in this stage.

    Staleness is the MODEL and only the model. That is what makes a stored vector unusable
    rather than merely old, and it is the one thing a row can be checked against without a
    column that repeats what `files` already knows; a frame whose content changed is
    revisited by the same rule that has it reclassified, and re-encoding it then costs the
    one CLIP call the stage was making anyway.
    """

    def __init__(self, conn: sqlite3.Connection, model: str, ids: set[int],
                 source: FeatureSource | None, now: str, stats: JunkStats,
                 enabled: bool) -> None:
        self._conn = conn
        self._model = model
        self._ids = ids
        self._source = source
        self._now = now
        self._stats = stats
        self._enabled = enabled

    def wanted(self, file_id: int) -> bool:
        """Does this frame want a vector in this run? (its own incrementality)"""
        return file_id in self._ids

    def needs_clip(self) -> bool:
        """Does this half need the CLIP row of a frame at all? — only if it has work."""
        return bool(self._ids)

    def vectors(self, paths: list[str]) -> list[np.ndarray | None]:
        """The features of the paths just scored, from the classifier's cache."""
        if self._source is None:
            return [None] * len(paths)
        return self._source(paths)

    def store(self, file_id: int, vec: np.ndarray | None, verdict: str | None) -> None:
        """Write the vector of one frame — or drop the row this frame must not have.

        Two things leave a frame without a row, and neither is a NULL: a verdict that is
        not a personal photograph (F120 — the embedding of a screenshot is noise in a
        search over personal photos, and a row a previous run left is removed), and a frame
        that did not encode at all.

        A frame that did not encode is therefore selected again by every later run. That is
        the accepted price of "no row rather than a NULL row": the alternative is a marker
        row saying "this one is hopeless", which is a claim about a file that may simply
        have been on a disconnected drive. The retry costs one decode attempt on a file the
        stage is already walking, and only for files that are actually broken.
        """
        if verdict is not None and verdict != QUALITY_VERDICT:
            self._conn.execute(
                "DELETE FROM clip_embeddings WHERE file_id = ?", (file_id,))
            return
        if vec is None:
            return
        self._conn.execute(_EMBEDDING_UPSERT, (file_id, self._model, int(np.size(vec)),
                                               pack_embedding(vec), self._now))
        self._stats.embeddings_stored += 1

    def purge(self) -> None:
        """Drop the rows of everything this run decided is not a personal photograph.

        The same statement, and for the same reason, as the `frame_quality` purge below
        it: incrementality skips a frame whose row already looks current, so a collection
        embedded before this rule would keep its screenshots PRECISELY because they are up
        to date, and the deep tier reclassifies frames after the fast pass wrote them.
        """
        if not self._enabled:
            return  # `features.store_embeddings` off: the table is not this run's business
        self._conn.execute(
            "DELETE FROM clip_embeddings WHERE file_id IN"
            " (SELECT file_id FROM media_class WHERE verdict != ?)", (QUALITY_VERDICT,))


# --- F141: the search index, a second vector with a model of its own ------------------
#
# The vector above is the CLASSIFICATION vector and it stays that. This one is the SEARCH
# vector, and the two are separate because the measurement said so twice over.
#
# `ViT-L-14` is accurate in English and does not work in Russian: over 217 hand-labelled
# judgements on 8 concepts it gives 22% precision at top-5 against 98% for
# `xlm-roberta-base-ViT-B-32`, and four of the eight concepts (cake, food, mountains,
# children) return nothing at all. The multilingual model is not weaker in English either
# (95% against 98% at top-5, three points on forty judgements), which was the objection
# that had to be ruled out before its smaller image tower could be trusted.
#
# The cheap fix — swap `naming.clip.*` — is the one thing that must not happen. The
# landmark threshold 0.85 with corroboration (F75), the animal threshold 0.70 (F122), the
# cascade selection at 0.50 (F130) and the junk classification are all calibrated on L14's
# numbers; a swap invalidates every one of them at once, and nothing in the pipeline would
# say so. So the search side pays for a pass of its own: ~10.5 minutes per 20 000 frames,
# behind `features.search_index`, off until a person switches it on.
#
# Everything else is `clip_embeddings`' arrangement, deliberately unchanged: the model
# name is written into every row (F128's rule, not weakened — a vector that does not say
# what computed it is rubbish that looks like data), a mismatch means recompute rather
# than use, the wire format is L2-normalized little-endian float32, and the population is
# canonical photographs only (F120).

_SEARCH_UPSERT = """INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(file_id) DO UPDATE SET
                        model = excluded.model, dim = excluded.dim,
                        vec = excluded.vec, updated_at = excluded.updated_at"""

# Which frames this pass owes a vector: a canonical photograph (F120 — the same population
# `clip_embeddings` and `frame_quality` have) whose stored vector is missing or was
# computed by another model. A frame with no verdict yet is included for the reason the
# F128 selection includes it: a first run has classified nothing, and the purge below
# settles whatever this run turns out to have decided.
_SEARCH_PENDING_SQL = """SELECT f.id, f.path
    FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
                 LEFT JOIN search_embeddings se ON se.file_id = f.id
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
      AND (mc.verdict IS NULL OR mc.verdict = ?)
      AND (se.model IS NULL OR se.model != ?)
    ORDER BY f.id"""


def search_index_model(cfg: Config) -> str:
    """What the search side computes with — `features.search_model`, as one name.

    A key of its own and not `naming.clip.*`: that model is what the classification
    thresholds are calibrated on, and the whole feature is the refusal to change it. The
    string is `<architecture>/<weights>` in the same spelling `embedding_model` writes, so
    a row of either table answers "which model" the same way.
    """
    features = getattr(cfg, "features", None) or FeaturesConfig()
    return str(getattr(features, "search_model", FeaturesConfig.search_model))


def search_index_enabled(cfg: Config) -> bool:
    """`features.search_index` — is the second pass wanted at all? (default: no)"""
    features = getattr(cfg, "features", None) or FeaturesConfig()
    return bool(getattr(features, "search_index", False))


def search_index_settings(s: NamingSettings, model: str) -> NamingSettings:
    """The CLIP settings of the SEARCH model: `s` with its model name replaced.

    Everything else about the encode is deliberately inherited — the batch size, the
    decode pool, and through them the shared preview cache — because this pass reads the
    very same previews the classification pass does (the brief's requirement, and the
    reason its ten minutes are ten and not thirty). Only the pair (architecture, weights)
    differs, and it is split here, once, so no caller has to know the name is a pair.
    """
    architecture, _, weights = model.partition("/")
    return replace(s, clip_model=architecture, clip_pretrained=weights)


def read_search_embeddings(conn: sqlite3.Connection, model: str,
                           file_ids: Sequence[int] | None = None,
                           ) -> dict[int, np.ndarray]:
    """Stored SEARCH vectors of this model by file_id — the same rule, the other table.

    A separate function rather than a table argument on `read_clip_embeddings`: the model
    filter is the safety property of both (mixing two spaces produces a plausible ranking
    nothing marks as wrong), and a parameter that selects which table to apply it to is
    one call site away from being passed the wrong one.
    """
    sql = "SELECT file_id, vec FROM search_embeddings WHERE model = ?"

    def rows(cursor: sqlite3.Cursor) -> dict[int, np.ndarray]:
        return {int(r["file_id"]): unpack_embedding(r["vec"]) for r in cursor}

    if file_ids is None:
        return rows(conn.execute(sql, (model,)))
    out: dict[int, np.ndarray] = {}
    for part in batched(list(file_ids), 500):
        out.update(rows(conn.execute(
            f"{sql} AND file_id IN ({','.join('?' * len(part))})",
            (model, *part))))
    return out


def search_image_encoder(s: NamingSettings) -> FeatureSource:  # pragma: no cover — ML
    """The image tower of the search model: paths -> a unit vector each, None where not.

    `landmarks.clip_classifier` and nothing new: it already decodes through the shared
    preview cache, in a pool, in one GPU batch, and returns None for a frame that would
    not decode. What this pass needs is its `encode` half alone — there are no prompts
    here, the vector IS the answer — so the classifier is built and its encoder taken.
    """
    classifier = clip_classifier(s)
    if not isinstance(classifier, CachingFeatureClassifier):
        raise TypeError(f"clip_classifier не отдаёт энкодер: {type(classifier).__name__}")
    return classifier.encode


class _SearchIndexPass:
    """F141: the second CLIP pass, the one that pays for itself in Russian queries.

    The only pass of this stage that ENCODES IMAGES AGAIN, and every property of it
    follows from that being expensive. It runs last, over the verdicts everything above
    has settled, so a frame the deep tier has just called a screenshot is never encoded;
    it is incremental on `search_embeddings.model`, so switching the model recomputes and
    a repeated run does nothing; and it is built lazily — a run with no frames to encode
    loads no weights, which is what makes leaving the toggle on cheap for a collection
    that is already indexed.

    The failures are the same shape as every other optional half here. A model that will
    not build leaves the run exactly as it was (the search index simply stays as it is,
    and search says so rather than ranking with the classification vectors); a chunk that
    will not encode is logged and skipped, and its frames are selected again next run,
    which is the same "no row rather than a wrong row" rule `_EmbeddingPass.store` states.
    """

    def __init__(self, conn: sqlite3.Connection, model: str,
                 encoder: Callable[[], FeatureSource], batch_size: int, now: str,
                 stats: JunkStats, enabled: bool) -> None:
        self._conn = conn
        self._model = model
        self._encoder = encoder
        self._batch = max(1, int(batch_size))
        self._now = now
        self._stats = stats
        self._enabled = enabled

    def run(self, report: _PhaseProgress) -> None:
        """Purge what must not be indexed, then encode what has no current vector."""
        if not self._enabled:
            return  # `features.search_index` off: the table is not this run's business
        self._purge()
        todo = [(int(r["id"]), str(r["path"])) for r in self._conn.execute(
            _SEARCH_PENDING_SQL, (QUALITY_VERDICT, self._model))]
        if not todo:
            return
        try:
            encode = self._encoder()
        except Exception as exc:  # noqa: BLE001 — the index is optional, must not crash
            _log.warning(
                "junk: модель поискового индекса недоступна (%s) — таблица "
                "search_embeddings остаётся как была, поиск скажет об этом сам", exc)
            return
        report.start(CLASSIFY_PHASE_SEARCH, len(todo))
        done = 0
        for chunk in batched(todo, self._batch):
            self._store(chunk, self._encode(encode, chunk))
            done += len(chunk)
            report.count(CLASSIFY_PHASE_SEARCH, len(chunk))
            report.step(done)

    def _encode(self, encode: FeatureSource,
                chunk: Sequence[tuple[int, str]]) -> list[np.ndarray | None]:
        """One batch through the tower; a batch that raises costs the run nothing.

        The encoder answers None per unreadable frame on its own — this catches the
        failure of the CALL, which is a batch of frames and not one of them, and turns it
        into the same "no vector" every one of them would have got individually.
        """
        try:
            return list(encode([path for _file_id, path in chunk]))
        except Exception as exc:  # noqa: BLE001 — one batch, not the stage
            _log.warning("junk: поисковый индекс — не закодирована пачка из %d кадров "
                         "(%s), они попадут в следующий прогон", len(chunk), exc)
            return [None] * len(chunk)

    def _store(self, chunk: Sequence[tuple[int, str]],
               vectors: Sequence[np.ndarray | None]) -> None:
        """Write the vectors of one batch, on the caller's thread (single writer)."""
        with self._conn:
            for (file_id, _path), vec in zip(chunk, vectors):
                if vec is None:
                    continue  # did not encode: no row, and selected again next run
                self._conn.execute(
                    _SEARCH_UPSERT, (file_id, self._model, int(np.size(vec)),
                                     pack_embedding(vec), self._now))
                self._stats.search_vectors_stored += 1

    def _purge(self) -> None:
        """Drop the rows of everything this run decided is not a personal photograph.

        The same statement and the same reason as `_EmbeddingPass.purge`: incrementality
        skips a frame whose row already looks current, so a screenshot indexed before its
        verdict changed would stay in the search index PRECISELY because its vector is up
        to date. Runs whenever the toggle is on, including on the runs that have no frame
        left to encode — that is the state a collection settles into.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM search_embeddings WHERE file_id IN"
                " (SELECT file_id FROM media_class WHERE verdict != ?)", (QUALITY_VERDICT,))


@dataclass
class JunkStats:
    total: int = 0        # canonical photos in total
    processed: int = 0    # processed in this run (rows whose tier != the active one)
    # F68: rows skipped as already handled by the active tier (total - processed).
    # On a repeated run without input changes it equals `total` and processed is 0 —
    # the observable sign that incrementality works.
    skipped_incremental: int = 0
    clip_used: bool = False
    by_verdict: dict[str, int] = field(default_factory=dict)
    vlm_candidates: int = 0  # #14/V1: files selected for the VLM (doc/product zone)
    vlm_applied: int = 0     # of those, actually reclassified by the VLM (without errors)
    # F113, the frame-quality cascade. quality_rows counts the rows the cheap tiers wrote
    # (sharpness always, pets when the toggle is on). F186 removed the pair next to it —
    # the uncertain band and how much of it the model answered — with the question they
    # priced.
    quality_rows: int = 0
    # The animals this run ended up marking — AFTER the F130 check, if it ran: the number
    # a user compares against the folder they get, not an intermediate the cascade
    # discarded on its way there.
    pets_found: int = 0
    # F130: frames shown to the pet check (`pet_score >= pet_candidate_threshold`) and how
    # many of them came back with an answer that parsed. The pair is what prices the check.
    pet_candidates: int = 0
    pet_verified: int = 0
    # F128: vectors written into `clip_embeddings` in this run. On a repeated run it is 0
    # and the table is unchanged — the observable sign that this half is incremental too.
    embeddings_stored: int = 0
    # F141: and the same number for the SEARCH index, counted apart from it because the
    # two are not the same cost. This one is a second CLIP pass — ~10.5 minutes per 20 000
    # frames — so "how many frames did that pass actually encode" is what prices the
    # toggle, and on a repeated run it is 0 like the counter above it.
    search_vectors_stored: int = 0
    # F140: photographs the rescue score was computed for, how many of them cleared
    # `features.junk_rescue_threshold` (the frames the deep tier is asked about), and how
    # many verdicts the model actually moved off `photo`. The first two price the gate the
    # way pet_candidates/pet_verified price the animal check; the third is the whole point
    # of the feature, and it is deliberately not the same number as the second — a
    # candidate the model calls a photograph stays one.
    junk_scored: int = 0
    junk_candidates: int = 0
    junk_rescued: int = 0
    # F154: the animal detector — frames the query selected, how many of them the model
    # actually looked at (the rest already had an answer from an earlier run, which is what
    # makes a repeated run cost nothing), and on how many an animal was found. The first
    # two price the cascade the way pet_candidates/pet_verified price the F130 check; the
    # last is the population of the animal slice this feature exists to widen.
    detector_candidates: int = 0
    detector_examined: int = 0
    detector_found: int = 0


def _non_photo_ids(conn: sqlite3.Connection, file_ids: Sequence[int]) -> set[int]:
    """Which of these frames `media_class` says are NOT personal photographs.

    One query over an indexed column, and it is asked AFTER the deep tier rather than
    trusted from the fast pass: the deep tier (and, since F140, the rescue) is the one
    thing that can move a verdict while a candidate list is standing, and `document` is
    precisely the class `vlm.exclude_classes` protects by default. A frame with no verdict
    at all is not in the answer — a first run has classified nothing yet, and that is not a
    reason to withhold it.
    """
    out: set[int] = set()
    for part in batched(list(file_ids), 500):
        out.update(int(r["file_id"]) for r in conn.execute(
            "SELECT file_id FROM media_class WHERE verdict != ? AND file_id IN"
            f" ({','.join('?' * len(part))})", (QUALITY_VERDICT, *part)))
    return out


class _QualityPass:
    """The frame-quality half of `classify`, kept out of its loop (F113).

    Owns three things and nothing else: which frames need quality work this run (its own
    incrementality, on `frame_quality.source`, mirroring how junk uses `media_class.tier`),
    what the cheap tiers say about a frame, and which frames that leaves for the pet check.
    It writes on the caller's thread, inside the caller's transaction — SQLite stays
    single-writer, as everywhere in this stage.

    F186 left it with ONE model question. The frame-quality prompt, its candidate band and
    the scope that narrowed it are gone (the eyes are geometry now, F179), so the pet check
    is the only thing here that can raise a model at all.
    """

    def __init__(self, conn: sqlite3.Connection, q: QualitySettings,
                 sharpness: SharpnessFn, source: str, ids: set[int],
                 now: str, stats: JunkStats,
                 pet_ask: PetAskFn | None = None, workers: int = 1) -> None:
        self._conn = conn
        self._q = q
        self._sharpness = sharpness
        self._source = source
        self._ids = ids
        self._now = now
        self._stats = stats
        # F130: the pet check and its candidate list — a CLIP score above
        # `pet_candidate_threshold`, which is the one population this pass still selects.
        self._pet_ask = pet_ask
        # F206: the preparation threads of that check — `vlm.workers`, the same knob and
        # the same pool the deep tier uses. Defaulted to 1 (the serial path) so a caller
        # that builds this pass by hand keeps the behaviour it had.
        self._workers = workers
        # (file_id, path, label the cheap tier wrote) — the third field is what keeps
        # `stats.pets_found` the FINAL count when an answer moves the label.
        self._pet_candidates: list[tuple[int, str, str | None]] = []
        # F140: (file_id, path) of every frame this run actually wrote a row for — the
        # population of the rescue score, which is the same one by construction: a score
        # belongs to a `frame_quality` row, and a row exists for personal photographs only.
        self._measured: list[tuple[int, str]] = []
        # F155: where the faces of the frames this run measures are. Read in one query
        # rather than one per frame, and only when the first frame actually asks — a
        # collection whose quality half is up to date must not pay for the query at all.
        self._faces: dict[int, FaceBoxes] | None = None

    @property
    def measured(self) -> list[tuple[int, str]]:
        """Frames whose row this run wrote — what `_JunkRescuePass` scores (F140)."""
        return self._measured

    def wanted(self, file_id: int) -> bool:
        """Does this frame need quality work in this run? (its own incrementality)"""
        return file_id in self._ids

    def _faces_of(self, file_id: int) -> FaceBoxes:
        """The face boxes of one frame — the whole map is read on the first call (F155)."""
        if self._faces is None:
            self._faces = read_face_boxes(self._conn, sorted(self._ids))
        return self._faces.get(file_id, NO_FACES)

    def needs_clip(self) -> bool:
        """Does the quality half need the CLIP row of a frame at all?

        One thing wants it: the pet group, which writes `pet`/`pet_score`. F130 adds no
        second reason (the check reads the score of that same group), and F186 removed the
        one there used to be — the subject score that decided half of the uncertainty band.
        """
        return self._q.pets

    def measure(self, file_id: int, path: str, probs_row: np.ndarray | None,
                verdict: str | None = None) -> None:
        """The cheap tiers for one frame: measure, write the row, note the candidates.

        F120: a frame this run decided is NOT a personal photograph is dropped here
        instead of measured, and any row a previous run left for it is removed — the
        first live run wrote 24 196 rows over the whole collection, and the answers on
        screenshots, products and documents were the noise that made the signal unusable.

        That same guard is what keeps `vlm.exclude_classes` out of the candidate list
        below: this population is personal photographs, and the excludable classes
        (document, product, screenshot, meme — `photo` is not one of them, see
        config.VLM_EXCLUDABLE_CLASSES) have already left by the time a candidate is noted.
        """
        if verdict is not None and verdict != QUALITY_VERDICT:
            self._conn.execute("DELETE FROM frame_quality WHERE file_id = ?", (file_id,))
            return
        # F155/F179: everything one decode yields — the whole frame, the sharpest face in
        # it where the faces stage found one, and how open the eyes of the largest face
        # are. The face crop is what actually separates a blurred frame from a detailed one
        # (see the module docstring); the whole-frame number is unchanged.
        measured = self._sharpness(path, self._faces_of(file_id))
        sharpness = measured.frame
        pet: str | None = None
        pet_score: float | None = None
        if self._q.pets and probs_row is not None:
            pet, pet_score = pet_verdict(probs_row, self._q.pet_threshold)
        self._conn.execute(_QUALITY_UPSERT, (file_id, sharpness, measured.face,
                                             measured.eyes, pet, pet_score,
                                             self._source, self._now))
        self._stats.quality_rows += 1
        self._measured.append((file_id, path))
        if pet is not None:
            self._stats.pets_found += 1
        # F130: the label written above is the UNVERIFIED one — `pet_score >=
        # pet_threshold` — and that is deliberate. It is what the frame keeps if the model
        # never answers about it, so the fallback is already in the table before anything
        # expensive is attempted, and a run that dies mid-check leaves today's answer
        # rather than half of tomorrow's.
        if (self._pet_ask is not None and pet_score is not None
                and pet_score >= self._q.pet_candidate_threshold):
            self._pet_candidates.append((file_id, path, pet))

    def _reclassified(self) -> set[int]:
        """Candidates that stopped being personal photographs while the list was standing.

        The list is built during the fast pass and asked after the deep tier, which is the
        one thing that can move a verdict in between — a frame the fast tier called a
        photograph can come back a `product` or a `document`. Its row is purged at the end
        of the stage either way, but the question would already have been asked by then,
        and `document` is precisely the class `vlm.exclude_classes` protects by default.
        One query over an indexed column is a cheap way not to show the model a passport.
        """
        return _non_photo_ids(
            self._conn, [fid for fid, _path, _before in self._pet_candidates])

    def ask_pets(self, report: _PhaseProgress) -> None:
        """The pet check over its candidates — one frame per call, one word back (F130).

        Three things can leave a frame unanswered, and all three mean the SAME thing: the
        row keeps the label the CLIP threshold gave it and `pet_vlm` stays NULL. The model
        raised on this frame; the model answered something nobody could read; the model was
        never built at all (the caller's graceful fallback). None of them is a "no" — a
        cheap tier that survives the failure of an expensive one is the rule this whole
        stage is built on, and guessing here would delete a label CLIP was right about.

        The phase is CLASSIFY_PHASE_PETS_VLM — its own name since F205, with its own
        caption in all three languages. It shared CLASSIFY_PHASE_VLM with the deep tier
        while the two cost the same per frame, and a shared name means the estimate can
        only charge one of them at the other's rate. The candidate list is known before
        the loop starts, so the bar reports a real (done, total) over it. The count
        reported is the list AFTER `_reclassified` has trimmed it, so the bar counts
        questions that will actually be asked.

        F206: THE ANSWERS COME THROUGH `_vlm_labels`, i.e. through the machinery the deep
        tier has used since F101 — `vlm.workers` threads decode and preprocess frames
        while this thread runs the model and writes. That is the whole change here: the
        0.42 frames/s the phase name above was measured at was this loop asking one frame
        at a time while the card idled through every decode, 116 minutes a run over the
        4 281 frames the two back-half questions see. Nothing about the question moves —
        the same prompt, the same token budget, the same input size, one frame per call —
        so no verdict may move either, and the order guarantee is the tier's own: answers
        arrive in the CANDIDATE order (a FIFO of futures), not in the order the
        preparations happen to finish, because an answer written against its neighbour's
        file is worse than a slow pass.
        """
        if self._pet_ask is None or not self._pet_candidates:
            return
        gone = self._reclassified()
        candidates = [c for c in self._pet_candidates if c[0] not in gone]
        if not candidates:
            return
        self._stats.pet_candidates = len(candidates)
        report.start(CLASSIFY_PHASE_PETS_VLM, len(candidates))
        report.count(CLASSIFY_PHASE_PETS_VLM, len(candidates))
        # closing(): the preparation threads live exactly as long as the pass, not until
        # the garbage collector gets round to the generator holding them (F101).
        answers = _vlm_labels(self._pet_ask, [path for _fid, path, _b in candidates],
                              self._workers)
        with closing(answers), self._conn:
            for i, ((file_id, _path, before), item) in enumerate(
                    zip(candidates, answers)):
                if isinstance(item, BaseException):
                    # A frame the model raised on keeps the label the CLIP threshold gave
                    # it — the same contract this loop has always had, only the raising
                    # moved into the generator.
                    _log.warning(
                        "junk: VLM-проверка животных не ответила по file_id=%s (%s) — "
                        "оставляю метку по порогу CLIP", file_id, item)
                    answer = ""
                else:
                    answer = item
                seen = parse_pet_answer(answer)
                if seen is not None:
                    # The same rule the fast half used, now with the model's word in hand.
                    # The score is passed as None because it cannot change the outcome —
                    # an answer outranks it — and routing both cases through one function
                    # is what keeps the column and the label from disagreeing.
                    after = pet_label(seen, None, self._q.pet_threshold)
                    self._conn.execute(
                        _PET_ANSWER_UPDATE, (seen, after, self._now, file_id))
                    self._stats.pet_verified += 1
                    self._stats.pets_found += (after is not None) - (before is not None)
                report.step(i + 1)


class _JunkRescuePass:
    """F140: the zero-shot score over the stored vectors, and the check it selects for.

    Owns the three things the other halves own and nothing else: which frames are scored
    (the ones this run wrote a `frame_quality` row for — the score belongs to that row, and
    that row exists for personal photographs only), where the numbers come from (the
    `clip_embeddings` table F128 filled, never a new pass over any image), and who is shown
    to the model. It writes on the caller's thread, inside the caller's transaction.

    Two failures leave a frame alone rather than junked, and they are the reason this is a
    gate and not a classifier: no stored vector (the score stays NULL and the frame is no
    candidate — a heuristics-only collection or `store_embeddings: false` simply does not
    have this feature), and no answer from the model (the fast verdict stands). Neither is
    ever read as "this is junk": the score alone is right ~85% of the time, and the F130
    lesson is that such a signal applied directly makes a better baseline worse.

    The text encoder is built LAZILY, inside `run`, and only when there are frames to
    score: it loads a model, and a run with nothing to ask must not pay for one.
    """

    def __init__(self, conn: sqlite3.Connection, q: QualitySettings, model: str,
                 encoder: Callable[[], TextEncoder], ask: JunkAskFn | None,
                 now: str, tier: str, stats: JunkStats, workers: int = 1) -> None:
        self._conn = conn
        self._q = q
        self._model = model
        self._encoder = encoder
        self._ask = ask
        self._now = now
        self._tier = tier
        self._stats = stats
        # F206: the preparation threads of the question below — `vlm.workers`, the same
        # knob the deep tier and the animal check use. Defaulted to 1 (the serial path)
        # for a caller that builds this pass by hand.
        self._workers = workers

    def _text_features(self) -> np.ndarray | None:
        """The prompts as a matrix of unit rows; None — the encoder could not be had.

        The same graceful fallback the model halves of this stage have, and for the same
        reason: an optional signal that cannot be computed must cost the run nothing. The
        rows are not cached between runs — five short strings through a text tower is
        nothing next to the pass that has just finished.
        """
        try:
            rows = np.asarray(self._encoder()(junk_rescue_prompts()), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — the stage must survive it
            _log.warning(
                "junk: текстовый энкодер для отбора мусора недоступен (%s) — "
                "оценка junk_score не считается, вердикты не меняются", exc)
            return None
        return unit_rows(rows)

    def run(self, frames: Sequence[tuple[int, str]], report: _PhaseProgress) -> None:
        """Score `frames`, store the scores, then ask the model about the candidates."""
        if not self._q.junk_rescue or not frames:
            return
        features = self._text_features()
        if features is None:
            return
        candidates = self._score(frames, features)
        if not candidates:
            return
        self._stats.junk_candidates = len(candidates)
        self._reclassify(candidates, report)

    def _score(self, frames: Sequence[tuple[int, str]],
               features: np.ndarray) -> list[tuple[int, str]]:
        """Write `junk_score` for every frame that has a vector; return the candidates.

        A frame the deep tier has since moved off `photo` is skipped entirely — it is junk
        already, its row is about to be purged, and scoring it would only inflate the
        counters the gate is priced by (brief test 7).
        """
        ids = [file_id for file_id, _path in frames]
        vectors = read_clip_embeddings(self._conn, self._model, ids)
        gone = _non_photo_ids(self._conn, ids)
        candidates: list[tuple[int, str]] = []
        with self._conn:
            for file_id, path in frames:
                if file_id in gone:
                    continue
                vec = vectors.get(file_id)
                score = None if vec is None else junk_rescue_score(vec, features)
                if score is None:
                    continue  # no vector of this model: NULL means "not computed"
                self._conn.execute(_JUNK_SCORE_UPDATE, (score, self._now, file_id))
                self._stats.junk_scored += 1
                if score >= self._q.junk_rescue_threshold:
                    candidates.append((file_id, path))
        return candidates

    def _reclassify(self, candidates: list[tuple[int, str]],
                    report: _PhaseProgress) -> None:
        """One question per candidate; only the model's answer moves a verdict.

        With the deep tier off there is no asker at all and this returns immediately: the
        scores are stored, the candidates are counted, and not one verdict of the run has
        changed — which is the promise the feature is switched on under.

        `photo` back from the model is an answer, not a refusal: the frame was a candidate
        because it LOOKS like a screenshot, and the model saying otherwise is exactly what
        the ~15% of candidates that are real photographs need. It is written as a verdict
        anyway (the source becomes `vlm`) so that a later reader can tell a frame the model
        confirmed from one it was never shown.

        F205: the phase is CLASSIFY_PHASE_RESCUE_VLM, not the deep tier's name — a price
        of its own, which the estimate can only quote if the log files it separately.

        F206: and the price is now the tier's, because the questions go through the tier's
        own pipeline (`_vlm_labels`, see `_QualityPass.ask_pets` for the measurement that
        ended the serial version of this loop). One question per candidate still, one
        frame per call still, the answers still consumed in the candidate order and still
        written from this thread inside this transaction — what changed is that the next
        candidate is being decoded while the model answers about this one.
        """
        if self._ask is None:
            return
        report.start(CLASSIFY_PHASE_RESCUE_VLM, len(candidates))
        report.count(CLASSIFY_PHASE_RESCUE_VLM, len(candidates))
        answers = _vlm_labels(self._ask, [path for _fid, path in candidates],
                              self._workers)
        with closing(answers), self._conn:
            for i, ((file_id, _path), item) in enumerate(zip(candidates, answers)):
                if isinstance(item, BaseException):
                    _log.warning(
                        "junk: VLM не ответила по кандидату file_id=%s (%s) — "
                        "остаётся вердикт быстрого яруса", file_id, item)
                    answer = ""
                else:
                    answer = item
                verdict = parse_junk_rescue_answer(answer)
                if verdict is not None:
                    self._conn.execute(_MEDIA_CLASS_UPSERT,
                                       (file_id, verdict, "vlm", None, self._now,
                                        self._tier))
                    if verdict != QUALITY_VERDICT:
                        self._stats.junk_rescued += 1
                        self._stats.by_verdict[QUALITY_VERDICT] = (
                            self._stats.by_verdict.get(QUALITY_VERDICT, 1) - 1)
                        self._stats.by_verdict[verdict] = (
                            self._stats.by_verdict.get(verdict, 0) + 1)
                report.step(i + 1)


# F154: who the animal query may rank at all. Canonical photographs the stage calls a
# photograph, plus the ones it has not classified yet (a first run has settled nothing, and
# that is not a reason to withhold a frame) — the F120 population. The detector never sees
# anything outside this list, and
# the ranking then cuts it down to `features.detector_candidates`.
_DETECTOR_POPULATION_SQL = """SELECT f.id, f.path FROM files f
    LEFT JOIN media_class mc ON mc.file_id = f.id
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
      AND (mc.verdict IS NULL OR mc.verdict = ?)
    ORDER BY f.id"""


class _DetectorPass:
    """F154: the object detector, over the candidates a query selects — never over a pass.

    The pipeline half of `detect.py`, and it owns the three things every other pass here
    owns: who the candidates are (the top `features.detector_candidates` frames of a
    zero-shot animal query over the vectors F128 already stores, inside the F120 population
    of personal photographs), which of them still need the model (its own incrementality —
    a row in `detections` written by the same detector means "already examined", and that
    covers a frame it found nothing on), and what the answer then decides
    (`frame_quality.pet`, through `detect.cascade_label`). It writes on the caller's thread
    inside its own transactions — SQLite stays single-writer, as everywhere in this stage.

    THERE IS NO CODE PATH THAT RUNS THE DETECTOR OVER EVERYTHING. That is the feature, not
    a safeguard: 83.8 ms per frame is 30.8 minutes over 22 096 photographs, for a signal
    that beats what the pipeline already has on exactly one slice out of three. Without
    stored vectors there are no candidates and the stage says so instead of falling back to
    a pass — a fallback nobody asked for is how three minutes become thirty-one.

    Three failures leave every label exactly as the cheaper tiers wrote it, and none of
    them is ever read as "no animal": no vectors to query (the reason is logged), an
    encoder or a detector that will not build (the graceful fallback every optional half of
    this stage has), and an error on one frame (that frame keeps its label and is examined
    again next run, because no row is written for it).

    Both the encoder and the detector are built LAZILY, inside `run`, and only when there
    is work: each loads a model, and a run with nothing to ask must not pay for one.
    """

    def __init__(self, conn: sqlite3.Connection, s: DetectorSettings, model: str,
                 encoder: Callable[[], TextEncoder], detector: Callable[[], DetectFn],
                 pet_threshold: float, now: str, stats: JunkStats) -> None:
        self._conn = conn
        self._s = s
        self._model = model
        self._encoder = encoder
        self._detector = detector
        self._pet_threshold = pet_threshold
        self._now = now
        self._stats = stats

    def run(self, report: _PhaseProgress) -> None:
        """Select the candidates, examine the ones nobody has, apply the label rule."""
        if not self._s.enabled:
            return
        candidates = self._candidates()
        if not candidates:
            return
        self._stats.detector_candidates = len(candidates)
        stored = self._stored(candidates)
        todo = [(file_id, path) for file_id, path in candidates if file_id not in stored]
        found = dict(stored)
        found.update(self._examine(todo, report))
        # Counted over the candidates that HAVE an answer, this run's and the stored ones
        # alike: the number a user compares against the folder they get does not depend on
        # which run happened to ask the question.
        self._stats.detector_found = sum(
            1 for boxes in found.values()
            if best_animal(boxes, self._s.threshold) is not None)
        self._relabel(candidates, found)

    def _candidates(self) -> list[tuple[int, str]]:
        """The frames the animal query ranks highest — (file_id, path), best first.

        The query runs over `clip_embeddings`, the CLASSIFICATION vectors: those are the
        rows this pipeline's prompts live in the space of, and the model filter inside
        `read_clip_embeddings` is what keeps a vector of another model out of the ranking.

        An empty table is a REASON and not an empty list (the F134 rule): "no candidates"
        and "nobody has ever computed a vector here" read identically in a count of zero,
        and only one of them is fixed by running the junk stage with
        `features.store_embeddings` on. So it is said, in the log, and nothing is asked.
        """
        rows = self._conn.execute(_DETECTOR_POPULATION_SQL, (QUALITY_VERDICT,)).fetchall()
        paths = {int(r["id"]): str(r["path"]) for r in rows}
        vectors = read_clip_embeddings(self._conn, self._model, list(paths))
        if not vectors:
            _log.warning(
                "junk: детектор объектов не запускается — в clip_embeddings нет векторов "
                "модели %s (нужен прогон junk с features.store_embeddings). Кандидатов "
                "нет, сплошного прохода по коллекции у этой стадии не бывает", self._model)
            return []
        features = self._text_features()
        if features is None:
            return []
        return [(file_id, paths[file_id])
                for file_id in rank_candidates(vectors, features, self._s.candidates)]

    def _text_features(self) -> np.ndarray | None:
        """The animal prompts as unit rows; None — the encoder could not be had.

        The same graceful fallback the model halves of this stage have, and for the same
        reason: an optional signal that cannot be computed must cost the run nothing.
        """
        try:
            rows = np.asarray(self._encoder()(list(ANIMAL_QUERY_PROMPTS)),
                              dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — the stage must survive it
            _log.warning(
                "junk: текстовый энкодер для отбора кандидатов детектора недоступен "
                "(%s) — детектор не запускается, метки животных остаются прежними", exc)
            return None
        return unit_rows(rows)

    def _stored(self, candidates: Sequence[tuple[int, str]]) -> dict[int, list[Detection]]:
        """What THIS detector already answered about these frames — incrementality.

        Keyed by the model, like every other marker in this stage: boxes from another
        detector are not this one's answer, so such a frame is examined again rather than
        trusted. A row that exists with no boxes in it is an answer too — "looked, found
        nothing" — and it is the reason a repeated run asks nothing at all.
        """
        out: dict[int, list[Detection]] = {}
        for part in batched([file_id for file_id, _path in candidates], 500):
            out.update({int(r["file_id"]): unpack_boxes(r["boxes"])
                        for r in self._conn.execute(
                            "SELECT file_id, boxes FROM detections WHERE model = ?"
                            f" AND file_id IN ({','.join('?' * len(part))})",
                            (self._s.model, *part))})
        return out

    def _examine(self, todo: Sequence[tuple[int, str]],
                 report: _PhaseProgress) -> dict[int, list[Detection]]:
        """Run the detector over the frames that have no answer yet, and store what it saw.

        A frame the model raises on gets NO ROW: it keeps whatever label it had and is
        examined again next run. That is the same "no row rather than a wrong row" rule
        `_EmbeddingPass.store` states, and here it also keeps one bad frame from being
        recorded as a frame with no animal on it.
        """
        if not todo:
            return {}
        try:
            examine = self._detector()
        except Exception as exc:  # noqa: BLE001 — the cascade is optional, must not crash
            _log.warning(
                "junk: детектор объектов недоступен (%s) — метки животных остаются "
                "за CLIP и F130-каскадом", exc)
            return {}
        report.start(CLASSIFY_PHASE_DETECT, len(todo))
        report.count(CLASSIFY_PHASE_DETECT, len(todo))
        seen: dict[int, list[Detection]] = {}
        with self._conn:
            for i, (file_id, path) in enumerate(todo):
                try:
                    boxes = list(examine(path))
                except Exception as exc:  # noqa: BLE001 — one frame, not the stage
                    _log.warning(
                        "junk: детектор не ответил по file_id=%s (%s) — кадр остаётся "
                        "с прежней меткой и попадёт в следующий прогон", file_id, exc)
                    # Outside the `else` for the F100 reason: a frame the model failed on
                    # is a frame this pass is done with, and a bar one short of its total
                    # for good is worse than an honest step.
                    report.step(i + 1)
                    continue
                kept = animal_boxes(boxes, STORE_FLOOR)
                best = best_animal(kept, self._s.threshold)
                self._conn.execute(_DETECTIONS_UPSERT, (
                    file_id, None if best is None else best.label,
                    None if best is None else float(best.score),
                    pack_boxes(kept), self._s.model, self._now))
                self._stats.detector_examined += 1
                seen[file_id] = kept
                report.step(i + 1)
        return seen

    def _relabel(self, candidates: Sequence[tuple[int, str]],
                 found: dict[int, list[Detection]]) -> None:
        """Write the label the cascade decides — for the frames the detector examined.

        Read before written, and only the rows that change are touched: this pass runs over
        candidates the earlier tiers have already labelled, and the great majority of them
        keep exactly what they had. `stats.pets_found` follows every change, so the number
        a run reports is the one the cascade ended on and not the fast tier's (see the
        counter's own note below for the one case where that difference is floored).

        A candidate with no `frame_quality` row is skipped rather than given one: that
        table's population is written by the quality half, under its own incrementality,
        and a detection is not a reason to invent a row with no sharpness in it. The boxes
        are stored either way — they are a fact about the frame, not about the label.
        """
        ids = [file_id for file_id, _path in candidates if file_id in found]
        rows = read_frame_quality(self._conn, ids)
        with self._conn:
            for file_id, row in rows.items():
                previous = pet_label(row.pet_vlm, row.pet_score, self._pet_threshold)
                after = cascade_label(
                    best_animal(found[file_id], self._s.threshold), examined=True,
                    verified=row.pet_vlm is not None, previous=previous,
                    animal=PET_CLASS)
                if after == row.pet:
                    continue
                self._conn.execute(_PET_DETECTOR_UPDATE, (after, self._now, file_id))
                # `pets_found` counts the animals THIS RUN ended up marking, and unlike
                # the F130 check this pass does not select from the frames the run
                # measured: on a collection that is otherwise up to date it has none. So a
                # label taken off such a frame cannot push the count below zero — there
                # was nothing there to subtract from.
                self._stats.pets_found = max(
                    0, self._stats.pets_found
                    + (after is not None) - (row.pet is not None))

    def purge(self) -> None:
        """Drop the boxes of everything this run decided is not a personal photograph.

        The same statement and the same reason as `_EmbeddingPass.purge`: incrementality
        skips a frame whose row already looks current, so a frame examined before its
        verdict moved to `document` would keep its boxes PRECISELY because they are up to
        date — and a document is the one class this project takes care not to describe.
        """
        if not self._s.enabled:
            return  # the detector is off: the table is not this run's business
        self._conn.execute(
            "DELETE FROM detections WHERE file_id IN"
            " (SELECT file_id FROM media_class WHERE verdict != ?)", (QUALITY_VERDICT,))


# --- F210: the preview of a sensitive frame does not stay on disk ---------------------
#
# Refusing to SHOW a document (slices.py withholds `thumb_url` for it) protects the
# screen and not the disk: the preview is written by the very stage that decides what the
# frame is — `decode_rgb_preview` decodes everything it is handed, deliberately, and by
# the time a verdict exists the JPEG is already in the cache. So the run that names a
# frame is the run that has to take its derivative away, exactly as it already drops the
# frame's quality row, its vector and its boxes (`_QualityPass`/`_EmbeddingPass`/
# `_DetectorPass.purge`).
#
# Which classes are sensitive is decided by the LIVE `vlm.exclude_classes` and by nothing
# else. There is no list written down here: `_JUNK_NO_PREVIEW` in the UI is the default of
# a parameter for direct calls, not a source of truth, and an EMPTY setting is an
# instruction rather than an oversight — somebody whose config says `exclude_classes: []`
# has said that no class is private, and for them this sweep must remove nothing at all.


def sweep_previews(conn: sqlite3.Connection, classes: frozenset[str]) -> int:
    """Remove the disk previews of every frame whose verdict is one of `classes`.

    Returns the number of preview files removed — a number for the caller's log, not a
    promise: a preview a reader holds open stays, and the next sweep gets it.

    An empty `classes` returns 0 without touching the database. That is the whole of
    requirement 4 of the brief: absence and emptiness are different wishes here as much as
    they are in `config._as_exclude_classes`, and the empty list is the one that says "no
    class of mine is private".
    """
    if not classes:
        return 0
    names = sorted(classes)
    rows = conn.execute(
        "SELECT f.path, f.mtime, f.size FROM files f"
        " JOIN media_class mc ON mc.file_id = f.id"
        f" WHERE mc.verdict IN ({','.join('?' * len(names))})", names).fetchall()
    removed = 0
    for r in rows:
        removed += imaging.preview_delete(r["path"], r["mtime"], r["size"])
    if removed:
        _log.info("junk: удалено превью чувствительных классов: %d (классы: %s)",
                  removed, ", ".join(names))
    return removed


def sweep_previews_for_new_classes(conn: sqlite3.Connection,
                                   before: Sequence[str], after: Sequence[str]) -> int:
    """The sweep a CHANGED `vlm.exclude_classes` calls for — over the ADDED classes only.

    The list is edited while the tool runs, and that is the moment the cleanup matters
    most: switching the protection on has to reach the previews ALREADY on disk, or the
    whole archive of documents survives it and the feature covers nothing but frames
    classified from now on.

    A class that LEFT the list sweeps nothing. Its previews are an ordinary cache entry
    again and are rebuilt on demand — there is nothing to remove and nothing lost.
    """
    return sweep_previews(conn, frozenset(after) - frozenset(before))


def classify(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    use_clip: bool = True,
    text_detector: TextFracDetector | None = None,
    text_detector_factory: TextFracDetectorFactory | None = None,
    vlm_classifier: VlmClassifyFn | None = None,
    vlm_classifier_factory: Callable[[str], VlmClassifyFn] | None = None,
    sharpness_detector: SharpnessFn | None = None,
    eye_landmarks_factory: Callable[[], EyeLandmarkFn] | None = None,
    pet_vlm: PetAskFn | None = None,
    pet_vlm_factory: Callable[[str], PetAskFn] | None = None,
    junk_rescue_vlm: JunkAskFn | None = None,
    junk_rescue_vlm_factory: Callable[[str], JunkAskFn] | None = None,
    junk_text_encoder: TextEncoder | None = None,
    junk_text_encoder_factory: Callable[[NamingSettings], TextEncoder] | None = None,
    search_encoder: FeatureSource | None = None,
    search_encoder_factory: Callable[[NamingSettings], FeatureSource] | None = None,
    detector: DetectFn | None = None,
    detector_factory: Callable[[str], DetectFn] | None = None,
    detector_text_encoder: TextEncoder | None = None,
    verdicts_only: bool = False,
    progress: ProgressCB | None = None,
) -> JunkStats:
    """Classify canonical photos into media_class.

    use_clip=False — heuristics only (source='heuristic', tier='heuristic'); such
    rows will be reprocessed by CLIP on the next run with use_clip=True.

    F68: incrementality is driven by `media_class.tier`, not by `source` — every row
    this run touches gets tier = active_tier ('heuristic' | 'clip' | 'vlm'), and only
    rows carrying a different tier are redone (see the module docstring for why
    `source` cannot serve as that marker).

    text_detector (F37, Phase A): (path, width, height) -> text_frac | None.
    By default an easyocr detector is built (lazily, once per run) — as with
    classifier, the caller passes its own (mock) in tests.

    text_detector_factory (F73): builds ONE detector per OCR worker thread; by
    default it wraps easyocr_text_frac_detector, and an explicit `text_detector`
    (above) is shared by every worker instead. Tests replace the factory to count how
    many detectors a run creates and to check the degradation path (a factory that
    fails on the second and further calls must shrink the pool, not kill the stage).
    The number of workers comes from `naming.ocr_workers` (see resolve_ocr_workers).

    vlm_classifier / vlm_classifier_factory (F37, Phase B): the deep tier,
    opt-in via cfg.naming.vlm_enabled (default False, gated by use_clip=True —
    a heuristics-only run does not touch deep). vlm_classifier — a ready
    classify_media(path)->label (a mock in tests, like classifier/text_detector);
    vlm_classifier_factory(model_name)->vlm_classifier — a factory for the real build
    (qwen_vlm_classifier_factory(cfg.vlm.max_edge) by default), replaced in tests to
    check the GRACEFUL FALLBACK: if the factory raises (no transformers, the model does
    not load, not enough VRAM), classify() catches the exception, logs it, and quietly
    continues on the fast tier (CLIP) — without crashing.

    F101: the deep pass is pipelined when the classifier exposes its halves (the real
    one does — SplitVlmClassifier) and `vlm.workers` is above 1: that many threads
    decode and preprocess frames while this thread runs the model and writes. An
    injected `vlm_classifier` (tests) has no halves and takes the serial path, which is
    why every verdict test below is unaffected by the pipeline.

    F102: everything about the MODEL — which one, at what input resolution, with how
    many preparation threads — comes from `cfg.vlm` (the `vlm:` config section, with the
    old `naming.*` keys still honoured by load_config). The tier toggle is the exception
    noted at the read below.

    sharpness_detector (F113): the frame-quality cascade, written into `frame_quality`
    alongside the classification. The detector is the laplacian over the shared preview (no
    toggle — milliseconds, and both the "best frame" and the "blurred junk" consumers need
    it); pets are a prompt group inside the CLIP call this stage already makes, behind
    `features.pets`. It is injectable for the same reason `classifier`/`text_detector` are:
    the suite must not load a model.

    F186: the third tier of that cascade — the model asked about the uncertain band — is
    gone, with its toggle, its scope and its parameters. It answered one question ("are the
    eyes open") that F179 answers off the eyelid geometry of a decode this stage already
    pays for, five times as often and at slightly better precision.

    F155: the detector now takes the frame's face boxes as well and answers with BOTH
    laplacians — over the whole preview and over the sharpest face in it — because they
    come out of one decode and a second pass over the collection for the second number is
    the one cost this signal is not worth.

    eye_landmarks_factory (F179): builds the 106-point contour model the eye number is
    fitted with, ONCE and on the first face of the run (`lazy_eye_landmarks`) — so a
    collection with no faces in it never builds one, and a machine that cannot build one
    loses `frame_quality.eye_openness` and nothing else. It is handed to the sharpness
    detector rather than run in a pass of its own, for the F155 reason: the pixels it needs
    are the ones that decode already has in memory. An injected `sharpness_detector` (every
    mock in the suite) answers for all three numbers and this factory is then unused.

    pet_vlm / pet_vlm_factory (F130): the animal check — the same shape and the same
    graceful fallback again, behind `features.pets_verify` (which needs `features.pets`).
    Frames whose pet score clears `features.pet_candidate_threshold` are shown the model
    one at a time and the answer decides `frame_quality.pet_vlm` and, through it, the
    label. A model that will not build, will not answer, or answers something nobody can
    read leaves every frame with the label `features.pet_threshold` gave it.

    F206: this check and the rescue below run through the SAME pipeline as the deep tier
    (`vlm.workers` preparation threads, `_vlm_labels`), which they did not until the
    regression in the module docstring was measured. An injected asker without halves —
    every mock in the suite — takes the serial path, exactly as an injected
    `vlm_classifier` does, which is why no verdict test below is affected by it.

    F186: the keeper question (F132) is gone too — which frame of a near-duplicate group to
    keep, asked once per group. Measured blind over 111 groups it agreed with the owner on
    32% of them against 30.4% for a coin, so there was no cheaper answer to move to and
    none was needed. `group_keeper` and the sharpness ranking the Duplicates tab shows are
    untouched, and no path of this stage has ever written `dedup_choice`.

    F128: the CLIP vector of every canonical photograph is stored in `clip_embeddings`
    (`features.store_embeddings`, on by default). No parameter of its own: the vector is
    taken from the classifier that has just scored the chunk — a `features(paths)` method
    over its cache, which the real one (landmarks.CachingFeatureClassifier) has — so a
    classifier injected as a plain function stores nothing, logs why once, and changes no
    other behaviour of the stage.

    junk_rescue_vlm / junk_rescue_vlm_factory / junk_text_encoder /
    junk_text_encoder_factory (F140): the rescue of the screenshots and receipts this stage
    called photographs, behind `features.junk_rescue`. The score is read off the stored
    CLIP vectors (F128) with the text encoder — injected here, and built lazily from
    `naming.clip.*` otherwise, so a run with nothing to score loads no model — and the
    frames it selects are shown the VLM, which needs the deep tier (`vlm.enabled`) to be on
    at all. Same shape and same graceful fallback as the three askers above: an encoder or
    a model that will not build leaves every verdict of the run exactly as the fast tier
    wrote it, and with the deep tier off nothing is reclassified even when the score is.

    search_encoder / search_encoder_factory (F141): the image tower of the SEARCH index,
    the one pass of this stage that encodes the frames a second time. Behind
    `features.search_index` (off by default — it is ~10.5 minutes per 20 000 frames), with
    a model of its own (`features.search_model`) that no threshold of this pipeline is
    calibrated on, which is the entire reason it exists next to `naming.clip.*` instead of
    replacing it. Injectable and built lazily for the same reasons the rescue encoder is:
    the suite must not load a model, and a run with nothing left to encode must not either.

    detector / detector_factory / detector_text_encoder (F154): the animal cascade's third
    tier — an object detector over the candidates a zero-shot query picks out of the stored
    CLIP vectors, behind `features.detector` AND `detect.enabled` (its own master switch,
    see config.detector_allowed: a detector is not a VLM and is not raised by `vlm.enabled`
    either way). Injectable and built lazily exactly like the encoders above, and for the
    same two reasons: the suite must not load a model, and a run with no candidate to
    examine must not either. Same graceful fallback as everything else here — an encoder or
    a detector that will not build leaves every animal label as F122/F130 wrote it, and
    there is no configuration in which the detector runs over the whole collection.

    F145: every one of the askers above is subordinate to `vlm.enabled`. Their own
    keys say WHAT to ask, not whether a model is raised — a run without deep analysis
    loads no weights whatever config.yaml holds, and each of them then behaves exactly as
    it does with its own key off (the graceful-fallback path they already had).

    verdicts_only (F165): run the halves that do NOT need the face signal and stop —
    the fast pass, the deep tier and the stored vectors, i.e. everything that ends in
    `media_class` or `clip_embeddings`. This is the `classify` stage, which the pipeline
    runs BEFORE faces so that the faces stage can skip what is already known not to be a
    photograph (see `faces._files_to_detect`); the quality cascade, the animal cascade, the
    rescue and the search index are left to the `junk` stage after it. Nothing
    else about the call changes: the same incrementality markers, so the second half finds
    the verdicts current and reclassifies nothing, and a lone `sorta junk` (or any caller
    that never passes this flag) still runs the whole stage exactly as before.

    progress (F100): the usual `(done, total)` callback; if it also carries a
    `phase(name)` channel (progress.TaskProgress, ui._StageProgress) the stage reports
    which of its phases it is in — CLASSIFY_PHASE_*. A plain function without that
    channel is not an error and gets the counter alone, as before.

    F147: those same phases are TIMED, and each one that ran leaves a
    `stage=junk phase=<name> elapsed=<sec> processed=<n>` line in the run log next to
    the stage summary. Independent of `progress` — a run with no callback measures
    itself just the same — and independent of the phases' own behaviour: this is the
    instrument the stage is about to be optimized with, so it changes nothing it
    measures.
    """
    s = naming_settings(cfg)
    rows = conn.execute(
        """SELECT f.id, f.path, f.width, f.height, f.camera_make, f.camera_model,
                  f.gps_lat,
                  EXISTS(SELECT 1 FROM faces fa
                         WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces,
                  mc.source AS mc_source, mc.tier AS mc_tier,
                  mc.verdict AS mc_verdict,
                  fq.source AS fq_source,
                  ce.model AS ce_model
           FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
                        LEFT JOIN frame_quality fq ON fq.file_id = f.id
                        LEFT JOIN clip_embeddings ce ON ce.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
           ORDER BY f.id"""
    ).fetchall()
    stats = JunkStats(total=len(rows))

    # F145: the master switch, read ONCE and required by every question below. Each of
    # them used to gate on its own key alone, so a run started without deep analysis
    # still loaded the weights whenever one subordinate key was true in config.yaml —
    # the hierarchy was assumed and never written down (see config.vlm_allowed). The
    # check stands HERE, before any factory is called: loading is five seconds and
    # gigabytes of memory, and somebody who cleared the checkbox does not pay for them.
    vlm_on = vlm_allowed(cfg)

    # F37 (Phase B): the tier gate. use_clip=False — an explicit heuristics-only
    # mode, deep does not enter there (symmetric with CLIP below).
    # F161: and `vlm.products` — the tier is a named line of the run screen with a price
    # of its own now, not the private effect of the master switch (see products_allowed,
    # which still requires `vlm_on`).
    vlm_fn: VlmClassifyFn | None = None
    if use_clip and products_allowed(cfg):
        if vlm_classifier is not None:
            vlm_fn = vlm_classifier
        else:
            factory = vlm_classifier_factory or qwen_vlm_classifier_factory(
                cfg.vlm.max_edge)
            try:
                vlm_fn = factory(cfg.vlm.model)
            except Exception as exc:  # noqa: BLE001 — deep is optional, must not crash
                _log.warning(
                    "junk: VLM недоступна (%s) — откат на fast-ярус (CLIP)", exc)
                vlm_fn = None

    # F165: every question below this one belongs to the half that runs AFTER faces. The
    # check sits next to each gate rather than around the whole block because that is where
    # a reader looks for the answer to "does the `classify` stage load this model?" — and
    # the answer has to be no for all of them, or the split would move the weights of the
    # deep passes ahead of the stage that has nothing to ask them.
    #
    # F186: the frame-quality question used to be resolved here, first of the three, and
    # its whole gate went with it — the toggle, the scope and the check that the scope was
    # satisfiable at all.
    q = quality_settings(cfg)

    # F130: the animal check, resolved exactly like the deep tier above and with the same
    # graceful fallback — a model that will not build must cost the cheap tiers nothing.
    # Its gate carries one extra condition — `features.pets`, because it verifies what the
    # CLIP pet group found and has nothing to verify without it. The model is the shared
    # one (F95), so switching this on next to another question costs a call per frame, not
    # a second set of weights.
    pet_ask: PetAskFn | None = None
    if use_clip and vlm_on and not verdicts_only and q.pets and q.pets_verify:
        if pet_vlm is not None:
            pet_ask = pet_vlm
        else:
            pet_factory = pet_vlm_factory or qwen_vlm_pet_factory(cfg.vlm.max_edge)
            try:
                pet_ask = pet_factory(cfg.vlm.model)
            except Exception as exc:  # noqa: BLE001 — the check is optional, must not crash
                _log.warning(
                    "junk: VLM-проверка животных недоступна (%s) — метка остаётся "
                    "по порогу CLIP", exc)
                pet_ask = None

    # F140: the rescue check, resolved the same way and with the same fallback again. Two
    # conditions gate it and they are different questions: `features.junk_rescue` says the
    # SCORE is wanted, the deep tier says there is somebody to answer for the candidates it
    # selects. With the tier off the score is still computed and stored — that is the state
    # the feature is meant to be tried in — and not one verdict moves.
    rescue_ask: JunkAskFn | None = None
    if use_clip and vlm_on and not verdicts_only and q.junk_rescue:
        if junk_rescue_vlm is not None:
            rescue_ask = junk_rescue_vlm
        else:
            r_factory = junk_rescue_vlm_factory or qwen_vlm_junk_rescue_factory(
                cfg.vlm.max_edge)
            try:
                rescue_ask = r_factory(cfg.vlm.model)
            except Exception as exc:  # noqa: BLE001 — the rescue is optional, must not crash
                _log.warning(
                    "junk: VLM-проверка кандидатов недоступна (%s) — вердикты остаются "
                    "за быстрым ярусом, счёт всё равно пишется", exc)
                rescue_ask = None

    # F68: incrementality runs on media_class.tier — the marker of WHICH TIER
    # processed the row, independent of `source` (what decided the verdict). Three
    # tiers: 'heuristic' (use_clip=False), 'clip' (the fast pass), 'vlm' (the fast
    # pass + the deep refinement of candidates). A row is redone only when its tier
    # differs from the active one — so any switch, upgrade or downgrade, reprocesses.
    active_tier = "heuristic" if not use_clip else ("vlm" if vlm_fn is not None else "clip")
    junk_ids = {r["id"] for r in rows if r["mc_tier"] != active_tier}
    # F113: the quality half keeps its OWN incrementality marker (`frame_quality.source`),
    # because the two halves go stale independently — switching `features.pets` on does not
    # change a single junk verdict, and a collection whose junk was classified before this
    # feature existed has no quality rows at all. A frame is walked when EITHER half wants
    # it, and each half then writes only its own table.
    quality_source = _quality_source(use_clip, q.pets, pet_ask,
                                     use_clip and q.junk_rescue, rescue_ask)
    # F120: only personal photographs are asked the quality questions. Selection uses the
    # verdict ALREADY STORED, because this run's verdict is not known until the frame is
    # walked; a frame with no verdict yet (a first run) is included and settled below, and
    # a frame whose class changes is picked up on the next run. The lag is one run and it
    # is on the cheap half of the cascade.
    # F165: and in the verdicts-only half there is no quality work at all — the population
    # is empty rather than the pass disabled, so `wanted()` says no about every frame and
    # the loop below writes nothing into `frame_quality`.
    quality_ids = ({r["id"] for r in rows
                    if r["fq_source"] != quality_source
                    and r["mc_verdict"] in (None, QUALITY_VERDICT)}
                   if use_clip and not verdicts_only else set())
    # F128: and the third half, with a marker of its own again — `clip_embeddings.model`.
    # A vector is stale when it was computed by another model, which is the only way a
    # stored vector becomes unusable rather than merely old. The population is the quality
    # half's, selected the same way and for the same F120 reason, with the same one-run lag
    # on a frame whose class changes. A heuristics-only run asks CLIP nothing, so there is
    # no vector to keep and no row is touched.
    store_embeddings = bool(getattr(
        getattr(cfg, "features", None) or FeaturesConfig(), "store_embeddings", True))
    embed_model = embedding_model(s)
    # Whether this run can produce a vector at all, decided BEFORE any frame is selected
    # for one. The real classifier hands its cache back (`features`, see
    # landmarks.CachingFeatureClassifier) and so does one built below; an injected plain
    # function — every mock in the suite, a caller with a scorer of its own — cannot, and
    # then this half is simply off. Selecting frames it could never write would send them
    # to CLIP for nothing and leave them selected again on the next run.
    features_of = getattr(classifier, "features", None) if classifier is not None else None
    feature_source: FeatureSource | None = features_of if callable(features_of) else None
    can_embed = classifier is None or feature_source is not None
    # F146: and it says so. The half switching itself off is the right behaviour and the
    # silence around it was not: an empty `clip_embeddings` reads exactly like a collection
    # nobody has processed yet, so a caller handing over a classifier that cannot produce
    # vectors — as the web app did from F128 until a production run in August 2026 — gets
    # no exception, no row and, until now, no hint of a reason.
    if use_clip and store_embeddings and not can_embed:
        _log.warning(
            "junk: классификатор (%s) не отдаёт CLIP-векторы — features.store_embeddings "
            "включён, но таблица clip_embeddings не наполняется",
            type(classifier).__name__)
    embed_ids = ({r["id"] for r in rows
                  if r["ce_model"] != embed_model
                  and r["mc_verdict"] in (None, QUALITY_VERDICT)}
                 if use_clip and store_embeddings and can_embed else set())
    work = [r for r in rows
            if r["id"] in junk_ids or r["id"] in quality_ids or r["id"] in embed_ids]
    # `processed`/`skipped_incremental` keep counting CLASSIFICATION, not the walk: they
    # are printed by the CLI and the web app as "how much of the collection was
    # reclassified", and a frame that was only measured for sharpness was not. What the
    # quality half did is its own counters (quality_rows / pets_found / quality_*).
    stats.processed = len(junk_ids)
    stats.skipped_incremental = len(rows) - len(junk_ids)
    now = utcnow_iso()
    # F165: the phases are filed under the stage that is actually running — the caller's
    # `stage_timer` opened `classify` or `junk`, and the two names have to agree.
    # F147: built here rather than next to the first `report.start` below, so the passes
    # that can have work when every other half of the stage is up to date are timed on
    # that path too instead of running under a throwaway reporter.
    report = _PhaseProgress(progress, VERDICTS_STAGE if verdicts_only else CLASSIFY_STAGE)
    # F141: the search index — a pass that can have work
    # when every other half of the stage is up to date. That is its ORDINARY case: the
    # toggle is switched on for a collection that is already classified, and an early
    # return that skipped it would leave the feature silently doing nothing. Its encoder
    # is a closure so that a run with nothing to encode never builds one.
    index_model = search_index_model(cfg)
    search_index = _SearchIndexPass(
        conn, index_model,
        (lambda: search_encoder) if search_encoder is not None
        else (lambda: (search_encoder_factory or search_image_encoder)(
            search_index_settings(s, index_model))),
        s.clip_batch_size, now, stats,
        use_clip and not verdicts_only and search_index_enabled(cfg))
    # F154: the animal detector — a second pass that can have work when every other half of
    # the stage is up to date, and for the same reason the search index can:
    # the toggle is switched on for a collection that is already classified. Both of its
    # models are closures so that a run with no candidate builds neither. A heuristics-only
    # run has no vectors to query and no CLIP tier to correct, so `use_clip` gates it too.
    d = detector_settings(cfg)
    # F165: and the verdicts-only half does not run it either — it is a cascade over
    # `frame_quality`, which belongs to the half after faces. The settings are left
    # untouched (unlike the heuristics-only case above) so that `purge` below still knows
    # the detector is on: a frame this half has just called a document must lose its boxes
    # in the run that renamed it, not in the next one.
    if not use_clip:
        d = replace(d, enabled=False)
    detect_pass = _DetectorPass(
        conn, d, embedding_model(s),
        (lambda: detector_text_encoder) if detector_text_encoder is not None
        else (lambda: clip_text_encoder(s)),
        (lambda: detector) if detector is not None
        else (lambda: (detector_factory or torchvision_detector)(d.model)),
        float(q.pet_threshold), now, stats)
    if not work:
        if not verdicts_only:
            detect_pass.run(report)
        with conn:
            detect_pass.purge()
        search_index.run(report)
        report.log_timings()
        return stats
    # F121: has the faces stage ever run here? One row is enough to tell — after that,
    # "this frame has no face" is a fact rather than an absence of evidence.
    faces_known = faces_stage_ran(conn)
    # F179: the eye number rides in the sharpness decode, so the model that produces it is
    # handed to that detector rather than to a pass of its own. Lazily, and only where
    # there are faces to fit it to: `lazy_eye_landmarks` builds on the first face of the
    # run, so a collection with none — or a first run, before `sorta faces` — pays nothing,
    # and a machine that cannot build it loses this one column and no more.
    quality = _QualityPass(
        conn, q,
        sharpness_detector or preview_sharpness_detector(
            q.sharpness_max_edge,
            lazy_eye_landmarks(eye_landmarks_factory or insightface_eye_landmarks)
            if faces_known else None),
        quality_source, quality_ids, now, stats, pet_ask, cfg.vlm.workers)
    # F140: the encoder is a closure and not an object, so that a run whose rescue has
    # nothing to score never builds one — see _JunkRescuePass.
    rescue = _JunkRescuePass(
        conn, q, embed_model,
        (lambda: junk_text_encoder) if junk_text_encoder is not None
        else (lambda: (junk_text_encoder_factory or clip_text_encoder)(s)),
        rescue_ask, now, active_tier, stats, cfg.vlm.workers)
    # F100: the phase channel of the callback, if it has one. The total is reported
    # right away, even if the stage is small/fast (#37); which phase the stage opens
    # with depends on the tier — a heuristics-only run classifies nothing, it only
    # writes verdicts.
    #
    # F147: this `start` is the only one that does NOT count units — its total is the
    # denominator of the whole fast pass, not the size of one phase's own work list, and
    # the phases below count what each of them actually touched.
    report.start(CLASSIFY_PHASE_CLIP if use_clip else CLASSIFY_PHASE_WRITE, len(work))

    heur_raw = {
        r["id"]: heuristic_verdict(
            r["path"], r["width"], r["height"], r["camera_make"], r["camera_model"],
        )
        for r in work
    }
    heur = {fid: v or "photo" for fid, v in heur_raw.items()}
    # F68: `tier` is written on every path and always equals active_tier — see
    # _MEDIA_CLASS_UPSERT, where the statement and that rule now live.
    upsert = _MEDIA_CLASS_UPSERT

    if not use_clip:
        with conn:
            for r in work:
                verdict = heur[r["id"]]
                conn.execute(upsert, (r["id"], verdict, "heuristic", None, now, active_tier))
                stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
        report.step(len(work))
        report.count(CLASSIFY_PHASE_WRITE, len(work))
        # F210: the heuristics-only tier writes verdicts like any other, so it owes the
        # same cleanup — see the call at the end of the stage for what it is and why.
        sweep_previews(conn, q.exclude_classes)
        report.log_timings()
        return stats

    # F113: a run can now reach this point with nothing to ask CLIP — a collection whose
    # junk classification is already current, on its first run after the frame_quality
    # table appeared, with both toggles off: only laplacians are missing. Loading a CLIP
    # model to compute those would be the whole cost of the stage for no question asked.
    needs_model = bool(junk_ids) or quality.needs_clip() or bool(embed_ids)
    if classifier is None:
        if needs_model:
            classifier = clip_classifier(s)  # pragma: no cover — ML, smoke test
        else:
            classifier = _unused_classifier
    # F128: the vectors come out of the classifier that has just been resolved — the cache
    # of `landmarks.CachingFeatureClassifier`, which is what makes this half free. A
    # classifier BUILT here without that method could only be a future regression of
    # clip_classifier, so it is logged rather than passed over: the table would otherwise
    # stay empty for a reason nobody could see.
    if embed_ids and feature_source is None:
        features_of = getattr(classifier, "features", None)
        feature_source = features_of if callable(features_of) else None
        if feature_source is None:
            _log.warning(
                "junk: классификатор не отдаёт CLIP-векторы — features.store_embeddings "
                "включён, но таблица clip_embeddings не наполняется")
    embeddings = _EmbeddingPass(conn, embed_model, embed_ids, feature_source, now, stats,
                                store_embeddings)
    # F73: the OCR pool — K worker threads, one own detector each, built lazily on
    # first use and reused for the whole run (see _OcrPool). The detector itself is no
    # longer built here: a run where the gate opens for nothing loads no model at all.
    ocr_workers = resolve_ocr_workers(cfg.raw)
    ocr = _OcrPool(
        text_detector_factory or _resolve_detector_factory(cfg, text_detector),
        ocr_workers)
    stats.clip_used = True
    # F90: every threshold of the verdict/gate logic in one place (see GateSettings) —
    # the same values scripts/measure_ocr_gate.py sweeps.
    g = gate_settings(cfg)
    # F113: the ONE main call of the stage answers both questions — the junk classes and,
    # when `features.pets` is on, the pet group appended to the very same prompt list. Not
    # a second pass and not a second call: the number of classifier calls a chunk makes is
    # what it was before this feature.
    prompts = clip_prompts(q.pets)
    doc_prompts = [prompt for _cls, prompt in _DOCUMENT_CLASSES]
    prod_prompts = [prompt for _cls, prompt in _PRODUCT_CLASSES]
    product_candidate_min = float(
        getattr(cfg.naming, "product_candidate_min", _DEFAULT_PRODUCT_CANDIDATE_MIN))
    # #14/V1: the VLM tier (deep) does NOT run on all frames — only on candidates:
    # files without faces where the fast tier doubts (verdict='document' OR the
    # document-CLIP is in a suspicious zone OR the product-CLIP is above the
    # threshold). Collect here, reclassify with the VLM after the fast pass.
    # (id, path, fast_verdict).
    vlm_candidates: list[tuple[int, str, str]] = []
    done = 0
    try:
        with conn:
            for chunk in batched(work, s.clip_batch_size):
                report.enter(CLASSIFY_PHASE_CLIP)
                paths = [r["path"] for r in chunk]
                # F113: a chunk may hold frames only one of the two halves asked for. The
                # junk half always needs its CLIP row, the quality half only when it has a
                # question for CLIP (pets, or the band's subject score) — so a frame nobody
                # needs a row for is not encoded, and everybody else is served by the one
                # call below.
                junk_idx = [i for i, r in enumerate(chunk) if r["id"] in junk_ids]
                clip_idx = sorted(set(junk_idx) | (
                    {i for i, r in enumerate(chunk) if quality.wanted(r["id"])}
                    if quality.needs_clip() else set()) | (
                    # F128: a frame whose vector is missing or was computed by another
                    # model needs the same one call — not one of its own.
                    {i for i, r in enumerate(chunk) if embeddings.wanted(r["id"])}
                    if embeddings.needs_clip() else set()))
                # F147: the unit of this phase is a frame ENCODED, not a frame walked —
                # incrementality can hand the loop a chunk whose CLIP rows are all
                # current, and counting those would price the encoder by work it never
                # did. The document/product passes below run over a subset of the same
                # frames, so they add no units of their own.
                report.count(CLASSIFY_PHASE_CLIP, len(clip_idx))
                probs: dict[int, np.ndarray] = {}
                vecs: dict[int, np.ndarray | None] = {}
                if clip_idx:
                    clip_paths = [paths[i] for i in clip_idx]
                    rows_probs = classifier(clip_paths, prompts)
                    probs = {i: rows_probs[k] for k, i in enumerate(clip_idx)}
                    if embeddings.needs_clip():
                        # F128: the vectors of the very call above, out of its cache —
                        # asked for right after it, while the chunk is still there.
                        vecs = {i: v for i, v in
                                zip(clip_idx, embeddings.vectors(clip_paths))}
                # F15: document-CLIP only for files without detected faces —
                # faces are an unconditional veto, a second pass for them is unneeded.
                noface_idx = [i for i in junk_idx if not chunk[i]["has_faces"]]
                doc_score: dict[int, float] = {}
                product_score: dict[int, float] = {}
                if noface_idx:
                    doc_probs = classifier([paths[i] for i in noface_idx], doc_prompts)
                    for k, i in enumerate(noface_idx):
                        doc_score[i] = _document_score(doc_probs[k])
                    if vlm_fn is not None:  # the product prefilter is only needed for the VLM gate
                        prod_probs = classifier([paths[i] for i in noface_idx], prod_prompts)
                        for k, i in enumerate(noface_idx):
                            product_score[i] = _product_score(prod_probs[k])
                # F73, phase 1: the pre-OCR verdict of every frame of the chunk, plus
                # the frames the OCR gate opens for. The gate condition itself and the
                # verdict logic are unchanged — only the OCR call left this loop.
                pre: dict[int, tuple[str, float]] = {}  # (verdict, score) before OCR
                ocr_jobs: list[OcrJob] = []
                for i in junk_idx:
                    r = chunk[i]
                    # F113: the junk classes read through _group_probs — with pets off that
                    # is the untouched row, with pets on it is the same row renormalized
                    # over the first three prompts, which is the same softmax the stage saw
                    # before the pet prompts joined the call.
                    p = _group_probs(probs[i], _JUNK_GROUP)
                    best = int(np.argmax(p))
                    verdict, score = clip_verdict(
                        _CLIP_CLASSES[best][0], float(p[best]), heur_raw[r["id"]],
                        doc_score.get(i), _is_real_photo(r), g)
                    pre[i] = (verdict, score)
                    if ocr_gate_open(bool(r["has_faces"]), verdict,
                                     doc_score.get(i, 0.0), g.text_rescue_docscore_min):
                        ocr_jobs.append((r["id"], r["path"], r["width"], r["height"]))
                # F73, phase 2: text_frac for the gated frames, in the pool. This is the
                # only part of the stage that leaves this thread.
                # F100: named only when the gate actually opened — a chunk with no OCR
                # jobs stays in the CLIP phase instead of flashing a caption for work
                # that is not happening.
                if ocr_jobs:
                    report.enter(CLASSIFY_PHASE_OCR)
                    report.count(CLASSIFY_PHASE_OCR, len(ocr_jobs))
                text_fracs = ocr.text_frac(ocr_jobs)
                # F73, phase 3: apply the OCR signal, then write — on this thread only
                # (single writer) and in the original per-chunk order, so the verdicts
                # and stats are exactly those of the serial version.
                report.enter(CLASSIFY_PHASE_WRITE)
                # F147: every frame of the chunk passes through here, and this phase is
                # where two of the six costs of the stage actually live — the laplacian
                # (`quality.measure`) and the stored vector (`embeddings.store`) — next
                # to the verdict writes themselves.
                report.count(CLASSIFY_PHASE_WRITE, len(chunk))
                for i, r in enumerate(chunk):
                    # F120: the quality half needs to know what this frame turned out to
                    # be. A frame the fast tier did not settle (`i not in pre`) keeps
                    # whatever media_class already says — hence the stored verdict as the
                    # starting value rather than a name that may never be assigned.
                    verdict = r["mc_verdict"]
                    if i in pre:
                        # missing / None both mean "no signal" — the verdict is left alone
                        verdict, score = pre[i]
                        verdict, score, source = apply_text_frac(
                            verdict, score, text_fracs.get(r["id"]), g)
                        if verdict == "photo" and _in_screenshots_dir(r["path"]):
                            # F29: the Screenshots folder is a "floor" for photo; we do not
                            # override document/meme (conservative, brief F29).
                            verdict = "screenshot"
                        conn.execute(upsert,
                                     (r["id"], verdict, source, score, now, active_tier))
                        stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
                        # #14/V1: selection into VLM candidates (deep refines doc/product/photo) —
                        # without faces, not screenshot/meme, and the fast tier doubts: already a
                        # document, OR the document-CLIP is in a suspicious zone, OR the
                        # product-CLIP is above the threshold. Clear personal photos (both scores
                        # low) are not touched by the VLM.
                        # F120: `vlm.exclude_classes` — classes no VLM is shown at all.
                        # The default holds `document`, and the cost is real and stated:
                        # the deep tier is what CORRECTS a wrong document verdict, so an
                        # excluded class keeps whatever the fast tier decided about it.
                        if (vlm_fn is not None and not r["has_faces"]
                                and verdict not in q.exclude_classes
                                and verdict not in ("screenshot", "meme")
                                and (verdict == "document"
                                     or doc_score.get(i, 0.0) >= g.text_rescue_docscore_min
                                     or product_score.get(i, 0.0) >= product_candidate_min)):
                            vlm_candidates.append((r["id"], r["path"], verdict))
                    # F113: the cheap half of the cascade — the laplacian always, the pet
                    # group when the toggle is on, and the note of which frames the animal
                    # check is worth asking about.
                    if quality.wanted(r["id"]):
                        quality.measure(r["id"], r["path"], probs.get(i), verdict)
                    # F128: and the vector of the same frame, kept instead of dropped —
                    # under the same verdict, so a screenshot gets no row here either.
                    if embeddings.wanted(r["id"]):
                        embeddings.store(r["id"], vecs.get(i), verdict)
                done += len(chunk)
                report.step(done)
    finally:
        # F73: the workers (and their Readers) live exactly as long as the stage does.
        # The count is logged so a run can be checked against "one Reader per worker,
        # not per frame" without a profiler.
        ocr.close()
        if ocr.detectors_built:
            _log.info("junk: OCR-детекторов создано %d (воркеров %d)",
                      ocr.detectors_built, ocr_workers)

    # #14/V1: the deep tier — the VLM only on the selected candidates (not all frames).
    # A VLM runtime error on one frame does NOT crash the run (closes #31) — the file
    # keeps its fast verdict.
    if vlm_fn is not None and vlm_candidates:
        stats.vlm_candidates = len(vlm_candidates)
        # F100: the phase whose numbers used to arrive without a word of explanation.
        # The denominator switches from the frames of the fast pass to the candidates
        # of the gate — honest, and readable only because the caption switches with
        # it: at the measured 1.38 frames/s the difference between a bar that quietly
        # restarted and "2 201 of 7 896" is the difference between "probably hung" and
        # "about an hour left".
        report.start(CLASSIFY_PHASE_VLM, len(vlm_candidates))
        report.count(CLASSIFY_PHASE_VLM, len(vlm_candidates))
        # F101: the labels arrive from _vlm_labels — in the candidate order, whether the
        # pass was pipelined or serial. A frame the model failed on comes back as its
        # exception instead of a label; everything below (the order of the writes, the
        # mapping to a verdict, the stats, the progress step) is what it was.
        # closing(): the preparation threads live exactly as long as the pass, the way
        # the OCR pool does — not until the garbage collector gets round to the
        # generator holding them.
        labels = _vlm_labels(vlm_fn, [path for _fid, path, _v in vlm_candidates],
                             cfg.vlm.workers)
        with closing(labels), conn:
            for j, ((fid, _path, fast_verdict), label) in enumerate(
                    zip(vlm_candidates, labels)):
                if isinstance(label, BaseException):  # deep is optional, do not crash the run
                    _log.warning("junk: VLM-ошибка на file_id=%s (%s) — оставляю fast-вердикт",
                                 fid, label)
                else:
                    verdict = _VLM_LABEL_TO_VERDICT.get(label, fast_verdict)
                    # source='vlm' — the VLM is what decided this verdict. The
                    # incrementality marker is `tier` (already written as 'vlm' by the
                    # fast pass above, for candidates and non-candidates alike), so a
                    # repeated run does not re-run the VLM on these files.
                    conn.execute(upsert, (fid, verdict, "vlm", None, now, active_tier))
                    if verdict != fast_verdict:
                        stats.by_verdict[fast_verdict] = stats.by_verdict.get(fast_verdict, 1) - 1
                        stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
                        stats.vlm_applied += 1
                # F100: outside the `else` — a frame the model failed on is a frame the
                # pass is done with (an error on the last candidate would otherwise
                # leave the bar one short of its total for good).
                report.step(j + 1)

    # F140: the rescue, over the frames the fast tier called photographs. Here and not in
    # the loop because it reads what the loop wrote — the vector of each frame — and after
    # the deep tier because that tier is the other thing that can move a verdict, and a
    # frame it has just called a `document` must not be asked about again. Before the
    # question below it because the verdict is what the rest of the pipeline depends on,
    # and a run interrupted afterwards has finished the part that matters most.
    rescue.run(quality.measured, report)
    # F130: the animal check, over the candidates the CLIP pet group turned up. After the
    # deep tier because both want the same GPU and the verdict is what the rest of the
    # pipeline depends on. F186 left it the last question this stage puts to a model.
    quality.ask_pets(report)
    # F154: the detector, last of the three tiers that can move an animal label. After the
    # VLM check because it reads what that check wrote — a frame the model has already
    # judged keeps its answer, since "is this animal alive" is not a question a box
    # detector can be asked (`detect.cascade_label`) — and after the deep tier because that
    # is what settles which frames are personal photographs at all. Its candidates come out
    # of `clip_embeddings`, which the loop above has just filled.
    # F165: and, like the two passes above it, not in the verdicts-only half — it reads
    # and writes `frame_quality`, whose rows the half after faces owns.
    if not verdicts_only:
        detect_pass.run(report)
    # F120: enforce "only a personal photograph has a quality row" DIRECTLY, and do it
    # LAST, when every verdict of this run is written — the deep tier above reclassifies
    # frames, so a purge any earlier would judge them by the fast tier's answer.
    #
    # This is not the same guard as the per-frame one in `_QualityPass.measure`, and both
    # are needed: incrementality skips a frame whose `source` already matches, so a
    # collection measured before this rule — 24 196 rows over everything, all
    # `source='vlm'` — would keep its screenshots and documents precisely BECAUSE they
    # look up to date. One statement on an indexed column settles it for good.
    #
    # F165: the three purges run in BOTH halves, and that is deliberate — the verdicts half
    # writes no quality row, no vector and no box, but it is the half that can RENAME a
    # frame into a document, and the rule is that the run which renames it is the run that
    # drops what described it. `sorta classify` on its own would otherwise leave a passport
    # with its old crop measurements until somebody happened to run `sorta junk`.
    with conn:
        conn.execute(
            "DELETE FROM frame_quality WHERE file_id IN"
            " (SELECT file_id FROM media_class WHERE verdict != ?)", (QUALITY_VERDICT,))
        # F128: the same rule over the same population, for the same reason (see
        # _EmbeddingPass.purge) — and after the deep tier, whose reclassifications it has
        # to see.
        embeddings.purge()
        # F154: and the boxes, under the same rule and in the same transaction — a frame
        # this run decided is a document must not keep a description of what is on it.
        detect_pass.purge()
    # F210: and the derivative that is not in the database — the preview JPEG the stage
    # itself wrote before it knew what the frame was. Here for the same reason the three
    # purges above are here and not earlier: the deep tier reclassifies frames, so a sweep
    # any sooner would judge a passport by the fast tier's answer. In BOTH halves, again
    # for the F165 reason — `sorta classify` is what can rename a frame into a document,
    # so it is what must take the picture of it away.
    sweep_previews(conn, q.exclude_classes)
    # F141: and last of all, the second CLIP pass — every verdict of this run is
    # written and the purges above have run, so a frame the deep tier has just called a
    # screenshot is not encoded. Ten minutes is too much to spend on rows that would be
    # deleted a moment later.
    search_index.run(report)
    # F147: the breakdown of the seconds the caller's `stage_timer` is about to report as
    # a single number — by now all of it is written except the pass that was running when
    # the stage reached its end. F166: a stage that raised or was cancelled no longer
    # loses its phases either; `stage_timer` closes them, and marks the unfinished one as
    # unfinished instead of letting it read as a profile of a run that never happened.
    report.log_timings()
    return stats
