"""F6 (Phase 5, FR-7): junk classification of canonical photos.

Contract: reads files (+faces as a signal), writes ONLY into media_class
(schema v3). Deletes and moves NOTHING — the layout into _Unsorted/junk is done
by F5-sorter based on this table.

Two-stage scheme (conservative — brief F13, junk is costlier for a missed piece of
trash than a real photo in the trash):
a) heuristics (fast, no ML) — only an explicit Screenshot_/"снимок экрана" name,
   source='heuristic';
b) CLIP zero-shot (the same model as landmarks, in a batch) — 3 classes, threshold
   naming.junk_threshold, source='clip', score is written. A file with camera EXIF/
   GPS OR detected faces — a veto, the CLIP verdict does not override it.
   Below the threshold — the heuristic verdict stays, but the row is marked
   source='clip' (the file was checked by CLIP and is not recomputed again —
   incrementality).
Files with verdict='photo' are also written (a "checked" mark).

F15: verdict='document' — a separate review category (not junk), detected
BEFORE the camera/GPS veto (a photographed document has camera EXIF — the target
case), but ONLY if the photo has no detected faces (portraits — the main FP source
from F13, they never contain documents). A separate CLIP run over the
document prompts (its own softmax normalization, does not interfere with the
junk_threshold of the main 3 classes) and a separate, higher threshold
naming.document_threshold (not yet typed in config.py — read via cfg.raw). For
files with faces the document-CLIP is not computed at all (the veto is
unconditional, saving a pass).

F37 (Phase A): CLIP zero-shot document is unreliable both ways (FP on scenes
with signs/menus, FN on genuinely photographed documents with a low CLIP score).
After the CLIP-stage verdict is computed, a text-density signal is applied
(easyocr, the fraction of the frame area under text boxes, `text_frac`) — only
to the document↔photo pair, only for files without faces (the same veto as the
document-CLIP above):
- FP gate: verdict='document', but text_frac < naming.text_frac_min → 'photo'
  (a beach/scene without dense text comes back from _Documents);
- FN rescue: verdict='photo', but text_frac >= naming.text_frac_document →
  'document', even if the CLIP score was low (catches photographed documents
  that CLIP missed).
In both cases source='ocr', score=text_frac. Screenshot/meme are not touched
(OCR is applied only if the verdict is already 'document' or 'photo'). The
thresholds — `getattr(cfg.naming, "text_frac_min"/"text_frac_document", default)`:
the fields are not yet typed in NamingConfig (getattr fallback, like
document_threshold once was).

F38 (validating F37-A on real data found 3 bugs): (1) the detector decodes
via `imaging.decode_rgb` (Unicode/HEIC-safe) + downscale before
`reader.detect()` — cv2 silently failed to read non-ASCII paths/HEIC, the box area
is now relative to the downscaled frame; (2) the FN rescue (the `verdict==
'photo'` branch) calls `text_detector` ONLY if `doc_score[i] >=
cfg.naming.text_rescue_docscore_min` — clear scenes (doc_score≈0) do not run
OCR, many fewer calls; the FP gate (`verdict=='document'`) is not gated, as
before. (3) the `text_frac_document` default was lowered 0.35 -> 0.15
(a real document at an angle gave text_frac=0.247 < 0.35 → the FN was not fixed).

F37 (Phase B): the deep tier, opt-in, default OFF (`naming.vlm_enabled`). Instead of
the CLIP+OCR pipeline (fast tier, Phase A, above), canonical photos are classified
by a VLM (Qwen2.5-VL, lazy-import — like easyocr above) with a 3-way prompt:
personal_photo/document/product -> verdict photo/document/product, source='vlm'.
An explicit Screenshot_ name (heuristic) still overrides the VLM — the deep tier
does not detect screenshots/memes, that stays the fast tier's job. GRACEFUL FALLBACK
(critical for an optional tier on weak hardware): a model-factory failure
(transformers not installed, the model does not load, not enough VRAM) is caught
ENTIRELY around building the classifier — a silent fall back to the fast tier (CLIP),
the error is only logged (`_log.warning`), `classify()` does not crash.

F68: incrementality runs on its OWN column `media_class.tier` (schema v11), not on
`source`. The two mean different things: `source` is WHAT decided the verdict
(heuristic | clip | ocr | vlm — user-facing, read by sorter.py), `tier` is WHICH
TIER processed the row (heuristic | clip | vlm — the incrementality marker). They
do not coincide: the OCR gate/rescue rewrites source to 'ocr' inside the fast pass,
and the VLM gate deliberately leaves clear personal photos on source='clip' — under
the old `source`-based marker both kinds of rows failed the "already processed"
check and were reclassified on EVERY run (with the deep tier on, that meant the
whole collection). `classify()` computes `active_tier` (see below) and writes it to
every row it touches; `todo` is the rows whose `tier` differs from it, so a
fast<->deep switch (either direction) reprocesses instead of losing rows, and a
repeated run with the same tier processes nothing.

F48 (#28, V1 profile): the junk-stage bottleneck is not the models but the SECOND
decode of the frame inside OCR (`imaging.decode_rgb(path, max_edge=1280)` — 315
ms/frame, ~80% of the junk stage). Reason: the default JPEG-draft headroom in
decode_rgb (margin=2×, see imaging.py) on typical camera frames (~4000px) does not
pass the first halving threshold for max_edge=1280 -> draft silently does not fire,
the full frame is decoded. `easyocr_text_frac_detector` now passes
`draft_margin=imaging._DRAFT_MARGIN_AGGRESSIVE` (1.0, an opt-in parameter of
decode_rgb) — draft kicks in, the decode is many times cheaper; `text_frac` (the
fraction of area under text) does not change from this, the document/photo verdict
accuracy is preserved (the ratio is scale-robust). Other decode_rgb consumers (thumbs
in ui.py/sorter.py, the VLM decode) stay on the default margin — unaffected.

F67 supersedes that decode path: OCR and VLM take the frame from the shared disk
preview cache (`imaging.decode_rgb_preview`) instead of decoding the original, so
the draft margin no longer matters here — a 1536px preview is cheap to decode by
itself, and the cost is shared with the pHash/CLIP stages.

F73: with the decode that cheap, what is left of the junk stage is `reader.detect`
itself — and it used to run strictly serially on the pipeline thread (py-spy on a
live 24.5k-frame run: 4.27 files/s, every decode worker idle). The detect calls are
independent, so they now go through `_OcrPool`: K threads, each with its OWN easyocr
Reader (a Reader holds a torch model and its buffers — sharing one is not safe, the
same reason F12.1 gives every faces worker its own FaceAnalysis). The per-chunk loop
is split into three phases — the pre-OCR verdict, the parallel text_frac, then the
verdicts and the DB writes — and only the middle one leaves the caller's thread, so
SQLite stays single-writer. This is a perf change only: the gate `run_ocr`, the
thresholds and the order in which verdicts are applied are untouched, so the
classification is byte-for-byte what K=1 produces.

F95: the VLM weights are loaded by `naming.shared_vlm`, not by this module — the
event-naming stage now runs the same Qwen2.5-VL and two copies do not fit in VRAM.
Only the loading moved: the prompt, the decode, the label parsing and the graceful
fallback around building the classifier are unchanged.

F90: OCR still runs on 28% of the frames and changes 2% of the verdicts (14:1), and
`text_rescue_docscore_min` — the number that decides that ratio — was set by eye and
never measured. The gate is worth its cost (it catches a real document CLIP scored
low, and letting one of those into the city folders is the expensive error), so the
threshold is not something a worker may quietly raise; it is a decision for the user
in front of a table. The tool that prints that table is
`scripts/measure_ocr_gate.py`, and for it to price the REAL gate the verdict/gate
branches now live in functions of their own — `clip_verdict`, `ocr_gate_open`,
`apply_text_frac`, over the thresholds of `gate_settings` — instead of inline in the
classify() loop. classify() calls exactly those functions, so the measurement cannot
drift away from the pipeline. Behaviour is unchanged; no threshold moved.

F100: the stage now names the phase it is in (CLASSIFY_PHASE_* below), through the
same optional `progress.phase(name)` channel F84 built for clustering. It used to name
nothing, and with the deep tier on that showed. Measured on the live run of
2026-07-28 (24 196 frames, 7 896 of them past the candidate gate): the counter runs
through the fast pass to 24 196/24 196 and then silently RE-BASES — `total` becomes
7 896, `done` restarts at zero — with `"phase": null` throughout. The numbers were
always honest; what was missing is the sentence explaining why the bar just jumped
back to the start against a threefold smaller denominator. The VLM phase reports a
real `(done, total)` over the gate's candidate list — unlike HDBSCAN it is
measurable, because the candidates are known before the loop starts.

The one place the bar could genuinely freeze is fixed here too: a VLM error used to
`continue` PAST the progress call, so the counter stopped for exactly as many frames
as the model failed on. Observability only: no verdict, threshold or gate is touched,
and a callback without a `phase` channel (the CLI, quiet mode, tests) behaves exactly
as it did.

F101: the deep tier earns its keep — on the live run of 2026-07-28 it changed 2 592 of
24 196 verdicts (10.7%), 2 202 of them into `product`, a class the fast tier does not
produce at all — but it took ~95 minutes at 1.38 frames/s, which is a weekend job, not
a default. The profile said the pass is not heavy but SEQUENTIAL: ~0.6 s of CPU
(decode + the processor's image preprocessing) then ~0.19 s of GPU per frame, strictly
alternating — 0.84 cores busy out of 24, the card at ~26%. Batching was ruled out by
that same measurement (a starved GPU does not want bigger portions), so the lever is
the one F87 used for faces: run the CPU half of several frames while the GPU half of
the previous one is running. `_vlm_labels` does that — `vlm_workers` threads prepare,
this thread generates and writes, and the queue is bounded so the frames in flight
cannot grow into RAM (the prepared tensors stay on the CPU, see naming.qwen_runtime,
so the VRAM peak is what it was).

Not one verdict may move because of it: labels come back in the CANDIDATE ORDER (a
FIFO of futures, not "whatever finishes first"), the model still sees one frame per
call with the same prompt and the same greedy decode, the writes still happen on this
thread alone, and a frame whose preparation fails still keeps its fast verdict with a
warning and still steps the progress bar (F100).

F113: this stage now also fills `frame_quality` — the per-frame signals a later consumer
needs to pick the best frame of a burst and to recognize a shot nobody meant to take.
Every signal is taken with the CHEAPEST TOOL THAT CAN ANSWER IT, and the next tier up is
only paid for where the cheap one is not sure:

* sharpness — the variance of the laplacian over the shared preview. Milliseconds, no
  model, no toggle: the data is wanted by every future consumer and costs nothing.
* pets (cat/dog/pet) — a prompt group APPENDED to the main CLIP call of the stage, behind
  `features.pets`. Not a second pass and not a second call. "Is there a cat in the frame"
  is a question about an object, which is what CLIP does well; the CLIP failure measured
  in F110 was about the PURPOSE of a frame, a different question. And the arithmetic is
  not close: the same coverage through the VLM is 19 757 x 0.78 s = 4.3 hours.
  `_group_probs` keeps the junk verdict exactly where it was — a renormalized slice of a
  softmax IS the softmax over that slice — so `naming.junk_threshold` does not move under
  a threshold that was measured against three prompts.
* eyes open / a subject at all / an accidental shot — the local VLM, behind `vlm.quality`,
  over the UNCERTAIN BAND only (sharpness in the zone where it decides nothing, or a CLIP
  junk-group score too low to mean anything) inside `vlm.quality_scope` (pHash groups by
  default; F125 adds `faces` — the frames a face was actually found on, which is the only
  population the eyes question has, and without a faces run that scope asks nothing at
  all). The answer is one line of keywords read leniently (the F96 lesson: asked for a
  composite format the model ignores it), and an answer that does not parse leaves NULL —
  never False. NULL means "not asked"; a consumer that reads it as "no" would decide that
  a frame nobody ever looked at has its eyes shut.

The quality half keeps its OWN incrementality marker (`frame_quality.source`), because the
two halves go stale independently: switching `features.pets` on does not change a single
junk verdict, and a collection classified before this feature existed has no quality rows
at all.

F128: the CLIP vector this stage computes for every frame it looks at is now KEPT
(`clip_embeddings`), where it used to be read for three scores and dropped. Nothing is
shown for it — the value is that the next feature of that class (search by words, an album
from a query, scene clustering, "frames like this one") reads a table instead of paying
for a full CLIP pass over the collection.

It is deliberately not a fourth pass over anything: the vector comes out of the caching
classifier that has just scored the chunk (`landmarks.CachingFeatureClassifier.features`),
so the number of model calls a run makes is exactly what it was. Three properties of the
row carry the reasoning, and the schema comment states them once more:

* `model` is written always. Vectors of different models are not comparable, so a row
  whose model differs from the current config is RECOMPUTED, never used — the same rule
  the F120 prompt fingerprint applies to the quality answers.
* the vector is stored L2-normalized in float32, little-endian. Normalized so cosine
  similarity is a plain dot product for every consumer. float32 is a MEASURED decision and
  not the obvious one: half precision would halve a table that reaches 920 MB at 300 000
  photos, so the brief proposed it and made it conditional on the ranking surviving. It
  does not — over 256 unit vectors of the real width, 18 of 20 queries come back in a
  different order in float16 (tests/test_clip_embeddings.py keeps that measurement) — and
  the pre-committed answer to that was float32 rather than a softer test.
* the population is the one `frame_quality` has, by the F120 argument — the embedding of a
  screenshot or a product shot is noise in a search over personal photographs — and this
  half, like the other two, keeps its own incrementality marker (`clip_embeddings.model`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Generator, Sequence

import numpy as np
from PIL import Image

from . import imaging
from .config import Config, FeaturesConfig

# F102 moved the workers knob to `vlm.workers` and this resolver along with it (the old
# `naming.vlm_workers` address is still honoured there) — but this module is where it was
# born and where the measurement scripts import it from, so the name stays re-exported.
from .config import resolve_vlm_workers  # noqa: F401
from .landmarks import Classifier, batched, clip_classifier
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

# F29: the folder signal — the file is in a Screenshots/Screenshot directory (any
# path segment, case-insensitive). A "floor" for verdict='photo': such a file
# cannot stay an ordinary photo (see the override in classify).
_SCREENSHOT_DIRS = {"screenshots", "screenshot"}


def _in_screenshots_dir(path: str) -> bool:
    """True if any path segment == screenshots|screenshot (case-insensitive).
    Splitting on both separators — in the DB paths come with both `\\` and `/`
    depending on the indexing platform."""
    return any(
        seg.lower() in _SCREENSHOT_DIRS for seg in re.split(r"[\\/]", path)
    )

# F15/F22: a separate CLIP run for documents, its own softmax group (does not
# share the probability mass with _CLIP_CLASSES). Anti-classes (an ordinary photo +
# street/outdoor scenes — F22: they pull probability mass away from travel photos of
# buildings with signs, which were otherwise caught as receipt/paper/scan) are
# excluded from the document score; the max is taken ONLY over the document
# subclasses.
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

# #14/V1: a cheap CLIP prefilter for "productness" — the same trick as the document
# score (its own softmax group, personal-photo anti-classes excluded from the
# product score). Serves ONLY as a candidate gate for the VLM (not a final verdict):
# files with a high product_score go to the expensive VLM, which decides
# product/document/personal_photo. That way the VLM is not run on every frame.
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

# F113: pets — the one frame-quality question CLIP can answer, so it is answered by the
# CLIP call this stage already makes. "Is there a cat in this frame" is a question about
# an OBJECT in the picture, which is what CLIP was trained on; the CLIP failure we
# measured (a document against a product, a beach 0.95 against a medical form 0.79) was a
# question about the PURPOSE of a frame — a different task. The arithmetic decides the
# rest: asking the VLM about pets across a whole collection is 19 757 x 0.78 s = 4.3
# hours, while these prompts ride along on frames CLIP is looking at anyway.
#
# They are APPENDED to the junk prompt list, not scored on a pass of their own, and the
# junk verdict is protected from them by `_group_probs`: a softmax restricted to a subset
# of its own inputs and renormalized IS the softmax over that subset. So the three junk
# classes keep exactly the probabilities they had before these prompts existed and
# `naming.junk_threshold` — a measured number — does not silently move under them.
_PET_POS_CLASSES: tuple[tuple[str, str], ...] = (
    ("cat", "a photo of a cat"),
    ("dog", "a photo of a dog"),
    # F121: was "a photo of a pet animal at home", and it was the worst class of the
    # three — a review of all 649 of its frames found people and children in it. "At
    # home" describes a SCENE, so the prompt attracted domestic scenes with a living
    # being in them rather than animals. Naming the animals instead keeps the class for
    # what it is for: the pets that are neither a cat nor a dog.
    ("pet", "a photo of a rabbit, a hamster, a bird, a horse or another animal"),
)
# Anti-classes for the pet group — the same device the document and product groups use.
# Without somewhere for the probability mass of a pet-less frame to go, every photo comes
# out as the most cat-like of the three cat prompts.
# F120: the first live run said the two anti-classes below were not enough. CLIP does not
# separate a thing from a PICTURE of a thing unless something else is offered to take that
# probability, so `a photo of a cat` matched drawn cats, a plush toy landed in `dog`
# alongside a hotdog, and people in fur coats came out as `pet`. Measured contamination
# before these prompts: 45% of `dog` and 15% of `cat` were not photographs of an animal.
#
# Each anti-class below answers one observed failure, and they are anti-classes rather
# than a higher threshold on purpose: a drawn cat is a CONFIDENT cat to CLIP, so no
# threshold separates it — the probability has to have somewhere else to go.
_PET_ANTI_CLASSES: tuple[tuple[str, str], ...] = (
    # F121: a review of the whole population after the first pass found people and
    # CHILDREN in the general class, so the people prompt names them.
    ("people", "a photo of a person, a child or a group of people, with no animal"),
    ("scene", "a photo of a place, a building or an object, with no animal in it"),
    # drawn cats in `cat`
    ("drawing", "a drawing, painting, cartoon or illustration of an animal"),
    # F121: the drawing prompt does not catch these, and it should not — a wallpaper of
    # a cat IS a photograph of a cat, and CLIP is right about that. The distinction the
    # collection needs is not "drawn or photographed" but "mine or somebody else's".
    ("stock", "a wallpaper, a stock photograph, a poster or a magazine picture"),
    # F121: the "puppies" frame. CLIP reads lettering and believes it over the picture —
    # the typographic weakness it has been known for since the original paper.
    ("text", "a picture with large text, a caption or lettering written on it"),
    # F121: two plush dogs still got through the previous wording; naming the toy as the
    # SUBJECT of the shot rather than as an adjective is what separates it from an animal.
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

# Where each group sits in the prompt list of the single call (start, stop).
_JUNK_GROUP = (0, len(_CLIP_CLASSES))
_PET_GROUP = (len(_CLIP_CLASSES), len(_CLIP_CLASSES) + len(_PET_CLASSES))


def clip_prompts(pets: bool) -> list[str]:
    """The prompts of the ONE main CLIP call of the stage; pets appended when asked for.

    With pets off the list is byte-for-byte what it always was — same prompts, same order,
    same text-embedding cache key — so a run with the toggle off is not changed by this
    feature in any way.
    """
    prompts = [prompt for _cls, prompt in _CLIP_CLASSES]
    if pets:
        prompts.extend(prompt for _cls, prompt in _PET_CLASSES)
    return prompts


def _group_probs(probs_row: np.ndarray, group: tuple[int, int]) -> np.ndarray:
    """The softmax over ONE prompt group, recovered from the row of the shared call.

    softmax(x)_i / sum over a subset == softmax over that subset, so renormalizing the
    slice gives exactly the probabilities a separate CLIP call over those prompts alone
    would have produced — which is what lets one call serve two independent questions.
    A row of zeros (a frame that did not decode) has no mass to renormalize and is
    returned as it is: score 0, the same "no signal" it has always meant.
    """
    part = probs_row[group[0]:group[1]]
    if len(part) == len(probs_row):
        return part  # nothing else in this softmax — already normalized, do not touch it
    total = float(part.sum())
    return part / total if total > 0 else part


def pet_verdict(probs_row: np.ndarray, threshold: float) -> tuple[str | None, float]:
    """(class, score) of the pet group -> the class is None below `threshold`.

    The score is returned either way and stored either way: a threshold that was chosen
    from a distribution has to be re-choosable from the stored scores, without a new pass
    over the collection.

    F122: ONE class is stored, whichever positive prompt won. A labelled sample of 320
    frames said the two halves of this signal are of very different quality — "is there
    an animal here" is right 92% of the time at 0.70, while WHICH animal was the part the
    review kept finding wrong (people landing in `dog`, a concert photo in the general
    class). So the ensemble of three prompts stays — it is what the 92% was measured on —
    and only its unreliable half stops being published.

    The three prompts are deliberately NOT collapsed into one. The score is the max over
    the positives of a softmax across the whole group; merging them would move the
    probability mass into a single class, raise every score, and invalidate the threshold
    the measurement chose.
    """
    group = _group_probs(probs_row, _PET_GROUP)
    if not len(group):
        return None, 0.0  # pets are off — this call had no pet prompts in it
    positives = group[:_N_PET_POS]
    score = float(positives[int(np.argmax(positives))])
    return (PET_CLASS if score >= threshold else None), score

# F37 (Phase A): defaults for naming.text_frac_min/text_frac_document, while the
# fields are not typed in NamingConfig (getattr pattern).
# text_frac_min — low (FP gate: almost no text -> not a document).
# F38: text_frac_document lowered 0.35 -> 0.15 (validation on real data:
# a document at an angle gave text_frac=0.247, scenes — 0.0-0.002; a large margin).
_DEFAULT_TEXT_FRAC_MIN = 0.08
_DEFAULT_TEXT_FRAC_DOCUMENT = 0.15

# F38: the OCR rescue (verdict='photo' -> 'document') is called only if the
# document-CLIP already "doubts whether it is a document" (doc_score in the zone
# 0.3..document_threshold) — clear scenes (doc_score≈0) do not run OCR, which is
# the perf win.
_DEFAULT_TEXT_RESCUE_DOCSCORE_MIN = 0.3

# F38: the detector decodes via imaging.decode_rgb and shrinks the frame before
# reader.detect() — a full-size decode is 1.2-3.2s/frame on large photos
# (F38 measurement), shrinking to ~1280px gives a x3-10 speedup.
_DEFAULT_TEXT_FRAC_DOWNSCALE_PX = 1280

TextFracDetector = Callable[[str, int | None, int | None], float | None]
# F73: builds the detector of ONE worker thread — see _OcrPool. Every thread needs
# its own (an easyocr Reader is not thread-safe), so the pool takes a factory, not a
# ready detector.
TextFracDetectorFactory = Callable[[], TextFracDetector]
# (file_id, path, width, height) — one OCR job for the pool. file_id keys the result
# back to the row, the pool does not preserve the input order.
OcrJob = tuple[int, str, int | None, int | None]

# F73: the default ceiling for naming.ocr_workers. Each worker keeps its own Reader
# (i.e. its own model copy in VRAM), so the default stays deliberately conservative —
# a higher value is a measurement on real hardware, not a default that may knock over
# a weak card.
_DEFAULT_OCR_WORKERS_CAP = 4


def resolve_ocr_workers(raw: dict | None) -> int:
    """How many OCR threads run in parallel — `naming.ocr_workers` in config.yaml.

    Read straight out of `cfg.raw`, the way hashing.resolve_workers reads
    `index.workers`: no typed field is added to NamingConfig for it. Default
    min(4, cpu_count) — see _DEFAULT_OCR_WORKERS_CAP on why it is that low.
    Absent / 0 / negative / garbage -> the default; the result is always >= 1.
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

    Worker threads are long-lived (they outlive a chunk) and pull jobs from a shared
    queue. Every worker builds its OWN detector on its first job — lazily, once,
    thread-local — and reuses it for every later frame and every later chunk: loading
    an easyocr Reader is expensive, building one per frame would cost far more than
    the detection it does. Nothing is shared between workers.

    VRAM degradation: if a worker cannot build its detector (typically no memory left
    for the second and further Readers), the stage does NOT crash — the pool shrinks
    to the detectors actually created (in the limit a single worker), the job goes
    back into the queue for a surviving worker, and the reason is logged. No silent
    fallback: it is exactly on such silence that the reason for a VLM refusal was
    lost once (F37-B lesson). If not a single detector can be built, text_frac()
    re-raises the build error — an unbuildable detector was a stage error before F73
    too.

    Results are returned to the caller's thread; nothing here touches SQLite.
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

        A detector error on one frame becomes None for that file_id ("no signal" —
        the gate/rescue leaves the verdict alone) and does not affect its neighbours,
        the same contract the try/except around reader.detect has always given.
        A file_id missing from the result also means "no signal".
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

        The slot is reserved under the lock but the factory runs outside it: loading
        several Readers in parallel is fine, and serializing it would only delay the
        start of the stage.
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

    An explicitly injected `text_detector` (a mock in tests, a caller with its own
    detector) is handed to every worker as it is — how it copes with threads is then
    the caller's business. Otherwise each worker builds its own easyocr detector.
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
    """Polygon area by the shoelace formula.

    easyocr boxes are quadrilaterals (slanted text is not a rectangle).
    """
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

    Lazy-import: the junk module is imported without easyocr (like faces with
    insightface). The Reader is built once and reused for the whole classify() run.

    F38: decode via imaging (not reader.detect(path) — cv2 silently does not read
    non-ASCII paths and HEIC, the frame dropped out of the OCR signal) + downscale to
    maxpx before detect() (a full-size decode — seconds/frame on large photos).
    The box area is computed RELATIVE to the downscaled frame.
    """
    import easyocr

    from .diagnostics import warn_if_gpu_mismatch

    # F63: easyocr(gpu=True) silently falls back to the CPU when torch is a CPU-only
    # build (verbose=False also hides easyocr's own "Using CPU" notice) — surface it.
    warn_if_gpu_mismatch()
    # verbose=False: suppresses the model-download progress bar (the █ / █ char),
    # which crashes the Windows cp1251 console (UnicodeEncodeError). The download
    # proceeds silently; the detector itself does not change from this.
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def text_frac(path: str, width: int | None, height: int | None) -> float | None:
        # F40: decode DIRECTLY at a reduced resolution (max_edge) — JPEG draft gives
        # a DCT downscale without a full decode (for large JPEGs — the main perf win);
        # decode_rgb finishes with a thumbnail down to max_edge, no separate one
        # needed. HEIC does not support draft (full decode), but detect still runs on
        # the shrunk frame.
        # F67: the frame now comes from the shared preview cache — the F48 aggressive
        # draft margin is no longer needed on this path (a 1536px preview is already
        # small, there is nothing left for draft to save). mtime/size for the cache
        # key come from a local stat: the TextFracDetector signature stays as it is.
        try:
            st = os.stat(path)
        except OSError:
            return None  # vanished/unreadable file — same contract as a decode error
        img = imaging.decode_rgb_preview(path, st.st_mtime, st.st_size, max_edge=maxpx)
        if img is None:
            return None  # could not decode (corrupt/unrecognized file)
        # detect() — box DETECTION only, without text recognition: for density the
        # areas are enough, and the easyocr recognition path fails on degenerate
        # crops (cv2.resize !ssize.empty). Faster and does not load the recognition model.
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


# F37 (Phase B): VLM 3-way classify_media(path) -> label; mapping to verdict
# below. An unrecognized model answer -> 'personal_photo' (conservative, the same
# principle as everywhere in junk.py — better to let a document/product through as
# a photo than to lose a real photo).
VlmClassifyFn = Callable[[str], str]

# F101: the label a frame gets without asking the model at all — it did not exist on
# disk any more, or did not decode. Conservative by the rule above, and unchanged from
# when these two returns sat inline in classify_media.
_VLM_FALLBACK_LABEL = "personal_photo"

_VLM_LABEL_TO_VERDICT: dict[str, str] = {
    "personal_photo": "photo",
    "document": "document",
    "product": "product",
}

# F95: the model name and its input size describe the MODEL, not this stage, and the
# naming stage now runs the same weights. F102 finished that thought — they are the
# `vlm:` config section, and these two are only the defaults for a caller that has no
# config in hand (a measurement, a test).
_DEFAULT_VLM_MODEL = DEFAULT_VLM_MODEL
_DEFAULT_VLM_MAX_EDGE = VLM_MAX_EDGE

# One label is one short word — a longer budget only buys the model room to explain
# itself, which the parser below would then have to wade through.
_VLM_MAX_NEW_TOKENS = 8

_VLM_PROMPT = (
    "Classify this image into exactly one category: personal_photo, document, "
    "or product.\n"
    "personal_photo = a personal/casual photograph of people, places, pets or "
    "everyday life.\n"
    "document = a photographed or scanned document, receipt, ID card, form, or "
    "other text-heavy paper.\n"
    "product = an item photographed for sale or a marketplace/e-commerce style "
    "listing photo (isolated object, catalog shot).\n"
    "Answer with exactly one word: personal_photo, document, or product."
)


@dataclass(frozen=True)
class PreparedFrame:
    """What the CPU half of the deep tier produces for one frame (F101).

    Either model `inputs` (the frame decoded and preprocessed, waiting for the GPU) or
    a ready `label` — a frame that vanished or would not decode never reaches the
    model, exactly as in the serial classifier, and carrying that answer through the
    pipeline keeps the GPU half free of file-system branches.
    """
    inputs: Any = None
    label: str | None = None


@dataclass(frozen=True)
class SplitVlmClassifier:
    """classify_media(path) as its CPU half and its GPU half (F101).

    It IS a VlmClassifyFn (calling it does both halves in turn, which is the serial
    classifier), so nothing that only knows the old interface has to change. The deep
    tier checks for this type to decide whether the pass can be pipelined at all.
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

    Everything that belongs to this stage lives here: the prompt, the decode and the
    parsing of the answer. Decode — via imaging.decode_rgb_preview (Unicode/HEIC-safe,
    the Phase A/F38 lesson; F67: through the shared preview cache), downscale to
    max_edge before feeding the model.

    F101: when the runtime offers its halves (naming.SplitVlm) so does the classifier —
    `prepare` is the whole CPU part of a frame (decode + the processor), which is what
    the pipeline in _vlm_labels moves off this thread, and `classify_prepared` is the
    GPU part plus the label parsing. A runtime without the halves gets the plain
    serial classifier, unchanged.
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

    F95: the weights are no longer loaded here — naming.shared_vlm hands out ONE
    runtime per model name for the whole process, because the naming stage now runs
    the same model and a second copy does not fit in VRAM (peak 20.5 GB). The load is
    still lazy (transformers is imported inside the loader) and still fails only when
    the classifier is actually built — which the caller in classify() wraps in
    try/except for a graceful fallback to the fast tier.
    """
    return vlm_classifier_from(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_classifier_factory(max_edge: int) -> Callable[[str], VlmClassifyFn]:
    """The default `vlm_classifier_factory` of classify(), carrying `vlm.max_edge` (F102).

    The factory interface stays (model_name) -> classifier — tests inject their own, and
    widening it would make every one of them care about a number they do not use — so
    the configured input size travels in the closure instead.
    """
    return lambda model_name: qwen_vlm_classifier(model_name, max_edge=max_edge)


def _vlm_labels(vlm_fn: VlmClassifyFn, paths: list[str],
                workers: int) -> Generator[str | BaseException, None, None]:
    """Labels for `paths` IN INPUT ORDER, pipelined when that is possible (F101).

    Yields one item per path, in the order given: the label, or the exception the
    classifier raised on that frame (the caller logs it and keeps the fast verdict —
    the same contract the try/except around vlm_fn(path) has always had, only the
    raising moved).

    The pipeline needs both halves from the runtime (SplitVlmClassifier) and more than
    one worker; anything else — an injected test classifier, a runtime without halves,
    vlm_workers=1 — takes the serial path, which is the pre-F101 loop verbatim.
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

    A FIFO of futures, not "first finished wins": the frame whose future is at the head
    is the next one yielded, so the output order is the input order no matter how the
    preparations interleave. The GPU half runs HERE, on the consumer's thread — one
    stream of generate() calls, as before; several would only queue up inside the
    driver and cost VRAM.

    The window (2 per worker) is the RAM bound the brief asks for: at most that many
    preprocessed frames exist at once, and they are CPU tensors (naming.qwen_runtime
    keeps them off the card), so the VRAM peak is one frame's inputs — what it was when
    the pass was serial.
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

    The only signal (brief F13): an explicit Screenshot_/"снимок экрана" name.
    Screen-ratio (3:4/4:3 — the usual proportions of phone photos) and
    messenger-name→meme (a forwarded photo is often a real one) were removed — they
    were the main FP source on real family photos.
    """
    if camera_make or camera_model:
        return None  # shot with a camera — not junk
    name = Path(path).name
    if _SCREENSHOT_NAME_RE.match(name):
        return "screenshot"
    return None


def _is_real_photo(row: sqlite3.Row) -> bool:
    """Camera EXIF/GPS or the presence of detected faces — a veto against CLIP.

    Messengers strip EXIF from forwarded photos, so camera/GPS alone do not protect
    real photos without metadata (brief F13) — a face in the photo is an equally
    reliable "this is not a document/meme/screenshot" sign, added as a third veto
    condition. Used against false CLIP verdicts.
    """
    return bool(
        row["camera_make"] or row["camera_model"]
        or row["gps_lat"] is not None or row["has_faces"]
    )


# F90: the fast-tier verdict and the OCR gate, lifted out of the classify() loop.
# The gate is priced by scripts/measure_ocr_gate.py, which sweeps
# text_rescue_docscore_min over a grid — and a measurement is only worth anything if
# it replays the decision the pipeline actually makes. A second copy of these three
# branches in the script would drift from this one and quietly price the wrong gate,
# so both call the same functions. classify() behaves exactly as before.


@dataclass(frozen=True)
class GateSettings:
    """The thresholds the CLIP verdict and the OCR gate/rescue are built from.

    Read through getattr with the module defaults (see the F37/F38 constants above):
    the fields appeared in NamingConfig later than the code reading them, and the
    getattr pattern is what junk.py has always used for them.
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

    The order of the branches is the contract, not a detail: an explicit
    Screenshot_/"снимок экрана" name wins over everything (F22), then a
    high-confidence document-CLIP — BEFORE the camera/GPS/faces veto, because a
    photographed document carries camera EXIF (F15), then the veto (F13), then the
    junk classes. `doc_score` is None for frames with faces: the document pass is not
    run for them at all.
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

    `rescue_docscore_min` is a parameter rather than a field of GateSettings because
    F90 sweeps exactly this number over a grid to price the gate; classify() passes
    the configured one. The rest is the F38 condition unchanged: OCR only for the
    document<->photo pair, never for frames with faces, and the FP gate
    (verdict=='document') is not limited by the threshold — there are few documents
    anyway, and letting one through is the expensive error.
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
            # FP gate: CLIP is sure it is "document", but there is almost no text —
            # a scene (beach), not a document.
            return "photo", text_frac, "ocr"
        if verdict == "photo" and text_frac >= g.text_frac_document:
            # FN rescue: dense text over the whole frame — a document, even if the
            # CLIP score was low.
            return "document", text_frac, "ocr"
    return verdict, score, "clip"


# --- F113: the frame-quality cascade ------------------------------------------------
#
# Three questions, three prices. Sharpness is a laplacian over the preview every other
# stage has already paid for (milliseconds, no toggle, written always). Pets are a prompt
# group inside the CLIP call above (free, `features.pets`). Everything left — are the eyes
# open, is there a subject at all, is this a pocket shot — is a VLM at ~0.78 s per frame,
# so it is asked ONLY about the frames the cheap tiers did not settle (`vlm.quality`).
#
# The population rule is the F109 result put to use: sending the model the least confident
# 30% of frames kept 98.2% of the findings. There it was worthless because the probe
# learned from the model's own labels (a closed circle); CLIP has no such circle — it
# labels without being trained on anything of ours.

# path -> the variance of the laplacian, or None if the frame did not decode.
SharpnessFn = Callable[[str], float | None]
# path -> the model's raw answer about one frame (parsed by `parse_quality_answer`).
QualityAskFn = Callable[[str], str]

# The tier that produced a frame_quality row, and with it the incrementality marker.
QUALITY_SOURCE_CLASSIC = "classic"   # sharpness only
QUALITY_SOURCE_CLIP = "clip"         # + pets
QUALITY_SOURCE_VLM = "vlm"           # + the model answers over the uncertain band


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


def preview_sharpness_detector(max_edge: int) -> SharpnessFn:
    """The real detector: the shared preview cache, at a FIXED resolution.

    Fixed because the variance of the laplacian is scale-dependent — the same photo
    measured at 512 and at 1536 px gives two different numbers, and a threshold over a
    mixture of the two means nothing. `features.sharpness_max_edge` is that resolution;
    changing it invalidates every threshold chosen against the old one.

    Decoded grayscale straight away (the measure only looks at luma) and through
    `decode_rgb_preview`, so the cost on any stage after the first is a small-JPEG decode,
    not a decode of the original. A vanished or undecodable file is None — "no signal",
    the same contract the OCR detector gives.
    """
    def sharpness(path: str) -> float | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        img = imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge, grayscale=True)
        if img is None:
            return None
        try:
            return laplacian_variance(img)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not break the stage
            _log.warning("junk: резкость не посчиталась для %s: %s", path, exc)
            return None

    return sharpness


# One short line of keywords. F96's lesson is the reason it is not three fields of JSON:
# asked for a composite format the model ignores the format and answers in prose, and a
# parser that then finds nothing writes False where it should write "not asked".
# F122: the "accidental" question is gone, and it was measured out rather than dropped
# on a hunch. On a labelled sample the model called 76% of what it was shown accidental;
# of those, 5% actually were — and the frames it called DELIBERATE held twice that rate
# (10%). A signal that is slightly anti-correlated with the thing it names is not
# something a threshold repairs, and every token it occupied was paid for on every frame.
_QUALITY_PROMPT = (
    "Look at this photo and answer with keywords from this list only:\n"
    "eyes_open or eyes_closed — whether the people in the photo have their eyes open "
    "(use neither word if there are no people);\n"
    "subject or no_subject — whether the photo has a clear subject.\n"
    "Answer with those keywords separated by spaces, nothing else."
)
_QUALITY_MAX_NEW_TOKENS = 16

# Keyword -> value, per field, IN PRIORITY ORDER. The negatives come first on purpose:
# "no_subject" contains "subject", and a scan that met the positive first would read
# every refusal as agreement.
# F122: `is_accidental` is no longer asked, so it is no longer parsed. The COLUMN stays
# and stays NULL — "not asked" is exactly what NULL means here, dropping it would need a
# table rebuild, and a retired question is cheaper to leave documented than to excise.
_QUALITY_KEYWORDS: tuple[tuple[str, tuple[tuple[str, bool], ...]], ...] = (
    ("eyes_open", (("eyes_closed", False), ("eyes_open", True))),
    ("has_subject", (("no_subject", False), ("subject", True))),
)
_NON_WORD_RE = re.compile(r"[^a-z]+")


@dataclass(frozen=True)
class QualityFlags:
    """The three model answers about one frame. None means NOT ASKED / not understood.

    Never a False by default: a consumer reading a defaulted False would conclude that a
    frame it has never shown to anything has its eyes closed.
    """
    eyes_open: bool | None = None
    has_subject: bool | None = None
    is_accidental: bool | None = None

    @property
    def known(self) -> bool:
        """True if the answer carried at least one flag — i.e. it parsed at all."""
        return any(v is not None for v in
                   (self.eyes_open, self.has_subject, self.is_accidental))


def parse_quality_answer(answer: str) -> QualityFlags:
    """The model's answer -> flags, read leniently; nothing recognized -> all None.

    Lenient in the two ways that cost nothing and buy the answers a model actually gives:
    everything that is not a letter becomes a separator (so "eyes open." and "Eyes-Open"
    read the same as "eyes_open"), and a keyword is looked for anywhere in the line rather
    than as a whole answer, because the model likes to explain itself.
    """
    text = "_" + _NON_WORD_RE.sub("_", (answer or "").lower()) + "_"
    values: dict[str, bool | None] = {}
    for field_name, keywords in _QUALITY_KEYWORDS:
        values[field_name] = next(
            (value for keyword, value in keywords if f"_{keyword}_" in text), None)
    return QualityFlags(**values)


def vlm_quality_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                      max_edge: int) -> QualityAskFn:
    """The quality question over an ALREADY LOADED runtime (naming.shared_vlm).

    Deliberately the plain, serial path and not the split halves the deep junk tier uses:
    this population is a band inside a scope, not the whole collection, and the pipeline
    machinery would cost more reading than it saves seconds. The decode goes through the
    shared preview cache, Unicode/HEIC-safe, exactly as everywhere else here; a frame that
    will not decode gets an empty answer, which parses to "not asked".
    """
    def ask(path: str) -> str:
        try:
            st = os.stat(path)
        except OSError:
            return ""
        img = imaging.decode_rgb_preview(
            path, st.st_mtime, st.st_size, max_edge=max_edge)
        if img is None:
            return ""
        return describe([img], _QUALITY_PROMPT, _QUALITY_MAX_NEW_TOKENS)

    return ask


def qwen_vlm_quality(model_name: str = _DEFAULT_VLM_MODEL,
                     max_edge: int = _DEFAULT_VLM_MAX_EDGE,
                     ) -> QualityAskFn:  # pragma: no cover — ML, smoke test
    """The real quality asker — the SAME weights as everything else (F95): one per run."""
    return vlm_quality_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_quality_factory(max_edge: int) -> Callable[[str], QualityAskFn]:
    """The default `quality_vlm_factory` of classify(), carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_quality(model_name, max_edge=max_edge)


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
    vlm_quality: bool
    vlm_scope: str
    # F120: media classes no VLM is shown (`vlm.exclude_classes`).
    exclude_classes: frozenset[str] = frozenset()


def quality_settings(cfg: Config) -> QualitySettings:
    """`features:` + the `vlm:` quality keys of a config (or of a measurement)."""
    f = getattr(cfg, "features", None) or FeaturesConfig()
    vlm = cfg.vlm
    return QualitySettings(
        pets=bool(f.pets),
        pet_threshold=float(f.pet_threshold),
        sharpness_max_edge=int(f.sharpness_max_edge),
        sharpness_band=(float(f.sharpness_band_min), float(f.sharpness_band_max)),
        subject_score_min=float(f.subject_score_min),
        vlm_quality=bool(getattr(vlm, "quality", False)),
        vlm_scope=str(getattr(vlm, "quality_scope", "groups")),
        exclude_classes=frozenset(getattr(vlm, "exclude_classes", ()) or ()),
    )


# F120: the quality questions — is there a pet, are the eyes open, was this shot an
# accident — are questions about a PERSONAL PHOTOGRAPH. Asked of a screenshot or a
# product shot they produce an answer that means nothing, and the first live run showed
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
    """Is this frame one the cheap tiers did NOT settle? — the VLM population.

    Two independent ways in, either of them enough: sharpness inside the band where it
    decides nothing (clearly blurred is below it, clearly sharp above), or a junk-group
    CLIP probability of "a photograph" low enough that CLIP is saying it does not know
    what it is looking at. A frame that did not decode has no sharpness signal and is
    judged on the second condition alone.
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


def quality_scope_ready(conn: sqlite3.Connection, scope: str) -> bool:
    """May the quality VLM run under this scope at all? (F125, `faces` only)

    `scope: faces` is a HARD dependency and not a filter that happens to come out empty:
    the user's rule is "ask about the face markup only — no faces pass, no feature". So
    the model half is not even built, the reason is logged, and the cheap tiers carry on
    measuring: an optional add-on must not take the stage down with it.

    Note this is NOT the F121 ambiguity and does not repeat its logic. There "no face on
    this frame" and "nobody looked" are the same empty row and the answer is KEPT so the
    signal does not switch off silently; here the scope itself is the face markup, so a
    missing faces run leaves nothing to ask about in the first place.
    """
    if scope != "faces" or faces_stage_ran(conn):
        return True
    _log.warning(
        "junk: vlm.quality_scope='faces', но найденных лиц в базе нет — сначала "
        "нужен прогон стадии faces; VLM-вопросы о качестве пропущены, резкость и "
        "животные считаются как обычно")
    return False


def quality_scope_ids(cfg: Config, conn: sqlite3.Connection,
                      scope: str) -> set[int] | None:
    """File ids the quality VLM may be asked about; None — no restriction (`all`).

    `groups` is the default because that is where the question is actually asked: five
    frames of the same moment, which one is the keeper. `events` widens it to everything
    inside an event, `all` gives up the restriction entirely — and on a 20k collection
    that is the 4.3-hour option, which is why it is neither the default nor undocumented.

    F125: `faces` is the population the eyes question actually has — photographs a face
    was FOUND on, 7 341 of them on the live collection against 19 757 for `all`. The
    `bbox != '[]'` half of the predicate is the whole feature: that marker means "processed,
    no faces here" and stands on nearly every file, so a predicate without it turns "by
    faces" into "by everything". A frame with no verdict yet is kept for the same reason
    the quality half keeps it (a first run has not classified anything); a frame that is
    not a photograph is dropped by the F120 gate anyway, on the fresh verdict rather than
    the stored one.
    """
    if scope == "all":
        return None
    if scope == "events":
        return {int(r["file_id"]) for r in conn.execute(
            "SELECT DISTINCT file_id FROM event_files")}
    if scope == "faces":
        return {int(r["id"]) for r in conn.execute(
            """SELECT f.id FROM files f
               LEFT JOIN media_class mc ON mc.file_id = f.id
               WHERE (mc.verdict IS NULL OR mc.verdict = ?)
                 AND EXISTS(SELECT 1 FROM faces fa WHERE fa.file_id = f.id
                            AND fa.bbox != ?)""",
            (QUALITY_VERDICT, NO_FACES_BBOX))}
    from . import dedup  # local: dedup imports imaging/hashing, junk is not its consumer

    groups = dedup.near_duplicate_groups(conn, cfg.index.phash_max_distance)
    return {int(r["id"]) for group in groups for r in group}


@dataclass(frozen=True)
class FrameQuality:
    """One `frame_quality` row as Python types — None stays None, and is not a False."""
    file_id: int
    sharpness: float | None = None
    pet: str | None = None
    pet_score: float | None = None
    eyes_open: bool | None = None
    has_subject: bool | None = None
    is_accidental: bool | None = None
    source: str = QUALITY_SOURCE_CLASSIC


def _bool_or_none(value: object) -> bool | None:
    """SQLite 0/1/NULL -> False/True/None. The one place the distinction is decided."""
    return None if value is None else bool(value)


def read_frame_quality(conn: sqlite3.Connection,
                       file_ids: Sequence[int] | None = None) -> dict[int, FrameQuality]:
    """`frame_quality` by file_id — the reading side of the "NULL is not False" rule.

    The consumers of this table (F114: the web app, the sorter, the events stage) must not
    each rebuild the 0/NULL distinction out of raw rows; one of them would get it wrong
    exactly once and quietly discard frames nobody had looked at.
    """
    sql = ("SELECT file_id, sharpness, pet, pet_score, eyes_open, has_subject,"
           " is_accidental, source FROM frame_quality")

    def rows(cursor: sqlite3.Cursor) -> dict[int, FrameQuality]:
        return {
            int(r["file_id"]): FrameQuality(
                file_id=int(r["file_id"]),
                sharpness=None if r["sharpness"] is None else float(r["sharpness"]),
                pet=r["pet"],
                pet_score=None if r["pet_score"] is None else float(r["pet_score"]),
                eyes_open=_bool_or_none(r["eyes_open"]),
                has_subject=_bool_or_none(r["has_subject"]),
                is_accidental=_bool_or_none(r["is_accidental"]),
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


# The fast half of the cascade writes the row; the model half updates it in place. The
# three model columns are reset to NULL by the fast half on purpose: this run has not
# asked yet, and a leftover answer from a previous run would describe a frame the current
# settings may never look at.
_QUALITY_UPSERT = """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score,
                         eyes_open, has_subject, is_accidental, source, updated_at)
                     VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                     ON CONFLICT(file_id) DO UPDATE SET
                         sharpness = excluded.sharpness, pet = excluded.pet,
                         pet_score = excluded.pet_score, eyes_open = NULL,
                         has_subject = NULL, is_accidental = NULL,
                         source = excluded.source, updated_at = excluded.updated_at"""
_QUALITY_ANSWER_UPDATE = """UPDATE frame_quality
                            SET eyes_open = ?, has_subject = ?, is_accidental = ?,
                                updated_at = ?
                            WHERE file_id = ?"""


def _as_int(value: bool | None) -> int | None:
    """A flag for SQLite: True/False -> 1/0, None stays NULL ("not asked")."""
    return None if value is None else int(value)


def _unused_classifier(paths: list[str], prompts: list[str]) -> np.ndarray:
    """The classifier of a run that asks CLIP nothing — being called is a bug, not a case.

    A backfill of sharpness alone (F113, both toggles off, junk already classified) needs
    no model, and building one would be the entire cost of such a run. This stands in for
    it so nothing downstream has to carry an optional classifier around.
    """
    raise AssertionError(  # pragma: no cover — unreachable by construction
        "junk: CLIP вызван в прогоне, где он не нужен")


def quality_prompt_fingerprint(pets: bool, *, with_vlm: bool) -> str:
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
    """
    parts = list(clip_prompts(pets))
    if pets:
        # F122: what the stored value MEANS is part of what makes a row stale, not only
        # the text that produced it. Collapsing three class names into one changed the
        # meaning of `frame_quality.pet` without touching a prompt, and a marker blind to
        # that would have left every row saying `cat` and looking fresh.
        parts.append(PET_CLASS)
    if with_vlm:
        parts.append(_QUALITY_PROMPT)
    raw = "\x00".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def quality_tier(source: str) -> str:
    """The tier out of a stored marker — `vlm#1a2b3c4d` -> `vlm`.

    Consumers care which tier answered, not which revision of the prompts did; the
    fingerprint is for invalidation alone.
    """
    return source.split("#", 1)[0]


def _quality_source(use_clip: bool, pets: bool, ask: QualityAskFn | None) -> str:
    """The tier marker this run writes — and therefore what it considers up to date."""
    if ask is not None:
        return f"{QUALITY_SOURCE_VLM}#{quality_prompt_fingerprint(pets, with_vlm=True)}"
    if use_clip and pets:
        return f"{QUALITY_SOURCE_CLIP}#{quality_prompt_fingerprint(pets, with_vlm=False)}"
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
CLASSIFY_PHASE_WRITE = "junk_write"


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
    """

    def __init__(self, progress: ProgressCB | None) -> None:
        self._progress = progress
        phase = getattr(progress, "phase", None)
        self._phase: PhaseCB | None = phase if callable(phase) else None
        self._current: str | None = None
        self._total: int | None = None

    def enter(self, name: str) -> None:
        """Relabel to phase `name`, keeping the counter as it is."""
        if name == self._current:
            return
        self._current = name
        if self._phase is not None:
            self._phase(name)

    def start(self, name: str, total: int) -> None:
        """Enter a phase that counts its OWN items: caption and denominator together."""
        self._total = total
        self._current = None
        self.enter(name)
        self.step(0)

    def step(self, done: int) -> None:
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
    # (sharpness always, pets when the toggle is on); quality_candidates/answered are the
    # uncertain band and how much of it the model actually answered about — the pair that
    # says whether the band is worth what it costs.
    quality_rows: int = 0
    pets_found: int = 0
    quality_candidates: int = 0
    quality_answered: int = 0
    # F128: vectors written into `clip_embeddings` in this run. On a repeated run it is 0
    # and the table is unchanged — the observable sign that this half is incremental too.
    embeddings_stored: int = 0


class _QualityPass:
    """The frame-quality half of `classify`, kept out of its loop (F113).

    Owns three things and nothing else: which frames need quality work this run (its own
    incrementality, on `frame_quality.source`, mirroring how junk uses `media_class.tier`),
    what the cheap tiers say about a frame, and which frames that leaves for the model. It
    writes on the caller's thread, inside the caller's transaction — SQLite stays
    single-writer, as everywhere in this stage.
    """

    def __init__(self, conn: sqlite3.Connection, q: QualitySettings,
                 sharpness: SharpnessFn, ask: QualityAskFn | None,
                 scope_ids: set[int] | None, source: str, ids: set[int],
                 now: str, stats: JunkStats, faces_known: bool = False) -> None:
        self._conn = conn
        self._q = q
        self._sharpness = sharpness
        self._ask = ask
        self._scope = scope_ids
        self._source = source
        self._ids = ids
        self._now = now
        self._stats = stats
        # F121: whether the faces stage has EVER run on this index. Without that, "no
        # face on this frame" and "nobody has looked for one" are the same row, and
        # dropping the eyes answer on both would silently switch the signal off for
        # everyone who has not run `faces`.
        self._faces_known = faces_known
        # (file_id, path, has_face) — the third field decides whether the eyes answer is
        # believed for that frame.
        self._candidates: list[tuple[int, str, bool]] = []

    @property
    def candidates(self) -> list[tuple[int, str, bool]]:
        """Frames of the uncertain band, in file order — the model's whole population."""
        return self._candidates

    def wanted(self, file_id: int) -> bool:
        """Does this frame need quality work in this run? (its own incrementality)"""
        return file_id in self._ids

    def needs_clip(self) -> bool:
        """Does the quality half need the CLIP row of a frame at all?

        Two things want it: the pet group, and the subject score that decides half of the
        uncertainty band. Without a row the band would read every frame as "CLIP says
        nothing", which is the reading that sends everything to the model.
        """
        return self._q.pets or self._ask is not None

    def measure(self, file_id: int, path: str, probs_row: np.ndarray | None,
                verdict: str | None = None, has_face: bool = False) -> None:
        """The cheap tiers for one frame: measure, write the row, note the band.

        F120: a frame this run decided is NOT a personal photograph is dropped here
        instead of measured, and any row a previous run left for it is removed — the
        first live run wrote 24 196 rows over the whole collection, and the answers on
        screenshots, products and documents were the noise that made the signal unusable.
        """
        if verdict is not None and verdict != QUALITY_VERDICT:
            self._conn.execute("DELETE FROM frame_quality WHERE file_id = ?", (file_id,))
            return
        sharpness = self._sharpness(path)
        pet: str | None = None
        pet_score: float | None = None
        if self._q.pets and probs_row is not None:
            pet, pet_score = pet_verdict(probs_row, self._q.pet_threshold)
        self._conn.execute(_QUALITY_UPSERT, (file_id, sharpness, pet, pet_score,
                                             self._source, self._now))
        self._stats.quality_rows += 1
        if pet is not None:
            self._stats.pets_found += 1
        if self._ask is None:
            return
        if self._scope is not None and file_id not in self._scope:
            return
        subject = (float(_group_probs(probs_row, _JUNK_GROUP)[0])
                   if probs_row is not None else 0.0)
        if uncertain_band(sharpness, subject, self._q):
            self._candidates.append((file_id, path, has_face))

    def ask_model(self, report: _PhaseProgress) -> None:
        """The band, one frame per call — a failure on one frame costs only that frame."""
        if self._ask is None or not self._candidates:
            return
        self._stats.quality_candidates = len(self._candidates)
        report.start(CLASSIFY_PHASE_VLM, len(self._candidates))
        with self._conn:
            for i, (file_id, path, has_face) in enumerate(self._candidates):
                try:
                    flags = parse_quality_answer(self._ask(path))
                except Exception as exc:  # noqa: BLE001 — the cheap tiers must survive it
                    _log.warning(
                        "junk: VLM-качество не ответило по file_id=%s (%s) — "
                        "оставляю NULL", file_id, exc)
                    flags = QualityFlags()
                # F121: the prompt says "use neither word if there are no people" and the
                # model does not obey it — the first review found cats answered as
                # eyes_open and people in glasses answered as eyes_closed. The detector
                # already knows where a face is, so the answer is believed only there:
                # asking is free (one prompt, three questions, one call), believing is
                # not. Only when `faces` has actually run — otherwise "no face here" is
                # indistinguishable from "nobody looked".
                if self._faces_known and not has_face:
                    flags = QualityFlags(has_subject=flags.has_subject,
                                         is_accidental=flags.is_accidental)
                if flags.known:
                    self._conn.execute(_QUALITY_ANSWER_UPDATE, (
                        _as_int(flags.eyes_open), _as_int(flags.has_subject),
                        _as_int(flags.is_accidental), self._now, file_id))
                    self._stats.quality_answered += 1
                report.step(i + 1)


def classify(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    use_clip: bool = True,
    text_detector: TextFracDetector | None = None,
    text_detector_factory: TextFracDetectorFactory | None = None,
    vlm_classifier: VlmClassifyFn | None = None,
    vlm_classifier_factory: Callable[[str], VlmClassifyFn] | None = None,
    sharpness_detector: SharpnessFn | None = None,
    quality_vlm: QualityAskFn | None = None,
    quality_vlm_factory: Callable[[str], QualityAskFn] | None = None,
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

    sharpness_detector / quality_vlm / quality_vlm_factory (F113): the frame-quality
    cascade, written into `frame_quality` alongside the classification. The detector is the
    laplacian over the shared preview (no toggle — milliseconds, and both the "best frame"
    and the "blurred junk" consumers need it); pets are a prompt group inside the CLIP call
    this stage already makes, behind `features.pets`; the model answers about the uncertain
    band only, behind `vlm.quality`, with the same graceful fallback as the deep tier — a
    factory that raises leaves the cheap tiers running. All three are injectable for the
    same reason `classifier`/`text_detector` are: the suite must not load a model.

    F128: the CLIP vector of every canonical photograph is stored in `clip_embeddings`
    (`features.store_embeddings`, on by default). No parameter of its own: the vector is
    taken from the classifier that has just scored the chunk — a `features(paths)` method
    over its cache, which the real one (landmarks.CachingFeatureClassifier) has — so a
    classifier injected as a plain function stores nothing, logs why once, and changes no
    other behaviour of the stage.

    progress (F100): the usual `(done, total)` callback; if it also carries a
    `phase(name)` channel (progress.TaskProgress, ui._StageProgress) the stage reports
    which of its phases it is in — CLASSIFY_PHASE_*. A plain function without that
    channel is not an error and gets the counter alone, as before.
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

    # F37 (Phase B): the tier gate. use_clip=False — an explicit heuristics-only
    # mode, deep does not enter there (symmetric with CLIP below).
    vlm_fn: VlmClassifyFn | None = None
    # F102: the toggle is read off cfg.naming and not off cfg.vlm on purpose — the two
    # agree after load_config, but `--deep` and the UI checkbox force the tier for one
    # run by replacing exactly this field on their own copy of the config.
    if use_clip and bool(getattr(cfg.naming, "vlm_enabled", False)):
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

    # F113: the quality VLM, resolved exactly like the deep tier above and with the same
    # graceful fallback — a model that will not build must cost the cheap tiers nothing.
    # Its own toggle: a user may want the deep junk tier without the quality band or the
    # other way round, and the two populations have nothing to do with each other.
    # F125: `vlm.quality_scope: faces` also has to be SATISFIABLE before a model is built
    # — without a faces run its population is empty by construction, and loading 20 GB of
    # weights to ask nothing is the one outcome worth a check up front.
    q = quality_settings(cfg)
    quality_ask: QualityAskFn | None = None
    if use_clip and q.vlm_quality and quality_scope_ready(conn, q.vlm_scope):
        if quality_vlm is not None:
            quality_ask = quality_vlm
        else:
            q_factory = quality_vlm_factory or qwen_vlm_quality_factory(cfg.vlm.max_edge)
            try:
                quality_ask = q_factory(cfg.vlm.model)
            except Exception as exc:  # noqa: BLE001 — the band is optional, must not crash
                _log.warning(
                    "junk: VLM-качество недоступно (%s) — остаются классика и CLIP", exc)
                quality_ask = None

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
    quality_source = _quality_source(use_clip, q.pets, quality_ask)
    # F120: only personal photographs are asked the quality questions. Selection uses the
    # verdict ALREADY STORED, because this run's verdict is not known until the frame is
    # walked; a frame with no verdict yet (a first run) is included and settled below, and
    # a frame whose class changes is picked up on the next run. The lag is one run and it
    # is on the cheap half of the cascade.
    quality_ids = ({r["id"] for r in rows
                    if r["fq_source"] != quality_source
                    and r["mc_verdict"] in (None, QUALITY_VERDICT)}
                   if use_clip else set())
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
    if not work:
        return stats
    now = utcnow_iso()
    # F121: has the faces stage ever run here? One row is enough to tell — after that,
    # "this frame has no face" is a fact rather than an absence of evidence.
    faces_known = faces_stage_ran(conn)
    quality = _QualityPass(
        conn, q, sharpness_detector or preview_sharpness_detector(q.sharpness_max_edge),
        quality_ask,
        quality_scope_ids(cfg, conn, q.vlm_scope) if quality_ask is not None else None,
        quality_source, quality_ids, now, stats, faces_known)
    # F100: the phase channel of the callback, if it has one. The total is reported
    # right away, even if the stage is small/fast (#37); which phase the stage opens
    # with depends on the tier — a heuristics-only run classifies nothing, it only
    # writes verdicts.
    report = _PhaseProgress(progress)
    report.start(CLASSIFY_PHASE_CLIP if use_clip else CLASSIFY_PHASE_WRITE, len(work))

    heur_raw = {
        r["id"]: heuristic_verdict(
            r["path"], r["width"], r["height"], r["camera_make"], r["camera_model"],
        )
        for r in work
    }
    heur = {fid: v or "photo" for fid, v in heur_raw.items()}
    # F68: `tier` is written on every path and always equals active_tier — a row the
    # active tier touched must never stay unmarked (or marked by an older tier),
    # otherwise it is reclassified on every run.
    upsert = """INSERT INTO media_class (file_id, verdict, source, score, updated_at,
                                         tier)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET verdict = excluded.verdict,
                    source = excluded.source, score = excluded.score,
                    updated_at = excluded.updated_at, tier = excluded.tier"""

    if not use_clip:
        with conn:
            for r in work:
                verdict = heur[r["id"]]
                conn.execute(upsert, (r["id"], verdict, "heuristic", None, now, active_tier))
                stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
        report.step(len(work))
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
                text_fracs = ocr.text_frac(ocr_jobs)
                # F73, phase 3: apply the OCR signal, then write — on this thread only
                # (single writer) and in the original per-chunk order, so the verdicts
                # and stats are exactly those of the serial version.
                report.enter(CLASSIFY_PHASE_WRITE)
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
                    # group when the toggle is on, and the note of which frames the two of
                    # them failed to settle.
                    if quality.wanted(r["id"]):
                        quality.measure(r["id"], r["path"], probs.get(i), verdict,
                                        bool(r["has_faces"]))
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

    # F113: the last and most expensive step of the cascade — the three questions neither
    # the laplacian nor CLIP answers, asked only about the band those two left uncertain.
    # It runs after the deep tier for a plain reason: both want the same GPU, and the
    # verdict is what the rest of the pipeline depends on.
    quality.ask_model(report)
    # F120: enforce "only a personal photograph has a quality row" DIRECTLY, and do it
    # LAST, when every verdict of this run is written — the deep tier above reclassifies
    # frames, so a purge any earlier would judge them by the fast tier's answer.
    #
    # This is not the same guard as the per-frame one in `_QualityPass.measure`, and both
    # are needed: incrementality skips a frame whose `source` already matches, so a
    # collection measured before this rule — 24 196 rows over everything, all
    # `source='vlm'` — would keep its screenshots and documents precisely BECAUSE they
    # look up to date. One statement on an indexed column settles it for good.
    with conn:
        conn.execute(
            "DELETE FROM frame_quality WHERE file_id IN"
            " (SELECT file_id FROM media_class WHERE verdict != ?)", (QUALITY_VERDICT,))
        # F128: the same rule over the same population, for the same reason (see
        # _EmbeddingPass.purge) — and after the deep tier, whose reclassifications it has
        # to see.
        embeddings.purge()
    return stats
