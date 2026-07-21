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
Incrementality: the "already processed" marker toggles between 'clip' and 'vlm'
depending on the active tier — a fast<->deep switch reprocesses rather than losing
rows (see `active_source` in `classify()`).

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
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import imaging
from .config import Config
from .landmarks import Classifier, batched, clip_classifier
from .naming import naming_settings, utcnow_iso

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

    F38: decode via imaging.decode_rgb (not reader.detect(path) — cv2 silently does
    not read non-ASCII paths and HEIC, the frame dropped out of the OCR signal) +
    downscale to maxpx before detect() (a full-size decode — seconds/frame on large
    photos). The box area is computed RELATIVE to the downscaled frame.
    """
    import easyocr

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
        # F48: profiling found that the default decode_rgb headroom (draft_margin=2×)
        # in practice KILLS draft for typical camera frames at max_edge=1280
        # (a 2× margin asks for more than the first halving gives) — the decode
        # was full (315 ms/frame). text_frac is the FRACTION of area under text, not
        # the recognition itself, so sub-pixel downscale sharpness is not needed ->
        # the aggressive margin (imaging._DRAFT_MARGIN_AGGRESSIVE) is safe here.
        img = imaging.decode_rgb(
            path, max_edge=maxpx, draft_margin=imaging._DRAFT_MARGIN_AGGRESSIVE)
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

_VLM_LABEL_TO_VERDICT: dict[str, str] = {
    "personal_photo": "photo",
    "document": "document",
    "product": "product",
}

_DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# The VLM input is not for fine details like OCR, a large frame is not needed; saves
# VRAM/generation time (the same downscale logic as text_detector).
_DEFAULT_VLM_MAX_EDGE = 896

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


def qwen_vlm_classifier(
    model_name: str = _DEFAULT_VLM_MODEL,
) -> VlmClassifyFn:  # pragma: no cover — ML, smoke test
    """The real VLM classifier (Qwen2.5-VL via transformers).

    Lazy-import: junk is imported without transformers (like faces with insightface,
    easyocr above) — the module loads even if the `[vlm]` extras are not installed;
    it fails ONLY when actually building the classifier (which the caller in
    classify() wraps in try/except for a graceful fallback to the fast tier).

    Decode — via imaging.decode_rgb (Unicode/HEIC-safe, the Phase A/F38 lesson),
    downscale to _DEFAULT_VLM_MAX_EDGE before feeding the model.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    def classify_media(path: str) -> str:
        img = imaging.decode_rgb(path, max_edge=_DEFAULT_VLM_MAX_EDGE)
        if img is None:
            return "personal_photo"  # could not decode — conservative
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _VLM_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            # #30 (V1): greedy, NOT sampling. Qwen's default generation_config is
            # do_sample=True: on some frames fp16 logits go to NaN/inf ->
            # softmax gives a zero distribution -> torch.multinomial triggers a
            # CUDA device-side assert that POISONS the context (all subsequent
            # frames also fail). Classification does not need sampling — it needs
            # the most probable label; do_sample=False removes multinomial entirely
            # (verified: 150/150 candidates without an assert, incl. the trigger).
            out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().lower()
        for label in ("personal_photo", "document", "product"):
            if label in answer:
                return label
        return "personal_photo"

    return classify_media


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


@dataclass
class JunkStats:
    total: int = 0        # canonical photos in total
    processed: int = 0    # processed in this run (excluding source='clip' rows)
    clip_used: bool = False
    by_verdict: dict[str, int] = field(default_factory=dict)
    vlm_candidates: int = 0  # #14/V1: files selected for the VLM (doc/product zone)
    vlm_applied: int = 0     # of those, actually reclassified by the VLM (without errors)


def classify(
    cfg: Config, conn: sqlite3.Connection,
    classifier: Classifier | None = None,
    use_clip: bool = True,
    text_detector: TextFracDetector | None = None,
    vlm_classifier: VlmClassifyFn | None = None,
    vlm_classifier_factory: Callable[[str], VlmClassifyFn] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> JunkStats:
    """Classify canonical photos into media_class.

    use_clip=False — heuristics only (source='heuristic'); such rows will be
    reprocessed by CLIP on the next run with use_clip=True.

    text_detector (F37, Phase A): (path, width, height) -> text_frac | None.
    By default an easyocr detector is built (lazily, once per run) — as with
    classifier, the caller passes its own (mock) in tests.

    vlm_classifier / vlm_classifier_factory (F37, Phase B): the deep tier,
    opt-in via cfg.naming.vlm_enabled (default False, gated by use_clip=True —
    a heuristics-only run does not touch deep). vlm_classifier — a ready
    classify_media(path)->label (a mock in tests, like classifier/text_detector);
    vlm_classifier_factory(model_name)->vlm_classifier — a factory for the real
    build (qwen_vlm_classifier by default), replaced in tests to check the GRACEFUL
    FALLBACK: if the factory raises (no transformers, the model does not load, not
    enough VRAM), classify() catches the exception, logs it, and quietly continues
    on the fast tier (CLIP) — without crashing.
    """
    s = naming_settings(cfg)
    rows = conn.execute(
        """SELECT f.id, f.path, f.width, f.height, f.camera_make, f.camera_model,
                  f.gps_lat,
                  EXISTS(SELECT 1 FROM faces fa
                         WHERE fa.file_id = f.id AND fa.bbox != '[]') AS has_faces,
                  mc.source AS mc_source
           FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
           WHERE f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'
           ORDER BY f.id"""
    ).fetchall()
    stats = JunkStats(total=len(rows))

    # F37 (Phase B): the tier gate. use_clip=False — an explicit heuristics-only
    # mode, deep does not enter there (symmetric with CLIP below).
    vlm_fn: VlmClassifyFn | None = None
    if use_clip and bool(getattr(cfg.naming, "vlm_enabled", False)):
        if vlm_classifier is not None:
            vlm_fn = vlm_classifier
        else:
            model_name = str(getattr(cfg.naming, "classify_vlm_model", _DEFAULT_VLM_MODEL))
            factory = vlm_classifier_factory or qwen_vlm_classifier
            try:
                vlm_fn = factory(model_name)
            except Exception as exc:  # noqa: BLE001 — deep is optional, must not crash
                _log.warning(
                    "junk: VLM недоступна (%s) — откат на fast-ярус (CLIP)", exc)
                vlm_fn = None

    # incrementality: the "already processed by this tier" marker toggles between
    # 'vlm' and 'clip' — a fast<->deep switch reprocesses old rows with the right
    # tier rather than losing them or re-running fresh ones in a loop.
    active_source = "vlm" if vlm_fn is not None else "clip"
    todo = [r for r in rows if r["mc_source"] != active_source]
    stats.processed = len(todo)
    if not todo:
        return stats
    if progress:
        progress(0, len(todo))  # total right away, even if the stage is small/fast (#37)

    heur_raw = {
        r["id"]: heuristic_verdict(
            r["path"], r["width"], r["height"], r["camera_make"], r["camera_model"],
        )
        for r in todo
    }
    heur = {fid: v or "photo" for fid, v in heur_raw.items()}
    now = utcnow_iso()
    upsert = """INSERT INTO media_class (file_id, verdict, source, score, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET verdict = excluded.verdict,
                    source = excluded.source, score = excluded.score,
                    updated_at = excluded.updated_at"""

    if not use_clip:
        with conn:
            for r in todo:
                verdict = heur[r["id"]]
                conn.execute(upsert, (r["id"], verdict, "heuristic", None, now))
                stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
        if progress:
            progress(len(todo), len(todo))
        return stats

    if classifier is None:
        classifier = clip_classifier(s)  # pragma: no cover — ML, smoke test
    if text_detector is None:
        downscale_px = int(
            getattr(cfg.naming, "text_frac_downscale_px", _DEFAULT_TEXT_FRAC_DOWNSCALE_PX))
        text_detector = easyocr_text_frac_detector(downscale_px)  # pragma: no cover — ML, smoke test
    stats.clip_used = True
    document_threshold = float(cfg.naming.document_threshold)
    text_frac_min = float(getattr(cfg.naming, "text_frac_min", _DEFAULT_TEXT_FRAC_MIN))
    text_frac_document = float(
        getattr(cfg.naming, "text_frac_document", _DEFAULT_TEXT_FRAC_DOCUMENT))
    # F38: the FN rescue (verdict='photo') runs OCR only when the document-CLIP
    # already "doubts whether it is a document" — clear scenes (doc_score≈0) do not
    # run OCR (perf). The FP gate (verdict='document') is not limited by this threshold.
    text_rescue_docscore_min = float(
        getattr(cfg.naming, "text_rescue_docscore_min", _DEFAULT_TEXT_RESCUE_DOCSCORE_MIN))
    prompts = [prompt for _cls, prompt in _CLIP_CLASSES]
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
    with conn:
        for chunk in batched(todo, s.clip_batch_size):
            paths = [r["path"] for r in chunk]
            probs = classifier(paths, prompts)
            # F15: document-CLIP only for files without detected faces —
            # faces are an unconditional veto, a second pass for them is unneeded.
            noface_idx = [i for i, r in enumerate(chunk) if not r["has_faces"]]
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
            for i, (r, p) in enumerate(zip(chunk, probs)):
                best = int(np.argmax(p))
                score = float(p[best])
                if heur_raw[r["id"]] == "screenshot":
                    # F22: an explicit Screenshot_/"снимок экрана" name — a strong
                    # signal, it overrides both the document detection and the face
                    # veto (an avatar in a screenshot does not make the file a real photo).
                    verdict = "screenshot"
                elif i in doc_score and doc_score[i] >= document_threshold:
                    # no faces + a high-confidence CLIP document → a separate
                    # review category (F15), goes BEFORE the camera/GPS veto.
                    verdict = "document"
                    score = doc_score[i]
                elif _is_real_photo(r):
                    # camera EXIF/GPS or faces in the photo → this is a shot photo; a
                    # meme/screenshot does not carry those. The CLIP verdict does not
                    # override (otherwise on real data most "junk" would be false).
                    verdict = "photo"
                elif score >= s.junk_threshold:
                    verdict = _CLIP_CLASSES[best][0]
                else:
                    verdict = heur[r["id"]]
                source = "clip"
                # F37 (Phase A): the OCR signal only for the document<->photo pair,
                # only without faces (the same veto as the document-CLIP above) —
                # screenshot/meme are not touched.
                # F38: the rescue branch (photo) runs OCR only if doc_score is
                # uncertain (>= text_rescue_docscore_min) — a clear scene
                # (doc_score≈0) spends no OCR call. The FP gate (document) — without
                # a limit, as before (there are few documents anyway).
                run_ocr = not r["has_faces"] and (
                    verdict == "document"
                    or (verdict == "photo"
                        and doc_score.get(i, 0.0) >= text_rescue_docscore_min)
                )
                if run_ocr:
                    text_frac = text_detector(r["path"], r["width"], r["height"])
                    if text_frac is not None:
                        if verdict == "document" and text_frac < text_frac_min:
                            # FP gate: CLIP is sure it is "document", but there is
                            # almost no text — a scene (beach), not a document.
                            verdict, score, source = "photo", text_frac, "ocr"
                        elif verdict == "photo" and text_frac >= text_frac_document:
                            # FN rescue: dense text over the whole frame — a document,
                            # even if the CLIP score was low.
                            verdict, score, source = "document", text_frac, "ocr"
                if verdict == "photo" and _in_screenshots_dir(r["path"]):
                    # F29: the Screenshots folder is a "floor" for photo; we do not
                    # override document/meme (conservative, brief F29).
                    verdict = "screenshot"
                conn.execute(upsert, (r["id"], verdict, source, score, now))
                stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
                # #14/V1: selection into VLM candidates (deep refines doc/product/photo) —
                # without faces, not screenshot/meme, and the fast tier doubts: already a
                # document, OR the document-CLIP is in a suspicious zone, OR the
                # product-CLIP is above the threshold. Clear personal photos (both scores
                # low) are not touched by the VLM.
                if (vlm_fn is not None and not r["has_faces"]
                        and verdict not in ("screenshot", "meme")
                        and (verdict == "document"
                             or doc_score.get(i, 0.0) >= text_rescue_docscore_min
                             or product_score.get(i, 0.0) >= product_candidate_min)):
                    vlm_candidates.append((r["id"], r["path"], verdict))
            done += len(chunk)
            if progress:
                progress(done, len(todo))

    # #14/V1: the deep tier — the VLM only on the selected candidates (not all frames).
    # Each call in try/except: a VLM runtime error on one frame does NOT crash the
    # run (closes #31) — the file keeps its fast verdict.
    if vlm_fn is not None and vlm_candidates:
        stats.vlm_candidates = len(vlm_candidates)
        with conn:
            for j, (fid, path, fast_verdict) in enumerate(vlm_candidates):
                try:
                    label = vlm_fn(path)
                except Exception as exc:  # noqa: BLE001 — deep is optional, do not crash the run
                    _log.warning("junk: VLM-ошибка на file_id=%s (%s) — оставляю fast-вердикт",
                                 fid, exc)
                    continue
                verdict = _VLM_LABEL_TO_VERDICT.get(label, fast_verdict)
                # mark source='vlm' ALWAYS (even if the verdict matched fast) —
                # so a repeated run does not re-run the VLM on these candidates
                # (incrementality: mc_source == active_source == 'vlm').
                conn.execute(upsert, (fid, verdict, "vlm", None, now))
                if verdict != fast_verdict:
                    stats.by_verdict[fast_verdict] = stats.by_verdict.get(fast_verdict, 1) - 1
                    stats.by_verdict[verdict] = stats.by_verdict.get(verdict, 0) + 1
                    stats.vlm_applied += 1
                if progress:
                    progress(j + 1, len(vlm_candidates))
    return stats
