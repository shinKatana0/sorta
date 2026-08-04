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
* eyes open — the local VLM, behind `vlm.quality` (F177 retired the two questions that
  used to share the call), over the UNCERTAIN BAND only (sharpness in the zone where it
  decides nothing, or a CLIP junk-group score too low to mean anything) inside
  `vlm.quality_scope` (pHash groups by
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

F141: and a SECOND vector next to it (`search_embeddings`), from a different model, for
search alone. The one pass of this stage that encodes the same frames twice, and the
justification is a measurement: `ViT-L-14` scores 22% at top-5 on Russian queries against
98% for `xlm-roberta-base-ViT-B-32`, with four of eight concepts returning nothing at all.
Swapping the model would be free and is exactly what must not happen — the landmark (F75),
animal (F122) and cascade (F130) thresholds are calibrated on L14's numbers — so the
search side pays for its own pass instead, behind `features.search_index`, off by default
because ten and a half minutes per 20 000 frames is a cost a person agrees to. It keeps
the F128 rules unchanged: the model in every row, a mismatch means recompute, and the F120
population.

F130: the animal question becomes a CASCADE — CLIP selects widely, the VLM checks
(`features.pets_verify`, default off). F122 measured where the CLIP-only answer stands:
92% precision at 0.70 and 54% recall, with ~466 animals sitting below 0.30 among 18 400
frames, which no threshold reaches. The two halves of that have one cause and one fix:

* the errors left at 92% are drawn cats, plush toys, fur coats and a hotdog — frames CLIP
  is STRUCTURALLY unable to judge, because it compares a picture to a text as a whole and
  a picture of a cat is a cat to it. "Alive, or a rendering?" is a question about the
  meaning of the scene, and that is the one thing the VLM is better at;
* and once something checks the answer, the SELECTION can be widened. 0.70 was high
  precisely because nothing did. `features.pet_candidate_threshold` (0.30) decides who is
  shown to the model — 1 331 frames of the live collection, ~17 minutes at 0.78 s each,
  against 4.3 hours for asking about everything.

The model outranks the score, and nothing else does: a frame whose answer did not parse,
or whose answer never came, falls back to `pet_score >= pet_threshold` — never to "no".
The answer is stored (`frame_quality.pet_vlm`) because "rejected" and "never asked" are
different facts: without the column every later run would pay 0.78 s again for each of
the ~500 frames the model already turned down, and the interface would have nothing to
explain a removed label with.

F132: the stage also answers the question the duplicates view has always had to guess —
WHICH FRAME OF A NEAR-DUPLICATE GROUP IS THE ONE TO KEEP — and it asks it the way the
model answers best, comparatively: one call per group with the frames of that group in a
single prompt, not a score per frame (`dedup.keeper_vlm`, default off).

What that adds over the sharpness ranking is precisely what the laplacian cannot see:
closed eyes, a head turned away, a hand across the lens, the expression. What it does NOT
add is a decision — the answer lands in `group_keeper` as a recommendation with its
source, nothing is deleted or marked, and `dedup_choice` stays what it has always been:
the user's own decision, written by hand. The stage has no code path that writes it.

Three sizes bound the cost, and all three are settings rather than constants because the
measurement is per collection:

* `dedup.keeper_min_group_size` — the population. On the live collection 85% of the 791
  groups are PAIRS, and on a pair the sharpness ranking already compares two frames of one
  scene at one scale, honestly; a value of 3 leaves the 115 groups where the choice is
  genuinely unclear.
* `dedup.keeper_max_frames` — how many frames one question may hold. The same collection
  holds a group of 38: a 3B model asked to compare 38 pictures answers nothing usable and
  the context grows for no return. The frames sent are the best N by sharpness and the
  answer applies to the group as a whole.
* the group key itself (`dedup.group_key`) — a sha1 over the sorted file ids, because
  groups have no id: they are recomputed by union-find on every call. Members change ->
  the key changes -> the answer is asked again. Nothing else could tie an answer to a
  group.

The answer is read leniently (F96 again) and an unreadable one is not a failure of the
feature: the group keeps the sharpness ranking, stored as `source='sharpness'`, which is
exactly the recommendation the interface showed before this existed.

F140: the search by words (F134) put memes and screenshots at the top of its results, and
the table it searches is clean — all 19 753 rows of it carry the verdict `photo`. So this
was not junk leaking into the index but THIS STAGE being wrong about ~4% of what it calls a
photograph, an error nothing made visible until a query did. Those frames are laid out by
city today, and they land in the duplicates, in the quality signals and in the albums.

They are found by a zero-shot query over the vectors F128 already stores — no new pass over
any model:

    junk_score = max(similarity to screenshot/meme/text/receipt)
               - max(similarity to a photograph)

and the whole feature is what is NOT done with that number. Reviewed by eye on the live
collection: above +0.05 the 93 frames are junk outright, but the band +0.02..+0.05 still
holds ~17% real photographs. Applied as a verdict at 85% precision it would take ~150
living pictures out of the layout — exactly the mistake F130 measured for animals, where a
signal of 80-85% applied directly makes a 92% baseline worse. So the score SELECTS and does
not judge:

* it is computed for every photograph with a stored vector and kept
  (`frame_quality.junk_score`); a frame without one (a heuristics-only run,
  `store_embeddings: false`) gets NULL and is no candidate;
* frames above `features.junk_rescue_threshold` (0.02, the measured row above — 955 frames,
  ~12 minutes at 0.78 s each) are shown to the VLM, which answers with a verdict rather
  than a resemblance, and only the model's answer moves `media_class`;
* with the deep tier off nothing is reclassified at all, and with `features.junk_rescue`
  off nothing is even computed. The score prompts and the model's question both enter
  `quality_prompt_fingerprint`, so editing either invalidates the scores it produced (the
  F120 rule).

It is the same device as the OCR rescue gate (F38): a cheap signal decides who is worth an
expensive check. Cheaper, in fact — this one is a matmul over a table that already exists.

F154: the animal label gets a third tier — an OBJECT DETECTOR (`detect.py`), and the shape
it takes is the whole decision. Measured on 200 hand-labelled frames (2026-08-02, at
confidence 0.5), a COCO detector is 62% precision / 87% recall on animals against the CLIP
label's 71% / 33% — but 42% precision on people where the face boxes are ~100% (F152), and
20% / 15% on food, which COCO has no class for at all. It earns its keep on one slice of
three, and a pass over the collection to get it is 83.8 ms x 22 096 = 30.8 minutes.

So it is a CASCADE, the same device F130 and F140 already pay for: a zero-shot query over
the stored vectors ranks the collection, the top `features.detector_candidates` (2 000)
frames go to the detector, and ~3 minutes replace ~31. Recall is then bounded by the
query's own at that depth, and precision rises from the query's 43% to the detector's 62%.
`_DetectorPass` below is that pass; `detect.py` holds the model, the classes and the rule.

Three properties are the feature rather than its implementation:

* it is subordinate to a master switch of its own — `detect.enabled`, NOT `vlm.enabled`
  (this is not a VLM), under the F145 rule that no subordinate key raises a model by
  itself. Without stored vectors there are no candidates, and the stage says so instead of
  falling back to a pass over everything;
* the answer OVERRIDES the CLIP label in both directions, and falls back to the previous
  rule — never to "no animal" — whenever the detector says nothing (see
  `detect.cascade_label`, which also states why a frame the F130 check has answered about
  keeps that answer: a box detector calls a drawn cat a cat);
* what it found is stored with the class, the confidence and the box (`detections`),
  because there is nowhere else those could come from. F122 closed the cat/dog split for
  CLIP, which could not tell the species apart; a detector can, and the table is what makes
  that a query rather than another pass.

WHERE THE ANSWER IS ACTUALLY READ, because it is not here: since F137 the CONSUMERS of the
animal slice — the album, the "Animals" tab, the Overview counter — do not read
`frame_quality.pet` at all. They derive the verdict when they read, out of `pet_score`,
`pet_vlm` and (F160) the `detections` row, through `sorter.animal_auto_sql`, so that a
threshold moved in the config moves the slice without a run. F154 shipped without that
branch and the gap was the worst kind: the stage ran, the boxes were in the database, and
nothing a user looks at moved. F160 wrote the tier into that expression and made the two
spellings of the rule — `pet_label` here, `animal_auto_sql` there — a case table that is
run through both, so the fifth source of this label cannot repeat the silence.

F155: the blur filter this stage feeds catches 6% of what a person calls blurred — 2 of 33
on a hand-checked sample of 200 frames — and the reason is not the threshold. The variance
of the laplacian over a whole frame answers "how much detail is in this picture", which is
a different question from "is it in focus": a detailed sharp street and a smooth blurred
face give the same number, and blurred frames sit in every band up to 400. Nothing about a
whole frame is comparable across frames, so no cut through that number can be.

A FACE IS COMPARABLE. It is the one object whose content is roughly constant from frame to
frame, so the same variance measured inside it means the same thing twice. On the 68 frames
of that sample that have a face (13 of them blurred): 62% recall at a threshold of 200,
against 15% for the whole frame at 300, for a comparable number of frames flagged.

`frame_quality.face_sharpness` is that number, and three properties of it are the feature:

* it costs NO NEW PASS. `preview_sharpness_detector` decodes the shared preview once, as
  before, and takes the second variance over a crop of the same array. What did change is
  that the decode now applies the EXIF orientation — the boxes are written in the rotated
  space — and the whole-frame number stands, because the laplacian kernel and its interior
  are symmetric under every rotation and mirror an orientation can express (what the
  resample leaves is ~0.01% on a rotated frame, four orders below the band it is read in);
* THE COORDINATES ARE RESCALED (`face_crop_boxes`), and that is where the measurement this
  feature is built on went wrong: `faces.bbox` is in pixels of the full original, the
  preview is a few hundred pixels, and boxes used as written fell off the frame. 39 of 68
  crops were dropped that way and the surviving 29 reported 100% recall instead of 62%. A
  broken crop flatters the result rather than failing;
* it RANKS AND DOES NOT JUDGE. ~25% precision at every threshold measured — three of four
  flagged frames are not blurred — and it covers only the third of a collection that has a
  face at all. So `features.face_sharpness_max` orders the blur list and nothing in this
  stage reads it: no verdict, no threshold, no deletion. NULL means not measured, as
  everywhere in `frame_quality`.

F179: the third number off that same decode, and the first one in this file that a MODEL
used to answer. "Are the eyes open" was asked of the local VLM for 92 minutes a run and
came back 60% right about 9% of the frames it was meant to find; the question is gone
(F177) and the ~948 frames it was about — 15.6% of everything with a face in it — did not
go with it. Measured against the same 249 hand labels (F178, scripts/measure_eye_state.py):

    way in                        threshold  precision  recall
    the VLM (retired)                     —      60%       9%
    eyelid geometry                    0.18      62%      48%
    a classifier over the eye crop      0.9      46%      57%
    CLIP over the eye crop              0.8      58%      49%

Five times the recall at slightly better precision, out of ARITHMETIC over contour points
— `eye_openness` is the spread of an eye's ring across its own long axis, so no threshold
here depends on a network's opinion, only on where the contour landed. CLIP over the same
crop lost while costing a pass of the network the stage already runs.

Three properties, and they are the same three `face_sharpness` has, which is not an
accident — it is the shape a signal in this table has to have:

* NO NEW PASS and NO NEW WEIGHTS. `2d106det` is 4.8 MB inside the `buffalo_l` set the faces
  stage already downloads, it is fitted to the box the crop above is taken from, and it is
  built on the first face of a run and never on a run without one (`lazy_eye_landmarks`);
* THE COORDINATES ARE RESCALED, through the very same `face_crop_boxes`. The F178
  measurement lists this as the mistake F155 had already made once: `faces.bbox` is in
  pixels of the full original and the preview is a few hundred pixels wide, and a box used
  as written lands on empty sky. It does not raise — it flatters;
* it RANKS AND DOES NOT JUDGE. 62% precision means one frame in three of that slice has its
  eyes open, so `features.eye_openness_max` orders the list and opens it to a window a
  person walks past; nothing in this stage deletes or reclassifies by it.

The one rule that differs is which face answers: the LARGEST, where sharpness takes the
sharpest. A frame where somebody at the back blinked is not a portrait with closed eyes,
and a frame where one of several faces is in focus is a photograph that worked — each rule
picks the face its own question is about.

WHAT IS NOT FIXED HERE, written down because it will look like this feature's bug. F178
reports that on a frame with EXIF orientation 6 the faces stage decodes through
`cv2.imdecode` (which has already applied the rotation) and then rotates a second time, so
`faces.bbox` is written in a SIDEWAYS frame — while this stage works in the upright one.
If that holds, every box on a rotated photograph lands wrong here, and it lands wrong for
`face_sharpness` in exactly the same way and has since F155. It is a finding about the
faces stage, that stage is out of this feature's bounds, and the same script measures how
often a stored box matches a fresh detection (`легли на бокс`) — so whoever owns it starts
from a number rather than from this paragraph.

F164: the first three levers pulled with the phase table of F147 in hand, and the first
lesson is that a phase NAME is not a bill of costs. `junk_write` looked like 19,4 ms of
SQLite per frame and turned out to be the laplacian — the writes are one transaction and
cost 0,005 ms a row, measured (scripts/measure_junk_write.py, the table is at
_MEDIA_CLASS_UPSERT). Nothing was batched, because there was nothing left to batch; what
the phase is really made of is written down instead, so the next person starts from the
right number.

The other two levers are the same lever at both ends of the stage — a THREAD CEILING
chosen before anything was measured — and both keep their value, for two different
reasons. `vlm.workers` was measured and 4 turns out to be past the knee already: one
frame's preparation uses about seven cores since F105 gave the runtime the fast image
processor, so 6, 8 and 12 threads came back SLOWER than 4 on the live collection, and
the card's 51% is what NO overlap looks like at 0,12 s of CPU per 0,19 s of GPU rather
than a queue that is too short (config.default_vlm_workers holds that table).
`_DEFAULT_OCR_WORKERS_CAP` is unmeasured on purpose: what it protects is VRAM — one
easyocr Reader per thread — and only a free card can price it, so the tool ships
(scripts/measure_ocr_workers.py) and the number waits for the run that earns it.

Neither ceiling touches a verdict, and that is what the tests are about rather than the
seconds: the OCR pool has always written on the caller's thread in chunk order (F73),
the deep tier has always applied its labels in the candidate order (F101), and
tests/test_worker_ceilings.py pins both across every thread count the sweeps went up to.

F165: THE STAGE RUNS IN TWO HALVES, and `verdicts_only` is which one. The faces stage is
46% of a full run and it walks 24 195 frames, 4 300 of which this stage already knows are
screenshots, documents, memes or products — it just used to find out AFTER faces had paid
for them. So the verdicts move ahead of faces (`sorta classify`, and the `classify` step of
the pipeline) and everything that needs the face signal stays behind it (`sorta junk`):

    index -> geo -> landmarks -> classify -> faces -> events -> junk -> phash

The split is by dependency and nothing else. `verdicts_only=True` runs the fast pass
(heuristics, CLIP, the OCR gate), the deep tier and the stored vectors of F128 — none of
which reads `frame_quality` — and leaves out every half that does: the quality cascade
(`face_sharpness` is measured inside the boxes the faces stage writes, F155), the animal
cascade, the rescue, the keeper and the search index. Swapping the two stages instead of
splitting them would have switched `face_sharpness` off silently on every first run, which
is the failure this note exists to prevent someone re-inventing.

Both halves are the SAME function and the same incrementality: the second call finds
`media_class.tier` already current and reclassifies nothing, so `sorta junk` alone still
does the whole thing and the pipeline pays for the fast pass once. What the frames of the
first call cost the second is one softmax over the cache the shared classifier already
holds (F19) — no frame is decoded or encoded twice within a run.

THE ONE THING THAT DOES CHANGE, stated where it will be found. The fast tier reads
`has_faces` in four places (the F13 veto, the F15 document pass, the F38 OCR gate and the
#14 VLM gate), and before the faces stage has ever run there is nothing to read: on a FIRST
run with `--faces` a frame that CLIP calls a meme is no longer vetoed by the face in it.
That is exactly what a default run (faces are opt-in, F53) has always done, and one run
later the faces are in the database and the veto is back for every frame that still gets
there — but a frame this stage calls junk is a frame `faces` now skips, so the veto cannot
reach it afterwards. The brief accepted that trade for the 18% (F165, "Оговорки", 2 and 3);
it is written down here because no test can show it and no log line will mention it.
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
from typing import TYPE_CHECKING, Any, Callable, Generator, Sequence

import numpy as np
from PIL import Image

from . import imaging
from .config import Config, FeaturesConfig, vlm_allowed
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

# F102 moved the workers knob to `vlm.workers` and this resolver along with it (the old
# `naming.vlm_workers` address is still honoured there) — but this module is where it was
# born and where the measurement scripts import it from, so the name stays re-exported.
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

if TYPE_CHECKING:  # F132: the group shape only, for the annotations of _KeeperPass —
    from .dedup import GroupFrame  # the module itself is imported where it is used

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

# F130: what `frame_quality.pet_vlm` holds — the model's answer to the one question CLIP
# is structurally unable to answer. The errors left at 92% precision are drawn cats, plush
# toys, fur coats and a hotdog: CLIP compares a picture to a text as a whole and cannot
# tell a cat from a picture of a cat, while "is this animal alive or is it a rendering of
# one" is a question about the MEANING of the scene, which is what a VLM does.
#
# NULL is not one of these three. It means the question was not asked — below the
# candidate threshold, the check off, the model unavailable, or an answer that did not
# parse — and `pet_label` treats it as such rather than as a "no".
PET_VLM_REAL = "real"
PET_VLM_DEPICTION = "depiction"
PET_VLM_NONE = "none"

# Where each group sits in the prompt list of the single call (start, stop).
_JUNK_GROUP = (0, len(_CLIP_CLASSES))
_PET_GROUP = (len(_CLIP_CLASSES), len(_CLIP_CLASSES) + len(_PET_CLASSES))

# --- F140: the screenshots and receipts this stage took for photographs ----------------
#
# These prompts are NOT part of the call above and are deliberately not appended to it: the
# score they produce is read off the STORED vector (F128), which is what makes it free and
# what makes it answerable for a collection whose junk classification is already current.
# Nothing here is encoded a second time — the images were encoded once, months ago
# perhaps, and this is a matmul against a handful of text vectors.
#
# The junk classes themselves are reused rather than re-worded (`_CLIP_PROMPT`), because
# the score is a statement about the same question the stage already asks and a second
# wording of "a screenshot" would be a second definition of one. What is added is what the
# review of the live collection actually found among the misclassified frames — a
# photographed screen, a page of text, a receipt.
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

    One list rather than two because it is one text-encoder call, and `junk_rescue_score`
    splits it back at `len(_JUNK_RESCUE_POS_PROMPTS)`. The measurement script builds its
    prompts through here as well, so it cannot price a score the stage does not compute.
    """
    return list(_JUNK_RESCUE_POS_PROMPTS) + list(_JUNK_RESCUE_NEG_PROMPTS)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows of a text-feature matrix, L2-normalized — so a dot product is a cosine.

    Done here even though the encoder already normalizes, for the reason
    `pack_embedding` normalizes a vector the image tower had normalized: the guarantee
    costs a norm over a few rows and is worth more than the trust. A zero row has no
    direction to preserve and is left as it is rather than divided by zero.
    """
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=m.copy(), where=norms > 0)


def junk_rescue_score(vec: np.ndarray, text_features: np.ndarray) -> float | None:
    """max(junk prompts) - max(photograph prompts) over one stored vector.

    A margin and not a probability: softmax would compress exactly the region the
    threshold lives in (the useful band is 0.02 wide), and the frames this looks for are
    the ones the softmax of the main call already called a photograph. Both sides are unit
    vectors, so every similarity here is a cosine.

    None — the widths do not match: a vector stored by another model, or a truncated blob.
    A number computed across two spaces would look exactly like a real score, which is the
    one thing a selection signal must not do (`search.search` drops such rows for the same
    reason).
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


def pet_label(pet_vlm: str | None, pet_score: float | None, threshold: float, *,
              candidate_threshold: float | None = None,
              detected: bool | None = None) -> str | None:
    """The animal label of one frame — the whole cascade, in one place.

    The model OUTRANKS the score, which is the reason the cascade exists: a frame scored
    0.95 and answered `depiction` is a plush toy, and no threshold over a CLIP score ever
    separates one from a dog (F120 measured that — a drawn cat is a confident cat).

    `pet_vlm IS NULL` — not asked, not understood, or the model never came up — falls back
    to the rule that ran before this feature existed. That is what makes the check
    optional in the strong sense: with it off, and on every frame it could not answer, the
    label is byte-for-byte the one the stage wrote yesterday. Never a guess in either
    direction (brief item 3.2).

    THIS FUNCTION AND `sorter.animal_auto_sql` ARE ONE RULE IN TWO SPELLINGS. It is written
    twice because the two questions have different shapes — this one labels the single
    frame a stage has just scored, that one answers "which files" over a whole index — and
    the price of the second spelling is that every new source of the label has to reach
    both. F160 is the feature that found the detector reaching only one of them, so the
    case table in `tests/test_detector_reaches_the_screen.py` now runs both side by side
    and a source that lands in only one fails a test instead of a slice.

    The two keyword arguments are what a READER knows and the stage does not:

    `detected` is the F154 tier — True: this detector examined the frame and found an
    animal at or above `features.detector_threshold`; False: it examined the frame and
    found none; None: it never examined it (below the candidate depth, switched off, the
    model unavailable, an error on that frame, or boxes from another detector). None falls
    through to the score, never to "no animal" — the same order `detect.cascade_label`
    states, VLM answer above the detector included, because a box detector cannot be asked
    whether the cat it sees is alive.

    `candidate_threshold` is the F137 gate on a STORED answer: it counts only for a frame
    the current `features.pet_candidate_threshold` would still show the model, and a frame
    whose gate has since risen falls back to the tiers below it. None — the caller has a
    fresh answer to a question it has just asked, and there is nothing to re-gate.
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
#
# F164 was asked to sweep this too — 28% of the collection behind the gate is generous
# for "is there text on this frame", and `junk_ocr` is the most expensive phase of the
# stage. It did not, and the reason is worth as much as a table would be. A sweep of this
# number needs BOTH columns: how many frames a threshold cuts, and how many documents go
# into the city folders with them. The first is a CLIP pass, the second is an OCR pass
# over everything the LOWEST threshold of the grid gates — and with the card occupied
# (see _DEFAULT_OCR_WORKERS_CAP below) only the first half could have been run. A sweep
# with the coverage column and no benefit column is exactly the table that gets a
# threshold raised for the wrong reason: F38 raised this number because a document
# landing among the holiday photographs is the expensive error here.
#
# The tool is `scripts/measure_ocr_gate.py` (F90 built it, and it caches its per-frame
# aggregates so a grid can be re-tried without paying for the models again), and it
# already prints both columns plus `--probe-below`, an estimate of the documents the gate
# misses at EVERY threshold of the grid. What must not happen is this constant moving
# without that output: F90 states the rule in the module docstring — the threshold is a
# decision for a person in front of the table, not something a worker changes quietly.
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
#
# F164 went looking for that measurement, because `junk_ocr` is the most expensive phase
# of the stage — 614,6 s over 6 793 frames on the live run of 2026-08-03, i.e. 90,5 ms a
# frame and 15% of the whole run — and because the one number that does exist says the
# ceiling was chosen too low: F73 measured x3,7 going from 1 to 4 workers, which is still
# almost linear where it was capped.
#
# It did not raise the ceiling, and the reason is not the result but WHAT MAY BE MEASURED
# ON WHAT. The resource this number protects is VRAM — a Reader per thread, i.e. a copy
# of the detector on the card — so 6 and 8 workers are priced only by a run on the GPU
# profile, where those copies are real and where a pool that cannot build its N-th Reader
# quietly shrinks to the ones it built (see _OcrPool: such a row measures a smaller pool
# than its own label says). The card was occupied by another process throughout F164, a
# CPU run prices a different machine, and a default that ships to other people's hardware
# is not moved on a table about the wrong resource.
#
# So the tool ships instead and the number waits for the run that earns it:
#
#     python scripts/measure_ocr_workers.py --sample 500 --workers 1 4 6 8
#
# on a free card. It prints ms/frame, the speedup, the Readers actually built and the
# VRAM peak per row, and it says in one line whether the ceiling moves: at least x1,15
# over the current default with every Reader built. That is a one-line change here, with
# its table under it. Lowering stays what it always was — a value in `naming.ocr_workers`,
# which nothing here caps.
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
# group inside the CLIP call above (free, `features.pets`). Everything left — since F177
# that is the eyes and nothing else — is a VLM at ~0.78 s per frame, so it is asked ONLY
# about the frames the cheap tiers did not settle (`vlm.quality`).
#
# The population rule is the F109 result put to use: sending the model the least confident
# 30% of frames kept 98.2% of the findings. There it was worthless because the probe
# learned from the model's own labels (a closed circle); CLIP has no such circle — it
# labels without being trained on anything of ours.

# F155: one box of the `faces` table, as it is written there — (x1, y1, x2, y2) in pixels
# of the FULL original frame, after its EXIF orientation has been applied (the faces stage
# detects on a rotated full-resolution decode, see faces._decode_for_faces).
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
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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


# One short line of keywords. F96's lesson is the reason it is not three fields of JSON:
# asked for a composite format the model ignores the format and answers in prose, and a
# parser that then finds nothing writes False where it should write "not asked".
# F122: the "accidental" question is gone, and it was measured out rather than dropped
# on a hunch. On a labelled sample the model called 76% of what it was shown accidental;
# of those, 5% actually were — and the frames it called DELIBERATE held twice that rate
# (10%). A signal that is slightly anti-correlated with the thing it names is not
# something a threshold repairs, and every token it occupied was paid for on every frame.
# F177: the "is there a clear subject" question follows it, and by inspection rather than
# by a labelled sample. The first live run asked 6 111 frames and called 212 of them
# subjectless; looked at by eye those 212 are ordinary photographs — city shots and studio
# work alike — so the signal separates nothing. One question is left, and the prompt that
# carries it got shorter.
_QUALITY_PROMPT = (
    "Look at this photo and answer with keywords from this list only:\n"
    "eyes_open or eyes_closed — whether the people in the photo have their eyes open "
    "(use neither word if there are no people).\n"
    "Answer with those keywords separated by spaces, nothing else."
)
_QUALITY_MAX_NEW_TOKENS = 16

# Keyword -> value, per field, IN PRIORITY ORDER — the order is what keeps a keyword that
# CONTAINS another from being read as it (see `_PET_VLM_KEYWORDS`, where it still matters).
# F122/F177: `is_accidental` and `has_subject` are no longer asked, so they are no longer
# parsed. The COLUMNS stay and stay NULL — "not asked" is exactly what NULL means here,
# dropping them would need a table rebuild, and a retired question is cheaper to leave
# documented than to excise.
_QUALITY_KEYWORDS: tuple[tuple[str, tuple[tuple[str, bool], ...]], ...] = (
    ("eyes_open", (("eyes_closed", False), ("eyes_open", True))),
)
_NON_WORD_RE = re.compile(r"[^a-z]+")


@dataclass(frozen=True)
class QualityFlags:
    """The `frame_quality` answer columns of one frame. None means NOT ASKED / not parsed.

    Never a False by default: a consumer reading a defaulted False would conclude that a
    frame it has never shown to anything has its eyes closed.

    Only `eyes_open` is still asked. `has_subject` (F177) and `is_accidental` (F122) are
    retired questions whose columns stay — they are here so the write below keeps setting
    them to NULL, which is what "not asked" means in that table.
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


def _frame_question(describe: Callable[[Sequence[Image.Image], str, int], str],
                    max_edge: int, prompt: str, max_new_tokens: int) -> QualityAskFn:
    """One prompt over one frame, over an ALREADY LOADED runtime (naming.shared_vlm).

    Deliberately the plain, serial path and not the split halves the deep junk tier uses:
    these populations are a band or a candidate list, not the whole collection, and the
    pipeline machinery would cost more reading than it saves seconds. The decode goes
    through the shared preview cache, Unicode/HEIC-safe, exactly as everywhere else here;
    a frame that will not decode gets an empty answer, which parses to "not asked".

    Shared by the two questions this stage asks a frame (quality, F113; the pet check,
    F130) because they differ in the prompt and the token budget and in nothing else — a
    second copy of the decode would be a second place for the cache key to go wrong.
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
        return describe([img], prompt, max_new_tokens)

    return ask


def vlm_quality_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                      max_edge: int) -> QualityAskFn:
    """The quality question (the eyes) over a loaded runtime — see _frame_question."""
    return _frame_question(describe, max_edge, _QUALITY_PROMPT, _QUALITY_MAX_NEW_TOKENS)


def qwen_vlm_quality(model_name: str = _DEFAULT_VLM_MODEL,
                     max_edge: int = _DEFAULT_VLM_MAX_EDGE,
                     ) -> QualityAskFn:  # pragma: no cover — ML, smoke test
    """The real quality asker — the SAME weights as everything else (F95): one per run."""
    return vlm_quality_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_quality_factory(max_edge: int) -> Callable[[str], QualityAskFn]:
    """The default `quality_vlm_factory` of classify(), carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_quality(model_name, max_edge=max_edge)


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


# --- F132: the keeper of a near-duplicate group ---------------------------------------
#
# paths -> the model's raw answer about WHICH of them to keep (parsed by
# `parse_keeper_answer`). The one asker of this stage that takes SEVERAL frames, because
# that is the whole feature: one comparative question over the group instead of a score per
# frame. A comparative question is also the form a small model handles best — "which of
# these is better" needs no calibrated scale, only an ordering, and the frames of a burst
# are the same scene at the same scale, so nothing else about them can carry the answer.
KeeperAskFn = Callable[[Sequence[str]], str]

# Numbered 1..N in the order given, and the answer is a bare number. Deliberately NOT a
# description of the winning frame: the caller has to map the answer back onto a file, and
# a model that answers "the one where the child is smiling" maps onto nothing.
#
# The criteria are the ones a person actually uses and sharpness cannot see (F120 measured
# what it does see: focus, and only inside a group). Closed eyes, a turned head, a hand
# across the lens — that list is why this question exists at all next to the laplacian.
_KEEPER_PROMPT = (
    "These {count} photos are nearly the same shot, numbered 1 to {count} in the order "
    "shown.\n"
    "Choose the single best one to keep: eyes open rather than closed, faces turned to "
    "the camera, no hand or object blocking the view, the most natural expression, and "
    "the sharpest picture when nothing else separates them.\n"
    "Answer with one number from 1 to {count} and nothing else."
)
# One number. The budget is a couple of tokens above the F130 label's for the same reason
# it is small there: everything past the answer is the model explaining itself, and the
# parser below then has to wade through the explanation.
_KEEPER_MAX_NEW_TOKENS = 8

# The fallback when a digit in range is nowhere in the answer. ORDINALS ONLY, never the
# cardinals: "the first one" means frame 1, while "the best one" means nothing at all, and
# a parser that read cardinals would turn the second into a confident vote for frame 1.
_KEEPER_ORDINALS: tuple[str, ...] = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)
_DIGITS_RE = re.compile(r"\d+")


def keeper_prompt(count: int) -> str:
    """The question about `count` frames — the only place the number is filled in."""
    return _KEEPER_PROMPT.format(count=count)


def parse_keeper_answer(answer: str, count: int) -> int | None:
    """The model's answer -> a 1-based frame number, or None when nothing was readable.

    Lenient in the same two ways the other answers of this stage are read (the F96 lesson:
    asked for a strict format the model answers in prose anyway) — the number is looked for
    anywhere in the line, and an ordinal word is accepted when no number is. The FIRST
    number that falls inside the group wins: a model that names two frames is comparing
    them out loud, and its own choice comes first ("2 is better than 3").

    Out of range is not a choice. A "6" about five frames means the model lost count, and
    guessing which frame it meant is how a recommendation becomes a wrong recommendation —
    None sends the group back to the sharpness ranking, which is never wrong in that way.
    """
    text = (answer or "").lower()
    for match in _DIGITS_RE.finditer(text):
        number = int(match.group())
        if 1 <= number <= count:
            return number
    spaced = "_" + _NON_WORD_RE.sub("_", text) + "_"
    for index, word in enumerate(_KEEPER_ORDINALS[:count], start=1):
        if f"_{word}_" in spaced:
            return index
    return None


def vlm_keeper_asker(describe: Callable[[Sequence[Image.Image], str, int], str],
                     max_edge: int) -> KeeperAskFn:
    """One prompt over SEVERAL frames, over an already loaded runtime (naming.shared_vlm).

    The decode goes through the shared preview cache like every other question here. What
    is specific to this one: a frame that does not decode ABANDONS THE WHOLE QUESTION
    instead of being skipped. The answer is a position in the list, so dropping a frame
    silently would renumber the rest and the caller would map the answer onto the wrong
    file — a wrong keeper is worse than no keeper, because the group then falls back to a
    ranking that at least knows what it is ranking.
    """
    def ask(paths: Sequence[str]) -> str:
        images: list[Image.Image] = []
        for path in paths:
            try:
                st = os.stat(path)
            except OSError:
                return ""
            img = imaging.decode_rgb_preview(
                path, st.st_mtime, st.st_size, max_edge=max_edge)
            if img is None:
                return ""
            images.append(img)
        if len(images) < 2:
            return ""  # nothing to compare — the caller keeps its sharpness answer
        return describe(images, keeper_prompt(len(images)), _KEEPER_MAX_NEW_TOKENS)

    return ask


def qwen_vlm_keeper(model_name: str = _DEFAULT_VLM_MODEL,
                    max_edge: int = _DEFAULT_VLM_MAX_EDGE,
                    ) -> KeeperAskFn:  # pragma: no cover — ML, smoke test
    """The real keeper asker — the SAME weights as everything else (F95): one per run."""
    return vlm_keeper_asker(shared_vlm(model_name), max_edge=max_edge)


def qwen_vlm_keeper_factory(max_edge: int) -> Callable[[str], KeeperAskFn]:
    """The default `keeper_vlm_factory` of classify(), carrying `vlm.max_edge`."""
    return lambda model_name: qwen_vlm_keeper(model_name, max_edge=max_edge)


def keeper_prompt_fingerprint() -> str:
    """Eight hex characters over the question that decides a stored keeper.

    The same device as `quality_prompt_fingerprint`, over the only text that reaches
    `group_keeper.source`: edit the wording and every answer it produced stops matching,
    so the groups are asked again instead of keeping an answer to a question nobody asks
    any more. The `{count}` placeholder is hashed unexpanded — it is one prompt, asked of
    groups of different sizes.
    """
    return hashlib.sha1(_KEEPER_PROMPT.encode("utf-8")).hexdigest()[:8]


def keeper_source() -> str:
    """The `group_keeper.source` this run writes when the MODEL chose — `vlm#<print>`."""
    return f"{QUALITY_SOURCE_VLM}#{keeper_prompt_fingerprint()}"


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
        pets_verify=bool(getattr(f, "pets_verify", False)),
        pet_candidate_threshold=float(getattr(f, "pet_candidate_threshold", 0.3)),
        junk_rescue=bool(getattr(f, "junk_rescue", False)),
        junk_rescue_threshold=float(getattr(f, "junk_rescue_threshold", 0.02)),
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
_QUALITY_ANSWER_UPDATE = """UPDATE frame_quality
                            SET eyes_open = ?, has_subject = ?, is_accidental = ?,
                                updated_at = ?
                            WHERE file_id = ?"""
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


def quality_prompt_fingerprint(pets: bool, *, with_vlm: bool,
                               verify_pets: bool = False, rescue: bool = False,
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


def _quality_source(use_clip: bool, pets: bool, ask: QualityAskFn | None,
                    pet_ask: PetAskFn | None = None, rescue: bool = False,
                    rescue_ask: JunkAskFn | None = None) -> str:
    """The tier marker this run writes — and therefore what it considers up to date.

    F130: the pet check is a model too, so a run that only does that one still writes the
    `vlm` tier — the marker names WHICH TIER processed the row, and a row whose animal
    label came from the model did not come from the CLIP tier.

    F140: the rescue score is a CLIP signal (text vectors against a stored image vector),
    so switching it on moves the row to the `clip` tier at worst — and to `vlm` when the
    deep tier is there to answer for its candidates. Either way the fingerprint carries the
    prompts, which is what makes the score recomputable after an edit.
    """
    if ask is not None or pet_ask is not None or rescue_ask is not None:
        fingerprint = quality_prompt_fingerprint(
            pets, with_vlm=ask is not None, verify_pets=pet_ask is not None,
            rescue=rescue, rescue_vlm=rescue_ask is not None)
        return f"{QUALITY_SOURCE_VLM}#{fingerprint}"
    if use_clip and (pets or rescue):
        return (f"{QUALITY_SOURCE_CLIP}#"
                f"{quality_prompt_fingerprint(pets, with_vlm=False, rescue=rescue)}")
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
    happens with no callback at all — every method here already works that way — and
    that the deep passes, which re-`start` CLASSIFY_PHASE_VLM over three different
    candidate lists, share one bucket instead of inventing three.

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
    # (sharpness always, pets when the toggle is on); quality_candidates/answered are the
    # uncertain band and how much of it the model actually answered about — the pair that
    # says whether the band is worth what it costs.
    quality_rows: int = 0
    # The animals this run ended up marking — AFTER the F130 check, if it ran: the number
    # a user compares against the folder they get, not an intermediate the cascade
    # discarded on its way there.
    pets_found: int = 0
    quality_candidates: int = 0
    quality_answered: int = 0
    # F130: frames shown to the pet check (`pet_score >= pet_candidate_threshold`) and how
    # many of them came back with an answer that parsed. The pair prices the check the way
    # quality_candidates/answered price the band.
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
    # F132: near-duplicate groups the keeper pass looked at (those of at least
    # `dedup.keeper_min_group_size` frames), how many were actually shown to the model, and
    # how many came back with a number that parsed. The three price the question the way
    # pet_candidates/pet_verified price the animal check — and the gap between the first
    # two is what a `keeper_min_group_size` of 3, or a group of screenshots, costs.
    keeper_groups: int = 0
    keeper_asked: int = 0
    keeper_answered: int = 0
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
    what the cheap tiers say about a frame, and which frames that leaves for the model. It
    writes on the caller's thread, inside the caller's transaction — SQLite stays
    single-writer, as everywhere in this stage.
    """

    def __init__(self, conn: sqlite3.Connection, q: QualitySettings,
                 sharpness: SharpnessFn, ask: QualityAskFn | None,
                 scope_ids: set[int] | None, source: str, ids: set[int],
                 now: str, stats: JunkStats, faces_known: bool = False,
                 pet_ask: PetAskFn | None = None) -> None:
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
        # F130: the pet check, and its own candidate list — a different question over a
        # different population (a CLIP score above `pet_candidate_threshold`, not an
        # uncertainty band inside a scope), so the two do not share one.
        self._pet_ask = pet_ask
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
    def candidates(self) -> list[tuple[int, str, bool]]:
        """Frames of the uncertain band, in file order — the model's whole population."""
        return self._candidates

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

        Two things want it: the pet group, and the subject score that decides half of the
        uncertainty band. Without a row the band would read every frame as "CLIP says
        nothing", which is the reading that sends everything to the model.

        F130 adds no third reason: the pet check reads the score of the pet group, which
        is already covered by the first one (the check needs `features.pets`).
        """
        return self._q.pets or self._ask is not None

    def measure(self, file_id: int, path: str, probs_row: np.ndarray | None,
                verdict: str | None = None, has_face: bool = False) -> None:
        """The cheap tiers for one frame: measure, write the row, note the band.

        F120: a frame this run decided is NOT a personal photograph is dropped here
        instead of measured, and any row a previous run left for it is removed — the
        first live run wrote 24 196 rows over the whole collection, and the answers on
        screenshots, products and documents were the noise that made the signal unusable.

        That same guard is what keeps `vlm.exclude_classes` out of both candidate lists
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
        # (see the module docstring); the frame number is unchanged and still what the
        # uncertainty band below reads.
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
        if self._ask is None:
            return
        if self._scope is not None and file_id not in self._scope:
            return
        subject = (float(_group_probs(probs_row, _JUNK_GROUP)[0])
                   if probs_row is not None else 0.0)
        if uncertain_band(sharpness, subject, self._q):
            self._candidates.append((file_id, path, has_face))

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

        The phase is CLASSIFY_PHASE_VLM rather than a name of its own: it IS the model
        phase, the caption is already localized for it (F100), and the candidate list is
        known before the loop starts, so the bar reports a real (done, total) over it.
        The count reported is the list AFTER `_reclassified` has trimmed it, so the bar
        counts questions that will actually be asked.
        """
        if self._pet_ask is None or not self._pet_candidates:
            return
        gone = self._reclassified()
        candidates = [c for c in self._pet_candidates if c[0] not in gone]
        if not candidates:
            return
        self._stats.pet_candidates = len(candidates)
        report.start(CLASSIFY_PHASE_VLM, len(candidates))
        report.count(CLASSIFY_PHASE_VLM, len(candidates))
        with self._conn:
            for i, (file_id, path, before) in enumerate(candidates):
                try:
                    answer = self._pet_ask(path)
                except Exception as exc:  # noqa: BLE001 — the cheap tier must survive it
                    _log.warning(
                        "junk: VLM-проверка животных не ответила по file_id=%s (%s) — "
                        "оставляю метку по порогу CLIP", file_id, exc)
                    answer = ""
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

    def ask_model(self, report: _PhaseProgress) -> None:
        """The band, one frame per call — a failure on one frame costs only that frame."""
        if self._ask is None or not self._candidates:
            return
        self._stats.quality_candidates = len(self._candidates)
        report.start(CLASSIFY_PHASE_VLM, len(self._candidates))
        report.count(CLASSIFY_PHASE_VLM, len(self._candidates))
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
                # already knows where a face is, so the answer is believed only there.
                # Only when `faces` has actually run — otherwise "no face here" is
                # indistinguishable from "nobody looked".
                # F177: the eyes are now the WHOLE answer, so a frame with no face in it
                # leaves nothing to write rather than an answer about its subject.
                if self._faces_known and not has_face:
                    flags = QualityFlags()
                if flags.known:
                    self._conn.execute(_QUALITY_ANSWER_UPDATE, (
                        _as_int(flags.eyes_open), _as_int(flags.has_subject),
                        _as_int(flags.is_accidental), self._now, file_id))
                    self._stats.quality_answered += 1
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
                 now: str, tier: str, stats: JunkStats) -> None:
        self._conn = conn
        self._q = q
        self._model = model
        self._encoder = encoder
        self._ask = ask
        self._now = now
        self._tier = tier
        self._stats = stats

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
        """
        if self._ask is None:
            return
        report.start(CLASSIFY_PHASE_VLM, len(candidates))
        report.count(CLASSIFY_PHASE_VLM, len(candidates))
        with self._conn:
            for i, (file_id, path) in enumerate(candidates):
                try:
                    answer = self._ask(path)
                except Exception as exc:  # noqa: BLE001 — one frame, not the stage
                    _log.warning(
                        "junk: VLM не ответила по кандидату file_id=%s (%s) — "
                        "остаётся вердикт быстрого яруса", file_id, exc)
                    answer = ""
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
# that is not a reason to withhold a frame) — the F120 population, spelled the way
# `quality_scope_ids` spells it. The detector never sees anything outside this list, and
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


class _KeeperPass:
    """F132: the keeper of each near-duplicate group — computed and stored, never applied.

    The one question of this stage that is asked about a GROUP rather than a frame, and the
    reason it is worth asking at all: sharpness ranks focus, and focus is not what a person
    chooses by. Closed eyes, a head turned away, a hand in the corner of the frame — the
    model sees those, and asked comparatively ("which of these five") it answers far more
    reliably than it would score five frames one at a time.

    Three rules hold the feature to being advice:

    * `dedup_choice` is NEVER written here. That table is the user's decision and the only
      thing the sorter and the trash button act on; this one is a recommendation the
      interface may show. The boundary is the feature.
    * every group of the population gets a row, and a group the model did not answer about
      gets the sharpness ranking under `source='sharpness'` — the recommendation that
      already existed. A refusal must not be an empty screen.
    * only a group where EVERY frame is a personal photograph is shown to the model. That
      is the F120 rule, and here it is also the privacy one: near-duplicate scans of a
      passport are exactly the group a model must not be handed, and `vlm.exclude_classes`
      (which holds `document` by default) is a subset of what this excludes.
    """

    def __init__(self, conn: sqlite3.Connection, cfg: Config, ask: KeeperAskFn | None,
                 now: str, stats: JunkStats) -> None:
        self._conn = conn
        self._cfg = cfg
        self._ask = ask
        self._now = now
        self._stats = stats

    def _photo_groups(self, groups: list[list["GroupFrame"]]) -> set[int]:
        """Indexes of the groups every frame of which is a personal photograph.

        One query over an indexed column for the whole population, and the predicate is
        the quality half's: a frame with no verdict yet (a collection whose junk stage has
        never run) is not held against its group, a frame with a verdict that is not
        `photo` takes the group out of the model's population for good.
        """
        wanted: set[int] = set()
        ids = [f.file_id for group in groups for f in group]
        blocked: set[int] = set()
        for part in batched(ids, 500):
            blocked.update(int(r["file_id"]) for r in self._conn.execute(
                "SELECT file_id FROM media_class WHERE verdict != ? AND file_id IN"
                f" ({','.join('?' * len(part))})", (QUALITY_VERDICT, *part)))
        for index, group in enumerate(groups):
            if not any(f.file_id in blocked for f in group):
                wanted.add(index)
        return wanted

    def run(self, report: _PhaseProgress) -> None:
        """Ask about the groups this run has not answered for, and store every answer.

        Incrementality is the group key plus the prompt fingerprint: a group whose stored
        `source` is exactly what this run would write is left alone, which covers both
        "nothing changed" (the same members, the same question) and "the question changed"
        (a prompt edit re-asks). A group recorded as `sharpness` is revisited every run on
        purpose — that value means the model did not answer, which is a transient state
        (it raised, it said something unreadable, the group was not eligible), and one
        retry per run is the cheapest way not to make a failure permanent.
        """
        if self._ask is None:
            return
        from . import dedup  # local, as in quality_scope_ids — junk is not dedup's consumer

        d = self._cfg.dedup
        groups = dedup.keeper_groups(
            self._conn, self._cfg.index.phash_max_distance,
            min_size=int(getattr(d, "keeper_min_group_size", 2)))
        if not groups:
            return
        self._stats.keeper_groups = len(groups)
        source = keeper_source()
        keys = [dedup.group_key([f.file_id for f in group]) for group in groups]
        stored = dedup.read_group_keepers(self._conn, keys)
        answered = {key for key, row in stored.items() if row.source == source}
        todo = [i for i, key in enumerate(keys) if key not in answered]
        if not todo:
            return
        askable = self._photo_groups(groups)
        max_frames = int(getattr(d, "keeper_max_frames", 5))
        report.start(CLASSIFY_PHASE_VLM, len(todo))
        report.count(CLASSIFY_PHASE_VLM, len(todo))
        with self._conn:
            for step, index in enumerate(todo):
                group = groups[index]
                keeper, chosen_by = group[0].file_id, dedup.KEEPER_SOURCE_SHARPNESS
                # The best N BY SHARPNESS, because a group can be larger than any question
                # worth asking (the live collection holds one of 38 frames) — `group` is
                # already ranked, so this is a slice. The answer applies to the group as a
                # whole: what the model picked out of the best frames is the keeper, and
                # the frames not shown were the ones the cheap signal ranked below them.
                shown = group[:max_frames] if max_frames > 1 else []
                if len(shown) > 1 and index in askable:
                    self._stats.keeper_asked += 1
                    try:
                        answer = self._ask([f.path for f in shown])
                    except Exception as exc:  # noqa: BLE001 — one group, not the stage
                        _log.warning(
                            "junk: VLM не выбрала кадр в группе дублей (%s) — остаётся "
                            "рекомендация по резкости", exc)
                        answer = ""
                    choice = parse_keeper_answer(answer, len(shown))
                    if choice is not None:
                        keeper, chosen_by = shown[choice - 1].file_id, source
                        self._stats.keeper_answered += 1
                dedup.store_group_keeper(
                    self._conn, keys[index], keeper, chosen_by, self._now)
                report.step(step + 1)


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
    quality_vlm: QualityAskFn | None = None,
    quality_vlm_factory: Callable[[str], QualityAskFn] | None = None,
    pet_vlm: PetAskFn | None = None,
    pet_vlm_factory: Callable[[str], PetAskFn] | None = None,
    keeper_vlm: KeeperAskFn | None = None,
    keeper_vlm_factory: Callable[[str], KeeperAskFn] | None = None,
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

    sharpness_detector / quality_vlm / quality_vlm_factory (F113): the frame-quality
    cascade, written into `frame_quality` alongside the classification. The detector is the
    laplacian over the shared preview (no toggle — milliseconds, and both the "best frame"
    and the "blurred junk" consumers need it); pets are a prompt group inside the CLIP call
    this stage already makes, behind `features.pets`; the model answers about the uncertain
    band only, behind `vlm.quality`, with the same graceful fallback as the deep tier — a
    factory that raises leaves the cheap tiers running. All three are injectable for the
    same reason `classifier`/`text_detector` are: the suite must not load a model.

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

    keeper_vlm / keeper_vlm_factory (F132): the keeper question — which frame of a
    near-duplicate group to keep, asked ONCE PER GROUP with the frames of that group in one
    prompt, behind `dedup.keeper_vlm`. Same shape and same graceful fallback as the two
    askers above; the answer is stored in `group_keeper` as a RECOMMENDATION, and nothing
    on any path of this stage writes `dedup_choice`. A model that will not build, will not
    answer or answers something unreadable leaves the group with the sharpness ranking the
    interface already showed.

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

    F145: every one of the four askers above is subordinate to `vlm.enabled`. Their own
    keys say WHAT to ask, not whether a model is raised — a run without deep analysis
    loads no weights whatever config.yaml holds, and each of them then behaves exactly as
    it does with its own key off (the graceful-fallback path they already had).

    verdicts_only (F165): run the halves that do NOT need the face signal and stop —
    the fast pass, the deep tier and the stored vectors, i.e. everything that ends in
    `media_class` or `clip_embeddings`. This is the `classify` stage, which the pipeline
    runs BEFORE faces so that the faces stage can skip what is already known not to be a
    photograph (see `faces._files_to_detect`); the quality cascade, the animal cascade, the
    rescue, the keeper and the search index are left to the `junk` stage after it. Nothing
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
    vlm_fn: VlmClassifyFn | None = None
    if use_clip and vlm_on:
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
    # F165: and every question below this one belongs to the half that runs AFTER faces.
    # The check sits next to each gate rather than around the whole block because that is
    # where a reader looks for the answer to "does the `classify` stage load this model?" —
    # and the answer has to be no for all four, or the split would move the weights of the
    # deep passes ahead of the stage that has nothing to ask them.
    q = quality_settings(cfg)
    quality_ask: QualityAskFn | None = None
    if (use_clip and vlm_on and not verdicts_only and q.vlm_quality
            and quality_scope_ready(conn, q.vlm_scope)):
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

    # F130: the animal check, resolved the same way and with the same fallback. Its gate
    # carries one extra condition — `features.pets`, because it verifies what the CLIP pet
    # group found and has nothing to verify without it. The model is the shared one (F95),
    # so switching this on next to `vlm.quality` costs a second question per frame, not a
    # second set of weights.
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

    # F132: the keeper question, resolved the same way and with the same fallback again.
    # Its toggle lives in `dedup:` and not in `vlm:` because it is a property of the
    # duplicates feature — which frame of a burst to keep — rather than of the runtime;
    # the runtime it uses is the shared one (F95), so switching it on next to the other
    # two questions costs calls, not a second set of weights.
    keeper_ask: KeeperAskFn | None = None
    if (use_clip and vlm_on and not verdicts_only
            and bool(getattr(cfg.dedup, "keeper_vlm", False))):
        if keeper_vlm is not None:
            keeper_ask = keeper_vlm
        else:
            k_factory = keeper_vlm_factory or qwen_vlm_keeper_factory(cfg.vlm.max_edge)
            try:
                keeper_ask = k_factory(cfg.vlm.model)
            except Exception as exc:  # noqa: BLE001 — advice is optional, must not crash
                _log.warning(
                    "junk: VLM-выбор кадра в группе недоступен (%s) — рекомендация "
                    "остаётся по резкости", exc)
                keeper_ask = None

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
    quality_source = _quality_source(use_clip, q.pets, quality_ask, pet_ask,
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
    # F132: the keeper question has a population of its own — groups, not frames — and its
    # own marker, so it can have work when every other half of the stage is up to date.
    # That is the ordinary case for it: the toggle is switched on for a collection whose
    # junk classification and quality rows are already current, and an early return here
    # would leave the feature silently doing nothing until something else went stale.
    # F147: built here rather than next to the first `report.start` below, so the keeper
    # — the one pass that can have work when every other half of the stage is up to date
    # — is timed on that path too instead of running under a throwaway reporter.
    # F165: the phases are filed under the stage that is actually running — the caller's
    # `stage_timer` opened `classify` or `junk`, and the two names have to agree.
    report = _PhaseProgress(progress, VERDICTS_STAGE if verdicts_only else CLASSIFY_STAGE)
    keeper = _KeeperPass(conn, cfg, keeper_ask, now, stats)
    # F141: the search index, and — like the keeper above it — a pass that can have work
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
    # F154: the animal detector — a third pass that can have work when every other half of
    # the stage is up to date, and for the same reason the keeper and the search index can:
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
        keeper.run(report)
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
        quality_ask,
        quality_scope_ids(cfg, conn, q.vlm_scope) if quality_ask is not None else None,
        quality_source, quality_ids, now, stats, faces_known, pet_ask)
    # F140: the encoder is a closure and not an object, so that a run whose rescue has
    # nothing to score never builds one — see _JunkRescuePass.
    rescue = _JunkRescuePass(
        conn, q, embed_model,
        (lambda: junk_text_encoder) if junk_text_encoder is not None
        else (lambda: (junk_text_encoder_factory or clip_text_encoder)(s)),
        rescue_ask, now, active_tier, stats)
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
    # frame it has just called a `document` must not be asked about again. Before the two
    # questions below it for the reason the pet check comes before the band: the verdict is
    # what the rest of the pipeline depends on, and a run interrupted afterwards has
    # finished the part that matters most.
    rescue.run(quality.measured, report)
    # F130: the animal check, over the candidates the CLIP pet group turned up. After the
    # deep tier for the same reason the quality band is (one GPU, and the verdict is what
    # the rest of the pipeline depends on), and before the band because it is the shorter
    # of the two lists — a run interrupted between them has finished the cheaper question.
    quality.ask_pets(report)
    # F113: the last and most expensive step of the cascade — the three questions neither
    # the laplacian nor CLIP answers, asked only about the band those two left uncertain.
    # It runs after the deep tier for a plain reason: both want the same GPU, and the
    # verdict is what the rest of the pipeline depends on.
    quality.ask_model(report)
    # F154: the detector, last of the three tiers that can move an animal label. After the
    # VLM check because it reads what that check wrote — a frame the model has already
    # judged keeps its answer, since "is this animal alive" is not a question a box
    # detector can be asked (`detect.cascade_label`) — and after the deep tier because that
    # is what settles which frames are personal photographs at all. Its candidates come out
    # of `clip_embeddings`, which the loop above has just filled.
    # F165: and, like the three passes above it, not in the verdicts-only half — it reads
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
    # F132: last of all, because it reads what everything above has just written — the
    # verdicts (a group with a screenshot in it is not shown to the model) and the
    # sharpness of each frame (the ranking that decides which frames go into the question
    # and what the group falls back to). It writes `group_keeper` and nothing else; no
    # path of this stage touches `dedup_choice`.
    keeper.run(report)
    # F141: and after even that, the second CLIP pass — every verdict of this run is
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
