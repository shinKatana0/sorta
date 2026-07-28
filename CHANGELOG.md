# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **City folders are named in the layout language** (F99): with `language: ru` the
  layout produced `Россия\St Petersburg\2023\` — a localized country over an English
  city. The geo layer writes the English anchor plus `places.city_geonameid` on purpose
  and calls translating it sort's job (G3), and sort was translating the country only.
  The defect was invisible while the online provider was in use (Nominatim stores names
  already in the config language); the first full offline run (2026-07-28) hit **16 881
  rows of 26 135**. The city name now comes from `GeoResolver.name(geonameid, lang)`,
  with one resolver per plan (`places.tsv` is 170 472 rows — loading it per file would
  turn seconds into hours), and the resolver's last fallback — the geonameid itself — is
  refused in favour of the English anchor: a folder called `498817` explains nothing,
  `Kapong` does. Rows without a geonameid (a hand-assigned place, `path_inferred`, a
  landmark, an online answer) keep the text the DB holds — there is nothing there to
  translate by. Measured on the live collection: 62 of 83 cities (15 284 files) get a
  Russian name, the remaining 21 (3 166 files) stay English because no Russian name for
  them exists. Switching the folder language re-plans the whole collection **without a
  single geo query or a write into `places`**, and the web app's cards now show the city
  the folder is named after instead of the anchor next to a localized folder.
- **A repeated `sort --apply` no longer duplicates what it already copied** (F97): the
  first real apply on the live collection (22 364 files, ~220 GB, copy mode) died with
  the machine; the user restarted it into the same destination and got **10 021 files
  copied a second time under `_1` names — 140.9 GB** of byte-identical twins. Cause:
  `_resolve_dst` could tell "the file is already AT its target" (an in-place layout)
  from a name conflict, but not "the target already holds exactly this file". It now
  compares the size and then the blake3 of the existing `dst` against the source hash
  **from the index** (no re-hashing of sources on a resume) and skips the file,
  counting it in a separate `skipped_already_copied` — separate on purpose, because
  "source and target are one path" and "the copy is already made" are different events.
  A different file with the same name still gets `_1`, and a same-size-different-hash
  file (a copy interrupted mid-write) counts as different — the existing file is never
  overwritten. Albums go through the same `_resolve_dst` and inherit the behaviour.

### Changed
- **The deep classification tier overlaps its CPU work with the GPU** (F101): the tier
  earns its keep — on the live run of 2026-07-28 it changed **2 592 verdicts of 24 196
  (10.7%)**, 2 202 of them into `product`, a class the fast tier does not produce at all
  — but at 1.38 frames/s it took ~95 minutes, which makes it a weekend job rather than
  something you leave on. Profiling said why, and it was not what a "heavy model" would
  suggest: the pass is **sequential**, ~0.6 s of CPU (decode + the processor's image
  preprocessing) then ~0.19 s of GPU per frame, strictly alternating — **0.84 cores busy
  out of 24** and the card at ~26%. Batching was ruled out by that same measurement: a
  starved GPU does not want bigger portions. Two levers instead. The processor is now
  built with `use_fast=True` (transformers had been warning about the slow one all
  along, and the slow one is pure Python over PIL — a good part of that 0.6 s). And the
  runtime is split into the two halves it always consisted of, so `naming.vlm_workers`
  threads prepare frames while this thread runs the model — the shape F87 gave faces
  (×1.47), with more headroom here because the skew is worse. **No verdict may move
  because of it**, and that is enforced rather than hoped for: labels come back in the
  candidate order (a FIFO of futures, not "whatever finishes first"), the model still
  sees one frame per call with the same prompt and the same greedy decode, writes still
  happen on one thread, and a frame whose preparation fails still keeps its fast verdict
  and still steps the progress bar. VRAM does not grow — prepared tensors stay on the
  CPU and only one frame's inputs are ever on the card — and the frames in flight are
  capped at two per worker. Each preparation thread gets **its own processor** (a
  processor is mutable state, the same reason every OCR worker has its own easyocr
  Reader since F73). `scripts/measure_vlm_speed.py` runs the old and the new path over
  the same frames on one load of the weights and prints median/p90 ms per frame, GPU
  load, peak VRAM, cores busy — and a label-by-label comparison that exits non-zero if a
  single verdict differs.

### Added
- **The "not personal photos" buckets can be reviewed — and fixed in bulk** (F103): the
  first live run of the deep VLM tier (2026-07-28) reclassified **2 592 frames of
  24 196**, including a `product` class the fast tier does not produce at all (2 202
  frames). Reviewed by eye, only a handful of those verdicts are wrong — but "a handful
  of 2 202" is dozens of frames, and there was nowhere to look at them: the buckets were
  visible only indirectly, as folders of the layout plan. The tool was confidently
  carrying every tenth frame of a collection into a separate folder without ever showing
  them first. A new tab lists every frame whose `media_class.verdict` is not `photo`,
  with per-bucket counters and a filter (product / document / screenshot / meme), and
  returns a whole selection to the normal city layout in one action. The correction goes
  through the existing `manual_overrides` mechanism (F77) as the new action `photo`:
  **`media_class` is never rewritten**, because the model's verdict is a measurement and
  a correction by eye is a separate layer on top of it — otherwise the next junk run
  would silently wipe the user's decisions and leave no trace of why. Only the sorter's
  ROUTE changes: a corrected frame falls through to the ordinary city/date branch.
  The `document` bucket answers **without a preview link** — those are passports,
  medical forms and bank papers, and such a frame is never decoded for display; the card
  carries a name and a date, which is enough to decide. Returning a document to the
  photos is still allowed, only its preview is not built. Nothing here reclassifies
  anything and no threshold moved.
- **The classification stage says which phase it is in — and how far the deep tier has
  got** (F100): with `--deep` the frame counter ran through the fast pass and then stood
  at **100% for the whole VLM tier**; on the live run of 2026-07-28 (24 196 frames) that
  was forty minutes in which the only way to tell a working model from a hung process
  was to look at the GPU load. `junk.classify` now reports its phases through the same
  optional `progress.phase(name)` channel clustering has used since F84 —
  `junk_clip` / `junk_ocr` / `junk_vlm` / `junk_write`, localized in the web app in
  ru/en/ja. Unlike HDBSCAN, the deep phase is **measurable**: the gate's candidate list
  exists before the loop starts, so it reports a real `(done, total)` over the
  candidates — «глубокий анализ (VLM): 1 200 из 1 843» instead of a full bar and a
  guess. The denominator changes from frames to candidates exactly when the caption
  does, which is what makes the switch readable; a frame the model errored on still
  moves the counter, so the bar always reaches its total. Nothing else moved: no
  verdict, threshold or gate is touched, a run without the deep tier looks and behaves
  as before, and a callback with no `phase` channel (the CLI path, quiet mode, tests) is
  not an error — it just gets the counter, as it always did.
- **Cancelling a layout from the UI** (F97): copying 220 GB takes an hour and a half and
  there was no way to stop it short of killing the process. `plan_and_sort` now takes
  `should_cancel`, polled at the start of each file, before the `moves` row is written;
  on cancel it **breaks rather than raises**, so the batch still gets its `finished_at`
  — an exception would fly past the code that closes it, and undo is exactly the tool
  the user reaches for next. `POST /api/sort/cancel` and a Cancel button next to the
  progress bar; the report says "cancelled, N of M", not a bare "done".
- **Rolling the last batch back from the UI** (F97): the "Moves" tab was a read-only
  manifest that told the user to go and type `sorta undo`, in precisely the situation
  the journal had been written for. It now has a Roll back button (and a second entry
  point in the result panel of a cancelled layout), behind a confirmation dialog that
  names the operation and the count from the manifest — *"N copies in `<dest>` will be
  deleted, the originals stay untouched"* / *"N files will go back to their original
  folders"*. `POST /api/undo` + `GET /api/undo/status` + `POST /api/undo/cancel`, the
  same shape as `/api/sort`, cross-locked with a layout and a pipeline run both ways.
  The rollback is itself cancellable (it re-hashes every copy) and idempotent: pressing
  the button again finishes what a cancel left. `undo` now also handles the **tail of an
  interrupted transfer** — rows still in `status='planned'` whose file exists: a hash
  match means it is our own complete file (deleted for copy, moved back for move), a
  mismatch means a broken copy, which is **not** deleted but reported by path for the
  user to check. A batch left with `finished_at = NULL` is closed by the rollback
  instead of looking like it is running forever.
- **Windows long paths** (F97): every filesystem call in `sorter.py` now goes through a
  `\\?\`-prefix helper — the copy, the destination `mkdir`, the existence check that
  resolves name conflicts (without it `exists()` lies "no" past 260 characters, the
  unsuffixed name is chosen and the write fails anyway), and the `unlink` in `undo` and
  in near-duplicate deletion. Paths in the DB stay plain absolute strings; the prefix
  lives only at the call boundary, and on non-Windows the helper is a no-op.
- **Event names that say what happened** (F95): the biggest events of a collection were
  called `2025-04-24..05-06 Тайланд` (1 359 files) — a date range and a country, which
  is exactly what the folder path above them already showed. A year later a trip is
  looked for by «Пхукет с детьми» or «свадьба в Праге», so the name has to carry the
  CONTENT. New naming provider `vlm`: it takes the 3–5 sample frames of an event, asks
  the local Qwen2.5-VL what is going on in them, and appends the answer to the template
  base — `2025-04-24..05-06 Тайланд пляжный отдых с детьми`. Dates and places are never
  asked of the model: they are known exactly from EXIF and geo, and a model asked for
  them invents them. One call per EVENT rather than per file (473 events — minutes, not
  hours), and the weights are the ones the deep junk tier already loads: `naming.py`
  hands out a single runtime per model name, because a second copy of a 20.5 GB peak
  does not fit on the card. Opt-in (`naming.provider: vlm`, needs the `[vlm]` extra) —
  the default stays `template`, and everything that can go wrong (no transformers, the
  model does not load, no VRAM, a garbage answer) falls back to the template name
  instead of breaking the naming stage. Names the user typed by hand are untouchable, as
  before. Documents and screenshots are now excluded from the sample frames of every
  provider — the filter sits before the provider is chosen, so it also covers the cloud
  one (`claude`), and it exists because the description becomes the name of a physical
  folder that then travels into backups and reports.
- **Place inherited from the trip** (F85a): a file with no GPS used to get a place only
  from the six hours around it, so a whole session where nobody had coordinates stayed
  in `_Unsorted/no_place` even when the trip around it was placed perfectly (1 758 files
  of the validation collection). The geo stage now groups its own sessions into trips by
  the same rule `events` uses (`events.trip_merge_gap_hours` / `events.trip_merge_max_km`)
  and lends the trip's place to what the session level could not reach. Two conditions
  keep it honest, because a file filed under a foreign city is worse than one in an empty
  folder: the trip's own GPS frames must agree about the city (the dominant one holds
  more than half of them), and the file must lie between two of those frames in time —
  the camera was in that city before it and after it. The new place carries
  `confidence = 'trip_inferred'`, told apart from `session_inferred` in
  `sorta geo`/`sorta stats`, the CSV plan and the HTML report. Precision is measurable
  without labelling: `scripts/measure_place_inference.py` hides the coordinates of files
  that have them and scores each level against the truth — 99.0% for the trip level on
  the validation collection (385 cases), where it places 839 files that had none.
- **Country from a folder name** (F85c, part 1): a file that no geometric signal could
  place takes the COUNTRY named by a folder on its path («Тайланд 2023», «Greece»). Only
  the country, and deliberately so — measured on the files that do have GPS, so the guess
  can be scored against the truth, a country read from a folder name is right 99.5% of
  2 105 hints while a CITY read the same way is right 4.3% of 1 152 (the bundled base
  holds 150 000 settlements, so any ordinary word finds a hamlet somewhere). The hint is
  the last rule to run: it never overrides GPS, session or trip inheritance, it only
  fills what nothing else reached, and it costs no dependency, no network and no model —
  the country names are already in the package. On the validation collection it takes
  520 files out of `_Unsorted/no_place` and into `<Country>/<year>/`. New confidence
  level `path_inferred`, reported by `sorta geo`.
- **Assigning a place to a whole group by hand** (F85c, part 2): after every automatic
  signal has had its turn, about 6 300 files of the validation collection still have
  none — old scans, forwarded pictures, frames shot with GPS off. No model will place
  them, because the information is not in them; it is in the person who took them. The
  web app now takes it in bulk: pick a group that is already a thing on screen (a whole
  event on the "Events" tab, a whole source folder from a row of the "Cities" plan),
  pick a city or a country from the bundled base, one action for all 500 files. The
  assignment is stored apart from `places` (which `geo` recomputes from scratch every
  run) and applied when the plan is built, so it survives a recompute; the plan, the CSV
  and the report show it as `manual`, so a place the user chose is never mistaken for
  one the program inferred. Files the camera placed itself are skipped and counted back
  rather than overwritten silently — overwriting them is a second, explicit answer.
  Undoing an assignment restores exactly the place the program had worked out.
- **Persistent geo cache** (F93): with `geo.provider: online` the provider's answers are
  stored in the DB (`geo_cache`) instead of the process, keyed by the city+district pair
  of the bundled base — adding photos no longer re-asks Nominatim about the whole
  collection, and all three languages are fetched at once, so switching folder language
  costs no network at all. The cache survives "Start over"; clear it with
  `sorta cache --clear-geo`, `sorta reset --clear-geo` or the checkbox in the reset
  dialog of `sorta ui`. Expiry: `geo.cache_max_age_days` (default 180, 0 — never).
- **Cache management in the web app** (F94): the "Process" tab shows what the preview
  and geo caches occupy (the same numbers `sorta cache` prints) and clears either of
  them with a confirmation that states the price — a cleared preview cache rebuilds
  itself, but the next run decodes at 336 ms per frame instead of 73. Both clears are
  refused while a run or a layout is in flight, and nothing is ever cleared
  automatically: there is no size ceiling and no eviction.

## [0.2.0] - 2026-07-25

### Added
- **Parallel faces inference** (F12.1): independent insightface sessions run in a
  thread pool — ~3× faster face detection on a GPU (measured on real data). Tune with
  `faces.infer_workers` (auto: 4 on CUDA, 1 on CPU); SQLite stays single-writer.
- **`sorta doctor`** and a startup GPU-health check (F63): warns when PyTorch is a
  CPU-only build while onnxruntime is on CUDA — so CLIP/OCR silently running on the
  CPU (GPU idle for hours) no longer goes unnoticed.
- **Configurable CLIP decode pool** (F64): `naming.clip.decode_workers` (auto
  `min(cpu, 16)`) speeds the decode-bound junk/landmarks stages on multi-core machines.
- README web-app hero GIF + Cities screenshot (captured on a synthetic demo), plus
  CI / release / license / Python badges, a CHANGELOG, and GitHub issue templates.

### Fixed
- Face clustering no longer crashes on very small collections — fewer detected faces
  than the HDBSCAN sample size previously raised and took down the whole faces stage
  (F3.2).

### Performance
- pHash / index decode threads are tunable via `index.workers` (the decode-bound pHash
  stage scales with cores).

## [0.1.0] - 2026-07-24

First public release — the MVP is feature-complete and CI is green.

### Added
- **Indexing:** scan a photo/video collection into a SQLite index (EXIF via exiftool,
  blake3 hashes, incremental re-runs by path/mtime/size then hash).
- **City/country sorting** from a single index — switching modes needs no re-scan.
- **Offline geolocation** (bundled GeoNames) with GPS + session inference; optional
  online Nominatim/OSM.
- **Faces & people:** local detection + clustering (insightface); label and merge
  clusters; per-person albums. Opt-in (`--faces`).
- **Events:** time-gap + city clustering with localized names; manual events. Opt-in
  (`--events`).
- **Duplicates:** exact (blake3) and near-duplicate (perceptual hash) with a
  batch-review UI.
- **Junk & documents:** screenshots/memes routed out; documents collected into a
  `_Documents/` review folder.
- **Albums:** collect a person/event slice into a named folder via hardlinks, copy, or
  move.
- **Local web app** (`sorta ui`): process a folder, review the plan, resolve
  duplicates, name people, materialize sorts/albums.
- **Safety:** dry-run by default; `moves.jsonl` journal written before each move;
  `sorta undo`; blake3 verification; never overwrites (suffix `_1`, `_2`); originals
  never modified.
- **Trilingual** UI and folder names (en/ru/ja); default UI language English.

[Unreleased]: https://github.com/shinKatana0/sorta/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shinKatana0/sorta/releases/tag/v0.2.0
[0.1.0]: https://github.com/shinKatana0/sorta/releases/tag/v0.1.0
