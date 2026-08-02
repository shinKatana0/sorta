# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **A search line in the "Slices" block** (F134): the field F133 drew and left disabled is
  wired to the F129 engine — type "cake" and the collection comes back ranked, with the
  score on every card and a "Gather into folder" button that is the existing album route
  with `kind='query'` and the words as the selector. The engine is untouched; what this
  feature is actually about is the state in which it cannot run. `clip_embeddings` is
  filled by the junk stage of an ordinary run, so a fresh collection — and any collection
  last processed before F128 — has nothing to rank, and an empty result list would read as
  "you have no photographs like that": a conclusion about somebody's own archive drawn
  from a table nobody filled. So the new `GET /api/search` carries the state of the index
  in **every** answer (`state`, `available`, `indexed`, `total`, `index_model`), the line
  stays **disabled** while there is nothing to search, and the reason stands next to it
  with the way to fix it — a button to "Overview", where the run is started. The two
  unavailable states are deliberately two sentences: an index that was never computed
  ("this is an ordinary run, no separate model") and an index computed by **another
  model**, which is named, because its vectors are not comparable with the query and
  mixing them would produce a plausible ranking nothing marks as wrong. Partial coverage
  does not block anything and says its denominator out loud ("searching 19 753 of 19 757
  photographs") — an incremental run is the normal way to live with a growing archive, and
  a person has to be able to tell "it is not in the collection" from "it is not in the
  index yet". No threshold is introduced anywhere: the list ranks, the reader stops where
  the resemblance runs out, and the interface promises no accuracy, since nobody has
  measured it on a real collection yet. The privacy rule of F133 is carried over
  unchanged — a frame whose class is in `vlm.exclude_classes` is ranked but hands out no
  thumbnail, so a search cannot become the way around what the slices already close. An
  empty query never reaches the model, in the browser or on the server.
- **The best frame of a duplicate group, asked comparatively** (F132): `dedup.keeper_vlm`,
  off by default, next to `dedup.keeper_max_frames` (5) and `dedup.keeper_min_group_size`
  (2). Sharpness already ranks a near-duplicate group honestly — inside a group the frames
  are one picture at one scale, which is the only place the laplacian answers the question
  it was measured for — but it cannot see what a person chooses by: closed eyes, a head
  turned away, a hand across the lens, the expression. So the model is asked, and asked
  **comparatively**: one call per group with the frames in a single prompt ("which of
  these is best"), not a score per frame. The comparative form is the whole bet of the
  feature — it needs no calibrated scale, only an ordering. The answer is a **number**,
  read leniently (an ordinal word counts, a number outside the group does not), and stored
  in the new `group_keeper` table (schema v18) with its `source`, so the interface can say
  whether sharpness or the model suggested this frame. Groups have no id — they are
  recomputed by union-find on every call — so a group is addressed by a sha1 over its
  sorted file ids, which invalidates itself: a burst that gained or lost a frame hashes
  differently and is asked again, an unchanged one is not. Nothing is ever deleted, moved
  or marked: `dedup_choice` remains the user's own decision and no path of the junk stage
  writes it. Every failure — an unreadable answer, a raise on one group, a model that will
  not build, a group holding something that is not a personal photograph (scans of a
  passport are exactly the burst a model must not be handed) — falls back to the sharpness
  ranking under `source='sharpness'`, never to an empty recommendation. Cost is measured
  before it is quoted: `scripts/measure_group_keeper.py` prints seconds per call on a
  sample of real groups and only then projects the full population, because the 0.78 s in
  hand was measured on a prompt with **one** image and this one carries up to five. On the
  reference collection the population is 791 groups, 676 of them pairs — which is what
  `keeper_min_group_size: 3` is for: 115 groups where the choice is genuinely unclear.
- **Taking a false animal mark off a frame — and putting a missing one back** (F124):
  the "Animals" tab gained the one action it was missing. The 0.70 threshold was
  measured at 92% precision, so of the 805 marked frames of the live collection **about
  64 are not animals**; the person who sees them in the list is the one who takes the
  mark off, and until now there was nowhere to put that decision. It goes into a table
  of its own, `manual_pet` (schema v17), and **not** into `frame_quality`: that table
  has exactly one writer (`junk`) and every run recomputes it from scratch — after F120
  a prompt fingerprint invalidates the rows outright — so a correction written there
  would live until the next run and no longer. The precedent is F85c's `manual_places`
  against `places`, word for word. It is not an action of `manual_overrides` either:
  that column is about the **layout**, and folding "this is not a cat" into it is how a
  file ends up dropped from the layout because of what is in the frame. The mark is
  applied **when read, never when written** — `junk` is untouched, and the consumers
  (the album slice, the tab, the "Overview" counter) read one shared expression,
  `sorter.ANIMAL_IDS_SQL` = `COALESCE(manual_pet.is_animal, frame_quality.pet IS NOT
  NULL)`. That is what makes an edit survive **any** recompute, a change of model, of
  prompts or of the threshold included, and what keeps the three numbers from drifting
  apart. The mark is two-way on purpose: a person takes a false one off and puts a
  missing one on, which is also the only way a frame the threshold missed reaches the
  album. A corrected frame **stays on the page**, dimmed, saying which way it was
  decided and offering the way back to the automatic verdict — a card that vanishes
  moves the counter for no visible reason and takes its own undo with it. There is
  deliberately **no** "unmark everything below 0.75": the feature exists because
  somebody looked at the frame, and a threshold is already there for the other case.
  A `reset` wipes the table with every other manual decision.
- **A cascade for animals: CLIP selects widely, the VLM checks** (F130):
  `features.pets_verify`, off by default, with a second and much lower selection
  threshold `features.pet_candidate_threshold` (0.30) next to the existing
  `features.pet_threshold` (0.70). F122 measured where the CLIP-only answer stands — 92%
  precision, 54% recall — and both halves of that have the same cause. The errors left at
  92% are drawn cats, plush toys, fur coats and a hotdog: CLIP compares a picture to a
  text **as a whole** and cannot tell a cat from a picture of a cat, which is not
  something a threshold repairs. And the recall cannot be raised by lowering the cut,
  because 0.70 was high precisely because nothing was checking the answer. Both change
  once something does. The model is asked **one** question with three outcomes — a live
  animal, a picture of one, no animal — and the species is deliberately not among them
  (F122 retired those labels by measurement; bringing them back through another model
  without a new one would be an unmeasured label that looks like data). The answer
  **outranks** the score: 0.95 and "plush toy" is not an animal, 0.35 and "real" is. Every
  way the expensive tier can fail — an unparsable answer, a raise on one frame, a model
  that will not build — falls back to `pet_score >= pet_threshold` and **never** to "no".
  The answer is stored in the new `frame_quality.pet_vlm` (schema v17) because "rejected"
  and "never asked" are different facts: without the column every later run would pay
  0.78 s again for each of the ~500 frames the model already turned down, and the
  interface would have nothing to explain a removed label with. The prompt joins
  `quality_prompt_fingerprint`, so editing it invalidates the rows it produced. Cost on
  the live collection, counted on stored scores rather than estimated: 1 331 candidates,
  ~17 minutes, against 4.3 hours for asking about everything. The expected 97-99% / 66%
  is a **prediction** extrapolated from the F122 labelling, not a measurement of this
  feature — `scripts/measure_frame_quality.py --features verify [--labels …]` is the tool
  that will confirm or refute it, and the brief pre-commits to reporting whichever way it
  goes.
- **A workspace for going through frames** (F126): the "Duplicates" tab became
  "Review", and duplicates became the first of its four slices — **duplicates ·
  blurred · closed eyes · no subject**. Those are four names for one job (look at a
  frame, decide whether it stays), and three of them were computed into `frame_quality`
  back in F113 and **reachable from nowhere**. Every slice keeps its place in the
  switcher with a counter, at zero as well: "you have no closed eyes" is an answer, a
  vanished entry is a riddle. The counters repeat in "Overview" over the same queries,
  so a number and the list it links to cannot disagree. Duplicates stay the **only
  grouped** slice, with the keeper choice, their own route and their own rendering
  untouched — that is the one path in the product that deletes files, and a regression
  there would cost more than the whole feature. The blurred list is ordered by
  ascending sharpness and opens as far as `features.blur_review_max` (90), with "show
  more" continuing past it: the window is a prefix of the same ordering, so the seam
  neither repeats a frame nor skips one. There is **no "delete everything below the
  threshold"** button, and that is the measurement talking rather than caution —
  reviewed by eye in bands, blurred frames turn up in every band up to 400, so
  sharpness ranks the list and a person decides each frame ("almost everything is
  blurred and I would delete it, except a couple I keep for the memory"). A decision is
  a row in the existing `dedup_choice` and nothing else: `to_delete` is already
  understood by the sorter, `file_id` is the primary key there, so a frame that shows
  up in two slices carries **one** decision and shows it in both — and `keep` survives
  the next recompute of `frame_quality`, which is what stops the same kept frames from
  being asked about after every run. Without a faces run the closed-eyes slice answers
  with the reason instead of a zero (F125), and the `vlm.quality_scope` dropdown gained
  the **"By faces"** entry that F125's value was waiting for.
- **Animals are visible and actionable in the web app** (F123): a checkbox on the
  "Process" tab, an "Animals" tab, a counter in "Overview" and an album. The signal has
  been computed since F113 and calibrated in F122 — 805 frames of the live collection at
  92% precision — and until now **not one of them was reachable from the interface**:
  no list, no counter, no action. The checkbox is deliberately built like `deep` and not
  like `faces`: animals are not a pipeline stage but three prompts inside the CLIP call
  the `junk` stage makes anyway, so `pets` overrides `features.pets` for the run
  (`/api/process`, `/api/process/rerun-optional`) and the list of stages is left exactly
  as it was — with `deep` it still means **one** run of `junk`, not two. Its hint says
  the thing that decides whether it is ever ticked: this one is almost free, unlike
  faces (17 minutes) and the deep tier (hours). The tab appears by data presence like
  "People" and "Events", pages its grid like the junk buckets, and is ordered **by
  confidence descending** with the score printed on every card: about 64 of those 805
  frames are not animals, and reading down until the quality stops is how a person finds
  where that border sits. `sorter.plan_album` gained `kind='animal'` — the slice needs
  no selector, journals into `move_batches.mode='album_animal'`, defaults to a dry run
  and inherits the F97 rule, so gathering the same album twice makes no `_1` copies.
  Taking a false mark off a frame is **not** part of this: `frame_quality` has one
  writer and every run recomputes it from scratch, so that correction needs a table of
  its own (F124).
- **The frame-quality cascade is reachable from the web app** (F119): `vlm.quality`,
  `vlm.quality_scope` and all six `features.*` keys now have controls in the settings
  column, applied without a restart like the rest of it. F113 shipped them into the
  config and F104's column never grew to match, so the only way to switch on the
  feature was editing `config.yaml` — the exact thing the column exists to avoid. The
  scope is a **dropdown rather than a text box**: `all` costs about 4 hours over a
  20 000-frame collection, and landing on it through a typo is expensive. The settings
  machinery gained two value kinds on the way — `float` (four of these keys are
  thresholds, and the spec knew only bool/str/int) and `choice`, a string closed to a
  fixed set so a misspelling is a 400 instead of a silent fallback. `_apply_settings`
  now loops over sections instead of hard-coding `cfg.vlm`, which is what let
  `features:` join at all.
- **A ceiling for the preview cache** (F117): `imaging.preview_cache_max_gb`, reachable
  from `config.yaml`, from `sorta cache` and from the settings column of the web app.
  The cache has had no bound since F67 — it grew until the disk did, and the only
  cleanup was `sorta cache --clear`, which empties it whole. Measured at ~150 KB a
  frame (12 GB for 38 485 files), which extrapolates to ~45 GB at 300 000 and ~75 GB at
  half a million. The default is **0, meaning no ceiling**, deliberately: the cache pays
  for itself on every full run (a frame is touched 3-5 times, and a cold frame costs
  336 ms against 73), so the answer to a full disk is a bound rather than switching the
  cache off — nothing is ever deleted unless a person sets a number. Over the ceiling
  the previews that have gone **longest without being read** are dropped (atime, with
  mtime as the fallback where Windows updates atime lazily), and only as many as it
  takes to fit: a preview later stages keep reading is cheaper to keep than to decode
  again, and the directory is never emptied wholesale. The size is checked every 512
  writes rather than per write — the check walks the whole directory — so the cache can
  overshoot by ~75 MB between checks, which is nothing against any ceiling worth
  setting. `sorta cache` now prints the ceiling and the share used, or says none is set
  and names the key that would set one.
- **Search by words, and an album from a query** (F129): `sorta search "cake"` prints a
  ranked list of paths and scores, `sorta album query "cake" --dest …` gathers the top of
  it into a folder with hardlinks. The engine is a new module, `sorta/search.py`; nothing
  new is computed for it — F128 already stores an L2-normalized CLIP vector per canonical
  photograph, so a query is one text encode and one matmul against what is on disk. What
  this changes is not only the interface: a slice of the collection stops being written
  code (a tab for animals, a tab for people) and becomes a **query**, so "food", "snow",
  "the sea" are no longer a programmer's work. Three properties are the feature.
  **Vectors of another model never rank** — mixing two embedding spaces produces a
  plausible order that nothing in the output marks as wrong, so such rows are skipped and,
  if that leaves nothing, the command says which of the two states it is in. **An empty
  table is a reason, not an empty list**: "nothing was found" and "nothing was ever
  computed" read identically in zero lines, and only one of them is fixed by running
  `sorta junk`. And **this ranks, it does not classify** — there is no "this really is a
  cake" threshold and there will not be one, for the same reason sharpness has none, so
  `features.search_limit` (200) is a **sample size**: how many frames the list prints and
  the album gathers. The known limits are written down rather than discovered later:
  compound queries ("a cake with candles on a table by the window") are weak where single
  subjects work well, and the population is personal photographs only — a screenshot or a
  document has no vector at all. The accuracy on a real collection has **not** been named:
  `scripts/measure_search.py` prints the top-N, the distribution over similarity bands and,
  from a filled-in worksheet, precision by depth and by band — F121/F122 is why a number
  nobody measured is not allowed to stand in for one. No interface button yet: the search
  box belongs in the "Slices" block of the web app, which F133 is rewriting.

### Changed
- **One run button instead of two** (F135). The "Process" tab offered "Start" and
  "Re-run selected" side by side, and the second one bought exactly three stages —
  `index` (34 s), `geo` (3 s) and `landmarks` (139 s) on the 380 GB validation run,
  about three minutes — in exchange for a permanent fork in the road: which of the two
  is the right button this time. Everything else was already skipped by the stages
  themselves (`junk` by the prompt fingerprint, `faces` by the marker, `events` by
  composition). So "Start" is the whole pipeline again and the stages skip what is done,
  which is what they were built to do. Two things make that an improvement rather than
  three minutes saved for a worse tab. The **source comes back by itself**: the status
  snapshot carries the path of the last run, and an empty field is filled from it — the
  browser's own memory covers a page reload, this covers a fresh profile against a live
  server, and between them a repeat run never means typing a path again. And the run now
  **says what it skipped**: the stages that can tell new work from work they recognised
  as already done — `index` and `junk` — report `processed`/`skipped` into the status,
  and the finished run prints a line per stage, the same pair the CLI has always printed.
  Without it a run that correctly skipped everything is indistinguishable from a run that
  did nothing, which is the reading the second button existed to avoid. The checkboxes
  are unchanged: unticked still means the stage does not run at all. `POST
  /api/process/rerun-optional` **keeps working** and its tests are untouched — it is in
  the API documentation and callable from outside, and retiring a public route is a
  decision of its own, not a side effect of tidying up the markup.
- **`sort`, `album`, `geo` and `ui` speak the configured language** (F118). F112 moved
  the CLI's output into the i18n catalog and reached `cli.py` only: `sorter.py` printed
  its plan summary, its in-place and `--move` warnings and its blocked-multi notice in
  **Russian whatever `language:` said**, and so did the `geo` progress fallback and the
  line `sorta ui` prints on start. Found while correcting the guides, which quoted that
  Russian output as if it were what an English reader sees. Eleven strings moved into
  the catalog in all three languages; the command echo itself (`sort --by city --apply`)
  stays untranslated on purpose — it is what the reader would type. The developer smoke
  tool behind `python -m sorta.faces` is deliberately left as it is: it is not a command
  a user runs.
- **The cloud naming provider defaults to Claude Opus 5** instead of Claude Opus 4.8.
  Not a fix — 4.8 is a live model — simply the current recommendation at the same price,
  and the provider stays opt-in behind `naming.provider: claude`, off by default.
- **Animals are reported as `animal` rather than as a species, and the threshold is
  0.7** (F122). Both changes come from the same measurement: 320 frames labelled by
  hand, sampled by score band and weighted back to the collection. The binary question
  — is there an animal here — was **92% right**; the species assignment on top of it was
  not, and the review that prompted this found drawn cats, plush toys, a hotdog and
  people in fur coats all filed as `cat` or `dog`. So the class is now the one call the
  numbers support. The three CLIP prompts still run: they are the ensemble the threshold
  was calibrated on, and collapsing them into one would move every score and void the
  calibration — only *which* of them won is no longer recorded. On the threshold, 0.70
  marks 805 frames at 92% precision / 54% recall against 895 at 89% / 58% for the old
  0.60, and it dominates 0.85 outright (same precision, nine more points of recall).
  With ~40 frames a band the interval is about ±8 points, so this is a justified
  preference rather than a proven optimum — and since `pet_score` is stored, re-choosing
  the threshold never needs another pass.
- **The VLM is no longer asked whether a shot was accidental** (F122). On the labelled
  sample the question was right **5% of the time**, and 10% of the frames it called
  deliberate were not — noise dressed as a signal, and one of the two categories a
  reviewer could not interpret at all. One of three questions in the prompt is gone;
  `frame_quality.is_accidental` stays in the schema and stays NULL, because NULL already
  means "not asked" and dropping a column in SQLite costs a table rebuild.

### Fixed
- **Frame quality is measured over photos only** (F120). Sharpness, the pet score and
  the VLM answers were computed over the whole collection — screenshots, documents,
  product shots and memes included. That is not a slow path, it is a wrong population:
  a screenshot scores 2854 on the laplacian against 1253 for a photograph, so "sharp"
  measured across both means nothing, and the frames that most often carried a pet class
  were stock wallpapers. The stage now selects photos, refuses to write a row for
  anything else, and purges rows left by earlier runs at the end of `classify` (the
  purge has to be last: verdicts are not written yet when the stage begins). Sharpness
  also joins the near-duplicate keeper recommendation, where it is finally a fair
  comparison — the same scene at the same scale.
- **The eyes answer is believed only where a face was detected** (F121). The prompt says
  to use neither word when there are no people; the model does not obey it, and the
  first live run answered `eyes_open` on cats and `eyes_closed` on people in glasses.
  Asking stays free — one prompt, one call — but the answer is now dropped where the
  detector found no face. Deliberately not dropped when `faces` has never run: there,
  "no face here" and "nobody looked" are the same empty table, and treating them alike
  would switch the signal off for everyone who skipped the stage, silently.
- **Editing a prompt invalidates the rows it produced** (F120/F121). `frame_quality.source`
  now carries a short fingerprint of the prompts that filled the row (`clip#abc12345`),
  so changing a prompt makes the stage recompute instead of skipping rows that look
  processed. Without it, every semantic change to a prompt needed a hand-written DELETE
  that nobody would remember to run — and the F122 class change is exactly that kind of
  change, invisible to a marker that only records the tier.
- **A release bump no longer fails the suite, and neither does a restarted exiftool
  session.** Four rounds of one bug: `test_exif_parallel` asserted exact process counts
  while `_ensure` transparently restarts a session that dies, so any exact count was a
  coin toss. Each earlier fix was scoped to the assertions in front of it rather than to
  a search of the file, and the next CI run found the next one. The file is now audited
  in full: every launch count is bounded on both sides against what the pool was asked
  for, and the only exact equalities left are `launches() == 0`, where nothing was
  started and there is nothing to restart.
- **Every configuration key is documented, and the watchdog now says so.** The guides
  carried a `config.yaml` sample and a page on the `vlm:` section; everything else was
  reachable only by reading the source or `config.example.yaml`. Widening the F115
  watchdog from one section to all ten turned that into a number: **51 keys across nine
  sections** had never been written up, `features:` (F113) among them in full, along
  with every threshold of `naming:`, the whole of `faces:` and `dedup:`, and the video
  keys of `imaging:`. §21 of all three guides now carries a per-section table — key,
  default, and what it actually does, with the measured reasoning where there is any
  (why `naming.clip_batch_size` is a weak lever and `naming.clip_decode_workers` is the
  real one; why the landmark threshold is not to be lowered; why trips merge by
  coordinates rather than by city id). The check requires the dotted form on purpose: a
  bare `preview_quality` inside a YAML block is easy to write and impossible to search
  for.
- **The English and Japanese guides no longer quote Russian output.** Both carried some
  thirty samples of CLI output in Russian, plus a glossary translating the words — all
  accurate until F112 made the output follow `language:` and defaulted it to `en`.
  Every sample is now rendered from the i18n catalog for its own guide rather than
  transcribed, and a test refuses Cyrillic inside a fenced block of either file.
  (Cyrillic in prose stays: the link to the Russian translation, and layout folder names
  like `Россия/…`, which are data rather than chrome.) The Russian guide had a stale
  sample of its own — `sorta geo` gained `path_inferred` with F85c and the example never
  did.

## [0.3.1] - 2026-07-31

A re-cut of 0.3.0 on a green commit. **No module under `sorta/` changed** — install
either and you get the same program; this tag exists so the released commit carries a
passing CI run.

### Fixed
- **A release bump no longer fails the test suite**: `test_cli_help.py` spelled the
  version out as a literal (`"Sorta v0.2.0 — …"`) inside a case that is about the three
  interface languages, not about the version, so bumping to 0.3.0 turned all four CI
  matrices red. It reads `sorta.__version__` now; the `v{version}` shape of the string
  is still pinned, by the catalog case that owns it. The local gate had gone green on
  the same tree by accident of timing — it started before the bump, and that test runs
  in the first minute of a twelve-minute suite.
- `uv.lock` still declared `sorta 0.2.0` after the version bump.

## [0.3.0] - 2026-07-31

### Fixed
- **The offline geo base is shipped inside the package** (F65): `v0.1.0` and `v0.2.0`
  were installable but not usable — `data/geo/*.tsv` (the bundled `places`/`names`/
  `admin1` tables) was never listed as package data, so a wheel-based install resolved
  no city at all and wrote garbage into `places`. The tables now live under
  `sorta/data/geo/` and are declared in `pyproject.toml`; a missing base fails loudly at
  startup instead of degrading into empty results. Anyone who installed an earlier
  release must re-run `sorta geo` after upgrading — the `places` rows written by those
  versions are wrong. **This is the reason 0.3.0 supersedes both earlier releases.**
- **A forwarded picture is one thing, not two** (F83): the same frame could be counted
  as junk by one signal and as "downloaded" by another, landing in two buckets and two
  folders of the layout. One verdict now decides where it goes.
- **The offline base answers when the online provider has no city** (F86): with
  `geo.provider: online`, coordinates the provider resolved to a region but not a city
  used to leave the file place-less; the bundled base is now asked as a second step.
- **Trips merge by where the files are, not by which city id they got** (F92): two legs
  of one journey were split whenever the geocoder handed neighbouring frames different
  geonameids for the same town.
- **The detector no longer runs the network twice per frame** (F88): the insightface
  input size was left unpinned, so the model was re-prepared per call.
- **The test suite passes on Linux** (F116): four failures were properties of the tests
  themselves — terminal styling forced under `GITHUB_ACTIONS`, a faked `os.name`
  building an impossible `WindowsPath`, a test keyed by thread ident. No product code
  changed; CI is green on both platforms.
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
- **The VLM has a config section, and its input resolution is a knob in it** (F102): the
  local Qwen2.5-VL was configured out of the `naming:` section — the toggle
  (`naming.vlm_enabled`), the model (`naming.classify_vlm_model`) and the preparation
  threads (`naming.vlm_workers`) all lived under "naming" because there was no other
  address, even though the first of them switches **junk classification** on. The knob
  that decides what the pass costs was not in the config at all: `VLM_MAX_EDGE = 896`, a
  constant in the code, in a project whose stated rule is that thresholds live in
  config.yaml — and after F101 removed the CPU half of the pass, the remaining time is
  the GPU half, which the input resolution is exactly what decides. The new `vlm:`
  section holds what describes the shared model **runtime** (`enabled`, `model`,
  `workers`, `max_edge`); what belongs to a consumer stays with the consumer
  (`naming.provider` picks who names events, `naming.product_candidate_min` is the junk
  gate deciding which frame is worth a call at all). **Nothing has to be edited**: the
  three old keys are still read, the new one wins when both are given, and a config using
  only the old address logs one warning per run rather than one per frame — silently
  ignoring a `naming.vlm_enabled` would mean switching a 20 GB tier on, or off, on
  somebody else's machine. Defaults are unchanged (`enabled: false`, 896,
  `min(4, cores)`), so a run without a config edit is the run that ran yesterday, and a
  bad value (a word where a number belongs, a negative, a quoted `"false"`) falls back to
  the default instead of starting something heavy or crashing. Token budgets deliberately
  did **not** move: 8 for a junk label and 32 for an event name are the protocol of the
  conversation with the model, not a user setting.
  `scripts/measure_vlm_resolution.py` is what the knob was made for — one load of the
  weights over one sample of the tier's own candidate frames at 896/672/448, with
  median/p90 ms, frames/s, peak VRAM and a label-by-label verdict comparison, judged by
  criteria registered in the code before the first run (agreement ≥ 98% on ≥ 300 frames,
  no more than 2% of documents lost to `photo`, speedup ≥ 40%) and printing which of the
  three outcomes the data reached.
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
- **The CLI answers in the language from the config** (F112, F114): every runtime message
  and, since F114, `--help` itself — command summaries and option descriptions — are
  read from the i18n dictionaries. **Behaviour change:** the default is now `en`, so a
  user with no explicit `language:` key sees English where 0.2.0 showed Russian; set
  `language: ru` to keep the old output word for word. `--help` peeks at `--config`
  without consuming it, and an unreadable or missing config falls back to the default
  language rather than a traceback.
- **`media_class.tier` separates the processing tier from the deciding signal** (F68):
  which tier looked at a frame and which signal produced its verdict were the same
  column, so re-running a tier erased the provenance of the answer.
- **A CPU-only install on a CUDA machine is surfaced, not silently slow** (F76):
  `sorta doctor` and the startup check report a PyTorch/onnxruntime mismatch instead of
  letting CLIP and OCR run for hours on the CPU with the GPU idle.
- **Comments and docstrings are English** (F111): an audit converted 41 comments and 81
  docstrings; the Russian that remains is the *subject* — folder-name parsing, country
  names that double as ordinary words, service folder names, test fixtures. No
  functional string was touched: CLI messages, i18n dictionaries and layout folder names
  are the product, not documentation about it.

### Added
- **Per-frame quality signals, each taken with the cheapest tool that answers it** (F113):
  choosing the best frame of a burst, or spotting the one nobody meant to take, needs
  facts the index did not hold — how sharp a frame is, whether there is an animal in it,
  whether the eyes are open, whether there is a subject at all. The new table
  `frame_quality` (schema v15) holds them, and the point of the feature is the price of
  each: **sharpness** is the variance of the laplacian over the preview every other stage
  has already decoded — milliseconds, no model, so it has no toggle and is written always;
  **pets** (cat/dog/pet) are a prompt group **appended to the CLIP call the junk stage
  already makes** — no second pass, no second call, behind `features.pets`; and only what
  neither of them can decide — eyes open, a subject at all, an accidental pocket shot —
  goes to the local VLM at ~0.78 s a frame, behind `vlm.quality`, and only over the
  **uncertain band** (sharpness in the zone where it decides nothing, or a CLIP score too
  low to mean anything) inside `vlm.quality_scope` (pHash groups by default, `events` or
  `all` on request — `all` is 4.3 hours on 20 000 frames and says so in the config).
  Asking the VLM about pets across a collection would have been exactly those 4.3 hours;
  CLIP does it in minutes because it is already looking at every frame, and "is there a
  cat in this picture" is a question about an object — which is what CLIP is good at,
  unlike the question about a frame's PURPOSE that F110 measured it failing. The junk
  verdicts do not move a millimetre under the added prompts: a softmax restricted to a
  subset of its own inputs and renormalized *is* the softmax over that subset, so
  `naming.junk_threshold` still means what it was measured to mean. **NULL means "not
  asked", never "no"** — an answer the model gives in prose instead of keywords leaves the
  columns NULL rather than guessing False, because a consumer that read a defaulted False
  would decide a frame nobody ever looked at has its eyes shut. The three thresholds ship
  as **provisional** values with the measurement that has to replace them:
  `python scripts/measure_frame_quality.py --features pets sharpness band` prints the
  score distribution, what each candidate threshold would fire on, and how large the VLM
  band would be — over at least 200 of your own frames, in aggregates that identify none
  of them.
- **The state of the collection, on one screen** (F108): every key number of an archive
  was reachable only through a hand-written SQL query. On 2026-07-28 the numbers that
  actually decided what to do next — **7 619 frames with no place, 2 202 products, 819
  documents, 2 592 verdicts changed by the deep tier, 360 events instead of 473 after a
  geo-provider switch, 22 364 files in the layout plan** — were all pulled out of the
  database by hand, and a user of the tool could not see a single one of them. The new
  **"Overview"** tab is the first one and opens by default whenever the index is not
  empty, in four groups: what is in the index (files, photos/videos, duplicates, read
  errors, events); how each frame got its place (exact GPS, inherited from the session /
  the trip / the folder name, recognised from the frame, set by hand) and **how many have
  no place at all, with the percentage** — the one number that predicts the quality of a
  layout, because each of those frames goes to the "no place" folder; what the classifier
  decided, split by verdict, by `source` and — separately — by the `tier` that handled it,
  which is what answers "did the deep VLM tier run at all"; and whether a layout ran, when,
  where, in which mode, over how many files, and **whether its batch was ever closed** (an
  open `finished_at` is the trace of an interrupted run and is stated as such). Every
  number that has a tab of its own is a link to it — products and documents to "Not
  personal photos", duplicates to "Duplicates", events to "Events": an overview without
  those is a report, and what is needed is a control panel. `GET /api/overview` is built
  from **plain SQL aggregates only** and is not cached: a plan of 24k frames takes minutes
  to build, and this is the screen opened right AFTER a run to see what changed, where a
  stale number is worse than a missing one. Aggregates only — no file path and no file id
  leaves the endpoint, and there are no previews on this tab at all. An empty index gets
  an invitation to pick a folder instead of a table of zeros.
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
- **A source-folder tree with pre-index exclusions** (F81, F82): before indexing, the
  source tree is shown as it is on disk and folders are switched off — tri-state, so a
  folder can be excluded from the *scan* entirely or scanned but kept out of the
  *layout*. Nothing is read that the user marked as not theirs.
- **Manual exclude and reassign from the web app** (F77): a single frame can be pulled
  out of the layout or moved to another place/person/event without editing the DB by
  hand; the correction is stored as a layer over the computed verdict and survives a
  re-run.
- **A settings column and a summary before the layout** (F104): toggles are switched in
  the interface and applied without restarting the server. The config writer is textual
  and line-based, so the user's comments survive an edit made from the UI. Undo moved
  out of the plan screen to the "Moves" tab, next to the manifest it acts on — an undo
  made from the plan screen is an undo made blind, and it deletes files.
- **Downloaded pictures get their own folder** (F78): frames saved from messengers and
  the web no longer share `_Unsorted` with undated shots — two different problems that
  needed two different answers.
- **Landmark matches are corroborated, not just thresholded** (F75): a zero-shot CLIP
  score above the bar is no longer enough on its own; a match must agree with a second
  signal before it names a place.
- **`sorta faces --rescan`, with labels that survive it** (F89): re-running detection on
  a changed collection no longer costs the names already assigned to clusters.
- **The clustering phase of `faces` is visible** (F84): HDBSCAN over tens of thousands of
  embeddings is minutes of silence; it now reports where it is.
- **Video files get a preview frame, and clips get a filmstrip** (F74, F80): videos enter
  the shared preview cache like stills, and a clip is shown as six frames instead of one
  tile — one frame is not enough to tell what a clip is.
- **A run log with stage timings and an environment header** (F69): every run writes what
  ran, how long each stage took, and on what hardware — the first thing to ask for when
  a run behaves differently than the last one.
- **The install section rewritten, and the documentation matches the product** (F79,
  F115): three statements in the guide were not gaps but errors — the default language,
  a "known limitation" already fixed, and a list of junk classes missing the biggest new
  one. A gap is noticed by a reader; a false statement is believed.

### Performance
- **EXIF through a pool of exiftool sessions, read with `-fast`** (F71, F72): the
  metadata stage runs several `-stay_open` sessions in parallel and uses the cheap read
  path, recovering by a second pass only the fields `-fast2` drops.
- **A lazy disk preview cache — one decode per frame** (F67): a frame is decoded once and
  reused by every later stage and by the web app (73 ms against 336 ms per frame on a
  cold cache).
- **Near-duplicate grouping sped up, "Duplicates" tab cached** (F66); **a lazy plan cache
  and a paged `/api/plan`** (F70), so a collection of tens of thousands of frames opens
  in the browser instead of timing out.
- **OCR in a pool of detector sessions** (F73) and **frame decode off the inference
  thread** (F87): the junk stage no longer blocks its own GPU session on I/O.
- **The original is decoded only where there is a face to crop** (F91).
- Measured and **rejected** on purpose, so the questions stay closed: raising the VLM
  candidate gate (F106 — 9.8% of the time for 6% of the findings), lowering the VLM input
  resolution (F102 — drops documents), sdpa attention in the vision tower and batching
  (F105 — 1.5× *slower*, twice the VRAM), a CLIP probe as a cheaper gate (F109, F110),
  StreetCLIP for places (F85b), an OCR gate (F90). The VLM pass gained ×1.20 in total,
  and the numbers behind each rejection are in the commit messages.

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
