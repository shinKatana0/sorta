# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Files can be reassigned to a folder the plan does not have** (F203). The owner
  reported on 2026‑08‑05 that the layout would move selected frames **only into folders
  the program had already proposed**: the picker listed the plan's own categories, so the
  country root — `Россия/` with no city and no year, a branch the layout has carried since
  F86 — and a directory that does not exist yet were both unsayable. This is the same gap
  as F202 one level down: the product suggests well and listens badly, and the layout is
  the last place where the person's decision should win, because the files really move.
  The target is now **typed**, with the plan's folders kept as suggestions beside the
  field, and it is checked as a **name** rather than as a path: a separator, a `..`, an
  absolute path, a colon or a control character is refused with a **reason** the page says
  in three languages, instead of being stored and silently dropped hours later when the
  layout ran. The naming rule is **one function** shared with the layout
  (`sorter.manual_target_parts`, built on the `_sanitize` every folder name goes through),
  so the folder the field accepts and the folder the apply writes cannot be spelled
  differently — and the sorter keeps asking it again when it reads the row, since
  `manual_overrides` is a file on disk. Nothing is created before `--apply`, a name
  conflict still gets its `_1` suffix instead of overwriting, and the moves journal and
  `undo` are untouched. There is deliberately **no bulk reassign of a whole slice**: one
  typo would carry a thousand frames.
- **A place can be a region, not only a city or a country** (F202). The owner typed
  «Алтай» into the place picker on 2026‑08‑05 and got two Mongolian towns; «Карелия»,
  «Крым» and «Тоскана» found nothing at all. The base knew only cities and countries,
  while people remember regions — and **7 492 frames** of the live collection have no
  city, many of them with a correct answer that exists at exactly that level. Nothing was
  downloaded for it: `admin1.tsv` (**3 865 regions**, each with its own geonameid) has
  shipped in the wheel all along, and city labels have been showing those names for as
  long — only the **search** and the **layout** could not name one. Now the picker answers
  **country, then region, then city** (the wider the level, the more visible a wrong pick
  is in the plan), a region option **says that it is one** (`место (область, страна)` —
  one word is regularly both a city and a region), `manual_places` stores it in one new
  column (`region_geonameid`, schema **v28**, older rows unchanged), and the layout gains
  a third branch — `<Country>/<Region>/<Year>` with its own reason `region_only`, so the
  plan says at which level the decision was made. The F85c rule does not weaken: a manual
  place is still taken **whole**, so an assigned region never keeps an inferred city, and
  a body asking for a city and a region at once is refused. A region is never **inferred**
  — that is a separate question with its own cost of being wrong. Names come from the same
  `names.tsv` in all three languages. Note the data's own spelling: 552548 is «Карельская
  республика» in Russian, so a Russian search finds it by the start of that name (or by
  the short «Karelia» of `admin1.tsv`), not by the word «Карелия».
- **A group of duplicates says which tier it is in** (F199). The tiers behaved correctly
  and named themselves nowhere: the owner opened the screen on 2026‑08‑05 and saw two
  outwardly identical pairs, one carrying "★ largest file" and the other carrying nothing,
  with the files differing in size in **both**. The behaviour was right — one pHash across
  a group means one picture stored twice, where "keep the largest" is a statement about
  **facts** (resolution and weight); several pHashes mean different photographs, where no
  rule may speak. Unsaid, that reads as randomness, and a person who reads it that way
  stops trusting the suggestion in the tier where it holds. So **every group now carries
  its tier in words** (`tier_caption`, `tier_why` in `/api/dupes`), shown as one line above
  its frames with the reasoning folded behind a *why*. The third tier's line states the
  **measured** number in the form F171 set for the other slices — over 111 groups labelled
  blind, **not one rule beat picking at random** (27–32% against 30.4%) — and says the
  thing the report was really about: the **file size has nothing to do with the tier**,
  which follows from whether it is one picture. Nothing about the tiers themselves moved:
  `dedup.group_tier` is unchanged, `phash_max_distance` is unchanged, the first tier is
  still a number rather than a list, and the third tier still preselects **nothing**.
- **Three tiers of sameness on the duplicates screen** (F194). One word, "duplicate", was
  covering three populations whose cost of a mistake differs by orders of magnitude — and
  the screen applied the same apparatus of "choose the one to keep" to all three. Counted
  on the live collection 2026‑08‑04: **12 350 byte‑identical files over 7 631 originals**
  (half the archive), **299 groups** of the same picture stored in different files, and
  **791 groups** of merely similar frames. Each now has its own default. **Byte copies are
  collapsed into a line with a number** — the bytes are the same bytes, so "which of these
  twelve thousand do you want" is a question about nothing; collapsed means off the screen,
  and **nothing is deleted**, the files stay on disk. **The same picture in several files
  suggests its largest copy** (★), because resolution and weight are checkable facts rather
  than taste — the one place a free rule works, and the one place the old screen never
  applied it. **Similar frames suggest nothing at all**: labelled blind over 111 groups by
  the owner, no signal we have is distinguishable from choosing at random (sharpness 27%,
  arithmetic 28%, cascade 28%, the model 32% — random itself 30.4%), and the interface was
  highlighting *sharpness*, the one that scores below a coin. A highlighted frame reads as
  an answer, so a person trusting it chose worse than by pointing blindly and never found
  out. The frames are still **sorted** by sharpness — that is what a ranking honestly is,
  and the caption now says "sorted by sharpness" instead of naming a best frame.
- **Keeping several frames of a burst** (F194). A series of five can hold three worth
  keeping — a portrait with the eyes open, another expression, a wide shot — and "the best
  one" threw two of them away. The keeper controls are **checkboxes now, not a radio**, and
  a group nobody ticked in **keeps every frame**, which is the third tier's default said in
  the only way that cannot go wrong: it writes nothing. `POST /api/dupes/choices` takes
  `keep_file_ids: [...]` (the single `keep_file_id` still works); an **empty** keep list is
  refused, because "delete the whole group" is the one sentence that route must not be able
  to say. `group_keeper` keeps being filled and keeps being read — as the **order** of a
  group, never as its answer.
- **Every slice behaves the same: pick some frames, name the folder, gather it** (F193).
  Three complaints of 2026-08-04 were one defect — the slices answered "what can I do with
  you" differently. An album was **all of a slice or nothing**; the folder could be **named**
  in memes and screenshots and nowhere else, so "With people", "Group" and "Portraits" were
  the odd ones out; and the **documents** bucket answered by saying nothing at all — no
  button, no sentence, no refusal. Fixed one at a time that question would have been
  answered three times, differently, and the fourth divergence would have arrived with the
  next slice. So there is now **one album row and one selection**, built in one place and
  used by every slice of the tab — the search line, a pinned query, the three face slices,
  the animals, every class bucket and the «Просмотр» workspace — the way `makePager`
  already gives them all the "show more" button. A slice added tomorrow gets the tick, the
  folder-name field, the destination and the button by calling it, and a slice that does
  not call it has no row at all. The tick on the cards feeds the album row above them:
  ticking the first frame turns the scope to **"only the selected"** by itself and
  unticking the last turns it back off, so the button never quietly gathers a selection
  that holds nothing. The ids **narrow** the slice and never widen it — they are ANDed onto
  the membership rule the slice is built from (`sorter.plan_album`) — so a request made
  past the interface cannot pull a frame out of a slice it is not in, and no guard a slice
  carries is something a selection can walk around. An **empty** selection is refused with
  a reason instead of gathering a folder of zero files, which would read as "this slice is
  empty" — a statement about somebody's archive that is not true. **Documents** keep the
  decision they were given: they are not gathered into a folder, and `vlm.exclude_classes`
  does not change that — the key decides what is SHOWN, never that a folder of somebody's
  passports, medical forms and bank papers may be assembled in one click. What changed is
  that the program now **says so**: the bucket carries the album row like every other slice
  and the refusal comes out of the route with a reason the interface puts into a sentence,
  because a hidden button forbade nothing and explained nothing. Every bucket now answers
  one of the two — the kind it gathers as, or the word for why it does not — including a
  class shipped without an album kind, which used to fall out of the interface in silence.
- **"Try to improve" on the frame you have opened, in every slice** (F168). The action
  shipped behind one door — the «Размытые» slice — and the measurement of 2026-08-03 found
  that door nearly shut: at its threshold the sharpness filter holds **8%** of the frames a
  person calls soft (it answers "how much detail is in the frame", not "is it in focus"),
  so a useful operation sat behind a detector we had measured to be almost blind. The
  second measurement (F169, 80 blind pairs against plain bicubic enlargement) says where it
  really belongs, and it is **not blur but SIZE**: 66% wins under 640 px, 58% at 640–1024,
  and by 1280 px a coin toss. So the entrance is now the **expanded frame**, which every
  slice already opens — cities, people, animals, events, search, pinned queries — and the
  offer stands while the frame is below `features.restore_max_edge`. Above the ceiling the
  button is **withdrawn and the reason is said**: the copy there would be rebuilt from a
  reduced version of itself and the measurement found no gain, which is a sentence beside
  the picture rather than a silence. Nothing about the operation changed: the same route,
  the same one frame per press, the same copy beside the original with the original
  untouched, and a second press still returns the copy that already exists instead of
  making another. Two refusals are enforced **in the route** and not by a missing button —
  a frame classed as a personal document (`vlm.exclude_classes`) is never decoded and drawn
  four times larger, and a clip is not an image the model has an answer for — so a request
  made past the interface collects the same refusal. The copy is a canonical file that lies
  in the city folder beside its source, so wherever it is opened it now names the frame it
  was processed from instead of reading as a second similar photograph from nowhere.
- **A name in the search line finds the person** (F189). Naming a face cluster and merging
  another into it made that person reachable by `sorta album person <name>` and by
  `sort --by person` — and by no query anybody could type: the search line knew only CLIP
  vectors, so «Ирина» asked the model for frames that resemble a **word**. The bridge is a
  parse of the query string and nothing else — no index, no threshold, no cluster work.
  The two answers stay **apart**, which is the whole feature: a query is a RANKING (recall
  grows with depth, precision falls) and a name is an **exact selection** (the frame is in
  that person's cluster or it is not), so a name gives a LIST — no threshold, no depth, no
  "show more by relevance", paging by count only — and the answer says which kind it is
  ("Кадры человека: Ирина — показано 200 из 214", and no closeness number on the cards).
  Which frames are the person's is not decided twice: the selection is the album's own
  (`sorter.plan_album(kind='person')`, the `merged_into` roots of F31), and a test compares
  the two SETS rather than trusting that they agree. The match is exact apart from case and
  stray blanks («ирина » and «Ирина» are one name); nothing fuzzy («Ира» does not find
  «Ирина» — a near miss on a name puts somebody else's photographs under it), and an
  unnamed cluster is found by nothing. **The word search never disappears**: a name that is
  also an ordinary word («Роза», «Марк») shows the person first and keeps the ranking one
  click away — `--words` in the terminal, a link above the grid in the web app. Pinning
  (F156) picks it up: a pinned name is an ordinary tab answering the same set, captioned as
  a person rather than as an estimate and gathering the person's album. Nothing is indexed
  for this and no model is loaded, so it also works on a collection whose search index is
  empty.

### Changed
- **The layout screen copies by default instead of moving** (F200). Everything else about
  a layout is built so a careless first click costs nothing — the plan is a dry run, the
  journal is written before the first file travels, `undo` puts them back, blake3 says the
  copy is the original and nothing is ever overwritten. "Move" preselected was the one
  place left where the first click is irreversible in substance: `undo` returns the files,
  but only while the journal is intact and nobody has tidied the tree by hand. A collection
  laid out by copying can always have its originals deleted afterwards; one laid out by
  moving has to be reconstructed from a log. **`move` is not removed** — it is one click
  away, in the same pair, under the same heading. A body of `POST /api/sort` that omits
  `mode` now means "copy" instead of being a 400: the screen and the parser answer one
  question, and two defaults in two places would be free to drift apart. **The terminal is
  untouched** — `sorta sort` still moves by default and still writes nothing without
  `--apply`; that is a separate entry point with a decision of its own.
- **The "Layout" tab asks two questions instead of showing thirteen controls** (F192).
  The tab opened onto a destination field, a move/copy pair, the apply button, expand and
  collapse, four buttons of manual corrections with a folder list, a place picker, a
  delete button and the plan tree — a control panel where two things are needed **every
  single time** and the other eleven are answered once. That is exactly the split F133
  made on the run screen, and the same remedy applies: the tab now opens onto a desk of
  two fields — **where** the collection goes, and **by what** it is grouped — with
  everything else one click away behind a gear that opens in place, above the tree it is
  used against. **Not one control was removed**: move-or-copy, the corrections, the place
  picker and the tree buttons are all still there, in a block with headings of their own,
  and a test pins the inventory as a set so a future tidy-up cannot quietly drop half of
  it. The second field is new only on the screen: the criterion is `sorter.MODES` — the
  same **city / person / event** `sorta sort --by` has laid a collection out by since F5,
  of which the web app could reach only the city one. `POST /api/sort` and
  `GET /api/sort/summary` now carry `by`; absent, it still means "city", which is what
  they meant before. **Layout and albums are not mixed here** — every album is gathered
  from "Review" or "Slices" and none of those boxes is on this tab — but the difference is
  now written where the choice is made, because "by person" can mean either: a layout
  moves the originals, an album is links beside them and moves nothing.
- **The web app is a package cut by tab, not one 14 427-line file** (F182). Nothing a
  person can see changed — this is the entry for a move, and the move is the point. On
  2026-08-03/04 **ten features queued for `ui.py` in a single day**, because two workers
  inside it is a guaranteed conflict: F152 came back with 18 divergences across 10 files,
  F160 with an import that vanished and that neither gate caught on its own. No other
  module ever had that problem — `junk.py`, `geo.py`, `search.py` and `scripts/` took
  features in pairs and threes without a collision. The cut is **by tab, not by layer**: a
  feature normally lives in one tab (F150 in «Разбор», F156 in «Срезы», F159 on the run
  screen), so two features in two tabs now stop meeting at all, whereas a cut into
  routes/queries/markup/script would have left every feature touching all four. F133 had
  already rebuilt the interface along that axis and the file was already marked with
  fifteen `# --- F126: рабочее место «Разбор»` seams; this made them real —
  `sorta/ui/{common,layout,slices,review,overview,moves,process,page,strings}.py` with the
  server and the route table in `__init__.py`. The other 42% of the file was never Python:
  the markup, the styles and the browser script now live in `sorta/web/` as `page.html`,
  `style.css` and `app/app.js`, so an editor highlights them, ruff and mypy stop reading
  the text of a page, and a conflict in one word of markup is no longer a conflict in
  `ui.py`. **Byte for byte the same page** — the rendering of all three languages was
  captured before the move and compared after it — and `sorta.ui` still answers to every
  name it answered to, which is what let the ~3 700 existing tests stay untouched apart
  from where a stub is attached. The frontend files are named among the wheel artifacts:
  F65 shipped a release whose data never made it into the package, and a page without its
  script would fail the same way, louder.
- **The screenshot slice says it is an opinion** (F171). On 350 hand-labelled frames the
  `screenshot` verdict is right about **59%** of what it points at (83% recall): every
  third frame in that bucket is an ordinary photograph. The live run of 2026-08-04
  reproduced the prediction made from that sample to within one frame — the rescue added
  **441 frames** to the bucket (1 782 against 1 341), and 41% of what it adds are
  photographs, so ~**181 personal pictures** left the city layout for a list a person
  reads as "these are your screenshots" and never opens. No verdict, threshold or file
  moves here: what changed is what the slice says and in what order it says it. The
  caption now names the **model** as the author of the verdict, states the measurement it
  was written from, and names returning a frame as an **ordinary step of the work** rather
  than the repair of a rare mistake. A bucket is answered as a **list in order** —
  `media_class.score` descending, the frames the classifier settled without a number
  keeping the path order behind them — and the page says whether that ordering happened
  (`ordered_by_score`), so the caption promises a ranking only where there is one; the
  "all" view keeps the path order, because four classes are four separate softmaxes and an
  order across them would be a comparison nobody measured. The way back is the action that
  already existed (`POST /api/overrides` with `photo`), above the grid where it was, and
  all three guides now carry the measurement, the run that confirmed it and the advice to
  look the list over before deleting anything.
- **Product recognition is a line of its own, with a price** (F161). "Deep analysis
  (VLM)" was the master switch of every question the model is asked (F145) and, by
  itself, the one thing that switched the deep junk tier on — which made it the only
  option on the run screen with an effect nobody had named and a cost nobody had stated.
  The effect is products: the deep tier is the **only producer of the `product` class**
  (the fast tier never emits one), and on the live run of 2026-07-28 it moved **2 202 of
  its 2 592 changed verdicts** into exactly that class. The cost is ~95 minutes over the
  24 196-photo collection, or ~12 when `features.junk_rescue` narrows the candidates to
  the band above its threshold (955 frames). So the tier is now **`vlm.products`**, a
  line under the master carrying its own price — measured from this machine's run log
  like every other line (F159), and following whichever of the two populations the
  config actually selects. The master keeps the veto and nothing else, and says so: its
  own price reads "0 — permission only", and the caption under it no longer promises
  hours it does not take. `vlm.products` defaults to **true**, the only subordinate key
  that does — a config written before it existed has to run what it ran yesterday, and a
  body without the field (`/api/process/rerun-optional`) still gets the tier. Also here
  because it is the same screen: on an empty collection the caret goes back into the
  path field (F133, lost while F135/F138 rebuilt the run controls).
- **"Blurred" is a list in order, not a window with a threshold** (F157). The morning of
  2026-08-02 measured the completeness of this filter at 6% and spent half a day calling
  that a catastrophe; the evening measurement cancelled the number, and not because the
  filter was better. The sample had been labelled in two sessions, five of the six
  features agreed between them (ratios 0.77-0.93) and blur diverged threefold — the
  criterion had moved, which the sharpness numbers show directly (softly labelled: median
  706 against 980 for the rest, one population; strictly: 254 against 1 022). Under the
  strict criterion the user chose — "visibly smeared", not "a little soft" — **a cutoff
  buys recall far faster than it loses precision**: at 90 it flags 7 frames of 300 and
  catches 2 of the 17 blurred (12% recall, 29% precision), at 300 it flags 47 and catches
  9 (53%, 19%), at 700 it flags 120 and catches 14 (82%, 12%). That is the profile where a
  threshold is the wrong instrument, so the slice is now **an ordering**: the softest frame
  first, `features.blur_review_max` is the **depth of the first page** (raised `90` → `300`
  off a sweep on a live collection — 523 frames at 90, 2 968 at 300, 7 859 at 700 out of
  19 211), "show more" continues down the same list rather than jumping a window, and the
  counter states **how many frames are shown** instead of how many blurred photographs
  there supposedly are. The caption says the same out loud: read from the top, stop where
  the resemblance ends, and know that a detailed sharp street and a smooth blurred face
  score alike. Where F155's `frame_quality.face_sharpness` exists, **the frames that have a
  face are ordered by the number measured inside it and come first** (62% of the blurred
  frames against 15% for the whole-frame number); on a database from before that column the
  list opens exactly as before. The album of the slice still gathers the first page and
  nothing below it, and nothing here is ever marked or deleted automatically.
- **"Closed eyes" is a real slice again, and it is arithmetic rather than a model**
  (F179). The question "are the eyes open" used to be asked of a local VLM: 60% precision
  over 9% of the frames it was meant to find, for ~92 minutes a run. It was retired
  (F177) and the population was not — the same labelling puts it at **~948 frames, 15.6%
  of everything with a face in it**, and nothing pointed at them. F178 priced three
  cheaper ways to see them against the SAME 249 hand labels, and eyelid **geometry** won:
  **62% precision at 48% recall** — five times the model's recall at slightly better
  precision — against 46% for a classifier over the eye crop and 58% for CLIP over the
  same crop. So `frame_quality.eye_openness` (schema v27) is now measured on every run:
  the height of the eye opening over its width, off the 106-point face contour of the
  `buffalo_l` set the faces stage already downloads. **No new pass and no new weights** —
  it rides in the decode the sharpness of F155 already pays for, the model is built on the
  first face of a run and never on a run without one, and a machine that cannot build it
  loses this one column and nothing else. **The coordinates are rescaled** from the
  original frame into the preview (`face_crop_boxes`, the same guard), which is the
  mistake that made the first version of this measurement report 100% recall over the
  crops that happened to survive; a broken crop flatters the result rather than failing.
  **Several faces → the largest one decides**: a frame where somebody at the back blinked
  is not a portrait with closed eyes. In the interface the slice is now **ordered from the
  most closed**, opens as far as `features.eye_openness_max` (default `0.18`) and
  continues past that window on "show more" — and its caption states the **measured 62%**
  rather than a count, because one frame in three of that list has its eyes open and
  nothing there is ever marked automatically. `vlm.quality` still asks its question and
  still fills `frame_quality.eyes_open`; what changed is that no list waits for it.
- **"Try to improve" no longer trades real pixels for invented ones in silence** (F169).
  The button of F149 scaled every frame to one ceiling before the model — `1024` px, a
  constant in `restore.py` — and the model is ×4. For a small frame that is a clean gain
  (800 px in, 3200 px out, the case the action was built for). For a phone shot it was a
  trade nobody was told about: **4032 × 3024 → 1024 × 768 → 4096 × 3072**, the same size
  out, through a quarter and back, with the real detail of the original dropped on the way
  in and plausible detail drawn in its place. The frame can come back looking sharper and
  holding less of what was there. Two things follow, and both are here. **The ceiling is a
  setting** — `features.restore_max_edge` — because it is the single number that decides
  what a person gets back, and a threshold in the code is a threshold nobody can move; the
  route now passes it, which it did not, so the constant really did decide for everybody.
  **A frame above it is told so**: the answer carries `rebuilt` with both numbers, the
  Review tab prints that beside "done" in all three languages, and a frame under the
  ceiling is handed to the model untouched and says nothing — because nothing was given
  up. What should ultimately happen to a 12-megapixel frame (tile it at native resolution,
  supersample back down, or close the action for that population and treat defocus as the
  different problem it is) is **not** decided here: `scripts/measure_restore.py` is the
  phase-0 measurement it will be decided from. It prints the three frame populations
  separately (< 1024 px, 1024–2500 px, > 2500 px), each against the original as the
  baseline, with size, weight, time, peak memory and the share of the frame's own pixels
  the model was even shown, at the current ceiling, at 2048 and at the full frame (a run
  that does not fit is a row, not a crash) — and lays out **blind** pairs for the eyes,
  both halves at the same size, the order seeded, the key in a file meant to be opened
  afterwards. It deliberately does not compute "better": the first probe of F149 used a
  model trained on clean bicubic downscaling and its numbers flattered a result a human
  eye then rejected, and "sharper" is not "truer".
- **«Not personal photos» was the wrong name for four different buckets** (F175). The
  slice showed **4 980** frames, which is exactly `product` 2 107 + `screenshot` 1 782 +
  `document` 1 015 + `meme` 76 — the classifier's whole output under one name — and that
  name was wrong three times over. **First, it collided with a different concept.**
  `files.not_personal` is a heuristic over the file NAME (`S01E05`, `1080p`, a rip group)
  that marks downloaded films: **three** files out of 38 485, computed by a different
  stage, filed into a different folder — and named almost identically. **Second, it was
  wrong on the facts.** A photograph of a shop receipt is personal, a screenshot of a
  conversation with your wife is personal, a passport more so; they are simply not
  photographs taken *for memory*, which is a different thing. Read as "not personal" the
  slice invites you to select everything and delete it, and 1 015 of the frames in it are
  documents that must not be deleted — a class that is private on top of that
  (`vlm.exclude_classes`, never rendered). The slice is **«Служебные кадры» / «Utility
  frames» / 「実用目的のコマ」** now. **Third, one caption cannot be honest about four
  different measurements.** Products are 78% precise at 81% recall (2026-08-03, 999
  frames), screenshots 59% at 83% (350 frames), and documents and memes have not been
  measured at all. So the caption of the whole slice names **no** percentage, each bucket
  states its own when it is the one open, and a bucket nobody has measured says **"not
  measured"** — the lookup falls back to it, so a class added later cannot quietly
  inherit a neighbour's number. **The documents are told apart before anything is
  selected:** the card carries the mark as a field of its own (`sensitive`, decided by
  the server from `vlm.exclude_classes`, not inferred by the browser from a missing
  thumbnail), gets a border and a "not for deletion" chip, and the note about what that
  bucket holds now sits **above** the button that selects the whole page. Nothing about
  the classification changed: the classes are neither merged nor split, `document` stays
  in `vlm.exclude_classes`, and `is_not_personal_video` is untouched — the guides simply
  say, in all three languages, that the flag and the slice are two different questions.
- **An action says where the frame will land, instead of «return to the photos»** (F174).
  Two of the marks the slices offer read as one movement to the person making it — "this
  frame does not belong here" — and the interface gave them two different names while
  saying nothing about what either one does. They are not the same movement. «This is not
  an animal» edits a MEMBERSHIP (`manual_pet`): the frame has been lying in its city folder
  the whole time, it is shown in the slice as a view over the canon, and the mark moves no
  file, ever. «Return to the photos» edits a route (`manual_overrides`): such a frame is
  **not** in the city layout at all, and returning it is a real transfer out of `_Products`
  on the next `sort --apply`. The button now reads the same in both slices, and what
  differs is stated **under** it: "goes into `Russia/Samara/2023`" there, "we take it out
  of the slice; the frame already lies in `Russia/Samara/2023`, the file will not move"
  here — which is the sentence that answers the fear the wording has to answer, that a mark
  deletes something. Where the frame lands is **not a guess**: the layout is a pure
  function of rows that are already in the database, so `sorter.destinations` answers for a
  page of cards with one query, through the same `_target_parts` and the same SELECT that
  `plan_and_sort` builds the plan from — and for the junk view it is asked with the
  correction **already assumed**, so the caption names the city the frame goes back to and
  not the service folder it is sitting in. A frame with no geodata is told so by name
  (`_Unsorted/no_place`, a third of the live collection) instead of being promised a city,
  and a country without a city is told the country level. A **bulk** return states the
  spread of the selection — "12 frames will return: 7 into cities, 5 into no_place" —
  because a person ticks dozens at a time and one folder name out of twelve deceives them;
  the count updates as the selection does, rather than appearing in the confirmation dialog
  where it is seen too late. Nothing is applied any earlier than before: the marks still
  pile up and land on `sort --apply`, the two tables stay two tables, and the layout rules
  are untouched. The test that matters is the one that pins the caption to the plan — the
  folder a card names is compared against what `plan_and_sort` builds for the same file,
  before and after the correction is written, so the two cannot drift apart.
- **Faces are not looked for where the classifier has already said "this is not a
  photograph"** (F165). The faces stage is the most expensive one there is — 30.9 minutes,
  46% of a full run — and it walked all **24 195** frames, among them 1 342 screenshots,
  682 documents and 76 memes, plus ~2 200 products once the deep tier has spoken: **up to
  4 300 frames, 18% of the collection**, at 77 ms each. The classifier knew about every one
  of them. It just used to run afterwards. So the stage is now **split by dependency**, not
  reordered: `classify` — the verdicts, which need nothing from faces — runs before them,
  and `junk` keeps everything that reads what the faces stage writes. The pipeline is
  `index → geo → landmarks → classify → faces → events → junk → phash`, and there is a
  `sorta classify` command for the front half.
  **Swapping the two stages instead would have been the silent kind of breakage**: `junk`
  reads the `faces` table in four places, and one of them is `frame_quality.face_sharpness`
  (F155), measured inside the boxes that stage writes. With the order flipped it would have
  stopped being computed on every first run — no error, no log line, just a NULL that means
  "not measured". Splitting keeps all four dependencies where they were.
  The economy is honest about its size and its edges: **~5% of the faces stage without the
  deep tier, up to 18% with it**, and it costs no new model and no new pass — only an
  order. A frame with **no verdict** is detected as before (NULL means "nobody asked", not
  "not a photograph"), so `sorta faces` on its own still walks the whole collection; a frame
  the deep tier later moves back to `photo` has no `faces` row yet and is picked up on the
  next run. What does change: before the faces stage has ever run there is no face to veto
  a CLIP verdict with (F13/F15), so on a **first** run with `--faces` a frame the classifier
  is confident about is no longer rescued by the face in it — which is exactly what a
  default run, where faces are opt-in, has always done.
- **Every ordered slice can be walked past its first page — search included** (F173). A
  query for «дети» came back with **exactly 200 frames** and no way further:
  `features.search_limit` was 200, and search had no paging at all, only a caption saying
  "200 frames". Animals, faces, screenshots and the Review each had a "show more" button;
  the one slice built by a QUERY rather than by a model's marks did not. That is the
  expensive one to miss, because the measurements of 2026-08-02/03 found exactly one
  confirmed lever of completeness — **the depth of the list**. Doubling it adds ~25 points
  on average, and on children it goes from **61% to 89%**: the second half of the ranking
  held nearly a third of all the frames the person was looking for, and the handle for it
  did not turn. `features.search_limit` is therefore **`features.search_page`** now — a
  ceiling cuts the answer off, a page only decides how much of it arrives first — and the
  old key keeps working, so an existing `config.yaml` loses nothing to the rename (it logs
  one line and reads the value). The counter says how many there are **in total** rather
  than how many are on screen: "showing 200" and "there are exactly 200" read identically,
  and for a ranking the second is almost never true. Beside the button there is one line
  about what depth costs — further down the list means more found **and** more missed —
  because that trade is measured and the person pressing the button is the one making it.
  There is no infinite scroll on purpose: a portion arrives when somebody asks for one.
  **The button is one mechanism now, not a fifth copy of one.** The server answers every
  paged slice through `ui._page_payload` (`items`/`total`/`offset`/`limit`/`has_more`, with
  `has_more` computed from the window actually served), the browser draws every one of them
  through a single `makePager`, and the caption is one catalog entry in the three languages
  instead of one per slice. The animal and face slices were moved onto it; the slices still
  ahead of this one in the queue (query slices, pinned queries, low resolution, blurred as
  a list) inherit the button by calling it rather than by copying it. The ranking order is
  untouched: this feature is about reaching the tail, not about how the tail is sorted.
- **The detector's threshold and depth are taken from the table it prints itself** (F162).
  Two defaults move and nothing else does: `features.detector_threshold` 0.5 → **0.6** and
  `features.detector_candidates` 2 000 → **4 000**. Both come off a re-measurement on
  **500** hand-labelled frames (36 animals), because the numbers F154 shipped were read on
  200, where fifteen animals made every one of them worth 6.7 points of recall. On the
  larger sample both figures moved by two dozen points — the detector at 0.50 is **78%
  precision / 69% recall**, not 62% / 87%, against **94% / 47%** for the CLIP label on the
  same frames. That spread is what a thin class does to a small sample, and it is the whole
  reason a row is chosen from a table and not in advance.
  **0.60 dominates 0.50 with nothing traded away**: the same 69% recall, 25 correct marks
  out of 29 instead of 25 out of 32. Three false ones go for free — a clean win rather than
  a compromise, which is why this one needs no decision about what a user prefers. 0.70
  buys no precision (86% again) and gives up two points of recall.
  The depth is a **ceiling**, not a threshold: it decides which frames the detector is
  shown at all, and an animal the query never ranked that high is not found at any
  confidence. Measured, 500 candidates reach 25% of the known animals, 1 000 reach 50%,
  2 000 reach 83% — so **17% of them were unreachable in principle** on the old default —
  and 4 000 reach **100%**. The price is named honestly: 4 000 frames is **5.6 minutes** at
  the measured 83.8 ms per frame, 2.8 minutes more than before, against the ~19 the animal
  stage already spends on the VLM and the 30.8 a pass over all 22 096 frames would cost.
  10 000 is pointless: the same ceiling for three times the time.
  Both tables are now written beside the values themselves — in `sorta/config.py`,
  `config.example.yaml` and the three user guides — so the numbers read as measured and
  paid for rather than picked. **`detect.enabled` does not move**: the detector stays off
  by default, and whether the cascade wants it is a decision for a live run, when the total
  time is on the table next to the few points that separate it from F158.

### Removed
- **The "is there a subject" question is gone, and so are the answers it collected**
  (F177). The frame-quality prompt asked two things in one call — are the eyes open, and
  does the frame have a clear subject. The second one ran for the first and only time on
  2026-08-03: **6,111 frames asked, 212 called subjectless**. Looked at by eye, those 212
  are ordinary photographs — street shots and studio work side by side — so the signal
  separates nothing, which is the same fate `is_accidental` met in F122 (measured at 5%
  precision and retired). Gone with it: the **"No subject" slice** of the Review
  workspace, its row in "Overview", and the **`no_subject` album kind** — deleted rather
  than hidden, because a hidden slice comes back at the first edit of the file that hides
  it, and `sorta album no_subject` is now refused as an unknown kind. **The stored
  answers are erased by a migration** (schema v26), and that is the part that had to be
  written rather than left to happen: nothing else would ever reach those rows — the
  question is out of the prompt so the stage cannot overwrite them, `vlm.quality` is off
  so the stage does not run at all, and a stale prompt fingerprint only means "recompute
  this row", never "the answer stored here is wrong". Without it the slice would keep
  listing 212 frames of a question nobody asks. **The eyes answers are NOT erased**: the
  migration touches one column. Editing the prompt does make every quality answer formally
  stale — they were given under a different wording — and the fingerprint will have them
  re-asked if quality is ever switched on again; but dropping the second question does not
  change the answer to the first, and those 6,083 answers (135 of them "eyes closed") are
  the only ones a person has checked by eye. The column `frame_quality.has_subject` stays
  and stays NULL, exactly as `is_accidental` does: NULL already means "not asked", and
  dropping a column in SQLite costs a table rebuild. The **blurred** slice is untouched
  and is not a VLM question at all (a Laplacian in the cheap tier, plus the sharpness
  inside the face box from F155), and the **closed eyes** question keeps working — its own
  cost is a separate conversation.
- **The cloud naming provider is gone, and with it the only code that could send a
  photograph anywhere** (F170). `naming.provider: claude` named events by uploading a
  few sample frames of each one to a vendor API; it was opt-in and off by default, and
  that was still the wrong shape for this product. What can be said about it changes:
  not "your photos do not leave your machine **unless** you turn one key on" but **"the
  product contains no code that sends images out"** — a sentence a reader checks with
  `grep` over the package instead of taking on trust. The key was also a trap by name:
  `naming.provider` reads as a choice of who invents the folder names, and nothing in it
  says that choosing that value uploads the archive. Three providers remain and all three
  run on the user's own hardware — `template` (the default), `vlm` (the local Qwen2.5-VL
  the deep tier already loads) and `local_vlm` (an ollama endpoint the user runs). The
  advantage that was deleted was never measured: nobody had compared the cloud model's
  event names against the local one, so this trades an unverified benefit for a checkable
  guarantee — the same trade this project already made for StreetCLIP, query translation
  and the phrasing ensemble. The honest cost is stated rather than argued away: **a
  machine with no GPU is now left with template names**, because the local model will
  crawl there. That is the loss of an ornament, not of a function — `template` is the
  default mode and the base of the product, and Qwen 3B does run on a CPU, slowly enough
  to matter for thousands of calls and not for the few hundred that naming events takes.
  An existing `config.yaml` that still selects the removed provider **keeps working**:
  the run logs one line saying the provider was removed, names the three that are
  available, and falls back to `template` — people upgrade with working files, and a
  removal must not kill somebody's run on its first line. `geo.provider: online`
  (Nominatim) is untouched and is a different conversation: it sends **coordinates**, not
  pictures, its name says what it is, and the guides describe it.

### Fixed
- **The place picker answers while the name is typed** (F201). The owner, assigning a place
  to an event on 2026-08-05: «при вводе места комбобокс ничего не открывает, не
  показывает». The field looked up a FULL name — right for `--where city=`, where the name
  is typed whole, and wrong for a combobox: «Моск» found 0, «Москв» found 0, «Москва» found
  1, and on the way «Мо» found a town in Norway, so for a whole word the only thing on
  screen was «такого места нет в базе — проверьте написание», sending the person to correct
  a word that was never misspelled. The search now matches the **beginning of a word** —
  the start of the name or any word inside a composite one, so «Новг» finds Нижний
  Новгород and a query may cross the space or hyphen it was split on («Нижний Новг»,
  «Ростов-на-До»). A start, not a substring: matching anywhere would answer «Рим» with
  every «Дурим» in a base of 150 000 settlements. The answer is **ordered** — the exact
  name first, then by population, then alphabetically — and **cut to twelve lines**, with
  the country still on top (a wrong country is visible in the plan at a glance, a wrong
  city is not) and never more than four of them, so the countries cannot crowd the cities
  out. Two letters is the floor for searching at all: one letter is not a request, and a
  name can be «Мо» or «東京». The empty list finally says **which** empty it is — the
  server reports whether it searched, so the field asks for more letters while the name is
  unfinished and only says "no such place" about something it actually looked for. Cost,
  measured on the bundled base (63 034 cities): a linear pass over all three languages was
  10–36 ms per keystroke, so the words are indexed once per language and bisected — a
  whole answer is 0.2–2.5 ms for six letters and 0.3–10 ms for three. No new dependency, no
  network, and nothing written: the picker reads the bundled base and only the bundled base.
- **"Try to improve" on a big frame refuses instead of working for nothing** (F198). The
  owner pressed it on a **4320 px** frame on 2026-08-05: the copy was made, and *then* the
  warning arrived — "the frame is larger than the limit (1024 px, this one is 4320), the
  copy was rebuilt from a reduced frame… this is not an improved original". The result was
  the original again, which is exactly what the measurement predicted: **35/35/30 on blind
  pairs above the ceiling — nothing**. Two entrances had drifted apart. The offer on the
  expanded frame **withdrew** the button above `features.restore_max_edge` (F168), and the
  route **did the work anyway** and said so afterwards (F169) — deliberately, because F169
  was written *before* the measurement and left the question to it. The verdict arrived on
  2026-08-04 and never came back to the code; the owner found it a day later. So the cost
  was not an impression but a **file**: a model run spent, a near-duplicate lying beside
  the original, a row in the index — for a warning that could only be read once all three
  existed. Above the ceiling the route now **refuses**, with the same reason code every
  other refusal travels as (`too_large`) and the wording the warning already had: the
  limit, the size of *this* frame, and `features.restore_max_edge`, so a person who
  disagrees with the threshold can see that the threshold is theirs. Both entrances read
  **one answer** now — the offer shows what the route enforces, so they cannot disagree
  again. **Below the ceiling nothing is narrowed** (62% against 10% for plain bicubic on
  small frames), the ceiling itself does not move, and the copies already made are **left
  alone**: deleting them is a decision a person makes, not a side effect of a fix.
- **A slice shows its frames as a grid — every slice, by one rule** (F195). The owner,
  reading «Животные · по запросу» on 2026-08-04: «все фотки растянуты — то есть одно фото
  на весь ряд». The panel of the pinned queries was the one grid nobody had written an
  `#id` rule for, so its cards fell out of the grid into block flow and stood **one to a
  row**, each stretched across the panel — while the four grids that happened to have a
  rule looked perfectly right, which is why the defect survived four features. The layout
  is now **one class on the container and one on the tile** (`.slice-grid` / `.slice-card`)
  worn by every panel that draws frames — the search line, the pinned queries, the three
  face slices, the animals, the class buckets and the «Просмотр» workspace — so how many
  frames stand in a row is decided by the **width of the panel** and never by which slice
  is open, and the next slice gets the layout by existing rather than by someone
  remembering to add a rule for it. What a slice keeps is what its cards **mean** (a
  struck-through animal, a bucket that must not be deleted), never the box they are drawn
  in. Two smaller things travel with it: the tile states `object-fit: cover` beside its
  sizes, so a card cannot scale a photograph along one axis — the other reading of
  "stretched" — and "loading" / "nothing found" / "it failed" now **span the row** instead
  of being squeezed into one 150 px column of a grid they are not a tile of. The tests
  walk the slices the interface actually draws instead of naming two of them.
- **"Try to improve" no longer leaves a file the index has never heard of** (F185). The
  copy was written under its final name and the row inserted by a SEPARATE call, so an
  insert that did not happen left a `_restored` file lying in the archive with nothing
  behind it — and the next `index` run reads such a file as a NEW photograph, so the
  collection grows a near-duplicate the person never made. Found on the live archive as
  **81 `_restored` files and zero rows**. The copy is now written to a staging name
  beside its destination, the row is written while it is still called that, and only then
  is it renamed into place — a rename inside one directory is atomic, and every other way
  out of that sequence takes the staging file with it (`restore.restore_and_record`).
  The click that found it also produced the second half: a **busy index is a reason now,
  not a stack trace** (`ERROR_DATABASE_BUSY`). SQLite allows one writer, an `index` stage
  can be running from the terminal, and the busy-guard (F145) only knows about runs
  started from the interface — so pressing the button during a terminal run raised
  `sqlite3.OperationalError` out of a request handler. It joins the three codes the
  interface already translates, and it is `_BUSY` rather than `_FAILED` on purpose:
  nothing is broken, the same press works a minute later, and the interface reads that
  difference off the name to decide whether offering "try again" is honest. No retrying
  happens inside `restore` — waiting for the index there would hold a connection and a
  thread for as long as a stage takes.
- **The run estimate is computed from measurements instead of baked-in constants**
  (F159). The run screen priced the "best frame of a group" stage at **1.32 s per group,
  whatever the group held**. Measured on 2026-08-03, one comparative question costs
  **0.45 s plus 1.03 s for every frame in the prompt** — 1.47 s for a pair, 2.45 for a
  triple, 3.46 for four, 4.56 for five. The 1.32 was the price of a PAIR, quoted for
  groups "of up to five", and on the reference collection it estimated the stage at half
  a minute against a measured **1.9 — a 3.7x understatement**, which is 90 invisible
  seconds on a small archive and hours on a 300 GB one full of groups of nine, ten and
  eleven. The line is now summed over the ACTUAL group sizes, capped at the frames
  `dedup.keeper_max_frames` really sends, and both numbers moved into a config section of
  their own (`estimate.keeper_call_sec` / `estimate.keeper_frame_sec`) with the
  measurement table in the comment above them.
  **The deeper fix is that the screen stopped carrying constants at all.** Since F147 the
  run log holds `stage=… elapsed=… processed=…` for every stage and phase of every run —
  the real speed of THIS machine — and the estimate now reads its rates from there
  (`runlog.read_measurements`): the four always-on stages behind the base line, faces,
  events, and the model questions priced off the VLM phase of the stage that asks them —
  `classify` for the deep tier and `junk` for the quality and animal ones, which F165
  split apart and which both call that phase `junk_vlm`. A constant is
  what is used when the log has nothing to say, and the screen **states which of the two
  it used** — "the numbers come from your own last run (2026-08-03)" against "a default
  estimate: this machine has no measurements of its own yet" — because somebody deciding
  whether to start a four-hour run is entitled to know whose four hours the estimate is
  describing. A timing is discarded rather than trusted when the version that wrote it is
  not the version asking (the `frame_quality.source` device: a stored answer is kept only
  while the question behind it is the same one), when no environment header vouches for
  it, or when it is older than `estimate.measurement_max_age_days`. Half a line measured
  is not a measured line: the base line covers four stages and falls back whole.
  **The same measurement also retired the premise the option was sold on.** From three
  frames up, one question over a group is **not** cheaper than asking about the frames one
  at a time (2.45 s against 2.31, 4.56 against 3.85) — the saving was only ever real for
  pairs, and `keeper_min_group_size: 3` stopped asking about those. The stage stays,
  because it answers something separate questions do not — which of the frames is the best
  one — but no caption calls it the cheap way to ask any more; they say the time grows
  with the group. The stage's own behaviour is untouched: this is a feature about
  estimating the time, not about spending less of it.
- **The detector's answer reaches the screen — and the rule is checked against itself**
  (F160). F154 shipped the object detector: it ranked the candidates, ran the model, stored
  `detections` and wrote `frame_quality.pet`. What it could not reach is the place the
  verdict is actually READ: since F137 the album, the "Animals" tab and the Overview
  counter derive it when they read, through `sorter.animal_auto_sql`, and that expression
  knew nothing about the new table — so a run spent three minutes, the boxes went into the
  database, and **not one number a user looks at moved**. The tier is now in that
  expression, with the same order of precedence it has in the stage (the user over
  everything, then the F130 answer — a box detector calls a drawn cat a cat — then the
  detector, then the CLIP score), and it reads the stored BOXES rather than the label
  column, so `features.detector_threshold` can be re-chosen in either direction without a
  new pass over a single image, exactly as F137 made the other two thresholds. With
  `detect.enabled` or `features.detector` off the expression is byte-for-byte the one that
  shipped before, boxes left behind by an earlier run included.
  **The real cause was wider than the missing branch.** "What counts as an animal" is
  written twice on purpose — `junk.pet_label` labels the one frame a stage has just scored,
  `animal_auto_sql` answers "which files" over a whole index — and by now four things
  decide it (the CLIP score F122, the VLM answer F130, the user F124, the detector F154).
  Every one of them had to be written into both halves and **nothing checked that it was**.
  So the case table — every combination of score, answer, detection and manual mark, at
  five sets of thresholds — is now run through **both** spellings and asserted equal row by
  row. A fifth source that lands in only one of them fails a test instead of a slice.
  The caption of the "Animals" tab states both measurements, because the switch chooses
  between two different promises: 82% precision at 64% recall for the cascade, 62% at 87%
  with the detector on. The default does not move — `detect.enabled` stays off, and trading
  20 points of precision for 23 of recall is the user's decision, taken with the numbers of
  a run in hand.
- **One city, one name — and one alphabet per language** (F172). A `language: ru` layout
  came out in three alphabets at once: «Санкт-Петербург», «Москва» and «Серик» next to
  `Nizhny Novgorod` (382 files), `Samara` (179) and `Ryazan` (109), with a Thai village in
  its own script between them. Worse, one city was filed twice — «Сочи» (385 frames) and
  `Sochi` (29) share a geonameid and are the same place. The bundled data was never the
  problem: `names.tsv` holds a Russian name for every one of those cities. The NAME had two
  sources that did not agree about the language. The bundled base was asked for the English
  anchor (`geo._CANONICAL_LANG`), the online provider answered in the language of the
  request — so whether a city came out Russian depended on whether Nominatim happened to
  name a suburb for it: the answers that stopped at the region were completed from the
  offline base, in English, and landed beside their own Russian twins. The rule is now
  written down once, in `geo._place_name`, and every source goes through it: `language` →
  `en` → the native name the source knew (the asciiname of `places.tsv` offline, the
  provider's own text online). A geonameid outranks text, so two files of one city cannot
  be named differently again; where there is no Russian name, the English one is used
  rather than an invented transliteration, and a place with no alternatives at all keeps
  its own script. Nothing about WHERE a file goes changed — only what the place is called.
  No schema change: the geonameid is still written next to the name, so the sorter keeps
  choosing the folder language when it READS the row (F99) and a change of `language`
  still costs no geo run at all.
- **"There is an animal here" is computed when read, not frozen when written** (F137).
  The thresholds of the animal cascade are deliberately **not** part of the prompt
  fingerprint — the scores and the model's answers are stored precisely so a threshold can
  be re-chosen without another pass over the collection, and hashing them would send
  sharpness, CLIP and the VLM round again on every edit. But the label itself was read off
  `frame_quality.pet`, i.e. off the config of whichever run last wrote it, so an edited
  threshold changed nothing anybody could see: the live collection kept **966** animals
  selected at a 0.30 candidate gate long after the gate went back to 0.50, where the stored
  answers say **848**, and nothing in the interface connected the number to the setting.
  The verdict is now derived at the moment of the read, from `pet_score`, `pet_vlm` and the
  thresholds in force — the check's stored answer where the current
  `features.pet_candidate_threshold` would still ask for one (the same replay F130 chose
  that threshold with), `pet_score >= features.pet_threshold` otherwise, and the user's own
  `manual_pet` verdict over both, unchanged (F124). Editing either threshold now moves the
  album, the "Animals" tab and the "Overview" counter **at once and without a run**.
  `frame_quality.pet` is still written and is still the column to query a database by, but
  no consumer decides by it — one of them doing so would reopen the same gap somewhere
  else. No schema change, no migration and no one-off recount: a stale label misleads
  nobody once nothing reads it.
- **No option raises the model by itself** (F145). A base run started **without** deep
  analysis called the VLM from the landmarks stage. The cause was not one setting: four of
  the five VLM questions of the pipeline gated on their own key alone — `vlm.quality`,
  `features.pets_verify`, `dedup.keeper_vlm`, `features.junk_rescue` — and the fifth,
  `features.landmarks_verify`, had no condition whatsoever, so any key left true in
  config.yaml from an earlier experiment raised 20 GB of weights on a run nobody asked to
  pay for them. It is a consequence of the project's own rule "a new feature gets a new
  toggle": F113, F130, F131 and F132 each honestly added one, and **nobody added the
  hierarchy** — it was assumed to exist and never checked. `vlm.enabled` is now the
  precondition for every one of them (`config.vlm_allowed`, read BEFORE any factory is
  called, so the weights are not merely unused but never loaded), and with it off each
  stage produces exactly what it produces with its own key off — the landmarks stage
  resolves the same set of places it resolved before F131 existed, down to the gate its
  proposals are collected at. Nothing in config.yaml is rewritten: a switched-off model is
  a state of the run, not a reason to erase somebody's settings.
- **The web app no longer offers what the run will not do** (F145). The three subordinate
  options on the run screen (the animal check, the frame-quality questions, the best frame
  of a group) go **dead rather than hidden** with the deep-analysis checkbox clear — a
  vanished option reads as "there is no such feature" — and their price on the run budget
  becomes zero rather than the old number, so the estimate adds up to what will actually
  happen. Nothing switches itself on in either direction.
- **Nothing that changes data is clickable while a run is in flight** (F145). The server
  refused these with 409 on eight routes and did not on fourteen others that write the
  index, the config file or files on disk — a race between two writers over
  `media_class`, `frame_quality` and `places`, which the pipeline rewrites wholesale.
  Every POST route is now either guarded or listed as deliberately exempt (only the three
  cancels and the folder picker), and the suite fails on a new one that is in neither set.
  In the interface the settings column, the duplicate-choice save, the review marks, the
  trash, "back to photos", album gathering and the layout are all disabled for the
  duration, each with a line saying why, and all come back on their own when the run ends
  — no page reload.
- **The overview block holds its height** (F145). While the index was empty it drew a stub
  with a button and swapped it for the full set of counters the moment the index stopped
  being empty — which is in the middle of a run, right after the `index` stage: everything
  below it, the run options among them, jumped down the page. It now draws the same rows
  with dashes from the first paint, so arriving numbers change the text and not the
  layout, and the empty state doubles as a statement of what a run will produce.

### Added
- **Pin a query of your own as a slice** (F156). The measurement cancelled ten features
  before it added one. On a random sample of 200 frames, **65 — a third — fall into no
  class at all**, so rather than inventing slices for them, each candidate was asked of the
  collection as a query and scored by how much of those 65 it actually reached: nature 26%,
  city 23%, plants 22%, sky 20%, signage 18%, sea 17%, transport 15%, interiors 12%, food
  12%, celebrations 6%. **Not one covers a third of a third**, and **food came out at 8
  frames** — smaller than sky or signage — though both the user and the author held it in
  mind as a large slice. Ten slices for 65 frames out of 200 would rebuild the
  thirteen-control remote F133 took apart. So the product **stops guessing which facets
  matter**: for one person they are mountains and children, for another receipts and cars,
  and the list belongs to the owner of the archive. Search for something, and when there
  are results **“Pin as a slice”** appears beside the line, asks for a name (the query
  itself is offered) and the query becomes a pin **indistinguishable from a built-in one**
  — the same grid, the same **Collect into folder** with link / copy / move
  (`move_batches.mode='album_query'`), the same **Show more**. The mechanism is untouched:
  F129 ranks, F151 pins, and all that is new is who writes the list. Four decisions carry
  it. **The pins live in `config.yaml`** (`features.saved_slices`, beside the three that
  ship) and **not in the index** — `sorta reset` and every re-processing rebuild the index,
  and a slice somebody named must not be one re-index away from gone; the file is written
  line by line, so the comments and settings around the key survive. **Unpinning** sits
  inside the slice, asks first, and removes a config entry and **not one photograph**.
  **The order** is the file's order and is changed with arrows on the panel — one way to
  reorder, not two. **The number of pins is bounded** (`features.max_pinned_slices`, 12) —
  F133's reason and not a resource one, since a pin costs a line and one matmul — and
  reaching the bound is **said out loud** rather than being a button that quietly does
  nothing. Two things are stated at the moment of pinning rather than afterwards: a pinned
  query is an **estimate** with the same numbers as every other query slice (~60% of what
  you want in the first portion, ~90% in a doubled one, precision falling with depth —
  «mountains» will bring hills and clouds), and **the wording goes to the model as it
  stands**, so a Russian or Japanese pin will rank badly until the multilingual index is
  built. There are **no suggestions to pin anything**, ever: a product that proposes the
  facets is the product this replaces.
- **An empty slice now says WHICH empty it is** (F156). The same feature settled that the
  standard slices — people, events, animals, duplicates — **do not unpin and are never
  hidden by hand**: they are the core of the product and there is already one control for
  them, the checkboxes on the run screen (do not need animals → do not compute animals →
  the slice is not there). A second way to take the same thing off the screen would
  inevitably disagree with the first, leaving a person to guess whether a slice is missing
  because they hid it or because they never computed it. What such a slice **does** owe is
  an account of its own emptiness, and it now gives one: **“this was not computed”** with a
  link straight to the run screen when the stage has never run (`features.pets: false`
  included — that is where the switch is), and the slice's own **“none were found”** when
  it has. `GET /api/tabs/visibility` carries the two as `reasons`, and a slice whose stage
  has not run is **shown** rather than hidden, the F152 rule — a pin that hides itself never
  gets to say why it is empty. This is the `frame_quality` principle of F125 applied to a
  whole slice: **NULL means “nobody asked”, not “there are none”**, and a bare zero reads
  as the second when it is nearly always the first.
- **A slice is a saved query, not a sixth filter with a threshold of its own** (F151).
  Measured on 2026-08-02 — 200 frames out of 22 096, labelled by hand, and the first time
  **recall** was measured rather than the precision of the top — the hand-written filters
  find **6%** of the blurred frames, **33%** of the animals, **0%** of the products, and
  there is no filter at all for children (an estimated 4 860 photographs) or people. The
  **same vectors**, asked in words, give **61%** for children, **65%** for products and
  **60%** for animals at the same depth, and **89% / 95% / 87%** at twice it. The price is
  zero: the vectors are the ones the junk stage already stores (F128/F141) and a query is
  one matrix multiplication (0.9 ms measured). So the **Slices** tab now pins three of them
  — **Children**, **Products**, **Animals · by query** — beside people and events, out of
  `features.saved_slices`: **a name and a list of phrases in the config file**, so a slice
  is retuned, added or deleted without code and without a release. The number of phrases is
  **not** why it works (one, three and six differ by less than the noise of the sample); the
  list is there to be edited, and the panel prints it so there is something to edit. Three
  decisions are the feature. **These lists are estimates and are captioned apart from the
  exact ones**: the `pets` label stays exactly as it was — 71% precision, checked by a model,
  answering "is this confidently an animal" — and stands next to the query slice under a
  different name, because with one label a reader takes an estimate for a fact. **There is
  no membership threshold and there will not be one**: the list is ranked, and where it
  stops being about the query is a judgement the person reading it makes. **Depth is the
  lever**, the only one the measurement confirmed, so "show more" is the primary button of
  this panel rather than a ghost at the bottom, and it continues the same ranking
  `features.search_page` frames at a time — no repeats, nothing skipped. Two populations are
  deliberately left out: **people**, because `faces` answers that exactly and for free (7 341
  frames against an estimate of 6 080 — F152 already draws those slices), and **blurred**,
  where the sharpness filter is 100% precise on that sample against 36% for the query and
  merging the two would sink the exact signal into the approximate one. An index that cannot
  rank says which of its states it is in, exactly as the search line does — a pinned slice
  must never come back as an empty list, because nobody typed anything to be wrong about.
- **A "low resolution" slice — the one signal in the product that needs no measurement**
  (F150). Measured on 22 095 photographs (2026-08-02): **706** frames hold fewer than a
  megapixel, and **682 of them are formally sharp** — the intersection with the blur
  window is **3%**, so until now no filter in the program showed them at all. The shape of
  the distribution says where they came from: 94 frames under 0.2 MP, 133 between 0.2 and
  0.5, 479 between 0.5 and 1, then a gap and 17 493 above 12 MP. Phones do not take
  pictures of that size; this is what arrived through a messenger or came off a download.
  It is the **fifth slice of the "Review" workspace**, beside duplicates, blurred frames,
  closed eyes — one job, one place, one decision per file — and not a sixth tab, because a
  tab for one signal would split the place where problem frames are sorted out. It is not
  folded into "Blurred" either: with a 3% overlap the two populations would hide inside
  each other. **Detection here is exact by construction.** `files.width * files.height` is
  a fact the indexer wrote down, not a model's estimate, so this slice has **no threshold
  chosen by taste, no labelling to check its precision against and no recall to measure**:
  it does not find frames that look small, it enumerates the frames that are small. The
  list is ordered by ascending pixels (the most damaged first — a ranking, like sharpness,
  never a verdict) below `features.low_resolution_mp` (**1.0** by default), each card
  prints the size the thumbnail cannot show (`1280×960 (1.2 MP)`), and the slice carries
  the same actions as its neighbours: keep, mark for deletion, gather into a folder
  (`sorta album low_resolution`), and **"Try to improve"** — for which this is the real
  addressee in the whole product, since Swin2SR ×4 is a super-resolution model and 640×480
  becomes 2560×1920, i.e. it does here exactly what it was trained for instead of working
  against its purpose on blur. **Small is not faulty**, and the hint says so: such a frame
  can be the only surviving photograph of somebody, so the slice is named after the fact
  and nothing in it is marked for deletion by default. Two limits are stated there too —
  megapixels say nothing about a large frame ruined by compression (a 4000×3000 picture
  full of JPEG artefacts is a different signal), and videos are not counted, their
  resolution having its own meaning. A counter in "Overview" like every other slice.
- **The run log says what is happening now, not what happened** (F166). F147 gave the junk
  stage a breakdown by phase, and the live run of 2026-08-03 showed what was still missing:
  all four phase lines carried the **same timestamp**, printed in one batch just before the
  stage summary, because `log_phase` was called for every phase at once at the end. An
  instrument that answers "where did the time go" at the moment the time has gone answers
  nothing about a two-hour stage — and if the run is cut short it answers **nothing at
  all**: the orchestrator interrupted `junk` mid-run and lost the numbers of three phases
  that had finished long before, along with everything the F159 estimate would have learned
  from them. Every timed unit of a run — the stage, and each phase inside it — now writes
  the same three kinds of line: `stage=<s>[ phase=<p>] started[ total=<n>]`, a periodic
  `... progress elapsed=<sec> processed=<n>[ total=<n>] rate=<n>/s`, and its summary **the
  moment that unit is over** rather than at the end of the stage. An interrupted or failed
  run therefore keeps every phase that finished, and marks the one that did not as
  `interrupted (...)`/`failed` with its seconds instead of silently dropping it. The
  periodic line is a heartbeat and not a channel of its own: one line per
  `logging.progress_interval_sec` (default **60 s**, `0` switches it off), through the same
  logger and at the same level as the summaries, and silent while a phase of that stage is
  open — the phase line already carries the same counters with a name on top of them. Those
  counters are the very pair `/api/process/status` serves, read from the same call that
  moves the progress bar, so the file and the interface cannot come to disagree about where
  a run is — F147's rule for the phase names, applied to the numbers next to them. Stages
  with no phases of their own (`index`, `geo`, `landmarks`, `faces`, `events`, `phash`) are
  no less readable for it. The shape of the F147 summary line is untouched: `stage=` and
  `elapsed=` are where they were, and every grep and estimate built on them keeps working.
- **The junk stage says where its seconds went** (F147). On the run of 2026-08-02 the
  stage took **2 070 seconds** — more than half of the whole hour — and the log held one
  line about it: `stage=junk elapsed=2070.208`. Six different things work inside it (CLIP
  over every frame, OCR behind the gate, the laplacian, the quality model, the animal
  cascade, the stored vectors), and which of them ate the 34 minutes was a guess — so
  every lever from the backlog (narrow the quality scope, raise the rescue threshold,
  change the batch size) was a guess too. Each phase that runs now writes its own
  `stage=junk phase=<name> elapsed=<sec> processed=<n> rate=<n>/s` line, in the same
  machine-readable shape, through the same logger and at the same level as the stage
  summary it breaks down, so the breakdown exists at the settings a long run is actually
  started with rather than only under DEBUG. The unit count is half the point: eighteen
  minutes over 1 362 model calls and eighteen minutes over 22 096 frames read the same
  without it. The phase names are the ones F100 already gave the progress bar
  (`junk_clip`, `junk_ocr`, `junk_vlm`, `junk_write`) — one name for the caption and for
  the stopwatch, so the two cannot drift — and a phase that did not run writes **nothing**
  rather than a zero, because `elapsed=0` reads as "it happened instantly". Measurement
  only: not one verdict, threshold or call count moves, which is the whole reason to build
  the instrument before pulling any lever with it.
- **Sharpness measured inside the face** (F155): a new column `frame_quality.face_sharpness`
  (schema v24) — the same laplacian the stage already computes, taken over the face boxes of
  the frame instead of over the whole of it. **The blur filter needed it because its recall
  was 6%**: on a hand-checked sample of 200 frames it caught 2 of the 33 blurred ones, and
  the reason is not the threshold. The variance of the laplacian over a whole frame answers
  "how much detail is in this picture", which is a different question from "is it in focus"
  — a detailed sharp street and a smooth blurred face give the same number, and blurred
  frames sit in every band up to 400. A face is the one object comparable across frames, so
  the same measure taken inside it means the same thing twice: on the 68 frames of that
  sample that have a face (13 of them blurred), **62% recall at a threshold of 200 against
  15% for the whole frame at 300**, for a comparable number of frames flagged. It costs no
  new pass — the preview is decoded once and the second variance comes off a crop of the
  same array — and **the coordinates are rescaled** from the original into preview pixels,
  which is the one thing that had to be right: `faces.bbox` is written in pixels of the full
  frame, and the measurement this feature came from used them as written, lost 39 of its 68
  crops off the edge and reported 100% recall over the survivors instead of the real 62%. A
  broken crop flatters the result rather than failing, so the suite pins the rescaling
  directly. Several faces give the sharpest one (if any face is in focus, the shot worked);
  the `bbox = '[]'` marker is not a face; a crop too small to measure is NULL, not zero.
  **It ranks and does not judge**: precision is ~25% at every threshold measured, and only
  the third of a collection that has a face is covered at all, so `features.face_sharpness_max`
  (`200.0`, provisional — `python scripts/measure_frame_quality.py --features sharpness`
  prints the sweep) orders the blur list and nothing in the pipeline acts on it.
- **An object detector as a second tier over a query, not a pass** (F154): the animal label
  gets a third tier (`features.detector`, off by default, behind its own master switch
  `detect.enabled`) — a COCO detector from torchvision, run over the candidates a zero-shot
  query picks out of the CLIP vectors the junk stage already stores. **It exists for one
  measured slice out of three.** On 200 hand-labelled frames at confidence 0.5 the detector
  is 62% precision / 87% recall on animals against the 71% / 33% of the label the pipeline
  writes today — it sees the cat in the corner of the frame, where CLIP compares the
  picture to a text as a whole — but on people it is 42% precision against the ~100% of the
  face boxes (F152), and on food 20% / 15%, because COCO has no `food` class at all (a
  banana, a pizza, a sandwich). So people and food are **not** detected here, and the class
  list says why where it draws the line. The shape follows from the price: a pass over the
  collection is 83.8 ms × 22 096 = **30.8 minutes**, while the ~2 000 candidates of the
  query (`features.detector_candidates`) are **~3 minutes** — the same cascade F130 and
  F140 already pay for, with recall bounded by the query's own at that depth and precision
  raised from 43% to 62%. The answer overrides the CLIP label in **both** directions, and
  every refusal — no stored vectors, no weights, an error on one frame, an answer the F130
  check already gave (a box detector calls a drawn cat a cat) — falls back to the label
  that was there, never to "no animal". What was found is stored with its class, confidence
  and box (`detections`, schema v23), which is also the incrementality marker: a row exists
  for a frame the detector found **nothing** on, so a repeated run asks about nothing at
  all. The threshold was not chosen in a brief: `scripts/measure_detector.py` prints
  precision and recall at 0.3 / 0.5 / 0.7 and the recall ceiling per candidate depth, next
  to what the current CLIP label scores on the same frames — the F130 lesson, where the
  0.30 a brief proposed turned out to be the worst row of its table. **No new dependency:**
  torchvision is installed with the CLIP side already and only the COCO weights are
  downloaded, and there is no configuration in which the detector runs over everything.
  One thing this does **not** yet reach, stated rather than hidden: since F137 the album,
  the "Animals" tab and the Overview counter derive the verdict when they read, out of
  `pet_score`/`pet_vlm` through `sorter.animal_auto_sql`, and that expression does not know
  about `detections` yet — so the detector fills its table and writes `frame_quality.pet`,
  while the slice a user sees is still the F130 answer until the read-time rule learns the
  new tier (one branch, in a file this feature does not own).
- **"Try to improve" — one frame, by request, a processed copy beside it** (F149). A third
  action in the **Review** tab next to "mark for deletion" and "keep": press it on ONE frame
  you opened and chose, and a model-processed copy appears **as a second card beside the
  original**, marked as processed, with the same actions on it — so the comparison is two
  pictures next to each other rather than a message saying a file was saved. Keep either,
  both or neither; choosing the copy marks nothing about the original, which is the same
  line between advice and action F148 drew. The model is `features.restore_model`
  (`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`, ~400 MB of weights and ~1 second per
  frame), loaded on the **first press** and never at startup, and it is `realworld` and not
  `classical` because that is what the measurement said: `swin2SR-classical` is trained on
  clean bicubic downscaling — a degradation a real archive does not contain — and lost to a
  plain unsharp mask, while `realworld-sr-x4` beat the mask outright.
  **The model does not bring back what was lost, it draws something plausible**, and for an
  archive that is more dangerous than the blur: a smeared frame is honestly smeared, a
  redrawn face looks real and is not. So the original is never touched (byte for byte, and
  that is the first test of the feature), the copy says what it is in its name
  (`<name>_restored.jpg`, beside the original, never over an existing file) and on its card,
  and **there is no bulk anything** — no stage, no CLI command, and a route that takes a
  single `file_id` and refuses a list. The input is scaled to 1024 px first (x4 over a full
  4000 px frame is 16000 px and a memory failure), and when there is no copy there is a
  **reason** instead — the weights come off the network, and being offline is an ordinary
  state for this program.
  The copy is an **ordinary member of the collection**: it gets its own `files` row, so it
  goes into the layout, the slices and albums, and it inherits the capture facts of its
  source rather than being re-read off a re-encoded JPEG (which would date it today and
  file it under this year instead of the year in the picture). Its link to the original is
  **stored** (`restored_files`, schema v23) and not guessed from a name, which is what keeps
  the pair from ever coming back as a duplicate to sort out: `dedup.near_duplicate_groups`
  leaves derived files out of its groups, so nobody spends the next run deciding about pairs
  they created themselves. Pressing the button twice returns the copy that exists.
  One thing this did **not** reach when it shipped, stated rather than hidden: since F137
  the album, the "Animals" tab and the Overview counter derive the verdict when they read,
  out of `pet_score`/`pet_vlm` through `sorter.animal_auto_sql`, and that expression did
  not know about `detections` — so the detector filled its table and wrote
  `frame_quality.pet` while the slice a user saw was still the F130 answer. F160 (above)
  closed that, and made the two spellings of the rule check each other.
- **Three slices over what the faces stage already found** (F152): **With people**, **Group
  photos** and **Portraits** in the Slices tab, with counters on the Overview and albums of
  their own (`sorta album people|group|portrait`, no selector — the collection holds exactly
  one of each). A hand-labelled sample of 200 frames put people in 27.5% of the archive and
  children in 22% — the largest populations there are, larger than animals, products and
  screenshots together — and not one slice pointed at them, while the signal had been on
  disk since phase 3: 12 952 detected faces over 7 341 photographs. Nothing here is a model,
  a pass or a guess: membership is a **fact** of the `faces` table, so the caption over the
  grid says exactly that and no card carries a confidence, because there is none to show.
  The two numbers the rules do need are geometric and live in `features:` —
  `group_photo_faces` (3, a count of face boxes) and `portrait_face_share` (0.08, the share
  of the frame one box covers, out of the box and `files.width/height`; a starting value
  stated as geometry rather than a measurement, and re-choosing it costs a query, not a
  pass). The one trap this feature is built around: a `faces` row with `bbox = '[]'` is not
  a face but the marker "processed, no faces here", and 24 195 of 24 196 live files carry
  one — a predicate that keeps it turns "with people" into "every photograph". It is
  excluded in exactly one place, `sorter.face_slice_ids_sql`, which the panel, the counters
  and the albums all read. Without a faces run the slices say **why** they are empty
  instead of showing a zero, on the pins, in the panel and on the Overview alike: nothing
  was measured, and "you have no photographs with people in them" would be a claim about
  somebody's own archive.
- **Any slice can be gathered into a folder, not only people and events** (F139). People,
  events, animals and a query had "Collect into folder"; products, screenshots and memes
  had only "back to photos", and blurred frames, closed eyes and "no subject" only the
  trash. That was never a decision — the album engine (`sorter.plan_album`, link/copy/move,
  `--name`) has been able to export any slice since F34, and the buttons simply grew where
  each slice happened to appear first. Six kinds join it: `product`, `screenshot` and
  `meme`, selected on `media_class.verdict`, and `blurred`, `eyes_closed` and `no_subject`,
  selected on `frame_quality` through the SAME expression that draws the "Review" workspace
  — the counter of a chip and the size of its album are now one number by construction, and
  the schema does not move. Three things this deliberately does not do. `blurred` gathers
  the **window** (`features.blur_review_max`), never the tail below it: past that ceiling
  sit thousands of frames nobody has looked at, and the whole point of the slice is that
  the decision is taken by eye. A class listed in `vlm.exclude_classes` (`document` by
  default) gets **no album**, refused at both ends — the payload the page draws its button
  from and the route itself, plus `plan_album` for the terminal — because a hidden button
  is not a rule; and `document` is not an album kind at all, so emptying that key lifts the
  preview rule and still does not assemble a folder of passports. And the destructive
  actions stay where they were, in their own block: one movement never both gathers and
  deletes. In the terminal the same six kinds take no selector —
  `sorta album product --dest …` is the whole command — and the folder they default to
  comes from the string catalog in all three languages.
- **Both indexes can answer one query** (F153): `features.search_fusion` — `off` | `rank` |
  `union`, **off by default**. Once the search index exists (F141) a photograph has two
  vectors, and the measurement of 2026-08-02 said something about them that neither number
  alone shows: over 217 hand-labelled judgements the classification model and the search
  model score the SAME at the top (88/96/98% at ranks 1/3/5) and return **different
  frames** — "it disagrees with xlm english on which photos, but both are good, even though
  they differ". Two rankings wrong in different places are the one case where merging beats
  either half, so a query can now be answered by both: `rank` weights a frame by its
  **place** in each list (reciprocal rank fusion — agreement between the models wins),
  `union` merges the two lists **as sets** (each frame keeps its best place, so what only
  one model found is not pushed out). What no mode does is add the scores up, and the
  merging function cannot: it is handed file ids and no numbers at all, because a cosine of
  `ViT-L-14` and a cosine of `xlm-roberta-base-ViT-B-32` are values of two different spaces
  that look comparable and are not — the mistake F128's `model` column exists to prevent.
  An index with nothing to rank does not silently halve the answer: the other one ranks and
  the fact is logged. The cost is one extra matmul over a stored table, ~1 ms of query
  time, and no pass over any image. **The default is off because the number that would
  choose otherwise does not exist yet**: precision cannot decide this — both models are at
  98% at top-5 — and what a merge is expected to raise is RECALL, which has never been
  measured for either of them. So the measurement is part of the feature rather than a
  promise about it: `scripts/measure_search.py --fusion` prints precision **and** recall at
  every depth for all four variants over one set of marks, states that its recall is
  relative to the labelled frames rather than to the collection, and says out loud when a
  variant's output is mostly unlabelled — a merge measured against a sample its competitor
  chose is not measured at all. Nothing else moved: the interface, the thresholds, the
  prompts and the schema are exactly as they were.
- **Search that answers Russian** (F141): `features.search_index`, off by default, with
  `features.search_model` (`xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k`) next to it and a
  new table `search_embeddings` (schema v22). Search by words was accurate in English and
  did not work in Russian, and 217 hand-labelled judgements over 8 concepts say how badly:
  22% precision at top-5 against 98% for the multilingual model, with four of the eight
  concepts — cake, food, mountains, children — returning **nothing at all**. That is not
  "worse", it is a feature that does not exist for half its users. The obvious fix, swapping
  `naming.clip.*`, is the one thing that must not happen: the landmark threshold 0.85 with
  corroboration (F75), the animal threshold 0.70 (F122), the cascade selection at 0.50
  (F130) and the junk classification are all calibrated on `ViT-L-14`'s numbers, and a swap
  invalidates every one of those measurements at once. So `ViT-L-14` stays the
  **classification** model and search gets a **second vector of its own**, written by a
  second CLIP pass over the same previews and the same population (canonical photographs,
  F120). The control that had to pass before that could be believed did: the smaller image
  tower is not weaker in English either — 95% against 98% at top-5, three points on forty
  judgements. The price is named in the config rather than hidden, which is why the toggle
  is off: **19 753 frames in 635 seconds** (~10.5 minutes) on the machine it was measured
  on, plus ~40 MB per 20 000 photographs. F128's rule is not weakened — every row carries
  the model that produced it, a mismatch means recompute and never use — and search reads
  the search index alone: with the toggle off it says there is nothing to rank (F134) rather
  than quietly falling back to vectors of another model, because a ranking produced by the
  wrong model looks exactly like a good one. Not one classification threshold moved.
  `scripts/measure_search.py` measures the search index it now reads.
- **The screenshots and receipts the classifier took for photographs** (F140):
  `features.junk_rescue`, off by default, with `features.junk_rescue_threshold` (0.02) next
  to it and a new column `frame_quality.junk_score` (schema v20). The search by words (F134)
  put memes and screenshots at the top of its results, and the table it searches turned out
  to be clean — all 19 753 rows carry the verdict `photo`. So nothing had leaked into the
  index: the junk stage is simply wrong about ~4% of what it calls a photograph, an error
  nothing made visible until a query did, and those ~800 frames go into the city layout, the
  duplicates, the quality signals and the albums like ordinary pictures. They are found by a
  zero-shot query over the vectors F128 already stores — `max(similarity to
  screenshot/meme/text/receipt) - max(similarity to a photograph)` — which costs **no pass
  over any image**: the vectors are on disk and the prompts are five short strings through
  the text tower. The whole feature is what is **not** done with that number. Reviewed by
  eye: above +0.05 the 93 frames are junk outright, but the band +0.02..+0.05 still holds
  ~17% real photographs, so applying the score as a verdict at ~85% precision would take
  ~150 living pictures out of the layout — exactly the mistake F130 measured for animals,
  where a signal of that accuracy applied directly makes a better baseline worse. The score
  therefore **selects and does not judge**: it is written for every photograph that has a
  vector (NULL without one, so a heuristics-only collection or `store_embeddings: false`
  simply does not have this feature), the frames above the threshold become candidates for
  the deep tier — 955 of them, ~12 minutes at the measured 0.78 s per frame — and only the
  model's answer moves a verdict, with its own three-way question (screenshot / document /
  photo) rather than a widened deep-tier prompt, which would have changed verdicts on runs
  where this feature is off. **With the deep tier off nothing is reclassified at all**: the
  score is stored, the candidates are counted, the run is otherwise unchanged. An unreadable
  answer, a model that raises, a model or a text encoder that will not build — every one of
  them leaves the fast verdict standing, never "junk". Both prompt texts enter
  `quality_prompt_fingerprint`, so editing either invalidates the scores it produced (the
  F120 rule), and `scripts/measure_junk_rescue.py` prints the distribution and what every
  threshold would select **before** one is chosen, over the stored vectors and without a
  model.
- **A landmark is checked before it becomes a place** (F131): `features.landmarks_verify`,
  off by default, with `features.landmark_candidate_threshold` (0.5) next to it. CLIP
  proposes a landmark, the local VLM is asked what well-known place the frame shows, and
  only a proposal the model names **itself** goes on. This cascade was not assumed to
  work — it was measured first, and the measurement could have closed the feature. F75
  established that CLIP fails here for a different reason than it fails on animals: not
  perception (a cat against a drawing of a cat, which a model that looks at the scene
  cures) but **discriminating knowledge** — the wrong cities scored 0.980 against 0.991
  for the right one, so no threshold splits them — and a 3-billion-parameter model could
  share exactly that weakness, in which case it would confirm wrong cities with an air of
  authority and be worse than no cascade at all. The probe (`measure_landmarks.py
  --probe`) asked 104 frames with a known answer, 24 of them hard negatives — proposals
  above 0.50 that CLIP believed and corroboration threw away — and the model confirmed a
  wrong city **zero** times, at 92% accuracy. The mechanism is not knowledge but silence:
  71 of the 104 answers named nothing at all, which is the behaviour a gate wants. The
  **form of the question** decided more than the model did, so it is fixed in code with
  the numbers behind it: "what place is this?" backed 80% of the right proposals in each
  of three runs, "was this taken at X?" backed 20/42/42% — naming the proposal is what
  turns a check into an agreement. Silence is a **rejection**, not a parse failure, and
  the row stays `unknown` rather than moving to the place the model did name. The order
  does not bend: CLIP proposes → the model checks → **F75 corroboration decides**, so a
  country named in the path still refutes a match both models agreed on. Answers are
  remembered in the new `landmark_checks` table (schema v20) with the score that produced
  them — a rejected frame stays `unknown` and comes back into the stage on every run, and
  without the table it would be re-asked forever (the F130 loss), while the score is what
  the next calibration needs and what the stage previously stored nowhere at all. The
  `model` column carries a fingerprint of the question, so rewording it invalidates the
  answers it produced. Every failure keeps the cheap tier intact: a model that raises on
  one frame, or will not build at all, leaves that proposal to the rule of a run without
  the check (`naming.landmark_threshold`), and an unavailable model also leaves the gate
  unwidened — a wider band with nothing checking it is the one outcome worse than no
  feature. With the toggle off the stage is unchanged in every respect, down to the score
  its proposals are collected at.
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
  `sorter.animal_ids_sql` = `COALESCE(manual_pet.is_animal, <the automatic rule>)`
  (F137). That is what makes an edit survive **any** recompute, a change of model, of
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
- **Three cheap levers on the junk stage, and the measurement kept all three shut**
  (F164). The phase table of F147 named where the 25,4 minutes of the stage go, and the
  three obvious levers under it were: `junk_write` at 19,4 ms a frame (a commit per row?),
  the OCR thread ceiling of 4 (set "so a weak card is not knocked over", never priced),
  and the four VLM preparation threads on a 24-core machine with the card at 51%. Each
  was measured before it was pulled, and none of them turned out to be a lever.
  **`junk_write` is not writing.** The stage already wraps its whole pass in ONE
  transaction, and this exact upsert costs **0,004–0,005 ms a row** across three runs —
  0,1 s of the 470 s the phase was billed for. Committing per chunk or per row costs
  between nothing and everything depending on whether the operating system really
  flushes (0,003 vs 0,157 ms a row for chunks, 0,014 vs 2,271 for rows — two runs of
  three never flushed, one paid 55 s for the same rows): batching would have been a
  gamble on that, for 0,02%, and would have traded away the property that makes an
  interrupted run safe (half of today's verdicts and half of yesterday's is a database
  whose incrementality marker lies). What the phase really holds is the laplacian of the
  quality half — **26,9 ms** a frame against 0,005 for the verdict and 0,06 for the
  stored vector — which, over the 79% of frames that get a quality row, is the 19,4 ms
  exactly. Recorded where the statement is, so the next person starts from the right
  number. **More VLM threads are slower, not faster.** The case for raising
  `vlm.workers` was F101's profile (~0,6 s of CPU per frame against ~0,19 s of GPU) and
  it expired when F105 gave the runtime the fast image processor: preparing a frame now
  takes ~0,12 s of about **seven cores**, so four threads on 24 cores already ask for
  more than exists. Measured over 120 frames of the live collection with the model's
  turn replaced by a sleep of its measured 0,19 s: 2 threads x1,26, 4 x1,16, 6 x1,06,
  8 x0,98, 12 x0,87 — and the frames in flight grow the process from 551 MB to 2,6 GB
  across that range. The 51% of the live run is what NO overlap looks like at those two
  numbers, not a starving card. The default stays min(4, cores). **The OCR ceiling stays
  unmeasured on purpose**: what it protects is VRAM (one easyocr Reader per thread), so
  only a run on a free GPU can price it — the sweep now exists
  (`scripts/measure_ocr_workers.py`) and the number waits for it. Nothing about a
  verdict moves in any of this: `scripts/measure_junk_write.py` and
  `scripts/measure_vlm_workers.py` are new measurement tools, and the two new test
  modules pin what the seconds rest on — the pass writes in one transaction and an
  interrupted run leaves the previous verdicts untouched, and 1 / 2 / 4 / 6 / 8 / 12
  threads at either end of the stage produce byte-identical `media_class` rows with the
  deep tier's labels still in candidate order (the F101 invariant).
- **What costs time is on the run screen, and it says how much** (F138). Three knobs
  worth between a quarter of an hour and four hours of a run — `vlm.quality` +
  `vlm.quality_scope` (95 minutes by faces, 4.3 hours over everything),
  `features.pets_verify` (~13 min) and `dedup.keeper_vlm` (~20 min) — sat in the
  settings column next to the number of preparation threads, while the one knob of that
  weight that was already a checkbox (the deep tier, ~29 min) sat on the run screen. The
  line ran through history rather than through the idea, and the idea is: **the run
  screen holds what costs THIS run time, the settings hold how the product is arranged**.
  All four are now on the run screen and none of them is in the column any more — a
  value with two homes acquires two truths and a question about which one is in force.
  They behave exactly as `deep` and `pets` have since F123: the checkbox **starts from
  config.yaml** (`/api/process/defaults`) and what a person changes decides **one run**,
  the file is never rewritten. What keeps this from being the console of switches F133
  removed is that the list now means something: **every line carries its price and the
  sum stands under them**, right where the eye is already going to the button. The
  numbers are computed, not written into the markup — the same checkbox is four hours on
  one collection and four minutes on another — so the new `GET /api/process/estimate`
  multiplies a **measured rate** (0.78 s a frame for a VLM question, F113; 1.32 s for the
  comparative group question, F132) by a count taken out of **this index**: the frames
  the deep tier answered on last time, the frames above `pet_candidate_threshold`, the
  near-duplicate groups at or above `keeper_min_group_size`, the population of each of
  the four quality scopes. All four scope prices travel at once, so switching the select
  costs no request and the sum moves the instant a box is ticked. Where a count cannot be
  taken — a fresh collection, a stage that has never run — the answer is a **dash, never
  a zero**: a zero reads as "free", and the estimate says out loud that it is an
  estimate, because a person promised twenty minutes who then waits two hours believes no
  figure on that screen again. The block stays at **seven lines** including the
  un-switchable first one, and the only nested control is the scope of the quality
  question, shown when its parent is on and hidden when it is not.
- **`landmarks` stops recomputing what has not changed** (F136). The other half of F135,
  and where its three minutes actually were: `index` (34 s) and `geo` (3 s) already skip
  what they recognise, `landmarks` spent **138 s of a 176 s run** putting the same 7 619
  frames through CLIP again. The stage was incremental in its **selection** only — a match
  leaves as `confidence='visual'`, everything else keeps `'unknown'` and came back every
  time, even when not one file, prompt or threshold had moved since. So a run now
  remembers what CLIP found for each frame it looked at, and a later run looks only at the
  frames whose answer could have changed. What makes an answer stale is fingerprinted the
  way F120 does it: the file itself (path + mtime + size, as the index records it), the
  landmark list including the country, city and geonameid a match would be written with,
  `naming.landmark_threshold`, `landmark_group_min`, `landmark_group_dominance`, the score
  proposals are collected at, and the prompt texts — the distractor classes included,
  since editing one moves every score. Any of them moves and the frame goes back to CLIP;
  a marker that matches in part is not a match at all. **Corroboration is not cached.** It
  is not a per-file rule — the group rule reads the company a match keeps — so skipping a
  frame that proposed something would thin out its folder and quietly change the verdict
  of its neighbours, which would be a wrong city bought with saved time and is the exact
  failure F75 exists to prevent. The proposals of the skipped frames are raised back out
  of the DB, and the F75 rules then run over the same set a full pass would have built:
  the test that pins this compares a partial run against a full run over the very same
  selection, rather than against expectations written by hand. The marker lives in
  `landmark_checks` under a reserved key, so the schema does not move and "start over"
  still clears it; it cannot live in `places`, which `geo` recomputes from scratch before
  this stage ever runs. With `features.landmarks_verify` on (F131), a skipped frame is
  neither re-scored nor re-asked, and a reused answer still carries the whole decision.
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
