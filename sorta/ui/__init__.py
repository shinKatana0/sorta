"""U1/U3/U4/F31/F32/F35/F36: a local web server — a live sort-plan report +
Duplicates (incl. batch saving) + deleting a single frame + a "People" tab (managing
face clusters) + person/event albums ("Collect into folder", on top of the F34
engine) + the "Process" entry point — running the pipeline
index→geo→landmarks→classify→faces→events→junk→phash from the web, on a background server thread.

Most routes are READ-ONLY (reading originals/decoding thumbnails by file_id from the
index). Writes go through six narrowly-scoped paths: (1) `dedup_choice` — the user's
decisions on near-duplicates (keep/to_delete), a soft mark, does not touch files;
this also includes the batch `POST /api/dupes/choices` (F32) — the same effect as
`_apply_choice`/`_skip_group` over many groups in one transaction (the whole body is
validated before the first write — an invalid item causes no partial write); (2)
`POST /api/dupes/trash` — the non-keeper frames of a group physically go to the OS
trash; (3) `POST /api/photo/trash` (U4) — one arbitrary frame (a Cities-leaf or a
Duplicates frame) to the trash. (2) and (3) use the same `_trash_files` — a single
trash path: `send2trash` (not permanent deletion) + DELETE of the `files`/`dedup_choice`
rows so the index does not diverge from the disk. Original files are otherwise not
modified. (4) `POST /api/clusters/label` and (5) `POST /api/clusters/merge` (F31) —
naming/merging face clusters via `faces.label_cluster`/`faces.merge` (the public
faces.py API, used read-only from a code-ownership standpoint; the functions
themselves write to `face_clusters`). Both accept only int ids from the JSON body,
never a path. (6) `POST /api/album` (F35) — exporting a person/event slice
(link/copy/move) via `sorter.plan_album` (a public API, used read-only from a
code-ownership standpoint; it writes `moves`/`move_batches`); the body accepts only
kind/selector/mode/where/name/apply — strings/ints/bool, never a path (the server
resolves dest itself from `cfg.sort.album_dir`, `plan_album` resolves file_id -> path
from the DB itself). `apply=False` — a preview (writes nothing), the client confirms
and re-sends with `apply=True`.

(7) `POST /api/process` (F36) — starts a background THREAD running the stages
index→geo→landmarks→classify→faces→events→junk→phash (the leaf functions indexer/geo/
landmarks/faces/events/junk/dedup/naming — NOT imported from cli.py, to avoid a
cycle); the body accepts `source_dir: str` (required) + optional
`deep: bool`/`geo_online: bool` (F50/#34, default False) — which override
`cfg.sources`/`cfg.naming.vlm_enabled`/`cfg.geo.provider` ONLY in this run's cfg copy
(`dataclasses.replace`) — the shared cfg read by the other routes' handlers is not
mutated. The thread opens its own sqlite connection (not transferable between
ThreadingHTTPServer threads). One run per server — a repeated `POST` while running ->
409 (`_ProcessState.try_start` is atomic under a shared lock). `GET /api/process/status`
— a thread-safe progress snapshot (polling); `POST /api/process/cancel` sets a flag
checked BETWEEN stages (not mid-stage). F135: the snapshot also carries `stage_stats`
— `{stage: {"processed", "skipped"}}` for the stages whose own counters separate new
work from work they recognised as already done (`index`, `junk`) — and it keeps
`source_dir` after the run ends, which is what refills an empty source field: with one
run button the path has to come back by itself. The pipeline moves no files — it only reads
source_dir and writes the index, so the layout FS invariants (the moves.jsonl journal,
hash verification) do not apply here. F138: the body also carries `pets_verify`/
`quality`/`quality_scope`/`keeper` — the same per-run override on
`features.pets_verify`/`vlm.quality`/`vlm.quality_scope`/`dedup.keeper_vlm`, since these
are what a run's TIME is spent on and they moved onto the run screen out of the settings
column. F161 adds `products` to that list (`vlm.products`, the deep junk tier), which is
the effect `deep` used to have of its own: `deep` is now permission and nothing else.
`GET /api/process/estimate` prices every line of that screen: a measured rate
times a count from this index, `null` (a dash, never a zero) where the index cannot say.
F222 adds `landmarks` to the body — a STAGE like `faces`/`events`, off by default, with
`features.landmarks` behind it — and puts what a run will DOWNLOAD next to what it will
cost: `GET /api/env` carries, per line of the screen, the weights it raises, which of
them are absent from this disk and how big they are, all out of the one `tier_states()`
probe `sorta doctor` reads. While a download is running the status snapshot carries
`download` ({stage, weights, mb}); a refusal to download reaches `error` as a sentence
naming the stage, the model and the size, with the traceback in the log.
F248: a `source_dir` that is not a folder is refused with a CODE (`_source_refusal`) —
`source_missing` for a mistyped path, `relocate_collection_moved` over a full index,
and the second one carries `relocate.old_prefix`, which is what draws the transfer
offer on this path as well as beside a failed `index` stage. Nothing is walked either
way; what the two answers differ in is what the screen then says and offers.

(8) `POST /api/process/reset` (F42, the "Start over" button) — wipes the ENTIRE index
via the ready `db.reset_index(conn)` (the same tables as the CLI `sorta reset`:
metadata, geo, faces/clusters with names, events with names, junk, dedup_choice,
moves). The body carries `{"clear_geo": bool}` from the checkbox of the reset dialog
(F93) and it reaches `db.reset_index(clear_geo=...)`: without it the cached provider
answers survive the reset, with it they go too — the same pair as the CLI
`sorta reset --clear-geo`. Blocked with 409 while `/api/process` is still `running` (the same
`_ProcessState.snapshot()`). Does not touch files on disk or already-sorted folders —
only the DB contents. PlanCache is invalidated right after the reset, so the next plan
request rebuilds it (an empty DB -> an empty plan, see PlanCache).

(9) `POST /api/sort` (F43, the "Layout" tab, the "Apply" button) — the real layout of
the collection: calls `sorter.plan_and_sort(cfg, conn, by, dest, apply=True,
copy=..., progress=...)` on a background thread with its own sqlite connection (the
`_ProcessState`/`_run_pipeline` pattern, but its own `_SortState` — no stages, one
operation). The body `{"dest": str|null|"", "mode": "move"|"copy", "by": mode?}`, where
`by` is the criterion (`sorter.MODES`, F192 — absent means "city", the only one this
route could apply before): `dest` empty/null
-> in-place (restructuring the source tree, `dest=None` in `plan_and_sort`, F28);
`mode` outside {move, copy} -> 400. The `moves`/`move_batches` journal, blake3
verification and name-conflict resolution — entirely in `plan_and_sort`, ui.py does
not duplicate this logic. Cross-locking with `/api/process`: while a sort is running —
`POST /api/process` and `POST /api/process/reset` answer 409 (and vice versa); a
repeated `POST /api/sort` while sorting — 409 (`_SortState.try_start`). A `ValueError`
from `plan_and_sort` (e.g. in-place with multiple `cfg.sources`) is caught and stored
in the state as an error, without crashing the thread/server. `GET /api/sort/status` —
a snapshot for polling. After a successful apply — `PlanCache.rebuild` with the same
conn (the city plan reads the new paths); the "Moves" tab learns about it from a reset
of `movesLoaded` in JS.

(9a) `POST /api/sort/cancel` (F97) — sets a flag on `_SortState`, exactly like
`/api/process/cancel`. `plan_and_sort` reads it as `should_cancel` between files and
BREAKS out of the loop, so the batch still gets its `finished_at`; the result then
carries `cancelled` and `moved`, i.e. "copied 4 000 of 22 364" rather than "done".
Copying 220 GB takes an hour and a half — before this the only way to stop it was to
kill the process.

(9b) `POST /api/undo` + `GET /api/undo/status` + `POST /api/undo/cancel` (F97, the
"Roll back" button on the "Moves" tab and in the panel of a cancelled layout) — the
same three-endpoint shape as `/api/sort`, with its own `_UndoState`. It rolls back the
LAST batch, the one the manifest on that tab is already showing (`_last_batch_id`),
and there is deliberately no batch selector: fewer ways to misfire a button that
deletes files. The engine is `sorter.undo` — the blake3 check before deleting a copy,
the tail of an interrupted transfer and the closing of a batch left with
`finished_at=NULL` all live there. Cross-locked with `/api/sort` and `/api/process`
both ways (409): a rollback changes paths of files on disk. Before F97 the UI sent the
user to the terminal for exactly the situation the journal was written for.

(10) `POST /api/browse` (F51, the "Browse…" button — next to the "Process" path field
and next to the layout destination field on the "Cities" tab) — opens a native
folder-picker dialog and returns `{"path": str}`, plus a `problem` code when there is no
path (F247: no picker, or one that lost its answer). Never a 500: the button is just a
convenience, manual path entry always works. The dialog — tkinter `askdirectory` in a
SEPARATE subprocess (`_browse_for_folder`): tkinter is not thread-safe, and the
POST handler runs on a ThreadingHTTPServer thread, not the process's main thread; a
fresh process = its own main thread, without a conflict with the server. The returned
path is not processed at all on the server — `POST /api/process` already validates
`source_dir` as an existing directory (no extra checks needed: the path is chosen by
the user in a native dialog on their own machine, there is no injection).

(11) `GET /api/sort/suggest-dest` — the default destination path for the city layout:
`{"dest": "<source>_sorted"}` (the source — `cfg.sources[0]` or the common root of the
indexed files; see `_suggested_sort_dest`). JS prefills the `#sort-dest` field only if
the user has not entered anything yet.

(12) `POST /api/overrides` (F77, the "Cities" tab) — the user's manual corrections to
the layout: `{"file_ids": [int,...], "action": "exclude"|"reassign"|"clear"|"photo",
"target": str?}`. `exclude` — "leave alone": the file is not moved anywhere by the next
`sort --apply`; `reassign` — lay it out into `target` (a folder relative to the sort
root) instead of wherever the automatic rules put it; `photo`
(F103) — "the classifier is wrong, this IS a photo": the junk/document/product verdict
stops deciding the route and the file goes back to the automatic city layout; `clear` —
drop the correction. One row per file in `manual_overrides` (PRIMARY KEY file_id), a
repeated correction overwrites it. Like every other write route, the body carries only
ints and (for reassign) a target string — no paths from the client to a file: the
target is a folder INSIDE the layout, and `sorter._manual_target_parts` validates it
against the sort root before any destination is built from it. This endpoint moves
nothing on disk — the physical move happens in the shared `sort --apply`.
F203: that target need NOT be a folder of the current plan. The plan's folders stay in
the suggestion list, but the field is typed into, so `Россия/` on its own — the country
root, which the layout has had a branch for since F86 and no way to ASK for — and a
folder nobody has created yet are both legitimate answers. The name is checked as a NAME
and not as a path (`_reassign_target_refusal` -> `sorter.manual_target_parts`, the very
function the layout cleans its folder names with): a separator, a `..`, an absolute path,
a colon or a control character comes back 400 with a `reason` the page turns into a
sentence, instead of being stored and silently dropped at apply time. Nothing appears on
disk here either — an unknown folder is created by the apply, and a name conflict inside
it gets the usual `_1` suffix rather than overwriting anything.
The mark is served back LIVE (`_overrides_map`, read per request in `PlanCache`) rather
than baked into the built plan: a correction has to show up in the UI the moment it is
saved, while invalidating a mode's plan would make the next tab interaction pay for a
full rebuild (F70) on every click. The preview plan is built with
`keep_manual_excluded=True`, so a frame marked "leave alone" stays in the list (framed
red, unmarkable) even after a rebuild — the sorting plan that actually moves files never
contains it.

(13) `GET /api/source-tree`, `GET|POST /api/source-tree/excludes` (F81/F82, the source
block of the "Process" tab) — choosing folders to leave out, in either of the two
meanings the program has for that. The GET returns the directory structure under a root
(folders only, each with a file count and the total size of its subtree; metadata via
`scandir`/`stat`, contents never read), bounded by `_TREE_MAX_NODES`/`_TREE_MAX_DEPTH`
with a `truncated` flag rather than a silent cut. The POST writes
`{"root": str, "skip_scan": [str, ...], "skip_layout": [str, ...]}` into the exclusion
file (`indexer.save_excludes` — atomic, keyed by root, other roots preserved) and
reports which entries were refused. `skip_scan` is "do not scan" (F81): those files
never enter the index at all. `skip_layout` is "do not lay out" (F82): they are indexed
as usual — searchable, counted, deduplicated — and only `sort` leaves them alone. A
folder is in at most one of the two: the tree carries one state per node, and
`save_excludes` resolves an overlap in favour of "do not scan". Both endpoints take a
path from the client, so both run it through `_validate_tree_root` first — the same
"absolute path to an existing directory" rule `POST /api/process` applies to
`source_dir`; every list entry goes through `indexer.normalize_exclude`, which lets an
exclusion narrow the walk and nothing else. This endpoint touches neither files nor the
index: the rows already indexed under a new "do not scan" are removed by the next
`index()` run.

(14) `GET /api/cache`, `POST /api/cache/clear` (F94, the bottom of the "Process" tab) —
the two caches the program keeps, from the web app instead of only from `sorta cache`.
The GET reports what they occupy (the preview directory: files + bytes via `_sum_dir`,
metadata only; `geo_cache`: rows via `geo.geo_cache_size`) — the same numbers the CLI
prints. It is a SEPARATE route on purpose and is never folded into the status poll: the
preview cache is tens of thousands of files, so walking it once a tick would be a
directory scan per second. The POST takes `{"target": "preview"|"geo"}` — anything else
is a 400 — and calls the ready `imaging.preview_cache_clear()` / `geo.clear_geo_cache()`;
neither is reimplemented here, this feature is only the way to reach them. Both are
idempotent (an empty cache clears to zero rows, not to an error) and both are refused
with 409 while a run or a layout is in flight, under the same `busy_lock` as
`/api/process/reset`: mid-run a geo clear would send the rest of the stage back to the
network and a preview clear would delete the frames the stage is writing right now. The
response carries the fresh sizes, so the client does not need a second request. Nothing
here decides on its own what to delete — there is no size ceiling and no eviction, a
cache goes away only when the user says so.

(15) `GET /api/places/search`, `POST /api/place` (F85c, the "Cities" and "Events" tabs) —
assigning a place to a whole GROUP by hand. About 6 300 files of the live collection
carry no place signal at all (no GPS, no neighbour in time with one, no landmark,
nothing readable in the folder name), and no model will place them — the information is
not in them, it is in the person who took them. So the feature is not another guess, it
is a cheap way to say it in bulk: pick a group the user already thinks in (a whole event,
a whole source folder), pick a place, one action. The GET resolves typed text against
the BUNDLED base only (`geodata.city_ids_by_name`/`country_cc_by_name`, the same pair
`--where` uses — full-name matches, so same-named cities come back as several candidates
told apart by region); it reads nothing but the data files. The POST takes
`{"kind": "event"|"source_dir", "selector": str, "action": "assign"|"clear",
"country": str?, "city_geonameid": int?, "include_gps": bool?}` and writes one row per
file into `manual_places` — never into `places`, which has a single writer (`geo`,
ARCHITECTURE §2) and is recomputed from scratch on every run. The sorter reads that
table when it builds the plan, so the assignment survives a geo recompute and shows up
as `place_confidence='manual'` in the plan, the CSV and the report — a place the user
chose is never mistaken for one the program inferred. Files with `exact_gps` are skipped
and counted back in `skipped_gps` unless `include_gps` is set: the camera knew the place
at the moment of the shot, so overwriting it is a separate, explicit decision. `selector`
for `source_dir` is compared against `files.path` as a string and never opened (see
`_is_under`), which is why this route accepts it at all. Nothing moves on disk here; the
plan cache IS dropped afterwards (unlike an F77 correction, an assignment changes the
target folder of every file of the group).

(16) `GET /api/junk` (F103, the "Utility frames" slice) — the buckets the classifier
carries out of the collection, shown AS buckets: every frame whose `media_class.verdict`
is not `photo`, with per-verdict counters and one bounded page of one bucket
(`?bucket=&offset=&limit=`, the plan-page bounds). Read-only and reclassifying nothing.
The deep tier moves ~10% of the collection into service folders and a few of those
verdicts are wrong; the fix is the EXISTING `POST /api/overrides` with the F103 action
`photo` over the selected frames — one row per file in `manual_overrides`, so the
sorter lays them out by city again while `media_class` keeps the model's verdict (a
re-run of the junk tier therefore cannot silently wipe the correction). A bucket named
by `vlm.exclude_classes` (F133; the default is `["document"]`) answers WITHOUT
`thumb_url`: those are passports, medical forms and bank papers, and the project rule is
that such a frame is never decoded for display — the card carries a name and a date
only. Returning one to the photos is still allowed; only its preview is not built. The
class list is the config key rather than a constant, so the same list that keeps a frame
away from the model keeps it off the screen — and emptying it lifts both at once.
F171: a bucket is answered as a LIST IN ORDER — `media_class.score` descending, the
frames with no estimate keeping the path order behind them — and `ordered_by_score` says
whether that ordering happened, which is what lets the caption promise a ranking only
where there is one. No schema, no verdict and no threshold moves with it: the screenshot
bucket is right about 59% of what it points at, and what changed is that the slice now
says so and reads from the confident end down.

(17) `GET|POST /api/settings` (F104, the settings column of the "Cities" tab) — the
knobs that used to be reachable only by editing config.yaml and restarting: the deep
VLM tier and its model, preparation threads and input size. POST validates against
`_SETTINGS_SPEC` (an unknown key, a wrong type or an out-of-range number is a 400 and
the file is not touched), then changes the RUNNING config and persists into config.yaml
through `config.save_setting`, which rewrites ONE line and leaves the user's comments
alone. 409 while `/api/process`, `/api/sort` or `/api/undo` is running, under the same
`busy_lock` as the rest — swapping the model mid-classification is not a setting. Each
knob is read at the start of a run, so applying one invalidates nothing (the reasoning
per knob is above `_SETTINGS_SPEC`); the folder language, which DOES invalidate the plan
cache, keeps its own route (`POST /api/config/language`, F65).

(18) `GET /api/sort/summary?dest=&by=` (F104) — the numbers the pre-apply dialog states:
files, folders, volume, how much goes into the two review folders, and how much is
already lying in that destination (with how much of it will be skipped as an identical
copy — the F97 rule, asked of the same functions the apply uses). All of it is read off
the SAME built plan the "Layout" tree draws, so the dialog and the tab cannot disagree —
which is why `by` (F192) is part of the question and not assumed to be "city".

(19) `GET /api/overview` (F108, the "Overview" tab, the first one) — a snapshot of the
whole collection in four groups: what is in the index, how each frame got its place (and
how many have none), what the classifier decided and by which tier, and whether a layout
ran at all. Read-only and, unlike everything else on this page, built ONLY from plain
SQL aggregates: no plan, no cache, nothing precomputed. Both properties are load-bearing
— the plan of a 24k collection takes minutes to build, and a cached number would answer
the question the user opens this tab with ("what did the run just change?") with the
state from before it. Aggregates only: no file path and no file id is in the answer.

(20) `GET /api/animals` + `POST /api/animals/mark` (F123/F124, the "Animals" tab) — one
bounded page of the frames the frame-quality stage's stored scores and answers make
animals under the thresholds in force (F137, over canonical, readable files), most
confident FIRST:
`pet_score DESC, id`. The score travels with every card, because the verdict is 92% right
and the remaining 8% are found by reading the list down until the quality stops. The album
this tab offers is the existing `POST /api/album` with the new `kind='animal'`.

Those wrong 8% are what the POST is for (F124): `{"file_ids": [int,...], "action":
"animal"|"not_animal"|"clear"}` writes ONE row per file into `manual_pet` — never into
`frame_quality`, which has a single writer (`junk`) and is recomputed from scratch on
every run, prompt fingerprint included (F120), so a correction written there would last
until the next run and no longer. It is not an action of `manual_overrides` either: that
column decides the LAYOUT, and "this is not a cat" must never drop a file out of it. The
mark is applied WHEN READ — `sorter.animal_ids_sql`, the one expression the album slice,
this tab and the "Overview" counter all read, so an edit survives any recompute and the
three numbers cannot drift apart. F137: the automatic half of that expression is read at
the same moment and from the same place, out of the stored `pet_score`/`pet_vlm` and the
thresholds of the LIVE config — an edited threshold moves these three numbers at once,
without a run. `clear` deletes the row and hands the frame back to the
automatic verdict, which is why the column is two-valued rather than a presence flag: a
person both takes a false mark OFF and puts a missing one ON. A frame with a manual mark
stays IN the list (struck through, with the mark named on the card) instead of vanishing
— a card that disappears takes its own undo button with it. There is deliberately no
route that marks a whole band at once: the feature exists because somebody LOOKED at the
frame, and a threshold is already there for the other case.

(21) `GET /api/review` + `POST /api/review/mark` (F126, the "Review" tab, which replaces
the "Duplicates" tab) — the things a person looks at in order to decide what stays:
near-duplicates, blurred frames, closed eyes and (F150) frames of low resolution. One
workspace with SLICES rather than that many tabs, because it is one job. Duplicates are
the only GROUPED slice and keep their own route and their own rendering untouched —
`/api/dupes` and the
four write routes above answer exactly as they did, since that is the one path in the
product that deletes files and the one that has been run against a live collection. The
GET carries the counters of all the slices (a slice with nothing in it stays in the
switcher with a zero — an empty slice is an answer, a missing one is a riddle) plus one
bounded page of the current flat slice, over photographs only (`media_class.verdict =
'photo'`, F120) that are canonical and readable. Every flat list is ORDERED by the number
it exists for, ascending in all three cases (little variance = blurred, a thin slit = a
closed eye, few pixels = low resolution), and each opens as far as its own window —
`features.blur_review_max`, `features.eye_openness_max` (F179),
`features.low_resolution_mp`; `beyond=1` continues past that window, which is a prefix of
the same ordering, so nothing is lost or repeated at the seam. F157: the blurred list is
that ordering and nothing more — its window is the depth of the first page, and where the
frames stop looking blurred is read off the screen rather than off a number. Where F155's
`frame_quality.face_sharpness` exists, the frames that have one are ordered by it first
(`blur_order` in the answer): measured inside the face it finds 62% of the blurred frames
against 15% for the whole-frame number, and the two scales never meet in one comparison.
Low resolution is the one slice here whose membership was never measured by anything: the
two columns are a fact the indexer wrote down, so the card carries the resolution itself
and the hint says what the pixel count does NOT catch (a large frame ruined by
compression). Without a faces run the
eyes slice answers `eyes_reason='no_faces_run'` rather than a zero (F125: the eyes are only
measured where a face was found). The POST writes the
decision into the EXISTING `dedup_choice` (`keep`/`to_delete`, or `clear` to drop the
row) — `file_id` is its primary key, so a frame that appears in two slices carries one
decision, and `to_delete` is already understood by the sorter. There is deliberately no
route that marks a whole slice at once: reviewed by eye, blurred frames turn up in every
band up to 400, so sharpness ranks the list and a person decides each frame.

(22) `GET /api/search` (F134, the query line of the "Slices" tab) — the F129 engine
behind the field F133 drew and left disabled: `q` is the words, `offset`/`limit` a PAGE of
the ranking (`features.search_page` frames by default, clamped, never a similarity
threshold — there is none and there will not be one), and the answer is that page as cards
with a score on each, plus `total`/`has_more` like every other paged slice (F173: the
ranking does not end where the page does, and the counter has to say so). Every answer also
carries the STATE of the index — `state` (empty / other_model / partial / ready),
`available`, `indexed`, `photos` and `index_model` — because the failure
this route exists to avoid is answering "nothing was found" when the truth is "nothing was
ever computed": the two are the same empty list on screen, and only one of them is a fact
about the person's photographs. An empty `q` returns that state and nothing else, without
loading a model, which is what the tab asks on open to decide whether the line may be
used at all. Sensitive classes follow the F133 rule unchanged — a frame whose
`media_class.verdict` is in `vlm.exclude_classes` is ranked but carries no `thumb_url`, so
a search cannot become the way around a protection the slices already apply. The one
action the results offer is the existing `POST /api/album` with `kind='query'` and the
words as the selector; both routes share one lazily loaded text encoder.

(23) `GET /api/face-slices` (F152, the three face pins of the "Slices" tab) — one bounded
page of "photographs with people" / "group photographs" / "portraits", plus the counters
of all three. What makes them different from the slices beside them is not the shape of
the route but the nature of the answer: membership is a FACT of the `faces` table (a
detector either found a box on the frame or it did not) rather than a place in a ranking,
so no card carries a score and none is invented. The rules live once, in
`sorter.face_slice_ids_sql`, which the albums and the "Overview" counters read too — and
the one thing all three exclude is the marker row `bbox = '[]'` ("processed, no faces"),
which 24 195 of 24 196 live files carry and which turns "with people" into "everything"
the moment it is forgotten. Two of the rules take a number out of `features:` and both
are geometric: `group_photo_faces` (3) is a count of boxes, `portrait_face_share` (0.08)
is the share of the frame one box covers, out of the bbox and `files.width/height`.
Without a faces run the answer is `reason='no_faces_run'` and counters of `null` — the
F125 rule, since a zero would read as a claim about the person's photographs. Sensitive
classes follow the F133 rule unchanged (listed, but no `thumb_url`). The one action these
slices offer is the existing `POST /api/album` with `kind='people'|'group'|'portrait'`.

(24) `GET /api/saved-slices` (F151, the pinned queries of the "Slices" tab) — the list of
saved slices out of `features.saved_slices` on every answer, plus one bounded page of the
one that was asked for (`slice=`, `offset=`, `limit=`). It is `/api/search` with the words
coming from the config instead of from a field: the same engine, the same page shape, the
same state of the index, the same F133 rule about sensitive classes — and the same absence
of a threshold, because "is this a child" is not a line anybody can draw. What the route
adds is the two things a pin needs: `queries`, the phrases the slice was ranked by (so the
panel can show what it asked and a reader can go and edit them), and the fact that these
lists are ESTIMATES. There is no count on a pin and none is invented: a ranking has no
size, and a number beside "children" would read like the archive holds exactly that many.
Asked without `slice` the route ranks nothing and loads no model — that is the call the
tab makes on open to build the row.

(25) F174 adds no route and changes no storage: `/api/junk` and `/api/animals` now carry
`dest`/`dest_reason`/`dest_group` on every card — WHERE that frame ends up. Two marks the
slices offer read as one movement to the person making it ("this frame does not belong
here") and neither said where the frame goes: taking an animal mark off changes a
membership and moves no file, while returning a product to the photos is a real transfer
into a city on the next apply. So the button reads the same in both (`slice_return_button`)
and the difference is stated under it — `dest_goes_to` there, `dest_stays_in` here — and a
bulk return states the SPREAD of the selection ("12 frames: 7 into cities, 5 into
no_place"), because one folder name out of twelve deceives a person who ticked dozens. The
folder is `sorter.destinations`, i.e. the code that builds the plan, asked with the
correction already assumed; nothing is applied any earlier than before.

(26) `POST /api/saved-slices/pin | /unpin | /move` (F156, a query of one's own) — the
three writes of `features.saved_slices`, and the only three. They add a typed query as a
named pin, take a pin away, or step one up/down the row; each answers with the WHOLE list,
which the pin row is redrawn from. The storage is `config.yaml` and not the index, because
`reset` and every re-processing rebuild the index and a slice somebody named must not be
one re-index away from gone. `features.max_pinned_slices` bounds what the interface may
add (F133's reason, not a resource one) and reaching it is an answer with `reason='limit'`
rather than a pin that quietly does not appear. Unpinning removes a config entry and
touches no file. Nothing here ranks anything or loads a model — a pin saves words.
`GET /api/tabs/visibility` gains `reasons` in the same feature: a built-in slice that is
empty says WHICH empty it is (`not_run`, and then the panel links to the run screen, or
`none_found`), because a bare zero reads as a claim about somebody's photographs.

(27) `POST /api/quit` (F209, the "Quit" button in the header) — closing the program
from the interface. It lives there rather than in a tray icon because the interface is
what works everywhere the product works: GNOME removed the tray in 3.26, so on Ubuntu
and Fedora an icon exists only through an extension the person installs by hand. A tray
is a shortcut to this route on the systems that have one, not a second implementation.
The stop is the Ctrl+C one and not a new one: `httpd.shutdown()` ends `serve_forever`,
and `serve`'s own `finally` then closes the server socket (the port is free for the next
start) and the connection this server read the index through. No `os._exit` — nothing
about the exit is skipped. The answer is written BEFORE anything stops, and the shutdown
runs on a thread of its own because `shutdown()` blocks until the serve loop has
returned: a person who pressed "Quit" has to see the program closing rather than a
severed connection, which reads as a crash.
A RUN is what this route mostly exists to protect. A pass over a real collection counts
for up to five hours, and losing it to one press would be the worst thing this feature
could do — so while `/api/process`, `/api/sort` or `/api/undo` is in flight the route
answers 409 with the reason `run_in_progress` and stops NOTHING, exactly like every
other writing route refuses while busy (F145). Only a second request carrying
`{"confirm": true}` closes the program, and it interrupts the run through the very flag
`/api/process/cancel` sets rather than inventing a second way to stop one. The refusal
is the server's and not the page's, per F133: a dialog the interface draws forbids
nothing, and a request sent past the interface would close the program all the same.

(28) `GET /api/relocate/suggest`, `POST /api/relocate` (F244, the run screen) — pointing
the index at a collection that moved, from the screen that refuses to index it. F242 gave
the product `sorta relocate` and the stop whose sentence already reaches the collapsed
error row; acting on it needed a terminal, which half the users of an equal-rights web
app will not open. The POST takes
`{"old_prefix": str, "new_prefix": str, "apply": bool=False}` and calls
`relocate.relocate` — which columns hold a path, where a prefix ends, the refusals and
the single transaction all stay there and are not repeated here. Without `apply` it
answers the plan and writes nothing; with it, the same plan carrying `applied`. A
`RelocateError` is an ANSWER and not a 500 (`{"error": "relocate_refused", "reason": ...}`
with HTTP 200): the engine writes nothing when it refuses, so the request did not fail.
In `_BUSY_GUARDED_ROUTES` like every other writer — it rewrites `files.path`, the column
each stage of a run is keyed by — and the plan cache is dropped after an applied move,
since every built plan is about the old paths. The GET prefills the old prefix from the
index (`_indexed_prefix`): the database knows that path, and a path typed by hand is a
path mistyped.

Security: the only entry to a file on disk for reading (`/thumb`, `/photo`) is a
file_id, resolved strictly via `SELECT path FROM files WHERE id = ?`. These routes
never accept a path directly from the request, so an arbitrary path (incl. `../..`)
does not resolve — a non-numeric/unknown id simply finds no row in files and answers
404. The write endpoints (`POST /api/dupes/*`, `POST /api/photo/trash`) also operate
only on a file_id from the JSON body (no paths from the client); before deleting a
`files` row or sending a path to the trash, the id is resolved by the same query
`SELECT ... FROM files WHERE id IN (...)` — unknown ids are silently ignored, not
substituted as a path. The server binds only to 127.0.0.1.

F208: the binding is what protects this port from the NETWORK; it protects nothing from
the user's own browser, which visits other people's pages and this port in one session. So
every POST must carry `Content-Type: application/json` and, when the request states an
`Origin`, that origin must be this server — see `_post_refusal` above `_make_handler` for
why the content type is the line that closes the class and the origin is only the second
one. A refusal is a 403 with a code and a sentence. GET is untouched: thumbnails and
previews are ordinary browser requests and carry no content type at all.

plan_and_sort (sorter, dry-run) — the single source of the plan; PlanCache calls it
with `write_reports=False` (no CSV/HTML side files from the UI path) and at most once
per mode per cache generation — LAZILY, on the first request for that mode (F70), so
neither the server start nor a `rebuild` blocks for the ~13 s a mode costs on a 26k
collection. `GET /api/plan?mode=` answers with a per-target-folder AGGREGATE
(folder -> count/size, kilobytes); the files of one folder come as an explicit page
(`&category=&offset=&limit=`), never as the whole 26k-element plan.

F182: where the code lives
--------------------------
This was one 14 427-line file, and on 2026-08-03/04 ten features queued for it in a
single day — two workers inside it is a guaranteed conflict (F152 came back with 18
divergences across 10 files; F160 with an import that vanished and that neither gate
caught alone). No other module in the project ever had that problem.

The cut is BY TAB, not by layer. A feature normally lives in one tab — F150 in
"Review", F156 in "Slices", F159 on the run screen — so two features in two tabs stop
meeting at all; a cut into routes/queries/markup/script would have left every feature
touching all four. F133 already rebuilt the interface along this axis and the file was
already marked with fifteen `# --- F126: the "Review" workspace` seams; this made them
real.

    common.py    what more than one tab needs — the connection, the paging window,
                 the image caches, the destination of a frame
    layout.py    "Layout": the plan, the canon, the places, the albums, the settings
    slices.py    "Slices": the queries, the pins, the built-in slices, the search line
    review.py    "Review": duplicates, blur, closed eyes, restoring
    overview.py  "Overview": the state of the collection in one screen
    moves.py     "Moves": the manifest of a batch, and the undo
    process.py   the run screen: the pipeline, the source tree, what a run costs
    page.py      the template and the `{{key}}` substitution
    strings.py   the chrome catalog every caption feature edits
    __init__.py  this docstring, the busy-route table, `_make_handler`, `serve` — and
                 the re-export of every name above, which is what keeps `sorta.ui` the
                 one module fifty test files import

    sorta/web/   page.html · style.css · app/app.js — the 42% of the old file that was
                 markup, styles and browser script inside triple quotes

The dependency direction is a DAG with `common` at the bottom and `__init__` at the
top; `tests/test_ui_package.py` fails on a cycle, on a tab that `sorta.ui` stops
re-exporting, and on `common` reaching back into a tab.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import logging
import mimetypes
import os
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, urlsplit

from send2trash import send2trash as send_to_trash

from .. import db, faces, i18n, imaging, restore
from ..config import (
    Config,
    # F160 replaced its own uses of FeaturesConfig with Config and dropped the import;
    # F149 landed `_restore_frame(db_path, features: FeaturesConfig, ...)` in main
    # meanwhile. Different lines, so git merged both without a word and left a name with
    # no import behind — the kind of break only a gate on the SUM can find.
    FeaturesConfig,
    SavedSlice,
    save_language,
    save_saved_slices,
    save_setting,
)
from ..dedup import (KEEPER_SOURCE_SHARPNESS, assign_duplicates, compute_phashes,
                    group_key, near_duplicate_groups, read_group_keepers)
from ..detect import detector_settings
from ..diagnostics import warn_if_geo_data_missing
from ..events import build_events
from ..faces import detect_and_cluster
from ..geo import clear_geo_cache, geo_cache_size, resolve_places
from ..geodata import GeoDataMissing, GeoResolver
from ..indexer import excludes_path, index as run_index, load_excludes, normalize_exclude
# `_has_column`: "does this database have that column yet". The indexer reads its own
# optional columns through it, and the blur list (F157) reads F155's `face_sharpness`
# through the same one — the two features were merged in either order on purpose.
from ..indexer import _has_column
from ..indexer import save_excludes as save_excludes_file
from ..junk import classify as classify_junk
from ..junk import (
    CLASSIFY_PHASE_VLM,
    CLASSIFY_STAGE,
    VERDICTS_STAGE,
    faces_stage_ran,
    search_index_model,
    search_index_settings,
    sweep_previews_for_new_classes,
)
from ..landmarks import Classifier, clip_classifier, detect_landmarks
from ..landmarks import batched
from ..naming import name_events, naming_settings
from ..runlog import (
    Measurement,
    log_environment,
    measurement_files,
    measurement_unit,
    read_measurements,
    stage_timer,
)
from ..search import (
    REASON_EMPTY,
    REASON_OTHER_MODEL,
    EmbeddingsMissing,
    TextEncoder,
    match_person,
    person_page,
    rank_queries,
    rank_text,
    text_encoder,
)
from ..sorter import (
    ALBUM_KINDS,
    ALBUM_MODES,
    CLASS_ALBUM_KINDS,
    FACE_SLICES,
    SELECTORLESS_ALBUM_KINDS,
    AlbumReport,
    Destination,
    PlanItem,
    animal_auto_sql,
    animal_ids_sql,
    destinations,
    face_slice_ids_sql,
    plan_album,
    plan_and_sort,
    quality_slice_from,
    quality_slice_where,
    undo,
)
# F104: the pre-apply summary has to say what the apply will DO, so it asks the two
# functions the apply itself uses rather than re-deriving the rule here — the moment
# the two answers can differ, the dialog is quoting numbers nobody has to honour.
# `_fs`: the long-path form a filesystem call needs on Windows; `_is_the_same_file`:
# "the file already lying at the target is byte-for-byte the one we would put there".
from ..sorter import _fs, _is_the_same_file

# F182: the tab modules, and every name they hold.
#
# The re-export is not tidiness, it is the property that makes the split checkable. Fifty
# test files import `sorta.ui` and reach for `ui._UI_STRINGS`, `ui.BUSY_REFUSED_ROUTES`,
# `ui._junk_payload` and two hundred others; a package that exported only what the
# dispatcher below calls would have turned a pure move into an edit of fifty files, and
# then nothing would prove the move changed no behaviour. So `sorta.ui` still answers to
# every name `sorta/ui.py` answered to — the imports above are the module's own, these are
# the tabs. (ruff would call the unused ones dead; per-file-ignores in pyproject.toml say
# why they are not.)
# F217: `tiers` and `wizard` ride along because `strings` reads the tier catalog out of
# them rather than spelling one out by hand — and the package re-exports every name a tab
# module defines, which the suite checks.
from .strings import (
    _BROWSE_APT_HINT, _BROWSE_NO_ANSWER, _BROWSE_UNAVAILABLE, _TIER_LANGS, _UI_STRINGS,
    _browse_strings, _doctor_line, _tier_strings, tiers, wizard,
)
from .common import (
    DEFAULT_PORT, STARTUP_CONFIG, STARTUP_DATABASE, STARTUP_ENVIRONMENT, STARTUP_GEO,
    STARTUP_GPU, STARTUP_PORT, STARTUP_SERVER, STARTUP_STEPS,
    TRASH_REFUSED_FAILED, TRASH_REFUSED_IN_USE, TRASH_REFUSED_NO_BIN,
    TRASH_REFUSED_PERMISSION,
    _CLUSTER_SAMPLE_LIMIT, _DEFAULT_ALBUM_DIRNAME, _DEST_GROUPS, _DEST_MODE,
    _EVENT_SAMPLE_LIMIT, _ImgCacheKey, _LANG_SELF_NAMES, _OVERVIEW_LIVE, _PLAN_PAGE_DEFAULT_LIMIT,
    _PLAN_PAGE_MAX_LIMIT, _PREVIEW_CACHE_MAX_ITEMS, _PREVIEW_MAX_EDGE, _ProgressCB,
    _StartupState, _SUPPORTED_MODES, _THUMB_CACHE_MAX_ITEMS, _THUMB_DECODE_CONCURRENCY,
    _THUMB_MAX_EDGE, _TRASH_PROBE_PREFIX, _WIN_SHARING_VIOLATION,
    _ThumbCacheKey, _UI_LANGS, _connect, _destination_json, _destinations_for, _encode_jpeg_cached,
    _is_under, _log, _page_payload, _parse_file_id, _parse_file_id_query, _parse_page_window,
    _preview_bytes, _preview_cache, _preview_cache_lock, _refusal_reason, _resolve_path,
    _startup_payload,
    _startup_state, _thumb_bytes, _thumb_cache,
    _thumb_cache_clear, _thumb_cache_lock, _thumb_decode_semaphore, _trash_files,
    _trash_volume_key, _volume_accepts_trash,
    _validate_file_id_payload, _validate_file_ids_payload, errno, startup_state,
)
from .layout import (
    PlanCache, _IMAGING_SETTING_ENV, _ManualPlace, _ModePlan, _OVERRIDE_ACTIONS, _PLACE_ACTIONS,
    _PLACE_COUNTRY_LIMIT, _PLACE_KINDS, _PLACE_REGION_LIMIT, _PLACE_SEARCH_LIMIT,
    _PLACE_SEARCH_MIN_QUERY,
    _SETTINGS_SPEC, _SETTING_SECTIONS, _SettingSpec, _SortState,
    _album_dest, _album_report_to_json, _apply_bulk_place, _apply_overrides, _apply_settings,
    _city_candidates, _city_option, _clusters_payload, _country_candidates, _country_label,
    _dest_occupancy, _events_payload,
    _geo_resolver, _geo_resolver_cache, _overrides_map, _place_target_ids,
    _places_search, _places_search_payload, _region_candidates, _region_option,
    _plan_category, _plan_item_to_json, _reassign_target_refusal, _run_sort,
    _settings_payload, _suggested_sort_dest,
    _summary_dest, _validate_album_payload, _validate_cluster_label_payload,
    _validate_cluster_merge_payload, _validate_language_payload, _validate_overrides_payload,
    _validate_place_payload, _validate_settings_payload, _validate_sort_payload,
)
from .slices import (
    _ALBUM_BLOCKED_ALL_BUCKETS, _ALBUM_BLOCKED_DOCUMENTS, _ALBUM_BLOCKED_NO_KIND,
    _ALBUM_BLOCKED_SENSITIVE, _ALBUM_NO_SELECTION, _ANIMALS_JOIN, _ANIMAL_MARK_ACTIONS,
    _FACE_COUNT_SQL, _FACE_FROM, _FACE_LIVE, _JUNK_NO_PREVIEW,
    _JUNK_ORDER, _LazyTextEncoder, _NEVER_ALBUM_CLASSES, _PIN_DUPLICATE, _PIN_EMPTY, _PIN_LIMIT,
    _SEARCH_AVAILABLE_STATES, _SEARCH_COVERED_SQL, _SEARCH_NAMES_SQL, _SEARCH_PARTIAL,
    _SEARCH_PHOTOS_SQL, _SEARCH_READY, _SEARCH_ROWS_SQL, _SLICE_NONE_FOUND, _SLICE_NOT_RUN,
    _album_refusal, _animal_item_to_json, _animals_count_sql, _animals_payload,
    _animals_population, _animals_select, _apply_animal_mark, _apply_saved_slices,
    _bucket_album, _face_item_to_json,
    _face_slice_count, _face_slice_where, _face_slices_payload, _junk_item_to_json, _junk_payload,
    _parse_face_slice_query, _parse_junk_query, _parse_saved_slice_query, _parse_search_query,
    _person_payload, _pinned_moved, _pinned_with, _pinned_without, _saved_slice_by_name,
    _saved_slices_payload, _search_index_state, _search_item_to_json, _search_items,
    _search_payload, _tabs_visibility_payload, _validate_animal_mark_payload,
    _validate_move_payload, _validate_pin_payload, _validate_slice_name_payload,
    album_selection, class_album_refusal,
)
from .review import (
    RESTORE_ERROR_SENSITIVE, RESTORE_ERROR_TOO_LARGE, RESTORE_ERROR_VIDEO,
    TIER_SAME_IMAGE, TIER_SIMILAR, _BLURRED_ORDER_WITH_FACE,
    _DUPES_CACHE_MAX_ITEMS, _TIER_CAPTIONS,
    _DupesCacheKey, _DupesFingerprint, _PENDING_JOIN, _REVIEW_MARK_ACTIONS, _REVIEW_SLICES,
    _REVIEW_SLICE_COLUMNS, _REVIEW_SLICE_KIND, _REVIEW_SLICE_ORDER, _apply_batch_choices,
    _apply_choice, _apply_review_mark, _blurred_order_column, _db_fingerprint, _dupes_cache,
    _dupes_cache_clear, _dupes_cache_lock, _dupes_payload, _order_by_size, _order_similar,
    _parse_review_query,
    _pending_dupe_groups, _restore_decision, _restore_frame, _restore_notice, _restore_offer,
    _restore_refusal,
    _restore_source_row, _restored_item_to_json, _restored_row, _restored_source_json,
    _review_count, _review_flat_counts, _review_from, _review_item_to_json, _review_order,
    _review_payload, _review_pending_count, _review_pending_counts, _review_where, _skip_group,
    _tier_captions,
    _trash_group, _validate_batch_choices_payload, _validate_group_payload, _validate_keep_ids,
    _validate_review_mark_payload,
)
from .overview import (
    _PLACE_CONFIDENCE_ORDER, _media_class_breakdown, _overview_layout, _overview_payload,
    _overview_place,
)
from .moves import (
    _UndoState, _last_batch_id, _moves_payload, _run_undo, _target_rel,
)
from .process import (
    _BROWSE_DIALOG_SCRIPT, _BROWSE_DIALOG_TIMEOUT_S, _BROWSE_NO_ANSWER_EXIT, _browse_text,
    BROWSE_CANCELLED, BROWSE_NO_ANSWER, BROWSE_UNAVAILABLE, CLASSIFY_PHASE_PETS_VLM, CLASSIFY_PHASE_RESCUE_VLM, RELOCATE_REFUSED, SOURCE_MISSING, _CACHE_TARGETS, _DEEP_TIER_SQL, _DEFAULT_RATES,
    _DownloadRefused,
    _ESTIMATE_CACHE_MAX_ITEMS, _LANDMARK_SCAN_KEY, _LIVE_PHOTOS_SQL, _LazyClassifierHolder,
    _OPTIONAL_STAGES, TIER_ABSENT, TIER_READY, TIER_WEIGHTS, _deep_tier_ran,
    _run_language, _gpu_present, _gpu_present_cache, _gpu_present_cache_clear,
    _gpu_present_lock, _parts_payload, _tier_state_name, _tiers_payload, _weights_payload,
    _PIPELINE_STAGE_NAMES, _PipelineCancelled, _ProcessState, _RATE_DEFAULT, _RATE_FIXED,
    _RATE_MEASURED, _RATE_UNITS, _Rate, _RunOptions, _SEC_PER_BASE_FRAME, _SEC_PER_EVENTS_FRAME,
    _SEC_PER_FACES_FRAME, _SEC_PER_LANDMARKS_FRAME, _SEC_PER_VLM_FRAME, _StageFn,
    _StageProgress, _StageStats,
    _TREE_MAX_DEPTH, _TREE_MAX_NODES, _any_truncated, _browse_for_folder, _browse_lock,
    _cache_payload, _env_payload, _estimate_cache, _estimate_cache_clear, _estimate_cache_lock,
    _excludes_payload, _indexed_prefix, _memory_payload, _pipeline_steps,
    _positive_or_none,
    _process_defaults_payload,
    _process_estimate_payload, _relocate_payload, _relocate_plan_to_json, _resolve_rates,
    _run_browse_dialog,
    _run_cfg, _run_log_fingerprint,
    _run_pipeline, _scan_dir, _source_refusal, _source_tree_payload, _stage_stats, _sum_dir,
    _validate_cache_clear_payload, _validate_excludes_payload, _validate_process_payload,
    _validate_relocate_payload, _validate_rerun_optional_payload, _validate_tree_root,
)
from .page import (
    _INDEX_HTML_TEMPLATE, _WEB_DIR, _load_index_template, _read_web, _render_index_html, _t,
)


# F145: which POST routes may not write while a run, a layout or an undo is in flight.
#
# The reason is not the look of a button. The pipeline rewrites `media_class`,
# `frame_quality` and `places` wholesale, and geo empties `places` before refilling it —
# a second writer over those tables mid-run is a race whose loser is the user's index.
# Eight routes already refused with their own 409 and their own wording; the rest, every
# one of which writes the database, the config file or files on disk, did not.
#
# The three sets are named rather than implied so that a route cannot join the server
# without a decision being made about it: the suite walks the dispatcher below and fails
# on any POST path that is in none of them.
_BUSY_SELF_GUARDED_ROUTES = frozenset({
    # These check the state themselves because they distinguish WHICH process is in the
    # way ("sort is running" / "undo is running") or take the lock for a critical section
    # of their own — the answer a caller gets from them is more specific, not less.
    "/api/process", "/api/process/rerun-optional", "/api/process/reset",
    "/api/cache/clear", "/api/config/language", "/api/settings",
    "/api/sort", "/api/undo",
    # F209: it writes nothing, and it is here for the strongest version of the reason
    # this table exists — a five-hour run is lost by closing the program on top of it.
    # It checks the state itself because it is the one route that may proceed anyway:
    # with `confirm` the answer is not a refusal but an interruption.
    "/api/quit",
})
_BUSY_GUARDED_ROUTES = frozenset({
    # Marks and choices in the index...
    "/api/dupes/choice", "/api/dupes/choices", "/api/dupes/skip",
    "/api/review/mark", "/api/animals/mark", "/api/overrides", "/api/place",
    # F149: writes a new file beside the original AND its row in the index — both halves
    # of what the busy guard exists for, and the run it would race with rewrites the very
    # tables the new row will need. The price of guarding it the ordinary way is that the
    # lock is held for the model call (~1 s, and the one-off weights download on the first
    # press), so another mark waits instead of being refused. That is the right way round:
    # a mark that arrives a second late is a slow click, a `files` row written into the
    # middle of a run is somebody's index.
    "/api/review/restore",
    "/api/clusters/label", "/api/clusters/merge",
    # ...files moved on disk...
    "/api/dupes/trash", "/api/photo/trash", "/api/photos/trash", "/api/album",
    # F244: it rewrites `files.path` in every row at once — the column each stage of a
    # run reads its work from — so a move crossing a run is the loudest version of the
    # race this table exists for.
    "/api/relocate",
    # ...and config.yaml.
    "/api/source-tree/excludes",
    # F156: the three writes of `features.saved_slices`. Nothing a run reads is touched by
    # them — the pins are read by the Slices tab and by nothing else — and the guard is
    # here for the FILE: each one is a read-modify-write of config.yaml, `busy_lock`
    # serializes them against each other and against the two config writers that were
    # already guarded (`/api/settings`, `/api/config/language`), and two of those cycles
    # crossing would lose one of the two edits entirely.
    "/api/saved-slices/pin", "/api/saved-slices/unpin", "/api/saved-slices/move",
})
# The POST routes that stay live on purpose: cancelling is what a person reaches for
# WHILE something runs, and the folder picker only reads.
_BUSY_EXEMPT_ROUTES = frozenset({
    "/api/process/cancel", "/api/sort/cancel", "/api/undo/cancel", "/api/browse",
})
# Every route that must answer 409 while busy, whichever half of the server refuses it.
BUSY_REFUSED_ROUTES = _BUSY_SELF_GUARDED_ROUTES | _BUSY_GUARDED_ROUTES


# F208: who is allowed to POST here at all.
#
# `127.0.0.1` keeps the network out. It does not keep out the user's own browser — and
# the browser is precisely the program that visits somebody else's page and this port in
# the same session. A page open in another tab could POST here: with
# `Content-Type: text/plain` the request is a "simple" one by the CORS rules, so the
# browser sends it WITHOUT asking permission first. The answer stays unreadable to that
# page, but the action has already happened — `/api/sort` moves files, `/api/photos/trash`
# empties them into the bin, `/api/settings` rewrites config.yaml. Every safety net this
# product is built on (dry-run by default, the journal, `undo`, blake3) protects against a
# human mistake and against nothing at all here.
#
# Requiring `application/json` is what closes the whole class rather than one route:
# that content type is not "simple", so the browser MUST ask first with an `OPTIONS`
# preflight, this server answers no such permission (there is no `do_OPTIONS` and no CORS
# header anywhere in it), and the real request is never sent. `Origin` is the second line
# and not the first: the header is not always present, so it can convict a foreign source
# but cannot vouch for a missing one.
#
# Deliberately NOT a token or a session: this is a local single-user server, and a secret
# is state that would have to be stored, rotated and handed to the page.
_JSON_MEDIA_TYPE = "application/json"
# The refusal codes. A code and a reason, like every other refusal here — whoever this
# breaks (a browser extension, somebody's own script) has to be able to read WHY, instead
# of getting a bare 400 with nothing in it.
REFUSED_CONTENT_TYPE = "content_type"
REFUSED_ORIGIN = "origin"
_POST_REFUSAL_DETAIL = {
    REFUSED_CONTENT_TYPE: f"POST requires Content-Type: {_JSON_MEDIA_TYPE}",
    REFUSED_ORIGIN: "POST is served only to a page of this server",
}


# F209: why `/api/quit` refused. A code and not a sentence, like every other refusal
# here — the interface turns it into the question it asks, and a caller past the
# interface can read what happened without parsing prose.
QUIT_RUN_IN_PROGRESS = "run_in_progress"
# The three long operations, by the name the answer carries. Which one it is decides the
# sentence on screen: "a run is going" and "a layout is going" are one refusal to this
# server and two different things to lose.
QUIT_RUNNING_NAMES = ("process", "sort", "undo")


def _media_type(raw: str | None) -> str:
    """`Content-Type` -> the media type alone: lowercased, without the parameters.

    `application/json; charset=utf-8` is what fetch() sends and it is the same type.
    """
    return (raw or "").split(";", 1)[0].strip().lower()


def _origin_is_ours(origin: str, host: str | None) -> bool:
    """Is this `Origin` the very server the request reached?

    Compared against the request's own `Host` header rather than against a constant: the
    port is chosen at start-up (0 in tests -> whatever the OS gave) and the page is opened
    under whatever name the user typed. `null` — a sandboxed frame, a `file://` page — has
    no netloc and is not ours.
    """
    parsed = urlsplit(origin.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    return parsed.netloc.lower() == (host or "").strip().lower()


def _post_refusal(content_type: str | None, origin: str | None,
                  host: str | None) -> str | None:
    """Why this POST may not be served — a refusal code, or None to let it through."""
    if _media_type(content_type) != _JSON_MEDIA_TYPE:
        return REFUSED_CONTENT_TYPE
    if origin is not None and not _origin_is_ours(origin, host):
        return REFUSED_ORIGIN
    return None


def _make_handler(db_path: Path, cache: PlanCache, cfg: Config,
                  process_state: _ProcessState,
                  sort_state: _SortState,
                  busy_lock: threading.Lock,
                  undo_state: _UndoState,
                  config_path: str | Path | None = None) -> type[BaseHTTPRequestHandler]:
    default_lang = i18n.normalize_lang(cfg.raw.get("language"))
    _index_html_cache: dict[i18n.Lang, bytes] = {
        default_lang: _render_index_html(default_lang).encode("utf-8"),
    }
    # F134: one text encoder per server, shared by `/api/search` and the `kind='query'`
    # album — loading CLIP twice for the two halves of the same feature would cost a
    # minute and a gigabyte for nothing.
    query_encoder = _LazyTextEncoder(cfg)

    def _resolve_query_lang(raw_values: list[str] | None) -> i18n.Lang:
        """`?lang=` from the query -> a valid code (ru/en/ja), otherwise `default_lang`
        (F39: an invalid/absent lang does not crash, just the config default)."""
        raw = (raw_values or [""])[0].strip().lower()
        if raw in _UI_LANGS:
            return raw  # type: ignore[return-value]
        return default_lang

    def _index_html_for(lang: i18n.Lang) -> bytes:
        html = _index_html_cache.get(lang)
        if html is None:
            html = _render_index_html(lang).encode("utf-8")
            _index_html_cache[lang] = html
        return html

    class Handler(BaseHTTPRequestHandler):
        server_version = "SortaUI/1"

        def log_message(self, fmt: str, *args: object) -> None:
            _log.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler contract)
            parts = urlsplit(self.path)
            path = parts.path
            if path == "/":
                self._serve_index(parse_qs(parts.query))
            elif path == "/api/plan":
                self._serve_plan(parse_qs(parts.query))
            elif path == "/api/dupes":
                self._serve_dupes()
            elif path == "/api/moves":
                self._serve_moves(parse_qs(parts.query))
            elif path == "/api/clusters":
                self._serve_clusters()
            elif path == "/api/events":
                self._serve_events()
            elif path == "/api/junk":
                self._serve_junk(parse_qs(parts.query))
            elif path == "/api/animals":
                self._serve_animals(parse_qs(parts.query))
            elif path == "/api/face-slices":
                self._serve_face_slices(parse_qs(parts.query))
            elif path == "/api/review":
                self._serve_review(parse_qs(parts.query))
            elif path == "/api/restore/offer":
                self._serve_restore_offer(parse_qs(parts.query))
            elif path == "/api/search":
                self._serve_search(parse_qs(parts.query))
            elif path == "/api/saved-slices":
                self._serve_saved_slices(parse_qs(parts.query))
            elif path == "/api/places/search":
                self._serve_places_search(parse_qs(parts.query))
            elif path == "/api/process/status":
                self._serve_process_status()
            elif path == "/api/process/defaults":
                self._send_json(_process_defaults_payload(cfg))
            elif path == "/api/process/estimate":
                self._send_json(_process_estimate_payload(cfg, db_path))
            elif path == "/api/config":
                self._send_json({"language": i18n.normalize_lang(cfg.raw.get("language"))})
            elif path == "/api/settings":
                self._send_json(_settings_payload(cfg))
            elif path == "/api/env":
                self._send_json(_env_payload())
            elif path == "/api/startup":
                # F227: the one route that is answered while the program is still getting
                # ready, so it reads nothing but a lock — the page polls it every half
                # second and the answer must not wait behind the launch it describes.
                self._send_json(_startup_payload())
            elif path == "/api/sort/status":
                self._serve_sort_status()
            elif path == "/api/undo/status":
                self._send_json(undo_state.snapshot())
            elif path == "/api/sort/suggest-dest":
                self._send_json({"dest": _suggested_sort_dest(cfg, db_path)})
            elif path == "/api/relocate/suggest":
                self._send_json({"old_prefix": _indexed_prefix(db_path)})
            elif path == "/api/sort/summary":
                self._serve_sort_summary(parse_qs(parts.query))
            elif path == "/api/tabs/visibility":
                self._send_json(_tabs_visibility_payload(db_path, cfg))
            elif path == "/api/overview":
                # F108: plain aggregates, computed per request. The plan cache is not
                # touched on purpose — building a layout here would cost minutes.
                self._send_json(_overview_payload(db_path, cfg))
            elif path == "/api/cache":
                self._send_json(_cache_payload(db_path))
            elif path == "/api/source-tree":
                self._serve_source_tree(parse_qs(parts.query))
            elif path == "/api/source-tree/excludes":
                self._serve_source_excludes(parse_qs(parts.query))
            elif path.startswith("/thumb/"):
                self._serve_thumb(path[len("/thumb/"):])
            elif path.startswith("/preview/"):
                self._serve_preview(path[len("/preview/"):])
            elif path.startswith("/frame/"):
                self._serve_frame(path[len("/frame/"):])
            elif path.startswith("/photo/"):
                self._serve_photo(path[len("/photo/"):])
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _anything_running(self) -> bool:
            """Is a pipeline, a layout or an undo in flight? — the busy state, one place."""
            return bool(process_state.snapshot()["running"]
                        or sort_state.snapshot()["running"]
                        or undo_state.snapshot()["running"])

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler contract)
            path = urlsplit(self.path).path
            # F208: before the route, before the body, before the busy guard — the
            # question here is WHO is asking, and a refused request must not have read a
            # single byte of what it wanted done.
            refusal = _post_refusal(self.headers.get("Content-Type"),
                                    self.headers.get("Origin"),
                                    self.headers.get("Host"))
            if refusal is not None:
                self._send_json({"error": "request refused", "reason": refusal,
                                 "detail": _POST_REFUSAL_DETAIL[refusal]},
                                status=HTTPStatus.FORBIDDEN)
                return
            if path not in _BUSY_GUARDED_ROUTES:
                self._dispatch_post(path)
                return
            # F145: held for the whole write and not just for the check — otherwise a run
            # could start in the window between the two and the very race this closes
            # would be back, one request narrower.
            with busy_lock:
                if self._anything_running():
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                self._dispatch_post(path)

        def _dispatch_post(self, path: str) -> None:
            if path == "/api/dupes/choice":
                self._handle_dupes_choice()
            elif path == "/api/dupes/choices":
                self._handle_dupes_choices()
            elif path == "/api/dupes/skip":
                self._handle_dupes_skip()
            elif path == "/api/dupes/trash":
                self._handle_dupes_trash()
            elif path == "/api/review/mark":
                self._handle_review_mark()
            elif path == "/api/review/restore":
                self._handle_review_restore()
            elif path == "/api/animals/mark":
                self._handle_animal_mark()
            elif path == "/api/photo/trash":
                self._handle_photo_trash()
            elif path == "/api/photos/trash":
                self._handle_photos_trash()
            elif path == "/api/overrides":
                self._handle_overrides()
            elif path == "/api/place":
                self._handle_place()
            elif path == "/api/clusters/label":
                self._handle_cluster_label()
            elif path == "/api/clusters/merge":
                self._handle_cluster_merge()
            elif path == "/api/album":
                self._handle_album()
            elif path == "/api/process":
                self._handle_process_start()
            elif path == "/api/process/rerun-optional":
                self._handle_process_rerun_optional()
            elif path == "/api/process/cancel":
                self._handle_process_cancel()
            elif path == "/api/process/reset":
                self._handle_process_reset()
            elif path == "/api/cache/clear":
                self._handle_cache_clear()
            elif path == "/api/config/language":
                self._handle_set_language()
            elif path == "/api/settings":
                self._handle_save_settings()
            elif path == "/api/saved-slices/pin":
                self._handle_pin_saved_slice()
            elif path == "/api/saved-slices/unpin":
                self._handle_unpin_saved_slice()
            elif path == "/api/saved-slices/move":
                self._handle_move_saved_slice()
            elif path == "/api/relocate":
                self._handle_relocate()
            elif path == "/api/browse":
                self._handle_browse()
            elif path == "/api/source-tree/excludes":
                self._handle_save_source_excludes()
            elif path == "/api/sort":
                self._handle_sort_start()
            elif path == "/api/sort/cancel":
                self._handle_sort_cancel()
            elif path == "/api/undo":
                self._handle_undo_start()
            elif path == "/api/undo/cancel":
                self._handle_undo_cancel()
            elif path == "/api/quit":
                self._handle_quit()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_index(self, query: dict[str, list[str]]) -> None:
            lang = _resolve_query_lang(query.get("lang"))
            self._send_bytes(_index_html_for(lang), "text/html; charset=utf-8")

        def _serve_plan(self, query: dict[str, list[str]]) -> None:
            # F70: two shapes on one route — without `category` an aggregate over the
            # target folders (kilobytes), with it one bounded page of that folder.
            # The full plan is not reachable from here anymore, by design.
            mode = (query.get("mode") or [""])[0]
            category = (query.get("category") or [""])[0]
            if not category:
                payload = cache.aggregate(mode)
            else:
                window = _parse_page_window(query)
                if window is None:
                    self._send_json({"error": "invalid offset/limit"},
                                    status=HTTPStatus.BAD_REQUEST)
                    return
                payload = cache.page(mode, category, window[0], window[1])
            if payload is None:
                self._send_json({"error": f"unsupported mode: {mode!r}"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)

        def _serve_dupes(self) -> None:
            self._send_json(_dupes_payload(db_path, cfg.index.phash_max_distance))

        def _serve_moves(self, query: dict[str, list[str]]) -> None:
            raw_batch = (query.get("batch") or [""])[0]
            batch_id = None
            if raw_batch:
                try:
                    batch_id = int(raw_batch)
                except ValueError:
                    batch_id = None
            self._send_json(_moves_payload(db_path, batch_id))

        def _serve_clusters(self) -> None:
            self._send_json(_clusters_payload(db_path))

        def _serve_events(self) -> None:
            self._send_json(_events_payload(db_path))

        def _serve_junk(self, query: dict[str, list[str]]) -> None:
            # F103: read-only. Nothing here reclassifies anything — the correction is a
            # POST to the existing /api/overrides.
            parsed = _parse_junk_query(query)
            if parsed is None:
                self._send_json({"error": "invalid offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            bucket, offset, limit = parsed
            # F133: read the sensitive set off the LIVE config, not a snapshot — the
            # settings panel can change `vlm.exclude_classes` without a restart, and a
            # privacy list that needs one is not a privacy list.
            self._send_json(_junk_payload(
                db_path, cfg, bucket, offset, limit,
                frozenset(cfg.vlm.exclude_classes)))

        def _serve_animals(self, query: dict[str, list[str]]) -> None:
            # F123: read-only, like /api/junk. The actions this tab offers are elsewhere
            # — gathering an album is the existing POST /api/album with kind='animal',
            # and correcting a mark is POST /api/animals/mark (F124, `manual_pet`).
            window = _parse_page_window(query)
            if window is None:
                self._send_json({"error": "invalid offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            offset, limit = window
            # F137: the thresholds off the LIVE config, for the reason `/api/junk` reads
            # its sensitive classes off it — the settings panel edits `pet_threshold`
            # without a restart, and a threshold that needs one is not a threshold.
            self._send_json(_animals_payload(db_path, cfg, offset, limit))

        def _serve_face_slices(self, query: dict[str, list[str]]) -> None:
            # F152: read-only. Nothing about these slices is decided here — the rules are
            # `sorter.face_slice_ids_sql`, and the one action they offer is the existing
            # POST /api/album with kind='people'|'group'|'portrait'. The sensitive classes
            # come off the LIVE config for the reason /api/junk reads them that way.
            parsed = _parse_face_slice_query(query)
            if parsed is None:
                self._send_json({"error": "invalid slice/offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            slice_, offset, limit = parsed
            self._send_json(_face_slices_payload(
                cfg, db_path, slice_, offset, limit,
                frozenset(cfg.vlm.exclude_classes)))

        def _serve_review(self, query: dict[str, list[str]]) -> None:
            # F126: read-only. The only write of this workspace is the decision itself
            # (`POST /api/review/mark` -> `dedup_choice`), and there is deliberately no
            # route that marks a whole slice: sharpness ranks frames, it does not
            # classify them, so "delete everything below X" would delete photographs
            # nobody looked at.
            parsed = _parse_review_query(query)
            if parsed is None:
                self._send_json({"error": "invalid slice/offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            slice_, offset, limit, beyond = parsed
            self._send_json(_review_payload(
                db_path, slice_, offset, limit, beyond=beyond,
                features=cfg.features,
                max_distance=cfg.index.phash_max_distance))

        def _serve_restore_offer(self, query: dict[str, list[str]]) -> None:
            # F168: read-only — what the expanded frame affords, asked once per frame the
            # person opens. It writes nothing and loads no model: the answer is a row of
            # the index plus the header of the file, so opening a photograph costs the
            # same as it did. The sensitive classes come off the LIVE config for the
            # reason `/api/junk` reads them that way.
            file_id = _parse_file_id_query(query)
            if file_id is None:
                self._send_json({"error": "invalid file_id"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            payload = _restore_offer(db_path, cfg.features, file_id,
                                     frozenset(cfg.vlm.exclude_classes))
            if payload is None:
                self._send_json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)

        def _serve_search(self, query: dict[str, list[str]]) -> None:
            # F134: read-only, and read-only in the strong sense — an empty `q` asks for
            # the state of the index alone and never reaches the model. The sensitive
            # classes come off the LIVE config for the reason `/api/junk` does that.
            parsed = _parse_search_query(query, cfg.features.search_page)
            if parsed is None:
                self._send_json({"error": "invalid offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            text, offset, limit, words = parsed
            self._send_json(_search_payload(cfg, db_path, text, offset, limit,
                                            encoder=query_encoder, words=words))

        def _serve_saved_slices(self, query: dict[str, list[str]]) -> None:
            # F151: read-only, and the slices come off the LIVE config for the reason
            # `/api/junk` reads its sensitive classes that way — `features.saved_slices`
            # is edited to retune a slice, and a pin that needs a restart to change is a
            # pin nobody will retune. Without `slice` nothing is ranked and no model is
            # loaded: that call is how the pin row is built.
            parsed = _parse_saved_slice_query(cfg, query, cfg.features.search_page)
            if parsed is None:
                self._send_json({"error": "invalid slice/offset/limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            name, offset, limit = parsed
            self._send_json(_saved_slices_payload(cfg, db_path, name, offset, limit,
                                                  encoder=query_encoder))

        def _serve_places_search(self, query: dict[str, list[str]]) -> None:
            # F85c: read-only, bundled data only. `?lang=` decides the language of the
            # LABELS; the search itself always tries all three, because a place is
            # looked up by the name the user knows it under.
            lang = _resolve_query_lang(query.get("lang"))
            raw = (query.get("q") or [""])[0]
            self._send_json(_places_search_payload(raw, lang))

        def _read_json_body(self) -> object | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return None
            if length <= 0:
                return None
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return None

        def _handle_dupes_choice(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, keep = parsed
            if keep is None or keep not in group:
                self._send_json({"error": "keep_file_id must be in group"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            _apply_choice(db_path, group, keep)
            self._send_json({"ok": True})

        def _handle_dupes_choices(self) -> None:
            parsed = _validate_batch_choices_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            groups, skip = parsed
            saved = _apply_batch_choices(db_path, groups, skip)
            self._send_json({"saved": saved})

        def _handle_dupes_skip(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, _keep = parsed
            _skip_group(db_path, group)
            self._send_json({"ok": True})

        def _handle_dupes_trash(self) -> None:
            parsed = _validate_group_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            group, keep = parsed
            if keep is None or keep not in group:
                self._send_json({"error": "keep_file_id must be in group"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            trashed, refused = _trash_group(db_path, group, keep)
            self._send_json({"trashed": trashed, "refused": refused})

        def _handle_review_mark(self) -> None:
            # F126: a soft mark and nothing else — the same `dedup_choice` the
            # duplicates half writes, so the sorter keeps its single deletion path.
            parsed = _validate_review_mark_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            ids, action = parsed
            marked = _apply_review_mark(db_path, ids, action)
            self._send_json({"ok": True, "marked": marked})

        def _handle_review_restore(self) -> None:
            # F149: ONE id, and the validator is where that is enforced — a body carrying
            # a list has no shape here at all, so there is no bulk route to find.
            file_id = _validate_file_id_payload(self._read_json_body())
            if file_id is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            # F168: the private classes come off the LIVE config, for the reason
            # `/api/junk` reads them that way — the settings panel can change
            # `vlm.exclude_classes` without a restart, and a guard that needed one would
            # be a guard the person thinks they have turned on.
            payload = _restore_frame(db_path, cfg.features, file_id,
                                     frozenset(cfg.vlm.exclude_classes))
            if payload.get("error") == "file not found":
                self._send_json(payload, status=HTTPStatus.NOT_FOUND)
                return
            # A model that will not load is not a bad request: the answer is 200 with a
            # reason the interface can say out loud, which is the whole point of it being
            # a code and not a stack trace.
            self._send_json(payload)

        def _handle_animal_mark(self) -> None:
            # F124: a row in `manual_pet` and nothing else — no file is touched, no
            # `frame_quality` row is rewritten (that table has one writer, `junk`).
            parsed = _validate_animal_mark_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            ids, action = parsed
            self._send_json({"ok": True,
                             **_apply_animal_mark(db_path, cfg, ids, action)})

        def _handle_photo_trash(self) -> None:
            file_id = _validate_file_id_payload(self._read_json_body())
            if file_id is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            # F241: a refusal is not a failed request — 200 with both halves named, so
            # the page can say what stayed instead of guessing behind a 500.
            trashed, refused = _trash_files(db_path, [file_id])
            self._send_json({"trashed": trashed, "refused": refused})

        def _handle_photos_trash(self) -> None:
            # bulk deletion of the selected (the shared _trash_files path, same as single)
            ids = _validate_file_ids_payload(self._read_json_body())
            if ids is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            trashed, refused = _trash_files(db_path, ids)
            self._send_json({"trashed": trashed, "refused": refused})

        def _handle_overrides(self) -> None:
            # F77: marks only — nothing is moved on disk here (the physical move is the
            # shared sort --apply). The plan cache is deliberately NOT invalidated: the
            # mark is served live by PlanCache, and a rebuild per click would cost the
            # whole mode (F70).
            body = self._read_json_body()
            # F203: the target is TYPED now — a folder the plan does not contain is a
            # legitimate answer — so a bad one comes back with a REASON instead of being
            # stored and dropped hours later by the sorter's own check. Asked of the raw
            # body, before its shape is validated, so `../../evil` earns the sentence
            # about leaving the sort root and not the flat "invalid body".
            refusal = _reassign_target_refusal(body)
            if refusal is not None:
                self._send_json({"error": "invalid target", "reason": refusal},
                                status=HTTPStatus.BAD_REQUEST)
                return
            parsed = _validate_overrides_payload(body)
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            file_ids, action, target = parsed
            applied = _apply_overrides(db_path, file_ids, action, target)
            self._send_json({"ok": True, "action": action, "target": target,
                             "file_ids": applied})

        def _handle_place(self) -> None:
            # F85c: unlike an F77 correction, this one changes the target FOLDER of
            # every file of the group, so the built plan is now stale — the cache is
            # dropped and the next request rebuilds it. Nothing is moved on disk here
            # either: the assignment is a row in the index, the layout is still the
            # shared `sort --apply`.
            parsed = _validate_place_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            kind, selector, action, place, include_gps = parsed
            result = _apply_bulk_place(db_path, kind, selector, action, place, include_gps)
            if result["affected"]:
                conn = _connect(db_path)
                try:
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            self._send_json(result)

        def _handle_cluster_label(self) -> None:
            parsed = _validate_cluster_label_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            cluster_id, name = parsed
            name = name.strip()
            if not name:
                self._send_json({"error": "name must not be empty"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            conn = _connect(db_path)
            try:
                root = faces.label_cluster(conn, cluster_id, name)
            except ValueError:
                self._send_json({"error": "cluster not found"}, status=HTTPStatus.NOT_FOUND)
                return
            finally:
                conn.close()
            self._send_json({"ok": True, "cluster_id": root, "label": name})

        def _handle_cluster_merge(self) -> None:
            parsed = _validate_cluster_merge_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            src, dst = parsed
            conn = _connect(db_path)
            try:
                root = faces.merge(conn, src, dst)
            except ValueError:
                self._send_json({"error": "cluster not found"}, status=HTTPStatus.NOT_FOUND)
                return
            finally:
                conn.close()
            self._send_json({"ok": True, "cluster_id": root})

        def _handle_album(self) -> None:
            body = self._read_json_body()
            # F139/F133/F193: a class with no album is refused here rather than in the
            # markup — a button the page does not draw is not a rule, and a request sent
            # past the interface would gather the folder all the same. `plan_album`
            # refuses a sensitive class a second time, for the terminal; this end answers
            # with a status and a REASON the interface can put into a sentence.
            #
            # Asked of the raw body, before the shape of it is validated at all: that is
            # what lets `document` come back with "documents are never gathered" instead
            # of the "invalid body" a name outside `ALBUM_KINDS` would otherwise earn —
            # the whole complaint about that bucket was that the program said nothing.
            refusal = class_album_refusal(cfg, body.get("kind")
                                          if isinstance(body, dict) else None)
            if refusal is not None:
                self._send_json({"error": "album refused", "reason": refusal},
                                status=HTTPStatus.FORBIDDEN)
                return
            parsed = _validate_album_payload(body)
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            kind, selector, mode, where, name, apply_, dest_str = parsed
            # F193: the frames a person ticked, for every kind alike. An empty selection
            # is a refusal with a reason, never an album of nothing — see `album_selection`.
            file_ids, selection_error = album_selection(body)
            if selection_error is not None:
                self._send_json({"error": "invalid selection",
                                 "reason": selection_error},
                                status=HTTPStatus.BAD_REQUEST)
                return
            dest = Path(dest_str) if dest_str else _album_dest(cfg, db_path)
            conn = _connect(db_path)
            try:
                # F134: `encoder` is the server's own text tower and is ignored by every
                # kind but `query` — without it `plan_album` would load a second copy of
                # CLIP for an album the search line has already ranked.
                report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                    where=where, apply=apply_, album_name=name,
                                    encoder=query_encoder, file_ids=file_ids)
            except EmbeddingsMissing as exc:
                # The button is only offered while the index is searchable, so this is a
                # race (a run emptied the table in between) — and it answers with the
                # REASON, never with an album of zero files, which would read as "your
                # collection holds none of these" (F134).
                self._send_json({"error": "search unavailable", "reason": exc.reason},
                                status=HTTPStatus.CONFLICT)
                return
            finally:
                conn.close()
            self._send_json(_album_report_to_json(report, apply_))

        def _serve_process_status(self) -> None:
            self._send_json(process_state.snapshot())

        def _handle_process_start(self) -> None:
            parsed = _validate_process_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            source_dir, options = parsed
            # F248: still a refusal and still no walk of a folder that is not there —
            # what changed is that it says WHICH of the two things happened, and the
            # moved one arrives with the way out attached.
            refusal = _source_refusal(db_path, cfg, source_dir)
            if refusal is not None:
                self._send_json(refusal, status=HTTPStatus.BAD_REQUEST)
                return
            # F43/F45: sort (layout apply) and process — both heavy operations
            # write/move files; the shared busy_lock makes the "the other is not
            # running" check + its own try_start an atomic critical section (otherwise
            # a TOCTOU between two parallel POSTs).
            with busy_lock:
                if sort_state.snapshot()["running"]:
                    self._send_json({"error": "sort is running"}, status=HTTPStatus.CONFLICT)
                    return
                if undo_state.snapshot()["running"]:
                    self._send_json({"error": "undo is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not process_state.try_start(source_dir):
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_pipeline,
                args=(db_path, cfg, source_dir, process_state, cache, options),
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_process_rerun_optional(self) -> None:
            # F62/F63: "Re-run selected" — the same _ProcessState/busy_lock as
            # /api/process; no source_dir from the client — indexing is not
            # overridden (_run_pipeline(source_dir=None) leaves cfg.sources).
            # deep -> junk with the VLM (naming.vlm_enabled=deep); F123: pets -> the
            # same junk stage with features.pets, and both together are one run of it.
            # F135: the web app no longer has a button for this — "Start" runs the
            # whole pipeline and the stages skip what is done. The ROUTE stays: it is
            # documented and callable from outside, and dropping a public endpoint is
            # a decision of its own, not a side effect of tidying up the markup.
            parsed = _validate_rerun_optional_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            faces, events, deep, pets = parsed
            # No "index is empty" guard here on purpose: re-running the optional
            # stages over nothing is a no-op, not a hazard — unlike a layout over a
            # half-built index or a reset mid-run, which the server does refuse.
            # "Pointless" is not the same answer as "dangerous".
            with busy_lock:
                if sort_state.snapshot()["running"]:
                    self._send_json({"error": "sort is running"}, status=HTTPStatus.CONFLICT)
                    return
                if undo_state.snapshot()["running"]:
                    self._send_json({"error": "undo is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not process_state.try_start(""):
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_pipeline,
                args=(db_path, cfg, None, process_state, cache,
                      _RunOptions(faces=faces, events=events, deep=deep, pets=pets)),
                kwargs={"only_optional": True},
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_process_cancel(self) -> None:
            process_state.request_cancel()
            self._send_json({"ok": True})

        def _handle_process_reset(self) -> None:
            # F93: the checkbox of the reset dialog rides in the body. Absent/garbage
            # body -> False, i.e. the geo cache survives: the destructive branch has to
            # be asked for explicitly, never fallen into.
            payload = self._read_json_body()
            clear_geo = bool(payload.get("clear_geo")) if isinstance(payload, dict) else False
            # F45: the reset also writes to the DB — hold busy_lock for the whole
            # reset, not just the check, otherwise sort/process could start in the
            # window between the check and db.reset_index itself.
            with busy_lock:
                if self._anything_running():
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
                conn = _connect(db_path)
                try:
                    db.reset_index(conn, clear_geo=clear_geo)
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            self._send_json({"ok": True, "clear_geo": clear_geo})

        def _handle_cache_clear(self) -> None:
            # F94: the button next to the size on the "Process" tab. The clearing
            # itself belongs to imaging/geo — this only decides that it may happen now.
            target = _validate_cache_clear_payload(self._read_json_body())
            if target is None:
                self._send_json({"error": "invalid target"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            # The same busy_lock guard as /api/process/reset, for the same reason:
            # mid-run a geo clear sends the rest of the stage back to the network, and
            # a preview clear deletes the frames that stage is writing right now.
            with busy_lock:
                if self._anything_running():
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                if target == "geo":
                    conn = _connect(db_path)
                    try:
                        removed = clear_geo_cache(conn)
                    finally:
                        conn.close()
                else:
                    # Counted before the removal: `preview_cache_clear` reports
                    # nothing, and "freed nothing" has to be distinguishable from
                    # "freed 12 GB" in the status line.
                    removed, freed = _sum_dir(imaging.preview_dir())
                    imaging.preview_cache_clear()
                    _log.info("preview cache cleared: %d files, %d bytes", removed, freed)
                payload = _cache_payload(db_path)
            self._send_json({"ok": True, "target": target, "removed": removed,
                             "cache": payload})

        def _handle_set_language(self) -> None:
            # F65: the "Folder language" selector — sets the OUTPUT language (folders/
            # names) for the plan preview and apply, separate from the interface `?lang`.
            # Persists into config.yaml (if known) so it survives restarts and CLI runs.
            lang = _validate_language_payload(self._read_json_body())
            if lang is None:
                self._send_json({"error": "invalid language"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            # hold busy_lock: the rebuild must not race a running sort/process that
            # reads cfg (the same guard as /api/process/reset).
            with busy_lock:
                if self._anything_running():
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                cfg.raw["language"] = lang
                cfg.language = lang
                if config_path is not None:
                    try:
                        save_language(config_path, lang)
                    except OSError as exc:
                        self._send_json({"error": f"could not save config: {exc}"},
                                        status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                conn = _connect(db_path)
                try:
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            self._send_json({"ok": True, "language": lang})

        def _handle_save_settings(self) -> None:
            # F104: the settings column. Modelled on _handle_set_language above — the
            # running cfg is changed first, then the file, under the same busy_lock.
            # Changing the model or the frame size in the middle of a classification is
            # not a setting but an accident, hence the 409: what the run would then be
            # doing is neither what the file says nor what the user saw.
            values = _validate_settings_payload(self._read_json_body())
            if values is None:
                self._send_json({"error": "invalid settings"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            with busy_lock:
                if self._anything_running():
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                # F210: what `vlm.exclude_classes` held BEFORE the save, so a class that
                # has just entered the list can be swept. Read here rather than after,
                # because _apply_settings replaces the whole dataclass.
                before = tuple(cfg.vlm.exclude_classes)
                _apply_settings(cfg, values)
                if config_path is not None:
                    try:
                        for key, value in values.items():
                            save_setting(config_path, key, value)  # type: ignore[arg-type]
                    except OSError as exc:
                        self._send_json({"error": f"could not save config: {exc}"},
                                        status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                # F210: turning the protection on has to reach the previews already on
                # disk — otherwise it covers only frames classified from now on and the
                # whole archive of documents stays in the cache. Inside busy_lock: no
                # stage is running (checked above), so nothing is writing previews back
                # while they are being removed. The rule itself lives in junk.py.
                conn = _connect(db_path)
                try:
                    sweep_previews_for_new_classes(
                        conn, before, cfg.vlm.exclude_classes)
                finally:
                    conn.close()
            self._send_json({"ok": True, "settings": _settings_payload(cfg)})

        # --- F156: the pinned queries a person makes for themselves ------------------

        def _pins_json(self) -> dict:
            """The answer every one of the three writes below sends back.

            The WHOLE list rather than the one entry that changed: the pin row is redrawn
            from it, and a client that patched its own copy would be the second place the
            order is decided — which is how a row on screen starts disagreeing with the
            file that holds it.
            """
            return {
                "ok": True,
                "slices": [{"slice": s.name, "queries": list(s.queries)}
                           for s in cfg.features.saved_slices],
                "max_pinned": int(cfg.features.max_pinned_slices),
            }

        def _write_saved_slices(self, slices: tuple[SavedSlice, ...]) -> bool:
            """Apply the new pin list and persist it; False -> the answer is already sent.

            The running config is changed first and rolled BACK if the file cannot be
            written, which is the one place this differs from `_handle_save_settings`: a
            pin exists to survive a restart, so a pin that applied and did not save would
            be a promise the next start breaks. Without a `config_path` (a server started
            without a config file) the pin still applies for this session — nothing else
            can be offered there, and refusing would take the feature away from it.
            """
            previous = cfg.features.saved_slices
            _apply_saved_slices(cfg, slices)
            if config_path is None:
                return True
            try:
                save_saved_slices(config_path, slices)
            except OSError as exc:
                _apply_saved_slices(cfg, previous)
                self._send_json({"error": f"could not save config: {exc}"},
                                status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return False
            return True

        def _handle_pin_saved_slice(self) -> None:
            # The query being pinned is NOT run here and no model is loaded: pinning saves
            # words, and the ranking happens when the slice is opened, exactly as it does
            # for the slices that ship in the config file.
            parsed = _validate_pin_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "empty query", "reason": _PIN_EMPTY},
                                status=HTTPStatus.BAD_REQUEST)
                return
            name, query = parsed
            result = _pinned_with(cfg, name, query)
            if isinstance(result, str):
                # The refusal is spelled out — a limit that is reached in silence looks
                # exactly like a button that does not work.
                self._send_json({"error": f"cannot pin: {result}", "reason": result,
                                 "max_pinned": int(cfg.features.max_pinned_slices)},
                                status=HTTPStatus.BAD_REQUEST)
                return
            if self._write_saved_slices(result):
                self._send_json({**self._pins_json(), "slice": name})

        def _handle_unpin_saved_slice(self) -> None:
            # Removes a config entry. No file is touched, no row is deleted, and the
            # frames the slice ranked are where they were — the confirmation the interface
            # asks for says exactly that, because "remove the slice" and "remove the
            # photographs" are one word apart in every language this speaks.
            name = _validate_slice_name_payload(self._read_json_body())
            slices = _pinned_without(cfg, name) if name is not None else None
            if slices is None:
                self._send_json({"error": "unknown slice"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            if self._write_saved_slices(slices):
                self._send_json(self._pins_json())

        def _handle_move_saved_slice(self) -> None:
            parsed = _validate_move_payload(self._read_json_body())
            slices = _pinned_moved(cfg, *parsed) if parsed is not None else None
            if slices is None:
                self._send_json({"error": "unknown slice"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            if self._write_saved_slices(slices):
                self._send_json(self._pins_json())

        def _handle_browse(self) -> None:
            path, problem = _browse_for_folder()
            payload = {"path": path}
            if problem:
                # A picker this machine cannot draw is not a cancel, and the button may
                # not answer both the same way: on Ubuntu without python3-tk it looked
                # like nothing happened at all.
                payload["problem"] = problem
            self._send_json(payload)

        def _handle_relocate(self) -> None:
            # F244: the engine is `relocate.relocate` and every decision about the move
            # is inside it — this end validates a body, chooses a status and drops the
            # plan cache, which is all that is left over.
            parsed = _validate_relocate_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            old_prefix, new_prefix, apply_ = parsed
            payload = _relocate_payload(db_path, old_prefix, new_prefix, apply=apply_)
            if payload.get("applied"):
                # Every built plan is about the old paths, so it is dropped here rather
                # than left to expire: the next request rebuilds it (F70, lazily).
                conn = _connect(db_path)
                try:
                    cache.rebuild(cfg, conn)
                finally:
                    conn.close()
            # A refusal keeps the 200: nothing was written, the answer says why, and the
            # page has a sentence to show instead of a failed request to explain.
            self._send_json(payload)

        # --- F81/F82: "do not scan" / "do not lay out" (the source block) --------

        def _serve_source_tree(self, query: dict[str, list[str]]) -> None:
            root = _validate_tree_root((query.get("path") or [""])[0])
            if root is None:
                self._send_json({"error": "not a directory"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            excludes = load_excludes(excludes_path(cfg))
            self._send_json(_source_tree_payload(
                root, sorted(excludes.for_root(root)),
                sorted(excludes.layout_for_root(root))))

        def _serve_source_excludes(self, query: dict[str, list[str]]) -> None:
            root = _validate_tree_root((query.get("path") or [""])[0])
            if root is None:
                self._send_json({"error": "not a directory"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(_excludes_payload(cfg, root))

        def _handle_save_source_excludes(self) -> None:
            # Writes the exclusion file, nothing else: the rows already indexed under
            # a new "do not scan" are dropped by the next `index()` run (indexer, §3),
            # not from here — one place decides what "not in the index" means.
            parsed = _validate_excludes_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            raw_root, values, layout = parsed
            root = _validate_tree_root(raw_root)
            if root is None:
                self._send_json({"error": "not a directory"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            rejected = [v for v in values + layout if isinstance(v, str)
                        and normalize_exclude(v) is None]
            try:
                save_excludes_file(excludes_path(cfg), root, values, layout)
            except OSError as exc:
                self._send_json({"error": f"could not save excludes: {exc}"},
                                status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            payload = _excludes_payload(cfg, root)
            payload["ok"] = True
            payload["rejected"] = rejected
            self._send_json(payload)

        def _serve_sort_status(self) -> None:
            self._send_json(sort_state.snapshot())

        def _serve_sort_summary(self, query: dict[str, list[str]]) -> None:
            # F104: what the confirmation dialog states before a layout starts. `dest`
            # comes from the form field, so the "already in the destination" numbers
            # are about the folder the user is actually about to write into.
            dest = (query.get("dest") or [""])[0].strip()
            # F192: `by` is the criterion the tab is showing (`sorter.MODES`); absent —
            # "city", the only one this route could summarize before.
            by = (query.get("by") or ["city"])[0].strip() or "city"
            payload = cache.summary(by, _summary_dest(cfg, dest or None))
            if payload is None:  # an unsupported criterion
                self._send_json({"error": "no plan"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)

        def _handle_sort_start(self) -> None:
            parsed = _validate_sort_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            dest, mode, by = parsed
            # F45: see the comment in _handle_process_start — the same shared
            # busy_lock, the same "other running -> own try_start" order.
            with busy_lock:
                if process_state.snapshot()["running"]:
                    self._send_json({"error": "process is running"}, status=HTTPStatus.CONFLICT)
                    return
                if undo_state.snapshot()["running"]:
                    self._send_json({"error": "undo is running"}, status=HTTPStatus.CONFLICT)
                    return
                if not sort_state.try_start():
                    self._send_json({"error": "already running"}, status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_sort,
                args=(db_path, cfg, dest, mode, sort_state, cache, by),
                daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_sort_cancel(self) -> None:
            # F97: a flag, exactly like /api/process/cancel. The engine reads it
            # between files and closes the batch itself — nothing here waits for the
            # thread, so the button answers instantly even mid-copy of a large file.
            sort_state.request_cancel()
            self._send_json({"ok": True})

        def _handle_undo_start(self) -> None:
            # F97: the rollback changes paths of files on disk, so it may not run
            # alongside a layout or a pipeline — the same busy_lock and the same
            # "other running -> own try_start" order as /api/sort, both ways.
            with busy_lock:
                if process_state.snapshot()["running"]:
                    self._send_json({"error": "process is running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                if sort_state.snapshot()["running"]:
                    self._send_json({"error": "sort is running"},
                                    status=HTTPStatus.CONFLICT)
                    return
                if not undo_state.try_start():
                    self._send_json({"error": "already running"},
                                    status=HTTPStatus.CONFLICT)
                    return
            thread = threading.Thread(
                target=_run_undo, args=(db_path, cfg, undo_state, cache), daemon=True,
            )
            thread.start()
            self._send_json({"ok": True})

        def _handle_undo_cancel(self) -> None:
            undo_state.request_cancel()
            self._send_json({"ok": True})

        # --- F209: closing the program from the page it is served on -----------------

        def _running_now(self) -> str | None:
            """Which of the three long operations is in flight — by name, or None.

            `_anything_running` answers the busy guard's question, which is a yes/no;
            this one answers the person's. What they are about to lose has a name, and
            the question the interface asks has to use it.
            """
            if process_state.snapshot()["running"]:
                return "process"
            if sort_state.snapshot()["running"]:
                return "sort"
            if undo_state.snapshot()["running"]:
                return "undo"
            return None

        def _handle_quit(self) -> None:
            # The body carries the confirmation and nothing else. Absent/garbage -> no
            # confirmation: the branch that interrupts a run is entered on purpose or not
            # at all (the rule `/api/process/reset` applies to `clear_geo`).
            #
            # `is True` rather than the `bool(...)` the other bodies are read with, and
            # this is the one place worth the difference: every truthy value would make
            # `{"confirm": "no"}` end a five-hour pass. The contract is the literal JSON
            # `true`, which is what the page sends; anything else gets the 409 and the
            # question, which costs a client one more request and loses nothing.
            payload = self._read_json_body()
            confirm = isinstance(payload, dict) and payload.get("confirm") is True
            running = self._running_now()
            if running is not None and not confirm:
                self._send_json({"error": "already running",
                                 "reason": QUIT_RUN_IN_PROGRESS, "running": running},
                                status=HTTPStatus.CONFLICT)
                return
            # The existing flag, not a second way to stop a run: this is what
            # `/api/process/cancel` and its two siblings set, and the engines read it
            # where they already read it. The three are mutually exclusive, so exactly
            # one of these branches is the one in flight.
            if running == "process":
                process_state.request_cancel()
            elif running == "sort":
                sort_state.request_cancel()
            elif running == "undo":
                undo_state.request_cancel()
            self._send_json({"ok": True, "quitting": True, "cancelled": running})
            self._stop_server()

        def _stop_server(self) -> None:
            """End `serve_forever` — on a thread of its own, after the answer is out.

            `shutdown()` blocks until the serve loop has returned, and this handler is
            running ON one of that server's threads: calling it here would hold the reply
            hostage to the very thing it is announcing. The answer above is already
            written (the handler's `wfile` is unbuffered), so the socket carries it while
            the loop winds down.

            Nothing is killed. `serve` closes the server and the index connection in its
            own `finally` and returns to `cli`, exactly as it does on Ctrl+C — a run
            thread is a daemon and dies with the process either way, which is why the
            confirmation above is where the decision is made and not here.
            """
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _serve_thumb(self, raw_id: str) -> None:
            file_id = _parse_file_id(raw_id)
            path = self._resolve(raw_id)
            if file_id is None or path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = _thumb_bytes(file_id, path)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, "image/jpeg")

        def _serve_photo(self, raw_id: str) -> None:
            path = self._resolve(raw_id)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), ctype)

        def _serve_preview(self, raw_id: str) -> None:
            # a large DECODED JPEG for the lightbox: HEIC/RAW, which the browser does
            # not render from the raw /photo, arrive here as JPEG (decode_rgb).
            file_id = _parse_file_id(raw_id)
            path = self._resolve(raw_id)
            if file_id is None or path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = _preview_bytes(file_id, path)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, "image/jpeg")

        def _serve_frame(self, raw: str) -> None:
            # F80: `/frame/<file_id>/<index>` — one frame of a clip's filmstrip. The
            # path is resolved from the DB by file_id exactly as /thumb and /preview
            # do; no path ever comes in from outside. An index that the clip does not
            # have (a short clip, a photo, a number past the strip) is a 404, not a
            # 500 — the lightbox uses it to find out how long the strip really is.
            raw_id, _, raw_index = raw.partition("/")
            file_id = _parse_file_id(raw_id)
            index = _parse_file_id(raw_index)
            path = self._resolve(raw_id)
            if file_id is None or index is None or index < 0 or path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = _preview_bytes(file_id, path, frame=index)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, "image/jpeg")

        def _resolve(self, raw_id: str) -> Path | None:
            """file_id (integer only) -> the path from files; otherwise None.

            A non-numeric/arbitrary segment (incl. with `../`) does not parse into an
            id and never reaches an FS read — the only path to a file is via
            SELECT path FROM files WHERE id = ?.
            """
            file_id = _parse_file_id(raw_id)
            if file_id is None:
                return None
            return _resolve_path(db_path, file_id)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_server(cfg: Config, conn: sqlite3.Connection, *,
                 port: int = DEFAULT_PORT,
                 config_path: str | Path | None = None) -> ThreadingHTTPServer:
    """Build (but do not start) the server bound to 127.0.0.1:port.

    port=0 asks the OS to pick a free port (used by tests and able to report the
    real port via server.server_port). `config_path` — the config.yaml to persist the
    folder-language choice into (POST /api/config/language); None disables the write
    (the running cfg is still updated in memory).
    """
    dest = Path(cfg.database).resolve().parent / "_sorta_ui_preview"
    cache = PlanCache(cfg, conn, dest)
    process_state = _ProcessState()
    sort_state = _SortState()
    undo_state = _UndoState()
    busy_lock = threading.Lock()
    handler_cls = _make_handler(Path(cfg.database).resolve(), cache, cfg,
                                process_state, sort_state, busy_lock, undo_state,
                                config_path=config_path)
    return ThreadingHTTPServer(("127.0.0.1", port), handler_cls)


def serve(cfg: Config, conn: sqlite3.Connection, *,
         port: int = DEFAULT_PORT, open_browser: bool = True,
         config_path: str | Path | None = None) -> None:
    """Start the local read-only plan server and block until Ctrl+C or `POST /api/quit`.

    127.0.0.1 only. A busy port -> RuntimeError with a clear message (the caller
    cli.py decides how to show it to the user). `config_path` is threaded to the
    server so the folder-language selector can persist into config.yaml.

    F209: the two ways out end in the same `finally`, because they are the same exit —
    the button asks `serve_forever` to return, which is what Ctrl+C does. `conn` is
    closed here rather than left to the interpreter: the connection belongs to the thread
    that called this, and after this returns nobody is going to read the index through it.
    """
    log_environment()  # F69: one environment header per server start
    warn_if_geo_data_missing()  # F65: an unreadable geo base empties every place
    try:
        httpd = build_server(cfg, conn, port=port, config_path=config_path)
    except OSError as exc:
        raise RuntimeError(f"sorta ui: port {port} is busy or unavailable: {exc}") from exc
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(i18n.cli_text("cli.ui.serving", i18n.normalize_lang(cfg.language), url=url))
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        conn.close()
