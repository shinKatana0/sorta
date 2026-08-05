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

(9) `POST /api/sort` (F43, the "Cities" tab, the "Sort" button) — the real layout of
the collection: calls `sorter.plan_and_sort(cfg, conn, "city", dest, apply=True,
copy=..., progress=...)` on a background thread with its own sqlite connection (the
`_ProcessState`/`_run_pipeline` pattern, but its own `_SortState` — no stages, one
operation). The body `{"dest": str|null|"", "mode": "move"|"copy"}`: `dest` empty/null
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
folder-picker dialog and returns `{"path": str}` (an empty string on cancel/error/no
GUI — not a 500, the button is just a convenience, manual path entry always works).
The dialog — tkinter `askdirectory` in a SEPARATE subprocess (`_browse_for_folder`,
`subprocess.run([sys.executable, "-c", ...])`): tkinter is not thread-safe, and the
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
`sort --apply`; `reassign` — lay it out into `target` (a folder of the current plan,
relative to the sort root) instead of wherever the automatic rules put it; `photo`
(F103) — "the classifier is wrong, this IS a photo": the junk/document/product verdict
stops deciding the route and the file goes back to the automatic city layout; `clear` —
drop the correction. One row per file in `manual_overrides` (PRIMARY KEY file_id), a
repeated correction overwrites it. Like every other write route, the body carries only
ints and (for reassign) a target string — no paths from the client to a file: the
target is a folder INSIDE the layout, and `sorter._manual_target_parts` validates it
against the sort root before any destination is built from it. This endpoint moves
nothing on disk — the physical move happens in the shared `sort --apply`.
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

(18) `GET /api/sort/summary?dest=` (F104) — the numbers the pre-apply dialog states:
files, folders, volume, how much goes into the two review folders, and how much is
already lying in that destination (with how much of it will be skipped as an identical
copy — the F97 rule, asked of the same functions the apply uses). All of it is read off
the SAME built plan the "Cities" tree draws, so the dialog and the tab cannot disagree.

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

Security: the only entry to a file on disk for reading (`/thumb`, `/photo`) is a
file_id, resolved strictly via `SELECT path FROM files WHERE id = ?`. These routes
never accept a path directly from the request, so an arbitrary path (incl. `../..`)
does not resolve — a non-numeric/unknown id simply finds no row in files and answers
404. The write endpoints (`POST /api/dupes/*`, `POST /api/photo/trash`) also operate
only on a file_id from the JSON body (no paths from the client); before deleting a
`files` row or sending a path to the trash, the id is resolved by the same query
`SELECT ... FROM files WHERE id IN (...)` — unknown ids are silently ignored, not
substituted as a path. The server binds only to 127.0.0.1.

plan_and_sort (sorter, dry-run) — the single source of the plan; PlanCache calls it
with `write_reports=False` (no CSV/HTML side files from the UI path) and at most once
per mode per cache generation — LAZILY, on the first request for that mode (F70), so
neither the server start nor a `rebuild` blocks for the ~13 s a mode costs on a 26k
collection. `GET /api/plan?mode=` answers with a per-target-folder AGGREGATE
(folder -> count/size, kilobytes); the files of one folder come as an explicit page
(`&category=&offset=&limit=`), never as the whole 26k-element plan.
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

from . import db, faces, i18n, imaging, restore
from .config import (
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
from .dedup import (KEEPER_SOURCE_SHARPNESS, assign_duplicates, compute_phashes,
                    group_key, near_duplicate_groups, read_group_keepers)
from .detect import detector_settings
from .diagnostics import warn_if_geo_data_missing
from .events import build_events
from .faces import detect_and_cluster
from .geo import clear_geo_cache, geo_cache_size, resolve_places
from .geodata import GeoDataMissing, GeoResolver
from .indexer import excludes_path, index as run_index, load_excludes, normalize_exclude
# `_has_column`: "does this database have that column yet". The indexer reads its own
# optional columns through it, and the blur list (F157) reads F155's `face_sharpness`
# through the same one — the two features were merged in either order on purpose.
from .indexer import _has_column
from .indexer import save_excludes as save_excludes_file
from .junk import classify as classify_junk
from .junk import (
    CLASSIFY_PHASE_VLM,
    CLASSIFY_STAGE,
    VERDICTS_STAGE,
    faces_stage_ran,
    search_index_model,
    search_index_settings,
)
from .landmarks import Classifier, clip_classifier, detect_landmarks
from .landmarks import batched
from .naming import name_events, naming_settings
from .runlog import (
    Measurement,
    log_environment,
    measurement_files,
    measurement_unit,
    read_measurements,
    stage_timer,
)
from .search import (
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
from .sorter import (
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
from .sorter import _fs, _is_the_same_file

_log = logging.getLogger(__name__)

DEFAULT_PORT = 8756
_THUMB_MAX_EDGE = 200
_CLUSTER_SAMPLE_LIMIT = 6
_EVENT_SAMPLE_LIMIT = 8
_SUPPORTED_MODES = ("city", "person", "event")
_DEFAULT_ALBUM_DIRNAME = "_Альбомы"
# F70: `/api/plan` never serves a whole mode again — a category page is bounded by a
# default and a hard maximum, so no query can ask the server for 26k items at once.
_PLAN_PAGE_DEFAULT_LIMIT = 200
_PLAN_PAGE_MAX_LIMIT = 1000

# F39: UI switcher languages — the same three as i18n.Lang; self-names for the
# selector options (not translated — this is a language's name in that language).
_UI_LANGS: tuple[str, ...] = ("ru", "en", "ja")
_LANG_SELF_NAMES: dict[str, str] = {"ru": "Русский", "en": "English", "ja": "日本語"}

_ProgressCB = Callable[[int, "int | None"], None]  # (done, total|None) — compatible with progress.ProgressCB


def _plan_item_to_json(item: PlanItem,
                       override: tuple[str, str | None] | None = None) -> dict:
    # G3: `item.city` already comes in the folder language (sorter._city_display_name)
    # — the grid of the "Cities"/"Events" tabs must not label a frame «St Petersburg»
    # while the target folder right next to it reads «Санкт-Петербург». One function
    # decides both, so the plan and the card can never disagree.
    geo = "/".join(p for p in (item.country, item.city) if p) or None
    payload = {
        "file_id": item.file_id,
        "name": item.src.name,
        # Where the file came FROM. Only the basename used to reach the UI, yet the
        # source folder is often the best evidence there is about a frame: 41% of this
        # collection sits in hand-named directories ("Тайланд 04.2025",
        # "Турция. Белек") — a person's own labelling of place and date. It is also
        # what you need in order to judge a wrong guess: a Colosseum match is plainly
        # wrong once you can see the file lives under "рускеала".
        "src_dir": item.src.parent.name,
        "src_path": str(item.src.parent),
        "target_rel": item.target_rel,
        "reason": item.reason,
        "date": item.taken_at,
        "geo": geo,
        # F85c: how confidently the place was determined — and, for `manual`, that it
        # was not determined at all but chosen by the user. The grid draws its own mark
        # off this, so a hand-assigned place never reads as something the program found.
        "place_confidence": item.place_confidence,
        "category": item.reason,
        "thumb_url": f"/thumb/{item.file_id}",
        # F80: video and photo tiles used to be indistinguishable in the grid. The
        # extension is enough (the indexer decides media_type the same way) and costs
        # no query — the plan carries no media_type of its own.
        "video": imaging.is_video_path(item.src),
    }
    if override is not None:
        # F77: only a corrected file carries the mark — the frontend draws a frame off
        # the presence of the key, so an uncorrected row must not carry a null.
        payload["override"] = override[0]
        payload["override_target"] = override[1]
    return payload


def _plan_category(item: PlanItem) -> str:
    """The target FOLDER of a plan item — the aggregation key of `/api/plan` (F70).

    `target_rel` is POSIX and always carries at least one directory segment (see
    sorter._target_parts — every branch returns a non-empty folder list), so the key
    is never empty; a pathological item without a folder falls back to target_rel.
    """
    head, sep, _name = item.target_rel.rpartition("/")
    return head if sep else item.target_rel


def _dest_occupancy(items: list[PlanItem], dest: Path | None) -> tuple[int, int]:
    """(taken, identical) target paths of `items` inside `dest` — F104.

    `taken` — the plan item's target name is already occupied; `identical` — by a
    byte-for-byte copy of that very file, i.e. the apply will SKIP it (F97) instead of
    writing a `_1` twin next to it. The difference between the two numbers is the file
    that will be written after all, under another name.

    The rule is asked of `sorter._is_the_same_file` rather than re-implemented: the
    dialog states what the apply is going to do, and the moment the two can disagree
    the numbers stop being a promise. `dest=None` — the destination could not be
    resolved (see `_summary_dest`), so nothing is claimed about it.
    """
    if dest is None:
        return 0, 0
    taken = identical = 0
    for item in items:
        head, sep, _name = item.target_rel.rpartition("/")
        target_dir = dest.joinpath(*head.split("/")) if sep else dest
        # The name the apply TRIES first — the `_1` suffixes come after this one.
        target = target_dir / item.src.name
        if not _fs(target).exists():
            continue
        taken += 1
        # src == dst is the in-place layout: the file IS the one lying at the target,
        # and it is skipped just as surely as an identical copy would be.
        if (os.path.normcase(str(target)) == os.path.normcase(str(item.src))
                or _is_the_same_file(target, item.src, item.db_hash, item.db_algo)):
            identical += 1
    return taken, identical


def _overrides_map(db_path: Path) -> dict[int, tuple[str, str | None]]:
    """F77: file_id -> (action, target) from `manual_overrides` — the live marks.

    Read per request instead of being stored in the built plan: a correction must be
    visible right after it is saved, and invalidating the plan of a mode would make the
    next request pay for a full rebuild (see PlanCache). The table holds one row per
    corrected file, so it is tiny next to the plan itself.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT file_id, action, target FROM manual_overrides").fetchall()
    finally:
        conn.close()
    return {int(r["file_id"]): (r["action"], r["target"]) for r in rows}


class _ModePlan:
    """One built mode: the plan items plus the per-folder index the routes serve.

    Both the aggregate rows and the per-category buckets are computed once, at build
    time — a request then only slices a ready list, so `/api/plan` costs milliseconds
    regardless of the collection size.
    """

    def __init__(self, items: list[PlanItem], sizes: dict[int, int]) -> None:
        self.items = items
        # F104: kept, not only folded into the rows below — the pre-apply summary sums
        # the volume of the files that will actually move, which is not the sum of the
        # folder rows (those include what the user marked "leave alone").
        self.sizes = sizes
        buckets: dict[str, list[PlanItem]] = defaultdict(list)
        for item in items:
            buckets[_plan_category(item)].append(item)
        self.by_category: dict[str, list[PlanItem]] = dict(buckets)
        self.categories: list[dict] = [
            {
                "category": name,
                "count": len(group),
                "size": sum(sizes.get(it.file_id, 0) for it in group),
            }
            for name, group in sorted(self.by_category.items())
        ]


class PlanCache:
    """An in-memory cache of report.plan by mode, built LAZILY — one mode on its
    first request — and dropped explicitly (`rebuild`) after `/api/process` (F36),
    a reset, an apply or a folder-language change, and NOT on every external DB update.

    F70: building all three modes eagerly cost ~40 s on a 26k collection, both at
    `sorta ui` start and on every rebuild, with the user staring at a dead window.
    Now `__init__`/`rebuild` only record what to build; the work happens on the
    thread that first asks for that mode, and only for the mode actually opened.

    sqlite3 connections are not transferable between threads (`check_same_thread`),
    and ThreadingHTTPServer serves each request on a new thread — so a lazy build
    opens its own short-lived connection from `cfg.database` instead of reusing the
    connection of whoever created the cache (see `_connect`, the same reason).

    Thread safety: a mode is built under its own lock, so a request burst from
    several ThreadingHTTPServer threads produces one build and one shared result.
    A `rebuild` that lands mid-build bumps the generation counter, and the finished
    (now stale) plan is simply not stored.
    """

    def __init__(self, cfg: Config, conn: sqlite3.Connection, dest: Path) -> None:
        self._dest = dest
        self._cfg = cfg
        self._db_path = Path(cfg.database).resolve()
        self._by_mode: dict[str, _ModePlan] = {}
        self._generation = 0
        self._state_lock = threading.Lock()
        self._build_locks = {mode: threading.Lock() for mode in _SUPPORTED_MODES}

    def rebuild(self, cfg: Config, conn: sqlite3.Connection) -> None:
        """Invalidate every built mode — the next request recomputes what it needs.

        The signature is kept as-is (the pipeline/sort threads call it with their own
        cfg/conn), but nothing is computed here anymore: a rebuild that blocks the
        caller for ~40 s is exactly what F70 removed. `conn` is deliberately unused —
        it belongs to the calling thread, and the lazy build runs on another one.
        """
        with self._state_lock:
            self._cfg = cfg
            self._db_path = Path(cfg.database).resolve()
            self._by_mode = {}
            self._generation += 1

    def _plan(self, mode: str) -> _ModePlan | None:
        """The built mode (building it if needed), or None for an unsupported mode."""
        if mode not in _SUPPORTED_MODES:
            return None
        with self._state_lock:
            built = self._by_mode.get(mode)
            if built is not None:
                return built
        with self._build_locks[mode]:
            with self._state_lock:
                built = self._by_mode.get(mode)
                if built is not None:
                    return built
                cfg, generation = self._cfg, self._generation
            built = self._build(cfg, mode)
            with self._state_lock:
                if generation == self._generation:
                    self._by_mode[mode] = built
            return built

    def _build(self, cfg: Config, mode: str) -> _ModePlan:
        """One dry-run plan + the file sizes the aggregate reports, in one connection.

        keep_manual_excluded=True (F77): a file marked "leave alone" is not moved by
        `sort --apply` (the sorter drops it from any plan that moves anything), but it
        must stay VISIBLE and unmarkable here — otherwise marking a frame would make it
        vanish from the grid on the next rebuild, with no way back.
        """
        conn = _connect(self._db_path)
        try:
            report = plan_and_sort(cfg, conn, mode, self._dest, apply=False,
                                   write_reports=False, keep_manual_excluded=True)
            sizes = {int(row["id"]): int(row["size"] or 0)
                     for row in conn.execute("SELECT id, size FROM files")}
        finally:
            conn.close()
        return _ModePlan(report.plan, sizes)

    def get(self, mode: str) -> list[PlanItem] | None:
        """The list of PlanItem for a mode, or None for an unsupported mode."""
        built = self._plan(mode)
        return None if built is None else built.items

    def aggregate(self, mode: str) -> dict | None:
        """`GET /api/plan?mode=` — target folders with counts/sizes, no file list.

        F77: the totals also say how many of the plan's files carry a manual correction
        (`overridden`) and how many of those are "leave alone" (`excluded`). The latter
        are LISTED (see `_build`) but will not be moved, so the apply confirmation counts
        `total - excluded`. Counted per request from the live table; the per-folder rows
        keep their existing shape (folder/count/size) — the marks themselves travel with
        the files, on the category page.
        """
        built = self._plan(mode)
        if built is None:
            return None
        marks = _overrides_map(self._db_path)
        actions = [marks[it.file_id][0] for it in built.items if it.file_id in marks]
        return {"mode": mode, "total": len(built.items),
                "overridden": len(actions),
                "excluded": sum(1 for a in actions if a == "exclude"),
                "categories": built.categories}

    def summary(self, mode: str, dest: Path | None) -> dict | None:
        """`GET /api/sort/summary` — the numbers the pre-apply dialog states (F104).

        Everything is read off the SAME built plan the "Cities" tree draws, so the
        dialog cannot quote a number the tab does not show: `files`/`dirs` leave out
        what the user marked "leave alone" (exactly as `aggregate` does), `bytes` is
        the volume of precisely those files, and the two review folders are counted by
        the plan's own reason codes — a folder NAME changes with the folder language,
        a reason does not.

        `dest` is the destination the form is about to send (None — it could not be
        resolved, see `_summary_dest`). What is already lying there is asked of the
        filesystem with the rule `sorter._resolve_dst` applies at apply time, so
        "already there, will be skipped" in the dialog means the same event that
        `report.skipped_already_copied`/`skipped_in_place` will count. That costs a
        stat per file (plus a hash where the size matches), which is why this is a
        request of its own and not part of every `/api/plan`.
        """
        built = self._plan(mode)
        if built is None:
            return None
        marks = _overrides_map(self._db_path)
        items = [it for it in built.items
                 if marks.get(it.file_id, ("", None))[0] != "exclude"]
        existing, same = _dest_occupancy(items, dest)
        return {
            "mode": mode,
            "dest": str(dest) if dest is not None else None,
            "files": len(items),
            "dirs": len({_plan_category(it) for it in items}),
            "bytes": sum(built.sizes.get(it.file_id, 0) for it in items),
            "products": sum(1 for it in items if it.reason == "product"),
            "documents": sum(1 for it in items if it.reason == "document"),
            "dest_existing": existing,
            "dest_same": same,
        }

    def page(self, mode: str, category: str, offset: int, limit: int) -> dict | None:
        """`GET /api/plan?mode=&category=&offset=&limit=` — one page of one folder.

        An unknown category is an empty page with `total: 0` (not an error): a folder
        can disappear between an aggregate and a click on it.
        """
        built = self._plan(mode)
        if built is None:
            return None
        items = built.by_category.get(category, [])
        page = items[offset:offset + limit]
        marks = _overrides_map(self._db_path) if page else {}
        return {
            "mode": mode,
            "category": category,
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "items": [_plan_item_to_json(it, marks.get(it.file_id)) for it in page],
        }


def _parse_page_window(query: dict[str, list[str]],
                       default_limit: int = _PLAN_PAGE_DEFAULT_LIMIT
                       ) -> tuple[int, int] | None:
    """(offset, limit) for any paged route, or None -> 400.

    A missing parameter falls back to the default; a non-integer or negative one is
    rejected rather than coerced — the one outcome that must never happen is quietly
    serving the whole category. A limit above the maximum is clamped, not rejected:
    an over-eager client gets less data, not an error.

    F173: `default_limit` is an argument because one route's page size is a setting rather
    than a constant — search opens to `features.search_page`. Everything else about the
    window is the same rule for every list, which is the point: a slice added tomorrow
    gets a validated window by calling this, not by writing a fourth copy of it.
    """
    raw_offset = (query.get("offset") or ["0"])[0]
    raw_limit = (query.get("limit") or [str(default_limit)])[0].strip()
    try:
        offset, limit = int(raw_offset), int(raw_limit or default_limit)
    except ValueError:
        return None
    if offset < 0 or limit < 0:
        return None
    return offset, min(limit, _PLAN_PAGE_MAX_LIMIT)


def _page_payload(items: list[dict], *, total: int, offset: int, limit: int) -> dict:
    """The five keys every paged slice answers with — F173's shared half on the server.

    Two of them are the feature. `total` is the length of the LIST, never the length of
    this page: "showing 200" and "there are 200" read identically, and for a ranking the
    second is almost never true. `has_more` is computed here, from the window the server
    actually served, so the button on the screen cannot disagree with the data behind it —
    a client deciding for itself would have to keep a running count and would be wrong the
    first time a page came back short.

    A slice merges its own keys into the result (`animals`, `counts`, the state of the
    search index): what is shared is the paging, not the payload.
    """
    return {
        "items": items,
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "has_more": int(offset) + len(items) < int(total),
    }


def _resolve_path(db_path: Path, file_id: int) -> Path | None:
    """The only legitimate way to reach a file on disk — by id from files.

    Opens a short-lived connection per call: ThreadingHTTPServer request handlers
    each run on their own thread, and an sqlite3 connection from another (calling)
    thread must not be passed here (see PlanCache).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
    finally:
        conn.close()
    return Path(row["path"]) if row is not None else None


def _parse_file_id(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


# F42: the People tab renders ~48 cluster cards at once (with
# _CLUSTER_SAMPLE_LIMIT previews each) -> ~288 concurrent GET /thumb/<id>.
# ThreadingHTTPServer spawns a thread per request — without a cache each request
# re-runs decode_rgb + JPEG-encode, hundreds of parallel decodes saturate the CPU,
# the server stops responding. Two independent measures:
# (1) _thumb_cache — an LRU of ready JPEG bytes by (file_id, mtime): a repeated/
#     concurrent request for the same frame never reaches imaging at all;
# (2) _thumb_decode_semaphore — limits the number of decode+encode running
#     CONCURRENTLY (not the total number of requests) — while the cache warms up,
#     a request spike does not spawn hundreds of CPU-heavy decodes at once.
_THUMB_CACHE_MAX_ITEMS = 512
_THUMB_DECODE_CONCURRENCY = max(2, min(8, os.cpu_count() or 4))
# Lightbox (F42/follow-up): a large DECODED JPEG instead of the raw original
# (`/photo`) — the browser cannot do HEIC/RAW, but decode_rgb can. Frames are viewed
# one at a time, so the cache is smaller than the thumbnail one; the edge is larger.
_PREVIEW_MAX_EDGE = 1600
_PREVIEW_CACHE_MAX_ITEMS = 64

# F80: the key carries the frame index too — a clip has one tile but a whole
# filmstrip behind the lightbox, and every frame of it is a separate JPEG. Photos and
# tiles are simply always frame 0.
_ImgCacheKey = tuple[int, float, int]
_ThumbCacheKey = _ImgCacheKey  # name backward-compatibility
_thumb_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_thumb_cache_lock = threading.Lock()
_preview_cache: OrderedDict[_ImgCacheKey, bytes] = OrderedDict()
_preview_cache_lock = threading.Lock()
# a shared semaphore: limits the TOTAL number of concurrent decode+encode (thumb and
# preview together), so a request spike does not spawn hundreds of CPU-heavy decodes.
_thumb_decode_semaphore = threading.Semaphore(_THUMB_DECODE_CONCURRENCY)


def _thumb_cache_clear() -> None:
    """Clear the in-process caches of decoded images (thumbnails + previews).
    Tests — isolation between cases; a DB reset — so a frame of a wiped id is not
    served (the mtime key almost rules out a collision anyway, but we clear for rigor)."""
    with _thumb_cache_lock:
        _thumb_cache.clear()
    with _preview_cache_lock:
        _preview_cache.clear()


def _encode_jpeg_cached(
    file_id: int, path: Path, *, max_edge: int, quality: int,
    cache: OrderedDict[_ImgCacheKey, bytes], cache_lock: threading.Lock,
    cache_max: int, frame: int = 0,
) -> bytes | None:
    """Ready JPEG bytes of a frame (decoded to max_edge), from cache or by decoding.

    The key (file_id, mtime, frame) — a change of mtime naturally invalidates the
    entry. A cache miss is rechecked AFTER acquiring the semaphore (another thread may
    have decoded and cached the same key while the current one waited in the queue) —
    avoids a needless re-decode under a request spike for one frame.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    mtime = stat.st_mtime
    key: _ImgCacheKey = (file_id, mtime, frame)
    with cache_lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    with _thumb_decode_semaphore:
        with cache_lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
        # F67: a gallery of thousands of tiles used to pay a full decode of the
        # ORIGINAL per tile (180-470 ms) — the preview cache turns that into a few ms
        # once the frame has been touched by any stage.
        # F80: video_frame with frame=0 IS decode_rgb_preview (photos included), so
        # every tile and the whole photo path stay on exactly the previous code.
        img = imaging.video_frame(
            path, mtime, stat.st_size, frame, max_edge=max_edge)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()

    with cache_lock:
        cache[key] = data
        cache.move_to_end(key)
        while len(cache) > cache_max:
            cache.popitem(last=False)
    return data


def _thumb_bytes(file_id: int, path: Path) -> bytes | None:
    """Ready JPEG thumbnail bytes for file_id (the _thumb_cache cache, F42)."""
    return _encode_jpeg_cached(
        file_id, path, max_edge=_THUMB_MAX_EDGE, quality=85,
        cache=_thumb_cache, cache_lock=_thumb_cache_lock,
        cache_max=_THUMB_CACHE_MAX_ITEMS)


def _preview_bytes(file_id: int, path: Path, frame: int = 0) -> bytes | None:
    """A large decoded JPEG for the lightbox (HEIC/RAW are rendered too).

    F80: `frame` > 0 asks for that frame of a clip's filmstrip — the same cache, one
    entry per frame (a strip is at most SORTA_VIDEO_FRAMES of them).
    """
    return _encode_jpeg_cached(
        file_id, path, max_edge=_PREVIEW_MAX_EDGE, quality=88,
        cache=_preview_cache, cache_lock=_preview_cache_lock,
        cache_max=_PREVIEW_CACHE_MAX_ITEMS, frame=frame)


def _connect(db_path: Path) -> sqlite3.Connection:
    """A short-lived per-call connection (see _resolve_path — the same reason:
    sqlite3 connections are not transferable between ThreadingHTTPServer threads)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# F66: near_duplicate_groups over tens of thousands of pHashes costs seconds, and the
# Duplicates tab re-requests it on every open. The payload is a few MB of JSON, so a
# couple of entries is all we keep (one per max_distance in practice).
_DUPES_CACHE_MAX_ITEMS = 2
_DupesFingerprint = tuple[tuple[int, int], ...]
_DupesCacheKey = tuple[str, int, _DupesFingerprint]
_dupes_cache: OrderedDict[_DupesCacheKey, list[dict]] = OrderedDict()
_dupes_cache_lock = threading.Lock()


def _dupes_cache_clear() -> None:
    """Drop the cached Duplicates payloads (test isolation)."""
    with _dupes_cache_lock:
        _dupes_cache.clear()


def _db_fingerprint(db_path: Path) -> _DupesFingerprint:
    """(st_mtime_ns, st_size) of the DB file AND its `-wal` sidecar.

    The schema runs in WAL mode, so a commit can land entirely in `<db>-wal` and
    leave the main file untouched — keying on the `.db` stat alone would serve stale
    groups after a pipeline run. A missing file contributes (-1, -1).
    """
    fingerprint: list[tuple[int, int]] = []
    for p in (db_path, Path(f"{db_path}-wal")):
        try:
            st = p.stat()
        except OSError:
            fingerprint.append((-1, -1))
        else:
            fingerprint.append((st.st_mtime_ns, st.st_size))
    return tuple(fingerprint)


def _dupes_payload(db_path: Path, max_distance: int) -> list[dict]:
    """near_duplicate_groups -> JSON-compatible groups for the Duplicates tab.

    recommended (F14): the best frame of the group by (width*height, then size) desc.
    action — the current decision from dedup_choice (keep/to_delete/None).

    keeper_id/keeper_source (F148): the STORED recommendation of the group, if it has
    one — the row `group_keeper` has been getting since F132 and which nothing read.
    Where it exists it names the recommended frame (the star and the preselected radio
    follow it), and `keeper_source` says who chose: `model` or `sharpness`. A group
    without a row — a pair, or one whose membership changed since it was asked about —
    carries `None` in both and is ranked here exactly as it was before.

    Cached (F66) under (db path, max_distance, _db_fingerprint): any write to the
    index changes the fingerprint and the payload is recomputed.
    """
    key: _DupesCacheKey = (str(db_path), max_distance, _db_fingerprint(db_path))
    with _dupes_cache_lock:
        cached = _dupes_cache.get(key)
        if cached is not None:
            _dupes_cache.move_to_end(key)
            return cached

    def remember(payload: list[dict]) -> list[dict]:
        with _dupes_cache_lock:
            _dupes_cache[key] = payload
            _dupes_cache.move_to_end(key)
            while len(_dupes_cache) > _DUPES_CACHE_MAX_ITEMS:
                _dupes_cache.popitem(last=False)
        return payload

    conn = _connect(db_path)
    try:
        groups = near_duplicate_groups(conn, max_distance=max_distance)
        if not groups:
            return remember([])
        all_ids = [r["id"] for g in groups for r in g]
        placeholders = ",".join("?" * len(all_ids))
        wh = {
            r["id"]: (r["width"], r["height"])
            for r in conn.execute(
                f"SELECT id, width, height FROM files WHERE id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
        choices = {
            r["file_id"]: r["action"]
            for r in conn.execute(
                f"SELECT file_id, action FROM dedup_choice WHERE file_id IN ({placeholders})",
                all_ids,
            ).fetchall()
        }
        # F120: sharpness, where it is finally comparable. Across the collection it is
        # not — a screenshot averages 2854 against a photograph's 1253, so a global
        # ranking sorts by content type rather than by focus. Inside a near-duplicate
        # group the frames ARE the same picture, which is the one place the number
        # answers the question it was measured for: which of these five is in focus.
        sharp = {
            r["file_id"]: r["sharpness"]
            for r in conn.execute(
                f"SELECT file_id, sharpness FROM frame_quality "
                f"WHERE file_id IN ({placeholders}) AND sharpness IS NOT NULL",
                all_ids,
            ).fetchall()
        }
        # F148: a group is addressed by a hash of its membership (dedup.group_key), so a
        # key that is missing here means the group has never been asked about (a pair
        # under `keeper_min_group_size`) or has gained/lost a frame since it was. Both
        # readings lead to the same behaviour: no stored recommendation, the ranking
        # below decides, and the tab looks like it did before this feature.
        keepers = read_group_keepers(
            conn, [group_key([r["id"] for r in g]) for g in groups])
    finally:
        conn.close()

    result = []
    for idx, group in enumerate(groups):
        frames = []
        for r in group:
            w, h = wh.get(r["id"], (None, None))
            frames.append({
                "file_id": r["id"],
                "name": Path(r["path"]).name,
                # Where the frame lies in the source, as the Cities tab shows it:
                # `src_dir` in the line, the full `src_path` in the tooltip. Deciding
                # which of two identical frames to keep is mostly a question of WHERE
                # they lie — the copy in "Sorted" beats the one in "Downloads".
                "src_dir": Path(r["path"]).parent.name,
                "src_path": str(Path(r["path"]).parent),
                "thumb_url": f"/thumb/{r['id']}",
                "width": w,
                "height": h,
                "size": r["size"],
                "sharpness": sharp.get(r["id"]),
                "action": choices.get(r["id"]),
                "recommended": False,
            })
        # Sharpness leads only when EVERY frame of the group has it. A partial comparison
        # would quietly prefer whichever frames happened to be measured — and after F120
        # only personal photographs are measured at all, so a mixed group is a real case,
        # not a corner one.
        by_sharpness = all(f["sharpness"] is not None for f in frames)
        best = min(
            frames,
            key=lambda f: (
                -(f["sharpness"] or 0.0) if by_sharpness else 0.0,
                -((f["width"] or 0) * (f["height"] or 0)),
                -(f["size"] or 0),
                f["file_id"],
            ),
        )
        # F148: the stored recommendation wins over the local ranking when the group has
        # one — that is the whole point of having computed it. It never widens what is
        # marked: it moves the star and the preselected keeper radio from one frame to
        # another, and `dedup_choice` is still written by the user's hand alone.
        keeper = keepers.get(group_key([f["file_id"] for f in frames]))
        keeper_source = None
        if keeper is not None:
            named = next((f for f in frames if f["file_id"] == keeper.keeper_id), None)
            if named is not None:
                best = named
                # Two words, not the prompt fingerprint the row carries: the user needs
                # to know WHO advises (trust in the advice depends on it), not which
                # revision of the question was asked.
                keeper_source = ("sharpness" if keeper.source == KEEPER_SOURCE_SHARPNESS
                                 else "model")
        best["recommended"] = True
        result.append({"group": idx, "frames": frames,
                       # Why this one — so the tab can say it instead of asking the user
                       # to trust a star. This is the LOCAL ranking's basis; when
                       # `keeper_source` is set, that is who named the starred frame.
                       "recommended_by": "sharpness" if by_sharpness else "resolution",
                       "keeper_id": best["file_id"] if keeper_source else None,
                       "keeper_source": keeper_source})
    return remember(result)


def _validate_group_payload(payload: object) -> tuple[list[int], int | None] | None:
    """Parse the body `{"group": [file_id,...], "keep_file_id": int?}`.

    None -> the body is invalid (not a JSON object / group is not a non-empty list of
    int / keep_file_id, if present, is not int). keep_file_id may be absent (skip).
    """
    if not isinstance(payload, dict):
        return None
    group = payload.get("group")
    if (not isinstance(group, list) or not group
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in group)):
        return None
    keep = payload.get("keep_file_id")
    if keep is not None and (not isinstance(keep, int) or isinstance(keep, bool)):
        return None
    return group, keep


def _apply_choice(db_path: Path, group: list[int], keep_file_id: int) -> None:
    """keeper -> action='keep', the other frames of the group -> 'to_delete'.

    Idempotent: ON CONFLICT overwrites the old decision (e.g. when moving the keeper
    to another frame of the same group).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for fid in group:
                action = "keep" if fid == keep_file_id else "to_delete"
                conn.execute(
                    """INSERT INTO dedup_choice (file_id, action, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, updated_at = excluded.updated_at""",
                    (fid, action, now),
                )
    finally:
        conn.close()


def _skip_group(db_path: Path, group: list[int]) -> None:
    """"Do not delete this group" — clears dedup_choice of the group's frames."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        with conn:
            conn.execute(
                f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
            )
    finally:
        conn.close()


def _validate_batch_choices_payload(
    payload: object,
) -> tuple[list[tuple[list[int], int]], list[list[int]]] | None:
    """Parse the body `{"groups": [{"group": [...], "keep_file_id": int}, ...],
    "skip": [[file_id,...], ...]}`. `skip` is optional (default []).

    None -> the body is invalid: `groups` is not a non-empty list / any entry does not
    pass `_validate_group_payload` or its `keep_file_id` is absent/not in `group` /
    `skip` is not a list of lists of int. The whole body is validated, before any DB
    write (F32: atomicity — 400 without a partial write).
    """
    if not isinstance(payload, dict):
        return None
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        return None
    groups: list[tuple[list[int], int]] = []
    for entry in raw_groups:
        parsed = _validate_group_payload(entry)
        if parsed is None:
            return None
        group, keep = parsed
        if keep is None or keep not in group:
            return None
        groups.append((group, keep))
    raw_skip = payload.get("skip", [])
    if not isinstance(raw_skip, list):
        return None
    skip: list[list[int]] = []
    for entry in raw_skip:
        if (not isinstance(entry, list) or not entry
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in entry)):
            return None
        skip.append(entry)
    return groups, skip


def _apply_batch_choices(
    db_path: Path, groups: list[tuple[list[int], int]], skip: list[list[int]]
) -> int:
    """Apply the keeper choice over all groups + clear the skipped ones, atomically.

    One transaction for the whole batch: either all groups are applied and all skips
    are cleared, or (on an exception before the call — validation already passed in
    _validate_batch_choices_payload) nothing changes. Returns the number of saved
    (not skipped) groups.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            for group, keep in groups:
                for fid in group:
                    action = "keep" if fid == keep else "to_delete"
                    conn.execute(
                        """INSERT INTO dedup_choice (file_id, action, updated_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(file_id) DO UPDATE SET
                               action = excluded.action, updated_at = excluded.updated_at""",
                        (fid, action, now),
                    )
            for group in skip:
                placeholders = ",".join("?" * len(group))
                conn.execute(
                    f"DELETE FROM dedup_choice WHERE file_id IN ({placeholders})", group
                )
    finally:
        conn.close()
    return len(groups)


def _target_rel(dst: str, dest_root: str) -> str:
    """dst relative to dest_root, as in PlanItem.target_rel (see sorter.py).

    ValueError (a path-case divergence on Windows, etc.) -> the full dst, the same
    fallback as in sorter._target_parts/plan_and_sort.
    """
    try:
        return Path(dst).relative_to(Path(dest_root)).as_posix()
    except ValueError:
        return Path(dst).as_posix()


def _moves_payload(db_path: Path, batch_id: int | None) -> dict:
    """The sort --apply batch manifest: batch metadata + the list of moves.

    batch_id=None -> the last batch (MAX(id) in move_batches). No batches ->
    {"batch": None, "moves": []}, without crashing. name/target_rel are computed from
    dst — independent of the current files row (a trashed file after a move still
    shows its path in the manifest, just without a preview).
    """
    conn = _connect(db_path)
    try:
        if batch_id is None:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, mode, dest_root, started_at, finished_at, operation "
                "FROM move_batches WHERE id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return {"batch": None, "moves": []}
        batch = dict(row)
        move_rows = conn.execute(
            "SELECT file_id, src, dst, status FROM moves "
            "WHERE batch_id = ? ORDER BY dst", (batch["id"],)
        ).fetchall()
    finally:
        conn.close()

    dest_root = batch["dest_root"]
    moves = [
        {
            "file_id": r["file_id"],
            "name": Path(r["dst"]).name,
            "src": r["src"],
            "dst": r["dst"],
            "target_rel": _target_rel(r["dst"], dest_root),
            "status": r["status"],
            "thumb_url": f"/thumb/{r['file_id']}",
            "video": imaging.is_video_path(r["dst"]),  # F80, as in _plan_item_to_json
        }
        for r in move_rows
    ]
    return {"batch": batch, "moves": moves}


def _trash_files(db_path: Path, ids: list[int]) -> list[dict]:
    """The single trash path: ids -> OS trash + DELETE of their files/dedup_choice rows.

    Reused by group deletion of duplicates (`_trash_group`, U3) and by deletion of a
    single frame (`/api/photo/trash`, U4). An id outside the current files (already
    deleted/unknown) is silently skipped — idempotent on a repeated call.
    """
    if not ids:
        return []
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, path FROM files WHERE id IN ({placeholders})", ids
        ).fetchall()
        trashed = []
        for r in rows:
            send_to_trash(r["path"])
            trashed.append({"file_id": r["id"], "name": Path(r["path"]).name})
        found_ids = [r["id"] for r in rows]
        if found_ids:
            ph2 = ",".join("?" * len(found_ids))
            with conn:
                conn.execute(f"DELETE FROM dedup_choice WHERE file_id IN ({ph2})", found_ids)
                # F149: both directions. Trashing a processed copy has to forget that it
                # existed (otherwise the button keeps answering "you already have one" for
                # a file that is gone), and trashing an ORIGINAL leaves its copy an
                # ordinary photograph — the derivation is a fact about a pair, and one half
                # of it is no longer there.
                conn.execute(
                    f"DELETE FROM restored_files "
                    f"WHERE file_id IN ({ph2}) OR source_file_id IN ({ph2})",
                    found_ids + found_ids)
                conn.execute(f"DELETE FROM files WHERE id IN ({ph2})", found_ids)
    finally:
        conn.close()
    return trashed


def _trash_group(db_path: Path, group: list[int], keep_file_id: int) -> list[dict]:
    """The group's non-keepers -> trash (see `_trash_files` — the shared trash path)."""
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(group))
        rows = conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", group
        ).fetchall()
        ids_to_trash = [r["id"] for r in rows if r["id"] != keep_file_id]
    finally:
        conn.close()
    return _trash_files(db_path, ids_to_trash)


def _validate_file_id_payload(payload: object) -> int | None:
    """Parse the body `{"file_id": int}`. None -> invalid (not dict / not int / bool)."""
    if not isinstance(payload, dict):
        return None
    file_id = payload.get("file_id")
    if not isinstance(file_id, int) or isinstance(file_id, bool):
        return None
    return file_id


def _validate_file_ids_payload(payload: object) -> list[int] | None:
    """Parse the body `{"file_ids": [int, ...]}` (bulk deletion of the selected).

    None -> invalid (not dict / not a non-empty list of int without bool). Duplicates
    are collapsed, order is preserved — `_trash_files` itself ignores ids outside the DB.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("file_ids")
    if not isinstance(raw, list) or not raw:
        return None
    seen: set[int] = set()
    ids: list[int] = []
    for v in raw:
        if not isinstance(v, int) or isinstance(v, bool):
            return None
        if v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


_OVERRIDE_ACTIONS = ("exclude", "reassign", "clear", "photo")


def _validate_overrides_payload(payload: object) -> tuple[list[int], str, str | None] | None:
    """Parse the body `POST /api/overrides` (F77):
    `{"file_ids": [int,...], "action": "exclude"|"reassign"|"clear"|"photo",
    "target": str?}`.

    None -> invalid (400): not an object, an unknown/absent action, file_ids that is not
    a non-empty list of ints (bool excluded, like everywhere else), or `reassign`
    without a non-empty target. The target is NOT resolved into a path here — it is a
    folder of the layout, and sorter._manual_target_parts validates it against the sort
    root before a destination is built from it.

    F103: `photo` ("the classifier is wrong, this IS a personal photo") carries no
    target — the whole point is that the file goes back to the AUTOMATIC city layout,
    not to a folder someone had to name.
    """
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if action not in _OVERRIDE_ACTIONS:
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    if action == "reassign":
        target = payload.get("target")
        if not isinstance(target, str) or not target.strip():
            return None
        return ids, action, target.strip()
    return ids, action, None


def _apply_overrides(db_path: Path, file_ids: list[int], action: str,
                     target: str | None) -> list[int]:
    """Write (or, for 'clear', delete) the manual marks; returns the affected file_ids.

    One row per file: a repeated correction of the same file overwrites it via ON
    CONFLICT rather than adding a second row. Ids outside `files` are silently skipped
    (the same rule as `_trash_files`; the FK on manual_overrides.file_id would reject
    them anyway). One transaction for the whole selection — a bulk correction either
    lands entirely or not at all.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(file_ids))
        known = [r["id"] for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", file_ids)]
        if not known:
            return []
        ph = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM manual_overrides WHERE file_id IN ({ph})", known)
            else:
                conn.executemany(
                    """INSERT INTO manual_overrides (file_id, action, target, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, target = excluded.target,
                           updated_at = excluded.updated_at""",
                    [(fid, action, target, now) for fid in known])
    finally:
        conn.close()
    return known


# --- F174: an action says WHERE the frame goes --------------------------------------
# Two of the marks the slices offer read as one movement to the person making it ("this
# frame does not belong in this slice"), and neither of them said where the frame ends
# up. Worse, they are not the same movement at all: taking an animal mark off changes
# a MEMBERSHIP and moves no file, while returning a product to the photos is a real
# transfer out of «_Товары» into a city on the next `sort --apply`. The fix is language,
# not storage — `manual_pet` and `manual_overrides` stay two tables.
#
# The folder name comes from `sorter.destinations`, i.e. from the code that builds the
# plan, never from a rule spelled a second time here. `city` is the mode because it is
# the mode the web app applies (see `_run_sort`), so the caption is about the layout the
# button will actually produce.
_DEST_MODE = "city"

# The plan's reason codes, grouped into the handful of answers a BULK caption can state:
# "12 frames will return: 7 into cities, 5 into no_place" is what the person needs before
# selecting dozens at once, and one folder name out of twelve would simply mislead them.
# A reason nobody grouped lands in `other` rather than being dropped — a group that
# silently loses frames would make the counts stop adding up to the selection.
_DEST_GROUPS: dict[str, str] = {
    "city": "city",
    "manual_reassign": "city",
    "country_only": "country",
    "no_place": "no_place",
    "low_date": "undated",
    "downloaded": "undated",
}


def _destination_json(dest: Destination | None) -> dict:
    """The three fields a card needs to name its destination, or empty for an unknown id.

    `folder` is what the caption prints, `reason` is what the explanation under it is
    looked up by (`dest_why_<reason>`, the `junk_bucket_<verdict>` pattern), and `group`
    is what the bulk breakdown counts. All three are decided HERE: a client that derived
    the group from the folder name would be a second copy of the layout rules, in JS.
    """
    if dest is None:
        return {}
    return {
        "dest": dest.folder,
        "dest_reason": dest.reason,
        "dest_group": _DEST_GROUPS.get(dest.reason, "other"),
    }


def _destinations_for(cfg: Config, conn: sqlite3.Connection, rows: list[sqlite3.Row],
                      assume_action: str | None = None) -> dict[int, Destination]:
    """`sorter.destinations` over the ids of one PAGE of cards, on the open connection.

    Bounded by the page the client asked for, so the cost does not grow with the archive.
    A failure to compute it is not a failure to show the page: geo data may be missing
    (`GeoResolver`) or the layout may raise on a config the slice has no say over, and a
    grid that 500s because a caption could not be phrased is worse than a grid without
    the caption. The cards then simply carry no `dest` field.
    """
    if not rows:
        return {}
    try:
        return destinations(cfg, conn, _DEST_MODE, [int(r["id"]) for r in rows],
                            assume_action)
    except (ValueError, sqlite3.Error, OSError) as exc:
        _log.warning("ui: не удалось вычислить назначение кадров: %s", exc)
        return {}


# --- F103: the "Utility frames" slice ------------------------------------------------
# The deep VLM tier carries away roughly every tenth frame of the collection into
# service folders (2 202 `product` alone on the live 24k run), and until now those
# buckets were visible only indirectly, as folders of the layout plan. A handful of
# those verdicts are wrong, and "a handful out of 2 202" is dozens of frames nobody
# could find. This view shows the buckets AS buckets and lets the wrong ones go back in
# one action. It reclassifies nothing: the fix is a row in `manual_overrides` (F77),
# `media_class` keeps whatever the model measured.

# The `document` bucket is passports, medical forms and bank papers. Those frames get a
# card with a name and a date and NO thumbnail — the project rule is that a document
# verdict is never decoded for display (a preview is a derived copy of the contents).
# Returning one to the photos is still allowed: the person knows what is in their own
# file, they just do not need it rendered to decide.
# F133: which classes those are is a CONFIG question, not a constant — `vlm.exclude_classes`
# already carries the list ("do not show this to the model") and defaults to
# `["document"]`. One visible list of sensitive classes beats two, of which the second
# gets forgotten. The tuple below is only the fallback for a caller that passes nothing:
# a privacy guard must never switch itself off through an omission (the F120 lesson,
# where a typo in the same key would silently have sent documents to the VLM).
_JUNK_NO_PREVIEW = ("document",)


def _junk_item_to_json(row: sqlite3.Row, restored: bool,
                       no_preview: frozenset[str] = frozenset(_JUNK_NO_PREVIEW),
                       dest: Destination | None = None) -> dict:
    """One card of the junk view. `thumb_url` is ABSENT for a no-preview bucket."""
    path = Path(row["path"])
    verdict = row["verdict"]
    payload = {
        "file_id": int(row["id"]),
        "verdict": verdict,
        "name": path.name,
        "date": row["taken_at"],
        # F175: said out loud, per card, for the same reason the page-level list is —
        # a card the person must not delete has to be visible AS one before the "select
        # everything" button is pressed, and a client inferring it from the missing
        # `thumb_url` would be a second copy of the privacy rule in JS.
        "sensitive": verdict in no_preview,
        # F77/F103: the frame already carries a manual "this is a photo" correction —
        # the card says so instead of offering the same action twice.
        "restored": restored,
        # F174: where the frame lands if it IS returned — the folder the plan will build
        # for it once the `photo` mark is in the table, not a folder named by this file.
        **_destination_json(dest),
    }
    if verdict not in no_preview:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


# F171: the order INSIDE one bucket — the model's own estimate, most confident first.
# `media_class.score` is the number the verdict was decided by (the CLIP probability of
# the winning class, or the text density for a document); NULL means "no estimate", never
# "unsure", so those frames keep the old path order at the END of the list instead of
# sinking to a score they were never given. The id is not needed as a tie-break: `f.path`
# is unique and already breaks every tie, so a card keeps its place between pages.
#
# It is applied to one bucket and never to the "all" view, for the reason F175 gives about
# the captions: four classes are four separate softmaxes, and an order across them would
# be a comparison nobody measured.
_JUNK_ORDER = "(mc.score IS NULL), mc.score DESC, f.path"


def _junk_payload(db_path: Path, cfg: Config, bucket: str | None,
                  offset: int, limit: int,
                  sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`GET /api/junk` — the buckets with their counts + one page of one bucket.

    F133: `sensitive` is `vlm.exclude_classes` — the config list that already means
    "handle this class as private", and whose default is `["document"]`. A class in it
    keeps its counter, its cards and the way back to the photos, and loses exactly one
    thing: `thumb_url`. That is the whole of the rule, and it has to be enforced HERE
    rather than in the markup — a card the browser was given a preview link for is a
    card whose contents have already been decoded and sent, whatever the page then
    chooses to draw. The card still carries a name and a date, which is what "open the
    documents in the common grid, do not enlarge them" (the brief) asks for.

    Reusing the VLM key instead of adding a second one is a deliberate trade: one
    visible list of sensitive classes beats two, of which the second gets forgotten.
    Emptying it therefore lifts both protections at once — the guide entry for the key
    is what has to say so.

    The selection is `media_class.verdict <> 'photo'` over canonical, readable files —
    the same `dup_of IS NULL AND error IS NULL` population `junk.classify` writes and
    the sorter lays out, so a bucket counter here matches what the plan will carry off.

    `bucket=None` — every non-photo frame; otherwise exactly the requested verdict. The
    `<> 'photo'` guard sits in the query itself rather than in the parameter check, so
    no value of `bucket` can turn this route into a way of listing personal photos.

    `buckets` is always the full set of counters (it is what the filter chips are drawn
    from), independent of the current filter; `total` is the size of the CURRENT
    selection. An unknown bucket is an empty page, not an error — the same rule as an
    unknown category in `PlanCache.page`.

    F139: `album_kind` is the album this bucket can be gathered into, or None — the
    server decides, because the answer depends on `sensitive` and a client that worked it
    out for itself would be a second copy of the privacy rule. It is None for the "all"
    view (an album of "everything the classifier carried off" is not a slice anybody
    asked for) and for a class in `vlm.exclude_classes`, which keeps its counter and gets
    neither a preview nor an album.

    F171: a bucket is a LIST IN ORDER — `_JUNK_ORDER`, the model's own estimate first —
    and `ordered_by_score` says whether it really was one, so the caption promises a
    ranking exactly where there is one to promise.
    """
    conn = _connect(db_path)
    try:
        counts = conn.execute(
            """SELECT mc.verdict AS verdict, COUNT(*) AS n
               FROM files f JOIN media_class mc ON mc.file_id = f.id
               WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
               GROUP BY mc.verdict"""
        ).fetchall()
        params: list[object] = []
        clause = ""
        if bucket is not None:
            clause = " AND mc.verdict = ?"
            params.append(bucket)
        total = conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      {clause}""", params).fetchone()[0]
        scored = 0 if bucket is None else int(conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      AND mc.score IS NOT NULL {clause}""", params).fetchone()[0])
        rows = conn.execute(
            f"""SELECT f.id, f.path, f.taken_at, mc.verdict
                FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      {clause}
                ORDER BY {_JUNK_ORDER if bucket is not None else 'f.path'}
                LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
        # F174: what the button on these cards will do — asked with the correction it
        # writes already assumed, so the caption names the city the frame goes back to
        # and not the service folder it is sitting in right now.
        dests = _destinations_for(cfg, conn, rows, "photo")
    finally:
        conn.close()
    marks = _overrides_map(db_path) if rows else {}
    buckets = [{"verdict": r["verdict"], "count": int(r["n"])} for r in counts]
    buckets.sort(key=lambda b: (-b["count"], b["verdict"]))
    return {
        "bucket": bucket,
        "buckets": buckets,
        "album_kind": (
            bucket if (bucket in CLASS_ALBUM_KINDS and bucket not in sensitive)
            else None),
        # Said out loud rather than inferred from a missing field: a card without
        # `thumb_url` is a class the server refuses to render, not a preview that failed
        # to build, and the two need different words on the screen.
        "sensitive": sorted(sensitive),
        # F171: whether this page really is the ranking its caption promises. `False` for
        # the "all" view (no ordering across four buckets) and for a bucket the classifier
        # settled without a number of its own — a heuristics-only run, or the frames the
        # deep tier rewrote, both of which store NULL rather than a confidence.
        "ordered_by_score": bool(scored),
        "total": int(total),
        "offset": offset,
        "limit": limit,
        "items": [
            _junk_item_to_json(
                r, (marks.get(int(r["id"])) or ("", None))[0] == "photo", sensitive,
                dests.get(int(r["id"])))
            for r in rows
        ],
    }


# --- F123: the "Animals" tab — the pet verdicts of the frame-quality stage ----------
# The signal has been computed since F113 and calibrated in F122 (805 frames of the live
# collection at 92% precision), and until now nobody could see a single one of them. The
# view is the junk grid's twin — a page of thumbnails over a read-only query — with one
# deliberate difference: the order is by CONFIDENCE, not by path. About 64 of those 805
# frames are not animals, and reading top-down until the quality runs out is how a person
# finds where that border sits, so the score travels to the card and is shown on it.


def _animal_item_to_json(row: sqlite3.Row, dest: Destination | None = None) -> dict:
    """One card of the animal view: a thumbnail, a name, a date and the pet score.

    F124: plus the two facts a card has to state about the mark itself — whether the
    frame counts as an animal right now (`is_animal`, straight out of the shared rule,
    never recomputed here in Python) and whether that answer came from a person
    (`manual`, the value of the `manual_pet` row, or None if there is none). A frame the
    user has taken the mark off stays on the card, struck through: it must be visible as
    marked BY HAND, otherwise the counter moves for no reason anybody can see and the
    decision cannot be taken back.
    """
    path = Path(row["path"])
    return {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # NULL is impossible for a frame that carries a verdict (junk writes the score
        # alongside it) — but a payload that pretends 0.0 was measured would lie about
        # exactly the number this tab exists to show.
        "score": None if row["pet_score"] is None else float(row["pet_score"]),
        "is_animal": bool(row["is_animal"]),
        "manual": None if row["manual"] is None else bool(row["manual"]),
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
        # F174: where the frame ALREADY lies. This slice is a view over the canon, not an
        # extraction from it, so the mark moves no file — and the card can only say so
        # convincingly by naming the folder the frame is in either way.
        **_destination_json(dest),
    }


_ANIMALS_JOIN = ("FROM files f LEFT JOIN frame_quality fq ON fq.file_id = f.id "
                 "LEFT JOIN manual_pet mp ON mp.file_id = f.id")


# F160: the animal rule now has a tier whose switches live outside `features:` — the
# detector's master switch `detect.enabled` (F145) and the model that wrote the boxes. So
# the helpers of this slice take the WHOLE live config, the way `_overview_payload` already
# does, and resolve both switches through the one function that ANDs them
# (`detector_settings`).
# Reading half of the pair here is exactly the mistake F145 was written about, and a slice
# still counting the boxes of a detector the user has switched off is the same bug in the
# other direction.
def _animals_population(cfg: Config) -> str:
    """What the TAB LISTS: the model's marks plus every frame a person has touched.

    Deliberately wider than the slice — a frame marked "not an animal" is no longer in the
    album and is still on this page, struck through, because a card that vanishes takes the
    undo button with it.

    F137: "the model's marks" is the automatic half of the shared rule (`animal_auto_sql`),
    not the `frame_quality.pet` cache — a threshold edit has to take frames OFF this page
    too, or the list and the counter it carries would disagree about the same collection.
    """
    return (f"({animal_auto_sql(cfg.features, 'fq', detector_settings(cfg))} "
            "OR mp.file_id IS NOT NULL) AND f.dup_of IS NULL AND f.error IS NULL")


def _animals_count_sql(cfg: Config) -> str:
    """What COUNTS as an animal: `sorter.animal_ids_sql` and nothing else, over the
    canonical, readable files every other counter in this file is built on. Used by this
    tab and by the "Overview" number, so the two cannot disagree with the album or with
    each other."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT COUNT(*) FROM files f
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.id IN ({ids})"""


def _animals_select(cfg: Config) -> str:
    """One card, one row shape — the page and the answer to a mark are the same SELECT, so
    a card redrawn after an edit says exactly what the same card would say on a reload."""
    ids = animal_ids_sql(cfg.features, detector_settings(cfg))
    return f"""SELECT f.id, f.path, f.taken_at, fq.pet_score,
           mp.is_animal AS manual, f.id IN ({ids}) AS is_animal
    {_ANIMALS_JOIN}"""


def _animals_payload(db_path: Path, cfg: Config, offset: int, limit: int) -> dict:
    """`GET /api/animals` — one page of the animal slice, most confident first.

    Two numbers, because after F124 they are two different questions: `total` is the
    length of the LIST (what the paging walks — model marks plus manual decisions), and
    `animals` is how many frames actually count as animals, by the one shared rule. The
    second is the number "Overview" shows and the album gathers; the first is what
    "showing 200 of N" is about.

    `ORDER BY pet_score DESC, f.id` — the id breaks ties, so two frames with an equal
    score keep a stable place between pages instead of swapping and being shown twice
    (or never) as the reader pages down. A manual decision does NOT move a card: the
    reader is walking down a list sorted by confidence, and a list that reshuffles under
    the frame just marked is a list nobody can finish reading.

    `cfg` is the LIVE config, for the reason `/api/junk` reads its sensitive classes off
    it: the thresholds this page is drawn with — and, since F160, whether the detector's
    tier counts at all — are the ones in force at the moment of the request, not the ones
    some run wrote into the database (F137).
    """
    population = _animals_population(cfg)
    conn = _connect(db_path)
    try:
        total = conn.execute(
            f"SELECT COUNT(*) {_ANIMALS_JOIN} WHERE {population}").fetchone()[0]
        animals = conn.execute(_animals_count_sql(cfg)).fetchone()[0]
        rows = conn.execute(
            f"""{_animals_select(cfg)}
                WHERE {population}
                ORDER BY fq.pet_score DESC, f.id
                LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
        # F174: no assumed correction — the question here is where the frame lies NOW,
        # which is the same folder it will lie in after the mark, because the mark
        # changes a membership and not a route.
        dests = _destinations_for(cfg, conn, rows)
    finally:
        conn.close()
    return {
        "animals": int(animals),
        **_page_payload([_animal_item_to_json(r, dests.get(int(r["id"])))
                         for r in rows],
                        total=int(total), offset=offset, limit=limit),
    }


# F124: "the model is wrong about this frame", the only three answers there are. `clear`
# drops the row and hands the frame back to the automatic verdict — which is not the same
# as `not_animal`, and the difference is the reason the row is two-valued rather than a
# presence flag.
_ANIMAL_MARK_ACTIONS = ("animal", "not_animal", "clear")


def _validate_animal_mark_payload(payload: object) -> tuple[list[int], str] | None:
    """Parse the body `POST /api/animals/mark`:
    `{"file_ids": [int,...], "action": "animal"|"not_animal"|"clear"}`.

    None -> invalid (400). The ids go through the same `_validate_file_ids_payload` as
    every other write route — ints only, never a path.
    """
    if not isinstance(payload, dict):
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    action = payload.get("action")
    if action not in _ANIMAL_MARK_ACTIONS:
        return None
    return ids, action


def _apply_animal_mark(db_path: Path, cfg: Config,
                       ids: list[int], action: str) -> dict:
    """Write the user's verdict into `manual_pet`; answer with the redrawn cards.

    One row per file (PRIMARY KEY file_id), so marking the same frame twice overwrites
    rather than piling up. Nothing here touches `frame_quality` — the whole point of the
    feature is that the model's own table keeps being recomputed from scratch and this
    mark is read on top of it (`sorter.animal_ids_sql`).

    An id outside the current index is skipped rather than written (the rule
    `_apply_review_mark`/`_trash_files` follow): a decision about a file the program does
    not know is not a decision about anything, and the FK would reject it anyway.

    The answer carries `items` (the marked frames as the tab's own cards) and `animals`
    (the fresh count by the shared rule) so the client can redraw one card and the
    counter in place. It could reload the page instead, and that is exactly what it must
    not do: this list is read top-down until the confidence runs out, and a reload sends
    the reader back to the first screen after every decision. `items` may come back
    SHORTER than the ids — a `clear` on a frame the model never marked leaves the list
    altogether — and the client drops those cards.
    """
    now = datetime.now(timezone.utc).isoformat()
    count_sql = _animals_count_sql(cfg)
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        known = [int(r["id"]) for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", ids).fetchall()]
        if not known:
            return {"marked": 0, "items": [],
                    "animals": int(conn.execute(count_sql).fetchone()[0])}
        known_placeholders = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM manual_pet WHERE file_id IN ({known_placeholders})",
                    known)
            else:
                is_animal = 1 if action == "animal" else 0
                conn.executemany(
                    """INSERT INTO manual_pet (file_id, is_animal, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           is_animal = excluded.is_animal,
                           updated_at = excluded.updated_at""",
                    [(fid, is_animal, now) for fid in known])
        rows = conn.execute(
            f"""{_animals_select(cfg)}
                WHERE {_animals_population(cfg)}
                  AND f.id IN ({known_placeholders})""",
            known).fetchall()
        animals = conn.execute(count_sql).fetchone()[0]
        # F174: the redrawn card has to say what a reload would say, and after F174 that
        # includes the folder the frame lies in — a caption that vanished on the first
        # click would look like the mark had moved the file after all.
        dests = _destinations_for(cfg, conn, rows)
    finally:
        conn.close()
    return {
        "marked": len(known),
        "animals": int(animals),
        "items": [_animal_item_to_json(r, dests.get(int(r["id"]))) for r in rows],
    }


# --- F152: the face slices — with people / group photos / portraits ----------------
# The three largest populations of the archive (people are 27.5% of a hand-labelled
# sample of 200 frames) had no slice at all, while the signal for them has been on disk
# since the faces stage: 12 952 real faces over 7 341 photographs. The rules themselves
# live in `sorter.face_slice_ids_sql`, exactly one copy of them, for the reason
# `ANIMAL_IDS_SQL` lives there — the album, this panel and the "Overview" counters must
# be talking about one collection.
#
# What is different from the slices around it is the CAPTION rather than the query:
# membership here is a fact of a detector's output, not a place in a ranking, so the
# panel says so and says nothing about confidence — there is no score to show.
#
# The one state that is not a number: without a faces run the honest answer is WHY there
# is nothing (`reason='no_faces_run'`, the F125 rule) and the counters travel as `null`
# rather than as zeros. A zero here reads as "no photograph of yours has a person on
# it" — a conclusion about somebody's own archive, drawn from a table nobody filled.

# Canonical and readable, the population every other counter in this file is built on.
# `media_type` is not filtered: the faces stage only ever writes rows for photographs,
# so a video cannot be in these slices anyway.
_FACE_LIVE = "f.dup_of IS NULL AND f.error IS NULL"

# `media_class` rides along for the F133 privacy rule alone — a frame of a sensitive
# class is listed but never given a `thumb_url`, so no preview of a document with a face
# on it is ever decoded.
_FACE_FROM = "FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id"

# How many real faces this frame carries — the one number a card of these slices shows,
# and the same `bbox != '[]'` rule the slices themselves are built on.
_FACE_COUNT_SQL = ("(SELECT COUNT(*) FROM faces fa WHERE fa.file_id = f.id "
                   "AND fa.bbox != '[]')")


def _face_slice_where(cfg: Config, slice_: str) -> tuple[str, list[object]]:
    """The WHERE of one face slice + its parameters, over the canonical population."""
    ids_sql, params = face_slice_ids_sql(cfg, slice_)
    return f"{_FACE_LIVE} AND f.id IN ({ids_sql})", params


def _face_slice_count(conn: sqlite3.Connection, cfg: Config, slice_: str) -> int:
    """How many frames one face slice holds, under the WHERE its page uses."""
    where, params = _face_slice_where(cfg, slice_)
    return int(conn.execute(
        f"SELECT COUNT(*) FROM files f WHERE {where}", params).fetchone()[0])


def _face_item_to_json(row: sqlite3.Row, sensitive: frozenset[str]) -> dict:
    """One card: a thumbnail, a name, a date and how many faces the frame holds.

    No score, because there is none to invent: the frame is in the slice because a box
    was found on it. The face count is on the card all the same — it is what makes the
    group slice checkable by eye, and on a portrait it says "one" out loud.
    """
    path = Path(row["path"])
    payload = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        "faces": int(row["faces"]),
    }
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _face_slices_payload(cfg: Config, db_path: Path, slice_: str, offset: int,
                         limit: int, sensitive: frozenset[str]) -> dict:
    """`GET /api/face-slices` — the three counters + one bounded page of the current one.

    `counts` is always the full set (it is what the pins draw), and every entry is `null`
    when the faces stage has not run: the counters are then not zero, they are unmeasured,
    and `reason` says which. Once the stage has run a zero IS the answer — "no group
    photographs were found" is a fact about the collection — and it is shown as one.

    `ORDER BY f.id`: these slices have no ranking of their own (there is no confidence in
    them to rank by), and index order is stable, which is what paging needs.
    """
    conn = _connect(db_path)
    try:
        ran = faces_stage_ran(conn)
        counts: dict[str, int | None] = {name: None for name in FACE_SLICES}
        items: list[dict] = []
        total = 0
        if ran:
            for name in FACE_SLICES:
                counts[name] = _face_slice_count(conn, cfg, name)
            where, params = _face_slice_where(cfg, slice_)
            total = int(counts[slice_] or 0)
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.taken_at, mc.verdict AS verdict,
                           {_FACE_COUNT_SQL} AS faces
                    {_FACE_FROM} WHERE {where}
                    ORDER BY f.id LIMIT ? OFFSET ?""",
                [*params, limit, offset]).fetchall()
            items = [_face_item_to_json(r, sensitive) for r in rows]
    finally:
        conn.close()
    return {
        "slice": slice_,
        "counts": [{"slice": name, "count": counts[name]} for name in FACE_SLICES],
        "reason": None if ran else "no_faces_run",
        # The thresholds travel with the answer so the hint above the grid can state the
        # rule the numbers were produced by instead of repeating a default in JS.
        "group_min": int(cfg.features.group_photo_faces),
        "portrait_share": float(cfg.features.portrait_face_share),
        **_page_payload(items, total=total, offset=offset, limit=limit),
    }


def _parse_face_slice_query(query: dict[str, list[str]]) -> tuple[str, int, int] | None:
    """(slice, offset, limit) for `GET /api/face-slices`, or None -> 400.

    An unknown slice is refused rather than answered with an empty page, the
    `_parse_review_query` rule: there are exactly three, so anything else is a client
    that has lost track of what it is asking for.
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    slice_ = ((query.get("slice") or [FACE_SLICES[0]])[0].strip() or FACE_SLICES[0])
    if slice_ not in FACE_SLICES:
        return None
    return slice_, window[0], window[1]


# --- F126: the "Review" workspace — duplicates, blur, closed eyes ------------------
# Three signals, one job: look at a frame and decide whether it stays. Duplicates have had
# a tab with the whole viewing-and-deleting machinery since U3; the other two have been
# computed into `frame_quality` since F113 and were not visible anywhere. So this is one
# place with SLICES rather than tabs — and the duplicates half is deliberately untouched:
# `/api/dupes` and its four write routes answer exactly as before, because that
# is the one path in the product that deletes files and it is the one path that has been
# run against the live collection.
#
# F177 removed a fourth slice, "no subject". The model was asked about 6 111 frames and
# called 212 of them subjectless; looked at by eye, those 212 are ordinary photographs, so
# the slice was showing a list assembled by nothing. It is deleted rather than hidden: a
# hidden slice comes back at the first edit of this file.
#
# Two rules the slices are built on:
#
# * a decision is a row in `dedup_choice` and nothing else. `to_delete` already means
#   "move into `_delete` on the next `sort --apply`" (sorter.py), and a second deletion
#   path in a program that moves 300 GB of somebody's photographs is a second way to lose
#   them. `file_id` is the primary key there, so a frame that shows up in two slices
#   carries ONE decision and shows it in both;
# * nothing is ever marked automatically. There is no "delete everything below the
#   threshold" route here, and the measurement is why: reviewed by eye in bands, blurred
#   frames turn up in every band up to 400, and the blurred frame that gets kept is the
#   only photograph of a person or a place. Sharpness ranks the list; a human decides.
#
# F150 adds a fifth, "low resolution", and it sits here rather than in a tab of its own
# for the same reason the other four share this one: all of them are "look at this and
# decide whether it stays". It is not folded INTO the blurred list either — measured on
# 22 095 photographs, the two populations intersect by 3% (682 of the 706 frames under a
# megapixel are formally sharp), so mixing them would hide each inside the other and leave
# a person sorting blur wondering why sharp little pictures keep appearing.
_REVIEW_SLICES = ("dupes", "blurred", "eyes", "low_resolution")

# F139: which album kind each flat slice gathers into — and, read the other way, the map
# that keeps the list and the album on one rule. The names differ because the switcher's
# are older than the album's (`eyes` is a chip label, `eyes_closed` is a folder), and
# renaming either half would move an API parameter for nothing. Duplicates have no kind:
# they are the grouped slice, the one where a keeper is chosen, and the one path in the
# program that deletes files — collecting them into a folder is not what they are for.
_REVIEW_SLICE_KIND = {"blurred": "blurred", "eyes": "eyes_closed",
                      "low_resolution": "low_resolution"}

# Every flat slice is ranked by the number it exists for, and for all three that means
# ASCENDING, so the most damaged frame is the first one a person sees: a blurred frame has
# little variance, a closed eye is a thin slit, a low-resolution frame has few pixels. None
# of the three orderings is a verdict — each decides what to look at first, not what to
# delete. F179 gave the eyes such a number; before it they went in index order, because the
# VLM answer behind them was a yes/no with nothing to sort by. `f.id` closes every one of
# them: frames of equal sharpness, equal openness or equal size must come back in the same
# order on every page, or paging would drop and repeat them at the seam.
_REVIEW_SLICE_ORDER = {
    "blurred": "fq.sharpness ASC, f.id",
    "eyes": "fq.eye_openness ASC, f.id",
    "low_resolution": "f.width * f.height ASC, f.id",
}

# F157 + F155: where a frame HAS a sharpness measured inside its face, that number orders
# it — and it orders it BEFORE every frame that has none. Two reasons, and neither is a
# preference:
#
# * the two numbers are not on one scale (a variance over a whole preview against one over
#   a 100-200 px crop, `features.face_sharpness_max` says why no factor converts them), so
#   they must never meet inside one comparison. `face_sharpness IS NULL` first, then each
#   group by its own number, is the only ordering that keeps that promise;
# * on frames that have a face the face number finds 62% of the blurred ones against 15%
#   for the whole-frame number (F155, 68 labelled frames). Reading the better signal first
#   is what a ranking is for.
#
# NULL keeps its schema meaning throughout — "not measured", never "sharp" — so a frame
# with no face, or one from a run before the column existed, simply sorts by the frame
# number in the second half of the list instead of dropping out of it.
_BLURRED_ORDER_WITH_FACE = ("(fq.face_sharpness IS NULL), fq.face_sharpness ASC, "
                            "fq.sharpness ASC, f.id")


def _blurred_order_column(conn: sqlite3.Connection) -> str:
    """Which number orders the blur list on THIS database — the F155 column, or the frame.

    The column is asked of the schema rather than assumed, because the order of F155 and
    F157 was never fixed: a database from before v25 has no `face_sharpness` at all, and
    the list has to open on it exactly as it does anywhere else. `_has_column` is the
    indexer's, which reads its own optional columns the same way (`files.orientation`);
    a second spelling of one PRAGMA would be a second thing to keep true.
    """
    return ("face_sharpness" if _has_column(conn, "frame_quality", "face_sharpness")
            else "sharpness")


def _review_order(conn: sqlite3.Connection, slice_: str) -> str:
    """The ORDER BY of one flat slice, against `_review_from`."""
    if slice_ == "blurred" and _blurred_order_column(conn) == "face_sharpness":
        return _BLURRED_ORDER_WITH_FACE
    return _REVIEW_SLICE_ORDER[slice_]


# The two extra columns a card carries, by slice — a card shows the number its slice is
# ABOUT and not every number the row happens to hold. The absent one is selected as NULL
# rather than left out so that one row shape feeds one `_review_item_to_json`; and for
# `low_resolution` there is no `fq` alias to read at all (`quality_slice_from`).
_REVIEW_SLICE_COLUMNS = {
    "blurred": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "eyes": "fq.sharpness AS sharpness, NULL AS width, NULL AS height",
    "low_resolution": "NULL AS sharpness, f.width AS width, f.height AS height",
}

# The membership rule itself lives in sorter.py (`quality_slice_where`,
# `quality_slice_from`) and is read from there rather than restated here: the album of a
# slice and the list of it must be the same set of frames, and two spellings of one
# condition drift.


def _review_from(slice_: str) -> str:
    """The FROM of one flat slice — the shared rule, by slice name."""
    return quality_slice_from(_REVIEW_SLICE_KIND[slice_])


def _review_where(slice_: str, features: FeaturesConfig, *,
                  beyond: bool = False) -> tuple[str, list[object]]:
    """The WHERE of one flat slice + its parameters — the shared rule, by slice name.

    `beyond` is "show more": the blurred list opens to `features.blur_review_max` and the
    closed-eyes list to `features.eye_openness_max` (F179), and each runs on without a
    ceiling once asked. Each window bounds its own slice alone.
    """
    return quality_slice_where(_REVIEW_SLICE_KIND[slice_], features, beyond=beyond)


def _review_count(conn: sqlite3.Connection, slice_: str,
                  features: FeaturesConfig) -> int:
    """How many frames one flat slice holds, under the same WHERE the page uses."""
    where, params = _review_where(slice_, features)
    return int(conn.execute(
        f"SELECT COUNT(*) {_review_from(slice_)} WHERE {where}", params).fetchone()[0])


def _review_flat_counts(conn: sqlite3.Connection,
                        features: FeaturesConfig) -> dict[str, int]:
    """The flat slice counters — plain aggregates, cheap enough for "Overview".

    EVERY slice is counted INSIDE its own window, so the chip, the "Overview" row and the
    length of the list the tab opens with are one number per slice. F179 made that true of
    the eyes too: the slice is a ranking now, and a counter that ignored the window would
    advertise every frame a face was measured on — the whole face population, not the
    closed eyes.
    """
    return {name: _review_count(conn, name, features)
            for name in _REVIEW_SLICES if name != "dupes"}


# F133: the same flat slices again, counting only the frames NOBODY has decided about.
# "Decided" is a row in `dedup_choice` and nothing else — the rule the marks are written
# by — so a slice empties as the person works through it, which is what makes the warning
# on the "Layout" tab disappear on its own.
_PENDING_JOIN = " LEFT JOIN dedup_choice dc ON dc.file_id = f.id"


def _review_pending_count(conn: sqlite3.Connection, slice_: str,
                          features: FeaturesConfig) -> int:
    """How many frames of one flat slice still carry no decision."""
    where, params = _review_where(slice_, features)
    return int(conn.execute(
        f"SELECT COUNT(*) {_review_from(slice_)}{_PENDING_JOIN} "
        f"WHERE {where} AND dc.action IS NULL", params).fetchone()[0])


def _review_pending_counts(conn: sqlite3.Connection,
                           features: FeaturesConfig) -> dict[str, int]:
    """The undecided part of each flat slice, under the same WHERE the page uses."""
    return {name: _review_pending_count(conn, name, features)
            for name in _REVIEW_SLICES if name != "dupes"}


def _pending_dupe_groups(groups: list[dict]) -> int:
    """Duplicate groups carrying no decision — no query, the payload already says so.

    A group counts as decided as soon as ONE of its frames carries an action: choosing a
    keeper writes `keep` on it and `to_delete` on the rest. "Do not delete this group"
    CLEARS those rows (`_skip_group`), so such a group is undecided again — which is the
    literal truth about it and the same thing the slice counters say.
    """
    return sum(
        1 for g in groups
        if not any(f.get("action") for f in g.get("frames", []))
    )


def _review_item_to_json(row: sqlite3.Row, action: str | None) -> dict:
    """One card of a flat slice: a thumbnail, a name, a date, the slice's number, the
    decision."""
    path = Path(row["path"])
    return {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # Where it lies, as on the Cities and Duplicates lists: with a burst of similar
        # frames the folder is often the only thing that tells them apart.
        "src_dir": path.parent.name,
        "src_path": str(path.parent),
        "sharpness": None if row["sharpness"] is None else float(row["sharpness"]),
        # F150: the size of the picture, on the card of the slice that is about it. A
        # thumbnail is the same 200 px whatever it was made from, so the pixels are the
        # one thing a person cannot see and the one thing they are deciding on.
        "width": None if row["width"] is None else int(row["width"]),
        "height": None if row["height"] is None else int(row["height"]),
        "action": action,
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
    }


def _review_payload(db_path: Path, slice_: str, offset: int, limit: int, *,
                    beyond: bool, features: FeaturesConfig,
                    max_distance: int) -> dict:
    """`GET /api/review` — the slice counters + one bounded page of the current slice.

    `counts` is always the full set (it is what the switcher draws, and a slice with
    nothing in it stays in the list showing a zero: "you have no closed eyes" is an
    answer, a vanished entry is a riddle). `dupes` counts GROUPS and comes from the
    cached `_dupes_payload` — the same payload the duplicates half of the tab renders
    from, so opening the workspace pays for it once.

    `slice='dupes'` carries no items: duplicates are the one grouped slice, and forcing
    them into the flat shape would cost the keeper choice that the whole view is for.
    The client renders that slice from `/api/dupes`, exactly as it did when it was a tab
    of its own.

    `eyes_reason='no_faces_run'` (F125) — the eye number is measured only where a face was
    found, so without a faces run the honest answer is why there is no data, not a zero
    that looks like "your subjects all had their eyes open".

    F150: `low_resolution_mp` travels with the answer for the same reason `blur_max`
    does — and `eye_max` with it since F179 — the hint above the grid states the rule the
    list was built by instead of repeating a default in JS.

    F179: `window_total` is the count of the CURRENT slice's window, because every flat
    slice has one now — blurred down to `features.blur_review_max`, closed eyes down to
    `features.eye_openness_max` — and "show more" walks either of them past its window
    into the frames the ranking is less sure about.

    F157: for the blurred slice that window is the depth of the FIRST PAGE, so
    `window_total`, the chip's counter and the length of the list the tab opens with are
    one number — a length, not a population. `blur_order` says which column ordered it.
    """
    conn = _connect(db_path)
    try:
        counts = _review_flat_counts(conn, features)
        pending = _review_pending_counts(conn, features)
        eyes_reason = None if faces_stage_ran(conn) else "no_faces_run"
        window_total = counts.get(slice_, counts["blurred"])
        blur_order = _blurred_order_column(conn)
        items: list[dict] = []
        total = 0
        if slice_ != "dupes":
            source = _review_from(slice_)
            where, params = _review_where(slice_, features, beyond=beyond)
            total = int(conn.execute(
                f"SELECT COUNT(*) {source} WHERE {where}", params).fetchone()[0])
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.taken_at, {_REVIEW_SLICE_COLUMNS[slice_]}
                    {source} WHERE {where}
                    ORDER BY {_review_order(conn, slice_)}
                    LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
            actions: dict[int, str] = {}
            if rows:
                ids = [int(r["id"]) for r in rows]
                placeholders = ",".join("?" * len(ids))
                actions = {
                    int(r["file_id"]): r["action"]
                    for r in conn.execute(
                        f"SELECT file_id, action FROM dedup_choice "
                        f"WHERE file_id IN ({placeholders})", ids).fetchall()
                }
            items = [_review_item_to_json(r, actions.get(int(r["id"]))) for r in rows]
    finally:
        conn.close()
    groups = _dupes_payload(db_path, max_distance)
    counts["dupes"] = len(groups)
    pending["dupes"] = _pending_dupe_groups(groups)
    if slice_ == "dupes":
        total = counts["dupes"]
    return {
        "slice": slice_,
        "grouped": slice_ == "dupes",
        # F139: the album kind of the CURRENT slice, or None for the duplicates. The
        # client draws its "gather into a folder" row from this and never from a table of
        # its own — see `_REVIEW_SLICE_KIND`.
        "album_kind": _REVIEW_SLICE_KIND.get(slice_),
        "counts": [{"slice": name, "count": counts[name]} for name in _REVIEW_SLICES],
        # F133: what the "Layout" tab warns about — the part of the workspace nobody has
        # answered yet. `pending_total` is the one number the warning shows; the per-slice
        # breakdown rides along because it costs nothing and says WHERE the work is left.
        "pending": [{"slice": name, "count": pending[name]} for name in _REVIEW_SLICES],
        "pending_total": sum(pending.values()),
        "eyes_reason": eyes_reason,
        "blur_max": float(features.blur_review_max),
        # F157: which number ordered the blur list — `face_sharpness` where F155's column
        # exists, `sharpness` where it does not. The caption says so out loud, because
        # "frames with a face are ordered by the sharpness of the face" is the one thing
        # that explains why a visibly sharp street can sit above a soft portrait.
        "blur_order": blur_order,
        # F179: the number the closed-eyes caption is shown with — and that caption states
        # the PRECISION measured at it, not a count, because 62% right is the fact a person
        # needs before looking at the list.
        "eye_max": float(features.eye_openness_max),
        "low_resolution_mp": float(features.low_resolution_mp),
        "window_total": window_total,
        "beyond": bool(beyond),
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


def _parse_review_query(
    query: dict[str, list[str]],
) -> tuple[str, int, int, bool] | None:
    """(slice, offset, limit, beyond) for `GET /api/review`, or None -> 400.

    An unknown slice is refused rather than answered with an empty page: the switcher
    offers exactly `_REVIEW_SLICES`, so anything else is a client that has lost track of
    what it is asking for. The window is parsed by the plan-page rules
    (`_parse_page_window`).
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    slice_ = ((query.get("slice") or [_REVIEW_SLICES[0]])[0].strip()
              or _REVIEW_SLICES[0])
    if slice_ not in _REVIEW_SLICES:
        return None
    beyond = (query.get("beyond") or ["0"])[0].strip() in ("1", "true")
    return slice_, window[0], window[1], beyond


_REVIEW_MARK_ACTIONS = ("keep", "to_delete", "clear")


def _validate_review_mark_payload(payload: object) -> tuple[list[int], str] | None:
    """Parse the body `POST /api/review/mark`:
    `{"file_ids": [int,...], "action": "keep"|"to_delete"|"clear"}`.

    None -> invalid (400). The ids go through the same `_validate_file_ids_payload` as
    every other bulk route — ints only, never a path.
    """
    if not isinstance(payload, dict):
        return None
    ids = _validate_file_ids_payload(payload)
    if ids is None:
        return None
    action = payload.get("action")
    if action not in _REVIEW_MARK_ACTIONS:
        return None
    return ids, action


def _apply_review_mark(db_path: Path, ids: list[int], action: str) -> int:
    """Write the decision of a flat slice into `dedup_choice`; returns how many landed.

    The same table and the same two values the duplicates half writes, on purpose: one
    decision per file, understood by one consumer (`sorter`, which moves `to_delete`
    into `_delete` on `--apply`). `clear` removes the row, i.e. "I have not decided",
    which is not the same as `keep` — and `keep` is what survives the next run, so the
    two or three blurred frames a person keeps for the memory are not asked about again.

    Nothing here touches a file on disk. An id outside the current index is skipped
    rather than written (`_trash_files` resolves ids the same way): a decision about a
    file the program does not know is not a decision about anything.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(ids))
        known = [int(r["id"]) for r in conn.execute(
            f"SELECT id FROM files WHERE id IN ({placeholders})", ids).fetchall()]
        if not known:
            return 0
        known_placeholders = ",".join("?" * len(known))
        with conn:
            if action == "clear":
                conn.execute(
                    f"DELETE FROM dedup_choice WHERE file_id IN ({known_placeholders})",
                    known)
            else:
                conn.executemany(
                    """INSERT INTO dedup_choice (file_id, action, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(file_id) DO UPDATE SET
                           action = excluded.action, updated_at = excluded.updated_at""",
                    [(fid, action, now) for fid in known])
    finally:
        conn.close()
    return len(known)


# --- F149: "try to improve" — one frame, by request, a copy beside it -----------------
# The third action of the Review tab, next to the two it has had ("mark for deletion",
# "keep"). What it does NOT do is most of the design (see `restore` for the measurement
# and the reasoning):
#
# * ONE frame per press. `{"file_id": int}` and no list shape at all — a body carrying
#   `file_ids` is refused by the validator like any other malformed one. There is no CLI
#   command either. A model that draws plausible detail, applied in bulk, turns an archive
#   into a collection of convincing forgeries;
# * the original is never opened for writing, and the copy carries `_restored` in its name;
# * a repeat press returns the copy that already exists (`restore.existing_copy`) instead
#   of making a second one;
# * keeping the copy does NOT mark the original for deletion. Two decisions, and the
#   second one is the person's — the same line between advice and action as F148. Nothing
#   on this path writes `dedup_choice`; the copy simply becomes a frame the existing
#   marking route can be used on, exactly like its source.


def _restored_item_to_json(row: sqlite3.Row, source_file_id: int) -> dict:
    """One card for the processed copy — the shape of a review card, plus what it is.

    `restored` and `source_file_id` are what the client draws the badge from and where it
    inserts the card: beside the original, not at the end of the list and not in a dialog
    of its own. `action` is always None: the copy has just been created, so nobody has
    decided anything about it yet, and it must not arrive with a decision attached.
    """
    item = _review_item_to_json(row, None)
    item["restored"] = True
    item["source_file_id"] = int(source_file_id)
    return item


# --- F168: the second entrance — the expanded frame, in every slice ------------------
# F149 drew the button in ONE place, the "blurred" slice, and the measurement of
# 2026-08-03 says that place is almost empty: the Laplacian filter at its threshold finds
# 8% of the frames a person calls soft (it answers "how much detail is in the frame", not
# "is it in focus"). So the action sat behind a detector we had measured to be nearly
# blind, and the only way to reach it was to be lucky enough to be in that list.
#
# The second measurement (F169, 80 blind pairs) says where the action really belongs. The
# gain is not about blur at all — it is about SIZE:
#
#     < 640 px    66% |  640-1024  58%  |  1024-1280  52%
#
# — a clean win on small frames, a coin toss by 1280. Hence the shape of this entrance:
# ONE input, on the frame a person has already expanded (the lightbox, which every slice
# opens), and offered only while the frame is below `features.restore_max_edge`. Above the
# ceiling the offer is withdrawn AND the reason is said out loud (`_restore_offer`): a
# button there would promise what the measurement did not find, and a frame silently
# rebuilt from a quarter of itself is exactly the trade F169 exists to disclose.
#
# The two bans below are enforced HERE, in the route, and not by not drawing a button —
# the F133 rule: a hidden control is not a rule, and a request made past the interface
# collects the same thing.
RESTORE_ERROR_SENSITIVE = "sensitive_class"
RESTORE_ERROR_VIDEO = "video"


def _restore_refusal(path: Path, verdict: str | None, media_type: str | None,
                     sensitive: frozenset[str]) -> str | None:
    """The code this frame may not be processed under, or None — the server-side bans.

    A private class (`vlm.exclude_classes`, `document` by default) is refused because
    processing one means decoding a passport or a medical form and drawing it four times
    larger — the one thing the product deliberately never renders. Video is refused
    because the engine is about images: a clip has no single frame to be the answer.
    """
    if verdict is not None and verdict in sensitive:
        return RESTORE_ERROR_SENSITIVE
    if media_type == "video" or imaging.is_video_path(path):
        return RESTORE_ERROR_VIDEO
    return None


def _restore_source_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    """The source's path and the two facts the bans are decided from, or None.

    A LEFT JOIN on purpose: a frame nobody has classified yet (`media_class` is written by
    a run that may not have happened) is an ordinary photograph, not a refusal.
    """
    return conn.execute(
        """SELECT f.id, f.path, f.media_type, mc.verdict AS verdict
           FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
           WHERE f.id = ?""", (file_id,)).fetchone()


def _restore_notice(src: Path, max_edge: int) -> dict:
    """F169: what the answer owes about the ceiling — `rebuilt` and the two numbers.

    Recomputed from the source rather than remembered, because the same sentence is owed
    on the press that REUSES a copy: the frame and the ceiling are what they are, so the
    second press must not quietly drop the warning the first one carried.
    """
    edge = restore.source_edge(src)
    return {"rebuilt": edge > int(max_edge) > 0, "source_edge": edge,
            "max_edge": int(max_edge)}


def _restore_frame(db_path: Path, features: FeaturesConfig, file_id: int,
                   sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`POST /api/review/restore` for ONE id -> the card of the copy, or the reason.

    Reads the source's path from the index (never from the request — the same rule every
    other route follows), hands it to `restore.restore_frame`, then indexes the result. A
    reason travels as a CODE (`restore.ERROR_*`), which the client translates: the weights
    come from the network and offline is an ordinary state for this product, so "the model
    is not here" has to be an answer a person can read rather than an empty result.

    F169: the ceiling comes from `features.restore_max_edge` and is PASSED — it used to be
    a constant the engine defaulted to, i.e. one number for every frame with nobody told —
    and the answer carries `rebuilt` whenever the frame was larger than it. The action is
    not refused for such a frame: what should happen to a 12 Mpx one is the measurement's
    decision (`scripts/measure_restore.py`), and until it is made the honest thing is to
    do the work and say what was done.

    F168: `sensitive` is `vlm.exclude_classes`, and the two bans it and `media_type` carry
    are refused HERE rather than by not drawing a button — this route is now reachable
    from every slice, and a rule that lives in the markup is a rule a request made past
    the interface never meets. The default is the fallback list for the same reason
    `_junk_payload` has one: a privacy guard must not switch itself off through an
    omission. Both refusals are ordinary reasons (200 + a code the client translates),
    not errors: the person pointed at a frame this action does not apply to, which is
    something the interface has to be able to say.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return {"ok": False, "error": "file not found"}
        refusal = _restore_refusal(Path(row["path"]), row["verdict"], row["media_type"],
                                   sensitive)
        if refusal is not None:
            return {"ok": False, "reason": refusal}
        model = features.restore_model
        notice = _restore_notice(Path(row["path"]), features.restore_max_edge)
        existing = restore.existing_copy(conn, file_id, model)
        if existing is not None:
            copy_id, copy_path = existing
            if Path(copy_path).exists():
                return {"ok": True, "reused": True, **notice,
                        "item": _restored_item_to_json(_restored_row(conn, copy_id), file_id)}
            # The person deleted it in their file manager. Answering "you already have one"
            # and drawing a card for a file that is gone is worse than doing the work again.
            restore.forget_copy(conn, copy_id)
        result = restore.restore_frame(Path(row["path"]), model,
                                       max_edge=features.restore_max_edge)
        if not result.ok or result.path is None:
            return {"ok": False, "reason": result.error, "detail": result.detail}
        notice = {"rebuilt": result.rebuilt, "source_edge": result.source_edge,
                  "max_edge": int(features.restore_max_edge)}
        copy_id = restore.record_restored(conn, file_id, result.path, model=model)
        item = _restored_item_to_json(_restored_row(conn, copy_id), file_id)
    finally:
        conn.close()
    # The copy is a new canonical file, so the cached duplicate payload and the cached
    # layout no longer describe the collection. It is never a duplicate of its source
    # (`dedup`), which is a statement about the GROUPS and not about the cache.
    _dupes_cache_clear()
    return {"ok": True, "reused": False, "item": item, **notice}


def _restored_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row:
    """The copy's row in the shape `_review_item_to_json` reads.

    `sharpness` is selected as NULL rather than joined: the copy has no `frame_quality`
    row and will not have one until the next run measures it, and a card that printed a
    zero would be claiming a measurement nobody made.

    F150: the size, on the other hand, is REAL and is selected as such. `record_restored`
    measures the copy it just wrote, and on the low-resolution slice — the model's proper
    addressee, where ×4 turns 640×480 into 2560×1920 — the change in size is the whole
    result of the operation. A card that hid it would leave the person comparing two
    thumbnails of identical width on screen.
    """
    return conn.execute(
        "SELECT id, path, taken_at, NULL AS sharpness, width, height "
        "FROM files WHERE id = ?", (file_id,)).fetchone()


def _restored_source_json(conn: sqlite3.Connection, file_id: int) -> dict | None:
    """Where this frame was processed FROM, or None if it is not a copy at all.

    The badge on the expanded frame is drawn from this, and the link comes out of
    `restored_files` rather than out of the name: the copy is an ordinary member of the
    collection now — it lies in the city folder beside its source, it can be gathered
    into an album — so wherever it turns up it has to say what it is, or it reads as a
    second similar photograph that came from nowhere.
    """
    row = conn.execute(
        """SELECT r.source_file_id AS file_id, f.path AS path
           FROM restored_files r JOIN files f ON f.id = r.source_file_id
           WHERE r.file_id = ?""", (file_id,)).fetchone()
    if row is None:
        return None
    return {"file_id": int(row["file_id"]), "name": Path(row["path"]).name}


def _restore_offer(db_path: Path, features: FeaturesConfig, file_id: int,
                   sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict | None:
    """`GET /api/restore/offer` — what the expanded frame affords; None -> 404.

    Read-only, and it is NOT a second implementation of the action: pressing still goes
    to the one route, and a reason still travels as the same code. This answers the two
    questions the expanded frame has to answer before anything is offered —

    * `available`: may this frame be processed at all (the bans the route enforces). A
      client that worked that out for itself would be a second copy of the privacy rule,
      which is the mistake F133 named;
    * `rebuilt`: is the frame ABOVE `features.restore_max_edge`, i.e. would the copy be
      rebuilt from a reduced version of itself. The measurement found the gain on small
      frames and nothing by 1280 px, so above the ceiling the offer is withdrawn and the
      two numbers are handed over for the sentence that says why. Silence there would be
      a promise the measurement does not support.

    `restored_from` is the other direction: this frame IS a copy, and here is the frame it
    was made from.

    A refused frame is not measured: the size comes off the file's header, and a frame
    classed as a personal document is one this program does not open for any purpose. The
    two numbers are what the "too large" sentence is built from, and there is no such
    sentence to build when the answer is already no.
    """
    conn = _connect(db_path)
    try:
        row = _restore_source_row(conn, file_id)
        if row is None:
            return None
        path = Path(row["path"])
        refusal = _restore_refusal(path, row["verdict"], row["media_type"], sensitive)
        notice = ({"rebuilt": False, "source_edge": 0,
                   "max_edge": int(features.restore_max_edge)} if refusal is not None
                  else _restore_notice(path, features.restore_max_edge))
        return {
            "file_id": int(row["id"]),
            "available": refusal is None,
            "reason": refusal,
            "restored_from": _restored_source_json(conn, file_id),
            **notice,
        }
    finally:
        conn.close()


def _parse_file_id_query(query: dict[str, list[str]]) -> int | None:
    """`?file_id=` as a positive int, or None -> 400 (the same rule as the POST body)."""
    raw = (query.get("file_id") or [""])[0].strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _parse_junk_query(query: dict[str, list[str]]) -> tuple[str | None, int, int] | None:
    """(bucket, offset, limit) for `GET /api/junk`, or None -> 400.

    An empty/absent `bucket` means "every non-photo frame"; the window is parsed by the
    same rules (and with the same bounds) as a plan page — a bad number is refused, an
    over-eager limit is clamped rather than rejected.
    """
    window = _parse_page_window(query)
    if window is None:
        return None
    raw_bucket = (query.get("bucket") or [""])[0].strip()
    return (raw_bucket or None), window[0], window[1]


# --- F85c: assigning a place to a whole group at once -------------------------------
# About 6 300 files of the live collection carry no place signal at all — no GPS, no
# neighbour in time with one, no landmark, and nothing readable in the folder name. No
# model will place them: the information is not in them. It is in the person who took
# them, and the only thing that stands between them and a correct place is that clicking
# six thousand times is not a thing anyone will do. Hence: pick a GROUP the user already
# thinks in (a whole event, a whole source folder), pick a place from the bundled base,
# one action.

_PLACE_KINDS = ("event", "source_dir")
_PLACE_ACTIONS = ("assign", "clear")
_PLACE_SEARCH_LIMIT = 12

_geo_resolver_cache: GeoResolver | None = None


def _geo_resolver() -> GeoResolver:
    """The bundled GeoNames resolver, loaded at most once per server process.

    The place picker asks it on every keystroke (debounced), and the data behind it is
    12 MB plus a KD-tree — building that per request would make the field unusable.
    """
    global _geo_resolver_cache
    if _geo_resolver_cache is None:
        _geo_resolver_cache = GeoResolver()
    return _geo_resolver_cache


@dataclasses.dataclass(frozen=True)
class _ManualPlace:
    """What a `manual_places` row holds: a country, optionally narrowed to one city."""

    country: str
    city: str | None = None
    city_geonameid: int | None = None


def _country_label(cc: str, lang: i18n.Lang) -> str:
    """The country name to SHOW: the curated dictionary first, then the bundled base."""
    curated = i18n.country(cc, lang)
    if curated != cc:
        return curated
    try:
        return _geo_resolver().country_name(cc, lang) or cc
    except GeoDataMissing:
        return cc


def _city_candidates(query: str, lang: i18n.Lang) -> list[dict]:
    """Cities of the bundled base whose name in ANY of the three languages is `query`.

    `city_ids_by_name` (F46) matches a FULL name, not a prefix, which is what makes this
    safe to offer: the same reverse index the `--where city=` filter is built on, and
    the same geonameids that land in `places.city_geonameid`. Same-named cities come
    back as several candidates, told apart by region and country — picking for the user
    would be guessing.
    """
    resolver = _geo_resolver()
    out: list[dict] = []
    seen: set[int] = set()
    for search_lang in _UI_LANGS:
        for gid in resolver.city_ids_by_name(query, search_lang):  # type: ignore[arg-type]
            if gid in seen:
                continue
            seen.add(gid)
            cc = resolver.country_of(gid)
            if not cc:
                # Without a country the place cannot be laid out (the layout starts at
                # the country folder), so such a city is not offered at all.
                continue
            region = resolver.region_key_of(gid)
            region_name = resolver.region_name(cc, region[1], lang) if region else None
            city_name = resolver.name(gid, lang)
            details = ", ".join(p for p in (region_name, _country_label(cc, lang)) if p)
            out.append({
                "kind": "city", "country": cc, "city_geonameid": gid,
                "city": resolver.name(gid, "en"),
                "label": f"{city_name} ({details})" if details else city_name,
            })
            if len(out) >= _PLACE_SEARCH_LIMIT:
                return out
    return out


def _places_search(query: str, lang: i18n.Lang) -> list[dict]:
    """`GET /api/places/search` — what the typed text may mean, country first.

    Country first because it is the safer answer: a wrong country is a mistake the user
    can see in one glance at the plan, and the country level is where a file with no
    other signal belongs anyway. Both halves read ONLY the bundled base — no network, no
    model, and nothing is written until the user picks one and confirms.
    """
    text = query.strip()
    if not text:
        return []
    results: list[dict] = []
    try:
        cc = i18n.country_cc_by_name(text)
        for search_lang in _UI_LANGS:
            if cc:
                break
            cc = _geo_resolver().country_cc_by_name(text, search_lang)  # type: ignore[arg-type]
        if cc:
            results.append({"kind": "country", "country": cc.upper(),
                            "city_geonameid": None, "city": None,
                            "label": _country_label(cc, lang)})
        results.extend(_city_candidates(text, lang))
    except GeoDataMissing:
        # The bundled base is the only source here; without it the picker offers
        # nothing rather than pretending an empty answer means "no such place".
        _log.warning("ui: гео-данные недоступны — поиск места вернёт пустой список")
        return []
    return results


def _validate_place_payload(
    payload: object,
) -> tuple[str, str, str, _ManualPlace | None, bool] | None:
    """Parse the body of `POST /api/place`:
    `{"kind": "event"|"source_dir", "selector": str, "action": "assign"|"clear",
      "country": str?, "city_geonameid": int?, "include_gps": bool?}`.

    None -> invalid (400). `assign` needs a country (a city alone would leave the layout
    without its top folder); `city_geonameid` is optional and narrows it to one city.
    The selector is NOT resolved here — an event id is looked up in the DB, and a source
    folder is only ever COMPARED against `files.path`, never opened (see
    `_place_target_ids`).
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    action = payload.get("action")
    selector = payload.get("selector")
    if kind not in _PLACE_KINDS or action not in _PLACE_ACTIONS:
        return None
    if not isinstance(selector, str) or not selector.strip():
        return None
    include_gps = bool(payload.get("include_gps"))
    if action == "clear":
        return kind, selector.strip(), action, None, include_gps
    country = payload.get("country")
    if not isinstance(country, str) or not country.strip():
        return None
    gid = payload.get("city_geonameid")
    if gid is not None and (not isinstance(gid, int) or isinstance(gid, bool)):
        return None
    city = None
    if gid is not None:
        try:
            city = _geo_resolver().name(gid, "en")
        except GeoDataMissing:
            return None
    return (kind, selector.strip(), action,
            _ManualPlace(country=country.strip().upper(), city=city, city_geonameid=gid),
            include_gps)


def _is_under(path: str, directory: str) -> bool:
    """Is `path` inside `directory`? A comparison of two strings, never of the disk.

    `files.path` is written by the indexer with the separators of the machine that
    indexed it, and the folder arrives from the client's own tree, so both are
    normalized (case and separator) before the prefix test. The boundary character is
    required — `/Photos/Greece2019` must not count as being inside `/Photos/Greece`.
    """
    root = os.path.normcase(directory.rstrip("\\/"))
    target = os.path.normcase(path)
    if not root:
        return False
    return target.startswith(root + os.sep) or target.startswith(root + "/")


def _place_target_ids(conn: sqlite3.Connection, kind: str, selector: str) -> list[int]:
    """The canonical files of the chosen group — one event, or one source folder.

    Only these two kinds exist on purpose: both are groups the user already sees as a
    thing (a card on the "Events" tab, a folder in the plan), and both are BOUNDED. "The
    whole collection in one action" is deliberately not offered — the larger the grab,
    the higher the price of a wrong pick, and undoing it means finding the files again.
    """
    if kind == "event":
        try:
            event_id = int(selector)
        except ValueError:
            return []
        rows = conn.execute(
            """SELECT f.id FROM event_files ef JOIN files f ON f.id = ef.file_id
               WHERE ef.event_id = ? AND f.dup_of IS NULL AND f.error IS NULL""",
            (event_id,),
        ).fetchall()
        return [int(r["id"]) for r in rows]
    rows = conn.execute(
        "SELECT id, path FROM files WHERE dup_of IS NULL AND error IS NULL").fetchall()
    return [int(r["id"]) for r in rows if _is_under(r["path"], selector)]


def _apply_bulk_place(db_path: Path, kind: str, selector: str, action: str,
                      place: _ManualPlace | None, include_gps: bool) -> dict:
    """Write (or drop) the manual place of a whole group. Returns what happened.

    Files with `confidence='exact_gps'` are SKIPPED unless `include_gps` is set: those
    were placed by the camera at the moment of the shot, and a memory of which city a
    trip was in is not better evidence than a coordinate. They are counted and reported
    back, so the client can offer to include them — an explicit second decision, never a
    silent overwrite. `clear` skips nothing: dropping a manual row can only restore what
    the program itself worked out.

    One transaction for the whole group — a bulk assignment either lands entirely or not
    at all, which is what makes "undo" a single action too.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        ids = _place_target_ids(conn, kind, selector)
        skipped_gps = 0
        if ids and action == "assign" and not include_gps:
            ph = ",".join("?" * len(ids))
            with_gps = {int(r["file_id"]) for r in conn.execute(
                f"""SELECT file_id FROM places
                    WHERE confidence = 'exact_gps' AND file_id IN ({ph})""", ids)}
            skipped_gps = len(with_gps)
            ids = [fid for fid in ids if fid not in with_gps]
        if ids:
            ph = ",".join("?" * len(ids))
            with conn:
                if action == "clear":
                    conn.execute(
                        f"DELETE FROM manual_places WHERE file_id IN ({ph})", ids)
                else:
                    assert place is not None  # guaranteed by _validate_place_payload
                    conn.executemany(
                        """INSERT INTO manual_places
                               (file_id, country, city, city_geonameid, updated_at)
                           VALUES (?, ?, ?, ?, ?)
                           ON CONFLICT(file_id) DO UPDATE SET
                               country = excluded.country, city = excluded.city,
                               city_geonameid = excluded.city_geonameid,
                               updated_at = excluded.updated_at""",
                        [(fid, place.country, place.city, place.city_geonameid, now)
                         for fid in ids])
    finally:
        conn.close()
    return {
        "ok": True, "action": action, "kind": kind, "selector": selector,
        "affected": len(ids), "skipped_gps": skipped_gps,
        "country": place.country if place else None,
        "city_geonameid": place.city_geonameid if place else None,
    }


def _clusters_payload(db_path: Path, sample_limit: int = _CLUSTER_SAMPLE_LIMIT) -> list[dict]:
    """Root clusters (`merged_into IS NULL`) with size/label/samples.

    size — the number of faces in the whole merge chain (the root + everything merged
    into it), not just faces whose `faces.cluster_id` points directly to the root
    (after `merge` it keeps pointing to the original cluster — see `faces.merge`).
    samples — up to `sample_limit` distinct file_ids, ordered by `faces.id`
    (deterministic, stable between requests). Noise clusters (`faces.cluster_id IS
    NULL`) are naturally excluded by the `WHERE cluster_id IS NOT NULL` filter. Sorted
    by descending size.
    """
    conn = _connect(db_path)
    try:
        cluster_rows = conn.execute(
            "SELECT id, label, merged_into FROM face_clusters"
        ).fetchall()
        face_rows = conn.execute(
            "SELECT cluster_id, file_id FROM faces "
            "WHERE cluster_id IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    merged_into = {r["id"]: r["merged_into"] for r in cluster_rows}
    labels = {r["id"]: r["label"] for r in cluster_rows}
    root_ids = [r["id"] for r in cluster_rows if r["merged_into"] is None]

    def root_of(cid: int) -> int:
        seen: set[int] = set()
        while merged_into.get(cid) is not None and cid not in seen:
            seen.add(cid)
            cid = merged_into[cid]
        return cid

    size: dict[int, int] = defaultdict(int)
    samples: dict[int, list[int]] = defaultdict(list)
    sample_seen: dict[int, set[int]] = defaultdict(set)
    for r in face_rows:
        root = root_of(r["cluster_id"])
        size[root] += 1
        seen_files = sample_seen[root]
        if r["file_id"] not in seen_files and len(samples[root]) < sample_limit:
            seen_files.add(r["file_id"])
            samples[root].append(r["file_id"])

    result = [
        {
            "cluster_id": rid,
            "size": size.get(rid, 0),
            "label": labels.get(rid),
            "samples": samples.get(rid, []),
        }
        for rid in root_ids
    ]
    result.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return result


def _validate_cluster_label_payload(payload: object) -> tuple[int, str] | None:
    """Parse `{"cluster_id": int, "name": str}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    cluster_id = payload.get("cluster_id")
    name = payload.get("name")
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        return None
    if not isinstance(name, str):
        return None
    return cluster_id, name


def _validate_cluster_merge_payload(payload: object) -> tuple[int, int] | None:
    """Parse `{"src": int, "dst": int}`. None -> invalid."""
    if not isinstance(payload, dict):
        return None
    src = payload.get("src")
    dst = payload.get("dst")
    if not isinstance(src, int) or isinstance(src, bool):
        return None
    if not isinstance(dst, int) or isinstance(dst, bool):
        return None
    return src, dst


def _album_dest(cfg: Config, db_path: Path) -> Path:
    """The album root: `cfg.sort.album_dir` if set in the config, otherwise the default next to the DB."""
    album_dir = getattr(cfg.sort, "album_dir", None)
    if album_dir:
        return Path(album_dir)
    return db_path.resolve().parent / _DEFAULT_ALBUM_DIRNAME


def _suggested_sort_dest(cfg: Config, db_path: Path) -> str:
    """The default destination path for the city layout: `<source>_sorted`.

    The source — the first `cfg.sources` (config.yaml); if empty — the common root of
    the indexed files from the DB. Nothing found → an empty string (the field stays
    for manual entry). A POSIX path (like sources in config).
    """
    root: Path | None = None
    if cfg.sources:
        root = Path(cfg.sources[0])
    else:
        try:
            conn = _connect(db_path)
            try:
                paths = [r[0] for r in conn.execute(
                    "SELECT path FROM files WHERE error IS NULL").fetchall()]
            finally:
                conn.close()
            if paths:
                common = os.path.commonpath(paths)
                # commonpath over files returns an ancestor directory; if it matched a
                # single file (the only path) — take its parent
                root = Path(common)
                if root.suffix:  # this is a file, not a directory
                    root = root.parent
        except (ValueError, OSError):
            root = None
    if root is None:
        return ""
    return (root.parent / (root.name + "_sorted")).as_posix()


def _events_payload(db_path: Path,
                    sample_limit: int = _EVENT_SAMPLE_LIMIT) -> list[dict]:
    """The event list for the "Events" tab: id/name/count/dates + up to
    `sample_limit` preview file_ids (clickable -> lightbox), by descending count."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT e.id, e.name, e.started_at, e.ended_at,
                      COUNT(ef.file_id) AS count
               FROM events e LEFT JOIN event_files ef ON ef.event_id = e.id
               GROUP BY e.id
               ORDER BY count DESC, e.id"""
        ).fetchall()
        # samples in a separate pass: the event's canonical frames by time,
        # up to sample_limit per event (as _clusters_payload accumulates in Python)
        samples: dict[int, list[int]] = defaultdict(list)
        for s in conn.execute(
            """SELECT ef.event_id, ef.file_id
               FROM event_files ef JOIN files f ON f.id = ef.file_id
               WHERE f.dup_of IS NULL AND f.error IS NULL
               ORDER BY ef.event_id, f.taken_at, f.id"""
        ):
            bucket = samples[s["event_id"]]
            if len(bucket) < sample_limit:
                bucket.append(s["file_id"])
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "count": r["count"],
            "started_at": r["started_at"],
            "ended_at": r["ended_at"],
            "samples": samples.get(r["id"], []),
        }
        for r in rows
    ]


# --- F156: why a built-in slice is empty --------------------------------------------
# A zero with no explanation reads as "your archive holds none of these", and far more
# often it means "nobody has looked yet" — the `frame_quality` rule of F125 (NULL is "not
# asked", not "no") applied to a whole slice. So each of the three exact slices answers
# with one of three things, and never with a bare emptiness:
#
#   None          the slice holds photographs
#   not_run       the stage that fills it never ran — the run screen is where that is
#                 fixed, and the interface links straight to it
#   none_found    the stage ran over this collection and there is nothing of the kind
#
# The two reasons are two on purpose: only one of them is a fact about the person's
# photographs, and only the other one has an action attached to it.
#
# `not_run` is also what a stage SWITCHED OFF looks like (`features.pets: false` — the
# quality stage runs and never asks about animals), and that is right rather than a
# compromise: the run screen holds that checkbox, so the sentence and the link lead to the
# same place either way. Which is the whole reason the standard slices are not made
# hideable a second time — one control for "do not compute animals", not two.
_SLICE_NOT_RUN = "not_run"
_SLICE_NONE_FOUND = "none_found"


def _tabs_visibility_payload(db_path: Path, cfg: Config) -> dict[str, object]:
    """F54: visibility of the "People"/"Events"/"Animals" tabs — by data presence
    (variant B, without a meta table). person ⇔ there is a faces row with a non-empty
    cluster_id (the same source as `_clusters_payload`); event ⇔ non-empty `events`;
    animal (F123) ⇔ some `frame_quality` row counts as an animal, which is false for
    every collection processed with `features.pets` off. Light EXISTS queries, we do not
    build the full payload.

    F156: ...or there is something to SAY. A slice whose stage has never run appears too,
    exactly as the face slices have since F152, because its emptiness is a sentence with a
    link in it and a pin that hides itself never gets to say it. `reasons` carries which
    of the two empty states each slice is in (`_SLICE_NOT_RUN` / `_SLICE_NONE_FOUND`, and
    `None` when the slice holds something) — the answer the panel captions itself with.
    A slice that ran and found nothing keeps hiding: there the zero IS the fact, the
    collection has already said it, and a pin over an empty page teaches nothing.

    The animal question is deliberately asked of what the tab would LIST
    (`_animals_population`) and not of what it would count: a user who has taken the mark
    off every frame has emptied the slice but not the tab, and the tab is where the undo
    button lives. F137 is the reason it is that expression rather than the older "some
    `frame_quality.pet` is set" — the cache column can claim a verdict the thresholds in
    force have withdrawn, and a tab shown for an empty page is exactly the drift this
    feature is about.

    F152: `face` is the one that is NOT asked of its own data, and that is the whole
    point. The three face slices appear as soon as the index holds a photograph the faces
    stage could have looked at — because without a run they have to be able to SAY there
    was no run (`no_faces_run`), and a pin that hides itself says nothing at all. It is
    the same question phase 3 asks of a collection (`faces._CANONICAL`), minus the join
    to the faces table.

    `indexed` rides along for the same cost: "re-run the selected stage" only makes
    sense over files that exist. Right after "Start over" the index is empty and
    ticking "faces" used to light the button up — offering to catch up a stage on
    nothing at all.
    """
    conn = _connect(db_path)
    try:
        person = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM faces WHERE cluster_id IS NOT NULL)"
        ).fetchone()[0])
        event = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM events)"
        ).fetchone()[0])
        animal = bool(conn.execute(
            f"SELECT EXISTS(SELECT 1 {_ANIMALS_JOIN} "
            f"WHERE {_animals_population(cfg)})"
        ).fetchone()[0])
        face = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files WHERE dup_of IS NULL AND error IS NULL "
            "AND media_type = 'photo')"
        ).fetchone()[0])
        indexed = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files)"
        ).fetchone()[0])
        # F156: which of the two empty states each of the three is in. One EXISTS per
        # slice, and each one asks whether the STAGE left anything behind — not whether
        # the slice came out non-empty, which is the question already answered above.
        #
        # faces: a real box (`faces_stage_ran` excludes the "processed, none here" marker
        #   row). events: the stage groups every canonical frame that carries a date, so
        #   its own output is the only marker there is — with dated frames and no events
        #   nothing has grouped them, and with no dated frames there was nothing to group.
        #   animals: a STORED `pet_score`, which the stage writes whether or not it reached
        #   the threshold and never writes with `features.pets` off. A fact of the table
        #   rather than the switch as it stands right now (F137's rule): the question is
        #   what was asked of THIS collection, and the switch may have moved since.
        dated = bool(conn.execute(
            f"SELECT EXISTS(SELECT 1 FROM files f WHERE {_OVERVIEW_LIVE} "
            "AND f.taken_at IS NOT NULL)").fetchone()[0])
        stage_ran = {
            "person": faces_stage_ran(conn),
            "event": bool(conn.execute(
                "SELECT EXISTS(SELECT 1 FROM events)").fetchone()[0]) or not dated,
            "animal": bool(conn.execute(
                "SELECT EXISTS(SELECT 1 FROM frame_quality WHERE pet_score IS NOT NULL)"
            ).fetchone()[0]),
        }
    finally:
        conn.close()
    found = {"person": person, "event": event, "animal": animal}
    reasons = {
        name: None if has else (_SLICE_NONE_FOUND if stage_ran[name]
                                else _SLICE_NOT_RUN)
        for name, has in found.items()
    }
    # A slice is offered when it holds photographs OR when it has something to say and a
    # collection to say it over — the population being the one its own stage walks
    # (canonical photographs for faces and animals, any indexed file for events).
    over = {"person": face, "event": indexed, "animal": face}
    visible = {name: has or (over[name] and reasons[name] == _SLICE_NOT_RUN)
               for name, has in found.items()}
    return {**visible, "face": face, "indexed": indexed, "reasons": reasons}


# --- F108: the "Overview" tab — the state of the collection in one screen -----------
# Every number below is a plain aggregate over the index, and the plan is deliberately
# NOT built: a layout of 24k frames costs minutes, while this is the screen a user opens
# right AFTER a run to see what changed. Nothing is cached either — a number that is one
# run out of date answers the question wrongly, which is worse than not answering it.
#
# Privacy: aggregates only. No file path and no file id leaves this endpoint; the single
# path in the payload is the destination FOLDER of the last layout, because "where did it
# go" is one of the four questions the layout group exists to answer.

# The order the place groups are shown in: from the place we know exactly, through the
# ones inherited from a neighbour, down to no place at all.
_PLACE_CONFIDENCE_ORDER = ("manual", "exact_gps", "session_inferred", "trip_inferred",
                           "path_inferred", "visual", "unknown")

# The population every per-file number is counted over — exactly the files the sorter
# lays out (`plan_and_sort`), so a counter here matches what an apply will carry off.
_OVERVIEW_LIVE = "f.dup_of IS NULL AND f.error IS NULL"


def _media_class_breakdown(conn: sqlite3.Connection, column: str) -> list[dict]:
    """`verdict`/`source`/`tier` -> [{"key": …, "count": n}], the biggest group first.

    The three breakdowns are counted over the same population, so each of them sums to
    the same `classes.total` — a `tier` split that does not add up to the number of
    classified files is exactly the confusion this tab exists to remove. `tier` is NULL
    for rows written before v11; that group travels as `key: null` and the view labels it.

    The column name is interpolated into the SQL — it never comes from a request, the
    three call sites below pass literals.
    """
    rows = conn.execute(
        f"""SELECT mc.{column} AS key, COUNT(*) AS n
            FROM files f JOIN media_class mc ON mc.file_id = f.id
            WHERE {_OVERVIEW_LIVE}
            GROUP BY mc.{column}""").fetchall()
    out = [{"key": r["key"], "count": int(r["n"])} for r in rows]
    out.sort(key=lambda b: (-b["count"], b["key"] or ""))
    return out


def _overview_place(conn: sqlite3.Connection) -> dict:
    """The place group: how each frame got its place, and how many have none at all.

    A manual place (F85c) wins over `places` as a whole, exactly as the sorter reads it —
    otherwise a frame the user placed by hand would be counted here as placeless. The
    `no_place` rule is `sorter._target_parts` verbatim: an unknown confidence, or neither
    a city nor a country. Every one of those frames ends up in `_Unsorted/no_place`, which
    is why this is the one number of the group that is shown even when it is zero.
    """
    total = conn.execute(
        f"SELECT COUNT(*) FROM files f WHERE {_OVERVIEW_LIVE}").fetchone()[0]
    rows = conn.execute(
        f"""SELECT CASE WHEN mp.file_id IS NOT NULL THEN 'manual'
                        ELSE COALESCE(p.confidence, 'unknown') END AS conf,
                   COUNT(*) AS n
            FROM files f
            LEFT JOIN places p ON p.file_id = f.id
            LEFT JOIN manual_places mp ON mp.file_id = f.id
            WHERE {_OVERVIEW_LIVE}
            GROUP BY conf""").fetchall()
    no_place = conn.execute(
        f"""SELECT COUNT(*) FROM files f
            LEFT JOIN places p ON p.file_id = f.id
            LEFT JOIN manual_places mp ON mp.file_id = f.id
            WHERE {_OVERVIEW_LIVE} AND mp.file_id IS NULL
                  AND (COALESCE(p.confidence, 'unknown') = 'unknown'
                       OR (p.city IS NULL AND p.country IS NULL
                           AND p.country_name IS NULL))""").fetchone()[0]
    counts = {r["conf"]: int(r["n"]) for r in rows}
    confidence = []
    for key in _PLACE_CONFIDENCE_ORDER:
        count = counts.pop(key, 0)
        if count:
            confidence.append({"key": key, "count": count})
    # A confidence value this list does not know about is still shown, under its raw name:
    # a place the index carries must never be invisible here.
    confidence += [{"key": key, "count": count}
                   for key, count in sorted(counts.items()) if count]
    return {
        "total": int(total),
        "confidence": confidence,
        "no_place": int(no_place),
        "no_place_percent": round(100.0 * no_place / total, 1) if total else 0.0,
    }


def _overview_layout(conn: sqlite3.Connection) -> dict:
    """The layout group: was anything moved, when, where, how, and was it finished.

    Only the LAST batch is described. `finished_at IS NULL` is the trace of an interrupted
    run — the tab says so explicitly instead of showing a batch that merely looks normal.
    """
    batches = conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0]
    unfinished = conn.execute(
        "SELECT COUNT(*) FROM move_batches WHERE finished_at IS NULL").fetchone()[0]
    last = conn.execute(
        """SELECT id, mode, operation, dest_root, started_at, finished_at
           FROM move_batches ORDER BY started_at DESC, id DESC LIMIT 1""").fetchone()
    payload: dict = {"batches": int(batches), "unfinished": int(unfinished), "last": None}
    if last is None:
        return payload
    counted = conn.execute(
        """SELECT COUNT(*) AS files, COALESCE(SUM(status = 'done'), 0) AS done
           FROM moves WHERE batch_id = ?""", (last["id"],)).fetchone()
    payload["last"] = {
        "mode": last["mode"],
        "operation": last["operation"],
        "dest_root": last["dest_root"],
        "started_at": last["started_at"],
        "finished_at": last["finished_at"],
        "unfinished": last["finished_at"] is None,
        "files": int(counted["files"]),
        "done": int(counted["done"]),
    }
    return payload


def _overview_payload(db_path: Path, cfg: Config) -> dict:
    """`GET /api/overview` — the four groups of numbers the tab draws.

    `empty` is the whole answer for a fresh index: the view then invites the user to pick
    a folder instead of drawing a table of zeros.

    F152: the three face slices are counted here by the same `sorter.face_slice_ids_sql`
    the panel and the albums use, and they are the one group of rows that can answer
    `null` — without a faces run they are unmeasured, not empty, and `faces_reason` says
    so. `cfg` (rather than the single `blur_max` this used to take) is what carries the
    thresholds those three rules read.

    F126: the flat review slices are counted here too, by the SAME queries the
    workspace itself uses (`_review_flat_counts`) — a counter that disagrees with the
    list it links to is worse than no counter. The blur window comes from the same
    `features` (`blur_review_max`), so this row and that list say one number. The
    duplicates row above stays what it always was: exact copies found by hash, not the
    phash groups of the workspace, which cost seconds to build and have no place on a
    tab made of plain aggregates.
    """
    # F137 needs the thresholds and F152 needs them too, so the whole config comes in and
    # the features are unpacked once here rather than threaded as a second argument.
    features = cfg.features
    conn = _connect(db_path)
    try:
        files = conn.execute(
            """SELECT COUNT(*) AS files,
                      COALESCE(SUM(media_type <> 'video'), 0) AS photos,
                      COALESCE(SUM(media_type = 'video'), 0) AS videos,
                      COALESCE(SUM(dup_of IS NOT NULL), 0) AS duplicates,
                      COALESCE(SUM(error IS NOT NULL), 0) AS errors
               FROM files""").fetchone()
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        # F123: counted over the same population as the "Animals" tab and the animal
        # album, so the three cannot disagree. F124: which now means the one shared rule
        # (`_animals_count_sql` -> `sorter.animal_ids_sql`) — a frame the user unmarked
        # leaves this number exactly as it leaves the album. F137: and a threshold the
        # user edited moves it here, in the tab and in the album together.
        animals = conn.execute(_animals_count_sql(cfg)).fetchone()[0]
        faces_ran = faces_stage_ran(conn)
        faces_counts: dict[str, int | None] = {
            name: (_face_slice_count(conn, cfg, name) if faces_ran else None)
            for name in FACE_SLICES
        }
        review = _review_flat_counts(conn, features)
        place = _overview_place(conn)
        classes_total = conn.execute(
            f"""SELECT COUNT(*) FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE {_OVERVIEW_LIVE}""").fetchone()[0]
        updated_at = conn.execute(
            f"""SELECT MAX(mc.updated_at) FROM files f
                JOIN media_class mc ON mc.file_id = f.id
                WHERE {_OVERVIEW_LIVE}""").fetchone()[0]
        tiers = _media_class_breakdown(conn, "tier")
        classes = {
            "total": int(classes_total),
            "verdicts": _media_class_breakdown(conn, "verdict"),
            "sources": _media_class_breakdown(conn, "source"),
            "tiers": tiers,
            # "Did the deep tier run at all" — the question that used to be answered by a
            # query into the database. A file the vlm tier deliberately skipped keeps
            # source='clip' but tier='vlm', so the TIER is what answers it (schema v11).
            "vlm_ran": any(t["key"] == "vlm" for t in tiers),
            "updated_at": updated_at,
        }
        layout = _overview_layout(conn)
    finally:
        conn.close()
    return {
        "empty": int(files["files"]) == 0,
        "collection": {
            "files": int(files["files"]),
            "photos": int(files["photos"]),
            "videos": int(files["videos"]),
            "duplicates": int(files["duplicates"]),
            "errors": int(files["errors"]),
            "events": int(events),
            "animals": int(animals),
            # F152: `null` where the faces stage never ran — the F125 rule, and the same
            # distinction `/api/face-slices` draws between "none" and "not asked".
            "with_people": faces_counts["people"],
            "group_photos": faces_counts["group"],
            "portraits": faces_counts["portrait"],
            "faces_reason": None if faces_ran else "no_faces_run",
            "blurred": review["blurred"],
            "eyes_closed": review["eyes"],
            # F150: the same query the slice itself runs, so the row and the list it
            # links to cannot say two different numbers.
            "low_resolution": review["low_resolution"],
        },
        "place": place,
        "classes": classes,
        "layout": layout,
    }


def _validate_album_payload(
    payload: object,
) -> tuple[str, str, str, list[str], str | None, bool, str | None] | None:
    """Parse the body `POST /api/album`. None -> invalid (400).

    kind/mode — from `ALBUM_KINDS`/`ALBUM_MODES` (sorter.py), selector — a non-empty
    string, `where` (opt.) — a list of strings, `name` (opt.) — a string (empty after
    strip is treated as absent — the default name is used), `apply` (opt., default
    False) — bool, `dest` (opt., F60) — the album destination path as a string;
    empty/absent -> None (the server resolves the default itself via `_album_dest`).

    F123: `kind='animal'` is the one kind with nothing to select — the collection has a
    single animal slice — so an empty selector is accepted there (and only there: for a
    person or an event an empty selector is a client that lost its subject, and
    gathering "everything" would be the wrong answer to it).
    F139: the class and quality slices join it, and F152 the three face slices, by the
    same rule and through the same shared list (`SELECTORLESS_ALBUM_KINDS`).

    Whether a KIND may be gathered at all is not decided here: that answer depends on
    `vlm.exclude_classes` and is given by the route, which has the config (F133).
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in ALBUM_KINDS:
        return None
    mode = payload.get("mode")
    if mode not in ALBUM_MODES:
        return None
    selectorless = kind in SELECTORLESS_ALBUM_KINDS
    selector = payload.get("selector", "" if selectorless else None)
    if not isinstance(selector, str):
        return None
    if not selectorless and not selector.strip():
        return None
    where = payload.get("where", [])
    if not isinstance(where, list) or not all(isinstance(w, str) for w in where):
        return None
    name = payload.get("name")
    if name is not None:
        if not isinstance(name, str):
            return None
        name = name.strip() or None
    apply_ = payload.get("apply", False)
    if not isinstance(apply_, bool):
        return None
    dest = payload.get("dest")
    if dest is not None:
        if not isinstance(dest, str):
            return None
        dest = dest.strip() or None
    return kind, selector, mode, where, name, apply_, dest


def _album_report_to_json(report: AlbumReport, applied: bool) -> dict:
    """`AlbumReport` -> the JSON response body of `POST /api/album`.

    For a preview (`applied=False`) `plan_album` does not compute `blocked_multi`
    (that is a side effect of the apply loop for mode='move') — here it is recomputed
    from `report.plan` with the same logic (`item.multi_person`), so the preview shows
    the expected blocking before the real move.
    """
    blocked = report.blocked_multi
    if not applied and report.mode == "move":
        blocked = sum(1 for it in report.plan if it.multi_person)
    return {
        "album_name": report.album_name,
        "dest": str(report.dest),
        "mode": report.mode,
        "kind": report.kind,
        "count": len(report.plan),
        "blocked_multi": blocked,
        "transferred": report.transferred,
        "failed": report.failed,
        "applied": applied,
    }


# --- F134: the search line of the "Slices" tab (`GET /api/search`) ------------------
# F129 built the engine and F133 left the line drawn but disabled; this is the wiring in
# between. It carries one idea and everything else follows from it: an interface that
# cannot search says WHY, and never by showing an empty result list.
#
# `clip_embeddings` is filled by the junk stage of an ordinary run, so a fresh collection
# — and any collection last processed before F128 — has nothing to rank. "Nothing was
# found for cake" and "nothing was ever encoded" are the same empty list on screen, and
# only one of them is a fact about the archive. A person who reads the first when the
# second is true concludes something false about their own photographs, which is the
# single most expensive mistake this feature can make. So the state of the index travels
# with every answer, the line is disabled while there is nothing to search, and the
# reason stands next to it:
#
#   empty         no vectors at all           -> process the collection (an ordinary run)
#   other_model   vectors of another model    -> process it again, that index is not
#                                                comparable with this query
#   partial       some of the collection      -> searchable, and it says N of M out loud
#   ready         all of it                   -> an ordinary search line
#
# The two unavailable states are deliberately two: "run it" and "run it AGAIN because the
# model changed" are different instructions, and a single sentence covering both teaches
# the reader nothing. The partial state is not a warning but an honest denominator — an
# incremental run is the normal way to live with a growing archive, and a person has to be
# able to tell "it is not in the collection" from "it is not in the index yet".
#
# What this route does NOT do: introduce a similarity threshold. The score orders frames
# against each other and means nothing in absolute terms (see search.py), so it travels to
# the card and the reader stops where the quality runs out — the same arrangement the
# animal slice and the sharpness list already use.

_SEARCH_READY = "ready"
_SEARCH_PARTIAL = "partial"
# The unavailable states are the engine's own codes, not a second spelling of them: the
# route can be reached before and after `search_text` raises, and the two paths must not
# be able to disagree about which state the index is in.
_SEARCH_AVAILABLE_STATES = (_SEARCH_READY, _SEARCH_PARTIAL)

# The population a search ranks over and the denominator of "N of M" — the same
# `dup_of IS NULL AND error IS NULL AND media_type = 'photo'` rule `search._CANDIDATES_SQL`
# selects on, counted here rather than imported as SQL because this is a COUNT of it.
_SEARCH_PHOTOS_SQL = """SELECT COUNT(*) FROM files
    WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'"""

# How much of that population this model has a vector for. Joined to `files` on purpose:
# a row whose frame has since become a duplicate or gone unreadable is not something a
# search can return, so counting it would inflate the numerator of a fraction whose whole
# job is to be honest.
#
# F141: the table is `search_embeddings`, the multilingual index the engine actually reads
# — not `clip_embeddings`, which holds the classification model's vectors and cannot
# answer a query. Counting the other table would make this line say "searching all 19 753
# photographs" over an index the search will refuse to use, which is the one thing this
# route exists to prevent.
_SEARCH_COVERED_SQL = """SELECT COUNT(*) FROM search_embeddings e
    JOIN files f ON f.id = e.file_id
    WHERE e.model = ? AND f.dup_of IS NULL AND f.error IS NULL AND f.media_type = 'photo'"""

# F189: whether anybody in this collection has a NAME — the roots of the `merged_into`
# chains, which is where `search.match_person` looks. It travels with the state because the
# line is DISABLED while the index cannot rank, and a name needs no index at all:
# `features.search_index` is off by default, so without this the feature would be invisible
# on a fresh collection — a person typing the name of their own daughter into a dead field.
_SEARCH_NAMES_SQL = """SELECT EXISTS(
    SELECT 1 FROM face_clusters WHERE merged_into IS NULL AND label IS NOT NULL)"""

# One card, and the same shape whichever state produced it. LEFT JOIN because a photograph
# usually has no `media_class` row at all — the class is what the privacy rule below reads.
_SEARCH_ROWS_SQL = """SELECT f.id, f.path, f.taken_at, mc.verdict
    FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
    WHERE f.id IN ({marks})"""

# F173: a limit is a SAMPLE SIZE (search.py) and a page of one at that, so the ceiling on
# how much may be rendered at once is `_PLAN_PAGE_MAX_LIMIT` like everywhere else — a
# client asking for more gets less rather than an error. It used to be a constant of its
# own with the same value, which is one more place a rule could drift.


def _search_index_state(conn: sqlite3.Connection, model: str) -> dict:
    """Which of the four states the index is in, plus the numbers that state it.

    `index_model` is what a person is told when the answer is "another model": the name of
    the model that actually produced the stored vectors, taken as the one with the most
    rows. Naming it is the difference between a sentence somebody can act on and a shrug —
    and the row count is how the name is chosen, because a table can hold leftovers of
    several models and only the dominant one is worth putting in front of a reader.

    `indexed` counts vectors of THIS model within the searchable population, `photos` the
    population itself. The pair is the "we are searching N of M photographs" line, and it
    is computed here, once, so the line and the availability of the field cannot disagree.
    """
    counts = {str(r["model"]): int(r["n"]) for r in conn.execute(
        "SELECT model, COUNT(*) AS n FROM search_embeddings GROUP BY model")}
    stored = counts.get(model, 0)
    photos = int(conn.execute(_SEARCH_PHOTOS_SQL).fetchone()[0])
    indexed = int(conn.execute(
        _SEARCH_COVERED_SQL, (model,)).fetchone()[0]) if stored else 0
    others = [(n, name) for name, n in counts.items() if name != model]
    if not counts:
        state = REASON_EMPTY
    elif not stored:
        state = REASON_OTHER_MODEL
    elif not indexed:
        # Vectors of this model exist and not one of them belongs to a frame a search may
        # return. There is nothing to rank and running the stage again is the fix, so this
        # is the empty state — exactly what `search._nothing_to_rank` calls it.
        state = REASON_EMPTY
    else:
        state = _SEARCH_PARTIAL if indexed < photos else _SEARCH_READY
    return {
        "state": state,
        "available": state in _SEARCH_AVAILABLE_STATES,
        "model": model,
        "index_model": model if stored else (max(others)[1] if others else None),
        "indexed": indexed,
        # F189: not part of `available` — the index is still in whatever state it is in,
        # and the sentence about it does not change. What this adds is that the line has
        # something to answer even so.
        "names": bool(conn.execute(_SEARCH_NAMES_SQL).fetchone()[0]),
        # F173: `photos`, not `total`. This route answers with a PAGE of a ranking now, and
        # in every paged payload of this server `total` means the length of the list being
        # walked. Two numbers called the same thing in one answer is how a counter starts
        # saying "showing 200 of 19 753 photographs in the collection" about a list of
        # 4 000 — so the coverage line's denominator got the name of what it counts.
        "photos": photos,
    }


def _search_item_to_json(row: sqlite3.Row, score: float, sensitive: frozenset[str],
                         scored: bool = True) -> dict:
    """One card of the ranking: the score is always on it, the thumbnail sometimes.

    F189: `scored=False` for a card of a SELECTION — a person's frames — and then the key
    is absent rather than zero. The number explains an order; this list has no order to
    explain, and «близость 0.000» under every frame of somebody's daughter would be a
    measurement nobody made.

    F133's rule, unchanged: a frame whose class sits in `vlm.exclude_classes` (documents
    by default) gets no `thumb_url`, so the browser never asks `/thumb` for it and no
    preview of a passport is ever decoded. The guard is here, on the server, for the
    reason it is there — a search that answered with a link would turn this route into
    the way around a protection the slices already apply.
    """
    path = Path(row["path"])
    payload: dict = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
    }
    if scored:
        # A ranking, not a filter: the number is what lets a reader see where the
        # relevance ran out, and a card without it would hide exactly that.
        payload["score"] = float(score)
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _search_items(conn: sqlite3.Connection, hits: Sequence[tuple[int, float]],
                  sensitive: frozenset[str], scored: bool = True) -> list[dict]:
    """The engine's (file_id, score) pairs -> cards, IN THE RANKING'S ORDER.

    The rows are fetched in chunks (a limit is user-set and SQLite has a ceiling on bound
    parameters — the `search.file_paths` reason) and then re-ordered by the ranking, never
    by whatever order SQLite returned: the order is the answer here.
    """
    rows: dict[int, sqlite3.Row] = {}
    for part in batched([fid for fid, _score in hits], 500):
        marks = ",".join("?" * len(part))
        rows.update({int(r["id"]): r for r in conn.execute(
            _SEARCH_ROWS_SQL.format(marks=marks), tuple(part))})
    return [_search_item_to_json(rows[fid], score, sensitive, scored)
            for fid, score in hits if fid in rows]


# --- F189: the search line answers a NAME with the person ------------------------------
# The question this closes: a cluster somebody named, and merged another cluster into, was
# reachable by `album person <name>` and by `sort --by person` and by no query anybody could
# type. «Ирина» in the search line asked CLIP for frames resembling a WORD.
#
# The bridge is a parse of the query string and nothing else — no index, no threshold, no
# cluster work — and it is deliberately in ONE place for the whole server: the typed line
# (`/api/search`) and a pinned slice of the same words (`/api/saved-slices`, F156) have to
# answer identically, or a pin becomes a second engine with a name.
#
# What travels to the client is two flags rather than a merged list:
#
#     person   the name this string is, whenever it is one — even when the answer being
#              served is the ranking, because the offer of the other answer is the point
#     exact    whether THIS payload is the person's frames. It decides the caption, and a
#              caption is how a reader tells an exact selection from the top of a ranking
#
# Requirement 4 lives in that pair: a name that is also an ordinary word («Роза», «Марк»)
# shows the person first and keeps the ranking one click away — the second answer never
# disappears silently, which is what a search line quietly hijacked by a name would do.


def _person_payload(conn: sqlite3.Connection, cfg: Config, label: str, offset: int,
                    limit: int) -> dict:
    """One page of a person's frames, in the shape every paged slice of this server has.

    `exact: true` is the whole difference on the wire, and the client draws a different
    sentence from it. The cards carry no score (`scored=False`): there is no order here to
    explain.
    """
    page = person_page(conn, label, limit=limit, offset=offset)
    return {
        "person": label,
        "exact": True,
        **_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes),
                          scored=False),
            total=page.total, offset=page.offset, limit=page.limit),
    }


def _search_payload(cfg: Config, db_path: Path, text: str, offset: int, limit: int,
                    encoder: TextEncoder | None = None, words: bool = False) -> dict:
    """`GET /api/search` — the state of the index always, a page of the ranking when there
    is one.

    The model is not asked anything unless there is a reason to: an empty query and an
    unavailable index both return before `rank_text`, which is what keeps a stray
    keystroke from loading CLIP and what makes "the line is disabled" cheap to render.

    `EmbeddingsMissing` is still caught, because the state was read a moment earlier and a
    run can empty the table in between; the answer then carries the engine's own reason
    rather than an empty `items` list, which is the one thing this route must never send.

    F173: a page rather than the whole answer, in the shape `_page_payload` gives every
    other paged slice. `total` is the length of the RANKING — the number the counter says
    and the number the "show more" button is decided by — and it comes back from the engine
    with the page, so the two cannot be computed out of step with each other. A state that
    ranks nothing still carries `total: 0` and `has_more: false`: the client draws the same
    controls whatever happened, and they are simply not there when there is nothing below.

    F189: a string that IS somebody's name is answered with that person's frames, before
    the index is consulted at all — a selection out of `face_clusters` needs no vector, so
    a name still finds the person on a collection nobody has indexed yet. `words=True` is
    how the client asks for the ranking anyway, which is the other half of the same rule:
    the name never takes the word search away, it only goes first.
    """
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)  # F141: the search model, not the classifier's
        payload = _search_index_state(conn, model)
        payload.update({"query": text, "person": None, "exact": False,
                        **_page_payload([], total=0, offset=offset, limit=limit)})
        # Computed even when the ranking is what gets served: the client offers the other
        # answer, and it can only offer what the payload names.
        person = match_person(conn, text)
        payload["person"] = person
        if person is not None and not words:
            payload.update(_person_payload(conn, cfg, person, offset, limit))
            return payload
        if not text.strip() or not payload["available"]:
            return payload
        try:
            page = rank_text(cfg, conn, text, limit=limit, offset=offset, encoder=encoder)
        except EmbeddingsMissing as exc:
            payload["state"] = exc.reason
            payload["available"] = False
            return payload
        payload.update(_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes)),
            total=page.total, offset=page.offset, limit=page.limit))
        return payload
    finally:
        conn.close()


def _parse_search_query(query: dict[str, list[str]],
                        default_limit: int) -> tuple[str, int, int, bool] | None:
    """(query text, offset, limit, words) for `GET /api/search`, or None -> 400.

    An absent/empty `q` is NOT an error: the client asks with one on purpose, to learn the
    state of the index without spending a model on it. The window is the shared
    `_parse_page_window` — a non-integer or a negative number is rejected, an over-eager
    limit is clamped — with `features.search_page` as the default size of a page.

    F189: `words=1` asks for the ranking even when the string names somebody. Anything else
    (absent, `0`, a typo) means the default, which is the person — a malformed flag must not
    be a 400 on a route whose whole job is to answer.
    """
    window = _parse_page_window(query, default_limit)
    if window is None:
        return None
    return ((query.get("q") or [""])[0], window[0], window[1],
            (query.get("words") or [""])[0] == "1")


# --- F151: the pinned queries of the "Slices" tab (`GET /api/saved-slices`) ------------
# A slice is a saved query. The measurement of 2026-08-02 (200 frames out of 22 096,
# labelled by hand, the first time RECALL was measured rather than the precision of the
# top) is what turned the feature around: the six hand-written filters find 6% of the
# blurred frames, 33% of the animals, 0% of the products and have nothing at all for
# children — while the SAME vectors, asked in words, give 61% for children, 65% for
# products and 60% for animals at the same depth, and 89% / 95% / 87% at twice it.
#
# So this route adds no model, no pass and no table: the vectors are the junk stage's
# (F128/F141), the ranking is F129's, the paging is F173's, and the only new thing on the
# server is WHERE the words come from — `features.saved_slices`, a config entry rather
# than code, so a slice can be retuned or added without a release.
#
# Three properties are the feature and each is a decision:
#
# * these lists are ESTIMATES and are labelled apart from the exact ones. The `pet` label
#   next to them is 71% precise and verified by a model; this ranking is 60% and verified
#   by nobody. Both slices stay, because they answer different questions ("is this
#   confidently an animal" against "show me every animal"), and if their captions matched
#   a reader would take one for the other;
# * no count on a pin, and no threshold anywhere. A ranking covers the whole index, so its
#   length is not a number of children; where the list stops being about the query is a
#   judgement, and the person reading it makes it;
# * depth is the lever. The page is `features.search_page` and "show more" continues the
#   same ranking — the one handle the measurement confirmed (61% -> 89%).
#
# Not here on purpose: PEOPLE (the signal is `faces`, 7 341 frames, exact and free — F152
# already draws it) and BLURRED (the sharpness filter is 100% precise on the sample and
# the query 36%; merging them is a different feature, and the exact half has to come
# first or it drowns).


def _saved_slice_by_name(cfg: Config, name: str) -> SavedSlice | None:
    for slice_ in cfg.features.saved_slices:
        if slice_.name == name:
            return slice_
    return None


def _saved_slices_payload(cfg: Config, db_path: Path, name: str | None, offset: int,
                          limit: int, encoder: TextEncoder | None = None) -> dict:
    """`GET /api/saved-slices` — the pins always, one page of the asked-for slice.

    The shape is `_search_payload`'s and deliberately so: a pinned slice IS a search, so
    the state of the index travels with every answer and an index that cannot rank says
    which of the two unavailable states it is in instead of coming back as an empty list.
    That rule is worth more here than in the search line — nobody types "children" into a
    pin, so an empty page would be read as a fact about the archive rather than as a
    question that missed.

    `name=None` is the tab's own call on open: the pins and the state, no ranking, no
    model. The phrases travel with the page because the panel prints them — a slice whose
    words are invisible cannot be edited by the person it is wrong for.

    F189: a pin whose single phrase is somebody's NAME answers with that person's frames,
    exactly as the search line does for the same string. Pinning is how a named person
    becomes an ordinary tab and it was supposed to cost nothing — but a pin that ranked
    «Ирина» by CLIP while the search line selected her cluster would be two answers under
    one word, and the divergence would be silent. A pin of SEVERAL phrases is a query and
    stays one: a name averaged with other words is not a name.
    """
    # The LIVE config, in the file's own order — that order is the order of the pins.
    slices = cfg.features.saved_slices
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)
        current = _saved_slice_by_name(cfg, name) if name else None
        payload = _search_index_state(conn, model)
        payload.update({
            "slices": [{"slice": s.name, "queries": list(s.queries)} for s in slices],
            "slice": current.name if current else None,
            "queries": list(current.queries) if current else [],
            # The one word the client needs to caption these lists apart from the exact
            # slices beside them. A constant rather than a per-slice flag: everything this
            # route serves is a ranking, and the day one of them is not, it will not be
            # served from here.
            "approximate": True,
            # F156: how many pins the interface may add (`features.max_pinned_slices`).
            # It travels with every answer so the "pin this" button can say the limit is
            # reached BEFORE somebody types a name for a slice that will be refused.
            "max_pinned": int(cfg.features.max_pinned_slices),
            # F189: the same two flags the search line sends, so the panel captions a
            # pinned person the way it captions a typed one.
            "person": None,
            "exact": False,
            **_page_payload([], total=0, offset=offset, limit=limit),
        })
        if current is None:
            return payload
        person = (match_person(conn, current.queries[0])
                  if len(current.queries) == 1 else None)
        if person is not None:
            payload.update(_person_payload(conn, cfg, person, offset, limit))
            # This list is a fact and not an estimate, and the word that says so is the
            # one the panel prints beside every ranking on this tab.
            payload["approximate"] = False
            return payload
        if not payload["available"]:
            return payload
        try:
            page = rank_queries(cfg, conn, current.queries, limit=limit, offset=offset,
                                encoder=encoder)
        except EmbeddingsMissing as exc:
            payload["state"] = exc.reason
            payload["available"] = False
            return payload
        payload.update(_page_payload(
            _search_items(conn, page.hits, frozenset(cfg.vlm.exclude_classes)),
            total=page.total, offset=page.offset, limit=page.limit))
        return payload
    finally:
        conn.close()


def _parse_saved_slice_query(cfg: Config, query: dict[str, list[str]],
                             default_limit: int) -> tuple[str | None, int, int] | None:
    """(slice name or None, offset, limit) for `GET /api/saved-slices`, or None -> 400.

    An absent `slice` is NOT an error — it is how the pin row is asked for. A `slice` that
    is not in the config IS one, the `_parse_face_slice_query` rule: answering it with an
    empty page would show a slice that does not exist as one holding no photographs.
    """
    window = _parse_page_window(query, default_limit)
    if window is None:
        return None
    name = (query.get("slice") or [""])[0].strip()
    if name and _saved_slice_by_name(cfg, name) is None:
        return None
    return (name or None), window[0], window[1]


# --- F156: pinning a query of one's own (`POST /api/saved-slices/{pin,unpin,move}`) ----
# The measurement that turned this feature around (2026-08-02, a random sample of 200
# frames): 65 of them — a third — fall into no class at all, and the ten candidate slices
# for those 65 cover 26%, 23%, 22%, 20%, 18%, 17%, 15%, 12%, 12%, 6%. Not one of them
# reaches a third of a third. Ten slices for 65 frames out of 200 is the thirteen-control
# remote F133 took apart, and food — which both the user and the author had in mind as a
# large slice — came out at 8 frames, smaller than sky or signage.
#
# So the product stops guessing which facets matter. For one person they are mountains and
# children, for another receipts and cars, and nobody but the owner of the archive knows
# which. The mechanism is unchanged (F129 ranks, F151 pins) — what is new is WHO writes
# the list.
#
# Three properties are the feature:
#
# * the list lives in `config.yaml`, beside the slices that ship. The index does not
#   survive `reset` or a re-processing and the config file does, and a slice somebody named
#   is not something to lose to a re-index;
# * a pin is a SAVED QUERY and nothing else, so a pinned slice is indistinguishable from a
#   built-in one on screen — the same grid, the same album, the same counter — and it is
#   removed by unpinning it, which deletes a config entry and touches no file;
# * the number of pins is bounded (`features.max_pinned_slices`) and reaching the bound is
#   SAID. A pin that silently does not appear is worse than no pin.
#
# No suggestions, ever: the product does not offer to pin anything for you. That is the
# whole point of the feature, and a "you might want to pin «food»" would be the guessing
# it replaces, wearing a friendlier hat.

# Why a pin was refused, in one word the client can caption. Not a sentence: the reason
# has to be shown in the interface language, so the server sends the code and the catalog
# holds the three sentences.
_PIN_EMPTY = "empty"          # nothing was typed — there is no query to save
_PIN_DUPLICATE = "duplicate"  # a pin of that name is already there
_PIN_LIMIT = "limit"          # `features.max_pinned_slices` is reached


def _validate_pin_payload(payload: object) -> tuple[str, str] | None:
    """Parse the body of `POST /api/saved-slices/pin` -> (name, query). None -> 400.

    `query` is the text that was typed and is required; `name` is optional and defaults to
    the query itself, which is what the field is pre-filled with. Both are stripped, and an
    empty query is refused HERE rather than in the interface: a slice with no words would
    rank the collection by an arbitrary direction and look exactly like an answer
    (`search.encode_queries` refuses it for the same reason).
    """
    if not isinstance(payload, dict):
        return None
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        return None
    return ((name or "").strip() or query.strip()), query.strip()


def _validate_slice_name_payload(payload: object) -> str | None:
    """`{"slice": "<name>"}` -> the name, for unpin and move. None -> 400."""
    if not isinstance(payload, dict):
        return None
    name = payload.get("slice")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _validate_move_payload(payload: object) -> tuple[str, int] | None:
    """`{"slice": …, "delta": -1|1}` -> (name, delta), the arrows. None -> 400.

    Arrows and not drag-and-drop, one of the two and deliberately the smaller: the order
    is a list of at most a dozen names, a keyboard reaches an arrow, and a drop target is a
    second way to say the same thing that would then have to agree with the first.
    """
    name = _validate_slice_name_payload(payload)
    if name is None or not isinstance(payload, dict):
        return None
    delta = payload.get("delta")
    if isinstance(delta, bool) or delta not in (-1, 1):
        return None
    return name, int(delta)


def _pinned_with(cfg: Config, name: str, query: str) -> tuple[SavedSlice, ...] | str:
    """The pinned list with one more slice in it, or the code saying why it cannot be.

    The new pin goes to the END of the list, where the person who made it will look for
    it: the order is theirs to change afterwards, and inserting somewhere clever would be
    the product having an opinion about a list it does not own.

    Emptiness is not one of the answers here — `_validate_pin_payload` has already
    refused it with `_PIN_EMPTY`, and a second copy of that rule is a second place for it
    to drift.
    """
    slices = tuple(cfg.features.saved_slices)
    if any(existing.name == name for existing in slices):
        return _PIN_DUPLICATE
    if len(slices) >= int(cfg.features.max_pinned_slices):
        return _PIN_LIMIT
    return (*slices, SavedSlice(name, (query.strip(),)))


def _pinned_without(cfg: Config, name: str) -> tuple[SavedSlice, ...] | None:
    """The pinned list with that slice gone, or None when there is no such slice.

    Nothing but the config entry is removed. Unpinning is not a deletion of anything on
    disk, and the confirmation the interface asks for says so — the frames the slice ranked
    are the collection's and were never the slice's to hold.
    """
    slices = tuple(cfg.features.saved_slices)
    kept = tuple(s for s in slices if s.name != name)
    return kept if len(kept) != len(slices) else None


def _pinned_moved(cfg: Config, name: str, delta: int) -> tuple[SavedSlice, ...] | None:
    """The pinned list with that slice one step up or down. None -> no such slice.

    A step off either end is a no-op rather than an error: the arrow at the top of the list
    does nothing, which is what an arrow at the top of a list does.
    """
    slices = list(cfg.features.saved_slices)
    index = next((i for i, s in enumerate(slices) if s.name == name), None)
    if index is None:
        return None
    target = index + delta
    if 0 <= target < len(slices):
        slices[index], slices[target] = slices[target], slices[index]
    return tuple(slices)


def _apply_saved_slices(cfg: Config, slices: tuple[SavedSlice, ...]) -> None:
    """Put the new pin list into the RUNNING config, `raw` included.

    `raw` is mirrored for the reason `_apply_settings` mirrors its own section: a later
    save of anything else must not write back the mapping this call just replaced.
    """
    cfg.features = dataclasses.replace(cfg.features, saved_slices=slices)
    section = cfg.raw.get("features")
    if not isinstance(section, dict):
        section = {}
        cfg.raw["features"] = section
    section["saved_slices"] = {s.name: list(s.queries) for s in slices}


class _LazyTextEncoder:
    """The CLIP text tower of this server: loaded on the first query, then reused.

    The same arrangement as `_LazyClassifierHolder` and for the same two reasons — the
    model must not be loaded by merely starting the UI (most sessions never search), and
    it must not be loaded twice, since the search route and the album route both encode
    text. Tests replace `ui.text_encoder`, so the whole feature runs without a model.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._encoder: TextEncoder | None = None
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> Any:
        with self._lock:
            if self._encoder is None:
                # F141: the SEARCH model's text tower — a query has to land in the space
                # the stored vectors live in, and since F141 that space is not the
                # classification model's.
                self._encoder = text_encoder(search_index_settings(
                    naming_settings(self._cfg), search_index_model(self._cfg)))
            encoder = self._encoder
        return encoder(texts)


# --- F36: "Process" — the background pipeline index→geo→landmarks→classify→faces→
# events→junk→phash from the web (POST /api/process), pollable progress (GET
# /api/process/status), cancel (POST /api/process/cancel). NOT imported from cli.py
# (to avoid a cli<->ui cycle) — the same leaf functions as `cli._pipeline_steps` are
# called directly from indexer/geo/landmarks/faces/events/junk/dedup/naming, +
# compute_phashes (dedup) as the last step.

# F165: `classify` — the front half of the junk stage (the verdicts, `verdicts_only`),
# placed before `faces` so that the faces stage skips the frames the classifier has
# already called screenshots, documents, memes or products. The back half keeps its
# place: everything left in it reads what `faces` writes.
_PIPELINE_STAGE_NAMES = ("index", "geo", "landmarks", "classify", "faces", "events",
                         "junk", "phash")

# F53/#39: faces and events — the heaviest/longest steps, opt-in via the "Process"
# checkboxes, default off. `_pipeline_steps()` still builds the FULL list (see the
# assert above by _PIPELINE_STAGE_NAMES) — filtering is up to the caller
# (`_run_pipeline`), with the same name list as `cli._OPTIONAL_STAGES`.
_OPTIONAL_STAGES = ("faces", "events")

# F135: with one button the run always walks the whole pipeline, and a stage that
# skipped everything looks exactly like a stage that did nothing. A step may report
# `{"processed": n, "skipped": m}` — the same two numbers the CLI prints ("skipped as
# already processed") — and the status snapshot carries them to the client. `None`
# means the stage cannot tell the two apart, and then nothing is claimed about it.
_StageStats = dict[str, int] | None
_StageFn = Callable[[Config, sqlite3.Connection, "_ProgressCB"], _StageStats]


def _stage_stats(stats: object, processed: tuple[str, ...], skipped: str) -> _StageStats:
    """Sum the `processed` counters of a stage's stats object and read `skipped` off it.

    None when any of the names is missing or does not hold a number. Stages are
    replaceable (tests swap the whole leaf function, a future one may stop returning
    stats at all), and a caption at the bottom of the page is worth neither an
    exception in the pipeline thread nor a fabricated zero — "skipped: 0" would claim
    a stage skipped nothing where in truth it said nothing.
    """
    values: list[int] = []
    for name in (*processed, skipped):
        value = getattr(stats, name, None)
        if not isinstance(value, int):
            return None
        values.append(value)
    return {"processed": sum(values[:-1]), "skipped": values[-1]}


class _LazyClassifierHolder:
    """Builds the CLIP classifier on the first call, reuses it between landmarks and
    junk within ONE `/api/process` run (the same reason as
    `cli._LazySharedClassifier`, F19: a shared image-feature cache for the whole run).
    Laziness preserves incrementality — a run without new unknown places and without
    new files for junk does not load the CLIP model at all.
    """

    def __init__(self, factory: Callable[[], Classifier]) -> None:
        self._factory = factory
        self._real: Classifier | None = None

    def __call__(self, paths: list[str], prompts: list[str]):
        if self._real is None:
            self._real = self._factory()
        return self._real(paths, prompts)

    def features(self, paths: list[str]) -> list[Any]:
        """The CLIP vectors of the paths already scored — the F128 half of the junk stage.

        F146: without this method the holder is not the classifier that stage expects.
        `junk.classify` decides whether it can fill `clip_embeddings` by looking for
        `features` on the object it was handed, so a wrapper forwarding `__call__` alone
        turned the whole half off — silently, and for every run started from the web app,
        which is where most runs are started.

        Laziness is untouched: a classifier that has not been built has scored nothing, so
        its cache holds nothing and every path is None — the same answer
        `landmarks.CachingFeatureClassifier` gives for a path nobody has scored, and no
        model is loaded to give it.
        """
        features_of = getattr(self._real, "features", None)
        if not callable(features_of):
            return [None] * len(paths)
        return list(features_of(paths))


def _pipeline_steps() -> list[tuple[str, _StageFn]]:
    """Processing steps in dependency order — the same as `cli._pipeline_steps`, plus
    `phash` last (canonically from cli _pipeline_steps).
    A fresh holder per call — a separate run does not share the CLIP classifier with
    the previous/next run.

    F135: a step returns `{"processed": n, "skipped": m}` where the stage's own stats
    can separate new work from what it recognised as already done — `index` (unchanged
    files) and `junk` (the F68 incremental skip). The rest return None: inventing a
    zero for a stage that does not count skips would claim something untrue.
    """
    holder: dict[str, _LazyClassifierHolder] = {}

    def _clip(cfg: Config) -> _LazyClassifierHolder:
        clf = holder.get("clip")
        if clf is None:
            clf = holder["clip"] = _LazyClassifierHolder(
                lambda: clip_classifier(naming_settings(cfg)))
        return clf

    def _index(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        assign_duplicates(conn, cfg.dedup.canonical_strategy)
        # `added + updated` is the work; `skipped` is what path+mtime+size recognised
        # as unchanged — the same split `cli._summarize_index` prints.
        return _stage_stats(stats, ("added", "updated"), "skipped")

    def _geo(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        resolve_places(cfg, conn, progress=cb)
        return None

    def _landmarks(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        detect_landmarks(cfg, conn, classifier=_clip(cfg), progress=cb)
        return None

    def _faces(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        detect_and_cluster(cfg, conn, progress=cb)
        return None

    def _events(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)
        return None

    def _classify(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg), verdicts_only=True,
                              progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _junk(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _phash(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        compute_phashes(cfg, conn, progress=cb)
        return None

    steps: list[tuple[str, _StageFn]] = [
        ("index", _index), ("geo", _geo), ("landmarks", _landmarks),
        ("classify", _classify), ("faces", _faces), ("events", _events),
        ("junk", _junk), ("phash", _phash),
    ]
    assert tuple(name for name, _fn in steps) == _PIPELINE_STAGE_NAMES
    return steps


class _PipelineCancelled(BaseException):
    """Pipeline cancellation from the progress callback (mid-stage). BaseException,
    not Exception, so an `except Exception` inside stages does not swallow it;
    caught only in `_run_pipeline`."""


class _ProcessState:
    """Thread-safe state of the background `/api/process` pipeline (F36).

    One run per server: `try_start` under the same `_lock` as all other mutations
    atomically rejects a repeated start while the previous one is still `running` —
    the `POST /api/process` handler turns False into 409. Updated by the stages'
    progress callbacks from the pipeline thread; read by `GET /api/process/status`
    from ThreadingHTTPServer request threads — hence a lock on every operation, not
    just a dataclass of fields.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.stage: str | None = None
        self.stage_index = 0
        self.stage_total = 0
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.finished = False
        self.source_dir: str | None = None
        self.phase: str | None = None
        self._phase_started = 0.0
        self._cancel_requested = False
        # F135: per-stage {"processed", "skipped"} of THIS run — see `_stage_stats`.
        self.stage_stats: dict[str, dict[str, int]] = {}

    def try_start(self, source_dir: str) -> bool:
        """True and switches to running if nothing is going now; otherwise False (409)."""
        with self._lock:
            if self.running:
                return False
            self._reset_locked()
            self.running = True
            self.source_dir = source_dir
            return True

    def set_stage_total(self, total: int) -> None:
        with self._lock:
            self.stage_total = total

    def set_stage(self, index: int, name: str) -> None:
        with self._lock:
            self.stage_index = index
            self.stage = name
            self.done = 0
            self.total = 0
            self.phase = None
            self._phase_started = 0.0

    def set_stage_stats(self, name: str, stats: dict[str, int]) -> None:
        """F135: what the finished stage `name` processed and what it skipped."""
        with self._lock:
            self.stage_stats[name] = dict(stats)

    def set_progress(self, done: int, total: int | None = None) -> None:
        """A signature superset of all stage ProgressCB variants (done, total|None).

        If cancellation is requested — raises _PipelineCancelled right from the
        callback: stages call progress often, so cancellation fires almost
        immediately (mid-stage), not only between stages.

        `total=None` zeroes the total instead of keeping the previous one (F84): a
        stage can go from a measurable phase to an unmeasurable one (faces: detection
        by frames -> HDBSCAN), and a total left over from the previous phase would
        keep drawing a filled bar with numbers that mean nothing.
        """
        with self._lock:
            cancel = self._cancel_requested
            if not cancel:
                self.done = done
                self.total = total if total is not None else 0
        if cancel:
            raise _PipelineCancelled()

    def set_phase(self, phase: str | None) -> None:
        """The named sub-phase of the current stage (F84), or None — no phase.

        The clock starts over on every change: on a phase without a percent the
        elapsed time is the only honest sign of life the bar can show.
        """
        with self._lock:
            self.phase = phase
            self._phase_started = time.monotonic() if phase else 0.0

    def request_cancel(self) -> None:
        with self._lock:
            if self.running:
                self._cancel_requested = True

    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def finish(self, error: str | None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.phase = None  # a finished run is not in any phase (F84)
            self._phase_started = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "stage_total": self.stage_total,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "finished": self.finished,
                "cancel_requested": self._cancel_requested,
                # F135: also what puts the source of the last run back into an empty
                # field — with one button the path has to come back by itself, and the
                # browser's own memory is not there in a fresh profile.
                "source_dir": self.source_dir,
                # F135: {stage: {"processed", "skipped"}} for the stages that can tell
                # new work from work recognised as already done.
                "stage_stats": {name: dict(values)
                                for name, values in self.stage_stats.items()},
                # F84: the sub-phase of the current stage and how long it has been
                # running. phase=None -> the stage reports no phases (every stage but
                # faces), and the client draws exactly what it drew before.
                "phase": self.phase,
                "phase_elapsed": (round(time.monotonic() - self._phase_started, 1)
                                  if self.phase else 0.0),
            }


class _StageProgress:
    """The callback a pipeline stage gets: `(done, total)` plus a `phase` channel (F84).

    Stages that know nothing about phases just call it, exactly as they called
    `state.set_progress` before. `faces` reports the phases of clustering through
    `.phase(name)` — the same duck-typed channel `progress.TaskProgress` gives the CLI.
    """

    def __init__(self, state: _ProcessState) -> None:
        self._state = state

    def __call__(self, done: int, total: int | None = None) -> None:
        self._state.set_progress(done, total)

    def phase(self, name: str) -> None:
        self._state.set_phase(name)


_BROWSE_DIALOG_TIMEOUT_S = 120
# Serialises the folder dialog — see _browse_for_folder.
_browse_lock = threading.Lock()

_BROWSE_DIALOG_SCRIPT = (
    "import tkinter, tkinter.filedialog, sys\n"
    "root = tkinter.Tk()\n"
    "root.withdraw()\n"
    "root.attributes('-topmost', True)\n"
    "path = tkinter.filedialog.askdirectory()\n"
    "root.destroy()\n"
    "sys.stdout.write(path or '')\n"
)


def _browse_for_folder() -> str:
    """F51: a native folder-picker dialog for the "Browse…" button.

    tkinter is not thread-safe and requires the process's main thread — the
    POST /api/browse handler runs on a ThreadingHTTPServer thread, so the dialog is
    opened in a SEPARATE process (its own main thread, without a conflict with the web
    server). Any failure (no display/GUI, cancel, timeout, exception) -> an empty
    string, not an error — the button is just a convenience, manual path entry always
    works.

    Only one dialog at a time: the subprocess takes a second or two to show a window,
    and every request that arrives meanwhile used to spawn another Explorer. The
    client disables its button too, but that cannot cover a second browser tab or a
    click that races the disable — the guard belongs here as well. A refused call
    returns "" (same contract as cancel), so the already-open dialog stays the one
    the user is talking to.
    """
    if not _browse_lock.acquire(blocking=False):
        return ""
    try:
        return _run_browse_dialog()
    finally:
        _browse_lock.release()


def _run_browse_dialog() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _BROWSE_DIALOG_SCRIPT],
            capture_output=True, text=True, timeout=_BROWSE_DIALOG_TIMEOUT_S,
            check=False,
        )
    except Exception:
        _log.exception("не удалось открыть диалог выбора папки")
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


# --- F81: the source folder tree ("do not scan") --------------------------------

# The response is bounded on purpose: on a pathological tree it must not blow up.
# Sizes are still summed over everything below, so the numbers stay truthful — only
# the node LIST is cut, and the answer says so instead of silently shortening.
_TREE_MAX_NODES = 2000
_TREE_MAX_DEPTH = 12


def _validate_tree_root(raw: object) -> Path | None:
    """The tree root arrives from the client, so it is checked before anything is read.

    The same rule the path behind the "Browse…" button meets in
    `_handle_process_start`: a non-empty ABSOLUTE path to an existing directory.
    Anything else (a relative path, a file, a directory that is not there) -> None ->
    400. The server never walks an arbitrary path just because it was asked to.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        return None
    try:
        if not path.is_dir():
            return None
        return path.resolve()
    except OSError:
        return None


def _sum_dir(directory: Path) -> tuple[int, int]:
    """(files, bytes) of a whole subtree — metadata only (`scandir`/`stat`)."""
    files = size = 0
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return 0, 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                sub_files, sub_size = _sum_dir(Path(entry.path))
                files += sub_files
                size += sub_size
                continue
            files += 1
            size += entry.stat(follow_symlinks=False).st_size
        except OSError:  # a vanished/unreadable entry is not worth failing the tree over
            continue
    return files, size


def _scan_dir(directory: Path, rel: str, name: str, depth: int,
              budget: list[int], max_depth: int) -> dict:
    node: dict = {"name": name, "rel": rel, "files": 0, "size": 0,
                  "children": [], "truncated": False}
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name.lower())
    except OSError:
        return node
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if not is_dir:
            node["files"] += 1
            try:
                node["size"] += entry.stat(follow_symlinks=False).st_size
            except OSError:
                pass
            continue
        child_rel = f"{rel}/{entry.name}" if rel else entry.name
        if depth < max_depth and budget[0] > 0:
            budget[0] -= 1
            child = _scan_dir(Path(entry.path), child_rel, entry.name, depth + 1,
                              budget, max_depth)
            node["children"].append(child)
            node["files"] += child["files"]
            node["size"] += child["size"]
        else:
            # over the limit: the folder is not sent, but its files still count
            sub_files, sub_size = _sum_dir(Path(entry.path))
            node["files"] += sub_files
            node["size"] += sub_size
            node["truncated"] = True
    return node


def _any_truncated(node: dict) -> bool:
    return bool(node["truncated"]) or any(_any_truncated(c) for c in node["children"])


def _source_tree_payload(root: Path, skip_scan: list[str], skip_layout: list[str],
                         max_nodes: int = _TREE_MAX_NODES,
                         max_depth: int = _TREE_MAX_DEPTH) -> dict:
    """§4: the directory structure under `root` — FOLDERS only, each with the number
    of files and the total size of its subtree. File contents are never read.

    Both exclusion lists ride along so the tree can be drawn with the state each folder
    already has (F82), in one request instead of two."""
    budget = [max_nodes]
    tree = _scan_dir(root, "", root.name or str(root), 0, budget, max_depth)
    return {
        "root": root.as_posix(),
        "tree": tree,
        "nodes": max_nodes - budget[0] + 1,
        "limit": max_nodes,
        "max_depth": max_depth,
        "truncated": _any_truncated(tree),
        "skip_scan": skip_scan,
        "skip_layout": skip_layout,
    }


def _excludes_payload(cfg: Config, root: Path) -> dict:
    """What is currently left out under `root`, in both meanings — the collapsed
    one-line summary of the source block shows the two numbers separately (§3).

    The size is measured for "do not scan" only: that is disk work the run will not do.
    A "do not lay out" folder is read and indexed exactly as before, so its size saves
    nothing and printing it would suggest otherwise.
    """
    excludes = load_excludes(excludes_path(cfg))
    scan = sorted(excludes.for_root(root))
    layout = sorted(excludes.layout_for_root(root))
    files = size = 0
    for rel in scan:
        sub_files, sub_size = _sum_dir(root.joinpath(*rel.split("/")))
        files += sub_files
        size += sub_size
    return {"root": root.as_posix(), "skip_scan": scan, "count": len(scan),
            "files": files, "size": size,
            "skip_layout": layout, "layout_count": len(layout)}


def _validate_excludes_payload(
        payload: object) -> tuple[str, list[object], list[object]] | None:
    """Parse `{"root": str, "skip_scan": [str, ...], "skip_layout": [str, ...]}`.
    None -> invalid body.

    The entries themselves are not judged here — `indexer.normalize_exclude` is the
    single place that decides whether a path may narrow the walk, and the handler
    reports back which ones it refused.
    """
    if not isinstance(payload, dict):
        return None
    root = payload.get("root")
    if not isinstance(root, str) or not root.strip():
        return None
    sections: list[list[object]] = []
    for key in ("skip_scan", "skip_layout"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            return None
        sections.append(values)
    return root, sections[0], sections[1]


def _process_defaults_payload(cfg: Config) -> dict:
    """F57: defaults for the "Process" checkboxes — JS sets .checked by these values
    on page init (otherwise the checkboxes always start empty regardless of
    config.yaml). `vlm_available` — whether the `transformers` package is installed
    (`find_spec`, WITHOUT importing the module/loading the model).

    F123: `pets` rides here for the same reason and from the same place — the config
    (`features.pets`), which the settings column also edits. Two entry points, one
    source of truth, exactly as `deep` lives next to `naming.vlm_enabled`.

    F138: the knobs that moved onto this screen out of the settings column ride here too,
    from the same place — `features.pets_verify`. The column no longer offers them, so the
    file is now their ONLY home and this is what a run starts from. F186 retired three of
    that set (`vlm.quality`, `vlm.quality_scope`, `dedup.keeper_vlm`) with the two
    questions they switched on.

    F161: `products` joins them from `vlm.products`, and its default is the reason the
    key exists — a file that never heard of it answers True here, so the screen opens
    showing the run that file has always described.
    """
    return {
        "deep": bool(cfg.naming.vlm_enabled),
        "products": bool(cfg.vlm.products),
        "geo_online": cfg.geo.provider == "online",
        "pets": bool(cfg.features.pets),
        "pets_verify": bool(cfg.features.pets_verify),
        "vlm_available": importlib.util.find_spec("transformers") is not None,
    }


# --- F138: what this run costs, said before it starts -------------------------
#
# Moving four expensive knobs onto the run screen risks bringing back the console of
# switches F133 took away. What stops it is that the list means something: every line
# carries its price and the sum stands under them, so the screen is a budget a person
# assembles rather than a row of toggles.
#
# A price is only worth showing if it is COMPUTED. The same checkbox is four hours on
# one collection and four minutes on another, so nothing here is a constant in the
# markup: each number is a measured rate multiplied by a count taken out of THIS index.
# Where a count cannot be taken — a fresh collection, a stage that has never run — the
# answer is None and the screen draws a dash. A zero would read as "free", and the one
# thing an estimate may not do is promise twenty minutes with two hours coming.
#
# F159: the rates below are no longer THE price. They are the price until this machine
# has measured its own, which after F147 it does on every run — the run log holds
# `stage=<s>[ phase=<p>] elapsed=<sec> processed=<n>` for everything the pipeline does,
# and a second per frame taken from there beats a second per frame measured once on
# somebody else's collection and shipped in a wheel. The screen says which of the two it
# used, because a person deciding whether to wait four hours needs to tell "this is how
# it went for YOU last time" from "this is how it went for the developer".
#
# The defaults, each with the measurement it comes from:
_SEC_PER_VLM_FRAME = 0.78    # F113: one frame in one prompt
# The faces stage over the reference collection — the ~17 minutes the changelog and the
# F123 note both quote — spread over its 19 757 photographs.
_SEC_PER_FACES_FRAME = 17 * 60 / 19757
# index + geo + landmarks + phash, the four that always run: ~5 minutes over the same
# collection.
_SEC_PER_BASE_FRAME = 5 * 60 / 19757
# events: a grouping pass over rows the DB already holds — under a minute there, and it
# is scaled per frame for the same reason as the others rather than pinned at "fast".
_SEC_PER_EVENTS_FRAME = 15.0 / 19757

# Where a rate comes from, as it travels to the browser next to the seconds it produced.
# `fixed` is neither: the animal line costs 0 because the prompts ride inside a CLIP call
# that runs anyway, and a structural zero has no pedigree to state.
_RATE_MEASURED = "measured"
_RATE_DEFAULT = "default"
_RATE_FIXED = "fixed"

# Which units of the run log price which rate, and the default each falls back to. A rate
# counts as measured only when EVERY unit behind it is: `base` covers four stages, and
# three measured ones plus a guessed fourth is a guess wearing a measurement's clothes.
#
# The model questions are read from TWO units, because F165 split the stage that asks them
# in half: the deep tier decides what a frame IS and runs ahead of faces (`classify`),
# while the quality and animal questions read what faces wrote and stay behind it
# (`junk`). Both phases are called `junk_vlm`, so pricing the deep line off the wrong one
# would quietly charge it the rate of a different population.
#
# F186 removed a fourth reader of that phase — the keeper question, which was priced from
# `estimate:` because the log could not tell its seconds from the per-frame ones. It is not
# asked any more, so nothing quotes a price for it.
_RATE_UNITS: dict[str, tuple[str, ...]] = {
    "base": tuple(measurement_unit(stage)
                  for stage in ("index", "geo", "landmarks", "phash")),
    "faces": (measurement_unit("faces"),),
    "events": (measurement_unit("events"),),
    "vlm_verdict": (measurement_unit(VERDICTS_STAGE, CLASSIFY_PHASE_VLM),),
    "vlm_frame": (measurement_unit(CLASSIFY_STAGE, CLASSIFY_PHASE_VLM),),
}
_DEFAULT_RATES: dict[str, float] = {
    "base": _SEC_PER_BASE_FRAME,
    "faces": _SEC_PER_FACES_FRAME,
    "events": _SEC_PER_EVENTS_FRAME,
    "vlm_verdict": _SEC_PER_VLM_FRAME,
    "vlm_frame": _SEC_PER_VLM_FRAME,
}


@dataclasses.dataclass(frozen=True)
class _Rate:
    """Seconds per unit, and where that number came from (F159)."""

    seconds: float
    source: str
    at: datetime | None = None


def _resolve_rates(measurements: dict[str, Measurement]) -> dict[str, _Rate]:
    """The run log's rates where it has them, the shipped defaults where it does not."""
    rates: dict[str, _Rate] = {}
    for name, units in _RATE_UNITS.items():
        found = [measurements[unit] for unit in units if unit in measurements]
        if len(found) == len(units):
            rates[name] = _Rate(sum(m.seconds_per_unit for m in found),
                                _RATE_MEASURED, max(m.at for m in found))
        else:
            rates[name] = _Rate(_DEFAULT_RATES[name], _RATE_DEFAULT)
    return rates


# The photographs a run actually works on: `sorta` skips a duplicate and a file it could
# not read, so counting them in would price frames nobody looks at. Same predicate the
# faces measurement script samples by.
_LIVE_PHOTOS_SQL = ("SELECT COUNT(*) FROM files "
                    "WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'")


def _positive_or_none(value: int) -> int | None:
    """A count of zero from a stage that has never run is "unknown", not "nothing"."""
    return value or None


# The estimate is asked for on every open of the first tab, and one of its counts is the
# near-duplicate grouping, which costs seconds over tens of thousands of pHashes (F66).
# Keyed like the Duplicates payload — any write to the index changes the fingerprint —
# plus the config values the arithmetic reads, so moving a threshold in the settings
# column re-prices immediately instead of serving the number the old one produced.
# F159 adds the run log to that list for the same reason: a run that has just written its
# own timings is exactly the moment the old prices stop being the right answer.
_ESTIMATE_CACHE_MAX_ITEMS = 2
_estimate_cache: OrderedDict[tuple, dict] = OrderedDict()
_estimate_cache_lock = threading.Lock()


def _estimate_cache_clear() -> None:
    """Drop the cached estimates (test isolation)."""
    with _estimate_cache_lock:
        _estimate_cache.clear()


def _run_log_fingerprint() -> tuple:
    """(mtime, size) of every file the measurements are read out of (F159)."""
    stats: list[tuple[str, int, int]] = []
    for path in measurement_files():
        try:
            st = path.stat()
        except OSError:
            continue
        stats.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(stats)


def _process_estimate_payload(cfg: Config, db_path: Path) -> dict:
    """`GET /api/process/estimate` — the seconds behind every line of the run budget.

    `counts` travels next to `seconds` on purpose: a number a person is asked to plan
    an evening around should be checkable against the collection it was derived from,
    not taken on faith. Both dicts use the same keys, and `None` in either means "this
    index does not know" — the screen draws a dash and the sum says so too.

    `pets` is 0.0 rather than None when there is anything to count: the animal prompts
    ride inside the CLIP call the junk stage makes anyway (F123), so the line genuinely
    adds nothing to the run — one of the two places a zero here is the truth. The other
    is `deep` since F161: a master switch that only grants permission does no work, and
    saying so with a number is the point of taking its old effect out into `products`.

    F159 adds `sources` and `measured_at`, on the same keys again. A rate is either
    `measured` — read out of this machine's own run log — or `default`, a number measured
    once elsewhere and shipped with the tool, and the difference is the whole point:
    somebody deciding whether to start a four-hour run is entitled to know whose four
    hours the estimate is describing. `fixed` is the third value and belongs to the one
    line that is structurally free.
    """
    key = (str(db_path), _db_fingerprint(db_path), cfg.index.phash_max_distance,
           float(cfg.features.pet_candidate_threshold),
           bool(cfg.features.junk_rescue), float(cfg.features.junk_rescue_threshold),
           float(cfg.estimate.measurement_max_age_days), _run_log_fingerprint())
    with _estimate_cache_lock:
        cached = _estimate_cache.get(key)
        if cached is not None:
            _estimate_cache.move_to_end(key)
            return cached
    conn = _connect(db_path)
    try:
        photos = int(conn.execute(_LIVE_PHOTOS_SQL).fetchone()[0])
        # The deep tier's gate picks its candidates from the CLIP probabilities of the
        # run in progress, so the only honest source for "how many frames it asks
        # about" is how many it answered on last time (`source='vlm'`).
        products = _positive_or_none(int(conn.execute(
            "SELECT COUNT(*) FROM media_class WHERE source = 'vlm'").fetchone()[0]))
        # F161: unless the F140 selection is on, and then the tier is shown the frames
        # that cleared `features.junk_rescue_threshold` instead of the whole candidate
        # list — 955 of the live collection's 24 196 against ~7 300, twelve minutes
        # against an hour and a half. The screen has to show the price of the run that
        # WILL happen, so the population follows the config rather than averaging the
        # two. A collection nobody has scored yet says nothing: no `junk_score` at all
        # is a dash, the same answer the pet check gives before its own pass has run.
        if cfg.features.junk_rescue:
            scored = int(conn.execute(
                "SELECT COUNT(*) FROM frame_quality"
                " WHERE junk_score IS NOT NULL").fetchone()[0])
            products = None if not scored else int(conn.execute(
                "SELECT COUNT(*) FROM frame_quality WHERE junk_score >= ?",
                (float(cfg.features.junk_rescue_threshold),)).fetchone()[0])
        # The pet check is shown the frames CLIP scored above the candidate threshold —
        # a number that exists only once the CLIP pet group has run at all.
        pet_scored = int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score IS NOT NULL"
        ).fetchone()[0])
        pets_verify = None if not pet_scored else int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score >= ?",
            (float(cfg.features.pet_candidate_threshold),)).fetchone()[0])
    finally:
        conn.close()
    rates = _resolve_rates(read_measurements(
        max_age_days=float(cfg.estimate.measurement_max_age_days)))
    counts: dict[str, int | None] = {
        "base": _positive_or_none(photos),
        "faces": _positive_or_none(photos),
        "events": _positive_or_none(photos),
        "pets": _positive_or_none(photos),
        "pets_verify": pets_verify,
        # F161: the master switch is priced over the frames of the run it permits, and
        # the rate is a structural zero — permission costs nothing. The line that costs
        # what this one used to is `products`.
        "deep": _positive_or_none(photos),
        "products": products,
    }
    per_line: dict[str, _Rate] = {
        "base": rates["base"],
        "faces": rates["faces"],
        "events": rates["events"],
        "pets": _Rate(0.0, _RATE_FIXED),
        "pets_verify": rates["vlm_frame"],
        # F161: the master switch itself. Zero and `fixed`, like the animal line and for
        # a kinder reason — that one rides on a pass that runs anyway, this one has no
        # pass at all.
        "deep": _Rate(0.0, _RATE_FIXED),
        # F165 moved the deep tier ahead of faces, into a stage of its own — so this is
        # the one model line whose rate comes from `classify` rather than from `junk`.
        "products": rates["vlm_verdict"],
    }
    seconds: dict[str, float | None] = {}
    for name, rate in per_line.items():
        count = counts[name]
        seconds[name] = None if count is None else round(count * rate.seconds, 1)
    measured = [rate.at for rate in per_line.values() if rate.at is not None]
    payload = {
        "seconds": seconds,
        "counts": counts,
        "sources": {name: rate.source for name, rate in per_line.items()},
        "measured_at": max(measured).date().isoformat() if measured else None,
    }
    with _estimate_cache_lock:
        _estimate_cache[key] = payload
        _estimate_cache.move_to_end(key)
        while len(_estimate_cache) > _ESTIMATE_CACHE_MAX_ITEMS:
            _estimate_cache.popitem(last=False)
    return payload


def _env_payload() -> dict:
    """F64: the environment for the UI banner. `gpu_profile` — whether the GPU profile
    is installed (the nvidia-* packages exist only in the `gpu` extra; `find_spec`
    without importing torch). CPU profile -> False -> a reduced-speed banner on the
    "Process" tab. (Detects the chosen profile, not "whether CUDA works right now" —
    on a broken GPU profile the runtime fallback fires, which is a separate symptom.)"""
    return {"gpu_profile": importlib.util.find_spec("nvidia") is not None}


# F94: the two caches the web app may look at and empty. The CLI (`sorta cache`) knows
# the same pair; the names are what travels in the body of `POST /api/cache/clear`.
_CACHE_TARGETS = ("preview", "geo")


def _cache_payload(db_path: Path) -> dict:
    """F94: what the preview and geo caches occupy — the numbers `sorta cache` prints.

    The preview side is a metadata-only walk (`_sum_dir`) of a directory that holds one
    JPEG per frame — tens of thousands of them on a real collection, which is exactly
    why this is its own route and not a field of the status snapshot. The geo side is a
    `COUNT(*)`, the unit `sorta cache` reports for it: rows, not bytes.

    A cache that was never written is not an error — a missing directory sums to
    (0, 0) and an empty table counts 0.
    """
    directory = imaging.preview_dir()
    files, size = _sum_dir(directory)
    conn = _connect(db_path)
    try:
        entries = geo_cache_size(conn)
    finally:
        conn.close()
    return {
        # F117: `max_gb` is 0 when no ceiling is set, and the front end renders that as
        # a state rather than as a limit of zero. The share is computed here so the two
        # entry points cannot disagree on the arithmetic.
        "preview": {"dir": str(directory), "files": files, "bytes": size,
                    "max_gb": imaging.preview_cache_max_gb()},
        "geo": {"entries": entries},
    }


def _validate_cache_clear_payload(payload: object) -> str | None:
    """Parse `{"target": "preview"|"geo"}` (F94). None -> invalid: not a dict, or a
    target outside the pair — deleting is not something to guess an object for."""
    if not isinstance(payload, dict):
        return None
    target = payload.get("target")
    if not isinstance(target, str) or target not in _CACHE_TARGETS:
        return None
    return target


@dataclasses.dataclass(frozen=True)
class _RunOptions:
    """The knobs of ONE run, exactly as the run screen sends them.

    Each is an override applied to a COPY of the config for this run and never written
    back to config.yaml: the screen starts from the file (`/api/process/defaults`) and
    what a person changes on it decides this run alone. That is F123's rule for `deep`
    and `pets`, and F138 extends it to the three knobs it took out of the settings
    column — a value with two homes acquires two truths and a question about which of
    them is the real one.

    F138 fields are `None` when the body did not carry them, meaning "the config
    decides" — the convention `cli._quality_overrides` already follows for
    `--quality/--no-quality`. The run screen always sends all four, so an unticked box
    there forces OFF (the F57 rule) rather than quietly falling back to config.yaml;
    `/api/process/rerun-optional`, which has no interface for them, leaves them alone.

    F161 adds `products` with the same convention, and the "config decides" half of it
    carries the compatibility promise: `/api/process/rerun-optional` sends `deep` and no
    `products`, so re-running the junk stage with the model does what it did before this
    key existed.
    """
    deep: bool = False
    products: bool | None = None
    geo_online: bool = False
    faces: bool = False
    events: bool = False
    pets: bool = False
    pets_verify: bool | None = None


def _validate_process_payload(payload: object) -> tuple[str, _RunOptions] | None:
    """Parse `{"source_dir": str, "deep": bool=False, "geo_online": bool=False,
    "faces": bool=False, "events": bool=False, "pets": bool=False,
    "products": bool?, "pets_verify": bool?}`
    (F50/#34: opt-in VLM tier / online geo for THIS run, without editing config.yaml;
    F53/#39: opt-in steps faces/events, the same principle — default False; F123:
    `pets` is an opt-in of the THIRD shape — neither a tier nor a step, but a config
    override on the junk stage, `features.pets`; F138: the same third shape for
    `features.pets_verify`. F186 retired the other three of that set — `vlm.quality`,
    the scope select and `dedup.keeper_vlm` — with the questions behind them.)
    None -> invalid: not dict / `source_dir` not a string or empty after strip / a flag
    given but not bool."""
    if not isinstance(payload, dict):
        return None
    source_dir = payload.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return None
    flags: dict[str, object] = {}
    for key in ("deep", "geo_online", "faces", "events", "pets"):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return None
        flags[key] = value
    for key in ("products", "pets_verify"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            return None
        flags[key] = value
    return source_dir.strip(), _RunOptions(**flags)  # type: ignore[arg-type]


def _validate_rerun_optional_payload(
        payload: object) -> tuple[bool, bool, bool, bool] | None:
    """Parse `{"faces": bool=False, "events": bool=False, "deep": bool=False,
    "pets": bool=False}` for F62/F63 `POST /api/process/rerun-optional` (re-running the
    SELECTED on an already-built index: faces / events / junk-with-VLM when deep).
    F123: `pets` re-runs the junk stage too — the animals are counted inside it — so
    `deep` and `pets` together still mean ONE junk run, not two. None -> invalid: not
    dict / a field is given but not bool / all four False (nothing to re-run)."""
    if not isinstance(payload, dict):
        return None
    flags: list[bool] = []
    for key in ("faces", "events", "deep", "pets"):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            return None
        flags.append(value)
    if not any(flags):
        return None
    faces, events, deep, pets = flags
    return faces, events, deep, pets


def _run_cfg(cfg: Config, source_dir: str | None, opts: _RunOptions) -> Config:
    """A COPY of the config with this run's overrides on it — the original, shared with
    the request handlers, is not mutated and config.yaml is not written (F138 §2).

    `deep`/`geo_online`/`pets` are full overrides (see `_run_pipeline`); the F138 knobs
    are applied only when the body carried them, so the one caller without an interface
    for them (`/api/process/rerun-optional`) keeps running by the file.
    """
    naming = dataclasses.replace(cfg.naming, vlm_enabled=opts.deep)
    geo = dataclasses.replace(cfg.geo,
                              provider="online" if opts.geo_online else "offline")
    features = dataclasses.replace(cfg.features, pets=opts.pets)
    if opts.pets_verify is not None:
        features = dataclasses.replace(features, pets_verify=opts.pets_verify)
    vlm_changed: dict[str, Any] = {}
    if opts.products is not None:
        vlm_changed["products"] = opts.products
    vlm = dataclasses.replace(cfg.vlm, **vlm_changed) if vlm_changed else cfg.vlm
    sources = [Path(source_dir).resolve()] if source_dir is not None else cfg.sources
    return dataclasses.replace(cfg, sources=sources, naming=naming, geo=geo,
                               features=features, vlm=vlm)


def _run_pipeline(db_path: Path, cfg: Config, source_dir: str | None,
                  state: _ProcessState, cache: PlanCache,
                  options: _RunOptions | None = None,
                  only_optional: bool = False) -> None:
    """The body of the `POST /api/process` background thread: its own sqlite
    connection (not transferable between threads), source_dir overrides cfg.sources
    only for this run (F28-style, like `cli._cmd_index` with a positional src) — the
    original cfg shared with request handlers is not mutated. `source_dir=None` (F62:
    opt-in re-run over the existing index) leaves `cfg.sources` as-is — `Path(None)`
    is not called.

    `deep`/`geo_online` (F50/#34, a full override since F57/#57) — authoritatively set
    `naming.vlm_enabled`/`geo.provider` on this run_cfg regardless of what is in
    config.yaml: `deep=False` forces the VLM off even if `cfg.naming.vlm_enabled=True`
    (similarly `geo_online=False` forces `provider="offline"`). So the UI checkboxes
    (initialized from cfg via `/api/process/defaults`) can be unchecked to disable what
    is enabled in config.yaml — previously an unchecked box did not force OFF but
    quietly took cfg (the F57 bug). The server cfg/config.yaml is not re-read or
    mutated — the override lives only in this run's run_cfg.

    `faces`/`events` (F53/#39) — opt-in steps, default off: without the checkboxes the
    run builds only `index/geo/landmarks/junk/phash`, the heaviest steps are skipped.
    `stage_total`/the "stage i/N" numbering are computed from the actual filtered list.

    `pets` (F123) — the same kind of override as `deep`, on `features.pets`, and NOT a
    stage: animals are three extra prompts inside the CLIP call the `junk` stage makes
    anyway, so the flag changes what that stage computes and leaves the list of stages
    exactly as it was. Making it an `_OPTIONAL_STAGES` entry would put a stage that does
    not exist into the run.

    `pets_verify`/`quality`/`quality_scope`/`keeper` (F138) — four more of that same
    third shape, all of them settings of the `junk` stage (`features.pets_verify`,
    `vlm.quality`, `vlm.quality_scope`, `dedup.keeper_vlm`), so the list of stages is
    again untouched and only what one of them computes changes. They are what the run
    screen prices: between a quarter of an hour and four hours each.

    `products` (F161) — the fifth of that shape, on `vlm.products`, and the one that took
    an effect away from `deep`: with it off the classify half runs its cheap tiers and
    asks the model nothing, whatever `deep` says. `deep` remains what decides whether a
    model may be raised at all.

    `only_optional` (F62/F63: "Re-run selected" — POST
    `/api/process/rerun-optional`) — steps are narrowed to the SELECTED stages over the
    already-built index: `faces` (with faces), `events` (with events), `junk` (with
    deep — reclassification with the VLM, `naming.vlm_enabled=deep` — or with `pets`,
    which recomputes the animal verdicts). `deep` and `pets` together are still ONE junk
    run: they are two settings of one stage. The other base ones
    (index/geo/landmarks/phash) are not run at all.

    F135: a step that returns `{"processed", "skipped"}` has it recorded into the
    state, so the finished run can say what it did and what it recognised as already
    done instead of showing the same "Done." for both.

    Cancellation is checked BETWEEN stages (not mid-stage — MVP). After a successful
    finish (without an error/cancel) the plan cache (the Cities tab) is recomputed
    with the same conn so the tabs show the new data right away; Duplicates/People/
    Events read the DB directly on each request and need no refresh.
    """
    opts = options or _RunOptions()
    conn = _connect(db_path)
    error: str | None = None
    try:
        run_cfg = _run_cfg(cfg, source_dir, opts)
        enabled_optional = {"faces": opts.faces, "events": opts.events}
        if only_optional:
            # F63: re-run the selected — faces/events by flags + junk with deep
            # (reclassification with the VLM). The order from _pipeline_steps is kept.
            # F123: pets asks for the same junk stage — a set, so two reasons to run it
            # still add up to one entry.
            rerun = {name for name in _OPTIONAL_STAGES if enabled_optional[name]}
            if opts.deep or opts.pets:
                rerun.add("junk")
            steps = [(name, fn) for name, fn in _pipeline_steps() if name in rerun]
        else:
            steps = [(name, fn) for name, fn in _pipeline_steps()
                     if name not in _OPTIONAL_STAGES or enabled_optional[name]]
        state.set_stage_total(len(steps))
        completed = True
        for i, (name, fn) in enumerate(steps, 1):
            if state.cancel_requested():
                completed = False
                break
            state.set_stage(i, name)
            try:
                # F69: the UI pipeline runs for hours in a background thread with
                # nobody watching the console — the per-stage timing has to reach the
                # run log, or "which stage ate the time" stays a guess.
                with stage_timer(name):
                    stats = fn(run_cfg, conn, _StageProgress(state))
                if stats is not None:
                    state.set_stage_stats(name, stats)
            except _PipelineCancelled:
                completed = False  # mid-stage cancellation via the progress callback
                break
            except Exception as exc:  # noqa: BLE001 — report via status, do not crash the thread
                error = str(exc)
                _log.exception("sorta ui: этап пайплайна %r упал", name)
                completed = False
                break
        if completed and error is None:
            try:
                cache.rebuild(cfg, conn)
            except Exception as exc:  # noqa: BLE001
                error = f"план не обновлён: {exc}"
    finally:
        conn.close()
        state.finish(error)


# --- F43: apply the city layout from the UI (`POST /api/sort`) — reuses the
# sorter.plan_and_sort(apply=True) engine one-to-one with the CLI `sort --by city
# --apply`; ui.py here is only background/progress (the _ProcessState/_run_pipeline
# pattern from F36) and request-body validation. The moves/move_batches journal,
# blake3 verification, name-conflict resolution and in-place semantics (dest=None) —
# entirely in sorter.py, not duplicated.

class _SortState:
    """Thread-safe state of the background `/api/sort` apply (F43) — modelled on
    `_ProcessState`, but without stages (one `plan_and_sort` operation).

    F97: it also carries a cancel flag now. Unlike `_ProcessState`, the flag is only
    READ (`cancel_requested` is handed to `plan_and_sort` as `should_cancel`) — it
    never raises out of a callback. The layout has a batch to close before it may
    stop, so the engine decides when to break, not the state object.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.finished = False
        self.result: dict | None = None
        self._cancel_requested = False

    def try_start(self) -> bool:
        """True and switches to running if nothing is going now; otherwise False (409)."""
        with self._lock:
            if self.running:
                return False
            self._reset_locked()
            self.running = True
            return True

    def set_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done = done
            self.total = total

    def request_cancel(self) -> None:
        with self._lock:
            if self.running:
                self._cancel_requested = True

    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def finish(self, error: str | None, result: dict | None) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.error = error
            self.result = result

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "done": self.done,
                "total": self.total,
                "error": self.error,
                "finished": self.finished,
                "result": self.result,
                "cancel_requested": self._cancel_requested,
            }


class _UndoState(_SortState):
    """Thread-safe state of the background `/api/undo` rollback (F97).

    Deliberately the same shape as `_SortState` (running/done/total/error/finished/
    result + a cancel flag): the client polls it with the same code, and a rollback is
    the same kind of thing as a layout — one long operation over a file list that has
    to be stoppable. A separate class rather than a second `_SortState` instance so
    the cross-lock in the handlers reads as what it is: sort, process and undo are
    three named things that may not run at the same time.
    """


def _validate_sort_payload(payload: object) -> tuple[str | None, str] | None:
    """Parse the body `POST /api/sort`: `{"dest": str|null|"", "mode": "move"|"copy"}`.

    None -> invalid (400): not dict / `mode` not in {move, copy} / `dest` not a string
    and not null. `dest` an empty/whitespace string or null -> None (in-place — layout
    inside the source folder, see `plan_and_sort` F28).
    """
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    if mode not in ("move", "copy"):
        return None
    dest = payload.get("dest")
    if dest is not None and not isinstance(dest, str):
        return None
    dest = dest.strip() if isinstance(dest, str) else None
    return (dest or None), mode


def _validate_language_payload(payload: object) -> str | None:
    """Parse the body `POST /api/config/language`: `{"language": "ru"|"en"|"ja"}`.

    None -> invalid (400): not a dict / `language` not one of the supported codes."""
    if not isinstance(payload, dict):
        return None
    lang = payload.get("language")
    if not isinstance(lang, str):
        return None
    lang = lang.strip().lower()
    return lang if lang in _UI_LANGS else None


# --- F104: the settings column of the "Cities" tab (`/api/settings`) ----------
# A toggle in the interface has to change what the tool DOES, not just what a file
# says — so for every knob here the question "what does writing it invalidate?" is
# answered explicitly, and the answer is what makes a restart unnecessary:
#
#   vlm.model    — which weights to load. Read when the model is first needed, i.e.
#                  inside the next run. Nothing to invalidate.
#   vlm.workers  — the frame-preparation pool. Read when the VLM pass starts.
#   vlm.max_edge — the input size of a frame. Read per frame from that run's config.
#
# The folder language is the one setting with a consequence — the plan preview is
# BUILT in that language — and it keeps its own endpoint (`POST /api/config/language`,
# F65), which rebuilds the plan cache. It is not folded in here precisely because its
# answer to the question above is different.


@dataclasses.dataclass(frozen=True)
class _SettingSpec:
    """What a settings key accepts.

    `minimum`/`maximum` apply to `kind` of "int" and "float"; `choices` to "choice",
    which is a string restricted to a fixed set (a select in the form, not a text box —
    a scope the server would refuse is not worth offering).
    """
    kind: str  # bool | str | int | float | choice
    minimum: float = 0
    maximum: float = 0
    choices: tuple[str, ...] = ()


# The bounds are sanity rails, not tuning advice: 0 threads or a 4-pixel frame is a
# typo, and a 40 000-pixel one is a typo that costs the whole VRAM budget. The `min`/
# `max` attributes of the number inputs in the template carry the same numbers — a test
# pins the two together, because a form that offers a value the server refuses is worse
# than no form.
_SETTINGS_SPEC: dict[str, _SettingSpec] = {
    "vlm.model": _SettingSpec("str"),
    "vlm.workers": _SettingSpec("int", 1, 32),
    "vlm.max_edge": _SettingSpec("int", 128, 4096),
    # F138: `vlm.enabled`, `vlm.quality`, `vlm.quality_scope` and `features.pets` are
    # NOT here any more. They decide what THIS run costs — between a quarter of an hour
    # and four hours each — so they live on the run screen with their price next to
    # them, and a knob that moved there leaves this column: two entry points for one
    # value give two truths and a question about which one is in force. What stays is
    # what costs a run nothing — the thresholds, the model, the pool, the input size,
    # the cache ceiling. The config FILE remains their home; the screen starts from it
    # (`/api/process/defaults`) and overrides it for one run only.
    "features.pet_threshold": _SettingSpec("float", 0.0, 1.0),
    "features.sharpness_max_edge": _SettingSpec("int", 64, 4096),
    "features.sharpness_band_min": _SettingSpec("float", 0.0, 10000.0),
    "features.sharpness_band_max": _SettingSpec("float", 0.0, 10000.0),
    "features.subject_score_min": _SettingSpec("float", 0.0, 1.0),
    # F117: the preview-cache ceiling in GB. 0 is a legal value and the default — it
    # means "no ceiling", the behaviour since F67, so the minimum cannot be 1. The
    # upper rail is a typo guard: nobody caps a preview cache at four terabytes.
    "imaging.preview_cache_max_gb": _SettingSpec("int", 0, 4096),
}

# Which config object each section's keys live on. `imaging:` is the exception and maps
# to the environment instead (config._IMAGING_ENV — imaging.py is a leaf module that
# pool workers call with a path and nothing else), so applying it means setting the
# variable rather than replacing a dataclass field.
_SETTING_SECTIONS = ("vlm", "features")
_IMAGING_SETTING_ENV = {
    "imaging.preview_cache_max_gb": imaging.ENV_PREVIEW_MAX_GB,
}


def _settings_payload(cfg: Config) -> dict:
    """`GET /api/settings` — the current values, straight out of the RUNNING config."""
    values = {
        key: getattr(getattr(cfg, key.split(".", 1)[0]), key.split(".", 1)[1])
        for key in _SETTINGS_SPEC if key.split(".", 1)[0] in _SETTING_SECTIONS
    }
    return {
        **values,
        # Read through imaging, not off cfg: the environment is the source of truth for
        # this one, and a shell export legitimately overrides the file.
        "imaging.preview_cache_max_gb": int(imaging.preview_cache_max_gb()),
    }


def _validate_settings_payload(payload: object) -> dict[str, object] | None:
    """Parse the body of `POST /api/settings`: `{"<key>": <value>, …}`. None -> 400.

    The WHOLE body is rejected on the first bad key or value — a half-applied save
    would leave the file and the running config disagreeing about which half of the
    form the user is looking at. An empty body is invalid too: it would answer "ok"
    without having done anything.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    values: dict[str, object] = {}
    for key, raw in payload.items():
        spec = _SETTINGS_SPEC.get(key)
        if spec is None:
            return None
        if spec.kind == "bool":
            if not isinstance(raw, bool):
                return None
            values[key] = raw
        elif spec.kind == "str":
            if not isinstance(raw, str) or not raw.strip():
                return None
            values[key] = raw.strip()
        elif spec.kind == "choice":
            # F119: a fixed set, so a misspelling is a 400 rather than a silent
            # fallback. `vlm.quality_scope` is the one where that matters: `all` is the
            # 4.3-hour option, and drifting into it by accident is expensive.
            if not isinstance(raw, str) or raw not in spec.choices:
                return None
            values[key] = raw
        elif spec.kind == "float":
            # bool is an int is not a float here either; ints are accepted and widened,
            # because a form posting `1` for a threshold of 1.0 is not an error.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            if not spec.minimum <= float(raw) <= spec.maximum:
                return None
            values[key] = float(raw)
        else:
            # bool is an int in Python — `workers: true` is garbage, not 1.
            if isinstance(raw, bool) or not isinstance(raw, int):
                return None
            if not spec.minimum <= raw <= spec.maximum:
                return None
            values[key] = raw
    return values


def _apply_settings(cfg: Config, values: dict[str, object]) -> None:
    """Put validated values into the RUNNING config (see the note above: nothing else
    has to be invalidated — every one of them is read when the next run starts)."""
    fields = {key: key.split(".", 1)[1] for key in _SETTINGS_SPEC}
    # F117: the imaging keys live in the environment rather than on a dataclass, so they
    # are applied separately and taken OUT before the vlm replace below — passing one to
    # dataclasses.replace would raise on an unknown field.
    # `values` is NOT mutated here: the caller iterates the same dict afterwards to
    # persist each key into config.yaml, and removing one would save a setting that
    # applied but never reached the file.
    for key, env_name in _IMAGING_SETTING_ENV.items():
        if key not in values:
            continue
        os.environ[env_name] = str(values[key])
        section = cfg.raw.get("imaging")
        if not isinstance(section, dict):
            section = {}
            cfg.raw["imaging"] = section
        section[fields[key]] = values[key]
    # F119: one loop per section instead of a hard-coded `cfg.vlm`. `features:` (the
    # F113 quality cascade) is a second dataclass section and behaves identically —
    # replace the fields on the running config, then mirror them into cfg.raw so a later
    # save writes the same values the form is showing.
    touched_vlm = False
    for name in _SETTING_SECTIONS:
        picked = {k: v for k, v in values.items() if k.startswith(f"{name}.")}
        if not picked:
            continue
        touched_vlm = touched_vlm or name == "vlm"
        # The values were type-checked one by one against _SETTINGS_SPEC, which mypy
        # cannot follow through a dict[str, object] — the cast says so rather than
        # widening the spec into something the validator would have to trust.
        changed: Any = {fields[key]: value for key, value in picked.items()}
        setattr(cfg, name, dataclasses.replace(getattr(cfg, name), **changed))
        section = cfg.raw.get(name)
        if not isinstance(section, dict):  # absent, or present and left empty
            section = {}
            cfg.raw[name] = section
        for key, value in picked.items():
            section[fields[key]] = value
    if touched_vlm:
        # F102: `naming.vlm_enabled`/`classify_vlm_model` are the effective per-run
        # toggle the junk stage reads, and load_config holds them equal to the `vlm:`
        # section. A write that skipped this would be a setting that saved and did not
        # apply.
        cfg.naming = dataclasses.replace(cfg.naming, vlm_enabled=cfg.vlm.enabled,
                                         classify_vlm_model=cfg.vlm.model)


def _summary_dest(cfg: Config, dest: str | None) -> Path | None:
    """The destination root the pre-apply summary must look into (F104).

    An empty destination means the in-place layout, whose root `plan_and_sort` takes
    from the single configured source (F28) — resolved the same way here, and None
    when that rule does not apply, so the dialog can say "the numbers about the
    destination are unknown" instead of inventing them.
    """
    if dest:
        return Path(dest)
    if len(cfg.sources) == 1:
        return Path(cfg.sources[0])
    return None


def _run_sort(db_path: Path, cfg: Config, dest: str | None, mode: str,
             state: _SortState, cache: PlanCache) -> None:
    """The body of the `POST /api/sort` background thread: its own sqlite connection
    (not transferable between threads, like `_run_pipeline`). Calls the ready
    `sorter.plan_and_sort(..., apply=True)` — the moves/move_batches journal, blake3
    verification and name-conflict resolution are the engine, here only
    progress/status and rebuilding PlanCache after a successful apply.

    `plan_and_sort` may raise `ValueError` (e.g. in-place with ≠1 source in
    `cfg.sources`) — caught and stored in the state as an error, the thread does not
    crash and the server stays alive.

    F97: `should_cancel` is the state's own flag, so `POST /api/sort/cancel` stops the
    copying between files. A cancelled run is NOT an error — it returns a result like
    any other, with `cancelled` set and `moved` telling how far it got.
    """
    conn = _connect(db_path)
    error: str | None = None
    result: dict | None = None
    try:
        dest_path = Path(dest) if dest else None
        try:
            report = plan_and_sort(cfg, conn, "city", dest_path, apply=True,
                                   copy=(mode == "copy"), progress=state.set_progress,
                                   should_cancel=state.cancel_requested)
        except ValueError as exc:
            error = str(exc)
        else:
            result = {
                "moved": report.moved,
                "failed": report.failed,
                "skipped_in_place": report.skipped_in_place,
                "skipped_already_copied": report.skipped_already_copied,
                "cancelled": report.cancelled,
                "total": len(report.plan),
                "dirs": report.dirs,
                "dest": str(report.dest),
                "in_place": report.in_place,
                "mode": mode,
            }
            # F45: rebuild is only an update of the cities-tree preview cache, the
            # apply already happened (files laid out, the moves journal written) —
            # a rebuild failure is NOT a layout error, only a soft signal for the UI.
            try:
                cache.rebuild(cfg, conn)
            except Exception:  # noqa: BLE001
                _log.exception("sorta ui: план не обновлён после apply раскладки")
                result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)


# --- F97: roll the last batch back from the UI (`POST /api/undo`) -------------
# The engine is `sorter.undo`, exactly the one behind the CLI `sorta undo` — the
# blake3 verification before deleting a copy, the interrupted tail and the closing of
# a batch left with finished_at=NULL all live there. Here, as with `/api/sort`, only
# the background thread, the progress snapshot and the cancel flag.
#
# The batch is resolved the same way `_moves_payload` resolves it — the LAST batch in
# move_batches, i.e. the very one the "Moves" tab is showing. `sorter.undo(None)` picks
# the last batch that has a 'done' move instead, which is a different batch in exactly
# the case this button exists for: a run interrupted before its first file finished.
# The button and the manifest next to it must talk about the same thing.


def _last_batch_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
    return int(row["id"]) if row is not None else None


def _run_undo(db_path: Path, cfg: Config, state: _UndoState, cache: PlanCache) -> None:
    """The body of the `POST /api/undo` background thread: its own sqlite connection
    (not transferable between threads, like `_run_sort`).

    No batches at all -> an error in the state, not an exception: the button is only
    reachable when the manifest shows a batch, so this is a race, not a user mistake.
    A cancelled rollback is a normal result with `cancelled` set — what was undone
    stays undone and pressing the button again finishes the rest.

    `stray` (copies of an interrupted transfer whose hash does not match) travels to
    the client as a list of paths: those files are still lying in the result and only
    a human can decide what they are.
    """
    conn = _connect(db_path)
    error: str | None = None
    result: dict | None = None
    try:
        batch_id = _last_batch_id(conn)
        if batch_id is None:
            error = "no batches to undo"
        else:
            try:
                stats = undo(conn, batch_id, progress=state.set_progress,
                             should_cancel=state.cancel_requested)
            except ValueError as exc:
                error = str(exc)
            else:
                result = {
                    "batch_id": stats.batch_id,
                    "undone": stats.undone,
                    "missing": stats.missing,
                    "failed": stats.failed,
                    "cancelled": stats.cancelled,
                    "stray": stats.stray,
                }
                # As in _run_sort: the rollback already happened, so a preview cache
                # that would not rebuild is a soft signal, never a rollback error.
                try:
                    cache.rebuild(cfg, conn)
                except Exception:  # noqa: BLE001
                    _log.exception("sorta ui: план не обновлён после отката")
                    result["preview_stale"] = True
    finally:
        conn.close()
        state.finish(error, result)


# F133: the tabs are named after what a person DOES, not after the code that computed
# the numbers. Three things can be done to a collection and they differ by what they do
# to the file system: one canon (a physical move), any number of slices (hardlinks, free
# to make and to drop) and the junk (a subtraction, and the dangerous one). "Overview"
# holds the state of the collection and the run that produces it — one question asked at
# two moments in time.
_UI_STRINGS: dict[str, dict[str, str]] = {
    # F126: the tab is the workspace, not one of its slices — duplicates are the first
    # of four things a person opens it to go through.
    "tab_review": {"ru": "Разбор", "en": "Review", "ja": "仕分け"},
    "tab_layout": {"ru": "Раскладка", "en": "Layout", "ja": "振り分け"},
    "tab_slices": {"ru": "Срезы", "en": "Slices", "ja": "スライス"},
    "tab_person": {"ru": "Люди", "en": "People", "ja": "人物"},
    "tab_event": {"ru": "События", "en": "Events", "ja": "イベント"},
    "tab_animal": {"ru": "Животные", "en": "Animals", "ja": "動物"},
    "tab_moves": {"ru": "Перемещения", "en": "Moves", "ja": "移動"},
    # F175: the slice used to be called "Not personal photos", and that name was wrong
    # twice over. A photograph of a receipt, a screenshot of a conversation with your
    # wife and a passport are all personal — they are simply not photographs taken FOR
    # MEMORY, which is a different thing; and read as "not personal" the slice invites
    # deleting it, while a thousand of the frames in it are documents that must not be
    # deleted. The old name also sat one letter away from `files.not_personal`, the flag
    # for downloaded films (three files of 38 485), which is about where a file came
    # from and not about what is in the frame — see the note in i18n._FOLDERS.
    "tab_junk": {"ru": "Служебные кадры", "en": "Utility frames",
                 "ja": "実用目的のコマ"},
    "process_intro": {
        "ru": "Укажите папку с фото и нажмите «Обработать» — индекс наполнится "
              "(гео, лица, события, мусор, почти-дубликаты). Файлы не перемещаются.",
        "en": "Enter a photo folder and click Process — the index fills in "
              "(geo, faces, events, junk, near-duplicates). Files are not moved.",
        "ja": "写真フォルダを指定して「処理する」を押すと、インデックスが作成されます"
              "（位置情報、顔、イベント、不要写真、類似写真）。ファイルは移動されません。",
    },
    "process_path_placeholder": {
        "ru": "Путь к папке с фото", "en": "Path to photo folder",
        "ja": "写真フォルダのパス",
    },
    "process_start_button": {"ru": "Обработать", "en": "Process", "ja": "処理する"},
    "process_browse_button": {"ru": "Обзор…", "en": "Browse…", "ja": "参照…"},
    "process_deep_label": {
        "ru": "Глубокий анализ (VLM)", "en": "Deep analysis (VLM)",
        "ja": "詳細分析（VLM）",
    },
    # F161: the hint used to open with "Slower", and that stopped being true here. The
    # checkbox is permission and nothing else since the deep tier became a line of its
    # own below it: what is slow are the lines it unlocks, each of which now says its own
    # price. Leaving "slower" on the master would price the same hours twice and leave
    # the one thing this switch really decides — whether a model may be raised at all —
    # unsaid.
    "process_deep_hint": {
        "ru": "Разрешает поднимать модель. Сам по себе не считает ничего: время "
              "показано у строк под ним. Нужен `uv sync --extra vlm` (иначе "
              "автоматический откат на быстрый анализ).",
        "en": "Permission to load the model. It computes nothing by itself — the time "
              "is on the lines below it. Requires `uv sync --extra vlm` (otherwise "
              "falls back to the fast tier automatically).",
        "ja": "モデルの読み込みを許可します。これ自体は何も計算しません（所要時間は"
              "下の各項目に表示されます）。`uv sync --extra vlm` が必要です"
              "（なければ自動的に高速分析にフォールバックします）。",
    },
    # F161: the effect that used to be the master switch's own, given its name back. It
    # is deliberately named after what it PRODUCES and not after how: "deep analysis" is
    # a technology, and 85% of what that technology did on the live run of 2026-07-28 was
    # find products (2 202 verdicts of 2 592).
    "process_products_label": {
        "ru": "Распознавание товаров", "en": "Product recognition", "ja": "商品の認識",
    },
    # And the hint says what a person GETS, in the two places they will look for it. The
    # last sentence is the one that matters: the fast tier does not produce the class at
    # all, so without this line the products slice is not thin — it is empty.
    "process_products_hint": {
        "ru": "Отсюда берутся папка «_Товары» в раскладке и одноимённый срез: снимки "
              "вещей на продажу отделяются от снимков на память. Без этой строки "
              "товаров не мало — их ноль.",
        "en": "The “_Products” folder of the layout and the slice of the same name come "
              "from here: pictures of things for sale are told apart from pictures kept "
              "for memory. Without this line products are not few — there are none.",
        "ja": "振り分けの「_商品」フォルダーと同名のスライスはここから作られます: "
              "売るために撮った写真を、思い出の写真と切り分けます。この項目がなければ"
              "商品は少ないのではなく、ゼロです。",
    },
    "process_deep_vlm_missing": {
        "ru": "VLM не установлен — будет использован быстрый ярус (CLIP). "
              "Доустановите: `uv sync --extra vlm`.",
        "en": "VLM is not installed — the fast tier (CLIP) will be used instead. "
              "Install it: `uv sync --extra vlm`.",
        "ja": "VLM がインストールされていません。代わりに高速ティア（CLIP）が"
              "使用されます。インストール: `uv sync --extra vlm`。",
    },
    "process_geo_online_label": {
        "ru": "Онлайн-гео (точнее заграница)", "en": "Online geo (more accurate abroad)",
        "ja": "オンライン位置情報（海外でより正確）",
    },
    "process_geo_online_hint": {
        "ru": "Точнее определяет места за границей, но отправляет GPS-координаты "
              "фото на сервер геокодирования (сами фото никуда не отправляются).",
        "en": "More accurate place names abroad, but sends photo GPS coordinates "
              "to a geocoding server (the photos themselves are never sent).",
        "ja": "海外の地名をより正確に特定しますが、写真のGPS座標をジオコーディング"
              "サーバーに送信します（写真自体は送信されません）。",
    },
    "process_faces_label": {
        "ru": "Разбор по лицам", "en": "Detect faces",
        "ja": "顔の検出",
    },
    "process_faces_hint": {
        "ru": "Самый долгий шаг (детекция + кластеризация); включай, если "
              "нужна раскладка/альбомы по людям.",
        "en": "The slowest step (detection + clustering); enable it if you "
              "need sorting or albums by person.",
        "ja": "最も時間のかかるステップです（検出とクラスタリング）。人物ごとの"
              "整理やアルバムが必要な場合に有効にしてください。",
    },
    "process_events_label": {
        "ru": "Разбор по событиям", "en": "Detect events",
        "ja": "イベントの検出",
    },
    # F123: the hint has one job — to say that this checkbox is NOT another long step.
    # It stands next to faces (17 minutes) and deep analysis (hours), and read as one of
    # them it simply never gets ticked; the animals ride on the CLIP pass the junk stage
    # makes anyway.
    "process_pets_label": {
        "ru": "Искать животных", "en": "Detect animals",
        "ja": "動物の検出",
    },
    "process_pets_hint": {
        "ru": "Почти бесплатно: едет на уже идущем проходе CLIP (не отдельный "
              "долгий шаг), добавляет вкладку «Животные» и альбом.",
        "en": "Almost free: it rides on the CLIP pass that already runs (not another "
              "long step), and adds the “Animals” tab and album.",
        "ja": "ほぼ無料です。すでに実行中の CLIP パスに相乗りするため（別の長い"
              "ステップではありません）、「動物」タブとアルバムが追加されます。",
    },
    "process_events_hint": {
        "ru": "Группировка в поездки/события по времени и месту (нужен geo); "
              "для раскладки/альбомов по событиям.",
        "en": "Groups photos into trips/events by time and place (needs "
              "geo); for sorting or albums by event.",
        "ja": "時間と場所に基づいて旅行やイベントにグループ化します"
              "（位置情報が必要）。イベントごとの整理やアルバムに使います。",
    },
    # --- F138: the run budget — the block that turns six switches into an estimate ---
    # The list has to MEAN something, or moving four expensive knobs here is just the
    # console of toggles F133 removed. The meaning is the price on every line and the
    # sum under them, right where the eye is already going to the button.
    "costs_title": {
        "ru": "Что посчитать", "en": "What to compute", "ja": "何を計算するか",
    },
    # The estimate says it is an estimate, in the block itself. A wrong exact number is
    # worse than an honest approximate one: promise twenty minutes, take two hours, and
    # no figure on this screen is believed again.
    "costs_estimate_note": {
        "ru": "Время — оценка по этой коллекции: замеренная скорость на число кадров "
              "в индексе. Не обещание; прочерк значит «по этой базе не сосчитать».",
        "en": "The times are an estimate for this collection: a measured rate times the "
              "frames this index holds. Not a promise; a dash means this index cannot "
              "tell.",
        "ja": "所要時間はこのコレクションに対する目安です（実測の速度 × インデックス内"
              "のコマ数）。約束ではありません。ダッシュは「このインデックスでは算出でき"
              "ない」という意味です。",
    },
    # F159: where the numbers came from, said next to them. A person deciding whether to
    # wait four hours needs to tell "this is how it went for YOU last time" from "this is
    # how it went for the developer" — the second is an honest guess, and calling it one
    # is what keeps the first believable.
    "costs_source_measured": {
        "ru": "Числа — по вашему прошлому прогону ({date}).",
        "en": "The numbers come from your own last run ({date}).",
        "ja": "数値は前回のご自身の実行（{date}）に基づいています。",
    },
    "costs_source_default": {
        "ru": "Оценка по умолчанию: своих замеров на этой машине ещё нет.",
        "en": "A default estimate: this machine has no measurements of its own yet.",
        "ja": "既定の見積もりです。この端末での実測値はまだありません。",
    },
    "costs_source_mixed": {
        "ru": "Часть чисел — по вашему прошлому прогону ({date}), остальные — оценка "
              "по умолчанию.",
        "en": "Some numbers come from your own last run ({date}), the rest are default "
              "estimates.",
        "ja": "一部の数値は前回のご自身の実行（{date}）に基づき、残りは既定の見積もりです。",
    },
    "costs_base_label": {
        "ru": "Города, места и дубли", "en": "Cities, places and duplicates",
        "ja": "都市・場所・重複",
    },
    "costs_always": {"ru": "всегда", "en": "always", "ja": "常に"},
    "costs_total_label": {
        "ru": "Примерно за прогон:", "en": "This run, roughly:", "ja": "実行の目安:",
    },
    # A sum with an unknown line in it is not a sum. It is still worth showing as a
    # floor — "at least this much" is a decision a person can make.
    "costs_total_at_least": {
        "ru": "не меньше {time}", "en": "at least {time}", "ja": "{time} 以上",
    },
    "costs_unknown": {"ru": "—", "en": "—", "ja": "—"},
    "costs_free": {
        "ru": "почти бесплатно", "en": "almost free", "ja": "ほぼ無料",
    },
    # F145: what a line under a cleared master switch costs. Not "almost free" — that is
    # said about a stage that RUNS and is cheap; this one does not run at all, and the
    # number says so plainly because the sum below has to add up with it.
    "costs_off": {
        "ru": "0 — не выполняется", "en": "0 — does not run",
        "ja": "0 — 実行されません",
    },
    # F161: and what the MASTER switch costs, which is also nothing — for the opposite
    # reason. A line under a cleared master does not run; this one has nothing to run.
    # Both numbers are zero and saying so with one string would hide the difference the
    # feature is about: permission is not work.
    "costs_permission_only": {
        "ru": "0 — только разрешение", "en": "0 — permission only",
        "ja": "0 — 許可のみ",
    },
    "costs_under_minute": {
        "ru": "меньше минуты", "en": "under a minute", "ja": "1 分未満",
    },
    "costs_minutes": {"ru": "~{minutes} мин", "en": "~{minutes} min", "ja": "約 {minutes} 分"},
    "costs_hours": {
        "ru": "~{hours} ч {minutes} мин", "en": "~{hours} h {minutes} min",
        "ja": "約 {hours} 時間 {minutes} 分",
    },
    # F130 moved out of config.yaml onto this screen: it costs ~13 minutes, and a knob
    # that costs a quarter of an hour belongs where the run is started.
    "process_pets_verify_label": {
        "ru": "Проверять животных моделью", "en": "Verify the animals with the model",
        "ja": "動物をモデルで確認",
    },
    "process_pets_verify_hint": {
        "ru": "Каждого кандидата показать модели: живое животное, изображение или его "
              "нет. Точнее, но это отдельные вопросы к модели по каждому кадру.",
        "en": "Every candidate is shown to the model: a live animal, a picture of one, "
              "or none. More accurate, but it is one model question per frame.",
        "ja": "候補を 1 枚ずつモデルに見せます（実際の動物か、その画像か、いないか）。"
              "精度は上がりますが、コマごとにモデルへ問い合わせます。",
    },
    # F145: said next to every option that asks the SAME model the "Deep analysis"
    # checkbox loads. With the checkbox clear each of them costs nothing and does
    # nothing — the line says which switch turns it back on, so a dead option does not
    # read as a missing feature.
    "process_needs_deep_hint": {
        "ru": "Работает только с «Глубоким анализом (VLM)» — без него модель не "
              "поднимается и этот пункт ничего не делает.",
        "en": "Works only with Deep analysis (VLM) — without it no model is loaded and "
              "this option does nothing.",
        "ja": "「詳細解析（VLM）」がオンのときのみ動作します。オフの場合モデルは読み込ま"
              "れず、この項目は何もしません。",
    },
    # --- F81/F82: the three blocks of the first tab + the exclusion tree ------
    # F82: the two mechanisms are now side by side in one tree, so the wording carries
    # the whole difference between them — "do not SCAN" (the files never enter the index
    # at all) and "do not LAY OUT" (`sort.exclude_dirs`: indexed, searched, deduplicated,
    # simply left where they are). Each gets a one-line explanation, because this is
    # exactly the distinction a live user got wrong. The F77 per-file corrections
    # ("leave alone") are a third thing and live on the "Cities" tab.
    "step_source_title": {"ru": "Источник", "en": "Source", "ja": "ソース"},
    "step_options_title": {
        "ru": "Параметры запуска", "en": "Run options", "ja": "実行オプション",
    },
    "step_actions_title": {"ru": "Действия", "en": "Actions", "ja": "アクション"},
    "step_change_button": {"ru": "изменить", "en": "change", "ja": "変更"},
    # The same button folds the step back: opening one and finding nothing to change
    # is the common case, and it used to leave the block open with no way back.
    "step_collapse_button": {"ru": "свернуть", "en": "collapse", "ja": "折りたたむ"},
    "step_needs_source_hint": {
        "ru": "Сначала укажите папку с фото.",
        "en": "Choose a photo folder first.",
        "ja": "先に写真フォルダを指定してください。",
    },
    "step_options_summary_prefix": {
        "ru": "Параметры: ", "en": "Options: ", "ja": "オプション: ",
    },
    "step_options_summary_default": {
        "ru": "по умолчанию", "en": "defaults", "ja": "既定",
    },
    "excludes_button": {
        "ru": "Исключить папки…", "en": "Leave folders out…", "ja": "フォルダを除外…",
    },
    "excludes_title": {
        "ru": "Какие папки исключить", "en": "Folders to leave out",
        "ja": "除外するフォルダ",
    },
    "excludes_hint": {
        "ru": "Нажимайте на значок слева от папки, чтобы переключить её состояние. "
              "Состояние родителя действует на всё поддерево.",
        "en": "Click the mark to the left of a folder to switch its state. A folder's "
              "state applies to its whole subtree.",
        "ja": "フォルダ左のマークをクリックして状態を切り替えます。親の状態は"
              "サブツリー全体に適用されます。",
    },
    # The three states, each with the one line that says what it actually does.
    "tri_none_label": {
        "ru": "обрабатывать", "en": "process", "ja": "処理する",
    },
    "tri_none_hint": {
        "ru": "как обычно: сканируется и раскладывается",
        "en": "as usual: scanned and laid out",
        "ja": "通常どおり: スキャンして振り分けます",
    },
    "tri_layout_label": {
        "ru": "не раскладывать", "en": "don't sort", "ja": "振り分けない",
    },
    "tri_layout_hint": {
        "ru": "уже разобрано руками: файлы остаются в индексе и на месте, "
              "дубликаты по ним ищутся, но раскладка их не трогает",
        "en": "already sorted by hand: the files stay in the index and where they "
              "are, duplicates still find them, the layout leaves them alone",
        "ja": "手作業で整理済み: ファイルはインデックスに残り、その場に置かれます。"
              "重複検索の対象にはなりますが、振り分けは行いません",
    },
    "tri_scan_label": {
        "ru": "не сканировать", "en": "don't scan", "ja": "スキャンしない",
    },
    "tri_scan_hint": {
        "ru": "не нужно совсем: папка не читается, её файлов не будет в индексе, "
              "они не попадут ни в поиск дубликатов, ни в статистику",
        "en": "not needed at all: the folder is not read, its files never enter the "
              "index and take part in neither duplicate search nor statistics",
        "ja": "まったく不要: フォルダは読み込まれず、ファイルはインデックスに"
              "入らないため、重複検索にも統計にも含まれません",
    },
    "excludes_save_button": {"ru": "Сохранить", "en": "Save", "ja": "保存"},
    "excludes_saved": {
        "ru": "Сохранено. «Не сканировать» исчезнет из индекса при следующей "
              "обработке, «не раскладывать» подействует на следующей раскладке.",
        "en": "Saved. «Don't scan» leaves the index on the next run, «don't sort» "
              "applies on the next layout.",
        "ja": "保存しました。「スキャンしない」は次回の処理でインデックスから消え、"
              "「振り分けない」は次回の振り分けから適用されます。",
    },
    "excludes_error_prefix": {
        "ru": "Не удалось получить дерево папок: ",
        "en": "Could not load the folder tree: ",
        "ja": "フォルダツリーを取得できません: ",
    },
    "excludes_save_error_prefix": {
        "ru": "Не удалось сохранить: ", "en": "Could not save: ", "ja": "保存できません: ",
    },
    "excludes_empty": {
        "ru": "Вложенных папок нет.", "en": "No subfolders here.",
        "ja": "サブフォルダはありません。",
    },
    "excludes_truncated": {
        "ru": "Дерево очень большое — показаны первые {limit} папок.",
        "en": "The tree is very large — the first {limit} folders are shown.",
        "ja": "ツリーが大きいため、最初の {limit} 件のフォルダのみ表示しています。",
    },
    "excludes_summary_none": {
        "ru": "обрабатывается целиком", "en": "processed in full", "ja": "全体を処理",
    },
    # Two numbers, never merged into one (§3): they mean different things, and one
    # total would hide which mechanism a folder ended up in.
    "excludes_summary": {
        "ru": "не сканируется папок: {count} ({size})",
        "en": "not scanned: {count} folder(s), {size}",
        "ja": "スキャンしないフォルダ: {count} 件 ({size})",
    },
    "excludes_summary_layout": {
        "ru": "не раскладывается папок: {count}",
        "en": "not sorted: {count} folder(s)",
        "ja": "振り分けないフォルダ: {count} 件",
    },
    "excludes_folder_meta": {
        "ru": "{count} файлов · {size}", "en": "{count} files · {size}",
        "ja": "{count} 件 · {size}",
    },
    "size_units": {
        "ru": "Б КБ МБ ГБ ТБ", "en": "B KB MB GB TB", "ja": "B KB MB GB TB",
    },
    # F135: one button, so the run has to say what it skipped. "Nothing happened" and
    # "everything was already done" look identical without these two lines.
    "process_summary_title": {
        "ru": "Что сделал прогон:",
        "en": "What the run did:",
        "ja": "この実行の内容:",
    },
    "process_summary_stage": {
        "ru": "{stage} — обработано: {processed}, пропущено как уже обработанные: {skipped}",
        "en": "{stage} — processed: {processed}, skipped as already processed: {skipped}",
        "ja": "{stage} — 処理: {processed} 件、処理済みのためスキップ: {skipped} 件",
    },
    "env_cpu_warning": {
        "ru": "Установлен CPU-профиль: обработка идёт на процессоре — распознавание "
              "людей, VLM и большие коллекции заметно медленнее. Для скорости "
              "поставьте GPU-профиль: uv tool install --force \".[gpu]\".",
        "en": "CPU profile installed: processing runs on the CPU — face recognition, "
              "VLM and large collections are noticeably slower. For speed, install "
              "the GPU profile: uv tool install --force \".[gpu]\".",
        "ja": "CPU プロファイルがインストールされています: 処理は CPU で実行され、"
              "顔認識・VLM・大規模なコレクションは著しく遅くなります。高速化するには "
              "GPU プロファイルをインストールしてください: uv tool install --force \".[gpu]\"。",
    },
    "process_cancel_button": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "process_enter_path": {
        "ru": "Введите путь к папке.", "en": "Enter a folder path.",
        "ja": "フォルダのパスを入力してください。",
    },
    "process_stage_progress": {
        "ru": "Этап {stage} ({index}/{total}): {done} из {all}",
        "en": "Stage {stage} ({index}/{total}): {done} of {all}",
        "ja": "ステージ {stage}（{index}/{total}）: {done}/{all}",
    },
    "process_stage_progress_indeterminate": {  # #37: total not yet known (e.g. indexing)
        "ru": "Этап {stage} ({index}/{total}): обработано {done}",
        "en": "Stage {stage} ({index}/{total}): {done} processed",
        "ja": "ステージ {stage}（{index}/{total}）: {done} 件処理済み",
    },
    "process_done": {
        "ru": "Обработка завершена.", "en": "Processing complete.",
        "ja": "処理が完了しました。",
    },
    "process_cancelled": {
        "ru": "Обработка остановлена.", "en": "Processing stopped.",
        "ja": "処理が中止されました。",
    },
    "process_cancel_requested": {
        "ru": "Отмена запрошена — остановка после текущего шага…",
        "en": "Cancel requested — stopping after the current step…",
        "ja": "キャンセルを要求しました — 現在のステップ後に停止します…",
    },
    "process_error_prefix": {
        "ru": "Ошибка обработки: ", "en": "Processing error: ", "ja": "処理エラー: ",
    },
    "process_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
    # F84: the sub-phases of clustering inside the `faces` stage. The keys mirror
    # faces.CLUSTER_PHASE_* ("process_phase_" + the key from /api/process/status).
    "process_phase_cluster_read": {
        "ru": "кластеры: чтение эмбеддингов", "en": "clusters: reading embeddings",
        "ja": "クラスタ: 埋め込みを読み込み中",
    },
    "process_phase_cluster_hdbscan": {
        "ru": "кластеры: группировка лиц", "en": "clusters: grouping faces",
        "ja": "クラスタ: 顔をグループ化中",
    },
    "process_phase_cluster_inherit": {
        "ru": "кластеры: перенос имён", "en": "clusters: carrying names over",
        "ja": "クラスタ: 名前を引き継ぎ中",
    },
    "process_phase_cluster_write": {
        "ru": "кластеры: запись", "en": "clusters: saving",
        "ja": "クラスタ: 保存中",
    },
    # F100: the sub-phases of the `junk` stage. Keys: junk.CLASSIFY_PHASE_*. All four
    # are measurable, the deep one included (the VLM gate knows its candidates before
    # the loop starts), so the caption is shown next to the real N / M — the
    # process_phase_elapsed form below is for phases that have no percent at all.
    # The stage line right above already says "классификация", so these captions name
    # only the phase — the same reason the clustering ones say "кластеры" and not "лица".
    "process_phase_junk_clip": {
        "ru": "быстрый разбор (CLIP)", "en": "fast pass (CLIP)",
        "ja": "高速判定 (CLIP)",
    },
    "process_phase_junk_ocr": {
        "ru": "поиск текста (OCR)", "en": "text detection (OCR)",
        "ja": "テキスト検出 (OCR)",
    },
    "process_phase_junk_vlm": {
        "ru": "глубокий анализ (VLM)", "en": "deep analysis (VLM)",
        "ja": "詳細解析 (VLM)",
    },
    "process_phase_junk_write": {
        "ru": "запись вердиктов", "en": "saving verdicts",
        "ja": "判定を保存中",
    },
    # F141: the second CLIP pass — the search index. Named apart from the fast pass above
    # because it is what `features.search_index` costs and nothing else, and a caption
    # saying "fast pass" over ten minutes of a second encode would be the wrong sentence.
    "process_phase_junk_search": {
        "ru": "поисковый индекс (CLIP)", "en": "search index (CLIP)",
        "ja": "検索インデックス (CLIP)",
    },
    # F154: the object detector over the candidates of the animal query. A caption of its
    # own for the reason the search index has one: it is a second model over a short list,
    # neither the fast CLIP pass nor the VLM tier, and its minutes are what
    # `features.detector` costs. (The only line this feature adds to this file — a phase
    # without a string surfaces as a raw identifier, which tests/test_ui_junk_phase.py
    # requires it not to.)
    "process_phase_junk_detect": {
        "ru": "детектор объектов", "en": "object detector",
        "ja": "物体検出",
    },
    "process_phase_elapsed": {  # a phase with no percent — the clock is the sign of life
        "ru": "{phase} — идёт {seconds} с",
        "en": "{phase} — {seconds}s so far",
        "ja": "{phase} — 経過 {seconds} 秒",
    },
    "process_stage_index": {"ru": "индексация", "en": "indexing", "ja": "インデックス作成"},
    "process_stage_geo": {"ru": "гео", "en": "geo", "ja": "位置情報"},
    "process_stage_landmarks": {"ru": "места", "en": "landmarks", "ja": "ランドマーク"},
    # F165: the two halves of the classification, and the chips have to tell them apart —
    # the first one decides WHAT a frame is (and lets the faces stage skip what is not a
    # photograph), the second one measures the photographs it left.
    "process_stage_classify": {"ru": "вердикты", "en": "verdicts", "ja": "判定"},
    "process_stage_faces": {"ru": "лица", "en": "faces", "ja": "顔"},
    "process_stage_events": {"ru": "события", "en": "events", "ja": "イベント"},
    "process_stage_junk": {"ru": "классификация", "en": "classification", "ja": "分類"},
    "process_stage_phash": {"ru": "почти-дубликаты", "en": "near-duplicates", "ja": "類似写真"},
    "process_reset_button": {
        "ru": "Начать заново", "en": "Start over", "ja": "最初からやり直す",
    },
    "process_reset_confirm": {
        "ru": "Сотрёт индекс, включая имена людей/событий и решения по дублям. "
              "Фото и уже разложенные папки НЕ тронет. Продолжить?",
        "en": "This will erase the index, including people/event names and "
              "duplicate decisions. Photos and already-sorted folders are NOT "
              "touched. Continue?",
        "ja": "人物名・イベント名・重複の判定を含むインデックスを消去します。"
              "写真や既に整理済みのフォルダには触れません。続行しますか?",
    },
    # F93: the geo cache survives "Start over" — the name of a point on the map does
    # not depend on which files the user keeps, and re-asking the provider costs ~10
    # minutes of network. But an invisible unresettable thing must not exist, so the
    # way out lives exactly where the user already decided to erase something. Default
    # UNCHECKED: the cache is normally what makes the next run fast.
    "process_reset_clear_geo_label": {
        "ru": "Также очистить кэш геоданных",
        "en": "Also clear the geo cache",
        "ja": "位置情報のキャッシュも消去する",
    },
    "process_reset_clear_geo_hint": {
        "ru": "Ответы онлайн-геокодера переживают сброс, поэтому повторный прогон не "
              "стоит сети. Ставьте галочку, если провайдер ответил неверно и город "
              "нужно переспросить (при provider: online это снова минуты сети).",
        "en": "The online geocoder's answers survive a reset, so the next run costs no "
              "network. Tick this if the provider got a city wrong and has to be asked "
              "again (with provider: online that is minutes of network once more).",
        "ja": "オンライン地理コーダーの応答はリセット後も残るため、次回の実行に通信は不要です。"
              "プロバイダーが誤った都市を返した場合のみチェックしてください"
              "(provider: online では再び数分の通信が必要になります)。",
    },
    "process_reset_confirm_ok": {
        "ru": "Стереть индекс", "en": "Erase the index", "ja": "インデックスを消去",
    },
    "process_reset_confirm_cancel": {
        "ru": "Отмена", "en": "Cancel", "ja": "キャンセル",
    },
    "process_reset_done": {
        "ru": "Индекс сброшен.", "en": "Index reset.", "ja": "インデックスをリセットしました。",
    },
    "process_reset_done_geo": {
        "ru": "Индекс сброшен, кэш геоданных очищен.",
        "en": "Index reset, geo cache cleared.",
        "ja": "インデックスをリセットし、位置情報のキャッシュを消去しました。",
    },
    "process_reset_error_prefix": {
        "ru": "Не удалось сбросить: ", "en": "Failed to reset: ", "ja": "リセットできません: ",
    },
    # F94: the caches were reachable only from `sorta cache`, while the web app is
    # advertised as a full entry point — so on a live collection 12 GB of previews had
    # no way out for anyone who does not use the terminal. Sizes are shown and both
    # clears are offered. F117 added a ceiling, and it does not change that stance: it
    # is 0 by default, so nothing is ever deleted unless a person sets a number — the
    # ceiling answers "my disk filled up", it is not a policy applied on their behalf.
    "cache_title": {"ru": "Кэши", "en": "Caches", "ja": "キャッシュ"},
    # F117: shown next to the size, because a size without its bound says nothing.
    "cache_limit": {
        "ru": "Потолок: {limit} ГБ — занято {percent}%",
        "en": "Ceiling: {limit} GB — {percent}% used",
        "ja": "上限: {limit} GB — {percent}% 使用",
    },
    "cache_no_limit": {
        "ru": "Потолок не задан — кэш растёт, пока есть место на диске",
        "en": "No ceiling — the cache grows for as long as the disk allows",
        "ja": "上限なし — ディスクの空きがある限り増えます",
    },
    "cache_sizes": {
        "ru": "Кэш превью: {preview} ({files} файлов) · Кэш геоданных: {geo} записей",
        "en": "Preview cache: {preview} ({files} files) · Geo cache: {geo} entries",
        "ja": "プレビューキャッシュ: {preview} ({files} 件) · 位置情報キャッシュ: {geo} 件",
    },
    "cache_hint": {
        "ru": "Кэш превью — уменьшенные копии кадров, он ускоряет прогон и "
              "пересобирается сам. Кэш геоданных — ответы онлайн-геокодера, они "
              "избавляют повторный прогон от сети. Сами по себе они не уменьшаются; "
              "если задать потолок, кэш превью удаляет самые давно не читанные копии, "
              "пока не уложится в него.",
        "en": "The preview cache holds downscaled copies of the frames: it speeds the "
              "run up and rebuilds itself. The geo cache holds the online geocoder's "
              "answers, which spare a repeat run the network. Neither shrinks on its "
              "own; with a ceiling set, the preview cache drops its least recently read "
              "copies until it fits.",
        "ja": "プレビューキャッシュは縮小したコマの控えで、処理を速くし、自動的に作り直されます。"
              "位置情報キャッシュはオンライン地理コーダーの応答で、再実行時の通信を省きます。"
              "自動では減りませんが、上限を設定すると、収まるまで最も長く読まれていない"
              "プレビューから削除されます。",
    },
    "cache_clear_preview_button": {
        "ru": "Очистить кэш превью", "en": "Clear the preview cache",
        "ja": "プレビューキャッシュを消去",
    },
    "cache_clear_geo_button": {
        "ru": "Очистить кэш геоданных", "en": "Clear the geo cache",
        "ja": "位置情報キャッシュを消去",
    },
    "cache_clear_preview_confirm": {
        "ru": "Удалить кэш превью ({preview})? Место освободится сразу, а кэш "
              "соберётся заново сам — но первый прогон после этого будет медленнее: "
              "336 мс на кадр против 73 мс на готовом кэше. Фото и индекс не тронет.",
        "en": "Delete the preview cache ({preview})? The space is freed at once and the "
              "cache rebuilds itself — but the first run after that is slower: 336 ms "
              "per frame against 73 ms on a warm cache. Photos and the index are NOT "
              "touched.",
        "ja": "プレビューキャッシュ ({preview}) を削除しますか? 容量はすぐに解放され、"
              "キャッシュは自動的に作り直されますが、次の処理は遅くなります"
              "(1 コマあたり 336 ミリ秒、キャッシュありなら 73 ミリ秒)。"
              "写真とインデックスには触れません。",
    },
    "cache_clear_geo_confirm": {
        "ru": "Удалить ответы онлайн-геокодера ({geo} записей)? У уже обработанных "
              "фото города останутся, но при provider: online следующий прогон "
              "снова сходит в сеть — это минуты. Делайте это, если провайдер "
              "ответил неверно.",
        "en": "Delete the online geocoder's answers ({geo} entries)? The photos already "
              "processed keep their cities, but with provider: online the next run goes "
              "to the network again — that is minutes. Do this if the provider got an "
              "answer wrong.",
        "ja": "オンライン地理コーダーの応答 ({geo} 件) を削除しますか? "
              "処理済みの写真の都市は残りますが、provider: online では次回の実行で"
              "再び通信が発生します(数分)。応答が誤っていた場合に実行してください。",
    },
    "cache_clear_preview_done": {
        "ru": "Кэш превью очищен: удалено файлов {n}.",
        "en": "Preview cache cleared: {n} files removed.",
        "ja": "プレビューキャッシュを消去しました: {n} 件を削除。",
    },
    "cache_clear_geo_done": {
        "ru": "Кэш геоданных очищен: удалено записей {n}.",
        "en": "Geo cache cleared: {n} entries removed.",
        "ja": "位置情報キャッシュを消去しました: {n} 件を削除。",
    },
    "cache_clear_error_prefix": {
        "ru": "Не удалось очистить кэш: ", "en": "Failed to clear the cache: ",
        "ja": "キャッシュを消去できません: ",
    },
    "lightbox_close": {"ru": "Закрыть", "en": "Close", "ja": "閉じる"},
    "lightbox_open": {"ru": "Открыть превью", "en": "Open preview", "ja": "プレビューを開く"},
    # F80: the filmstrip of a clip — the tile marker and the frame pager.
    "video_badge": {"ru": "Видео", "en": "Video", "ja": "動画"},
    "video_open": {
        "ru": "Открыть кадры видео", "en": "Open video frames", "ja": "動画のフレームを開く",
    },
    "frame_prev": {"ru": "Предыдущий кадр", "en": "Previous frame", "ja": "前のフレーム"},
    "frame_next": {"ru": "Следующий кадр", "en": "Next frame", "ja": "次のフレーム"},
    "frame_of": {
        "ru": "Кадр {n} из {all}", "en": "Frame {n} of {all}", "ja": "フレーム {all} 中 {n}",
    },
    "delete_remember_label": {
        "ru": "Не спрашивать подтверждение удаления в этой сессии",
        "en": "Don't ask for delete confirmation this session",
        "ja": "このセッション中は削除の確認をしない",
    },
    "expand_all": {"ru": "Развернуть всё", "en": "Expand all", "ja": "すべて展開"},
    "collapse_all": {"ru": "Свернуть всё", "en": "Collapse all", "ja": "すべて折りたたむ"},
    "back_to_top": {"ru": "Наверх", "en": "Top", "ja": "上へ"},
    "loading": {"ru": "Загрузка...", "en": "Loading...", "ja": "読み込み中..."},
    "save_all_choices": {
        "ru": "Сохранить весь выбор", "en": "Save all choices", "ja": "すべての選択を保存",
    },
    "merge_selected": {"ru": "Слить выбранные", "en": "Merge selected", "ja": "選択を統合"},
    "theme_light": {"ru": "Светлая", "en": "Light", "ja": "ライト"},
    "theme_dark": {"ru": "Тёмная", "en": "Dark", "ja": "ダーク"},
    "error_loading_plan": {
        "ru": "Ошибка загрузки плана: ", "en": "Error loading plan: ",
        "ja": "プラン読み込みエラー: ",
    },
    # F70: the plan tab loads a folder page by page — the counter and the button
    # that asks for the next page.
    "plan_shown_of": {
        "ru": "показано {n} из {all}", "en": "showing {n} of {all}",
        "ja": "{all} 件中 {n} 件を表示",
    },
    "plan_load_more": {
        "ru": "Загрузить ещё", "en": "Load more", "ja": "さらに読み込む",
    },
    "plan_empty": {
        "ru": "План пуст — нечего раскладывать.",
        "en": "The plan is empty — nothing to lay out.",
        "ja": "プランは空です — 整理する対象がありません。",
    },
    # --- F173: the three captions of the shared pager (`makePager`) --------------------
    # One button, one counter, one warning, for every ordered slice there is and every one
    # there will be. They are not per-slice keys because the fifth copy of "Показать ещё"
    # is how a new slice ships without the button at all: a slice that has to add a string
    # to get one has a reason to skip it, and search shipped without one for that reason.
    "slice_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    # THE fix of the counter. "Показано 200" is indistinguishable from "нашлось ровно 200",
    # and for a ranking the second is almost never true — so the denominator is the length
    # of the list, always, and the numerator only says how far down it the reader is.
    "slice_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    # The price of depth, in one line and only where something is actually ranked. Measured
    # on 2026-08-02/03: doubling the list adds ~25 points of completeness on average, and
    # the query «дети» goes from 61% to 89% — while the frames that arrive with the second
    # page are exactly the ones the model was least sure about. Pressing the button buys
    # coverage with precision, and a person choosing that has to know it is a trade.
    "slice_depth_hint": {
        "ru": "Дальше по списку — больше находок и больше промахов: вторая половина "
              "заметно полнее, но модель в ней уверена меньше.",
        "en": "Further down the list means more found and more missed: the second half is "
              "noticeably more complete, and the model is less sure of it.",
        "ja": "リストを下るほど、見つかる数は増え、外れも増えます。後半は網羅性が高い"
              "一方で、モデルの確信度は低くなります。",
    },
    "error_loading_moves": {
        "ru": "Ошибка загрузки перемещений: ", "en": "Error loading moves: ",
        "ja": "移動読み込みエラー: ",
    },
    "error_loading_dupes": {
        "ru": "Ошибка загрузки дублей: ", "en": "Error loading duplicates: ",
        "ja": "重複読み込みエラー: ",
    },
    "error_loading_clusters": {
        "ru": "Ошибка загрузки кластеров: ", "en": "Error loading clusters: ",
        "ja": "クラスター読み込みエラー: ",
    },
    "confirm_delete_photo": {
        "ru": "Удалить этот файл в корзину?", "en": "Move this file to trash?",
        "ja": "このファイルをごみ箱に移動しますか?",
    },
    "delete": {"ru": "Удалить", "en": "Delete", "ja": "削除"},
    "delete_selected": {
        "ru": "Удалить выбранное", "en": "Delete selected", "ja": "選択を削除",
    },
    "select_for_delete": {
        "ru": "Выбрать для удаления", "en": "Select for deletion", "ja": "削除対象に選択",
    },
    "confirm_delete_selected": {
        "ru": "Удалить {n} файлов в корзину?", "en": "Move {n} files to trash?",
        "ja": "{n} 件のファイルをごみ箱に移動しますか?",
    },
    "status_planned": {"ru": "запланировано", "en": "planned", "ja": "予定"},
    "status_done": {"ru": "выполнено", "en": "done", "ja": "完了"},
    "status_undone": {"ru": "отменено", "en": "undone", "ja": "取消"},
    "status_failed": {"ru": "ошибка", "en": "failed", "ja": "失敗"},
    "status_deleted": {"ru": "удалено", "en": "deleted", "ja": "削除済み"},
    "batch_label": {"ru": "Батч", "en": "Batch", "ja": "バッチ"},
    "started_label": {"ru": "начат", "en": "started", "ja": "開始"},
    "finished_label": {"ru": "завершён", "en": "finished", "ja": "終了"},
    "in_progress_label": {"ru": "в процессе", "en": "in progress", "ja": "進行中"},
    "files_count_label": {"ru": "файлов", "en": "files", "ja": "ファイル数"},
    "no_moves_yet": {
        "ru": "Перемещений ещё не выполнялось.", "en": "No moves have been made yet.",
        "ja": "まだ移動は実行されていません。",
    },
    "unnamed": {"ru": "без имени", "en": "unnamed", "ja": "名前なし"},
    "faces_unit": {"ru": "лиц", "en": "faces", "ja": "顔"},
    "person_name_placeholder": {"ru": "Имя человека", "en": "Person's name", "ja": "人物名"},
    "name_button": {"ru": "Назвать", "en": "Name", "ja": "名前を設定"},
    "alert_enter_name": {
        "ru": "Введите имя.", "en": "Enter a name.", "ja": "名前を入力してください。",
    },
    "select_for_merge": {
        "ru": "выбрать для слияния", "en": "select for merge", "ja": "統合対象として選択",
    },
    "no_clusters": {
        "ru": "Кластеры лиц не найдены.", "en": "No face clusters found.",
        "ja": "顔クラスターが見つかりません。",
    },
    "recommended_badge": {
        "ru": "★ рекомендовано", "en": "★ recommended", "ja": "★ おすすめ",
    },
    # F148: what the group's STORED recommendation says under the frame it names. The
    # source is part of the caption and not a detail: how much an advice is worth
    # depends on who gives it, and these two are given by different judges.
    "keeper_badge_model": {
        "ru": "рекомендуем оставить · по модели",
        "en": "recommended to keep · by the model",
        "ja": "残すのがおすすめ · モデルの判断",
    },
    "keeper_badge_sharpness": {
        "ru": "рекомендуем оставить · по резкости",
        "en": "recommended to keep · by sharpness",
        "ja": "残すのがおすすめ · 鮮明さで判定",
    },
    # What the recommendation does NOT say, in the one place it can be read: there is
    # always exactly one per group, and a burst of six can hold two moments both worth
    # keeping. Keeping several frames is allowed and normal — advising several is what
    # the program cannot do.
    "keeper_badge_hint": {
        "ru": "Рекомендация одна на группу. В серии может быть несколько удачных "
              "кадров — оставить можно любой из них и не один.",
        "en": "One recommendation per group. A burst can hold more than one frame worth "
              "keeping — you may keep any of them, and more than one.",
        "ja": "推奨はグループにつき1件です。連写には残す価値のあるコマが複数ある"
              "こともあり、どれでも、また複数でも残せます。",
    },
    "action_keep": {"ru": "оставить", "en": "keep", "ja": "保持"},
    "action_to_delete": {"ru": "к удалению", "en": "to delete", "ja": "削除予定"},
    "skip_group_label": {
        "ru": "не удалять эту группу", "en": "don't delete this group",
        "ja": "このグループを削除しない",
    },
    "delete_dupes_button": {
        "ru": "Удалить дубли", "en": "Delete duplicates", "ja": "重複を削除",
    },
    "confirm_trash_group": {
        "ru": "Удалить в корзину все кадры группы {n}, кроме выбранного?",
        "en": "Move all frames in group {n} to trash, except the selected one?",
        "ja": "選択したもの以外、グループ{n}のすべてのフレームをごみ箱に移動しますか?",
    },
    "alert_choose_keeper": {
        "ru": "Выберите кадр, который нужно оставить.", "en": "Select the frame to keep.",
        "ja": "残すフレームを選択してください。",
    },
    "no_dupes": {
        "ru": "Почти-дубликаты не найдены.", "en": "No near-duplicates found.",
        "ja": "ほぼ重複が見つかりません。",
    },
    "select_group_to_save": {
        "ru": "Отметьте хотя бы одну группу для сохранения.",
        "en": "Mark at least one group to save.",
        "ja": "保存するグループを少なくとも1つ選択してください。",
    },
    "saved_groups": {
        "ru": "Сохранено групп: {n}", "en": "Groups saved: {n}", "ja": "保存したグループ数: {n}",
    },
    "group_title": {
        "ru": "Группа {n} ({count} кадра)", "en": "Group {n} ({count} frames)",
        "ja": "グループ{n}（{count}枚）",
    },
    "album_button": {
        "ru": "Собрать в папку", "en": "Gather into folder", "ja": "フォルダにまとめる",
    },
    "album_mode_link": {"ru": "Ссылка (link)", "en": "Link", "ja": "リンク"},
    "album_mode_copy": {"ru": "Копия", "en": "Copy", "ja": "コピー"},
    "album_mode_move": {"ru": "Перемещение", "en": "Move", "ja": "移動"},
    "album_where_placeholder": {
        "ru": "Фильтр, напр. city=Барселона", "en": "Filter, e.g. city=Barcelona",
        "ja": "フィルター（例: city=Barcelona）",
    },
    "album_name_placeholder": {
        "ru": "Имя папки альбома", "en": "Album folder name", "ja": "アルバムフォルダ名",
    },
    "album_dest_placeholder": {
        "ru": "Путь назначения альбома", "en": "Album destination path",
        "ja": "アルバムの保存先パス",
    },
    "album_name_first_hint": {
        "ru": "Сначала назовите кластер", "en": "Name the cluster first",
        "ja": "先にクラスターに名前を付けてください",
    },
    "album_preview_text": {
        "ru": "{n} файлов → {dest}", "en": "{n} files → {dest}", "ja": "{n} ファイル → {dest}",
    },
    "album_blocked_text": {
        "ru": "; move заблокирует {k} мульти-кадров",
        "en": "; move will block {k} multi-person frames",
        "ja": "；moveは{k}件のマルチ人物フレームをブロックします",
    },
    "album_confirm_move": {
        "ru": "Внимание: перемещение изымет файлы из общего пула сортировки. Продолжить?",
        "en": "Warning: moving will remove files from the common sorting pool. Continue?",
        "ja": "警告: 移動するとファイルは共通の振り分けプールから除外されます。続行しますか?",
    },
    "album_confirm_generic": {
        "ru": "Собрать альбом?", "en": "Gather the album?", "ja": "アルバムをまとめますか?",
    },
    "album_result_text": {
        "ru": "Собрано {n}, ошибок {f}", "en": "Gathered {n}, errors {f}",
        "ja": "収集済み{n}、エラー{f}",
    },
    "album_in_progress": {
        "ru": "Идёт сбор альбома...", "en": "Gathering album...", "ja": "アルバムを収集中...",
    },
    "no_events": {
        "ru": "События не найдены.", "en": "No events found.", "ja": "イベントが見つかりません。",
    },
    "error_loading_events": {
        "ru": "Ошибка загрузки событий: ", "en": "Error loading events: ",
        "ja": "イベント読み込みエラー: ",
    },
    # --- F43: apply the city layout (the "Cities" tab) -----------------
    "sort_dest_placeholder": {
        "ru": "Папка назначения (пусто = в исходной папке)",
        "en": "Destination folder (empty = in the source folder)",
        "ja": "移動先フォルダ（空欄 = 元のフォルダ内）",
    },
    "sort_dest_hint": {
        "ru": "Пусто — коллекция раскладывается внутри исходной папки (in-place).",
        "en": "Empty — the collection is sorted inside the source folder (in-place).",
        "ja": "空欄の場合、コレクションは元のフォルダ内で振り分けられます（in-place）。",
    },
    "sort_dest_inplace_label": {
        "ru": "исходная папка (in-place)", "en": "source folder (in-place)",
        "ja": "元のフォルダ（in-place）",
    },
    "sort_mode_move": {"ru": "Переместить", "en": "Move", "ja": "移動"},
    "sort_mode_copy": {"ru": "Копировать", "en": "Copy", "ja": "コピー"},
    "sort_apply_button": {"ru": "Разложить", "en": "Apply", "ja": "振り分ける"},
    "folder_lang_label": {
        "ru": "Язык папок", "en": "Folder language", "ja": "フォルダの言語",
    },
    "folder_lang_saved": {
        "ru": "Язык папок сохранён — план пересчитан.",
        "en": "Folder language saved — the plan was recomputed.",
        "ja": "フォルダの言語を保存しました — プランを再計算しました。",
    },
    # F104: `sort_confirm_summary` (F43) lived here — the single line of a window.confirm
    # that said only "N files, M folders". It is gone with that dialog: the summary is
    # built from `sort_summary_*` below, off /api/sort/summary, and names the volume, the
    # review folders and what is already in the destination as well.
    # F97: the text used to send the user to the terminal (`sorta undo`) — there is a
    # button on the "Moves" tab now, so it points at the button.
    "sort_confirm_move": {
        "ru": "ВНИМАНИЕ: оригиналы будут ПЕРЕМЕЩЕНЫ. "
              "Откатить можно кнопкой на вкладке «Перемещения».",
        "en": "WARNING: originals will be MOVED. "
              "You can roll this back with the button on the Moves tab.",
        "ja": "警告: オリジナルファイルが移動されます。"
              "「移動」タブのボタンで元に戻せます。",
    },
    "sort_confirm_inplace": {
        "ru": "ВНИМАНИЕ: реструктурируется ИСХОДНОЕ дерево коллекции, а не копия "
              "в отдельной папке.",
        "en": "WARNING: this restructures the SOURCE tree of the collection, "
              "not a copy in a separate folder.",
        "ja": "警告: これは別フォルダのコピーではなく、コレクションの元のツリー"
              "構造そのものを再編成します。",
    },
    "sort_confirm_copy": {
        "ru": "Оригиналы останутся на месте — будут созданы копии.",
        "en": "Originals stay in place — copies will be created.",
        "ja": "オリジナルはそのまま残り、コピーが作成されます。",
    },
    "sort_progress_line": {
        "ru": "Готово {done} из {all}", "en": "Done {done} of {all}",
        "ja": "完了 {done}/{all}",
    },
    "sort_done_text": {
        "ru": "Разложено {n}, ошибок {f} (+ пропущено {p} на месте)",
        "en": "Sorted {n}, errors {f} (+ {p} skipped in place)",
        "ja": "振り分け済み {n}、エラー {f}（+ その場でスキップ {p}）",
    },
    "sort_error_prefix": {
        "ru": "Ошибка раскладки: ", "en": "Sort error: ", "ja": "振り分けエラー: ",
    },
    "sort_preview_stale_warning": {
        "ru": "Превью плана не обновилось — обновите вкладку.",
        "en": "Plan preview did not refresh — reload the tab.",
        "ja": "プレビューが更新されませんでした — タブを再読み込みしてください。",
    },
    "sort_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
    # --- F97: cancelling a layout + rolling back from the "Moves" tab ---------
    "sort_cancel_button": {"ru": "Отменить", "en": "Cancel", "ja": "中止"},
    "sort_cancel_requested": {
        "ru": "Отмена запрошена — текущий файл будет дописан…",
        "en": "Cancellation requested — the current file will be finished…",
        "ja": "中止をリクエストしました — 現在のファイルは書き終えます…",
    },
    "sort_cancelled_text": {
        "ru": "Отменено: разложено {n} из {all}, ошибок {f}.",
        "en": "Cancelled: sorted {n} of {all}, errors {f}.",
        "ja": "中止しました: {all} 件中 {n} 件を振り分け、エラー {f} 件。",
    },
    "sort_already_copied_note": {
        "ru": " Уже было на месте: {c}.", "en": " Already there: {c}.",
        "ja": " すでに配置済み: {c} 件。",
    },
    "sort_undo_hint": {
        "ru": "Разложенное можно откатить — вкладка «Перемещения».",
        "en": "What was sorted can be rolled back on the Moves tab.",
        "ja": "振り分けた結果は「移動」タブで元に戻せます。",
    },
    "undo_button": {"ru": "Откатить", "en": "Roll back", "ja": "元に戻す"},
    "undo_cancel_button": {"ru": "Отменить откат", "en": "Cancel rollback", "ja": "中止"},
    "undo_confirm_copy": {
        "ru": "Будет удалено {n} копий в {dest}. Оригиналы не тронутся.",
        "en": "{n} copies in {dest} will be deleted. The originals stay untouched.",
        "ja": "{dest} 内のコピー {n} 件を削除します。オリジナルはそのまま残ります。",
    },
    "undo_confirm_move": {
        "ru": "{n} файлов вернутся в исходные папки.",
        "en": "{n} files will go back to their original folders.",
        "ja": "{n} 件のファイルが元のフォルダに戻ります。",
    },
    "undo_confirm_ok": {"ru": "Откатить", "en": "Roll back", "ja": "元に戻す"},
    "undo_confirm_cancel": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "undo_progress_line": {
        "ru": "Откачено {done} из {all}", "en": "Rolled back {done} of {all}",
        "ja": "元に戻した件数 {done}/{all}",
    },
    "undo_done_text": {
        "ru": "Откачено {n}, отсутствовало {m}, ошибок {f}",
        "en": "Rolled back {n}, missing {m}, errors {f}",
        "ja": "元に戻した {n} 件、見つからない {m} 件、エラー {f} 件",
    },
    "undo_cancelled_text": {
        "ru": "Отменено: откачено {n}. Нажмите «Откатить» ещё раз, чтобы доделать.",
        "en": "Cancelled: {n} rolled back. Press Roll back again to finish.",
        "ja": "中止しました: {n} 件を元に戻しました。続けるにはもう一度「元に戻す」を押してください。",
    },
    "undo_stray_title": {
        "ru": "Битые копии прерванного переноса — не удалены, проверьте вручную:",
        "en": "Broken copies from an interrupted transfer — not deleted, check by hand:",
        "ja": "中断された転送による壊れたコピー — 削除していません。手動で確認してください:",
    },
    "undo_nothing_to_undo": {
        "ru": "Откатывать нечего.", "en": "Nothing to roll back.",
        "ja": "元に戻すものがありません。",
    },
    "undo_error_prefix": {
        "ru": "Ошибка отката: ", "en": "Rollback error: ", "ja": "元に戻す処理のエラー: ",
    },
    "undo_start_error_prefix": {
        "ru": "Не удалось запустить откат: ", "en": "Failed to start the rollback: ",
        "ja": "元に戻す処理を開始できません: ",
    },
    "undo_cancel_requested": {
        "ru": "Отмена отката запрошена…", "en": "Rollback cancellation requested…",
        "ja": "元に戻す処理の中止をリクエストしました…",
    },
    # --- F104: the settings column + the summary before a layout ------------
    "settings_title": {"ru": "Настройки", "en": "Settings", "ja": "設定"},
    "settings_hint": {
        "ru": "Меняются прямо здесь и сохраняются в config.yaml. Перезапускать "
              "«sorta ui» не нужно — новые значения берёт следующий прогон.",
        "en": "Changed right here and saved into config.yaml. No need to restart "
              "`sorta ui` — the next run picks the new values up.",
        "ja": "ここで変更すると config.yaml に保存されます。`sorta ui` の再起動は不要 — "
              "次の処理から新しい値が使われます。",
    },
    # F138: the column says out loud that the expensive knobs are not missing but
    # elsewhere — a person who used to switch the deep tier on from here has to be told
    # where it went, not left looking for it.
    "settings_costs_moved_hint": {
        "ru": "Здесь то, что не стоит времени прогона: пороги, модель, потоки. Что "
              "стоит часов — глубокий разбор, качество кадров, животные, лучший кадр "
              "в группе — живёт на экране запуска, рядом со своей ценой.",
        "en": "What is here costs a run nothing: thresholds, the model, the pools. What "
              "costs hours — the deep tier, frame quality, animals, the best frame of a "
              "group — lives on the run screen, next to its price.",
        "ja": "ここにあるのは実行時間を増やさない項目です（しきい値・モデル・スレッド）。"
              "時間のかかる項目 — 詳細解析、コマの品質、動物、グループ内のベストショット "
              "— は実行画面にあり、そこで所要時間が示されます。",
    },
    "settings_vlm_model_label": {"ru": "Модель", "en": "Model", "ja": "モデル"},
    "settings_vlm_workers_label": {
        "ru": "Потоки подготовки", "en": "Preparation threads", "ja": "前処理スレッド数",
    },
    "settings_vlm_workers_hint": {
        "ru": "Сколько кадров готовится к отправке в модель параллельно. Каждый поток "
              "держит кадр в памяти — больше не значит быстрее.",
        "en": "How many frames are prepared for the model in parallel. Every thread "
              "holds a frame in RAM — more is not automatically faster.",
        "ja": "モデルに渡すフレームを同時に何枚準備するか。各スレッドがフレームを"
              "メモリに保持するため、増やせば速くなるとは限りません。",
    },
    "settings_vlm_max_edge_label": {
        "ru": "Разрешение кадра, px", "en": "Frame resolution, px", "ja": "フレーム解像度 (px)",
    },
    "settings_vlm_max_edge_hint": {
        "ru": "Длинная сторона кадра, который видит модель. Меньше — быстрее и "
              "экономнее по видеопамяти, но мелкий текст на снимке различим хуже.",
        "en": "The long edge of the frame the model sees. Smaller is faster and easier "
              "on VRAM, but fine text in a shot becomes harder to make out.",
        "ja": "モデルが見るフレームの長辺。小さいほど高速で VRAM も節約できますが、"
              "写真内の細かい文字は読み取りにくくなります。",
    },
    # F119: the F113 quality cascade. Each signal is taken by the cheapest instrument
    # that answers it, and the hints say which — a person deciding whether to switch
    # something on needs to know what it will cost, not only what it does.
    "settings_quality_title": {
        "ru": "Качество кадра", "en": "Frame quality", "ja": "コマの品質",
    },
    "settings_quality_hint": {
        "ru": "Необязательные признаки: помогают выбрать лучший кадр из серии и найти "
              "случайные снимки. Всё выключено по умолчанию и считается только на "
              "прогоне. Сгруппировано по тому, ЧЕМ признак считается, — это и есть "
              "разница в цене.",
        "en": "Optional signals: they help pick the best frame of a burst and spot the "
              "shots nobody meant to take. All off by default and computed during a "
              "run. Grouped by WHAT answers each one, because that is where the cost "
              "difference is.",
        "ja": "任意のシグナル: 連写から最良のコマを選び、意図しない撮影を見つけるのに"
              "役立ちます。既定はすべて無効で、実行中にのみ計算されます。**何が**答える"
              "かで分けてあります — 費用の差はそこにあるからです。",
    },
    "settings_quality_cheap_title": {
        "ru": "Без VLM", "en": "No VLM needed", "ja": "VLM 不要",
    },
    "settings_quality_cheap_hint": {
        "ru": "Считается на проходе, который и так идёт: CLIP и обычная арифметика по "
              "превью. Включать можно, даже если глубокий анализ выключен.",
        "en": "Computed on a pass that runs anyway: CLIP and plain arithmetic over the "
              "preview. Safe to switch on even with deep analysis off.",
        "ja": "どのみち走るパスで計算されます: CLIP と、プレビューに対する単純な演算。"
              "詳細解析が無効でも有効にできます。",
    },
    "settings_quality_gate_title": {
        "ru": "Кого спрашивать у модели",
        "en": "Who reaches the model",
        "ja": "モデルに届くコマ",
    },
    "settings_quality_gate_hint": {
        "ru": "Пороги, которые решают, у каких кадров вообще спрашивать. Считаются "
              "дёшево, а экономят дорогое: чем уже полоса, тем меньше кадров уйдёт в "
              "модель.",
        "en": "The thresholds that decide which frames are worth asking about at all. "
              "Cheap to compute and what saves the expensive part: the narrower the "
              "band, the fewer frames reach the model.",
        "ja": "そもそもどのコマについて尋ねるかを決めるしきい値です。計算は安価で、"
              "高価な部分を節約します — 帯が狭いほど、モデルに届くコマは減ります。",
    },
    "settings_features_pet_threshold_label": {
        "ru": "Порог уверенности для животных",
        "en": "Animal confidence threshold",
        "ja": "動物の信頼度しきい値",
    },
    "settings_features_subject_score_min_label": {
        "ru": "Порог «это вообще фотография»",
        "en": "“This is a photograph at all” threshold",
        "ja": "「そもそも写真か」のしきい値",
    },
    "settings_features_subject_score_min_hint": {
        "ru": "Второй вход в модель. Это вероятность от CLIP: если он оценивает кадр "
              "как фотографию ниже этого порога — значит, сам не понял, на что смотрит, "
              "и такой кадр стоит показать модели.",
        "en": "The second way into the model. This is CLIP's own probability: scoring a "
              "frame as “a photograph” below this threshold is CLIP saying it does not "
              "know what it is looking at, and such a frame is worth showing to the "
              "model.",
        "ja": "モデルへの 2 つ目の入口です。これは CLIP 自身の確率で、「写真である」の"
              "スコアがこのしきい値を下回るのは、CLIP が何を見ているか分からないという"
              "ことであり、そのコマはモデルに見せる価値があります。",
    },
    "settings_features_sharpness_max_edge_hint": {
        "ru": "Сама резкость считается всегда и бесплатно — это дисперсия лапласиана по "
              "превью, которое другие стадии уже построили. Модель здесь ни при чём.",
        "en": "Sharpness itself is always computed and costs nothing — the variance of "
              "a Laplacian over the preview other stages have already built. No model "
              "is involved.",
        "ja": "鮮鋭度そのものは常に計算され、費用はかかりません — 他の段階がすでに作った"
              "プレビューに対するラプラシアンの分散です。モデルは関係ありません。",
    },
    "settings_features_sharpness_band_min_label": {
        "ru": "Резкость: нижняя граница",
        "en": "Sharpness: lower bound",
        "ja": "鮮鋭度: 下限",
    },
    "settings_features_sharpness_band_max_label": {
        "ru": "Резкость: верхняя граница",
        "en": "Sharpness: upper bound",
        "ja": "鮮鋭度: 上限",
    },
    "settings_features_sharpness_band_hint": {
        "ru": "Ниже нижней кадр однозначно смазан, выше верхней — однозначно резкий; "
              "спрашивать модель незачем ни там, ни там. К модели уходит только полоса "
              "между ними.",
        "en": "Below the lower bound a frame is plainly blurred, above the upper one it "
              "is plainly sharp, and neither is worth asking a model about. Only the "
              "band between them reaches the model.",
        "ja": "下限より下は明らかにぶれており、上限より上は明らかに鮮明で、どちらもモデル"
              "に尋ねる価値はありません。モデルに届くのは、その間の帯だけです。",
    },
    "settings_features_sharpness_max_edge_label": {
        "ru": "Размер кадра для оценки резкости, px",
        "en": "Frame size for the sharpness measure, px",
        "ja": "鮮鋭度を測るコマのサイズ (px)",
    },
    # F117: the ceiling belongs in the settings column rather than next to the cache
    # sizes, because it is a stored preference and the numbers next to the buttons are
    # a measurement. 0 is spelled out in the hint: an empty-looking limit is the one
    # value a person is most likely to misread.
    "settings_preview_max_gb_label": {
        "ru": "Потолок кэша превью, ГБ",
        "en": "Preview cache ceiling, GB",
        "ja": "プレビューキャッシュ上限 (GB)",
    },
    "settings_preview_max_gb_hint": {
        "ru": "0 — без потолка: кэш растёт, пока есть место (около 150 КБ на снимок, "
              "то есть ~45 ГБ на 300 тысячах). С потолком удаляются самые давно не "
              "читанные превью, пока кэш не уложится; выключать кэш ради места не "
              "стоит — холодный кадр стоит 336 мс против 73.",
        "en": "0 means no ceiling: the cache grows while there is room (about 150 KB a "
              "shot, so ~45 GB at 300 000). With a ceiling the least recently read "
              "previews are dropped until it fits. Do not switch the cache off to save "
              "space — a cold frame costs 336 ms against 73.",
        "ja": "0 は上限なし: 空きがある限り増えます (1 枚およそ 150 KB、30 万枚で ~45 GB)。"
              "上限を設けると、収まるまで最も長く読まれていないプレビューから削除されます。"
              "容量のためにキャッシュを切るのは得策ではありません — 未キャッシュのコマは "
              "73 ms に対し 336 ms かかります。",
    },
    "settings_folders_title": {"ru": "Папки", "en": "Folders", "ja": "フォルダ"},
    "settings_folder_lang_hint": {
        "ru": "Язык названий папок раскладки. План ниже пересчитывается сразу.",
        "en": "The language of the layout's folder names. The plan below is recomputed "
              "immediately.",
        "ja": "振り分けフォルダ名の言語。下のプランはすぐに再計算されます。",
    },
    "settings_saved": {"ru": "Сохранено.", "en": "Saved.", "ja": "保存しました。"},
    "settings_error_prefix": {
        "ru": "Не удалось сохранить настройку: ", "en": "Could not save the setting: ",
        "ja": "設定を保存できませんでした: ",
    },
    "settings_busy": {
        "ru": "Идёт прогон — настройки не меняются на ходу. Дождитесь окончания.",
        "en": "A run is in progress — settings do not change mid-run. Wait for it to end.",
        "ja": "処理の実行中です — 途中で設定は変更できません。終了までお待ちください。",
    },
    # F145: the same statement for everything else that writes — marks, the trash, an
    # album, a layout. The server has always answered 409; this is the sentence that
    # says so BEFORE the click instead of after it.
    "actions_busy": {
        "ru": "Идёт прогон — действия, меняющие данные, недоступны. "
              "Вернутся сами по окончании.",
        "en": "A run is in progress — actions that change data are unavailable. "
              "They come back on their own when it ends.",
        "ja": "処理の実行中です — データを変更する操作は利用できません。"
              "終了すると自動的に戻ります。",
    },
    "selection_delete_hint": {
        "ru": "Файлы уедут в корзину системы — не мимо неё.",
        "en": "The files go to the system trash, not past it.",
        "ja": "ファイルはシステムのゴミ箱に移動します（完全削除ではありません）。",
    },
    "sort_confirm_title": {
        "ru": "Разложить коллекцию?", "en": "Lay the collection out?",
        "ja": "コレクションを振り分けますか?",
    },
    "sort_confirm_ok": {"ru": "Разложить", "en": "Apply", "ja": "振り分ける"},
    "sort_confirm_cancel": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "sort_summary_dest": {
        "ru": "Куда: {dest}", "en": "Where to: {dest}", "ja": "移動先: {dest}",
    },
    "sort_summary_mode_move": {
        "ru": "Перемещение — оригиналы будут перенесены",
        "en": "Move — the originals will be transferred",
        "ja": "移動 — オリジナルが移されます",
    },
    "sort_summary_mode_copy": {
        "ru": "Копирование — оригиналы останутся на месте",
        "en": "Copy — the originals stay where they are",
        "ja": "コピー — オリジナルはその場に残ります",
    },
    "sort_summary_files": {
        "ru": "{n} файлов в {dirs} папок, {size}",
        "en": "{n} files into {dirs} folders, {size}",
        "ja": "{n} 件のファイルを {dirs} 個のフォルダへ、{size}",
    },
    "sort_summary_existing": {
        "ru": "В назначении уже лежит {n} из них; {same} совпадут и будут пропущены",
        "en": "{n} of them are already in the destination; {same} match and will be skipped",
        "ja": "そのうち {n} 件はすでに移動先にあります。{same} 件は一致するためスキップされます",
    },
    "sort_summary_existing_none": {
        "ru": "В назначении ничего из этого ещё нет",
        "en": "None of this is in the destination yet",
        "ja": "これらはまだ移動先にありません",
    },
    "sort_summary_existing_unknown": {
        "ru": "Что уже лежит в назначении — неизвестно: папка не задана, а источник не один",
        "en": "What is already in the destination is unknown: no folder given and more "
              "than one source",
        "ja": "移動先に何があるかは不明です: フォルダ未指定でソースが複数あります",
    },
    "sort_summary_service": {
        "ru": "В служебные папки: товары — {products}, документы — {documents}",
        "en": "Into the review folders: products — {products}, documents — {documents}",
        "ja": "確認用フォルダへ: 商品 — {products} 件、書類 — {documents} 件",
    },
    "sort_summary_empty": {
        "ru": "Раскладывать нечего: план пуст. Обработайте коллекцию на вкладке "
              "«Обработка» — или снимите пометки «не трогать».",
        "en": "There is nothing to lay out: the plan is empty. Process the collection on "
              "the Process tab — or unmark the frames left alone.",
        "ja": "振り分ける対象がありません: プランが空です。「処理」タブでコレクションを"
              "処理するか、「そのままにする」の指定を解除してください。",
    },
    "sort_summary_error": {
        "ru": "Не удалось посчитать сводку: ", "en": "Could not compute the summary: ",
        "ja": "サマリーを計算できませんでした: ",
    },
    # --- F77: manual corrections to the layout (the "Cities" tab) ----------
    "override_exclude_button": {
        "ru": "Не трогать", "en": "Leave alone", "ja": "そのままにする",
    },
    "override_clear_button": {
        "ru": "Снять правку", "en": "Clear correction", "ja": "修正を解除",
    },
    "override_move_button": {
        "ru": "Перенести в…", "en": "Move to…", "ja": "移動先…",
    },
    "override_target_placeholder": {
        "ru": "папка раскладки…", "en": "layout folder…", "ja": "振り分け先フォルダ…",
    },
    "override_exclude_folder_button": {
        "ru": "Не трогать папку", "en": "Leave folder alone", "ja": "フォルダをそのままに",
    },
    "override_exclude_folder_confirm": {
        "ru": "Исключить из раскладки все файлы этой папки ({n})? Они останутся там, "
              "где лежат.",
        "en": "Exclude all {n} files of this folder from the layout? They stay exactly "
              "where they are.",
        "ja": "このフォルダの {n} 件すべてを振り分けから除外しますか? "
              "ファイルは現在の場所に残ります。",
    },
    "override_excluded_mark": {
        "ru": "не трогать", "en": "left alone", "ja": "移動しない",
    },
    "override_reassigned_mark": {
        "ru": "перенос → {target}", "en": "moved → {target}", "ja": "移動先 → {target}",
    },
    "override_hint": {
        "ru": "Ручные правки сильнее автоматики: помеченные «не трогать» остаются на "
              "месте, перенесённые уходят в выбранную папку при раскладке.",
        "en": "Manual corrections outrank the automatic rules: files marked «leave "
              "alone» stay put, moved ones go to the chosen folder when you apply.",
        "ja": "手動の修正は自動判定より優先されます。「そのままにする」を付けた"
              "ファイルは移動せず、移動先を指定したものは振り分け時にそのフォルダへ入ります。",
    },
    "override_alert_choose_target": {
        "ru": "Выберите папку для переноса.", "en": "Choose a destination folder.",
        "ja": "移動先のフォルダを選択してください。",
    },
    "override_error_prefix": {
        "ru": "Не удалось сохранить правку: ", "en": "Could not save the correction: ",
        "ja": "修正を保存できません: ",
    },
    # F85c: assigning a place to a whole group by hand
    "place_search_placeholder": {
        "ru": "Город или страна", "en": "City or country", "ja": "都市または国",
    },
    "place_assign_button": {
        "ru": "Назначить место", "en": "Assign place", "ja": "場所を指定",
    },
    "place_clear_button": {
        "ru": "Отменить назначение", "en": "Undo assignment", "ja": "指定を取り消す",
    },
    "place_folder_button": {
        "ru": "Место для исходной папки", "en": "Place for the source folder",
        "ja": "元フォルダの場所",
    },
    "place_not_found": {
        "ru": "Такого места нет в базе — проверьте написание.",
        "en": "No such place in the bundled data — check the spelling.",
        "ja": "その場所は同梱データにありません。綴りを確認してください。",
    },
    "place_alert_choose": {
        "ru": "Сначала выберите место из списка.",
        "en": "Pick a place from the list first.",
        "ja": "先に一覧から場所を選んでください。",
    },
    "place_assign_confirm": {
        "ru": "Назначить место «{place}» файлам этой группы ({n})?",
        "en": "Assign the place «{place}» to the files of this group ({n})?",
        "ja": "このグループのファイル（{n}）に場所「{place}」を指定しますか？",
    },
    "place_folder_confirm": {
        "ru": "Назначить место «{place}» всем файлам исходной папки «{dir}»?",
        "en": "Assign the place «{place}» to every file of the source folder «{dir}»?",
        "ja": "元フォルダ「{dir}」のすべてのファイルに場所「{place}」を指定しますか？",
    },
    "place_event_clear_confirm": {
        "ru": "Снять назначенное место с файлов этого события ({n})?",
        "en": "Remove the assigned place from the files of this event ({n})?",
        "ja": "このイベントのファイル（{n}）から指定した場所を解除しますか？",
    },
    "place_folder_clear_confirm": {
        "ru": "Снять назначенное место с файлов исходной папки «{dir}»?",
        "en": "Remove the assigned place from the files of the source folder «{dir}»?",
        "ja": "元フォルダ「{dir}」のファイルから指定した場所を解除しますか？",
    },
    "place_assigned_status": {
        "ru": "Назначено: {n}", "en": "Assigned: {n}", "ja": "指定しました: {n}",
    },
    "place_cleared_status": {
        "ru": "Назначение снято: {n}", "en": "Assignment removed: {n}",
        "ja": "指定を解除しました: {n}",
    },
    "place_skipped_gps": {
        "ru": " · с точным GPS пропущено: {n}",
        "en": " · skipped, they have exact GPS: {n}",
        "ja": " · GPS があるためスキップ: {n}",
    },
    "place_include_gps_confirm": {
        "ru": "{n} файлов уже имеют координаты из камеры — они не тронуты. "
              "Перезаписать место и у них?",
        "en": "{n} files already carry camera coordinates and were left alone. "
              "Overwrite their place too?",
        "ja": "{n} 件はカメラの座標を持つためそのままです。これらの場所も上書きしますか？",
    },
    "place_manual_mark": {
        "ru": "место назначено вручную", "en": "place assigned by hand",
        "ja": "場所は手動指定",
    },
    "place_hint": {
        "ru": "Место назначается группе целиком — событию или исходной папке. Оно "
              "переживает пересчёт гео и видно в плане как «вручную».",
        "en": "A place is assigned to a whole group — an event or a source folder. It "
              "survives a geo recompute and shows up in the plan as «manual».",
        "ja": "場所はグループ単位（イベントまたは元フォルダ）で指定します。位置情報の"
              "再計算後も残り、プランには「手動」と表示されます。",
    },
    "place_error_prefix": {
        "ru": "Не удалось назначить место: ", "en": "Could not assign the place: ",
        "ja": "場所を指定できません: ",
    },
    # F103: the "Utility frames" view — the buckets the classifier carries out of
    # the collection, and the bulk way back for the frames it got wrong.
    # F175: the caption of the WHOLE slice names no percentage, and deliberately. Behind
    # one name lie four buckets measured separately (products 78%, screenshots 59%,
    # documents and memes not measured at all), and a single number over them would be
    # honest about none of them. What it does say is the thing a person has to know
    # before ticking everything: one of the four is not to be deleted.
    "junk_intro": {
        "ru": "Кадры, снятые не ради памяти: товары, скриншоты, документы, мемы. Это "
              "четыре разные корзины с разной надёжностью — откройте любую, и подпись "
              "назовёт её точность. Документы удалять нельзя: там паспорта, справки и "
              "чеки. Отметьте кадры, попавшие сюда зря, и верните их — они снова "
              "разложатся по городам. Вердикт модели при этом не переписывается.",
        "en": "Frames that were not taken for memory: products, screenshots, documents, "
              "memes. These are four different buckets of different reliability — open "
              "any one of them and the caption names its precision. The documents are "
              "not to be deleted: passports, certificates and receipts live there. Tick "
              "the frames that landed here by mistake and return them — they go back "
              "into the city layout. The model's verdict itself is not rewritten.",
        "ja": "思い出のためではなく実用のために撮られたコマです: 商品、"
              "スクリーンショット、書類、ミーム。信頼度の異なる 4 つの別々のバケットで、"
              "いずれかを開くとその精度が説明に出ます。書類は削除できません — "
              "パスポート、証明書、レシートが入っています。誤って入ったコマに"
              "チェックを入れて戻すと、再び都市ごとに振り分けられます。モデルの"
              "判定自体は書き換えません。",
    },
    # F175: precision belongs to a CLASS, not to the slice. Each line below is one
    # measurement with its date and its sample size, shown when that bucket is the one
    # open. A class nobody has measured gets `junk_accuracy_unmeasured` — the lookup in
    # the client falls back to it, so a class added later says "not measured" instead of
    # quietly inheriting somebody else's number.
    "junk_accuracy_product": {
        "ru": "Точность 78% при полноте 81% (замер 2026-08-03, 999 кадров): примерно "
              "каждый пятый кадр здесь — не товар.",
        "en": "Precision 78% at 81% recall (measured 2026-08-03 on 999 frames): about "
              "one frame in five here is not a product.",
        "ja": "精度 78%、再現率 81%（2026-08-03、999 コマで測定）: ここにあるコマの"
              "およそ 5 枚に 1 枚は商品ではありません。",
    },
    # F171: this bucket states an OPINION and has to be read as one. The rescue of
    # 2026-08-04 added 441 frames to it (1 782 against 1 341) and 41% of what it adds is
    # an ordinary photograph — about 181 personal pictures leaving the city layout for a
    # bucket a person reads as "these are your screenshots" and does not look through.
    # So the caption names the model as the author of the verdict, and names returning a
    # frame as the ordinary next step rather than as the repair of a rare mistake.
    "junk_accuracy_screenshot": {
        "ru": "Модель считает эти кадры экранными — это её оценка, а не факт. Точность "
              "59% при полноте 83% (замер 2026-08-03, 350 кадров): каждый "
              "третий кадр здесь — обычная фотография. Просмотрите список перед "
              "удалением и верните такие кадры в раскладку — здесь это обычный шаг "
              "работы, а не исправление редкой ошибки.",
        "en": "The model considers these frames screen captures — that is its estimate "
              "and not a fact. Precision 59% at 83% recall (measured 2026-08-03 on 350 "
              "frames): every third frame here is an ordinary photograph. Look the list "
              "over before deleting anything and return such frames to the layout — "
              "here that is an ordinary step of the work, not the repair of a rare "
              "mistake.",
        "ja": "モデルはこれらのコマを画面のコマだと考えています — 事実ではなく推定です。"
              "精度 59%、再現率 83%（2026-08-03、350 コマで測定）: ここにあるコマの"
              "3 枚に 1 枚は普通の写真です。削除する前にリストを見て、そうしたコマは"
              "振り分けに戻してください — ここではそれが通常の作業であり、まれな誤りの"
              "修正ではありません。",
    },
    "junk_accuracy_unmeasured": {
        "ru": "Точность этой корзины не измерена — сколько здесь ошибок, неизвестно.",
        "en": "The precision of this bucket has not been measured — how many frames "
              "here are wrong is not known.",
        "ja": "このバケットの精度は測定されていません — 誤りがどれだけあるかは"
              "分かりません。",
    },
    # F171: appended to the caption of the bucket that is open, and ONLY where the server
    # says the page was actually ordered by the model's own estimate (`ordered_by_score`).
    # A promise about the order that is true on one collection and silent on another is
    # the F157 rule: the sentence appears exactly where the ordering it describes does.
    "junk_order_hint": {
        "ru": " Список идёт от кадров, в которых модель уверена больше, к сомнительным: "
              "читайте сверху и остановитесь, где сходство кончилось.",
        "en": " The list runs from the frames the model is most sure of down to the "
              "doubtful ones: read from the top and stop where the resemblance ends.",
        "ja": " リストはモデルの確信が強いコマから弱いコマへ並びます。上から読み、"
              "似ていると思えなくなった所で止めてください。",
    },
    "junk_bucket_product": {"ru": "Товары", "en": "Products", "ja": "商品"},
    "junk_bucket_document": {"ru": "Документы", "en": "Documents", "ja": "書類"},
    "junk_bucket_screenshot": {"ru": "Скриншоты", "en": "Screenshots",
                               "ja": "スクリーンショット"},
    "junk_bucket_meme": {"ru": "Мемы", "en": "Memes", "ja": "ミーム"},
    "junk_empty": {
        "ru": "Здесь пусто — таких кадров нет.",
        "en": "Nothing here — there are no such frames.",
        "ja": "ここは空です。該当するフレームはありません。",
    },
    "junk_restore_confirm": {
        "ru": "{n} кадров вернутся в раскладку: {breakdown}. Продолжить?",
        "en": "{n} frames will return to the layout: {breakdown}. Continue?",
        "ja": "{n} 件が振り分けに戻ります: {breakdown}。続けますか？",
    },
    "junk_undo_restore_button": {
        "ru": "Отменить возврат", "en": "Undo the return", "ja": "戻すのを取り消す",
    },
    # F174: nothing has moved yet — the mark applies on the next `sort --apply`, and a
    # past tense here would promise a transfer that has not happened.
    "junk_restored_mark": {
        "ru": "вернётся в раскладку", "en": "will return to the layout",
        "ja": "振り分けに戻ります",
    },
    "junk_select_all": {"ru": "Выбрать всё на странице",
                        "en": "Select everything on this page",
                        "ja": "このページをすべて選択"},
    "junk_select_none": {"ru": "Снять выделение", "en": "Clear the selection",
                         "ja": "選択を解除"},
    "junk_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "junk_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    "junk_document_no_preview": {
        "ru": "без превью", "en": "no preview", "ja": "プレビューなし",
    },
    # F175: the hint says what a document IS before it says how it is shown. This slice
    # reads as "junk, select all, delete", and the frames the sentence is about are the
    # ones a person needs most — so the warning has to arrive above the grid, before the
    # selection, and not as an explanation of a missing thumbnail.
    "junk_document_hint": {
        "ru": "Документы здесь — не на удаление: это паспорта, справки, чеки и "
              "медицинские бланки, и они помечены отдельно. Sorta их не открывает и не "
              "показывает; видно имя файла и дату — этого хватает, чтобы решить.",
        "en": "The documents here are not for deletion: they are passports, "
              "certificates, receipts and medical forms, and they are marked out "
              "separately. Sorta neither opens nor renders them; the file name and the "
              "date are shown — enough to decide.",
        "ja": "ここにある書類は削除の対象ではありません: パスポート、証明書、"
              "レシート、診断書であり、別に印が付いています。Sorta はそれらを開かず"
              "表示もしません。判断にはファイル名と日付で十分です。",
    },
    # The same warning ON the card, because the hint above the grid is read once and the
    # selection is made card by card.
    "junk_document_mark": {
        "ru": "не удалять", "en": "not for deletion", "ja": "削除しない",
    },
    "junk_error_prefix": {
        "ru": "Не удалось вернуть кадры: ", "en": "Could not return the frames: ",
        "ja": "フレームを戻せません: ",
    },
    "error_loading_junk": {
        "ru": "Не удалось загрузить корзины: ", "en": "Could not load the buckets: ",
        "ja": "バケットを読み込めません: ",
    },
    # --- F174: the action names its destination ---------------------------------------
    # ONE name for one intention. "This is not an animal" and "return to the photos" are
    # the same movement to the person making it — the frame does not belong in this slice
    # — so the button reads the same in both, and what differs (a real transfer versus a
    # membership) is said UNDER it, in `dest_goes_to` / `dest_stays_in`. Two buttons for
    # one intention was the whole complaint.
    "slice_return_button": {
        "ru": "Вернуть в раскладку", "en": "Return to the layout", "ja": "振り分けに戻す",
    },
    "dest_goes_to": {
        "ru": "попадёт в {folder}", "en": "goes into {folder}",
        "ja": "{folder} に入ります",
    },
    "dest_stays_in": {
        "ru": "уберём из среза; кадр и так лежит в {folder}, файл не двинется",
        "en": "we take it out of the slice; the frame already lies in {folder}, "
              "the file will not move",
        "ja": "区分から外すだけです。コマはすでに {folder} にあり、ファイルは動きません",
    },
    "dest_unknown": {
        "ru": "папку назначения посчитать не удалось",
        "en": "the destination folder could not be computed",
        "ja": "保存先フォルダーを計算できませんでした",
    },
    # Looked up as `dest_why_<reason>` — the plan's own reason codes. A reason without a
    # key simply gets no explanation, the way an unknown bucket gets no chip label.
    "dest_why_no_place": {
        "ru": "у кадра нет геоданных", "en": "the frame carries no geodata",
        "ja": "このコマに位置情報がありません",
    },
    "dest_why_country_only": {
        "ru": "город не определился — известна только страна",
        "en": "no city resolved — only the country is known",
        "ja": "都市は不明で、国だけが分かっています",
    },
    "dest_why_low_date": {
        "ru": "у кадра нет надёжной даты съёмки",
        "en": "the frame carries no reliable capture date",
        "ja": "このコマに信頼できる撮影日がありません",
    },
    "dest_why_downloaded": {
        "ru": "ни даты съёмки, ни следов камеры — это скачанный кадр",
        "en": "no capture date and no camera trace — a downloaded frame",
        "ja": "撮影日もカメラの痕跡もありません。ダウンロードされたコマです",
    },
    # The bulk caption groups by destination instead of naming one folder: a person
    # selects dozens at a time, and one folder name out of twelve deceives them.
    "dest_bulk_summary": {
        "ru": "{n} кадров вернутся: {breakdown}",
        "en": "{n} frames will return: {breakdown}",
        "ja": "{n} 件が戻ります: {breakdown}",
    },
    "dest_bulk_item": {"ru": "{n} {group}", "en": "{n} {group}", "ja": "{n} 件 {group}"},
    "dest_group_city": {"ru": "в города", "en": "into cities", "ja": "都市へ"},
    "dest_group_country": {
        "ru": "на уровень страны", "en": "to the country level", "ja": "国のレベルへ",
    },
    "dest_group_no_place": {
        "ru": "в «без места»", "en": "into “no place”", "ja": "「場所不明」へ",
    },
    "dest_group_undated": {
        "ru": "в «без даты»", "en": "into “no date”", "ja": "「日付不明」へ",
    },
    "dest_group_other": {
        "ru": "в другие папки", "en": "into other folders", "ja": "その他のフォルダーへ",
    },
    # --- F123: the "Animals" tab -----------------------------------------------------
    # F160: the caption states BOTH measurements, because the slice is two different
    # promises and a config switch chooses between them. The cascade alone is 82%
    # precision at 64% recall; with the object detector on (`features.detector` +
    # `detect.enabled`) it is 62% at 87% — a quarter more animals found and a fifth of the
    # confidence given up for them. A caption naming one number while the other rule is in
    # force buys recall with the reader's trust, which is the mistake F158 measured on the
    # very same line.
    "animals_intro": {
        "ru": "Кадры с животными, сверху — те, в которых модель уверена больше. "
              "Точность около 82% при полноте 64%; с включённым детектором объектов "
              "(features.detector) размен другой — точность около 62% при полноте 87%: "
              "животных находится заметно больше, а шуб и игрушек среди них тоже. "
              "Ниже по списку видно, где проходит граница.",
        "en": "Frames with animals, the ones the model is most confident about first. "
              "Precision is about 82% at 64% recall; with the object detector on "
              "(features.detector) the trade is a different one — about 62% precision at "
              "87% recall: noticeably more animals found, and more fur coats and plush "
              "toys among them. Further down the list is where the border of confidence "
              "runs.",
        "ja": "動物が写ったコマです。モデルの確信度が高い順に並びます。精度は約 82%、"
              "再現率は 64% です。物体検出を有効にすると (features.detector) 精度は約 "
              "62%、再現率は 87% になり、見つかる動物は増えますが毛皮のコートや"
              "ぬいぐるみも増えます。下に行くほど確信度の境目が見えてきます。",
    },
    "animals_empty": {
        "ru": "Здесь пусто — животные не найдены.",
        "en": "Nothing here — no animals were found.",
        "ja": "ここは空です。動物は見つかりませんでした。",
    },
    "animals_score_label": {
        "ru": "уверенность {score}", "en": "confidence {score}", "ja": "確信度 {score}",
    },
    # F173: the button and the counter of this slice are `slice_load_more` /
    # `slice_shown_label` now — the shared pager's, like every other ordered list.
    "error_loading_animals": {
        "ru": "Не удалось загрузить животных: ", "en": "Could not load the animals: ",
        "ja": "動物を読み込めません: ",
    },
    # --- F124: taking a false mark off a frame (and putting a missing one back) --------
    # The two buttons are one toggle: the card offers the answer the frame does NOT have
    # right now. The third string is the way back to the automatic verdict, which is a
    # different thing from "not an animal" and therefore says so in words.
    # F174: the "take it off this frame" half is `slice_return_button` now — the same
    # words the junk view uses, because it is the same intention. What the two do differ
    # in is stated under the button (`dest_stays_in` here, `dest_goes_to` there).
    "animals_mark_animal": {
        "ru": "Это животное", "en": "This is an animal", "ja": "これは動物",
    },
    "animals_mark_clear": {
        "ru": "Вернуть автоматически", "en": "Back to automatic", "ja": "自動判定に戻す",
    },
    "animals_manual_excluded": {
        "ru": "снято вручную", "en": "unmarked by hand", "ja": "手動で解除",
    },
    "animals_manual_included": {
        "ru": "отмечено вручную", "en": "marked by hand", "ja": "手動で設定",
    },
    "animals_counted_label": {
        "ru": "Животных: {n}", "en": "Animals: {n}", "ja": "動物: {n} 件",
    },
    "animals_error_prefix": {
        "ru": "Не удалось сохранить отметку: ", "en": "Could not save the mark: ",
        "ja": "マークを保存できません: ",
    },
    # --- F126: the "Review" workspace -------------------------------------------------
    # The switcher labels are the slices; the duplicates one keeps the wording the tab
    # had, because that is what the user has been calling it since U3.
    "review_slice_dupes": {"ru": "Дубли", "en": "Duplicates", "ja": "重複"},
    "review_slice_blurred": {"ru": "Размытые", "en": "Blurred", "ja": "ぼやけ"},
    "review_slice_eyes": {"ru": "Закрытые глаза", "en": "Closed eyes", "ja": "目を閉じた"},
    # F150: named after the FACT and never after a judgement. "Bad" or "junk" would be a
    # verdict the program has no business passing on a picture somebody was sent once and
    # never got again.
    "review_slice_low_resolution": {
        "ru": "Низкое разрешение", "en": "Low resolution", "ja": "低解像度",
    },
    "review_intro": {
        "ru": "Одно место для всего, что надо просмотреть глазами и частью удалить. "
              "Отметка «удалить» — это пометка, а не удаление: файлы уедут в папку "
              "«_удалить» на следующей раскладке. Отметка «оставить» переживает "
              "пересчёт и больше не спросится.",
        "en": "One place for everything that has to be looked at by eye and partly "
              "deleted. Marking “delete” is a mark, not a deletion: those files go to "
              "the “_delete” folder on the next layout. A “keep” survives a recompute "
              "and is not asked about again.",
        "ja": "目で確認して一部を削除する作業を、ここ一か所にまとめています。「削除」は"
              "印であって削除ではありません。対象は次回の振り分けで「_削除」フォルダへ"
              "移ります。「残す」は再計算後も保持され、再び尋ねられません。",
    },
    # F157: the caption of a RANKING. It used to describe a window ("the list opens down
    # to 90"), which read as a verdict about the frames inside it — and the number behind
    # that reading catches 12% of what a person calls blurred. The list is now ordered from
    # the softest frame, `{max}` is only how far the first page reaches, and the sentence
    # says the two things a reader has to know: read from the top and stop where the
    # resemblance ends, and this number cannot tell a detailed sharp street from a smooth
    # blurred face. The "delete everything below the threshold" line stays: a button this
    # feature deliberately does not have has to be named, or somebody adds it.
    "review_hint_blurred": {
        "ru": "Это порядок, а не приговор. Сверху кадры, которые почти наверняка "
              "смазаны; ниже резкость растёт, и где-то начинаются нормальные "
              "фотографии — читайте сверху вниз и остановитесь, где сходство кончилось. "
              "Первая страница открыта до резкости {max}, «показать ещё» идёт дальше по "
              "списку. Признак грубый: детализированная резкая улица и гладкое размытое "
              "лицо дают близкие числа, поэтому кнопки «удалить всё ниже порога» здесь "
              "нет и по умолчанию не удаляется ничего.",
        "en": "This is an order, not a verdict. At the top are frames that are almost "
              "certainly smeared; further down the sharpness grows and at some point "
              "ordinary photographs begin — read from the top and stop where the "
              "resemblance ends. The first page opens down to a sharpness of {max}, and "
              "“show more” simply continues down the list. The signal is coarse: a "
              "detailed sharp street and a smooth blurred face score alike, so there is "
              "no “delete everything below the threshold” button here and nothing is "
              "marked by default.",
        "ja": "これは判定ではなく並び順です。上にあるのはほぼ確実にぶれているコマで、"
              "下にいくほど鮮鋭度は上がり、どこかで普通の写真が始まります。上から読み、"
              "似ていると思えなくなった所で止めてください。最初のページは鮮鋭度 {max} "
              "まで開き、「さらに表示」はその先へ続きます。この指標は粗いものです。"
              "細部の多い鮮明な街並みと、なめらかにぼけた顔は近い値になるため、"
              "「しきい値以下をすべて削除」というボタンはなく、既定では何も削除しません。",
    },
    # F155 + F157: shown only where `frame_quality.face_sharpness` exists, because only
    # there is it true. It is the answer to "why is this sharp-looking street above that
    # soft portrait": the frames with a face are ordered by a different number, measured
    # inside the face, which finds 62% of the blurred ones against 15% for the whole frame.
    "review_hint_blurred_faces": {
        "ru": " Кадры с лицами идут первыми и упорядочены по резкости самого лица — "
              "по кадру целиком этот признак их почти не находит.",
        "en": " Frames with a face come first and are ordered by the sharpness measured "
              "inside the face — over the whole frame this signal barely finds them.",
        "ja": " 顔のあるコマが先に並び、顔の内側で測った鮮鋭度で順序づけられます。"
              "コマ全体で測ると、この指標はそれらをほとんど拾えません。",
    },
    # F179: the caption states the MEASURED PRECISION and not a count. "Found 730 frames"
    # reads as a verdict about 730 photographs; on 249 hand-labelled frames this list is
    # right about 62% of what it points at, which is the one thing a person needs to know
    # before opening it. The list is ordered from the most closed, so the top is where the
    # 62% lives and "show more" walks into the doubtful part on purpose.
    "review_hint_eyes": {
        "ru": "Кадры, на которых у людей, скорее всего, закрыты глаза: посчитано по "
              "геометрии век самого крупного лица, а не спрошено у модели. На 249 "
              "размеченных кадрах такой список верен в 62% случаев — каждый третий кадр "
              "здесь на самом деле с открытыми глазами, поэтому ничего не удаляется само. "
              "Сверху самые закрытые; «показать ещё» продолжает список за порог {max}, в "
              "сомнительную часть. Мерится только там, где найдено лицо.",
        "en": "Frames where the people most likely have their eyes closed — computed from "
              "the eyelid geometry of the largest face, not asked of a model. On 249 "
              "hand-labelled frames a list like this is right 62% of the time: one frame "
              "in three here actually has its eyes open, so nothing is ever deleted "
              "automatically. The most closed come first, and “show more” continues past "
              "the {max} mark into the doubtful part. Measured only where a face was found.",
        "ja": "最も大きい顔のまぶたの形状から算出した、目を閉じている可能性が高いコマです"
              "（モデルへの質問ではありません）。手作業でラベル付けした 249 コマでは、この"
              "一覧の正解率は 62% です。3 コマに 1 コマは実際には目が開いているため、自動的"
              "な削除は行いません。閉じている度合いの高い順に並び、「もっと見る」はしきい値 "
              "{max} を越えて確度の低い部分へ進みます。顔が検出されたコマのみで計測します。",
    },
    "review_eyes_no_faces": {
        "ru": "Данных нет: стадия «лица» не запускалась, а глаза мерятся только там, где "
              "найдено лицо. Прогоните лица и повторите разбор.",
        "en": "No data: the faces stage never ran, and the eyes are only measured where a "
              "face was found. Run faces and come back to this slice.",
        "ja": "データがありません。顔ステージが実行されておらず、目の計測は顔が検出された"
              "コマにのみ行われます。顔ステージを実行してから、この区分を開いてください。",
    },
    # F150: the whole boundary of the slice, said out loud. Three things a person has to
    # know before pressing anything here: it is a fact and not a verdict (the frame may be
    # the only copy of something), megapixels say nothing about a big frame ruined by
    # compression, and videos are not in this list at all.
    "review_hint_low_resolution": {
        "ru": "Кадры меньше {mp} мегапикселя, сначала самые мелкие. Это факт из индекса, "
              "а не оценка: ширина и высота записаны при индексации, ничего не "
              "измерялось. Малое разрешение — не признак брака: это может быть "
              "единственная сохранившаяся фотография, присланная десять лет назад, "
              "поэтому по умолчанию не удаляется ничего. Пережатое сюда не попадает: "
              "кадр 4000×3000, убитый JPEG-артефактами, формально большой — это другой "
              "сигнал и другой разговор. Видео не считаем.",
        "en": "Frames smaller than {mp} megapixels, the smallest first. This is a fact "
              "out of the index rather than an estimate: width and height were written "
              "down when the file was indexed and nothing was measured. A small frame is "
              "not a faulty one — it can be the only surviving photograph, sent ten "
              "years ago — so nothing is marked for deletion by default. Over-compressed "
              "frames are not here: a 4000×3000 picture ruined by JPEG artefacts is "
              "formally large, and that is a different signal and a different "
              "conversation. Videos are not counted.",
        "ja": "{mp} メガピクセル未満のコマを、小さい順に並べています。これは推定ではなく"
              "索引に記録された事実です。幅と高さは登録時に書き込まれたもので、何も"
              "測定していません。解像度が低いことは欠陥ではありません — 十年前に送られて"
              "きた唯一の一枚かもしれないので、既定では何も削除の印を付けません。"
              "圧縮で潰れたコマはここには入りません。JPEG のノイズで壊れた 4000×3000 の"
              "画像は形式上は大きく、それは別の指標であり別の話です。動画は数えません。",
    },
    # The size of the picture, as a person reads it off a camera: the two sides and the
    # megapixels they come to.
    "review_resolution_label": {
        "ru": "{w}×{h} ({mp} Мп)", "en": "{w}×{h} ({mp} MP)", "ja": "{w}×{h}（{mp} MP）",
    },
    "review_empty": {
        "ru": "Здесь пусто — таких кадров нет.",
        "en": "Nothing here — there are no such frames.",
        "ja": "ここは空です。該当するフレームはありません。",
    },
    "review_sharpness_label": {
        "ru": "резкость {value}", "en": "sharpness {value}", "ja": "鮮鋭度 {value}",
    },
    "review_mark_delete": {
        "ru": "Пометить на удаление", "en": "Mark for deletion", "ja": "削除の印を付ける",
    },
    "review_mark_keep": {"ru": "Оставить", "en": "Keep", "ja": "残す"},
    "review_mark_clear": {
        "ru": "Снять отметку", "en": "Clear the mark", "ja": "印を外す",
    },
    # --- F149: "try to improve" — the third action, on one frame ----------------------
    # Every string here says PROCESSED, never "improved". The model draws something
    # plausible instead of bringing back what was lost, and an interface that calls that
    # an improvement is the one thing this feature must not do: a person has to know at
    # every moment which of the two pictures in front of them is the photograph.
    "review_restore": {
        "ru": "Попробовать улучшить", "en": "Try to improve", "ja": "補正を試す",
    },
    "review_restore_hint": {
        "ru": "Один кадр за раз: выберите ровно один. Модель НЕ возвращает утраченное — "
              "она дорисовывает правдоподобное, поэтому рядом появится помеченная "
              "копия, а оригинал останется как есть. Первое нажатие качает веса "
              "(~400 МБ), дальше — около секунды на кадр.",
        "en": "One frame at a time: select exactly one. The model does NOT bring back "
              "what was lost — it draws something plausible — so what appears beside the "
              "original is a marked copy, and the original stays as it is. The first "
              "press downloads the weights (~400 MB), after that it is about a second "
              "per frame.",
        "ja": "一度に 1 枚だけです。ちょうど 1 枚を選んでください。モデルは失われた情報を"
              "復元するのではなく、それらしく描き足します。そのため元の写真の隣には印の"
              "付いた複製が現れ、元の写真はそのまま残ります。初回は重み (約 400 MB) を"
              "取得し、その後は 1 枚あたり約 1 秒です。",
    },
    "review_restore_running": {
        "ru": "Обрабатываем кадр…", "en": "Processing the frame…", "ja": "処理中…",
    },
    "review_restore_badge": {
        "ru": "обработано моделью", "en": "processed by a model", "ja": "モデルによる処理",
    },
    "review_restore_badge_hint": {
        "ru": "Это НЕ фотография, а копия, дорисованная моделью: детали на ней "
              "правдоподобные, но выдуманные. Оригинал не изменён и лежит рядом. "
              "Оставить можно любую, обе или ни одной — выбор копии сам по себе ничего "
              "не помечает на удаление.",
        "en": "This is NOT a photograph but a copy a model drew over: its detail is "
              "plausible and invented. The original is unchanged and lies beside it. Keep "
              "either, both or neither — choosing the copy marks nothing for deletion by "
              "itself.",
        "ja": "これは写真ではなく、モデルが描き足した複製です。細部はそれらしく見えますが"
              "作られたものです。元の写真は変更されず隣にあります。どちらを残しても、"
              "両方でも、どちらも残さなくても構いません。複製を選んでも、それだけでは"
              "何も削除対象になりません。",
    },
    "review_restore_done": {
        "ru": "Готово: обработанная копия рядом с оригиналом. Оригинал не изменён.",
        "en": "Done: the processed copy is beside the original. The original is unchanged.",
        "ja": "完了しました。処理済みの複製が元の写真の隣にあります。元の写真は変更されて"
              "いません。",
    },
    "review_restore_reused": {
        "ru": "Такая копия уже была — показываем её, второй не делаем.",
        "en": "That copy already existed — here it is; a second one is not made.",
        "ja": "その複製はすでに存在します。既存のものを表示し、二つ目は作りません。",
    },
    # F169: the sentence a full-sized frame is owed. The model is x4 and cannot be shown
    # the whole frame, so a big one is REDUCED first and blown back up to about its own
    # size — the copy comes out the same size and holds less of what was really there.
    # Said next to "done", every time it happens, because it is the one outcome a person
    # cannot see by looking: the copy usually looks sharper, and sharper is not truer.
    "review_restore_rebuilt": {
        "ru": "Внимание: кадр больше предела ({max_edge} px по длинной стороне, здесь "
              "{source_edge}). Копия пересобрана из уменьшенной: настоящая детализация "
              "оригинала не попала в модель, и на её месте дорисована правдоподобная. "
              "Это не улучшение оригинала — предел меняется ключом "
              "features.restore_max_edge.",
        "en": "Note: this frame is larger than the limit ({max_edge} px on the longer "
              "side, this one is {source_edge}). The copy was rebuilt from a reduced "
              "frame: the real detail of the original never reached the model, and "
              "plausible detail was drawn in its place. This is not an improved original "
              "— the limit is the features.restore_max_edge key.",
        "ja": "注意: このコマは上限 (長辺 {max_edge} px、このコマは {source_edge} px) を"
              "超えています。複製は縮小した画像から作り直されました。元の写真の本当の"
              "細部はモデルに渡らず、代わりにそれらしい細部が描き足されています。"
              "元の写真が良くなったわけではありません。上限は "
              "features.restore_max_edge で変えられます。",
    },
    # --- F168: the same action, reached from the expanded frame in any slice ----------
    # The hint says the same things as `review_restore_hint` minus the one sentence that
    # belongs to the Review grid ("select exactly one"): here the frame IS the one being
    # looked at, and there is nothing to select.
    "review_restore_expanded_hint": {
        "ru": "Модель НЕ возвращает утраченное — она дорисовывает правдоподобное. Рядом "
              "с оригиналом появится помеченная копия, а оригинал останется как есть. "
              "Первое нажатие качает веса (~400 МБ), дальше — около секунды на кадр.",
        "en": "The model does NOT bring back what was lost — it draws something "
              "plausible — so what appears beside the original is a marked copy, and the "
              "original stays as it is. The first press downloads the weights (~400 MB), "
              "after that it is about a second per frame.",
        "ja": "モデルは失われた情報を復元するのではなく、それらしく描き足します。その"
              "ため元の写真の隣には印の付いた複製が現れ、元の写真はそのまま残ります。"
              "初回は重み (約 400 MB) を取得し、その後は 1 枚あたり約 1 秒です。",
    },
    # F168/F169: why the action is NOT offered on a big frame. The gain the measurement
    # found belongs to small frames (66% under 640 px, a coin toss by 1280), and above the
    # ceiling the copy would be rebuilt from a quarter of the original. Withdrawing the
    # button without a word would be the silent half of the same promise.
    "review_restore_too_large": {
        "ru": "Кадр крупнее предела ({max_edge} px по длинной стороне, здесь "
              "{source_edge}): копию пришлось бы пересобирать из уменьшенной, а на таких "
              "кадрах замер пользы не показал. Поэтому здесь действие не предлагается — "
              "предел меняется ключом features.restore_max_edge.",
        "en": "This frame is larger than the limit ({max_edge} px on the longer side, "
              "this one is {source_edge}): the copy would be rebuilt from a reduced "
              "frame, and on frames this size the measurement found no gain. So the "
              "action is not offered here — the limit is the features.restore_max_edge "
              "key.",
        "ja": "このコマは上限 (長辺 {max_edge} px、このコマは {source_edge} px) を超えて"
              "います。複製は縮小した画像から作り直すことになり、この大きさのコマでは"
              "効果が確認できませんでした。そのためここでは操作を提供しません。上限は "
              "features.restore_max_edge で変えられます。",
    },
    # The copy is a canonical file: it lies in the city folder beside its source and turns
    # up in every slice the source does. Wherever it is opened it says what it is and
    # which frame it was made from — otherwise it reads as a second similar photograph
    # that came from nowhere.
    "review_restore_source_badge": {
        "ru": "обработано моделью из {name}",
        "en": "processed by a model from {name}",
        "ja": "{name} をモデルで処理した複製",
    },
    "review_restore_error_sensitive_class": {
        "ru": "Кадр отнесён к личным документам (vlm.exclude_classes): такие кадры "
              "продукт не разворачивает и не обрабатывает. Ничего не создано.",
        "en": "This frame is classed as a personal document (vlm.exclude_classes): the "
              "product neither enlarges nor processes those. Nothing was created.",
        "ja": "このコマは個人的な書類 (vlm.exclude_classes) に分類されています。"
              "拡大も処理も行いません。何も作成されていません。",
    },
    "review_restore_error_video": {
        "ru": "Это видео, а модель работает с изображениями. Ничего не создано.",
        "en": "This is a video and the model works on images. Nothing was created.",
        "ja": "これは動画で、モデルは画像を扱います。何も作成されていません。",
    },
    "review_restore_error_model_unavailable": {
        "ru": "Модель не загрузилась. Веса качаются из сети и нужен дополнительный "
              "набор пакетов ([vlm]); офлайн и без скачанных весов эта кнопка работать "
              "не будет. Ничего не создано.",
        "en": "The model did not load. The weights come from the network and need the "
              "extra package set ([vlm]); offline and without cached weights this button "
              "cannot work. Nothing was created.",
        "ja": "モデルを読み込めませんでした。重みはネットワークから取得され、追加の"
              "パッケージ ([vlm]) が必要です。オフラインで重みが未取得の場合、この"
              "ボタンは動作しません。何も作成されていません。",
    },
    "review_restore_error_decode_failed": {
        "ru": "Кадр не читается — обрабатывать нечего. Ничего не создано.",
        "en": "The frame will not read — there is nothing to process. Nothing was created.",
        "ja": "このコマを読み込めないため、処理できません。何も作成されていません。",
    },
    "review_restore_error_write_failed": {
        "ru": "Копию не удалось записать рядом с оригиналом. Оригинал не изменён.",
        "en": "The copy could not be written beside the original. The original is unchanged.",
        "ja": "元の写真の隣に複製を書き込めませんでした。元の写真は変更されていません。",
    },
    "review_select_label": {"ru": "выбрать", "en": "select", "ja": "選択"},
    "review_select_all": {"ru": "Выбрать всё на странице",
                          "en": "Select everything on this page",
                          "ja": "このページをすべて選択"},
    "review_select_none": {"ru": "Снять выделение", "en": "Clear the selection",
                           "ja": "選択を解除"},
    "review_marked_status": {
        "ru": "Отмечено кадров: {n}", "en": "Frames marked: {n}", "ja": "印を付けたコマ: {n}",
    },
    "review_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "review_load_more_beyond": {
        "ru": "Показать за пределами окна", "en": "Show past the window",
        "ja": "表示範囲の先も表示",
    },
    "review_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    # F157: the counter of a ranking says how long the LIST is and never how many blurred
    # frames there are. "Showing 2 210" read as "you have 2 210 blurred photographs" —
    # a claim the signal cannot make (four of five frames on that page are not blurred),
    # and one that grows or shrinks the moment somebody edits a number in the config.
    "review_shown_ranked": {
        "ru": "Показано {shown}; дальше по списку резкость растёт",
        "en": "Showing {shown}; further down the list the sharpness grows",
        "ja": "{shown} 件を表示中。リストの先へ進むほど鮮鋭度は上がります",
    },
    "review_error_prefix": {
        "ru": "Не удалось сохранить отметку: ", "en": "Could not save the mark: ",
        "ja": "印を保存できません: ",
    },
    "error_loading_review": {
        "ru": "Не удалось загрузить разбор: ", "en": "Could not load the review: ",
        "ja": "仕分けを読み込めません: ",
    },
    # --- F108: the "Overview" tab ---------------------------------------------------
    "tab_overview": {"ru": "Обзор", "en": "Overview", "ja": "概要"},
    # F145: the caption over the SAME rows of counters, drawn with dashes. It replaced an
    # invitation with a button, which was a block of a different height: it was swapped
    # for the full one in the middle of a run, right after the `index` stage, and
    # everything below — the run options among them — jumped down the page.
    "overview_empty": {
        "ru": "Данных пока нет: ниже — то, что появится после прогона. "
              "Укажите папку с фото и нажмите «Обработать».",
        "en": "No data yet: below is what shows up after a run. Enter a photo folder "
              "and click Process.",
        "ja": "まだデータがありません。以下は処理後に表示される項目です。"
              "写真フォルダを指定して「処理する」を押してください。",
    },
    "overview_group_collection": {"ru": "Коллекция", "en": "Collection", "ja": "コレクション"},
    "overview_group_place": {"ru": "Место", "en": "Place", "ja": "場所"},
    "overview_group_classes": {"ru": "Разбор", "en": "Classification", "ja": "分類"},
    "overview_group_layout": {"ru": "Раскладка", "en": "Layout", "ja": "振り分け"},
    "overview_files": {"ru": "Файлов в индексе", "en": "Files in the index",
                       "ja": "インデックス内のファイル"},
    "overview_photos": {"ru": "Фото", "en": "Photos", "ja": "写真"},
    "overview_videos": {"ru": "Видео", "en": "Videos", "ja": "動画"},
    "overview_duplicates": {"ru": "Дубликатов", "en": "Duplicates", "ja": "重複"},
    "overview_errors": {"ru": "Ошибок чтения", "en": "Read errors", "ja": "読み込みエラー"},
    "overview_events": {"ru": "Событий", "en": "Events", "ja": "イベント"},
    "overview_animals": {"ru": "С животными", "en": "With animals", "ja": "動物あり"},
    # F152: the three face slices. They are the only rows of this card that can show a
    # dash instead of a number — without a faces run they are unmeasured, not empty.
    "overview_with_people": {"ru": "С людьми", "en": "With people", "ja": "人物あり"},
    "overview_group_photos": {"ru": "Групповых", "en": "Group photos", "ja": "集合写真"},
    "overview_portraits": {"ru": "Портретов", "en": "Portraits", "ja": "ポートレート"},
    # F126: the review slices that have a number of their own. Blurred is counted
    # inside the window the list opens to, so the row and the list agree.
    "overview_blurred": {"ru": "Размытых", "en": "Blurred", "ja": "ぼやけ"},
    "overview_eyes_closed": {"ru": "С закрытыми глазами", "en": "With closed eyes",
                             "ja": "目を閉じた"},
    # F150: counted under `features.low_resolution_mp`, the same ceiling the slice lists.
    "overview_low_resolution": {"ru": "Низкого разрешения", "en": "Low resolution",
                                "ja": "低解像度"},
    "overview_place_exact_gps": {"ru": "Точный GPS", "en": "Exact GPS", "ja": "正確なGPS"},
    "overview_place_manual": {"ru": "Указано вручную", "en": "Set by hand", "ja": "手動指定"},
    "overview_place_session_inferred": {
        "ru": "Унаследовано от съёмки", "en": "Inherited from the session",
        "ja": "撮影セッションから継承",
    },
    "overview_place_trip_inferred": {
        "ru": "Унаследовано от поездки", "en": "Inherited from the trip",
        "ja": "旅行から継承",
    },
    "overview_place_path_inferred": {
        "ru": "Унаследовано от имени папки", "en": "Inherited from the folder name",
        "ja": "フォルダ名から継承",
    },
    "overview_place_visual": {
        "ru": "Определено по кадру", "en": "Recognised from the frame", "ja": "画像から判定",
    },
    "overview_no_place": {
        "ru": "Без места вообще", "en": "No place at all", "ja": "場所が全く不明",
    },
    "overview_no_place_hint": {
        "ru": "Эти кадры уедут в «_Без места».",
        "en": "These frames end up in the “no place” folder.",
        "ja": "これらは「場所なし」フォルダーに入ります。",
    },
    "overview_classified": {
        "ru": "Разобрано кадров", "en": "Frames classified", "ja": "分類済みフレーム",
    },
    "overview_verdict_photo": {
        "ru": "Личные фото", "en": "Personal photos", "ja": "個人写真",
    },
    "overview_by_source": {"ru": "Чем решено", "en": "Decided by", "ja": "判定の根拠"},
    "overview_by_tier": {"ru": "Каким ярусом", "en": "Tier that handled it",
                         "ja": "処理したティア"},
    "overview_source_heuristic": {"ru": "Эвристика", "en": "Heuristics", "ja": "ヒューリスティック"},
    "overview_source_clip": {"ru": "CLIP", "en": "CLIP", "ja": "CLIP"},
    "overview_source_ocr": {"ru": "OCR", "en": "OCR", "ja": "OCR"},
    "overview_source_vlm": {"ru": "VLM", "en": "VLM", "ja": "VLM"},
    "overview_tier_heuristic": {"ru": "Быстрый (эвристика)", "en": "Fast (heuristics)",
                                "ja": "高速（ヒューリスティック）"},
    "overview_tier_clip": {"ru": "Быстрый (CLIP)", "en": "Fast (CLIP)", "ja": "高速（CLIP）"},
    "overview_tier_vlm": {"ru": "Глубокий (VLM)", "en": "Deep (VLM)", "ja": "詳細（VLM）"},
    "overview_tier_none": {"ru": "Ярус не записан", "en": "Tier not recorded",
                           "ja": "ティア未記録"},
    "overview_vlm_ran": {
        "ru": "Глубокий ярус (VLM) прогонялся.",
        "en": "The deep tier (VLM) has run.",
        "ja": "詳細ティア（VLM）は実行済みです。",
    },
    "overview_vlm_not_ran": {
        "ru": "Глубокий ярус (VLM) не прогонялся.",
        "en": "The deep tier (VLM) has not run.",
        "ja": "詳細ティア（VLM）は未実行です。",
    },
    "overview_updated_at": {
        "ru": "Последнее изменение разбора: {at}",
        "en": "Classification last changed: {at}",
        "ja": "分類の最終更新: {at}",
    },
    "overview_not_classified": {
        "ru": "Разбор ещё не запускался.", "en": "The classifier has not run yet.",
        "ja": "分類はまだ実行されていません。",
    },
    "overview_layout_none": {
        "ru": "Раскладка ещё не запускалась — файлы лежат там же, где лежали.",
        "en": "No layout has run yet — the files are still where they were.",
        "ja": "まだ振り分けは実行されていません。ファイルは元の場所のままです。",
    },
    "overview_layout_batches": {"ru": "Раскладок было", "en": "Layout runs",
                                "ja": "振り分けの回数"},
    "overview_layout_started": {"ru": "Начата", "en": "Started", "ja": "開始"},
    "overview_layout_finished": {"ru": "Завершена", "en": "Finished", "ja": "完了"},
    "overview_layout_dest": {"ru": "Куда", "en": "Destination", "ja": "振り分け先"},
    "overview_layout_mode": {"ru": "Режим", "en": "Mode", "ja": "モード"},
    "overview_layout_files": {"ru": "Файлов в раскладке", "en": "Files in the batch",
                              "ja": "バッチ内のファイル"},
    "overview_layout_done": {"ru": "Из них перенесено", "en": "Of them moved",
                             "ja": "うち移動済み"},
    "overview_layout_unfinished": {
        "ru": "Батч не закрыт — прогон был прерван.",
        "en": "The batch is not closed — the run was interrupted.",
        "ja": "バッチが閉じられていません。実行が中断されました。",
    },
    "overview_op_move": {"ru": "перенос", "en": "move", "ja": "移動"},
    "overview_op_copy": {"ru": "копия", "en": "copy", "ja": "コピー"},
    "overview_goto_hint": {
        "ru": "Открыть вкладку «{tab}»", "en": "Open the {tab} tab", "ja": "「{tab}」タブを開く",
    },
    "error_loading_overview": {
        "ru": "Не удалось загрузить обзор: ", "en": "Could not load the overview: ",
        "ja": "概要を読み込めません: ",
    },
    # --- F133: the "Slices" tab, the layout warning and the settings drawer -----------
    "slices_intro": {
        "ru": "Срез — это подборка поверх канона: кадры с людьми, групповые, портреты, "
              "имена, события, животные, товары, скриншоты, документы. Альбом среза — "
              "жёсткие ссылки, их можно собрать и удалить сколько угодно раз.",
        "en": "A slice is a selection on top of the canon: frames with people, group "
              "photos, portraits, names, events, animals, products, screenshots, "
              "documents. An album of a slice is hardlinks — gather it and drop it as "
              "often as you like.",
        "ja": "スライスは正本の上に重ねる抽出です（人物あり・集合写真・ポートレート・"
              "名前・イベント・動物・商品・スクリーンショット・書類）。スライスの"
              "アルバムはハードリンクなので、何度でも作成・削除できます。",
    },
    # --- F134: the search line itself. The place F133 reserved is wired now, so the
    # placeholder names what actually goes in it — words, not the name of a slice.
    "search_placeholder": {
        "ru": "Найти словами: торт, снег, море…",
        "en": "Search by words: cake, snow, the sea…",
        "ja": "言葉で検索: ケーキ、雪、海…",
    },
    "search_button": {"ru": "Найти", "en": "Search", "ja": "検索"},
    # Shown until the first answer about the index arrives. Not "search is unavailable":
    # the state is not known yet, and guessing it in either direction is a lie that lasts
    # exactly as long as the request.
    "search_state_checking": {
        "ru": "Проверяем индекс поиска…", "en": "Checking the search index…",
        "ja": "検索インデックスを確認しています…",
    },
    # THE state of this feature: nothing was ever encoded. An empty result list would read
    # as "you have no photographs like that", which is a conclusion about somebody's own
    # archive drawn from a table that was never filled.
    # F141 corrected this sentence: the index is no longer a by-product of an ordinary
    # run. It is a second CLIP pass with a multilingual model, ~10.5 minutes per 20 000
    # frames, behind `features.search_index` — so the setting has to be named, or the
    # reader follows an instruction that will not fill the table.
    "search_state_empty": {
        "ru": "Искать пока не по чему: индекс поиска пуст. Включите "
              "features.search_index: true и запустите обработку коллекции — это "
              "отдельный проход CLIP многоязычной моделью (~10,5 минут на 20 000 кадров).",
        "en": "There is nothing to search yet: the search index is empty. Switch on "
              "features.search_index: true and process the collection — it is a separate "
              "CLIP pass with a multilingual model (~10.5 minutes per 20 000 frames).",
        "ja": "検索できる対象がまだありません。検索インデックスが空です。"
              "features.search_index: true を有効にしてコレクションを処理してください — "
              "多言語モデルによる別途の CLIP パスです（2 万コマあたり約 10.5 分）。",
    },
    # The other unavailable state, and deliberately a different sentence: the fix is the
    # same run, but the reason is that the stored vectors belong to another model and are
    # not comparable with this query. Mixing them silently would produce a plausible
    # ranking that nothing on screen marks as wrong.
    "search_state_other_model": {
        "ru": "Индекс поиска посчитан другой моделью ({model}): её векторы несравнимы с "
              "текущей, поэтому выдача была бы правдоподобной чушью. Нужен повторный "
              "прогон коллекции.",
        "en": "The search index was computed by another model ({model}): its vectors are "
              "not comparable with the current one, so the ranking would be plausible "
              "nonsense. The collection has to be processed again.",
        "ja": "検索インデックスは別のモデル（{model}）で作成されています。ベクトルに"
              "互換性がなく、もっともらしい誤った結果になります。コレクションを再度"
              "処理してください。",
    },
    # Available, and honest about the denominator: an incremental run is the normal way to
    # live with a growing archive, and a person must be able to tell "it is not in the
    # collection" from "it is not in the index yet".
    "search_state_partial": {
        "ru": "Ищем по {n} из {all} фотографий: остальные попадут в индекс на следующем "
              "прогоне.",
        "en": "Searching {n} of {all} photographs: the rest join the index on the next "
              "run.",
        "ja": "{all} 枚中 {n} 枚を検索対象にしています。残りは次回の処理でインデックスに"
              "追加されます。",
    },
    "search_state_ready": {
        "ru": "Ищем по всем {all} фотографиям коллекции.",
        "en": "Searching all {all} photographs of the collection.",
        "ja": "コレクションの {all} 枚すべてを検索します。",
    },
    "search_goto_overview": {
        "ru": "К «Обзору»", "en": "Go to Overview", "ja": "「概要」へ",
    },
    # No threshold exists and none will (search.py): the score orders frames against each
    # other and says nothing in absolute terms. The line says so instead of promising an
    # accuracy nobody has measured.
    "search_ranking_hint": {
        "ru": "Это ранжирование, а не фильтр: список отсортирован по близости к запросу, "
              "порога «точно оно» нет. Смотрите сверху вниз и остановитесь, где кончится "
              "похожее.",
        "en": "This is a ranking, not a filter: the list is sorted by closeness to the "
              "query and there is no “this really is it” threshold. Read top-down and "
              "stop where the resemblance runs out.",
        "ja": "これはフィルタではなくランキングです。クエリとの近さで並んでおり、"
              "「確実に該当」というしきい値はありません。上から順に見て、似ていないと"
              "感じたところで止めてください。",
    },
    "search_score_label": {
        "ru": "близость {score}", "en": "closeness {score}", "ja": "近さ {score}",
    },
    # F173: the numerator AND the denominator. The old wording ("{n} frames") was true of
    # the page and read as a fact about the collection — «200 кадров» for a query whose
    # ranking is four thousand long, with the half that matters below the fold.
    "search_shown_label": {
        "ru": "Запрос «{q}»: показано {shown} из {total}, от самого близкого",
        "en": "Query “{q}”: showing {shown} of {total}, closest first",
        "ja": "クエリ「{q}」: {total} 件中 {shown} 件を表示（近い順）",
    },
    # An available index always ranks everything it holds, so an empty list means the
    # index itself is empty of frames a search may return — never "there are no such
    # photographs".
    "search_no_frames": {
        "ru": "Ранжировать нечего: в индексе поиска нет ни одного кадра, который можно "
              "показать.",
        "en": "There is nothing to rank: the search index holds no frame that could be "
              "shown.",
        "ja": "並べ替える対象がありません。検索インデックスに表示できるコマがありません。",
    },
    "error_loading_search": {
        "ru": "Не удалось выполнить поиск: ", "en": "Could not run the search: ",
        "ja": "検索を実行できません: ",
    },
    # --- F189: the same line, answering with a person ----------------------------------
    # Said in front of the index's own reason rather than instead of it: the ranking still
    # cannot run and the way to fix that is still on screen — what changes is that the
    # field is not dead while there is somebody to find in it.
    "search_state_names_only": {
        "ru": "Имя названного человека здесь найдётся и без индекса — наберите имя.",
        "en": "The name of a person you have labelled is found here without the index — "
              "type a name.",
        "ja": "名前を付けた人物は、インデックスがなくてもここで見つかります — "
              "名前を入力してください。",
    },
    # The caption is the feature as much as the selection is. A reader who cannot tell an
    # exact answer from the top of a ranking has been handed one thing and shown another,
    # so this sentence says what it is and the ranking's sentence stays where it was.
    "search_person_shown_label": {
        "ru": "Кадры человека: {name} — показано {shown} из {total}",
        "en": "Frames of a person: {name} — showing {shown} of {total}",
        "ja": "人物のコマ: {name} — {total} 件中 {shown} 件を表示",
    },
    "search_person_hint": {
        "ru": "Это точный отбор по кластеру лиц, а не ранжирование: кадр либо в кластере "
              "этого человека, либо нет. Порога и «похожести» здесь нет, список полный — "
              "он лишь показывается по частям.",
        "en": "This is an exact selection by face cluster, not a ranking: a frame is "
              "either in this person's cluster or it is not. There is no threshold and no "
              "“closeness” here — the list is complete and merely shown in portions.",
        "ja": "これはランキングではなく、顔クラスタによる正確な抽出です。コマがこの人物の"
              "クラスタに入っているかどうかだけで決まります。しきい値も「近さ」もなく、"
              "一覧は完全で、分割して表示しているだけです。",
    },
    # The depth warning of a ranking does not apply to a list: the next page is more of the
    # same fact, not a worse guess.
    "search_person_more_hint": {
        "ru": "Дальше — продолжение того же списка: кадры не становятся менее «точными».",
        "en": "Further on is the same list continued: the frames do not get less certain.",
        "ja": "この先も同じ一覧の続きです。コマの確かさが下がることはありません。",
    },
    # Requirement 4 on screen: a name can be an ordinary word («Роза», «Марк»), and the
    # other answer is one click away instead of gone.
    "search_person_words_link": {
        "ru": "Искать «{q}» по картинке",
        "en": "Search for “{q}” as an image",
        "ja": "「{q}」を画像として検索",
    },
    "search_words_person_link": {
        "ru": "Показать кадры человека: {name}",
        "en": "Show the frames of a person: {name}",
        "ja": "人物のコマを表示: {name}",
    },
    # A named cluster all of whose frames are duplicates or unreadable. Rare, and still not
    # "nothing was found": the person exists, the frames a search may show do not.
    "search_person_no_frames": {
        "ru": "У этого человека нет кадров, которые можно показать: все они дубли или "
              "нечитаемые файлы.",
        "en": "This person has no frame that can be shown: all of them are duplicates or "
              "unreadable files.",
        "ja": "この人物には表示できるコマがありません。すべて重複か読み取れない"
              "ファイルです。",
    },
    # --- F152: the three face slices ---------------------------------------------------
    # The labels are deliberately not the label of the cluster slice next to them: "Люди"
    # there answers "who is this", these answer "is anybody in the frame".
    "face_slice_people": {"ru": "С людьми", "en": "With people", "ja": "人物あり"},
    "face_slice_group": {"ru": "Групповые", "en": "Group photos", "ja": "集合写真"},
    "face_slice_portrait": {"ru": "Портреты", "en": "Portraits", "ja": "ポートレート"},
    # THE line that has to differ from the caption of an approximate slice. A query slice
    # is a ranking and says so; this one is a fact of the detector, and the sentence says
    # what the fact is and where its errors come from, without a percentage nobody
    # measured.
    "face_slices_intro": {
        "ru": "Эти срезы — не оценка: кадр в них потому, что детектор нашёл на нём лицо. "
              "Порога «похоже на человека» здесь нет, ошибки бывают только у самого "
              "детектора. Служебная отметка «файл обработан, лиц нет» исключена везде.",
        "en": "These slices are not an estimate: a frame is here because the detector "
              "found a face on it. There is no “looks like a person” threshold — the only "
              "errors are the detector's own. The “processed, no faces” marker row is "
              "excluded everywhere.",
        "ja": "これらのスライスは推定ではありません。検出器がその写真で顔を見つけたから"
              "入っています。「人物らしさ」のしきい値はなく、誤りは検出器そのものの誤り"
              "だけです。「処理済み・顔なし」の内部記録はすべて除外されます。",
    },
    "face_hint_people": {
        "ru": "Хотя бы одно лицо в кадре.",
        "en": "At least one face in the frame.",
        "ja": "写真に顔が 1 つ以上あります。",
    },
    "face_hint_group": {
        "ru": "Лиц в кадре — {n} и больше (features.group_photo_faces).",
        "en": "{n} faces or more in the frame (features.group_photo_faces).",
        "ja": "顔が {n} 個以上（features.group_photo_faces）。",
    },
    "face_hint_portrait": {
        "ru": "Ровно одно лицо, и оно занимает не меньше {share}% кадра "
              "(features.portrait_face_share).",
        "en": "Exactly one face, covering at least {share}% of the frame "
              "(features.portrait_face_share).",
        "ja": "顔がちょうど 1 つで、写真の {share}% 以上を占めます"
              "（features.portrait_face_share）。",
    },
    # F125's rule: the reason, never a zero. Without a faces run nothing was measured, and
    # "0 photographs with people" is a statement about somebody's archive that no table
    # in this index supports.
    "face_no_faces_run": {
        "ru": "Стадия «лица» не запускалась — считать нечего. Запустите обработку с "
              "галочкой «Разбор по лицам», и срезы наполнятся сами.",
        "en": "The faces stage has not run — there is nothing to count yet. Process the "
              "collection with “Detect faces” ticked and these slices fill in by "
              "themselves.",
        "ja": "顔の処理がまだ実行されていないため、集計できません。「顔の検出」を"
              "有効にして処理すると、これらのスライスが表示されます。",
    },
    "face_empty": {
        "ru": "В этом срезе пусто: таких кадров не нашлось.",
        "en": "This slice is empty — no such frames were found.",
        "ja": "このスライスは空です。該当するコマは見つかりませんでした。",
    },
    "face_count_label": {
        "ru": "лиц: {n}", "en": "{n} faces", "ja": "顔 {n}",
    },
    # F173: the shared pager's button and counter here too. What this slice does NOT take
    # from it is `slice_depth_hint` — nothing is ranked here (a frame is in the slice
    # because the detector found a face), so there is no precision to trade for depth and
    # a line saying otherwise would be a warning about a risk this list does not carry.
    "error_loading_face_slices": {
        "ru": "Не удалось загрузить срезы по лицам: ",
        "en": "Could not load the face slices: ",
        "ja": "顔のスライスを読み込めません: ",
    },
    # --- F151: the pinned queries ------------------------------------------------------
    # The labels of the three slices `features.saved_slices` ships with. A name that is not
    # in this catalog is shown as it stands in the config — the row must not refuse to draw
    # a slice somebody added, and a made-up translation would be worse than the key itself.
    "query_slice_children": {"ru": "Дети", "en": "Children", "ja": "子ども"},
    "query_slice_products": {"ru": "Товары", "en": "Products", "ja": "商品"},
    "query_slice_animals": {"ru": "Животные", "en": "Animals", "ja": "動物"},
    # THE caption rule of this feature. «Животные» the pin and «Животные» the pet label are
    # two different slices of one archive — 60% precision against 71%, a ranking against a
    # verdict a model checked — and with the same label a reader would take the estimate for
    # the fact. So every pinned query wears the mark, including the ones with no exact
    # counterpart: what is marked is the METHOD, not the collision.
    "query_slice_pin": {
        "ru": "{name} · по запросу", "en": "{name} · by query", "ja": "{name}・クエリ",
    },
    "query_slice_intro": {
        "ru": "Это оценка, а не метка: срез собран запросом к тем же векторам, и ни одна "
              "модель его не проверяла. Порога «точно оно» здесь нет — список идёт от "
              "самого близкого, и где он перестаёт быть про запрос, решаете вы. На "
              "размеченной выборке из 200 кадров такой срез находит около 60% нужного в "
              "первой порции и около 90% в удвоенной, поэтому «Показать ещё» здесь — "
              "главная кнопка, а не украшение.",
        "en": "This is an estimate, not a label: the slice is a query over the same "
              "vectors and no model has checked it. There is no “this really is it” "
              "threshold — the list runs from the closest down, and where it stops being "
              "about the query is yours to decide. On a hand-labelled sample of 200 "
              "frames a slice like this finds about 60% of what you are after in the "
              "first portion and about 90% in a doubled one, which is why “Show more” is "
              "the main button here rather than a decoration.",
        "ja": "これはラベルではなく推定です。同じベクトルへの問い合わせで集めた"
              "スライスであり、モデルによる確認は行われていません。「確実に該当」と"
              "いうしきい値はなく、近い順に並ぶだけなので、どこで終わりにするかは"
              "あなたが決めます。200 コマの人手ラベル付き標本では、最初の一覧で約 "
              "60%、倍の深さで約 90% を拾えます。だからこそ「さらに表示」が主役の"
              "ボタンです。",
    },
    # What the slice actually asked, on screen — the half that makes "editable without
    # code" real rather than stated. The phrases stay English whatever `language:` says:
    # they go to a CLIP text tower and not to a reader, and the measured numbers were
    # produced by this wording.
    "query_slice_phrases": {
        "ru": "Запрос среза: {phrases}. Правится в features.saved_slices; формулировки "
              "английские — язык интерфейса на выдачу не влияет.",
        "en": "The slice asks: {phrases}. Edit it in features.saved_slices; the phrases "
              "are English — the interface language does not change this list.",
        "ja": "このスライスの問い合わせ: {phrases}。features.saved_slices で編集でき"
              "ます。表現は英語です（表示言語はこの一覧に影響しません）。",
    },
    "query_slice_shown_label": {
        "ru": "Срез «{name}»: показано {shown} из {total}, от самого близкого",
        "en": "Slice “{name}”: showing {shown} of {total}, closest first",
        "ja": "スライス「{name}」: {total} 件中 {shown} 件を表示（近い順）",
    },
    "error_loading_saved_slices": {
        "ru": "Не удалось загрузить срез по запросу: ",
        "en": "Could not load the query slice: ",
        "ja": "クエリのスライスを読み込めません: ",
    },
    # --- F156: pinning a query of one's own --------------------------------------------
    # The product stops guessing which facets matter (the sample of 200 says there is no
    # such thing as "the" facets: ten candidate slices covered 26% of the unclassed frames
    # at best), so these strings are all about one act — a person saving THEIR query.
    "pin_slice_button": {
        "ru": "Закрепить как срез", "en": "Pin as a slice", "ja": "スライスとして固定",
    },
    # The name is asked for, with the query itself offered: the query is usually the best
    # name there is, and a dialog that demands a different one is a dialog that gets «мое1».
    "pin_slice_prompt": {
        "ru": "Название среза (запрос: {query})",
        "en": "Name of the slice (the query: {query})",
        "ja": "スライスの名前（クエリ: {query}）",
    },
    # THE warning of this feature, and it is said BEFORE the pin rather than afterwards.
    # The phrases go to the model as they stand and the search index is English until F141
    # reaches this collection, so a Russian or Japanese pin will rank badly — a person who
    # learns that a week later concludes the feature is broken.
    "pin_slice_language_warning": {
        "ru": "Запрос не на английском. Индекс пока английский, поэтому такой срез будет "
              "работать заметно хуже — формулировка уходит в модель как есть.",
        "en": "The query is not in English. The index is English for now, so a slice like "
              "this will work noticeably worse — the wording goes to the model as it is.",
        "ja": "クエリが英語ではありません。索引は現時点で英語なので、このスライスの精度は"
              "目に見えて落ちます（表現はそのままモデルに渡されます）。",
    },
    "pin_slice_done": {
        "ru": "Срез «{name}» закреплён.", "en": "The slice “{name}” is pinned.",
        "ja": "スライス「{name}」を固定しました。",
    },
    # Every refusal is a sentence, never a button that does nothing.
    "pin_error_empty": {
        "ru": "Пустой запрос закрепить нельзя.",
        "en": "An empty query cannot be pinned.",
        "ja": "空のクエリは固定できません。",
    },
    "pin_error_duplicate": {
        "ru": "Срез с таким названием уже закреплён.",
        "en": "A slice with that name is already pinned.",
        "ja": "その名前のスライスはすでに固定されています。",
    },
    # F133's reason and not a resource one — and the number is in the sentence, so the
    # person knows what to unpin and what to raise.
    "pin_error_limit": {
        "ru": "Закреплено {max} срезов — это предел. Открепите ненужный или поднимите "
              "features.max_pinned_slices.",
        "en": "{max} slices are pinned — that is the limit. Unpin one, or raise "
              "features.max_pinned_slices.",
        "ja": "固定できるスライスは {max} 件までです。不要なものを外すか、"
              "features.max_pinned_slices を増やしてください。",
    },
    "pin_error_generic": {
        "ru": "Не удалось закрепить срез: ", "en": "Could not pin the slice: ",
        "ja": "スライスを固定できません: ",
    },
    "pin_unpin_button": {
        "ru": "Открепить срез", "en": "Unpin the slice", "ja": "スライスを外す",
    },
    # The confirmation says what is removed AND what is not: "delete the slice" and
    # "delete the photographs" are one word apart, and only one of them is happening.
    "pin_unpin_confirm": {
        "ru": "Открепить срез «{name}»? Удалится только закрепление — файлы останутся "
              "на месте.",
        "en": "Unpin the slice “{name}”? Only the pin is removed — the files stay where "
              "they are.",
        "ja": "スライス「{name}」を外しますか？外れるのは固定だけで、ファイルはそのまま"
              "残ります。",
    },
    "pin_move_up": {"ru": "Выше", "en": "Move up", "ja": "上へ"},
    "pin_move_down": {"ru": "Ниже", "en": "Move down", "ja": "下へ"},
    # The album gathers what a single query ranks (`sorta album query`), and a slice asking
    # several phrases is ranked by their average — one selector cannot reproduce it, so the
    # button is not offered rather than gathering a different list under the same name.
    "pin_album_one_query": {
        "ru": "Альбом собирается по одной формулировке, а этот срез спрашивает несколько. "
              "Оставьте в features.saved_slices одну — и кнопка появится.",
        "en": "An album is gathered by a single wording, and this slice asks several. "
              "Leave one of them in features.saved_slices and the button appears.",
        "ja": "アルバムは 1 つの表現でまとめます。このスライスは複数を問い合わせている"
              "ため、features.saved_slices に 1 つだけ残すとボタンが表示されます。",
    },
    # --- F156: why a built-in slice is empty -------------------------------------------
    # The `frame_quality` rule of F125, said out loud on a whole slice: a zero with no
    # explanation reads as "there are none of these in your archive", and far more often
    # the truth is that nobody has looked yet. The counterpart answer — "it was computed
    # and there is nothing" — is the slice's own existing line ("События не найдены."),
    # which is why only this half needed writing.
    "slice_not_computed": {
        "ru": "Это не считалось: стадия, которая наполняет срез, не запускалась. "
              "Пусто здесь означает «не спрашивали», а не «в архиве нет».",
        "en": "This was not computed: the stage that fills this slice has not run. Empty "
              "here means “nobody asked”, not “there are none”.",
        "ja": "これは計算されていません。このスライスを埋める処理が実行されていません。"
              "ここでの空は「該当なし」ではなく「未確認」という意味です。",
    },
    "slice_goto_process": {
        "ru": "К экрану прогона", "en": "Go to the run screen", "ja": "実行画面へ",
    },
    "slices_pinned_label": {
        "ru": "Закреплённые срезы", "en": "Pinned slices", "ja": "固定スライス",
    },
    "slices_empty": {
        "ru": "Срезов пока нет: обработайте коллекцию — люди, события, животные и "
              "классы появятся здесь.",
        "en": "No slices yet: process the collection — people, events, animals and "
              "classes show up here.",
        "ja": "スライスはまだありません。コレクションを処理すると、人物・イベント・"
              "動物・分類がここに表示されます。",
    },
    "layout_review_warning": {
        "ru": "В «Разборе» осталось без решения: {n}. Отмеченные к удалению уезжают в "
              "«_delete» во время раскладки, а альбомы — ссылки из канона: собрав их "
              "раньше, вы получите ссылки на выброшенное. Раскладку это не запрещает.",
        "en": "The Review still holds {n} undecided. Frames marked for deletion leave "
              "for “_delete” during the layout, and albums are links out of the canon: "
              "gather them earlier and you get links to what you threw away. This does "
              "not block the layout.",
        "ja": "「仕分け」に未決定が {n} 件残っています。削除指定のコマは振り分けの際に"
              "「_delete」へ移動し、アルバムは正本からのリンクです。先にアルバムを作ると"
              "捨てたものへのリンクが残ります。振り分け自体は禁止されません。",
    },
    "layout_review_goto": {
        "ru": "К «Разбору»", "en": "Go to Review", "ja": "「仕分け」へ",
    },
    "settings_open_button": {
        "ru": "Настройки", "en": "Settings", "ja": "設定",
    },
    "settings_close_button": {
        "ru": "Закрыть", "en": "Close", "ja": "閉じる",
    },
}


def _t(key: str, lang: i18n.Lang) -> str:
    """Resolve a chrome UI string: exact language -> en -> the key itself (see F33)."""
    entry = _UI_STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


# F182: the page, its stylesheet and its script live in `sorta/web/` as the files they
# are, not as a 6 100-line string literal in the middle of the Python. `page.html` keeps
# two seams — `{{style}}` and `{{script}}` — and they are filled once, here, at import:
# what the server holds afterwards is the same template as before, byte for byte.
_WEB_DIR = Path(__file__).resolve().parent / "web"


def _read_web(root: Path, *parts: str) -> str:
    r"""Read one frontend file of `sorta/web/`.

    Text mode on purpose. The template is assembled with "\n" throughout, and a
    checkout that materialises these files with CRLF (the Windows default) must not
    change a single byte of what is served — universal newlines make that impossible.
    """
    return root.joinpath(*parts).read_text(encoding="utf-8")


def _load_index_template(root: Path | None = None) -> str:
    """Put the three files back together into the template `_render_index_html` fills.

    `root` is for the tests only — the server always assembles from `sorta/web/`.
    """
    web = _WEB_DIR if root is None else root
    return (_read_web(web, "page.html")
            .replace("{{style}}", _read_web(web, "style.css"))
            .replace("{{script}}", _read_web(web, "app", "app.js")))


_INDEX_HTML_TEMPLATE = _load_index_template()


def _render_index_html(lang: i18n.Lang) -> str:
    """Fills the chrome `{{key}}` placeholders and the `window.I18N` JSON (F33).

    Placeholders are literal `{{...}}` tokens, replaced via `str.replace` (not
    `.format`): the CSS/JS in the template is full of single `{`/`}`, which `.format`
    would interpret as substitution fields.
    """
    i18n_map = {key: _t(key, lang) for key in _UI_STRINGS}
    lang_options = "".join(
        f'<option value="{code}"{" selected" if code == lang else ""}>{name}</option>'
        for code, name in _LANG_SELF_NAMES.items()
    )
    html = _INDEX_HTML_TEMPLATE.replace("{{lang}}", lang)
    html = html.replace("{{lang_options}}", lang_options)
    # F80: how many frames the lightbox may page through. The real strip of a short
    # clip can be shorter — the pager finds that out from the first 404 and clamps.
    html = html.replace("{{video_frames}}", str(imaging.video_frames()))
    html = html.replace("{{i18n_json}}", json.dumps(i18n_map, ensure_ascii=False))
    for key, value in i18n_map.items():
        html = html.replace("{{" + key + "}}", value)
    return html


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
            elif path == "/api/sort/status":
                self._serve_sort_status()
            elif path == "/api/undo/status":
                self._send_json(undo_state.snapshot())
            elif path == "/api/sort/suggest-dest":
                self._send_json({"dest": _suggested_sort_dest(cfg, db_path)})
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
            self._send_json({"query": raw.strip(),
                             "results": _places_search(raw, lang)})

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
            trashed = _trash_group(db_path, group, keep)
            self._send_json({"trashed": trashed})

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
            trashed = _trash_files(db_path, [file_id])
            self._send_json({"trashed": trashed})

        def _handle_photos_trash(self) -> None:
            # bulk deletion of the selected (the shared _trash_files path, same as single)
            ids = _validate_file_ids_payload(self._read_json_body())
            if ids is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            trashed = _trash_files(db_path, ids)
            self._send_json({"trashed": trashed})

        def _handle_overrides(self) -> None:
            # F77: marks only — nothing is moved on disk here (the physical move is the
            # shared sort --apply). The plan cache is deliberately NOT invalidated: the
            # mark is served live by PlanCache, and a rebuild per click would cost the
            # whole mode (F70).
            parsed = _validate_overrides_payload(self._read_json_body())
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
            parsed = _validate_album_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            kind, selector, mode, where, name, apply_, dest_str = parsed
            # F139/F133: a sensitive class has no album, and the refusal lives here
            # rather than in the markup — a button the page does not draw is not a rule,
            # and a request sent past the interface would gather the folder all the same.
            # `plan_album` refuses it a second time, for the terminal; this end answers
            # with a status instead of a traceback. The settings panel can change
            # `vlm.exclude_classes` without a restart, so the key is read per request.
            if kind in CLASS_ALBUM_KINDS and kind in frozenset(cfg.vlm.exclude_classes):
                self._send_json({"error": "sensitive class"},
                                status=HTTPStatus.FORBIDDEN)
                return
            dest = Path(dest_str) if dest_str else _album_dest(cfg, db_path)
            conn = _connect(db_path)
            try:
                # F134: `encoder` is the server's own text tower and is ignored by every
                # kind but `query` — without it `plan_album` would load a second copy of
                # CLIP for an album the search line has already ranked.
                report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                    where=where, apply=apply_, album_name=name,
                                    encoder=query_encoder)
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
            if not Path(source_dir).is_dir():
                self._send_json({"error": "not a directory"}, status=HTTPStatus.BAD_REQUEST)
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
                _apply_settings(cfg, values)
                if config_path is not None:
                    try:
                        for key, value in values.items():
                            save_setting(config_path, key, value)  # type: ignore[arg-type]
                    except OSError as exc:
                        self._send_json({"error": f"could not save config: {exc}"},
                                        status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
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
            self._send_json({"path": _browse_for_folder()})

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
            payload = cache.summary("city", _summary_dest(cfg, dest or None))
            if payload is None:  # only an unsupported mode, which "city" is not
                self._send_json({"error": "no plan"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)

        def _handle_sort_start(self) -> None:
            parsed = _validate_sort_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            dest, mode = parsed
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
                target=_run_sort, args=(db_path, cfg, dest, mode, sort_state, cache),
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
    """Start the local read-only plan server and block until Ctrl+C.

    127.0.0.1 only. A busy port -> RuntimeError with a clear message (the caller
    cli.py decides how to show it to the user). `config_path` is threaded to the
    server so the folder-language selector can persist into config.yaml.
    """
    log_environment()  # F69: one environment header per server start
    warn_if_geo_data_missing()  # F65: an unreadable geo base empties every place
    try:
        httpd = build_server(cfg, conn, port=port, config_path=config_path)
    except OSError as exc:
        raise RuntimeError(f"sorta ui: порт {port} занят или недоступен: {exc}") from exc
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
