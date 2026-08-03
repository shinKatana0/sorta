"""U1/U3/U4/F31/F32/F35/F36: a local web server — a live sort-plan report +
Duplicates (incl. batch saving) + deleting a single frame + a "People" tab (managing
face clusters) + person/event albums ("Collect into folder", on top of the F34
engine) + the "Process" entry point — running the pipeline
index→geo→landmarks→faces→events→junk→phash from the web, on a background server thread.

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
index→geo→landmarks→faces→events→junk→phash (the leaf functions indexer/geo/
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
column. `GET /api/process/estimate` prices every line of that screen: a measured rate
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

(16) `GET /api/junk` (F103, the "Not personal photos" tab) — the buckets the classifier
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
the "Duplicates" tab) — the four things a person looks at in order to decide what stays:
near-duplicates, blurred frames, closed eyes, frames with no subject. One workspace with
four SLICES rather than four tabs, because it is one job. Duplicates are the only GROUPED
slice and keep their own route and their own rendering untouched — `/api/dupes` and the
four write routes above answer exactly as they did, since that is the one path in the
product that deletes files and the one that has been run against a live collection. The
GET carries the counters of all four slices (a slice with nothing in it stays in the
switcher with a zero — an empty slice is an answer, a missing one is a riddle) plus one
bounded page of the current flat slice, over photographs only (`media_class.verdict =
'photo'`, F120) that are canonical and readable. The blurred list is ordered by ascending
sharpness and opens as far as `features.blur_review_max`; `beyond=1` continues past that
window, which is a prefix of the same ordering, so nothing is lost or repeated at the
seam. Without a faces run the eyes slice answers `eyes_reason='no_faces_run'` rather than
a zero (F125: the question is only asked where a face was found). The POST writes the
decision into the EXISTING `dedup_choice` (`keep`/`to_delete`, or `clear` to drop the
row) — `file_id` is its primary key, so a frame that appears in two slices carries one
decision, and `to_delete` is already understood by the sorter. There is deliberately no
route that marks a whole slice at once: reviewed by eye, blurred frames turn up in every
band up to 400, so sharpness ranks the list and a person decides each frame.

(22) `GET /api/search` (F134, the query line of the "Slices" tab) — the F129 engine
behind the field F133 drew and left disabled: `q` is the words, `limit` a SAMPLE SIZE
(`features.search_limit` by default, clamped, never a similarity threshold — there is
none and there will not be one), and the answer is the ranking as cards with a score on
each. Every answer also carries the STATE of the index — `state` (empty / other_model /
partial / ready), `available`, `indexed`, `total` and `index_model` — because the failure
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

from . import db, faces, i18n, imaging
from .config import (
    VLM_QUALITY_SCOPES,
    Config,
    FeaturesConfig,
    save_language,
    save_setting,
)
from .dedup import assign_duplicates, compute_phashes, near_duplicate_groups
from .diagnostics import warn_if_geo_data_missing
from .events import build_events
from .faces import detect_and_cluster
from .geo import clear_geo_cache, geo_cache_size, resolve_places
from .geodata import GeoDataMissing, GeoResolver
from .indexer import excludes_path, index as run_index, load_excludes, normalize_exclude
from .indexer import save_excludes as save_excludes_file
from .junk import classify as classify_junk
from .junk import (
    faces_stage_ran,
    quality_scope_ids,
    search_index_model,
    search_index_settings,
)
from .landmarks import Classifier, clip_classifier, detect_landmarks
from .landmarks import batched
from .naming import name_events, naming_settings
from .runlog import log_environment, stage_timer
from .search import (
    REASON_EMPTY,
    REASON_OTHER_MODEL,
    EmbeddingsMissing,
    TextEncoder,
    search_text,
    text_encoder,
)
from .sorter import (
    ALBUM_KINDS,
    ALBUM_MODES,
    CLASS_ALBUM_KINDS,
    FACE_SLICES,
    QUALITY_FROM,
    SELECTORLESS_ALBUM_KINDS,
    AlbumReport,
    PlanItem,
    animal_auto_sql,
    animal_ids_sql,
    face_slice_ids_sql,
    plan_album,
    plan_and_sort,
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


def _parse_page_window(query: dict[str, list[str]]) -> tuple[int, int] | None:
    """(offset, limit) for a `/api/plan` category page, or None -> 400.

    A missing parameter falls back to the default; a non-integer or negative one is
    rejected rather than coerced — the one outcome that must never happen is quietly
    serving the whole category. A limit above the maximum is clamped, not rejected:
    an over-eager client gets less data, not an error.
    """
    raw_offset = (query.get("offset") or ["0"])[0]
    raw_limit = (query.get("limit") or [str(_PLAN_PAGE_DEFAULT_LIMIT)])[0]
    try:
        offset, limit = int(raw_offset), int(raw_limit)
    except ValueError:
        return None
    if offset < 0 or limit < 0:
        return None
    return offset, min(limit, _PLAN_PAGE_MAX_LIMIT)


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
        best["recommended"] = True
        result.append({"group": idx, "frames": frames,
                       # Why this one — so the tab can say it instead of asking the user
                       # to trust a star.
                       "recommended_by": "sharpness" if by_sharpness else "resolution"})
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


# --- F103: the "Not personal photos" view -------------------------------------------
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
                       no_preview: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """One card of the junk view. `thumb_url` is ABSENT for a no-preview bucket."""
    path = Path(row["path"])
    verdict = row["verdict"]
    payload = {
        "file_id": int(row["id"]),
        "verdict": verdict,
        "name": path.name,
        "date": row["taken_at"],
        # F77/F103: the frame already carries a manual "this is a photo" correction —
        # the card says so instead of offering the same action twice.
        "restored": restored,
    }
    if verdict not in no_preview:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _junk_payload(db_path: Path, bucket: str | None,
                  offset: int, limit: int,
                  sensitive: frozenset[str] = frozenset(_JUNK_NO_PREVIEW)) -> dict:
    """`GET /api/junk` — the buckets with their counts + one page of one bucket.

    F133: `sensitive` is `vlm.exclude_classes` — the config list that already means
    "handle this class as private", and whose default is `["document"]`. A class in it
    keeps its COUNTER and loses its CONTENT: no paths, no rows, and therefore no
    thumbnails, because a thumbnail is fetched by a path this route hands out. The guard
    lives here rather than in the markup for exactly that reason — hiding a button in
    the browser is not privacy when the data has already been sent to it.

    Reusing the VLM key instead of adding a second one is a deliberate trade: one
    visible list of sensitive classes beats two, of which the second gets forgotten.
    Emptying it therefore lifts both protections at once, which the guide says out loud.

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
        rows = conn.execute(
            f"""SELECT f.id, f.path, f.taken_at, mc.verdict
                FROM files f JOIN media_class mc ON mc.file_id = f.id
                WHERE mc.verdict <> 'photo' AND f.dup_of IS NULL AND f.error IS NULL
                      {clause}
                ORDER BY f.path
                LIMIT ? OFFSET ?""", [*params, limit, offset]).fetchall()
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
        # The client draws the counter-only state from this — it must not have to guess
        # which classes came back empty on purpose and which are simply empty.
        "sensitive": sorted(sensitive),
        "total": int(total),
        "offset": offset,
        "limit": limit,
        "items": [
            _junk_item_to_json(
                r, (marks.get(int(r["id"])) or ("", None))[0] == "photo", sensitive)
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


def _animal_item_to_json(row: sqlite3.Row) -> dict:
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
    }


_ANIMALS_JOIN = ("FROM files f LEFT JOIN frame_quality fq ON fq.file_id = f.id "
                 "LEFT JOIN manual_pet mp ON mp.file_id = f.id")


def _animals_population(features: FeaturesConfig) -> str:
    """What the TAB LISTS: the model's marks plus every frame a person has touched.

    Deliberately wider than the slice — a frame marked "not an animal" is no longer in the
    album and is still on this page, struck through, because a card that vanishes takes the
    undo button with it.

    F137: "the model's marks" is the automatic half of the shared rule (`animal_auto_sql`),
    not the `frame_quality.pet` cache — a threshold edit has to take frames OFF this page
    too, or the list and the counter it carries would disagree about the same collection.
    """
    return (f"({animal_auto_sql(features, 'fq')} OR mp.file_id IS NOT NULL) "
            "AND f.dup_of IS NULL AND f.error IS NULL")


def _animals_count_sql(features: FeaturesConfig) -> str:
    """What COUNTS as an animal: `sorter.animal_ids_sql` and nothing else, over the
    canonical, readable files every other counter in this file is built on. Used by this
    tab and by the "Overview" number, so the two cannot disagree with the album or with
    each other."""
    return f"""SELECT COUNT(*) FROM files f
    WHERE f.dup_of IS NULL AND f.error IS NULL AND f.id IN ({animal_ids_sql(features)})"""


def _animals_select(features: FeaturesConfig) -> str:
    """One card, one row shape — the page and the answer to a mark are the same SELECT, so
    a card redrawn after an edit says exactly what the same card would say on a reload."""
    return f"""SELECT f.id, f.path, f.taken_at, fq.pet_score,
           mp.is_animal AS manual, f.id IN ({animal_ids_sql(features)}) AS is_animal
    {_ANIMALS_JOIN}"""


def _animals_payload(db_path: Path, features: FeaturesConfig,
                     offset: int, limit: int) -> dict:
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

    `features` is the LIVE config's, for the reason `/api/junk` reads its sensitive
    classes off it: the thresholds this page is drawn with are the ones in force at the
    moment of the request, not the ones some run wrote into the database (F137).
    """
    population = _animals_population(features)
    conn = _connect(db_path)
    try:
        total = conn.execute(
            f"SELECT COUNT(*) {_ANIMALS_JOIN} WHERE {population}").fetchone()[0]
        animals = conn.execute(_animals_count_sql(features)).fetchone()[0]
        rows = conn.execute(
            f"""{_animals_select(features)}
                WHERE {population}
                ORDER BY fq.pet_score DESC, f.id
                LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
    finally:
        conn.close()
    return {
        "total": int(total),
        "animals": int(animals),
        "offset": offset,
        "limit": limit,
        "items": [_animal_item_to_json(r) for r in rows],
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


def _apply_animal_mark(db_path: Path, features: FeaturesConfig,
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
    count_sql = _animals_count_sql(features)
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
            f"""{_animals_select(features)}
                WHERE {_animals_population(features)}
                  AND f.id IN ({known_placeholders})""",
            known).fetchall()
        animals = conn.execute(count_sql).fetchone()[0]
    finally:
        conn.close()
    return {
        "marked": len(known),
        "animals": int(animals),
        "items": [_animal_item_to_json(r) for r in rows],
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
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
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


# --- F126: the "Review" workspace — duplicates, blur, closed eyes, no subject -------
# Four signals, one job: look at a frame and decide whether it stays. Duplicates have had
# a tab with the whole viewing-and-deleting machinery since U3; the other three have been
# computed into `frame_quality` since F113 and were not visible anywhere. So this is one
# place with four SLICES rather than four tabs — and the duplicates half is deliberately
# untouched: `/api/dupes` and its four write routes answer exactly as before, because that
# is the one path in the product that deletes files and it is the one path that has been
# run against the live collection.
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
_REVIEW_SLICES = ("dupes", "blurred", "eyes", "subject")

# F139: which album kind each flat slice gathers into — and, read the other way, the map
# that keeps the list and the album on one rule. The names differ because the switcher's
# are older than the album's (`eyes` is a chip label, `eyes_closed` is a folder), and
# renaming either half would move an API parameter for nothing. Duplicates have no kind:
# they are the grouped slice, the one where a keeper is chosen, and the one path in the
# program that deletes files — collecting them into a folder is not what they are for.
_REVIEW_SLICE_KIND = {"blurred": "blurred", "eyes": "eyes_closed",
                      "subject": "no_subject"}

# Blurred is ranked by the number the slice exists for; the other two have no ranking of
# their own, so they go in index order — stable between pages, which is what paging needs.
_REVIEW_SLICE_ORDER = {
    "blurred": "fq.sharpness ASC, f.id",
    "eyes": "f.id",
    "subject": "f.id",
}

# The membership rule itself lives in sorter.py (`quality_slice_where`, `QUALITY_FROM`)
# and is read from there rather than restated here: the album of a slice and the list of
# it must be the same set of frames, and two spellings of one condition drift.
_REVIEW_FROM = QUALITY_FROM


def _review_where(slice_: str, blur_max: float | None) -> tuple[str, list[object]]:
    """The WHERE of one flat slice + its parameters — the shared rule, by slice name.

    `blur_max` is the window the blurred list opens to (`features.blur_review_max`) and
    applies to that slice alone; None — "show more" has been pressed and the list runs on
    without a ceiling.
    """
    return quality_slice_where(_REVIEW_SLICE_KIND[slice_], blur_max)


def _review_count(conn: sqlite3.Connection, slice_: str,
                  blur_max: float | None) -> int:
    """How many frames one flat slice holds, under the same WHERE the page uses."""
    where, params = _review_where(slice_, blur_max)
    return int(conn.execute(
        f"SELECT COUNT(*) {_REVIEW_FROM} WHERE {where}", params).fetchone()[0])


def _review_flat_counts(conn: sqlite3.Connection, blur_max: float) -> dict[str, int]:
    """The three flat slice counters — plain aggregates, cheap enough for "Overview".

    Blurred is counted INSIDE the window, so the chip, the "Overview" row and the length
    of the list the tab opens with are the same number.
    """
    return {
        "blurred": _review_count(conn, "blurred", blur_max),
        "eyes": _review_count(conn, "eyes", None),
        "subject": _review_count(conn, "subject", None),
    }


# F133: the same three slices again, counting only the frames NOBODY has decided about.
# "Decided" is a row in `dedup_choice` and nothing else — the rule the marks are written
# by — so a slice empties as the person works through it, which is what makes the warning
# on the "Layout" tab disappear on its own.
_REVIEW_PENDING_FROM = f"{_REVIEW_FROM} LEFT JOIN dedup_choice dc ON dc.file_id = f.id"


def _review_pending_count(conn: sqlite3.Connection, slice_: str,
                          blur_max: float | None) -> int:
    """How many frames of one flat slice still carry no decision."""
    where, params = _review_where(slice_, blur_max)
    return int(conn.execute(
        f"SELECT COUNT(*) {_REVIEW_PENDING_FROM} WHERE {where} AND dc.action IS NULL",
        params).fetchone()[0])


def _review_pending_counts(conn: sqlite3.Connection, blur_max: float) -> dict[str, int]:
    """The undecided part of each flat slice, under the same WHERE the page uses."""
    return {
        "blurred": _review_pending_count(conn, "blurred", blur_max),
        "eyes": _review_pending_count(conn, "eyes", None),
        "subject": _review_pending_count(conn, "subject", None),
    }


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
    """One card of a flat slice: a thumbnail, a name, a date, sharpness, the decision."""
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
        "action": action,
        "thumb_url": f"/thumb/{int(row['id'])}",
        "video": imaging.is_video_path(path),
    }


def _review_payload(db_path: Path, slice_: str, offset: int, limit: int, *,
                    beyond: bool, blur_max: float, max_distance: int) -> dict:
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

    `eyes_reason='no_faces_run'` (F125) — the eyes question is asked only where a face
    was found, so without a faces run the honest answer is why there is no data, not a
    zero that looks like "your subjects all had their eyes open".
    """
    conn = _connect(db_path)
    try:
        counts = _review_flat_counts(conn, blur_max)
        pending = _review_pending_counts(conn, blur_max)
        eyes_reason = None if faces_stage_ran(conn) else "no_faces_run"
        window_total = counts["blurred"]
        items: list[dict] = []
        total = 0
        if slice_ != "dupes":
            ceiling = None if (beyond or slice_ != "blurred") else blur_max
            where, params = _review_where(slice_, ceiling)
            total = int(conn.execute(
                f"SELECT COUNT(*) {_REVIEW_FROM} WHERE {where}", params).fetchone()[0])
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.taken_at, fq.sharpness
                    {_REVIEW_FROM} WHERE {where}
                    ORDER BY {_REVIEW_SLICE_ORDER[slice_]}
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
        "blur_max": float(blur_max),
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
    offers exactly four, so anything else is a client that has lost track of what it is
    asking for. The window is parsed by the plan-page rules (`_parse_page_window`).
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


def _tabs_visibility_payload(db_path: Path, features: FeaturesConfig) -> dict[str, bool]:
    """F54: visibility of the "People"/"Events"/"Animals" tabs — by data presence
    (variant B, without a meta table). person ⇔ there is a faces row with a non-empty
    cluster_id (the same source as `_clusters_payload`); event ⇔ non-empty `events`;
    animal (F123) ⇔ some `frame_quality` row counts as an animal, which is false for
    every collection processed with `features.pets` off. Light EXISTS queries, we do not
    build the full payload.

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
            f"WHERE {_animals_population(features)})"
        ).fetchone()[0])
        face = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files WHERE dup_of IS NULL AND error IS NULL "
            "AND media_type = 'photo')"
        ).fetchone()[0])
        indexed = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files)"
        ).fetchone()[0])
    finally:
        conn.close()
    return {"person": person, "event": event, "animal": animal, "face": face,
            "indexed": indexed}


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

    F126: the three flat review slices are counted here too, by the SAME queries the
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
        animals = conn.execute(_animals_count_sql(features)).fetchone()[0]
        faces_ran = faces_stage_ran(conn)
        faces_counts: dict[str, int | None] = {
            name: (_face_slice_count(conn, cfg, name) if faces_ran else None)
            for name in FACE_SLICES
        }
        review = _review_flat_counts(conn, features.blur_review_max)
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
            "no_subject": review["subject"],
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

# One card, and the same shape whichever state produced it. LEFT JOIN because a photograph
# usually has no `media_class` row at all — the class is what the privacy rule below reads.
_SEARCH_ROWS_SQL = """SELECT f.id, f.path, f.taken_at, mc.verdict
    FROM files f LEFT JOIN media_class mc ON mc.file_id = f.id
    WHERE f.id IN ({marks})"""

# A limit is a SAMPLE SIZE (search.py), so a client asking for more than the server is
# willing to render gets less rather than an error — the `_parse_page_window` rule.
_SEARCH_MAX_LIMIT = 1000


def _search_index_state(conn: sqlite3.Connection, model: str) -> dict:
    """Which of the four states the index is in, plus the numbers that state it.

    `index_model` is what a person is told when the answer is "another model": the name of
    the model that actually produced the stored vectors, taken as the one with the most
    rows. Naming it is the difference between a sentence somebody can act on and a shrug —
    and the row count is how the name is chosen, because a table can hold leftovers of
    several models and only the dominant one is worth putting in front of a reader.

    `indexed` counts vectors of THIS model within the searchable population, `total` the
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
        "total": photos,
    }


def _search_item_to_json(row: sqlite3.Row, score: float,
                         sensitive: frozenset[str]) -> dict:
    """One card of the ranking: the score is always on it, the thumbnail sometimes.

    F133's rule, unchanged: a frame whose class sits in `vlm.exclude_classes` (documents
    by default) gets no `thumb_url`, so the browser never asks `/thumb` for it and no
    preview of a passport is ever decoded. The guard is here, on the server, for the
    reason it is there — a search that answered with a link would turn this route into
    the way around a protection the slices already apply.
    """
    path = Path(row["path"])
    payload = {
        "file_id": int(row["id"]),
        "name": path.name,
        "date": row["taken_at"],
        # A ranking, not a filter: the number is what lets a reader see where the
        # relevance ran out, and a card without it would hide exactly that.
        "score": float(score),
    }
    verdict = row["verdict"]
    if verdict is None or str(verdict) not in sensitive:
        payload["thumb_url"] = f"/thumb/{int(row['id'])}"
        payload["video"] = imaging.is_video_path(path)
    return payload


def _search_items(conn: sqlite3.Connection, hits: Sequence[tuple[int, float]],
                  sensitive: frozenset[str]) -> list[dict]:
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
    return [_search_item_to_json(rows[fid], score, sensitive)
            for fid, score in hits if fid in rows]


def _search_payload(cfg: Config, db_path: Path, text: str, limit: int,
                    encoder: TextEncoder | None = None) -> dict:
    """`GET /api/search` — the state of the index always, the ranking when there is one.

    The model is not asked anything unless there is a reason to: an empty query and an
    unavailable index both return before `search_text`, which is what keeps a stray
    keystroke from loading CLIP and what makes "the line is disabled" cheap to render.

    `EmbeddingsMissing` is still caught, because the state was read a moment earlier and a
    run can empty the table in between; the answer then carries the engine's own reason
    rather than an empty `items` list, which is the one thing this route must never send.
    """
    conn = _connect(db_path)
    try:
        model = search_index_model(cfg)  # F141: the search model, not the classifier's
        payload = _search_index_state(conn, model)
        payload.update({"query": text, "limit": limit, "items": []})
        if not text.strip() or not payload["available"]:
            return payload
        try:
            hits = search_text(cfg, conn, text, limit=limit, encoder=encoder)
        except EmbeddingsMissing as exc:
            payload["state"] = exc.reason
            payload["available"] = False
            return payload
        payload["items"] = _search_items(
            conn, hits, frozenset(cfg.vlm.exclude_classes))
        return payload
    finally:
        conn.close()


def _parse_search_query(query: dict[str, list[str]],
                        default_limit: int) -> tuple[str, int] | None:
    """(query text, limit) for `GET /api/search`, or None -> 400.

    An absent/empty `q` is NOT an error: the client asks with one on purpose, to learn the
    state of the index without spending a model on it. `limit` follows the
    `_parse_page_window` rule — a non-integer or a negative one is rejected, an
    over-eager one is clamped.
    """
    text = (query.get("q") or [""])[0]
    raw_limit = (query.get("limit") or [str(default_limit)])[0].strip()
    try:
        limit = int(raw_limit or default_limit)
    except ValueError:
        return None
    if limit < 0:
        return None
    return text, min(limit, _SEARCH_MAX_LIMIT)


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


# --- F36: "Process" — the background pipeline index→geo→landmarks→faces→events→
# junk→phash from the web (POST /api/process), pollable progress (GET
# /api/process/status), cancel (POST /api/process/cancel). NOT imported from cli.py
# (to avoid a cli<->ui cycle) — the same leaf functions as `cli._pipeline_steps` are
# called directly from indexer/geo/landmarks/faces/events/junk/dedup/naming, +
# compute_phashes (dedup) as the last step.

_PIPELINE_STAGE_NAMES = ("index", "geo", "landmarks", "faces", "events", "junk", "phash")

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

    def _junk(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        stats = classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb)
        return _stage_stats(stats, ("processed",), "skipped_incremental")

    def _phash(cfg: Config, conn: sqlite3.Connection, cb: _ProgressCB) -> _StageStats:
        compute_phashes(cfg, conn, progress=cb)
        return None

    steps: list[tuple[str, _StageFn]] = [
        ("index", _index), ("geo", _geo), ("landmarks", _landmarks),
        ("faces", _faces), ("events", _events), ("junk", _junk), ("phash", _phash),
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

    F138: the three knobs that moved onto this screen out of the settings column ride
    here too, from the same place — `vlm.quality`/`vlm.quality_scope`,
    `features.pets_verify`, `dedup.keeper_vlm`. The column no longer offers them, so
    the file is now their ONLY home and this is what a run starts from.
    """
    return {
        "deep": bool(cfg.naming.vlm_enabled),
        "geo_online": cfg.geo.provider == "online",
        "pets": bool(cfg.features.pets),
        "pets_verify": bool(cfg.features.pets_verify),
        "quality": bool(cfg.vlm.quality),
        "quality_scope": str(cfg.vlm.quality_scope),
        "keeper": bool(cfg.dedup.keeper_vlm),
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
# The rates, each with the measurement it comes from:
_SEC_PER_VLM_FRAME = 0.78    # F113: one frame in one prompt
_SEC_PER_VLM_GROUP = 1.32    # F132: one comparative question over a whole group
# The faces stage over the reference collection — the ~17 minutes the changelog and the
# F123 note both quote — spread over its 19 757 photographs.
_SEC_PER_FACES_FRAME = 17 * 60 / 19757
# index + geo + landmarks + phash, the four that always run: ~5 minutes over the same
# collection.
_SEC_PER_BASE_FRAME = 5 * 60 / 19757
# events: a grouping pass over rows the DB already holds — under a minute there, and it
# is scaled per frame for the same reason as the others rather than pinned at "fast".
_SEC_PER_EVENTS_FRAME = 15.0 / 19757

# The photographs a run actually works on: `sorta` skips a duplicate and a file it could
# not read, so counting them in would price frames nobody looks at. Same predicate the
# faces measurement script samples by.
_LIVE_PHOTOS_SQL = ("SELECT COUNT(*) FROM files "
                    "WHERE dup_of IS NULL AND error IS NULL AND media_type = 'photo'")


def _positive_or_none(value: int) -> int | None:
    """A count of zero from a stage that has never run is "unknown", not "nothing"."""
    return value or None


def _quality_scope_counts(cfg: Config, conn: sqlite3.Connection,
                          photos: int, group_ids: int | None) -> dict[str, int | None]:
    """How many frames each `vlm.quality_scope` would hand the model.

    The scope is a select next to the checkbox, and the four options differ by hours —
    so all four prices travel to the browser at once and switching between them costs
    no request. `groups` reuses the near-duplicate grouping computed for the keeper
    line above rather than asking `junk.quality_scope_ids` to build it a second time
    (it costs seconds on a real collection).

    Unknown, i.e. a dash, wherever the population is not a population yet: no pHashes
    (`groups`), no events built (`events`), no faces run (`faces` — which `junk` refuses
    outright, see `quality_scope_ready`).
    """
    counts: dict[str, int | None] = {"all": _positive_or_none(photos),
                                     "groups": group_ids}
    events = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    counts["events"] = (None if not events else
                        len(quality_scope_ids(cfg, conn, "events") or ()))
    counts["faces"] = (len(quality_scope_ids(cfg, conn, "faces") or ())
                       if faces_stage_ran(conn) else None)
    return counts


# The estimate is asked for on every open of the first tab, and one of its counts is the
# near-duplicate grouping, which costs seconds over tens of thousands of pHashes (F66).
# Keyed like the Duplicates payload — any write to the index changes the fingerprint —
# plus the config values the arithmetic reads, so moving a threshold in the settings
# column re-prices immediately instead of serving the number the old one produced.
_ESTIMATE_CACHE_MAX_ITEMS = 2
_estimate_cache: OrderedDict[tuple, dict] = OrderedDict()
_estimate_cache_lock = threading.Lock()


def _estimate_cache_clear() -> None:
    """Drop the cached estimates (test isolation)."""
    with _estimate_cache_lock:
        _estimate_cache.clear()


def _process_estimate_payload(cfg: Config, db_path: Path) -> dict:
    """`GET /api/process/estimate` — the seconds behind every line of the run budget.

    `counts` travels next to `seconds` on purpose: a number a person is asked to plan
    an evening around should be checkable against the collection it was derived from,
    not taken on faith. Both dicts use the same keys, and `None` in either means "this
    index does not know" — the screen draws a dash and the sum says so too.

    `pets` is 0.0 rather than None when there is anything to count: the animal prompts
    ride inside the CLIP call the junk stage makes anyway (F123), so the line genuinely
    adds nothing to the run — the one place a zero here is the truth.
    """
    key = (str(db_path), _db_fingerprint(db_path), cfg.index.phash_max_distance,
           int(cfg.dedup.keeper_min_group_size),
           float(cfg.features.pet_candidate_threshold))
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
        deep = _positive_or_none(int(conn.execute(
            "SELECT COUNT(*) FROM media_class WHERE source = 'vlm'").fetchone()[0]))
        # The pet check is shown the frames CLIP scored above the candidate threshold —
        # a number that exists only once the CLIP pet group has run at all.
        pet_scored = int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score IS NOT NULL"
        ).fetchone()[0])
        pets_verify = None if not pet_scored else int(conn.execute(
            "SELECT COUNT(*) FROM frame_quality WHERE pet_score >= ?",
            (float(cfg.features.pet_candidate_threshold),)).fetchone()[0])
        hashed = bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM files WHERE phash IS NOT NULL)").fetchone()[0])
        keeper: int | None = None
        group_frames: int | None = None
        if hashed:
            groups = near_duplicate_groups(conn, cfg.index.phash_max_distance)
            keeper = sum(1 for g in groups
                         if len(g) >= int(cfg.dedup.keeper_min_group_size))
            group_frames = sum(len(g) for g in groups)
        scopes = _quality_scope_counts(cfg, conn, photos, group_frames)
    finally:
        conn.close()
    counts: dict[str, int | None] = {
        "base": _positive_or_none(photos),
        "faces": _positive_or_none(photos),
        "events": _positive_or_none(photos),
        "pets": _positive_or_none(photos),
        "pets_verify": pets_verify,
        "deep": deep,
        "keeper": keeper,
        **{f"quality_{scope}": value for scope, value in scopes.items()},
    }
    rates = {
        "base": _SEC_PER_BASE_FRAME,
        "faces": _SEC_PER_FACES_FRAME,
        "events": _SEC_PER_EVENTS_FRAME,
        "pets": 0.0,
        "pets_verify": _SEC_PER_VLM_FRAME,
        "deep": _SEC_PER_VLM_FRAME,
        "keeper": _SEC_PER_VLM_GROUP,
        **{f"quality_{scope}": _SEC_PER_VLM_FRAME for scope in scopes},
    }
    seconds = {name: (None if count is None else round(count * rates[name], 1))
               for name, count in counts.items()}
    payload = {"seconds": seconds, "counts": counts}
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
    """
    deep: bool = False
    geo_online: bool = False
    faces: bool = False
    events: bool = False
    pets: bool = False
    pets_verify: bool | None = None
    quality: bool | None = None
    quality_scope: str | None = None
    keeper: bool | None = None


def _validate_process_payload(payload: object) -> tuple[str, _RunOptions] | None:
    """Parse `{"source_dir": str, "deep": bool=False, "geo_online": bool=False,
    "faces": bool=False, "events": bool=False, "pets": bool=False,
    "pets_verify": bool?, "quality": bool?, "quality_scope": str?, "keeper": bool?}`
    (F50/#34: opt-in VLM tier / online geo for THIS run, without editing config.yaml;
    F53/#39: opt-in steps faces/events, the same principle — default False; F123:
    `pets` is an opt-in of the THIRD shape — neither a tier nor a step, but a config
    override on the junk stage, `features.pets`; F138: the same third shape for
    `features.pets_verify`, `vlm.quality`, `dedup.keeper_vlm` and the scope select).
    None -> invalid: not dict / `source_dir` not a string or empty after strip / a flag
    given but not bool / `quality_scope` outside VLM_QUALITY_SCOPES — a misspelling
    there is the 4.3-hour option, so it is refused rather than defaulted (the rule
    `_validate_settings_payload` set for the same key)."""
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
    for key in ("pets_verify", "quality", "keeper"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            return None
        flags[key] = value
    scope = payload.get("quality_scope")
    if scope is not None and (not isinstance(scope, str)
                              or scope not in VLM_QUALITY_SCOPES):
        return None
    flags["quality_scope"] = scope
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
    if opts.quality is not None:
        vlm_changed["quality"] = opts.quality
    if opts.quality_scope is not None:
        vlm_changed["quality_scope"] = opts.quality_scope
    vlm = dataclasses.replace(cfg.vlm, **vlm_changed) if vlm_changed else cfg.vlm
    dedup_cfg = (cfg.dedup if opts.keeper is None
                 else dataclasses.replace(cfg.dedup, keeper_vlm=opts.keeper))
    sources = [Path(source_dir).resolve()] if source_dir is not None else cfg.sources
    return dataclasses.replace(cfg, sources=sources, naming=naming, geo=geo,
                               features=features, vlm=vlm, dedup=dedup_cfg)


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
    "tab_junk": {"ru": "Не личные фото", "en": "Not personal photos",
                 "ja": "個人写真ではない"},
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
    "process_deep_hint": {
        "ru": "Медленнее; нужен `uv sync --extra vlm` (иначе автоматический откат "
              "на быстрый анализ).",
        "en": "Slower; requires `uv sync --extra vlm` (otherwise falls back to "
              "the fast tier automatically).",
        "ja": "処理が遅くなります。`uv sync --extra vlm` が必要です"
              "（なければ自動的に高速分析にフォールバックします）。",
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
    "process_quality_label": {
        "ru": "Качество кадров", "en": "Frame quality", "ja": "コマの品質",
    },
    "process_quality_hint": {
        "ru": "Открыты ли глаза, есть ли сюжет, не случаен ли кадр — то, чего не "
              "решить дёшево. Нужен `uv sync --extra vlm`.",
        "en": "Whether the eyes are open, whether there is a subject, whether the shot "
              "was an accident — what nothing cheap can decide. Needs "
              "`uv sync --extra vlm`.",
        "ja": "目が開いているか、被写体があるか、意図しない撮影ではないか — 安価な手段"
              "では決められないものです。`uv sync --extra vlm` が必要です。",
    },
    "process_quality_scope_label": {
        "ru": "У каких кадров спрашивать",
        "en": "Which frames to ask about",
        "ja": "どのコマについて尋ねるか",
    },
    "process_keeper_label": {
        "ru": "Лучший кадр в группе", "en": "Best frame of a group",
        "ja": "グループ内のベストショット",
    },
    "process_keeper_hint": {
        "ru": "Один сравнительный вопрос модели на группу почти-дублей: какой кадр "
              "оставить. Ничего не удаляет — только подсказка на вкладке «Разбор».",
        "en": "One comparative question per near-duplicate group: which frame to keep. "
              "It deletes nothing — it is a recommendation on the Review tab.",
        "ja": "類似写真のグループごとに 1 回、どのコマを残すかをモデルに比較させます。"
              "削除は行いません —「確認」タブでの推奨にとどまります。",
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
    "process_phase_elapsed": {  # a phase with no percent — the clock is the sign of life
        "ru": "{phase} — идёт {seconds} с",
        "en": "{phase} — {seconds}s so far",
        "ja": "{phase} — 経過 {seconds} 秒",
    },
    "process_stage_index": {"ru": "индексация", "en": "indexing", "ja": "インデックス作成"},
    "process_stage_geo": {"ru": "гео", "en": "geo", "ja": "位置情報"},
    "process_stage_landmarks": {"ru": "места", "en": "landmarks", "ja": "ランドマーク"},
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
    "settings_scope_groups": {
        "ru": "Группы похожих", "en": "Near-duplicate groups", "ja": "類似のグループ",
    },
    "settings_scope_events": {"ru": "События", "en": "Events", "ja": "イベント"},
    # F125 added the value; the entry in the list is F126's, which owns this file.
    "settings_scope_faces": {"ru": "По лицам", "en": "By faces", "ja": "顔で"},
    "settings_scope_all": {"ru": "Все кадры", "en": "Every frame", "ja": "すべてのコマ"},
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
    # F103: the "Not personal photos" view — the buckets the classifier carries out of
    # the collection, and the bulk way back for the frames it got wrong.
    "junk_intro": {
        "ru": "Кадры, которые классификатор посчитал не личными фото. Отметьте те, "
              "что попали сюда зря, и верните их — они снова разложатся по городам. "
              "Вердикт модели при этом не переписывается.",
        "en": "Frames the classifier judged not to be personal photos. Tick the ones "
              "that landed here by mistake and return them — they go back into the "
              "city layout. The model's verdict itself is not rewritten.",
        "ja": "分類器が個人写真ではないと判断したフレームです。誤って入ったものに"
              "チェックを入れて戻すと、再び都市ごとに振り分けられます。モデルの"
              "判定自体は書き換えません。",
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
    "junk_restore_button": {
        "ru": "Вернуть в фото", "en": "Return to photos", "ja": "写真に戻す",
    },
    "junk_restore_confirm": {
        "ru": "Вернуть в обычную раскладку по городам: {n}?",
        "en": "Return to the normal city layout: {n}?",
        "ja": "通常の都市別振り分けに戻します: {n} 件？",
    },
    "junk_undo_restore_button": {
        "ru": "Отменить возврат", "en": "Undo the return", "ja": "戻すのを取り消す",
    },
    "junk_restored_mark": {
        "ru": "возвращено в фото", "en": "returned to photos", "ja": "写真に戻しました",
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
    "junk_document_hint": {
        "ru": "Документы не открываются и не показываются: в этой корзине паспорта, "
              "справки и медицинские бланки. Видно имя файла и дату — этого хватает, "
              "чтобы решить.",
        "en": "Documents are neither opened nor rendered: this bucket holds passports, "
              "certificates and medical forms. The file name and the date are shown — "
              "enough to decide.",
        "ja": "書類は開かず表示もしません。このバケットにはパスポート、証明書、"
              "診断書が含まれます。判断にはファイル名と日付で十分です。",
    },
    "junk_error_prefix": {
        "ru": "Не удалось вернуть кадры: ", "en": "Could not return the frames: ",
        "ja": "フレームを戻せません: ",
    },
    "error_loading_junk": {
        "ru": "Не удалось загрузить корзины: ", "en": "Could not load the buckets: ",
        "ja": "バケットを読み込めません: ",
    },
    # --- F123: the "Animals" tab -----------------------------------------------------
    "animals_intro": {
        "ru": "Кадры с животными, сверху — те, в которых модель уверена больше. "
              "Точность около 92%: ниже по списку начинают попадаться шубы и игрушки, "
              "и видно, где проходит граница.",
        "en": "Frames with animals, the ones the model is most confident about first. "
              "Precision is about 92%: fur coats and plush toys start showing up "
              "further down, which is where the border of confidence is.",
        "ja": "動物が写ったコマです。モデルの確信度が高い順に並びます。精度は約 92% "
              "で、下に行くほど毛皮のコートやぬいぐるみが混じり始め、そこが確信度の"
              "境目です。",
    },
    "animals_empty": {
        "ru": "Здесь пусто — животные не найдены.",
        "en": "Nothing here — no animals were found.",
        "ja": "ここは空です。動物は見つかりませんでした。",
    },
    "animals_score_label": {
        "ru": "уверенность {score}", "en": "confidence {score}", "ja": "確信度 {score}",
    },
    "animals_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "animals_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    "error_loading_animals": {
        "ru": "Не удалось загрузить животных: ", "en": "Could not load the animals: ",
        "ja": "動物を読み込めません: ",
    },
    # --- F124: taking a false mark off a frame (and putting a missing one back) --------
    # The two buttons are one toggle: the card offers the answer the frame does NOT have
    # right now. The third string is the way back to the automatic verdict, which is a
    # different thing from "not an animal" and therefore says so in words.
    "animals_mark_not_animal": {
        "ru": "Это не животное", "en": "Not an animal", "ja": "動物ではない",
    },
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
    # The switcher labels are the four slices; the duplicates one keeps the wording the
    # tab had, because that is what the user has been calling it since U3.
    "review_slice_dupes": {"ru": "Дубли", "en": "Duplicates", "ja": "重複"},
    "review_slice_blurred": {"ru": "Размытые", "en": "Blurred", "ja": "ぼやけ"},
    "review_slice_eyes": {"ru": "Закрытые глаза", "en": "Closed eyes", "ja": "目を閉じた"},
    "review_slice_subject": {"ru": "Без сюжета", "en": "No subject", "ja": "被写体なし"},
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
    "review_hint_blurred": {
        "ru": "Список открыт до резкости {max} и отсортирован от самых размытых. Это "
              "окно, а не приговор: размытые кадры встречаются во всех полосах вплоть "
              "до 400, поэтому кнопки «удалить всё ниже порога» здесь нет и по "
              "умолчанию не удаляется ничего.",
        "en": "The list opens down to a sharpness of {max} and starts with the "
              "blurriest. That is a window, not a verdict: blurred frames turn up in "
              "every band up to 400, so there is no “delete everything below the "
              "threshold” button here and nothing is marked by default.",
        "ja": "リストは鮮鋭度 {max} まで開き、ぼやけの強い順に並びます。これは判定では"
              "なく表示範囲です。ぼやけたコマは 400 までのどの帯にも現れるため、"
              "「しきい値以下をすべて削除」というボタンはなく、既定では何も削除しません。",
    },
    "review_hint_eyes": {
        "ru": "Кадры, на которых у людей закрыты глаза. Вопрос задаётся только там, где "
              "найдено лицо.",
        "en": "Frames where the people have their eyes closed. The question is only "
              "asked where a face was found.",
        "ja": "人物が目を閉じているコマです。この質問は顔が検出されたコマにのみ行われます。",
    },
    "review_hint_subject": {
        "ru": "Кадры, в которых модель не нашла осмысленного сюжета: снятый пол, "
              "смазанная стена, случайное нажатие.",
        "en": "Frames where the model found no subject at all: a shot of the floor, a "
              "smeared wall, an accidental press.",
        "ja": "モデルが被写体を見つけられなかったコマです。床の写り込み、ぶれた壁、"
              "誤操作などです。",
    },
    "review_eyes_no_faces": {
        "ru": "Данных нет: стадия «лица» не запускалась, а про глаза спрашивают только "
              "там, где найдено лицо. Прогоните лица и повторите разбор.",
        "en": "No data: the faces stage never ran, and the eyes question is only asked "
              "where a face was found. Run faces and come back to this slice.",
        "ja": "データがありません。顔ステージが実行されておらず、目の質問は顔が検出された"
              "コマにのみ行われます。顔ステージを実行してから、この区分を開いてください。",
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
    # F126: the three review slices that have a number of their own. Blurred is counted
    # inside the window the list opens to, so the row and the list agree.
    "overview_blurred": {"ru": "Размытых", "en": "Blurred", "ja": "ぼやけ"},
    "overview_eyes_closed": {"ru": "С закрытыми глазами", "en": "With closed eyes",
                             "ja": "目を閉じた"},
    "overview_no_subject": {"ru": "Без сюжета", "en": "With no subject",
                            "ja": "被写体なし"},
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
    "search_shown_label": {
        "ru": "Запрос «{q}»: {n} кадров, от самого близкого",
        "en": "Query “{q}”: {n} frames, closest first",
        "ja": "クエリ「{q}」: {n} 件（近い順）",
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
    "face_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "face_shown_label": {
        "ru": "Показано {shown} из {total}",
        "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    "error_loading_face_slices": {
        "ru": "Не удалось загрузить срезы по лицам: ",
        "en": "Could not load the face slices: ",
        "ja": "顔のスライスを読み込めません: ",
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


_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="{{lang}}"><head><meta charset="utf-8">
<title>Sorta UI</title>
<style>
:root {
  color-scheme: light;
  --bg: #F7F8FB;
  --surface: #FFFFFF;
  --head-bg: #FBFCFE;
  --card: #FFFFFF;
  --chip: #F3F5F9;
  --field: #FFFFFF;
  --track: #E7EBF2;
  --ink: #1A2230;
  --muted: #5B6675;
  --line: #E3E7EE;
  --accent: #2F5BD0;
  --accent-soft: #B9C8EF;
  --on-accent: #FFFFFF;
  --tab-active-ink: #1A2230;
  --tab-active-bg: #FFFFFF;
  --tab-active-line: #DBE1EA;
  --good: #1E9E6A;
  --good-soft: #BFE7D5;
  --danger: #D14343;
  --danger-soft: #EAB6B6;
  --radius-sm: 5px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-pill: 999px;
  --shadow-sm: 0 1px 2px rgba(20,30,50,.06);
  --shadow-lg: 0 8px 24px rgba(20,30,50,.05);
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 12px;
  --space-lg: 16px;
  --space-xl: 24px;
  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #141A22;
    --surface: #181F29;
    --head-bg: #171E27;
    --card: #181F29;
    --chip: #1E2731;
    --field: #121821;
    --track: #232D39;
    --ink: #E6EAF0;
    --muted: #8A96A6;
    --line: #28323F;
    --accent: #6E9BFF;
    --accent-soft: #31456E;
    --on-accent: #0B1220;
    --tab-active-ink: #E6EAF0;
    --tab-active-bg: #212B37;
    --tab-active-line: #334053;
    --good: #3ECB95;
    --good-soft: #204A3A;
    --danger: #F0736F;
    --danger-soft: #5A2C2C;
    --shadow-sm: none;
    --shadow-lg: none;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #141A22;
  --surface: #181F29;
  --head-bg: #171E27;
  --card: #181F29;
  --chip: #1E2731;
  --field: #121821;
  --track: #232D39;
  --ink: #E6EAF0;
  --muted: #8A96A6;
  --line: #28323F;
  --accent: #6E9BFF;
  --accent-soft: #31456E;
  --on-accent: #0B1220;
  --tab-active-ink: #E6EAF0;
  --tab-active-bg: #212B37;
  --tab-active-line: #334053;
  --good: #3ECB95;
  --good-soft: #204A3A;
  --danger: #F0736F;
  --danger-soft: #5A2C2C;
  --shadow-sm: none;
  --shadow-lg: none;
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #F7F8FB;
  --surface: #FFFFFF;
  --head-bg: #FBFCFE;
  --card: #FFFFFF;
  --chip: #F3F5F9;
  --field: #FFFFFF;
  --track: #E7EBF2;
  --ink: #1A2230;
  --muted: #5B6675;
  --line: #E3E7EE;
  --accent: #2F5BD0;
  --accent-soft: #B9C8EF;
  --on-accent: #FFFFFF;
  --tab-active-ink: #1A2230;
  --tab-active-bg: #FFFFFF;
  --tab-active-line: #DBE1EA;
  --good: #1E9E6A;
  --good-soft: #BFE7D5;
  --danger: #D14343;
  --danger-soft: #EAB6B6;
  --shadow-sm: 0 1px 2px rgba(20,30,50,.06);
  --shadow-lg: 0 8px 24px rgba(20,30,50,.05);
}
* { box-sizing: border-box; }
html, body { max-width: 100%; overflow-x: hidden; }
body {
  font-family: var(--font-sans);
  margin: 0;
  padding: var(--space-lg) var(--space-xl) var(--space-xl);
  background: var(--bg);
  color: var(--ink);
  font-size: 14px;
  line-height: 1.45;
}
h1, h2, h3 { font-weight: 600; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: var(--radius-sm); }
@media (prefers-reduced-motion: no-preference) {
  .btn, .tab-btn, .top-btn, details > summary, .stage-chip, .thumb-skel img { transition: background .12s ease, border-color .12s ease, color .12s ease, opacity .12s ease, transform .12s ease; }
}

/* --- таблицы -------------------------------------------------------- */
.table-wrap { width: 100%; max-width: 100%; overflow-x: auto; border-radius: var(--radius-md); border: 1px solid var(--line); }
table { border-collapse: collapse; width: 100%; background: var(--surface); font-variant-numeric: tabular-nums; }
td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tbody tr:nth-child(even), table tr:nth-child(even) { background: var(--chip); }
table tr:hover { background: var(--accent-soft); }
img { width: 56px; height: 56px; object-fit: cover; border-radius: var(--radius-sm); border: 1px solid var(--line);
      vertical-align: middle; margin-right: var(--space-sm); background: var(--chip); }
details { margin-left: var(--space-md); }
summary { cursor: pointer; font-weight: 600; margin: var(--space-sm) 0; overflow-wrap: anywhere; list-style-position: outside; }
details .table-wrap { margin: 0.3rem 0 0.8rem var(--space-md); width: calc(100% - 1rem); }
/* F70: «показано N из M» под страницей файлов раскрытой папки. */
.plan-page-status { margin: 0 0 0.4rem var(--space-md); font-size: 0.8rem; color: var(--muted); }

/* --- кнопки ----------------------------------------------------------- */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--font-sans); font-size: 13px; font-weight: 500; line-height: 1;
  padding: 7px 12px; margin: 0; cursor: pointer;
  background: var(--chip); color: var(--ink);
  border: 1px solid var(--line); border-radius: var(--radius-md);
}
.btn:hover { border-color: var(--accent); }
.btn:active { transform: translateY(1px); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.btn:disabled:hover { border-color: var(--line); }
.btn svg { width: 14px; height: 14px; flex: none; }
.btn-primary { background: var(--accent); color: var(--on-accent); border-color: var(--accent); font-weight: 600; }
.btn-primary:hover { filter: brightness(1.06); border-color: var(--accent); }
.btn-ghost { background: transparent; }
.btn-danger { background: transparent; color: var(--danger); border-color: var(--danger-soft); }
.btn-danger:hover { background: var(--danger-soft); border-color: var(--danger); }
.btn-sm { padding: 4px 9px; font-size: 12px; }
/* Plan-row buttons (delete / keep / folder place) sat in the cell with no layout:
   labels of different widths drifted vertically and the three buttons read as
   three different levels. One line, one gap. */
.plan-actions { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }

/* --- шапка / вордмарк --------------------------------------------------- */
.header-bar { display: flex; align-items: center; justify-content: space-between;
      margin: 0 0 var(--space-lg) 0; gap: var(--space-md); flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 8px; }
.brand-mark { width: 26px; height: 26px; color: var(--accent); flex: none; }
.brand-name { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.01em; }
.header-controls { display: flex; align-items: center; gap: var(--space-sm); flex-wrap: wrap; }
.lang-field { display: inline-flex; align-items: center; gap: 5px; background: var(--chip);
      border: 1px solid var(--line); border-radius: var(--radius-md); padding: 0 4px 0 8px; }
.lang-field svg { width: 14px; height: 14px; color: var(--muted); flex: none; }
.lang-select { padding: 6px 4px; cursor: pointer; border: none; background: transparent; color: var(--ink);
      font-family: var(--font-sans); font-size: 13px; }
/* нативный option-попап в тёмной теме иначе белый со светлым текстом (плохой
   контраст) — явные цвета по токенам темы + color-scheme выше делают его читаемым */
.lang-select option { background: var(--surface); color: var(--ink); }
.theme-toggle-btn svg { width: 15px; height: 15px; }

/* --- вкладки ------------------------------------------------------------ */
.tabs { display: flex; gap: 4px; margin-bottom: var(--space-lg); border-bottom: 1px solid var(--line);
      overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; }
.tab-btn {
  flex: none; padding: 8px 16px; cursor: pointer; font-family: var(--font-sans); font-size: 13.5px;
  font-weight: 500; color: var(--muted); background: transparent;
  border: 1px solid transparent; border-bottom: none; border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.tab-btn:hover { color: var(--ink); background: var(--chip); }
.tab-btn.active { background: var(--tab-active-bg); color: var(--tab-active-ink);
      border-color: var(--tab-active-line); font-weight: 600; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
/* --- F133: a slice is a panel INSIDE the "Slices" tab, switched by a pin rather than by
   a tab of its own. The pins are drawn from data (see SLICE_PINS in the script), because
   F129 turns the list into a query and a fixed row of buttons would have to be redrawn a
   feature later. --- */
.slice-panel { display: none; }
.slice-panel.active { display: block; }
.slice-search { display: flex; flex-direction: column; gap: 4px; margin-bottom: var(--space-md); }
.slice-query-field { display: flex; align-items: center; gap: var(--space-sm);
      padding: 6px var(--space-sm); border: 1px solid var(--line); border-radius: var(--radius-md);
      background: var(--card); color: var(--muted); }
.slice-query-field svg { width: 15px; height: 15px; flex: none; }
.slice-query-field input { flex: 1; min-width: 0; border: 0; background: transparent;
      color: inherit; padding: 2px 0; }
/* --- F134: the line is a control now, and the two things beside it are the feature:
   the reason it cannot be used, and the way to fix it. A disabled field with nothing
   next to it is how a person concludes their archive is empty. --- */
.slice-query-row { display: flex; align-items: center; gap: var(--space-sm); }
.slice-query-row .slice-query-field { flex: 1; }
.slice-query-field input:disabled { cursor: not-allowed; }
#search-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: var(--space-md); }
.search-card { border: 1px solid var(--line); border-radius: var(--radius-md);
      padding: var(--space-sm); background: var(--card); display: flex;
      flex-direction: column; gap: var(--space-xs); }
.search-card img { width: 100%; height: 110px; margin: 0; }
.search-card-name { font-size: 0.8rem; word-break: break-all; }
.search-card-meta { font-size: 0.75rem; color: var(--muted); }
/* The score is not decoration: it is the only thing that explains the order, and the
   reader stops where it stops being convincing. */
.search-card-score { font-size: 0.75rem; color: var(--muted);
      font-variant-numeric: tabular-nums; }
/* --- F133: the order warning. Loud enough to be read, and a hint and nothing more: the
   collection is alive, "gather" happens again and again, and a person coming back for one
   album must not be walked through steps. --- */
.layout-warning { display: flex; align-items: center; gap: var(--space-sm); flex-wrap: wrap;
      margin-bottom: var(--space-md); padding: var(--space-sm) var(--space-md);
      border: 1px solid var(--danger-soft); border-left: 4px solid var(--danger);
      border-radius: var(--radius-md);
      background: var(--card); font-size: 0.9rem; }

/* --- F138: the run budget. A price is a HINT, not a heading — it is set as secondary
   text and pushed to the right edge of its line, so the list still reads as a list of
   what to compute and the numbers come second. The total is the one thing here set
   like a heading: it is what the eye meets on its way to the run button. --- */
.cost-block { display: flex; flex-direction: column; gap: var(--space-sm); }
.cost-head { font-weight: 600; font-size: 0.9rem; }
.cost-row { display: grid; grid-template-columns: 1fr auto auto; align-items: baseline;
      column-gap: var(--space-sm); }
.cost-name { font-size: 0.85rem; }
.cost-always { font-size: 0.8rem; color: var(--muted); }
.cost-price { font-size: 0.8rem; color: var(--muted); text-align: right;
      font-variant-numeric: tabular-nums; white-space: nowrap; }
.cost-row .cost-hint, .cost-row .process-toggle-warn { grid-column: 1 / -1; }
.cost-child { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr auto;
      align-items: baseline; column-gap: var(--space-sm); margin-left: 20px; }
.cost-child .cost-hint { grid-column: 1 / -1; }
.cost-child select { font-size: 0.8rem; padding: 4px 6px; margin-left: 6px; }
.cost-total { display: flex; align-items: baseline; gap: var(--space-sm);
      padding-top: var(--space-sm); border-top: 1px solid var(--line); }
.cost-total-label { font-size: 0.85rem; color: var(--muted); }
.cost-total-value { font-weight: 600; font-variant-numeric: tabular-nums; }

/* --- общие карточки ------------------------------------------------------ */
.card { border: 1px solid var(--line); border-radius: var(--radius-lg); padding: var(--space-md) var(--space-lg);
      margin-bottom: var(--space-md); background: var(--card); box-shadow: var(--shadow-sm); }
.card.named { border-color: var(--accent-soft); border-width: 1.5px; }
.card h3 { margin: 0 0 var(--space-sm) 0; font-size: 0.95rem; }

/* --- бейджи / чипы -------------------------------------------------------- */
.badge { display: inline-flex; align-items: center; gap: 3px; color: var(--good); font-weight: 600;
      margin-left: 6px; font-size: 0.85em; }
.badge svg { width: 12px; height: 12px; }
.chip { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: var(--radius-pill);
      font-size: 0.78rem; font-weight: 500; background: var(--chip); color: var(--muted); }
.chip-good { background: var(--good-soft); color: var(--good); }
.chip-accent { background: var(--accent-soft); color: var(--accent); }
.chip-danger { background: var(--danger-soft); color: var(--danger); }

/* --- инпуты/селекты --------------------------------------------------- */
input[type="text"], select {
  font-family: var(--font-sans); font-size: 13px; color: var(--ink); background: var(--field);
  border: 1px solid var(--line); border-radius: var(--radius-md); padding: 7px 9px;
}
input[type="text"]:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
input[type="checkbox"], input[type="radio"] { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
label { cursor: pointer; }

/* --- состояния: пусто/загрузка/ошибка ---------------------------------- */
.state-msg { display: flex; align-items: center; gap: 8px; padding: var(--space-md) var(--space-lg);
      border-radius: var(--radius-md); color: var(--muted); background: var(--chip); }
.state-error { color: var(--danger); background: var(--danger-soft); }
.state-msg svg { width: 15px; height: 15px; flex: none; }
@media (prefers-reduced-motion: no-preference) {
  .state-loading svg { animation: sorta-spin 0.9s linear infinite; }
}
@keyframes sorta-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.tree-controls { margin: 0 0 var(--space-md) 0; display: flex; gap: var(--space-sm); }
.top-btn { position: fixed; right: 1.2rem; bottom: 1.2rem; padding: 9px 14px;
      cursor: pointer; border-radius: var(--radius-pill); opacity: 0.9; z-index: 1000;
      background: var(--surface); box-shadow: var(--shadow-lg); }
.top-btn:hover { opacity: 1; }
.dupes-controls { margin: 0 0 var(--space-md) 0; display: flex; align-items: center; gap: var(--space-sm); }
.dupes-controls #dupes-save-status { color: var(--good); font-size: 0.85rem; }
.dupe-group .table-wrap { margin-bottom: var(--space-sm); }
.skip-label { display: inline-flex; align-items: center; gap: 5px; font-size: 0.85rem; color: var(--muted);
      margin-right: var(--space-md); }
.cluster-controls { margin: 0 0 var(--space-md) 0; }
#clusters-grid, #events-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: var(--space-md); }
.cluster-thumbs { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: var(--space-sm); }
.thumb-skel { width: 44px; height: 44px; border-radius: var(--radius-sm); background: var(--track);
      overflow: hidden; }
.thumb-skel img { width: 100%; height: 100%; margin: 0; object-fit: cover; cursor: zoom-in;
      display: block; opacity: 0; }
.thumb-skel.loaded { background: transparent; }
.thumb-skel.loaded img { opacity: 1; }
.thumb-skel img:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
.cluster-meta { font-size: 0.85rem; color: var(--muted); margin: 0 0 var(--space-sm) 0; }
.cluster-name-form { display: flex; gap: 5px; margin-bottom: var(--space-sm); }
.cluster-name-form input { flex: 1; min-width: 0; }
.cluster-merge-select { font-size: 0.8rem; display: flex; align-items: center; gap: 5px; color: var(--muted); }
.album-controls { display: flex; align-items: center; gap: 5px; margin-top: var(--space-sm); flex-wrap: wrap; }
.album-controls select { font-size: 0.8rem; padding: 6px 7px; }
.album-controls input[type="text"] { flex: 1; min-width: 90px; font-size: 0.8rem; padding: 6px 7px; }
.album-status { font-size: 0.8rem; color: var(--good); margin-left: 2px; }
.album-hint { font-size: 0.8rem; color: var(--muted); margin-top: var(--space-sm); font-style: italic; }
.event-meta { font-size: 0.85rem; color: var(--muted); margin: 0 0 var(--space-sm) 0; }
.event-thumbs { display: flex; flex-wrap: wrap; gap: 3px; margin: 0 0 var(--space-sm) 0; }
/* единый вид кликабельной миниатюры-превью (Города/Дубли/Перемещения/События) */
/* фон-плейсхолдер виден, пока lazy-<img> не загрузился — отклик вместо «пусто» */
.clickable-thumb { cursor: zoom-in; background: var(--track); }
.clickable-thumb:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
/* F80: в сетке видео и фото были неотличимы. Значок «плёнки» поверх угла плитки —
   обёртка появляется ТОЛЬКО у видео, у фото плитка остаётся голым <img>. */
.thumb-video { position: relative; display: inline-block; line-height: 0; }
.thumb-video-badge { position: absolute; left: 3px; bottom: 3px; display: inline-flex;
      align-items: center; gap: 3px; padding: 1px 4px; border-radius: var(--radius-sm);
      background: rgba(10,14,22,.72); color: #fff; font-size: 0.7rem; line-height: 1.4;
      pointer-events: none; }
.thumb-video-badge svg { width: 11px; height: 11px; display: block; }
.thumb-name { display: block; font-size: 0.8rem; color: var(--muted); word-break: break-all; margin-top: 2px; }
.event-name-input { width: 100%; margin-bottom: var(--space-sm); box-sizing: border-box; }

/* --- F103: корзины «не личные фото» ----------------------------------- */
/* Сетка плиток, а не таблица: здесь смотрят глазами — «это правда товар?» —
   и решение принимается по картинке, а не по колонкам. */
#junk-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: var(--space-md); }
.junk-card { border: 1px solid var(--line); border-radius: var(--radius-md);
      padding: var(--space-sm); background: var(--card); display: flex;
      flex-direction: column; gap: var(--space-xs); }
.junk-card.restored { outline: 2px solid var(--good); outline-offset: -2px;
      background: var(--good-soft); }
.junk-card img { width: 100%; height: 110px; margin: 0; }
/* Документ вместо превью получает нейтральную заглушку того же размера: сетка не
   ломается, а содержимое паспорта не декодируется вообще. */
.junk-doc-box { height: 110px; display: flex; align-items: center; justify-content: center;
      border-radius: var(--radius-sm); border: 1px dashed var(--line); background: var(--chip);
      color: var(--muted); font-size: 0.8rem; }
.junk-card-name { font-size: 0.8rem; word-break: break-all; }
.junk-card-meta { font-size: 0.75rem; color: var(--muted); }
.junk-card-select { display: flex; align-items: center; gap: 5px; font-size: 0.8rem; }

/* --- F123: the "Animals" tab -------------------------------------------- */
/* The same tile grid as the junk buckets: the decision is made by looking. The one
   difference is the confidence score on the card — the list is sorted by it, and the
   reader is looking for the place where the quality runs out. */
#animals-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: var(--space-md); }
.animal-card { border: 1px solid var(--line); border-radius: var(--radius-md);
      padding: var(--space-sm); background: var(--card); display: flex;
      flex-direction: column; gap: var(--space-xs); }
.animal-card img { width: 100%; height: 110px; margin: 0; }
.animal-card-name { font-size: 0.8rem; word-break: break-all; }
.animal-card-meta { font-size: 0.75rem; color: var(--muted); }
.animal-card-score { font-size: 0.75rem; color: var(--muted);
      font-variant-numeric: tabular-nums; }
/* F124: a frame the user has taken the mark off stays on the page, dimmed, with its
   verdict named and its way back one click away. Making it disappear would move the
   counter for no visible reason and hide the undo inside the vanished card. */
.animal-card.not-animal { opacity: 0.55; }
.animal-card.not-animal img { filter: grayscale(1); }
.animal-card-manual { font-size: 0.75rem; color: var(--muted); }
.animal-card-actions { display: flex; gap: var(--space-xs); flex-wrap: wrap;
      align-items: center; }

/* --- F152: the face slices -------------------------------------------------
   The same tile as the animal grid; no score line, because these slices have no
   confidence to print — a frame is in one because a face was detected on it. */
#face-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: var(--space-md); }
.face-card { border: 1px solid var(--line); border-radius: var(--radius-md);
      padding: var(--space-sm); background: var(--card); display: flex;
      flex-direction: column; gap: var(--space-xs); }
.face-card img { width: 100%; height: 110px; margin: 0; }
.face-card-name { font-size: 0.8rem; word-break: break-all; }
.face-card-meta { font-size: 0.75rem; color: var(--muted); }

/* --- F126: the "Review" workspace ---------------------------------------- */
/* The switcher looks like the junk bucket chips, for the same reason: a row of
   named counters is how a person picks which pile to go through next. Every slice
   stays in the row at zero — a slice that disappears when it empties turns an answer
   into a question about the interface. The flat slices reuse the tile grid; only
   duplicates keep their table, because only there is a keeper chosen. */
.review-slices { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center;
      margin: var(--space-md) 0; }
.review-slice-btn.active { background: var(--accent); color: var(--on-accent);
      border-color: var(--accent); font-weight: 600; }
.review-slice-count { margin-left: 6px; font-variant-numeric: tabular-nums; }
#review-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: var(--space-md); }
.review-card { border: 1px solid var(--line); border-radius: var(--radius-md);
      padding: var(--space-sm); background: var(--card); display: flex;
      flex-direction: column; gap: var(--space-xs); }
/* The decision is visible on the card itself: a frame already decided is not asked
   about again, in this slice or in any other it shows up in. */
.review-card.marked-delete { outline: 2px solid var(--danger); outline-offset: -2px;
      background: var(--danger-soft); }
.review-card.marked-keep { outline: 2px solid var(--good); outline-offset: -2px;
      background: var(--good-soft); }
.review-card img { width: 100%; height: 110px; margin: 0; }
.review-card-name { font-size: 0.8rem; word-break: break-all; }
.review-card-meta { font-size: 0.75rem; color: var(--muted);
      font-variant-numeric: tabular-nums; }
.review-card-select { display: flex; align-items: center; gap: 5px; font-size: 0.8rem; }

/* --- F108: вкладка «Обзор» ---------------------------------------------- */
/* Четыре группы рядом, а не одна длинная простыня: вопрос «что с архивом»
   распадается ровно на них, и ответ должен читаться без прокрутки. */
.overview-groups { display: grid; gap: var(--space-md);
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.overview-row { display: flex; align-items: baseline; justify-content: space-between;
      gap: var(--space-sm); padding: 5px 0; border-top: 1px solid var(--line); }
.overview-row:first-of-type { border-top: none; }
.overview-label { color: var(--muted); font-size: 0.85rem; }
.overview-value { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
/* Главное число группы (без места, файлов в индексе) — крупнее остальных строк. */
.overview-row-main .overview-value { font-size: 1.15rem; }
.overview-row-main .overview-label { color: var(--ink); font-weight: 500; }
/* Число, у которого есть своя вкладка, само является переходом на неё: обзор без
   переходов — отчёт, а нужен пульт. */
.overview-value-link { font-family: var(--font-sans); font-size: inherit; font-weight: 600;
      font-variant-numeric: tabular-nums; color: var(--accent); background: none;
      border: none; padding: 0; cursor: pointer; text-decoration: underline;
      text-underline-offset: 2px; }
.overview-value-link:hover { color: var(--ink); }
.overview-subtitle { margin: var(--space-md) 0 var(--space-xs) 0; color: var(--muted);
      font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.overview-note { margin: var(--space-sm) 0 0 0; color: var(--muted); font-size: 0.8rem; }
.overview-note-warn { color: var(--danger); }
/* Строки-значения (путь назначения, дата) переносятся, в отличие от чисел. */
.overview-text { white-space: normal; word-break: break-word; font-weight: 500;
      text-align: right; }
.process-intro { max-width: 46rem; color: var(--muted); }
/* F51: вертикальные группы (путь / каждый тумблер+hint / кнопки), а не один
   плоский flex — там .process-toggle-hint с flex-basis:100% уезжал в конец
   контейнера, после всех кнопок, оторвано от своего чекбокса. */
.process-controls { display: flex; flex-direction: column; gap: var(--space-sm);
      margin: var(--space-md) 0; max-width: 42rem; }
.process-path-row { display: flex; gap: var(--space-sm); align-items: center; flex-wrap: wrap; }
.process-path-row input[type="text"] { flex: 1; min-width: 220px; padding: 8px 10px; }
.process-option { display: flex; flex-direction: column; gap: 2px; }
/* F81: три блока первой вкладки. Настроенный блок схлопывается в одну строку,
   ненастроенный остаётся раскрытым; следующие блоки приглушены пояснением, но НЕ
   заблокированы — экран открывают многократно, и визард наказывает каждый
   следующий заход. */
.step { border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--surface);
      padding: var(--space-md); display: flex; flex-direction: column; gap: var(--space-sm); }
.step-head { display: flex; align-items: center; gap: var(--space-sm); flex-wrap: wrap; }
.step-title { font-weight: 600; font-size: 0.9rem; }
.step-summary { display: none; color: var(--muted); font-size: 0.85rem; overflow-wrap: anywhere; }
.step.collapsed .step-summary { display: inline; }
.step.collapsed .step-body { display: none; }
.step-edit-btn { display: none; padding: 3px 8px; font-size: 0.78rem; }
/* Кнопка видна всегда, когда шаг МОЖНО свернуть (источник задан): в свёрнутом
   виде она «изменить», в раскрытом — «свернуть». Раньше её показывал только
   .collapsed, поэтому раскрытый шаг обратно не складывался. */
.step.can-collapse .step-edit-btn { display: inline-flex; }
.step-body { display: flex; flex-direction: column; gap: var(--space-sm); }
.step-hint { display: none; font-size: 0.8rem; color: var(--muted); }
.step.step-dimmed { opacity: 0.65; }
.step.step-dimmed .step-hint { display: block; }
.excludes-panel { display: flex; flex-direction: column; gap: var(--space-sm);
      border: 1px solid var(--line); border-radius: var(--radius-md); padding: var(--space-md);
      background: var(--chip); }
.excludes-tree { max-height: 22rem; overflow: auto; background: var(--surface);
      border: 1px solid var(--line); border-radius: var(--radius-md); padding: var(--space-sm); }
.excludes-tree ul { list-style: none; margin: 0; padding-left: var(--space-md); }
.excludes-tree > ul { padding-left: 0; }
.excludes-tree li { margin: 2px 0; }
.excludes-row { display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; }
.excludes-meta { color: var(--muted); font-size: 0.78rem; }
/* F82: три состояния узла. Значок + подпись прямо в кнопке — состояние должно
   читаться взглядом по дереву, без наведения и без легенды под рукой; цвет —
   вторая, а не единственная опора (те же две роли, что у строк плана в F77). */
.tri-state { font: inherit; font-size: 0.85rem; line-height: 1.2; cursor: pointer;
      padding: 1px 7px; border-radius: var(--radius-pill); border: 1px solid var(--line);
      background: var(--surface); color: var(--muted); white-space: nowrap; }
.tri-state[data-state="layout"] { border-color: var(--accent); background: var(--accent-soft);
      color: var(--accent); font-weight: 500; }
.tri-state[data-state="scan"] { border-color: var(--danger); background: var(--danger-soft);
      color: var(--danger); font-weight: 500; }
.tri-state:disabled { cursor: default; opacity: 0.55; }
.excludes-legend { list-style: none; margin: 0; padding: 0; font-size: 0.8rem;
      color: var(--muted); display: flex; flex-direction: column; gap: 2px; }
.excludes-legend .tri-mark { font-weight: 600; color: var(--ink); }
.process-toggle-label { font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px; }
.process-toggle-hint { font-size: 0.8rem; color: var(--muted); margin-left: 20px; }
.process-toggle-warn { color: var(--danger); }
.process-actions { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center; }
.process-progress { width: 100%; max-width: 40rem; display: block; margin: var(--space-sm) 0; height: 8px;
      appearance: none; border: none; border-radius: var(--radius-pill); overflow: hidden; background: var(--track); }
.process-progress::-webkit-progress-bar { background: var(--track); border-radius: var(--radius-pill); }
.process-progress::-webkit-progress-value { background: var(--accent); border-radius: var(--radius-pill); }
.process-progress::-moz-progress-bar { background: var(--accent); border-radius: var(--radius-pill); }
/* #37: total ещё неизвестен (индексация сканирует дерево) — вместо «0 из 0»
   бегущая полоса «идёт работа». Определённый прогресс (total>0) заполняется как
   обычно (::progress-value выше). */
.process-progress.indeterminate { background-image: linear-gradient(90deg,
      var(--track) 0%, var(--accent-soft) 40%, var(--accent) 50%, var(--accent-soft) 60%, var(--track) 100%);
      background-size: 240% 100%; background-repeat: no-repeat; }
.process-progress.indeterminate::-webkit-progress-bar { background: transparent; }
.process-progress.indeterminate::-webkit-progress-value { background: transparent; }
.process-progress.indeterminate::-moz-progress-bar { background: transparent; }
@media (prefers-reduced-motion: no-preference) {
  .process-progress.indeterminate { animation: process-indeterminate 1.2s linear infinite; }
}
@keyframes process-indeterminate { from { background-position: 120% 0; } to { background-position: -120% 0; } }
.process-status { margin: var(--space-sm) 0; color: var(--muted); }
/* F135: what the finished run actually did — one line per stage that can tell
   "processed" from "skipped as already done". Without it a run that skipped
   everything is indistinguishable from a run that did nothing. */
.process-summary { margin: var(--space-sm) 0; font-size: 0.85rem; color: var(--muted); }
.process-summary-title { display: block; }
.process-summary-line { display: block; margin-left: var(--space-sm); }
/* F84: caption of the current sub-phase, right under the bar — on a phase without a
   percent (clustering) it is the only thing that says the run is alive. */
.process-phase { margin: calc(-1 * var(--space-sm)) 0 var(--space-sm); font-size: 0.85rem;
      color: var(--muted); }
/* F64: инфо-баннер о CPU-профиле (амбер, читается в обеих темах через --ink) */
/* F94: the caches — a quiet block at the bottom of the tab. It is housekeeping, not a
   step of the run, so it looks like a card but is not numbered among the steps. */
/* Width matches the three blocks above (.process-controls, capped at 42rem): this
   one sits OUTSIDE that container and stretched across the full page, which made
   the tab look assembled from two different layouts. */
.cache-block { margin-top: var(--space-md); max-width: 42rem; border: 1px solid var(--line);
      border-radius: var(--radius-md); background: var(--surface); padding: var(--space-md);
      display: flex; flex-direction: column; gap: var(--space-sm); }
/* F119: a sub-heading inside the settings block. The frame-quality knobs are answered
   by three different instruments (CLIP, nothing at all, the VLM) and grouping them
   under one heading said they were all the VLM's — which is what a reader concluded. */
.settings-subhead { margin-top: var(--space-sm); font-size: 0.82rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
.cache-head { display: flex; align-items: baseline; gap: var(--space-sm); flex-wrap: wrap; }
.cache-sizes { color: var(--muted); font-size: 0.85rem; overflow-wrap: anywhere; }
.cache-status { font-size: 0.8rem; color: var(--muted); }
.env-warning { margin-top: var(--space-md); padding: 10px 13px; font-size: 0.85rem;
      border-radius: var(--radius-md); color: var(--ink); line-height: 1.45;
      background: rgba(214, 158, 46, 0.13); border: 1px solid rgba(214, 158, 46, 0.42); }
.stage-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: var(--space-sm) 0; }
.stage-chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: var(--radius-pill);
      font-size: 0.78rem; font-weight: 500; background: var(--chip); color: var(--muted); border: 1px solid var(--line); }
.stage-chip svg { width: 11px; height: 11px; }
.stage-chip.done { background: var(--good-soft); color: var(--good); border-color: transparent; }
.stage-chip.now { background: var(--accent-soft); color: var(--accent); border-color: transparent; font-weight: 600; }

/* --- F77: ручные правки раскладки ------------------------------------- */
/* Два РАЗНЫХ состояния строки, которые нельзя путать: «не трогать» — красная
   рамка (предложение пользователя), «перенесено» — синяя пунктирная. Рамка на
   строке, а не на превью: список плана — таблица, и обводка целой строки заметнее
   при взгляде по сетке. */
tr.override-exclude, tr.override-exclude:hover { outline: 2px solid var(--danger);
      outline-offset: -2px; background: var(--danger-soft); }
tr.override-reassign, tr.override-reassign:hover { outline: 2px dashed var(--accent);
      outline-offset: -2px; background: var(--accent-soft); }
/* F103: «возвращено в фото» — третье состояние, и оно не отрицательное: кадр
   возвращается в обычную раскладку, поэтому зелёная рамка, а не красная. */
tr.override-photo, tr.override-photo:hover { outline: 2px solid var(--good);
      outline-offset: -2px; background: var(--good-soft); }
.override-mark { margin-left: var(--space-sm); }
.override-folder-btn { margin-left: var(--space-sm); font-weight: 500; }
.override-controls { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center;
      margin: 0 0 var(--space-md) 0; }
.override-hint { flex-basis: 100%; font-size: 0.8rem; color: var(--muted); margin: 0; }
.override-status { font-size: 0.8rem; color: var(--danger); }

/* F85c: назначение места группе — та же панель, что и у ручных правок */
.place-controls { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center;
      margin: 0 0 var(--space-md) 0; }
.place-input { min-width: 200px; }
.place-options { max-width: 320px; }
.place-row-btn { margin-left: var(--space-sm); }
.place-manual { margin-left: var(--space-sm); }

.sort-controls { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center; margin: var(--space-md) 0; }
.sort-controls input[type="text"] { flex: 1; min-width: 220px; padding: 8px 10px; }
.sort-dest-hint { flex-basis: 100%; font-size: 0.8rem; color: var(--muted); }
.sort-mode-label { font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px; }

/* --- F104 (layout A): the settings move into a right-hand column so the action row
   can go back to being a row about STARTING a layout. What made the old row dangerous
   was not the number of buttons but their neighbourhood: "Apply" transfers hundreds of
   gigabytes, "Delete selected" erases files and "Expand all" does nothing at all — one
   slip of the mouse costs wildly different amounts. The column drops below the tree on
   a narrow screen (see the media query): the plan itself matters more. --- */
.city-layout { display: grid; grid-template-columns: minmax(0, 1fr);
      gap: var(--space-lg); align-items: start; }
.city-main { min-width: 0; }
/* --- F133: the settings are configuration, not a working surface. Thirteen keys people
   come back to about once a month used to hold a third of the screen at all times; they
   now live behind the gear in the header, in a drawer over the page. Nothing about their
   behaviour moves: the same GET/POST /api/settings, applied without a restart and
   refused while a run is in flight. --- */
.settings-panel { position: fixed; inset: 0; z-index: 2100; display: flex;
      justify-content: flex-end; background: rgba(10,14,22,.5); }
.settings-panel[hidden] { display: none; }
.settings-panel-box { width: min(380px, 100%); height: 100%; overflow-y: auto;
      padding: var(--space-lg); background: var(--bg); border-left: 1px solid var(--line);
      display: flex; flex-direction: column; gap: var(--space-md); }
.settings-panel-head { display: flex; align-items: center; justify-content: space-between;
      gap: var(--space-sm); }
.settings-side { display: flex; flex-direction: column; gap: var(--space-md); }
.settings-block { display: flex; flex-direction: column; gap: var(--space-sm);
      padding: var(--space-md); border: 1px solid var(--line); border-radius: var(--radius-md);
      background: var(--card); }
.settings-head { font-weight: 600; }
.settings-field { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; }
.settings-field input, .settings-field select { padding: 6px 8px; }
/* Deleting lives in the context of a selection: the row appears only once frames are
   ticked, and never stands next to the button that starts a layout. */
.selection-controls { display: flex; gap: var(--space-sm); flex-wrap: wrap; align-items: center;
      margin: var(--space-md) 0; padding: var(--space-sm); border-radius: var(--radius-md);
      border: 1px solid var(--line); background: var(--card); }
.sort-dialog-list { margin: 0 0 var(--space-md) 0; padding-left: var(--space-lg); }
.sort-dialog-list li { margin-bottom: 4px; }

/* --- F93: подтверждение сброса. window.confirm не умеет галочку, а галочка
   «очистить кэш геоданных» обязана быть именно здесь: пользователь вспоминает про
   кэш в момент «хочу переделать начисто», а не в настройках. --- */
.reset-dialog { position: fixed; inset: 0; z-index: 2100; display: flex; align-items: center;
      justify-content: center; padding: var(--space-xl); background: rgba(10,14,22,.86); }
.reset-dialog[hidden] { display: none; }
.reset-dialog-box { max-width: 520px; padding: var(--space-lg); border-radius: var(--radius-md);
      background: var(--surface); box-shadow: var(--shadow-lg); }
.reset-dialog-text { margin: 0 0 var(--space-md) 0; }
.reset-dialog-actions { display: flex; gap: var(--space-sm); justify-content: flex-end;
      margin-top: var(--space-md); }

/* --- лайтбокс (F42): один переиспользуемый оверлей для крупного просмотра --- */
.lightbox { position: fixed; inset: 0; z-index: 2000; display: flex; align-items: center;
      justify-content: center; padding: var(--space-xl); background: rgba(10,14,22,.86);
      cursor: zoom-out; }
.lightbox[hidden] { display: none; }
.lightbox img { width: auto; height: auto; max-width: 100%; max-height: 100%;
      object-fit: contain; cursor: default;
      border-radius: var(--radius-md); box-shadow: var(--shadow-lg); background: var(--surface); }

/* --- F80: листалка кадров видео внутри лайтбокса (у фото скрыта) --- */
.lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); cursor: pointer;
      display: flex; align-items: center; justify-content: center; width: 44px; height: 44px;
      padding: 0; border: 0; border-radius: 50%; color: #fff; background: rgba(10,14,22,.55); }
.lightbox-nav:hover { background: rgba(10,14,22,.85); }
.lightbox-nav[hidden] { display: none; }
.lightbox-nav svg { width: 22px; height: 22px; }
.lightbox-prev { left: var(--space-md); }
.lightbox-next { right: var(--space-md); }
.lightbox-dots { position: absolute; left: 0; right: 0; bottom: var(--space-md);
      display: flex; justify-content: center; gap: 7px; }
.lightbox-dots[hidden] { display: none; }
.lightbox-dot { width: 10px; height: 10px; padding: 0; border-radius: 50%; cursor: pointer;
      border: 1px solid rgba(255,255,255,.75); background: transparent; }
.lightbox-dot.active { background: #fff; }

@media (max-width: 1000px) {
  .settings-panel-box { width: 100%; }
}

@media (max-width: 640px) {
  body { padding: var(--space-md); }
  #clusters-grid, #events-list { grid-template-columns: 1fr; }
  #junk-grid, #animals-grid, #search-grid, #face-grid {
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
  .process-path-row { flex-direction: column; align-items: stretch; }
  .process-path-row input[type="text"] { min-width: 100%; }
}
</style></head><body>
<div class="header-bar">
<div class="brand">
<svg class="brand-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M4 7l8-3 8 3v10l-8 3-8-3V7z"/><path d="M4 7l8 3 8-3"/><path d="M12 10v10"/>
</svg>
<span class="brand-name">Sorta</span>
</div>
<div class="header-controls">
<label class="lang-field">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.7 3.8 6 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-6-3.8-9s1.3-6.3 3.8-9z"/>
</svg>
<select id="lang-select" class="lang-select">{{lang_options}}</select>
</label>
<button type="button" id="theme-toggle-btn" class="btn btn-ghost theme-toggle-btn">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z"/></svg>
<span id="theme-toggle-label">{{theme_dark}}</span></button>
<button type="button" id="settings-toggle-btn" class="btn btn-ghost settings-toggle-btn"
        title="{{settings_open_button}}" aria-expanded="false">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/>
<path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34
1.7 1.7 0 0 0-1 1.56V21a2 2 0 1 1-4 0v-.09A1.7 1.7 0 0 0 8.5 19.3a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06
a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1H2a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 3.7 8.5a1.7 1.7 0 0 0-.34-1.87l-.06-.06
a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 8 4.14 1.7 1.7 0 0 0 9 2.58V2a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1 1.56
1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87V9a1.7 1.7 0 0 0 1.56 1H22a2 2 0 1 1 0 4h-.09
a1.7 1.7 0 0 0-1.56 1z"/></svg>
<span class="settings-toggle-label">{{settings_open_button}}</span></button>
</div>
</div>
<div class="tabs" role="tablist">
<button type="button" class="tab-btn active" id="tab-btn-overview">{{tab_overview}}</button>
<button type="button" class="tab-btn" id="tab-btn-review">{{tab_review}}</button>
<button type="button" class="tab-btn" id="tab-btn-layout">{{tab_layout}}</button>
<button type="button" class="tab-btn" id="tab-btn-slices">{{tab_slices}}</button>
<button type="button" class="tab-btn" id="tab-btn-moves">{{tab_moves}}</button>
</div>
<p id="delete-remember-row" style="display:none"><label><input type="checkbox" id="delete-remember">
{{delete_remember_label}}</label></p>

<section id="tab-overview" class="tab-panel active">
<div id="overview-body"><div class="state-msg state-loading">{{loading}}</div></div>
<p class="process-intro">{{process_intro}}</p>
<div class="process-controls">
<div class="step" id="step-source">
<div class="step-head">
<span class="step-title">{{step_source_title}}</span>
<span class="step-summary" id="step-source-summary"></span>
<button type="button" class="btn btn-ghost step-edit-btn" id="step-source-edit">{{step_change_button}}</button>
</div>
<div class="step-body">
<div class="process-path-row">
<input type="text" id="process-source-dir" placeholder="{{process_path_placeholder}}">
<button type="button" id="process-browse-btn" class="btn btn-ghost">{{process_browse_button}}</button>
<button type="button" id="process-excludes-btn" class="btn btn-ghost">{{excludes_button}}</button>
</div>
<div id="excludes-panel" class="excludes-panel" style="display:none">
<p class="step-title">{{excludes_title}}</p>
<span class="process-toggle-hint">{{excludes_hint}}</span>
<ul class="excludes-legend" id="excludes-legend">
<li><span class="tri-mark">&#9744; {{tri_none_label}}</span> — {{tri_none_hint}}</li>
<li><span class="tri-mark">&#9680; {{tri_layout_label}}</span> — {{tri_layout_hint}}</li>
<li><span class="tri-mark">&#9746; {{tri_scan_label}}</span> — {{tri_scan_hint}}</li>
</ul>
<div id="excludes-tree" class="excludes-tree"></div>
<div class="process-actions">
<button type="button" id="excludes-save-btn" class="btn btn-primary">{{excludes_save_button}}</button>
<button type="button" id="excludes-close-btn" class="btn btn-ghost">{{process_cancel_button}}</button>
<span id="excludes-status" class="override-status"></span>
</div>
</div>
</div>
</div>
<div class="step" id="step-options">
<div class="step-head">
<span class="step-title">{{step_options_title}}</span>
<span class="step-summary" id="step-options-summary"></span>
<button type="button" class="btn btn-ghost step-edit-btn" id="step-options-edit">{{step_change_button}}</button>
</div>
<span class="step-hint">{{step_needs_source_hint}}</span>
<div class="step-body">
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="process-geo-online-checkbox"> {{process_geo_online_label}}</label>
<span class="process-toggle-hint">{{process_geo_online_hint}}</span>
</div>
<div class="cost-block" id="process-costs">
<div class="cost-head">{{costs_title}}</div>
<span class="process-toggle-hint">{{costs_estimate_note}}</span>
<span class="process-toggle-hint busy-hint" style="display:none">{{settings_busy}}</span>
<div class="cost-row">
<span class="cost-name">{{costs_base_label}}</span>
<span class="cost-always">{{costs_always}}</span>
<span class="cost-price" data-cost="base"></span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-faces-checkbox"> {{process_faces_label}}</label>
<span class="cost-price" data-cost="faces"></span>
<span class="process-toggle-hint cost-hint">{{process_faces_hint}}</span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-events-checkbox"> {{process_events_label}}</label>
<span class="cost-price" data-cost="events"></span>
<span class="process-toggle-hint cost-hint">{{process_events_hint}}</span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-pets-checkbox"> {{process_pets_label}}</label>
<span class="cost-price" data-cost="pets"></span>
<span class="process-toggle-hint cost-hint">{{process_pets_hint}}</span>
<span class="cost-child" id="process-pets-verify-row" style="display:none">
<label class="process-toggle-label"><input type="checkbox" id="process-pets-verify-checkbox"> {{process_pets_verify_label}}</label>
<span class="cost-price" data-cost="pets_verify"></span>
<span class="process-toggle-hint cost-hint">{{process_pets_verify_hint}}</span>
<span class="process-toggle-hint cost-hint vlm-off-hint" style="display:none">{{process_needs_deep_hint}}</span>
</span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-deep-checkbox"> {{process_deep_label}}</label>
<span class="cost-price" data-cost="deep"></span>
<span class="process-toggle-hint cost-hint">{{process_deep_hint}}</span>
<span id="process-deep-vlm-missing" class="process-toggle-hint process-toggle-warn" style="display:none">{{process_deep_vlm_missing}}</span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-quality-checkbox"> {{process_quality_label}}</label>
<span class="cost-price" data-cost="quality"></span>
<span class="process-toggle-hint cost-hint">{{process_quality_hint}}</span>
<span class="process-toggle-hint cost-hint vlm-off-hint" style="display:none">{{process_needs_deep_hint}}</span>
<span class="cost-child" id="process-quality-scope-row" style="display:none">
<label class="process-toggle-label" for="process-quality-scope">{{process_quality_scope_label}}
<select id="process-quality-scope"><option value="groups">{{settings_scope_groups}}</option><option value="events">{{settings_scope_events}}</option><option value="faces">{{settings_scope_faces}}</option><option value="all">{{settings_scope_all}}</option></select></label>
</span>
</div>
<div class="cost-row">
<label class="process-toggle-label"><input type="checkbox" id="process-keeper-checkbox"> {{process_keeper_label}}</label>
<span class="cost-price" data-cost="keeper"></span>
<span class="process-toggle-hint cost-hint">{{process_keeper_hint}}</span>
<span class="process-toggle-hint cost-hint vlm-off-hint" style="display:none">{{process_needs_deep_hint}}</span>
</div>
</div>
</div>
</div>
<div class="step" id="step-actions">
<div class="step-head">
<span class="step-title">{{step_actions_title}}</span>
</div>
<span class="step-hint">{{step_needs_source_hint}}</span>
<div class="step-body">
<div class="cost-total" id="process-budget">
<span class="cost-total-label">{{costs_total_label}}</span>
<span class="cost-total-value" id="process-budget-value"></span>
</div>
<div class="process-actions">
<button type="button" id="process-start-btn" class="btn btn-primary">{{process_start_button}}</button>
<button type="button" id="process-cancel-btn" class="btn btn-ghost process-cancel-btn" style="display:none">{{process_cancel_button}}</button>
<button type="button" id="process-reset-btn" class="btn btn-danger">{{process_reset_button}}</button>
</div>
</div>
</div>
</div>
<progress id="process-progress" class="process-progress" max="0" value="0" style="display:none"></progress>
<div id="process-phase" class="process-phase" style="display:none"></div>
<div id="process-stages" class="stage-chips"></div>
<div id="process-status" class="process-status"></div>
<div id="process-summary" class="process-summary"></div>
<div id="env-cpu-warning" class="env-warning" style="display:none">⚠ {{env_cpu_warning}}</div>
<div class="cache-block" id="cache-block">
<div class="cache-head">
<span class="step-title">{{cache_title}}</span>
<span id="cache-sizes" class="cache-sizes">{{loading}}</span>
</div>
<span id="cache-limit" class="cache-sizes"></span>
<span class="process-toggle-hint">{{cache_hint}}</span>
<div class="process-actions">
<button type="button" id="cache-clear-preview-btn" class="btn btn-ghost">{{cache_clear_preview_button}}</button>
<button type="button" id="cache-clear-geo-btn" class="btn btn-ghost">{{cache_clear_geo_button}}</button>
<span id="cache-status" class="cache-status"></span>
</div>
</div>
</section>

<section id="tab-layout" class="tab-panel">
<div class="city-layout">
<div class="city-main">
<div id="layout-review-warning" class="layout-warning" style="display:none">
<span id="layout-review-warning-text"></span>
<button type="button" id="layout-review-goto-btn" class="btn btn-ghost btn-sm">{{layout_review_goto}}</button>
</div>
<div class="sort-controls">
<input type="text" id="sort-dest" placeholder="{{sort_dest_placeholder}}">
<button type="button" id="sort-browse-btn" class="btn btn-ghost">{{process_browse_button}}</button>
<label class="sort-mode-label"><input type="radio" name="sort-mode" value="move" checked> {{sort_mode_move}}</label>
<label class="sort-mode-label"><input type="radio" name="sort-mode" value="copy"> {{sort_mode_copy}}</label>
<button type="button" id="sort-apply-btn" class="btn btn-primary" disabled>{{sort_apply_button}}</button>
<span class="sort-dest-hint">{{sort_dest_hint}}</span>
<span class="sort-dest-hint" id="sort-empty-hint" style="display:none">{{sort_summary_empty}}</span>
<span class="sort-dest-hint busy-hint" style="display:none">{{actions_busy}}</span>
</div>
<progress id="sort-progress" class="process-progress" max="0" value="0" style="display:none"></progress>
<div class="process-actions">
<button type="button" id="sort-cancel-btn" class="btn btn-ghost" style="display:none">{{sort_cancel_button}}</button>
</div>
<div id="sort-status" class="process-status"></div>
<div id="sort-warning" class="process-status"></div>
<div class="tree-controls">
<button type="button" class="btn btn-ghost expand-all-btn">{{expand_all}}</button>
<button type="button" class="btn btn-ghost collapse-all-btn">{{collapse_all}}</button>
</div>
<div class="override-controls">
<button type="button" id="city-override-exclude-btn" class="btn btn-danger" disabled>{{override_exclude_button}}<span id="city-override-count"></span></button>
<select id="city-override-target"><option value="">{{override_target_placeholder}}</option></select>
<button type="button" id="city-override-move-btn" class="btn" disabled>{{override_move_button}}</button>
<button type="button" id="city-override-clear-btn" class="btn btn-ghost" disabled>{{override_clear_button}}</button>
<span id="override-status" class="override-status"></span>
<span class="override-hint busy-hint" style="display:none">{{actions_busy}}</span>
<p class="override-hint">{{override_hint}}</p>
</div>
<div class="place-controls" id="city-place-controls">
<span id="city-place-picker" class="place-picker"></span>
<span id="place-status" class="override-status"></span>
<p class="override-hint">{{place_hint}}</p>
</div>
<div class="selection-controls" id="city-selection-controls" style="display:none">
<button type="button" id="city-delete-selected-btn" class="btn btn-danger" disabled>{{delete_selected}}<span id="city-delete-selected-count"></span></button>
<span class="override-hint">{{selection_delete_hint}}</span>
<span class="override-hint busy-hint" style="display:none">{{actions_busy}}</span>
</div>
<div id="tree-city"><div class="state-msg state-loading">{{loading}}</div></div>
</div>
</div>
</section>

<div id="settings-panel" class="settings-panel" hidden>
<div class="settings-panel-box">
<div class="settings-panel-head">
<span class="settings-head">{{settings_title}}</span>
<button type="button" id="settings-close-btn" class="btn btn-ghost btn-sm">{{settings_close_button}}</button>
</div>
<aside class="settings-side">
<div class="settings-block">
<div class="settings-head">{{settings_title}}</div>
<span class="process-toggle-hint">{{settings_hint}}</span>
<span class="process-toggle-hint">{{settings_costs_moved_hint}}</span>
<label class="settings-field" for="setting-vlm-model">{{settings_vlm_model_label}}
<input type="text" id="setting-vlm-model"></label>
<label class="settings-field" for="setting-vlm-workers">{{settings_vlm_workers_label}}
<input type="number" id="setting-vlm-workers" min="1" max="32" step="1"></label>
<span class="process-toggle-hint">{{settings_vlm_workers_hint}}</span>
<label class="settings-field" for="setting-vlm-max-edge">{{settings_vlm_max_edge_label}}
<input type="number" id="setting-vlm-max-edge" min="128" max="4096" step="1"></label>
<span class="process-toggle-hint">{{settings_vlm_max_edge_hint}}</span>
<div class="settings-head">{{settings_quality_title}}</div>
<span class="process-toggle-hint">{{settings_quality_hint}}</span>
<div class="settings-subhead">{{settings_quality_cheap_title}}</div>
<span class="process-toggle-hint">{{settings_quality_cheap_hint}}</span>
<label class="settings-field" for="setting-features-pet-threshold">{{settings_features_pet_threshold_label}}
<input type="number" id="setting-features-pet-threshold" min="0" max="1" step="0.05"></label>
<label class="settings-field" for="setting-features-sharpness-max-edge">{{settings_features_sharpness_max_edge_label}}
<input type="number" id="setting-features-sharpness-max-edge" min="64" max="4096" step="1"></label>
<span class="process-toggle-hint">{{settings_features_sharpness_max_edge_hint}}</span>
<div class="settings-subhead">{{settings_quality_gate_title}}</div>
<span class="process-toggle-hint">{{settings_quality_gate_hint}}</span>
<label class="settings-field" for="setting-features-sharpness-band-min">{{settings_features_sharpness_band_min_label}}
<input type="number" id="setting-features-sharpness-band-min" min="0" max="10000" step="1"></label>
<label class="settings-field" for="setting-features-sharpness-band-max">{{settings_features_sharpness_band_max_label}}
<input type="number" id="setting-features-sharpness-band-max" min="0" max="10000" step="1"></label>
<span class="process-toggle-hint">{{settings_features_sharpness_band_hint}}</span>
<label class="settings-field" for="setting-features-subject-score-min">{{settings_features_subject_score_min_label}}
<input type="number" id="setting-features-subject-score-min" min="0" max="1" step="0.05"></label>
<span class="process-toggle-hint">{{settings_features_subject_score_min_hint}}</span>
<label class="settings-field" for="setting-imaging-preview-cache-max-gb">{{settings_preview_max_gb_label}}
<input type="number" id="setting-imaging-preview-cache-max-gb" min="0" max="4096" step="1"></label>
<span class="process-toggle-hint">{{settings_preview_max_gb_hint}}</span>
<span class="process-toggle-hint busy-hint" style="display:none">{{settings_busy}}</span>
<div id="settings-status" class="override-status"></div>
</div>
<div class="settings-block">
<div class="settings-head">{{settings_folders_title}}</div>
<label class="settings-field" for="folder-lang-select">{{folder_lang_label}}
<select id="folder-lang-select"><option value="ru">Русский</option><option value="en">English</option><option value="ja">日本語</option></select></label>
<span class="process-toggle-hint">{{settings_folder_lang_hint}}</span>
<span class="process-toggle-hint busy-hint" style="display:none">{{settings_busy}}</span>
</div>
</aside>
</div>
</div>

<section id="tab-review" class="tab-panel">
<p class="process-intro">{{review_intro}}</p>
<div id="review-slices" class="review-slices">
<button type="button" class="btn btn-sm review-slice-btn active" id="review-slice-dupes">{{review_slice_dupes}}<span class="review-slice-count" id="review-count-dupes"></span></button>
<button type="button" class="btn btn-sm review-slice-btn" id="review-slice-blurred">{{review_slice_blurred}}<span class="review-slice-count" id="review-count-blurred"></span></button>
<button type="button" class="btn btn-sm review-slice-btn" id="review-slice-eyes">{{review_slice_eyes}}<span class="review-slice-count" id="review-count-eyes"></span></button>
<button type="button" class="btn btn-sm review-slice-btn" id="review-slice-subject">{{review_slice_subject}}<span class="review-slice-count" id="review-count-subject"></span></button>
</div>
<div id="review-dupes">
<div class="dupes-controls">
<button type="button" id="dupes-save-all-btn" class="btn btn-primary">{{save_all_choices}}</button>
<span id="dupes-save-status"></span>
<span class="override-hint busy-hint" style="display:none">{{actions_busy}}</span>
</div>
<div id="dupes-list"><div class="state-msg state-loading">{{loading}}</div></div>
</div>
<div id="review-flat" style="display:none">
<p id="review-hint" class="override-hint"></p>
<div class="override-controls">
<button type="button" id="review-delete-btn" class="btn btn-danger" disabled>{{review_mark_delete}}<span id="review-selected-count"></span></button>
<button type="button" id="review-keep-btn" class="btn btn-primary" disabled>{{review_mark_keep}}</button>
<button type="button" id="review-clear-btn" class="btn btn-ghost" disabled>{{review_mark_clear}}</button>
<button type="button" id="review-select-all-btn" class="btn btn-ghost">{{review_select_all}}</button>
<button type="button" id="review-select-none-btn" class="btn btn-ghost">{{review_select_none}}</button>
<span id="review-status" class="override-status"></span>
<span class="override-hint busy-hint" style="display:none">{{actions_busy}}</span>
</div>
<div id="review-album" class="album-controls"></div>
<div id="review-grid"><div class="state-msg state-loading">{{loading}}</div></div>
<div class="process-actions">
<button type="button" id="review-more-btn" class="btn btn-ghost" style="display:none">{{review_load_more}}</button>
<span id="review-shown" class="override-hint"></span>
</div>
</div>
</section>

<section id="tab-slices" class="tab-panel">
<p class="process-intro">{{slices_intro}}</p>
<div class="slice-search">
<div class="slice-query-row">
<label class="slice-query-field" for="slice-query">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4.3-4.3"/></svg>
<input type="search" id="slice-query" placeholder="{{search_placeholder}}" disabled>
</label>
<button type="button" id="slice-query-btn" class="btn btn-primary btn-sm" disabled>{{search_button}}</button>
<button type="button" id="slice-query-goto" class="btn btn-ghost btn-sm" style="display:none">{{search_goto_overview}}</button>
</div>
<span id="slice-query-hint" class="process-toggle-hint">{{search_state_checking}}</span>
</div>
<div id="slice-pins" class="review-slices" aria-label="{{slices_pinned_label}}"></div>

<div id="tab-search" class="slice-panel">
<p class="process-intro">{{search_ranking_hint}}</p>
<div id="search-album" class="album-controls"></div>
<div id="search-grid"></div>
<div class="process-actions">
<span id="search-shown" class="override-hint"></span>
</div>
</div>

<div id="tab-person" class="slice-panel">
<div class="cluster-controls">
<button type="button" id="clusters-merge-btn" class="btn btn-primary" disabled>
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="4.5" r="1.6"/><circle cx="18" cy="4.5" r="1.6"/>
<circle cx="12" cy="19.5" r="1.6"/><path d="M6 6v3c0 2.5 2 4 4 4h1M18 6v3c0 2.5-2 4-4 4h-1M12 13v5"/></svg>
{{merge_selected}}</button>
</div>
<div id="clusters-grid"><div class="state-msg state-loading">{{loading}}</div></div>
</div>

<div id="tab-event" class="slice-panel">
<div id="events-list"><div class="state-msg state-loading">{{loading}}</div></div>
</div>

<div id="tab-face" class="slice-panel">
<p class="process-intro">{{face_slices_intro}}</p>
<p id="face-hint" class="override-hint"></p>
<div id="face-album" class="album-controls"></div>
<div id="face-grid"><div class="state-msg state-loading">{{loading}}</div></div>
<div class="process-actions">
<button type="button" id="face-more-btn" class="btn btn-ghost" style="display:none">{{face_load_more}}</button>
<span id="face-shown" class="override-hint"></span>
</div>
</div>

<div id="tab-animal" class="slice-panel">
<p class="process-intro">{{animals_intro}}</p>
<div id="animals-album" class="album-controls"></div>
<div id="animals-grid"><div class="state-msg state-loading">{{loading}}</div></div>
<div class="process-actions">
<button type="button" id="animals-more-btn" class="btn btn-ghost" style="display:none">{{animals_load_more}}</button>
<span id="animals-shown" class="override-hint"></span>
<span id="animals-counted" class="override-hint"></span>
<span id="animals-mark-status" class="album-status"></span>
</div>
</div>

<div id="tab-junk" class="slice-panel">
<p class="process-intro">{{junk_intro}}</p>
<div class="override-controls">
<button type="button" id="junk-restore-btn" class="btn btn-primary" disabled>{{junk_restore_button}}<span id="junk-selected-count"></span></button>
<button type="button" id="junk-select-all-btn" class="btn btn-ghost">{{junk_select_all}}</button>
<button type="button" id="junk-select-none-btn" class="btn btn-ghost">{{junk_select_none}}</button>
<span id="junk-status" class="override-status"></span>
<span class="override-hint busy-hint" style="display:none">{{actions_busy}}</span>
</div>
<div id="junk-doc-hint" class="override-hint" style="display:none">{{junk_document_hint}}</div>
<div id="junk-album" class="album-controls"></div>
<div id="junk-grid"><div class="state-msg state-loading">{{loading}}</div></div>
<div class="process-actions">
<button type="button" id="junk-more-btn" class="btn btn-ghost" style="display:none">{{junk_load_more}}</button>
<span id="junk-shown" class="override-hint"></span>
</div>
</div>

<div id="slice-empty" class="state-msg state-empty" style="display:none">{{slices_empty}}</div>
</section>

<section id="tab-moves" class="tab-panel">
<div id="moves-summary"></div>
<div class="process-actions">
<button type="button" id="undo-btn" class="btn btn-danger" disabled>{{undo_button}}</button>
<button type="button" id="undo-cancel-btn" class="btn btn-ghost" style="display:none">{{undo_cancel_button}}</button>
</div>
<progress id="undo-progress" class="process-progress" max="0" value="0" style="display:none"></progress>
<div id="undo-status" class="process-status"></div>
<div id="undo-stray" class="process-status"></div>
<div class="tree-controls">
<button type="button" class="btn btn-ghost expand-all-btn">{{expand_all}}</button>
<button type="button" class="btn btn-ghost collapse-all-btn">{{collapse_all}}</button>
</div>
<div id="tree-moves"><div class="state-msg state-loading">{{loading}}</div></div>
</section>

<button type="button" id="top-btn" class="btn top-btn" title="{{back_to_top}}">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
{{back_to_top}}</button>
<div id="reset-dialog" class="reset-dialog" hidden>
<div class="reset-dialog-box">
<p class="reset-dialog-text">{{process_reset_confirm}}</p>
<div class="process-option">
<label class="process-toggle-label"><input type="checkbox" id="reset-clear-geo-checkbox"> {{process_reset_clear_geo_label}}</label>
<span class="process-toggle-hint">{{process_reset_clear_geo_hint}}</span>
</div>
<div class="reset-dialog-actions">
<button type="button" id="reset-dialog-cancel" class="btn btn-ghost">{{process_reset_confirm_cancel}}</button>
<button type="button" id="reset-dialog-ok" class="btn btn-danger">{{process_reset_confirm_ok}}</button>
</div>
</div>
</div>
<div id="undo-dialog" class="reset-dialog" hidden>
<div class="reset-dialog-box">
<p class="reset-dialog-text" id="undo-dialog-text"></p>
<div class="reset-dialog-actions">
<button type="button" id="undo-dialog-cancel" class="btn btn-ghost">{{undo_confirm_cancel}}</button>
<button type="button" id="undo-dialog-ok" class="btn btn-danger">{{undo_confirm_ok}}</button>
</div>
</div>
</div>
<div id="sort-dialog" class="reset-dialog" hidden>
<div class="reset-dialog-box">
<p class="reset-dialog-text" id="sort-dialog-text"></p>
<ul class="sort-dialog-list" id="sort-dialog-list"></ul>
<p class="reset-dialog-text" id="sort-dialog-warning"></p>
<div class="reset-dialog-actions">
<button type="button" id="sort-dialog-cancel" class="btn btn-ghost">{{sort_confirm_cancel}}</button>
<button type="button" id="sort-dialog-ok" class="btn btn-danger">{{sort_confirm_ok}}</button>
</div>
</div>
</div>
<div id="lightbox" class="lightbox" hidden title="{{lightbox_close}}">
<img id="lightbox-img" src="" alt="">
<button type="button" id="lightbox-prev" class="lightbox-nav lightbox-prev" hidden
        title="{{frame_prev}}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
        ><path d="M15 18l-6-6 6-6"/></svg></button>
<button type="button" id="lightbox-next" class="lightbox-nav lightbox-next" hidden
        title="{{frame_next}}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
        ><path d="M9 6l6 6-6 6"/></svg></button>
<div id="lightbox-dots" class="lightbox-dots" hidden></div>
</div>
<script>window.I18N = {{i18n_json}};</script>
<script>window.VIDEO_FRAMES = {{video_frames}};</script>
<script>
(function () {
  var I18N = window.I18N;
  var THEME_KEY = "sorta-ui-theme";
  // F80: сколько кадров ленты может листать лайтбокс (SORTA_VIDEO_FRAMES). У
  // короткого ролика кадров реально меньше — это выясняется по первому 404.
  var VIDEO_FRAMES = window.VIDEO_FRAMES || 1;

  // --- инлайн-SVG иконки (U1: без иконочных шрифтов/эмодзи) --------------
  var ICONS = {
    folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 ' +
        '2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/><path d="M12 12v4M10 14h4"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4 12 22l-9-9V4a1 1 ' +
        '0 0 1 1-1h9l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.2"/></svg>',
    merge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="4.5" r="1.6"/>' +
        '<circle cx="18" cy="4.5" r="1.6"/><circle cx="12" cy="19.5" r="1.6"/>' +
        '<path d="M6 6v3c0 2.5 2 4 4 4h1M18 6v3c0 2.5-2 4-4 4h-1M12 13v5"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 ' +
        '2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13"/><path d="M10 11v6M14 11v6"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
    spinner: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round"><circle cx="12" cy="12" r="9" opacity="0.25"/>' +
        '<path d="M21 12a9 9 0 0 0-9-9"/></svg>',
    warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 22 20H2L12 3z"/>' +
        '<path d="M12 10v4M12 17h.01"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>' +
        '<path d="M12 8h.01M11 11.5h1v5.5h1"/></svg>',
    film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" ' +
        'height="14" rx="2"/><path d="M7 5v14M17 5v14M3 12h18"/></svg>',
    pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s7-6.3 7-11a7 7 0 1 ' +
        '0-14 0c0 4.7 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/></svg>',
  };

  function icon(name) {
    var tmp = document.createElement("div");
    tmp.innerHTML = ICONS[name] || "";
    var el = tmp.firstElementChild;
    if (el) el.setAttribute("aria-hidden", "true");
    return el;
  }

  // Кнопка с опциональной иконкой: variant — "primary"/"ghost"/"danger"/null.
  function makeBtn(variant, iconName, label, extraClass) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn" + (variant ? " btn-" + variant : "") + (extraClass ? " " + extraClass : "");
    if (iconName) btn.appendChild(icon(iconName));
    btn.appendChild(document.createTextNode(label));
    return btn;
  }

  // Единый спокойный вид для пустых/загрузочных/ошибочных состояний вкладок.
  function stateEl(kind, text) {
    var div = document.createElement("div");
    div.className = "state-msg state-" + kind;
    var iconName = kind === "error" ? "warn" : kind === "loading" ? "spinner" : "info";
    var ic = icon(iconName);
    if (ic) div.appendChild(ic);
    div.appendChild(document.createTextNode(text));
    return div;
  }

  function wrapTable(table) {
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.appendChild(table);
    return wrap;
  }

  function fmt(template, vals) {
    return template.replace(/\\{(\\w+)\\}/g, function (_, key) {
      return Object.prototype.hasOwnProperty.call(vals, key) ? vals[key] : "";
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.getElementById("theme-toggle-label").textContent =
        theme === "dark" ? I18N.theme_light : I18N.theme_dark;
  }

  function initTheme() {
    var saved = null;
    try { saved = window.localStorage.getItem(THEME_KEY); } catch (e) { saved = null; }
    var theme = saved || ((window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light");
    applyTheme(theme);
  }

  document.getElementById("theme-toggle-btn").addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    var next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
  });

  initTheme();

  var LANG_KEY = "sorta_lang";
  var SUPPORTED_LANGS = ["ru", "en", "ja"];

  function urlWithLang(lang) {
    var url = new URL(window.location.href);
    url.searchParams.set("lang", lang);
    return url.toString();
  }

  function initLang() {
    var select = document.getElementById("lang-select");
    var currentLang = document.documentElement.lang;
    var saved = null;
    try { saved = window.localStorage.getItem(LANG_KEY); } catch (e) { saved = null; }
    if (saved && SUPPORTED_LANGS.indexOf(saved) !== -1 && saved !== currentLang) {
      window.location.replace(urlWithLang(saved));
      return;
    }
    if (select) {
      select.addEventListener("change", function () {
        var next = select.value;
        try { window.localStorage.setItem(LANG_KEY, next); } catch (e) { /* ignore */ }
        window.location.href = urlWithLang(next);
      });
    }
  }

  initLang();

  // F65: the "Folder language" selector (Cities tab) — the OUTPUT language of
  // folders/names, separate from the interface language. Reads the current value
  // from /api/config, and on change persists it (POST /api/config/language) and
  // re-renders the city plan preview with the new folder names.
  function initFolderLang() {
    var select = document.getElementById("folder-lang-select");
    if (!select) return;
    fetch("/api/config")
      .then(function (r) { return r.json(); })
      .then(function (cfg) { if (cfg && cfg.language) select.value = cfg.language; })
      .catch(function () { /* keep the default option */ });
    select.addEventListener("change", function () {
      var next = select.value;
      select.disabled = true;
      postJson("/api/config/language", { language: next }).then(function (resp) {
        select.disabled = false;
        if (resp && resp.ok) {
          renderPlanTab("city", "tree-city");
          settingsStatus(I18N.folder_lang_saved);
        } else {
          settingsStatus((resp && resp.error === "already running")
              ? I18N.settings_busy
              : I18N.settings_error_prefix + ((resp && resp.error) || "error"));
        }
      }).catch(function () { select.disabled = false; });
    });
  }

  initFolderLang();

  // F104: the settings column of the "Cities" tab. Every control writes ONE key
  // through POST /api/settings; the server puts it into the RUNNING config and into
  // config.yaml, so none of this needs `sorta ui` restarted. The whole reason these
  // knobs got an interface is that a text editor plus a restart is not a switch.
  //
  // A rejected save (a run is in progress -> 409, garbage -> 400) is not swallowed:
  // the control is put back to the value the SERVER holds, so the form can never show
  // a setting the tool is not actually using.
  //
  // F138: the knobs that cost a run TIME are not in this list any more — they are on
  // the run screen with their price beside them, and each has exactly one place.
  var SETTING_CONTROLS = [
    { key: "vlm.model", id: "setting-vlm-model", kind: "text" },
    { key: "vlm.workers", id: "setting-vlm-workers", kind: "int" },
    { key: "vlm.max_edge", id: "setting-vlm-max-edge", kind: "int" },
    { key: "features.pet_threshold", id: "setting-features-pet-threshold", kind: "float" },
    { key: "features.sharpness_max_edge", id: "setting-features-sharpness-max-edge", kind: "int" },
    { key: "features.sharpness_band_min", id: "setting-features-sharpness-band-min", kind: "float" },
    { key: "features.sharpness_band_max", id: "setting-features-sharpness-band-max", kind: "float" },
    { key: "features.subject_score_min", id: "setting-features-subject-score-min", kind: "float" },
    { key: "imaging.preview_cache_max_gb", id: "setting-imaging-preview-cache-max-gb", kind: "int" }
  ];
  var settingsValues = {};

  function settingsStatus(text) {
    var el = document.getElementById("settings-status");
    if (el) el.textContent = text;
  }

  function renderSettings(data) {
    if (data) settingsValues = data;
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (!el || !(control.key in settingsValues)) return;
      if (control.kind === "bool") el.checked = !!settingsValues[control.key];
      else el.value = settingsValues[control.key];
    });
  }

  function readSetting(control) {
    var el = document.getElementById(control.id);
    if (!el) return null;
    if (control.kind === "bool") return el.checked;
    if (control.kind === "int") {
      var n = parseInt(el.value, 10);
      // An empty or non-numeric field is sent AS IS: the server owns the range and
      // answers 400, and one refusal in one place beats two copies of the rule.
      return isNaN(n) ? el.value : n;
    }
    if (control.kind === "float") {
      var f = parseFloat(el.value);
      return isNaN(f) ? el.value : f;
    }
    return el.value.trim();
  }

  function saveSetting(control) {
    var body = {};
    body[control.key] = readSetting(control);
    settingsStatus("");
    postJson("/api/settings", body).then(function (resp) {
      if (resp && resp.ok) {
        renderSettings(resp.settings);
        settingsStatus(I18N.settings_saved);
        return;
      }
      renderSettings(null);
      settingsStatus((resp && resp.error === "already running")
          ? I18N.settings_busy
          : I18N.settings_error_prefix + ((resp && resp.error) || "error"));
    }).catch(function () {
      renderSettings(null);
      settingsStatus(I18N.settings_error_prefix + "network");
    });
  }

  function initSettings() {
    fetch("/api/settings")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderSettings(data); })
      .catch(function () { /* the column keeps its empty fields */ });
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (!el) return;
      el.addEventListener("change", function () { saveSetting(control); });
    });
  }

  initSettings();

  // Дерево по списку элементов — осталось для вкладки «Перемещения»: там приходит
  // ОДИН батч (ограниченный по размеру), а не весь план коллекции, поэтому строить
  // его из готового списка по-прежнему нормально. План города/людей/событий с F70
  // ходит другим путём — через агрегат ниже.
  function countFiles(node) {
    var n = node.files.length;
    Object.keys(node.children).forEach(function (k) { n += countFiles(node.children[k]); });
    return n;
  }

  function buildTree(items) {
    var root = { files: [], children: {} };
    items.forEach(function (item) {
      var parts = (item.target_rel || "").split("/");
      parts.pop();
      var node = root;
      parts.forEach(function (part) {
        if (!node.children[part]) node.children[part] = { files: [], children: {} };
        node = node.children[part];
      });
      node.files.push(item);
    });
    return root;
  }

  // Ленивое построение узла: содержимое папки создаётся ТОЛЬКО при первом
  // раскрытии <details> — строки со всеми <img> сразу подвешивали вкладку.
  function renderNode(name, node, depth, renderFilesFn) {
    var renderFn = renderFilesFn || renderFiles;
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    summary.textContent = name + " (" + countFiles(node) + ")";
    details.appendChild(summary);
    var built = false;
    details.addEventListener("toggle", function () {
      if (!details.open || built) return;
      built = true;
      if (node.files.length) details.appendChild(renderFn(node.files));
      Object.keys(node.children).sort().forEach(function (childName) {
        details.appendChild(renderNode(childName, node.children[childName], depth + 1, renderFn));
      });
    });
    return details;
  }

  // F70: дерево строится из АГРЕГАТА (папка -> количество), а не из списка файлов —
  // сервер больше не отдаёт 26 тысяч элементов одним куском. Каждый узел знает
  // суммарное количество файлов в своей ветке; лист (`category`) знает ключ, по
  // которому у сервера запрашивается страница файлов.
  function buildCategoryTree(categories) {
    var root = { count: 0, children: {}, category: null };
    categories.forEach(function (row) {
      var parts = String(row.category || "").split("/");
      var node = root;
      node.count += row.count;
      parts.forEach(function (part, i) {
        if (!node.children[part]) {
          node.children[part] = { count: 0, children: {}, category: null };
        }
        node = node.children[part];
        node.count += row.count;
        if (i === parts.length - 1) node.category = row.category;
      });
    });
    return root;
  }

  // --- удаление отдельного кадра (общий путь для обеих вкладок) --------

  function deletePhoto(fileId, onSuccess) {
    var remember = document.getElementById("delete-remember").checked;
    if (!remember && !window.confirm(I18N.confirm_delete_photo)) return;
    postJson("/api/photo/trash", { file_id: fileId }).then(function (resp) {
      if (resp.trashed && resp.trashed.length) onSuccess();
    });
  }

  // Массовое удаление выбранного (общий путь _trash_files, что и одиночный).
  // onSuccess получает список реально отправленных в корзину file_id.
  function deletePhotos(fileIds, onSuccess) {
    postJson("/api/photos/trash", { file_ids: fileIds }).then(function (resp) {
      if (resp.trashed) {
        onSuccess(resp.trashed.map(function (t) { return t.file_id; }));
      }
    });
  }

  // F145: the rule that used to hold for the layout button alone, stated for everything
  // that WRITES. The server refuses all of it with 409 while a run, a layout or an undo
  // is in flight (see BUSY_REFUSED_ROUTES), and a control that is alive for an action
  // that cannot happen teaches that the interface lies — you find that out by clicking.
  // So: dead while busy, with a line saying why (the `.busy-hint` spans), and alive again
  // the moment it ends, without reloading the page — hence `= busy` everywhere below and
  // never a one-way disable.
  //
  // Declared here, above the first control that uses it: the three flags themselves are
  // set further down (they belong to the polls that own them), and until a poll has run
  // nothing is running, which is what `undefined` means here anyway.
  function uiBusy() {
    return !!(sortRunning || processRunning || undoRunning);
  }

  // Some of these controls have a rule of their own ("nothing selected -> dead") and are
  // redrawn by their own tab. They register that redraw here instead of being listed by
  // id, so the two rules meet in one place and neither can undo the other.
  var busyRefreshers = [];

  function registerBusyRefresh(fn) {
    busyRefreshers.push(fn);
    fn();
  }

  // Переиспользуемый множественный выбор + «Удалить выбранное» для любого
  // контейнера со строками, где есть чекбокс `.row-select` (value=file_id).
  // Делегирование на контейнер — работает и с лениво построенными строками.
  // F104: barId — the row the button lives in; it is SHOWN only while something is
  // selected. A permanently visible "Delete selected" next to "Apply" is a destructive
  // button one row away from the button that moves the whole collection; in the context
  // of a selection it is the obvious action, and nowhere near the layout controls.
  function wireBulkDelete(containerId, buttonId, countId, barId) {
    var container = document.getElementById(containerId);
    var button = document.getElementById(buttonId);
    var countEl = countId ? document.getElementById(countId) : null;
    var barEl = barId ? document.getElementById(barId) : null;
    function checked() {
      return Array.prototype.slice.call(container.querySelectorAll(".row-select:checked"));
    }
    function refresh() {
      var n = checked().length;
      if (countEl) countEl.textContent = n ? " (" + n + ")" : "";
      button.disabled = uiBusy() || n === 0;   // F145: files go to the trash from here
      if (barEl) barEl.style.display = n === 0 ? "none" : "";
    }
    registerBusyRefresh(refresh);
    container.addEventListener("change", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("row-select")) refresh();
    });
    button.addEventListener("click", function () {
      var boxes = checked();
      if (!boxes.length) return;
      var ids = boxes.map(function (b) { return parseInt(b.value, 10); });
      if (!window.confirm(fmt(I18N.confirm_delete_selected, { n: ids.length }))) return;
      deletePhotos(ids, function (trashedIds) {
        var done = {};
        trashedIds.forEach(function (id) { done[id] = true; });
        boxes.forEach(function (b) {
          if (done[parseInt(b.value, 10)]) {
            var tr = b.closest("tr");
            if (tr) tr.remove();
          }
        });
        refresh();
      });
    });
    refresh();
  }

  // Единое поведение превью по всему UI: клик по миниатюре (Города/Дубли/
  // Перемещения/События/Люди) открывает лайтбокс с крупным /preview, а не новую
  // вкладку с сырым /photo. samples/index позволяют листать соседние кадры (для
  // одиночных строк — [fileId]/0). thumbUrl опционален (по умолчанию /thumb/id).
  // F70: раскрытая папка — это до PLAN_PAGE_SIZE строк, то есть столько же
  // одновременных GET /thumb/<id>. Сервер ограничивает число параллельных декодов,
  // но очередь запросов браузера ничем не ограничена. Два простых ограничения:
  // (1) src ставится только когда картинка реально видна (IntersectionObserver);
  // (2) одновременно грузится не больше THUMB_CONCURRENCY штук — остальные ждут в
  // очереди. Слот освобождается по load/error, поэтому очередь не может застрять.
  var THUMB_CONCURRENCY = 6;
  var thumbQueue = [];
  var thumbActive = 0;

  function releaseThumbSlot() {
    thumbActive -= 1;
    pumpThumbQueue();
  }

  function pumpThumbQueue() {
    while (thumbActive < THUMB_CONCURRENCY && thumbQueue.length) {
      var next = thumbQueue.shift();
      thumbActive += 1;
      next.img.addEventListener("load", releaseThumbSlot);
      next.img.addEventListener("error", releaseThumbSlot);
      next.img.src = next.url;
    }
  }

  function queueThumb(img, url) {
    thumbQueue.push({ img: img, url: url });
    pumpThumbQueue();
  }

  var thumbObserver = null;
  if (window.IntersectionObserver) {
    thumbObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        thumbObserver.unobserve(entry.target);
        queueThumb(entry.target, entry.target.getAttribute("data-thumb-src"));
      });
    }, { rootMargin: "200px" });
  }

  function loadThumbWhenVisible(img, url) {
    if (!thumbObserver) { queueThumb(img, url); return; }
    img.setAttribute("data-thumb-src", url);
    thumbObserver.observe(img);
  }

  // F80: у видео плитка получает значок — до этого ролик в сетке был неотличим от
  // фото. Обёртка создаётся ТОЛЬКО для видео: у фото в ячейке остаётся тот же голый
  // <img>, что и раньше, поэтому вёрстка фото-строк не меняется вовсе.
  function videoBadge() {
    var badge = document.createElement("span");
    badge.className = "thumb-video-badge";
    var mark = icon("film");
    if (mark) badge.appendChild(mark);
    badge.appendChild(document.createTextNode(I18N.video_badge));
    return badge;
  }

  function clickableThumb(fileId, samples, index, thumbUrl, isVideo) {
    var img = document.createElement("img");
    loadThumbWhenVisible(img, thumbUrl || ("/thumb/" + fileId));
    img.alt = "";
    img.className = "clickable-thumb";
    img.title = isVideo ? I18N.video_open : I18N.lightbox_open;
    img.addEventListener("click", function () {
      openLightbox(samples || [fileId], index || 0, isVideo ? VIDEO_FRAMES : 0);
    });
    if (!isVideo) return img;
    var wrap = document.createElement("span");
    wrap.className = "thumb-video";
    wrap.appendChild(img);
    wrap.appendChild(videoBadge());
    return wrap;
  }

  // --- F77: ручные правки раскладки (не трогать / перенести в папку) -----
  // Правка только помечает файл в БД: физически ничего не двигается до общей
  // раскладки. Пометка приходит вместе со страницей плана (item.override), поэтому
  // после перерисовки список остаётся размеченным.

  var PLAN_ID_PAGE_SIZE = 1000;  // серверный максимум limit для страницы плана

  // Строка помечается ДВУМЯ разными способами: исключённая (красная рамка) и
  // перенесённая (синяя пунктирная) — это разные состояния, путать их нельзя.
  function markOverrideRow(tr, action, target) {
    tr.classList.remove("override-exclude", "override-reassign", "override-photo");
    var old = tr.querySelector(".override-mark");
    if (old) old.remove();
    tr.dataset.override = action || "";
    var btn = tr.querySelector(".override-row-btn");
    if (btn) {
      btn.textContent = action ? I18N.override_clear_button : I18N.override_exclude_button;
    }
    if (!action) {
      tr.removeAttribute("title");
      return;
    }
    // F103: третье состояние — «возвращено в фото» (правка из вкладки «Не личные
    // фото»). Строка плана должна показывать его отдельно: это не «не трогать» и не
    // «перенесено в папку», а снятие вердикта классификатора.
    var excluded = action === "exclude";
    var restored = action === "photo";
    tr.classList.add(excluded ? "override-exclude"
        : restored ? "override-photo" : "override-reassign");
    var label = excluded ? I18N.override_excluded_mark
        : restored ? I18N.junk_restored_mark
        : fmt(I18N.override_reassigned_mark, { target: target || "" });
    tr.title = label;
    var chip = document.createElement("span");
    chip.className = "chip override-mark " + (excluded ? "chip-danger"
        : restored ? "chip-good" : "chip-accent");
    chip.textContent = label;
    var meta = tr.querySelector(".plan-meta");
    if (meta) meta.appendChild(chip);
  }

  // Пометить уже отрисованные строки внутри scope (контейнер/узел дерева) —
  // «список обновляется без перезагрузки страницы».
  function markRowsOverride(scope, fileIds, action, target) {
    var wanted = {};
    fileIds.forEach(function (id) { wanted[id] = true; });
    Array.prototype.slice.call(scope.querySelectorAll(".row-select")).forEach(function (box) {
      if (!wanted[parseInt(box.value, 10)]) return;
      var tr = box.closest("tr");
      if (tr) markOverrideRow(tr, action, target);
    });
  }

  function overrideStatusEl() {
    return document.getElementById("override-status");
  }

  function applyOverride(action, fileIds, target, onSuccess) {
    var body = { file_ids: fileIds, action: action };
    if (target) body.target = target;
    return postJson("/api/overrides", body).then(function (resp) {
      if (resp && resp.ok) {
        onSuccess(resp.file_ids || fileIds);
      } else {
        overrideStatusEl().textContent = I18N.override_error_prefix +
            ((resp && resp.error) || "");
      }
    }).catch(function (err) {
      overrideStatusEl().textContent = I18N.override_error_prefix + err;
    });
  }

  // Все file_id папки — страницами у сервера, поэтому «не трогать папку» работает
  // и для нераскрытой папки, и для папки больше одной страницы.
  function fetchCategoryIds(mode, category) {
    var ids = [];
    function step(offset) {
      return fetch("/api/plan?mode=" + encodeURIComponent(mode) +
                   "&category=" + encodeURIComponent(category) +
                   "&offset=" + offset + "&limit=" + PLAN_ID_PAGE_SIZE)
        .then(function (r) { return r.json(); })
        .then(function (page) {
          var items = page.items || [];
          items.forEach(function (it) { ids.push(it.file_id); });
          if (items.length && ids.length < page.total) return step(offset + items.length);
          return ids;
        });
    }
    return step(0);
  }

  // Кнопка правки в самой строке: одиночный файл — частый случай, ради него не
  // нужно идти в выделение. Метка/подпись кнопки переключаются по состоянию строки.
  function overrideRowButton(tr, item) {
    var btn = makeBtn(null, null, I18N.override_exclude_button, "btn-sm override-row-btn");
    btn.addEventListener("click", function () {
      var action = tr.dataset.override ? "clear" : "exclude";
      applyOverride(action, [item.file_id], null, function () {
        markOverrideRow(tr, action === "clear" ? null : "exclude", null);
      });
    });
    return btn;
  }

  // Панель над деревом: правка применяется к ВЫДЕЛЕНИЮ (те же чекбоксы
  // .row-select, что и «Удалить выбранное»); одиночный файл — выделение из одного.
  function wireOverrideControls(containerId) {
    var container = document.getElementById(containerId);
    var excludeBtn = document.getElementById("city-override-exclude-btn");
    var moveBtn = document.getElementById("city-override-move-btn");
    var clearBtn = document.getElementById("city-override-clear-btn");
    var select = document.getElementById("city-override-target");
    var countEl = document.getElementById("city-override-count");

    function selectedIds() {
      return Array.prototype.slice.call(container.querySelectorAll(".row-select:checked"))
          .map(function (b) { return parseInt(b.value, 10); });
    }
    function refresh() {
      var n = selectedIds().length;
      countEl.textContent = n ? " (" + n + ")" : "";
      var dead = uiBusy() || n === 0;    // F145: these write `manual_overrides`
      excludeBtn.disabled = dead;
      clearBtn.disabled = dead;
      moveBtn.disabled = dead;
    }
    registerBusyRefresh(refresh);
    function apply(action) {
      var ids = selectedIds();
      if (!ids.length) return;
      var target = null;
      if (action === "reassign") {
        target = select.value;
        if (!target) { window.alert(I18N.override_alert_choose_target); return; }
      }
      applyOverride(action, ids, target, function (applied) {
        markRowsOverride(container, applied, action === "clear" ? null : action, target);
      });
    }
    container.addEventListener("change", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("row-select")) refresh();
    });
    excludeBtn.addEventListener("click", function () { apply("exclude"); });
    moveBtn.addEventListener("click", function () { apply("reassign"); });
    clearBtn.addEventListener("click", function () { apply("clear"); });
    refresh();
  }

  // Список целей переноса = папки текущего плана из уже загруженного агрегата
  // (отдельный эндпойнт не нужен). Перетаскивание плитки в узел дерева не
  // реализуем: дерево ленивое, узел нераскрытой (и потому отсутствующей в DOM)
  // папки не может быть целью drop — список даёт доступ ко ВСЕМ папкам раскладки,
  // как и требует фича.
  function fillOverrideTargets(categories) {
    var select = document.getElementById("city-override-target");
    var previous = select.value;
    select.textContent = "";
    var empty = document.createElement("option");
    empty.value = "";
    empty.textContent = I18N.override_target_placeholder;
    select.appendChild(empty);
    categories.forEach(function (row) {
      var opt = document.createElement("option");
      opt.value = row.category;
      opt.textContent = row.category;
      select.appendChild(opt);
    });
    select.value = previous;
  }

  // --- F85c: место, назначенное человеком, — сразу на всю группу ---------
  // У этих файлов не осталось ни одного сигнала: ни GPS, ни соседей по времени,
  // ни имени папки. Место знает только владелец, поэтому задача не «угадать
  // точнее», а дать назначить его ПАЧКОЙ — событию целиком или исходной папке
  // целиком. Пишется отдельно от places (её geo перезаписывает целиком) и
  // применяется при построении плана; на диске здесь ничего не двигается.

  var PLACE_SEARCH_DELAY = 250;  // мс: поиск идёт по нажатию клавиш, не по каждой

  // Язык интерфейса берём из <html lang>: он уже проставлен сервером, отдельного
  // состояния для этого заводить незачем. (initLang() держит одноимённую локальную
  // переменную — имя здесь другое намеренно.)
  function uiLang() {
    return document.documentElement.getAttribute("lang") || "en";
  }

  // Поле выбора места. Сервер отвечает ТОЧНЫМИ совпадениями по локальной базе
  // (та же пара city_ids_by_name/country_cc_by_name, что и у --where), поэтому
  // список короткий и однозначный: одноимённые города различаются регионом.
  function renderPlacePicker(container) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "place-input";
    input.placeholder = I18N.place_search_placeholder;
    var select = document.createElement("select");
    select.className = "place-options";
    select.disabled = true;
    var results = [];
    var timer = null;

    function fill(list) {
      results = list || [];
      select.textContent = "";
      results.forEach(function (r, i) {
        var opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = r.label;
        select.appendChild(opt);
      });
      select.disabled = results.length === 0;
    }

    function search() {
      var q = input.value.trim();
      if (!q) { fill([]); return; }
      fetch("/api/places/search?lang=" + encodeURIComponent(uiLang()) +
            "&q=" + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (data) { fill(data && data.results); })
        .catch(function () { fill([]); });
    }

    input.addEventListener("input", function () {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(search, PLACE_SEARCH_DELAY);
    });
    container.appendChild(input);
    container.appendChild(select);
    return {
      chosen: function () {
        if (!results.length) return null;
        return results[parseInt(select.value, 10)] || null;
      },
      typed: function () { return input.value.trim(); }
    };
  }

  function placeStatusEl() {
    return document.getElementById("place-status");
  }

  function postPlace(body, statusEl, onDone) {
    return postJson("/api/place", body).then(function (resp) {
      if (!resp || !resp.ok) {
        statusEl.textContent = I18N.place_error_prefix + ((resp && resp.error) || "");
        return;
      }
      var text = fmt(body.action === "clear" ? I18N.place_cleared_status
                                             : I18N.place_assigned_status,
                     { n: resp.affected });
      if (resp.skipped_gps) text += fmt(I18N.place_skipped_gps, { n: resp.skipped_gps });
      statusEl.textContent = text;
      // Кадры с точными координатами не перезаписываются молча: камера знала
      // место в момент съёмки лучше, чем память о поездке. Это отдельное решение,
      // и спрашивают о нём ровно один раз.
      if (resp.skipped_gps && !body.include_gps &&
          window.confirm(fmt(I18N.place_include_gps_confirm, { n: resp.skipped_gps }))) {
        body.include_gps = true;
        return postPlace(body, statusEl, onDone);
      }
      if (onDone) onDone(resp);
    }).catch(function (err) {
      statusEl.textContent = I18N.place_error_prefix + err;
    });
  }

  // Одно действие на группу: подтверждение называет и место, и размер захвата —
  // цена ошибки тем выше, чем крупнее группа.
  function assignPlace(picker, kind, selector, confirmKey, confirmVals, statusEl, onDone) {
    var chosen = picker.chosen();
    if (!chosen) {
      statusEl.textContent = picker.typed() ? I18N.place_not_found : "";
      window.alert(I18N.place_alert_choose);
      return;
    }
    confirmVals.place = chosen.label;
    if (!window.confirm(fmt(I18N[confirmKey], confirmVals))) return;
    postPlace({ kind: kind, selector: selector, action: "assign",
                country: chosen.country, city_geonameid: chosen.city_geonameid },
              statusEl, onDone);
  }

  function clearPlace(kind, selector, confirmKey, confirmVals, statusEl, onDone) {
    if (!window.confirm(fmt(I18N[confirmKey], confirmVals))) return;
    postPlace({ kind: kind, selector: selector, action: "clear" }, statusEl, onDone);
  }

  var cityPlacePicker = null;

  // Кнопка в строке плана: место назначается ИСХОДНОЙ папке кадра целиком — по ней
  // и видно, что кадры одной поездки лежат вместе. Строка с уже назначенным местом
  // предлагает обратное действие, как и кнопка ручных правок рядом.
  function placeRowButton(item) {
    var manual = item.place_confidence === "manual";
    var btn = makeBtn(null, "pin", manual ? I18N.place_clear_button
                                          : I18N.place_folder_button,
        "btn-sm place-row-btn");
    btn.disabled = !item.src_path;
    btn.addEventListener("click", function () {
      var statusEl = placeStatusEl();
      var vals = { dir: item.src_dir || item.src_path };
      var done = function () { renderPlanTab("city", "tree-city"); };
      if (manual) {
        clearPlace("source_dir", item.src_path, "place_folder_clear_confirm",
                   vals, statusEl, done);
      } else {
        assignPlace(cityPlacePicker, "source_dir", item.src_path,
                    "place_folder_confirm", vals, statusEl, done);
      }
    });
    return btn;
  }

  function renderFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdSelect = document.createElement("td");
      var checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "row-select";
      checkbox.value = String(item.file_id);
      checkbox.title = I18N.select_for_delete;
      tdSelect.appendChild(checkbox);
      tr.appendChild(tdSelect);
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url, item.video));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      nameEl.title = item.src_path ? item.src_path + "\\\\" + item.name : item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      tdMeta.className = "plan-meta";
      // Исходная папка идёт первой: по ней чаще всего и видно, верна ли догадка
      // («Колизей» из папки «рускеала» — очевидная ошибка). Полный путь — в тултипе.
      tdMeta.textContent = [item.src_dir, item.date, item.geo, item.category]
          .filter(Boolean).join(" \\u00b7 ");
      if (item.src_path) { tdMeta.title = item.src_path; }
      // F85c: место, выбранное человеком, помечено отдельно — иначе его не отличить
      // от выведенного программой, а это разные по надёжности вещи.
      if (item.place_confidence === "manual") {
        var placeChip = document.createElement("span");
        placeChip.className = "chip chip-good place-manual";
        placeChip.textContent = I18N.place_manual_mark;
        tdMeta.appendChild(placeChip);
      }
      tr.appendChild(tdMeta);
      var tdActions = document.createElement("td");
      tdActions.className = "plan-actions";
      var btnDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnDelete.addEventListener("click", function () {
        deletePhoto(item.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnDelete);
      tdActions.appendChild(overrideRowButton(tr, item));
      tdActions.appendChild(placeRowButton(item));
      tr.appendChild(tdActions);
      // F77: пометка из ответа плана — строка приходит уже размеченной.
      markOverrideRow(tr, item.override || null, item.override_target || null);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  // F70: страница файлов одной папки. Первая грузится при раскрытии узла,
  // следующие — по кнопке «Загрузить ещё»; `total` из ответа показывается как
  // «показано N из M». DOM-узлы существуют только для реально загруженных строк.
  var PLAN_PAGE_SIZE = 200;

  function renderCategoryFiles(mode, category) {
    var wrap = document.createElement("div");
    var status = document.createElement("div");
    status.className = "plan-page-status";
    var moreBtn = makeBtn("ghost", null, I18N.plan_load_more, "btn-sm");
    moreBtn.style.display = "none";
    wrap.appendChild(status);
    wrap.appendChild(moreBtn);
    var loaded = 0;
    var busy = false;

    function loadNext() {
      if (busy) return;
      busy = true;
      moreBtn.disabled = true;
      fetch("/api/plan?mode=" + encodeURIComponent(mode) +
            "&category=" + encodeURIComponent(category) +
            "&offset=" + loaded + "&limit=" + PLAN_PAGE_SIZE)
        .then(function (r) { return r.json(); })
        .then(function (page) {
          var items = page.items || [];
          if (items.length) wrap.insertBefore(renderFiles(items), status);
          loaded += items.length;
          busy = false;
          moreBtn.disabled = false;
          status.textContent = fmt(I18N.plan_shown_of, { n: loaded, all: page.total });
          moreBtn.style.display = (items.length && loaded < page.total) ? "" : "none";
        })
        .catch(function (err) {
          busy = false;
          moreBtn.disabled = false;
          status.textContent = I18N.error_loading_plan + err;
        });
    }

    moreBtn.addEventListener("click", loadNext);
    loadNext();
    return wrap;
  }

  // Ленивое построение узла дерева: содержимое папки (страница файлов + дочерние
  // папки) создаётся ТОЛЬКО при первом раскрытии <details>, и файлы при этом
  // запрашиваются у сервера отдельным запросом — до раскрытия ни одного файла
  // папки в браузере нет вообще.
  function renderCategoryNode(mode, name, node) {
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    summary.textContent = name + " (" + node.count + ")";
    if (node.category) {
      // F77: «не трогать» на папку целиком — кнопка в заголовке категории.
      // Клик внутри <summary> иначе раскрывает/сворачивает узел, поэтому событие
      // до <details> не доходит.
      var folderBtn = makeBtn("danger", null, I18N.override_exclude_folder_button,
          "btn-sm override-folder-btn");
      folderBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (!window.confirm(fmt(I18N.override_exclude_folder_confirm, { n: node.count }))) return;
        folderBtn.disabled = true;
        fetchCategoryIds(mode, node.category).then(function (ids) {
          if (!ids.length) { folderBtn.disabled = false; return; }
          return applyOverride("exclude", ids, null, function (applied) {
            markRowsOverride(details, applied, "exclude", null);
          });
        }).then(function () { folderBtn.disabled = false; })
          .catch(function (err) {
            folderBtn.disabled = false;
            overrideStatusEl().textContent = I18N.override_error_prefix + err;
          });
      });
      summary.appendChild(folderBtn);
    }
    details.appendChild(summary);
    var built = false;
    details.addEventListener("toggle", function () {
      if (!details.open || built) return;
      built = true;
      if (node.category) details.appendChild(renderCategoryFiles(mode, node.category));
      Object.keys(node.children).sort().forEach(function (childName) {
        details.appendChild(renderCategoryNode(mode, childName, node.children[childName]));
      });
    });
    return details;
  }

  // F43: счётчики последнего city-плана.
  // F104: the numbers of the confirmation itself now come from /api/sort/summary (it
  // also knows the volume and what is already in the destination); what stays here is
  // the one question the START button needs answered — is there anything to lay out at
  // all. `cityPlanLoaded` keeps "nothing to lay out" apart from "not counted yet".
  var cityPlanCount = 0;
  var cityPlanLoaded = false;

  // renderPlanTab: дерево папок плана режима (city/person/event) из агрегата —
  // общий код, переиспользуемый всеми план-вкладками (U2).
  function renderPlanTab(mode, containerId) {
    var container = document.getElementById(containerId);
    fetch("/api/plan?mode=" + encodeURIComponent(mode))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var categories = data.categories || [];
        if (mode === "city") {
          // F77: помеченные «не трогать» остаются в списке, но НЕ переезжают —
          // в подтверждении раскладки их считать нельзя.
          cityPlanCount = (data.total || 0) - (data.excluded || 0);
          cityPlanLoaded = true;
          updateBusyControlsDisabled();
          fillOverrideTargets(categories);
        }
        container.textContent = "";
        if (!categories.length) {
          container.appendChild(stateEl("empty", I18N.plan_empty));
          return;
        }
        var root = buildCategoryTree(categories);
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderCategoryNode(mode, name, root.children[name]));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_plan + err));
      });
  }

  cityPlacePicker = renderPlacePicker(document.getElementById("city-place-picker"));
  renderPlanTab("city", "tree-city");
  wireBulkDelete("tree-city", "city-delete-selected-btn", "city-delete-selected-count",
                 "city-selection-controls");
  wireOverrideControls("tree-city");

  document.querySelectorAll(".expand-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = true; });
    });
  });
  document.querySelectorAll(".collapse-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = false; });
    });
  });
  document.getElementById("top-btn").addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // --- вкладки ---------------------------------------------------------

  var dupesLoaded = false;
  var reviewLoaded = false;
  var movesLoaded = false;
  var clustersLoaded = false;
  var eventsLoaded = false;
  var junkLoaded = false;
  var animalsLoaded = false;

  // F133: four tabs named after what a person does with the collection, plus "Moves" as
  // it was. "Overview" holds the state AND the run that produces it; "Slices" holds
  // people/events/animals and the classifier's classes as switchable panels of its own.
  var TAB_NAMES = ["overview", "review", "layout", "slices", "moves"];

  function activateTab(name) {
    TAB_NAMES.forEach(function (t) {
      document.getElementById("tab-btn-" + t).classList.toggle("active", t === name);
      document.getElementById("tab-" + t).classList.toggle("active", t === name);
    });
    // #36: чекбокс «не спрашивать удаление» релевантен только там, где удаляют
    // (Раскладка/Разбор) — на остальных вкладках это шум, прячем.
    document.getElementById("delete-remember-row").style.display =
        (name === "layout" || name === "review") ? "" : "none";
    if (name === "review" && !reviewLoaded) {
      reviewLoaded = true;
      loadReview();
    }
    if (name === "slices") loadSlices();
    if (name === "moves" && !movesLoaded) {
      movesLoaded = true;
      loadMoves();
    }
    // F133: the order warning is re-asked on every open, for the same reason the numbers
    // of "Overview" are — the person has just come back from the Review, and a warning
    // one decision out of date is the one that teaches people to ignore warnings.
    if (name === "layout") loadLayoutWarning();
    // F108: обзор — единственная вкладка без флага «уже загружено». Его открывают
    // ПОСЛЕ прогона, чтобы увидеть изменения, и устаревшая цифра здесь хуже
    // отсутствующей — поэтому числа перезапрашиваются на каждом открытии.
    if (name === "overview") loadOverview();
  }

  TAB_NAMES.forEach(function (t) {
    document.getElementById("tab-btn-" + t).addEventListener("click", function () {
      activateTab(t);
    });
  });

  // --- F133: срезы -----------------------------------------------------------
  // The pin row is BUILT, never written out in the markup: F129 replaces the fixed list
  // with a query, and a row of hand-written buttons would have to be thrown away then.
  // Three of the pins (people/events/animals) show a panel that used to be a tab; the
  // rest are the classifier's classes — products, screenshots, documents and the others —
  // and they all share the one panel `/api/junk` already fills, with its counts, its
  // paging and its rule that a document is never decoded for display.

  // Which classes go first. The order is the product's, not the counter's: a person looks
  // for "products, screenshots, documents", and whichever of them happens to be biggest
  // this month is not a reason to reshuffle the row under them.
  var SLICE_CLASS_ORDER = ["product", "screenshot", "document"];

  var slicePins = [];
  var sliceCurrent = null;
  var slicePending = null;
  var sliceVisibility = { person: false, event: false, animal: false, face: false };
  var junkBucketCounts = [];
  // F152: the counters of the three face pins, `null` for each of them until the faces
  // stage has run — the pin then carries no number at all, because "0 photographs with
  // people" is a claim and "nobody has looked yet" is the truth.
  var faceSliceCounts = {};

  function sliceKeyId(key) {
    return "slice-pin-" + key.replace(/[^a-z0-9]+/g, "-");
  }

  function slicePanelId(key) {
    if (key.indexOf("junk") === 0) return "tab-junk";
    if (key.indexOf("face:") === 0) return "tab-face";
    return "tab-" + key;
  }

  function buildSlicePins() {
    var pins = [];
    // F152: first in the row, and deliberately: on the live collection these are the
    // largest slices there are, and until now the row opened with the smallest.
    if (sliceVisibility.face) {
      FACE_SLICES.forEach(function (name) {
        var count = faceSliceCounts[name];
        pins.push({ key: "face:" + name, label: I18N["face_slice_" + name],
                    count: (count === null || count === undefined) ? undefined : count,
                    faceSlice: name });
      });
    }
    if (sliceVisibility.person) pins.push({ key: "person", label: I18N.tab_person });
    if (sliceVisibility.event) pins.push({ key: "event", label: I18N.tab_event });
    if (sliceVisibility.animal) pins.push({ key: "animal", label: I18N.tab_animal });
    var rest = junkBucketCounts.slice().sort(function (a, b) {
      var ai = SLICE_CLASS_ORDER.indexOf(a.verdict);
      var bi = SLICE_CLASS_ORDER.indexOf(b.verdict);
      if (ai !== bi) return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
      return a.verdict < b.verdict ? -1 : 1;
    });
    rest.forEach(function (b) {
      pins.push({ key: "junk:" + b.verdict, label: junkBucketLabel(b.verdict),
                  count: b.count, bucket: b.verdict });
    });
    if (rest.length) {
      pins.push({ key: "junk", label: I18N.tab_junk, bucket: null,
                  count: rest.reduce(function (acc, b) { return acc + b.count; }, 0) });
    }
    return pins;
  }

  function renderSlicePins() {
    slicePins = buildSlicePins();
    var box = document.getElementById("slice-pins");
    box.textContent = "";
    slicePins.forEach(function (pin) {
      var label = pin.label + (pin.count === undefined ? "" : " (" + pin.count + ")");
      var btn = makeBtn(null, null, label, "btn-sm review-slice-btn");
      btn.id = sliceKeyId(pin.key);
      if (pin.key === sliceCurrent) btn.classList.add("active");
      btn.addEventListener("click", function () { selectSlice(pin.key); });
      box.appendChild(btn);
    });
    // F134: "no slices yet" must not sit under a search that is working — the search
    // line is a slice of its own the moment it has results on screen.
    document.getElementById("slice-empty").style.display =
        (slicePins.length || searchActive) ? "none" : "";
  }

  function selectSlice(key) {
    var pin = null;
    slicePins.forEach(function (p) { if (p.key === key) pin = p; });
    if (!pin) return;
    sliceCurrent = key;
    // F134: the search results are a panel of this tab like any other, so picking a pin
    // puts them away — one panel is visible at a time, whichever one it is.
    searchActive = false;
    var panelId = slicePanelId(key);
    ["tab-person", "tab-event", "tab-animal", "tab-junk", "tab-face",
     "tab-search"].forEach(function (id) {
      document.getElementById(id).classList.toggle("active", id === panelId);
    });
    slicePins.forEach(function (p) {
      var btn = document.getElementById(sliceKeyId(p.key));
      if (btn) btn.classList.toggle("active", p.key === key);
    });
    // F152: three pins, one panel — the junk-bucket arrangement. The page is refetched
    // whenever the slice changes, because the panel holds one slice at a time.
    if (pin.faceSlice !== undefined && (faceSlice !== pin.faceSlice || !faceLoaded)) {
      faceLoaded = true;
      faceSlice = pin.faceSlice;
      loadFaceSlice();
    }
    if (key === "person" && !clustersLoaded) {
      clustersLoaded = true;
      loadClusters();
    }
    if (key === "event" && !eventsLoaded) {
      eventsLoaded = true;
      loadEvents();
    }
    if (key === "animal" && !animalsLoaded) {
      animalsLoaded = true;
      loadAnimals();
    }
    if (pin.bucket !== undefined && (junkBucket !== pin.bucket || !junkLoaded)) {
      junkLoaded = true;
      junkBucket = pin.bucket;
      loadJunk();
    }
  }

  function loadSlices() {
    // F134: the state of the search index is asked for on every open, for the reason the
    // numbers of "Overview" are — the person may have just come back from a run, and a
    // line that stays disabled after the run that enabled it is the worst of both states.
    fetchSearchState();
    // The counters of the class pins come from the route that already serves them, asked
    // for zero items: the counts are the whole answer here. F152 asks its own route the
    // same way — a page of zero cards, three numbers back.
    return Promise.all([
      fetch("/api/junk?offset=0&limit=0")
        .then(function (r) { return r.json(); })
        .then(function (data) { junkBucketCounts = data.buckets || []; })
        .catch(function () {}),
      fetch("/api/face-slices?offset=0&limit=0")
        .then(function (r) { return r.json(); })
        .then(function (data) { applyFaceCounts(data); })
        .catch(function () {}),
    ])
      .then(function () {
        renderSlicePins();
        if (!slicePins.length || searchActive) return;
        var want = slicePending;
        slicePending = null;
        var still = false;
        slicePins.forEach(function (p) {
          if (p.key === (want || sliceCurrent)) still = true;
        });
        selectSlice(still ? (want || sliceCurrent) : slicePins[0].key);
      });
  }

  // A number on "Overview" leads to its SLICE, not merely to the tab holding it — and
  // the pins may not have been built yet, so the wish is remembered and honoured by the
  // load that the tab switch starts.
  function gotoSlice(key) {
    slicePending = key;
    activateTab("slices");
  }

  // --- F134: поиск словами в блоке «Срезы» -----------------------------------
  // The line F133 drew and left disabled. Everything here is arranged around one state
  // that is not a failure: an index nobody has computed yet. `/api/search` answers with
  // the state of the index on EVERY request — including the empty query this tab asks on
  // open, which never reaches the model — so the line can stay disabled with the reason
  // beside it instead of ranking nothing. An empty list of results would read as "you
  // have no photographs like that": a conclusion about somebody's own archive, drawn
  // from a table that was never filled.

  var searchState = null;
  var searchActive = false;   // результаты на экране -> панель среза занята поиском

  function searchStateText(state) {
    if (!state) return I18N.search_state_checking;
    // Two unavailable states, two sentences: "run it" and "run it AGAIN, that index was
    // computed by another model" are different instructions, and the model is named
    // because a reason nobody can act on is not a reason.
    if (state.state === "other_model") {
      return fmt(I18N.search_state_other_model, { model: state.index_model || "?" });
    }
    if (state.state === "empty") return I18N.search_state_empty;
    if (state.state === "partial") {
      return fmt(I18N.search_state_partial, { n: state.indexed, all: state.total });
    }
    return fmt(I18N.search_state_ready, { all: state.total });
  }

  function applySearchState(state) {
    searchState = state;
    var available = !!(state && state.available);
    document.getElementById("slice-query").disabled = !available;
    document.getElementById("slice-query-btn").disabled = !available;
    document.getElementById("slice-query-hint").textContent = searchStateText(state);
    // The way out of both unavailable states is a run of the collection, and the run
    // lives on "Overview" — a reason without the way to it is a dead end.
    document.getElementById("slice-query-goto").style.display =
        (state && !available) ? "" : "none";
  }

  function fetchSearchState() {
    // An empty `q`: the state of the index is the whole question here, and the server
    // loads no model to answer it.
    return fetch("/api/search?q=")
      .then(function (r) { return r.json(); })
      .then(function (data) { applySearchState(data); })
      .catch(function () {});
  }

  function showSearchPanel() {
    searchActive = true;
    ["tab-person", "tab-event", "tab-animal", "tab-junk"].forEach(function (id) {
      document.getElementById(id).classList.remove("active");
    });
    document.getElementById("tab-search").classList.add("active");
    slicePins.forEach(function (p) {
      var btn = document.getElementById(sliceKeyId(p.key));
      if (btn) btn.classList.remove("active");
    });
    sliceCurrent = null;
    document.getElementById("slice-empty").style.display = "none";
  }

  function renderSearchCard(item) {
    var card = document.createElement("div");
    card.className = "search-card";
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // F133's rule, and the search must not become the way around it: a sensitive class
      // is never decoded for display. The server sent no link, so nothing here asks
      // /thumb for one.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "search-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "search-card-meta";
    meta.textContent = item.date || "";
    card.appendChild(meta);
    // The score is on every card because it is the only thing that explains the order —
    // this ranks, it does not classify, and the reader decides where the list stops
    // being about their query.
    var score = document.createElement("span");
    score.className = "search-card-score";
    score.textContent = fmt(I18N.search_score_label,
                            { score: Number(item.score).toFixed(3) });
    card.appendChild(score);
    return card;
  }

  // The album of a query is the album route that already exists: kind='query' and the
  // words themselves as the selector, through the same dry-run-then-confirm path every
  // other album goes through.
  function renderSearchAlbumControls(query) {
    var box = document.getElementById("search-album");
    box.textContent = "";
    if (!query) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      gatherAlbum("query", query, modeSelect.value, null, null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  function renderSearchResults(data) {
    applySearchState(data);
    var grid = document.getElementById("search-grid");
    grid.textContent = "";
    var items = data.items || [];
    items.forEach(function (it) { grid.appendChild(renderSearchCard(it)); });
    if (!items.length) {
      // Never "nothing was found": a usable index ranks everything it holds, so an empty
      // list is a fact about the index and the answer says which one.
      grid.appendChild(stateEl("empty",
          data.available ? I18N.search_no_frames : searchStateText(data)));
    }
    document.getElementById("search-shown").textContent = items.length
        ? fmt(I18N.search_shown_label, { q: data.query, n: items.length }) : "";
    renderSearchAlbumControls(items.length ? data.query : "");
  }

  function runSearch() {
    var q = document.getElementById("slice-query").value.trim();
    // An empty query goes nowhere near the model — not from here and not on the server.
    if (!q || !(searchState && searchState.available)) return;
    showSearchPanel();
    var grid = document.getElementById("search-grid");
    grid.textContent = "";
    grid.appendChild(stateEl("loading", I18N.loading));
    document.getElementById("search-shown").textContent = "";
    renderSearchAlbumControls("");
    return fetch("/api/search?q=" + encodeURIComponent(q))
      .then(function (r) { return r.json(); })
      .then(function (data) { renderSearchResults(data); })
      .catch(function (err) {
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_search + err));
      });
  }

  document.getElementById("slice-query-btn").addEventListener("click", runSearch);
  document.getElementById("slice-query").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); runSearch(); }
  });
  document.getElementById("slice-query-goto").addEventListener("click", function () {
    activateTab("overview");
  });

  // F54: «Люди»/«События» скрыты по умолчанию (без мигания) и раскрываются
  // по факту наличия данных в БД (вариант B, stateless) — фетч дешёвых
  // EXISTS-проверок, вызывается при инициализации и после каждого прогона
  // (refreshTabsAfterProcess), т.к. прогон мог впервые породить кластеры/события.
  // F133: эти три больше не вкладки, а закреплённые срезы, и правило то же —
  // среза нет, пока в базе нечего показать.
  function applyTabVisibility() {
    fetch("/api/tabs/visibility")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        sliceVisibility = {
          person: !!data.person,
          event: !!data.event,
          // F123: "Animals" follows the same rule — the slice exists exactly when there
          // is something to show (features.pets off => no verdicts at all).
          animal: !!data.animal,
          // F152: and the face slices deliberately do NOT follow it. They appear as soon
          // as the index holds a photograph, before any faces run, because their empty
          // state is a SENTENCE ("the faces stage has not run") and a pin that hides
          // itself never gets to say it.
          face: !!data.face,
        };
        renderSlicePins();
        // A slice that has just disappeared must not stay selected — but the person is
        // never pulled off a TAB, so the fallback is the first slice, not another tab.
        // And only while the tab is open: this runs on page load too, where selecting a
        // slice would fetch a grid nobody has asked to see.
        var still = false;
        slicePins.forEach(function (p) { if (p.key === sliceCurrent) still = true; });
        if (!still) {
          sliceCurrent = null;
          if (slicePins.length &&
              document.getElementById("tab-slices").classList.contains("active")) {
            selectSlice(slicePins[0].key);
          }
        }
      })
      .catch(function () {});
  }

  applyTabVisibility();

  // --- F133: предупреждение о порядке на «Раскладке» -------------------------
  // Отмеченные к удалению кадры уезжают в «_delete» ВО ВРЕМЯ sort --apply, тогда же,
  // когда строится канон, а альбомы — hardlink'и ИЗ канона. Собрав альбомы раньше, чем
  // выкинут мусор, человек получит ссылки на то, что решил выбросить.
  //
  // Подсказка, и только: ни одна кнопка раскладки здесь не трогается. Коллекция живая,
  // «собрать» происходит снова и снова, и вернувшемуся за одним альбомом шаги мешают —
  // запертая вкладка стоила бы дороже, чем ошибка, от которой она защищает.
  function renderLayoutWarning(data) {
    var box = document.getElementById("layout-review-warning");
    var pending = data ? Number(data.pending_total || 0) : 0;
    document.getElementById("layout-review-warning-text").textContent =
        fmt(I18N.layout_review_warning, { n: pending });
    box.style.display = pending ? "" : "none";
  }

  function loadLayoutWarning() {
    // slice=dupes carries no items — the counters are the whole answer.
    return fetch("/api/review?slice=dupes&offset=0&limit=0")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderLayoutWarning(data); })
      .catch(function () { renderLayoutWarning(null); });
  }

  document.getElementById("layout-review-goto-btn").addEventListener("click", function () {
    activateTab("review");
  });

  // --- F133: настройки за шестерёнкой ---------------------------------------
  // Ровно та же панель и те же /api/settings — переехало только место, откуда её
  // открывают. Тринадцать ключей, к которым возвращаются раз в месяц, больше не держат
  // треть экрана постоянно.
  function toggleSettingsPanel(open) {
    var panel = document.getElementById("settings-panel");
    panel.hidden = !open;
    document.getElementById("settings-toggle-btn")
        .setAttribute("aria-expanded", open ? "true" : "false");
  }

  document.getElementById("settings-toggle-btn").addEventListener("click", function () {
    toggleSettingsPanel(document.getElementById("settings-panel").hidden);
  });
  document.getElementById("settings-close-btn").addEventListener("click", function () {
    toggleSettingsPanel(false);
  });
  document.getElementById("settings-panel").addEventListener("click", function (e) {
    if (e.target === this) toggleSettingsPanel(false);
  });

  // --- вкладка «Обзор» (F108) --------------------------------------------
  // Все числа приходят одним запросом /api/overview (простые агрегаты по индексу,
  // без построения плана) и рисуются четырьмя карточками: коллекция, место,
  // разбор, раскладка.

  // F145: the empty state draws the SAME rows with a dash where the number will be.
  // Before, it drew an invitation with a button instead — a block of a different height,
  // swapped for the full one the moment the index stopped being empty, i.e. in the middle
  // of a run, right after the `index` stage. Everything below it, the run options among
  // them, jumped down the page while a person was reading. So: the block holds its height
  // from the first paint, the numbers arriving change the text and not the layout, and
  // the list doubles as a statement of what a run will produce.
  var overviewEmpty = false;

  // Числа читают глазами: 7 619 против 7619. toLocaleString берёт разделитель
  // разрядов из локали браузера.
  function overviewNum(n) {
    return Number(n || 0).toLocaleString();
  }

  // F145: the value column of an overview row — the number, or a dash while there is no
  // index to take it from. Separate from overviewNum, which the review slice counters on
  // another tab also use and which must stay a plain formatter.
  function overviewStat(n) {
    if (overviewEmpty) return "\\u2014";
    return overviewNum(n);
  }

  function overviewValue(text, extraClass) {
    var el = document.createElement("span");
    el.className = "overview-value" + (extraClass ? " " + extraClass : "");
    el.textContent = text;
    return el;
  }

  // Число, у которого есть своя вкладка, само является переходом на неё. Ноль
  // ссылкой не делаем: вести на заведомо пустую вкладку не за чем.
  // F126: a review number leads to its SLICE, not just to the tab — the workspace has
  // four of them and landing on the wrong one is the same as landing nowhere.
  // F133: the same is now true of "Slices", where people, events, animals and the
  // classifier's classes live side by side.
  function overviewCount(count, tab, slice) {
    if (!tab || !count) return overviewValue(overviewStat(count));
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "overview-value-link";
    btn.textContent = overviewStat(count);
    btn.title = fmt(I18N.overview_goto_hint, { tab: I18N["tab_" + tab] || tab });
    btn.addEventListener("click", function () {
      if (tab === "slices") {
        gotoSlice(slice);
        return;
      }
      activateTab(tab);
      if (slice) selectReviewSlice(slice);
    });
    return btn;
  }

  // F152: a face-slice number, or a dash when the faces stage never ran. `overviewStat`
  // cannot answer this one — it dashes on an empty INDEX, and here the index is full
  // while this particular question has not been asked of it.
  function overviewFaceCount(count, slice) {
    if (count === null || count === undefined) return overviewValue("\\u2014");
    return overviewCount(count, "slices", slice);
  }

  function overviewRow(label, valueEl, main) {
    var row = document.createElement("div");
    row.className = "overview-row" + (main ? " overview-row-main" : "");
    var name = document.createElement("span");
    name.className = "overview-label";
    name.textContent = label;
    row.appendChild(name);
    row.appendChild(valueEl);
    return row;
  }

  function overviewCard(title) {
    var card = document.createElement("div");
    card.className = "card overview-card";
    var head = document.createElement("h3");
    head.textContent = title;
    card.appendChild(head);
    return card;
  }

  function overviewSubtitle(text) {
    var el = document.createElement("p");
    el.className = "overview-subtitle";
    el.textContent = text;
    return el;
  }

  function overviewNote(text, warn) {
    var el = document.createElement("p");
    el.className = "overview-note" + (warn ? " overview-note-warn" : "");
    el.textContent = text;
    return el;
  }

  function overviewPlaceLabel(key) {
    return I18N["overview_place_" + key] || key;
  }

  function overviewVerdictLabel(key) {
    return key === "photo" ? I18N.overview_verdict_photo : junkBucketLabel(key);
  }

  function overviewSourceLabel(key) {
    return I18N["overview_source_" + key] || key;
  }

  function overviewTierLabel(key) {
    return key ? (I18N["overview_tier_" + key] || key) : I18N.overview_tier_none;
  }

  function overviewCollectionCard(data) {
    var c = data.collection;
    var card = overviewCard(I18N.overview_group_collection);
    card.appendChild(overviewRow(I18N.overview_files, overviewValue(overviewStat(c.files)), true));
    card.appendChild(overviewRow(I18N.overview_photos, overviewValue(overviewStat(c.photos))));
    card.appendChild(overviewRow(I18N.overview_videos, overviewValue(overviewStat(c.videos))));
    card.appendChild(overviewRow(I18N.overview_duplicates,
                                 overviewCount(c.duplicates, "review", "dupes")));
    card.appendChild(overviewRow(I18N.overview_errors, overviewValue(overviewStat(c.errors))));
    card.appendChild(overviewRow(I18N.overview_events,
                                 overviewCount(c.events, "slices", "event")));
    card.appendChild(overviewRow(I18N.overview_animals,
                                 overviewCount(c.animals, "slices", "animal")));
    // F152: the three face slices, each leading to its own pin. They are the only rows
    // here that can be a dash: without a faces run there is no measurement, and a zero
    // would read as "no photograph of yours has a person on it".
    card.appendChild(overviewRow(I18N.overview_with_people,
                                 overviewFaceCount(c.with_people, "face:people")));
    card.appendChild(overviewRow(I18N.overview_group_photos,
                                 overviewFaceCount(c.group_photos, "face:group")));
    card.appendChild(overviewRow(I18N.overview_portraits,
                                 overviewFaceCount(c.portraits, "face:portrait")));
    if (c.faces_reason === "no_faces_run") {
      card.appendChild(overviewNote(I18N.face_no_faces_run));
    }
    // F126: the three slices of the review workspace that have a number of their own.
    card.appendChild(overviewRow(I18N.overview_blurred,
                                 overviewCount(c.blurred, "review", "blurred")));
    card.appendChild(overviewRow(I18N.overview_eyes_closed,
                                 overviewCount(c.eyes_closed, "review", "eyes")));
    card.appendChild(overviewRow(I18N.overview_no_subject,
                                 overviewCount(c.no_subject, "review", "subject")));
    return card;
  }

  function overviewPlaceCard(data) {
    var p = data.place;
    var card = overviewCard(I18N.overview_group_place);
    // Главное число группы: каждый такой кадр уедет в «_Без места» — это и есть
    // качество будущей раскладки, поэтому доля в процентах стоит рядом.
    card.appendChild(overviewRow(
        I18N.overview_no_place,
        overviewValue(overviewEmpty ? overviewStat(p.no_place)
                      : overviewStat(p.no_place) + " (" + p.no_place_percent + "%)"),
        true));
    p.confidence.forEach(function (row) {
      // «unknown» — ровно те кадры, что уже названы строкой выше (правилом
      // раскладки); второй раз их не повторяем.
      if (row.key === "unknown") return;
      card.appendChild(overviewRow(overviewPlaceLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    card.appendChild(overviewNote(I18N.overview_no_place_hint));
    return card;
  }

  function overviewClassesCard(data) {
    var cl = data.classes;
    var card = overviewCard(I18N.overview_group_classes);
    card.appendChild(overviewRow(I18N.overview_classified,
                                 overviewValue(overviewStat(cl.total)), true));
    if (!cl.total) {
      card.appendChild(overviewNote(I18N.overview_not_classified));
      return card;
    }
    cl.verdicts.forEach(function (row) {
      // F133: всё, что не «личное фото», — закреплённый срез своего класса; ведём
      // прямо в него, а не в общий список.
      card.appendChild(overviewRow(
          overviewVerdictLabel(row.key),
          overviewCount(row.count, row.key === "photo" ? null : "slices",
                        "junk:" + row.key)));
    });
    card.appendChild(overviewSubtitle(I18N.overview_by_source));
    cl.sources.forEach(function (row) {
      card.appendChild(overviewRow(overviewSourceLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    card.appendChild(overviewSubtitle(I18N.overview_by_tier));
    cl.tiers.forEach(function (row) {
      card.appendChild(overviewRow(overviewTierLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    // Прогонялся ли глубокий ярус — вопрос, который раньше решался запросом в БД.
    card.appendChild(overviewNote(
        cl.vlm_ran ? I18N.overview_vlm_ran : I18N.overview_vlm_not_ran));
    if (cl.updated_at) {
      card.appendChild(overviewNote(fmt(I18N.overview_updated_at, { at: cl.updated_at })));
    }
    return card;
  }

  function overviewLayoutCard(data) {
    var lay = data.layout;
    var card = overviewCard(I18N.overview_group_layout);
    if (!lay.last) {
      card.appendChild(overviewNote(I18N.overview_layout_none));
      return card;
    }
    var last = lay.last;
    // The batch mode is `city` — the canon — and the tab that builds it is "Layout".
    var mode = last.mode === "city"
        ? I18N.tab_layout : (I18N["tab_" + last.mode] || last.mode);
    var op = last.operation === "copy" ? I18N.overview_op_copy : I18N.overview_op_move;
    card.appendChild(overviewRow(I18N.overview_layout_files,
                                 overviewValue(overviewStat(last.files)), true));
    card.appendChild(overviewRow(I18N.overview_layout_done,
                                 overviewCount(last.done, "moves")));
    card.appendChild(overviewRow(I18N.overview_layout_mode,
                                 overviewValue(mode + " \\u00b7 " + op, "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_started,
                                 overviewValue(last.started_at, "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_finished,
                                 overviewValue(last.finished_at || "\\u2014", "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_dest,
                                 overviewValue(last.dest_root, "overview-text")));
    if (lay.batches > 1) {
      card.appendChild(overviewRow(I18N.overview_layout_batches,
                                   overviewValue(overviewStat(lay.batches))));
    }
    // Незакрытый батч — след прерванного прогона; о нём говорим явно.
    if (last.unfinished || lay.unfinished) {
      card.appendChild(overviewNote(I18N.overview_layout_unfinished, true));
    }
    return card;
  }

  function renderOverview(data) {
    var body = document.getElementById("overview-body");
    body.textContent = "";
    // F145: one flag, read by overviewNum — the four cards below are built the same way
    // either way, and an empty index differs only in what stands in the value column.
    overviewEmpty = !!data.empty;
    if (overviewEmpty) body.appendChild(overviewNote(I18N.overview_empty));
    var groups = document.createElement("div");
    groups.className = "overview-groups";
    groups.appendChild(overviewCollectionCard(data));
    groups.appendChild(overviewPlaceCard(data));
    groups.appendChild(overviewClassesCard(data));
    groups.appendChild(overviewLayoutCard(data));
    body.appendChild(groups);
  }

  function loadOverview() {
    var body = document.getElementById("overview-body");
    body.textContent = "";
    body.appendChild(stateEl("loading", I18N.loading));
    return fetch("/api/overview")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderOverview(data); })
      .catch(function (err) {
        body.textContent = "";
        body.appendChild(stateEl("error", I18N.error_loading_overview + err));
      });
  }

  // --- вкладка «Обработать» (F36: запуск пайплайна из веба + polling) ----

  // F57: чекбоксы deep/geo-online должны стартовать по факту config.yaml
  // (cfg.naming.vlm_enabled / cfg.geo.provider), а не всегда пустыми — иначе
  // сложно понять, что реально включено, и нельзя увидеть текущее состояние
  // до первого клика. vlmAvailable — установлен ли пакет transformers;
  // приглушённая пометка «VLM не установлен» показывается только когда
  // чекбокс отмечен, но пакета нет (запрос VLM ≠ его реальный запуск —
  // junk.classify штатно фолбэчит на CLIP).
  var vlmAvailable = true;

  function updateVlmMissingWarning() {
    var checked = document.getElementById("process-deep-checkbox").checked;
    document.getElementById("process-deep-vlm-missing").style.display =
        (checked && !vlmAvailable) ? "" : "none";
  }

  function applyProcessDefaults() {
    fetch("/api/process/defaults")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        document.getElementById("process-deep-checkbox").checked = !!data.deep;
        document.getElementById("process-geo-online-checkbox").checked = !!data.geo_online;
        document.getElementById("process-pets-checkbox").checked = !!data.pets;
        // F138: the four that moved here out of the settings column start from the
        // config exactly as deep/pets do — the file is where they live, this screen is
        // where one run overrides them.
        document.getElementById("process-pets-verify-checkbox").checked = !!data.pets_verify;
        document.getElementById("process-quality-checkbox").checked = !!data.quality;
        document.getElementById("process-keeper-checkbox").checked = !!data.keeper;
        if (data.quality_scope) {
          document.getElementById("process-quality-scope").value = data.quality_scope;
        }
        vlmAvailable = !!data.vlm_available;
        updateVlmMissingWarning();
        renderCosts();
        updateStepLayout();  // сводка блока «Параметры запуска» — по фактическим галочкам
      })
      .catch(function () {});
  }

  // --- F138: the run budget -----------------------------------------------
  //
  // The prices come from the server ONCE (they depend on the collection, not on the
  // checkboxes) and the sum is recomputed here on every click. Asking the server per
  // click would put a request between a person and a toggle they are still deciding
  // about — and there is nothing to ask: switching a box does not change what the
  // index holds. A run does, so the estimate is re-fetched after one.
  //
  // A missing price is null, and null is a DASH, never a zero: a zero reads as "free",
  // and this screen may not promise twenty minutes with two hours coming. The same rule
  // carries into the sum — an unknown line makes it a floor ("at least"), not a total.
  var costEstimate = null;
  var COST_ROWS = [
    { key: "base", always: true },
    { key: "faces", id: "process-faces-checkbox" },
    { key: "events", id: "process-events-checkbox" },
    { key: "pets", id: "process-pets-checkbox" },
    { key: "pets_verify", id: "process-pets-verify-checkbox",
      parent: "process-pets-checkbox", vlm: true },
    { key: "deep", id: "process-deep-checkbox" },
    { key: "quality", id: "process-quality-checkbox", scoped: true, vlm: true },
    { key: "keeper", id: "process-keeper-checkbox", vlm: true }
  ];

  // --- F145: "Deep analysis (VLM)" is the master switch ----------------------
  //
  // The three lines marked `vlm` above ask the SAME weights this checkbox loads, and
  // until F145 each of them could raise those weights by itself — a run started without
  // the checkbox still spent 20 GB and hours because one key was true in config.yaml.
  // The server now refuses to load a model without it (config.vlm_allowed), and this
  // screen has to say the same thing before the run rather than after it:
  //
  //   * the options stay VISIBLE and go dead. A vanished option reads as "there is no
  //     such thing", and there is;
  //   * their price becomes zero, not the old number. The estimate has to add up to what
  //     the run will actually do;
  //   * nothing is switched on or off automatically. Clearing the master leaves the
  //     subordinate boxes exactly as they were — one movement, one consequence.
  var VLM_SUBORDINATE_IDS = ["process-pets-verify-checkbox", "process-quality-checkbox",
                             "process-quality-scope", "process-keeper-checkbox"];

  function vlmMasterOn() {
    return document.getElementById("process-deep-checkbox").checked;
  }

  function updateVlmSubordinatesDisabled() {
    var off = !vlmMasterOn();
    VLM_SUBORDINATE_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = off || processRunning; }
    });
    document.querySelectorAll(".vlm-off-hint").forEach(function (el) {
      el.style.display = off ? "" : "none";
    });
  }

  function currentQualityScope() {
    return document.getElementById("process-quality-scope").value;
  }

  // The seconds behind one line, or null when this index cannot say. `quality` is the
  // only line whose price depends on a second control — the scope select carries four
  // populations that differ by hours.
  function costSeconds(row) {
    // F145: a subordinate line costs nothing with the master off, whatever the box next
    // to it says — that IS the run, and a dash here would mean "unknown" rather than
    // "free".
    if (row.vlm && !vlmMasterOn()) return 0;
    if (!costEstimate) return null;
    var key = row.scoped ? "quality_" + currentQualityScope() : row.key;
    var value = costEstimate[key];
    return (typeof value === "number") ? value : null;
  }

  function formatCost(seconds) {
    if (seconds === null) return I18N.costs_unknown;
    if (seconds <= 0) return I18N.costs_free;
    if (seconds < 60) return I18N.costs_under_minute;
    var minutes = Math.round(seconds / 60);
    if (minutes < 60) return fmt(I18N.costs_minutes, { minutes: minutes });
    return fmt(I18N.costs_hours,
               { hours: Math.floor(minutes / 60), minutes: minutes % 60 });
  }

  function costRowEnabled(row) {
    if (row.always) return true;
    if (row.parent && !document.getElementById(row.parent).checked) return false;
    return document.getElementById(row.id).checked;
  }

  function renderCosts() {
    // The subordinate controls exist only while their parent is on: a scope for a
    // question nobody is asking is a choice about nothing.
    var petsOn = document.getElementById("process-pets-checkbox").checked;
    var qualityOn = document.getElementById("process-quality-checkbox").checked;
    document.getElementById("process-pets-verify-row").style.display = petsOn ? "" : "none";
    document.getElementById("process-quality-scope-row").style.display =
        qualityOn ? "" : "none";
    updateVlmSubordinatesDisabled();
    var total = 0;
    var unknown = false;
    var vlmOff = !vlmMasterOn();
    COST_ROWS.forEach(function (row) {
      var seconds = costSeconds(row);
      var cell = document.querySelector('[data-cost="' + row.key + '"]');
      // F145: a line the master switch has turned off is priced at zero and says why —
      // "almost free" is what a stage that RUNS and is cheap gets, and this one does not
      // run at all.
      if (cell) {
        cell.textContent = (row.vlm && vlmOff) ? I18N.costs_off : formatCost(seconds);
      }
      if (!costRowEnabled(row)) return;
      if (seconds === null) unknown = true;
      else total += seconds;
    });
    var value = document.getElementById("process-budget-value");
    if (unknown && total <= 0) value.textContent = I18N.costs_unknown;
    else if (unknown) {
      value.textContent = fmt(I18N.costs_total_at_least, { time: formatCost(total) });
    } else value.textContent = formatCost(total);
  }

  function loadCostEstimate() {
    fetch("/api/process/estimate")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        costEstimate = (data && data.seconds) || null;
        renderCosts();
      })
      .catch(function () { renderCosts(); });
  }

  applyProcessDefaults();
  loadCostEstimate();
  document.getElementById("process-deep-checkbox")
      .addEventListener("change", updateVlmMissingWarning);
  ["process-faces-checkbox", "process-events-checkbox", "process-pets-checkbox",
   "process-pets-verify-checkbox", "process-deep-checkbox", "process-quality-checkbox",
   "process-quality-scope", "process-keeper-checkbox"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", renderCosts);
  });
  // Draw once before either answer arrives: dashes and the right nested rows, rather
  // than a block of blank price slots for as long as the two requests take.
  renderCosts();

  // F64: баннер о CPU-профиле (обработка на процессоре — медленно для лиц/VLM).
  fetch("/api/env").then(function (r) { return r.json(); })
    .then(function (data) {
      if (data && !data.gpu_profile) {
        document.getElementById("env-cpu-warning").style.display = "";
      }
    }).catch(function () {});

  var PROCESS_POLL_MS = 1500;
  var processPollTimer = null;

  function processStageLabel(stage) {
    return stage ? (I18N["process_stage_" + stage] || stage) : "";
  }

  // Чипы-этапы (F41): done/now/pending по стадиям пайплайна — тот же порядок,
  // что и сервер (_PIPELINE_STAGE_NAMES), только для отображения. F53/#39:
  // faces/events opt-in — currentProcessStages фиксируется по чекбоксам в
  // момент запуска (сервер фильтрует steps так же), иначе индексы чипов
  // разъедутся со stage_index отфильтрованного прогона.
  var ALL_PROCESS_STAGES = ["index", "geo", "landmarks", "faces", "events", "junk", "phash"];
  var OPTIONAL_PROCESS_STAGES = { faces: true, events: true };
  var currentProcessStages = ALL_PROCESS_STAGES.slice();

  function filterProcessStages(faces, events) {
    var enabled = { faces: faces, events: events };
    return ALL_PROCESS_STAGES.filter(function (name) {
      return !OPTIONAL_PROCESS_STAGES[name] || enabled[name];
    });
  }

  // F135: there is no "Re-run selected" any more — one run button, and the stages
  // skip what is already done by themselves. The /api/process/rerun-optional ROUTE is
  // still there (it is public, see the API documentation): the button went, not it.

  // The last known state of the pipeline. The status poll runs once a tick while the
  // handlers on this tab fire instantly — without this flag they used to re-enable
  // what the tick had just disabled for the duration of a run.
  var processRunning = false;

  // Всё, что задаёт вход пайплайна, на время прогона недоступно: менять источник
  // у уже идущей обработки бессмысленно, а диалог выбора папки ещё и открывает
  // отдельное окно поверх работающего процесса. Галки шагов и ярусов — там же:
  // параметры уходят на сервер один раз, в момент старта, поэтому снятая на
  // середине прогона галка «лица» ничего не отменяет, а выглядит так, будто
  // отменила — и это выясняется через час, когда лица всё-таки посчитались.
  function updateProcessInputsDisabled() {
    ["process-browse-btn", "process-source-dir", "process-excludes-btn",
     "process-deep-checkbox", "process-geo-online-checkbox",
     "process-faces-checkbox", "process-events-checkbox",
     "process-pets-checkbox"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = processRunning; }
    });
    // F145: the options under the master switch have two reasons to be dead and one
    // place that applies both — listing them here as well would re-enable, on the next
    // status tick, boxes the cleared checkbox had just switched off.
    updateVlmSubordinatesDisabled();
  }

  function renderStageChips(data) {
    var container = document.getElementById("process-stages");
    container.textContent = "";
    if (!data.running && !data.finished) return;
    var success = !data.running && data.finished && !data.error;
    currentProcessStages.forEach(function (name, idx) {
      var stepIndex = idx + 1;
      var cls = "pending";
      if (success || stepIndex < data.stage_index) cls = "done";
      else if (data.running && stepIndex === data.stage_index) cls = "now";
      var chip = document.createElement("span");
      chip.className = "stage-chip " + cls;
      if (cls === "done") chip.appendChild(icon("check"));
      chip.appendChild(document.createTextNode(processStageLabel(name)));
      container.appendChild(chip);
    });
  }

  // F84: a stage can name the phase it is in (clustering inside "faces"); an empty
  // phase means the stage reports none — then nothing is drawn and the screen looks
  // exactly as it did before. On an unmeasurable phase (total is unknown, HDBSCAN is
  // one blocking call) there is no honest percent to show, so the caption carries a
  // stopwatch instead: an invented percent would discredit the bar for good.
  function renderProcessPhase(data) {
    var el = document.getElementById("process-phase");
    var key = data.running && !data.cancel_requested ? data.phase : null;
    var label = key ? (I18N["process_phase_" + key] || key) : "";
    if (!label) { el.textContent = ""; el.style.display = "none"; return; }
    el.textContent = data.total > 0 ? label : fmt(I18N.process_phase_elapsed, {
      phase: label,
      seconds: Math.round(data.phase_elapsed || 0),
    });
    el.style.display = "";
  }

  function refreshTabsAfterProcess() {
    dupesLoaded = false;
    reviewLoaded = false;  // F126: a run recomputes every signal the slices are built on
    clustersLoaded = false;
    eventsLoaded = false;
    movesLoaded = false;
    junkLoaded = false;  // F103: прогон junk-яруса меняет состав корзин
    animalsLoaded = false;  // F123: the same run recomputes the animal verdicts
    faceLoaded = false;     // F152: a faces run is what turns the reason into numbers
    renderPlanTab("city", "tree-city");
    applyTabVisibility();
    loadCacheSizes();  // F94: a run is what makes the preview cache grow
    // F138: a run is also what makes the estimate knowable — the deep tier's candidate
    // count, the pet scores and the near-duplicate groups all come out of it. The
    // dashes of a fresh collection turn into numbers here and nowhere else.
    loadCostEstimate();
    // F133: a run recomputes what the order warning is about; refresh it where it is
    // shown rather than waiting for the next open of the tab.
    if (document.getElementById("tab-layout").classList.contains("active")) {
      loadLayoutWarning();
    }
    if (document.getElementById("tab-slices").classList.contains("active")) {
      loadSlices();
    }
    // F108: обзор перечитывается при каждом открытии, но если человек смотрит на
    // него прямо сейчас — обновляем немедленно: прогон только что изменил числа.
    if (document.getElementById("tab-overview").classList.contains("active")) {
      loadOverview();
    }
  }

  // F135: the source of the last run comes back into the field by itself. The
  // browser's own memory (SOURCE_DIR_KEY) covers a page reload but not a fresh profile
  // or a second browser — and "Start" in one click is half of what merging the two
  // buttons is for. A field that already has something in it is left alone: putting a
  // path over what someone is typing is worse than not restoring it at all.
  function adoptRememberedSource(data) {
    var input = document.getElementById("process-source-dir");
    if (!data.source_dir || input.value.trim()) return;
    input.value = data.source_dir;
    rememberSourceDir();
    loadExcludesInfo();
    updateStepLayout();
  }

  // F135: with one button the run walks the whole pipeline every time, and without
  // this summary "everything was already done" reads exactly like "nothing happened".
  // It shows what the CLI prints: how much the stage processed and how much it skipped
  // as already processed. Stages without such a counter are not sent by the server and
  // do not appear here — an invented zero would be a lie, not a line of a report.
  function renderProcessSummary(data) {
    var box = document.getElementById("process-summary");
    box.textContent = "";
    var stats = data.stage_stats || {};
    var names = ALL_PROCESS_STAGES.filter(function (name) { return stats[name]; });
    if (!names.length) return;
    var title = document.createElement("span");
    title.className = "process-summary-title";
    title.textContent = I18N.process_summary_title;
    box.appendChild(title);
    names.forEach(function (name) {
      var line = document.createElement("span");
      line.className = "process-summary-line";
      line.textContent = fmt(I18N.process_summary_stage, {
        stage: processStageLabel(name),
        processed: stats[name].processed || 0,
        skipped: stats[name].skipped || 0,
      });
      box.appendChild(line);
    });
  }

  function renderProcessStatus(data) {
    var startBtn = document.getElementById("process-start-btn");
    var cancelBtn = document.getElementById("process-cancel-btn");
    var bar = document.getElementById("process-progress");
    var statusEl = document.getElementById("process-status");
    processRunning = !!data.running;
    startBtn.disabled = processRunning;
    updateProcessInputsDisabled();
    updateBusyControlsDisabled();  // раскладка и «начать заново» — пока идёт прогон
    adoptRememberedSource(data);
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    bar.style.display = data.running ? "" : "none";
    if (!data.running) bar.classList.remove("indeterminate");
    renderStageChips(data);
    renderProcessPhase(data);
    renderProcessSummary(data);
    if (data.running) {
      if (data.cancel_requested) {
        // отмена запрошена — показываем фидбэк, пока стадия прерывается/дорабатывает
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
        statusEl.textContent = I18N.process_cancel_requested;
        return;
      }
      // #37: total>0 -> определённый прогресс (заполняется); total<=0 (индексация,
      // total неизвестен) -> бегущая indeterminate-полоса + «обработано X».
      if (data.total > 0) {
        bar.classList.remove("indeterminate");
        bar.max = data.total;
        bar.value = data.done || 0;
      } else {
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
      }
      statusEl.textContent = fmt(
        data.total > 0 ? I18N.process_stage_progress : I18N.process_stage_progress_indeterminate, {
        stage: processStageLabel(data.stage),
        index: data.stage_index,
        total: data.stage_total,
        done: data.done,
        all: data.total,
      });
      return;
    }
    if (!data.finished) {
      statusEl.textContent = "";
      return;
    }
    if (data.error) {
      statusEl.textContent = I18N.process_error_prefix + data.error;
    } else if (data.cancel_requested) {
      statusEl.textContent = I18N.process_cancelled;
      refreshTabsAfterProcess();
    } else {
      statusEl.textContent = I18N.process_done;
      refreshTabsAfterProcess();
    }
  }

  function pollProcessStatus() {
    fetch("/api/process/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderProcessStatus(data);
        if (data.running) {
          processPollTimer = setTimeout(pollProcessStatus, PROCESS_POLL_MS);
        }
      });
  }

  document.getElementById("process-start-btn").addEventListener("click", function () {
    var input = document.getElementById("process-source-dir");
    var path = input.value.trim();
    if (!path) { window.alert(I18N.process_enter_path); return; }
    // запуск = «настроено»: оба блока схлопываются, экран остаётся про прогресс
    stepSourceOpen = false;
    stepOptionsOpen = false;
    rememberSourceDir();
    updateStepLayout();
    var deep = document.getElementById("process-deep-checkbox").checked;
    var geoOnline = document.getElementById("process-geo-online-checkbox").checked;
    var faces = document.getElementById("process-faces-checkbox").checked;
    var events = document.getElementById("process-events-checkbox").checked;
    // F123: pets does NOT go into filterProcessStages — it is a setting of the junk
    // stage, not a stage, and the chip row has to show the run that will actually happen.
    var pets = document.getElementById("process-pets-checkbox").checked;
    // F138: three more settings of that same junk stage, and the scope of the quality
    // question. All four are sent EXPLICITLY, ticked or not, so an unticked box forces
    // OFF what config.yaml switched on (the F57 rule) instead of quietly deferring to
    // the file. `pets_verify` needs `pets` — the row is hidden without it — so it is
    // sent as false rather than as a check the junk stage would refuse anyway.
    var petsVerify = pets && document.getElementById("process-pets-verify-checkbox").checked;
    currentProcessStages = filterProcessStages(faces, events);
    postJson("/api/process", {
      source_dir: path, deep: deep, geo_online: geoOnline, faces: faces, events: events,
      pets: pets, pets_verify: petsVerify,
      quality: document.getElementById("process-quality-checkbox").checked,
      quality_scope: currentQualityScope(),
      keeper: document.getElementById("process-keeper-checkbox").checked,
    }).then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("process-status").textContent =
            I18N.process_start_error_prefix + resp.error;
        return;
      }
      if (processPollTimer) clearTimeout(processPollTimer);
      pollProcessStatus();
    });
  });

  // Диалог появляется через секунду-две, а кнопка всё это время оставалась
  // активной — каждый лишний клик открывал ещё один проводник. Блокируем на время
  // запроса; сервер тоже отказывает во втором диалоге (см. _browse_for_folder),
  // потому что вкладку можно открыть и вторую.
  function browseIntoField(btn, apply) {
    if (btn.disabled) { return; }
    btn.disabled = true;
    postJson("/api/browse", {})
      .then(function (resp) { if (resp && resp.path) { apply(resp.path); } })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  }

  document.getElementById("process-browse-btn").addEventListener("click", function () {
    browseIntoField(this, function (path) {
      document.getElementById("process-source-dir").value = path;
      sourceDirChanged();
    });
  });

  // --- F81: «не сканировать» + три блока первой вкладки ------------------

  // Путь помнится между открытиями страницы: этот экран открывают многократно, и
  // вводить один и тот же источник каждый раз — ровно тот штраф, который фича
  // убирает.
  var SOURCE_DIR_KEY = "sorta.sourceDir";
  // Что сейчас исключено под текущим источником — для схлопнутой строки блока
  // «Источник». Два числа хранятся раздельно: «не сканировать» и «не раскладывать» —
  // разные вещи. root пустой = про этот источник ещё не спрашивали.
  var excludesInfo = { root: "", scan: [], count: 0, size: 0,
                       layout: [], layoutCount: 0 };
  var stepSourceOpen = false;
  var stepOptionsOpen = false;

  function currentSourceDir() {
    return document.getElementById("process-source-dir").value.trim();
  }

  function formatSize(bytes) {
    var units = I18N.size_units.split(" ");
    var value = bytes || 0;
    var i = 0;
    while (value >= 1024 && i < units.length - 1) { value = value / 1024; i += 1; }
    return value.toFixed(i === 0 || value >= 100 ? 0 : 1) + " " + units[i];
  }

  function excludesSummaryText() {
    if (excludesInfo.root !== currentSourceDir()) return I18N.excludes_summary_none;
    var parts = [];
    if (excludesInfo.count) {
      parts.push(fmt(I18N.excludes_summary,
                     { count: excludesInfo.count, size: formatSize(excludesInfo.size) }));
    }
    if (excludesInfo.layoutCount) {
      parts.push(fmt(I18N.excludes_summary_layout, { count: excludesInfo.layoutCount }));
    }
    return parts.length ? parts.join(" · ") : I18N.excludes_summary_none;
  }

  function optionsSummaryText() {
    var on = [];
    [["process-deep-checkbox", I18N.process_deep_label],
     ["process-geo-online-checkbox", I18N.process_geo_online_label],
     ["process-faces-checkbox", I18N.process_faces_label],
     ["process-events-checkbox", I18N.process_events_label],
     ["process-pets-checkbox", I18N.process_pets_label],
     ["process-pets-verify-checkbox", I18N.process_pets_verify_label],
     ["process-quality-checkbox", I18N.process_quality_label],
     ["process-keeper-checkbox", I18N.process_keeper_label]].forEach(function (pair) {
      if (document.getElementById(pair[0]).checked) on.push(pair[1]);
    });
    return I18N.step_options_summary_prefix +
        (on.length ? on.join(", ") : I18N.step_options_summary_default);
  }

  // Настроенный блок — одна строка с «изменить», ненастроенный раскрыт. Следующие
  // блоки приглушены пояснением, но НЕ заблокированы: кнопка запуска доступна
  // всегда, когда источник задан (визард штрафует каждый следующий заход).
  // Кнопка шага — переключатель, а не «открыть»: открыл, посмотрел, ничего не менял
  // — и складываешь обратно тем же местом, куда нажал. Сворачивание чисто визуальное
  // и НИЧЕГО не отменяет: введённый путь и снятые галки остаются как есть (иначе это
  // была бы «отмена», а она в шаге, который применяется сразу, только путает).
  // Сворачивать нечего, пока источник не задан, — там кнопка скрыта.
  function updateStepToggle(stepId, buttonId, open, canCollapse) {
    var step = document.getElementById(stepId);
    var button = document.getElementById(buttonId);
    step.classList.toggle("collapsed", canCollapse && !open);
    step.classList.toggle("can-collapse", canCollapse);
    button.textContent = open ? I18N.step_collapse_button : I18N.step_change_button;
    button.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function updateStepLayout() {
    var src = currentSourceDir();
    document.getElementById("step-source-summary").textContent =
        src + " · " + excludesSummaryText();
    document.getElementById("step-options-summary").textContent = optionsSummaryText();
    updateStepToggle("step-source", "step-source-edit", stepSourceOpen, !!src);
    updateStepToggle("step-options", "step-options-edit", stepOptionsOpen, !!src);
    var options = document.getElementById("step-options");
    options.classList.toggle("step-dimmed", !src);
    document.getElementById("step-actions").classList.toggle("step-dimmed", !src);
  }

  function rememberSourceDir() {
    try { window.localStorage.setItem(SOURCE_DIR_KEY, currentSourceDir()); } catch (e) {}
  }

  function excludesInfoOf(src, data) {
    return { root: src, scan: data.skip_scan || [], count: data.count || 0,
             size: data.size || 0, layout: data.skip_layout || [],
             layoutCount: data.layout_count || 0 };
  }

  function loadExcludesInfo() {
    var src = currentSourceDir();
    if (!src) {
      excludesInfo = { root: "", scan: [], count: 0, size: 0, layout: [], layoutCount: 0 };
      updateStepLayout();
      return;
    }
    fetch("/api/source-tree/excludes?path=" + encodeURIComponent(src))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) return;
        excludesInfo = excludesInfoOf(src, data);
        updateStepLayout();
      })
      .catch(function () {});
  }

  function sourceDirChanged() {
    // Шаг НЕ схлопывается на выборе папки. Исключения относятся к конкретному корню
    // и являются частью этого же шага, поэтому сворачивать его в момент, когда
    // пользователь как раз собирается их отметить, — значит заставлять возвращаться
    // назад через «изменить». Схлопнутым шаг стартует только при загрузке страницы с
    // уже запомненным источником: там правда нечего делать.
    stepSourceOpen = true;
    rememberSourceDir();
    loadExcludesInfo();
    loadSourceTree();
    updateStepLayout();
  }

  // F82: три состояния узла — "" обрабатывать, "layout" не раскладывать, "scan" не
  // сканировать. Одно поле на узел, поэтому «отмечено и то и другое» невозможно по
  // построению: переключение на одно автоматически снимает другое.
  var TRI_STATES = ["", "layout", "scan"];

  function triText(state) {
    if (state === "scan") return "☒ " + I18N.tri_scan_label;
    if (state === "layout") return "◐ " + I18N.tri_layout_label;
    return "☐";
  }

  function triHint(state) {
    if (state === "scan") return I18N.tri_scan_hint;
    if (state === "layout") return I18N.tri_layout_hint;
    return I18N.tri_none_hint;
  }

  function setTriState(btn, state) {
    btn.setAttribute("data-state", state);
    btn.textContent = triText(state);
    btn.title = triHint(state);
  }

  function setSubtreeState(ul, state) {
    var marks = ul.querySelectorAll("button.tri-state");
    for (var i = 0; i < marks.length; i++) {
      setTriState(marks[i], state);
      // исключённое поддерево не редактируется по частям: состояние родителя — состояние всего
      marks[i].disabled = !!state;
    }
  }

  function renderExcludesNode(node, states, parentState) {
    var li = document.createElement("li");
    var row = document.createElement("div");
    row.className = "excludes-row";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tri-state";
    btn.setAttribute("data-rel", node.rel);
    setTriState(btn, parentState || states[node.rel] || "");
    btn.disabled = !!parentState;
    row.appendChild(btn);
    row.appendChild(document.createTextNode(node.name));
    var meta = document.createElement("span");
    meta.className = "excludes-meta";
    meta.textContent = fmt(I18N.excludes_folder_meta,
                           { count: node.files, size: formatSize(node.size) });
    row.appendChild(meta);
    li.appendChild(row);
    var ul = null;
    if (node.children && node.children.length) {
      ul = document.createElement("ul");
      node.children.forEach(function (child) {
        ul.appendChild(renderExcludesNode(child, states, btn.getAttribute("data-state")));
      });
      li.appendChild(ul);
    }
    btn.addEventListener("click", function () {
      var next = TRI_STATES[(TRI_STATES.indexOf(btn.getAttribute("data-state")) + 1)
                            % TRI_STATES.length];
      setTriState(btn, next);
      if (ul) setSubtreeState(ul, next);
    });
    return li;
  }

  function renderExcludesTree(data) {
    var container = document.getElementById("excludes-tree");
    container.textContent = "";
    var states = {};
    (data.skip_layout || []).forEach(function (rel) { states[rel] = "layout"; });
    // «не сканировать» пишется вторым: при странном файле, где папка попала в оба
    // раздела, сервер уже решил в пользу scan — дерево не должно спорить с ним
    (data.skip_scan || []).forEach(function (rel) { states[rel] = "scan"; });
    var children = (data.tree && data.tree.children) || [];
    if (!children.length) {
      container.appendChild(stateEl("empty", I18N.excludes_empty));
      return;
    }
    var ul = document.createElement("ul");
    children.forEach(function (child) {
      ul.appendChild(renderExcludesNode(child, states, ""));
    });
    container.appendChild(ul);
    if (data.truncated) {
      // ответ ограничен — говорим об этом прямо, а не молча показываем часть дерева
      var note = document.createElement("p");
      note.className = "process-toggle-hint";
      note.textContent = fmt(I18N.excludes_truncated, { limit: data.limit });
      container.appendChild(note);
    }
  }

  function collectExcludes() {
    // только верхние отмеченные: потомки отмеченной папки заблокированы и не нужны
    var result = { skip_scan: [], skip_layout: [] };
    var marks = document.getElementById("excludes-tree")
        .querySelectorAll("button.tri-state");
    for (var i = 0; i < marks.length; i++) {
      if (marks[i].disabled) continue;
      var state = marks[i].getAttribute("data-state");
      if (state === "scan") result.skip_scan.push(marks[i].getAttribute("data-rel"));
      else if (state === "layout") result.skip_layout.push(marks[i].getAttribute("data-rel"));
    }
    return result;
  }

  // Вынесено из обработчика кнопки: дерево показывается и по кнопке, и сразу после
  // выбора папки — «выбрал источник, вижу его структуру» это один шаг, а не два.
  function loadSourceTree(announce) {
    var src = currentSourceDir();
    if (!src) { if (announce) { window.alert(I18N.process_enter_path); } return; }
    document.getElementById("excludes-panel").style.display = "";
    document.getElementById("excludes-status").textContent = "";
    var container = document.getElementById("excludes-tree");
    container.textContent = "";
    container.appendChild(stateEl("loading", I18N.loading));
    fetch("/api/source-tree?path=" + encodeURIComponent(src))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) {
          container.textContent = "";
          container.appendChild(stateEl(
              "error", I18N.excludes_error_prefix + ((data && data.error) || "")));
          return;
        }
        renderExcludesTree(data);
      })
      .catch(function (e) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.excludes_error_prefix + e));
      });
  }

  document.getElementById("process-excludes-btn").addEventListener("click", function () {
    loadSourceTree(true);
  });

  document.getElementById("excludes-save-btn").addEventListener("click", function () {
    var src = currentSourceDir();
    if (!src) return;
    var statusEl = document.getElementById("excludes-status");
    var picked = collectExcludes();
    postJson("/api/source-tree/excludes",
             { root: src, skip_scan: picked.skip_scan, skip_layout: picked.skip_layout })
      .then(function (resp) {
        if (!resp || resp.error) {
          statusEl.textContent =
              I18N.excludes_save_error_prefix + ((resp && resp.error) || "");
          return;
        }
        excludesInfo = excludesInfoOf(src, resp);
        statusEl.textContent = I18N.excludes_saved;
        updateStepLayout();
      })
      .catch(function (e) { statusEl.textContent = I18N.excludes_save_error_prefix + e; });
  });

  document.getElementById("excludes-close-btn").addEventListener("click", function () {
    document.getElementById("excludes-panel").style.display = "none";
  });

  document.getElementById("step-source-edit").addEventListener("click", function () {
    stepSourceOpen = !stepSourceOpen;
    // Свернули источник — панель исключений уходит вместе с ним: она часть этого
    // шага и висеть отдельно от него не должна.
    if (!stepSourceOpen) {
      document.getElementById("excludes-panel").style.display = "none";
    }
    updateStepLayout();
  });

  document.getElementById("step-options-edit").addEventListener("click", function () {
    stepOptionsOpen = !stepOptionsOpen;
    updateStepLayout();
  });

  document.getElementById("process-source-dir")
      .addEventListener("input", updateStepLayout);
  document.getElementById("process-source-dir")
      .addEventListener("change", sourceDirChanged);
  ["process-deep-checkbox", "process-geo-online-checkbox", "process-faces-checkbox",
   "process-events-checkbox", "process-pets-checkbox",
   "process-pets-verify-checkbox", "process-quality-checkbox",
   "process-keeper-checkbox"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", updateStepLayout);
  });

  (function restoreSourceDir() {
    var input = document.getElementById("process-source-dir");
    if (!input.value.trim()) {
      var saved = null;
      try { saved = window.localStorage.getItem(SOURCE_DIR_KEY); } catch (e) { saved = null; }
      if (saved) input.value = saved;
    }
    loadExcludesInfo();
    updateStepLayout();
  })();

  document.getElementById("process-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("process-status").textContent = I18N.process_cancel_requested;
    renderProcessPhase({});  // the phase caption is stale now, do not wait for a tick
    postJson("/api/process/cancel", {});
  });

  // F93: сброс подтверждается своим диалогом, а не window.confirm — в нём живёт
  // галочка «также очистить кэш геоданных». Галочка каждый раз сбрасывается: очистка
  // кэша — разовое решение, а не режим, который тихо остаётся включённым.
  var resetDialogEl = document.getElementById("reset-dialog");
  var resetClearGeoEl = document.getElementById("reset-clear-geo-checkbox");

  function closeResetDialog() {
    resetDialogEl.hidden = true;
  }

  document.getElementById("process-reset-btn").addEventListener("click", function () {
    resetClearGeoEl.checked = false;
    resetDialogEl.hidden = false;
  });

  document.getElementById("reset-dialog-cancel").addEventListener("click", closeResetDialog);

  resetDialogEl.addEventListener("click", function (e) {
    if (e.target === resetDialogEl) closeResetDialog();  // клик по фону — отмена
  });

  document.getElementById("reset-dialog-ok").addEventListener("click", function () {
    var clearGeo = resetClearGeoEl.checked;
    closeResetDialog();
    postJson("/api/process/reset", { clear_geo: clearGeo }).then(function (resp) {
      var statusEl = document.getElementById("process-status");
      if (resp && resp.error) {
        statusEl.textContent = I18N.process_reset_error_prefix + resp.error;
        return;
      }
      statusEl.textContent = clearGeo ? I18N.process_reset_done_geo : I18N.process_reset_done;
      // F135: the summary of the last run counted files of an index that is gone now.
      renderProcessSummary({});
      refreshTabsAfterProcess();
    });
  });

  // --- F94: the caches ------------------------------------------------------
  // Sizes are asked for rarely and on purpose: the preview cache is tens of
  // thousands of files, so the status poll must never touch it. Page load, the end
  // of a run and a clear are the only three moments the number can have changed.
  var cacheInfo = { previewBytes: 0, previewFiles: 0, geoEntries: 0 };

  function applyCacheInfo(data) {
    if (!data || !data.preview || !data.geo) return;
    cacheInfo.previewBytes = data.preview.bytes || 0;
    cacheInfo.previewFiles = data.preview.files || 0;
    cacheInfo.previewMaxGb = data.preview.max_gb || 0;
    cacheInfo.geoEntries = data.geo.entries || 0;
    document.getElementById("cache-sizes").textContent = fmt(I18N.cache_sizes, {
      preview: formatSize(cacheInfo.previewBytes),
      files: cacheInfo.previewFiles,
      geo: cacheInfo.geoEntries,
    });
    // F117: 0 is "no ceiling", a state — not a limit of zero, which would read as a
    // cache that may hold nothing at all.
    var limitEl = document.getElementById("cache-limit");
    if (cacheInfo.previewMaxGb > 0) {
      var used = cacheInfo.previewBytes / (cacheInfo.previewMaxGb * 1e9) * 100;
      limitEl.textContent = fmt(I18N.cache_limit, {
        limit: cacheInfo.previewMaxGb,
        percent: Math.round(used),
      });
    } else {
      limitEl.textContent = I18N.cache_no_limit;
    }
  }

  var cacheSizesPending = false;

  function loadCacheSizes() {
    // Page load and "the run has finished" can land together (a reload right after a
    // run) — one walk of the preview directory per moment, not two.
    if (cacheSizesPending) return;
    cacheSizesPending = true;
    fetch("/api/cache").then(function (r) { return r.json(); })
      .then(function (data) { cacheSizesPending = false; applyCacheInfo(data); })
      .catch(function () { cacheSizesPending = false; });
  }

  // Both clears are irreversible and neither is free, so each states its own price
  // before it happens — the preview one that the next run pays 336 ms per frame
  // instead of 73, the geo one that with provider: online it pays the network again.
  function clearCache(target, confirmText, doneText) {
    if (!window.confirm(confirmText)) return;
    var statusEl = document.getElementById("cache-status");
    statusEl.textContent = "";
    postJson("/api/cache/clear", { target: target }).then(function (resp) {
      if (!resp || resp.error) {
        statusEl.textContent =
            I18N.cache_clear_error_prefix + ((resp && resp.error) || "");
        return;
      }
      statusEl.textContent = fmt(doneText, { n: resp.removed || 0 });
      applyCacheInfo(resp.cache);
    });
  }

  document.getElementById("cache-clear-preview-btn").addEventListener("click", function () {
    clearCache("preview",
               fmt(I18N.cache_clear_preview_confirm,
                   { preview: formatSize(cacheInfo.previewBytes) }),
               I18N.cache_clear_preview_done);
  });

  document.getElementById("cache-clear-geo-btn").addEventListener("click", function () {
    clearCache("geo",
               fmt(I18N.cache_clear_geo_confirm, { geo: cacheInfo.geoEntries }),
               I18N.cache_clear_geo_done);
  });

  loadCacheSizes();

  pollProcessStatus();

  // --- вкладка «Города»: apply раскладки (F43) ----------------------------
  // Дерево-превью вкладки — уже dry-run; кнопка сразу открывает подтверждение
  // (текст зависит от режима/dest), только потом POST /api/sort. Фон +
  // прогресс — тот же паттерн polling, что и «Обработать» (F36) выше.

  var SORT_POLL_MS = 1500;
  var sortPollTimer = null;

  function updateSortApplyBtnStyle() {
    var btn = document.getElementById("sort-apply-btn");
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var move = !checked || checked.value === "move";
    btn.classList.toggle("btn-danger", move);
    btn.classList.toggle("btn-primary", !move);
  }

  document.querySelectorAll('input[name="sort-mode"]').forEach(function (r) {
    r.addEventListener("change", updateSortApplyBtnStyle);
  });
  updateSortApplyBtnStyle();

  // F104: before a layout the user sees NUMBERS, not a question "are you sure?". They
  // come from /api/sort/summary — the same built plan the tab's tree is drawn from, so
  // the dialog cannot name a figure the tab does not show.
  var sortDialogEl = document.getElementById("sort-dialog");

  function sortSummaryLines(data, dest, mode) {
    var lines = [fmt(I18N.sort_summary_dest,
                     { dest: data.dest || dest || I18N.sort_dest_inplace_label })];
    lines.push(mode === "move" ? I18N.sort_summary_mode_move : I18N.sort_summary_mode_copy);
    lines.push(fmt(I18N.sort_summary_files,
                   { n: data.files, dirs: data.dirs, size: formatSize(data.bytes) }));
    if (data.dest === null) lines.push(I18N.sort_summary_existing_unknown);
    else if (!data.dest_existing) lines.push(I18N.sort_summary_existing_none);
    else lines.push(fmt(I18N.sort_summary_existing,
                        { n: data.dest_existing, same: data.dest_same }));
    if (data.products || data.documents) {
      lines.push(fmt(I18N.sort_summary_service,
                     { products: data.products, documents: data.documents }));
    }
    return lines;
  }

  function openSortDialog(data, dest, mode) {
    document.getElementById("sort-dialog-text").textContent = I18N.sort_confirm_title;
    var list = document.getElementById("sort-dialog-list");
    list.textContent = "";
    sortSummaryLines(data, dest, mode).forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      list.appendChild(li);
    });
    // A line of its own goes to what the numbers cannot say: an in-place run
    // restructures the SOURCE tree rather than a copy in a separate folder.
    document.getElementById("sort-dialog-warning").textContent =
        dest ? (mode === "move" ? I18N.sort_confirm_move : I18N.sort_confirm_copy)
             : I18N.sort_confirm_inplace;
    sortDialogEl.hidden = false;
  }

  function closeSortDialog() {
    sortDialogEl.hidden = true;
  }

  // Раскладка во время прогона запрещена и на сервере (409 «process is running»
  // под общим busy_lock), но кнопка до этого оставалась живой — про запрет
  // узнавали кликом. Хуже другое: на середине прогона плана попросту нет.
  // geo чистит places перед записью, junk ещё не заполнил media_class — то есть
  // раскладка, начатая сейчас, разложила бы коллекцию по недостроенному индексу.
  var sortRunning = false;

  // Кнопки, которые обязаны быть мертвы, пока занят ЛЮБОЙ из двух процессов
  // (прогон пайплайна или раскладка). Сервер их и так отбивает 409 под общим
  // busy_lock, но «Начать заново» сперва показывает страшное подтверждение и
  // только потом ошибку, а раскладка на середине прогона разложила бы коллекцию
  // по недостроенному индексу (places очищены, media_class ещё пуст).
  // F94: очистки кэшей — там же: подтверждение с ценой действия ради ответа 409
  // ничем не лучше, а превью на середине прогона пишет тот самый шаг.
  // F97: откат — третий такой же процесс: он двигает файлы на диске, совмещать его
  // с раскладкой или прогоном нельзя (сервер отбивает 409 под тем же busy_lock).
  // undoAvailable/undoBatchInfo наполняет манифест «Перемещений» (applyUndoAvailability):
  // кнопка отката жива только когда есть что откатывать, а диалог берёт числа оттуда же.
  var undoRunning = false;
  var undoAvailable = false;
  var undoBatchInfo = null;

  function updateBusyControlsDisabled() {
    var busy = uiBusy();
    ["sort-browse-btn", "sort-dest",
     "process-reset-btn",
     "cache-clear-preview-btn", "cache-clear-geo-btn",
     // F145: saving the whole set of duplicate choices writes `dedup_choice` for every
     // group on the tab at once — the largest single write the review side has.
     "dupes-save-all-btn",
     "folder-lang-select"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = busy; }
    });
    // F145: the settings column. The server has answered 409 here since F104 ("swapping
    // the model mid-classification is not a setting but an accident") — what was missing
    // was the ban being VISIBLE: the fields stayed live, you moved one, and learned about
    // the refusal afterwards.
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (el) { el.disabled = busy; }
    });
    // Album buttons are built by four different tabs and none of them exists until its
    // tab is drawn, so they are swept by class rather than by id.
    document.querySelectorAll(".album-gather-btn").forEach(function (btn) {
      btn.disabled = busy;
    });
    document.querySelectorAll(".busy-hint").forEach(function (el) {
      el.style.display = busy ? "" : "none";
    });
    busyRefreshers.forEach(function (fn) { fn(); });
    var undoBtn = document.getElementById("undo-btn");
    // «Откатить» дополнительно требует батча в манифесте — см. applyUndoAvailability
    if (undoBtn) { undoBtn.disabled = busy || !undoAvailable; }
    // F104: an empty plan disables the start button and says WHY, instead of opening a
    // dialog full of zeroes. Until the plan has arrived the button is dead too, but
    // silently — "nothing to lay out" and "not counted yet" are different statements.
    var applyBtn = document.getElementById("sort-apply-btn");
    if (applyBtn) { applyBtn.disabled = busy || cityPlanCount === 0; }
    var emptyHint = document.getElementById("sort-empty-hint");
    if (emptyHint) {
      emptyHint.style.display = (cityPlanLoaded && cityPlanCount === 0) ? "" : "none";
    }
  }

  function renderSortStatus(data) {
    var bar = document.getElementById("sort-progress");
    var statusEl = document.getElementById("sort-status");
    var warnEl = document.getElementById("sort-warning");
    var cancelBtn = document.getElementById("sort-cancel-btn");
    sortRunning = !!data.running;
    // F104: "Cancel" is a contextual button — it exists exactly while a layout runs.
    // A permanent cancel button next to the start button cancels nothing.
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    updateBusyControlsDisabled();
    bar.style.display = data.running ? "" : "none";
    if (data.running) {
      bar.max = data.total || 0;
      bar.value = data.done || 0;
      statusEl.textContent = data.cancel_requested
          ? I18N.sort_cancel_requested
          : fmt(I18N.sort_progress_line, { done: data.done, all: data.total });
      warnEl.textContent = "";
      return;
    }
    if (!data.finished) {
      statusEl.textContent = ""; warnEl.textContent = ""; return;
    }
    if (data.error) {
      statusEl.textContent = I18N.sort_error_prefix + data.error;
      warnEl.textContent = "";
      return;
    }
    var r = data.result || {};
    // F97: отменённый прогон обязан говорить «сколько из скольких», а не «готово».
    // F104: what stayed next to it is the HINT pointing at the "Moves" tab, not a roll
    // back button. The manifest that says WHAT exactly would be rolled back lives
    // there; rolling back from the plan screen is rolling back blind.
    if (r.cancelled) {
      statusEl.textContent = fmt(I18N.sort_cancelled_text,
          { n: r.moved || 0, all: r.total || 0, f: r.failed || 0 });
    } else {
      statusEl.textContent = fmt(I18N.sort_done_text,
          { n: r.moved || 0, f: r.failed || 0, p: r.skipped_in_place || 0 });
    }
    if (r.skipped_already_copied) {
      statusEl.textContent += fmt(I18N.sort_already_copied_note,
          { c: r.skipped_already_copied });
    }
    warnEl.textContent = r.preview_stale ? I18N.sort_preview_stale_warning
        : (r.cancelled ? I18N.sort_undo_hint : "");
    movesLoaded = false;
    refreshUndoAvailability();
    renderPlanTab("city", "tree-city");
  }

  function pollSortStatus() {
    fetch("/api/sort/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderSortStatus(data);
        if (data.running) sortPollTimer = setTimeout(pollSortStatus, SORT_POLL_MS);
      });
  }

  document.getElementById("sort-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("sort-status").textContent = I18N.sort_cancel_requested;
    postJson("/api/sort/cancel", {});
  });

  function startSort() {
    var dest = document.getElementById("sort-dest").value.trim();
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var mode = checked ? checked.value : "move";
    postJson("/api/sort", { dest: dest || null, mode: mode }).then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("sort-status").textContent =
            I18N.sort_start_error_prefix + resp.error;
        return;
      }
      if (sortPollTimer) clearTimeout(sortPollTimer);
      pollSortStatus();
    });
  }

  document.getElementById("sort-apply-btn").addEventListener("click", function () {
    // An empty plan never gets here (the button is dead, see updateBusyControlsDisabled)
    // — a dialog full of zeroes is not an explanation.
    if (!cityPlanCount) return;
    var dest = document.getElementById("sort-dest").value.trim();
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var mode = checked ? checked.value : "move";
    var statusEl = document.getElementById("sort-status");
    statusEl.textContent = "";
    fetch("/api/sort/summary?dest=" + encodeURIComponent(dest))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) {
          statusEl.textContent = I18N.sort_summary_error + ((data && data.error) || "");
          return;
        }
        openSortDialog(data, dest, mode);
      })
      .catch(function (err) {
        statusEl.textContent = I18N.sort_summary_error + err;
      });
  });

  document.getElementById("sort-dialog-cancel").addEventListener("click", closeSortDialog);

  sortDialogEl.addEventListener("click", function (e) {
    if (e.target === sortDialogEl) closeSortDialog();  // клик по фону — отмена
  });

  document.getElementById("sort-dialog-ok").addEventListener("click", function () {
    closeSortDialog();
    startSort();
  });

  document.getElementById("sort-browse-btn").addEventListener("click", function () {
    browseIntoField(this, function (path) {
      document.getElementById("sort-dest").value = path;
    });
  });

  // Дефолт пути назначения = <источник>_sorted (сервер знает источник); только
  // если пользователь ещё ничего не ввёл — свой ввод не затираем.
  fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
    .then(function (resp) {
      var input = document.getElementById("sort-dest");
      if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
    }).catch(function () {});

  pollSortStatus();

  // --- вкладка «Перемещения» (U5, read-only манифест sort --apply) -------

  var MOVE_STATUS_LABELS = {
    planned: I18N.status_planned, done: I18N.status_done, undone: I18N.status_undone,
    failed: I18N.status_failed, deleted: I18N.status_deleted,
  };

  function moveStatusLabel(status) {
    return MOVE_STATUS_LABELS[status] || status;
  }

  var MOVE_STATUS_CHIP_CLASS = {
    done: "chip-good", planned: "chip-accent", failed: "chip-danger", deleted: "chip-danger",
    undone: "chip",
  };

  function renderMoveFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url, item.video));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      var pathLine = document.createElement("div");
      pathLine.textContent = item.src + " → " + item.dst;
      tdMeta.appendChild(pathLine);
      var statusChip = document.createElement("span");
      statusChip.className = "chip " + (MOVE_STATUS_CHIP_CLASS[item.status] || "chip");
      statusChip.textContent = moveStatusLabel(item.status);
      tdMeta.appendChild(statusChip);
      tr.appendChild(tdMeta);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  function batchSummaryText(batch, count) {
    var parts = [I18N.batch_label + " #" + batch.id, batch.mode, batch.operation || "move",
        I18N.started_label + " " + batch.started_at];
    parts.push(batch.finished_at ? I18N.finished_label + " " + batch.finished_at
        : I18N.in_progress_label);
    parts.push(I18N.files_count_label + ": " + count);
    return parts.join(" · ");
  }

  function loadMoves() {
    var container = document.getElementById("tree-moves");
    var summary = document.getElementById("moves-summary");
    fetch("/api/moves")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        container.textContent = "";
        summary.textContent = "";
        applyUndoAvailability(data);
        if (!data.batch) {
          summary.appendChild(stateEl("empty", I18N.no_moves_yet));
          return;
        }
        summary.textContent = batchSummaryText(data.batch, data.moves.length);
        var root = buildTree(data.moves);
        if (root.files.length) container.appendChild(renderMoveFiles(root.files));
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderNode(name, root.children[name], 0, renderMoveFiles));
        });
      })
      .catch(function (err) {
        applyUndoAvailability(null);
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_moves + err));
      });
  }

  // --- F97: откат последнего батча кнопкой (POST /api/undo) ----------------
  // Кнопка живёт рядом с манифестом и откатывает ровно тот батч, который манифест
  // показывает — селектора батчей нет намеренно: меньше способов ошибиться кнопкой,
  // которая удаляет файлы. Вторая точка входа — панель результата после отменённой
  // раскладки; эндпоинт и диалог у них общие.

  var UNDO_POLL_MS = 1000;
  var undoPollTimer = null;
  var undoDialogEl = document.getElementById("undo-dialog");

  // Строки, которые откат реально трогает: 'done' и хвост прерванного переноса
  // ('planned' — журнал коммитится ДО операции, статус мог не успеть записаться).
  function undoableCount(moves) {
    var n = 0;
    moves.forEach(function (m) {
      if (m.status === "done" || m.status === "planned") n += 1;
    });
    return n;
  }

  function applyUndoAvailability(data) {
    if (!data || !data.batch) {
      undoAvailable = false;
      undoBatchInfo = null;
    } else {
      undoBatchInfo = {
        operation: data.batch.operation || "move",
        dest_root: data.batch.dest_root || "",
        count: undoableCount(data.moves || []),
      };
      undoAvailable = undoBatchInfo.count > 0;
    }
    updateBusyControlsDisabled();
  }

  function refreshUndoAvailability() {
    movesLoaded = true;  // манифест перезагружаем прямо сейчас, повтор по клику не нужен
    loadMoves();
  }

  // Диалог называет операцию своими словами и числами из манифеста: без числа
  // страшную кнопку не нажимают вообще, а эта кнопка удаляет файлы.
  function undoConfirmText() {
    if (!undoBatchInfo) return I18N.undo_nothing_to_undo;
    if (undoBatchInfo.operation === "move") {
      return fmt(I18N.undo_confirm_move, { n: undoBatchInfo.count });
    }
    return fmt(I18N.undo_confirm_copy,
        { n: undoBatchInfo.count, dest: undoBatchInfo.dest_root });
  }

  function openUndoDialog() {
    if (!undoAvailable) {
      document.getElementById("undo-status").textContent = I18N.undo_nothing_to_undo;
      return;
    }
    document.getElementById("undo-dialog-text").textContent = undoConfirmText();
    undoDialogEl.hidden = false;
  }

  function closeUndoDialog() {
    undoDialogEl.hidden = true;
  }

  function renderUndoStatus(data) {
    var bar = document.getElementById("undo-progress");
    var statusEl = document.getElementById("undo-status");
    var strayEl = document.getElementById("undo-stray");
    var cancelBtn = document.getElementById("undo-cancel-btn");
    undoRunning = !!data.running;
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    updateBusyControlsDisabled();
    bar.style.display = data.running ? "" : "none";
    if (data.running) {
      bar.max = data.total || 0;
      bar.value = data.done || 0;
      statusEl.textContent = data.cancel_requested
          ? I18N.undo_cancel_requested
          : fmt(I18N.undo_progress_line, { done: data.done, all: data.total });
      strayEl.textContent = "";
      return;
    }
    if (!data.finished) { statusEl.textContent = ""; strayEl.textContent = ""; return; }
    if (data.error) {
      statusEl.textContent = I18N.undo_error_prefix + data.error;
      strayEl.textContent = "";
      return;
    }
    var r = data.result || {};
    statusEl.textContent = r.cancelled
        ? fmt(I18N.undo_cancelled_text, { n: r.undone || 0 })
        : fmt(I18N.undo_done_text,
              { n: r.undone || 0, m: r.missing || 0, f: r.failed || 0 });
    // Битые копии называются поимённо: они остались лежать в результате и выглядят
    // как обычные фото — молча их не удаляем и молча про них не забываем.
    strayEl.textContent = (r.stray && r.stray.length)
        ? I18N.undo_stray_title + " " + r.stray.join(", ") : "";
    refreshUndoAvailability();
    renderPlanTab("city", "tree-city");
  }

  function pollUndoStatus() {
    fetch("/api/undo/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderUndoStatus(data);
        if (data.running) undoPollTimer = setTimeout(pollUndoStatus, UNDO_POLL_MS);
      });
  }

  document.getElementById("undo-btn").addEventListener("click", openUndoDialog);
  document.getElementById("undo-dialog-cancel").addEventListener("click", closeUndoDialog);

  undoDialogEl.addEventListener("click", function (e) {
    if (e.target === undoDialogEl) closeUndoDialog();  // клик по фону — отмена
  });

  document.getElementById("undo-dialog-ok").addEventListener("click", function () {
    closeUndoDialog();
    var statusEl = document.getElementById("undo-status");
    statusEl.textContent = "";
    document.getElementById("undo-stray").textContent = "";
    postJson("/api/undo", {}).then(function (resp) {
      if (resp && resp.error) {
        statusEl.textContent = I18N.undo_start_error_prefix + resp.error;
        return;
      }
      if (undoPollTimer) clearTimeout(undoPollTimer);
      pollUndoStatus();
    });
  });

  document.getElementById("undo-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("undo-status").textContent = I18N.undo_cancel_requested;
    postJson("/api/undo/cancel", {});
  });

  pollUndoStatus();
  refreshUndoAvailability();

  // --- альбомы (F35): кнопка «Собрать в папку» на карточках Люди/События ---

  // F145: the album button moves files, so it is dead while anything runs — and the
  // reason is written next to it, the same `.busy-hint` the static blocks carry.
  function appendAlbumBusyHint(box) {
    var hint = document.createElement("span");
    hint.className = "override-hint busy-hint";
    hint.textContent = I18N.actions_busy;
    hint.style.display = uiBusy() ? "" : "none";
    box.appendChild(hint);
    return hint;
  }

  function albumModeSelect() {
    var select = document.createElement("select");
    ["link", "copy", "move"].forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = I18N["album_mode_" + m];
      select.appendChild(opt);
    });
    return select;
  }

  // Поле пути назначения альбома + «Обзор…» (F60, тот же мотив, что и
  // sort-dest/process-source-dir): дефолт = <источник>_sorted с сервера,
  // префилл только если поле ещё пустое (свой ввод не затираем).
  function appendAlbumDestControls(box) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "album-dest-input";
    input.placeholder = I18N.album_dest_placeholder;
    box.appendChild(input);
    var browseBtn = makeBtn("ghost", null, I18N.process_browse_button, "btn-sm album-browse-btn");
    browseBtn.addEventListener("click", function () {
      browseIntoField(this, function (path) { input.value = path; });
    });
    box.appendChild(browseBtn);
    fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
      .then(function (resp) {
        if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
      }).catch(function () {});
    return input;
  }

  function albumPreviewText(resp) {
    var txt = fmt(I18N.album_preview_text, { n: resp.count, dest: resp.dest });
    if (resp.mode === "move" && resp.blocked_multi) {
      txt += fmt(I18N.album_blocked_text, { k: resp.blocked_multi });
    }
    return txt;
  }

  // Превью (apply=false) -> подтверждение (текст зависит от режима, move явно
  // предупреждает об изъятии из пула) -> apply=true. statusEl получает
  // прогресс/результат; при успешном apply сбрасывается кэш вкладки
  // «Перемещения», чтобы следующий заход её перезагрузил (F35 п.4).
  function gatherAlbum(kind, selector, mode, where, name, dest, statusEl) {
    var body = { kind: kind, selector: selector, mode: mode, apply: false };
    if (where) body.where = [where];
    if (name) body.name = name;
    if (dest) body.dest = dest;
    statusEl.textContent = I18N.album_in_progress;
    postJson("/api/album", body).then(function (resp) {
      if (resp.error) { statusEl.textContent = resp.error; return; }
      var confirmMsg = albumPreviewText(resp) + "\\n" +
          (mode === "move" ? I18N.album_confirm_move : I18N.album_confirm_generic);
      if (!window.confirm(confirmMsg)) { statusEl.textContent = ""; return; }
      body.apply = true;
      statusEl.textContent = I18N.album_in_progress;
      postJson("/api/album", body).then(function (resp2) {
        if (resp2.error) { statusEl.textContent = resp2.error; return; }
        statusEl.textContent = fmt(I18N.album_result_text,
            { n: resp2.transferred, f: resp2.failed });
        movesLoaded = false;
      });
    });
  }

  // F139: the gather row of a slice that has no subject to choose inside it — a class
  // bucket ("Products") or a quality slice ("Blurred"). The same three controls every
  // other album has (mode, an optional folder name, a destination) and the same
  // dry-run-then-confirm path; the only thing that varies is the `kind` the server was
  // asked to gather, and `kind` = null takes the row away entirely, which is what a
  // sensitive class and the duplicates look like.
  //
  // Rebuilt only when the kind CHANGES: the row is drawn from inside the paging render,
  // and re-creating it per page would ask the server for a default destination again and
  // wipe a path somebody had typed.
  function renderSliceAlbumControls(boxId, kind) {
    var box = document.getElementById(boxId);
    if (box.getAttribute("data-kind") === (kind || "")) return;
    box.setAttribute("data-kind", kind || "");
    box.textContent = "";
    if (!kind) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "album-name-input";
    nameInput.placeholder = I18N.album_name_placeholder;
    box.appendChild(nameInput);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      gatherAlbum(kind, "", modeSelect.value, null, nameInput.value.trim() || null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  // --- лайтбокс (F42): один переиспользуемый оверлей поверх /photo/<id> ---
  // Заполняется по клику (не N скрытых оверлеев). Клик по фону/Esc закрывает;
  // стрелки ←/→ листают переданный список sample-кадров (опц., F42).
  //
  // F80: у ВИДЕО те же стрелки листают кадры ОДНОГО ролика (/frame/<id>/<i>), а не
  // соседние файлы: воспроизведения нет, и несколько кадров — единственный способ
  // понять, что там снято. Для фото поведение не меняется ни на шаг: lightboxFrames
  // остаётся нулём, кадр берётся всё тем же /preview/<id>.
  //
  // Кадры тянутся лениво: src ставится ровно одному кадру, тому, что показан. Сетка
  // плиток по-прежнему знает только /thumb — шесть кадров на плитку никто не грузит.

  var lightboxEl = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");
  var lightboxPrev = document.getElementById("lightbox-prev");
  var lightboxNext = document.getElementById("lightbox-next");
  var lightboxDots = document.getElementById("lightbox-dots");
  var lightboxSamples = null;
  var lightboxIndex = 0;
  var lightboxFrames = 0;   // > 0 <=> открыто видео, столько кадров у ленты
  var lightboxFrame = 0;

  function renderLightboxDots() {
    lightboxDots.textContent = "";
    for (var i = 0; i < lightboxFrames; i++) {
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "lightbox-dot" + (i === lightboxFrame ? " active" : "");
      dot.title = fmt(I18N.frame_of, { n: i + 1, all: lightboxFrames });
      dot.addEventListener("click", (function (frame) {
        return function (e) { e.stopPropagation(); showLightboxFrame(frame); };
      })(i));
      lightboxDots.appendChild(dot);
    }
    var multi = lightboxFrames > 1;
    lightboxDots.hidden = !multi;
    lightboxPrev.hidden = !multi;
    lightboxNext.hidden = !multi;
  }

  function showLightboxFrame(frame) {
    lightboxFrame = frame;
    lightboxImg.src = "/frame/" + lightboxSamples[lightboxIndex] + "/" + frame;
    renderLightboxDots();
  }

  function showLightboxAt(index) {
    lightboxIndex = index;
    if (lightboxFrames) { showLightboxFrame(0); return; }
    // /preview — крупный ДЕКОДИРОВАННЫЙ JPEG (HEIC/RAW рендерятся), не сырой /photo
    lightboxImg.src = "/preview/" + lightboxSamples[index];
  }

  function stepLightboxFrame(delta) {
    showLightboxFrame((lightboxFrame + delta + lightboxFrames) % lightboxFrames);
  }

  function openLightbox(samples, index, videoFrames) {
    lightboxSamples = samples;
    lightboxFrames = videoFrames || 0;
    lightboxFrame = 0;
    renderLightboxDots();
    showLightboxAt(index);
    lightboxEl.hidden = false;
  }

  function closeLightbox() {
    lightboxEl.hidden = true;
    lightboxImg.src = "";
    lightboxSamples = null;
    lightboxFrames = 0;
    lightboxFrame = 0;
    renderLightboxDots();
  }

  // Короткий ролик отдаёт меньше кадров, чем настроено, и недостающий индекс — это
  // честный 404. Обрезаем ленту по первому промаху и возвращаемся на прошлый кадр:
  // сервер не обязан заранее знать, сколько кадров вытащится из конкретного файла.
  lightboxImg.addEventListener("error", function () {
    if (!lightboxFrames || lightboxFrame < 1) return;
    lightboxFrames = lightboxFrame;
    showLightboxFrame(lightboxFrame - 1);
  });

  lightboxEl.addEventListener("click", closeLightbox);
  lightboxImg.addEventListener("click", function (e) { e.stopPropagation(); });
  lightboxPrev.addEventListener("click", function (e) {
    e.stopPropagation();
    stepLightboxFrame(-1);
  });
  lightboxNext.addEventListener("click", function (e) {
    e.stopPropagation();
    stepLightboxFrame(1);
  });
  document.addEventListener("keydown", function (e) {
    if (lightboxEl.hidden) return;
    if (e.key === "Escape") { closeLightbox(); return; }
    if (lightboxFrames > 1) {
      if (e.key === "ArrowRight") stepLightboxFrame(1);
      else if (e.key === "ArrowLeft") stepLightboxFrame(-1);
      return;
    }
    if (!lightboxSamples || lightboxSamples.length < 2) return;
    if (e.key === "ArrowRight") showLightboxAt((lightboxIndex + 1) % lightboxSamples.length);
    else if (e.key === "ArrowLeft") {
      showLightboxAt((lightboxIndex - 1 + lightboxSamples.length) % lightboxSamples.length);
    }
  });

  // --- вкладка «Люди» (F31, управление кластерами лиц) --------------------

  var clustersById = {};
  var selectedForMerge = {};
  var selectedForMergeCount = 0;

  function updateMergeButton() {
    // F145: merging two clusters rewrites `faces.cluster_id` for both of them.
    document.getElementById("clusters-merge-btn").disabled =
        uiBusy() || selectedForMergeCount !== 2;
  }

  registerBusyRefresh(updateMergeButton);

  function toggleMergeSelection(clusterId, checked) {
    if (checked) {
      if (!(clusterId in selectedForMerge)) selectedForMergeCount += 1;
      selectedForMerge[clusterId] = true;
    } else {
      if (clusterId in selectedForMerge) selectedForMergeCount -= 1;
      delete selectedForMerge[clusterId];
    }
    updateMergeButton();
  }

  function renderClusterCard(c) {
    var card = document.createElement("div");
    card.className = "card" + (c.label ? " named" : "");

    var thumbs = document.createElement("div");
    thumbs.className = "cluster-thumbs";
    // Скелетон рисуется сразу (карточка отзывчива, пока идёт /thumb) —
    // сама миниатюра грузится лениво и фоном; onload плавно проявляет её и
    // снимает скелетон-заглушку (F42).
    c.samples.forEach(function (fileId, idx) {
      var skel = document.createElement("div");
      skel.className = "thumb-skel";
      var img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.addEventListener("load", function () { skel.className = "thumb-skel loaded"; });
      img.addEventListener("click", function () { openLightbox(c.samples, idx); });
      img.src = "/thumb/" + fileId;
      skel.appendChild(img);
      thumbs.appendChild(skel);
    });
    card.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "cluster-meta";
    meta.textContent = (c.label ? c.label : I18N.unnamed) + " \\u00b7 " + c.size + " " +
        I18N.faces_unit;
    card.appendChild(meta);

    var form = document.createElement("div");
    form.className = "cluster-name-form";
    var input = document.createElement("input");
    input.type = "text";
    input.value = c.label || "";
    input.placeholder = I18N.person_name_placeholder;
    form.appendChild(input);
    var btnName = makeBtn("primary", "tag", I18N.name_button, "btn-sm");
    btnName.addEventListener("click", function () {
      var name = input.value.trim();
      if (!name) { window.alert(I18N.alert_enter_name); return; }
      postJson("/api/clusters/label", { cluster_id: c.cluster_id, name: name })
        .then(function (resp) { if (resp && resp.ok) loadClusters(); });
    });
    form.appendChild(btnName);
    card.appendChild(form);

    var mergeLabel = document.createElement("label");
    mergeLabel.className = "cluster-merge-select";
    var checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.addEventListener("change", function () {
      toggleMergeSelection(c.cluster_id, checkbox.checked);
    });
    mergeLabel.appendChild(checkbox);
    mergeLabel.appendChild(document.createTextNode(" " + I18N.select_for_merge));
    card.appendChild(mergeLabel);

    if (c.label) {
      var albumBox = document.createElement("div");
      albumBox.className = "album-controls";
      var modeSelect = albumModeSelect();
      albumBox.appendChild(modeSelect);
      var destInput = appendAlbumDestControls(albumBox);
      var whereInput = document.createElement("input");
      whereInput.type = "text";
      whereInput.placeholder = I18N.album_where_placeholder;
      albumBox.appendChild(whereInput);
      var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
      albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
      var albumStatus = document.createElement("span");
      albumStatus.className = "album-status";
      albumBtn.addEventListener("click", function () {
        var where = whereInput.value.trim();
        gatherAlbum("person", c.label, modeSelect.value, where || null, null,
            destInput.value.trim() || null, albumStatus);
      });
      albumBox.appendChild(albumBtn);
      albumBox.appendChild(albumStatus);
      appendAlbumBusyHint(albumBox);
      card.appendChild(albumBox);
    } else {
      var hint = document.createElement("div");
      hint.className = "album-hint";
      hint.textContent = I18N.album_name_first_hint;
      card.appendChild(hint);
    }

    return card;
  }

  function loadClusters() {
    var container = document.getElementById("clusters-grid");
    fetch("/api/clusters")
      .then(function (r) { return r.json(); })
      .then(function (clusters) {
        container.textContent = "";
        clustersById = {};
        selectedForMerge = {};
        selectedForMergeCount = 0;
        updateMergeButton();
        if (!clusters.length) {
          container.appendChild(stateEl("empty", I18N.no_clusters));
          return;
        }
        var named = clusters.filter(function (c) { return c.label; });
        var unnamed = clusters.filter(function (c) { return !c.label; });
        named.concat(unnamed).forEach(function (c) {
          clustersById[c.cluster_id] = c;
          container.appendChild(renderClusterCard(c));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_clusters + err));
      });
  }

  document.getElementById("clusters-merge-btn").addEventListener("click", function () {
    var ids = Object.keys(selectedForMerge).map(Number);
    if (ids.length !== 2) return;
    var a = clustersById[ids[0]];
    var b = clustersById[ids[1]];
    var dst = a.size >= b.size ? a.cluster_id : b.cluster_id;
    var src = dst === a.cluster_id ? b.cluster_id : a.cluster_id;
    postJson("/api/clusters/merge", { src: src, dst: dst })
      .then(function (resp) { if (resp && resp.ok) loadClusters(); });
  });

  // --- вкладка «События» (F35: список событий + «Собрать в папку») --------

  function renderEventCard(e) {
    var card = document.createElement("div");
    card.className = "card";

    var meta = document.createElement("div");
    meta.className = "event-meta";
    meta.textContent = e.count + " " + I18N.files_count_label + " \\u00b7 " +
        [e.started_at, e.ended_at].filter(Boolean).join(" \\u2013 ");
    card.appendChild(meta);

    // превью-кадры события (клик -> лайтбокс, стрелки листают кадры события)
    if (e.samples && e.samples.length) {
      var thumbs = document.createElement("div");
      thumbs.className = "event-thumbs";
      e.samples.forEach(function (fileId, idx) {
        thumbs.appendChild(clickableThumb(fileId, e.samples, idx));
      });
      card.appendChild(thumbs);
    }

    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "event-name-input";
    nameInput.value = e.name || "";
    nameInput.placeholder = I18N.album_name_placeholder;
    card.appendChild(nameInput);

    var albumBox = document.createElement("div");
    albumBox.className = "album-controls";
    var modeSelect = albumModeSelect();
    albumBox.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(albumBox);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      var name = nameInput.value.trim();
      gatherAlbum("event", String(e.id), modeSelect.value, null, name || null,
          destInput.value.trim() || null, albumStatus);
    });
    albumBox.appendChild(albumBtn);
    albumBox.appendChild(albumStatus);
    appendAlbumBusyHint(albumBox);
    card.appendChild(albumBox);

    // F85c: событие — самая осязаемая группа, какая есть: это одна поездка, и место
    // у неё одно. Назначение на всё событие целиком — одно действие вместо e.count.
    var placeBox = document.createElement("div");
    placeBox.className = "place-controls";
    var picker = renderPlacePicker(placeBox);
    var placeStatus = document.createElement("span");
    placeStatus.className = "override-status";
    var assignBtn = makeBtn("primary", "pin", I18N.place_assign_button,
        "btn-sm place-assign-btn");
    assignBtn.addEventListener("click", function () {
      assignPlace(picker, "event", String(e.id), "place_assign_confirm",
                  { n: e.count }, placeStatus, null);
    });
    var clearBtn = makeBtn("ghost", null, I18N.place_clear_button,
        "btn-sm place-clear-btn");
    clearBtn.addEventListener("click", function () {
      clearPlace("event", String(e.id), "place_event_clear_confirm",
                 { n: e.count }, placeStatus, null);
    });
    placeBox.appendChild(assignBtn);
    placeBox.appendChild(clearBtn);
    placeBox.appendChild(placeStatus);
    card.appendChild(placeBox);

    return card;
  }

  function loadEvents() {
    var container = document.getElementById("events-list");
    fetch("/api/events")
      .then(function (r) { return r.json(); })
      .then(function (events) {
        container.textContent = "";
        if (!events.length) {
          container.appendChild(stateEl("empty", I18N.no_events));
          return;
        }
        events.forEach(function (e) { container.appendChild(renderEventCard(e)); });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_events + err));
      });
  }

  // --- F103: вкладка «Не личные фото» -------------------------------------
  // Корзины классификатора видно КАК корзины: чипы-фильтры со счётчиком, сетка
  // плиток, отметка нескольких кадров и ОДИН возврат на всё выделение (по одному
  // это десятки кликов на «пару штук из 2 202»). Возврат — это POST /api/overrides
  // с action="photo" (готовый механизм F77): вердикт в media_class не переписывается,
  // поэтому повторный прогон яруса не сотрёт правку.

  var JUNK_PAGE_SIZE = 200;
  var junkBucket = null;   // null — «Все»
  var junkOffset = 0;
  var junkSelected = {};

  function junkBucketLabel(verdict) {
    return I18N["junk_bucket_" + verdict] || verdict;
  }

  function junkSelectedIds() {
    return Object.keys(junkSelected).map(Number);
  }

  function refreshJunkControls() {
    var n = junkSelectedIds().length;
    document.getElementById("junk-selected-count").textContent = n ? " (" + n + ")" : "";
    // F145: "back to photos" rewrites `media_class` — the table the run in flight owns.
    document.getElementById("junk-restore-btn").disabled = uiBusy() || n === 0;
  }

  registerBusyRefresh(refreshJunkControls);

  // F133: корзины классификатора — это и есть закреплённые срезы «товары / скриншоты /
  // документы»; отдельного ряда чипов больше нет, счётчики уезжают в ряд срезов.
  function renderJunkBuckets(buckets) {
    junkBucketCounts = buckets || [];
    renderSlicePins();
  }

  function renderJunkCard(item) {
    var card = document.createElement("div");
    card.className = "junk-card" + (item.restored ? " restored" : "");
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // Документ: превью не строим вовсе — сервер не прислал ссылку, и запроса к
      // /thumb здесь нет. Заглушка того же размера, чтобы сетка не разъезжалась.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "junk-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "junk-card-meta";
    meta.textContent = [junkBucketLabel(item.verdict), item.date || ""]
        .filter(Boolean).join(" \\u00b7 ");
    card.appendChild(meta);
    if (item.restored) {
      var chip = document.createElement("span");
      chip.className = "chip chip-good";
      chip.textContent = I18N.junk_restored_mark;
      card.appendChild(chip);
      var undoBtn = makeBtn("ghost", null, I18N.junk_undo_restore_button, "btn-sm");
      undoBtn.addEventListener("click", function () { applyJunkAction([item.file_id], "clear"); });
      card.appendChild(undoBtn);
      return card;
    }
    var label = document.createElement("label");
    label.className = "junk-card-select";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.className = "junk-select";
    box.value = String(item.file_id);
    box.checked = !!junkSelected[item.file_id];
    box.addEventListener("change", function () {
      if (box.checked) junkSelected[item.file_id] = true;
      else delete junkSelected[item.file_id];
      refreshJunkControls();
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(I18N.junk_restore_button));
    card.appendChild(label);
    return card;
  }

  function renderJunkPage(data, append) {
    var grid = document.getElementById("junk-grid");
    // F139: the bucket is gathered into a folder like any other slice — or it is not,
    // and the server says which (a sensitive class keeps its counter and gets neither a
    // preview nor an album). The "back to photos" row above is untouched: one movement
    // must not be able to both gather and delete.
    renderSliceAlbumControls("junk-album", data.album_kind);
    if (!append) grid.textContent = "";
    var items = data.items || [];
    items.forEach(function (it) { grid.appendChild(renderJunkCard(it)); });
    var shown = grid.querySelectorAll(".junk-card").length;
    // Пустая корзина — внятное «здесь пусто», а не вечный спиннер.
    if (!shown) grid.appendChild(stateEl("empty", I18N.junk_empty));
    document.getElementById("junk-shown").textContent =
        shown ? fmt(I18N.junk_shown_label, { shown: shown, total: data.total }) : "";
    document.getElementById("junk-more-btn").style.display =
        shown && shown < data.total ? "" : "none";
    // Пояснение про документы — только там, где карточки без превью реально есть
    // (считаем по всей сетке, а не по последней подгруженной странице).
    document.getElementById("junk-doc-hint").style.display =
        grid.querySelector(".junk-doc-box") ? "" : "none";
    junkOffset = shown;
  }

  function fetchJunk(offset, append) {
    var grid = document.getElementById("junk-grid");
    if (!append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    var url = "/api/junk?offset=" + offset + "&limit=" + JUNK_PAGE_SIZE +
        (junkBucket ? "&bucket=" + encodeURIComponent(junkBucket) : "");
    return fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderJunkBuckets(data.buckets || []);
        renderJunkPage(data, append);
      })
      .catch(function (err) {
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_junk + err));
      });
  }

  function loadJunk() {
    junkSelected = {};
    refreshJunkControls();
    document.getElementById("junk-status").textContent = "";
    return fetchJunk(0, false);
  }

  function applyJunkAction(ids, action) {
    var status = document.getElementById("junk-status");
    status.textContent = "";
    return postJson("/api/overrides", { file_ids: ids, action: action })
      .then(function (resp) {
        if (resp && resp.ok) {
          junkSelected = {};
          refreshJunkControls();
          fetchJunk(0, false);
        } else {
          status.textContent = I18N.junk_error_prefix + ((resp && resp.error) || "");
        }
      })
      .catch(function (err) { status.textContent = I18N.junk_error_prefix + err; });
  }

  document.getElementById("junk-restore-btn").addEventListener("click", function () {
    var ids = junkSelectedIds();
    if (!ids.length) return;
    if (!window.confirm(fmt(I18N.junk_restore_confirm, { n: ids.length }))) return;
    applyJunkAction(ids, "photo");
  });
  document.getElementById("junk-select-all-btn").addEventListener("click", function () {
    document.querySelectorAll("#junk-grid .junk-select").forEach(function (box) {
      box.checked = true;
      junkSelected[parseInt(box.value, 10)] = true;
    });
    refreshJunkControls();
  });
  document.getElementById("junk-select-none-btn").addEventListener("click", function () {
    document.querySelectorAll("#junk-grid .junk-select").forEach(function (box) {
      box.checked = false;
    });
    junkSelected = {};
    refreshJunkControls();
  });
  document.getElementById("junk-more-btn").addEventListener("click", function () {
    fetchJunk(junkOffset, true);
  });
  refreshJunkControls();

  // --- F123: the "Animals" tab -------------------------------------------
  // A page of tiles ordered by confidence, plus the one action the slice affords:
  // gather it into an album. Paged for the same reason as the junk grid (F70) — 805
  // cards with previews are not put into the DOM at once. The score is printed on the
  // card: the verdict is 92% right, and the only way to see where the wrong 8% start
  // is to read down a list that is sorted by exactly that number.

  var ANIMALS_PAGE_SIZE = 200;
  var animalsOffset = 0;
  // The length of the LIST, kept so a card redrawn after a mark can restate "showing
  // N of M" without asking the server for a page it already has.
  var animalsTotal = 0;

  function hasManualPet(item) {
    return item.manual !== null && item.manual !== undefined;
  }

  function renderAnimalCard(item) {
    var card = document.createElement("div");
    // F124: `is_animal` comes from the server (the one shared rule), it is never
    // recomputed here — a second spelling of that rule in JS is exactly how the tab
    // and the album start reporting different collections.
    card.className = "animal-card" + (item.is_animal ? "" : " not-animal");
    card.dataset.fileId = String(item.file_id);
    card.appendChild(
        clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    var name = document.createElement("span");
    name.className = "animal-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "animal-card-meta";
    meta.textContent = item.date || "";
    card.appendChild(meta);
    if (item.score !== null && item.score !== undefined) {
      var score = document.createElement("span");
      score.className = "animal-card-score";
      score.textContent = fmt(I18N.animals_score_label,
                              { score: Number(item.score).toFixed(2) });
      card.appendChild(score);
    }
    // A frame decided by hand says so, and says which way: without it the counter
    // moves for no visible reason and a dimmed card looks like a rendering fault.
    if (hasManualPet(item)) {
      var manual = document.createElement("span");
      manual.className = "animal-card-manual";
      manual.textContent = item.manual ? I18N.animals_manual_included
                                       : I18N.animals_manual_excluded;
      card.appendChild(manual);
    }
    var actions = document.createElement("div");
    actions.className = "animal-card-actions";
    // One toggle offering the answer the frame does NOT have right now, per card and
    // never over a band: the whole feature is that somebody looked at this frame.
    var toggle = makeBtn("ghost", null,
        item.is_animal ? I18N.animals_mark_not_animal : I18N.animals_mark_animal,
        "btn-sm animal-mark-btn");
    toggle.addEventListener("click", function () {
      markAnimal(item.file_id, item.is_animal ? "not_animal" : "animal");
    });
    actions.appendChild(toggle);
    if (hasManualPet(item)) {
      var back = makeBtn("ghost", null, I18N.animals_mark_clear,
                         "btn-sm animal-clear-btn");
      back.addEventListener("click", function () { markAnimal(item.file_id, "clear"); });
      actions.appendChild(back);
    }
    card.appendChild(actions);
    return card;
  }

  function animalCardEl(fileId) {
    return document.querySelector('#animals-grid .animal-card[data-file-id="' +
                                 fileId + '"]');
  }

  // Both numbers of the page: how much of the LIST is on screen, and how many of it
  // count as animals. After a manual mark those are different questions — the card
  // stays in the list and leaves the count.
  function renderAnimalsCounts(shown, total, animals) {
    document.getElementById("animals-shown").textContent =
        shown ? fmt(I18N.animals_shown_label, { shown: shown, total: total }) : "";
    document.getElementById("animals-counted").textContent =
        shown ? fmt(I18N.animals_counted_label, { n: animals }) : "";
  }

  function renderAnimalsPage(data, append) {
    var grid = document.getElementById("animals-grid");
    if (!append) grid.textContent = "";
    (data.items || []).forEach(function (it) {
      grid.appendChild(renderAnimalCard(it));
    });
    var shown = grid.querySelectorAll(".animal-card").length;
    if (!shown) grid.appendChild(stateEl("empty", I18N.animals_empty));
    animalsTotal = data.total;
    renderAnimalsCounts(shown, data.total, data.animals);
    document.getElementById("animals-more-btn").style.display =
        shown && shown < data.total ? "" : "none";
    animalsOffset = shown;
  }

  // The answer redraws the card in place instead of reloading the page: this list is
  // read top-down until the confidence runs out, and a reload after every decision
  // would send the reader back to the first screen. The redrawn card comes from the
  // server, so it says what a reload would say.
  function markAnimal(fileId, action) {
    var status = document.getElementById("animals-mark-status");
    status.textContent = "";
    return postJson("/api/animals/mark", { file_ids: [fileId], action: action })
      .then(function (resp) {
        if (!resp || !resp.ok) {
          status.textContent = I18N.animals_error_prefix + ((resp && resp.error) || "");
          return;
        }
        var card = animalCardEl(fileId);
        var fresh = (resp.items || [])[0];
        if (card && fresh) {
          card.parentNode.replaceChild(renderAnimalCard(fresh), card);
        } else if (card) {                    // it left the list entirely (a `clear`
          card.parentNode.removeChild(card);  // on a frame the model never marked)
          animalsTotal = Math.max(0, animalsTotal - 1);
        }
        var grid = document.getElementById("animals-grid");
        var shown = grid.querySelectorAll(".animal-card").length;
        if (!shown) grid.appendChild(stateEl("empty", I18N.animals_empty));
        animalsOffset = shown;
        renderAnimalsCounts(shown, animalsTotal, resp.animals);
      })
      .catch(function (err) { status.textContent = I18N.animals_error_prefix + err; });
  }

  function fetchAnimals(offset, append) {
    var grid = document.getElementById("animals-grid");
    if (!append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    return fetch("/api/animals?offset=" + offset + "&limit=" + ANIMALS_PAGE_SIZE)
      .then(function (r) { return r.json(); })
      .then(function (data) { renderAnimalsPage(data, append); })
      .catch(function (err) {
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_animals + err));
      });
  }

  // The album controls of the People/Events cards, one per tab instead of one per
  // card: the slice is single, so there is nothing to pick a subject from. The
  // selector goes out empty and the server ignores it (kind='animal'), and the album
  // name is left to the server too — it is a folder name, and it follows `language:`.
  function renderAnimalsAlbumControls() {
    var box = document.getElementById("animals-album");
    if (box.childNodes.length) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      gatherAlbum("animal", "", modeSelect.value, null, null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  function loadAnimals() {
    renderAnimalsAlbumControls();
    return fetchAnimals(0, false);
  }

  document.getElementById("animals-more-btn").addEventListener("click", function () {
    fetchAnimals(animalsOffset, true);
  });

  // --- F152: the face slices -----------------------------------------------
  // Three pins over one panel — the junk-bucket arrangement, because these are three
  // questions of one kind and a panel each would be three copies of the same grid. What
  // is NOT shared with the slices around it is the caption: there is no score on a card
  // and no ranking hint above the grid, because a frame is here by a fact of the
  // detector and not by a position in a list. The one line that changes with the slice
  // is the rule it was selected by, thresholds and all.
  //
  // The empty state is a sentence, not a zero. Without a faces run the server answers
  // `reason='no_faces_run'` and `null` counters, and both the pins and this panel say
  // that instead of showing a number nobody measured (F125).

  var FACE_SLICES = ["people", "group", "portrait"];
  var FACE_PAGE_SIZE = 200;
  var faceSlice = "people";
  var faceOffset = 0;
  var faceLoaded = false;
  var faceReason = null;

  function applyFaceCounts(data) {
    faceReason = (data && data.reason) || null;
    faceSliceCounts = {};
    ((data && data.counts) || []).forEach(function (row) {
      faceSliceCounts[row.slice] = row.count;
    });
  }

  // Why this slice holds what it holds, in one line above the grid — with the numbers
  // the server actually selected by, so the rule on screen is the rule that ran.
  function faceHintText(data) {
    if (faceReason === "no_faces_run") return I18N.face_no_faces_run;
    if (faceSlice === "group") {
      return fmt(I18N.face_hint_group, { n: data.group_min });
    }
    if (faceSlice === "portrait") {
      return fmt(I18N.face_hint_portrait,
                 { share: (Number(data.portrait_share) * 100).toFixed(1) });
    }
    return I18N.face_hint_people;
  }

  function renderFaceCard(item) {
    var card = document.createElement("div");
    card.className = "face-card";
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // F133's rule, unchanged: a sensitive class is listed but never decoded for
      // display. A document with a face on it is exactly the frame that rule is for.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "face-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "face-card-meta";
    meta.textContent = [item.date || "", fmt(I18N.face_count_label, { n: item.faces })]
        .filter(Boolean).join(" \\u00b7 ");
    card.appendChild(meta);
    return card;
  }

  function renderFacePage(data, append) {
    var grid = document.getElementById("face-grid");
    if (!append) grid.textContent = "";
    (data.items || []).forEach(function (it) { grid.appendChild(renderFaceCard(it)); });
    var shown = grid.querySelectorAll(".face-card").length;
    if (!shown) {
      grid.appendChild(stateEl("empty",
          faceReason === "no_faces_run" ? I18N.face_no_faces_run : I18N.face_empty));
    }
    document.getElementById("face-shown").textContent =
        shown ? fmt(I18N.face_shown_label, { shown: shown, total: data.total }) : "";
    document.getElementById("face-more-btn").style.display =
        shown && shown < data.total ? "" : "none";
    faceOffset = shown;
  }

  function fetchFaceSlice(offset, append) {
    var grid = document.getElementById("face-grid");
    if (!append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    return fetch("/api/face-slices?slice=" + faceSlice + "&offset=" + offset +
                 "&limit=" + FACE_PAGE_SIZE)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        applyFaceCounts(data);
        renderSlicePins();
        document.getElementById("face-hint").textContent = faceHintText(data);
        renderFacePage(data, append);
      })
      .catch(function (err) {
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_face_slices + err));
      });
  }

  // One album per slice, the animal arrangement: the selector goes out empty and the
  // server ignores it (the collection holds a single slice of each kind), and the album
  // name is left to the server — it is a folder name and follows `language:`. Rebuilt on
  // every open because the KIND changes with the pin.
  function renderFaceAlbumControls() {
    var box = document.getElementById("face-album");
    box.textContent = "";
    if (faceReason === "no_faces_run") return;   // nothing to gather, and no button for it
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    var kind = faceSlice;
    albumBtn.addEventListener("click", function () {
      gatherAlbum(kind, "", modeSelect.value, null, null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  function loadFaceSlice() {
    return fetchFaceSlice(0, false).then(function () { renderFaceAlbumControls(); });
  }

  document.getElementById("face-more-btn").addEventListener("click", function () {
    fetchFaceSlice(faceOffset, true);
  });

  // --- F126: the "Review" workspace ----------------------------------------
  // One tab, four slices, one job: look and decide. The switcher keeps every slice in
  // place at zero, because "you have no closed eyes" is an answer and a vanished entry
  // is a riddle. Duplicates are rendered by the code below this block, untouched — they
  // are the only grouped slice, the only one where a keeper is chosen, and the only path
  // in the program that deletes files. The three flat slices share the tile grid and the
  // one action they afford: a mark in `dedup_choice`, which the sorter already reads.
  // Paged like every other grid since F70 — 530 cards with previews do not go into the
  // DOM at once.

  var REVIEW_PAGE_SIZE = 200;
  var REVIEW_SLICES = ["dupes", "blurred", "eyes", "subject"];
  var reviewSlice = "dupes";
  var reviewOffset = 0;
  // Blur opens to `features.blur_review_max` and continues past it only when asked:
  // the number is a window, not a verdict.
  var reviewBeyond = false;
  var reviewWindowTotal = 0;
  var reviewSelected = {};

  function reviewSelectedIds() {
    return Object.keys(reviewSelected).map(Number);
  }

  function refreshReviewControls() {
    var n = reviewSelectedIds().length;
    document.getElementById("review-selected-count").textContent = n ? " (" + n + ")" : "";
    var dead = uiBusy() || n === 0;   // F145: a mark is a row in `dedup_choice`
    ["review-delete-btn", "review-keep-btn", "review-clear-btn"].forEach(function (id) {
      document.getElementById(id).disabled = dead;
    });
  }

  registerBusyRefresh(refreshReviewControls);

  function renderReviewCounts(counts) {
    counts.forEach(function (row) {
      var el = document.getElementById("review-count-" + row.slice);
      if (el) el.textContent = " (" + overviewNum(row.count) + ")";
    });
  }

  // Why this slice looks the way it does, in one line above the grid. For closed eyes
  // it is also where the F125 answer lands: without a faces run there is no data, and
  // saying so beats showing a zero that reads as "nobody blinked".
  function reviewHintText(data) {
    if (reviewSlice === "blurred") {
      return fmt(I18N.review_hint_blurred, { max: data.blur_max });
    }
    if (reviewSlice === "eyes") {
      return data.eyes_reason === "no_faces_run"
          ? I18N.review_eyes_no_faces : I18N.review_hint_eyes;
    }
    return I18N.review_hint_subject;
  }

  function renderReviewCard(item) {
    var card = document.createElement("div");
    card.className = "review-card" +
        (item.action === "to_delete" ? " marked-delete" : "") +
        (item.action === "keep" ? " marked-keep" : "");
    card.appendChild(
        clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    var name = document.createElement("span");
    name.className = "review-card-name";
    name.textContent = item.name;
    name.title = item.src_path || item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "review-card-meta";
    var sharp = item.sharpness === null || item.sharpness === undefined ? "" :
        fmt(I18N.review_sharpness_label, { value: Number(item.sharpness).toFixed(0) });
    meta.textContent = [item.src_dir, item.date || "", sharp, actionLabel(item.action)]
        .filter(Boolean).join(" \\u00b7 ");
    card.appendChild(meta);
    var label = document.createElement("label");
    label.className = "review-card-select";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.className = "review-select";
    box.value = String(item.file_id);
    box.checked = !!reviewSelected[item.file_id];
    box.addEventListener("change", function () {
      if (box.checked) reviewSelected[item.file_id] = true;
      else delete reviewSelected[item.file_id];
      refreshReviewControls();
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + I18N.review_select_label));
    card.appendChild(label);
    return card;
  }

  function renderReviewPage(data, append) {
    var grid = document.getElementById("review-grid");
    if (!append) grid.textContent = "";
    (data.items || []).forEach(function (it) { grid.appendChild(renderReviewCard(it)); });
    var shown = grid.querySelectorAll(".review-card").length;
    if (!shown) {
      grid.appendChild(stateEl("empty",
          data.eyes_reason === "no_faces_run" && reviewSlice === "eyes"
              ? I18N.review_eyes_no_faces : I18N.review_empty));
    }
    document.getElementById("review-shown").textContent =
        shown ? fmt(I18N.review_shown_label, { shown: shown, total: data.total }) : "";
    // Past the end of the window the button changes its meaning, not just its target:
    // the next page is no longer "more of the same list" but a step outside the window
    // the list opened to.
    var beyondNext = reviewSlice === "blurred" && !reviewBeyond &&
        shown >= reviewWindowTotal;
    var more = shown < data.total || beyondNext;
    var moreBtn = document.getElementById("review-more-btn");
    moreBtn.textContent = beyondNext ? I18N.review_load_more_beyond : I18N.review_load_more;
    moreBtn.style.display = more ? "" : "none";
    reviewOffset = shown;
  }

  function fetchReview(offset, append) {
    var flat = reviewSlice !== "dupes";
    var grid = document.getElementById("review-grid");
    if (flat && !append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    var url = "/api/review?slice=" + reviewSlice + "&offset=" + offset +
        "&limit=" + REVIEW_PAGE_SIZE + (reviewBeyond ? "&beyond=1" : "");
    return fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderReviewCounts(data.counts || []);
        reviewWindowTotal = data.window_total || 0;
        // F139: the flat slices are gathered into a folder like people and events are;
        // the duplicates are not (`album_kind` is null there), and the marking row above
        // stays exactly where it was — gathering and deleting are two movements.
        renderSliceAlbumControls("review-album", data.album_kind);
        if (!flat) return;
        document.getElementById("review-hint").textContent = reviewHintText(data);
        renderReviewPage(data, append);
      })
      .catch(function (err) {
        if (!flat) return;   // the duplicates list reports its own failures
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_review + err));
      });
  }

  function selectReviewSlice(slice) {
    if (REVIEW_SLICES.indexOf(slice) < 0) return;
    reviewSlice = slice;
    reviewBeyond = false;
    reviewSelected = {};
    refreshReviewControls();
    document.getElementById("review-status").textContent = "";
    REVIEW_SLICES.forEach(function (name) {
      document.getElementById("review-slice-" + name)
          .classList.toggle("active", name === slice);
    });
    var grouped = slice === "dupes";
    document.getElementById("review-dupes").style.display = grouped ? "" : "none";
    document.getElementById("review-flat").style.display = grouped ? "none" : "";
    if (grouped && !dupesLoaded) {
      dupesLoaded = true;
      loadDupes();
    }
    return fetchReview(0, false);
  }

  function applyReviewMark(action) {
    var ids = reviewSelectedIds();
    if (!ids.length) return;
    var status = document.getElementById("review-status");
    status.textContent = "";
    return postJson("/api/review/mark", { file_ids: ids, action: action })
      .then(function (resp) {
        if (resp && resp.ok) {
          status.textContent = fmt(I18N.review_marked_status, { n: resp.marked });
          reviewSelected = {};
          refreshReviewControls();
          fetchReview(0, false);
        } else {
          status.textContent = I18N.review_error_prefix + ((resp && resp.error) || "");
        }
      })
      .catch(function (err) { status.textContent = I18N.review_error_prefix + err; });
  }

  function loadReview() {
    return selectReviewSlice(reviewSlice);
  }

  REVIEW_SLICES.forEach(function (name) {
    document.getElementById("review-slice-" + name).addEventListener("click", function () {
      selectReviewSlice(name);
    });
  });
  document.getElementById("review-more-btn").addEventListener("click", function () {
    if (reviewSlice === "blurred" && !reviewBeyond && reviewOffset >= reviewWindowTotal) {
      reviewBeyond = true;
    }
    fetchReview(reviewOffset, true);
  });
  document.getElementById("review-delete-btn").addEventListener("click", function () {
    applyReviewMark("to_delete");
  });
  document.getElementById("review-keep-btn").addEventListener("click", function () {
    applyReviewMark("keep");
  });
  document.getElementById("review-clear-btn").addEventListener("click", function () {
    applyReviewMark("clear");
  });
  document.getElementById("review-select-all-btn").addEventListener("click", function () {
    document.querySelectorAll("#review-grid .review-select").forEach(function (box) {
      box.checked = true;
      reviewSelected[parseInt(box.value, 10)] = true;
    });
    refreshReviewControls();
  });
  document.getElementById("review-select-none-btn").addEventListener("click", function () {
    document.querySelectorAll("#review-grid .review-select").forEach(function (box) {
      box.checked = false;
    });
    reviewSelected = {};
    refreshReviewControls();
  });
  refreshReviewControls();

  // --- the duplicates slice (U3/F32), unchanged inside the new workspace ---

  function postJson(url, data) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(function (r) { return r.json(); });
  }

  var currentGroups = [];

  function groupFileIds(g) {
    return g.frames.map(function (f) { return f.file_id; });
  }

  function selectedKeeper(g) {
    var radios = document.getElementsByName("keep-" + g.group);
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) return parseInt(radios[i].value, 10);
    }
    return null;
  }

  function groupSkipped(g) {
    var checkbox = document.getElementById("skip-" + g.group);
    return !!(checkbox && checkbox.checked);
  }

  function actionLabel(action) {
    if (action === "keep") return I18N.action_keep;
    if (action === "to_delete") return I18N.action_to_delete;
    return "";
  }

  function renderGroup(g) {
    var box = document.createElement("div");
    box.className = "card dupe-group";

    var title = document.createElement("h3");
    title.textContent = fmt(I18N.group_title, { n: g.group + 1, count: g.frames.length });
    box.appendChild(title);

    var table = document.createElement("table");
    // клик по кадру группы -> лайтбокс; стрелки листают кадры этого дубль-набора
    var groupSamples = g.frames.map(function (fr) { return fr.file_id; });
    g.frames.forEach(function (f, frameIdx) {
      var tr = document.createElement("tr");

      var tdRadio = document.createElement("td");
      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "keep-" + g.group;
      radio.value = String(f.file_id);
      radio.checked = f.action === "keep" || (!f.action && f.recommended);
      tdRadio.appendChild(radio);
      tr.appendChild(tdRadio);

      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(f.file_id, groupSamples, frameIdx, f.thumb_url));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = f.name;
      // Как во вкладке «Города»: полный путь — в тултипе имени. У дублей имена
      // совпадают по построению, поэтому единственное, чем кадры различаются на
      // глаз, — это где они лежат.
      nameEl.title = f.src_path ? f.src_path + "\\\\" + f.name : f.name;
      tdThumb.appendChild(nameEl);
      if (f.recommended) {
        var badge = document.createElement("span");
        badge.className = "badge";
        badge.appendChild(icon("check"));
        badge.appendChild(document.createTextNode(I18N.recommended_badge));
        tdThumb.appendChild(badge);
      }
      tr.appendChild(tdThumb);

      var tdMeta = document.createElement("td");
      var dims = f.width && f.height ? f.width + "×" + f.height : "?";
      var kb = Math.round((f.size || 0) / 1024) + " KB";
      // Исходная папка первой, как в «Городах»: при выборе, какой из одинаковых
      // кадров оставить, решает обычно именно она.
      tdMeta.textContent = [f.src_dir, dims, kb, actionLabel(f.action)]
          .filter(Boolean).join(" · ");
      if (f.src_path) { tdMeta.title = f.src_path; }
      tr.appendChild(tdMeta);

      var tdActions = document.createElement("td");
      tdActions.className = "plan-actions";
      var btnFrameDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnFrameDelete.addEventListener("click", function () {
        deletePhoto(f.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnFrameDelete);
      tr.appendChild(tdActions);

      table.appendChild(tr);
    });
    box.appendChild(wrapTable(table));

    var skipLabel = document.createElement("label");
    skipLabel.className = "skip-label";
    var skipCheckbox = document.createElement("input");
    skipCheckbox.type = "checkbox";
    skipCheckbox.id = "skip-" + g.group;
    skipLabel.appendChild(skipCheckbox);
    skipLabel.appendChild(document.createTextNode(" " + I18N.skip_group_label));
    box.appendChild(skipLabel);

    var btnTrash = makeBtn("danger", "trash", I18N.delete_dupes_button);
    btnTrash.addEventListener("click", function () {
      var keep = selectedKeeper(g);
      if (keep === null) { window.alert(I18N.alert_choose_keeper); return; }
      var remember = document.getElementById("delete-remember").checked;
      if (!remember && !window.confirm(fmt(I18N.confirm_trash_group, { n: g.group + 1 }))) {
        return;
      }
      postJson("/api/dupes/trash", { group: groupFileIds(g), keep_file_id: keep })
        .then(loadDupes);
    });
    box.appendChild(btnTrash);

    return box;
  }

  function loadDupes() {
    document.getElementById("dupes-save-status").textContent = "";
    fetch("/api/dupes")
      .then(function (r) { return r.json(); })
      .then(function (groups) {
        currentGroups = groups;
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        if (!groups.length) {
          container.appendChild(stateEl("empty", I18N.no_dupes));
          return;
        }
        groups.forEach(function (g) { container.appendChild(renderGroup(g)); });
      })
      .catch(function (err) {
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_dupes + err));
      });
  }

  document.getElementById("dupes-save-all-btn").addEventListener("click", function () {
    var statusEl = document.getElementById("dupes-save-status");
    var groups = [];
    var skip = [];
    currentGroups.forEach(function (g) {
      if (groupSkipped(g)) {
        skip.push(groupFileIds(g));
        return;
      }
      var keep = selectedKeeper(g);
      if (keep === null) return;
      groups.push({ group: groupFileIds(g), keep_file_id: keep });
    });
    if (!groups.length) {
      statusEl.textContent = I18N.select_group_to_save;
      return;
    }
    postJson("/api/dupes/choices", { groups: groups, skip: skip }).then(function (resp) {
      if (resp && typeof resp.saved === "number") {
        statusEl.textContent = fmt(I18N.saved_groups, { n: resp.saved });
      }
      loadDupes();
    });
  });
})();
</script>
</body></html>
"""


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
    "/api/clusters/label", "/api/clusters/merge",
    # ...files moved on disk...
    "/api/dupes/trash", "/api/photo/trash", "/api/photos/trash", "/api/album",
    # ...and config.yaml.
    "/api/source-tree/excludes",
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
            elif path == "/api/search":
                self._serve_search(parse_qs(parts.query))
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
                self._send_json(_tabs_visibility_payload(db_path, cfg.features))
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
                db_path, bucket, offset, limit,
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
            self._send_json(_animals_payload(db_path, cfg.features, offset, limit))

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
                blur_max=cfg.features.blur_review_max,
                max_distance=cfg.index.phash_max_distance))

        def _serve_search(self, query: dict[str, list[str]]) -> None:
            # F134: read-only, and read-only in the strong sense — an empty `q` asks for
            # the state of the index alone and never reaches the model. The sensitive
            # classes come off the LIVE config for the reason `/api/junk` does that.
            parsed = _parse_search_query(query, cfg.features.search_limit)
            if parsed is None:
                self._send_json({"error": "invalid limit"},
                                status=HTTPStatus.BAD_REQUEST)
                return
            text, limit = parsed
            self._send_json(_search_payload(cfg, db_path, text, limit,
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

        def _handle_animal_mark(self) -> None:
            # F124: a row in `manual_pet` and nothing else — no file is touched, no
            # `frame_quality` row is rewritten (that table has one writer, `junk`).
            parsed = _validate_animal_mark_payload(self._read_json_body())
            if parsed is None:
                self._send_json({"error": "invalid body"}, status=HTTPStatus.BAD_REQUEST)
                return
            ids, action = parsed
            self._send_json({"ok": True,
                             **_apply_animal_mark(db_path, cfg.features, ids, action)})

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
