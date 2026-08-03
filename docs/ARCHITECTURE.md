# Sorta — architecture

## 1. Overview

```
                 ┌──────────────────────── SQLite (sorta.db) ────────────────────────┐
                 │  files ──► places ──► events/event_files                               │
                 │    │         ▲                                                        │
                 │    └──► faces/face_clusters          move_batches/moves (journal)     │
                 └───────▲──────────▲──────────▲──────────────▲──────────────────────────┘
                         │          │          │              │
  disk ──► [indexer] ────┘   [geo] ─┘  [faces]─┘   [events]   │
                                                              │
                 [sorter] ◄── reads all, writes moves + moves files ◄── CLI --by ...
```

**Central principle**: the single source of truth is the SQLite index. Each
pipeline module is a pure transformation — "reads some tables → writes its own
tables". Sorting is the materialization of an index view into the filesystem.
Switching the sort mode does not require re-running the pipelines.

## 2. Modules and boundaries

| Module | Files | Reads | Writes |
|---|---|---|---|
| core | `config.py`, `db/`, `hashing.py`, `dates.py`, `exif.py`, `imaging.py` | FS (decode) | — |
| indexer | `indexer.py`, `dedup.py` | FS | `files` |
| geo | `geo.py` | `files`, `geo_cache` | `places`, `geo_cache` (online provider only) |
| faces | `faces.py` | `files` | `faces`, `face_clusters` |
| events | `events.py` | `files`, `places` | `events`, `event_files` |
| naming | `naming.py`, `landmarks.py`, `junk.py` | `files`, `places`, `events` | `places` (unknown only), `media_class`, `events.name` (name_is_manual=0 only) |
| sorter | `sorter.py` | all | `move_batches`, `moves`, FS |
| ui/cli | `cli.py`, `ui.py` | everything (read) | `manual_overrides`, `manual_places`, `manual_pet`, `dedup_choice` — the user's OWN decisions; otherwise orchestrate module calls |

**Architectural boundary invariants:**
1. Modules do NOT import each other (except `core`). Data exchange happens only
   through DB tables; a module's interface = the tables it reads/writes.
2. Each table has exactly one writer (see §3). The only exception is
   `events.name`: `naming` writes it ONLY into rows with `name_is_manual = 0`
   (a predicate in the UPDATE).
3. Pipeline modules are idempotent: re-running recomputes their tables from
   scratch (except protected manual edits — face labels, manual event names).

## 3. Data contracts (stable interfaces between modules)

### files (written only by indexer)
- `path` — absolute, POSIX separators; `dup_of IS NULL` = canonical file.
  All downstream modules work ONLY with the canonical ones (`WHERE dup_of IS NULL
  AND error IS NULL`).
- `taken_at` ISO 8601 + `taken_at_source` (exif|filename|mtime) +
  `taken_at_confidence` (high|medium|low).
- `gps_lat/gps_lon` — WGS84 in degrees, NULL if absent.

### places (written only by geo)
- 1:1 with files; `confidence`: `exact_gps` | `session_inferred` | `trip_inferred` |
  `path_inferred` | `visual` | `unknown`. The inferred levels are told apart on purpose:
  reports, the CSV plan and the UI show how confidently a place was determined (F85a),
  and `path_inferred` (F85c) is country-only by construction — it comes from a folder
  NAME, not from geometry.
- Idempotency: re-running geo fully recomputes the rows. A place the USER assigned is
  therefore not stored here at all — see `manual_places`.

### manual_places (written only by ui) — F85c
- A place the user assigned to a whole group (an event, a source folder) by hand. It
  cannot live in `places`: one writer, and every geo run recomputes that table from
  scratch, so a manual place there would last exactly until the next run.
- The sorter prefers this row over `places` when it builds the plan and reports the file
  as `place_confidence='manual'` — a place the user chose is never presented as one the
  program inferred. The whole place comes from one source: a manual row replaces country,
  city and district together (a country-only assignment leaves city/geonameid NULL and
  lands in the `country_only` branch of the layout).
- Wiped by `reset_index` like every other manual decision.

### manual_pet (written only by ui) — F124
- The user's verdict on one frame's animal mark: `is_animal = 0` takes a false mark off,
  `is_animal = 1` puts a missing one on. Both directions, because a person does both.
- It cannot live in `frame_quality`: one writer (`junk`), recomputed from scratch on every
  run, prompt fingerprint included — the same reasoning as `manual_places` against
  `places`. It is not an action of `manual_overrides` either: that column decides the
  layout, and "this is not a cat" must never drop a file out of it.
- Applied WHEN READ, never when written: `junk` is untouched and keeps computing
  `frame_quality`, while the consumers (the album slice, the "Animals" tab, the
  Overview counter) read `COALESCE(manual_pet.is_animal, <the automatic rule>)` through the
  single expression `sorter.animal_ids_sql`. That is what makes the edit survive a change
  of model, of prompts or of the threshold — it is not in what gets recomputed — and what
  keeps the three numbers from drifting apart.
- F137: the automatic half of that expression is derived at the same moment, out of
  `frame_quality.pet_score`/`pet_vlm` and the thresholds the config holds right then
  (`sorter.animal_auto_sql`) — the stored answer of the check where the current
  `features.pet_candidate_threshold` would still ask for one, `pet_score >=
  features.pet_threshold` otherwise. `frame_quality.pet` stays written as a cache and as
  the column to query a database by, and nothing reads it to decide anything: the
  thresholds are deliberately not part of `quality_prompt_fingerprint` (hashing them would
  re-run the whole cascade on every edit), so a label read as a verdict froze the config of
  whichever run last wrote it.
- F160: the detector (F154) is a tier of that same expression — between the F130 answer and
  the CLIP score, over the `detections` row this detector left, and read off the stored
  BOXES so `features.detector_threshold` is re-chosen without a run like the two thresholds
  above it. With `detect.enabled` or `features.detector` off the expression is byte-for-byte
  the F137 one. The rule is written twice — `junk.pet_label` for the stage,
  `sorter.animal_auto_sql` for every reader — and since F160 a case table is run through
  both and asserted equal, because four sources decide this label and nothing was checking
  that each reached both halves.
- Wiped by `reset_index` like every other manual decision.

### geo_cache (written only by geo, online provider) — F93
- What the online provider SAID, never where a file ends up: the key is the
  `(city_geonameid, district_geonameid)` pair of the bundled base (a grid cell rounded
  to `geo.cache_coord_digits` for coordinates that base cannot place) plus the provider;
  the value is country + city/district/country names in ALL THREE languages, so
  switching folder language costs no network.
- A row is written only for a COMPLETE answer (all three languages resolved) — a failed
  request must not be frozen into the collection. Rows older than
  `geo.cache_max_age_days` are asked again.
- The only table `reset_index` spares: a reset is about the user's files, not about the
  name of a point on the map. Clear it explicitly with `sorta cache --clear-geo`,
  `sorta reset --clear-geo` or the checkbox in the reset dialog of `sorta ui`.

### faces / face_clusters (written only by faces)
- `embedding` — BLOB float32 (512, ArcFace little-endian).
- `face_clusters.label` — person name; `merged_into` — the merge chain, the
  effective cluster = the root of the chain.
- Re-clustering must preserve labels (matching old clusters to new ones by the
  intersection of their face sets).

### events / event_files (written by events; naming edits only name)
- Event = a time cluster (gap > config) × place; `origin` auto|manual
  (manual — `events add`, recomputation does not recreate them).
- `name_is_manual = 1` — the name is not overwritten by recomputation (F4) or by
  the name provider (F6). F6 (naming) writes `events.name` ONLY into rows with
  name_is_manual=0 — the only permitted cross-module write, protected by a
  predicate in the UPDATE.

### media_class (written only by naming/junk)
- 1:1 with files; `verdict`: photo | screenshot | meme | document | product;
  `source`: heuristic | clip | ocr | vlm (a later one overrides the earlier);
  `score` — NULL for heuristics. Two-tier classifier: fast (CLIP zero-shot + OCR)
  by default, deep (VLM) — opt-in. verdict != photo → the sorter puts the file in
  a separate branch (documents/products/junk), not the main layout.
- `tier`: heuristic | clip | vlm — WHICH tier produced the row, and the only thing
  incrementality is decided on: a run redoes exactly the rows whose `tier` differs
  from the one it is running (so a repeated run with the same tier processes
  nothing, and the deep tier is paid for once).
- A missing row = "not classified" — the sorter treats it as photo.

### move_batches / moves (written only by sorter)
- A row in `moves` with `status='planned'` is created BEFORE the FS operation;
  after verify — `done`. `undo` walks the journal in reverse order.

## 4. Key scenarios

### index (Phase 1, implemented)
walk → filter by extension/size → incremental check path+size+mtime →
batch of 200: exiftool -json -n (or Pillow fallback) + blake3 + pHash + date cascade →
UPSERT in one transaction per batch (Ctrl+C-safe) → dedup pass.

### geo (Phase 2, implemented)
batch reverse_geocoder (`mode=1`, offline) for files with GPS → sessions by taken_at
(gap from `geo.session_gap_hours`) → place inheritance for files without GPS
(only high/medium confidence, the nearest-in-time neighbour with GPS) → a second pass
over TRIPS (F85a): geo groups its own sessions the way `events` does (the same
`events.trip_merge_gap_hours`/`trip_merge_max_km`, because on a clean run the `events`
table does not exist yet and `places` has a single writer), and a still place-less file
inherits the trip's place when the trip's GPS frames agree about the city (the dominant
one holds > 50% of them) and the file lies between two of those frames in time → a last
pass over what none of that reached (F85c): the COUNTRY named by a folder on the file's
path, country-only because a country read from a folder name is right 99.5% of the time
and a city 4.3% → full idempotent recomputation of places in one transaction.
With `geo.provider: online` the network sits in front of that: coordinates are grouped
by their city+district pair from the bundled base, each group is looked up in
`geo_cache`, and only a miss goes to Nominatim — three requests (ru/en/ja) about the
median point of the group, then one row. The recompute itself is unchanged.

### faces (Phase 3, implemented)
insightface buffalo_l (CUDA with a CPU fallback; `_enable_cuda_dll_dirs` for
pip-wheel CUDA) → quality filter (min_face_px, det_threshold) → embeddings
into faces (the bbox='[]' marker = "processed, no faces") → HDBSCAN on normalized
vectors, preserving labels across recomputation (>50% intersection) →
label/merge/contact sheets.

### sort --by city (Phase 2)
plan: `SELECT ... JOIN places` → template `Country/City/YYYY/name` → name-conflict
resolution → dry-run report (console + CSV). apply: for each file
journal(planned) → move (rename or copy+verify+delete) → journal(done).

### Failure handling
- A corrupt file → `files.error`, processing continues.
- Interrupting index → the unfinished batch is rolled back by the transaction; a
  re-run finishes indexing.
- Interrupting sort --apply → `moves.status='planned'` marks the stop point;
  `undo` reverses what finished, a repeated `sort` continues.

## 5. Technology choices and their reasons
- **SQLite + WAL** — single user, local, transactions; embeddings fit into a BLOB,
  100k rows is a trivial volume.
- **exiftool in batches** — the only tool that reliably reads HEIC/RAW/video;
  batching removes the process-startup cost (the main performance killer).
- **blake3** — many times faster than sha256 on large files; sha256 fallback for
  environments without the package (the algorithm is recorded in `hash_algo`).
- **Fallbacks everywhere** (exiftool→Pillow, blake3→sha256, typer→argparse,
  imagehash→skip pHash) — the core is testable on bare Python, and degradation is
  explicit rather than a crash.
- **insightface + hdbscan** (Phase 3) — the best open out-of-the-box stack for
  face embeddings; HDBSCAN does not require knowing the number of clusters ahead.
- **Face clustering on GPU, but HDBSCAN on CPU** — embeddings number in the
  hundreds of thousands, the CPU handles it in minutes.
