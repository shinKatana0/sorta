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
| geo | `geo.py` | `files` | `places` |
| faces | `faces.py` | `files` | `faces`, `face_clusters` |
| events | `events.py` | `files`, `places` | `events`, `event_files` |
| naming | `naming.py`, `landmarks.py`, `junk.py` | `files`, `places`, `events` | `places` (unknown only), `media_class`, `events.name` (name_is_manual=0 only) |
| sorter | `sorter.py` | all | `move_batches`, `moves`, FS |
| ui/cli | `cli.py`, `ui.py` | everything (read) | — (orchestrate module calls) |

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
- 1:1 with files; `confidence`: `exact_gps` | `session_inferred` | `visual` | `unknown`.
- Idempotency: re-running geo fully recomputes the rows (protected manual edits,
  should they appear, would be behind a separate flag).

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
  `source`: heuristic | clip | vlm (a later one overrides the earlier);
  `score` — NULL for heuristics. Two-tier classifier: fast (CLIP zero-shot + OCR)
  by default, deep (VLM) — opt-in. verdict != photo → the sorter puts the file in
  a separate branch (documents/products/junk), not the main layout.
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
(only high/medium confidence, the nearest-in-time neighbour with GPS) →
full idempotent recomputation of places in one transaction.

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
