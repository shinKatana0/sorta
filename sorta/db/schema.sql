-- Sorta: index schema.
PRAGMA journal_mode = WAL;
PRAGMA user_version = 20;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,            -- absolute POSIX path
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    ext TEXT NOT NULL,
    media_type TEXT NOT NULL,             -- photo | raw | video
    hash TEXT,
    hash_algo TEXT,                       -- blake3 | sha256
    phash TEXT,
    taken_at TEXT,                        -- ISO 8601, local capture time
    taken_at_source TEXT,                 -- exif | filename | mtime
    taken_at_confidence TEXT,             -- high | medium | low
    gps_lat REAL,
    gps_lon REAL,
    camera_make TEXT,
    camera_model TEXT,
    width INTEGER,
    height INTEGER,
    orientation INTEGER,                  -- EXIF 274: 1..8, NULL if absent (v2)
    not_personal INTEGER NOT NULL DEFAULT 0, -- 1 = not personal media (movie/series,
    --                                          F17-video-guard, v5): sorted
    --                                          into _Unsorted/not_personal
    dup_of INTEGER REFERENCES files(id),  -- NULL = canonical instance
    error TEXT,                           -- processing error text, NULL if ok
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS idx_files_taken ON files(taken_at);
CREATE INDEX IF NOT EXISTS idx_files_dup ON files(dup_of);

-- Phase 2 (owner: F2-geo)
CREATE TABLE IF NOT EXISTS places (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    country TEXT,                         -- ISO cc (RU)
    country_name TEXT,                    -- v10 (online): full country name from Nominatim in the config language; offline NULL (name from i18n.country by cc)
    region TEXT,                          -- DEPRECATED (G2 does not write it; NULL) — kept, dropped later
    city TEXT,                            -- canonical name (asciiname/en) for --where/CSV/landmark fallback
    city_geonameid INTEGER,               -- G2: city geonameid (GeoNames), NULL for landmark/unknown
    district_geonameid INTEGER,           -- G2: district geonameid, NULL if none/landmark
    district_name TEXT,                   -- G2b (online): district name from Nominatim; offline NULL (district from geonameid)
    confidence TEXT NOT NULL,             -- exact_gps | session_inferred | trip_inferred (F85a) | path_inferred (F85c) | visual | unknown
    updated_at TEXT NOT NULL
);

-- Phase 3 (owner: F3-faces)
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    bbox TEXT NOT NULL,                   -- JSON [x1,y1,x2,y2]
    embedding BLOB NOT NULL,              -- float32 ArcFace
    cluster_id INTEGER
);
CREATE TABLE IF NOT EXISTS face_clusters (
    id INTEGER PRIMARY KEY,
    label TEXT,                           -- person name, NULL until named
    merged_into INTEGER REFERENCES face_clusters(id)
);
CREATE INDEX IF NOT EXISTS idx_faces_file ON faces(file_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);

-- Phase 4 (owner: F4-events)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    place_city TEXT,
    name TEXT NOT NULL,
    name_is_manual INTEGER NOT NULL DEFAULT 0, -- a manual name survives recomputation
    origin TEXT NOT NULL DEFAULT 'auto'        -- auto | manual: manual events (events add,
                                               -- v4) are not recreated by recomputation, only
                                               -- their date-range files are reattached
);
CREATE TABLE IF NOT EXISTS event_files (
    event_id INTEGER NOT NULL REFERENCES events(id),
    file_id INTEGER NOT NULL REFERENCES files(id),
    PRIMARY KEY (event_id, file_id)
);

-- Phase 5 (owner: F6-naming, v3): photo/junk classification
CREATE TABLE IF NOT EXISTS media_class (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    verdict TEXT NOT NULL,                -- photo | screenshot | meme | document
    source TEXT NOT NULL,                 -- which signal decided: heuristic | clip | ocr | vlm
    score REAL,                           -- classifier confidence, NULL for heuristics
    updated_at TEXT NOT NULL,
    -- v11: which TIER processed the row (heuristic | clip | vlm). Distinct from
    -- `source`: a file the vlm tier deliberately skipped (the gate judged it a clear
    -- personal photo) still keeps source='clip' but was fully handled by tier='vlm'.
    -- Conflating the two made every run reclassify the whole collection.
    tier TEXT
);

-- v15 (F113): per-frame quality signals, each taken with the CHEAPEST tool that can
-- answer it — the laplacian for sharpness, the junk stage's own CLIP call for pets, and
-- the VLM only for what neither of them decides. One row per canonical photo the frame
-- quality stage touched.
--
-- NULL MEANS "NOT ASKED", NOT "NO". The distinction is the whole point of the nullable
-- columns: a consumer that reads a NULL `eyes_open` as "eyes closed" would throw away
-- frames nobody ever looked at. Only `source` is NOT NULL — a row exists exactly because
-- some tier processed it.
CREATE TABLE IF NOT EXISTS frame_quality (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    sharpness REAL,                       -- variance of the laplacian over the shared
    --                                       preview, at a FIXED resolution (the number is
    --                                       scale-dependent — see features.sharpness_max_edge).
    --                                       NULL = the frame did not decode.
    pet TEXT,                             -- 'animal' when pet_score reached
    --                                       features.pet_threshold; NULL = below it, or
    --                                       features.pets is off (then pet_score is NULL too).
    --                                       F122: it used to be cat | dog | pet. The three
    --                                       prompts still run — they are the ensemble the
    --                                       threshold was calibrated on — but WHICH of them
    --                                       won is not recorded, because on a labelled sample
    --                                       the binary call was 92% right and the class
    --                                       assignment was not (drawn cats, plush toys and
    --                                       fur coats all landed as a species).
    pet_score REAL,                       -- the pet-group CLIP score, written whether or not
    --                                       it reached the threshold (so a threshold can be
    --                                       re-measured without a new pass)
    -- v17 (F130): what the VLM answered about this frame — real | depiction | none — when
    -- the pet score put it past features.pet_candidate_threshold and features.pets_verify
    -- was on. NULL means NOT ASKED, here as everywhere in this table: below the candidate
    -- threshold, the check switched off, the model unavailable, or an answer that did not
    -- parse. The column is what keeps a rejected frame distinguishable from an unasked one
    -- — without it every later run would re-ask the ~500 frames the model already turned
    -- down, at 0.78 s each, and the interface would have nothing to explain a removed
    -- label with. `pet` above is the DECISION: 'animal' when pet_vlm = 'real', or when
    -- pet_vlm IS NULL and pet_score >= features.pet_threshold.
    pet_vlm TEXT,
    eyes_open INTEGER,                    -- the VLM answers: 1 | 0 | NULL (not asked, or the
    has_subject INTEGER,                  -- answer did not parse — never guessed as 0).
    --                                       eyes_open is kept only where a face was detected
    --                                       (F121): the model answers it on cats otherwise.
    is_accidental INTEGER,                -- RETIRED by F122, always NULL. The question was
    --                                       right 5% of the time on a labelled sample. The
    --                                       column stays because NULL already means "not
    --                                       asked", which is the truth, and dropping a column
    --                                       in SQLite costs a table rebuild.
    source TEXT NOT NULL,                 -- classic | clip | vlm — WHICH TIER processed the
    --                                       row, and with it the incrementality marker (the
    --                                       F68 lesson, one column that means the tier)
    updated_at TEXT NOT NULL
);

-- v16 (F128): the CLIP vector of a canonical photograph, kept instead of thrown away.
-- The junk stage computes one for every frame it looks at, reads three scores off it and
-- used to discard it — so every feature of that class (search by words, an album from a
-- query, scene clustering, "frames like this one") began with a full CLIP pass over the
-- collection. This table is that pass, already paid for.
--
-- A table of its own and not a column on `files`: the data is bulky and optional, and
-- `files` is read by everything.
--
-- `model` IS THE FEATURE, not decoration. Vectors from different models are not
-- comparable, and a vector without a record of what produced it is rubbish that looks
-- like data — a search over it returns plausible nonsense and nobody can tell why. So the
-- row carries what computed it, and a mismatch with the current config means RECOMPUTE,
-- never use (the same rule the F120 prompt fingerprint follows).
--
-- `vec` is L2-NORMALIZED FLOAT32, little-endian (`dim` numbers, 4 bytes each) — the same
-- wire format `faces.embedding` uses. Normalized so cosine similarity is a dot product and
-- no consumer has to renormalize per query.
--
-- FLOAT16 IS SETTLED: DO NOT CONVERT. It is the obvious saving — the table halves, ~30 MB
-- instead of ~61 for 19 753 photos, ~461 MB instead of ~922 at 300 000 — and it has been
-- measured twice, from both ends, and rejected both times.
--
-- (1) Ranking (F128, tests/test_clip_embeddings.py): over 256 unit vectors of the real
--     width, 18 of 20 queries come back in a different order in half precision. Repeated
--     on the 19 753 REAL vectors (which cluster far tighter than random ones): 137 of 200
--     queries reorder inside the top 50 — but the first result never changed once, and
--     every reordered pair sits within 2.1e-04 of a cosine of its neighbour, none above
--     1e-3. So on its own this objection is weak: nothing a person would notice moves.
--
-- (2) SPEED, and this is the one that decides it. numpy has no native float16 matmul: it
--     upcasts the whole matrix on EVERY query, so the saved memory is paid for by doing
--     the work twice. Measured on this machine over 19 753 vectors:
--
--         float32   0.9 ms per query,  61 MB   ->  300 000 photos:   14 ms,  922 MB
--         float16  70.2 ms per query,  30 MB   ->  300 000 photos: 1066 ms,  461 MB
--
--     78x slower. Half a gigabyte of RAM against a full second of latency on every
--     keystroke of a search — and latency is the only reason this table exists at all.
--
-- The trap is that (1) alone reads like "a fifth-decimal reordering, who cares, put it
-- back to float16". Whoever thinks that will be right about the ranking and will still
-- make the search unusable. Storage is 38% of the database (61 MB of 160) — the cheapest
-- of the three costs, not the one to optimise.
--
-- Population: canonical photographs, the same as `frame_quality` and for the F120 reason
-- — the embedding of a screenshot is noise in a search over personal photos. A frame CLIP
-- was never run on (a heuristics-only run) has no row: NULL does not happen here, the row
-- simply is not there.
CREATE TABLE IF NOT EXISTS clip_embeddings (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    model TEXT NOT NULL,                  -- "<open_clip model>/<weights>" — see above
    dim INTEGER NOT NULL,                 -- numbers in `vec` (768 for ViT-L-14)
    vec BLOB NOT NULL,                    -- dim x float32, little-endian, L2-normalized
    updated_at TEXT NOT NULL
);

-- v18 (F132): which frame of a near-duplicate group is worth keeping — a RECOMMENDATION
-- and nothing else. Nothing here deletes, moves or marks a file; the only table that
-- records a DECISION about a duplicate is `dedup_choice` below, and it is written by the
-- user's hand alone. The two tables are deliberately separate for exactly that reason.
--
-- `group_key` IS THE PRIMARY KEY BECAUSE GROUPS HAVE NO ID. They are not stored anywhere:
-- `dedup.near_duplicate_groups` recomputes them on every call with a union-find over the
-- pHashes, so there is no stable identifier to reference and no row that could own one.
-- The key is a sha1 over the SORTED file ids of the group, which makes it self-invalidating:
-- add a frame to the burst or delete one from it and the group hashes to something else, so
-- the stored answer is not found and the question is asked again about the group that
-- actually exists now. The same device as the F120 prompt fingerprint, and here it is the
-- only way to tie an answer to a group at all.
--
-- `source` says WHO chose: `sharpness` — the ranking the interface has always used (the
-- laplacian inside a group, where it is finally comparable, then resolution and size), or
-- `vlm#<fingerprint>` — the model, with the fingerprint of the question it was asked.
-- The fingerprint is what makes a prompt edit invalidate the answers it produced, and the
-- distinction is what lets the interface say which of the two suggested this frame instead
-- of asking the user to trust a star.
CREATE TABLE IF NOT EXISTS group_keeper (
    group_key TEXT PRIMARY KEY,           -- sha1 of the sorted file ids of the group
    keeper_id INTEGER NOT NULL REFERENCES files(id),
    source TEXT NOT NULL,                 -- sharpness | vlm#<prompt fingerprint>
    updated_at TEXT NOT NULL
);

-- v20 (F131): what the local VLM answered about a landmark CLIP proposed.
--
-- The stage writes ONLY into `places`, and only when a match survives; a proposal it
-- threw away used to leave no trace at all. That cost twice over. A rejected frame stays
-- `confidence='unknown'`, so the next run selects it again, proposes the same landmark
-- again and pays for the same VLM question again (the F130 loss, in a stage where one
-- question is ~0.8 s). And the SCORE that produced the proposal was nowhere either, which
-- is why the size of the uncertainty band could not be answered with a query and had to
-- be re-measured by `scripts/measure_landmarks.py --probe`.
--
-- One row per (file, proposed landmark): the question asked is about that pair, and CLIP
-- proposing a different landmark next time is a different question rather than a stale
-- answer. `model` carries the runtime AND the fingerprint of the question ("<model>#<8
-- hex>"), the group_keeper device above — editing the prompt has to invalidate the
-- answers it produced, and nobody should have to remember to empty a table by hand.
--
-- NOT a place: this table never decides where a file goes. It records what was asked and
-- what came back; F75 corroboration still has the last word over every confirmation.
CREATE TABLE IF NOT EXISTS landmark_checks (
    file_id INTEGER NOT NULL REFERENCES files(id),
    landmark TEXT NOT NULL,               -- the proposed landmark's name, as in the list
    score REAL,                           -- the CLIP probability that proposed it
    verdict TEXT NOT NULL,                -- confirmed | rejected
    model TEXT NOT NULL,                  -- <vlm model>#<prompt fingerprint>
    updated_at TEXT NOT NULL,
    PRIMARY KEY (file_id, landmark)
);

-- v7 (U3): user decisions on near-duplicates from the web app (sorta ui).
-- action='to_delete' — the sorter (U3b) moves the file into the _delete folder on sort --apply;
-- 'keep' — the kept frame of the group. Trash (send2trash) deletes the files rows immediately.
CREATE TABLE IF NOT EXISTS dedup_choice (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    action TEXT NOT NULL,                 -- keep | to_delete
    updated_at TEXT NOT NULL
);

-- v12 (F77): manual corrections made in the web app, which outrank every automatic
-- rule — the user looked at the frame, the classifier did not. One row per file: it
-- is either excluded from the layout or reassigned, never both. Wiped by
-- `reset_index` like every other manual decision (face labels, event names, dedup
-- choices): a from-scratch reindex starts from a clean slate.
CREATE TABLE IF NOT EXISTS manual_overrides (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    action TEXT NOT NULL,                 -- exclude | reassign
    -- reassign: destination folder relative to the sort root, POSIX separators.
    -- NULL for exclude. Comes from outside, so the sorter must validate it before
    -- building a path from it.
    target TEXT,
    updated_at TEXT NOT NULL
);

-- v14 (F85c): a place the USER assigned, to a whole event or a whole source folder at
-- once. It cannot live in `places`: that table has exactly one writer (`geo`) and every
-- geo run recomputes it from scratch, so a manual place written there would survive
-- until the next run and no longer. Here it survives every recompute and is applied
-- where the layout is decided — the sorter prefers this row over `places` and reports
-- the file as confidence='manual', so a place the user chose is never mistaken for one
-- the program inferred.
--
-- The whole place comes from ONE source (the same rule F86 follows for the online/
-- offline mix): a manual row replaces country + city together, district included, and
-- never has a Nominatim country glued onto a hand-picked city. A country-only
-- assignment leaves city/city_geonameid NULL and lands in the `country_only` branch.
CREATE TABLE IF NOT EXISTS manual_places (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    country TEXT NOT NULL,                -- ISO cc; always known (a city implies its country)
    city TEXT,                            -- canonical en/asciiname anchor, NULL = country only
    city_geonameid INTEGER,               -- GeoNames id of the city, NULL = country only
    updated_at TEXT NOT NULL
);

-- v17 (F124): the user's verdict on ONE frame's animal mark, which outranks the model's.
-- The pet threshold is 92% right, so out of 805 marked frames of the live collection some
-- 64 are not animals; the person who sees them in the "Animals" tab is the one who takes
-- the mark off, and it needs somewhere to live.
--
-- It cannot live in `frame_quality`: that table has exactly one writer (`junk`) and every
-- run recomputes it from scratch — after F120 the prompt fingerprint invalidates the rows
-- outright — so a manual mark written there would last until the next run and no longer.
-- The same reasoning, and the same shape, as `manual_places` (F85c) against `places`.
--
-- It is also NOT an action of `manual_overrides`: that column is about the LAYOUT
-- (exclude | reassign), and folding "this is not a cat" into it is how a file ends up
-- dropped from the layout because of what is in the frame.
--
-- The mark is applied WHEN READ, never when written: `junk` keeps computing
-- `frame_quality.pet` exactly as before, and the consumers (the album slice in
-- sorter.py, the tab in ui.py) prefer this row over it — one rule, `sorter.ANIMAL_IDS_SQL`.
-- That is what makes the edit survive any recompute at all, including a change of model,
-- of prompts or of the threshold: it does not live in what gets recomputed.
--
-- Two-way on purpose: a person takes a false mark OFF and puts a missing one ON. The
-- second direction is not hypothetical — it is how a frame the threshold missed gets into
-- the album at all. Wiped by `reset_index` with every other manual decision (face labels,
-- event names, dedup choices): a from-scratch reindex starts from a clean slate.
CREATE TABLE IF NOT EXISTS manual_pet (
    file_id INTEGER PRIMARY KEY REFERENCES files(id),
    is_animal INTEGER NOT NULL,           -- 1 = an animal (put the mark back), 0 = not one
    updated_at TEXT NOT NULL
);

-- Phase 2/5 (owner: F5-sorter): move journal for undo
CREATE TABLE IF NOT EXISTS move_batches (
    id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL,                   -- city | person | event
    dest_root TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    operation TEXT NOT NULL DEFAULT 'move'  -- v8 (C16): move | copy — undo distinguishes them
);
CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES move_batches(id),
    file_id INTEGER NOT NULL REFERENCES files(id),
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    hash TEXT NOT NULL,
    status TEXT NOT NULL                  -- planned | done | undone | failed
);

-- v13 (owner: F93-geo-cache): answers of the ONLINE geo provider, kept ACROSS runs.
-- geo recomputes places from scratch every time (session inheritance needs it), and
-- the in-memory cache died with the process — so adding 200 photos cost the same ~35
-- minutes of Nominatim as a full run. This table is the network, not the decision:
-- it caches what the provider said, never where a file ends up.
--
-- `key` is built by the code, not by SQL, because it has two shapes and SQLite would
-- treat NULLs in a composite PK as distinct rows:
--   "c:<city_geonameid>/<district_geonameid>"  — the normal key. The local base
--       already partitions coordinates by MEANING, which beats any grid: measured on
--       a 14 254-file collection, 603 requests against 6 219 for a 110 m grid, and
--       zero localities mixed against 0.9%.
--   "g:<lat>/<lon>"  — the fallback for coordinates the local base cannot resolve,
--       rounded to 3 digits (~110 m).
--
-- All three interface languages are stored side by side: language is a property of
-- the DATA, not of the run. Switching folder language must not cost a network pass,
-- and it used to leave the cities in the old language until the next full geo.
CREATE TABLE IF NOT EXISTS geo_cache (
    provider TEXT NOT NULL,               -- only 'online' writes here; offline never does
    key TEXT NOT NULL,                    -- see the two shapes above
    country TEXT,                         -- ISO cc — the same in every language
    country_name_ru TEXT, country_name_en TEXT, country_name_ja TEXT,
    city_ru TEXT, city_en TEXT, city_ja TEXT,
    district_ru TEXT, district_en TEXT, district_ja TEXT,
    updated_at TEXT NOT NULL,             -- for the staleness policy (borders do move)
    PRIMARY KEY (provider, key)
);
