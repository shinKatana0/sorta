-- Sorta: index schema.
PRAGMA journal_mode = WAL;
PRAGMA user_version = 17;

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
-- no consumer has to renormalize per query. Half precision would halve the table (~30 MB
-- instead of ~60 for 19 757 photos, ~460 MB instead of ~920 for 300 000) and was rejected
-- by measurement, not by taste: over 256 unit vectors of the real width, 18 of 20 queries
-- rank differently in float16 (tests/test_clip_embeddings.py). The reordered pairs are
-- always within 3e-5 of a cosine of each other, but the ranking is what every consumer of
-- this table reads, so it is stored at the precision the encoder produced.
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
