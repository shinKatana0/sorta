-- Sorta: index schema.
PRAGMA journal_mode = WAL;
PRAGMA user_version = 10;

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
    confidence TEXT NOT NULL,             -- exact_gps | session_inferred | visual | unknown
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
    source TEXT NOT NULL,                 -- heuristic | clip
    score REAL,                           -- classifier confidence, NULL for heuristics
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
