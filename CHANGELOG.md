# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
