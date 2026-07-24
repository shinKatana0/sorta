# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/shinKatana0/sorta/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shinKatana0/sorta/releases/tag/v0.1.0
