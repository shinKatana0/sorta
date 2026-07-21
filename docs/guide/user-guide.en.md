# Sorta — User Guide (English)

> Languages: **English** · [Русский](user-guide.ru.md) · [日本語](user-guide.ja.md)

Sorta is a command‑line and local‑web tool that **indexes a large photo/video
collection** (tested on 60+ GB, designed for 300+ GB) and **sorts the files into a
new folder structure** — by **city/country**, by **person**, or by **event** — with
full safety guarantees (dry‑run by default, a move journal, and one‑command undo).

- **Local by default.** All ML models (faces, scene/text detection) run offline on
  your machine (GPU recommended). Nothing is uploaded unless you explicitly enable
  an online provider in the config.
- **Your originals are never modified.** Sorting *moves* or *copies* files; EXIF is
  never rewritten. With `--copy`/`--link` the originals stay exactly where they are.
- **Two ways to use it:** a guided **web UI** (`sorta ui`) or the **CLI**. They wrap
  the same engine — pick whichever you prefer.

---

## 1. Contents

1. [Requirements](#2-requirements)
2. [Installation](#3-installation)
3. [Configuration](#4-configuration)
4. [Core concepts](#5-core-concepts)
5. [Quick start — Web UI (recommended)](#6-quick-start--web-ui-recommended)
6. [Quick start — CLI](#7-quick-start--cli)
7. [The processing pipeline](#8-the-processing-pipeline)
8. [Sorting: cities, people, events](#9-sorting-cities-people-events)
9. [Duplicates](#10-duplicates)
10. [People & face clusters](#11-people--face-clusters)
11. [Events](#12-events)
12. [Albums (collect a slice into a folder)](#13-albums)
13. [Junk, screenshots & documents](#14-junk-screenshots--documents)
14. [Safety, undo & privacy](#15-safety-undo--privacy)
15. [Full command reference](#16-full-command-reference)
16. [Configuration reference](#17-configuration-reference)
17. [Troubleshooting](#18-troubleshooting)

---

## 2. Requirements

| Component | Requirement |
|---|---|
| OS | Windows, Linux or macOS |
| Python | 3.11 – 3.14 (`requires-python >=3.11,<3.15`) |
| Package/env manager | [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip` |
| `exiftool` | **Required** for HEIC/RAW/video metadata (dates, GPS, orientation) — essentially any modern phone photo. Without it Sorta falls back to Pillow, which only reads JPEG/PNG/TIFF/WEBP and no video. |
| Disk space | Enough for the new structure. `--copy` duplicates data (×N). `--link` (hardlinks) uses almost no extra space (same volume, NTFS/ext4/APFS). Plus the SQLite index and (optionally) thumbnails, both small relative to the photo collection. |

Sorta's ML backend (faces, CLIP/OCR for junk classification) is installed via one of
two **mutually exclusive install profiles** — pick the one matching your hardware:

| | CPU profile (`--extra cpu`) | GPU profile (`--extra gpu`) |
|---|---|---|
| Hardware | Any x86‑64 machine, no GPU needed | NVIDIA GPU + driver supporting **CUDA 13** (verified on Blackwell/RTX 5090) |
| Backend | `onnxruntime` (CPU) + CPU‑build torch/torchvision | `onnxruntime-gpu` + CUDA 13/cuDNN 9 runtime (pip wheels) + CUDA‑build torch/torchvision |
| Faces / CLIP speed | Works, correctly, just **slow** — expect hours of `faces`/`junk`/`landmarks` on a large collection. Fine for city‑sorting + duplicates (faces/events are opt‑in anyway, see §8), usable for smaller collections with faces/events on. | Fast. Reference timings from our own 6,298‑photo test collection, faces+events+junk enabled: **≈ 45 min** (fast/CLIP tier), **≈ 77 min** with the optional deep VLM tier (`naming.vlm_enabled` / `uv sync --extra vlm`). |
| RAM | 8 GB+ recommended (indexing/hashing is the RAM‑heavy part, independent of profile) | Same, plus whatever the GPU driver reserves |
| VRAM | n/a | **~3 GB** for base + faces (measured on RTX 5090: CLIP ViT‑L ≈2.0 GB + buffalo_l ≈0.6 GB) — a **≥4 GB** GPU is comfortable. The optional deep VLM tier (Qwen2.5‑VL‑3B) adds ≈7 GB (estimated from the 3B fp16 model, not measured) → **≥8 GB** total |

The timings and VRAM figures are observations from our hardware, not a guarantee —
your mileage will vary with collection composition (video previews, RAW files, and
faces per photo are the main cost drivers).

---

## 3. Installation

```bash
git clone https://github.com/shinKatana0/sorta.git
cd sorta

# Install exiftool — REQUIRED for HEIC/RAW/video metadata:
#   Windows: winget install OliverBetz.ExifTool
#   Debian/Ubuntu: sudo apt install libimage-exiftool-perl
#   macOS: brew install exiftool

# Create your config from the template
cp config.example.yaml config.yaml
```

Sorta's ML backend (faces, CLIP/OCR) needs exactly one hardware profile installed
— `cpu` or `gpu`, mutually exclusive (see §2). There are two supported ways to
install it, both **"set once, then just run `sorta`"** — pick based on what you're
doing:

### A) Global install with `uv tool install` (recommended for regular use)

```bash
uv tool install ".[cpu]"        # no NVIDIA GPU
# or
uv tool install ".[gpu]"        # NVIDIA GPU + CUDA 13 driver
```

This resolves `pyproject.toml`'s profile/index setup (the `pytorch-cu130` /
`pytorch-cpu` indexes) exactly like `uv sync` does, and installs a `sorta` command
onto your PATH — verified to give a real CUDA 13 torch build on the `gpu` profile
(`torch.cuda.is_available()` → `True`). From here on, just run `sorta ui`, `sorta
index …`, etc. from any terminal, in any directory — no `uv run`, no active
virtualenv.

- **Switch profile** (moved to different hardware, or picked the wrong one) —
  reinstall with `--force` and the other extra:
  `uv tool install --force ".[gpu]"` (or `".[cpu]"`).
- **Update after `git pull`** — `uv tool install --force ".[<profile>]"`. This
  installs a fresh snapshot of the current code; it is **not** an editable
  install, so local edits need path B below to be picked up automatically.
- Once Sorta is published to PyPI, the same idea becomes
  `uv tool install "sorta[gpu]"` (or `"sorta[cpu]"`) — no local checkout needed.

### B) Project venv with `uv sync` (for developing on the code)

```bash
uv sync --extra cpu --extra dev      # no NVIDIA GPU
# or
uv sync --extra gpu --extra dev      # NVIDIA GPU + CUDA 13 driver

# Activate it once per shell session:
.\.venv\Scripts\Activate.ps1         # Windows PowerShell
source .venv/bin/activate            # Linux/macOS/bash
```

With the venv active, `sorta …` runs straight out of your checkout (editable
install) — code edits are visible immediately, no reinstall step.

> **Don't run `uv run sorta …` as your everyday command.** `uv run <cmd>`
> re‑syncs the environment against `pyproject.toml`'s base dependency set before
> every invocation — unless you repeat `--extra <profile>` on that exact command
> every single time, the resync silently drops your GPU packages (torch falls
> back to a CPU build) each time you run it. Path A (`uv tool install`) and path B
> (an activated venv) both sidestep this entirely, which is the whole point of
> installing once instead of invoking through `uv run`.

Always pass an explicit `--extra cpu` or `--extra gpu` in either path — they're
marked mutually exclusive in `pyproject.toml` so `uv` resolves the right
torch/onnxruntime build (GPU wheels pull the CUDA 13 runtime as regular pip
packages, no system CUDA Toolkit needed). Neither profile is chosen for you.

`--extra dev` adds the dev tools (ruff, mypy, pytest) — needed if you'll run
`scripts/check.py` or the test suite, not required just to run `sorta`. There's also
an optional `--extra vlm` for the deep VLM classification tier (`naming.vlm_enabled`);
without it, that tier gracefully falls back to the fast CLIP tier.

---

## 4. Configuration

Sorta reads `config.yaml` (copy it from `config.example.yaml`). The two settings you
must review:

```yaml
sources:
  - "D:/Photos"          # folder(s) with your photos/videos (scanned recursively)
database: "sorta.db"     # where the SQLite index is stored
language: ru             # UI/folder language: ru | en | ja  (default ru)
```

- **`sources`** — one or more root folders to scan. You can also pass the folder on
  the command line (`sorta index /path/to/photos`), which overrides this.
- **`language`** — controls the language of generated **folder names** (e.g.
  `Россия/…` vs `Russia/…`) and the **web UI** chrome. Supported: `ru`, `en`, `ja`.

> **Note:** `language` does **not** affect the CLI's own console messages (progress
> lines like `Готово: +13 новых, ...`) — those are fixed text, independent of config.
> Folder names and the web UI *are* fully localized. See the worked examples in §9
> for what real CLI output looks like, and §18 if it renders as `????` in your
> terminal.

See the [Configuration reference](#17-configuration-reference) for every option.

---

## 5. Core concepts

**Index is separate from sorting.** First a pipeline fills a SQLite **index**
(metadata, geolocation, face embeddings, clusters, events, junk classification).
Sorting is just *applying a view* of that index to the filesystem. Switching sort
modes (city ↔ person ↔ event) does **not** require re‑scanning.

**Dry‑run by default.** `sort` and `album` print a plan and write nothing until you
add `--apply`. Always review the plan first.

**Journal & undo.** Every move/copy/link is written to a journal *before* the
filesystem operation; `sorta undo` reverses the last batch. Hashes (blake3) are
verified before moving; name conflicts get a `_1`, `_2` suffix — an existing file is
never overwritten.

**Three transfer modes.**
- **move** (default for `sort`) — relocates the file. One structure on disk.
- **copy** — duplicates the file; originals untouched. Multiple structures possible,
  at ×N disk cost.
- **link** (hardlink, default for `album`) — a second name for the same bytes; near
  zero extra space; falls back to copy across volumes/filesystems.

**Canonical structure + albums.** The recommended model: a single **canonical**
structure by city, plus on‑demand **albums** (a specific person / event) collected
into separate named folders via hardlinks.

---

## 6. Quick start — Web UI (recommended)

The web UI is the easiest path and needs no terminal knowledge beyond starting it.

```bash
sorta ui                       # opens a local server on http://127.0.0.1:8756
```

Then in the browser:

1. **Process** tab → enter the path to your photo folder. Two checkboxes, both
   **unchecked by default**: **"Detect faces"** and **"Detect events"** — the
   pipeline's slowest stages, opt‑in on purpose (see §8). Leave them off for a fast
   city‑sorting‑only run, or tick what you need. Click **Process**: it runs in the
   background with per‑stage progress (index → geo → landmarks → [faces] → [events]
   → junk → near‑duplicates — faces/events only if ticked). You can close the tab;
   processing continues.
2. **Cities** tab → review the proposed structure (`Country/City/Year/District`).
   Always visible.
3. **Duplicates** tab → review near‑duplicate groups; the recommended keeper is
   pre‑selected. Adjust where you disagree, then click **Save all choices** once.
   Always visible.
4. **People** tab → only appears once face clusters exist (you ticked "Detect faces"
   at least once, or ran `sorta faces`). Name clusters and merge duplicates of the
   same person.
5. **Events** tab → only appears once events exist (you ticked "Detect events" at
   least once, or ran `sorta events`). Rename events; collect any person/event into
   a folder with **Collect into folder**.
6. **Moves** tab → after you apply a sort/album, see exactly what went where. Always
   visible.

The **Process** tab has two more checkboxes beyond faces/events, both reflecting
`config.yaml` and acting as a full override for this run only (checked = force on,
unchecked = force off) — the UI equivalent of the CLI's `--deep`/`--no-deep` and
`--geo online`/`--geo offline` (§8):

- **"Deep analysis (VLM)"** — use the deep VLM tier instead of the fast CLIP tier
  for junk/document classification. It only actually takes effect if it's *both*
  requested (this checkbox, `--deep`, or `naming.vlm_enabled: true` in config)
  *and* installed (the `vlm` extra, e.g. `uv tool install ".[gpu,vlm]"` or
  `uv sync --extra gpu --extra vlm --extra dev`) — without that extra it silently
  falls back to the fast CLIP tier, and the UI hint under the checkbox says so.
- **"Online geo (more accurate abroad)"** — use online Nominatim reverse‑geocoding
  instead of the bundled offline GeoNames data for this run; sends only GPS
  coordinates, never photos (see §15).

People/Events staying hidden on a fresh Process run is expected — it means faces/
events weren't enabled for that run, not that something broke; re‑run with the
checkbox ticked (or `sorta faces`/`sorta events`) and the tab appears.

The server binds to `127.0.0.1` only (not reachable from the network). Stop it with
`Ctrl+C`.

---

## 7. Quick start — CLI

These examples assume `sorta` is already on your PATH — via `uv tool install` or
an activated venv (§3). Don't prefix them with `uv run`; see the warning in §3 for
why.

```bash
# 1) Index a folder (metadata, hashes, exact duplicates)
sorta index /path/to/photos

# 2) Run the base pipeline (geo, landmarks, junk) + near-dup hashes — no faces/events
sorta run
sorta phash

# 2b) ...or opt into faces/events too (the slow stages, see §8):
sorta run --faces --events

# 3) Preview the city sort (dry-run — writes a CSV + HTML plan, moves nothing)
sorta sort --by city --dest /path/to/sorted

# 4) Apply it (copy is non-destructive; drop --copy to MOVE)
sorta sort --by city --dest /path/to/sorted --copy --apply

# Undo the last batch if needed
sorta undo
```

### Worked example, start to finish

Everything below is **real command output**, captured against a small synthetic test
collection (13 generated JPEGs with embedded EXIF/GPS: a 2‑day "Paris" trip, a
"Tokyo" day that's too small to become an event, an exact duplicate, a near‑duplicate,
a screenshot, and two placeholder "face" images used only to exercise the pipeline —
not real photographs of anyone). It's here so you know exactly what to expect; the
full walkthrough of every mode continues in §9–§13.

```
$ sorta index -c config.yaml
Готово: +13 новых, ~0 обновлено, 0 пропущено, 0 ошибок, 1 дубликатов помечено

$ sorta geo -c config.yaml
Готово: 12 файлов — exact_gps 10, session_inferred 1, unknown 1

$ sorta faces -c config.yaml
Детекция: 12 файлов, 0 лиц, 12 без лиц, 0 ошибок
Кластеры: 0 (лиц в кластерах: 0, шум: 0, имён сохранено: 0)

$ sorta events -c config.yaml
События: 1 авто (7 файлов, имён сохранено: 0), 0 ручных (0 файлов)

$ sorta junk -c config.yaml
Классификация: 12/12 обработано (photo: 11, screenshot: 1)

$ sorta phash -c config.yaml
pHash посчитан для 13 фото. Отчёт: sorta dupes --near

$ sorta stats -c config.yaml
Файлов в индексе: 13 (+0 с ошибками)
  с GPS:            11 (84%)
  дата из exif     : 13 (100%)
  дата из filename : 0 (0%)
  дата из mtime    : 0 (0%)
  дубликатов:       1
Гео (places): 12
  exact_gps       : 10 (83%)
  unknown         : 1 (8%)
  session_inferred: 1 (8%)
```

A few things worth noticing here (real, not edited for effect):

- **CLI messages print in Russian regardless of `language`** — see the note in §4.
  The numbers are what matter; a rough gloss: *"Готово: +13 новых"* = "Done: +13 new",
  *"дубликатов"* = duplicates, *"с GPS"* = with GPS, *"Детекция"* = Detection,
  *"Кластеры"* = Clusters, *"События"* = Events, *"Классификация"* = Classification.
- `index` found **13** files but `stats` later also says 13 — the exact‑duplicate
  file *is* indexed (with `dup_of` set), it just doesn't get its own place/event/junk
  row, which is why `geo`/`junk` report **12**.
- `faces` genuinely found **0 faces** in the two placeholder images — a real
  photographic‑face detector doesn't fire on flat vector art, which is exactly why we
  didn't fabricate a "found 2 faces, named Alice" example here. See §11 for how the
  person workflow looks once you point it at real photos.
- `events` built **1** event from **7** files (the Paris trip); the 4 Tokyo files
  stayed below `events.min_event_size` (5) and fall back to a `no_event` bucket in
  event‑mode sorting (§9) — a real, useful demonstration of that threshold, not a bug.
- `landmarks` isn't shown above because on this data it found nothing to do (no
  GPS‑less file sits near enough to a real landmark in the bundled catalogue) — see
  §9 for how it fits in when it does.

---

## 8. The processing pipeline

`sorta run` (or the UI **Process** button) executes these stages in order. Each is
also a standalone command and is **incremental** (re‑running only processes
new/changed files):

| Stage | Command | Runs by default? | What it does |
|---|---|---|---|
| Index | `sorta index [dir]` | always | Scan files, read EXIF/dates, compute blake3 hashes, mark exact duplicates. |
| Geo | `sorta geo` | always | Resolve each file's place from GPS; infer place for GPS‑less files from time‑adjacent neighbours (offline GeoNames, or online Nominatim if enabled). |
| Landmarks | `sorta landmarks` | always | Visual place guess for GPS‑less scenes, conservative threshold — fills in city for e.g. an indoor landmark photo with no GPS. |
| Faces | `sorta faces` | **opt‑in** (`--faces`) | Detect faces (insightface), compute embeddings, cluster people (HDBSCAN). The slowest stage; skipped unless you ask for it. |
| Events | `sorta events` | **opt‑in** (`--events`) | Group photos into events by time gaps + city; name them by date + city. Independent of faces — enable either, both, or neither. |
| Junk | `sorta junk` | always | Classify each photo: `photo` / `screenshot` / `meme` / `document` (heuristics + CLIP + text‑density). |
| Near‑dup hashes | `sorta phash` | always (UI); separate command in the CLI (`sorta run` doesn't call it — run `sorta phash` yourself) | Compute perceptual hashes for near‑duplicate detection. |

**`sorta run` flags** (all optional, all overrides for *this run only* — nothing is
written to `config.yaml`):

```
--faces / --no-faces       Run face detection + clustering this run (default: off)
--events / --no-events     Build events this run (default: off)
--deep / --no-deep         Use the deep VLM classification tier for junk this run
                            (needs `uv sync --extra vlm`; gracefully falls back to
                            the fast CLIP tier without it). Default: from config.yaml
                            (naming.vlm_enabled).
--geo offline|online       Reverse-geocoding provider for this run. `online` is more
                            accurate abroad but sends GPS coordinates (never images)
                            to Nominatim. Default: from config.yaml (geo.provider).
--by city|person|event     Also print a dry-run sort plan at the end (see §9)
--dest DIR                 Destination for that plan (omit for in-place)
```

The **base run** (`sorta run`, no flags) is deliberately the fast path: city sorting
and duplicate detection, nothing else. Enable `--faces`/`--events` when you actually
want people/event sorting or albums — running `sorta faces`/`sorta events` on their
own afterwards works exactly the same and is fully incremental either way. Check
coverage anytime with `sorta stats`.

---

## 9. Sorting: cities, people, events

```bash
sorta sort --by city   --dest <dir> [--apply] [--copy|--move] [--where …] [--dedupe]
sorta sort --by person --dest <dir> [--apply] …
sorta sort --by event  --dest <dir> [--apply] …
```

- **`--by city`** → `Country/City/Year/District/…` (localized names).
- **`--by person`** → a folder per **named** person (name clusters first — see §11).
- **`--by event`** → `Year/EventName/…`.
- **`--dest`** — target root. If omitted, sorting is **in‑place** (restructures the
  source folder itself — dry‑run, journal and undo still apply).
- **`--copy` / `--move`** — copy (originals kept) or move (default).
- **`--where`** — filter the plan, repeatable: `--where "country=DE" --where "year>=2020"`.
- **`--dedupe`** — route lower‑quality near‑duplicates to a `_Duplicates` folder.
- **`--exclude <path>`** — skip an already‑sorted subfolder.

Files that don't fit a mode land in review folders: `_Unsorted/` (no place / no
date / junk), `_Documents/` (see §14).

Without `--apply` you get a **dry‑run**: a CSV + a browsable HTML plan in `report_output/` next to
the database, and **nothing is moved**.

### Worked example — `--by city`

Continuing the synthetic collection from §7 (index/geo/junk already ran):

```
$ sorta sort --by city --dest sorted -c config.yaml
sort --by city (dry-run): 12 файлов -> 4 каталогов; план: …\report_output\sort_plan_city_20260721_113247.csv, …\report_output\sort_plan_city_20260721_113247.html
```

The CSV plan (one row per file — `target` is relative to `--dest`) — trimmed to the
columns that matter here:

| path | country | city | target | reason |
|---|---|---|---|---|
| `Screenshots/shot_01.jpg` | | | `_Unsorted/junk/screenshot/shot_01.jpg` | junk |
| `paris_01.jpg` | FR | Paris | `France/Paris/2023/paris_01.jpg` | city |
| `paris_02.jpg` | FR | Paris | `France/Paris/2023/paris_02.jpg` | city |
| `paris_02_edited.jpg` | FR | Paris | `France/Paris/2023/paris_02_edited.jpg` | city |
| `paris_03.jpg` | FR | Paris | `France/Paris/2023/paris_03.jpg` | city |
| `paris_04.jpg` | FR | Paris | `France/Paris/2023/paris_04.jpg` | city |
| `paris_05_nogps.jpg` (no GPS) | FR | Paris | `France/Paris/2023/paris_05_nogps.jpg` | city — place **inherited** from a time‑adjacent Paris photo |
| `tokyo_01.jpg` | JP | Tokyo | `Japan/Tokyo/2023/tokyo_01.jpg` | city |
| `tokyo_02.jpg` | JP | Tokyo | `Japan/Tokyo/2023/tokyo_02.jpg` | city |
| `tokyo_03.jpg` | JP | Katsushika‑ku | `Japan/Katsushika-ku/2023/tokyo_03.jpg` | city — a different GPS point resolved to a different Tokyo ward, which is correct: cities aren't merged just because they're both "Tokyo‑ish" |

Applying it (`--copy` so the originals stay put — drop it to move instead) and the
resulting tree:

```
$ sorta sort --by city --dest sorted_apply --copy --apply -c config.yaml
sort --by city --apply: 12 файлов -> 4 каталогов; план: …
Скопировано 12, на месте 0, ошибок 0. Откат: sorta undo

$ find sorted_apply -type f
sorted_apply/France/Paris/2023/paris_01.jpg
sorted_apply/France/Paris/2023/paris_02.jpg
sorted_apply/France/Paris/2023/paris_02_edited.jpg
sorted_apply/France/Paris/2023/paris_03.jpg
sorted_apply/France/Paris/2023/paris_04.jpg
sorted_apply/France/Paris/2023/paris_05_nogps.jpg
sorted_apply/France/Paris/2023/person_a_1.jpg
sorted_apply/Japan/Katsushika-ku/2023/tokyo_03.jpg
sorted_apply/Japan/Tokyo/2023/person_a_2.jpg
sorted_apply/Japan/Tokyo/2023/tokyo_01.jpg
sorted_apply/Japan/Tokyo/2023/tokyo_02.jpg
sorted_apply/_Unsorted/junk/screenshot/shot_01.jpg

$ sorta undo -c config.yaml
Откат батча 2: возвращено 12, отсутствовало 0, ошибок 0

$ find sorted_apply -type f
(nothing — undo removed every copy)
```

`--dest` folders are localized too — the exact same `sort --by city` on `language: ru`
produces `Франция/Париж/2023/…` and `Япония/Токио/2023/…`; `language: ja` produces
`フランス/パリ/2023/…` and, interestingly, `日本/東京都/2023/桜丘町/…` — a **district**
subfolder (Sakuragaoka‑chō) that doesn't appear for `en`/`ru` on the same file. That's
not a bug: the bundled GeoNames data has Japanese‑localized district names that don't
exist for `en`/`ru`, and `naming.drop_unlocalized_district` (default on) hides a
district segment for a language it can't localize rather than showing a raw
transliterated code.

**Filtering with `--where`:**

```
$ sorta sort --by city --dest sorted_fr --where "country=FR" -c config.yaml
sort --by city (dry-run): 7 файлов -> 1 каталогов; план: …
```

Only the 7 French‑resolved files are planned; everything else is left out of the plan
entirely (not routed to `_Unsorted`).

### Worked example — `--by event`

```
$ sorta sort --by event --dest sorted_event -c config.yaml
sort --by event (dry-run): 12 файлов -> 3 каталогов; план: …
```

| path | event | target |
|---|---|---|
| `paris_01.jpg` … `person_a_1.jpg` (7 files) | `2023-06-10..06-11 Paris` | `2023/2023-06-10..06-11 Paris/<name>.jpg` |
| `tokyo_01.jpg`, `tokyo_02.jpg`, `tokyo_03.jpg`, `person_a_2.jpg` (4 files) | *(none — below `events.min_event_size`)* | `2023/11/<name>.jpg` — the `no_event` fallback, grouped by year/month instead |
| `shot_01.jpg` | | `_Unsorted/junk/screenshot/shot_01.jpg` — junk always wins regardless of mode |

This is the same `min_event_size` threshold from §7/§12 in action: the Tokyo day had
real GPS, real timestamps, a real place — everything except *enough files* to clear
the bar for becoming a named event on its own.

### Worked example — `--by person`

Person‑mode needs **named face clusters** first (§11), which in turn needs `sorta
faces` to actually find faces in real photographs — something our synthetic
placeholder images can't demonstrate honestly (see the note in §7 and the caveat in
§11). Once you've named a couple of clusters on a real collection, the shape of it is:

```bash
sorta sort --by person --dest /path/to/sorted --apply
```

producing `<dest>/<PersonName>/<file>.jpg` for every photo where that person is the
(or the primary, see `sort.multi_person` in §17) named face — everything else that
lacks a named person still needs a place to go, so unnamed‑person photos fall back to
`_Unsorted/`. Junk/screenshot routing and `--where`/`--copy`/`--move`/`--apply` all
work exactly as in the city/event examples above.

---

## 10. Duplicates

- **Exact duplicates** (identical bytes) are detected during `index`; only the
  canonical copy is sorted, the rest stay in place.
- **Near‑duplicates** (visually similar, different size/name) are found via
  perceptual hashing (`sorta phash`, then `sorta dupes --near` or the UI
  **Duplicates** tab).

In the UI **Duplicates** tab: each group shows a recommended keeper (★). Adjust the
radio where you disagree, tick *"don't delete this group"* to skip a group, then
click **Save all choices** once (no per‑group clicking). On the next sort/copy, the
non‑keepers are routed to a `_delete` folder (recoverable) — or use the per‑photo
**Delete** button / **Delete duplicates** to send them to the OS recycle bin
immediately.

Real output on the synthetic collection from §7 (after `sorta phash`):

```
$ sorta dupes -c config.yaml
paris_01_copy.jpg
  -> дубликат paris_01.jpg

Всего: 1

$ sorta dupes --near -c config.yaml
Группа из 2 похожих:
  paris_02.jpg  (7424 байт)
  paris_02_edited.jpg  (5908 байт)
Группа из 2 похожих:
  person_a_1.jpg  (14742 байт)
  person_a_2.jpg  (14742 байт)

Групп: 2 (порог Хэмминга: 5)
```

`paris_02_edited.jpg` is a genuinely recompressed/resized copy of `paris_02.jpg` —
exactly the "same photo, edited or re‑exported" case perceptual hashing is for. The
second group is a false‑but‑instructive positive: our two placeholder face images are
pixel‑for‑pixel identical (we generated them from the same procedure), so pHash
correctly calls them near‑duplicates even though `sorta faces` treats them as unrelated
files (no faces detected in either). On a real collection two different photos of the
same person are usually *not* near‑duplicates — pHash compares whole‑image similarity,
not identity.

---

## 11. People & face clusters

Face detection produces **clusters** (groups of the same face). Before person
sorting is meaningful, name the clusters:

- **UI → People tab:** each cluster shows sample faces; type a name and **Name** it;
  select two clusters and **Merge** if they're the same person.
- **CLI:** `sorta faces label <cluster_id> "Mom"`, `sorta faces merge <src> <dst>`,
  `sorta faces sheet <cluster_id> out.html` (contact sheet to identify a cluster).

Once named, `sorta sort --by person` (or a person **album**, §13) uses the names.

`sorta faces` needs `--faces` on `sorta run` / "Detect faces" ticked in the UI (§8) —
it doesn't run on a base pipeline. Real output from §7's synthetic collection:

```
$ sorta faces -c config.yaml
Детекция: 12 файлов, 0 лиц, 12 без лиц, 0 ошибок
Кластеры: 0 (лиц в кластерах: 0, шум: 0, имён сохранено: 0)
```

Genuinely zero — buffalo_l is trained on real photographs and correctly does not fire
on our synthetic placeholder images (flat vector shapes, not real facial texture).
That's expected, not a bug in Sorta or in this guide: point `sorta faces` at an
actual photo collection and it detects real faces. Once it has (a real run, not this
synthetic one, would print something like `Детекция: 340 файлов, 512 лиц, 8 без лиц,
0 ошибок` / `Кластеры: 6 (лиц в кластерах: 480, шум: 32, имён сохранено: 0)`), naming
and sorting are exactly the commands above — `sorta faces label 3 "Mom"` names cluster
`3`, then `sorta sort --by person --dest … --apply` files that person's photos under
`<dest>/Mom/`.

---

## 12. Events

Events group photos by time gaps and city. `sorta events` (re)builds them:

- Small clusters (below `events.min_event_size`) are not turned into events.
- Same‑city sessions within `events.trip_merge_gap_hours` merge into one trip.
- Name = date range + localized city (e.g. `2023-11-29..12-02 Sochi`).

Manual control:
- `sorta events add "Conference" 2025-05-21 2025-05-23` — a manual event over a date
  range (survives recompute).
- `sorta events rename <event_id> "IEEE conference Tokyo"` — a manual name.

`sorta events` needs `--events` on `sorta run` / "Detect events" ticked in the UI
(§8). Real output from §7's synthetic collection (7 Paris files clear the default
`min_event_size` of 5; the 4‑file Tokyo day doesn't):

```
$ sorta events -c config.yaml
События: 1 авто (7 файлов, имён сохранено: 0), 0 ручных (0 файлов)
```

---

## 13. Albums

An **album** collects a specific slice — one person (optionally filtered) or one
event — into its own named folder, without disturbing the canonical city structure.

```bash
# All photos of "Mom", as hardlinks (default), preview then apply:
sorta album person "Mom" --dest /path/to/albums
sorta album person "Mom" --dest /path/to/albums --apply

# "Mom" but only in Barcelona:
sorta album person "Mom" --where "city=Barcelona" --dest /path/to/albums --apply

# A specific event with a custom folder name, as copies:
sorta album event "2025-05-21..05-23 Tokyo" --dest /path/to/albums \
      --name "IEEE conference Tokyo" --copy --apply
```

- Default mode is **link** (hardlink, ~0 extra space; a photo can appear in several
  albums *and* in the city structure).
- **`--copy`** makes independent copies; **`--move`** *removes the files from the
  general pool* (prints a warning). A photo with **2+ named people** cannot be moved
  into one album (ambiguous) — those are blocked; use link/copy.
- In the UI, use **Collect into folder** on People/Events cards.

Real output — collecting the Paris event from §12 into an album, as copies:

```
$ sorta album event "2023-06-10..06-11 Paris" --dest albums --copy --apply -c config.yaml
album event '2023-06-10..06-11 Paris' --apply [copy]: 7 файлов -> …\albums\2023-06-10..06-11 Paris
Альбом «2023-06-10..06-11 Paris»: выгружено 7, ошибок 0. Откат: sorta undo

$ find albums -type f
albums/2023-06-10..06-11 Paris/paris_01.jpg
albums/2023-06-10..06-11 Paris/paris_02.jpg
albums/2023-06-10..06-11 Paris/paris_02_edited.jpg
albums/2023-06-10..06-11 Paris/paris_03.jpg
albums/2023-06-10..06-11 Paris/paris_04.jpg
albums/2023-06-10..06-11 Paris/paris_05_nogps.jpg
albums/2023-06-10..06-11 Paris/person_a_1.jpg
```

Because we passed `--copy`, these are independent files — `sorta undo` here removes
only the album copies, never touching the originals (see §15).

---

## 14. Junk, screenshots & documents

`sorta junk` classifies each photo so sorting can route non‑memories out of your
city/person/event folders:

- **`screenshot`**, **`meme`** → `_Unsorted/junk/…`. Files in a `Screenshots/`
  folder are detected by folder name too.
- **`document`** (passports, receipts, forms, medical papers…) → `_Documents/` — a
  **review folder**, *not* junk. Detection combines CLIP with a **text‑density**
  signal (documents are text‑dense; beaches and product shots are not).

`_Documents/` deliberately **over‑collects** (a real photo landing there is easy to
pull out; a real document leaking into your city memories is worse). Review it
manually. Note: automatic separation of "for‑sale item" photos from documents is a
known limitation — those may co‑mingle in `_Documents/`.

> **Privacy:** documents may contain personal data. Sorta processes them **locally**
> and never uploads them (unless you enable an online provider). See §15.

Real output on §7's synthetic collection — the one photo saved under `Screenshots/`
is picked up by the folder‑name heuristic, the rest classify as ordinary photos:

```
$ sorta junk -c config.yaml
Классификация: 12/12 обработано (photo: 11, screenshot: 1)
```

---

## 15. Safety, undo & privacy

- **Dry‑run by default** — nothing moves until `--apply`.
- **Move journal** — every operation is recorded *before* it happens.
- **Undo** — `sorta undo` reverses the last batch (`--batch <id>` for a specific
  one). For copy/link batches, undo deletes the copies/links, never the originals.
- **Hash‑verified, never overwrites** — blake3 checked before moving; name conflicts
  get `_1`, `_2`.
- **Originals untouched with copy/link.** With move, files relocate but content and
  EXIF are unchanged.
- **Local by default.** Face/scene/text models run on your machine. Online providers
  are **opt‑in** in `config.yaml`: `geo.provider: online` (Nominatim) sends only GPS
  coordinates, never images; `naming.provider: claude` sends a handful of sample
  photos per event to the Claude API (the one feature that does leave your machine
  with real photo content) — see [SECURITY.md](../../SECURITY.md) for exactly what
  each provider sends. Keep them off for maximum privacy.
- The web UI binds to `127.0.0.1` only.

---

## 16. Full command reference

```
sorta index [DIR]                 Scan sources (or DIR) → metadata, hashes, exact dupes
sorta run [--src DIR] [--faces] [--events] [--deep/--no-deep] [--geo offline|online]
          [--by city|person|event] [--dest DIR]
                                  Base pipeline (index→geo→landmarks→junk); --src
                                  overrides config sources for this run; --faces/
                                  --events opt into the slow stages (default: off,
                                  independent of each other); --deep/--geo override
                                  config.yaml for this run only; with --by, also
                                  prints a dry-run plan at the end
sorta geo                         Resolve places (GPS + session inference)
sorta landmarks                   Visual place guess for GPS-less scenes (conservative)
sorta faces                       Detect faces + cluster people
sorta faces label <cluster> <name>    Name a cluster
sorta faces merge <src> <dst>          Merge two clusters (same person)
sorta faces sheet <cluster> <out.html> Contact sheet to identify a cluster
sorta events                      (Re)build events
sorta events add <name> <from> <to>    Manual event over a date range
sorta events rename <id> <name>        Manual event name
sorta junk                        Classify photo/screenshot/meme/document
sorta phash                       Perceptual hashes (for near-duplicates)
sorta stats                       Index coverage (GPS, date sources, duplicates)
sorta dupes [--near]              List exact / near duplicates
sorta sort --by MODE [--dest DIR] [--apply] [--copy|--move]
           [--where …] [--dedupe] [--delete-worse-dupes] [--exclude PATH] [--thumbnails]
                                  Plan/apply a sort (dry-run without --apply)
sorta album person|event <selector> --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
                                  Collect a slice into a named folder (hardlink by default)
sorta undo [--batch ID]           Reverse the last (or a specific) batch
sorta reset [--yes]               Wipe the index (DB) and start over — leaves your
                                  photos and any already-sorted folders untouched
                                  (names of people/events and dup decisions are lost)
sorta ui [--port 8756]            Local web app (Process / Cities / Duplicates / People / Events / Moves)
```

Every command takes `-c/--config <path>` (default `config.yaml`).

---

## 17. Configuration reference

Key sections of `config.yaml` (see `config.example.yaml` for the full template):

```yaml
sources: ["D:/Photos"]         # folders to scan (recursive)
database: "sorta.db"           # SQLite index path
language: ru                   # ru | en | ja — folder & UI language

index:
  min_file_size_kb: 5          # ignore tiny files
  workers: 8                   # parallel hashing
  skip_dirs: [".thumbnails", "@eaDir", "$RECYCLE.BIN", "System Volume Information"]

geo:
  provider: offline            # offline (bundled GeoNames) | online (Nominatim/OSM)
  session_gap_hours: 6         # gap that splits GPS-inference sessions
  nominatim_url: "https://nominatim.openstreetmap.org"   # only if provider: online
  nominatim_user_agent: "sorta-photo-organizer"          # required by OSM policy

events:
  gap_hours: 6                 # gap that starts a new session
  trip_merge_gap_hours: 48     # same-city sessions within this merge into a trip
  min_event_size: 5            # smaller groups don't become events

sort:
  multi_person: primary        # multi-person photo → largest face's person
  exclude_dirs: []             # subfolders to skip during sort
  album_dir: null              # root for albums (default: _Albums next to the DB)
  report_dir: null             # where sort plans (CSV/HTML) are written (default: report_output/ next to the DB)

faces:
  min_face_px: 40              # ignore faces smaller than this
  det_threshold: 0.7           # detector confidence
  min_cluster_size: 5          # min faces per cluster (HDBSCAN)
  max_distance: 0.5            # cosine similarity threshold

naming:
  landmark_threshold: 0.85     # CLIP threshold for visual place (conservative)
  junk_threshold: 0.85         # CLIP threshold for screenshot/meme
  document_threshold: 0.9      # CLIP threshold for documents
  text_frac_document: 0.15     # text-area fraction above which a photo → document
  text_rescue_docscore_min: 0.3  # only run OCR on photos with this doc-score+
  vlm_enabled: false           # deep VLM classification tier (needs `--extra vlm`);
                               #   same as `--deep` / the UI "Deep analysis" checkbox
```

---

## 18. Troubleshooting

- **`uv sync` (no extras) leaves `sorta faces`/`sorta junk` broken or inconsistent**
  — expected. Always install with an explicit profile: `uv sync --extra cpu --extra
  dev` or `uv sync --extra gpu --extra dev` (§2/§3). `cpu`/`gpu` are mutually
  exclusive; switching hardware later just means re‑running `uv sync` with the other
  one.
- **`No module named ruff` / dev tools missing** — add `--extra dev` to your `uv sync`
  (it's separate from the cpu/gpu profile, see above).
- **HEIC/RAW dates, previews, or video metadata missing** — install `exiftool` (see
  §3); it's required for those formats, Pillow only covers JPEG/PNG/TIFF/WEBP.
- **Faces/CLIP very slow on the GPU profile** — confirm `uv sync --extra gpu` actually
  ran (not `cpu`) and that your driver supports CUDA 13; `sorta faces`/`sorta junk`
  print which onnxruntime execution provider they picked (`CUDAExecutionProvider` vs
  `CPUExecutionProvider`) near the start of their output.
- **Classification/faces slow even though you installed `--extra gpu`** — you're
  probably invoking `sorta` through bare `uv run sorta …`. `uv run` re‑syncs the
  environment to `pyproject.toml`'s base dependencies before each run, which drops
  the GPU torch build back to CPU unless you repeat `--extra gpu` on that exact
  command every time (see §3). Run the tool‑installed binary (`uv tool install
  ".[gpu]"`, then plain `sorta …`) or an activated venv instead — neither resyncs
  on every invocation. Verify the GPU is actually in use:
  `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
  To change hardware profile, reinstall with the other extra — that's the same
  "change profile = reinstall with a different extra" step described in §3.
- **Deliberately forcing CPU on a GPU‑profile install** (e.g. to debug, or the GPU is
  busy with something else) — set `CUDA_VISIBLE_DEVICES=` (empty) for the command;
  both torch and onnxruntime respect it and fall back to CPU:
  ```bash
  CUDA_VISIBLE_DEVICES= sorta faces          # bash/macOS/Linux
  ```
  ```powershell
  $env:CUDA_VISIBLE_DEVICES=''; sorta faces  # PowerShell
  ```
- **`buffalo_l` re‑downloading every run** — the model cache
  (`~/.insightface/models/buffalo_l`) got deleted or isn't writable; make sure that
  path (or a symlink/junction to wherever you keep the model) persists across runs.
- **`database is locked`** — another Sorta process is writing (e.g. a pipeline run).
  Wait for it to finish; don't run two writers at once.
- **A folder with non‑ASCII name (e.g. Cyrillic) seemed skipped by OCR** — fixed:
  images are decoded via an Unicode‑safe path; update to the latest version.
- **CLI console messages print in Russian even with `language: en`/`ja`** — expected,
  not a bug: `language` controls folder names and the web UI, not the CLI's own
  progress text (see §4 and the worked example in §7). If it also renders as `????`
  in your terminal, that's a separate, purely cosmetic encoding issue (next bullet).
- **Cyrillic/Japanese text garbled in a Windows console** — cosmetic; the files and
  the web UI are unaffected. Use the web UI, a UTF‑8 terminal, or set
  `PYTHONUTF8=1` before running `sorta`.
- **`sorta landmarks` (or another command) fails with a relative‑path error like
  `data/landmarks.yaml` not found** — that path (`naming.landmarks_file` in
  `config.yaml`) is resolved relative to your **current directory**, not the repo.
  Either run `sorta` from the repo root, or set an absolute path for
  `naming.landmarks_file` in your `config.yaml`.

---

*Sorta keeps your originals safe and works locally. Review every plan before
`--apply`, and use `sorta undo` if anything looks wrong.*
