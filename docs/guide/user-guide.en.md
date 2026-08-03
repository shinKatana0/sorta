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
16. [Maintenance & diagnostics](#17-maintenance--diagnostics)
17. [Preview cache](#18-preview-cache)
18. [The run log](#19-the-run-log)
19. [Offline models](#20-offline-models)
20. [Configuration reference](#21-configuration-reference)
21. [The Review workspace](#22-the-review-workspace)
22. [Search by words](#23-search-by-words)
23. [Animals and frame quality](#24-animals-and-frame-quality)
24. [Troubleshooting](#25-troubleshooting)

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

| | CPU profile (the `cpu` extra) | GPU profile (the `gpu` extra) |
|---|---|---|
| Hardware | Any x86‑64 machine, no GPU needed | NVIDIA GPU + driver supporting **CUDA 13** (verified on Blackwell/RTX 5090) |
| Backend | `onnxruntime` (CPU) + CPU‑build torch/torchvision | `onnxruntime-gpu` + CUDA 13/cuDNN 9 runtime (pip wheels) + CUDA‑build torch/torchvision |
| Faces / CLIP speed | Works, correctly, just **slow** — expect hours of `faces`/`junk`/`landmarks` on a large collection. Fine for city‑sorting + duplicates (faces/events are opt‑in anyway, see §8), usable for smaller collections with faces/events on. | Fast. Measured on 2026‑07‑28 on a 24,196‑photo collection, faces+events+junk enabled: **≈ 40 min** for a full run without the deep tier. The optional deep VLM tier (`vlm.enabled` + the `vlm` extra) adds **+122 min**, but only once — per‑stage breakdown in §8. |
| RAM | 8 GB+ recommended (indexing/hashing is the RAM‑heavy part, independent of profile) | Same, plus whatever the GPU driver reserves |
| VRAM | n/a | **~3 GB** for base + faces (measured on RTX 5090: CLIP ViT‑L ≈2.0 GB + buffalo_l ≈0.6 GB) — a **≥4 GB** GPU is comfortable. The optional deep VLM tier (Qwen2.5‑VL‑3B) adds ≈7 GB (estimated from the 3B fp16 model, not measured) → **≥8 GB** total |

The timings and VRAM figures are observations from our hardware, not a guarantee —
your mileage will vary with collection composition (video previews, RAW files, and
faces per photo are the main cost drivers).

---

## 3. Installation

### 3.1 Before you install

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

Without `exiftool` Sorta falls back to Pillow, which reads JPEG/PNG/TIFF/WEBP only:
no video at all, and no dates, GPS or orientation from HEIC/RAW. On a phone
collection that is most of the metadata Sorta sorts by, so treat it as required, not
optional.

### 3.2 Pick a hardware profile

Sorta's ML backend (faces, CLIP/OCR) needs exactly one hardware profile. `cpu` and
`gpu` are **mutually exclusive** extras (`tool.uv.conflicts` in `pyproject.toml`) —
install one, never both. What each combination actually resolves to (checked with
`uv pip compile` on 2026‑07‑26):

| Extra | What gets installed |
|---|---|
| `cpu` | `torch==2.13.0+cpu`, `onnxruntime` |
| `gpu` | `torch==2.13.0+cu130`, `onnxruntime-gpu` |
| `gpu,vlm` | the `gpu` set + `transformers==4.51.3` |
| `cpu,vlm` | the `cpu` set + `transformers==4.51.3` |

`vlm` is the optional deep VLM classification tier (`vlm.enabled` / `--deep`,
see §8); without it that tier silently falls back to the fast CLIP tier. `dev` adds
the quality-gate tools (ruff, mypy, pytest) — needed only to run `scripts/check.py`
or the test suite.

### 3.3 Install the `sorta` command (`uv tool install`)

> **`uv tool install` has no `--extra` flag.** The extra goes *inside the package
> specification*, in the same quoted argument as the path:
> `uv tool install "C:\path\to\sorta[gpu]"`. This is exactly where a real install
> went wrong: the command was run without the extra, so it resolved the plain
> package, quietly built the **CPU** profile, and torch arrived as `+cpu` on a
> machine with a perfectly good GPU.

```bash
# CPU only
uv tool install "C:\path\to\sorta[cpu]"

# GPU (NVIDIA / CUDA 13)
uv tool install "C:\path\to\sorta[gpu]"

# GPU + the deep VLM tier
uv tool install "C:\path\to\sorta[gpu,vlm]"

# CPU + the deep VLM tier
uv tool install "C:\path\to\sorta[cpu,vlm]"

# …any of them editable, for local code changes
uv tool install -e "C:\path\to\sorta[gpu]"
```

`C:\path\to\sorta` is your checkout; `.` works when you are already inside it
(`uv tool install ".[gpu]"`). Always quote the whole argument — unquoted `[...]` is
glob syntax in most shells.

- **Use `-e` if you are going to edit the code.** A non‑editable install is a
  snapshot: you change a file, run `sorta` again, and it keeps behaving like the old
  copy, with nothing on screen to explain why. With `-e` your edits are live and no
  reinstall step exists to forget.
- **Switching profile, or updating a non‑editable install after `git pull`** —
  reinstall with `--force` and the extra you want:
  `uv tool install --force "C:\path\to\sorta[gpu]"`.
- Once Sorta is published to PyPI, the same shape becomes
  `uv tool install "sorta[gpu]"` — no local checkout needed.

This resolves `pyproject.toml`'s profile/index setup (the `pytorch-cu130` /
`pytorch-cpu` indexes) exactly like `uv sync` does, and puts a `sorta` command on
your PATH: from here on run `sorta ui`, `sorta index …` and the rest from any
terminal, in any directory — no `uv run`, no activated virtualenv.

### 3.4 A development environment (`uv sync`)

```bash
uv sync --extra gpu --extra dev      # NVIDIA GPU + CUDA 13 driver
# or
uv sync --extra cpu --extra dev      # no NVIDIA GPU

# Activate it once per shell session:
.\.venv\Scripts\Activate.ps1         # Windows PowerShell
source .venv/bin/activate            # Linux/macOS/bash
```

`uv sync` is the one that takes `--extra` as a flag, and it needs the profile spelled
out just as much: pass `--extra cpu` or `--extra gpu` explicitly, because neither is
chosen for you. GPU wheels pull the CUDA 13 runtime as ordinary pip packages — no
system CUDA Toolkit required. With the venv active, `sorta …` runs straight out of
your checkout and code edits are visible immediately.

> **Don't run `uv run sorta …` as your everyday command.** `uv run <cmd>`
> re‑syncs the environment against `pyproject.toml`'s base dependency set before
> every invocation — unless you repeat `--extra <profile>` on that exact command
> every single time, the resync silently drops your GPU packages (torch falls
> back to a CPU build) each time you run it. A tool install (§3.3) and an activated
> venv both sidestep this entirely, which is the whole point of installing once
> instead of invoking through `uv run`.

### 3.5 Right after installing: `sorta doctor`

`sorta doctor` is the one command to run before anything else. It touches no
database and downloads nothing — it just reports what the install actually became:

```
$ sorta doctor
torch: 2.13.0+cu130 (CUDA available: yes, device: NVIDIA GeForce RTX 5090 Laptop GPU)
onnxruntime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider (CUDA: yes)
mismatch: no
geo data: C:\...\sorta\data\geo\places.tsv (9.2 MB)
Run log: C:\Users\you\AppData\Local\sorta\logs\sorta.log
Preview cache: C:\Users\you\AppData\Local\sorta\previews
```

How to read it, line by line:

- **`torch: … (CUDA available: yes|no)`** — the build you ended up with. On the `gpu`
  profile this must say `+cu130` and `CUDA available: yes`. A `+cpu` build here means
  the extra never made it into the install command (§3.3).
- **`onnxruntime providers: …`** — what face detection can run on. `CUDAExecutionProvider`
  in the list means the GPU is available to it; a list without it means faces run on
  the CPU (§3.6).
- **`mismatch: yes`** — torch is a CPU‑only build while onnxruntime does have CUDA.
  Faces would still use the GPU while CLIP and OCR quietly run on the CPU, which is
  the slow, silent failure this line exists to catch.
- **`geo data: …places.tsv (N MB)`** — the bundled GeoNames database, the thing city
  sorting resolves coordinates against. It must exist and be non‑empty; if the line is
  prefixed with `⚠` and says FILE NOT FOUND / FILE IS EMPTY, every coordinate resolves
  to an empty place and the fix is a reinstall (or `python scripts/build_geodata.py`
  in a checkout).
- **`Run log:`** — where the run log is written (§19).
- **`Preview cache:`** — where decoded previews are cached (§18); a
  ` (DISABLED)` suffix means the cache is switched off.

The capture above was taken with `language: ru`, which is why those last two labels are
Russian; with the default `en` they read "Run log:" and "Preview cache:". The two health
summaries above them (`torch`/`onnxruntime` and `geo data`) come from the diagnostics
module and stay English in every language. `sorta doctor` reads `config.yaml` only to
learn the output language — it works fine without one, just in the default language —
so the preview path it prints is the default one plus any `SORTA_PREVIEW_DIR` in the
environment.

### 3.6 Known trap: `onnxruntime` overwriting `onnxruntime-gpu`

`insightface` (face detection) depends on the plain `onnxruntime` package, while the
`gpu` extra installs `onnxruntime-gpu`. They are two different distributions that
unpack into the *same* `onnxruntime` directory, so whichever is installed last wins —
and if that is the CPU one, face detection quietly moves to the CPU with no error
anywhere.

The symptom is visible in `sorta doctor`: the providers line has **no**
`CUDAExecutionProvider`, e.g.

```
onnxruntime providers: AzureExecutionProvider, CPUExecutionProvider (CUDA: no)
```

The workaround is to reinstall the GPU package on top, without letting pip resolve
dependencies again:

```bash
python -m pip install --force-reinstall --no-deps onnxruntime-gpu
```

`--no-deps` is not optional here — without it pip re‑resolves `insightface` and drags
the CPU `onnxruntime` right back in. Run it inside the environment Sorta is installed
into (your activated project venv), then re‑run `sorta doctor` and confirm that
`CUDAExecutionProvider` is back in the list.

---

## 4. Configuration

Sorta reads `config.yaml` (copy it from `config.example.yaml`). The two settings you
must review:

```yaml
sources:
  - "D:/Photos"          # folder(s) with your photos/videos (scanned recursively)
database: "sorta.db"     # where the SQLite index is stored
language: ru             # UI/folder language: ru | en | ja  (default en)
```

- **`sources`** — one or more root folders to scan. You can also pass the folder on
  the command line (`sorta index /path/to/photos`), which overrides this.
- **`language`** — controls the language of generated **folder names** (e.g.
  `Россия/…` vs `Russia/…`), the **web UI** chrome and the **CLI's console
  messages**. Supported: `ru`, `en`, `ja`. With the key absent, the language is `en`.

> **Note:** the one exception is the `--help` texts. They live in typer decorators and
> are evaluated at import time, before any config is read, so `sorta --help` and
> `sorta sort --help` stay English whatever `language` says. Everything else — folder
> names, the web UI, progress lines and command summaries — is localized. See the
> worked examples in §7 and §9 for what real CLI output looks like, and §25 if it
> renders as `????` in your terminal.

See the [Configuration reference](#21-configuration-reference) for every option.

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

Then in the browser there are five tabs: **Overview · Review · Layout · Slices ·
Moves**. They are named after **what you do to the collection**, not after the pipeline
stages, because there are exactly three things you can do to it and they differ in what
happens to the files:

- **the canon** — one city layout; the files physically move, and there is only one
  such structure;
- **slices** — people, events, animals, products, screenshots, a query in words: as
  many as you like, gathered as hardlinks, so making and dropping one is free;
- **the junk** — a subtraction: the frames that have to be looked at by eye and partly
  thrown away.

1. **Overview** tab → the state of the collection on one screen: how many files are
   indexed, photos and videos, duplicates, read errors, events; where the place came
   from (exact GPS, set by hand, inherited from the session/trip/folder name,
   recognised from the frame) and how many frames have no place at all; what the
   classifier decided (personal photos, products, documents, screenshots, memes — by
   which signal and which tier); whether a layout has run, where to, in which mode and
   how many files made it. The numbers are **clickable**: a click takes you to the tab
   where you can act on that group. **Starting a run lives here too**: the path to the
   photo folder, the stage checkboxes and the **Process** button — the state of the
   collection and the run that changes it are one question asked at two moments in
   time. Enter the path, tick **"Detect faces"** and **"Detect events"** if you want
   them (both **unchecked by default** — the pipeline's slowest stages, opt‑in on
   purpose, see §8) and click **Process**: it runs in the background with per‑stage
   progress (index → geo → landmarks → [faces] → [events] → junk → near‑duplicates).
   You can close the tab; processing continues.
2. **Review** tab → one workspace for everything that has to be looked at by eye and
   partly deleted: **Duplicates**, **Blurred**, **Closed eyes**, **No subject** — four
   slices of a single tab, described in §22.
3. **Layout** tab → the canon: the proposed structure (`Country/City/Year/District`),
   where to lay it out, move or copy, and the button that starts it. Always visible.
4. **Slices** tab → everything built **on top of** the canon: **With people**, **Group
   photos**, **Portraits**, **People** (the named face clusters), **Events**,
   **Animals** and the classifier's buckets — **Products**, **Documents**,
   **Screenshots**, **Memes**. At the top of it sits the **search line** (§23). Any
   slice is gathered into its own folder with **Collect into folder**, hardlinks by
   default, so it can be gathered and dropped as often as you like. In the classifier's
   buckets, tick the frames that landed there by mistake and press **Return to
   photos** — they go back into the city layout; the model's verdict is not rewritten,
   and the return itself is reversible with **Undo the return**.
   **Documents are shown without thumbnails** — file name and date only. That is a
   rule, not an unfinished corner: this bucket holds passports, certificates and
   medical forms, and Sorta neither opens nor renders them even locally; the name and
   the date are enough to decide.
   **The first three slices are different in kind from the rest** and their caption says
   so. **With people** / **Group photos** / **Portraits** are read straight off the face
   detection: a frame is there because a box was found on it, not because a score cleared
   a line, so there is no confidence on the cards and nothing to tune. Their only two
   numbers are geometric — `features.group_photo_faces` (3 faces or more make a group
   photograph) and `features.portrait_face_share` (one face covering ≥ 8% of the frame
   area makes a portrait), both in §21. Until the faces stage has run they say **so** —
   "the faces stage has not run" — instead of showing a zero, because nothing was
   measured and "you have no photographs with people in them" would be a claim about
   your archive. From the terminal these are `sorta album people|group|portrait` (§13).
5. **Moves** tab → after you apply a layout/album, see exactly what went where. This is
   also where a layout is **rolled back** (§15). Always visible.

**Where the familiar tabs went.** Nothing was removed, the grouping changed: Process
moved inside Overview, Cities is now called Layout, Duplicates became the first slice of
Review, and People, Events and "Not personal photos" became panels of Slices. Inside
Slices, People and Events appear only once face clusters or events exist — that is,
after a run with faces/events enabled (or after `sorta faces`/`sorta events`).

**Why Review comes before Layout.** Frames marked for deletion leave for the `_delete`
folder during `sort --apply` — the same moment the canon is built — and the albums of a
slice are hardlinks *out of the canon*. Gather the albums before you have been through
the junk and you get links to what you threw away. It is not forbidden: the Layout tab
shows a warning while the Review still holds undecided frames, and nothing more. The
collection is alive, "gather" happens again and again, and a locked tab would cost more
than the mistake it guards against.

**There is no "Re-run selected" button any more.** The stages skip what is already done
by themselves — a repeat run touches only new and changed files — so starting a run is
one **Process** button rather than a choice of stages. The button went, not the route:
the HTTP endpoint `POST /api/process/rerun-optional` is still there for external callers.

The run block on the **Overview** tab has two more checkboxes beyond faces/events, both
reflecting `config.yaml` and acting as a full override for this run only (checked =
force on, unchecked = force off) — the UI equivalent of the CLI's `--deep`/`--no-deep`
and `--geo online`/`--geo offline` (§8):

- **"Deep analysis (VLM)"** — use the deep VLM tier instead of the fast CLIP tier
  for junk/document classification. It only actually takes effect if it's *both*
  requested (this checkbox, `--deep`, or `vlm.enabled: true` in config)
  *and* installed (the `vlm` extra, e.g. `uv tool install ".[gpu,vlm]"` or
  `uv sync --extra gpu --extra vlm --extra dev`) — without that extra it silently
  falls back to the fast CLIP tier, and the UI hint under the checkbox says so.
- **"Online geo (more accurate abroad)"** — use online Nominatim reverse‑geocoding
  instead of the bundled offline GeoNames data for this run; sends only GPS
  coordinates, never photos (see §15).

People/Events staying hidden in Slices after a run is expected — it means faces/
events weren't enabled for that run, not that something broke; re‑run with the
checkbox ticked (or `sorta faces`/`sorta events`) and the panel appears.

### The Settings panel

The **Settings** button in the header opens the settings panel. It edits
`config.yaml` for real: the value is written to the file straight away and **there is
no need to restart `sorta ui`** — the next run reads the new one. What lives there:

- **Deep analysis (VLM)** — `vlm.enabled` (§21). Unlike the checkbox of the same name
  on the Process tab, which only covers a single run, this toggle writes the value
  into the config for good.
- **Model**, **Preparation threads**, **Frame resolution, px** — `vlm.model`,
  `vlm.workers`, `vlm.max_edge`.
- **Frame quality** — the F113 set, off by default. Grouped by WHAT answers each
  signal, because that is where the whole difference in cost is:
  - **No VLM needed** — computed on a pass that runs anyway: **Look for animals**
    (`features.pets`, threshold `features.pet_threshold`) is answered by **CLIP**, not
    by the model, and works with deep analysis switched off; `features.sharpness_max_edge`
    is the preview size for sharpness, and sharpness itself is the variance of a
    Laplacian — no model at all.
  - **Through the VLM** — **Ask the model about quality** (`vlm.quality`) and **Which
    frames to ask about** (`vlm.quality_scope` — a dropdown, because “Every frame” costs
    about 4 hours over a 20 000-frame collection). Needs `vlm.enabled` and the `vlm`
    extra.
  - **Who reaches the model** — the two ways into the expensive part: the sharpness
    uncertainty band (`features.sharpness_band_min` / `features.sharpness_band_max`) and
    `features.subject_score_min`, the CLIP probability below which CLIP is saying it does
    not know what it is looking at.
- **Preview cache ceiling, GB** — `imaging.preview_cache_max_gb` (§18).
- **Folders → folder‑name language** — `language`. The plan below is recomputed
  immediately, with no restart.

Settings do not change mid‑run: while a run is in progress the column says so and asks
you to wait for it to finish.

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
full walkthrough of every mode continues in §9–§13. The output below is what the
default `language: en` prints; set `language: ru` or `ja` and the same commands say the
same thing in that language (§4).

```
$ sorta index -c config.yaml
Done: +13 new, ~0 updated, 0 skipped, 0 errors, 1 duplicates marked

$ sorta geo -c config.yaml
Done: 12 files — exact_gps 10, session_inferred 1, trip_inferred 0, path_inferred 0, unknown 1

$ sorta faces -c config.yaml
Detection: 12 files, 0 faces, 12 without faces, 0 errors
Clusters: 0 (faces in clusters: 0, noise: 0, names kept: 0)

$ sorta events -c config.yaml
Events: 1 automatic (7 files, names kept: 0), 0 manual (0 files)

$ sorta junk -c config.yaml
Classification: 12/12 processed (photo: 11, screenshot: 1)

$ sorta phash -c config.yaml
pHash computed for 13 photos. Report: sorta dupes --near

$ sorta stats -c config.yaml
Files in the index: 13 (+0 with errors)
  with GPS:           11 (84%)
  date from exif     : 13 (100%)
  date from filename : 0 (0%)
  date from mtime    : 0 (0%)
  duplicates:         1
Geo (places): 12
  exact_gps       : 10 (83%)
  unknown         : 1 (8%)
  session_inferred: 1 (8%)
```

A few things worth noticing here (real, not edited for effect):
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
| Geo | `sorta geo` | always | Resolve each file's place from GPS; infer place for GPS‑less files from time‑adjacent neighbours, then from the whole trip when its GPS frames agree on the city and surround the file in time (offline GeoNames, or online Nominatim if enabled). |
| Landmarks | `sorta landmarks` | always | Visual place guess for GPS‑less scenes, conservative threshold — fills in city for e.g. an indoor landmark photo with no GPS. |
| Faces | `sorta faces` | **opt‑in** (`--faces`) | Detect faces (insightface), compute embeddings, cluster people (HDBSCAN). The slowest stage; skipped unless you ask for it. |
| Events | `sorta events` | **opt‑in** (`--events`) | Group photos into events by time gaps + city; name them by date + city. Independent of faces — enable either, both, or neither. |
| Junk | `sorta junk` | always | Classify each photo: `photo` / `screenshot` / `meme` / `document` / `product` (heuristics + CLIP + text‑density). The `product` class (an item photographed for sale) comes from the deep VLM tier only — the fast tier never produces it, see §14. |
| Near‑dup hashes | `sorta phash` | always (UI); separate command in the CLI (`sorta run` doesn't call it — run `sorta phash` yourself) | Compute perceptual hashes for near‑duplicate detection. |

**`sorta run` flags** (all optional, all overrides for *this run only* — nothing is
written to `config.yaml`):

```
--faces / --no-faces       Run face detection + clustering this run (default: off)
--events / --no-events     Build events this run (default: off)
--deep / --no-deep         Use the deep VLM classification tier for junk this run
                            (needs `uv sync --extra vlm`; gracefully falls back to
                            the fast CLIP tier without it). Default: from config.yaml
                            (vlm.enabled).
--geo offline|online       Reverse-geocoding provider for this run. `online` is more
                            accurate abroad but sends GPS coordinates (never images)
                            to Nominatim. Default: from config.yaml (geo.provider).
--pets / --no-pets         Look for animals this run (features.pets, §24). CLIP answers
                            it on a pass that runs anyway, so it is cheap.
--quality / --no-quality   Ask the model about frame quality this run (vlm.quality):
                            are the eyes open, is there a subject at all. Needs the
                            `vlm` extra.
--quality-scope groups|events|faces|all
                           Which frames reach those questions (vlm.quality_scope). The
                            price is measured: `all` ≈ 4.3 hours on 20 thousand frames,
                            `faces` ≈ 95 minutes on 7,341 (and needs a `faces` run —
                            without one the population is empty and nothing is asked).
--by city|person|event     Also print a dry-run sort plan at the end (see §9)
--dest DIR                 Destination for that plan (omit for in-place)
```

`sorta junk` takes the same three quality flags — it is the same stage run on its own.
Like `--deep`/`--geo`, they apply to the current run only and write nothing into
`config.yaml`.

The **base run** (`sorta run`, no flags) is deliberately the fast path: city sorting
and duplicate detection, nothing else. Enable `--faces`/`--events` when you actually
want people/event sorting or albums — running `sorta faces`/`sorta events` on their
own afterwards works exactly the same and is fully incremental either way. Check
coverage anytime with `sorta stats`.

### What it costs

Measured on 2026‑07‑28 on a live collection of **24,196 photos** (RTX 5090, the `gpu`
profile, faces and events enabled):

| Stage | Time |
|---|---|
| `index` | 6.1 min |
| `landmarks` | 2.8 min |
| `faces` | 16.1 min |
| `junk` (fast tier) | 14.1 min |
| `geo`, `events`, `phash` | seconds |
| **Full run without the deep tier** | **≈ 40 min** |
| The deep VLM tier (7,896 candidates out of 24,196) | **+122 min, once** |

"Once" is literal: junk incrementality runs on `media_class.tier`, so a repeated run
only touches new files and frames already labelled by the deep tier are never sent to
the model a second time. The first `--deep` run is an entry price, not a permanent tax
on every processing run.

These are observations from our hardware, not a guarantee: on yours it depends on what
the collection is made of (video, RAW, the number of faces per frame are the main cost
drivers).

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
date / junk), `_Documents/` and `_Products/` (see §14).

Without `--apply` you get a **dry‑run**: a CSV + a browsable HTML plan in `report_output/` next to
the database, and **nothing is moved**.

### Worked example — `--by city`

Continuing the synthetic collection from §7 (index/geo/junk already ran):

```
$ sorta sort --by city --dest sorted -c config.yaml
sort --by city (dry-run): 12 files -> 4 folders; plan: …\report_output\sort_plan_city_20260721_113247.csv, …\report_output\sort_plan_city_20260721_113247.html
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
sort --by city --apply: 12 files -> 4 folders; plan: …
Copied 12, in place 0, errors 0. Undo: sorta undo

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
Undo of batch 2: 12 restored, 0 missing, 0 errors

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
sort --by city (dry-run): 7 files -> 1 folders; plan: …
```

Only the 7 French‑resolved files are planned; everything else is left out of the plan
entirely (not routed to `_Unsorted`).

### Worked example — `--by event`

```
$ sorta sort --by event --dest sorted_event -c config.yaml
sort --by event (dry-run): 12 files -> 3 folders; plan: …
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
(or the primary, see `sort.multi_person` in §21) named face — everything else that
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
  -> duplicate of paris_01.jpg

Total: 1

$ sorta dupes --near -c config.yaml
A group of 2 similar:
  paris_02.jpg  (7424 bytes)
  paris_02_edited.jpg  (5908 bytes)
A group of 2 similar:
  person_a_1.jpg  (14742 bytes)
  person_a_2.jpg  (14742 bytes)

Groups: 2 (Hamming threshold: 5)
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
Detection: 12 files, 0 faces, 12 without faces, 0 errors
Clusters: 0 (faces in clusters: 0, noise: 0, names kept: 0)
```

Genuinely zero — buffalo_l is trained on real photographs and correctly does not fire
on our synthetic placeholder images (flat vector shapes, not real facial texture).
That's expected, not a bug in Sorta or in this guide: point `sorta faces` at an
actual photo collection and it detects real faces. Once it has (a real run, not this
synthetic one, would print something like `Detection: 340 files, 512 faces, 8 without
faces, 0 errors` / `Clusters: 6 (faces in clusters: 480, noise: 32, names kept: 0)`), naming
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
Events: 1 automatic (7 files, names kept: 0), 0 manual (0 files)
```

---

## 13. Albums

An **album** collects a specific slice — one person, one event, every animal, or the
answer to a query in words — into its own named folder, without disturbing the canonical
city structure.

```bash
# All photos of "Mom", as hardlinks (default), preview then apply:
sorta album person "Mom" --dest /path/to/albums
sorta album person "Mom" --dest /path/to/albums --apply

# "Mom" but only in Barcelona:
sorta album person "Mom" --where "city=Barcelona" --dest /path/to/albums --apply

# A specific event with a custom folder name, as copies:
sorta album event "2025-05-21..05-23 Tokyo" --dest /path/to/albums \
      --name "IEEE conference Tokyo" --copy --apply

# Every frame with an animal — NO selector: the collection has exactly one animal
# slice (§24):
sorta album animal --dest /path/to/albums --apply

# An album from a query in words (§23) — ask in English:
sorta album query "cake" --dest /path/to/albums --apply

# A classifier bucket (§14) — products, screenshots or memes, no selector:
sorta album product --dest /path/to/albums --name "Products" --apply
sorta album screenshot --dest /path/to/albums --apply
sorta album meme --dest /path/to/albums --apply

# A quality slice of the Review workspace (§22) — no selector either:
sorta album blurred --dest /path/to/albums --apply
sorta album eyes_closed --dest /path/to/albums --apply
sorta album no_subject --dest /path/to/albums --apply

# The face slices (§6) — also without a selector: the collection has exactly one of
# each. Every frame a face was found on, the group photographs, the portraits:
sorta album people --dest /path/to/albums --apply
sorta album group --dest /path/to/albums --apply
sorta album portrait --dest /path/to/albums --apply
```

- **Slices without a selector**: `animal`, `product`, `screenshot`, `meme`, `blurred`,
  `eyes_closed`, `no_subject`, `people`, `group`, `portrait`. There is nothing to choose
  inside them — the collection has exactly one products bucket and exactly one blurred
  list — so `sorta album <kind> --dest …` is the whole command, and the folder is named
  after the slice unless `--name` says otherwise.
- **`person` / `event` / `query`** require the selector, because the selector *is* the
  subject of the album (a person's name, an event's name, the words themselves). A
  missing one is an **error**, not "the whole collection": an album quietly gathered
  from everything is indistinguishable from a correct one.
- **`document` is not an album kind**, and no class listed in `vlm.exclude_classes` is:
  that bucket is passports, medical forms and bank papers. It keeps its counter and gets
  neither a preview nor a folder — assembling one in a single click is exactly what the
  key exists to prevent. Move a class *into* `vlm.exclude_classes` and its album goes
  with its preview.
- **`blurred` is a window, not a threshold.** It gathers what the Review workspace lists
  — the frames under `features.blur_review_max` — and never the whole tail below it: the
  point of that slice is that the decision is taken by eye, on what was shown.

- Default mode is **link** (hardlink, ~0 extra space; a photo can appear in several
  albums *and* in the city structure).
- **`--copy`** makes independent copies; **`--move`** *removes the files from the
  general pool* (prints a warning). A photo with **2+ named people** cannot be moved
  into one album (ambiguous) — those are blocked; use link/copy.
- In the UI, use **Collect into folder** — on the People/Events cards, on the
  **Animals** slice, on a classifier bucket (Products, Screenshots, Memes) and on the
  three quality slices of **Review** (Blurred, Closed eyes, No subject). It is the same
  row everywhere: mode, an optional folder name, a destination. The marking buttons
  ("Return to photos", "To trash") stay in their own block — one movement never both
  gathers and deletes.

Real output — collecting the Paris event from §12 into an album, as copies:

```
$ sorta album event "2023-06-10..06-11 Paris" --dest albums --copy --apply -c config.yaml
album event '2023-06-10..06-11 Paris' --apply [copy]: 7 files -> …\albums\2023-06-10..06-11 Paris
Album 2023-06-10..06-11 Paris: 7 exported, 0 errors. Undo: sorta undo

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
- **`product`** (an item photographed for sale: a listing shot, a marketplace frame)
  → `_Products/` — also a **review folder**, neither junk nor a memory. This class
  comes from the **deep VLM tier only** (§8): the fast tier does not produce it at
  all, and there such frames stay `photo` or end up in `_Documents/`. On a live
  collection that is every tenth frame — 2,202 out of 24,196.

`_Documents/` deliberately **over‑collects** (a real photo landing there is easy to
pull out; a real document leaking into your city memories is worse). Review it
manually.

These buckets are easier to review from the web UI: the **Not personal photos** tab
(§6) shows them side by side, returns the frames you tick back into the normal layout
in bulk, and **renders no thumbnails for documents** — deliberately, because that
bucket is where personal papers are.

> **Privacy:** documents may contain personal data. Sorta processes them **locally**
> and never uploads them (unless you enable an online provider). See §15.

Real output on §7's synthetic collection — the one photo saved under `Screenshots/`
is picked up by the folder‑name heuristic, the rest classify as ordinary photos:

```
$ sorta junk -c config.yaml
Classification: 12/12 processed (photo: 11, screenshot: 1)
```

---

## 15. Safety, undo & privacy

- **Dry‑run by default** — nothing moves until `--apply`.
- **A summary before the start (web UI)** — the layout button first opens a dialog
  with the numbers: where to, move or copy, how many files into how many folders and
  at what size, how many of them are already in the destination, and how many go into
  `_Products/` and `_Documents/`. An empty plan says exactly that: there is nothing to
  lay out.
- **A Cancel button while it runs** — a layout can be stopped without waiting for the
  end: the file in flight is finished whole, the rest are left alone, and the result
  says how much did get sorted. What was sorted rolls back from the Moves tab.
- **Move journal** — every operation is recorded *before* it happens.
- **Undo** — `sorta undo` reverses the last batch (`--batch <id>` for a specific
  one). For copy/link batches, undo deletes the copies/links, never the originals. In
  the web UI the **Roll back** button on the **Moves** tab does the same — with a
  confirmation, a progress line and a cancel button of its own; an interrupted
  rollback continues when you press it again.
- **Running it again does not duplicate anything** — a second layout into the same
  destination recognises the files the previous one copied there (by content, not by
  name) and skips them instead of dropping a `_1` copy next to them. The suffix stays
  where it belongs: on a *different* file that happens to share a name.
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
sorta index [DIR] [--exclude-dir NAME]
                                  Scan sources (or DIR) → metadata, hashes, exact dupes;
                                  --exclude-dir keeps a subfolder out of the scan (§21a)
sorta index --refresh-exif        Re-read metadata of already-indexed files (§17)
sorta run [--src DIR] [--faces/--no-faces] [--events/--no-events] [--deep/--no-deep]
          [--geo offline|online] [--pets/--no-pets] [--quality/--no-quality]
          [--quality-scope groups|events|faces|all] [--by city|person|event] [--dest DIR]
                                  Base pipeline (index→geo→landmarks→junk); --src
                                  overrides config sources for this run; --faces/
                                  --events opt into the slow stages (default: off,
                                  independent of each other); --deep/--geo/--pets/
                                  --quality/--quality-scope override config.yaml for
                                  this run only (§8, §24); with --by, also prints a
                                  dry-run plan at the end
sorta geo                         Resolve places (GPS + session inference)
sorta landmarks                   Visual place guess for GPS-less scenes (conservative)
sorta faces [--rescan] [--limit N]     Detect faces + cluster people; --rescan redoes
                                  files already processed, --limit N caps that rescan
                                  at the first N files (only together with --rescan)
sorta faces label <cluster> <name>    Name a cluster
sorta faces merge <src> <dst>          Merge two clusters (same person)
sorta faces sheet <cluster> <out.html> Contact sheet to identify a cluster
sorta events                      (Re)build events
sorta events add <name> <from> <to>    Manual event over a date range
sorta events rename <id> <name>        Manual event name
sorta junk [--pets/--no-pets] [--quality/--no-quality]
           [--quality-scope groups|events|faces|all]
                                  Classify photo/screenshot/meme/document/product; the
                                  frame-quality flags override config for this run (§24)
sorta phash                       Perceptual hashes (for near-duplicates)
sorta stats                       Index coverage (GPS, date sources, duplicates)
sorta dupes [--near]              List exact / near duplicates
sorta search "<words>" [--limit N]
                                  Find frames by words: the CLIP ranking, best first
                                  (§23). --limit N is how many lines to print
sorta sort --by MODE [--dest DIR] [--apply] [--copy|--move]
           [--where …] [--dedupe] [--delete-worse-dupes] [--exclude PATH] [--thumbnails]
                                  Plan/apply a sort (dry-run without --apply)
sorta album person|event|query <selector> --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
sorta album animal --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
sorta album product|screenshot|meme --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
sorta album blurred|eyes_closed|no_subject --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
sorta album people|group|portrait --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
                                  Collect a slice into a named folder (hardlink by
                                  default); only person/event/query take a selector,
                                  every other kind is a single slice (§13, §6)
sorta undo [--batch ID]           Reverse the last (or a specific) batch
sorta reset [--yes|-y] [--clear-geo]
                                  Wipe the index (DB) and start over — leaves your
                                  photos and any already-sorted folders untouched
                                  (names of people/events and dup decisions are lost).
                                  The online-geo cache survives unless --clear-geo
sorta ui [--port 8756]            Local web app (Overview / Review / Layout / Slices / Moves)
sorta doctor                      Environment check: torch/onnxruntime, GPU, geo data,
                                  log path, preview cache path (§3.5)
sorta cache [--clear] [--clear-geo] [--preview-max-gb N]
                                  Caches: preview size or deletion (§18); --clear-geo
                                  drops the cached online-geo answers, --preview-max-gb
                                  sets the cache ceiling for this invocation
                                  (0 = no ceiling)
sorta --install-completion        Install shell completion for `sorta`
sorta --show-completion           Print the completion script without installing it
```

Every command takes `-c/--config <path>` (default `config.yaml`) — except
`sorta doctor`, which reads no config at all.

---

## 17. Maintenance & diagnostics

Three commands that are not part of the pipeline but come up in day‑to‑day use.

### `sorta doctor`

The install/environment check described in §3.5: torch and onnxruntime devices, the
bundled geo database, the run‑log path and the preview‑cache path. Run it after every
install or profile change, and first whenever something is unexpectedly slow.

### `sorta index --refresh-exif`

Re‑reads the metadata of files **that are already in the index**, without reindexing
them. The summary line has this shape (`<…>` are the counters of your run):

```
$ sorta index --refresh-exif -c config.yaml
Re-read: <N> files, <N> updated; coordinates recovered: <N>, capture dates: <N>; without EXIF: <N>, errors: <N>
New coordinates appeared — re-run: sorta geo (and sorta events)
```

(*"Re‑read: N files, N updated; coordinates recovered: N, capture dates: N; no EXIF:
N, errors: N"* — the second line only appears when coordinates actually came back.)

- **Why it exists.** A plain `sorta index` skips those files on purpose: it compares
  path + size + mtime, and none of them changed. If the *reader* improved — a Sorta
  update that fixed metadata extraction — a normal run will never revisit them.
- **When to run it.** After an update whose notes mention EXIF/metadata reading, or
  whenever `sorta stats` shows fewer files with GPS or EXIF dates than you expect.
- **What it touches.** GPS, camera make/model, dimensions, capture date and
  orientation only. Hashes, pHashes and duplicate marks are **left alone** — the file
  content did not change, only what could be read out of it. Files are never read
  whole and nothing is re‑hashed.
- **Afterwards.** New coordinates mean places have to be resolved again: run
  `sorta geo`, and `sorta events` too if events matter to you. The command prints
  that reminder itself when it actually recovered coordinates.

### `sorta cache`

Reports the preview cache (§18), or clears it:

```
$ sorta cache -c config.yaml
Preview cache: C:\Users\you\AppData\Local\sorta\previews
  files: 34887, size: 12.60 GB
  ceiling: none set (imaging.preview_cache_max_gb)

$ sorta cache --clear -c config.yaml
Preview cache removed: C:\Users\you\AppData\Local\sorta\previews
```

`--preview-max-gb N` sets the cache ceiling (§18) for this invocation without touching
`config.yaml`: `sorta cache --preview-max-gb 40` reports the size against 40 GB, and
`--preview-max-gb 0` takes off a ceiling the config had set. The zero has to be spelled
out — it is the value that means "no ceiling".

Deleting the cache is safe at any moment: it is lazy and rebuilds itself, one frame
at a time, in whichever stage next needs that frame. The only cost is that those
frames get decoded from the originals once more.

---

## 18. Preview cache

**What it is.** Each frame is decoded **once**. The result — a downscaled JPEG,
1536 px on the long edge — is written to a shared cache directory, and every later
stage (pHash, CLIP, OCR, the deep VLM tier, UI thumbnails) reads that small copy
instead of decoding the original again.

**Why.** Decoding is what those stages actually spend their time on. A full HEIC
decode costs ≈470 ms; the same frame out of the preview cache costs single‑digit
milliseconds (≈8 ms). The same photo used to be decoded three to five times per run,
because the stages run one after another.

**Where it lives.**

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\sorta\previews` |
| Linux / macOS | `~/.cache/sorta/previews` |

**How much space it takes: budget ≈150 KB per photo, and expect gigabytes.** That is
the design figure for a 1536 px q88 JPEG; detailed frames run larger. A real cache
measured here held 34,887 previews in **12.60 GB** (≈360 KB each), so on a collection
of tens of thousands of photos this is several gigabytes of disk, not a rounding
error. It is a user‑level cache, never written inside your photo collection, and
`sorta cache` tells you the current size at any time (§17).

**Small images are never cached.** If the source is already no larger than
`preview_max_edge`, a preview would be a copy rather than a saving, so it is skipped —
a collection of screenshots costs nothing here.

**Configuration** — the `imaging:` section of `config.yaml`:

```yaml
imaging:
  preview_cache: true     # false → every stage decodes originals again
  # preview_dir: D:/sorta-previews   # default: see the table above
  preview_max_edge: 1536  # long edge of the cached JPEG
  preview_quality: 88     # JPEG quality of the cached copy
  # preview_cache_max_gb: 40   # ceiling in GB; 0 (the default) means none
```

`imaging.preview_max_edge` is the long edge of the cached JPEG, and it covers every
consumer with headroom (OCR asks for 1280, the VLM 896, CLIP 448, pHash 96).
`imaging.preview_quality` is the JPEG quality: at 88 a frame weighs about 150 KB and is
visually indistinguishable from a full decode at these sizes.

Each key has an environment variable of the same shape — `SORTA_PREVIEW_CACHE`,
`SORTA_PREVIEW_DIR`, `SORTA_PREVIEW_MAX_EDGE`, `SORTA_PREVIEW_QUALITY`,
`SORTA_PREVIEW_MAX_GB` — and the variable **wins** over `config.yaml` when both are
set.

**A ceiling: `imaging.preview_cache_max_gb`.** The default is `0` — no ceiling, and
the cache grows for as long as the disk allows. That is deliberate: the cache pays for
itself on every full run, so the answer to a full disk is to **bound the cache, not to
switch it off**. Switching it off puts the decode of the original back into every
stage — 336 ms a frame instead of 73.

With a ceiling set and the cache over it, the previews that have gone **longest
without being read** are deleted, and only as many as it takes to fit. Not the oldest:
a preview later stages keep reading is cheaper to keep than to decode again. The
directory is never emptied wholesale — the only reason to delete a file is that the
total does not fit.

The size is checked **every 512 previews rather than on each write**, because the
check walks the whole directory and that is tens of thousands of files. Between checks
the cache can exceed the ceiling by roughly 75 MB, which is nothing against any
ceiling worth setting.

To pick a number: about 150 KB a photo, so ~45 GB at 300 000 frames and ~75 GB at half
a million. `sorta cache` prints the current size and the ceiling (§17); the "Process"
tab of `sorta ui` shows the same, and the settings column changes the value without a
restart.

**Video goes into this cache too.** A frame is extracted from the clip and stored
beside the photo previews: 68% of the videos in a real collection turned out to be
HEVC, which browsers do not play, so a frame is the one thing that shows the same
everywhere. No pipeline stage decodes video at all — these previews exist for the
interface.

```yaml
imaging:
  # video_previews: true   # false → clips get no tile in the UI
  # video_workers: 4       # frame-extraction threads
  # video_frames: 6        # frames in the lightbox filmstrip; 1 → a single tile
```

`imaging.video_previews` switches these tiles on or off entirely, `imaging.video_workers`
sets the number of extraction threads, and `imaging.video_frames` is the filmstrip in
the lightbox: one frame says what the clip is, six say whether it is yours and where it
was shot. The strip is built in one pass and only once the lightbox is actually opened.
The environment variables are `SORTA_VIDEO_PREVIEWS`, `SORTA_VIDEO_WORKERS`,
`SORTA_VIDEO_FRAMES`.

---

## 19. The run log

Nothing of a long run survives on the console — `sorta ui` in particular lives for
hours in a background window nobody watches. Every command therefore also writes a
log file.

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\sorta\logs\sorta.log` |
| Linux / macOS | `~/.cache/sorta/logs/sorta.log` |

It rotates at **5 MB, keeping 5 files** (`sorta.log.1` … `sorta.log.5`), so it cannot
grow without bound. A pipeline run (`sorta run`) and every `sorta ui` start also write
an environment header (Sorta and Python versions, platform, where the `sorta` package
was imported from, GPU state, geo data, working directory) — the same facts
`sorta doctor` prints, recorded at the moment of the run.

**Stage timings.** Every stage of a pipeline run (`sorta run` or the UI **Process**
button) writes a machine‑greppable pair of lines with a stable
`stage=<name> … elapsed=<seconds>` shape, so a profile of a run can be read without
parsing prose. A real profile from a run on 2026‑07‑25:

```
2026-07-25T23:37:10.345 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=index started
2026-07-25T23:37:43.854 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=index elapsed=33.509
2026-07-25T23:37:43.855 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=geo started
2026-07-25T23:37:47.727 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=geo elapsed=3.872
2026-07-25T23:37:47.727 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=landmarks started
2026-07-25T23:41:34.228 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=landmarks elapsed=226.501
2026-07-25T23:41:34.229 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=junk started
2026-07-26T00:10:34.736 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=junk elapsed=1740.508
2026-07-26T00:10:34.736 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=phash started
2026-07-26T00:26:09.900 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=phash elapsed=935.164
```

Read straight off: `junk` took 29 minutes and `phash` 15.6 — together 91 % of the
49‑minute run — while `index` (33 s) and `geo` (4 s) are noise. That is where to point a
`--extra gpu` install, `naming.clip.decode_workers` or `naming.ocr_workers` (§21), and
it is also how you tell "it hung" from "it is in the slow stage".

Other shapes of the same line: `stage=<name> failed elapsed=…` (with the traceback
after it) and `stage=<name> interrupted (…) elapsed=…` for a cancelled run — pressing
**Cancel** in the UI is not an error. A stage that reports a count also appends
` processed=<n> rate=<n>/s`.

**Overrides.**

- `SORTA_LOG_FILE=<path>` — write the log somewhere else.
- `SORTA_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` — the level of the **file** sink
  (default `INFO`).

The file level is independent of the console one: `log_level` in `config.yaml`
controls only what the console prints (default `WARNING` — warnings and errors), while
the file keeps the `INFO` detail including the stage timings. Turning the console
quiet does not make the log file quiet.

---

## 20. Offline models

The ML models (CLIP, easyocr, and the VLM tier if you use it) are downloaded **once**.
After that Sorta needs no network for them at all — which is what "local by default"
has to mean in practice.

The switch is automatic: as soon as the Hugging Face cache holds at least one
downloaded model, Sorta starts every command with offline mode on
(`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`), so no stage phones the Hub to check a
revision for weights that are already on disk. A fresh machine with an empty cache is
left alone, so the first download still works.

- `SORTA_ALLOW_MODEL_DOWNLOAD=1` turns the automatic switch **off** — use it when you
  need to fetch a *new* model on a machine that already has others cached (e.g.
  enabling the VLM tier for the first time).
- If you set `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` yourself, Sorta never
  overrides your value.

Note that face detection (insightface/`buffalo_l`) keeps its own cache in
`~/.insightface/models` and is not affected by these variables.

---

## 21. Configuration reference

Key sections of `config.yaml` (see `config.example.yaml` for the full template):

```yaml
sources: ["D:/Photos"]         # folders to scan (recursive)
database: "sorta.db"           # SQLite index path
language: ru                   # ru | en | ja — folder, UI and CLI-message language (default en)

index:
  min_file_size_kb: 5          # ignore tiny files
  workers: 8                   # parallel hashing
  exif_workers: 8              # parallel exiftool sessions (0/absent = auto min(8, cores))
  skip_dirs: [".thumbnails", "@eaDir", "$RECYCLE.BIN", "System Volume Information"]

geo:
  provider: offline            # offline (bundled GeoNames) | online (Nominatim/OSM)
  session_gap_hours: 6         # gap that splits GPS-inference sessions
  nominatim_url: "https://nominatim.openstreetmap.org"   # only if provider: online
  nominatim_user_agent: "sorta-photo-organizer"          # required by OSM policy
  cache_max_age_days: 180      # how long a cached online answer stays fresh (0 = never expire)

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
  ocr_workers: 4               # parallel OCR detectors (default min(4, cores))
  product_candidate_min: 0.4   # product-CLIP above this → the frame goes to the VLM tier
  clip:
    batch_size: 16             # GPU forward batch for CLIP
    decode_workers: 0          # CLIP decode threads; 0 = auto min(cores, 16)

vlm:                           # the local VLM runtime — shared by junk and event naming
  enabled: false               # deep VLM classification tier (needs the `vlm` extra);
                               #   same as `--deep` / the UI "Deep analysis" checkbox
  model: "Qwen/Qwen2.5-VL-3B-Instruct"
  workers: 4                   # frame-preparation threads (default min(4, cores))
  max_edge: 896                # the long edge of the frame the model sees

imaging:                       # the preview cache — see §18
  preview_cache: true
  preview_max_edge: 1536
  preview_quality: 88
  preview_cache_max_gb: 0      # 0 = no ceiling; otherwise evict down to the limit

log_level: WARNING             # console verbosity only; the run log stays at INFO (§19)
```

### The complete list of keys

Above are the ones people change most often. Below is everything else, by section. The
default given is the one that applies when the key is absent from `config.yaml`
entirely.

**`index:` — what gets indexed**

| Key | Default | What it does |
|---|---|---|
| `index.extensions` | three lists: `photo`, `raw`, `video` | Which extensions count as a photo, a RAW file and a video. A file whose extension is in none of the lists is not indexed at all. |
| `index.min_file_size_kb` | `5` | Smaller files are skipped — those are icons and artefacts, not shots. |
| `index.compute_phash` | `true` | Whether `index` computes pHash as it goes. With `false`, near-duplicates appear only after a separate `sorta phash`. |
| `index.phash_max_distance` | `5` | The Hamming distance below which two frames count as near-duplicates (§10). |
| `index.skip_dirs` | `.thumbnails`, `@eaDir`, `$RECYCLE.BIN`, `System Volume Information` | Service directories the walk never enters. To exclude **your own** folders, see §21a. |

**`dedup:` — which file of a duplicate group is the real one**

| Key | Default | What it does |
|---|---|---|
| `dedup.canonical_strategy` | `prefer_exif_then_largest` | Among exact duplicates the canonical file is the one that has EXIF, and the larger one when that ties. The rest are marked duplicates and stay out of the layout. |
| `dedup.keeper_vlm` | `false` | Ask the local VLM which frame of a near-duplicate group is the one to keep — one comparative question ("which of these is best") per group, not a score per frame. It sees what sharpness cannot: closed eyes, a head turned away, a hand across the lens. Needs the VLM extra installed; the answer is stored in the `group_keeper` table as a **recommendation** together with its source (`sharpness` or the model), and nothing is deleted, moved or marked by it — the decision about a duplicate stays yours. With this off the Duplicates tab keeps recommending by sharpness, exactly as before. |
| `dedup.keeper_max_frames` | `5` | How many frames of one group go into that question — the best N by sharpness; the rest are not shown and the answer applies to the whole group. Groups of a few dozen frames do occur, and a 3B model asked to compare 38 pictures answers nothing usable. |
| `dedup.keeper_min_group_size` | `3` | The smallest group worth asking the model about. `2` is every group; set `3` to pay only where the choice is genuinely unclear — on a **pair** of the same scene sharpness already compares the two frames honestly, and pairs are the large majority of groups (791 groups on the reference collection, 115 of them with three frames or more). |

**`geo:` — how a place is decided**

| Key | Default | What it does |
|---|---|---|
| `geo.session_gap_hours` | `6` | The time gap that separates shooting sessions. Within a session, a frame with no GPS takes its place from its neighbours in time. |
| `geo.nominatim_url` | `https://nominatim.openstreetmap.org` | The online geocoder's address. Your own server removes both the rate limit and the privacy question. |
| `geo.nominatim_user_agent` | `sorta-photo-organizer` | Required by OSM policy: the public service rejects requests without a meaningful User-Agent. |
| `geo.nominatim_timeout` | `10` | Timeout of a single geocoder request, in seconds. |
| `geo.cache_coord_digits` | `3` | How far coordinates are rounded for the cache key. Three digits is about 110 m, so neighbouring frames of one place ask the provider once. |
| `geo.cache_max_age_days` | `180` | How long a stored provider answer stays fresh; `0` never expires. The cache lives in the database and survives "Start over" (§18). |

**`events:` — how frames become events**

| Key | Default | What it does |
|---|---|---|
| `events.gap_hours` | `6` | The gap that starts a new shooting session. |
| `events.merge_gap_hours` | `18` | Adjacent sessions closer than this merge into one event, so an evening and the next morning of one trip do not drift apart. |
| `events.trip_merge_gap_hours` | `48` | Sessions within this span and `trip_merge_max_km` count as one trip. |
| `events.trip_merge_max_km` | `120` | The furthest the files' coordinates may lie apart for two sessions to still be one trip. Merging goes by coordinates rather than by city id: otherwise a trip through villages falls into pieces. |
| `events.min_event_size` | `5` | Smaller groups do not become events. |

**`faces:` — faces and clusters**

| Key | Default | What it does |
|---|---|---|
| `faces.min_face_px` | `40` | Faces smaller than this are not embedded: they yield no stable vector and only add noise to the clusters. |
| `faces.det_threshold` | `0.7` | The detector's confidence threshold. |
| `faces.min_cluster_size` | `5` | The minimum number of faces in a cluster (HDBSCAN); anything smaller stays noise. |
| `faces.max_distance` | `0.5` | The cosine distance threshold between embeddings inside a cluster. |

**`sort:` — how the layout is built**

| Key | Default | What it does |
|---|---|---|
| `sort.multi_person` | `primary` | A frame with several people goes to the person with the largest face. |
| `sort.exclude_dirs` | `[]` | Subfolders that are not laid out (their files stay where they are). |
| `sort.thumbnail_workers` | `0` (auto) | Threads building the thumbnails of the HTML plan. |
| `sort.album_dir` | `null` | The album root; by default `_Albums` next to the database. |
| `sort.report_dir` | `null` | Where `sort` writes its plans (CSV/HTML); by default `report_output/` next to the database. |
| `sort.drop_unlocalized_district` | `true` | Do not add the district segment when its name does not translate into the layout language — a folder called `498817` explains nothing. |

**`naming:` — classification thresholds and the naming providers**

| Key | Default | What it does |
|---|---|---|
| `naming.landmark_threshold` | `0.85` | The CLIP threshold for placing a frame by a landmark. Lowering it is dangerous: a wrong city is worse than no city. |
| `naming.landmark_group_min` | `5` | How many frames in a row must agree on one landmark before it is believed (corroboration, F75). |
| `naming.landmark_group_dominance` | `0.6` | The share of the group one landmark must hold to count as corroborated. |
| `naming.junk_threshold` | `0.85` | The CLIP threshold for `screenshot`/`meme`. |
| `naming.document_threshold` | `0.9` | The CLIP threshold for documents. |
| `naming.text_frac_min` | `0.08` | A frame CLIP called a document, but with less than this share of its area under text, goes back to being a scene — the false-positive gate. |
| `naming.text_frac_document` | `0.15` | A frame with more text than this counts as a document even where CLIP hesitated — the rescue for missed documents. |
| `naming.text_rescue_docscore_min` | `0.3` | OCR runs only on frames whose doc-score is at least this: text detection is expensive and clear scenes are not worth checking. |
| `naming.text_frac_downscale_px` | `1280` | The size a frame is reduced to before text detection. |
| `naming.product_candidate_min` | `0.4` | A frame whose product-CLIP score is above this goes to the deep VLM tier. The threshold is measured: from 0.4 to 0.7 the curve is flat, and raising it costs 10% of the time for lost findings. |
| `naming.clip_model` | `ViT-L-14-quickgelu` | The open_clip model. The `quickgelu` variant is required for the `openai` weights, otherwise the two mismatch. |
| `naming.clip_pretrained` | `openai` | Which weights to load for that model. |
| `naming.clip_batch_size` | `16` | The CLIP forward batch on the GPU. A weak lever: the path is bound by decoding, not by the forward pass (measured: 16 → 64 gained 2%). |
| `naming.clip_decode_workers` | `0` (auto `min(cores, 16)`) | CLIP decode threads — this is the real lever: 8 threads gave 61 frames/s, 20 gave 113. |
| `naming.max_samples` | `4` | How many frames of an event the naming model is shown. |
| `naming.vlm_base_url` | `http://localhost:11434` | The ollama address — used only with `naming.provider: local_vlm`. |
| `naming.vlm_model` | `llava` | The ollama model for that same provider. |
| `naming.vlm_timeout` | `120` | Timeout of an ollama request, in seconds. |
| `naming.claude_model` | `claude-opus-5` | The cloud provider's model — used only with `naming.provider: claude`. |
| `naming.claude_api_key_env` | `ANTHROPIC_API_KEY` | The name of the environment variable holding the key. The key itself is never written into the config. |
| `naming.claude_timeout` | `60` | Timeout of a cloud request, in seconds. |

**`features:` — per-frame quality signals (F113, all off by default)**

| Key | Default | What it does |
|---|---|---|
| `features.pets` | `false` | Whether to look for animals. That is a question about an OBJECT in the frame, so it is asked of CLIP rather than the VLM: through the model it would have cost 4.3 hours over the collection, through the CLIP pass that already runs it costs minutes. |
| `features.pet_threshold` | `0.7` | The confidence threshold. **Measured** on 320 hand-labelled frames, sampled by score band and weighted back to the collection: at `0.7` it marks 805 frames at 92% precision and 54% recall; `0.6` marked 895 at 89% and 58%. Raising it further buys nothing — `0.85` has the same precision for nine points less recall. The scores are stored, so a different threshold needs no new pass. |
| `features.pets_verify` | `false` | Whether to **check each candidate with the local VLM** before the animal label is written: one question with three answers — a live animal, a picture of one (a drawing, a plush toy, a print, a statue), or no animal. It needs `features.pets` (it verifies what CLIP found, it does not search by itself) and it uses the same weights as the rest of the `vlm:` section, not a second model. The reason is that the errors left at 92% precision are the ones no threshold removes: CLIP compares a picture to a text as a whole and cannot tell a cat from a picture of a cat. The answer outranks the score — a frame scored 0.95 and answered "plush toy" is not an animal — while a frame the model could not answer about, or never reached, keeps exactly the label `features.pet_threshold` gives it. With this off a run is unchanged in every respect. |
| `features.pet_candidate_threshold` | `0.3` | Who is shown to the model when `features.pets_verify` is on — the second, much lower threshold. `0.7` above is high because nothing was checking CLIP's answer; once something is, the selection can be widened, and that is where the missing recall lives (about 466 animals sit below `0.3`). **Measured** on 500 random hand-labelled frames, scored by the rule the product itself applies: the bare `0.7` threshold marks 18 frames at 94% precision and 47% recall, the cascade at `0.5` marks 20 at 90% and 50%, the cascade at `0.3` marks 28 at 82% and 64%. `0.3` is the value the cascade exists for — seventeen points of recall for twelve of precision — and the check earns its place there: it removed 6 of the 11 false marks and lost no correct one. **Precision is deliberately lower than the bare threshold gives**; recall is what you are buying. Counted on the stored scores of a 19 757-photo collection at the measured 0.78 s per frame: `0.7` selects 805 frames (10.5 min), `0.5` — 993 (12.9 min), `0.3` — 1 331 (17.3 min), `0.2` — 1 679 (21.8 min), everything — 19 757 (4.3 hours). Ignored when the check is off. |
| `features.junk_rescue` | `false` | Whether to look for the **screenshots, photographed screens and receipts this stage has already filed as photographs**. A search by words put memes and screenshots at the top of its results while every row of the embedding table said `photo` — so those frames were not junk leaking into the index, they were a few percent of the "photographs" being misclassified, and they go into the city layout, the duplicates and the albums like ordinary pictures. They are found with a zero-shot query over the vectors `features.store_embeddings` already keeps: `junk_score = max(looks like a screenshot / meme / text / receipt) - max(looks like a photograph)`, written to `frame_quality.junk_score` for every photograph that has a vector (NULL without one). It costs no pass over any image — the vectors are on disk and the prompts are five short strings. **With the deep tier (`vlm.enabled`) off no verdict changes at all**: the score is stored, the candidates are counted, and the run is otherwise the one you had. With it on, the frames above the threshold below are shown to the model, and only the model's answer moves a verdict — a candidate it calls a photograph stays one. |
| `features.junk_rescue_threshold` | `0.02` | Who is shown to the model. Reviewed by eye on a 19 753-photo collection: `+0.05` selects 93 frames (0.5%) and they are junk outright; `+0.02` selects 955 (4.8%), and the band between the two still holds about 17% real photographs; `0.00` selects 5 688 (28.8%), where junk is down to single figures. This is a **selection** threshold and not a verdict: at about 85% precision, reclassifying by it directly would take some 150 living photographs out of the layout, which is exactly the mistake measured for the animal label — so the model gets the last word. 955 frames is ~12 minutes at the measured 0.78 s per frame. Read it off your own collection first: `python scripts/measure_junk_rescue.py` prints the distribution and what every threshold would select, before anything is switched on. |
| `features.detector` | `false` | Whether the animal label gets a **third tier: an object detector** (COCO, torchvision) over the candidates a query selects. It needs `detect.enabled` — its own master switch, not `vlm.enabled`, because a detector is not a VLM. It exists for **one measured slice out of three**: on 200 hand-labelled frames at confidence 0.5 the detector is 62% precision / 87% recall on animals, against 71% / 33% for the label the pipeline writes today — while on people it is 42% precision against ~100% from the face boxes (§6), and on food 20% / 15%, because COCO has no `food` class at all (it has a banana, a pizza, a sandwich). So people and food are deliberately **not** detected here. It is a cascade and never a pass: a full run of the detector over 22 096 photographs is 30.8 minutes at the measured 83.8 ms per frame, while the ~2 000 candidates of the query cost ~3 minutes. The answer overrides the CLIP label in **both** directions — an animal found where the score was too low is labelled, a frame CLIP called an animal with nothing on it loses the label — and every refusal (no stored vectors, no weights, an error on one frame) falls back to the label you already had, never to "no animal". What was found is stored with its class, confidence and box (table `detections`), so a "cats apart from dogs" slice is a query and not another pass. Needs `features.store_embeddings`: without stored vectors there are no candidates, and the stage says so instead of falling back to a pass over everything. |
| `features.detector_candidates` | `2000` | How deep into the query ranking the candidate list goes — the **one** number that decides what this costs, because the detector sees nothing else. 2 000 frames is ~3 minutes at the measured 83.8 ms per frame. Recall is bounded by the query's own recall at this depth (87% on the labelled sample), so raising it buys the animals the query ranked lower and costs time linearly; `python scripts/measure_detector.py` prints that ceiling per depth on your own collection. |
| `features.detector_threshold` | `0.5` | The confidence at which a detected box counts as an animal. **Choose it from the table, not in advance:** `python scripts/measure_detector.py` prints precision and recall at 0.3 / 0.5 / 0.7 over a labelled sample of your own, next to what the current CLIP label scores on the same frames. `0.5` is the row that was measured (62% / 87%). The lesson behind that rule is `features.pet_candidate_threshold`, where the value a brief proposed in advance turned out to be the worst row of its table. Every box is stored with its score, so re-choosing this costs a query and no new pass over any image. |
| `features.landmarks_verify` | `false` | Whether to **check each landmark CLIP proposes with the local VLM** before the place is written: the model is asked what well-known place the frame shows, and only a proposal it names itself goes on. The order does not bend — CLIP proposes, the model checks, the corroboration behind `naming.landmark_threshold` decides; agreement between the two models never overrules a country named in the path. It exists because CLIP's failure here is not one of perception but of knowledge: the wrong cities scored 0.980 against 0.991 for the right one, and no threshold splits them. That is also why it was measured before it was built. On 104 frames with a known answer — 24 of them proposals CLIP believed and corroboration threw away — the model confirmed a wrong city zero times, at 92% accuracy; not because it knows every landmark but because it stays silent when it does not (71 of the 104 answers named nothing). Needs the `[vlm]` extra; if the weights will not load, the run behaves exactly as it does with this off, and with it off it is unchanged in every respect. |
| `features.landmark_candidate_threshold` | `0.5` | Who is shown to the model when `features.landmarks_verify` is on — the second, much lower threshold, exactly like the pair above. `naming.landmark_threshold` is high because nothing was checking CLIP's proposal; once something is, the selection widens. Measured on the 7 619 place-less frames of a live collection: `0.85` gives 10 proposals (8 kept by corroboration, 2 dropped), `0.7` — 66 (52 / 14), `0.5` — 151 (127 / 24). 151 questions is a couple of minutes of VLM. Never set above `naming.landmark_threshold` — the check is there to widen the band, not to narrow it, and a higher value is clamped back down. Ignored when the check is off. |
| `features.group_photo_faces` | `3` | How many detected faces make a photograph a **group photograph** — the second of the three face slices (§6). Not a confidence threshold: those slices are read straight off the `faces` table, so a frame is in one because the detector found a box on it. Three, because two people in a frame are a couple or a passer-by and the slice exists to find the gatherings. Raise it to keep only the crowds. |
| `features.portrait_face_share` | `0.08` | What makes a frame a **portrait**: exactly one detected face, covering at least this share of the frame area (the face box over width × height). Geometry, not confidence — a box over 8% of the area is about 28% of each side, i.e. head‑and‑shoulders rather than a person standing in a landscape. This is a starting value and **not a measurement on a real collection**; the boxes are stored, so re‑choosing it costs a query and no new pass over any image. A frame whose dimensions the index never learned is not in this slice — the share cannot be computed for it. |
| `features.sharpness_max_edge` | `512` | The size a preview is reduced to before sharpness is measured (the variance of a Laplacian). |
| `features.sharpness_band_min` | `30` | Below this a frame is plainly blurred and there is nothing to ask a model about. |
| `features.sharpness_band_max` | `300` | Above this it is plainly sharp. Between the two lies the band of uncertainty, and only that band reaches the VLM. |
| `features.subject_score_min` | `0.9` | The "there is a subject in this frame at all" threshold — what separates a shot from a pocket accident. |
| `features.blur_review_max` | `90.0` | How far down the blur review list opens by default. **Not a "blurred" verdict** — the measurement says the opposite. Reviewed by eye in bands, blurred frames turned up in **every** band up to 400: sharpness ranks, it does not classify, and no cutoff separates junk from a soft but wanted photograph. What the bands did show is where the yield falls off, around 90–120 (below 70: 378 frames, below 90: 530, below 120: 785, below 160: 1215, on a 19 757-photo collection). The list opens here and continues on demand. **Nothing is ever deleted by this number.** |
| `features.store_embeddings` | `true` | Whether to keep the CLIP vector of every canonical photograph (table `clip_embeddings`) instead of discarding it once the junk stage has read its scores off it. Nothing is shown for it: it is what lets a later feature — search by words, an album from a query, "frames like this one" — work without a fresh CLIP pass over the whole collection. On by default, because the price is small (~60 MB per 20 000 photos, written inside a pass that runs anyway) and the off state is the one where each such feature costs a full pass. Turn it off on very large collections: 300 000 photos are ~920 MB. Vectors are stored L2-normalized in float32 together with the model that produced them — change `naming.clip.model` or `naming.clip.pretrained` and the stored rows are recomputed rather than used, because vectors of different models are not comparable. |
| `features.search_limit` | `200` | How many frames a search by words takes: `sorta search "cake"` prints this many, and `sorta album query "cake" --dest …` gathers this many. **Not a similarity threshold**, and there will not be one, for the same reason sharpness has none — a CLIP score orders frames against each other and means nothing in absolute terms, so "this really is a cake" is not a line anybody can draw. What this number chooses is the size of the sample a person then looks through: raise it to see further down the ranking, lower it for a shorter list. Two limits of the method are worth knowing before you rely on it — compound queries ("a cake with candles on a table by the window") are weak while single subjects are what CLIP does well, and the population is personal photographs only (a screenshot or a document has no vector at all). Needs `features.search_index`. |
| `features.search_index` | `false` | Whether to build the **search index**: a second CLIP vector per photograph (table `search_embeddings`), computed by a multilingual model and read by search alone. **Off by default because it is not free** — unlike `features.store_embeddings` it is a second CLIP pass over the collection, 19 753 frames in 635 seconds (~10.5 minutes) on the machine it was measured on, plus ~40 MB per 20 000 photographs. What it buys was measured on 217 hand-labelled judgements over 8 concepts: Russian queries go from 22% to 98% precision at top-5, and four of the eight (cake, food, mountains, children) go from returning **nothing at all** to working, while English does not regress (95% against 98%). With it off `sorta search` and `sorta album query "…"` have nothing to rank and say so — the classification vectors are deliberately not used instead, because a ranking produced by the wrong model looks exactly like a good one. |
| `features.search_model` | `xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k` | The model of the search side, `<open_clip architecture>/<weights>`. A key of its own rather than `naming.clip.*`, and that separation is the whole feature: the landmark (0.85), animal (0.70) and cascade (0.50) thresholds are calibrated on the classification model's numbers, so search is not allowed to change it. Change this one and every stored search vector goes stale — the next run recomputes them, and vectors of two models are never mixed. |

### The `vlm:` section

Everything that describes the **runtime of the local model itself** lives in the
`vlm:` section, and both of its consumers use it: the deep junk classification tier
(§14) and the `naming.provider: vlm` event namer (§12). The weights are the same, one
copy per process, so the settings are shared.

| Key | What it does |
|---|---|
| `vlm.enabled` | Turns the deep tier on for good, in the config. Default `false`. For a single run the same is done by `--deep`/`--no-deep` and by the "Deep analysis (VLM)" checkbox in the web UI. Without the `vlm` extra a run falls back gracefully to the fast CLIP tier. |
| `vlm.model` | The model id. Default `Qwen/Qwen2.5-VL-3B-Instruct`. |
| `vlm.workers` | Threads preparing frames (decode + preprocessing) while the GPU classifies the previous one. Default `min(4, cores)`. It does not affect verdicts — labels are applied in candidate order whatever it is set to. |
| `vlm.max_edge` | The long edge the frame is scaled to before the model sees it — the main lever on what the tier costs. Default `896`. Lowering it is not free: documents are recognised by small text. |
| `vlm.quality` | A toggle of its own for the questions about a frame's quality: are the eyes open, is there a subject worth keeping. Default `false`. A third question — is this an accidental shot — was asked until F122 and has been retired: on a labelled sample it was right 5% of the time, which is noise. The eyes answer is believed only where the face detector found a face. Sharpness and pets are computed without it — by a Laplacian and by CLIP, both free — and the model is asked only about what neither of them decides. |
| `vlm.quality_scope` | Who gets asked: `groups` (frames of near-duplicate groups, the default), `events` (plus a sample from every event), `all` (every live photo). On 20 000 frames `all` means hours of GPU, which is why the default is narrow. |
| `vlm.exclude_classes` | **Privacy:** classes no VLM is ever shown. The default is `[document]` — that bucket holds passports, medical forms and bank papers, and the project already refuses to DECODE them for display. The model is local and nothing leaves the machine, but the call is yours. **The cost is real:** the deep tier is what *corrects* a wrong `document` verdict (a beach photo scored 0.95 as a document on a live run), so an excluded class keeps whatever the fast tier decided. Accepted: `document`, `product`, `screenshot`, `meme`; `[]` shows everything. `photo` cannot be excluded. |

> **An old config needs no editing.** The previous addresses
> `naming.vlm_enabled`, `naming.classify_vlm_model` and `naming.vlm_workers` are
> **still read** and work as aliases for `vlm.enabled` / `vlm.model` / `vlm.workers`.
> A key given in both places is taken from `vlm:`; a key given only at the old address
> makes Sorta log one warning per run saying that `naming.<key>` is deprecated and
> should be renamed. It is a warning, not an error — nothing broke, and you can rename
> it whenever it suits you. `vlm.max_edge` has no old address: it used to be a
> constant in the code.

### The `detect:` section

Everything that describes the **runtime of the object detector** lives in the `detect:`
section — a second kind of model, so a section of its own next to `vlm:`. Its one
consumer is the animal cascade (`features.detector`, §24).

| Key | What it does |
|---|---|
| `detect.enabled` | **The master switch of the detector**, default `false`. `features.detector` says what a detector is wanted for; this says whether one may be loaded at all — the same hierarchy `vlm.enabled` has over the VLM questions, and deliberately a separate switch: a detector costs 83.8 ms per frame and a few hundred megabytes, a VLM 0.78 s and 20 GB, so wanting one says nothing about wanting the other. No new dependency — torchvision is installed with the CLIP side already, and only the COCO weights are downloaded: once, by torchvision itself, into the torch hub cache (`TORCH_HOME`), which is a different cache from the Hugging Face one §20 is about. |
| `detect.model` | The torchvision detection model, by the name of its builder function. Default `fasterrcnn_mobilenet_v3_large_fpn` — the checkpoint every number above was measured with, and the reason the cascade is affordable. A heavier one (`fasterrcnn_resnet50_fpn`) finds a little more and costs several times the 83.8 ms. Change it and the stored boxes go stale: the rows record which detector produced them, so the frames are examined again rather than trusted. |

### Throughput settings

Four knobs decide how fast the heavy stages run. All of them are optional — the
defaults are chosen to be safe on modest hardware, and `config.example.yaml` carries
the full commentary and the measurements behind them.

| Setting | What it does | When to touch it |
|---|---|---|
| `index.exif_workers` | Parallel `exiftool` sessions during `index`. exiftool is a separate process, so this scales almost linearly (measured 11.8 → 2.0 ms per file going from 1 to 8 sessions). Default: auto, `min(8, cores)`. | Raise on a many‑core machine if `stage=index` dominates your log. |
| `naming.ocr_workers` | Parallel OCR detectors in `junk`. Each worker holds its **own** model copy in VRAM, so the default is deliberately conservative: `min(4, cores)`. | Raise only with VRAM to spare; lower it if OCR is what runs you out of memory. |
| `naming.clip.batch_size` | CLIP forward batch on the GPU. The CLIP path is decode‑bound, so this barely moves the needle. Default: 16. | Rarely worth changing. |
| `naming.clip.decode_workers` | Decode threads feeding CLIP — the real lever for `junk`/`landmarks` speed. Default: auto, `min(cores, 16)`. | Raise if `stage=junk` / `stage=landmarks` dominate and the GPU is idle. |

---

## 21a. Skipping folders, and correcting the layout by hand

Two different problems, two different tools. Getting them the wrong way round is the
usual source of confusion, so the wording is deliberate:

- **"do not scan"** — the files never enter the index at all.
- **"do not lay out"** — the files are indexed, but stay where they are.

### Excluding source folders before the scan

A source folder can be excluded *before* the walk reaches it, which is the only option
that saves any work: an excluded subtree costs no `stat`, no hash and no later stage.
On a real 400 GB collection, excluding one `Movies` folder removed 847 files and
54 GB from every stage of the run.

In the web app the first tab shows the folder tree of the source. Every folder has three
states: ☐ process, ◐ don't sort, ☒ don't scan — clicking the mark cycles through them,
and a folder's state applies to its whole subtree. From the command line:

```bash
sorta index --exclude-dir Movies --exclude-dir Screenshots
```

Both write the same file — `excludes.yaml` next to the database, or wherever
`index.excludes_file` points — so a folder excluded in one place stays excluded in the
other. The file is **keyed by source root**, and within a root by what the exclusion
means:

```yaml
"D:/Photos":
  skip_scan:      # do not scan: the files never enter the index
    - Movies
    - DCIM3/Screenshots
  skip_layout:    # do not lay out: indexed, left where they are
    - foto/Greece
    - foto/France
"E:/Archive":
  skip_scan:
    - temp
```

A folder lands in exactly one section: marking one state clears the other. A file in the
old shape (a plain list under the root) is read as `skip_scan` — there is nothing to
rewrite.

Because of that, switching sources needs no decision about old exclusions: each root
carries its own set and coming back restores it. Note that files already indexed under
a newly excluded path are **removed from the index** on the next `index` run — "do not
scan" and "is in the index" cannot both be true. The move journal is never touched.

### Correcting individual files

Automatic rules cannot see what you see. In the web app any file (or a selection) can
be:

- **excluded from the layout** — it is not copied or moved, and does not appear in the
  plan at all;
- **reassigned** — sent to any folder of the current layout.

Both outrank every automatic rule: the duplicate decision, the junk verdict, geo. They
live in the index and are wiped by a from-scratch reindex, like face labels and event
names.

### Where a file came from

Each row shows the folder the file was taken from, with the full path in the tooltip.
This is usually the fastest way to judge a wrong guess: a Colosseum match is plainly
wrong once you can see the file lives under a folder named after a Karelian park.

---

## 21b. Files with no reliable date

A photo whose date could not be established does not go into the same bucket as a
downloaded picture. The split is by whether there is any trace of a camera at all
(make, model or GPS):

- a camera shot with a lost date — the "no date" folder;
- no camera trace whatsoever — the **downloaded** folder.

On the validation collection this line fell almost perfectly: of 1 059 undated files,
1 057 had no camera trace and were messenger cache with names like
`10083666931142353280.JPG`; exactly 2 were real shots. The folder is deliberately not
named as a verdict — forwarded pictures are often worth looking through.


---

## 22. The Review workspace

The **Review** tab (§6) is one workspace for everything that has to be looked at by eye
and partly deleted. Four slices, switched by the buttons at the top, and each button
carries the number of frames still undecided:

- **Duplicates** — near‑duplicate groups, with the recommended keeper (★) pre‑selected
  (§10).
- **Blurred** — frames from the blurriest down; the list opens as far as
  `features.blur_review_max` (default `90`) and continues on a button, including past
  that window.
- **Closed eyes** — the model's answers about frame quality (`vlm.quality`, §24). The
  question is only asked where the detector found a face, so without a `faces` run this
  slice is empty and says so.
- **No subject** — frames where the model found no subject at all: a shot of the floor,
  a smeared wall, an accidental press.

**The decision goes into one shared journal, one per file.** There are three buttons:
**Mark for deletion**, **Keep**, **Clear the mark**. Marking "delete" is a mark, not a
deletion: those files leave for the `_delete` folder on the next layout (§9), and until
then nothing happens. A **Keep survives a recompute** — a frame you have decided about
is not asked about a second time, even if the next run scores it as blurred again. The
journal is shared by all four slices, so a file that turns up both in "blurred" and in
"no subject" has one decision, not two.

**There is no "delete everything below the threshold" button, and there will not be
one.** That follows from a measurement rather than from caution: reviewing by sharpness
bands, blurred frames turned up **in every band up to 400**. Sharpness, then, ranks but
does not classify, and the line that separates a write‑off from a soft but wanted frame
simply does not exist. So `features.blur_review_max` (§21) is how far the list opens,
not a verdict, and **nothing is ever deleted by that number**. What the bands did show is
where the returns fall off — around 90–120: below 70 there are 378 frames, below 90 530,
below 120 785, below 160 1,215, on a collection of 19,757 photographs.

While the Review holds undecided frames, the Layout tab shows a warning — see §6 for why
the order is what it is.

---

## 23. Search by words

The search goes by the image, not by file names and not by tags: every personal
photograph has a CLIP vector (`features.store_embeddings`, §21), and the query is
compared against it.

```bash
sorta search "cake"                 # the ranking: rank, closeness, path
sorta search "cake" --limit 50      # how many lines to print
sorta album query "cake" --dest /path/to/albums --apply   # the same thing, as a folder
```

In the web UI this is the search line at the top of the **Slices** tab (§6): type the
words, press **Search**, then gather what came back into a folder with the album button —
like any other slice.

**It is a ranking, not a filter.** The list is sorted by closeness to the query and there
is no "this really is it" threshold, nor will there be one: the CLIP score orders frames
against each other and means nothing in absolute terms. Read top‑down and stop where the
resemblance runs out. `features.search_limit` (default `200`) is a sample size, not a
similarity threshold.

**Four states of the search index.** The line under the input always says which one it
is in:

1. **The index is empty** — there is nothing to search yet. An ordinary run over the
   collection fixes it: no separate model and no extra setting are needed, the vectors
   are written on the same pass that classifies the frames anyway.
2. **The index was computed by another model** — its vectors are not comparable with the
   current one, so the ranking would be plausible nonsense. Those rows never enter the
   ranking; the collection has to be processed again, which recomputes them.
3. **Part of the collection is covered** — "searching N of M photographs": the rest join
   the index on the next run. That is the normal life of a growing archive, and telling
   "it is not in the collection" from "it is not in the index yet" matters more than a
   tidy number.
4. **Everything is ready** — "searching all M photographs of the collection".

**A limit measured on 2026‑08‑02: ask in English.** The index is computed by the
`ViT-L-14` model (`naming.clip_model`, `ViT-L-14-quickgelu` by default), trained on
English captions, so **queries in other languages work noticeably worse than English
ones**. Checked on eight pairs of queries: `cake` finds cakes, its Russian translation
does not. Until a multilingual index exists, the advice is blunt: phrase the query in
English even when the rest of your work is in another language.

Two more things worth knowing in advance: CLIP is weak on compound queries ("a cake with
candles on a table by the window") and good at single subjects; and the population is
personal photographs only — a screenshot or a document has no vector at all, so a
passport can never surface in the results.

---

## 24. Animals and frame quality

The signals about the frame itself, rather than about the place or the people in it. All
of it is **off by default** — this is the tier you pay for in time — and it is switched
on either by a flag for one run (§8) or by a key in `config.yaml` (§21).

### Animals

- **There is exactly one class, `animal`.** There is no species and there will not be
  one: the measurement says the binary question "is there an animal in this frame" is
  answered correctly in 92% of cases, and the species question is not. Hence one
  **Animals** slice in the UI, and one album in the terminal:
  `sorta album animal --dest …`, with no selector (§13).
- **CLIP answers it, not the VLM** — on a pass that runs anyway. That is why `--pets`
  costs minutes: the same coverage through the model would cost 4.3 hours.
- **The threshold `features.pet_threshold` is `0.70`.** Measured on 320 hand‑labelled
  frames: at `0.70` it marks 805 frames with a precision of 92% and a recall of 54%.
- **The verifying cascade** is switched on by `features.pets_verify`: CLIP proposes
  widely, the local VLM confirms or drops (it can tell a cat from a plush cat, CLIP
  cannot). Candidates are then selected by a separate, far lower threshold,
  `features.pet_candidate_threshold` — **`0.30`**. Measured on 500 random hand‑labelled
  frames: at `0.30` the cascade marks 28 frames at 82% precision and 64% recall, against
  20 at 90% and 50% for `0.50` and 18 at 94% and 47% for the bare threshold with no
  cascade at all. The wide gate buys recall with precision on purpose — expect about one
  wrong mark in five — and costs roughly 19 minutes against 12 on a collection of 20
  thousand photographs.
- **A false mark can be taken off by hand.** The card offers **Not an animal** / **This
  is an animal** / **Back to automatic**. The correction is stored in the index and
  **survives any recompute**: the next run will not put back a mark you removed. The
  tab, the counter and the `animal` album all read the same rule, so they cannot
  disagree with each other.

### Frame quality

Sharpness is the variance of a Laplacian, computed with no model at all, while "are the
eyes open" and "is there a subject" are asked of the local VLM (`vlm.quality`) — those
are the ones that cost time. Who gets asked is decided by `vlm.quality_scope`:

| Value | Who is asked | Price |
|---|---|---|
| `groups` | the frames of near‑duplicate groups (default) | the narrowest population |
| `events` | plus a sample of each event's frames | wider than `groups`, narrower than `all` |
| `faces` | the frames a face was found on | **≈ 95 minutes on 7,341 frames** |
| `all` | every live photograph | **≈ 4.3 hours on 20 thousand frames** |

Both numbers are measurements, not estimates. `faces` needs a `faces` run: that is a
dependency and not a filter — without one the population is empty and nothing is asked at
all. The answers show up in the Review (§22), and for a single run the whole thing is
switched on with `--quality`/`--no-quality` and `--quality-scope` on `sorta run` and
`sorta junk` (§8).

---

## 25. Troubleshooting

- **`sorta doctor` says `torch: 2.13.0+cpu` on a GPU machine** — the extra never
  reached the install command. `uv tool install` has no `--extra` flag; the extra
  belongs inside the quoted spec: `uv tool install --force "C:\path\to\sorta[gpu]"`
  (§3.3).
- **`sorta doctor` shows no `CUDAExecutionProvider`** while torch reports CUDA — the
  plain `onnxruntime` (pulled in by `insightface`) overwrote `onnxruntime-gpu`, so
  face detection is on the CPU. Fix and explanation in §3.6.
- **Your code changes have no effect** — a non‑editable `uv tool install` is a
  snapshot of the code at install time. Reinstall with `-e` (§3.3) or use a project
  venv (§3.4).
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
- **Free disk space dropped by gigabytes during a run** — most likely the preview
  cache (§18): ≈150 KB per photo and up. Check it with `sorta cache`, delete it with
  `sorta cache --clear`, move it elsewhere with `imaging.preview_dir`, or switch it
  off with `imaging.preview_cache: false` (at the cost of decoding every frame again
  in every stage).
- **A model refuses to download on a machine that already has other models** —
  expected: Sorta switches Hugging Face to offline mode once its cache is non‑empty
  (§20). Run the command once with `SORTA_ALLOW_MODEL_DOWNLOAD=1` to fetch the new
  weights.
- **A folder with non‑ASCII name (e.g. Cyrillic) seemed skipped by OCR** — fixed:
  images are decoded via an Unicode‑safe path; update to the latest version.
- **`sorta --help` is English even though the config says `language: ru`/`ja`** —
  expected, not a bug: the help texts live in typer decorators and are evaluated at
  import time, before the config is read. Everything else — progress lines, command
  summaries, folder names, the web UI — speaks the language from `language` (see §4).
  If messages come out in the right language but render as `????` in your terminal,
  that's a separate, purely cosmetic encoding issue (next bullet).
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
