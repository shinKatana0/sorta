<img src="docs/assets/icon.png" alt="" width="72" align="left" hspace="12">

# Sorta

[![CI](https://github.com/shinKatana0/sorta/actions/workflows/check.yml/badge.svg)](https://github.com/shinKatana0/sorta/actions/workflows/check.yml)
[![Release](https://img.shields.io/github/v/release/shinKatana0/sorta)](https://github.com/shinKatana0/sorta/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> Languages: **English** · [Русский](README.ru.md) · [日本語](README.ja.md)

**Index and sort a large photo/video collection** (60 GB+ tested, 300 GB+ by design)
into a clean folder structure — by **city/country**, **person**, or **event** — with
safety first: dry‑run by default, a move journal, and one‑command undo.

Sorta runs **locally** (ML models for faces, scenes and text run on your machine),
**never modifies your originals**, and offers both a **CLI** and a guided **local web
app**.

![Sorta — the local web app: sort by city, switch folder language, review duplicates, undo](docs/assets/hero.gif)

<sub>The local web app (`sorta ui`) on a synthetic demo collection — city tree, live folder‑language switch (en/ru/ja), and duplicate review.</sub>

> ⚡ **An NVIDIA GPU (CUDA 13) is what this is built for.** 6 GB of VRAM covers
> everything except the deep VLM tier, which peaks at a measured **20.5 GB** and
> wants a 24 GB card. The CPU profile runs the rest and has never been timed on a
> large collection — see [System requirements](#system-requirements).

> 📖 **User guide:** [English](docs/guide/user-guide.en.md) ·
> [Русский](docs/guide/user-guide.ru.md) · [日本語](docs/guide/user-guide.ja.md)

---

## Highlights

- **City / person / event sorting** from a single index — switching modes needs no
  re‑scan.
- **Offline geolocation** (bundled GeoNames) with GPS + session inference; optional
  online Nominatim/OSM.
- **Fast basic run by default.** The full pipeline over ~38 000 files takes **67 minutes**;
  turning the deep VLM tier on takes **272**, so it is **off by default** and named on the
  run screen with what it costs. The deep tier buys one thing the fast one cannot produce
  at all — the `product` class — and is incremental, so its price is paid once.
- **Faces and events are opt‑in** (`--faces`/`--events`, or the matching checkboxes) —
  they're the slowest stages and not everyone needs them.
- **Faces & people:** local detection + clustering (insightface), once enabled; name and
  merge clusters, then sort or build per‑person albums. **A name typed into the search box
  finds that person** — including clusters you merged — as an exact selection rather than
  a ranking.
- **Search your collection in words** (multilingual CLIP), and **pin any query as a tab**
  of its own. The built‑in slices are saved queries too, so a pinned one is not a lesser
  kind of thing.
- **Slices for what a photograph IS and what it turned out like:** people, children,
  animals, screenshots, memes, documents, products, downloaded images, blurred, closed
  eyes, low resolution.
- **Every slice states what it measured.** A caption saying "the model calls these
  on‑screen; about one in three is an ordinary photograph, check before deleting" is
  worth more than a number pretending to be a verdict. Lists are **ranked, not
  thresholded** — the product orders them and the person decides where to stop.
- **Duplicates:** exact (blake3) and near‑duplicate (perceptual hash) with a
  batch‑review UI.
- **Restore a soft frame** where it helps: measured on blind pairs, the model beats plain
  enlargement on **small** frames (62% against 10%) and does nothing measurable above the
  ceiling, so that is exactly where the action is offered.
- **Albums:** collect any slice or query into a named folder via **hardlinks** (near‑zero
  extra space), copy, or move — with a subset selection when you don't want all of it.
- **Local web app** (`sorta ui`), five tabs arranged by what they do to your files rather
  than by pipeline stage: **Overview** (the collection's condition on one screen),
  **Review** (duplicates, blur, closed eyes), **Layout** (where things go and by what),
  **Slices** (queries, pins and built‑ins), **Moves** (what the last batch did, and undo).
- **Trilingual** UI and folder names: **ru / en / ja**.
- **Nothing leaves the machine.** Every model runs locally; the only outbound path is
  optional Nominatim, which receives rounded **coordinates** and never a photograph.
- **Safe by design:** dry‑run, journal + `undo`, blake3 verification, never overwrites
  (suffix `_1`, `_2`).

---

![The Overview tab — the condition of the collection on one screen, every number a link into the frames behind it](docs/assets/process.png)

![The Layout tab — the two things needed every time (where, and by what) on the desk; everything else behind the gear](docs/assets/layout.png)

### Two entry points, one job each

The **CLI runs the pipeline** — every stage, `sort`, `album`, `search`, and a per‑run flag
for every option. The **web app is where you decide**: resolving duplicates, correcting a
place, pinning a query, restoring a frame. `sorta dupes` lists duplicates; it does not
resolve them, because resolving them from a command line means choosing without seeing the
frames — and on blind pairs no rule we measured beat chance at that. Everything
reproducible is scriptable; nothing that needs an opinion pretends to be.

---

## System requirements

> [!WARNING]
> **Sorta is built for a machine with an NVIDIA GPU.** Every stage that costs real time —
> face detection, CLIP classification, the deep VLM tier — is a neural network, and the
> numbers below were measured on one.
>
> **The deep tier needs ~20.5 GB of VRAM** (a measured peak, not an estimate: Qwen2.5‑VL‑3B
> holds that much and a second copy does not fit). A 24 GB card runs it; a 8–12 GB card
> does not, and the run will fail rather than crawl. The rest of the pipeline is far
> lighter — about 3 GB — so **city sorting, duplicates, faces and events are comfortable on
> a 6–8 GB card**.
>
> **The CPU profile works and has never been measured end to end.** It exists so that a
> machine without a GPU can still index, geolocate, find duplicates and sort by city. Take
> it as: small collections, or a large one with faces, events and the deep tier switched
> off. Nobody has run 38 000 files through it, and we will not pretend to know how long
> that takes.


| | CPU profile (the `cpu` extra) | GPU profile (the `gpu` extra) |
|---|---|---|
| Hardware | Any x86‑64 machine | NVIDIA GPU + driver supporting **CUDA 13** |
| VRAM | n/a | **~3 GB** for everything except the deep tier (measured on RTX 5090: CLIP ViT‑L 2.0 GB + buffalo_l 0.6 GB) — **6 GB** comfortable. The **deep VLM tier peaks at 20.5 GB**, measured, so it wants a **24 GB** card |
| Speed | Works; **never measured end to end** on a large collection | Measured 2026‑08‑05, 38 485 files from cold: index 5.3 min · geo 3 s offline · landmarks 4.9 · classify 32.4 · faces 14.2 · events 2 s · junk 19.3 · phash 45 s — **77.5 min** in all, and ~10 minutes less with the preview cache warm |
| Best for | City sorting + duplicates on any machine; smaller collections with faces/events on | Large collections (300 GB+) with faces/events routinely on |

### Running on a smaller card

The 20.5 GB peak is the deep tier at its default input size. There are knobs, and here is
what each one is actually known to buy:

| knob | what it does | measured |
|---|---|---|
| `vlm.enabled: false` | the deep tier does not load at all | the rest of the pipeline needs ~3 GB — this is the real answer for a small card |
| `vlm.max_edge` | the frame size the model sees; tokens grow with area | 896 → 672 is **×1.48 faster**, and **7.5% of documents become "photo"** (300 frames, F102). The default stays 896 because that trade was judged bad; the knob is there for whoever judges otherwise |
| `vlm.workers` | threads preparing frames for the GPU | changes CPU-side preparation, not VRAM |
| `imaging.preview_cache_max_gb` | ceiling on the thumbnail cache | disk, not memory |

**Be plain about the shape of it**: nothing here turns 20.5 GB into 8. Lowering
`vlm.max_edge` reduces the peak because the token count falls with the area, but by how
much has not been measured — the resolution study priced speed and verdicts, not memory.
The honest advice for a card under 24 GB is to leave the deep tier off: it produces one
class (`product`) and nothing else in the product depends on it.

Common to both: Python **3.11–3.14**, [`uv`](https://docs.astral.sh/uv/), and
**`exiftool` on PATH** (required for HEIC/RAW/video metadata — without it Sorta falls
back to Pillow, which only reads JPEG/PNG/TIFF/WEBP and no video). Disk space for the
index (SQLite) and thumbnails scales with collection size; `--copy` sorting needs
roughly ×2 the collection size, `--link` (hardlink) needs almost none.

Timings above are from our own hardware, not a guarantee. Full breakdown, including
RAM/VRAM notes, in the [user guide](docs/guide/user-guide.en.md#2-requirements).

---

## Quick start

### Windows: the installer (no `uv`, no terminal)

`sorta-<version>-setup.exe` on the [releases page](https://github.com/shinKatana0/sorta/releases/latest)
carries the **base tier** whole and works with no network afterwards: index, EXIF, geo,
duplicates and sorting by city. Its shortcut opens the web app with an icon in the tray
and no console window. The heavier tiers — faces (~400 MB), search by words (~3 GB),
NVIDIA/CUDA 13 (~2.5 GB), the deep VLM tier (~7 GB) — are offered once by the first‑run
wizard (`sorta-setup`, re‑runnable at any time), and **saying no to all of them leaves a
working product**, not a stub.

> ⚠️ **The installer is not signed**, so Windows SmartScreen greets it with "Windows
> protected your PC". Click **More info** → **Run anyway**. The release page publishes a
> `sha256` beside the file — check it with `Get-FileHash sorta-<version>-setup.exe` if you
> would rather look before running. A code‑signing certificate is on the owner's list, not
> in this release; see
> [packaging/windows/README.md](packaging/windows/README.md) for how the installer is
> built.

Everything below is the developer/CLI path — it is unchanged, and it is not what the
installer does.

```bash
# Install once — pick ONE of these. The extra goes INSIDE the package spec and
# selects your hardware profile (`cpu` and `gpu` are mutually exclusive).
uv tool install "C:\path\to\sorta[cpu]"       # no NVIDIA GPU — puts `sorta` on PATH
uv tool install "C:\path\to\sorta[gpu]"       # NVIDIA GPU + CUDA 13 driver
uv tool install "C:\path\to\sorta[gpu,vlm]"   # ...also the deep VLM tier
uv tool install -e "C:\path\to\sorta[gpu]"    # editable — for local code changes

sorta doctor                            # check what you actually installed (do this first)
cp config.example.yaml config.yaml      # set `sources` and `language`
# exiftool is required for HEIC/RAW/video — install it first (see Requirements)

# Easiest: the web app
sorta ui                                # http://127.0.0.1:8756 → Process a folder → review
sorta-tray                              # ...the same, with an icon in the tray (extra: tray)

# Or the CLI
sorta index /path/to/photos             # scan
sorta run                               # geo, landmarks, junk + near-dup hashes (city+dupes)
sorta run --faces --events              # ...also detect faces and build events
sorta sort --by city --dest /path/to/sorted            # dry-run plan (CSV + HTML)
sorta sort --by city --dest /path/to/sorted --copy --apply   # apply (copy = non-destructive)
sorta undo                              # reverse the last batch if needed
```

`sorta-tray` (F207) is the same web app with an icon in the notification area — for a
Sorta started from a shortcut rather than from a terminal. Double-click the icon or pick
**Open** to open the window, **Quit** to close the program (it asks first if a run is
going). Add the `tray` extra for it (`…sorta[gpu,tray]`); without the extra, or on a
desktop that has no tray, it still serves — just with no icon.

> **`uv tool install` has no `--extra` flag** — the extra belongs in the quoted
> package spec, as above. Installed without it, you silently get the **CPU** profile
> (`torch==2.13.0+cpu`) on a GPU machine. `sorta doctor` is how you catch that: it
> prints the torch build, the onnxruntime providers, the geo database, and the log
> and preview‑cache paths.

Developing on the code instead? Use a project venv (`uv sync --extra gpu --extra
dev`, activate it, then run the same `sorta …` commands with live edits) — see
[Installation](docs/guide/user-guide.en.md#3-installation) in the user guide for
every install variant, `sorta doctor` output explained, the `onnxruntime`/
`onnxruntime-gpu` trap, and why a bare `uv run sorta …` isn't one of the paths.

Two things worth knowing before a big run: Sorta keeps a **preview cache** (one
decode per frame, ≈150 KB per photo — gigabytes on a large collection; inspect with
`sorta cache`, clear with `sorta cache --clear`) and writes a **run log** with
per‑stage timings to `%LOCALAPPDATA%\sorta\logs\sorta.log`
(`~/.cache/sorta/logs/sorta.log` elsewhere). Details in the user guide.

Full walkthrough (with real command output), command reference and config reference
are in the [user guide](docs/guide/user-guide.en.md).

---

## Safety & privacy

- **Originals are never modified.** Sorting moves/copies files; EXIF is not rewritten.
- **Dry‑run by default;** every operation is journaled before it runs; `sorta undo`
  reverses it.
- **No image ever leaves your machine.** Not "off by default" — there is no code
  left that sends one. The cloud event‑naming provider was removed together with its
  upload, and a test keeps it from returning quietly. The single outbound path is
  optional Nominatim geocoding, which receives **rounded coordinates** and never a
  photograph — see [SECURITY.md](SECURITY.md).
- **Documents** (passports, receipts, medical papers…) are collected into a local
  `_Documents/` review folder and processed only on your machine.
- The web app binds to `127.0.0.1` only.

See [SECURITY.md](SECURITY.md) for details.

---

## Documentation

- **[User guide](docs/guide/user-guide.en.md)** — install, config, workflows,
  command & config reference, troubleshooting (EN / RU / JA)
- `docs/ARCHITECTURE.md` — architecture, module ownership, data contracts
- `CONTRIBUTING.md` — how to contribute · `SECURITY.md` — privacy & reporting ·
  `NOTICE` — third‑party data attribution (GeoNames, OpenStreetMap/Nominatim)

---

## Development

```bash
uv sync --extra cpu --extra dev         # or --extra gpu
uv run python scripts/check.py          # gates: ruff + mypy + pytest (with coverage)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup and quality-gate details.

## License

MIT — see [LICENSE](LICENSE). Bundled/queried third‑party geo data has its own
attribution requirements — see [NOTICE](NOTICE).
