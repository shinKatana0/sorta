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

> ⚡ **For full‑speed use** — face recognition, the deep VLM tier, or large
> collections — an **NVIDIA GPU (CUDA 13) with ≥ 4 GB VRAM** is recommended
> (**≥ 8 GB** for the VLM tier). Everything still runs on CPU, just noticeably
> slower for those — see [System requirements](#system-requirements).

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

![The Cities tab — the proposed Country / City / Year folder tree, reviewed before anything moves](docs/assets/process.png)

### Two entry points, one job each

The **CLI runs the pipeline** — every stage, `sort`, `album`, `search`, and a per‑run flag
for every option. The **web app is where you decide**: resolving duplicates, correcting a
place, pinning a query, restoring a frame. `sorta dupes` lists duplicates; it does not
resolve them, because resolving them from a command line means choosing without seeing the
frames — and on blind pairs no rule we measured beat chance at that. Everything
reproducible is scriptable; nothing that needs an opinion pretends to be.

---

## System requirements

| | CPU profile (the `cpu` extra) | GPU profile (the `gpu` extra) |
|---|---|---|
| Hardware | Any x86‑64 machine | NVIDIA GPU + driver supporting **CUDA 13** |
| VRAM | n/a | **~3 GB** base + faces (measured on RTX 5090: CLIP ViT‑L 2.0 GB + buffalo_l 0.6 GB) — **≥ 4 GB** comfortable, **≥ 8 GB** for the deep VLM tier (Qwen2.5‑VL‑3B, ~7 GB est.) |
| Faces / CLIP speed | Works, but **slow** (hours on a large, faces/events‑enabled collection) | Fast — measured 2026‑07‑28: 24,196 photos, faces+events+junk ≈ **40 min** without the deep tier; the optional deep VLM tier adds **+122 min**, once |
| Best for | City sorting + duplicates on any machine; smaller collections with faces/events on | Large collections (300 GB+) with faces/events routinely on |

Common to both: Python **3.11–3.14**, [`uv`](https://docs.astral.sh/uv/), and
**`exiftool` on PATH** (required for HEIC/RAW/video metadata — without it Sorta falls
back to Pillow, which only reads JPEG/PNG/TIFF/WEBP and no video). Disk space for the
index (SQLite) and thumbnails scales with collection size; `--copy` sorting needs
roughly ×2 the collection size, `--link` (hardlink) needs almost none.

Timings above are from our own hardware, not a guarantee. Full breakdown, including
RAM/VRAM notes, in the [user guide](docs/guide/user-guide.en.md#2-requirements).

---

## Quick start

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

# Or the CLI
sorta index /path/to/photos             # scan
sorta run                               # geo, landmarks, junk + near-dup hashes (city+dupes)
sorta run --faces --events              # ...also detect faces and build events
sorta sort --by city --dest /path/to/sorted            # dry-run plan (CSV + HTML)
sorta sort --by city --dest /path/to/sorted --copy --apply   # apply (copy = non-destructive)
sorta undo                              # reverse the last batch if needed
```

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
