# Sorta

> Languages: **English** · [Русский](README.ru.md) · [日本語](README.ja.md)

**Index and sort a large photo/video collection** (60 GB+ tested, 300 GB+ by design)
into a clean folder structure — by **city/country**, **person**, or **event** — with
safety first: dry‑run by default, a move journal, and one‑command undo.

Sorta runs **locally** (ML models for faces, scenes and text run on your machine),
**never modifies your originals**, and offers both a **CLI** and a guided **local web
app**.

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
- **Fast basic run by default:** `sorta run` / the UI **Process** button do city‑level
  sorting + duplicate detection (index → geo → landmarks → junk → near‑dup hashes).
  **Faces and events are opt‑in** (`--faces`/`--events`, or the matching checkboxes) —
  they're the slowest stages and not everyone needs them.
- **Faces & people:** local detection + clustering (insightface), once enabled; name
  and merge clusters, then sort or build per‑person albums.
- **Events:** time‑gap + city clustering with localized names, once enabled; manual
  events too.
- **Duplicates:** exact (blake3) and near‑duplicate (perceptual hash) with a
  batch‑review UI.
- **Junk & documents:** screenshots/memes routed out; documents collected into a
  `_Documents/` review folder (CLIP + text‑density).
- **Albums:** collect a person/event slice into a named folder via **hardlinks**
  (near‑zero extra space), copy, or move.
- **Local web app** (`sorta ui`): process a folder, review the plan, resolve
  duplicates, name people, and materialize sorts/albums — all in the browser. The
  **People**/**Events** tabs only appear once you've actually run those stages.
- **Trilingual** UI and folder names: **ru / en / ja**.
- **Safe by design:** dry‑run, journal + `undo`, blake3 verification, never
  overwrites (suffix `_1`, `_2`).

---

## System requirements

| | CPU profile (`--extra cpu`) | GPU profile (`--extra gpu`) |
|---|---|---|
| Hardware | Any x86‑64 machine | NVIDIA GPU + driver supporting **CUDA 13** |
| VRAM | n/a | **~3 GB** base + faces (measured on RTX 5090: CLIP ViT‑L 2.0 GB + buffalo_l 0.6 GB) — **≥ 4 GB** comfortable, **≥ 8 GB** for the deep VLM tier (Qwen2.5‑VL‑3B, ~7 GB est.) |
| Faces / CLIP speed | Works, but **slow** (hours on a large, faces/events‑enabled collection) | Fast — reference: 6,298 photos, faces+events+junk ≈ **45 min** (fast tier) / ≈ **77 min** with the optional deep VLM tier |
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
# Install once — pick the profile that matches your hardware
uv tool install ".[cpu]"                # no NVIDIA GPU — puts `sorta` on PATH
# or
uv tool install ".[gpu]"                # NVIDIA GPU + CUDA 13 driver
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

Developing on the code instead? Use a project venv (`uv sync --extra cpu --extra
dev`, activate it, then run the same `sorta …` commands with live edits) — see
[Installation](docs/guide/user-guide.en.md#3-installation) in the user guide for
both set‑once paths, how to switch profiles, and why a bare `uv run sorta …` isn't
one of them.

Full walkthrough (with real command output), command reference and config reference
are in the [user guide](docs/guide/user-guide.en.md).

---

## Safety & privacy

- **Originals are never modified.** Sorting moves/copies files; EXIF is not rewritten.
- **Dry‑run by default;** every operation is journaled before it runs; `sorta undo`
  reverses it.
- **Local by default.** All ML runs on your machine. Online providers (Nominatim
  geocoding: GPS coordinates only; Claude API event naming: a handful of sample
  photos per event, only if you opt in) are off by default — see
  [SECURITY.md](SECURITY.md) for exactly what each one sends.
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
