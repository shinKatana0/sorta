# Security & Privacy

Sorta processes personal photos and videos — including, potentially, images of
**identity and financial documents**. Privacy and data safety are core design goals.

## Data handling principles

- **Local by default.** All machine‑learning models (face detection/embeddings,
  scene/landmark classification, text/OCR detection) run **on your machine**. Sorta
  does not upload your images.
- **No image ever leaves your machine — by construction, not by default.** There is
  no code left in Sorta that sends a picture anywhere. The cloud event‑naming
  provider (`naming.provider: claude`) was **removed** along with the upload it
  performed, and a test refuses to let that string come back unnoticed. Event names
  are produced by a local template or by a local VLM.
- **One outbound path, and it carries coordinates.** `geo.provider: online` looks a
  place up through Nominatim/OSM: it sends **rounded GPS coordinates**, never an
  image, and the key says so in its name. Off by default — `offline` uses the
  GeoNames data bundled with the package and needs no network at all.
- **The program does fetch its own model weights, and nothing of yours travels with
  them.** The first run of a tier that needs a model downloads it — CLIP and the VLM
  from `huggingface.co`, the face model from insightface's own storage — and every run
  after that is offline, because the weights are cached on disk. A request for a file
  carries no photograph, no path and no metadata. A run says what it is about to
  fetch, and how large it is, before it starts. To keep even that from happening, let
  the Windows installer's wizard fetch them once during the install, or set
  `HF_HUB_OFFLINE=1` — a setting Sorta never overrides, and under which a missing
  model is an error rather than a download.
- **Originals are never modified.** Sorting moves/copies files and never rewrites
  EXIF. With `--copy`/`--link`, originals stay exactly where they are.
- **Deleting is the recycle bin, not `unlink`.** The review screens hand a file to
  `send2trash`, so the operating system holds it until the bin is emptied. `sorta
  undo` is the move journal played backwards and covers moves: a trashed file comes
  back from the bin, not from `undo`.
- **Documents are collected locally.** Detected documents go to a local
  `_Documents/` review folder for you to handle. They are processed only on your
  machine.
- **Local web app.** `sorta ui` binds to `127.0.0.1` only and is not reachable from
  the network.

## Your responsibilities

- **Have a backup of the collection before the first `--apply`.** Dry‑run, the
  journal, `undo`, blake3 verification and the recycle bin are a design meant to make
  one unnecessary — they are not a guarantee, and no reading of this document should
  be taken as one. This is the advice we would give about any tool that moves files.
- The generated index (`sorta.db`) and any HTML plans/thumbnails contain metadata and
  derived thumbnails of your photos. Store them where you'd store the photos
  themselves, and exclude them from any accidental sharing.
- If you enable an online provider, review that provider's terms and privacy policy.
- Sorta does not encrypt your files or the index; use OS‑level protections as needed.

## Reporting a vulnerability

If you discover a security or privacy issue, please report it **privately** rather
than opening a public issue:

- Open a private security advisory on the repository (GitHub → *Security* →
  *Report a vulnerability*).

Please include steps to reproduce and the affected version/commit. We aim to
acknowledge reports promptly and coordinate a fix and disclosure.

## Scope

This project is a local tool with no server component beyond the opt‑in localhost web
app. There is no user account system, telemetry, or remote data storage in Sorta
itself.
