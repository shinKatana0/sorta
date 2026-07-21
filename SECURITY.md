# Security & Privacy

Sorta processes personal photos and videos — including, potentially, images of
**identity and financial documents**. Privacy and data safety are core design goals.

## Data handling principles

- **Local by default.** All machine‑learning models (face detection/embeddings,
  scene/landmark classification, text/OCR detection) run **on your machine**. Sorta
  does not upload your images.
- **Opt‑in online providers only.** The only network features are explicitly enabled
  in `config.yaml`:
  - `geo.provider: online` — reverse geocoding via Nominatim/OSM. It sends **GPS
    coordinates**, never images. Off by default (`offline` uses bundled data).
  - `naming.provider: claude` — event naming via the Claude API. **This is the one
    feature that does send images**: up to `naming.max_samples` (default 4) sample
    photos from an event, so the model can describe what's happening in them. Off
    by default (`template` generates names locally, no network; `local_vlm` also
    sends sample images, but to a **local** endpoint you control, e.g. Ollama —
    not a cloud service).
  Keep these off for maximum privacy; `naming.provider: claude` is the only opt-in
  that leaves your machine with actual photo content, and only the small event
  samples you configured, not your whole library.
- **Originals are never modified.** Sorting moves/copies files and never rewrites
  EXIF. With `--copy`/`--link`, originals stay exactly where they are.
- **Documents are collected locally.** Detected documents go to a local
  `_Documents/` review folder for you to handle. They are processed only on your
  machine.
- **Local web app.** `sorta ui` binds to `127.0.0.1` only and is not reachable from
  the network.

## Your responsibilities

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
