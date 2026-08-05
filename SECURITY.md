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
