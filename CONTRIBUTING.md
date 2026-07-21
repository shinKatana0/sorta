# Contributing to Sorta

Thanks for your interest in improving Sorta! This document explains how to set up a
dev environment, the quality bar, and how the project is organized.

## Development setup

```bash
git clone https://github.com/shinKatana0/sorta.git
cd sorta
uv sync --extra cpu --extra dev  # no NVIDIA GPU
# or
uv sync --extra gpu --extra dev  # NVIDIA GPU + CUDA 13 driver
```

`cpu`/`gpu` are mutually exclusive install profiles for the ML backend
(torch/onnxruntime) — see the [user guide](docs/guide/user-guide.en.md#2-requirements)
for which one fits your machine. Always pass one of them explicitly:
plain `uv sync` (no extras) does not reliably resolve a consistent
torch/onnxruntime pair. `--extra dev` adds the dev tools (ruff, mypy,
pytest) on top and is required for the commands below.

## Quality gates

All changes must pass the gate script before being committed:

```bash
uv run --extra cpu --extra dev python scripts/check.py  # or --extra gpu
# ruff (lint) + mypy (types) + pytest (with coverage)
```

Pass the same profile you installed with (`cpu` or `gpu`) plus `dev`. A bare
`uv run python scripts/check.py` re‑syncs the environment to the base
dependencies and drops the dev tools — always include the extras.

- **ruff** — linting/formatting.
- **mypy** — static typing.
- **pytest** — tests, with a coverage floor enforced in `pyproject.toml`.

Tests must not touch a real photo collection — use `tmp_path` and synthetic fixtures
for filesystem operations. ML‑heavy paths (faces, CLIP, OCR) are **mocked** in tests
(no model downloads in CI).

## Project layout

- `sorta/` — the package, organized by layer:
  `indexer`, `geo`, `geodata`, `faces`, `events`, `sorter`, `junk`, `landmarks`,
  `dedup`, `imaging`, `ui`, `db`, `config`, `cli`, `i18n`.
- `tests/` — pytest suite.
- `docs/guide/` — user guide (EN/RU/JA).
- `docs/ARCHITECTURE.md` — architecture, module ownership, and data contracts.

## Conventions

- **Config, not constants.** Thresholds (face size, clustering, event gaps, CLIP/OCR
  thresholds) live in `config.yaml` / `config.py`, not hardcoded.
- **Safety first.** Anything that moves/copies files defaults to dry‑run, journals
  before acting, verifies hashes, and supports `undo`. Never overwrite an existing
  file (suffix `_1`, `_2`).
- **Local by default.** Cloud/online calls are opt‑in via config; never send images
  off the machine implicitly.
- **Incremental.** Long stages should reprocess only new/changed files.
- **i18n.** User‑facing folder names and the web UI support ru/en/ja; keep new
  strings translated.

## Pull requests

1. Keep changes focused; describe what and why.
2. Ensure `scripts/check.py` is green.
3. Add/update tests and docs (including the user guide if behavior changes).

## License

By contributing, you agree that your contributions are licensed under the project's
[LICENSE](LICENSE).
