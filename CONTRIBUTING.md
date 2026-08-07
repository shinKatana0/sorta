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

All changes must pass the gate script before being committed. The gate is split in
two halves by how long they take, and you are expected to run both:

```bash
uv run --extra cpu --extra dev python scripts/check.py --fast  # ~3 s, before committing
uv run --extra cpu --extra dev python scripts/check.py --slow  # the test suite, ~4 min
uv run --extra cpu --extra dev python scripts/check.py         # everything, in order
```

| Invocation | What runs | When |
|---|---|---|
| `--fast` | version sync + ruff + mypy | Before every commit — it takes seconds and catches what makes a diff not worth reading. |
| `--slow` | pytest with coverage | Start it in the background and wait for it; a fast pass is not a green gate on its own. |
| no flags | both, fast half first | What CI runs and what a merge is checked with. |

The slow half runs the suite **twice**: everything under `-n auto` (one pytest worker
per core, whole files at a time), then the handful of tests marked `serial` in a single
process. A test is `serial` when it asserts about elapsed time or runs the real server
on a port — under load those measure the machine rather than the code. **The marker is
not a place to put a test that went red once**: a failure that only happens under
`-n auto` is shared process state until proven otherwise, and it gets fixed rather than
marked. Every use carries its reason next to it, and `tests/test_gate_parallel.py` fails
the gate if one does not. The four-minute figure is what the two passes plus the
coverage report took on a 24-core machine; the run prints its own duration at the end,
so the next time this number goes stale it is visible on the spot.

Pass the same profile you installed with (`cpu` or `gpu`) plus `dev`. A bare
`uv run python scripts/check.py` re‑syncs the environment to the base
dependencies and drops the dev tools — always include the extras.

- **ruff** — linting/formatting.
- **mypy** — static typing.
- **pytest** — tests, with a coverage floor enforced in `pyproject.toml`.

Any half exits non‑zero on the first failed check and says which one it was;
committing is blocked until the run is green.

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
- **Comments and docstrings in English.** The repository is public, so the prose that
  explains *why* the code is what it is has to be readable without Russian. Russian
  stays where it is the subject — a folder name the product creates (`_удалить`), a
  spelling people type for a trip folder (`Тайланд 2023`) — quoted, so it reads as data.
  The exception is the command docstrings in `sorta/cli.py`: Typer prints them as
  `--help`, so they follow the interface language. `tests/test_comments_english.py`
  enforces this.

## Pull requests

1. Keep changes focused; describe what and why.
2. Ensure `scripts/check.py` is green.
3. Add/update tests and docs (including the user guide if behavior changes).

## Release process

The version lives in `pyproject.toml` (source of truth), mirrored in
`sorta/__init__.py` and the top `CHANGELOG.md` entry. Before tagging:

```bash
python scripts/release.py check     # asserts the three versions agree
python scripts/release.py notes     # prints the CHANGELOG section for gh release
```

`.gitignore` is an **allow-list** — everything is ignored unless explicitly
re-included, so a stray file can't be committed by accident. A genuinely new
top-level path needs its own `!/path` entry.

## License

By contributing, you agree that your contributions are licensed under the project's
[LICENSE](LICENSE).
