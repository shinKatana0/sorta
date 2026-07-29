"""Shared pytest configuration for the test suite.

On GitHub Actions Windows runners the `%TEMP%` variable points to an 8.3-short
path (`C:\\Users\\RUNNER~1\\...`), whereas the code under test stores paths via
`Path(...).resolve()` — the long canonical form (`...\\runneradmin\\...`).
Because of this, comparisons of "a path from the code" vs "a path assembled from a
tempfile fixture" diverge only on CI. We canonicalize the base temp directory once at
conftest load (before test collection) — every `tempfile.TemporaryDirectory()`/`mkdtemp()`
gets the already-resolved form. On Linux/macOS this is a no-op (realpath does not change the path).
"""
import os
import tempfile

tempfile.tempdir = os.path.realpath(tempfile.gettempdir())

# The suite must not touch the user's real state. Both of these default to
# %LOCALAPPDATA%\sorta\... and are reached implicitly: configure_logging() attaches
# the run log, and decode_rgb_preview() is now on the thumb/CLIP/OCR paths. Caught in
# practice — a test run polluted a live sorta.log so thoroughly that the pipeline's own
# stage lines were unreadable, and previews of real photos (including document-verdict
# ones) had been written into the user's cache. Setting these before collection covers
# every test, including ones that never think about logging or previews.
_SANDBOX = os.path.join(tempfile.gettempdir(), "sorta-tests")
os.environ["SORTA_LOG_FILE"] = os.path.join(_SANDBOX, "logs", "sorta.log")
os.environ["SORTA_PREVIEW_DIR"] = os.path.join(_SANDBOX, "previews")

# Typer decides once, at import of typer.rich_utils, whether to style `--help`:
# `FORCE_TERMINAL = True if getenv("GITHUB_ACTIONS") ...`. So on CI — and only there —
# rich wraps every option and every `--flag` inside a docstring in ANSI escapes, and
# tests that look for a plain substring in `result.output` fail on text that is
# actually present ("--rescan" arrives as "\x1b[1m-\x1b[0m\x1b[1m-rescan\x1b[0m").
# What those tests are about is the wording of the help, not how a terminal paints it,
# so we turn the styling off for the whole run through typer's own switch. It must be
# set before typer is imported, hence conftest at collection time rather than a fixture.
# The switch itself is verified by an assertion on real output —
# tests/test_comments_english.py::test_the_help_output_carries_no_terminal_styling.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
