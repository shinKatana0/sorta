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
