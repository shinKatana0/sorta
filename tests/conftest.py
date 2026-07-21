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
