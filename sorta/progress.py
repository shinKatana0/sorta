"""Progress for long operations: rich bars and a wrapper for the `sorta run` pipeline.

A step callback has the form `progress(done, total)`; `total=None` — the total is
not yet known (spinner + counter). Outside a tty (pipe/log file) or with
`quiet=True`, a no-op callback is returned so rich control codes do not clutter logs.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

ProgressCB = Callable[[int, Optional[int]], None]


def _noop(done: int, total: Optional[int]) -> None:
    pass


@contextmanager
def progress_task(description: str, *, quiet: Optional[bool] = None) -> Iterator[ProgressCB]:
    """A context with a rich bar; yields the callback `progress(done, total)`.

    `quiet=None` (default) → auto: quiet if stdout is not a tty (pipe/log).
    """
    if quiet is None:
        quiet = not sys.stdout.isatty()
    if quiet:
        yield _noop
        return
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:  # pragma: no cover — rich is in the dependencies
        yield _noop
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as prog:
        task_id = prog.add_task(description, total=None)

        def cb(done: int, total: Optional[int]) -> None:
            prog.update(task_id, completed=done, total=total)

        yield cb
