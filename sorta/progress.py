"""Progress for long operations: rich bars and a wrapper for the `sorta run` pipeline.

A step callback has the form `progress(done, total)`; `total=None` — the total is
not yet known (spinner + counter). Outside a tty (pipe/log file) or with
`quiet=True`, a no-op callback is returned so rich control codes do not clutter logs.

A step made of several internal phases (F84: clustering inside `faces`) additionally
reports the phase KEY through `progress.phase(name)` — the caption is the caller's
business (`phase_labels` here, the localized strings of the served UI).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping, Optional

ProgressCB = Callable[[int, Optional[int]], None]
PhaseCB = Callable[[str], None]


class TaskProgress:
    """What `progress_task` yields: a `(done, total)` callback with a `phase` channel.

    A step that knows nothing about phases just calls the object — the previous
    contract, unchanged. A step that does relabels the bar, so a long phase without a
    percent (one blocking HDBSCAN call) at least says what it is busy with. `update=
    None` — quiet mode: every call is a no-op.
    """

    def __init__(self, description: str, update: Callable[..., None] | None,
                 labels: Mapping[str, str] | None = None) -> None:
        self._description = description
        self._update = update
        self._labels = dict(labels or {})

    def __call__(self, done: int, total: Optional[int] = None) -> None:
        if self._update is not None:
            self._update(completed=done, total=total)

    def phase(self, name: Optional[str]) -> None:
        """Append the caption of phase `name` to the bar's description.

        An unknown key is shown as-is: a raw identifier next to the bar is still
        better than a bar that stands there saying nothing.

        F229: `None` — no phase any more, the description goes back to the bare name of
        the step. The same meaning `_ProcessState.set_phase(None)` has always had on the
        other screen; a caption that can be put up and never taken down would leave
        "waiting for the model" standing over the frames the stage went on to count.
        """
        if self._update is None:
            return
        if name is None:
            self._update(description=self._description)
            return
        self._update(description=f"{self._description} · {self._labels.get(name, name)}")


@contextmanager
def progress_task(description: str, *, quiet: Optional[bool] = None,
                  phase_labels: Mapping[str, str] | None = None) -> Iterator[TaskProgress]:
    """A context with a rich bar; yields the callback `progress(done, total)`.

    `quiet=None` (default) → auto: quiet if stdout is not a tty (pipe/log).
    `phase_labels` — captions for the phase keys a step may report (see TaskProgress).
    """
    if quiet is None:
        quiet = not sys.stdout.isatty()
    if quiet:
        yield TaskProgress(description, None, phase_labels)
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
        yield TaskProgress(description, None, phase_labels)
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

        def update(**fields: object) -> None:
            prog.update(task_id, **fields)  # type: ignore[arg-type]

        yield TaskProgress(description, update, phase_labels)
