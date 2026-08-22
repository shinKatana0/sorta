"""F245: what went wrong, as a value — so a screen can say it in its own language.

An exception that has crossed one `except` is a string: `str(exc)` is all that is left of
it, and a string cannot be translated after the fact. The classes this package raises
therefore carry the failure itself alongside the sentence — `code` says WHAT happened and
`params` holds the values the sentence mentions.

The sentence itself does not change and stays English (F238/F239): the log, the terminal
and the file attached to a complaint are the record of a run, and a record nobody but its
author can read is not one. Only the web app renders from `code`, in the language of its
interface; everything it has no key for is shown as it arrived.
"""
from __future__ import annotations


class Fault(Exception):
    """One of our own failures: an English sentence plus the fact behind it.

    Mixed in FIRST, before the built-in this exception is a kind of
    (`class X(Fault, ValueError)`), so that `super().__init__(message)` reaches that
    built-in and `str(exc)`, `args` and `except ValueError` all behave as they did before
    the class carried anything.

    `params` travels to a browser as JSON: strings and numbers belong in it, `Path` and
    exception objects do not — convert at the raise site.
    """

    def __init__(self, message: str, code: str, **params: object) -> None:
        super().__init__(message)
        self.code = code
        self.params = params


def fault_code(exc: BaseException) -> str | None:
    """The code of `exc`, or None for anything not ours (sqlite3, OSError, MemoryError)."""
    return exc.code if isinstance(exc, Fault) else None


def fault_params(exc: BaseException) -> dict[str, object]:
    """The values `exc` names, or an empty dict for anything not ours."""
    return dict(exc.params) if isinstance(exc, Fault) else {}
