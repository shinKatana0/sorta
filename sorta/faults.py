"""F245: what went wrong, as a value — so a screen can say it in its own language.

An exception that has crossed one `except` is a string, and a string cannot be
translated after the fact. So the classes this package raises carry the failure itself
next to the sentence: `code` says WHAT happened, `params` holds the values it names.

The sentence stays English (F238/F239) — the log and the terminal are the record of a
run, and a record only its author can read is not one. The web app renders from `code`;
anything it has no key for it shows as it arrived.
"""
from __future__ import annotations


class Fault(Exception):
    """One of our own failures: an English sentence plus the fact behind it.

    Mixed in FIRST, before the built-in this exception is a kind of
    (`class X(Fault, ValueError)`), so `super().__init__(message)` reaches that built-in
    and `str(exc)`, `args` and `except ValueError` behave as they did before.

    `params` travels to a browser as JSON: strings and numbers belong in it, `Path` and
    exception objects do not — convert at the raise site.

    `codes` is every code the class can raise with. It is what the guard asks a class
    for: the raise sites are spread over a module and one of them builds its code from a
    value (`EmbeddingsMissing`), so they cannot be read off the source. Adding a code
    here is what turns into a demand for three translations.
    """

    codes: tuple[str, ...] = ()

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
