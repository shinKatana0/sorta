"""The entry point a shortcut runs: the window first, the program second.

Importing `sorta.tray` costs 3.59 s of `sorta.ui` alone (payload interpreter, warm cache),
and a window opened inside it cannot be on the screen during that. This module imports
stdlib only, puts the window up, and then imports the program.
"""
from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from .splash import open_splash

    splash = None if "--no-splash" in (argv if argv is not None else sys.argv[1:]) \
        else open_splash()
    try:
        from .tray import main as tray_main
    except BaseException:
        if splash is not None:
            splash.close()
        raise
    return tray_main(argv, splash=splash)


if __name__ == "__main__":  # pragma: no cover — the shortcut and the console script
    sys.exit(main())
