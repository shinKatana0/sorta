"""F245: the run screen speaks the interface's language; the log stays English.

Three things are held here, and the order is the order of the error's journey.

1. A failure of ours carries WHAT happened (`faults.Fault`) and still says it in the
   same English sentence it said before: `str(exc)`, `args` and the built-in class it is
   a kind of are exactly what the terminal, the log and every existing `except` saw.
2. Every class of ours has a key in all three languages, and the page renders that key
   instead of the sentence. The classes are FOUND (F239) — a seventh one added tomorrow
   is red until it is given a key.
3. The reverse, and the one that matters more: with the interface in `ru` or `ja` the
   file `sorta.log` still receives English. A feature that "helps" by translating the
   log is caught here rather than a month later, by the log attached to a complaint.
"""
from __future__ import annotations

import unittest

from sorta import faults


class TestAFaultIsStillTheExceptionItWas(unittest.TestCase):
    """The English text does not move. Everything else is additive."""

    def test_str_and_args_are_the_message_alone(self):
        class Refusal(faults.Fault, RuntimeError):
            pass

        exc = Refusal("no index at /tmp/photos.db.", "relocate_no_index",
                      path="/tmp/photos.db")
        self.assertEqual(str(exc), "no index at /tmp/photos.db.")
        self.assertEqual(exc.args, ("no index at /tmp/photos.db.",))

    def test_the_builtin_it_is_a_kind_of_still_catches_it(self):
        class Unsafe(faults.Fault, ValueError):
            pass

        with self.assertRaises(ValueError):
            raise Unsafe("path must be absolute", "exif_relative_path", path="x")

    def test_the_code_and_the_params_travel_with_it(self):
        class Missing(faults.Fault, FileNotFoundError):
            pass

        exc = Missing("places.tsv is not at /x", "geo_data_missing", path="/x")
        self.assertEqual(faults.fault_code(exc), "geo_data_missing")
        self.assertEqual(faults.fault_params(exc), {"path": "/x"})

    def test_someone_elses_exception_has_neither(self):
        """`sqlite3.OperationalError`, `OSError`, `MemoryError` — the page may not
        pretend to have translated one of those."""
        self.assertIsNone(faults.fault_code(OSError("disk gone")))
        self.assertEqual(faults.fault_params(OSError("disk gone")), {})


if __name__ == "__main__":
    unittest.main()
