"""The progress helper and the step order of the `sorta run` pipeline."""
import unittest

from sorta.progress import progress_task


class TestProgressTask(unittest.TestCase):
    def test_quiet_yields_working_noop(self):
        with progress_task("шаг", quiet=True) as cb:
            self.assertTrue(callable(cb))
            cb(0, 100)      # must not crash
            cb(50, 100)
            cb(1, None)     # unknown total

    def test_active_bar_updates_without_error(self):
        # rich is in the dependencies; force non-quiet — updates must not crash
        with progress_task("шаг", quiet=False) as cb:
            cb(0, 10)
            cb(5, 10)
            cb(10, 10)


class TestPipelineSteps(unittest.TestCase):
    def test_order_and_dependencies(self):
        from sorta.cli import _pipeline_steps
        names = [name for name, _fn in _pipeline_steps()]
        self.assertEqual(names, ["index", "geo", "landmarks", "faces", "events", "junk"])
        # dependency invariants
        self.assertLess(names.index("geo"), names.index("landmarks"))
        self.assertLess(names.index("faces"), names.index("junk"))


if __name__ == "__main__":
    unittest.main()
