"""The progress helper and the step order of the `sorta run` pipeline."""
import unittest

from sorta.progress import TaskProgress, progress_task


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


class TestPhaseChannel(unittest.TestCase):
    """F84: a step with internal phases relabels its own bar."""

    def test_description_gets_the_caption_of_the_phase(self):
        seen: list[dict] = []
        cb = TaskProgress("faces", lambda **fields: seen.append(fields),
                          {"cluster_read": "кластеры: чтение"})
        cb(2, 5)
        cb.phase("cluster_read")
        cb.phase("who_is_this")  # an unknown key is shown as-is, not swallowed
        self.assertEqual(seen[0], {"completed": 2, "total": 5})
        self.assertEqual(seen[1]["description"], "faces · кластеры: чтение")
        self.assertEqual(seen[2]["description"], "faces · who_is_this")

    def test_quiet_phase_is_a_noop(self):
        with progress_task("шаг", quiet=True, phase_labels={"a": "фаза"}) as cb:
            cb.phase("a")  # no bar to relabel — must not crash
            cb(1, 2)

    def test_active_bar_accepts_phases(self):
        with progress_task("шаг", quiet=False, phase_labels={"a": "фаза"}) as cb:
            cb(0, 10)
            cb.phase("a")
            cb(5, None)

    def test_cli_labels_cover_every_cluster_phase(self):
        # A phase without a caption would show a raw identifier next to the bar.
        from sorta import faces
        from sorta.cli import _CLUSTER_PHASE_LABELS
        keys = {value for name, value in vars(faces).items()
                if name.startswith("CLUSTER_PHASE_")}
        self.assertEqual(set(_CLUSTER_PHASE_LABELS), keys)


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
