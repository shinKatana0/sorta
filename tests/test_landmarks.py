"""F64: the size of the CLIP decode pool (config override / auto default by cpu count).

`_decode_pool_size` is a pure function — tested directly, without touching
`clip_classifier` (that one loads open_clip: ML, no-cover).
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from sorta.landmarks import _decode_pool_size


def _settings(**kwargs):
    """A lightweight settings object — only the fields the helper reads."""
    return SimpleNamespace(**kwargs)


class TestDecodePoolSize(unittest.TestCase):
    def _with_cpus(self, n):
        return mock.patch("sorta.landmarks.os.cpu_count", return_value=n)

    def test_override_wins_over_cpu_count(self):
        with self._with_cpus(4):
            self.assertEqual(_decode_pool_size(_settings(clip_decode_workers=20)), 20)

    def test_override_above_cap_is_respected(self):
        # the cap is only for the auto default — an explicit choice is the user's
        with self._with_cpus(24):
            self.assertEqual(_decode_pool_size(_settings(clip_decode_workers=32)), 32)

    def test_zero_override_means_auto(self):
        with self._with_cpus(24):
            self.assertEqual(_decode_pool_size(_settings(clip_decode_workers=0)), 16)

    def test_missing_attribute_means_auto(self):
        # the config field is optional: settings without it must still work
        with self._with_cpus(24):
            self.assertEqual(_decode_pool_size(_settings()), 16)

    def test_auto_is_capped_at_16(self):
        with self._with_cpus(32):
            self.assertEqual(_decode_pool_size(_settings()), 16)

    def test_auto_follows_cpu_count_below_the_cap(self):
        with self._with_cpus(4):
            self.assertEqual(_decode_pool_size(_settings()), 4)

    def test_auto_falls_back_when_cpu_count_unknown(self):
        with self._with_cpus(None):
            self.assertEqual(_decode_pool_size(_settings()), 4)

    def test_never_below_one(self):
        for cpus in (None, 0, 1, 4, 24, 32):
            with self._with_cpus(cpus):
                self.assertGreaterEqual(_decode_pool_size(_settings()), 1)
                self.assertGreaterEqual(
                    _decode_pool_size(_settings(clip_decode_workers=0)), 1)


if __name__ == "__main__":
    unittest.main()
