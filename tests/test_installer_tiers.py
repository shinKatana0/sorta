"""F211: the watchdog — the installer's tiers against the extras of `pyproject.toml`.

The same trick F115 used on the config keys, and it has already paid for itself once: an
extra added to the project and forgotten everywhere else is invisible until somebody
notices the feature is not installable. Here it fails the gate.

Both directions are guarded, and they fail differently:

* an extra the project declares and no tier carries — the wizard would never offer it,
  and whoever added it would find out from a user;
* an extra a tier names that the project does not have any more — the installer
  promising something `uv` will refuse to resolve.

The catalog itself is checked for the things a person actually reads: every tier has a
size, the base one is not optional and carries the profile the shortcut needs, and the
CUDA index is the one the project resolves torch from.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from sorta import wizard

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_installer.py"


def _load_script():
    """Import scripts/build_installer.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("build_installer", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_script()
PYPROJECT = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


class TestEveryExtraIsAccountedFor(unittest.TestCase):
    """The guard itself, on the real project."""

    def test_no_extra_of_the_project_is_missing_from_the_installer(self):
        missing, stale = builder.unaccounted_extras(PYPROJECT)
        self.assertEqual(missing, set(), "extras the installer never heard of")
        self.assertEqual(stale, set(), "extras the installer names and the project lacks")

    def test_the_guard_notices_an_extra_added_past_the_installer(self):
        """The case this exists for, written as the change that must fail it."""
        with_new_extra = PYPROJECT.replace(
            '[project.optional-dependencies]\n',
            '[project.optional-dependencies]\nocr = ["something>=1.0"]\n', 1)
        missing, stale = builder.unaccounted_extras(with_new_extra)
        self.assertEqual(missing, {"ocr"})
        self.assertEqual(stale, set())

    def test_the_guard_notices_an_extra_the_project_dropped(self):
        without_vlm = PYPROJECT.replace(
            'vlm = ["transformers>=4.49,<4.52", "accelerate>=0.34", '
            '"qwen-vl-utils>=0.0.8"]\n', "", 1)
        missing, stale = builder.unaccounted_extras(without_vlm)
        self.assertEqual(missing, set())
        self.assertEqual(stale, {"vlm"})

    def test_the_build_refuses_to_produce_an_installer_that_is_out_of_date(self):
        """The watchdog is not only a test: the build runs it too, so a stale tier list
        cannot become a released file even if nobody ran the suite."""
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn("unaccounted_extras(pyproject)", source)

    def test_an_extra_that_is_deliberately_not_shipped_says_why(self):
        """`dev` is the one, and the reason is the point: a name on a list with no
        sentence behind it is how the next one gets added by reflex."""
        self.assertIn("dev", wizard.NOT_SHIPPED)
        for extra, reason in wizard.NOT_SHIPPED.items():
            with self.subTest(extra=extra):
                self.assertGreater(len(reason.split()), 5, extra)
                self.assertNotIn(extra, {e for t in wizard.TIERS for e in t.extras})


class TestTheTierCatalog(unittest.TestCase):
    """What the tiers claim, checked against the project rather than against a memory."""

    def test_the_base_tier_is_not_optional_and_carries_what_the_shortcut_needs(self):
        base = wizard.BASE_TIER
        self.assertFalse(base.optional)
        self.assertNotIn(base, wizard.OPTIONAL_TIERS)
        # `cpu` is the hardware profile the installer can carry offline; `tray` is what
        # the desktop shortcut starts (F207), so a base tier without it is a shortcut
        # to an icon that never appears.
        self.assertEqual(set(base.extras), {"cpu", "tray"})

    def test_the_profiles_stay_mutually_exclusive(self):
        """`cpu` and `gpu` conflict in pyproject.toml, so they may never be one tier."""
        for tier in wizard.TIERS:
            with self.subTest(tier=tier.key):
                self.assertFalse({"cpu", "gpu"} <= set(tier.extras))

    def test_every_tier_key_is_unique_and_reachable(self):
        keys = [tier.key for tier in wizard.TIERS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(wizard.TIERS_BY_KEY), set(keys))
        self.assertEqual(wizard.tier_keys(), tuple(keys))

    def test_every_optional_tier_costs_something_and_says_what(self):
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertGreater(tier.download_mb, 0)
                self.assertTrue(tier.extras or tier.weights,
                                f"{tier.key}: a tier that adds nothing")

    def test_the_deep_tier_is_the_vlm_extra_and_the_qwen_weights(self):
        deep = wizard.TIERS_BY_KEY["deep"]
        self.assertEqual(deep.extras, ("vlm",))
        self.assertTrue(any("Qwen" in weight for weight in deep.weights))

    def test_the_cuda_tier_names_the_index_the_project_resolves_torch_from(self):
        self.assertIn(f'url = "{wizard.PYTORCH_CU130_INDEX}"', PYPROJECT)
        self.assertEqual(wizard.TIERS_BY_KEY["gpu"].index_url, wizard.PYTORCH_CU130_INDEX)

    def test_the_shipped_interpreter_is_one_the_project_supports(self):
        """`requires-python` is held by the CUDA wheels (F55), so the version the
        installer carries has to sit inside it rather than next to it."""
        import re

        bounds = re.search(r'requires-python\s*=\s*">=([\d.]+),<([\d.]+)"', PYPROJECT)
        self.assertIsNotNone(bounds)
        low = tuple(int(part) for part in bounds.group(1).split("."))
        high = tuple(int(part) for part in bounds.group(2).split("."))
        shipped = tuple(int(part) for part in builder.PYTHON_VERSION.split("."))
        self.assertLessEqual(low[:len(shipped)], shipped)
        self.assertLess(shipped, high[:len(shipped)])


if __name__ == "__main__":
    unittest.main()
