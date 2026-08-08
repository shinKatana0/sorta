"""F216: `sorta doctor` names the tiers that are actually on this machine.

The installer ships one tier and offers four, so "which of them are here" is the first
question of the person who installed it by hand AND of the workflow that installs it on
a clean machine — and until now answering it meant reading a directory listing.

What is under test is the ANSWER, not the machine the suite runs on: every probe is
injected, because a test that passed only where buffalo_l happened to be cached would be
the kind of check this whole feature exists to argue against.
"""
from __future__ import annotations

import tempfile
import unittest
from collections.abc import Collection
from pathlib import Path
from unittest.mock import patch

from sorta import cli, i18n, install, wizard

_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")


def _states(*, packages: Collection[str] = (),
            weights: Collection[str] = ()) -> list[cli.TierState]:
    """The tier states of a machine that has exactly these packages and weights."""
    return cli.tier_states(package_present=lambda name: name in packages,
                           weights_cached=lambda name: name in weights)


def _by_key(states: list[cli.TierState]) -> dict[str, cli.TierState]:
    return {state.key: state for state in states}


class TestTheStateOfATierIsTwoIndependentHalves(unittest.TestCase):
    """Packages and weights fail differently, so they are reported apart — the same
    distinction the wizard draws when it says "chosen" rather than "installed"."""

    def test_every_tier_of_the_catalog_is_answered_for(self):
        self.assertEqual([state.key for state in _states()],
                         [tier.key for tier in wizard.TIERS])

    def test_a_tier_whose_weights_are_not_downloaded_is_not_missing(self):
        faces = _by_key(_states())["faces"]
        self.assertEqual(faces.missing_weights, ("buffalo_l",))
        self.assertEqual(faces.missing_packages, ())
        self.assertFalse(faces.ready)

    def test_a_tier_with_its_weights_on_disk_is_ready(self):
        self.assertTrue(_by_key(_states(weights={"buffalo_l"}))["faces"].ready)

    def test_a_tier_names_the_packages_it_is_missing(self):
        """By distribution name — what `uv pip install` would be asked for, which is
        also what a person types when they repair it by hand."""
        deep = _by_key(_states(weights={"Qwen2.5-VL-3B"}))["deep"]
        self.assertIn("transformers", deep.missing_packages)
        self.assertIn("accelerate", deep.missing_packages)
        self.assertEqual(deep.missing_weights, ())

    def test_a_version_bound_is_not_part_of_the_name(self):
        for requirement, expected in (("onnxruntime>=1.27.0", "onnxruntime"),
                                      ("torch >= 2.10.0", "torch"),
                                      ("qwen-vl-utils>=0.0.8 ; extra == 'vlm'",
                                       "qwen-vl-utils"),
                                      ("uvicorn[standard]", "uvicorn")):
            with self.subTest(requirement=requirement):
                self.assertEqual(cli._distribution_name(requirement), expected)

    def test_the_base_tier_is_ready_when_its_packages_are_installed(self):
        installed = {cli._distribution_name(requirement)
                     for requirement in wizard.tier_requirements(wizard.BASE_TIER)}
        self.assertTrue(installed, "the suite runs against installed metadata")
        self.assertTrue(_by_key(_states(packages=installed))["base"].ready)


class TestNothingIsClaimedThatWasNotChecked(unittest.TestCase):
    """The failure this feature is against: a green line that verified nothing."""

    def test_without_metadata_a_tier_with_extras_is_not_called_present(self):
        """A source directory that was never installed has no requirements to read. The
        honest answer is "missing", by extra name — `in place` would be a statement
        nobody made."""
        states = _by_key(cli.tier_states(
            package_present=lambda _name: True,
            weights_cached=lambda _name: True,
        ))
        with patch.object(wizard, "tier_requirements", lambda _tier: ()):
            blind = _by_key(cli.tier_states(package_present=lambda _name: True,
                                            weights_cached=lambda _name: True))
        self.assertTrue(states["deep"].ready)
        self.assertFalse(blind["deep"].ready)
        self.assertEqual(blind["deep"].missing_packages, ("extra:vlm",))
        # ...and a tier that installs no packages at all is unaffected by any of it.
        self.assertTrue(blind["faces"].ready)


class TestTheWeightsProbeLooksWhereTheLoadersWrite(unittest.TestCase):
    """Two caches, because two libraries download them: insightface keeps buffalo_l in
    a directory of its own, everything else comes through huggingface_hub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.insightface = self.root / "insightface"
        self.hub = self.root / "hub"
        self.insightface.mkdir()
        self.hub.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _cached(self, name: str) -> bool:
        return cli._weights_cached(name, insightface=self.insightface, hub=self.hub)

    def test_nothing_downloaded_is_answered_as_nothing_downloaded(self):
        for name in ("buffalo_l", "ViT-L-14", "XLM-RoBERTa", "Qwen2.5-VL-3B"):
            with self.subTest(weights=name):
                self.assertFalse(self._cached(name))

    def test_an_empty_directory_is_not_a_downloaded_model(self):
        """An interrupted download leaves the directory and no files in it."""
        (self.insightface / "buffalo_l").mkdir()
        self.assertFalse(self._cached("buffalo_l"))
        (self.insightface / "buffalo_l" / "det_10g.onnx").write_bytes(b"x")
        self.assertTrue(self._cached("buffalo_l"))

    def _hub_model(self, entry: str) -> Path:
        """A finished hub download: a revision under `snapshots/` with a file in it.

        F225: a directory is no longer the answer — an interrupted download leaves one
        behind too, and the probe has to tell the two apart (see
        test_a_download_that_stopped_halfway.py, where the rule itself is pinned).
        """
        snapshot = self.hub / entry / "snapshots" / "0123456789abcdef"
        snapshot.mkdir(parents=True)
        (snapshot / "open_clip_pytorch_model.bin").write_bytes(b"x")
        return self.hub / entry

    def test_the_hub_cache_is_matched_by_what_the_loader_asked_for(self):
        """The catalog names a model the way a person reads it; the cache names it the
        way the hub does, and for ViT-L-14 those two strings share nothing."""
        self._hub_model("models--timm--vit_large_patch14_clip_224.openai")
        self.assertTrue(self._cached("ViT-L-14"))
        self.assertFalse(self._cached("Qwen2.5-VL-3B"))
        self._hub_model("models--Qwen--Qwen2.5-VL-3B-Instruct")
        self.assertTrue(self._cached("Qwen2.5-VL-3B"))

    def test_a_missing_cache_directory_is_not_a_crash(self):
        self.assertFalse(cli._weights_cached("buffalo_l",
                                             insightface=self.root / "nope",
                                             hub=self.root / "also-nope"))

    def test_every_weight_of_the_catalog_is_known_to_the_probe(self):
        """The watchdog, the same shape as the extras one in the build script: a tier
        that gains a model file and does not say what it is called on disk would report
        "not downloaded" for a model that is right there."""
        named = {name for tier in wizard.TIERS for name in tier.weights}
        self.assertEqual(named - set(cli._WEIGHT_MARKERS), set())


class TestTheLinesAPersonReads(unittest.TestCase):
    """Three states, three sentences, three languages."""

    def test_a_state_gets_the_sentence_that_belongs_to_it(self):
        # F230: the way out at the end of the block is the one THIS install has, so the
        # kind is stated here — the words are what this case is about, and the suite runs
        # from a checkout while the sentence below belongs to an installed copy.
        lines = cli._doctor_tier_lines("en", [
            cli.TierState("base"),
            cli.TierState("faces", missing_weights=("buffalo_l",)),
            cli.TierState("deep", missing_packages=("transformers",)),
        ], kind=install.KIND_INSTALLED)
        self.assertEqual(lines[0], "Installed tiers:")
        self.assertIn("in place", lines[1])
        # The middle state says both what is missing and what it will cost — a size
        # nobody states is a size nobody agreed to.
        self.assertIn("buffalo_l", lines[2])
        self.assertIn("400 MB", lines[2])
        self.assertIn("not installed", lines[3])
        self.assertIn("transformers", lines[3])
        self.assertIn("sorta-setup", lines[-1])

    def test_the_way_out_is_only_offered_when_something_is_missing(self):
        complete = cli._doctor_tier_lines("en", [cli.TierState(tier.key)
                                                 for tier in wizard.TIERS])
        self.assertEqual(len(complete), len(wizard.TIERS) + 1)
        self.assertNotIn("sorta-setup", complete[-1])

    def test_a_long_list_of_missing_packages_stays_one_line(self):
        """The gpu tier names eight packages, and a line that wraps three times is a
        line nobody reads."""
        line = cli._doctor_tier_lines("en", [
            cli.TierState("gpu", missing_packages=("a", "b", "c", "d", "e", "f"))])[1]
        self.assertIn("a, b, c, d, +2", line)

    def test_every_sentence_exists_in_three_languages(self):
        states = [cli.TierState("base"),
                  cli.TierState("faces", missing_weights=("buffalo_l",)),
                  cli.TierState("deep", missing_packages=("transformers",))]
        rendered = {lang: cli._doctor_tier_lines(lang, states) for lang in _LANGS}
        for lang, lines in rendered.items():
            with self.subTest(lang=lang):
                # a header, one line per tier, and the way out
                self.assertEqual(len(lines), len(states) + 2)
                for line in lines:
                    self.assertTrue(line.strip(), lang)
        # ...and they are three translations rather than one string printed three times.
        for index in range(len(states) + 2):
            with self.subTest(line=index):
                self.assertEqual(len({rendered[lang][index] for lang in _LANGS}), 3)


class TestDoctorPrintsTheBlock(unittest.TestCase):
    """The wiring: the wizard's check screen IS this command (F211), so a tier that is
    missing has to be visible from it and not only from a helper."""

    def test_the_block_is_between_the_environment_and_the_health_lines(self):
        import io
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import patch

        health = SimpleNamespace(summary="health", available=True)
        buffer = io.StringIO()
        with patch.object(cli, "gpu_health", lambda **_kw: health), \
                patch.object(cli, "geo_data_health", lambda: health), \
                patch.object(cli, "tier_states",
                             lambda: [cli.TierState("faces",
                                                    missing_weights=("buffalo_l",))]), \
                patch.object(cli, "default_log_path", lambda: "run.log"), \
                redirect_stdout(buffer):
            cli._cmd_doctor("no-such-config.yaml")
        printed = buffer.getvalue().splitlines()
        self.assertIn("Installed tiers:", printed)
        self.assertLess(printed.index("Installed tiers:"), printed.index("health"))
        self.assertTrue(any("buffalo_l" in line for line in printed), printed)


if __name__ == "__main__":
    unittest.main()
