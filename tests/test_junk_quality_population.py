"""F120: who is measured for quality, and who is never shown to a VLM.

Both rules come out of the first live run of F113. The quality half wrote 24 196 rows
over the WHOLE collection and the answers were unusable for a reason that is structural
rather than a tuning miss: "are the eyes open" and "is there a pet" are questions about a
personal photograph. Measured on that run — 45% of the `dog` class and 45% of the
sharpest frames were not photographs, and screenshots average a laplacian of 2854 against
a photograph's 1253, so a global sharpness ranking sorted by content type.

The privacy rule is separate and is the user's call rather than a correctness one:
`vlm.exclude_classes` names classes no VLM is shown at all, and it holds `document` by
default because that bucket is passports, medical forms and bank papers.
"""
from __future__ import annotations

import unittest

from sorta.config import VLM_EXCLUDABLE_CLASSES, VlmConfig, _vlm_from
from sorta.junk import QUALITY_VERDICT, quality_settings


class _Cfg:
    """The two sections `quality_settings` reads, and nothing else."""

    def __init__(self, vlm: VlmConfig) -> None:
        self.vlm = vlm
        self.features = None


class TestExcludeClassesParsing(unittest.TestCase):
    def test_documents_are_excluded_unless_asked_otherwise(self):
        self.assertEqual(_vlm_from({}).exclude_classes, ("document",))

    def test_an_empty_list_really_means_send_everything(self):
        """An explicit `[]` is an answer, not an absence — it must survive the parser."""
        self.assertEqual(_vlm_from({"vlm": {"exclude_classes": []}}).exclude_classes, ())

    def test_a_list_of_typos_does_not_turn_the_protection_off(self):
        """The dangerous direction. Somebody writing this key down wants MORE protection,
        so a misspelling falls back to the default rather than to nothing excluded."""
        got = _vlm_from({"vlm": {"exclude_classes": ["documnet", "docs"]}})
        self.assertEqual(got.exclude_classes, ("document",))

    def test_names_are_normalized_and_deduplicated(self):
        got = _vlm_from({"vlm": {"exclude_classes": ["DOCUMENT", " document ", "product"]}})
        self.assertEqual(got.exclude_classes, ("document", "product"))

    def test_photo_is_not_excludable(self):
        """Excluding personal photographs would leave the tier with nothing to do, so the
        name is not in the accepted set and is dropped like any other unknown one."""
        self.assertNotIn("photo", VLM_EXCLUDABLE_CLASSES)
        got = _vlm_from({"vlm": {"exclude_classes": ["photo"]}})
        self.assertEqual(got.exclude_classes, ("document",))

    def test_the_setting_reaches_the_stage(self):
        q = quality_settings(_Cfg(VlmConfig(exclude_classes=("document", "product"))))
        self.assertEqual(q.exclude_classes, frozenset({"document", "product"}))


class TestQualityPopulation(unittest.TestCase):
    def test_quality_is_asked_only_of_personal_photographs(self):
        """A single constant, so the selection query and the per-frame check cannot
        drift apart — they are the two places that would have to agree."""
        self.assertEqual(QUALITY_VERDICT, "photo")
