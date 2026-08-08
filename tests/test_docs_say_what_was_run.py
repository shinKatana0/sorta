"""F235: the public documents state the size that was measured, and nothing more.

The README said *60 GB+ tested, 300 GB+ by design* while the only full run ever made was
**380 GB / 38 485 files**, and nine features had landed over 2026-08-07/09 without the
guides following them. Two failure modes, and neither is caught by
`test_docs_guides.py` — that suite reads config keys and the CLI surface, and a sentence
about a collection size belongs to neither:

* a **retired claim that comes back** — the numbers below were replaced by a bigger
  measured one, so there is no context in these files where they are still true;
* a **claim that lives in one language only.** The three versions are one text in three
  languages; a number, a command or a tier name present in `en` and missing from `ja` is
  the drift that produced this feature in the first place.

The tokens are language-independent on purpose (digits, commands, log keys), so the check
says nothing about prose and cannot be satisfied by translating a sentence badly.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_GUIDE_DIR = _ROOT / "docs" / "guide"

GUIDES = {lang: _GUIDE_DIR / f"user-guide.{lang}.md" for lang in ("en", "ru", "ja")}
READMES = {"en": _ROOT / "README.md", "ru": _ROOT / "README.ru.md",
           "ja": _ROOT / "README.ja.md"}
PUBLIC = {**{f"guide.{k}": v for k, v in GUIDES.items()},
          **{f"readme.{k}": v for k, v in READMES.items()}}

# A thousands separator, and ONLY that: `ru` writes `38 485` with a non-breaking space,
# `en`/`ja` write `38,485`. The comma has to be followed by exactly three digits, because
# `ru` also writes a decimal comma (`1,6 ГБ`) — stripping that one would read
# 1.6 GB as 16 GB and pass on a document that says the wrong thing.
_THOUSANDS = re.compile(r"(?<=\d)(?:[\s  ](?=\d)|,(?=\d{3}(?!\d)))")

# The superseded collection size. The lookbehind is load-bearing: the preview-cache
# section legitimately measures a cache at `12.60 GB`, and a bare "60 GB" matches inside
# it. A future `60 GB` about something else fails here on purpose — the number that made
# this feature necessary is worth one line in a diff.
_RETIRED_SIZE = re.compile(r"(?<![\d.,])(?:60|300)\s*(?:GB|ГБ)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def whole(text: str) -> str:
    """The text with thousands separators removed, so `38 485` and `38,485` are one word."""
    return _THOUSANDS.sub("", text)


def spellings(number: str) -> tuple[str, ...]:
    """A decimal written the way each language writes it — `2.5` and `2,5`."""
    return (number, number.replace(".", ","))


def stated_in(number: str, text: str) -> bool:
    return any(form in text for form in spellings(number))


class TestTheMeasuredCollectionSize(unittest.TestCase):
    """380 GB / 38 485 files — the 2026-07-26/27 run, and the only one there is."""

    def test_the_superseded_size_is_in_no_public_document(self):
        for name, path in PUBLIC.items():
            found = _RETIRED_SIZE.search(read(path))
            with self.subTest(document=name):
                self.assertIsNone(found, f"{path.name}: {found.group(0) if found else ''}")

    def test_every_public_document_states_the_size_that_was_measured(self):
        """Both halves, everywhere: the size alone was what the retired sentence said, and
        it is the file count beside it that makes the claim checkable."""
        for name, path in PUBLIC.items():
            text = whole(read(path))
            for token in ("380", "38485"):
                with self.subTest(document=name, token=token):
                    self.assertIn(token, text)

    def test_the_readmes_state_it_where_a_reader_meets_it(self):
        """A README opens with the claim; buried on line 900 it is not the same statement."""
        for lang, path in READMES.items():
            opening = whole(read(path))[:1200]
            with self.subTest(lang=lang):
                self.assertIn("380", opening)
                self.assertIn("38485", opening)


class TestNoPromiseAboutMacOS(unittest.TestCase):
    """Nothing has been run on a Mac, so nothing is claimed about one.

    The CI job exists and is advisory; the guides have to say which of the two that is.
    """

    RETIRED = ("brew install exiftool", "Linux / macOS", "macOS/bash", "bash/macOS",
               "Linux, macOS", "Linux или macOS", "Linux or macOS",
               "Linux、macOS")

    def test_no_document_lists_macos_as_a_platform_it_runs_on(self):
        for name, path in PUBLIC.items():
            text = read(path)
            for phrase in self.RETIRED:
                with self.subTest(document=name, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_every_guide_says_the_macos_job_is_advisory(self):
        for lang, path in GUIDES.items():
            text = read(path)
            with self.subTest(lang=lang):
                self.assertIn("macos-latest", text)
                self.assertIn("advisory", text)


class TestTheNineFeaturesReachedAllThreeLanguages(unittest.TestCase):
    """2026-08-07/09 landed F222–F230; a guide that names one and not the others lies.

    Language-independent tokens only — a command, a log key, a measured number. Adding a
    row here is how the next feature's claim is made to arrive in `ru` and `ja` on the day
    it arrives in `en`, rather than at the next audit.
    """

    REQUIRED = (
        # F227 — the launch, in the order it happens
        "F227",
        "startup step=",
        # F230 — the acceleration tier, and the way back off it
        "sorta-setup --restore-cpu",
        "sorta-setup --tiers gpu",
        "sorta-setup --tiers deep",
        "r580",
        # F226 — exiftool: bundled on Windows, a package manager's job on Linux
        "sudo apt install libimage-exiftool-perl",
        # F224 — the way out is an item beside the way in
        "Uninstall Sorta",
        "sorta cache --clear-models",
        # F223 — two tiers priced apart, not one line
        "ViT-L-14",
        "XLM-RoBERTa",
        # the three install forms, so no advice is written for the wrong one
        "uv sync --extra gpu --extra dev",
        "install profile:",
    )

    def test_every_token_is_present_in_every_guide(self):
        for token in self.REQUIRED:
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, token=token):
                    self.assertIn(token, read(path))


class TestTheNumbersAgreeAcrossTheLanguages(unittest.TestCase):
    """A number stated in one language and not in the others is a number nobody checked."""

    # The launch measurement of F227 and the tier prices of F223, both of which the guides
    # quote and neither of which any of the three may quote alone.
    IN_EVERY_GUIDE = ("380", "38485", "5.65", "13.96", "1.6", "1.4", "2.5", "400")
    # The VRAM peak is a README statement in all three languages and in none of the guides,
    # which is a legitimate split — the READMEs are what it has to agree across.
    IN_EVERY_README = ("380", "38485", "20.5", "1.6", "1.4", "2.5", "400")

    def _agrees(self, number: str, documents: dict[str, Path]) -> None:
        present = {lang for lang, path in documents.items()
                   if stated_in(number, whole(read(path)))}
        self.assertEqual(present, set(documents), f"{number}: only in {sorted(present)}")

    def test_every_number_appears_in_all_three_guides(self):
        for number in self.IN_EVERY_GUIDE:
            with self.subTest(number=number):
                self._agrees(number, GUIDES)

    def test_every_number_appears_in_all_three_readmes(self):
        for number in self.IN_EVERY_README:
            with self.subTest(number=number):
                self._agrees(number, READMES)

    def test_a_decimal_comma_does_not_pass_as_a_thousands_separator(self):
        """The normalisation above is the one thing here that could silently agree.

        `1,6 GB` normalised to `16` would make a guide claiming 16 GB satisfy a check for
        1.6 GB, and every Russian tier price is written that way.
        """
        self.assertEqual(whole("38 485"), "38485")
        self.assertEqual(whole("38,485"), "38485")
        self.assertEqual(whole("1,6"), "1,6")
        self.assertEqual(whole("12.60"), "12.60")


if __name__ == "__main__":
    unittest.main()
