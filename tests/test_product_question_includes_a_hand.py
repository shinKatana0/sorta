"""F196: the product question includes an item held in a hand.

The deep tier asked a question that narrowed itself — `isolated object, catalog shot` —
and a thing held up to the camera fell into `everyday life`. The owner labelled 20 misses
against a list of reasons fixed before the frames were opened and got `narrow` 17 (85%),
`borderline` 3, and nothing at all in `feature_missing` or `other`. Re-asking 733
already-asked frames with the wider wording on 2026-08-05 priced the edit: recall 80% ->
94%, precision 78% -> 75%, ~2 107 -> ~2 604 frames marked.

What the cases below pin, in the order the brief asks for them:

1. the QUESTION IS A DIFFERENT ONE — its fingerprint moved, so an answer stored by the old
   wording is an answer to something else and cannot be treated as fresh;
2. the answer format did not move with it: one word out of the same three, parsed by the
   same rule, and the same verdict for each;
3. `personal_photo` and `document` are word for word what they were — the measurement
   priced ONE edit;
4. the numbers of that measurement stand next to the prompt, so the next clarification is
   made by somebody who knows what the previous one cost.

Case 1 is the fingerprint of the TEXT. It is deliberately not a claim about the database:
`media_class.tier` is a bare tier name (F68) and carries no fingerprint of the question,
unlike `frame_quality.source` (F120) — see the comment above `_VLM_PROMPT` for what a
populated collection has to do to see this edit.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta import junk
from sorta.junk import _VLM_LABEL_TO_VERDICT, _VLM_MAX_NEW_TOKENS, _VLM_PROMPT, _vlm_label

_ROOT = Path(__file__).resolve().parent.parent

# The question as it stood before F196, verbatim. It is here rather than in a fixture
# because the two categories that must NOT have moved are read out of it below: a
# paraphrase would let a rewrite of `document` pass as unchanged.
_PROMPT_BEFORE_F196 = (
    "Classify this image into exactly one category: personal_photo, document, "
    "or product.\n"
    "personal_photo = a personal/casual photograph of people, places, pets or "
    "everyday life.\n"
    "document = a photographed or scanned document, receipt, ID card, form, or "
    "other text-heavy paper.\n"
    "product = an item photographed for sale or a marketplace/e-commerce style "
    "listing photo (isolated object, catalog shot).\n"
    "Answer with exactly one word: personal_photo, document, or product."
)


def _fingerprint(text: str) -> str:
    """Eight hex characters over a prompt — the shape `quality_prompt_fingerprint` uses."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _category_line(prompt: str, label: str) -> str:
    """The one line of the question that defines `label`."""
    for line in prompt.split("\n"):
        if line.startswith(f"{label} = "):
            return line
    raise AssertionError(f"промпт не определяет категорию {label}")


class TestTheQuestionIsADifferentOne(unittest.TestCase):
    """Test 1: the fingerprint moved — a stored answer to the old wording is stale."""

    def test_the_fingerprint_of_the_question_moved(self):
        self.assertNotEqual(_fingerprint(_VLM_PROMPT), _fingerprint(_PROMPT_BEFORE_F196))

    def test_the_narrowing_words_are_gone(self):
        # The two phrases the labelling named as the cause: a question that describes a
        # catalog cannot be answered `product` about a thing somebody is holding.
        self.assertNotIn("isolated object", _VLM_PROMPT)
        self.assertNotIn("catalog shot", _VLM_PROMPT)

    def test_a_hand_in_the_frame_is_answered_in_the_question(self):
        product = _category_line(_VLM_PROMPT, "product")
        self.assertIn("held in a hand", product)
        self.assertIn("a hand in the frame does not make it a personal photo",
                      _VLM_PROMPT)


class TestTheAnswerFormatDidNotMove(unittest.TestCase):
    """Test 2: still exactly one word out of three, read by the same rule."""

    def test_the_question_asks_for_one_word_out_of_the_same_three(self):
        self.assertTrue(_VLM_PROMPT.endswith(
            "Answer with exactly one word: personal_photo, document, or product."))
        self.assertEqual(_VLM_MAX_NEW_TOKENS, 8)

    def test_the_question_defines_exactly_three_categories(self):
        defined = re.findall(r"^(\w+) = ", _VLM_PROMPT, flags=re.MULTILINE)
        self.assertEqual(sorted(defined), ["document", "personal_photo", "product"])
        self.assertEqual(sorted(_VLM_LABEL_TO_VERDICT), sorted(defined))

    def test_the_answers_are_read_exactly_as_before(self):
        for answer, label in (("product", "product"),
                              ("Product.", "product"),
                              ("document", "document"),
                              ("personal_photo", "personal_photo"),
                              ("a hand holding a mug", "personal_photo"),
                              ("", "personal_photo")):
            self.assertEqual(_vlm_label(answer), label)

    def test_each_label_still_maps_to_the_verdict_it_did(self):
        self.assertEqual(_VLM_LABEL_TO_VERDICT, {"personal_photo": "photo",
                                                 "document": "document",
                                                 "product": "product"})

    def test_the_widened_question_is_what_reaches_the_model(self):
        # The constant and the call site, tied together: an edit that lands in the module
        # but not in the call is the failure this feature is about.
        with tempfile.TemporaryDirectory() as tmp:
            frame = str(Path(tmp) / "held-in-a-hand.jpg")
            Image.new("RGB", (64, 48), (10, 100, 200)).save(frame, "JPEG")
            seen: list[tuple[str, int]] = []

            def describe(images, prompt, max_new_tokens):
                seen.append((prompt, max_new_tokens))
                return "product"

            self.assertEqual(junk.vlm_classifier_from(describe)(frame), "product")
        self.assertEqual(seen, [(_VLM_PROMPT, _VLM_MAX_NEW_TOKENS)])
        self.assertIn("held in a hand", seen[0][0])


class TestTheOtherTwoCategoriesAreWordForWord(unittest.TestCase):
    """Test 3: the measurement priced ONE edit, so only one line may have changed."""

    def test_personal_photo_is_unchanged(self):
        self.assertEqual(_category_line(_VLM_PROMPT, "personal_photo"),
                         _category_line(_PROMPT_BEFORE_F196, "personal_photo"))

    def test_document_is_unchanged(self):
        self.assertEqual(_category_line(_VLM_PROMPT, "document"),
                         _category_line(_PROMPT_BEFORE_F196, "document"))

    def test_the_opening_line_is_unchanged(self):
        self.assertEqual(_VLM_PROMPT.split("\n")[0],
                         _PROMPT_BEFORE_F196.split("\n")[0])


class TestTheMeasurementStandsNextToTheQuestion(unittest.TestCase):
    """Test 4: the guard on the reasoning — a wording nobody can price is re-guessed."""

    def comment_above_the_prompt(self) -> str:
        source = (_ROOT / "sorta" / "junk.py").read_text(encoding="utf-8")
        lines = source.split("\n")
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("_VLM_PROMPT = ("))
        block: list[str] = []
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
            block.append(lines[start])
        return "\n".join(reversed(block))

    def test_both_halves_of_the_trade_are_written_down(self):
        comment = self.comment_above_the_prompt()
        for number in ("78%", "75%", "80%", "94%", "~2 107", "~2 604", "~290", "~190"):
            self.assertIn(number, comment,
                          f"{number} потерялось из обоснования рядом с промптом")

    def test_the_price_of_landing_it_is_written_down(self):
        # An edit here does not invalidate anything by itself — the marker of the deep
        # tier is a bare tier name. Whoever ships it has to read that off the comment.
        comment = self.comment_above_the_prompt()
        self.assertIn("media_class.tier", comment)
        self.assertIn("90 minutes", comment)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
