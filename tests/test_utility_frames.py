"""F175: "Utility frames" — one name that used to cover four different buckets.

The slice was called "Not personal photos" and showed 4 980 frames, which is exactly
`product` + `screenshot` + `document` + `meme`: the whole of the classifier's output
under a single name. Three things were wrong with that at once, and this module pins
the fix for each of them.

* **The name.** A photograph of a receipt is personal, a screenshot of a conversation
  with your wife is personal, a passport more so. They are not photographs taken FOR
  MEMORY, which is a different thing — and the old name, read as "not personal", made
  the slice look like a bin, with a thousand documents inside it that must not be
  deleted.
* **The collision.** `files.not_personal` is a heuristic over the file NAME (downloaded
  films: three files out of 38 485) and lands in a folder of its own. Two different
  questions, computed by different stages, must not answer to the same words — so the
  slice is renamed and the guides say in so many words that the two are unrelated.
* **The precision.** The buckets are measured separately (products 78%, screenshots
  59%, documents and memes not at all), so the slice as a whole names no number and
  each bucket names its own. A class with no measurement says "not measured" rather
  than inheriting a neighbour's percentage — the fallback below is what guarantees it
  for the classes that get added later, too.

The fourth requirement — a document is tellable apart BEFORE a person selects
everything — is checked on the payload and on the rendered client: the card carries the
mark as a field of its own, not as the absence of a thumbnail.
"""
from __future__ import annotations

import dataclasses
import re
import unittest
from pathlib import Path

from sorta import i18n, ui

from tests.test_ui_junk_buckets import JunkViewTestBase

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = {
    "guide.en": _ROOT / "docs" / "guide" / "user-guide.en.md",
    "guide.ru": _ROOT / "docs" / "guide" / "user-guide.ru.md",
    "guide.ja": _ROOT / "docs" / "guide" / "user-guide.ja.md",
    "readme.en": _ROOT / "README.md",
    "readme.ru": _ROOT / "README.ru.md",
    "readme.ja": _ROOT / "README.ja.md",
}

# The dead name, in the three languages it was written in. The Russian pattern is
# deliberately narrow: «не личность» (§22, about pHash comparing a frame and not an
# identity) is an unrelated sentence and must keep working. The English one requires a
# SPACE — `files.not_personal` and `_Unsorted/not_personal/` are identifiers, they are
# what the guides now point at to say "that is the other thing".
_DEAD_NAME = re.compile(
    r"не[\s ]+личны|не[\s ]+личн(?:ое|ого)|not[\s ]+personal"
    r"|個人写真ではない",
    re.IGNORECASE)

LANGS = ("ru", "en", "ja")

# The measurements the caption of a bucket is allowed to state, and where they come
# from. Anything not listed here has no number and must say so.
MEASURED = {
    "product": ("78%", "81%", "2026-08-03", "999"),
    "screenshot": ("59%", "83%", "2026-08-03", "350"),
}
UNMEASURED = ("document", "meme")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestTheDeadNameIsGone(unittest.TestCase):
    def test_no_guide_or_readme_calls_the_slice_that(self):
        for name, path in _DOCS.items():
            found = _DEAD_NAME.search(read(path))
            with self.subTest(doc=name):
                self.assertIsNone(found, f"{path.name}: {found.group(0) if found else ''}")

    def test_no_caption_in_the_catalog_calls_the_slice_that(self):
        for key, entry in ui._UI_STRINGS.items():
            for lang, value in entry.items():
                found = _DEAD_NAME.search(value)
                with self.subTest(key=key, lang=lang):
                    self.assertIsNone(found, f"{key}/{lang}: {found.group(0) if found else ''}")

    def test_the_unrelated_russian_sentence_still_reads(self):
        """A guard on the guard: «не личность» is about pHash, not about this slice.

        A pattern wide enough to catch it would make the case above unfixable — the
        author would have to rewrite a paragraph that was never wrong.
        """
        self.assertIsNone(_DEAD_NAME.search("pHash сравнивает похожесть кадра, а "
                                            "не личность."))

    def test_the_slice_has_a_name_in_all_three_languages(self):
        entry = ui._UI_STRINGS["tab_junk"]
        self.assertEqual(set(entry), set(LANGS))
        for lang, value in entry.items():
            with self.subTest(lang=lang):
                self.assertTrue(value.strip(), f"tab_junk/{lang} is empty")

    def test_the_rendered_page_carries_the_new_name(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.assertIn(ui._UI_STRINGS["tab_junk"][lang],
                              ui._render_index_html(lang))


class TestTheSliceAndTheDownloadedFilmsFolderAreTwoThings(unittest.TestCase):
    """Requirement 2: `files.not_personal` is about where a file came from.

    The flag itself is untouched (three files out of 38 485, and the heuristic is
    accurate) — what had to stop was the two of them answering to one set of words.
    """

    def test_the_slice_and_the_folder_are_not_named_alike(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                slice_name = ui._UI_STRINGS["tab_junk"][lang].casefold()
                folder = i18n.folder("not_personal", lang).casefold()
                self.assertNotEqual(slice_name, folder)
                self.assertNotIn(folder, slice_name)
                self.assertNotIn(slice_name, folder)

    def test_the_flag_still_has_its_folder(self):
        # Renaming was one of the two ways out and it is NOT the one taken: an existing
        # collection already has this folder on disk, and the guides carry the
        # distinction instead. The case is here so that a later rename is a deliberate
        # decision with a test to update, not a silent one.
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.assertTrue(i18n.folder("not_personal", lang).strip())

    def test_every_guide_says_the_two_are_different(self):
        """The flag by its identifier, the folder by a name a reader will see on disk.

        Which of the two spellings that is depends on the guide: the `ru` one shows the
        layout the way `language: ru` builds it («не_личное»), the other two keep the
        English folder names they use everywhere else.
        """
        for lang in LANGS:
            text = read(_DOCS[f"guide.{lang}"])
            folders = {i18n.folder("not_personal", lang),
                       i18n.folder("not_personal", "en")}
            with self.subTest(lang=lang):
                self.assertIn("files.not_personal", text)
                self.assertTrue(any(folder in text for folder in folders), folders)
                self.assertIn(ui._UI_STRINGS["tab_junk"][lang], text)


class TestPrecisionBelongsToTheBucket(unittest.TestCase):
    def test_the_slice_caption_names_no_percentage(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                intro = ui._UI_STRINGS["junk_intro"][lang]
                self.assertNotIn("%", intro)
                for numbers in MEASURED.values():
                    for token in numbers:
                        self.assertNotIn(token.rstrip("%"), intro)

    def test_a_measured_bucket_states_its_number_its_date_and_its_sample(self):
        for verdict, tokens in MEASURED.items():
            entry = ui._UI_STRINGS[f"junk_accuracy_{verdict}"]
            self.assertEqual(set(entry), set(LANGS))
            for lang, value in entry.items():
                for token in tokens:
                    with self.subTest(verdict=verdict, lang=lang, token=token):
                        self.assertIn(token, value)

    def test_an_unmeasured_bucket_has_no_caption_of_its_own(self):
        # It must fall through to `junk_accuracy_unmeasured`. A key of its own here —
        # even an honest one — would be a place for a number to appear later without
        # anybody having measured anything.
        for verdict in UNMEASURED:
            with self.subTest(verdict=verdict):
                self.assertNotIn(f"junk_accuracy_{verdict}", ui._UI_STRINGS)

    def test_the_fallback_says_not_measured_and_states_no_number(self):
        entry = ui._UI_STRINGS["junk_accuracy_unmeasured"]
        self.assertEqual(set(entry), set(LANGS))
        for lang, value in entry.items():
            with self.subTest(lang=lang):
                self.assertTrue(value.strip())
                self.assertNotIn("%", value)
                self.assertFalse(re.search(r"\d", value), value)

    def test_the_client_looks_the_caption_up_and_falls_back(self):
        html = ui._render_index_html("en")
        self.assertIn(
            'return I18N["junk_accuracy_" + verdict] || I18N.junk_accuracy_unmeasured;',
            html)
        # ...and the whole slice ("all buckets") gets no line at all.
        self.assertIn('if (!verdict) return "";', html)
        self.assertIn('id="junk-accuracy"', html)

    def test_the_line_is_rewritten_for_the_bucket_that_is_open(self):
        html = ui._render_index_html("en")
        self.assertIn("accuracy.textContent = junkAccuracyText(data.bucket);", html)


class TestDocumentsAreTellableApart(JunkViewTestBase):
    """Requirement 4: the bucket that must not be deleted is visible as such.

    Before any selection, and without asking the client to infer it from a missing
    preview link — a card with no `thumb_url` could be one whose preview failed to
    build, and the two need different words on the screen.
    """

    def test_a_document_card_is_marked_and_a_product_card_is_not(self):
        self.add_classified("passport.jpg", "document")
        self.add_classified("chair.jpg", "product")
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        self.assertTrue(by_verdict["document"]["sensitive"])
        self.assertFalse(by_verdict["product"]["sensitive"])

    def test_the_mark_comes_from_the_config_key_like_the_preview_does(self):
        self.add_classified("passport.jpg", "document")
        self.add_classified("chair.jpg", "product")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        self.assertTrue(by_verdict["product"]["sensitive"])
        self.assertFalse(by_verdict["document"]["sensitive"])

    def test_the_mark_travels_with_the_bucket_filter_too(self):
        self.add_classified("passport.jpg", "document")
        self.start_server()
        self.assertTrue(self.junk("?bucket=document")["items"][0]["sensitive"])

    def test_the_card_carries_its_own_class_and_its_own_chip(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('(item.sensitive ? " sensitive" : "")', html)
        self.assertIn("I18N.junk_document_mark", html)
        self.assertIn(".junk-card.sensitive", html)

    def test_the_note_stands_above_the_button_that_selects_everything(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        hint = html.index('id="junk-doc-hint"')
        select_all = html.index('id="junk-select-all-btn"')
        grid = html.index('id="junk-grid"')
        self.assertLess(hint, select_all)
        self.assertLess(select_all, grid)

    def test_the_note_appears_exactly_where_such_cards_are(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('grid.querySelector(".junk-card.sensitive") ? "" : "none"', html)


class TestNewStringsAreTranslated(unittest.TestCase):
    KEYS = ("tab_junk", "junk_intro", "junk_accuracy_product",
            "junk_accuracy_screenshot", "junk_accuracy_unmeasured",
            "junk_document_hint", "junk_document_mark")

    def test_every_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), set(LANGS))
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_captions_reach_the_page_in_the_chosen_language(self):
        for lang in LANGS:
            html = ui._render_index_html(lang)
            for key in ("junk_intro", "junk_document_hint"):
                with self.subTest(lang=lang, key=key):
                    self.assertIn(ui._UI_STRINGS[key][lang], html)


if __name__ == "__main__":
    unittest.main()
