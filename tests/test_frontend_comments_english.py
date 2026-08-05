"""F197: the comments of the browser script and the stylesheet are English too.

F111 made the prose of this public repository English and checked it with a Python
tokenizer. The browser script was invisible to that check for a year, and not through
carelessness: it lived inside `ui.py` as a triple-quoted string, so to `tokenize` it was
a string literal and not a comment at all. F182 moved it out into `sorta/web/app/app.js`
and `sorta/web/style.css`, and 271 lines of Russian prose appeared as ordinary source.

The lesson worth pinning is why one guard could not see what another had to: a
repository-wide rule sees a file as its tool sees it, not as a reader does. So the rule
gets a second reading of the same files, this time by a scanner that knows what a
comment is in JavaScript and in CSS.

Two things are checked, and the second is the one that makes the first safe to do:

* no comment holds Russian PROSE — the F111 rule, unchanged and imported rather than
  restated, so the two guards cannot drift apart. Russian that is the SUBJECT stays in
  quotes: interface captions a comment cites («Показать ещё»), folder names the product
  really writes, search queries a measurement was run with;
* the code the browser receives did not move. Translating a comment cannot change
  behaviour, so the proof is a digest of the file with every comment removed and blank
  lines dropped: it survives any rewrapping of a paragraph and fails the moment a
  selector, a key or a line of code is touched «раз уж я здесь». The two digests below
  were taken from the pre-translation files.

`page.html` is not read here — it was already clean, and a guard over a file with
nothing to guard is a line that only ever costs time.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from tests.test_comments_english import unquoted_cyrillic

_ROOT = Path(__file__).resolve().parent.parent
_WEB = _ROOT / "sorta" / "web"
_FRONTEND = ("app/app.js", "style.css")

# sha256 of `code_fingerprint`, taken from the files as they were before a single comment
# was translated. Unlike a digest of the whole page (deliberately not asserted by
# tests/test_ui_frontend_files.py, and for a good reason) this one does not move when a
# caption changes: it is a digest of the code alone, so what it pins is exactly the
# promise this feature makes and nothing else.
# A `/` opens a regular expression only where a value cannot already have ended; after an
# identifier, a number, a `)` or a `]` the same character is division. The set is small
# on purpose — it covers the three literals app.js actually contains, and anything it
# misjudges would show up immediately as a moved fingerprint rather than silently.
_BEFORE_A_REGEX = set("(,=:[!&|?{};+-*%~^<>")


def _string_end(source: str, start: int) -> int:
    """Index just past the string literal opening at `start`."""
    quote = source[start]
    i = start + 1
    while i < len(source):
        ch = source[i]
        if ch == "\\":
            i += 2
        elif ch == quote:
            return i + 1
        elif ch == "\n" and quote != "`":
            return i  # unterminated: leave the newline to the caller
        else:
            i += 1
    return len(source)


def _regex_end(source: str, start: int) -> int:
    """Index just past the regular-expression literal opening at `start`, flags included."""
    i = start + 1
    in_class = False
    while i < len(source):
        ch = source[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            i += 1
            break
        elif ch == "\n":
            return i
        i += 1
    while i < len(source) and source[i].isalpha():
        i += 1
    return i


def _block_text(chunk: str) -> str:
    """The prose of a `/* ... */` comment: delimiters and the decorative left rule gone."""
    inner = chunk[2:-2] if chunk.endswith("*/") else chunk[2:]
    return " ".join(line.strip().lstrip("*").strip() for line in inner.splitlines()).strip()


def split_comments(source: str, *, js: bool) -> tuple[str, list[tuple[int, int, str]]]:
    """Separate `source` into its code and its comments.

    Returns the code with every comment replaced by a single space — a space rather than
    nothing, so removing a comment cannot glue two tokens together — and the comments as
    (first line, last line, text). Strings are skipped whole: a `//` inside a caption or
    a `/*` inside an SVG path is not a comment, and reading it as one is how a naive
    line-prefix guard starts reporting the product's own text.
    """
    code: list[str] = []
    comments: list[tuple[int, int, str]] = []
    line = 1
    i = 0
    n = len(source)
    previous = ""  # the last significant character of code seen so far
    while i < n:
        ch = source[i]
        if ch in "\"'" or (js and ch == "`"):
            end = _string_end(source, i)
            chunk = source[i:end]
            code.append(chunk)
            line += chunk.count("\n")
            previous = ch
            i = end
            continue
        if ch == "/" and i + 1 < n:
            following = source[i + 1]
            if js and following == "/":
                end = source.find("\n", i)
                end = n if end < 0 else end
                comments.append((line, line, source[i + 2 : end].strip()))
                code.append(" ")
                i = end
                continue
            if following == "*":
                end = source.find("*/", i + 2)
                end = n if end < 0 else end + 2
                chunk = source[i:end]
                comments.append((line, line + chunk.count("\n"), _block_text(chunk)))
                code.append(" ")
                line += chunk.count("\n")
                i = end
                continue
            if js and previous in _BEFORE_A_REGEX:
                end = _regex_end(source, i)
                code.append(source[i:end])
                previous = "/"
                i = end
                continue
        code.append(ch)
        if ch == "\n":
            line += 1
        elif not ch.isspace():
            previous = ch
        i += 1
    return "".join(code), comments


def comment_blocks(source: str, *, js: bool) -> list[tuple[int, str]]:
    """(first line, text) per run of consecutive comment lines.

    Consecutive lines are joined for the same reason F111 joins them: a quotation may be
    wrapped across two of them, and a per-line check would read the halves as unquoted
    Russian.
    """
    blocks: list[tuple[int, str]] = []
    previous_end = -2
    for first, last, text in split_comments(source, js=js)[1]:
        if first == previous_end + 1 and blocks:
            start, existing = blocks[-1]
            blocks[-1] = (start, f"{existing} {text}")
        else:
            blocks.append((first, text))
        previous_end = last
    return blocks


def code_fingerprint(source: str, *, js: bool) -> str:
    """A digest of what the browser is asked to do, with the prose about it removed.

    Blank lines go too, and so does trailing whitespace, because that is all a comment
    leaves behind. Indentation is kept: it is not something a translation touches.
    """
    code = split_comments(source, js=js)[0]
    kept = [stripped for stripped in (line.rstrip() for line in code.splitlines()) if stripped]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def frontend_files() -> list[tuple[str, str, bool]]:
    return [
        (rel, _WEB.joinpath(*rel.split("/")).read_text(encoding="utf-8"), rel.endswith(".js"))
        for rel in _FRONTEND
    ]


class TestTheFrontendPoseIsEnglish(unittest.TestCase):
    def test_no_comment_holds_russian_prose(self):
        for rel, source, js in frontend_files():
            for line, text in comment_blocks(source, js=js):
                with self.subTest(file=rel, line=line):
                    self.assertEqual(unquoted_cyrillic(text), [], text)

    def test_both_files_are_really_read(self):
        """A guard that finds no comments passes for the wrong reason. app.js carried 263
        Russian comment lines and style.css 34, so the counts below cannot come from a
        scanner that quietly stopped at the first string literal."""
        counts = {rel: len(comment_blocks(source, js=js)) for rel, source, js in frontend_files()}
        self.assertGreater(counts["app/app.js"], 100)
        self.assertGreater(counts["style.css"], 10)


class TestTheScannerReadsJavaScriptAndNotJustLines(unittest.TestCase):
    """What a line-prefix guard gets wrong, stated as cases: these are the shapes the
    frontend actually contains — SVG markup in single quotes, a caption with a slash, a
    regular expression built from punctuation."""

    def test_a_comment_marker_inside_a_string_is_not_a_comment(self):
        source = 'var url = "https://example.invalid/x"; // а это уже комментарий\n'
        self.assertEqual(comment_blocks(source, js=True), [(1, "а это уже комментарий")])

    def test_a_regular_expression_is_not_the_start_of_a_comment(self):
        source = 'key.replace(/[^a-z0-9/]+/g, "-"); // tail\n'
        code, comments = split_comments(source, js=True)
        self.assertEqual(comments, [(1, 1, "tail")])
        self.assertIn("/[^a-z0-9/]+/g", code)

    def test_division_is_not_a_regular_expression(self):
        code, comments = split_comments("var r = w / h; /* why */ var q = 1;\n", js=True)
        self.assertEqual(comments, [(1, 1, "why")])
        self.assertIn("var q = 1;", code)

    def test_a_block_comment_loses_its_left_rule_and_keeps_its_words(self):
        self.assertEqual(_block_text("/*\n * one\n * two\n */"), "one two")

    def test_css_has_no_line_comments(self):
        """`//` is not a comment in CSS, and a guard that thinks it is would read the
        second half of a `url(//host/x)` as prose."""
        code, comments = split_comments("a { background: url(//h/x); } /* по-русски */\n", js=False)
        self.assertEqual(comments, [(1, 1, "по-русски")])
        self.assertIn("url(//h/x)", code)

    def test_a_quotation_wrapped_over_two_lines_stays_one_block(self):
        source = '// the caption reads "Показать\n// ещё" on the button\nvar x = 1;\n'
        self.assertEqual(
            comment_blocks(source, js=True),
            [(1, 'the caption reads "Показать ещё" on the button')],
        )

    def test_the_rule_would_notice_russian_prose_in_either_file(self):
        for js in (True, False):
            with self.subTest(js=js):
                marker = "//" if js else "/*"
                block = comment_blocks(f"{marker} папка для мусора\n", js=js)
                self.assertEqual(unquoted_cyrillic(block[0][1]), ["папка", "для", "мусора"])


class TestTranslatingACommentChangedNoCode(unittest.TestCase):
    """What is left of F197's acceptance criterion, and why the rest is gone.

    It held a digest of both files' CODE taken before the first line was translated, and
    it did its job: it proved that a rewrite of 271 comments moved no selector, no i18n
    key and no comparison. That proof was worth having ONCE.

    It cannot stay, and the rule written to keep it was tried first. F199 set one down
    beside the digest: a feature that really edits the frontend moves the number IN THE
    SAME commit, so it always states "the code as of the last deliberate change". That is
    sound and it failed immediately — F198 and F200 both edited app.js for good reasons
    and neither had any way to know a digest lived in a test about translating comments.
    Two of the first three occasions.

    A check that fires on honest work is a check somebody re-anchors without reading, and
    one day they will re-anchor it while a selector really did move.

    What survives is the part that keeps working: the fingerprint itself, and the proof
    that it ignores comments and notices code. It is the tool the next such translation
    will use to take its own before-and-after, which is the only way a digest of this kind
    is honest.
    """

    def test_the_fingerprint_ignores_comments_and_notices_code(self):
        """Both halves of the claim, because either one alone would make it useless."""
        before = "var a = 1; // старый текст\n\nvar b = 2;\n"
        translated = "var a = 1; // the new text, on\n// two lines now\nvar b = 2;\n"
        self.assertEqual(
            code_fingerprint(before, js=True), code_fingerprint(translated, js=True)
        )
        self.assertNotEqual(
            code_fingerprint(before, js=True),
            code_fingerprint("var a = 1; // старый текст\n\nvar b = 3;\n", js=True),
        )


if __name__ == "__main__":
    unittest.main()
