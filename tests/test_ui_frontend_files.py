"""F182: the page, the stylesheet and the script are files, not a string literal.

Of the 14 427 lines `ui.py` had grown to, 6 100 — 42% — were inside triple quotes:
the markup, the styles and the whole browser script. Nothing about them is Python.
They made ruff and mypy read the text of a page, they hid every syntax error the
editor would have shown, and a conflict in one word of markup arrived as a conflict
in `ui.py`, the one file ten features were already queuing for.

So `sorta/web/` holds them as what they are, and `page.html` keeps two seams —
`{{style}}` and `{{script}}` — that are filled once at import.

What is pinned here is that the move stays a move:

* the template the server holds is the three files put back together, byte for byte
  (test 2 — the main one: it is the same comparison that was made against the
  pre-split rendering at migration time, written so it keeps being made). The
  reference taken before the move, for the record — sha256 of `_render_index_html`:

      ru  0699b639aef1233c8abd6e79832f03946529575d21383c9381a060f1fd9ac0c6
      en  ab1d2c60fdc49c347d71e2238d88b407e4963adce439133fb1b38be4a22db523
      ja  8024dc3408261be415da1a48d957ed3f4de5fae747e1b0994969d556fd8855b0

  They are not asserted here on purpose: a digest of the whole page turns the next
  legitimate caption change into a gate failure with nothing to read. What is asserted
  is the property that made them equal — the seams, and only the seams, join the files;
* a checkout with CRLF line endings serves the same bytes as one with LF (test 3);
* the three languages still substitute, and no `{{placeholder}}` survives (test 4);
* the files reach the built wheel (test 5) — the F65 trap, where the geo base stayed
  outside the package and two releases had to be marked broken.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from sorta import ui

_ROOT = Path(__file__).resolve().parents[1]
_WEB = _ROOT / "sorta" / "web"
_FRONTEND = ("page.html", "style.css", "app/app.js")


class TestTheFrontendLivesInFiles(unittest.TestCase):
    def test_the_three_files_are_there_and_carry_the_page(self):
        for rel in _FRONTEND:
            with self.subTest(rel):
                path = _WEB.joinpath(*rel.split("/"))
                self.assertTrue(path.is_file(), f"{rel} is missing from sorta/web")
                self.assertGreater(len(path.read_text(encoding="utf-8")), 1000)

    def test_the_template_is_the_three_files_put_back_together(self):
        """The clean-transfer proof: assembled here independently of the loader, and
        the two seams are the only thing that joins them."""
        page = (_WEB / "page.html").read_text(encoding="utf-8")
        self.assertIn("{{style}}", page)
        self.assertIn("{{script}}", page)
        expected = page.replace(
            "{{style}}", (_WEB / "style.css").read_text(encoding="utf-8")
        ).replace(
            "{{script}}", (_WEB / "app" / "app.js").read_text(encoding="utf-8")
        )
        self.assertEqual(ui._INDEX_HTML_TEMPLATE, expected)

    def test_crlf_on_disk_does_not_change_a_byte_of_the_page(self):
        """A Windows checkout materialises text files with CRLF. The template is
        assembled with "\\n" everywhere, so reading these files in binary — or with
        `newline=""` — would serve a different page on one platform than on the other.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in _FRONTEND:
                dst = root.joinpath(*rel.split("/"))
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = _WEB.joinpath(*rel.split("/")).read_text(encoding="utf-8")
                dst.write_bytes(src.replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(ui._load_index_template(root=root), ui._INDEX_HTML_TEMPLATE)

    def test_every_language_still_substitutes_and_nothing_is_left_unfilled(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang):
                html = ui._render_index_html(lang)
                self.assertIn(f'<html lang="{lang}"', html)
                self.assertNotIn("{{", html)
                self.assertIn("window.I18N", html)


class TestTheFrontendReachesTheWheel(unittest.TestCase):
    """F65 again, and it is the reason this class exists rather than a config assertion:
    the geo base was declared, looked declared, and was absent from the wheel — the CLI
    resolved every coordinate to an empty place and two releases were marked broken.
    A page that ships without its script fails the same way, louder."""

    def test_pyproject_names_the_frontend_among_the_wheel_artifacts(self):
        """The cheap half: it never skips, so `sorta/web` cannot quietly drop out of
        the build configuration on a machine without uv."""
        data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        artifacts = data["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
        for pattern in ("sorta/web/*.html", "sorta/web/*.css", "sorta/web/app/*.js"):
            with self.subTest(pattern):
                self.assertIn(pattern, artifacts)

    def test_a_built_wheel_carries_the_page_the_stylesheet_and_the_script(self):
        uv = shutil.which("uv")
        if uv is None:
            self.skipTest("uv is not on PATH; the gate is documented to run under it")
        with tempfile.TemporaryDirectory() as tmp:
            built = subprocess.run(
                [uv, "build", "--wheel", "--out-dir", tmp],
                cwd=_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            wheels = list(Path(tmp).glob("*.whl"))
            self.assertEqual(len(wheels), 1, wheels)
            inside = set(zipfile.ZipFile(wheels[0]).namelist())
        for rel in _FRONTEND:
            with self.subTest(rel):
                self.assertIn(f"sorta/web/{rel}", inside)


if __name__ == "__main__":
    unittest.main()
