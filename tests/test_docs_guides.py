"""F79: the user guides must stay in parity, resolvable and factual about install.

Documentation drifted a whole session behind the code, and the install section
described a command shape that does not exist (`uv tool install --extra gpu`), which
cost a live user a full install round. These tests are the cheap part of preventing a
repeat: they do not judge prose, they check the properties a reader depends on —

* every in-document anchor and every relative link actually resolves;
* the three languages carry the same numbered sections (a section added to `en` and
  forgotten in `ru`/`ja` fails here);
* the topics documented in this session are present in all three;
* the English files hold no Russian prose (quoted CLI output and the links to the
  translations are the deliberate exceptions);
* the wrong install form never comes back.
"""
from __future__ import annotations

import re
import unittest
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_GUIDE_DIR = _ROOT / "docs" / "guide"

GUIDES = {
    "en": _GUIDE_DIR / "user-guide.en.md",
    "ru": _GUIDE_DIR / "user-guide.ru.md",
    "ja": _GUIDE_DIR / "user-guide.ja.md",
}
READMES = {
    "en": _ROOT / "README.md",
    "ru": _ROOT / "README.ru.md",
    "ja": _ROOT / "README.ja.md",
}

_FENCE = re.compile(r"(?ms)^```.*?^```")
_INLINE_CODE = re.compile(r"(?s)`[^`]*`")
_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.*)$")
_SECTION = re.compile(r"(?m)^##\s+(\d+)\.\s+(.*)$")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]+")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def without_code(text: str) -> str:
    """Drop fenced blocks and inline code — the parts allowed to hold anything."""
    return _INLINE_CODE.sub("", _FENCE.sub("", text))


def slug(heading: str) -> str:
    """GitHub-style anchor: lowercase, punctuation dropped, spaces to hyphens.

    Deliberately unicode-aware (the ru/ja guides have Cyrillic and Japanese headings),
    and deliberately strict about punctuation: an em dash or a non-breaking hyphen in
    a heading disappears from the anchor, which is exactly the mismatch that used to
    leave the ru table of contents pointing at nothing.
    """
    out = []
    for ch in heading.strip().lower():
        if ch in "-_ " or unicodedata.category(ch)[0] in ("L", "N", "M"):
            out.append(ch)
    return "".join(out).replace(" ", "-")


def headings(text: str) -> list[str]:
    return [m.group(2) for m in _HEADING.finditer(without_code(text))]


def sections(text: str) -> list[tuple[int, str]]:
    """The numbered `## N. Title` sections, in file order."""
    return [(int(m.group(1)), m.group(2).strip()) for m in _SECTION.finditer(without_code(text))]


class TestLinksResolve(unittest.TestCase):
    """Anchors and relative paths a reader can click."""

    def test_in_document_anchors_resolve_to_a_heading(self):
        for lang, path in {**GUIDES, **READMES}.items():
            text = read(path)
            available = {slug(h) for h in headings(text)}
            for target in _LINK.findall(without_code(text)):
                if not target.startswith("#"):
                    continue
                with self.subTest(lang=lang, anchor=target):
                    self.assertIn(target[1:], available)

    def test_relative_links_point_at_existing_files(self):
        for lang, path in {**GUIDES, **READMES}.items():
            text = without_code(read(path))
            for target in _LINK.findall(text):
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                file_part, _, anchor = target.partition("#")
                if not file_part:
                    continue
                resolved = (path.parent / file_part).resolve()
                with self.subTest(lang=lang, link=target):
                    self.assertTrue(resolved.exists(), f"{path.name}: {target}")
                    if anchor and resolved.suffix == ".md":
                        available = {slug(h) for h in headings(read(resolved))}
                        self.assertIn(anchor, available, f"{path.name}: {target}")


class TestLanguageParity(unittest.TestCase):
    """A section added in one language has to appear in the other two."""

    def test_all_guides_have_the_same_section_numbers(self):
        numbering = {lang: [n for n, _ in sections(read(path))] for lang, path in GUIDES.items()}
        self.assertEqual(numbering["en"], numbering["ru"])
        self.assertEqual(numbering["en"], numbering["ja"])

    def test_section_numbers_are_a_gapless_sequence(self):
        for lang, path in GUIDES.items():
            with self.subTest(lang=lang):
                numbers = [n for n, _ in sections(read(path))]
                self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_the_table_of_contents_lists_every_section(self):
        """The contents list is §1 itself plus one line per remaining section."""
        for lang, path in GUIDES.items():
            with self.subTest(lang=lang):
                text = read(path)
                anchors = [t for t in _LINK.findall(without_code(text)) if t.startswith("#")]
                expected = [f"#{slug(f'{n}. {title}')}" for n, title in sections(text)][1:]
                self.assertEqual(anchors[:len(expected)], expected)


class TestDocumentedTopics(unittest.TestCase):
    """Everything this session added has to be findable in all three languages."""

    # Language-independent tokens: commands, env vars, config keys, pinned versions.
    REQUIRED = [
        # §3 installation — the resolved profiles
        "torch==2.13.0+cpu",
        "torch==2.13.0+cu130",
        "transformers==4.51.3",
        'uv tool install "C:\\path\\to\\sorta[cpu]"',
        'uv tool install "C:\\path\\to\\sorta[gpu]"',
        'uv tool install "C:\\path\\to\\sorta[gpu,vlm]"',
        'uv tool install "C:\\path\\to\\sorta[cpu,vlm]"',
        'uv tool install -e "C:\\path\\to\\sorta[gpu]"',
        "uv sync --extra gpu --extra dev",
        "tool.uv.conflicts",
        # §3.5/§3.6 doctor and the onnxruntime trap
        "sorta doctor",
        "CUDA available: yes",
        "places.tsv",
        "CUDAExecutionProvider",
        "python -m pip install --force-reinstall --no-deps onnxruntime-gpu",
        # §17 new commands
        "sorta index --refresh-exif",
        "sorta cache",
        "--clear",
        # §18 preview cache
        "preview_cache",
        "preview_dir",
        "preview_max_edge",
        "preview_quality",
        "SORTA_PREVIEW_CACHE",
        "SORTA_PREVIEW_DIR",
        "SORTA_PREVIEW_MAX_EDGE",
        "SORTA_PREVIEW_QUALITY",
        "150",  # KB per photo — the disk budget has to be stated
        # §19 run log
        "sorta.log",
        "stage=",
        "elapsed=",
        "SORTA_LOG_FILE",
        "SORTA_LOG_LEVEL",
        # §20 offline models
        "SORTA_ALLOW_MODEL_DOWNLOAD",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        # §21 new config fields
        "index.exif_workers",
        "naming.ocr_workers",
        "naming.clip.batch_size",
        "naming.clip.decode_workers",
    ]

    def test_every_topic_is_documented_in_every_language(self):
        for lang, path in GUIDES.items():
            text = read(path)
            for token in self.REQUIRED:
                with self.subTest(lang=lang, token=token):
                    self.assertIn(token, text)

    def test_readmes_point_at_doctor_and_the_spec_form_of_the_extra(self):
        for lang, path in READMES.items():
            text = read(path)
            with self.subTest(lang=lang):
                self.assertIn("sorta doctor", text)
                self.assertIn('uv tool install "C:\\path\\to\\sorta[gpu]"', text)
                self.assertIn("sorta cache", text)


class TestNoWrongInstallForm(unittest.TestCase):
    """`uv tool install` takes the extra in the package spec, never as a flag."""

    def test_no_document_shows_uv_tool_install_with_an_extra_flag(self):
        pattern = re.compile(r"uv tool install[^\n`]*--extra")
        for lang, path in {**GUIDES, **READMES}.items():
            with self.subTest(lang=lang):
                self.assertIsNone(pattern.search(read(path)))


class TestEnglishFilesStayEnglish(unittest.TestCase):
    """No Russian prose in the English files.

    Two deliberate exceptions, both already in the guide and both correct: the links
    to the translations, and quoted samples of CLI output (which is fixed Russian).
    Everything inside code fences and inline code is out of scope by construction.
    """

    def test_no_russian_prose_outside_code_and_quotes(self):
        for path in (READMES["en"], GUIDES["en"]):
            for number, line in enumerate(without_code(read(path)).splitlines(), 1):
                if not _CYRILLIC.search(line):
                    continue
                with self.subTest(file=path.name, line=number):
                    quoted = line.count('"') >= 2
                    translation_link = "Русский" in line
                    self.assertTrue(quoted or translation_link, line.strip())


if __name__ == "__main__":
    unittest.main()
