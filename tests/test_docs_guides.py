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

F115 adds the other half of the same problem. The 2026-07-29 audit found not gaps but
false statements — a `ru` default language, a junk class list without `product`, a
superseded timing reference — so the cases below read each of those facts out of the
module that owns it (`config.Config`, `config.VlmConfig`, `scripts/check.py`) instead
of hard-coding the prose. A key added to `vlm:` fails the suite until it is written up.

F142 widens that principle from the config file to the command line. Twelve features
landed on 2026-08-01/02 and the guides stayed silent about every one of them, because
the only watchdog here read `config.yaml` keys: `sorta search` had one accidental
mention, `sorta album query` one, `--pets`, `--quality-scope` and `--preview-max-gb`
none at all. So `TestEveryCommandAndFlagIsDocumented` walks the application `cli.py`
builds and requires each command and each of its options to be named in all three
guides. The source is the built application on purpose — a list of commands kept in
this file would drift exactly the way the prose did.
"""
from __future__ import annotations

import dataclasses
import re
import subprocess
import unittest
import unicodedata
from pathlib import Path

from sorta import cli, config
from sorta.sorter import ALBUM_KINDS

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

# Every section of config.yaml and the object that owns its keys. `imaging:` is the odd
# one out and maps to a plain dict of env names: imaging.py is a leaf module that pool
# workers call with a path and nothing else, so that section has no config dataclass.
_CONFIG_SECTIONS: dict[str, object] = {
    "index": config.IndexConfig,
    "dedup": config.DedupConfig,
    "geo": config.GeoConfig,
    "events": config.EventsConfig,
    "faces": config.FacesConfig,
    "sort": config.SortConfig,
    "naming": config.NamingConfig,
    "features": config.FeaturesConfig,
    "vlm": config.VlmConfig,
    # F154: the detector's runtime — a second section of the `vlm:` kind, and it joins the
    # watchdog on the day it lands rather than at the next audit.
    "detect": config.DetectConfig,
    "imaging": config._IMAGING_ENV,
}

_FENCE = re.compile(r"(?ms)^```.*?^```")
_INLINE_CODE = re.compile(r"(?s)`[^`]*`")
_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.*)$")
_SECTION = re.compile(r"(?m)^##\s+(\d+)\.\s+(.*)$")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]+")
_QUOTED = re.compile(r"(?s)\u00AB.*?\u00BB|\"[^\"]*\"")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def without_code(text: str) -> str:
    """Drop fenced blocks and inline code — the parts allowed to hold anything."""
    return _INLINE_CODE.sub("", _FENCE.sub("", text))


def without_quotes(text: str) -> str:
    """Blank quoted spans, keeping newlines so line numbers still point home.

    Whole-text, not line-by-line: a quotation wraps, and a per-line reading sees an
    opening guillemet with no closing one and calls a citation a violation.
    """
    return _QUOTED.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


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


def cli_surface() -> list[tuple[str, list[str]]]:
    """Every command of the CLI as `("sorta faces label", ["--config", "-c"])`.

    Walked off the application `cli.build_app` returns rather than parsed out of the
    source: typer turns the decorated functions into a tree of commands, so a command
    or a flag added to `cli.py` appears here on the next run with nothing to update.
    Sub-applications (`faces`, `events`) are groups and recurse.

    The walk duck-types on `.commands` instead of importing click: typer 0.27 vendors
    click as `typer._click`, and `import click` fails in this environment.
    """
    import typer.main

    def walk(command: object, path: list[str]) -> list[tuple[str, list[str]]]:
        found = []
        for name, sub in sorted(getattr(command, "commands", {}).items()):
            found.extend(walk(sub, [*path, name]))
        opts: list[str] = []
        for param in command.params:  # type: ignore[attr-defined]
            opts.extend([*param.opts, *param.secondary_opts])
        # `--pets/--no-pets` is two option strings and both have to be documented: the
        # negative half is the one that switches OFF what config.yaml switched on.
        found.append((" ".join(path), sorted({o for o in opts if o.startswith("-")})))
        return found

    return walk(typer.main.get_command(cli.build_app("en")), ["sorta"])


class TestEveryCommandAndFlagIsDocumented(unittest.TestCase):
    """F142: the terminal surface, checked against `cli.py` rather than against a list.

    A command that exists and is documented nowhere is invisible: `sorta search` shipped
    and the only mention of it in 1,600 lines of guide was one a worker had added in
    passing. These cases are cheap to satisfy (a line in §16 counts) and impossible to
    satisfy by accident, which is the point — the next command is caught the day it
    lands, not at the next audit.
    """

    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback, no typer
            self.skipTest("typer is not installed")
        self.surface = cli_surface()

    def test_the_walk_actually_found_the_command_line(self):
        """A green suite must not be the result of an empty walk.

        Everything below is a loop over what `cli_surface` returned, so a typer release
        that renames `.commands` or `.params` would silently check nothing at all.
        """
        names = {name for name, _opts in self.surface}
        self.assertIn("sorta search", names)
        self.assertIn("sorta faces label", names)
        self.assertGreaterEqual(len(names), 15)
        flags = {opt for _name, opts in self.surface for opt in opts}
        # Two flags of two different commands, so an empty walk cannot pass and neither
        # can one that stops at the top level. `--quality-scope` stood here until F186
        # retired the question it chose a population for.
        self.assertIn("--preview-max-gb", flags)
        self.assertIn("--no-pets", flags)

    # `assertTrue` rather than `assertIn`: the container here is a 1,600-line guide, and
    # a failure that prints all of it buries the one word it is about.
    def test_every_command_is_named_in_every_guide(self):
        for name, _opts in self.surface:
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, command=name):
                    self.assertTrue(name in read(path),
                                    f"{path.name}: undocumented command `{name}`")

    def test_every_flag_is_named_in_every_guide(self):
        for name, opts in self.surface:
            for opt in opts:
                for lang, path in GUIDES.items():
                    with self.subTest(lang=lang, command=name, flag=opt):
                        self.assertTrue(
                            opt in read(path),
                            f"{path.name}: undocumented flag {opt} of `{name}`")

    def test_every_album_kind_is_named_in_every_guide(self):
        """The kinds are positional arguments, so the flag loop above cannot see them.

        `animal` and `query` are exactly the pair this case exists for: they arrived with
        F123/F129 and the guides described albums as "one person or one event" for the
        whole week after.
        """
        for kind in ALBUM_KINDS:
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, kind=kind):
                    self.assertTrue(f"album {kind}" in read(path),
                                    f"{path.name}: undocumented album kind {kind!r}")


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
        # F115 §8 — the deep tier is paid for once, and this column is the reason why
        "media_class.tier",
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


class TestGuidesAgreeWithTheCode(unittest.TestCase):
    """F115: the claims that had drifted — checked against the modules, not by eye.

    The 2026-07-29 audit found the guides asserting things the code had stopped doing
    a day earlier (a `ru` default language, a CLI that ignores `language`, a junk
    class list without `product`). Prose can drift again; what these cases pin is the
    pairing — every fact below is read out of the module that owns it, so the guide
    and the code cannot disagree silently.
    """

    def test_the_guides_do_not_promise_ru_as_the_default_language(self):
        """`config.py` defaults to `en`; the guides used to promise `ru`."""
        self.assertEqual(config.Config().language, "en")
        wrong = re.compile(r"(?:default|по умолчанию|既定)[ 　]*`?ru\b")
        for lang, path in {**GUIDES, **READMES}.items():
            with self.subTest(lang=lang):
                found = wrong.search(read(path))
                self.assertIsNone(found, found.group(0) if found else "")

    def test_every_junk_verdict_the_sorter_routes_is_documented(self):
        """A class that gets its own layout branch has to be named in the guides.

        `product` is the one this test was written for: it is every tenth frame of a
        deep-tier run and a review folder of its own, and the guides listed four
        classes for a long time after it appeared.
        """
        for verdict in ("photo", "screenshot", "meme", "document", "product"):
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, verdict=verdict):
                    self.assertIn(f"`{verdict}`", read(path))

    def test_no_russian_output_is_quoted_in_the_other_guides(self):
        """F118: a sample of CLI output in the en/ja guides must be in that language.

        Until F112 the CLI spoke only Russian, so quoting Russian output was accurate
        and both guides did it — around thirty samples each, plus a glossary translating
        the words. F112 made the output follow `language:` and defaulted it to `en`,
        which turned every one of those samples into a statement about behaviour that no
        longer exists. This is the F115 failure mode exactly: not a gap, a false
        statement, and a reader has no way to tell.

        Fenced blocks only. Cyrillic in prose is legitimate — the links to the Russian
        translation, and the layout folder names (`Россия/…`), which are data produced by
        `language: ru` rather than chrome.
        """
        for lang, path in (("en", GUIDES["en"]), ("ja", GUIDES["ja"])):
            offenders = []
            in_fence = False
            for number, line in enumerate(read(path).splitlines(), 1):
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence and _CYRILLIC.search(line):
                    offenders.append(f"{number}: {line.strip()}")
            with self.subTest(lang=lang):
                self.assertEqual(offenders, [], f"{path.name}: {offenders}")

    def test_every_configuration_key_is_documented(self):
        """Every key of every section, read off the class that owns it.

        This began as a watchdog over `vlm:` alone, written after an audit found not
        gaps but false statements. Watching one section turned out to be the weakness
        rather than the design: `imaging.preview_cache_max_gb` was added and all three
        guides stayed silent, so the suite was green while the documentation described a
        cache with no ceiling. Widening it then showed the real backlog — 51 keys across
        nine sections had never been written up at all, `features:` (F113) among them,
        in full.

        The check is the dotted form on purpose. A bare `preview_quality` inside a YAML
        block is easy to write and impossible to search for; `imaging.preview_quality`
        is what a reader greps and what the rest of the guide cites.
        """
        for section, owner in _CONFIG_SECTIONS.items():
            keys = ([f.name for f in dataclasses.fields(owner)]
                    if dataclasses.is_dataclass(owner) else list(owner))
            for key in keys:
                for lang, path in GUIDES.items():
                    with self.subTest(lang=lang, key=f"{section}.{key}"):
                        self.assertIn(f"{section}.{key}", read(path))

    def test_the_legacy_naming_aliases_are_documented(self):
        """A live config.yaml still holds the old keys — the guides have to say so."""
        for legacy in ("naming.vlm_enabled", "naming.classify_vlm_model",
                       "naming.vlm_workers"):
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, key=legacy):
                    self.assertIn(legacy, read(path))

    def test_the_detector_defaults_are_quoted_as_they_are_configured(self):
        """The guides state each default in a column of its own, and a stale one there is
        worse than none: F162 re-measured both detector numbers on 500 frames and moved
        them (0.5 -> 0.6, 2 000 -> 4 000), so the column is read off the class that owns
        them rather than trusted to have been edited along with the prose.
        """
        features = config.FeaturesConfig()
        for key in ("detector_candidates", "detector_threshold"):
            head = f"| `features.{key}` |"
            for lang, path in GUIDES.items():
                rows = [line for line in read(path).splitlines()
                        if line.startswith(head)]
                with self.subTest(lang=lang, key=key):
                    self.assertEqual(len(rows), 1)
                    self.assertTrue(rows[0].startswith(
                        f"{head} `{getattr(features, key)}` |"), rows[0][:140])

    def test_the_superseded_timings_do_not_come_back(self):
        """The 6,298-photo reference run was replaced by the 24,196-photo measurement."""
        stale = re.compile(r"6[ ,]?298")
        for lang, path in {**GUIDES, **READMES}.items():
            with self.subTest(lang=lang):
                self.assertIsNone(stale.search(read(path)))

    def test_no_retired_key_is_still_documented(self):
        """The other direction of the watchdog above, and the one F186 needed.

        `test_every_configuration_key_is_documented` walks the dataclasses, so a key that
        LEFT them is invisible to it: the guides went on describing `vlm.quality`, the
        scope that chose who it was asked of and the comparative keeper question for as
        long as anybody cared to read them. A documented key that does not exist is worse
        than an undocumented one — it is a setting a person writes into their config.yaml
        and then waits for something to happen.

        `config.example.yaml` is checked with them: it is the file people copy, and a
        retired key sitting in it would be written into every new config in the world.
        The schema is checked with them too — its column comments are the field list, and
        `frame_quality.eyes_open` was described there as a column a live key still wrote.
        """
        retired = ("vlm.quality", "vlm.quality_scope", "dedup.keeper_vlm",
                   "estimate.keeper_call_sec", "estimate.keeper_frame_sec",
                   "quality_scope:", "keeper_vlm:", "keeper_call_sec:",
                   "keeper_frame_sec:", "--quality-scope", "--no-quality")
        example = _ROOT / "config.example.yaml"
        schema = _ROOT / "sorta" / "db" / "schema.sql"
        for lang, path in {**GUIDES, "example": example, "schema": schema}.items():
            text = read(path)
            for key in retired:
                with self.subTest(lang=lang, key=key):
                    self.assertNotIn(key, text)

    def test_the_keys_that_outlived_the_retired_ones_are_still_documented(self):
        """The other half — a watchdog that passed on an emptied section would be worse
        than none. These three sat next to what F186 removed and are still read."""
        example = read(_ROOT / "config.example.yaml")
        for key in ("dedup.keeper_max_frames", "dedup.keeper_min_group_size",
                    "vlm.max_edge"):
            for lang, path in GUIDES.items():
                with self.subTest(lang=lang, key=key):
                    self.assertIn(key, read(path))
        for key in ("keeper_max_frames:", "keeper_min_group_size:",
                    "measurement_max_age_days:"):
            with self.subTest(key=key, file="config.example.yaml"):
                self.assertIn(key, example)


class TestContributingDescribesTheGate(unittest.TestCase):
    """The gate script grew two halves; CONTRIBUTING has to describe the one that runs."""

    def test_both_halves_are_documented_and_exist(self):
        gate = read(_ROOT / "scripts" / "check.py")
        contributing = read(_ROOT / "CONTRIBUTING.md")
        for flag in ("--fast", "--slow"):
            with self.subTest(flag=flag):
                self.assertIn(f'"{flag}"', gate)
                self.assertIn(f"scripts/check.py {flag}", contributing)


def english_documents() -> list[Path]:
    """Every published English document, found rather than listed.

    A hand-written list is what let this rule be true and useless at once: it named the
    README and the guide, so two whole sections of `DECISIONS.md` were published in
    Russian (2026-08-09, found by the owner reading the file).
    """
    tracked = subprocess.run(["git", "ls-files", "*.md", "*.yaml"], cwd=_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    return [_ROOT / name for name in tracked
            if not name.endswith((".ru.md", ".ja.md")) and "data/geo/" not in name]


class TestEnglishFilesStayEnglish(unittest.TestCase):
    """No Russian prose in the English files.

    Every deliberate exception is a form of QUOTATION: the links to the translations,
    CLI output, folder names, search words a measurement used. Code fences and inline
    code are out of scope by construction.
    """

    def test_the_search_finds_the_documents_it_is_meant_to_judge(self):
        """Guards the guard: a glob matching nothing would pass every case below."""
        found = {path.name for path in english_documents()}
        for expected in ("README.md", "DECISIONS.md", "ARCHITECTURE.md", "CHANGELOG.md",
                         "CONTRIBUTING.md", "user-guide.en.md", "landmarks.yaml"):
            self.assertIn(expected, found)

    def test_no_russian_prose_outside_code_and_quotes(self):
        for path in english_documents():
            text = without_quotes(without_code(read(path)))
            for number, line in enumerate(text.splitlines(), 1):
                if not _CYRILLIC.search(line) or "Русский" in line:
                    continue
                with self.subTest(file=path.name, line=number):
                    self.fail(line.strip())


if __name__ == "__main__":
    unittest.main()
