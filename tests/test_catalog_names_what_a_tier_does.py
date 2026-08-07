"""F223: a tier is named by what it DOES, and the wizard does not close over its answer.

The defect, in one line of the old catalog:

    Tier("search", weights=("ViT-L-14", "XLM-RoBERTa"), download_mb=3000)

ViT-L-14 is what the classification stage loads — the stage nobody is asked about, the
one without which screenshots, documents and product shots ride into the city folders
among the photographs. XLM-RoBERTa is what a search by words needs and nothing else. Glued
into one 3.0 GB line called "Search by words", they made a person who did not want to
search by words switch off the classification without a word being said. That is what
happened to the owner on 2026-08-07: almost nothing chosen in the wizard, and the run then
stopped on the verdicts because they went to fetch 1.6 GB by themselves.

What is pinned here, in the brief's own order:

1. no tier glues a model the run always needs to one it does not — the guard on the pair
   "what a stage loads" ↔ "what the tier is called", which is the defect itself;
2. choosing search by words brings in the tier it stands on, and the wizard SAYS so;
3. the default answer: yes for the tier the layout needs, no for every other;
4. a download that refuses leaves a working install, not a failed one;
5. weights already on the disk are not offered for downloading a second time;
6. the F216 watchdog still holds — every weight of the catalog is known to the probe;
7. the tier summary of `sorta-install.json` and the guides agree with the catalog;
8. every string of it exists in three languages.

And the second defect: the window closed over the answer. `sorta-setup` ends by holding a
console that belongs to it, and returns straight to the prompt in one that does not.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

from sorta import i18n, tiers, wizard

_LANGS = ("ru", "en", "ja")
_ROOT = Path(__file__).resolve().parent.parent
_GUIDES = {"ru": _ROOT / "docs" / "guide" / "user-guide.ru.md",
           "en": _ROOT / "docs" / "guide" / "user-guide.en.md",
           "ja": _ROOT / "docs" / "guide" / "user-guide.ja.md"}


def _load_build_script():
    """scripts/build_installer.py — a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "build_installer_f223", _ROOT / "scripts" / "build_installer.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nothing_cached() -> list[tiers.TierState]:
    """A machine with none of it — the state a fresh install is in."""
    return [tiers.TierState(tier.key,
                            missing_packages=tuple(f"extra:{e}" for e in tier.extras),
                            missing_weights=tier.weights)
            for tier in wizard.TIERS]


def everything_cached() -> list[tiers.TierState]:
    """A machine that has been through this once: packages and weights all in place."""
    return [tiers.TierState(tier.key) for tier in wizard.TIERS]


class Screen:
    """The console, and what the wizard was answered on it."""

    def __init__(self, answers: bool | None = None) -> None:
        self.lines: list[str] = []
        self.questions: list[tuple[str, bool]] = []
        self.answers = answers  # None — press Enter, i.e. take the tier's own default
        self.commands: list[list[str]] = []
        self.downloaded: list[str] = []
        self.download_ok = True

    def say(self, text: str) -> None:
        self.lines.append(text)

    def ask(self, question: str, default: bool = False) -> bool:
        self.questions.append((question, default))
        return default if self.answers is None else self.answers

    def doctor(self, config_path: str) -> None:
        pass

    def install(self, command) -> int:
        self.commands.append(list(command))
        return 0

    def download(self, tier, lang, config_path="config.yaml", *, say) -> bool:
        self.downloaded.append(tier.key)
        if not self.download_ok:
            say(i18n.cli_text("cli.setup.weights_failed", lang,
                              weights=", ".join(tier.weights), error="no network"))
        return self.download_ok

    def run(self, **kwargs) -> int:
        options = dict(manifest={}, states=nothing_cached(), say=self.say, ask=self.ask,
                       doctor=self.doctor, install=self.install, download=self.download)
        options.update(kwargs)
        return wizard.run_setup("en", **options)

    @property
    def said(self) -> str:
        return "\n".join(self.lines)


class TestNoTierIsNamedAfterOneOfTheThingsItCarries(unittest.TestCase):
    """Test 1 — the guard on the defect this feature exists for.

    It is deliberately not a test for "there is a tier called vision": that would pass
    the day somebody puts a second mandatory model back inside an optional tier. What it
    pairs is what a stage LOADS against what refusing a tier would cost.
    """

    def _always_needed(self) -> set[str]:
        return {weight for part in tiers.RUN_PARTS if not part.optional
                for weight in part.weights}

    def test_a_model_every_run_needs_never_shares_a_tier_with_an_optional_one(self):
        """The old `search` tier is exactly this failure: refusing it — a perfectly
        reasonable answer to "do you want to search by words?" — switched off the
        classification, which nobody is asked about at all."""
        always = self._always_needed()
        self.assertIn("ViT-L-14", always)  # the case, named
        for tier in wizard.TIERS:
            needed = [weight for weight in tier.weights if weight in always]
            if not needed:
                continue
            with self.subTest(tier=tier.key):
                self.assertEqual(list(tier.weights), needed,
                                 f"{tier.key}: refusing it would silently switch off "
                                 f"{needed}, which every run loads")

    def test_such_a_tier_is_offered_as_something_the_run_needs(self):
        """...and it is offered accordingly: the answer defaults to yes and the weights
        are fetched here, while somebody can see the progress and refuse in words."""
        for weight in self._always_needed():
            key = tiers.weight_tier(weight)
            with self.subTest(weight=weight):
                self.assertIsNotNone(key)
                tier = wizard.TIERS_BY_KEY[str(key)]
                self.assertTrue(tier.default_yes, f"{tier.key}: Enter has to mean yes")
                self.assertTrue(tier.preload, f"{tier.key}: fetched at the screen")

    def test_a_no_states_the_consequence_instead_of_warning_about_it(self):
        """The refusal has to stay possible and honest: the stage fetches the same
        1.6 GB on its first run, in the middle of a run and away from this screen."""
        for lang in _LANGS:
            with self.subTest(lang=lang):
                without = wizard.TIERS_BY_KEY["vision"].without(lang)
                self.assertIn("1,6" if lang == "ru" else "1.6", without)

    def test_the_two_models_are_priced_apart_and_add_up_to_what_was_measured(self):
        vision = wizard.TIERS_BY_KEY["vision"]
        search = wizard.TIERS_BY_KEY["search"]
        self.assertEqual(vision.weights, ("ViT-L-14",))
        self.assertEqual(search.weights, ("XLM-RoBERTa",))
        self.assertEqual(vision.download_mb, tiers.weights_size_mb(vision.weights))
        self.assertEqual(search.download_mb, tiers.weights_size_mb(search.weights))
        # The 3.0 GB line the two used to be: nothing was invented, it was split.
        self.assertEqual(vision.download_mb + search.download_mb, 3000)


class TestOneTierNeedingAnother(unittest.TestCase):
    """Test 2 — the first dependency in this catalog, and it is spoken out loud."""

    def test_search_by_words_stands_on_the_tier_that_encodes_the_pictures(self):
        self.assertEqual(wizard.TIERS_BY_KEY["search"].requires, ("vision",))

    def test_choosing_it_brings_the_required_tier_with_it(self):
        screen = Screen()
        screen.run(chosen=("search",))
        self.assertEqual(screen.downloaded, ["vision"])  # ...and it was fetched

    def test_the_wizard_says_it_rather_than_doing_it_quietly(self):
        screen = Screen()
        screen.run(chosen=("search",))
        self.assertIn(i18n.cli_text(
            "cli.setup.requires", "en", name=wizard.TIERS_BY_KEY["search"].name("en"),
            required=wizard.TIERS_BY_KEY["vision"].name("en")), screen.said)

    def test_a_tier_refused_by_hand_and_then_required_is_not_left_in_the_skipped_list(self):
        """Somebody answers no to the vision tier and yes to search by words. The tier
        comes back — the sentence above says so — and a summary that then reported it as
        skipped would contradict the same screen."""
        answers = iter([False, False, True])
        screen = Screen()
        screen.ask = lambda question, default=False: next(answers)  # type: ignore[method-assign]
        screen.run(tiers=(wizard.TIERS_BY_KEY["vision"], wizard.TIERS_BY_KEY["faces"],
                          wizard.TIERS_BY_KEY["search"]), chosen=None)
        vision = wizard.TIERS_BY_KEY["vision"].name("en")
        self.assertIn(i18n.cli_text("cli.setup.added", "en",
                                    names=f"{vision}, "
                                          f"{wizard.TIERS_BY_KEY['search'].name('en')}"),
                      screen.said)
        self.assertNotIn(i18n.cli_text("cli.setup.skipped", "en", names=vision),
                         screen.said)

    def test_a_requirement_already_on_the_disk_is_not_announced(self):
        """Nothing will be downloaded for it, so a line about it is noise — and this
        screen is the one that must not have any."""
        screen = Screen()
        screen.run(chosen=("search",), states=everything_cached())
        self.assertEqual(screen.downloaded, [])
        self.assertNotIn("does not work without", screen.said)

    def test_the_resolution_is_transitive_and_keeps_the_catalog_order(self):
        deep, search = wizard.TIERS_BY_KEY["deep"], wizard.TIERS_BY_KEY["search"]
        resolved, pulled = wizard.with_requirements([deep, search])
        self.assertEqual([tier.key for tier in resolved], ["vision", "search", "deep"])
        self.assertEqual([(a.key, b.key) for a, b in pulled], [("search", "vision")])
        # ...and nothing is pulled in twice when it was asked for by hand.
        resolved, pulled = wizard.with_requirements(
            [wizard.TIERS_BY_KEY["vision"], search])
        self.assertEqual([tier.key for tier in resolved], ["vision", "search"])
        self.assertEqual(pulled, ())


class TestWhatPressingEnterAnswers(unittest.TestCase):
    """Test 3 — the default answer, per tier."""

    def test_only_the_tier_the_layout_needs_defaults_to_yes(self):
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertEqual(tier.default_yes, tier.key == "vision")

    def test_pressing_enter_through_the_whole_wizard_takes_exactly_that_one(self):
        screen = Screen(answers=None)
        code = screen.run(chosen=None)
        self.assertEqual(code, 0)
        self.assertEqual(screen.downloaded, ["vision"])
        self.assertIn(i18n.cli_text("cli.setup.added", "en",
                                    names=wizard.TIERS_BY_KEY["vision"].name("en")),
                      screen.said)

    def test_the_question_shows_which_way_enter_goes(self):
        screen = Screen(answers=None)
        screen.run(chosen=None)
        asked = dict((question, default) for question, default in screen.questions)
        self.assertTrue(asked[i18n.cli_text("cli.setup.question_yes", "en")])
        self.assertFalse(asked[i18n.cli_text("cli.setup.question", "en")])
        self.assertIn("[Y/n]", i18n.cli_text("cli.setup.question_yes", "en"))

    def test_the_console_takes_enter_as_the_default_and_eof_as_a_no(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertTrue(wizard.ask_console("Add it?", True))
            self.assertFalse(wizard.ask_console("Add it?", False))
        for answer in ("n", "no", "нет"):
            with self.subTest(answer=answer):
                with mock.patch("builtins.input", return_value=answer):
                    self.assertFalse(wizard.ask_console("Add it?", True))
        # Nobody at the screen: the whole point of downloading here is that somebody is.
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(wizard.ask_console("Add it?", True))


class TestTheWizardFetchesTheWeightsItself(unittest.TestCase):
    """The third thing the catalog did not have: an install that downloads a model."""

    def test_the_progress_says_how_much_has_arrived(self):
        """1.6 GB with nothing on screen reads as a hang — it already did once, and it
        cost the owner an hour."""
        said: list[str] = []
        arrived = iter([0, 400 * 1_000_000, 900 * 1_000_000])

        def fetch(tier, config_path):
            pass

        with mock.patch.object(wizard.threading, "Thread", _StepThread):
            ok = wizard.download_weights(wizard.TIERS_BY_KEY["vision"], "en",
                                         say=said.append, fetch=fetch,
                                         measure=lambda: next(arrived), tick=0.0)
        self.assertTrue(ok)
        self.assertIn(i18n.cli_text("cli.setup.weights_progress", "en",
                                    done=wizard.human_size(400, "en"),
                                    size=wizard.human_size(1600, "en")), said)
        self.assertIn(i18n.cli_text("cli.setup.weights_ready", "en", weights="ViT-L-14",
                                    size=wizard.human_size(1600, "en")), said)

    def test_a_download_that_refuses_is_a_sentence_and_not_a_traceback(self):
        said: list[str] = []

        def fetch(tier, config_path):
            raise OSError("SSL: CERTIFICATE_VERIFY_FAILED")

        ok = wizard.download_weights(wizard.TIERS_BY_KEY["vision"], "en", say=said.append,
                                     fetch=fetch, measure=lambda: 0, tick=0.01)
        self.assertFalse(ok)
        self.assertIn(i18n.cli_text("cli.setup.weights_failed", "en", weights="ViT-L-14",
                                    error="SSL: CERTIFICATE_VERIFY_FAILED"), said)

    def test_test_4_a_refused_download_leaves_a_working_install(self):
        """The tier is not added, the exit code is zero, and the person is told that the
        stage will fetch the weights on its first run (which F222 announces)."""
        screen = Screen()
        screen.download_ok = False
        code = screen.run(chosen=("vision",))
        self.assertEqual(code, 0)
        self.assertEqual(screen.commands, [])
        self.assertNotIn(i18n.cli_text(
            "cli.setup.added", "en",
            names=wizard.TIERS_BY_KEY["vision"].name("en")), screen.said)
        self.assertIn("no network", screen.said)
        self.assertIn(i18n.cli_text("cli.setup.works_anyway", "en"), screen.said)

    def test_every_weight_a_preloading_tier_carries_can_actually_be_fetched(self):
        for tier in wizard.TIERS:
            if not tier.preload:
                continue
            for weight in tier.weights:
                with self.subTest(tier=tier.key, weight=weight):
                    self.assertIn(weight, wizard._FETCHERS)

    def test_a_weight_with_no_downloader_is_named_rather_than_ignored(self):
        with self.assertRaises(LookupError):
            wizard.fetch_weights(wizard.Tier("x", weights=("Qwen2.5-VL-3B",)))

    def test_the_model_fetched_is_the_one_the_config_will_load(self):
        with mock.patch.object(wizard, "_FETCHERS", {}):
            pass  # the table is a table; what follows is the pair it is asked for
        self.assertEqual(wizard.clip_weight_names("nowhere/config.yaml"),
                         ("ViT-L-14-quickgelu", "openai"))

    def test_a_config_that_names_another_model_is_honoured(self):
        from sorta.config import Config, NamingConfig

        configured = Config(naming=NamingConfig(clip_model="ViT-B-32",
                                                clip_pretrained="laion2b"))
        with mock.patch("sorta.config.load_config", return_value=configured):
            self.assertEqual(wizard.clip_weight_names("config.yaml"),
                             ("ViT-B-32", "laion2b"))

    def test_progress_is_measured_on_the_disk(self):
        with mock.patch.object(wizard, "hf_cache_dir",
                               return_value=Path("nowhere-at-all")):
            self.assertEqual(wizard.downloaded_bytes(), 0)

    def test_the_measurement_adds_up_the_files_of_the_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--timm--vit" / "blobs"
            root.mkdir(parents=True)
            (root / "part").write_bytes(b"x" * 2048)
            self.assertEqual(wizard.downloaded_bytes(Path(tmp)), 2048)


class _StepThread:
    """A thread that is alive for exactly one progress line, then done.

    The real one is `threading.Thread`; what the test needs is a deterministic number of
    turns through the loop, not a race with a sleep.
    """

    def __init__(self, target=None, daemon=None) -> None:
        self._target = target
        self._turns = 0

    def start(self) -> None:
        if self._target is not None:
            self._target()

    def join(self, timeout=None) -> None:
        self._turns += 1

    def is_alive(self) -> bool:
        return self._turns < 2


class TestWeightsAlreadyOnTheDisk(unittest.TestCase):
    """Test 5 — a reinstall over a full cache offers nothing to download twice."""

    def test_a_tier_that_is_in_place_is_not_offered(self):
        screen = Screen(answers=True)
        code = screen.run(chosen=None, states=everything_cached())
        self.assertEqual(code, 0)
        self.assertEqual(screen.questions, [])
        self.assertEqual(screen.downloaded, [])
        self.assertEqual(screen.commands, [])
        for tier in wizard.OPTIONAL_TIERS:
            with self.subTest(tier=tier.key):
                self.assertIn(i18n.cli_text("cli.setup.in_place", "en",
                                            name=tier.name("en")), screen.said)

    def test_the_summary_says_it_was_already_there_and_not_that_nothing_was_chosen(self):
        screen = Screen(answers=True)
        screen.run(chosen=None, states=everything_cached())
        self.assertIn(i18n.cli_text(
            "cli.setup.already", "en",
            names=", ".join(tier.name("en") for tier in wizard.OPTIONAL_TIERS)),
            screen.said)
        self.assertNotIn(i18n.cli_text("cli.setup.works_anyway", "en"), screen.said)

    def test_a_half_installed_tier_is_still_offered(self):
        """Packages in place and weights missing is neither installed nor absent — the
        state F216 built, and the one where a download is still owed."""
        states = [tiers.TierState(tier.key) for tier in wizard.TIERS
                  if tier.key != "vision"]
        states.append(tiers.TierState("vision", missing_weights=("ViT-L-14",)))
        screen = Screen(answers=True)
        screen.run(chosen=None, states=states)
        self.assertEqual(screen.downloaded, ["vision"])

    def test_the_probe_is_the_one_the_doctor_reads(self):
        """Not a second answer to "is it on disk" — the F216/F217 rule this inherits."""
        with mock.patch("sorta.tiers.tier_states", return_value=[]) as probe:
            self.assertEqual(wizard.probe_tiers(), [])
        probe.assert_called_once_with()


class TestTheWindowDoesNotCloseOverTheAnswer(unittest.TestCase):
    """The second defect of the brief: `sorta-setup` ended, and Windows took the screen.

    The owner chose a tier from the Start menu shortcut and the window shut — no summary,
    no error, nothing. Worse: until F221 the tier install failed on a certificate, and
    that message disappeared at exactly the same speed.
    """

    def test_a_console_of_our_own_is_held_until_it_is_read(self):
        said: list[str] = []
        waited: list[str] = []
        wizard.hold_console("en", say=said.append, wait=lambda prompt: waited.append(
            prompt) or "", owns=lambda: True)
        self.assertEqual(said, [i18n.cli_text("cli.setup.press_enter", "en")])
        self.assertEqual(waited, [""])

    def test_an_inherited_console_is_not_held(self):
        """`sorta-setup` typed into a terminal that was already open has to return to
        the prompt like any other command."""
        said: list[str] = []
        wizard.hold_console("en", say=said.append,
                            wait=lambda prompt: self.fail("waited in a terminal"),
                            owns=lambda: False)
        self.assertEqual(said, [])

    def test_a_closed_stdin_does_not_turn_the_pause_into_a_crash(self):
        wizard.hold_console("en", say=lambda text: None,
                            wait=mock.Mock(side_effect=EOFError), owns=lambda: True)

    def test_nothing_is_held_anywhere_but_windows(self):
        self.assertFalse(wizard.owns_console("posix"))

    def test_windows_asks_how_many_processes_share_the_console(self):
        """One process attached — the window was created for us and dies with us."""
        kernel32 = mock.Mock()
        kernel32.GetConsoleProcessList.return_value = 1
        with mock.patch.object(wizard, "os") as fake_os:
            fake_os.name = "nt"
            with mock.patch("ctypes.windll", mock.Mock(kernel32=kernel32), create=True):
                self.assertTrue(wizard.owns_console("nt"))
                kernel32.GetConsoleProcessList.return_value = 2
                self.assertFalse(wizard.owns_console("nt"))

    def test_a_machine_that_cannot_answer_is_not_paused_on(self):
        with mock.patch("ctypes.windll", None, create=True):
            self.assertFalse(wizard.owns_console("nt"))

    def test_the_entry_point_holds_the_console_whatever_happened(self):
        """Both paths: the ordinary end AND the one where the arguments were wrong. The
        second is the one that used to disappear fastest."""
        with mock.patch.object(wizard, "hold_console") as hold:
            with mock.patch.object(wizard, "run_setup", return_value=0):
                self.assertEqual(wizard.main(["--tiers", "none"]), 0)
            self.assertEqual(hold.call_count, 1)
            self.assertEqual(wizard.main(["--tiers", "telepathy"]), 2)
            self.assertEqual(hold.call_count, 2)

    def test_the_post_install_step_of_the_installer_runs_the_same_wizard(self):
        """The `[Run]` entry of sorta.iss goes through this module, so it inherits the
        hold above rather than needing one of its own (the .iss is not touched)."""
        iss = (_ROOT / "packaging" / "windows" / "sorta.iss").read_text(encoding="utf-8")
        run_section = iss.split("[Run]")[1]
        self.assertIn("sorta.wizard", run_section)


class TestTheCatalogIsStatedTheSameEverywhere(unittest.TestCase):
    """Test 7 — the manifest and the guides against the catalog, and test 6 unbroken."""

    def test_the_installer_manifest_states_every_field_a_tier_now_has(self):
        builder = _load_build_script()
        summary = {entry["key"]: entry for entry in builder.tier_summary()}
        self.assertEqual(set(summary), {tier.key for tier in wizard.TIERS})
        for tier in wizard.TIERS:
            with self.subTest(tier=tier.key):
                entry = summary[tier.key]
                self.assertEqual(entry["weights"], list(tier.weights))
                self.assertEqual(entry["download_mb"], tier.download_mb)
                self.assertEqual(entry["requires"], list(tier.requires))
                self.assertEqual(entry["default_yes"], tier.default_yes)
                self.assertEqual(entry["preload"], tier.preload)

    def test_every_weight_of_the_catalog_is_known_to_the_probe(self):
        """Test 6: the F216 watchdog, which the new tier must not break — a model whose
        name on disk nobody stated would be reported as missing while it is right there."""
        named = {weight for tier in wizard.TIERS for weight in tier.weights}
        self.assertEqual(named - set(tiers._WEIGHT_MARKERS), set())
        self.assertEqual(named - set(tiers._WEIGHT_MB), set())

    def test_the_guides_price_the_tiers_the_way_the_catalog_does(self):
        """A table that still says "search by words — 3 GB" is the old catalog written
        out in words, and it is the version most people read."""
        for lang, path in _GUIDES.items():
            with self.subTest(lang=lang):
                sizes = _guide_tier_sizes(path)
                expected = sorted(_normalised(wizard.human_size(tier.download_mb, lang))
                                  for tier in wizard.OPTIONAL_TIERS)
                self.assertEqual(sorted(sizes), expected)

    def test_the_guides_name_the_models_each_tier_carries(self):
        for lang, path in _GUIDES.items():
            table = _normalised(_guide_tier_table(_GUIDES[lang]))
            for tier in wizard.OPTIONAL_TIERS:
                for weight in tier.weights:
                    with self.subTest(lang=lang, weight=weight):
                        self.assertIn(_normalised(weight), table)

    def test_the_guides_say_that_this_one_download_happens_during_the_setup(self):
        """The wizard now downloads something, which no guide said before — and a person
        who is not told will read a 1.6 GB fetch as a stuck installer."""
        for lang, path in _GUIDES.items():
            with self.subTest(lang=lang):
                self.assertIn(_normalised(wizard.human_size(1600, lang)),
                              _normalised(path.read_text(encoding="utf-8")))


def _guide_tier_table(path: Path) -> str:
    """The rows of the tier table of a guide — the one the wizard section prints."""
    text = path.read_text(encoding="utf-8")
    for chunk in text.split("\n|---|---|---|---|\n")[1:]:
        rows = chunk.split("\n\n")[0]
        if "buffalo_l" in rows:
            return rows
    raise AssertionError(f"{path.name}: no tier table with the tiers in it")


def _normalised(text: str) -> str:
    """The same string as a comparison can use it, whichever guide it came from.

    The guides are typeset: non-breaking hyphens in `ViT‑L‑14`, a decimal comma in
    Russian, `~` / `約` in front of a size, and `7 GB` where the catalog formats `7.0 GB`.
    None of that is drift; the number and the name are what this compares.
    """
    for source, target in (("‑", "-"), ("‐", "-"), (" ", " "),
                           (",", "."), ("~", ""), ("≈", ""), ("約", "")):
        text = text.replace(source, target)
    return re.sub(r"\.0(?=\s*(GB|ГБ))", "", text).strip()


def _guide_tier_sizes(path: Path) -> list[str]:
    """The download column of that table, one entry per row."""
    sizes = []
    for row in _guide_tier_table(path).splitlines():
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) >= 2:
            sizes.append(_normalised(cells[1]))
    return sizes


class TestTheStringsExistInThreeLanguages(unittest.TestCase):
    """Test 8. A wizard that falls back to English on the one screen a person meets
    first is a wizard nobody trusts with the rest."""

    def test_every_sentence_this_feature_adds_is_translated(self):
        added = ("question_yes", "requires", "in_place", "already", "press_enter",
                 "weights_downloading", "weights_progress", "weights_ready",
                 "weights_failed", "tier.vision.name", "tier.vision.benefit",
                 "tier.vision.without")
        for name in added:
            key = f"cli.setup.{name}"
            with self.subTest(key=key):
                entry = i18n._CLI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang in _LANGS:
                    self.assertTrue(entry[lang].strip())

    def test_the_new_tier_is_named_by_what_it_gives_in_every_language(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                benefit = wizard.TIERS_BY_KEY["vision"].benefit(lang)
                marker = {"ru": "скриншот", "en": "screenshot", "ja": "スクリーン"}[lang]
                self.assertIn(marker, benefit)

    def test_the_search_tier_no_longer_claims_the_model_it_lost(self):
        """It carries the text tower and nothing else now; a benefit line still promising
        places by sight would send somebody to the wrong tier for it."""
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = (wizard.TIERS_BY_KEY["search"].benefit(lang)
                        + wizard.TIERS_BY_KEY["search"].without(lang))
                self.assertNotIn("ViT-L-14", text)


if __name__ == "__main__":
    unittest.main()
