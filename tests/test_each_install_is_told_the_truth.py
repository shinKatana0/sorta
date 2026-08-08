"""F230: every install is told the truth about itself, and the watchdog that keeps it so.

Two paths are legitimate and neither may be broken to fix the other:

    a checkout of the sources      a developer, `uv`, commands like `uv sync --extra gpu`
    an installed copy              the `sorta-setup` wizard, an item in the Start menu

Three lines of advice were being written for whichever path their author happened to be
on, and all three were found by the owner in a virtual machine over two days:
`diagnostics._FIX_PROFILE` printed `uv sync --extra gpu --extra dev` to a copy that has no
project directory; the help of `--deep` named `uv sync --extra vlm` (the web app's version
of the same sentence had been fixed by F217 a release earlier); and the tier hint chose
BY OPERATING SYSTEM — `"cli.doctor.tier_hint" if os_name == "nt"` — so a developer on
Windows was sent to a Start menu item that only an installed copy has.

**The watchdog is the feature.** Three cases caught one at a time by a person in two days
means a fourth is already being written, so the check here is about FILES: every string of
the catalog that names a command has to be part of a family chosen by install kind
(`install.INSTALL_ADVICE`), or be justified by hand below. Same shape as F228's `ast` walk
over every subprocess launch and F218's payload guard — a check that does not let a fifth
case in silently.

The rest pins the behaviour the guard cannot see: that BOTH variants survive (the
developer's `uv sync` stays where it is true — the owner says outright that the repository
path is the one he prefers), that the wizard asks about the acceleration tier knowing what
card is in the machine, and that `doctor` says which profile won afterwards.
"""
from __future__ import annotations

import inspect
import string
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import cli, diagnostics, i18n, install, tiers, wizard
from sorta.i18n import _CLI_STRINGS

_LANGS = ("ru", "en", "ja")


# --- what counts as an instruction a person can paste ----------------------------------
#
# Deliberately about COMMANDS and the one place a command hides behind a picture (the
# Start menu item, which is `sorta-setup` with a mouse). A sentence with none of these in
# it describes a state; a sentence with one of them tells somebody to do something, and
# whether they CAN depends on which install they have.
_COMMAND_MARKERS = (
    "uv ", "pip ", "winget ", "brew ", "apt ", "dnf ", "sorta-setup",
    "меню «Пуск»", "Start menu", "スタートメニュー",
)


def command_markers(text: str) -> tuple[str, ...]:
    return tuple(marker for marker in _COMMAND_MARKERS if marker in text)


def advice_strings() -> dict[str, tuple[str, ...]]:
    """Every key of the CLI catalog that names a command, with the markers found in it."""
    found: dict[str, tuple[str, ...]] = {}
    for key, entry in _CLI_STRINGS.items():
        markers = tuple(sorted({marker for text in entry.values()
                                for marker in command_markers(text)}))
        if markers:
            found[key] = markers
    return found


def in_a_family(key: str) -> bool:
    """Is this key one of the three variants `install.advice_key` can return?"""
    return any(key in install.advice_keys(base) for base in install.INSTALL_ADVICE)


# The command-bearing strings that are NOT install-path advice, and why. Every entry is
# pinned in both directions below: a key that stops naming a command has to leave this
# list, or the list stops describing the catalog and starts excusing it.
_NOT_INSTALL_ADVICE: dict[str, str] = {
    "cli.doctor.exiftool_linux":
        "exiftool is a THIRD-PARTY program, and which package manager installs it is a "
        "property of the operating system, not of our install: `apt`/`dnf` is the right "
        "answer for a checkout and for an installed copy alike on that machine. Chosen by "
        "`cli._exiftool_hint_key(sys.platform)`, which is the one signal that legitimately "
        "stays the platform.",
    "cli.doctor.exiftool_windows":
        "the same, for Windows — `winget install OliverBetz.ExifTool` is what a machine "
        "with winget needs whichever way Sorta got onto it.",
    "cli.doctor.exiftool_macos":
        "the same, for macOS — `brew install exiftool`, chosen by platform for the same "
        "reason.",
    "cli.download.failed":
        "`sorta-setup` here is not a way to install PACKAGES (which is what differs "
        "between the three installs) but a way to fetch model WEIGHTS in advance, and it "
        "is a console script of the wheel on all three paths — `uv run sorta-setup` in a "
        "checkout, the command itself after `uv tool install`, the Start menu item on an "
        "installed copy. There is nothing to choose between.",
}


class TestTheWatchdog(unittest.TestCase):
    """About files: a fifth case cannot be added silently."""

    def test_every_advice_that_names_a_command_is_chosen_by_install_kind(self):
        unaccounted = {key: markers for key, markers in advice_strings().items()
                       if not in_a_family(key) and key not in _NOT_INSTALL_ADVICE}
        self.assertEqual(
            unaccounted, {},
            "these strings tell somebody to run something, and they are reachable "
            "without a choice by install kind — so one of the two paths is being given "
            "the other one's command. Split the key into "
            "`<base>.checkout` / `<base>.installed` / `<base>.tool`, register the base in "
            "`install.INSTALL_ADVICE`, and reach it through `install.advice_key`. If it "
            "genuinely is not install-path advice, say why in _NOT_INSTALL_ADVICE.")

    def test_nothing_on_the_exemption_list_is_stale(self):
        """The other direction — see F228's `_NOT_THROUGH_THE_HELPER`."""
        naming = advice_strings()
        for key, reason in _NOT_INSTALL_ADVICE.items():
            with self.subTest(key=key):
                self.assertIn(key, naming,
                              f"{key} no longer names a command — delete its entry "
                              f"({reason})")

    def test_every_exemption_carries_a_reason(self):
        for key, reason in _NOT_INSTALL_ADVICE.items():
            with self.subTest(key=key):
                self.assertGreater(len(reason), 60, reason)

    def test_the_watchdog_is_reading_a_real_catalog(self):
        """A scanner pointed at nothing finds nothing and looks exactly like a green gate
        — the failure mode of every check that walks a collection."""
        self.assertGreater(len(_CLI_STRINGS), 200)
        self.assertGreaterEqual(len(advice_strings()), len(_NOT_INSTALL_ADVICE) + 6)

    def test_the_watchdog_goes_red_on_a_new_case(self):
        """A check nobody has seen fail is not a check (F182, F216). A fifth line with a
        command in it, added the old way, has to be found."""
        key = "cli.f230.new_case"
        _CLI_STRINGS[key] = {  # type: ignore[assignment]
            lang: "To fix it: uv sync --extra gpu" for lang in _LANGS}
        try:
            self.assertIn(key, advice_strings())
            self.assertFalse(in_a_family(key))
            self.assertNotIn(key, _NOT_INSTALL_ADVICE)
        finally:
            del _CLI_STRINGS[key]

    def test_a_sentence_about_a_state_is_not_flagged(self):
        """The guard must stay quiet about lines that describe rather than instruct, or it
        becomes noise and noise gets switched off."""
        for text in ("Command sorta: not on PATH",
                     "torch: 2.13.0+cu130 (CUDA available: yes)",
                     "`sorta doctor` prints what the install actually became.",
                     "Ярусы установки:"):
            with self.subTest(text=text):
                self.assertEqual(command_markers(text), ())


class TestEveryFamilyIsWholeAndDistinct(unittest.TestCase):
    """A family is three keys, and the whole point is that they differ."""

    def test_all_three_variants_exist_for_every_registered_family(self):
        for base in install.INSTALL_ADVICE:
            for key in install.advice_keys(base):
                with self.subTest(key=key):
                    self.assertIn(key, _CLI_STRINGS)
                    for lang in _LANGS:
                        self.assertTrue(_CLI_STRINGS[key][lang].strip())

    def test_the_checkout_variant_keeps_the_uv_command(self):
        """Requirement 4 of the brief: the developer's path must not be broken to fix the
        others. `uv` has to stay where it is true, and this is the half of the guard that
        says so — the previous fixes REPLACED a command instead of choosing between two."""
        for base in install.INSTALL_ADVICE:
            key = install.advice_key(base, install.KIND_CHECKOUT)
            with self.subTest(key=key):
                self.assertIn("uv ", _CLI_STRINGS[key]["en"])

    def test_no_uv_command_reaches_an_installed_copy(self):
        """There is no project directory there and no `uv sync` to run in it."""
        for base in install.INSTALL_ADVICE:
            key = install.advice_key(base, install.KIND_INSTALLED)
            with self.subTest(key=key):
                for lang in _LANGS:
                    self.assertNotIn("uv ", _CLI_STRINGS[key][lang])

    def test_only_an_installed_copy_is_sent_to_the_start_menu(self):
        """The defect in one assertion: the Start menu is the Windows installer's, and it
        was being offered by OPERATING SYSTEM to whoever ran on Windows."""
        menus = ("Start menu", "меню «Пуск»", "スタートメニュー")
        for base in install.INSTALL_ADVICE:
            for kind in (install.KIND_CHECKOUT, install.KIND_TOOL):
                key = install.advice_key(base, kind)
                with self.subTest(key=key):
                    for lang in _LANGS:
                        for menu in menus:
                            self.assertNotIn(menu, _CLI_STRINGS[key][lang])

    def test_a_checkout_and_an_installed_copy_never_get_the_same_sentence(self):
        """The defect class in one assertion. Two of the three variants MAY coincide — a
        tool install and an installed copy both add a tier with `sorta-setup`, and pretending
        otherwise would be inventing a difference — but the developer's path and the
        installer's path cannot, because their commands do not exist on each other's
        machines."""
        for base in install.INSTALL_ADVICE:
            with self.subTest(base=base):
                for lang in _LANGS:
                    checkout = _CLI_STRINGS[install.advice_key(
                        base, install.KIND_CHECKOUT)][lang]
                    installed = _CLI_STRINGS[install.advice_key(
                        base, install.KIND_INSTALLED)][lang]
                    self.assertNotEqual(checkout, installed)

    def test_every_variant_of_a_family_takes_the_same_fields(self):
        """`advice_key` returns a name and the caller formats it with one set of fields —
        a placeholder in one variant only would raise on one install kind only."""
        def fields(template: str) -> set[str]:
            return {name for _, name, _, _ in string.Formatter().parse(template) if name}

        for base in install.INSTALL_ADVICE:
            sizes = {len(fields(_CLI_STRINGS[key]["en"]))
                     for key in install.advice_keys(base)}
            with self.subTest(base=base):
                # Either nobody takes a field or they all take the same number of them;
                # `command_hint.installed` names the interpreter and the others do not,
                # which is why this is a bound and not an equality (the caller passes the
                # path unconditionally and `str.format` ignores what a template omits).
                self.assertLessEqual(max(sizes), 1)


class TestWhichInstallIsThis(unittest.TestCase):
    """One answer, and the OS is not part of it."""

    def test_a_manifest_means_an_installed_copy(self):
        self.assertEqual(install.install_kind({"python": "python.exe"}),
                         install.KIND_INSTALLED)

    def test_sources_above_the_package_mean_a_checkout(self):
        self.assertEqual(install.install_kind({}), install.KIND_CHECKOUT)

    def test_no_manifest_and_no_sources_mean_a_wheel(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(
                install.install_kind({}, package=Path(empty) / "site-packages" / "sorta"),
                install.KIND_TOOL)

    def test_the_suite_itself_runs_in_a_checkout(self):
        """The property the golden outputs of the other test files rely on."""
        self.assertEqual(install.install_kind({}), install.KIND_CHECKOUT)

    def test_the_operating_system_is_not_an_input_at_all(self):
        """The whole diagnosis of part A: `os.name` decided this, and the fix is not a
        better use of that signal — it is that the signal is gone. Asserted on the
        signatures, because a patch of `os.name` cannot prove the absence of a read (and
        breaks pathlib on Windows while trying)."""
        for function in (install.install_kind, install.advice_key, tiers._tier_hint_key):
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                self.assertNotIn("os_name", parameters)
                self.assertNotIn("platform", parameters)
        self.assertNotIn("os.name", inspect.getsource(install.install_kind))

    def test_every_kind_composes_a_key(self):
        self.assertEqual(install.advice_keys("cli.doctor.tier_hint"),
                         ("cli.doctor.tier_hint.checkout",
                          "cli.doctor.tier_hint.installed",
                          "cli.doctor.tier_hint.tool"))

    def test_an_explicit_kind_is_not_re_probed(self):
        with mock.patch.object(install, "manifest_path") as probe:
            self.assertEqual(install.advice_key("cli.doctor.tier_hint",
                                                install.KIND_TOOL),
                             "cli.doctor.tier_hint.tool")
        probe.assert_not_called()


class TestTheThreeKnownPlaces(unittest.TestCase):
    """The three lines the owner met, each read out of the module that prints it."""

    def test_the_tier_hint_follows_the_install_and_not_the_os(self):
        self.assertEqual(tiers._tier_hint_key(install.KIND_CHECKOUT),
                         "cli.doctor.tier_hint.checkout")
        self.assertEqual(tiers._tier_hint_key(install.KIND_INSTALLED),
                         "cli.doctor.tier_hint.installed")
        self.assertEqual(tiers._tier_hint_key(install.KIND_TOOL),
                         "cli.doctor.tier_hint.tool")

    def test_a_windows_checkout_is_not_sent_to_the_start_menu(self):
        """The acceptance criterion, word for word: a checkout on Windows with a card is
        advised `uv sync --extra gpu` and hears nothing about the Start menu."""
        with mock.patch("os.name", "nt"):
            text = i18n.cli_text(tiers._tier_hint_key(install.KIND_CHECKOUT), "en")
        self.assertIn("uv sync --extra gpu", text)
        self.assertNotIn("Start menu", text)

    def test_an_installed_copy_is_not_offered_uv(self):
        text = i18n.cli_text(tiers._tier_hint_key(install.KIND_INSTALLED), "en")
        self.assertIn("sorta-setup", text)
        self.assertNotIn("uv ", text)

    def test_the_deep_flag_names_a_command_this_install_has(self):
        checkout = i18n.cli_text(
            "cli.help.run.deep", "en",
            how=i18n.cli_text(install.advice_key("cli.help.run.deep_how",
                                                 install.KIND_CHECKOUT), "en"))
        installed = i18n.cli_text(
            "cli.help.run.deep", "en",
            how=i18n.cli_text(install.advice_key("cli.help.run.deep_how",
                                                 install.KIND_INSTALLED), "en"))
        self.assertIn("uv sync --extra vlm", checkout)
        self.assertNotIn("uv ", installed)
        self.assertIn("Sorta setup", installed)

    def test_the_gpu_repair_command_is_the_one_this_install_can_run(self):
        self.assertEqual(diagnostics.fix_profile(install.KIND_CHECKOUT),
                         "uv sync --extra gpu --extra dev")
        for kind in (install.KIND_INSTALLED, install.KIND_TOOL):
            with self.subTest(kind=kind):
                self.assertIn("sorta-setup", diagnostics.fix_profile(kind))
                self.assertNotIn("uv ", diagnostics.fix_profile(kind))

    def test_an_installed_copy_is_never_told_to_run_pip(self):
        """It is a `uv pip install --target` tree: there is no pip in it, and the tier the
        wizard installs with `--reinstall` is the same repair done by the thing that knows
        where the packages are."""
        self.assertIn("pip ", diagnostics.fix_ort(install.KIND_CHECKOUT))
        for kind in (install.KIND_INSTALLED, install.KIND_TOOL):
            with self.subTest(kind=kind):
                self.assertNotIn("pip ", diagnostics.fix_ort(kind))

    def test_the_problem_sentences_carry_the_right_fix(self):
        for kind in install.KINDS:
            health = diagnostics.GpuHealth(
                torch_version="2.13.0+cpu", torch_cuda_available=False,
                torch_device_name=None, ort_providers=("CPUExecutionProvider",),
                gpu_present=True, install_kind=kind)
            with self.subTest(kind=kind):
                self.assertEqual(len(health.problems), 2)
                self.assertIn(diagnostics.fix_profile(kind), health.problems[0])
                self.assertIn(diagnostics.fix_ort(kind), health.problems[1])
                self.assertIn(kind, health.summary)

    def test_the_doctor_answers_once_and_hands_the_answer_on(self):
        """One reading of the manifest per command: a second probe is how an output starts
        disagreeing with itself (the F225 lesson, one screen over)."""
        lines = cli._doctor_install_lines(
            "en", command=None, scripts=cli.Path("scripts"), exiftool="exiftool",
            installed_python="python.exe", kind=install.KIND_INSTALLED)
        self.assertIn("python.exe", "\n".join(lines))
        self.assertNotIn("uv ", "\n".join(lines))
        checkout = cli._doctor_install_lines(
            "en", command=None, scripts=cli.Path("scripts"), exiftool="exiftool",
            kind=install.KIND_CHECKOUT)
        self.assertIn("uv run sorta", "\n".join(checkout))


class TestTheWizardKnowsAboutTheCard(unittest.TestCase):
    """Part B: the person with a card does not have to guess about acceleration."""

    def _card(self, **fields) -> diagnostics.NvidiaCard:
        base = {"name": "NVIDIA GeForce RTX 4080", "driver": "581.15",
                "present": True, "probed": True}
        return diagnostics.NvidiaCard(**{**base, **fields})

    def test_a_card_with_a_current_driver_is_usable(self):
        self.assertTrue(self._card().usable)
        self.assertEqual(self._card().driver_state, diagnostics.DRIVER_OK)

    def test_an_old_driver_is_not_usable_and_says_so(self):
        card = self._card(driver="551.86")
        self.assertFalse(card.usable)
        self.assertEqual(card.driver_state, diagnostics.DRIVER_OLD)
        line = wizard.card_line(card, "en")
        self.assertIn("551.86", line)
        self.assertIn("580", line)
        self.assertIn("driver", line)

    def test_a_driver_that_cannot_be_read_is_not_treated_as_old(self):
        """nvidia-smi answered, so the card works — refusing the tier over a parsing
        failure would leave an RTX machine on the CPU for nothing."""
        card = self._card(driver=None)
        self.assertEqual(card.driver_state, diagnostics.DRIVER_UNKNOWN)
        self.assertTrue(card.usable)

    def test_no_card_names_the_reason_and_the_size_it_saves(self):
        line = wizard.card_line(diagnostics.NvidiaCard(probed=True), "en")
        self.assertIn("2.5 GB", line)
        self.assertIn("processor", line)

    def test_the_card_is_named_in_all_three_languages(self):
        for lang in _LANGS:
            for card in (self._card(), self._card(driver="551.86"),
                         diagnostics.NvidiaCard(probed=True)):
                with self.subTest(lang=lang, driver=card.driver, present=card.present):
                    line = wizard.card_line(card, lang)
                    self.assertTrue(line.strip())
                    self.assertNotIn("{", line)

    def test_the_probe_parses_name_and_driver(self):
        completed = mock.Mock(returncode=0,
                             stdout="NVIDIA GeForce RTX 4080, 581.15\n")
        card = diagnostics.nvidia_card(lambda: completed)
        self.assertEqual(card.name, "NVIDIA GeForce RTX 4080")
        self.assertEqual(card.driver, "581.15")
        self.assertTrue(card.present)
        self.assertTrue(card.probed)

    def test_a_missing_binary_is_a_probed_absence(self):
        def boom():
            raise FileNotFoundError("nvidia-smi")

        card = diagnostics.nvidia_card(boom)
        self.assertFalse(card.present)
        self.assertTrue(card.probed)
        self.assertFalse(card.usable)

    def test_an_unprobed_card_is_not_an_absent_one(self):
        """The default instance means "nobody looked", and the wizard must behave exactly
        as it did before F230 in that state rather than announce an absence."""
        self.assertFalse(diagnostics.NvidiaCard().probed)


class TestTheOfferItself(unittest.TestCase):
    """What the wizard says and what Enter answers, per state of the machine."""

    def setUp(self):
        self.gpu = wizard.TIERS_BY_KEY[wizard.GPU_TIER_KEY]
        self.states = [tiers.TierState(
            tier.key, missing_packages=tuple(f"extra:{e}" for e in tier.extras),
            missing_weights=tier.weights) for tier in wizard.TIERS]

    def _run(self, card, *, answers=None):
        said: list[str] = []
        asked: list[tuple[str, bool]] = []
        commands: list[list[str]] = []

        def ask(question: str, default: bool = False) -> bool:
            asked.append((question, default))
            return default if answers is None else answers

        code = wizard.run_setup(
            "en", manifest={}, states=self.states, tiers=(self.gpu,), card=card,
            say=said.append, ask=ask, doctor=lambda *a, **k: None,
            install=lambda command: commands.append(list(command)) or 0,
            download=lambda *a, **k: True)
        return code, "\n".join(said), asked, commands

    def _card(self, **fields) -> diagnostics.NvidiaCard:
        base = {"name": "RTX 4080", "driver": "581.15", "present": True, "probed": True}
        return diagnostics.NvidiaCard(**{**base, **fields})

    def test_a_machine_with_a_card_is_offered_the_tier_with_yes_by_default(self):
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            code, said, asked, commands = self._run(self._card())
        self.assertEqual(code, 0)
        self.assertEqual([default for _q, default in asked], [True])
        self.assertIn(i18n.cli_text("cli.setup.question_yes", "en"),
                      [question for question, _d in asked])
        self.assertIn("RTX 4080", said)
        self.assertTrue(commands, "pressing Enter has to install it, not skip it")

    def test_the_price_of_a_refusal_is_named_in_hours(self):
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            _code, said, _asked, _commands = self._run(self._card())
        self.assertIn(i18n.cli_text("cli.setup.card_refusal_cost", "en"), said)
        self.assertIn("hours", said)

    def test_a_machine_with_no_card_is_not_asked_at_all(self):
        code, said, asked, commands = self._run(diagnostics.NvidiaCard(probed=True))
        self.assertEqual(code, 0)
        self.assertEqual(asked, [])
        self.assertEqual(commands, [])
        self.assertIn(i18n.cli_text("cli.setup.card_absent", "en",
                                    size=wizard.human_size(self.gpu.download_mb, "en")),
                      said)

    def test_an_old_driver_is_told_instead_of_2_5_gb_being_downloaded(self):
        code, said, asked, commands = self._run(self._card(driver="551.86"))
        self.assertEqual(code, 0)
        self.assertEqual(asked, [])
        self.assertEqual(commands, [])
        self.assertIn("551.86", said)
        self.assertIn(str(diagnostics.MIN_DRIVER_MAJOR), said)

    def test_an_explicit_tiers_gpu_still_will_not_install_wheels_that_cannot_load(self):
        """`--tiers gpu` on a machine whose driver is too old is honoured by SAYING why:
        an install command must not put wheels on a machine that cannot import them."""
        said: list[str] = []
        commands: list[list[str]] = []
        code = wizard.run_setup(
            "en", manifest={}, states=self.states, tiers=(self.gpu,),
            card=self._card(driver="551.86"), chosen=("gpu",), say=said.append,
            doctor=lambda *a, **k: None,
            install=lambda command: commands.append(list(command)) or 0,
            download=lambda *a, **k: True)
        self.assertEqual(code, 0)
        self.assertEqual(commands, [])
        self.assertIn("551.86", "\n".join(said))

    def test_nothing_changes_when_the_hardware_was_never_probed(self):
        """Every wizard test written before F230 passes no card, and must keep getting the
        wizard it was written against: the tier offered, the default a no."""
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            _code, said, asked, _commands = self._run(None)
        self.assertEqual([default for _q, default in asked], [False])
        self.assertNotIn("nvidia-smi", said)

    def test_the_entry_point_probes_the_card_and_hands_it_over(self):
        """The product path: `main` is where the one `nvidia-smi` call belongs."""
        card = self._card()
        with mock.patch.object(wizard, "accelerator", return_value=card) as probe, \
                mock.patch.object(wizard, "run_setup", return_value=0) as setup, \
                mock.patch.object(wizard, "hold_console"):
            self.assertEqual(wizard.main(["--tiers", "none"]), 0)
        probe.assert_called_once_with()
        self.assertIs(setup.call_args[1]["card"], card)

    def test_the_real_probe_is_what_the_entry_point_uses(self):
        with mock.patch("sorta.diagnostics.nvidia_card") as probe:
            wizard.accelerator()
        probe.assert_called_once_with()


class TestTheWayBackToTheCpuProfile(unittest.TestCase):
    """The acceleration tier is installed with `--reinstall`: it takes the CPU profile
    with it, and until F230 there was nothing to return to."""

    def test_the_command_is_the_mirror_of_the_tier(self):
        commands: list[list[str]] = []
        said: list[str] = []
        with mock.patch.object(wizard, "tier_requirements",
                               lambda tier: ("onnxruntime>=1.27.0", "torch>=2.10.0")):
            code = wizard.restore_cpu("en", {}, say=said.append,
                                      install=lambda c: commands.append(list(c)) or 0)
        self.assertEqual(code, 0)
        command = commands[0]
        self.assertIn("--reinstall", command)
        self.assertIn("torch>=2.10.0", command)
        # No CUDA index: the plain wheels have to take the place of the cu130 ones.
        self.assertNotIn("--index", command)
        self.assertNotIn(wizard.PYTORCH_CU130_INDEX, command)

    def test_it_ends_by_sending_the_person_to_the_doctor(self):
        said: list[str] = []
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            wizard.restore_cpu("en", {}, say=said.append, install=lambda c: 0)
        self.assertIn("sorta doctor", "\n".join(said))

    def test_a_failed_restore_leaves_the_install_as_it_was_and_says_so(self):
        said: list[str] = []
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            code = wizard.restore_cpu("en", {}, say=said.append, install=lambda c: 7)
        self.assertEqual(code, 1)
        self.assertIn(i18n.cli_text("cli.setup.restore_cpu_failed", "en", status=7),
                      said)

    def test_no_package_metadata_claims_nothing(self):
        said: list[str] = []
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ()):
            code = wizard.restore_cpu("en", {}, say=said.append,
                                      install=lambda c: 0)
        self.assertEqual(code, 1)
        self.assertIn(i18n.cli_text("cli.setup.tier.cpu_profile.name", "en"),
                      "\n".join(said))

    def test_the_flag_reaches_it_and_asks_nothing_else(self):
        with mock.patch.object(wizard, "restore_cpu", return_value=0) as restore, \
                mock.patch.object(wizard, "run_setup") as setup, \
                mock.patch.object(wizard, "hold_console"):
            self.assertEqual(wizard.main(["--restore-cpu"]), 0)
        restore.assert_called_once()
        setup.assert_not_called()

    def test_the_flag_is_what_an_installed_copy_is_told_about(self):
        self.assertIn("--restore-cpu",
                      i18n.cli_text("cli.setup.cpu_back.installed", "en"))
        self.assertIn("uv sync --extra cpu",
                      i18n.cli_text("cli.setup.cpu_back.checkout", "en"))

    def test_the_wizard_names_it_the_moment_it_replaces_the_profile(self):
        said: list[str] = []
        outcome = wizard.Outcome()
        with mock.patch.object(wizard, "tier_requirements", lambda tier: ("torch",)):
            wizard._add_tiers([wizard.TIERS_BY_KEY[wizard.GPU_TIER_KEY]], "en", {},
                              outcome, say=said.append, install=lambda command: 0)
        text = "\n".join(said)
        # The suite runs in a checkout, so the way back it is told about is the checkout's
        # one — which is the whole point: the sentence follows the install, here too.
        self.assertIn(i18n.cli_text(wizard.cpu_back_key(), "en"), text)
        self.assertIn(i18n.cli_text("cli.setup.profile_changed", "en"), text)

    def test_a_tier_that_only_adds_says_nothing_about_profiles(self):
        said: list[str] = []
        outcome = wizard.Outcome()
        wizard._add_tiers([wizard.TIERS_BY_KEY["faces"]], "en", {}, outcome,
                          say=said.append, install=lambda command: 0,
                          download=lambda *a, **k: True)
        text = "\n".join(said)
        self.assertNotIn(i18n.cli_text(wizard.cpu_back_key(), "en"), text)
        self.assertNotIn(i18n.cli_text("cli.setup.profile_changed", "en"), text)

    def test_the_rollback_is_not_a_tier_of_the_catalog(self):
        """It is asked for by name, not offered on a screen — see `wizard.CPU_PROFILE`."""
        self.assertNotIn(wizard.CPU_PROFILE, wizard.TIERS)
        self.assertNotIn(wizard.CPU_PROFILE.key, wizard.TIERS_BY_KEY)
        self.assertIsNone(wizard.selected_tiers(None))
        with self.assertRaises(ValueError):
            wizard.selected_tiers(wizard.CPU_PROFILE.key)


class TestTheDoctorSaysWhichProfileWon(unittest.TestCase):
    """F76: the two packages unpack into one directory, so after the profile is replaced
    only this can tell what happened."""

    def _health(self, torch_version: str, providers: tuple[str, ...]):
        return diagnostics.GpuHealth(
            torch_version=torch_version, torch_cuda_available="+cu" in torch_version,
            torch_device_name=None, ort_providers=providers,
            install_kind=install.KIND_INSTALLED)

    def test_both_stacks_on_cuda_is_the_gpu_profile(self):
        health = self._health("2.13.0+cu130", ("CUDAExecutionProvider",))
        self.assertEqual(health.profile, diagnostics.PROFILE_GPU)
        self.assertIn("install profile: gpu", health.summary)

    def test_both_stacks_plain_is_the_cpu_profile(self):
        health = self._health("2.13.0+cpu", ("CPUExecutionProvider",))
        self.assertEqual(health.profile, diagnostics.PROFILE_CPU)
        self.assertIn("install profile: cpu", health.summary)

    def test_one_of_each_is_named_mixed_rather_than_guessed(self):
        self.assertEqual(
            self._health("2.13.0+cpu", ("CUDAExecutionProvider",)).profile,
            diagnostics.PROFILE_MIXED)
        self.assertEqual(
            self._health("2.13.0+cu130", ("CPUExecutionProvider",)).profile,
            diagnostics.PROFILE_MIXED)

    def test_no_torch_at_all_is_unknown_and_not_cpu(self):
        self.assertEqual(self._health("not installed", ()).profile,
                         diagnostics.PROFILE_UNKNOWN)

    def test_the_profile_is_read_off_the_build_and_not_off_the_driver(self):
        """A CUDA build on a machine with too old a driver answers no to
        `torch.cuda.is_available()` — and is still the gpu profile, which is exactly the
        state somebody needs to see named after an install."""
        health = diagnostics.GpuHealth(
            torch_version="2.13.0+cu130", torch_cuda_available=False,
            torch_device_name=None, ort_providers=("CUDAExecutionProvider",),
            install_kind=install.KIND_INSTALLED)
        self.assertEqual(health.profile, diagnostics.PROFILE_GPU)

    def test_the_wizard_sends_people_here_after_changing_the_profile(self):
        self.assertIn("sorta doctor",
                      i18n.cli_text("cli.setup.profile_changed", "en"))
        self.assertIn("install profile",
                      i18n.cli_text("cli.setup.profile_changed", "en"))


if __name__ == "__main__":
    unittest.main()
