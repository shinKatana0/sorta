"""F247: the folder the dialog returned has to reach the field, and a refusal has to
name what actually happened.

Reported from a clean VM (Windows 11, 2026-08-23): "Browse…" opened Explorer, a folder
was chosen, and the screen answered that this machine has no folder picker and that
`sudo apt install python3-tk` would give it one. Three defects in one sentence — the path
was lost on the write to stdout, the refusal described something that had not happened,
and the advice was for another operating system.

The encoding test below is the one that matters: it fails on the shipped
`sys.stdout.write(path)` for exactly the reason the owner's log showed.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from sorta import ui

# Cyrillic and an emoji: cp1251 (the locale of the machine that reported this) has a
# character for the first half and none for the second.
_HARD_PATH = "C:/Фото/Отпуск 📷/北京"

_LANGS = ("ru", "en", "ja")


def _stub_dialog(answer: str, break_stdout: bool = False) -> str:
    """The shipped child script with tkinter replaced by a stub that answers `answer`.

    Everything below the dialog — the encoding, the write, the exit code — stays the
    code that ships, and it runs in a real interpreter read by the real parent. The
    answer is spelled with `ascii()` so that the command line carries no non-ASCII of its
    own: what is being measured is the child's stdout, not its argv.
    """
    prelude = [
        "import sys, types",
        "tk = types.ModuleType('tkinter')",
        "fd = types.ModuleType('tkinter.filedialog')",
        "class _Root:",
        "    def withdraw(self): pass",
        "    def attributes(self, *a): pass",
        "    def destroy(self): pass",
        "tk.Tk = _Root",
        f"fd.askdirectory = lambda: {ascii(answer)}",
        "tk.filedialog = fd",
        "sys.modules['tkinter'] = tk",
        "sys.modules['tkinter.filedialog'] = fd",
    ]
    if break_stdout:
        # Candidate B of the brief: the stream the answer goes to is not there. The
        # chain in the VM is launcher -> `sorta ui` -> dialog, and every link has its
        # own captured streams.
        prelude.append("sys.stdout = None")
    return "\n".join(prelude) + "\n" + ui.process._BROWSE_DIALOG_SCRIPT


class _RealDialogCase(unittest.TestCase):
    """Runs the real child in a real interpreter, with a locale encoding that cannot
    spell the answer."""

    def run_dialog(self, answer: str, break_stdout: bool = False) -> tuple[str, str]:
        script = _stub_dialog(answer, break_stdout)
        with mock.patch.dict(os.environ, {"PYTHONIOENCODING": "cp1251"}), \
                mock.patch.object(ui.process, "_BROWSE_DIALOG_SCRIPT", script):
            return ui._run_browse_dialog()


class TestThePathSurvivesTheLocale(_RealDialogCase):
    def test_a_path_the_locale_cannot_encode_arrives_whole(self):
        self.assertEqual(self.run_dialog(_HARD_PATH), (_HARD_PATH, ui.BROWSE_CANCELLED))

    def test_an_ascii_path_is_unchanged_by_the_same_route(self):
        self.assertEqual(self.run_dialog("C:/Photos"), ("C:/Photos", ui.BROWSE_CANCELLED))

    def test_a_cancelled_dialog_is_still_an_empty_path_and_no_problem(self):
        self.assertEqual(self.run_dialog(""), ("", ui.BROWSE_CANCELLED))


class TestTheChildDoesNotFallOverOnTheAnswer(_RealDialogCase):
    def test_a_stdout_that_cannot_be_written_is_a_code_and_not_a_traceback(self):
        with self.assertLogs("sorta.ui", level="WARNING") as logs:
            self.assertEqual(self.run_dialog(_HARD_PATH, break_stdout=True),
                             ("", ui.BROWSE_NO_ANSWER))
        line = "\n".join(logs.output)
        self.assertIn("the picked path did not reach stdout", line)
        self.assertNotIn("Traceback", line)
        self.assertIn(str(ui._BROWSE_NO_ANSWER_EXIT), line)


class TestTheThreeOutcomesAreToldApart(unittest.TestCase):
    def _dialog_exits(self, returncode: int, stderr: bytes = b"") -> tuple[str, str]:
        with mock.patch.object(ui.process, "subprocess") as fake:
            fake.TimeoutExpired = subprocess.TimeoutExpired
            fake.run.return_value = mock.Mock(returncode=returncode, stdout=b"",
                                              stderr=stderr)
            with self.assertLogs("sorta.ui", level="WARNING"):
                return ui._run_browse_dialog()

    def test_a_machine_without_tkinter_says_the_picker_is_missing(self):
        self.assertEqual(self._dialog_exits(1, b"ModuleNotFoundError: tkinter"),
                         ("", ui.BROWSE_UNAVAILABLE))

    def test_a_lost_answer_has_a_code_of_its_own(self):
        self.assertEqual(self._dialog_exits(ui._BROWSE_NO_ANSWER_EXIT),
                         ("", ui.BROWSE_NO_ANSWER))

    def test_a_dialog_nobody_answered_is_not_a_missing_picker(self):
        """The window was drawn and stood open for two minutes. Saying "no picker on
        this machine" here is the same lie the feature exists to remove."""
        with mock.patch.object(ui.process, "subprocess") as fake:
            fake.TimeoutExpired = subprocess.TimeoutExpired
            fake.run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=120)
            with self.assertLogs("sorta.ui", level="WARNING"):
                self.assertEqual(ui._run_browse_dialog(), ("", ui.BROWSE_NO_ANSWER))

    def test_a_dialog_that_could_not_be_started_is_a_missing_picker(self):
        with mock.patch.object(ui.process, "subprocess") as fake:
            fake.TimeoutExpired = subprocess.TimeoutExpired
            fake.run.side_effect = OSError("no display")
            with self.assertLogs("sorta.ui", level="ERROR"):
                self.assertEqual(ui._run_browse_dialog(), ("", ui.BROWSE_UNAVAILABLE))

    def test_the_three_codes_are_three_different_values(self):
        codes = (ui.BROWSE_CANCELLED, ui.BROWSE_UNAVAILABLE, ui.BROWSE_NO_ANSWER)
        self.assertEqual(len(set(codes)), 3)


class TestTheLogStaysEnglishAndKeepsTheEvidence(unittest.TestCase):
    """F245: the log is the source of truth and is not translated — and this line is
    what found the defect in the first place."""

    def test_the_warning_carries_the_exit_code_and_the_child_stderr(self):
        with mock.patch.object(ui.process, "subprocess") as fake:
            fake.TimeoutExpired = subprocess.TimeoutExpired
            fake.run.return_value = mock.Mock(
                returncode=9, stdout=b"", stderr="ImportError: нет модуля".encode("utf-8"))
            with self.assertLogs("sorta.ui", level="WARNING") as logs:
                ui._run_browse_dialog()
        line = logs.output[0]
        self.assertIn("browse: the folder dialog exited 9", line)
        self.assertIn(ui.BROWSE_UNAVAILABLE, line)
        # The child's own words are quoted whatever alphabet they are in; ours are English.
        self.assertIn("ImportError: нет модуля", line)
        self.assertNotIn("Диалог", line)


class TestTheMessagesSayWhatHappened(unittest.TestCase):
    def test_both_refusals_are_in_the_catalog_in_three_languages(self):
        for key in ("browse_unavailable", "browse_no_answer"):
            entry = ui._UI_STRINGS[key]
            self.assertEqual(set(entry), set(_LANGS), key)
            for lang, text in entry.items():
                with self.subTest(key=key, lang=lang):
                    self.assertTrue(text.strip(), f"{key}/{lang} is empty")

    def test_the_two_refusals_are_not_the_same_sentence(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertNotEqual(ui._UI_STRINGS["browse_unavailable"][lang],
                                    ui._UI_STRINGS["browse_no_answer"][lang])

    def test_the_lost_answer_never_claims_the_machine_has_no_picker(self):
        """The whole complaint: the picker WAS there and had just been used."""
        for lang, missing in (("ru", "недоступ"), ("en", "unavailable"),
                              ("ja", "利用できません")):
            text = ui._UI_STRINGS["browse_no_answer"][lang]
            with self.subTest(lang=lang):
                self.assertNotIn(missing, text)
                self.assertNotIn("apt", text)

    def test_the_lost_answer_still_sends_the_person_to_the_field(self):
        for lang, field in (("ru", "поле"), ("en", "field"), ("ja", "入力欄")):
            with self.subTest(lang=lang):
                self.assertIn(field, ui._UI_STRINGS["browse_no_answer"][lang])


class TestThePackageAdviceFollowsThePlatform(unittest.TestCase):
    def test_windows_is_told_nothing_about_apt(self):
        for platform in ("win32", "darwin"):
            strings = ui._browse_strings(platform)["browse_unavailable"]
            for lang in _LANGS:
                with self.subTest(platform=platform, lang=lang):
                    self.assertNotIn("apt", strings[lang])
                    self.assertNotIn("python3-tk", strings[lang])

    def test_linux_is_told_which_package_gives_the_picker(self):
        strings = ui._browse_strings("linux")["browse_unavailable"]
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertIn("sudo apt install python3-tk", strings[lang])

    def test_the_advice_is_an_addition_to_one_sentence_not_a_second_one(self):
        """Same first sentence everywhere — only the package line comes and goes."""
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertTrue(ui._browse_strings("linux")["browse_unavailable"][lang]
                                .startswith(ui._BROWSE_UNAVAILABLE[lang]))

    def test_the_catalog_of_this_machine_matches_this_platform(self):
        self.assertEqual(ui._UI_STRINGS["browse_unavailable"],
                         ui._browse_strings(ui.sys.platform)["browse_unavailable"])


class TestTheAlertPicksItsMessageByCode(unittest.TestCase):
    def test_the_page_branches_on_the_problem_code(self):
        html = ui._render_index_html("ru")
        self.assertIn('resp.problem === "no_answer"', html)
        self.assertIn("I18N.browse_no_answer", html)
        self.assertIn("I18N.browse_unavailable", html)


if __name__ == "__main__":
    unittest.main()
