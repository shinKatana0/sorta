"""F224: an uninstall can leave nothing behind — but it asks first.

The owner cleaned a virtual machine by hand on 2026-08-07 and found that `unins000.exe`
had left `%APPDATA%\\sorta`, `%LOCALAPPDATA%\\sorta`, 1.6 GB of CLIP weights in
`~/.cache/huggingface/hub` and 0.3 GB of buffalo_l in `~/.insightface/models` — none of
which is named after Sorta. The missing half was the weights, so `sorta cache` grew
`--models` and `--clear-models`, and the Windows uninstaller calls that command instead
of repeating its logic.

The cases here are the ones that make the difference between "removes the gigabytes" and
"removes a neighbour's gigabytes":

* nothing goes without an answer, and the answer defaults to no;
* only the models the tier catalog names go, one directory at a time — the rest of a
  shared cache is left exactly as it was found;
* a junction is removed AS a junction and what it points at survives — checked on a real
  one, because that is the case `shutil.rmtree` gets wrong on Windows;
* the size stated before the deletion is the number of bytes the deletion returns;
* a silent uninstall deletes nothing at all;
* photographs and the move journal are never touched in any mode.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta import cli, i18n, tiers, weights, wizard

_ISS = Path(__file__).resolve().parent.parent / "packaging" / "windows" / "sorta.iss"


def read_iss() -> str:
    """The installer script. `utf-8-sig`: it carries a BOM, which Inno 6 requires of a
    script holding anything but ASCII — and this one holds three languages."""
    return _ISS.read_text(encoding="utf-8-sig")


def code_only(script: str) -> str:
    """The Pascal without its comments — a rule stated in a comment is not a rule."""
    return "\n".join(line for line in script.splitlines()
                     if not line.strip().startswith("//"))


def make_junction(link: Path, target: Path) -> bool:
    """A REAL junction, or False when this machine cannot make one.

    `mklink /J` needs no privileges (F218), which is the whole reason the link cases
    below can be tested at all instead of being simulated with a stub that answers
    `is_link` the way the test wants.
    """
    if os.name != "nt":
        return False
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    return done.returncode == 0 and link.exists()


class CacheCase(unittest.TestCase):
    """Two caches of the shape the real ones have, under a temporary home."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.hub = self.home / ".cache" / "huggingface" / "hub"
        self.models = self.home / ".insightface" / "models"
        self.hub.mkdir(parents=True)
        self.models.mkdir(parents=True)

    def write(self, path: Path, size: int = 1000) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def clip(self, size: int = 4000) -> Path:
        """What ViT-L-14 is CALLED once open_clip has downloaded it."""
        directory = self.hub / "models--timm--vit_large_patch14_clip_224.openai"
        self.write(directory / "blobs" / "weights.bin", size)
        return directory

    def buffalo(self, size: int = 2000) -> Path:
        directory = self.models / "buffalo_l"
        self.write(directory / "det_10g.onnx", size)
        return directory

    def found(self) -> list[weights.Downloaded]:
        return weights.downloaded(insightface=self.models, hub=self.hub)

    def use_as_real_caches(self) -> None:
        """Point the DEFAULTS at the fixture, so the command under test is the command
        a person runs and not one handed a pair of paths by the test."""
        env = patch.dict(os.environ, {"HF_HUB_CACHE": str(self.hub)})
        env.start()
        self.addCleanup(env.stop)
        insightface = patch.object(tiers, "_INSIGHTFACE_MODELS", self.models)
        insightface.start()
        self.addCleanup(insightface.stop)


class TestNothingGoesWithoutAnAnswer(CacheCase):
    """The default is «leave it». Uninstalling a program is not a request to delete
    somebody's data, and neither is asking what is on the disk."""

    def setUp(self):
        super().setUp()
        self.use_as_real_caches()
        self.clip()
        self.buffalo()

    def test_listing_the_models_removes_nothing(self):
        cli._cmd_cache("config.yaml", models=True, confirm=None)
        self.assertTrue((self.hub / "models--timm--vit_large_patch14_clip_224.openai"
                         ).is_dir())
        self.assertTrue((self.models / "buffalo_l").is_dir())

    def test_saying_no_leaves_everything(self):
        """The question is injected, exactly as `sorta reset` injects its own: typer
        aborts the command from inside `confirm`, and nothing after it runs."""
        def refuse(text: str) -> None:
            raise SystemExit(1)

        with self.assertRaises(SystemExit):
            cli._cmd_cache("config.yaml", clear_models=True, confirm=refuse)
        self.assertTrue((self.models / "buffalo_l").is_dir())
        self.assertTrue(self.found())

    def test_the_question_states_the_size_before_it_is_asked(self):
        asked: list[str] = []
        cli._cmd_cache("config.yaml", clear_models=True, confirm=asked.append)
        self.assertEqual(len(asked), 1)
        # 6000 bytes of fixture, printed the way the catalog prints sizes.
        self.assertIn("0.00 GB", asked[0])
        self.assertEqual(self.found(), [])

    def test_the_command_works_with_no_config_at_all(self):
        """The person this exists for is removing the program — quite possibly with
        their config.yaml already gone, and always from a directory nobody chose."""
        cli._cmd_cache(str(self.home / "nothing-here.yaml"), models=True, confirm=None)


class TestWhatCannotBeDoneIsSaidRatherThanRaised(CacheCase):
    """A cleanup that ends in a traceback is a cleanup nobody finishes — and this one
    runs inside an uninstaller, where a traceback goes nowhere at all."""

    def setUp(self):
        super().setUp()
        self.use_as_real_caches()

    def test_an_empty_disk_is_an_answer(self):
        printed: list[str] = []
        with patch("builtins.print", lambda *args: printed.append(" ".join(map(str, args)))):
            cli._cmd_cache("config.yaml", clear_models=True, confirm=None)
        self.assertEqual(printed, [i18n.cli_text("cli.cache.models_none", "en")])

    def test_a_directory_that_will_not_go_is_reported(self):
        self.clip()
        printed: list[str] = []
        with patch.object(weights, "_delete",
                          side_effect=OSError("the file is in use")), \
                patch("builtins.print",
                      lambda *args: printed.append(" ".join(map(str, args)))):
            cli._cmd_cache("config.yaml", clear_models=True, confirm=None)
        self.assertTrue(any("the file is in use" in line for line in printed))
        self.assertTrue(self.found())  # still there, and said to be


class TestOnlyTheCatalogModelsGo(CacheCase):
    """A shared cache is a cache somebody else is also using."""

    def setUp(self):
        super().setUp()
        self.ours_hub = self.clip()
        self.ours_models = self.buffalo()
        self.stranger_hub = self.hub / "models--openai--whisper-large-v3"
        self.write(self.stranger_hub / "blobs" / "weights.bin", 5000)
        self.stranger_models = self.models / "antelopev2"
        self.write(self.stranger_models / "det.onnx", 5000)
        self.loose = self.write(self.hub / "version.txt", 10)

    def test_the_catalog_models_are_the_ones_found(self):
        found = {entry.path for entry in self.found()}
        self.assertEqual(found, {self.ours_hub, self.ours_models})

    def test_what_is_not_ours_survives_the_removal(self):
        result = weights.remove(self.found())
        self.assertEqual(set(result.removed), {self.ours_hub, self.ours_models})
        self.assertFalse(self.ours_hub.exists())
        self.assertFalse(self.ours_models.exists())
        self.assertTrue((self.stranger_hub / "blobs" / "weights.bin").is_file())
        self.assertTrue((self.stranger_models / "det.onnx").is_file())
        self.assertTrue(self.loose.is_file())
        # ...and the caches themselves are still caches, not two empty holes.
        self.assertTrue(self.hub.is_dir())
        self.assertTrue(self.models.is_dir())

    def test_a_cache_that_holds_nothing_of_ours_is_a_quiet_answer(self):
        weights.remove(self.found())
        self.assertEqual(self.found(), [])


class TestLinksAreRemovedAsLinks(CacheCase):
    """The case that makes this more than one `Remove-Item`.

    `~/.insightface` is a junction to `C:\\AI\\buffalo` on the owner's machine, and a
    recursive delete through it destroys weights that live somewhere else entirely —
    which is what `shutil.rmtree` does on Windows, where `os.path.islink` answers False
    for a junction. Both cases are checked on a REAL junction: a stub would only prove
    that the code agrees with the test's idea of a link.
    """

    def setUp(self):
        super().setUp()
        self.elsewhere = self.home / "elsewhere"
        self.write(self.elsewhere / "buffalo_l" / "det_10g.onnx", 9000)

    def test_a_model_directory_that_is_a_junction_goes_as_a_link(self):
        if not make_junction(self.models / "buffalo_l", self.elsewhere / "buffalo_l"):
            self.skipTest("this machine cannot create a junction")
        entry, = [item for item in self.found() if item.weight == "buffalo_l"]
        self.assertTrue(entry.link)
        self.assertEqual(entry.size, 0)  # nothing here to free — the bytes are elsewhere
        result = weights.remove([entry])
        self.assertEqual(result.freed, 0)
        self.assertEqual(result.removed, (self.models / "buffalo_l",))
        self.assertFalse((self.models / "buffalo_l").exists())
        self.assertTrue((self.elsewhere / "buffalo_l" / "det_10g.onnx").is_file())

    def test_a_model_found_behind_a_junction_is_left_where_it_is(self):
        """The `~/.insightface -> C:\\AI\\buffalo` shape: the model directory is real,
        and it is real inside somebody else's store."""
        linked = self.home / "linked-models"
        self.write(linked / "buffalo_l" / "det_10g.onnx", 9000)
        (self.models).rmdir()
        if not make_junction(self.models, linked):
            self.skipTest("this machine cannot create a junction")
        entry, = [item for item in self.found() if item.weight == "buffalo_l"]
        self.assertEqual(entry.behind, self.models)
        self.assertFalse(entry.removable)
        result = weights.remove([entry])
        self.assertEqual(result.removed, ())
        self.assertEqual(result.kept, (entry,))
        self.assertTrue((linked / "buffalo_l" / "det_10g.onnx").is_file())

    def test_a_link_inside_a_model_directory_is_not_walked_through(self):
        directory = self.clip()
        if not make_junction(directory / "blobs-elsewhere", self.elsewhere):
            self.skipTest("this machine cannot create a junction")
        weights.remove(self.found())
        self.assertFalse(directory.exists())
        self.assertTrue((self.elsewhere / "buffalo_l" / "det_10g.onnx").is_file())


class TestTheSizeShownIsTheSizeFreed(CacheCase):
    """«Frees 1.9 GB» is an answer; «clear the cache» is a riddle."""

    def test_the_total_is_what_the_disk_gives_back(self):
        self.clip(40_000)
        self.buffalo(20_000)
        found = self.found()
        stated = weights.total_bytes(found)
        before = sum(path.stat().st_size
                     for path in self.home.rglob("*") if path.is_file())
        result = weights.remove(found)
        after = sum(path.stat().st_size
                    for path in self.home.rglob("*") if path.is_file())
        self.assertEqual(stated, 60_000)
        self.assertEqual(result.freed, stated)
        self.assertEqual(before - after, stated)

    def test_a_link_is_counted_as_the_nothing_it_frees(self):
        self.write(self.home / "elsewhere" / "buffalo_l" / "det.onnx", 9000)
        if not make_junction(self.models / "buffalo_l",
                             self.home / "elsewhere" / "buffalo_l"):
            self.skipTest("this machine cannot create a junction")
        self.clip(40_000)
        self.assertEqual(weights.total_bytes(self.found()), 40_000)


class TestPhotographsAreNeverTouched(CacheCase):
    """Sorta only ever moved them, and the journal of those moves is data too."""

    def test_nothing_outside_the_two_caches_is_ever_named(self):
        pictures = self.home / "Pictures" / "2019" / "Kyoto"
        self.write(pictures / "IMG_0001.jpg", 3000)
        self.write(pictures / "buffalo_l.jpg", 3000)  # a name is not a model
        journal = self.write(self.home / "AppData" / "sorta" / "moves.jsonl", 500)
        self.clip()
        self.buffalo()
        for entry in self.found():
            self.assertIn(entry.path.parent, (self.hub, self.models))
        weights.remove(self.found())
        self.assertTrue((pictures / "IMG_0001.jpg").is_file())
        self.assertTrue((pictures / "buffalo_l.jpg").is_file())
        self.assertTrue(journal.is_file())


class TestTheCatalogIsReadAndNotCopied(unittest.TestCase):
    """F223 is editing the tier catalog as this lands. A second list of model names
    would disagree with the first one the same day."""

    def test_the_models_are_the_ones_the_catalog_names(self):
        self.assertEqual(
            weights.catalog_weights(),
            tuple(dict.fromkeys(name for tier in wizard.TIERS
                                for name in tier.weights)))
        self.assertIn("buffalo_l", weights.catalog_weights())

    def test_a_model_added_to_a_tier_is_removable_without_another_edit(self):
        tier = wizard.Tier("f224", weights=("some-new-model",), download_mb=1)
        with patch.object(wizard, "TIERS", (*wizard.TIERS, tier)):
            self.assertIn("some-new-model", weights.catalog_weights())


class TestTheUninstallerAsksAndCallsTheCommand(unittest.TestCase):
    """The Inno half, read as text — the compiler is not on the machine that runs the
    suite, so what can be pinned here is the shape of the decision, not the pixels."""

    def setUp(self):
        self.script = read_iss()
        self.code = code_only(self.script)

    def test_both_ticks_start_empty(self):
        self.assertIn("DataBox.Checked := False", self.script)
        self.assertIn("ModelsBox.Checked := False", self.script)
        self.assertNotIn("Checked := True", self.script)

    def test_a_silent_uninstall_asks_nobody_and_deletes_nothing(self):
        page = self.script.split("function InitializeUninstall")[1]
        silent = page.index("if UninstallSilent then")
        # The guard comes before the page, and both answers are already False by then.
        self.assertLess(silent, page.index("AskWhatToRemove(ModelBytes, DataBytes)"))
        self.assertLess(page.index("RemoveData := False"), silent)
        self.assertLess(page.index("RemoveModels := False"), silent)

    def test_the_models_are_removed_by_calling_the_command(self):
        """Not repeated here: the dangerous rule (shared caches, junctions) lives where
        ordinary tests reach it. F211's precedent — the wizard calls `sorta doctor`."""
        self.assertIn("-m sorta.cli cache --clear-models --yes", self.script)
        for stranger in ("huggingface", "insightface", "FindFirst"):
            self.assertNotIn(stranger, self.script)

    def test_the_sizes_come_from_the_program_itself(self):
        self.assertIn("from sorta import weights; print(weights.report())", self.script)

    def test_the_data_directories_are_not_followed_through_a_link(self):
        self.assertIn("FILE_ATTRIBUTE_REPARSE_POINT", self.script)
        removal = self.code.split("procedure RemoveOurDir")[1]
        self.assertLess(removal.index("FILE_ATTRIBUTE_REPARSE_POINT"),
                        removal.index("DelTree"))

    def test_the_command_runs_while_the_program_is_still_on_the_disk(self):
        step = self.code.split("procedure CurUninstallStepChanged")[1]
        self.assertIn("CurUninstallStep <> usUninstall", step)
        self.assertNotIn("usPostUninstall", step)

    def test_the_report_is_two_numbers_and_not_a_translated_sentence(self):
        self.assertEqual(weights.report().splitlines()[0].split()[0], "models")
        self.assertEqual(weights.report().splitlines()[1].split()[0], "data")
        for line in weights.report().splitlines():
            self.assertTrue(line.split()[1].isdigit(), line)


class TestThreeLanguages(unittest.TestCase):
    """Every string a person sees, in the three languages of the product."""

    KEYS = (
        "cli.cache.models_header", "cli.cache.models_entry", "cli.cache.models_link",
        "cli.cache.models_behind_link", "cli.cache.models_total",
        "cli.cache.models_none", "cli.cache.models_confirm",
        "cli.cache.models_cleared", "cli.cache.models_kept", "cli.cache.models_failed",
        "cli.help.cache.models", "cli.help.cache.clear_models", "cli.help.cache.yes",
    )
    MESSAGES = ("CleanupCaption", "CleanupIntro", "CleanupData", "CleanupDataNote",
                "CleanupModels", "CleanupModelsNote")

    def test_every_new_cli_string_speaks_three(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = i18n._CLI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for text in entry.values():
                    self.assertTrue(text.strip())

    def test_every_uninstaller_message_speaks_three(self):
        script = read_iss()
        for name in self.MESSAGES:
            for lang in ("en", "ru", "ja"):
                with self.subTest(message=name, lang=lang):
                    self.assertIn(f"\n{lang}.{name}=", script)

    def test_the_sizes_reach_the_uninstaller_page_as_numbers(self):
        """A tick that does not say what it costs is the "delete the cache" riddle."""
        script = read_iss()
        for name in ("CleanupData", "CleanupModels"):
            for lang in ("en", "ru", "ja"):
                line = script.split(f"\n{lang}.{name}=")[1].splitlines()[0]
                self.assertIn("%1", line)


class TestTheFlagsExist(unittest.TestCase):
    """The terminal surface, since the uninstaller types it."""

    def setUp(self):
        if cli.app is None:  # pragma: no cover — the argparse fallback, no typer
            self.skipTest("typer is not installed")

    def test_cache_carries_the_three_new_options(self):
        import typer.main
        command = typer.main.get_command(cli.build_app("en")).commands["cache"]
        options = {opt for param in command.params for opt in param.opts}
        self.assertLessEqual({"--models", "--clear-models", "--yes", "-y"}, options)
