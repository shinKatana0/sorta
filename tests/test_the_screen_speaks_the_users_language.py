"""F245: the run screen speaks the interface's language; the log stays English.

Three things are held here, and the order is the order of an error's journey.

1. A failure of ours carries WHAT happened (`faults.Fault`) and still says it in the
   same English sentence it said before: `str(exc)`, `args` and the built-in class it is
   a kind of are exactly what the terminal, the log and every existing `except` saw.
2. Every failure class of ours has a key in all three languages, and the page renders
   that key instead of the sentence. The classes are FOUND (F239) — a seventh one added
   tomorrow is red until it is given a key.
3. The reverse, and the one that matters more: with the interface in `ru` or `ja` the
   file `sorta.log` still receives English. A feature that "helps" by translating the log
   is caught here rather than a month later, by the log attached to a complaint.

What is walked for (2) is every class of `sorta/` that subclasses `Exception`. A class
that derives straight from `BaseException` is out of scope by a property and not by a
name: `except Exception` in the pipeline cannot catch one, so it can never become the
error of a run — in this package that is cancellation, which is a signal and not a
failure anybody is shown.
"""
from __future__ import annotations

import ast
import importlib
import json
import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import exif, faults, geodata, relocate, search, sorter, tray
from sorta.config import Config
from sorta.db import connect
from sorta.faults import Fault
from sorta.i18n import Lang
from sorta.runlog import setup_file_logging
from sorta.ui import process
from sorta.ui.page import _render_index_html
from sorta.ui.strings import _UI_STRINGS

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "sorta"
_APP_JS = _PACKAGE / "web" / "app" / "app.js"
_LANGS: tuple[Lang, ...] = ("ru", "en", "ja")

# Cyrillic, hiragana, katakana, CJK — everything the interface can be in and the log
# may not be.
_NOT_ENGLISH = re.compile(r"[\u0400-\u04FF\u3040-\u30FF\u3400-\u9FFF]")


def modules() -> list[str]:
    """Every importable module of the package, walked and never listed (F239)."""
    names = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        parts = path.relative_to(_ROOT).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def exception_classes() -> dict[str, type[BaseException]]:
    """`module.Class` -> every failure class the package DEFINES.

    Found by reading each module for its own `class` statements and then asking the
    imported module what those names are: `ast` alone cannot tell that
    `class GeoDataMissing(FileNotFoundError)` is an exception without resolving the
    base, and a list of base names is the guard that passes for the next class written
    against a base nobody thought of.
    """
    found: dict[str, type[BaseException]] = {}
    for name in modules():
        module = importlib.import_module(name)
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            obj = getattr(module, node.name, None)
            if isinstance(obj, type) and issubclass(obj, BaseException):
                found[f"{name}.{node.name}"] = obj
    return found


def shown_to_a_person() -> dict[str, type[Exception]]:
    """The classes the rule is about: the ones an `except Exception` can catch.

    `Fault` itself is not one of them — it is the protocol the others are written
    against, and nothing raises it.
    """
    return {name: cls for name, cls in exception_classes().items()
            if issubclass(cls, Exception) and cls is not Fault}


def without_a_key(classes: dict[str, type[Exception]]) -> dict[str, str]:
    """Class -> why it is red. Empty when every one of them can be drawn in three
    languages. A function rather than a `for` inside the test, so the guard itself can
    be shown going red over a class that does not exist in the package."""
    red: dict[str, str] = {}
    for name, cls in classes.items():
        if not issubclass(cls, Fault):
            red[name] = f"{name} is not a faults.Fault — it reaches a screen as a string"
            continue
        if not cls.codes:
            red[name] = f"{name} declares no codes"
            continue
        for code in cls.codes:
            entry = _UI_STRINGS.get(f"fault_{code}")
            missing = [lang for lang in _LANGS if not (entry or {}).get(lang)]
            if missing:
                red[name] = f"{name}: fault_{code} has no {', '.join(missing)}"
    return red


def constructed_codes() -> dict[str, set[str]]:
    """Class name -> every code a call to it passes as a literal, over the package.

    The second half of the declaration check: `codes` is what the guard reads, and a
    raise site that quietly invents a code outside that tuple would be a failure with no
    translation and nothing to notice it. A code computed from a value (`search`) is not
    a literal and is not seen here — those two are constructed by hand below.
    """
    names = {name.rsplit(".", 1)[1] for name in exception_classes()}
    used: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in names or len(node.args) < 2:
                continue
            code = node.args[1]
            if isinstance(code, ast.Constant) and isinstance(code.value, str):
                used.setdefault(node.func.id, set()).add(code.value)
    return used


def assigned_codes() -> set[str]:
    """Every code the package writes down without an exception class behind it.

    The run screen can fail outside a stage — the plan cache after a run that otherwise
    finished — and that failure has a code too. Found by reading what is assigned to
    `error_code`, so the next one written that way owes three translations as well.
    """
    found: set[str] = set()
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "error_code":
                    found.add(node.value.value)
    return found


def render(code: str, params: dict[str, object], lang: Lang) -> str:
    """The sentence the page builds, in `lang` — the Python side of `processErrorText`.

    Only the two substitutions the browser makes: a `stage` is a key and is named by the
    catalog `i18n.stage_label` writes into, `size_mb` becomes a size in words. The rest
    of the values are printed as they arrive.
    """
    values = dict(params)
    if values.get("stage"):
        values["stage"] = _UI_STRINGS[f"stage_name_{values['stage']}"][lang]
    if "size_mb" in values:
        mb = int(str(values["size_mb"]))
        values["size"] = (_UI_STRINGS["tier_size_mb"][lang].format(mb=mb) if mb < 1000
                          else _UI_STRINGS["tier_size_gb"][lang].format(gb=f"{mb / 1000:.1f}"))
    text = _UI_STRINGS[f"fault_{code}"][lang].format(**values)
    if values.get("offline_variable"):
        text += " " + _UI_STRINGS["fault_download_offline"][lang].format(
            variable=values["offline_variable"])
    return text


def a_fault_of_each_kind() -> list[tuple[BaseException, str]]:
    """One real exception per code, raised the way its module raises it.

    Built by calling the constructors rather than by writing the codes down: the message
    under test is the one the module produces, so a message edited without its catalog
    entry fails the comparison instead of being copied into it.
    """
    made: list[BaseException] = [
        exif.UnsafeExifPath("exiftool: path must be absolute, got 'a/b.jpg'",
                            "exif_relative_path", path="'a/b.jpg'"),
        geodata.GeoDataMissing(
            geodata._MISSING_DATA_HINT.format(path="/x/places.tsv"),
            "geo_data_missing", path="/x/places.tsv"),
        relocate.RelocateError("the old and the new prefix are the same path (/a).",
                               "relocate_same_prefix", prefix="/a"),
        relocate.RelocateError("no index at /a/photos.db.", "relocate_no_index",
                               path="/a/photos.db"),
        relocate.RelocateError(
            "/b does not exist — nothing was written. The new location has to be there "
            "before the index is pointed at it.",
            "relocate_target_missing", prefix="/b"),
        relocate.RelocateError(
            "no value in the index starts with /a — nothing was written. Check the old "
            "prefix against a path the index actually holds; the match is "
            "case-sensitive.",
            "relocate_no_rows", prefix="/a"),
        relocate.RelocateError(
            "3 paths would collide with rows that are already there (for example /b/c.jpg)"
            " — nothing was written.",
            "relocate_collisions", count=3, sample="/b/c.jpg"),
        relocate.CollectionMoved(
            relocate._MOVED_HINT.format(roots="/a, /b", rows=17, sample="/a/c.jpg"),
            "relocate_collection_moved", roots="/a, /b", rows=17, sample="/a/c.jpg"),
        search.EmbeddingsMissing(search.REASON_EMPTY, "ViT-L-14", 0, 0),
        search.EmbeddingsMissing(search.REASON_OTHER_MODEL, "ViT-L-14", 19757, 0),
        sorter.TransferError("copy failed: /a/x.jpg -> /b/x.jpg: disk full",
                             "sorter_copy_failed", src="/a/x.jpg", dst="/b/x.jpg",
                             error="disk full"),
        sorter.TransferError("the hash of the copy did not match, copy deleted: "
                             "/a/x.jpg -> /b/x.jpg",
                             "sorter_hash_mismatch", src="/a/x.jpg", dst="/b/x.jpg"),
        sorter.TransferError("dst already exists, overwriting is forbidden: /b/x.jpg",
                             "sorter_dst_exists", dst="/b/x.jpg"),
        sorter.TransferError("the check after the transfer did not pass: /b/x.jpg",
                             "sorter_check_failed", dst="/b/x.jpg"),
        tray.TrayUnavailable("cannot read favicon.ico: broken", "tray_icon_unreadable",
                             icon="favicon.ico", error="broken"),
        tray.TrayUnavailable("pystray is not installed: No module named 'pystray'",
                             "tray_no_pystray", error="No module named 'pystray'"),
        tray.TrayUnavailable("no tray on this system: no DISPLAY", "tray_no_backend",
                             error="no DISPLAY"),
    ]
    return [(exc, faults.fault_code(exc) or "") for exc in made]


class TestAFaultIsStillTheExceptionItWas(unittest.TestCase):
    """The English text does not move. Everything else is additive."""

    def test_str_and_args_are_the_message_alone(self):
        exc = relocate.RelocateError("no index at /tmp/photos.db.", "relocate_no_index",
                                     path="/tmp/photos.db")
        self.assertEqual(str(exc), "no index at /tmp/photos.db.")
        self.assertEqual(exc.args, ("no index at /tmp/photos.db.",))

    def test_the_builtin_it_is_a_kind_of_still_catches_it(self):
        with self.assertRaises(ValueError):
            raise exif.UnsafeExifPath("path must be absolute", "exif_relative_path",
                                      path="x")
        with self.assertRaises(FileNotFoundError):
            raise geodata.GeoDataMissing("places.tsv is not at /x", "geo_data_missing",
                                         path="/x")

    def test_the_code_and_the_params_travel_with_it(self):
        exc = geodata.GeoDataMissing("places.tsv is not at /x", "geo_data_missing",
                                     path="/x")
        self.assertEqual(faults.fault_code(exc), "geo_data_missing")
        self.assertEqual(faults.fault_params(exc), {"path": "/x"})

    def test_someone_elses_exception_has_neither(self):
        """`sqlite3.OperationalError`, `OSError`, `MemoryError` — the page may not
        pretend to have translated one of those."""
        self.assertIsNone(faults.fault_code(OSError("disk gone")))
        self.assertEqual(faults.fault_params(OSError("disk gone")), {})

    def test_the_message_a_module_raises_is_the_one_it_raised_before(self):
        """Two of the six, checked through the real refusal rather than by hand."""
        relative = Path("relative/a.jpg")
        with self.assertRaises(exif.UnsafeExifPath) as caught:
            exif._require_absolute([relative])
        self.assertEqual(str(caught.exception),
                         f"exiftool: path must be absolute, got {str(relative)!r}")
        self.assertEqual(render("exif_relative_path",
                                faults.fault_params(caught.exception), "en"),
                         str(caught.exception))
        with self.assertRaises(relocate.RelocateError) as refused:
            relocate.relocate("/no/such.db", "/a", "/a")
        self.assertTrue(str(refused.exception).startswith("the old and the new prefix"))


class TestEveryFailureOfOursCanBeDrawn(unittest.TestCase):
    """F245 requirement 5: the classes are found, and each one owes three languages."""

    def test_every_exception_class_of_the_package_is_a_fault_with_translated_codes(self):
        red = without_a_key(shown_to_a_person())
        self.assertEqual(red, {}, "\n".join(red.values()))

    def test_a_class_that_derives_from_baseexception_is_out_of_scope_by_its_type(self):
        """Not by its name: `except Exception` cannot catch one, so it never becomes the
        error of a run. Cancellation is the only such class today."""
        signals = {name: cls for name, cls in exception_classes().items()
                   if not issubclass(cls, Exception)}
        for name, cls in signals.items():
            with self.subTest(signal=name):
                self.assertTrue(issubclass(cls, BaseException))
                self.assertFalse(issubclass(cls, Exception))

    def test_a_seventh_class_added_tomorrow_is_red_until_it_has_a_key(self):
        """The watchdog, seen barking: a class of the shape the next feature will add."""
        class NewRefusal(Fault, RuntimeError):
            codes = ("brand_new_refusal",)

        class Undeclared(Fault, RuntimeError):
            pass

        class NotAFaultAtAll(RuntimeError):
            pass

        red = without_a_key({"sorta.new.NewRefusal": NewRefusal,
                             "sorta.new.Undeclared": Undeclared,
                             "sorta.new.NotAFaultAtAll": NotAFaultAtAll})
        self.assertEqual(sorted(red), ["sorta.new.NewRefusal", "sorta.new.NotAFaultAtAll",
                                       "sorta.new.Undeclared"])
        self.assertIn("fault_brand_new_refusal", red["sorta.new.NewRefusal"])
        self.assertIn("declares no codes", red["sorta.new.Undeclared"])
        self.assertIn("not a faults.Fault", red["sorta.new.NotAFaultAtAll"])

    def test_the_walk_really_reaches_the_classes_it_is_about(self):
        """A scan pointed at nothing finds nothing and looks exactly like a green gate."""
        found = exception_classes()
        self.assertGreaterEqual(len(found), 9)
        for name in ("sorta.exif.UnsafeExifPath", "sorta.relocate.RelocateError",
                     "sorta.relocate.CollectionMoved", "sorta.search.EmbeddingsMissing",
                     "sorta.sorter.TransferError", "sorta.tray.TrayUnavailable",
                     "sorta.geodata.GeoDataMissing",
                     "sorta.ui.process._DownloadRefused",
                     "sorta.ui.process._PipelineCancelled"):
            with self.subTest(cls=name):
                self.assertIn(name, found)

    def test_no_raise_site_invents_a_code_its_class_does_not_declare(self):
        for class_name, used in constructed_codes().items():
            declared = {name.rsplit(".", 1)[1]: cls
                        for name, cls in exception_classes().items()}[class_name].codes
            with self.subTest(cls=class_name):
                self.assertEqual(sorted(used - set(declared)), [])

    def test_the_two_codes_built_from_a_reason_are_the_declared_ones(self):
        """`EmbeddingsMissing` names its code after `reason`, so no literal to read."""
        made = {faults.fault_code(search.EmbeddingsMissing(reason, "m", 0, 0))
                for reason in (search.REASON_EMPTY, search.REASON_OTHER_MODEL)}
        self.assertEqual(made, set(search.EmbeddingsMissing.codes))


class TestTheEnglishScreenDidNotChange(unittest.TestCase):
    """F245 criterion 2/6: `en` shows the sentence it always showed, character for
    character — which is only true while the catalog's `en` IS the exception's message."""

    def test_every_fault_renders_in_english_as_its_own_message(self):
        for exc, code in a_fault_of_each_kind():
            with self.subTest(code=code):
                self.assertEqual(render(code, faults.fault_params(exc), "en"), str(exc))

    def test_every_fault_says_the_same_thing_in_the_other_two_languages(self):
        """Not the same text — the same VALUES. A translation that drops the path or the
        count is a sentence about nothing, which is the failure this whole feature is
        meant to prevent."""
        for exc, code in a_fault_of_each_kind():
            for lang in ("ru", "ja"):
                with self.subTest(code=code, lang=lang):
                    text = render(code, faults.fault_params(exc), lang)
                    self.assertNotEqual(text, "")
                    self.assertNotIn("{", text)
                    for value in faults.fault_params(exc).values():
                        self.assertIn(str(value), text)

    def test_a_failure_outside_a_stage_has_a_code_and_three_languages_too(self):
        """The plan cache after a run that otherwise finished — the one failure of the
        run screen with no exception class of ours behind it. It used to be the last
        Russian sentence in `ui/process.py`, shown whatever the interface was."""
        self.assertEqual(assigned_codes(), {"plan_not_rebuilt"})
        for code in assigned_codes():
            for lang in _LANGS:
                with self.subTest(code=code, lang=lang):
                    self.assertTrue(_UI_STRINGS[f"fault_{code}"][lang])
        self.assertEqual(render("plan_not_rebuilt", {"error": "database is locked"}, "en"),
                         "the plan was not rebuilt: database is locked")

    def test_the_download_refusal_is_still_the_sentence_the_console_prints(self):
        """It is generated from the CLI catalog, so the two cannot drift; asserted
        because a copy here would be the second wording of one fact."""
        from sorta import tiers

        params = {"stage": "landmarks", "weights": "ViT-L-14", "size_mb": 1600,
                  "error": "SSL: CERTIFICATE_VERIFY_FAILED", "offline_variable": ""}
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertEqual(
                    render("download_refused", params, lang),
                    tiers.download_failure("landmarks", ("ViT-L-14",), lang,
                                           "SSL: CERTIFICATE_VERIFY_FAILED"))


class TestThePersonalityReachesThePage(unittest.TestCase):
    """F245 requirements 2 and 3: the code travels with the error, and the page draws
    from it. What the browser does with the payload cannot be run here (no JS engine —
    see tests/test_ui_js_sanity.py), so what is pinned is that it reads the fields the
    server sends and renders them through the catalog."""

    def test_finish_and_the_snapshot_carry_the_code_and_the_values(self):
        state = process._ProcessState()
        state.try_start("/photos")
        state.finish("no index at /a.db.", "index", "relocate_no_index", {"path": "/a.db"})
        snapshot = state.snapshot()
        self.assertEqual(snapshot["error"], "no index at /a.db.")
        self.assertEqual(snapshot["error_code"], "relocate_no_index")
        self.assertEqual(snapshot["error_params"], {"path": "/a.db"})
        # The payload is JSON on the way to the browser: a `Path` or an exception in
        # `params` would be a 500 at the moment of the failure it describes.
        self.assertEqual(json.loads(json.dumps(snapshot))["error_params"],
                         {"path": "/a.db"})

    def test_an_error_that_is_not_ours_leaves_the_code_empty(self):
        state = process._ProcessState()
        state.try_start("/photos")
        state.finish("database is locked", "index")
        self.assertIsNone(state.snapshot()["error_code"])
        self.assertEqual(state.snapshot()["error_params"], {})

    def test_the_page_draws_the_code_and_falls_back_to_the_english_text(self):
        source = _APP_JS.read_text(encoding="utf-8")
        self.assertIn('I18N["fault_" + data.error_code]', source)
        self.assertIn("I18N.process_error_unknown", source)
        self.assertIn("processErrorText(data)", source)
        # The two params that are keys rather than text.
        self.assertIn('I18N["stage_name_" + vals.stage]', source)
        self.assertIn("downloadSize(vals.size_mb)", source)

    def test_the_wrapper_for_someone_elses_error_exists_in_three_languages(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                text = _UI_STRINGS["process_error_unknown"][lang]
                self.assertIn("{stage}", text)

    def test_the_page_the_browser_gets_carries_every_key_the_renderer_asks_for(self):
        """A catalog the server holds and the page does not ship is a code drawn as
        nothing. What is checked is `window.I18N` of the real rendered page."""
        for lang in _LANGS:
            served = json.loads(re.search(r"window\.I18N = (\{.*\});",
                                          _render_index_html(lang)).group(1))
            for _exc, code in a_fault_of_each_kind():
                with self.subTest(lang=lang, code=code):
                    self.assertIn(f"fault_{code}", served)
            for key in ("fault_download_refused", "fault_download_offline",
                        "process_error_unknown", "stage_name_landmarks"):
                with self.subTest(lang=lang, key=key):
                    self.assertTrue(served[key])


class _NoCache:
    """A plan cache that would notice being rebuilt — a failed run must not reach it."""

    def rebuild(self, cfg: Config, conn: object) -> None:  # pragma: no cover
        raise AssertionError("the plan cache was rebuilt after a failed run")


class TestTheLogStaysEnglish(unittest.TestCase):
    """F245 requirement 6, the reverse guard: whatever the interface speaks, the file
    that gets attached to a complaint is English.

    The real pipeline is run with a stage that raises a real refusal of ours, with the
    real rotating file sink attached — the same one `sorta ui` writes `sorta.log` with.
    """

    def _run_and_read_the_log(self, lang: str) -> tuple[str, dict]:
        moved = relocate.CollectionMoved(
            relocate._MOVED_HINT.format(roots="D:/photos", rows=38485,
                                        sample="D:/photos/a.jpg"),
            "relocate_collection_moved", roots="D:/photos", rows=38485,
            sample="D:/photos/a.jpg")

        def boom(cfg, conn, progress=None):
            raise moved

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "photos.db"
            connect(db).close()
            cfg = Config(language=lang, raw={"language": lang})
            state = process._ProcessState()
            state.try_start(str(root))
            log_file = root / "sorta.log"
            logger = logging.getLogger()
            before = list(logger.handlers)
            setup_file_logging(log_file, "DEBUG")
            try:
                with mock.patch.object(process, "_pipeline_steps",
                                       lambda notify: [("index", boom)]):
                    process._run_pipeline(db, cfg, str(root), state, _NoCache())
            finally:
                for handler in [h for h in logger.handlers if h not in before]:
                    logger.removeHandler(handler)
                    handler.close()
            return log_file.read_text(encoding="utf-8"), state.snapshot()

    def test_the_log_is_english_whatever_the_interface_speaks(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                written, _snapshot = self._run_and_read_the_log(lang)
                self.assertEqual(_NOT_ENGLISH.findall(written), [])
                self.assertIn("the collection was not found", written)
                self.assertIn("D:/photos/a.jpg", written)

    def test_the_screen_gets_the_code_while_the_log_gets_the_sentence(self):
        written, snapshot = self._run_and_read_the_log("ru")
        self.assertEqual(snapshot["error_code"], "relocate_collection_moved")
        self.assertEqual(snapshot["error_params"]["rows"], 38485)
        self.assertEqual(snapshot["error_stage"], "index")
        self.assertIn(snapshot["error"], written)
        # ...and the Russian the reader of that screen sees was never written down here.
        self.assertNotIn(render("relocate_collection_moved",
                                snapshot["error_params"], "ru"), written)

    def test_the_guard_would_notice_a_translated_log(self):
        """The watchdog, seen barking: the same run with the sentence localized the way
        it was before F245 — which is what the next feature to "help" would produce."""
        russian = render("relocate_collection_moved",
                         {"roots": "D:/photos", "rows": 38485,
                          "sample": "D:/photos/a.jpg"}, "ru")
        self.assertNotEqual(_NOT_ENGLISH.findall(russian), [])


if __name__ == "__main__":
    unittest.main()
