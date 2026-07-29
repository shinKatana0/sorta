"""F69: the run log file — everything through tmp_path, no real %LOCALAPPDATA%."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from sorta import runlog

LOGGER_NAME = "sorta.runlog"


@pytest.fixture(autouse=True)
def clean_logging():
    """Remove our handlers and restore levels — otherwise the file sink leaks into
    the rest of the pytest session (duplicated output, an open file in tmp_path)."""
    root = logging.getLogger()
    sorta_logger = logging.getLogger("sorta")
    root_level, sorta_level = root.level, sorta_logger.level
    # The file handler sees a record only after the emitting logger passes it; other
    # test modules call config.configure_logging, which pins `sorta` to WARNING.
    sorta_logger.setLevel(logging.DEBUG)
    yield
    for handler in list(root.handlers):
        if getattr(handler, "_sorta_runlog_handler", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(root_level)
    sorta_logger.setLevel(sorta_level)
    logging.captureWarnings(False)


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    """A clean environment: the machine's own SORTA_LOG_* must not affect the tests."""
    monkeypatch.delenv(runlog.ENV_LOG_FILE, raising=False)
    monkeypatch.delenv(runlog.ENV_LOG_LEVEL, raising=False)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers
            if getattr(h, "_sorta_runlog_handler", False)]


class TestSetupFileLogging:
    def test_creates_file_and_captures_pipeline_warning(self, tmp_path):
        path = tmp_path / "x.log"
        used = runlog.setup_file_logging(path)

        logging.getLogger("sorta.whatever").warning("junk: VLM недоступна")

        assert used == path
        assert path.exists()
        assert "junk: VLM недоступна" in read(path)

    def test_root_handler_catches_every_module_logger(self, tmp_path):
        runlog.setup_file_logging(tmp_path / "x.log")

        logging.getLogger("sorta.junk").warning("junk warning")
        logging.getLogger("sorta.geo").warning("geo warning")
        logging.getLogger("some.third.party").warning("third party warning")

        content = read(tmp_path / "x.log")
        assert "junk warning" in content
        assert "geo warning" in content
        assert "third party warning" in content

    def test_cyrillic_survives_the_round_trip(self, tmp_path):
        """Regression on encoding: the default cp1251 on Windows breaks the write."""
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)

        logging.getLogger("sorta.geo").warning("гео-база не загрузилась: Санкт-Петербург")

        content = read(path)  # utf-8 read must not raise
        assert "гео-база не загрузилась: Санкт-Петербург" in content

    def test_repeated_setup_does_not_duplicate_lines(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)
        runlog.setup_file_logging(path)

        logging.getLogger("sorta.junk").warning("сообщение ровно один раз")

        assert len(our_handlers()) == 1
        assert read(path).count("сообщение ровно один раз") == 1

    def test_repeated_setup_with_a_different_case_is_the_same_file(self, tmp_path):
        """`os.path.normcase` — on Windows C:\\X.LOG and c:\\x.log are one file."""
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)
        runlog.setup_file_logging(Path(str(path).upper()) if os.name == "nt" else path)

        assert len(our_handlers()) == 1

    def test_format_has_time_level_logger_and_thread(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)

        logging.getLogger("sorta.faces").warning("boom")

        line = read(path).strip().splitlines()[-1]
        assert "WARNING" in line
        assert "sorta.faces" in line
        assert "[MainThread]" in line
        assert line.startswith("20")  # ISO date, local time

    def test_rotation_is_configured_5mb_x5(self, tmp_path):
        runlog.setup_file_logging(tmp_path / "x.log")

        handler = our_handlers()[0]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 5
        assert (handler.encoding or "").lower() == "utf-8"

    def test_enables_capture_warnings(self, tmp_path):
        with mock.patch("logging.captureWarnings") as capture:
            runlog.setup_file_logging(tmp_path / "x.log")

        capture.assert_called_once_with(True)

    def test_captures_warnings_module(self, tmp_path):
        import warnings

        runlog.setup_file_logging(tmp_path / "x.log")
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("сторонняя библиотека ворчит", UserWarning)

        assert "сторонняя библиотека ворчит" in read(tmp_path / "x.log")

    def test_level_default_is_info(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)

        logging.getLogger("sorta.indexer").info("info line")
        logging.getLogger("sorta.indexer").debug("debug line")

        content = read(path)
        assert "info line" in content
        assert "debug line" not in content

    def test_explicit_level_is_applied(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path, level="DEBUG")

        logging.getLogger("sorta.indexer").debug("debug line")

        assert "debug line" in read(path)


class TestEnvOverrides:
    def test_env_file_is_used_without_an_argument(self, tmp_path, monkeypatch):
        path = tmp_path / "from_env.log"
        monkeypatch.setenv(runlog.ENV_LOG_FILE, str(path))

        used = runlog.setup_file_logging()

        logging.getLogger("sorta.junk").warning("env path")
        assert used == path
        assert "env path" in read(path)

    def test_argument_wins_over_env_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv(runlog.ENV_LOG_FILE, str(tmp_path / "from_env.log"))
        explicit = tmp_path / "explicit.log"

        used = runlog.setup_file_logging(explicit)

        assert used == explicit
        assert not (tmp_path / "from_env.log").exists()

    def test_env_level_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv(runlog.ENV_LOG_LEVEL, "debug")
        path = tmp_path / "x.log"

        runlog.setup_file_logging(path)

        logging.getLogger("sorta.indexer").debug("debug via env")
        assert "debug via env" in read(path)

    def test_env_level_by_number(self, tmp_path, monkeypatch):
        monkeypatch.setenv(runlog.ENV_LOG_LEVEL, str(logging.ERROR))
        runlog.setup_file_logging(tmp_path / "x.log")

        assert our_handlers()[0].level == logging.ERROR

    def test_argument_wins_over_env_level(self, tmp_path, monkeypatch):
        monkeypatch.setenv(runlog.ENV_LOG_LEVEL, "ERROR")
        runlog.setup_file_logging(tmp_path / "x.log", level="DEBUG")

        assert our_handlers()[0].level == logging.DEBUG

    def test_garbage_env_level_falls_back_to_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv(runlog.ENV_LOG_LEVEL, "не уровень")
        runlog.setup_file_logging(tmp_path / "x.log")

        assert our_handlers()[0].level == logging.INFO


# Faking `os.name` cuts BOTH ways, and the posix side is the loud one: `Path()` picks
# its flavour from `os.name` at instantiation, so with "nt" faked on Linux it builds a
# WindowsPath — which refuses to exist there and raises NotImplementedError. Worse, the
# failure is reported while the patch is still in place, so pytest's own
# `Path(os.getcwd())` raises too and the run dies with an INTERNALERROR instead of a
# test name (caught by CI on 2026-07-29; the local gate runs on Windows and never saw
# it). Each direction is therefore tested only on the platform where it is real.
_WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt",
    reason="Path() picks its flavour from os.name at instantiation, so faking nt on "
           "posix builds a WindowsPath that cannot be instantiated at all.",
)


class TestDefaultLogPath:
    @_WINDOWS_ONLY
    def test_windows_uses_localappdata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        assert runlog.default_log_path() == tmp_path / "sorta" / "logs" / "sorta.log"

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Path() picks its flavour from os.name at instantiation, so faking "
               "posix on Windows builds an unusable PosixPath. The same fallback "
               "branch is covered natively by test_windows_without_localappdata.",
    )
    def test_posix_uses_cache_home(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")

        expected = Path.home() / ".cache" / "sorta" / "logs" / "sorta.log"
        assert runlog.default_log_path() == expected

    @_WINDOWS_ONLY
    def test_windows_without_localappdata_falls_back(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        assert runlog.default_log_path() == Path.home() / ".cache" / "sorta" / "logs" / "sorta.log"


class TestFailureIsNotFatal:
    def _blocked_path(self, tmp_path: Path) -> Path:
        """A regular file in the middle of the path — mkdir cannot create the parent."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        return blocker / "logs" / "sorta.log"

    def test_unopenable_path_does_not_raise(self, tmp_path):
        target = self._blocked_path(tmp_path)

        used = runlog.setup_file_logging(target)

        assert used == target  # something meaningful: the path we tried to use
        assert our_handlers() == []

    def test_logging_keeps_working_after_a_failed_setup(self, tmp_path, caplog):
        target = self._blocked_path(tmp_path)

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            runlog.setup_file_logging(target)

        assert any("файл лога" in r.getMessage() for r in caplog.records)
        logging.getLogger("sorta.junk").warning("жизнь продолжается")  # must not raise

    def test_a_working_path_still_works_afterwards(self, tmp_path):
        runlog.setup_file_logging(self._blocked_path(tmp_path))
        good = tmp_path / "good.log"

        runlog.setup_file_logging(good)

        logging.getLogger("sorta.junk").warning("после отказа")
        assert "после отказа" in read(good)


class TestStageTimer:
    def test_success_line_is_greppable(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with runlog.stage_timer("index"):
                pass

        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("stage=index started") for m in messages)
        summary = [m for m in messages if m.startswith("stage=index elapsed=")]
        assert len(summary) == 1

    def test_total_is_logged_at_the_start(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with runlog.stage_timer("faces", total=40287):
                pass

        assert "stage=faces started total=40287" in [r.getMessage() for r in caplog.records]

    def test_processed_and_rate_reach_the_summary(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with runlog.stage_timer("classify", total=10) as result:
                assert result.name == "classify"
                assert result.total == 10
                result.processed = 1234

        summary = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("stage=classify elapsed=")][0]
        assert "processed=1234" in summary
        assert "rate=" in summary

    def test_summary_without_processed_has_no_counters(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with runlog.stage_timer("geo"):
                pass

        summary = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("stage=geo elapsed=")][0]
        assert "processed=" not in summary
        assert "rate=" not in summary

    def test_exception_is_logged_as_error_and_propagates(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with pytest.raises(RuntimeError, match="VLM не загрузилась"):
                with runlog.stage_timer("classify"):
                    raise RuntimeError("VLM не загрузилась")

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert errors[0].getMessage().startswith("stage=classify failed elapsed=")
        assert errors[0].exc_info is not None  # traceback, not just the message

    def test_failed_stage_keeps_the_partial_count(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            with pytest.raises(ValueError):
                with runlog.stage_timer("index") as result:
                    result.processed = 7
                    raise ValueError("boom")

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert "processed=7" in errors[0]

    def test_summary_lands_in_the_file(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)

        with runlog.stage_timer("sort") as result:
            result.processed = 3

        content = read(path)
        assert "stage=sort started" in content
        assert "stage=sort elapsed=" in content
        assert "processed=3" in content


class TestLogEnvironment:
    def test_writes_a_non_empty_block(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        assert len(caplog.records) == 1  # one INFO portion, not interleaved lines
        message = caplog.records[0].getMessage()
        for expected in ("environment:", "sorta:", "python:", "platform:",
                         "package:", "gpu:", "geo data:", "cwd:"):
            assert expected in message

    def test_reports_where_the_package_was_imported_from(self, caplog):
        """The line that tells the installed uv-tool from the working tree (F65)."""
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        message = caplog.records[0].getMessage()
        assert str(Path(runlog.__file__).resolve().parent) in message
        assert "repo working tree" in message or "site-packages" in message

    def test_survives_missing_torch_and_onnxruntime(self, caplog):
        with mock.patch.dict(sys.modules, {"torch": None, "onnxruntime": None}):
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                runlog.log_environment()  # must not raise

        message = caplog.records[0].getMessage()
        assert "gpu:" in message
        assert "not installed" in message

    def test_survives_a_broken_diagnostics_layer(self, caplog):
        with mock.patch("sorta.diagnostics.gpu_health", side_effect=RuntimeError("нет CUDA")):
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                runlog.log_environment()

        assert "недоступны" in caplog.records[0].getMessage()

    def test_reports_the_geo_data_directory(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        assert "places.tsv:" in caplog.records[0].getMessage()

    def test_reports_the_directory_the_resolver_actually_reads(self, caplog):
        """The header probed <repo>/data/geo while F65 ships the base in the package,
        so a correct install was reported as a missing base — in the very line that
        exists to catch a missing one. Asserting the substring was not enough."""
        from sorta.geodata import GeoResolver

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        assert str(GeoResolver().data_dir) in caplog.records[0].getMessage()

    def test_the_bundled_base_is_reported_as_present(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        assert "places.tsv: yes" in caplog.records[0].getMessage()

    def test_falls_back_to_the_legacy_directory(self, caplog, tmp_path, monkeypatch):
        """A pre-F65 layout (the base in <repo>/data/geo) is still reported honestly."""
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "places.tsv").write_text("", encoding="utf-8")
        monkeypatch.setattr(runlog, "_GEO_DATA_DIR", tmp_path / "absent")
        monkeypatch.setattr(runlog, "_LEGACY_GEO_DATA_DIR", legacy)

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        message = caplog.records[0].getMessage()
        assert str(legacy) in message
        assert "places.tsv: yes" in message

    def test_a_missing_base_is_still_reported_as_missing(self, caplog, tmp_path,
                                                         monkeypatch):
        monkeypatch.setattr(runlog, "_GEO_DATA_DIR", tmp_path / "absent")
        monkeypatch.setattr(runlog, "_LEGACY_GEO_DATA_DIR", tmp_path / "also-absent")

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            runlog.log_environment()

        assert "places.tsv: no" in caplog.records[0].getMessage()

    def test_goes_into_the_file(self, tmp_path):
        path = tmp_path / "x.log"
        runlog.setup_file_logging(path)

        runlog.log_environment()

        assert "environment:" in read(path)
