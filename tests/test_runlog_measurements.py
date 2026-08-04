"""F159: the run log is read back, so an estimate can stop carrying constants.

`runlog` has written `stage=<s>[ phase=<p>] elapsed=<sec> processed=<n>` since F147 and
nobody read it. These cases pin the reader — which lines count as a measurement, and,
more importantly, which do NOT: a stale timing is worse than no timing, because it is
believed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from sorta import __version__, runlog

BUILD = __version__
NOW = datetime(2026, 8, 3, 12, 0, 0)


def stamp(at: datetime) -> str:
    return f"{at:%Y-%m-%dT%H:%M:%S}.000"


def line(at: datetime, message: str, level: str = "INFO") -> str:
    """One record in the shape `runlog._FORMAT` writes."""
    return f"{stamp(at)} {level:<8} sorta.runlog [MainThread] {message}"


def header(at: datetime, build: str = BUILD) -> str:
    """The environment header `log_environment` writes at the start of every run."""
    return "\n".join([
        line(at, "environment:"),
        f"  sorta: {build}",
        "  python: 3.10.0 (C:\\Python310\\python.exe)",
    ])


def write_log(path, *blocks: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return path


def read(path, **kwargs):
    kwargs.setdefault("now", NOW)
    return runlog.read_measurements(path, **kwargs)


class TestWhatCountsAsAMeasurement:
    def test_a_stage_summary_becomes_a_rate(self, tmp_path):
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 20),
                 "stage=faces elapsed=1200.000 processed=600 rate=0.5/s"),
        )
        found = read(log)

        assert set(found) == {"stage=faces"}
        assert found["stage=faces"].seconds_per_unit == pytest.approx(2.0)
        assert found["stage=faces"].processed == 600
        assert found["stage=faces"].at == datetime(2026, 8, 3, 10, 20)
        assert found["stage=faces"].build == BUILD

    def test_a_phase_is_read_under_the_name_the_log_gives_it(self, tmp_path):
        """The estimate prices the model questions by the junk stage's VLM phase, so the
        key has to be the file's own `stage=... phase=...` and not a second vocabulary."""
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 11, 0),
                 "stage=junk phase=junk_vlm elapsed=800.000 processed=1000 rate=1.2/s"),
        )
        found = read(log)

        unit = runlog.measurement_unit("junk", "junk_vlm")
        assert unit == "stage=junk phase=junk_vlm"
        assert found[unit].seconds_per_unit == pytest.approx(0.8)

    def test_the_latest_run_of_a_stage_wins(self, tmp_path):
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 1, 10, 0)),
            line(datetime(2026, 8, 1, 10, 10),
                 "stage=geo elapsed=100.000 processed=100 rate=1.0/s"),
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=geo elapsed=400.000 processed=100 rate=0.2/s"),
        )
        assert read(log)["stage=geo"].seconds_per_unit == pytest.approx(4.0)

    def test_started_progress_and_the_broken_exits_are_not_measurements(self, tmp_path):
        """`started` and `progress` describe a unit that is not over; `failed` and
        `interrupted` describe one that stopped early, where the seconds are real but the
        denominator is not. A rate off either would promise a run nobody has ever had."""
        at = datetime(2026, 8, 3, 10, 30)
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(at, "stage=junk started total=1000"),
            line(at, "stage=junk progress elapsed=10.000 processed=100 total=1000"
                     " rate=10.0/s"),
            line(at, "stage=junk failed elapsed=10.000 processed=100 rate=10.0/s",
                 level="ERROR"),
            line(at, "stage=events interrupted (KeyboardInterrupt) elapsed=5.000"
                     " processed=50 rate=10.0/s"),
            line(at, "stage=junk phase=junk_clip interrupted (Cancelled) elapsed=5.000"
                     " processed=50 rate=10.0/s"),
        )
        assert read(log) == {}

    def test_a_summary_without_a_denominator_is_not_a_rate(self, tmp_path):
        """`processed=` is absent whenever the stage never reported a count. Seconds
        alone cannot be divided by anything, and inventing a 1 would price a whole stage
        at the cost of a single frame."""
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 10), "stage=landmarks elapsed=100.000"),
            line(datetime(2026, 8, 3, 10, 20),
                 "stage=phash elapsed=100.000 processed=0"),
            line(datetime(2026, 8, 3, 10, 30),
                 "stage=index elapsed=0.000 processed=100 rate=0.0/s"),
        )
        assert read(log) == {}

    def test_a_log_that_is_not_there_is_simply_no_measurements(self, tmp_path):
        assert read(tmp_path / "nothing" / "sorta.log") == {}

    def test_prose_around_the_timings_is_ignored(self, tmp_path):
        """The file is a real log: warnings, tracebacks and third-party chatter share it
        with the lines this reads."""
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 1),
                 "junk: VLM unavailable, falling back to CLIP", level="WARNING"),
            "Traceback (most recent call last):",
            "  File \"x.py\", line 1, in <module>",
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=events elapsed=60.000 processed=600 rate=10.0/s"),
        )
        assert read(log)["stage=events"].seconds_per_unit == pytest.approx(0.1)


class TestAMeasurementCanGoStale:
    """Requirement 5: a timing of a stage that has changed since is not a timing."""

    def test_a_timing_from_another_build_is_not_used(self, tmp_path):
        """The version is the fingerprint of the code that produced the number — the
        same device `frame_quality.source` uses for the prompts behind a verdict."""
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 3, 10, 0), build="0.0.1-something-else"),
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=faces elapsed=1200.000 processed=600 rate=0.5/s"),
        )
        assert read(log) == {}

    def test_a_timing_no_header_vouches_for_is_not_used(self, tmp_path):
        """A number whose pedigree cannot be established is exactly the kind of number
        this feature exists to stop showing."""
        log = write_log(
            tmp_path / "sorta.log",
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=faces elapsed=1200.000 processed=600 rate=0.5/s"),
        )
        assert read(log) == {}

    def test_the_build_applies_from_its_own_header_onwards(self, tmp_path):
        """Two runs of different versions share one file; only the current one counts."""
        log = write_log(
            tmp_path / "sorta.log",
            header(datetime(2026, 8, 1, 10, 0), build="0.0.1"),
            line(datetime(2026, 8, 1, 10, 10),
                 "stage=geo elapsed=100.000 processed=100 rate=1.0/s"),
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=faces elapsed=200.000 processed=100 rate=0.5/s"),
        )
        assert set(read(log)) == {"stage=faces"}

    def test_a_timing_older_than_the_window_is_not_used(self, tmp_path):
        old = NOW - timedelta(days=200)
        log = write_log(
            tmp_path / "sorta.log",
            header(old),
            line(old, "stage=faces elapsed=1200.000 processed=600 rate=0.5/s"),
        )
        assert read(log) == {}
        assert set(read(log, max_age_days=0)) == {"stage=faces"}


class TestWhereItReadsFrom:
    def test_the_previous_file_is_read_when_rotation_split_a_run(self, tmp_path):
        """Rotation knows nothing about runs: the 5 MB boundary can fall between a
        header and the summaries that belong to it."""
        log = tmp_path / "sorta.log"
        write_log(log.with_name("sorta.log.1"),
                  header(datetime(2026, 8, 3, 10, 0)),
                  line(datetime(2026, 8, 3, 10, 10),
                       "stage=geo elapsed=100.000 processed=100 rate=1.0/s"))
        write_log(log,
                  header(datetime(2026, 8, 3, 11, 0)),
                  line(datetime(2026, 8, 3, 11, 10),
                       "stage=faces elapsed=200.000 processed=100 rate=0.5/s"))

        assert set(read(log)) == {"stage=geo", "stage=faces"}
        assert [p.name for p in runlog.measurement_files(log)] == ["sorta.log.1",
                                                                  "sorta.log"]

    def test_the_env_var_decides_when_no_path_is_given(self, tmp_path, monkeypatch):
        """`SORTA_LOG_FILE` already redirects the WRITER; the reader has to follow it to
        the same place or it would price a run off somebody else's file."""
        log = write_log(
            tmp_path / "elsewhere.log",
            header(datetime(2026, 8, 3, 10, 0)),
            line(datetime(2026, 8, 3, 10, 10),
                 "stage=events elapsed=60.000 processed=600 rate=10.0/s"),
        )
        monkeypatch.setenv(runlog.ENV_LOG_FILE, str(log))

        assert set(runlog.read_measurements(now=NOW)) == {"stage=events"}
        assert runlog.measurement_files() == [log]


class TestTheWriterAndTheReaderAgree:
    """The one property that matters more than any regex: what `runlog` writes, it can
    read. A format change on one side and not the other would leave the estimate
    silently back on its constants, which is exactly the failure F159 is about."""

    def test_a_real_stage_timer_round_trips(self, tmp_path, monkeypatch):
        import logging
        import time

        log = tmp_path / "sorta.log"
        monkeypatch.setenv(runlog.ENV_LOG_LEVEL, "INFO")
        logging.getLogger("sorta").setLevel(logging.INFO)
        root = logging.getLogger()
        root_level = root.level
        runlog.setup_file_logging(log)
        try:
            runlog.log_environment()
            with runlog.stage_timer("faces", total=100) as stage:
                runlog.log_phase("faces", "faces_detect", 12.5, 250)
                # The summary writes `elapsed=%.3f`, and a stage this test does nothing
                # in rounds to 0.000 — which the reader rejects, correctly: a rate needs
                # both halves. A few milliseconds of real work make it a real summary.
                time.sleep(0.01)
                stage.processed = 100
        finally:
            for handler in list(root.handlers):
                if getattr(handler, "_sorta_runlog_handler", False):
                    root.removeHandler(handler)
                    handler.close()
            root.setLevel(root_level)

        found = runlog.read_measurements(log)
        assert set(found) == {"stage=faces", "stage=faces phase=faces_detect"}
        assert found["stage=faces phase=faces_detect"].processed == 250
        assert found["stage=faces"].processed == 100
