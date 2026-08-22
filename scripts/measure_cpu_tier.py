#!/usr/bin/env python3
"""F243 phase 0: the rig that gives the CPU tier a number, and proves the number is CPU.

The CPU profile is the only tier of this project with no measurement of any kind, and on
a machine without a video card it is the ONLY tier there is. Closing that hole takes one
run of the whole collection — phase 1, not this script. This is the rig phase 1 runs, and
the reason it is code instead of a stopwatch is the trap below.

THE TRAP. The repository's own venv is the GPU profile (`nvidia-*`, CUDA torch). Running
"the CPU measurement" in it is the easiest thing in the world, and what comes out is a
number about CUDA wearing the label of a number about the CPU. That is the class this
project has been catching in itself since 2026-08-07 — a check that answers about its own
machine instead of about its subject — and it cost four red gates. So the rig REFUSES to
start unless it can prove it stands on a CPU-only stack, and it writes the proof INTO the
report: providers, versions, cores, processor, interpreter. A measurement nobody can
prove was about the CPU is worse than no measurement at all: it lies with the face of a
number, and the report has to be readable in six months without its author.

WHAT IT DOES NOT DO. It moves no file, writes to no production index (`--db` defaults to
a file of its own, and pointing it at the owner's base needs `--allow-real-db`), fetches
no model and edits no documentation. Faces and the deep tier are out of scope on purpose:
their price on the CPU is predictably unacceptable, and measuring it would spend a day of
the machine on a number that changes nothing.

    uv sync --extra cpu                     # a venv of its own, NOT the repository's
    python scripts/measure_cpu_tier.py --src D:/Photos --out cpu_tier.json

The stage timings are READ BACK from the run log the product writes itself (`stage=` /
`elapsed=`, F219/F235) rather than taken with a second stopwatch here: two ways of
measuring one thing disagree eventually and then nobody knows which was right. What the
log cannot say is reported as a stage with no timing, never as an absence.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sorta import accel, exif, runlog  # noqa: E402 — after the path insert
from sorta.config import Config, configure_logging, load_config  # noqa: E402
from sorta.db import connect  # noqa: E402
from sorta.dedup import assign_duplicates, compute_phashes, near_duplicate_groups  # noqa: E402
from sorta.geo import resolve_places  # noqa: E402
from sorta.hashing import resolve_workers  # noqa: E402
from sorta.indexer import index as run_index  # noqa: E402

REPORT_SCHEMA = 1

# --- the proof, without which nothing runs -------------------------------------------

TORCH_RUNTIME = "torch.cuda.is_available()"
TORCH_BUILD = "torch build"
ONNX_PROVIDERS = "onnxruntime providers"


@dataclass(frozen=True)
class Check:
    """One question about the stack, its answer, and the evidence behind the answer.

    `cpu_only` is the verdict of this check alone; the run needs every one of them.
    """

    name: str
    cpu_only: bool
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {"check": self.name, "cpu_only": self.cpu_only, "detail": self.detail}


def import_or_none(name: str) -> Any | None:
    """The module, or None when this environment does not have it."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def torch_checks(torch: Any | None) -> list[Check]:
    """What torch says about CUDA, asked twice — at runtime and about the build itself.

    The build is a separate question and a separate refusal on purpose. A CUDA wheel on a
    machine whose card is absent, busy or driver-less answers `is_available() -> False`,
    and the run would then price the GPU profile's stack (505 MB of torch, onnxruntime-gpu
    beside it) while calling the result a CPU number. The profile is what the report is
    about, not the weather on the card.
    """
    if torch is None:
        return [Check(TORCH_RUNTIME, True, "torch is not installed — there is no CUDA to run on"),
                Check(TORCH_BUILD, True, "torch is not installed")]
    version = str(getattr(torch, "__version__", "unknown"))
    try:
        available = bool(torch.cuda.is_available())
        runtime = Check(TORCH_RUNTIME, not available,
                        f"torch {version}: cuda.is_available() -> {available}")
    except Exception as exc:
        # Unanswered is not the same fact as "no", and the rig may only refuse in the
        # direction that cannot produce a false number.
        runtime = Check(TORCH_RUNTIME, False,
                        f"torch {version}: cuda.is_available() raised {type(exc).__name__}: {exc}")
    built = getattr(getattr(torch, "version", None), "cuda", None)
    return [runtime,
            Check(TORCH_BUILD, built is None and "+cu" not in version,
                  f"torch {version}, built against CUDA {built or 'nothing'}")]


def onnx_checks(onnxruntime: Any | None) -> list[Check]:
    """The providers this onnxruntime OFFERS — the faces and junk stages' accelerator.

    TRAP: `accel.onnx_providers` is the wrong function to ask here. It returns the
    historical `[CUDA, CPU]` pair even on a runtime that offers neither, because a session
    is allowed to ask for what it cannot get. `available_providers` is what the machine
    really has.
    """
    if onnxruntime is None:
        return [Check(ONNX_PROVIDERS, True, "onnxruntime is not installed")]
    offered = accel.available_providers(onnxruntime)
    return [Check(ONNX_PROVIDERS, accel.CUDA_PROVIDER not in offered,
                  f"offered: {', '.join(offered) or 'none'}")]


def cpu_only_checks(torch: Any | None, onnxruntime: Any | None) -> list[Check]:
    """Every question that has to be answered before a stage is allowed to run."""
    return [*torch_checks(torch), *onnx_checks(onnxruntime)]


def refusals(checks: Sequence[Check]) -> list[Check]:
    """The checks that did not prove a CPU-only stack; empty means the run may start."""
    return [check for check in checks if not check.cpu_only]


# --- the stages this rig prices -------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One measurable stage: its name in the run log, and how to run it.

    `run` returns the number of items the stage processed — the denominator that turns
    seconds into a rate, and the same count the run log's summary line carries.
    """

    name: str
    run: Callable[[Config, sqlite3.Connection, Any], int]


def _stage_index(cfg: Config, conn: sqlite3.Connection, cb: Any) -> int:
    """The walk, the metadata and the exact-duplicate roles — `_pipeline_steps`' own pair.

    `assign_duplicates` is inside the stage and not beside it because that is where the
    product puts it, and a stage measured in a different shape from the one that ships is
    not a measurement of the product.
    """
    stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
    assign_duplicates(conn, cfg.dedup.canonical_strategy)
    return stats.scanned


def _stage_geo(cfg: Config, conn: sqlite3.Connection, cb: Any) -> int:
    return resolve_places(cfg, conn, progress=cb).total


def _stage_phash(cfg: Config, conn: sqlite3.Connection, cb: Any) -> int:
    """Returns what it COMPUTED, which on a repeat run is 0 — see `no_timing_note`."""
    return compute_phashes(cfg, conn, progress=cb)


def _stage_dupes(cfg: Config, conn: sqlite3.Connection, cb: Any) -> int:
    """The near-duplicate report. Processed = the population it compares, counted here
    because the function returns groups and a group count is not a denominator."""
    population = conn.execute(
        """SELECT COUNT(*) FROM files
           WHERE phash IS NOT NULL AND dup_of IS NULL AND error IS NULL"""
    ).fetchone()[0]
    near_duplicate_groups(conn, cfg.index.phash_max_distance)
    return int(population)


# In dependency order, which is also the order `--stages` runs them in whatever order it
# was given them: phash after index and dupes after phash is not a preference.
STAGES: tuple[Stage, ...] = (
    Stage("index", _stage_index),
    Stage("geo", _stage_geo),
    Stage("phash", _stage_phash),
    Stage("dupes", _stage_dupes),
)

DEFAULT_STAGES = tuple(stage.name for stage in STAGES)

# What the total below deliberately does NOT include, so the report cannot be read as a
# price for the whole pipeline.
OUT_OF_SCOPE = ("landmarks", "classify", "faces", "events", "junk")
OUT_OF_SCOPE_WHY = ("the deep tier and faces are out of F243's scope by decision: their "
                    "cost on a CPU is predictably unacceptable and measuring it would "
                    "spend a day of the machine on a number that changes nothing")


def select_stages(names: str) -> list[Stage]:
    """`--stages` -> the stages to run, in pipeline order regardless of how it was typed."""
    wanted = [part for part in names.replace(",", " ").split() if part]
    known = {stage.name: stage for stage in STAGES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"--stages: unknown stage(s) {', '.join(unknown)}; "
                         f"this rig knows {', '.join(known)}")
    if not wanted:
        raise SystemExit("--stages: nothing to measure")
    return [stage for stage in STAGES if stage.name in set(wanted)]


# --- running them, and reading the timings back out of the run log --------------------

MEASURED = "measured"
NO_TIMING = "no timing in the run log"


@dataclass(frozen=True)
class StageOutcome:
    """What the report says about one stage — including the ones that produced no timing.

    A stage that ran and left no summary is reported as itself with `seconds: null` and
    the reason, because a stage missing from the list would read as a stage nobody asked
    for.
    """

    name: str
    status: str
    seconds: float | None
    processed: int | None
    note: str = ""


def _quiet(done: int, total: int | None = None) -> None:
    """The progress callback the stages are handed: the run log is the only reader."""


def run_stages(stages: Iterable[Stage], cfg: Config,
               conn: sqlite3.Connection) -> dict[str, int]:
    """Run each stage under the product's own `stage_timer`; returns name -> processed.

    The timer is what writes `stage=<name> elapsed=<sec> processed=<n>` into the run log,
    which is where `collect_outcomes` reads the seconds from a moment later.
    """
    counts: dict[str, int] = {}
    for stage in stages:
        print(f"stage {stage.name}: running", flush=True)
        with runlog.stage_timer(stage.name) as timed:
            processed = stage.run(cfg, conn, runlog.observe(timed, _quiet))
            timed.processed = processed
        counts[stage.name] = processed
    return counts


def measurements_since(log: Path, started: datetime) -> dict[str, runlog.Measurement]:
    """The run log's summaries, this run's only.

    TRAP: the log is append-only and the file may already hold an earlier run of the same
    build. A timing from before this run started is not this run's, and attributing it
    would be the same defect the whole script exists against — with no way to notice it
    afterwards. The log's own timestamps have a one-second resolution, hence the truncation.
    """
    since = started.replace(microsecond=0)
    return {unit: found
            for unit, found in runlog.read_measurements(log, max_age_days=0).items()
            if found.at >= since}


def no_timing_note(processed: int | None) -> str:
    """Why a stage that ran holds no timing — the one thing a bare gap cannot say."""
    if processed == 0:
        return ("the stage processed 0 items, and a summary without a denominator is "
                "dropped by runlog.read_measurements — nothing was skipped, there was "
                "nothing to do")
    return ("the stage ran, but the run log holds no summary line for it — read the log "
            "before believing any total below")


def collect_outcomes(measurements: dict[str, runlog.Measurement],
                     counts: dict[str, int], order: Sequence[str]) -> list[StageOutcome]:
    """One row per REQUESTED stage, in the order they ran — including the empty ones."""
    outcomes = []
    for name in order:
        found = measurements.get(runlog.measurement_unit(name))
        if found is None:
            outcomes.append(StageOutcome(name, NO_TIMING, None, counts.get(name),
                                         no_timing_note(counts.get(name))))
        else:
            outcomes.append(StageOutcome(name, MEASURED, round(found.seconds, 3),
                                         found.processed))
    return outcomes


def measured_seconds(outcomes: Iterable[StageOutcome]) -> float:
    return round(sum(o.seconds or 0.0 for o in outcomes), 3)


# --- what the report has to carry so it can be read without its author ----------------

# Every package the speed of the base tier depends on. `onnxruntime-gpu` is in the list
# precisely because it should be absent: its version here would name the wrong profile.
MEASURED_PACKAGES = ("sorta", "torch", "onnxruntime", "onnxruntime-gpu", "numpy", "pillow",
                     "imagehash", "pyexiftool", "reverse-geocoder", "blake3")


def package_versions(names: Sequence[str] = MEASURED_PACKAGES) -> dict[str, str]:
    """name -> version, or "not installed" — which is an answer and is written down."""
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except Exception:
            found[name] = "not installed"
    return found


def machine_facts() -> dict[str, Any]:
    """The machine, and WHICH interpreter ran — the evidence for the venv the rig stood in."""
    return {
        "platform": platform.platform(),
        # platform.processor() is empty on some Linux builds; the Windows environment
        # variable is what the owner's machine answers with.
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "architecture": platform.machine(),
        "cores_logical": os.cpu_count(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "exiftool": exif.exiftool_binary() or "not found",
    }


def collection_facts(cfg: Config, conn: sqlite3.Connection) -> dict[str, Any]:
    """The collection as the index sees it after the run — the report's denominator."""
    counted = conn.execute(
        """SELECT COUNT(*) AS rows_total,
                  SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                  SUM(CASE WHEN dup_of IS NULL AND error IS NULL THEN 1 ELSE 0 END) AS canonical
           FROM files"""
    ).fetchone()
    return {
        "sources": [str(path) for path in cfg.sources],
        "database": str(cfg.database),
        "files": int(counted["rows_total"] or 0),
        "errors": int(counted["errors"] or 0),
        "canonical": int(counted["canonical"] or 0),
        "workers": resolve_workers(cfg.raw),
    }


def build_report(*, checks: Sequence[Check], outcomes: Sequence[StageOutcome],
                 machine: dict[str, Any], collection: dict[str, Any],
                 log: Path, started: datetime, finished: datetime) -> dict[str, Any]:
    """The whole document, in the order somebody reading it cold needs it."""
    return {
        "schema": REPORT_SCHEMA,
        "feature": "F243",
        "tool": "scripts/measure_cpu_tier.py",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "cpu_only": not refusals(checks),
        "proof": [check.as_json() for check in checks],
        "machine": machine,
        "packages": package_versions(),
        "collection": collection,
        "run_log": str(log),
        "stages": [asdict(outcome) for outcome in outcomes],
        "measured_seconds_total": measured_seconds(outcomes),
        "not_measured": {"stages": list(OUT_OF_SCOPE), "why": OUT_OF_SCOPE_WHY},
    }


def format_report(report: dict[str, Any]) -> str:
    """The console form — the same facts, for the person standing in front of the run."""
    lines = [f"{'stage':<10} {'status':<24} {'seconds':>10} {'processed':>10}"]
    for stage in report["stages"]:
        seconds = "—" if stage["seconds"] is None else f"{stage['seconds']:.3f}"
        processed = "—" if stage["processed"] is None else str(stage["processed"])
        lines.append(f"{stage['name']:<10} {stage['status']:<24} {seconds:>10} {processed:>10}")
    lines.append(f"{'total':<10} {'':<24} {report['measured_seconds_total']:>10.3f} "
                 f"{report['collection']['files']:>10}")
    lines.append("not measured: " + ", ".join(report["not_measured"]["stages"]))
    return "\n".join(lines)


# --- the run log this rig writes and then reads ---------------------------------------


def attach_run_log(path: Path, cfg: Config) -> None:
    """Point the product's own logging at `path` and open it.

    Through `SORTA_LOG_FILE` and `configure_logging` rather than by calling
    `setup_file_logging` directly: `configure_logging` is what every command funnels
    through, and it is also the thing that lowers the `sorta` logger to INFO so the
    `stage=` lines are not dropped before any handler sees them.

    `log_environment` is not decoration here — its `sorta: <version>` header is what
    `read_measurements` matches a timing's build against, and without it every
    measurement of this run is discarded as unvouched-for.
    """
    os.environ[runlog.ENV_LOG_FILE] = str(path)
    configure_logging(cfg.log_level)
    runlog.log_environment()


def detach_run_log(path: Path) -> None:
    """Close the file sink this run opened.

    TRAP: a RotatingFileHandler keeps the file open, and on Windows an open handle is
    enough to stop the directory it lives in from being removed — which is how a test's
    temporary directory turns into a `PermissionError` at teardown.
    """
    wanted = os.path.normcase(os.path.abspath(str(path)))
    root = logging.getLogger()
    for handler in list(root.handlers):
        base = getattr(handler, "baseFilename", None)
        if base is not None and os.path.normcase(os.path.abspath(base)) == wanted:
            root.removeHandler(handler)
            handler.close()


# --- the database this rig is allowed to write ----------------------------------------

# The two names the product ships and the owner runs. A measurement has no business in
# either, and the flag below is what makes an exception a decision rather than a typo.
PRODUCTION_DB_NAMES = ("photos.db", "sorta.db")

DEFAULT_DB = "cpu_tier_measure.db"
DEFAULT_OUT = "cpu_tier_measure.json"


def _same_file(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def is_production_db(db: Path, configured: Path | None = None) -> bool:
    """Would writing here touch a real index? By name, and by what config.yaml points at."""
    if db.name.lower() in PRODUCTION_DB_NAMES:
        return True
    return configured is not None and _same_file(db, configured)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--src", help="the collection to measure (overrides config sources)")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"a database of its own for the measurement (default {DEFAULT_DB}); "
                             "the production index needs --allow-real-db")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                        help=f"stages to measure (default {', '.join(DEFAULT_STAGES)} — the "
                             "base tier, without faces and without the deep tier)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"the JSON report (default {DEFAULT_OUT})")
    parser.add_argument("--log", help="the run log to write and read the timings back from "
                                      "(default: the database path with a .log suffix)")
    parser.add_argument("--allow-real-db", action="store_true",
                        help="measure into photos.db/sorta.db anyway — say it out loud")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stages = select_stages(args.stages)

    checks = cpu_only_checks(import_or_none("torch"), import_or_none("onnxruntime"))
    refused = refusals(checks)
    if refused:
        print("REFUSED: this stack cannot be proven to be CPU-only, so a number from it "
              "would be about something else.")
        for check in refused:
            print(f"  {check.name}: {check.detail}")
        print(f"  interpreter: {sys.executable}")
        print("Install a venv of its own (`uv sync --extra cpu`) and run the rig from that "
              "one, not from the repository's.")
        return 2

    cfg = load_config(args.config)
    db = Path(args.db)
    if is_production_db(db, cfg.database) and not args.allow_real_db:
        print(f"REFUSED: {db} is a production index, and a measurement may not write into "
              "one. Point --db somewhere else, or pass --allow-real-db if you mean it.")
        return 2
    cfg.database = db
    if args.src:
        cfg.sources = [Path(args.src).resolve()]
    if not cfg.sources:
        raise SystemExit("no collection to measure: pass --src <dir> or list sources in "
                         f"{args.config}")

    log = Path(args.log) if args.log else db.with_suffix(".log")
    out = Path(args.out)
    started = datetime.now()
    attach_run_log(log, cfg)
    conn = connect(db)
    try:
        counts = run_stages(stages, cfg, conn)
        outcomes = collect_outcomes(measurements_since(log, started), counts,
                                    [stage.name for stage in stages])
        report = build_report(checks=checks, outcomes=outcomes, machine=machine_facts(),
                              collection=collection_facts(cfg, conn), log=log,
                              started=started, finished=datetime.now())
    finally:
        conn.close()
        detach_run_log(log)

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    print(f"report: {out}")
    print(f"run log: {log}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
