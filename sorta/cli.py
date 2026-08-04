"""CLI. Typer if available; otherwise a minimal argparse fallback (for CI/sandboxes)."""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import typer
except ImportError:  # pragma: no cover — a sandbox/CI without typer, see _argparse_main
    _TYPER_AVAILABLE = False
else:
    _TYPER_AVAILABLE = True

from . import __version__, imaging
from .config import VLM_QUALITY_SCOPES, configure_logging, load_config
from .db import connect, reset_index
from .dedup import assign_duplicates, compute_phashes, near_duplicate_groups
from .diagnostics import (
    geo_data_health,
    gpu_health,
    warn_if_geo_data_missing,
    warn_if_gpu_mismatch,
)
from .events import add_manual_event, build_events, rename_event
from .faces import (
    CLUSTER_PHASE_CLUSTER,
    CLUSTER_PHASE_INHERIT,
    CLUSTER_PHASE_READ,
    CLUSTER_PHASE_WRITE,
    detect_and_cluster,
    export_contact_sheet,
    label_cluster,
)
from .faces import merge as merge_clusters
from .geo import clear_geo_cache, geo_cache_size, resolve_places
from .i18n import Lang, normalize_lang
from .i18n import cli_text as _t
from .indexer import index as run_index
from .indexer import excludes_path, refresh_exif, save_excludes
from .junk import (
    CLASSIFY_PHASE_CLIP,
    CLASSIFY_PHASE_OCR,
    CLASSIFY_PHASE_VLM,
    CLASSIFY_PHASE_WRITE,
)
from .junk import classify as classify_junk
from .landmarks import Classifier, clip_classifier, detect_landmarks
from .naming import name_events, naming_settings
from .progress import progress_task
from .runlog import default_log_path, log_environment, observe, stage_timer
from .search import REASON_OTHER_MODEL, EmbeddingsMissing, file_paths, search_text
from .sorter import SELECTORLESS_ALBUM_KINDS, plan_album, plan_and_sort
from .sorter import undo as undo_batch


def _configure_runtime() -> None:
    """Process-wide setup that must happen before any ML import.

    Both of these are read at import time by the libraries they affect, so they
    cannot be moved next to the model code.
    """
    _ensure_utf8_console()
    from .offline import configure_model_offline

    configure_model_offline()


def _ensure_utf8_console() -> None:
    """The Windows console defaults to cp1251 — it does not encode characters like
    `->` arrows, `⚠`, or emoji in the output, which makes print/rich (incl. `--help`)
    crash with UnicodeEncodeError. We force UTF-8 at the CLI entry point (like
    scripts/check.py)."""
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc and enc.lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except (AttributeError, ValueError, OSError):  # pragma: no cover — environment without reconfigure
                pass


# --- F112: the output language ----------------------------------------------
# Every user-visible string of this module goes through `i18n.cli_text` (imported as
# `_t`) with the language from `cfg.language`. The commands that load the config
# anyway read it off `cfg`; the few checks that fire BEFORE the config is read (the
# `typer.BadParameter` guards) use `_lang_of` below.
#
# F114: the `--help` texts are in that same catalog now. They could not be: a
# `typer.Option(..., help=...)` runs when this module is imported, and by then nothing
# has read the config. So the interface is no longer built at import time either — it
# is assembled by `build_app(lang)` after `_startup_lang()` has peeked at argv for
# `--config` and read the language off it.

def _lang(cfg) -> Lang:
    """The output language of a command that already holds a loaded config."""
    return normalize_lang(getattr(cfg, "language", None))


def _lang_of(config_path: str) -> Lang:
    """The output language for a check that runs before the config is loaded.

    An unreadable/absent config.yaml must not swallow the very message it was needed
    for (a bad `--geo` value is still a bad `--geo` value without a config), so any
    failure here falls back to the default language.
    """
    try:
        return _lang(load_config(config_path))
    except Exception:  # noqa: BLE001 — any unreadable config, the message still goes out
        return normalize_lang(None)


_DEFAULT_CONFIG = "config.yaml"


def _peek_config_path(argv: list[str]) -> str:
    """The value of `--config`/`-c` in a command line, without parsing it.

    F114: the help language comes out of the config, and the config path comes out of
    the command line — but the command line cannot be parsed yet, because the parser is
    what we are about to build. So this only LOOKS: it consumes nothing, validates
    nothing and never raises. Whatever is wrong with the arguments is still typer's to
    say, in typer's words, a moment later.

    Every spelling click accepts is recognised — `--config x`, `--config=x`, `-c x`,
    `-c=x`, `-cx` — and a repeated flag keeps its LAST value, the way click does. A bare
    `--` ends the scan: after it click sees only positional arguments.
    """
    path = _DEFAULT_CONFIG
    rest = list(argv)
    while rest:
        token = rest.pop(0)
        if token == "--":
            break
        if token in ("--config", "-c"):
            if rest:
                path = rest.pop(0)
        elif token.startswith("--config="):
            path = token[len("--config="):]
        elif token.startswith("-c="):
            path = token[len("-c="):]
        elif token.startswith("-c") and not token.startswith("--"):
            path = token[2:]
    return path


def _startup_lang(argv: list[str] | None = None) -> Lang:
    """The interface language, decided before anything has parsed the command line.

    `_lang_of` swallows every read error, so `--help` keeps working with no config at
    all and with a broken one — the person reading the help is precisely the one who
    has not set anything up yet.
    """
    return _lang_of(_peek_config_path(sys.argv[1:] if argv is None else list(argv)))


# --- Stage summaries (a single format for the standalone commands and the `run` pipeline) ----
# Each helper returns a ready summary string for a step (multi-line where needed).
# Used BOTH by the same-named command `_cmd_<step>` AND by the `_pipeline_steps`
# step, so the output does not diverge (backlog #9 / F20).

def _summarize_index(stats, dups: int, lang: Lang) -> str:
    return _t("cli.index.done", lang, added=stats.added, updated=stats.updated,
              skipped=stats.skipped, errors=stats.errors, dups=dups)


def _summarize_geo(stats, lang: Lang) -> str:
    return _t("cli.geo.done", lang, total=stats.total, exact_gps=stats.exact_gps,
              session_inferred=stats.session_inferred,
              trip_inferred=stats.trip_inferred,
              path_inferred=stats.path_inferred, unknown=stats.unknown)


def _summarize_landmarks(stats, lang: Lang) -> str:
    lines = [_t("cli.landmarks.done", lang, scanned=stats.scanned, matched=stats.matched)]
    for name, n in sorted(stats.by_landmark.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name}: {n}")  # landmark names are data, not chrome
    return "\n".join(lines)


# F84: captions for the phases `cluster_faces` reports — the rich bar shows them in
# its description, so the `faces` step no longer goes silent for the whole of
# clustering. Keys: sorta.faces.CLUSTER_PHASE_*.
def _cluster_phase_labels(lang: Lang) -> dict[str, str]:
    return {
        CLUSTER_PHASE_READ: _t("cli.phase.cluster_read", lang),
        CLUSTER_PHASE_CLUSTER: _t("cli.phase.cluster_cluster", lang),
        CLUSTER_PHASE_INHERIT: _t("cli.phase.cluster_inherit", lang),
        CLUSTER_PHASE_WRITE: _t("cli.phase.cluster_write", lang),
    }


# F100: the same for the junk stage. Keys: sorta.junk.CLASSIFY_PHASE_*. The VLM one
# is the phase that matters here: with the deep tier on it is the long half of the
# stage AND the one that changes the denominator under the reader — the counter
# switches from every frame to the candidates of the gate (24 196 -> 7 896 on the
# live run of 2026-07-28). Without a caption the bar simply restarts at zero against
# a smaller number, which reads as a bar that lost its place rather than as a new
# kind of work. An unknown key is shown as-is by TaskProgress.phase, so a phase added
# later is never fatal here — it just goes unlabelled until someone names it.
def _junk_phase_labels(lang: Lang) -> dict[str, str]:
    return {
        CLASSIFY_PHASE_CLIP: _t("cli.phase.junk_clip", lang),
        CLASSIFY_PHASE_OCR: _t("cli.phase.junk_ocr", lang),
        CLASSIFY_PHASE_VLM: _t("cli.phase.junk_vlm", lang),
        CLASSIFY_PHASE_WRITE: _t("cli.phase.junk_write", lang),
    }


def _summarize_faces(face_stats, cl_stats, lang: Lang) -> str:
    lines = [
        _t("cli.faces.detected", lang, files=face_stats.files_processed,
           faces=face_stats.faces_found, no_faces=face_stats.no_face_files,
           errors=face_stats.errors),
        _t("cli.faces.clusters", lang, clusters=cl_stats.clusters,
           clustered=cl_stats.faces - cl_stats.noise, noise=cl_stats.noise,
           labels_kept=cl_stats.labels_kept),
    ]
    if cl_stats.malformed:
        lines.append(_t("cli.faces.malformed", lang, n=cl_stats.malformed))
    return "\n".join(lines)


def _summarize_events(stats, lang: Lang) -> str:
    return _t("cli.events.done", lang, auto_events=stats.auto_events,
              auto_files=stats.auto_files, names_preserved=stats.names_preserved,
              manual_events=stats.manual_events, manual_files=stats.manual_files)


def _summarize_junk(stats, lang: Lang) -> str:
    kinds = ", ".join(f"{v}: {n}" for v, n in sorted(stats.by_verdict.items()))
    line = _t("cli.junk.done", lang, processed=stats.processed, total=stats.total,
              kinds=kinds)
    # F68: makes incrementality observable — on a repeat run with nothing new this
    # should account for everything and `processed` should be 0.
    if getattr(stats, "skipped_incremental", 0):
        line += _t("cli.junk.skipped_incremental", lang, n=stats.skipped_incremental)
    if getattr(stats, "vlm_candidates", 0):
        line += _t("cli.junk.vlm", lang, applied=stats.vlm_applied,
                   candidates=stats.vlm_candidates)
    return line


def _cmd_index(config_path: str, src: str | None = None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    if src:  # a positional source overrides config sources for this run
        cfg.sources = [Path(src).resolve()]
    if not cfg.sources:
        raise ValueError(_t("cli.index.no_source", lang))
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.index", lang)) as cb:
        stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        dups = assign_duplicates(conn, cfg.dedup.canonical_strategy)
    print(_summarize_index(stats, dups, lang))


def _summarize_refresh(stats, lang: Lang) -> str:
    return _t("cli.refresh.done", lang, scanned=stats.scanned, updated=stats.updated,
              gps=stats.recovered_gps, dates=stats.recovered_date,
              empty=stats.still_empty, errors=stats.errors)


def _cmd_refresh_exif(config_path: str) -> None:
    """F71: re-read metadata of already-indexed files.

    A plain `index` skips them — `_needs_update` compares path+size+mtime and none of
    those changed, only the exiftool flag did.
    """
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with stage_timer("refresh-exif") as stage, progress_task(
            _t("cli.progress.refresh_exif", lang)) as cb:
        stats = refresh_exif(cfg, conn, progress=observe(stage, cb))
    print(_summarize_refresh(stats, lang))
    if stats.recovered_gps:
        print(_t("cli.refresh.rerun_geo", lang))


def _cmd_add_excludes(config_path: str, src: str | None, values: list[str]) -> None:
    """F81: persist --exclude-dir into the excludes file before indexing.

    Written rather than applied for this run only: the UI reads the same file, so a
    folder excluded from the CLI stays excluded in the web app, and the next run does
    not silently start scanning it again.
    """
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    root = Path(src).resolve() if src else (cfg.sources[0] if cfg.sources else None)
    if root is None:
        raise SystemExit(_t("cli.excludes.no_source", lang))
    path = excludes_path(cfg)
    accepted = save_excludes(path, root, values)
    print(_t("cli.excludes.saved", lang, root=root, values=", ".join(accepted) or "—"))
    print(_t("cli.excludes.file", lang, path=path))


def _cmd_geo(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.geo", lang)) as cb:
        stats = resolve_places(cfg, conn, progress=cb)
    print(_summarize_geo(stats, lang))


def _cmd_faces(config_path: str, rescan: bool = False, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    title = _t("cli.progress.faces_rescan" if rescan else "cli.progress.faces", lang)
    with progress_task(title, phase_labels=_cluster_phase_labels(lang)) as cb:
        face_stats, cl_stats = detect_and_cluster(cfg, conn, progress=cb,
                                                  rescan=rescan, limit=limit)
    print(_summarize_faces(face_stats, cl_stats, lang))


def _cmd_landmarks(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.landmarks", lang)) as cb:
        stats = detect_landmarks(cfg, conn, progress=cb)
    print(_summarize_landmarks(stats, lang))


def _cmd_phash(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.phash", lang)) as cb:
        n = compute_phashes(cfg, conn, progress=cb)
    print(_t("cli.phash.done", lang, n=n))


def _quality_overrides(cfg, *, pets: bool | None = None, quality: bool | None = None,
                       quality_scope: str | None = None):
    """F127: the frame-quality knobs of ONE run, from flags instead of config.yaml.

    The same principle `--deep`/`--geo` have followed since F50: a copy of the config
    for this run (`dataclasses.replace`), never a write to the file. `None` means the
    flag was not passed and the value stays as the config has it — which is what makes
    `--no-pets` able to switch OFF what `features.pets: true` switched on, instead of
    the flag only ever being able to add.

    The three live in one helper because they are one cascade and two commands
    (`junk` and `run`) offer all three: `features.pets` is computed inside the junk
    stage's CLIP call, and `vlm.quality`/`vlm.quality_scope` decide which of those
    frames the quality VLM is then asked about.
    """
    if pets is not None:
        cfg = dataclasses.replace(
            cfg, features=dataclasses.replace(cfg.features, pets=pets))
    changed: dict[str, object] = {}
    if quality is not None:
        changed["quality"] = quality
    if quality_scope is not None:
        changed["quality_scope"] = quality_scope
    if changed:
        cfg = dataclasses.replace(cfg, vlm=dataclasses.replace(cfg.vlm, **changed))
    return cfg


def _cmd_junk(config_path: str, *, pets: bool | None = None,
              quality: bool | None = None, quality_scope: str | None = None) -> None:
    cfg = _quality_overrides(load_config(config_path), pets=pets, quality=quality,
                             quality_scope=quality_scope)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.junk", lang),
                       phase_labels=_junk_phase_labels(lang)) as cb:
        stats = classify_junk(cfg, conn, progress=cb)
    print(_summarize_junk(stats, lang))


def _cmd_classify(config_path: str) -> None:
    """F165: the verdicts alone — the half of the junk stage that runs before faces.

    No flags of its own: the three the `junk` command offers (`--pets`,
    `--quality`/`--quality-scope`) all belong to the frame-quality cascade, and that
    cascade is precisely what this half does not run.
    """
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.classify", lang),
                       phase_labels=_junk_phase_labels(lang)) as cb:
        stats = classify_junk(cfg, conn, verdicts_only=True, progress=cb)
    print(_summarize_junk(stats, lang))


def _cmd_events(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.events", lang)) as cb:
        stats = build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)  # naming by the provider (template by default)
    print(_summarize_events(stats, lang))


# --- The `sorta run` pipeline -----------------------------------------------

class _LazySharedClassifier:
    """Builds the real CLIP classifier on the FIRST call and reuses it between
    landmarks and junk within one `run` (F19): their image features share the
    `CachingFeatureClassifier` cache, so each photo is decoded+encoded once for the
    whole run, not separately in landmarks, the junk classes, and the document pass.

    Laziness preserves incrementality: a `run` with no new data (landmarks and junk
    with no rows) does NOT invoke the classifier and the CLIP model is not loaded.
    The factory is injected — in tests it is replaced with a fake without ML.
    """

    def __init__(self, factory: Callable[[], Classifier]) -> None:
        self._factory = factory
        self._real: Classifier | None = None

    def __call__(self, paths: list[str], prompts: list[str]) -> np.ndarray:
        if self._real is None:
            self._real = self._factory()
        return self._real(paths, prompts)

    def features(self, paths: list[str]) -> list[np.ndarray | None]:
        """The CLIP vectors of the paths already scored — see `ui._LazyClassifierHolder`.

        F146: the hole was found on the UI wrapper, and this one is built on the same
        pattern, so it is closed on both rather than only where it was noticed. The junk
        stage looks for `features` on the object it was handed to decide whether it can
        fill `clip_embeddings`; a wrapper that forwards `__call__` alone switches that
        half off without a word.

        Laziness is untouched: an unbuilt classifier has scored nothing and so has no
        vector to hand back, which is exactly what None per path means here.
        """
        features_of = getattr(self._real, "features", None)
        if not callable(features_of):
            return [None] * len(paths)
        return list(features_of(paths))


# F53/#39: faces and events — the heaviest/longest steps, not needed for the basic
# scenario (cities + dupes) — opt-in via --faces/--events, default off.
# `_pipeline_steps()` still builds the FULL list; filtering is up to the caller
# (`_cmd_run`), see below.
_OPTIONAL_STAGES = ("faces", "events")


def _pipeline_steps() -> list[tuple[str, object]]:
    """Full-analysis steps in dependency order: (name, fn(cfg, conn, cb)).

    Order matters: geo before landmarks (landmarks writes only unknown places),
    faces before junk (junk uses the face-presence signal). landmarks and junk share
    ONE lazy CLIP classifier (F19) — a shared image-feature cache for the whole run.

    F165: `classify` is the front half of `junk` — the verdicts, which depend on nothing
    here — and it runs before `faces` so that the faces stage can skip the screenshots and
    the documents instead of detecting on them first and being told afterwards. The back
    half keeps its place after `faces`, because everything left in it (the quality cascade,
    `face_sharpness`, the animal cascade) reads the table that stage writes. Both halves
    share the classifier with `landmarks` for the same F19 reason: within one run the
    second call scores frames the first one already encoded.
    """
    shared: dict[str, _LazySharedClassifier] = {}

    def _clip(cfg) -> _LazySharedClassifier:
        clf = shared.get("clip")
        if clf is None:
            clf = shared["clip"] = _LazySharedClassifier(
                lambda: clip_classifier(naming_settings(cfg)))
        return clf

    def _index(cfg, conn, cb) -> str:
        stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        dups = assign_duplicates(conn, cfg.dedup.canonical_strategy)
        return _summarize_index(stats, dups, _lang(cfg))

    def _geo(cfg, conn, cb) -> str:
        return _summarize_geo(resolve_places(cfg, conn, progress=cb), _lang(cfg))

    def _landmarks(cfg, conn, cb) -> str:
        return _summarize_landmarks(
            detect_landmarks(cfg, conn, classifier=_clip(cfg), progress=cb), _lang(cfg))

    def _faces(cfg, conn, cb) -> str:
        face_stats, cl_stats = detect_and_cluster(cfg, conn, progress=cb)
        return _summarize_faces(face_stats, cl_stats, _lang(cfg))

    def _events(cfg, conn, cb) -> str:
        stats = build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)
        return _summarize_events(stats, _lang(cfg))

    def _classify(cfg, conn, cb) -> str:
        return _summarize_junk(
            classify_junk(cfg, conn, classifier=_clip(cfg), verdicts_only=True,
                          progress=cb), _lang(cfg))

    def _junk(cfg, conn, cb) -> str:
        return _summarize_junk(
            classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb), _lang(cfg))

    return [
        ("index", _index),
        ("geo", _geo),
        ("landmarks", _landmarks),
        ("classify", _classify),
        ("faces", _faces),
        ("events", _events),
        ("junk", _junk),
    ]


def _cmd_run(config_path: str, by: str | None = None, dest: str | None = None,
             deep: bool | None = None, geo: str | None = None,
             faces: bool = False, events: bool = False,
             src: str | None = None, pets: bool | None = None,
             quality: bool | None = None, quality_scope: str | None = None) -> None:
    """`deep`/`geo` (F50/#34) — an opt-in override for THIS run, not written to
    config.yaml: `deep` -> `naming.vlm_enabled`, `geo` ("offline"|"online") ->
    `geo.provider`. None (flag not passed) -> the value stays from config.

    `src` (F59) — the source directory for this run, overrides config.sources (like
    the positional src of `index`).

    `faces`/`events` (F53/#39) — opt-in steps, default off: the basic run builds only
    `index/geo/landmarks/classify/junk`, the heaviest/longest steps are skipped.
    Independent of each other and of `deep`/`geo`.

    `pets`/`quality`/`quality_scope` (F127) — the same kind of per-run override as
    `deep`, on the frame-quality cascade (see `_quality_overrides`). NOT stages: they
    change what the `junk` stage computes and leave the list of steps as it was."""
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    log_environment()  # F69: versions, package origin, GPU, geo data — once per run
    warn_if_gpu_mismatch()  # F63: loud if torch is CPU-only while a GPU is expected
    warn_if_geo_data_missing()  # F65: an unreadable geo base empties every place
    if src:  # an explicit source overrides config sources for this run
        cfg.sources = [Path(src).resolve()]
    if not cfg.sources:
        raise SystemExit(_t("cli.run.no_source", lang))
    if deep is not None:
        cfg = dataclasses.replace(cfg, naming=dataclasses.replace(cfg.naming, vlm_enabled=deep))
    if geo is not None:
        cfg = dataclasses.replace(cfg, geo=dataclasses.replace(cfg.geo, provider=geo))
    cfg = _quality_overrides(cfg, pets=pets, quality=quality,
                             quality_scope=quality_scope)
    conn = connect(cfg.database)
    try:
        enabled_optional = {"faces": faces, "events": events}
        steps = [(name, fn) for name, fn in _pipeline_steps()
                 if name not in _OPTIONAL_STAGES or enabled_optional[name]]
        for i, (name, fn) in enumerate(steps, 1):
            print(_t("cli.run.stage", lang, index=i, total=len(steps), name=name))
            # F69: the per-stage timing goes to the run log, so "which stage ate the
            # three hours" is answerable after the fact instead of by eye.
            # F100: one map for the whole pipeline — the keys of the two stages that
            # report phases do not overlap (cluster_* vs junk_*), and the loop does not
            # know which stage it is about to run.
            with stage_timer(name) as stage, progress_task(
                    name,
                    phase_labels={**_cluster_phase_labels(lang),
                                  **_junk_phase_labels(lang)}) as cb:
                # F166: the run log reads the same callback the bar does, so a stage
                # with no phases of its own (index, geo, faces, ...) also says where it
                # is while it runs instead of only where it ended.
                summary = fn(cfg, conn, observe(stage, cb))  # type: ignore[operator]
            for line in str(summary).splitlines():
                print(f"  {line}")
        if by:
            plan_dest = Path(dest) if dest else None  # None -> in-place (source root)
            print(_t("cli.run.plan", lang, by=by, dest=dest or "in-place"))
            with progress_task(f"plan {by}") as cb:
                plan_and_sort(cfg, conn, by, plan_dest, apply=False, progress=cb)
    finally:
        conn.close()
    print("\n" + _t("cli.run.finished", lang))


def _cmd_stats(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    total = q("SELECT COUNT(*) FROM files WHERE error IS NULL")
    if not total:
        print(_t("cli.stats.empty", lang))
        return
    print(_t("cli.stats.files", lang, total=total,
             errors=q("SELECT COUNT(*) FROM files WHERE error IS NOT NULL")))
    with_gps = q("SELECT COUNT(*) FROM files WHERE gps_lat IS NOT NULL")
    print(_t("cli.stats.gps", lang, n=with_gps, pct=with_gps * 100 // total))
    for src in ("exif", "filename", "mtime"):
        n = conn.execute("SELECT COUNT(*) FROM files WHERE taken_at_source = ?", (src,)).fetchone()[0]
        print(_t("cli.stats.date_source", lang, source=src, n=n, pct=n * 100 // total))
    print(_t("cli.stats.dupes", lang,
             n=q("SELECT COUNT(*) FROM files WHERE dup_of IS NOT NULL")))
    places_total = q("SELECT COUNT(*) FROM places")
    if places_total:
        print(_t("cli.stats.geo_total", lang, n=places_total))
        for conf, n in conn.execute(
            "SELECT confidence, COUNT(*) FROM places GROUP BY confidence ORDER BY 2 DESC"
        ):
            print(_t("cli.stats.geo_confidence", lang, confidence=conf, n=n,
                     pct=n * 100 // places_total))
    n_faces = q("SELECT COUNT(*) FROM faces WHERE bbox != '[]'")
    if n_faces:
        n_clusters = q("SELECT COUNT(*) FROM face_clusters WHERE merged_into IS NULL")
        n_named = q("SELECT COUNT(*) FROM face_clusters "
                    "WHERE merged_into IS NULL AND label IS NOT NULL")
        print(_t("cli.stats.faces", lang, faces=n_faces, clusters=n_clusters,
                 named=n_named))


def _cmd_dupes(config_path: str, near: bool = False) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    if near:
        have_phash = conn.execute(
            "SELECT COUNT(*) FROM files WHERE phash IS NOT NULL").fetchone()[0]
        if not have_phash:
            print(_t("cli.dupes.no_phash", lang))
            return
        groups = near_duplicate_groups(conn, cfg.index.phash_max_distance)
        if not groups:
            print(_t("cli.dupes.near_none", lang))
            return
        for group in groups:
            print(_t("cli.dupes.near_group", lang, n=len(group)))
            for r in group:
                print(_t("cli.dupes.near_item", lang, path=r["path"], size=r["size"]))
        print("\n" + _t("cli.dupes.near_total", lang, n=len(groups),
                        threshold=cfg.index.phash_max_distance))
        return
    rows = conn.execute(
        """SELECT c.path AS canon, f.path AS dup FROM files f
           JOIN files c ON f.dup_of = c.id ORDER BY c.path"""
    ).fetchall()
    if not rows:
        print(_t("cli.dupes.exact_none", lang))
        return
    for r in rows:
        print(_t("cli.dupes.exact_item", lang, dup=r["dup"], canon=r["canon"]))
    print("\n" + _t("cli.dupes.exact_total", lang, n=len(rows)))


def _stub(name: str, doc: str, lang: Lang):
    def cmd(*_a, **_k):
        print(_t("cli.stub.next_phase", lang, name=name, doc=doc))
        raise SystemExit(2)
    return cmd


# --- The rest of the command bodies -----------------------------------------
# What `build_app` registers with typer below is a thin shell: flags, arguments and
# their help. The work itself lives here, in functions that know nothing about the
# interface — which is also what keeps them callable from the argparse fallback.


def _cmd_doctor(config_path: str) -> None:
    """F112: `--config` is here only to know the output language — the command still
    works without a readable config (`_lang_of` falls back to the default), it just
    prints in the default language then. The two health summaries below come from
    diagnostics.py, which this feature does not own, so they stay as that module
    writes them.
    """
    lang = _lang_of(config_path)
    print(gpu_health().summary)
    # F65: the geo base failing to load is invisible at runtime (every coordinate just
    # resolves to an empty place), so the doctor has to state it outright.
    geo = geo_data_health()
    print(("" if geo.available else "⚠ ") + geo.summary)
    print(_t("cli.doctor.log", lang, path=default_log_path()))
    print(_t("cli.cache.preview_dir", lang, path=imaging.preview_dir())
          + ("" if imaging.preview_cache_enabled()
             else _t("cli.cache.preview_disabled", lang)))


def _cmd_cache(config_path: str, *, clear: bool = False, clear_geo: bool = False,
               preview_max_gb: float | None = None) -> None:
    # F127: the ceiling for this run, without editing `imaging.preview_cache_max_gb`.
    # The env variable IS the override: imaging.py is a leaf module the pool workers
    # call with a path and nothing else, so the config file seeds these variables and
    # only when they are not already set (config._apply_imaging_config) — an exported
    # variable wins over config.yaml, and so does the flag. `0` = no ceiling, as in the
    # config, and it has to be set rather than skipped: it is the value that switches a
    # configured ceiling OFF for this run.
    if preview_max_gb is not None:
        os.environ[imaging.ENV_PREVIEW_MAX_GB] = str(preview_max_gb)
    cfg = load_config(config_path)  # applies the imaging: section onto the env
    lang = _lang(cfg)
    directory = imaging.preview_dir()
    if clear_geo:
        conn = connect(cfg.database)
        try:
            removed = clear_geo_cache(conn)
        finally:
            conn.close()
        print(_t("cli.cache.geo_cleared", lang, n=removed))
    if clear:
        imaging.preview_cache_clear()
        print(_t("cli.cache.preview_cleared", lang, path=directory))
    if clear or clear_geo:
        return
    files, size = imaging.preview_cache_size()
    print(_t("cli.cache.preview_dir", lang, path=directory))
    print(_t("cli.cache.preview_stats", lang, files=files, size_gb=size / 1e9))
    # F117: a size means little without the bound it is measured against.
    limit_gb = imaging.preview_cache_max_gb()
    if limit_gb > 0:
        print(_t("cli.cache.preview_limit", lang, limit_gb=limit_gb,
                 percent=100.0 * size / (limit_gb * 1e9)))
    else:
        print(_t("cli.cache.preview_no_limit", lang))
    conn = connect(cfg.database)
    try:
        print(_t("cli.cache.geo_size", lang, n=geo_cache_size(conn)))
    finally:
        conn.close()


def _cmd_ui(config_path: str, port: int) -> None:
    from .ui import serve as ui_serve
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    warn_if_gpu_mismatch()  # F63: loud if torch is CPU-only while a GPU is expected
    conn = connect(cfg.database)
    ui_serve(cfg, conn, port=port, config_path=config_path)


def _cmd_faces_label(config_path: str, cluster_id: int, name: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    root = label_cluster(connect(cfg.database), cluster_id, name)
    print(_t("cli.faces.labeled", _lang(cfg), cluster=root, name=name))


def _cmd_faces_merge(config_path: str, src_id: int, dst_id: int) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    root = merge_clusters(connect(cfg.database), src_id, dst_id)
    print(_t("cli.faces.merged", _lang(cfg), src=src_id, dst=root))


def _cmd_faces_sheet(config_path: str, cluster_id: int, out_html: Path) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    n = export_contact_sheet(connect(cfg.database), cluster_id, out_html)
    print(_t("cli.faces.sheet_done", _lang(cfg), n=n, path=out_html))


def _cmd_events_rename(config_path: str, event_id: int, name: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    rename_event(connect(cfg.database), event_id, name)
    print(_t("cli.events.renamed", _lang(cfg), event_id=event_id, name=name))


def _cmd_events_add(config_path: str, name: str, date_from: str,
                    date_to: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    eid = add_manual_event(connect(cfg.database), name, date_from, date_to)
    print(_t("cli.events.added", _lang(cfg), event_id=eid, name=name,
             date_from=date_from, date_to=date_to))


def _cmd_sort(config_path: str, by: str, dest: Path | None = None, *,
              apply: bool = False, copy: bool = False,
              where: list[str] | None = None, thumbnails: bool = False,
              dedupe: bool = False, delete_worse_dupes: bool = False,
              exclude: list[str] | None = None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(f"sort --by {by}") as cb:
        report = plan_and_sort(cfg, conn, by, dest, apply=apply, copy=copy,
                               where=where or [],
                               thumbnails=thumbnails, dedupe=dedupe,
                               delete_worse_dupes=delete_worse_dupes,
                               exclude=exclude or [], progress=cb)
    if apply:
        # Copy and move are two whole sentences, not one sentence with the verb
        # pasted in: the word order around the counts is not the same everywhere.
        extra = (_t("cli.sort.deleted_dupes", lang, n=report.deleted)
                 if report.deleted else "")
        print(_t("cli.sort.copied" if copy else "cli.sort.moved", lang,
                 moved=report.moved, in_place=report.skipped_in_place,
                 failed=report.failed, extra=extra))


def _search_unavailable(exc: EmbeddingsMissing, lang: Lang) -> str:
    """F129: why a search cannot run, in a sentence that says what to do about it.

    An empty result would read as "nothing matched", which is the one thing this state is
    not — and the two states need two different sentences: a table that was never filled is
    fixed by running the junk stage, a table full of another model's vectors by running it
    again after the model change (F128 recomputes them; it does not mix them in).
    """
    if exc.reason == REASON_OTHER_MODEL:
        return _t("cli.search.other_model", lang, model=exc.model, n=exc.total)
    return _t("cli.search.no_embeddings", lang)


def _cmd_search(config_path: str, query: str, limit: int | None = None) -> None:
    """F129: print the CLIP ranking for a query — paths and scores, best first.

    The scores are printed because they are the only thing that tells a reader how far down
    the list stopped being about their words: this ranks, it does not classify, so the line
    where a query runs out is something only a human can see. The lines themselves carry no
    words in any language (a rank, a score, a path — data, like the landmark names in
    `_summarize_landmarks`); the sentence around them goes through the catalog.
    """
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    if not query.strip():
        raise SystemExit(_t("cli.search.empty_query", lang))
    conn = connect(cfg.database)
    try:
        try:
            hits = search_text(cfg, conn, query, limit=limit)
        except EmbeddingsMissing as exc:
            raise SystemExit(_search_unavailable(exc, lang)) from None
        paths = file_paths(conn, [file_id for file_id, _score in hits])
        for rank, (file_id, score) in enumerate(hits, 1):
            print(f"{rank:>4}. {score:.3f}  {paths.get(file_id, '')}")
        print(_t("cli.search.done", lang, n=len(hits), query=query))
    finally:
        conn.close()


def _cmd_album(config_path: str, kind: str, selector: str, dest: Path, *,
               copy: bool = False, move: bool = False,
               where: list[str] | None = None, name: str | None = None,
               apply: bool = False) -> None:
    mode = "move" if move else "copy" if copy else "link"
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(f"album {kind} {selector}"):
        try:
            report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                where=where or [], apply=apply, album_name=name)
        except EmbeddingsMissing as exc:  # F129: only kind='query' can raise this
            raise SystemExit(_search_unavailable(exc, lang)) from None
    if apply:
        extra = (_t("cli.album.blocked_multi", lang, n=report.blocked_multi)
                 if report.blocked_multi else "")
        print(_t("cli.album.done", lang, name=report.album_name,
                 transferred=report.transferred, failed=report.failed, extra=extra))


def _cmd_reset(config_path: str, *, clear_geo: bool = False,
               confirm: Callable[[str], None] | None = None) -> None:
    """`confirm` is injected instead of called here: the question is typer's
    (`typer.confirm(..., abort=True)`), and this function has to stay callable where
    typer is not installed. None means nothing is asked — which is what `--yes` does.
    """
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    if confirm is not None:
        confirm(_t("cli.reset.confirm", lang,
                   extra=_t("cli.reset.confirm_geo", lang) if clear_geo else ""))
    conn = connect(cfg.database)
    try:
        reset_index(conn, clear_geo=clear_geo)
    finally:
        conn.close()
    print(_t("cli.reset.done", lang,
             extra=_t("cli.reset.done_geo", lang) if clear_geo else ""))


def _cmd_undo(config_path: str, batch: int | None = None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    with progress_task("undo") as cb:
        stats = undo_batch(connect(cfg.database), batch, progress=cb)
    print(_t("cli.undo.done", _lang(cfg), batch=stats.batch_id, undone=stats.undone,
             missing=stats.missing, failed=stats.failed))


# --- Typer interface (primary) ----------------------------------------------
# A factory instead of a module-level application (F114): the help of an option is
# written inside `typer.Option(...)`, which runs when this module is imported — too
# early for anything to know the language. So the interface is assembled here, once,
# after the language has been read. Nothing else about it moved: the same commands,
# the same flags, the same guards raising typer's own errors.
#
# The bodies live in the `_cmd_*` functions above. What is left in each shell is the
# part that IS the interface: the flags, and the two or three checks that have to
# answer with `typer.BadParameter` before any work starts.


def build_app(lang: Lang) -> typer.Typer:
    """The whole command line, with every `--help` text in `lang`."""

    def h(key: str, **fields: object) -> str:
        return _t(key, lang, **fields)

    app = typer.Typer(help=h("cli.help.app", version=__version__))
    cfg_opt = typer.Option(_DEFAULT_CONFIG, "--config", "-c",
                           help=h("cli.help.opt.config"))
    # F127: the frame-quality flags are offered by two commands (`junk` and `run`) and
    # are one and the same override, so they are declared once. `None` by default —
    # "as in config.yaml" — which is what lets `--no-pets` turn OFF what the file
    # turned on (an option that defaulted to False could only ever add).
    pets_opt = typer.Option(None, "--pets/--no-pets", help=h("cli.help.opt.pets"))
    quality_opt = typer.Option(None, "--quality/--no-quality",
                               help=h("cli.help.opt.quality"))
    quality_scope_opt = typer.Option(None, "--quality-scope",
                                     help=h("cli.help.opt.quality_scope"))

    def check_quality_scope(value: str | None, config: str) -> None:
        """A closed list, and a typo has to say so.

        A silent fallback to the default is the failure this guards against: the scope
        decides how many frames a 20 GB model is shown, and a misspelt one would run
        the wrong population for hours without ever printing why.
        """
        if value is not None and value not in VLM_QUALITY_SCOPES:
            raise typer.BadParameter(
                _t("cli.quality.scope_choice", _lang_of(config),
                   values=", ".join(VLM_QUALITY_SCOPES)))

    @app.command(help=h("cli.help.index"))
    def index(
        src: str = typer.Argument(None, help=h("cli.help.index.src")),
        config: str = cfg_opt,
        refresh_exif: bool = typer.Option(
            False, "--refresh-exif", help=h("cli.help.index.refresh_exif")),
        exclude_dir: list[str] = typer.Option(
            None, "--exclude-dir", help=h("cli.help.index.exclude_dir")),
    ):
        if refresh_exif:
            _cmd_refresh_exif(config)
            return
        if exclude_dir:
            _cmd_add_excludes(config, src, exclude_dir)
        _cmd_index(config, src=src)

    @app.command(help=h("cli.help.stats"))
    def stats(config: str = cfg_opt):
        _cmd_stats(config)

    @app.command(help=h("cli.help.dupes"))
    def dupes(
        near: bool = typer.Option(False, "--near", help=h("cli.help.dupes.near")),
        config: str = cfg_opt,
    ):
        _cmd_dupes(config, near=near)

    @app.command(help=h("cli.help.geo"))
    def geo(config: str = cfg_opt):
        _cmd_geo(config)

    @app.command(help=h("cli.help.landmarks"))
    def landmarks(config: str = cfg_opt):
        _cmd_landmarks(config)

    @app.command(help=h("cli.help.phash"))
    def phash(config: str = cfg_opt):
        _cmd_phash(config)

    @app.command(help=h("cli.help.classify"))
    def classify(config: str = cfg_opt):
        _cmd_classify(config)

    @app.command(help=h("cli.help.junk"))
    def junk(
        pets: bool = pets_opt,
        quality: bool = quality_opt,
        quality_scope: str = quality_scope_opt,
        config: str = cfg_opt,
    ):
        check_quality_scope(quality_scope, config)
        _cmd_junk(config, pets=pets, quality=quality, quality_scope=quality_scope)

    @app.command(help=h("cli.help.doctor"))
    def doctor(config: str = cfg_opt):
        _cmd_doctor(config)

    @app.command("cache", help=h("cli.help.cache"))
    def cache_cmd(
        clear: bool = typer.Option(False, "--clear", help=h("cli.help.cache.clear")),
        clear_geo: bool = typer.Option(
            False, "--clear-geo", help=h("cli.help.cache.clear_geo")),
        preview_max_gb: float = typer.Option(
            None, "--preview-max-gb", min=0,
            help=h("cli.help.cache.preview_max_gb")),
        config: str = cfg_opt,
    ):
        _cmd_cache(config, clear=clear, clear_geo=clear_geo,
                   preview_max_gb=preview_max_gb)

    @app.command(help=h("cli.help.ui"))
    def ui(port: int = typer.Option(8756, "--port", help=h("cli.help.ui.port")),
           config: str = cfg_opt):
        _cmd_ui(config, port)

    # A group's own help comes from `typer.Typer(help=...)`, which typer prefers over
    # anything its callback says — so the callbacks below carry no help of their own.
    faces_app = typer.Typer(help=h("cli.help.faces"))
    app.add_typer(faces_app, name="faces")

    @faces_app.callback(invoke_without_command=True)
    def faces_main(
        ctx: typer.Context,
        rescan: bool = typer.Option(
            False, "--rescan", help=h("cli.help.faces.rescan")),
        limit: int = typer.Option(None, "--limit", help=h("cli.help.faces.limit")),
        config: str = cfg_opt,
    ):
        if ctx.invoked_subcommand is not None:
            return
        if limit is not None and not rescan:
            raise typer.BadParameter(
                _t("cli.faces.limit_needs_rescan", _lang_of(config)))
        if limit is not None and limit <= 0:
            raise typer.BadParameter(_t("cli.faces.limit_positive", _lang_of(config)))
        _cmd_faces(config, rescan=rescan, limit=limit)

    @faces_app.command("label", help=h("cli.help.faces.label"))
    def faces_label(cluster_id: int, name: str, config: str = cfg_opt):
        _cmd_faces_label(config, cluster_id, name)

    @faces_app.command("merge", help=h("cli.help.faces.merge"))
    def faces_merge(src_id: int, dst_id: int, config: str = cfg_opt):
        _cmd_faces_merge(config, src_id, dst_id)

    @faces_app.command("sheet", help=h("cli.help.faces.sheet"))
    def faces_sheet(cluster_id: int, out_html: Path, config: str = cfg_opt):
        _cmd_faces_sheet(config, cluster_id, out_html)

    events_app = typer.Typer(help=h("cli.help.events"))
    app.add_typer(events_app, name="events")

    @events_app.callback(invoke_without_command=True)
    def events_main(ctx: typer.Context, config: str = cfg_opt):
        if ctx.invoked_subcommand is None:
            _cmd_events(config)

    @events_app.command("rename", help=h("cli.help.events.rename"))
    def events_rename(event_id: int, name: str, config: str = cfg_opt):
        _cmd_events_rename(config, event_id, name)

    @events_app.command("add", help=h("cli.help.events.add"))
    def events_add(name: str, date_from: str, date_to: str, config: str = cfg_opt):
        _cmd_events_add(config, name, date_from, date_to)

    @app.command(help=h("cli.help.sort"))
    def sort(
        by: str = typer.Option(..., help=h("cli.help.sort.by")),
        dest: Path = typer.Option(None, "--dest", help=h("cli.help.sort.dest")),
        apply: bool = typer.Option(False, "--apply", help=h("cli.help.sort.apply")),
        copy: bool = typer.Option(False, "--copy", help=h("cli.help.sort.copy")),
        where: list[str] = typer.Option(None, "--where", help=h("cli.help.sort.where")),
        thumbnails: bool = typer.Option(
            False, "--thumbnails", help=h("cli.help.sort.thumbnails")),
        dedupe: bool = typer.Option(False, "--dedupe", help=h("cli.help.sort.dedupe")),
        delete_worse_dupes: bool = typer.Option(
            False, "--delete-worse-dupes", help=h("cli.help.sort.delete_worse_dupes")),
        exclude: list[str] = typer.Option(
            None, "--exclude", help=h("cli.help.sort.exclude")),
        config: str = cfg_opt,
    ):
        _cmd_sort(config, by, dest, apply=apply, copy=copy, where=where,
                  thumbnails=thumbnails, dedupe=dedupe,
                  delete_worse_dupes=delete_worse_dupes, exclude=exclude)

    @app.command("search", help=h("cli.help.search"))
    def search_cmd(
        query: str = typer.Argument(..., help=h("cli.help.search.query")),
        limit: int = typer.Option(None, "--limit", min=1,
                                  help=h("cli.help.search.limit")),
        config: str = cfg_opt,
    ):
        _cmd_search(config, query, limit=limit)

    @app.command(help=h("cli.help.album"))
    def album(
        kind: str = typer.Argument(..., help=h("cli.help.album.kind")),
        selector: str = typer.Argument(None, help=h("cli.help.album.selector")),
        dest: Path = typer.Option(..., "--dest", help=h("cli.help.album.dest")),
        copy: bool = typer.Option(False, "--copy", help=h("cli.help.album.copy")),
        move: bool = typer.Option(False, "--move", help=h("cli.help.album.move")),
        where: list[str] = typer.Option(
            None, "--where", help=h("cli.help.album.where")),
        name: str = typer.Option(None, "--name", help=h("cli.help.album.name")),
        apply: bool = typer.Option(False, "--apply", help=h("cli.help.album.apply")),
        config: str = cfg_opt,
    ):
        if copy and move:
            raise typer.BadParameter(
                _t("cli.album.copy_move_exclusive", _lang_of(config)))
        # F127: a slice with nothing to select INSIDE it takes no selector — the
        # collection has a single animal view, and since F139 a single products bucket
        # and a single blurred list — so there `sorta album <kind> --dest ...` is the
        # whole command. For a person and an event the selector is the subject itself —
        # and for a query (F129) it is the words — so a missing one has to be an error
        # here, said out loud, rather than an album quietly gathered from something else.
        # F152: the face slices join that list too, hence the shared constant — the rule
        # is a property of the kinds and belongs where they are declared.
        if kind not in SELECTORLESS_ALBUM_KINDS and not (selector or "").strip():
            raise typer.BadParameter(
                _t("cli.album.selector_required", _lang_of(config)))
        _cmd_album(config, kind, selector or "", dest, copy=copy, move=move,
                   where=where, name=name, apply=apply)

    @app.command(help=h("cli.help.reset"))
    def reset(
        yes: bool = typer.Option(False, "--yes", "-y", help=h("cli.help.reset.yes")),
        clear_geo: bool = typer.Option(
            False, "--clear-geo", help=h("cli.help.reset.clear_geo")),
        config: str = cfg_opt,
    ):
        def ask(text: str) -> None:
            typer.confirm(text, abort=True)  # aborts the command on "no"

        _cmd_reset(config, clear_geo=clear_geo, confirm=None if yes else ask)

    @app.command(help=h("cli.help.undo"))
    def undo(
        batch: int = typer.Option(None, "--batch", help=h("cli.help.undo.batch")),
        config: str = cfg_opt,
    ):
        _cmd_undo(config, batch)

    @app.command(help=h("cli.help.run"))
    def run(
        by: str = typer.Option(None, "--by", help=h("cli.help.run.by")),
        dest: Path = typer.Option(None, "--dest", help=h("cli.help.run.dest")),
        deep: bool = typer.Option(
            None, "--deep/--no-deep", help=h("cli.help.run.deep")),
        geo: str = typer.Option(None, "--geo", help=h("cli.help.run.geo")),
        faces: bool = typer.Option(
            False, "--faces/--no-faces", help=h("cli.help.run.faces")),
        events: bool = typer.Option(
            False, "--events/--no-events", help=h("cli.help.run.events")),
        src: str = typer.Option(None, "--src", help=h("cli.help.run.src")),
        pets: bool = pets_opt,
        quality: bool = quality_opt,
        quality_scope: str = quality_scope_opt,
        config: str = cfg_opt,
    ):
        if geo is not None and geo not in ("offline", "online"):
            raise typer.BadParameter(_t("cli.run.geo_choice", _lang_of(config)))
        check_quality_scope(quality_scope, config)
        _cmd_run(config, by=by, dest=str(dest) if dest else None, deep=deep, geo=geo,
                 faces=faces, events=events, src=src, pets=pets, quality=quality,
                 quality_scope=quality_scope)

    return app


# --- The argparse fallback (no typer) ---------------------------------------
# Localized for the same reason as the interface above: help whose language depends on
# which packages happen to be installed is the least predictable kind there is (F114).

_FALLBACK_COMMANDS = ("index", "stats", "dupes", "geo", "phash", "landmarks", "classify",
                      "junk", "faces", "events", "run")


def _argparse_main(lang: Lang, argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="sorta", description=_t("cli.help.app", lang, version=__version__))
    parser.add_argument("command", choices=list(_FALLBACK_COMMANDS),
                        help=_t("cli.help.fallback.command", lang))
    parser.add_argument("-c", "--config", default=_DEFAULT_CONFIG,
                        help=_t("cli.help.opt.config", lang))
    parser.add_argument("--near", action="store_true",
                        help=_t("cli.help.dupes.near", lang))
    args = parser.parse_args(argv)
    if args.command == "dupes":
        _cmd_dupes(args.config, near=args.near)
    else:
        {"index": _cmd_index, "stats": _cmd_stats, "geo": _cmd_geo, "phash": _cmd_phash,
         "landmarks": _cmd_landmarks, "classify": _cmd_classify, "junk": _cmd_junk,
         "faces": _cmd_faces, "events": _cmd_events,
         "run": _cmd_run}[args.command](args.config)


# The application the tests and the entry point reach for. Built at import, which in a
# `sorta ...` process is the same moment as `main()` — the argv it peeks at is the one
# the user typed.
app = build_app(_startup_lang()) if _TYPER_AVAILABLE else None


def main():
    _configure_runtime()
    if app is not None:
        app()
    else:  # no typer: the same commands and the same help, through argparse
        _argparse_main(_startup_lang())


if __name__ == "__main__":
    main()
