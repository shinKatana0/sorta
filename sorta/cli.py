"""CLI. Typer if available; otherwise a minimal argparse fallback (for CI/sandboxes)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, imaging
from .config import configure_logging, load_config
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
from .runlog import default_log_path, log_environment, stage_timer
from .sorter import plan_album, plan_and_sort
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
# `--help` texts are deliberately NOT localized — see the note above `_CLI_STRINGS`
# in i18n.py: they are evaluated at import time, before any config is read.

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
    with stage_timer("refresh-exif"), progress_task(
            _t("cli.progress.refresh_exif", lang)) as cb:
        stats = refresh_exif(cfg, conn, progress=cb)
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


def _cmd_junk(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    lang = _lang(cfg)
    conn = connect(cfg.database)
    with progress_task(_t("cli.progress.junk", lang),
                       phase_labels=_junk_phase_labels(lang)) as cb:
        stats = classify_junk(cfg, conn, progress=cb)
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

    def _junk(cfg, conn, cb) -> str:
        return _summarize_junk(
            classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb), _lang(cfg))

    return [
        ("index", _index),
        ("geo", _geo),
        ("landmarks", _landmarks),
        ("faces", _faces),
        ("events", _events),
        ("junk", _junk),
    ]


def _cmd_run(config_path: str, by: str | None = None, dest: str | None = None,
             deep: bool | None = None, geo: str | None = None,
             faces: bool = False, events: bool = False,
             src: str | None = None) -> None:
    """`deep`/`geo` (F50/#34) — an opt-in override for THIS run, not written to
    config.yaml: `deep` -> `naming.vlm_enabled`, `geo` ("offline"|"online") ->
    `geo.provider`. None (flag not passed) -> the value stays from config.

    `src` (F59) — the source directory for this run, overrides config.sources (like
    the positional src of `index`).

    `faces`/`events` (F53/#39) — opt-in steps, default off: the basic run builds only
    `index/geo/landmarks/junk`, the heaviest/longest steps are skipped. Independent
    of each other and of `deep`/`geo`."""
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
            with stage_timer(name), progress_task(
                    name,
                    phase_labels={**_cluster_phase_labels(lang),
                                  **_junk_phase_labels(lang)}) as cb:
                summary = fn(cfg, conn, cb)  # type: ignore[operator]
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


# --- Typer interface (primary) --------------------------------------------
try:
    import typer

    app = typer.Typer(help=f"Sorta v{__version__} — сортировка фотоколлекции")
    _CFG = typer.Option("config.yaml", "--config", "-c", help="Путь к config.yaml")

    @app.command()
    def index(
        src: str = typer.Argument(
            None, help="Каталог с фото (рекурсивно); переопределяет config sources"),
        config: str = _CFG,
        refresh_exif: bool = typer.Option(
            False, "--refresh-exif",
            help="Перечитать метаданные уже проиндексированных файлов "
                 "(вместо сканирования). Содержимое файлов не читается."),
        exclude_dir: list[str] = typer.Option(
            None, "--exclude-dir",
            help="Не сканировать эту папку источника (путь относительно корня). "
                 "Можно повторять. Сохраняется в файл исключений."),
    ):
        """Сканировать источники, извлечь метаданные, пометить дубликаты."""
        if refresh_exif:
            _cmd_refresh_exif(config)
            return
        if exclude_dir:
            _cmd_add_excludes(config, src, exclude_dir)
        _cmd_index(config, src=src)

    @app.command()
    def stats(config: str = _CFG):
        """Покрытие индекса: GPS, источники дат, дубликаты."""
        _cmd_stats(config)

    @app.command()
    def dupes(
        near: bool = typer.Option(False, "--near", help="Показать почти-дубликаты (pHash)"),
        config: str = _CFG,
    ):
        """Список точных дубликатов; с --near — группы почти-дубликатов."""
        _cmd_dupes(config, near=near)

    @app.command()
    def geo(config: str = _CFG):
        """Определить место каждого файла: GPS + наследование по сессиям."""
        _cmd_geo(config)

    @app.command()
    def landmarks(config: str = _CFG):
        """Места без GPS по известным достопримечательностям (CLIP). Запускать после geo."""
        _cmd_landmarks(config)

    @app.command()
    def phash(config: str = _CFG):
        """Посчитать pHash для почти-дубликатов (для `dupes --near`)."""
        _cmd_phash(config)

    @app.command()
    def junk(config: str = _CFG):
        """Классифицировать фото/мусор (screenshot|meme|document) для сортировки."""
        _cmd_junk(config)

    @app.command()
    def doctor(config: str = _CFG):
        """Диагностика окружения: torch/onnxruntime, GPU, гео-база, лог-файл.

        F112: `--config` is here only to know the output language — the command still
        works without a readable config (`_lang_of` falls back to the default), it just
        prints in the default language then. The two health summaries below come from
        diagnostics.py, which this feature does not own, so they stay as that module
        writes them.
        """
        lang = _lang_of(config)
        print(gpu_health().summary)
        # F65: the geo base failing to load is invisible at runtime (every coordinate
        # just resolves to an empty place), so the doctor has to state it outright.
        geo = geo_data_health()
        print(("" if geo.available else "⚠ ") + geo.summary)
        print(_t("cli.doctor.log", lang, path=default_log_path()))
        print(_t("cli.cache.preview_dir", lang, path=imaging.preview_dir())
              + ("" if imaging.preview_cache_enabled()
                 else _t("cli.cache.preview_disabled", lang)))

    @app.command("cache")
    def cache_cmd(
        clear: bool = typer.Option(
            False, "--clear", help="Удалить кэш превью (он пересоберётся сам)"),
        clear_geo: bool = typer.Option(
            False, "--clear-geo",
            help="Удалить кэш ответов онлайн-геокодера (F93): следующий `sorta geo` "
                 "при provider: online снова сходит в сеть"),
        config: str = _CFG,
    ):
        """Кэши: показать путь и размер, при --clear/--clear-geo — удалить.

        Кэш превью безопасно удалять в любой момент: он ленивый и пересоздаётся той
        стадией, которой первой понадобится кадр. Смысл команды — освободить место
        (порядка 150 КБ на фото) или заставить перегенерировать превью после смены
        настроек.

        Кэш геоданных (F93) — ответы онлайн-провайдера в таблице geo_cache. Он
        переживает и повторный прогон, и «Начать заново», поэтому --clear-geo —
        единственный способ переспросить провайдера, если он однажды ответил неверно.
        """
        cfg = load_config(config)  # applies the imaging: section onto the env
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
        files = sum(1 for _ in directory.rglob("*.jpg")) if directory.exists() else 0
        size = sum(f.stat().st_size for f in directory.rglob("*.jpg")) if files else 0
        print(_t("cli.cache.preview_dir", lang, path=directory))
        print(_t("cli.cache.preview_stats", lang, files=files, size_gb=size / 1e9))
        conn = connect(cfg.database)
        try:
            print(_t("cli.cache.geo_size", lang, n=geo_cache_size(conn)))
        finally:
            conn.close()

    @app.command()
    def ui(port: int = typer.Option(8756, "--port", help="Порт локального сервера (127.0.0.1)"),
           config: str = _CFG):
        """Локальный веб-интерфейс: живой отчёт плана (пока режим city). Ctrl+C — стоп."""
        from .ui import serve as ui_serve
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        warn_if_gpu_mismatch()  # F63: loud if torch is CPU-only while a GPU is expected
        conn = connect(cfg.database)
        ui_serve(cfg, conn, port=port, config_path=config)

    faces_app = typer.Typer(help="Лица: детекция, кластеры, именование.")
    app.add_typer(faces_app, name="faces")

    @faces_app.callback(invoke_without_command=True)
    def faces_main(
        ctx: typer.Context,
        rescan: bool = typer.Option(
            False, "--rescan",
            help="Пересчитать лица заново: стереть строки faces и продетектировать "
                 "все канонические фото (имена кластеров переносятся по файлам). "
                 "Нужен после смены детектора; без флага шаг инкрементальный"),
        limit: int = typer.Option(
            None, "--limit",
            help="Только с --rescan: пересчитать N случайных файлов, остальные не "
                 "трогать (замер шага на живом пайплайне)"),
        config: str = _CFG,
    ):
        """Без подкоманды: найти лица в новых фото и пересчитать кластеры."""
        if ctx.invoked_subcommand is not None:
            return
        if limit is not None and not rescan:
            raise typer.BadParameter(_t("cli.faces.limit_needs_rescan", _lang_of(config)))
        if limit is not None and limit <= 0:
            raise typer.BadParameter(_t("cli.faces.limit_positive", _lang_of(config)))
        _cmd_faces(config, rescan=rescan, limit=limit)

    @faces_app.command("label")
    def faces_label(cluster_id: int, name: str, config: str = _CFG):
        """Назвать кластер: sorta faces label 3 "Мама"."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        root = label_cluster(connect(cfg.database), cluster_id, name)
        print(_t("cli.faces.labeled", _lang(cfg), cluster=root, name=name))

    @faces_app.command("merge")
    def faces_merge(src_id: int, dst_id: int, config: str = _CFG):
        """Слить кластер src в dst (это один человек)."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        root = merge_clusters(connect(cfg.database), src_id, dst_id)
        print(_t("cli.faces.merged", _lang(cfg), src=src_id, dst=root))

    @faces_app.command("sheet")
    def faces_sheet(cluster_id: int, out_html: Path, config: str = _CFG):
        """Экспорт контактного листа кластера в HTML."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        n = export_contact_sheet(connect(cfg.database), cluster_id, out_html)
        print(_t("cli.faces.sheet_done", _lang(cfg), n=n, path=out_html))

    events_app = typer.Typer(help="События: автокластеризация, имена, ручные события.")
    app.add_typer(events_app, name="events")

    @events_app.callback(invoke_without_command=True)
    def events_main(ctx: typer.Context, config: str = _CFG):
        """Без подкоманды: пересчитать события (время × место)."""
        if ctx.invoked_subcommand is None:
            _cmd_events(config)

    @events_app.command("rename")
    def events_rename(event_id: int, name: str, config: str = _CFG):
        """Переименовать событие (имя переживает пересчёт)."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        rename_event(connect(cfg.database), event_id, name)
        print(_t("cli.events.renamed", _lang(cfg), event_id=event_id, name=name))

    @events_app.command("add")
    def events_add(name: str, date_from: str, date_to: str, config: str = _CFG):
        """Ручное событие на диапазон дат: events add "Конференция" 2024-01-01 2024-01-10."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        eid = add_manual_event(connect(cfg.database), name, date_from, date_to)
        print(_t("cli.events.added", _lang(cfg), event_id=eid, name=name,
                 date_from=date_from, date_to=date_to))

    @app.command()
    def sort(
        by: str = typer.Option(..., help="city | person | event"),
        dest: Path = typer.Option(
            None, "--dest", help="Каталог назначения; без него — in-place раскладка в корень источника (единственный sources)"),
        apply: bool = typer.Option(False, "--apply", help="Реально переместить (иначе dry-run)"),
        copy: bool = typer.Option(
            False, "--copy", help="Копировать в новую структуру, оригиналы на месте (C16; иначе перемещение)"),
        where: list[str] = typer.Option(
            None, "--where", help='Фильтр, повторяемый: "country=DE", "year>=2020"'),
        thumbnails: bool = typer.Option(
            False, "--thumbnails", help="Миниатюры в HTML-отчёте (медленно: декод всех фото)"),
        dedupe: bool = typer.Option(
            False, "--dedupe", help="Почти-дубли: лучший — по режиму, худшие — в _Duplicates (нужен sorta phash)"),
        delete_worse_dupes: bool = typer.Option(
            False, "--delete-worse-dupes", help="С --dedupe: БЕЗВОЗВРАТНО удалять худшие (не откатывается)"),
        exclude: list[str] = typer.Option(
            None, "--exclude", help="Не сортировать файлы из этого каталога (повторяемый); объединяется с sort.exclude_dirs"),
        config: str = _CFG,
    ):
        """Разложить файлы перемещением. По умолчанию — dry-run с планом (CSV+HTML)."""
        cfg = load_config(config)
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

    @app.command()
    def album(
        kind: str = typer.Argument(..., help="person | event"),
        selector: str = typer.Argument(..., help="имя человека / имя или id события"),
        dest: Path = typer.Option(..., "--dest", help="Куда выгрузить альбом"),
        copy: bool = typer.Option(False, "--copy", help="Копировать (иначе hardlink)"),
        move: bool = typer.Option(
            False, "--move", help="Изъять из пула (перемещение); иначе hardlink"),
        where: list[str] = typer.Option(
            None, "--where", help='Доп. фильтр среза: "city=Барселона", "year>=2020"'),
        name: str = typer.Option(None, "--name", help="Имя папки альбома (иначе имя человека/события)"),
        apply: bool = typer.Option(False, "--apply", help="Реально выгрузить (иначе dry-run)"),
        config: str = _CFG,
    ):
        """Выгрузить срез (человека/события) в отдельную папку. По умолчанию — hardlink, dry-run."""
        if copy and move:
            raise typer.BadParameter(
                _t("cli.album.copy_move_exclusive", _lang_of(config)))
        mode = "move" if move else "copy" if copy else "link"
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        lang = _lang(cfg)
        conn = connect(cfg.database)
        with progress_task(f"album {kind} {selector}"):
            report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                where=where or [], apply=apply, album_name=name)
        if apply:
            extra = (_t("cli.album.blocked_multi", lang, n=report.blocked_multi)
                     if report.blocked_multi else "")
            print(_t("cli.album.done", lang, name=report.album_name,
                     transferred=report.transferred, failed=report.failed, extra=extra))

    @app.command()
    def reset(
        yes: bool = typer.Option(False, "--yes", "-y", help="Без подтверждения"),
        clear_geo: bool = typer.Option(
            False, "--clear-geo",
            help="Заодно очистить кэш ответов онлайн-геокодера (F93); без флага он "
                 "переживает сброс, и повторный прогон geo не стоит сети"),
        config: str = _CFG,
    ):
        """Стереть индекс (БД) и начать с нуля. Фото и разложенные папки НЕ трогает.

        Внимание: пропадут имена людей/событий и решения по дублям. Кэш геоданных
        (F93) остаётся — названия точек на карте не зависят от того, какие файлы лежат
        у пользователя; стереть и его — `--clear-geo`.
        """
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        lang = _lang(cfg)
        if not yes:
            typer.confirm(
                _t("cli.reset.confirm", lang,
                   extra=_t("cli.reset.confirm_geo", lang) if clear_geo else ""),
                abort=True)
        conn = connect(cfg.database)
        try:
            reset_index(conn, clear_geo=clear_geo)
        finally:
            conn.close()
        print(_t("cli.reset.done", lang,
                 extra=_t("cli.reset.done_geo", lang) if clear_geo else ""))

    @app.command()
    def undo(
        batch: int = typer.Option(None, "--batch", help="ID батча (по умолчанию последний)"),
        config: str = _CFG,
    ):
        """Откатить перемещения последнего (или указанного) запуска sort по журналу."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        with progress_task("undo") as cb:
            stats = undo_batch(connect(cfg.database), batch, progress=cb)
        print(_t("cli.undo.done", _lang(cfg), batch=stats.batch_id, undone=stats.undone,
                 missing=stats.missing, failed=stats.failed))

    @app.command()
    def run(
        by: str = typer.Option(None, "--by", help="city|person|event — построить dry-run план в конце"),
        dest: Path = typer.Option(
            None, "--dest", help="Каталог назначения для плана с --by; без него — in-place"),
        deep: bool = typer.Option(
            None, "--deep/--no-deep",
            help="Глубокий анализ VLM на этот прогон: медленнее, нужен "
                 "`uv sync --extra vlm` (иначе откат на быстрый ярус); "
                 "без флага — как в config.yaml (naming.vlm_enabled)"),
        geo: str = typer.Option(
            None, "--geo",
            help="offline|online — online точнее для мест за границей, но "
                 "отправляет GPS-координаты фото серверу геокодирования "
                 "(Nominatim), сами фото не отправляются; без флага — как в "
                 "config.yaml (geo.provider)"),
        faces: bool = typer.Option(
            False, "--faces/--no-faces",
            help="Разбор по лицам (детекция + кластеризация) — самый долгий "
                 "шаг; по умолчанию выключен, доступен отдельно как `sorta "
                 "faces`"),
        events: bool = typer.Option(
            False, "--events/--no-events",
            help="Группировка в события по времени/месту; по умолчанию "
                 "выключена, доступна отдельно как `sorta events`"),
        src: str = typer.Option(
            None, "--src",
            help="Каталог-источник для этого прогона; переопределяет "
                 "config sources (как позиционный аргумент у `index`)"),
        config: str = _CFG,
    ):
        """Анализ одним прогоном: index -> geo -> landmarks -> junk (+faces/+events с флагами).

        Ничего не перемещает. С --by в конце строит dry-run план (в --dest либо
        in-place в корень источника, если --dest не задан).
        """
        if geo is not None and geo not in ("offline", "online"):
            raise typer.BadParameter(_t("cli.run.geo_choice", _lang_of(config)))
        _cmd_run(config, by=by, dest=str(dest) if dest else None, deep=deep, geo=geo,
                  faces=faces, events=events, src=src)

    def main():
        _configure_runtime()
        app()

except ImportError:  # pragma: no cover — fallback without typer
    def main():
        _configure_runtime()
        import argparse
        p = argparse.ArgumentParser(prog="sorta")
        p.add_argument("command", choices=["index", "stats", "dupes", "geo", "phash",
                                            "landmarks", "junk", "faces", "events", "run"])
        p.add_argument("-c", "--config", default="config.yaml")
        p.add_argument("--near", action="store_true")
        a = p.parse_args()
        if a.command == "dupes":
            _cmd_dupes(a.config, near=a.near)
        else:
            {"index": _cmd_index, "stats": _cmd_stats, "geo": _cmd_geo, "phash": _cmd_phash,
             "landmarks": _cmd_landmarks, "junk": _cmd_junk, "faces": _cmd_faces,
             "events": _cmd_events, "run": _cmd_run}[a.command](a.config)


if __name__ == "__main__":
    main()
