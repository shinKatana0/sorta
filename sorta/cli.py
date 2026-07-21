"""CLI. Typer if available; otherwise a minimal argparse fallback (for CI/sandboxes)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .config import configure_logging, load_config
from .db import connect, reset_index
from .dedup import assign_duplicates, compute_phashes, near_duplicate_groups
from .events import add_manual_event, build_events, rename_event
from .faces import detect_and_cluster, export_contact_sheet, label_cluster
from .faces import merge as merge_clusters
from .geo import resolve_places
from .indexer import index as run_index
from .junk import classify as classify_junk
from .landmarks import Classifier, clip_classifier, detect_landmarks
from .naming import name_events, naming_settings
from .progress import progress_task
from .sorter import plan_album, plan_and_sort
from .sorter import undo as undo_batch


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


# --- Stage summaries (a single format for the standalone commands and the `run` pipeline) ----
# Each helper returns a ready summary string for a step (multi-line where needed).
# Used BOTH by the same-named command `_cmd_<step>` AND by the `_pipeline_steps`
# step, so the output does not diverge (backlog #9 / F20).

def _summarize_index(stats, dups: int) -> str:
    return (f"Готово: +{stats.added} новых, ~{stats.updated} обновлено, "
            f"{stats.skipped} пропущено, {stats.errors} ошибок, {dups} дубликатов помечено")


def _summarize_geo(stats) -> str:
    return (f"Готово: {stats.total} файлов — exact_gps {stats.exact_gps}, "
            f"session_inferred {stats.session_inferred}, unknown {stats.unknown}")


def _summarize_landmarks(stats) -> str:
    lines = [f"Места без GPS: просмотрено {stats.scanned}, определено {stats.matched}"]
    for name, n in sorted(stats.by_landmark.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name}: {n}")
    return "\n".join(lines)


def _summarize_faces(face_stats, cl_stats) -> str:
    lines = [
        f"Детекция: {face_stats.files_processed} файлов, {face_stats.faces_found} лиц, "
        f"{face_stats.no_face_files} без лиц, {face_stats.errors} ошибок",
        f"Кластеры: {cl_stats.clusters} (лиц в кластерах: "
        f"{cl_stats.faces - cl_stats.noise}, шум: {cl_stats.noise}, "
        f"имён сохранено: {cl_stats.labels_kept})",
    ]
    if cl_stats.malformed:
        lines.append(f"⚠ повреждённых эмбеддингов пропущено: {cl_stats.malformed}")
    return "\n".join(lines)


def _summarize_events(stats) -> str:
    return (f"События: {stats.auto_events} авто ({stats.auto_files} файлов, "
            f"имён сохранено: {stats.names_preserved}), "
            f"{stats.manual_events} ручных ({stats.manual_files} файлов)")


def _summarize_junk(stats) -> str:
    kinds = ", ".join(f"{v}: {n}" for v, n in sorted(stats.by_verdict.items()))
    line = f"Классификация: {stats.processed}/{stats.total} обработано ({kinds})"
    if getattr(stats, "vlm_candidates", 0):
        line += f"; VLM: {stats.vlm_applied}/{stats.vlm_candidates} кандидатов переклассифицировано"
    return line


def _cmd_index(config_path: str, src: str | None = None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    if src:  # a positional source overrides config sources for this run
        cfg.sources = [Path(src).resolve()]
    if not cfg.sources:
        raise ValueError(
            "не задан источник: укажите каталог — sorta index <src_dir> — "
            "или заполните секцию 'sources' в config.yaml")
    conn = connect(cfg.database)
    with progress_task("index: сканирование") as cb:
        stats = run_index(cfg, conn, progress=lambda s: cb(s.scanned, None))
        dups = assign_duplicates(conn, cfg.dedup.canonical_strategy)
    print(_summarize_index(stats, dups))


def _cmd_geo(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("geo: места") as cb:
        stats = resolve_places(cfg, conn, progress=cb)
    print(_summarize_geo(stats))


def _cmd_faces(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("faces: детекция") as cb:
        face_stats, cl_stats = detect_and_cluster(cfg, conn, progress=cb)
    print(_summarize_faces(face_stats, cl_stats))


def _cmd_landmarks(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("landmarks: места без GPS") as cb:
        stats = detect_landmarks(cfg, conn, progress=cb)
    print(_summarize_landmarks(stats))


def _cmd_phash(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("phash: почти-дубликаты") as cb:
        n = compute_phashes(cfg, conn, progress=cb)
    print(f"pHash посчитан для {n} фото. Отчёт: sorta dupes --near")


def _cmd_junk(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("junk: классификация") as cb:
        stats = classify_junk(cfg, conn, progress=cb)
    print(_summarize_junk(stats))


def _cmd_events(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    with progress_task("events: кластеризация") as cb:
        stats = build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)  # naming by the provider (template by default)
    print(_summarize_events(stats))


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
        return _summarize_index(stats, dups)

    def _geo(cfg, conn, cb) -> str:
        return _summarize_geo(resolve_places(cfg, conn, progress=cb))

    def _landmarks(cfg, conn, cb) -> str:
        return _summarize_landmarks(
            detect_landmarks(cfg, conn, classifier=_clip(cfg), progress=cb))

    def _faces(cfg, conn, cb) -> str:
        face_stats, cl_stats = detect_and_cluster(cfg, conn, progress=cb)
        return _summarize_faces(face_stats, cl_stats)

    def _events(cfg, conn, cb) -> str:
        stats = build_events(cfg, conn, progress=cb)
        name_events(cfg, conn)
        return _summarize_events(stats)

    def _junk(cfg, conn, cb) -> str:
        return _summarize_junk(
            classify_junk(cfg, conn, classifier=_clip(cfg), progress=cb))

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
    if src:  # an explicit source overrides config sources for this run
        cfg.sources = [Path(src).resolve()]
    if not cfg.sources:
        raise SystemExit(
            "не задан источник: укажите --src <каталог> или заполните "
            "'sources' в config.yaml")
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
            print(f"[этап {i}/{len(steps)}] {name}")
            with progress_task(name) as cb:
                summary = fn(cfg, conn, cb)  # type: ignore[operator]
            for line in str(summary).splitlines():
                print(f"  {line}")
        if by:
            plan_dest = Path(dest) if dest else None  # None -> in-place (source root)
            print(f"[план] dry-run sort --by {by} -> {dest or 'in-place'}")
            with progress_task(f"plan {by}") as cb:
                plan_and_sort(cfg, conn, by, plan_dest, apply=False, progress=cb)
    finally:
        conn.close()
    print("\nАнализ завершён. Индекс наполнен; просмотрите план и запустите sort при необходимости.")


def _cmd_stats(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    total = q("SELECT COUNT(*) FROM files WHERE error IS NULL")
    if not total:
        print("Индекс пуст — запустите: sorta index")
        return
    print(f"Файлов в индексе: {total} (+{q('SELECT COUNT(*) FROM files WHERE error IS NOT NULL')} с ошибками)")
    print(f"  с GPS:            {q('SELECT COUNT(*) FROM files WHERE gps_lat IS NOT NULL')} "
          f"({q('SELECT COUNT(*) FROM files WHERE gps_lat IS NOT NULL') * 100 // total}%)")
    for src in ("exif", "filename", "mtime"):
        n = conn.execute("SELECT COUNT(*) FROM files WHERE taken_at_source = ?", (src,)).fetchone()[0]
        print(f"  дата из {src:9}: {n} ({n * 100 // total}%)")
    print(f"  дубликатов:       {q('SELECT COUNT(*) FROM files WHERE dup_of IS NOT NULL')}")
    places_total = q("SELECT COUNT(*) FROM places")
    if places_total:
        print(f"Гео (places): {places_total}")
        for conf, n in conn.execute(
            "SELECT confidence, COUNT(*) FROM places GROUP BY confidence ORDER BY 2 DESC"
        ):
            print(f"  {conf:16}: {n} ({n * 100 // places_total}%)")
    n_faces = q("SELECT COUNT(*) FROM faces WHERE bbox != '[]'")
    if n_faces:
        n_clusters = q("SELECT COUNT(*) FROM face_clusters WHERE merged_into IS NULL")
        n_named = q("SELECT COUNT(*) FROM face_clusters "
                    "WHERE merged_into IS NULL AND label IS NOT NULL")
        print(f"Лица: {n_faces} (кластеров: {n_clusters}, именованных: {n_named})")


def _cmd_dupes(config_path: str, near: bool = False) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    conn = connect(cfg.database)
    if near:
        have_phash = conn.execute(
            "SELECT COUNT(*) FROM files WHERE phash IS NOT NULL").fetchone()[0]
        if not have_phash:
            print("pHash ещё не посчитан — запустите: sorta phash")
            return
        groups = near_duplicate_groups(conn, cfg.index.phash_max_distance)
        if not groups:
            print("Почти-дубликатов не найдено")
            return
        for group in groups:
            print(f"Группа из {len(group)} похожих:")
            for r in group:
                print(f"  {r['path']}  ({r['size']} байт)")
        print(f"\nГрупп: {len(groups)} (порог Хэмминга: {cfg.index.phash_max_distance})")
        return
    rows = conn.execute(
        """SELECT c.path AS canon, f.path AS dup FROM files f
           JOIN files c ON f.dup_of = c.id ORDER BY c.path"""
    ).fetchall()
    if not rows:
        print("Точных дубликатов не найдено")
        return
    for r in rows:
        print(f"{r['dup']}\n  -> дубликат {r['canon']}")
    print(f"\nВсего: {len(rows)}")


def _stub(name: str, doc: str):
    def cmd(*_a, **_k):
        print(f"'{name}' будет реализована в следующей фазе: {doc}")
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
    ):
        """Сканировать источники, извлечь метаданные, пометить дубликаты."""
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
    def ui(port: int = typer.Option(8756, "--port", help="Порт локального сервера (127.0.0.1)"),
           config: str = _CFG):
        """Локальный веб-интерфейс: живой отчёт плана (пока режим city). Ctrl+C — стоп."""
        from .ui import serve as ui_serve
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        conn = connect(cfg.database)
        ui_serve(cfg, conn, port=port, config_path=config)

    faces_app = typer.Typer(help="Лица: детекция, кластеры, именование.")
    app.add_typer(faces_app, name="faces")

    @faces_app.callback(invoke_without_command=True)
    def faces_main(ctx: typer.Context, config: str = _CFG):
        """Без подкоманды: найти лица в новых фото и пересчитать кластеры."""
        if ctx.invoked_subcommand is None:
            _cmd_faces(config)

    @faces_app.command("label")
    def faces_label(cluster_id: int, name: str, config: str = _CFG):
        """Назвать кластер: sorta faces label 3 "Мама"."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        root = label_cluster(connect(cfg.database), cluster_id, name)
        print(f"Кластер {root} назван: {name}")

    @faces_app.command("merge")
    def faces_merge(src_id: int, dst_id: int, config: str = _CFG):
        """Слить кластер src в dst (это один человек)."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        root = merge_clusters(connect(cfg.database), src_id, dst_id)
        print(f"Слито: {src_id} -> {root}")

    @faces_app.command("sheet")
    def faces_sheet(cluster_id: int, out_html: Path, config: str = _CFG):
        """Экспорт контактного листа кластера в HTML."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        n = export_contact_sheet(connect(cfg.database), cluster_id, out_html)
        print(f"Готово: {n} лиц -> {out_html}")

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
        print(f"Событие {event_id}: {name}")

    @events_app.command("add")
    def events_add(name: str, date_from: str, date_to: str, config: str = _CFG):
        """Ручное событие на диапазон дат: events add "Конференция" 2024-01-01 2024-01-10."""
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        eid = add_manual_event(connect(cfg.database), name, date_from, date_to)
        print(f"Ручное событие {eid}: {name} ({date_from}..{date_to})")

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
        conn = connect(cfg.database)
        with progress_task(f"sort --by {by}") as cb:
            report = plan_and_sort(cfg, conn, by, dest, apply=apply, copy=copy,
                                   where=where or [],
                                   thumbnails=thumbnails, dedupe=dedupe,
                                   delete_worse_dupes=delete_worse_dupes,
                                   exclude=exclude or [], progress=cb)
        if apply:
            verb = "Скопировано" if copy else "Перемещено"
            extra = f", удалено дублей {report.deleted}" if report.deleted else ""
            print(f"{verb} {report.moved}, на месте {report.skipped_in_place}, "
                  f"ошибок {report.failed}{extra}. Откат: sorta undo")

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
            raise typer.BadParameter("--copy и --move взаимоисключающи")
        mode = "move" if move else "copy" if copy else "link"
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        conn = connect(cfg.database)
        with progress_task(f"album {kind} {selector}"):
            report = plan_album(cfg, conn, kind, selector, dest, mode=mode,
                                where=where or [], apply=apply, album_name=name)
        if apply:
            extra = f", заблокировано (мульти) {report.blocked_multi}" if report.blocked_multi else ""
            print(f"Альбом «{report.album_name}»: выгружено {report.transferred}, "
                  f"ошибок {report.failed}{extra}. Откат: sorta undo")

    @app.command()
    def reset(
        yes: bool = typer.Option(False, "--yes", "-y", help="Без подтверждения"),
        config: str = _CFG,
    ):
        """Стереть индекс (БД) и начать с нуля. Фото и разложенные папки НЕ трогает.

        Внимание: пропадут имена людей/событий и решения по дублям.
        """
        cfg = load_config(config)
        configure_logging(cfg.log_level)
        if not yes:
            typer.confirm(
                "Стереть весь индекс? Имена людей/событий и решения по дублям "
                "пропадут; фото и уже разложенные папки НЕ тронутся",
                abort=True)
        conn = connect(cfg.database)
        try:
            reset_index(conn)
        finally:
            conn.close()
        print("Индекс стёрт. Запустите `sorta index`/`sorta run` заново.")

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
        print(f"Откат батча {stats.batch_id}: возвращено {stats.undone}, "
              f"отсутствовало {stats.missing}, ошибок {stats.failed}")

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
            raise typer.BadParameter("--geo должен быть offline или online")
        _cmd_run(config, by=by, dest=str(dest) if dest else None, deep=deep, geo=geo,
                  faces=faces, events=events, src=src)

    def main():
        _ensure_utf8_console()
        app()

except ImportError:  # pragma: no cover — fallback without typer
    def main():
        _ensure_utf8_console()
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
