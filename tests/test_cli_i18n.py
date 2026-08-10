"""F112: the command line speaks the config language (ru|en|ja), not always Russian.

The `ru` expectations below are GOLDEN: they were captured from the command output
BEFORE the strings moved into `i18n._CLI_STRINGS`, so any drift in the Russian wording
fails here instead of in a user's terminal. That regression is the whole point of the
feature — `language: ru` has to keep printing exactly what it printed yesterday.
"""
from __future__ import annotations

import contextlib
import io
import re
import string
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sorta import cli, i18n, install
from sorta.db import connect

_LANGS = ("ru", "en", "ja")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# F216: the tier block of `doctor`, fixed so that the golden output is about the words
# and not about which models happen to be cached on the machine running the suite. One
# tier of each state: in place, packages in place with the weights still to come, and
# not installed at all.
DOCTOR_TIERS = [
    cli.TierState("base"),
    cli.TierState("faces", missing_weights=("buffalo_l",)),
    cli.TierState("deep", missing_packages=("transformers",)),
]


def command_callback(app, name: str):
    """The function typer registered for `sorta <name>` (F114: the commands are built
    by `cli.build_app`, so they are reached through an application, not by module
    attribute)."""
    for info in app.registered_commands:
        if (info.name or info.callback.__name__) == name:
            return info.callback
    raise KeyError(name)


def group_callback(app, name: str):
    """The callback of a sub-application — `sorta faces` without a subcommand."""
    for info in app.registered_groups:
        if info.name == name:
            return info.typer_instance.registered_callback.callback
    raise KeyError(name)


def _seed(db_path: Path) -> None:
    """Three good files (one a duplicate, two with GPS) + one with an error, two
    places and one named face cluster — enough for every branch of `stats`/`dupes`."""
    conn = connect(db_path)
    conn.executemany(
        "INSERT INTO files (path, size, mtime, ext, media_type, taken_at_source, "
        "gps_lat, gps_lon, error, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,'2026-01-01')",
        [("/a.jpg", 100, 1.0, ".jpg", "photo", "exif", 50.0, 30.0, None),
         ("/b.jpg", 200, 1.0, ".jpg", "photo", "filename", None, None, None),
         ("/c.jpg", 300, 1.0, ".jpg", "photo", "mtime", 10.0, 20.0, None),
         ("/d.jpg", 400, 1.0, ".jpg", "photo", "exif", None, None, "boom")])
    conn.execute("UPDATE files SET dup_of = 1 WHERE path = '/c.jpg'")
    conn.execute("INSERT INTO places (file_id, confidence, updated_at) "
                 "VALUES (1, 'exact_gps', '2026-01-01')")
    conn.execute("INSERT INTO places (file_id, confidence, updated_at) "
                 "VALUES (2, 'unknown', '2026-01-01')")
    conn.execute("INSERT INTO face_clusters (id, merged_into, label) VALUES (1, NULL, 'X')")
    conn.execute("INSERT INTO faces (file_id, cluster_id, bbox, embedding) "
                 "VALUES (1, 1, '[1,2,3,4]', X'00')")
    conn.commit()
    conn.close()


class _Env:
    """A temp project (config + seeded DB) for one language, plus the whole set of
    strings the CLI can print. Fixed fake stats everywhere, so the same call renders
    the same numbers in every language and only the words may differ."""

    def __init__(self, tmp: Path, lang: str, *, data: str = "X") -> None:
        self.root = tmp
        self.lang = lang
        self.data = data  # value substituted where the CLI echoes user data back
        self.db = tmp / f"{lang}.db"
        self.cfg = tmp / f"config-{lang}.yaml"
        self.cfg.write_text(
            f'sources: ["{(tmp / "src").as_posix()}"]\n'
            f'database: "{self.db.as_posix()}"\n'
            f'language: {lang}\n', encoding="utf-8")
        self.no_source_cfg = tmp / f"nosrc-{lang}.yaml"
        self.no_source_cfg.write_text(
            f'database: "{self.db.as_posix()}"\nlanguage: {lang}\n', encoding="utf-8")
        self.empty_cfg = tmp / f"empty-{lang}.yaml"
        self.empty_cfg.write_text(
            f'database: "{(tmp / f"empty-{lang}.db").as_posix()}"\n'
            f'language: {lang}\n', encoding="utf-8")
        (tmp / "src").mkdir(exist_ok=True)
        self.previews = tmp / f"previews-{lang}"
        self.previews.mkdir()

    def _cap(self, fn) -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
        except SystemExit as exc:
            buf.write(f"{exc}")
        except ValueError as exc:
            buf.write(f"{exc}")
        return buf.getvalue()

    def summaries(self) -> dict[str, str]:
        lang = self.lang
        return {
            "index": cli._summarize_index(
                SimpleNamespace(added=2, updated=1, skipped=3, errors=1), 4, lang),
            "geo": cli._summarize_geo(SimpleNamespace(
                total=10, exact_gps=4, session_inferred=2, trip_inferred=1,
                path_inferred=2, unknown=1), lang),
            "landmarks": cli._summarize_landmarks(SimpleNamespace(
                scanned=5, matched=2, by_landmark={self.data: 2}), lang),
            "faces": cli._summarize_faces(
                SimpleNamespace(files_processed=8, faces_found=12, no_face_files=2,
                                errors=1),
                SimpleNamespace(clusters=3, faces=12, noise=2, labels_kept=1,
                                malformed=4), lang),
            "faces_clean": cli._summarize_faces(
                SimpleNamespace(files_processed=1, faces_found=1, no_face_files=0,
                                errors=0),
                SimpleNamespace(clusters=1, faces=1, noise=0, labels_kept=0,
                                malformed=0), lang),
            "events": cli._summarize_events(SimpleNamespace(
                auto_events=3, auto_files=20, names_preserved=1, manual_events=1,
                manual_files=5), lang),
            "junk": cli._summarize_junk(SimpleNamespace(
                total=100, processed=40, by_verdict={"photo": 30, "screenshot": 10}),
                lang),
            "junk_full": cli._summarize_junk(SimpleNamespace(
                total=100, processed=40, by_verdict={"photo": 30},
                skipped_incremental=7, vlm_candidates=9, vlm_applied=4), lang),
            "refresh": cli._summarize_refresh(SimpleNamespace(
                scanned=10, updated=4, recovered_gps=2, recovered_date=3,
                still_empty=1, errors=0), lang),
        }

    def commands(self) -> dict[str, str]:
        """Everything the commands print, with the heavy work faked out."""
        cfg, out = str(self.cfg), {}
        out["stats_empty"] = self._cap(lambda: cli._cmd_stats(str(self.empty_cfg)))
        _seed(self.db)
        out["stats"] = self._cap(lambda: cli._cmd_stats(cfg))
        out["dupes_exact"] = self._cap(lambda: cli._cmd_dupes(cfg))
        out["dupes_no_phash"] = self._cap(lambda: cli._cmd_dupes(cfg, near=True))
        out["dupes_exact_none"] = self._cap(
            lambda: cli._cmd_dupes(str(self.empty_cfg)))

        steps = [("alpha", lambda c, n, cb: "one"), ("beta", lambda c, n, cb: "two")]
        # F237: pinned, or the golden output would depend on how much memory the machine
        # running the suite happens to have free. The sentence itself has its own cases.
        with patch.object(cli, "_pipeline_steps", lambda: steps), \
                patch.object(cli, "memory_health",
                             lambda *_a, **_kw: SimpleNamespace(low=False)):
            out["run"] = self._cap(lambda: cli._cmd_run(cfg))
        out["run_no_source"] = self._cap(
            lambda: cli._cmd_run(str(self.no_source_cfg)))
        out["index_no_source"] = self._cap(
            lambda: cli._cmd_index(str(self.no_source_cfg)))
        out["excludes_no_source"] = self._cap(
            lambda: cli._cmd_add_excludes(str(self.no_source_cfg), None, ["x"]))
        out["excludes_saved"] = self._cap(
            lambda: cli._cmd_add_excludes(cfg, None, ["junk"]))

        with patch.object(cli, "compute_phashes", lambda c, n, progress=None: 7):
            out["phash"] = self._cap(lambda: cli._cmd_phash(cfg))
        refreshed = SimpleNamespace(scanned=10, updated=4, recovered_gps=2,
                                    recovered_date=3, still_empty=1, errors=0)
        with patch.object(cli, "refresh_exif", lambda c, n, progress=None: refreshed):
            out["refresh_exif"] = self._cap(lambda: cli._cmd_refresh_exif(cfg))

        with patch.object(cli, "label_cluster", lambda n, cid, name: 3):
            out["faces_label"] = self._cap(
                lambda: cli._cmd_faces_label(cfg, 3, self.data))
        with patch.object(cli, "merge_clusters", lambda n, a, b: 5):
            out["faces_merge"] = self._cap(lambda: cli._cmd_faces_merge(cfg, 7, 5))
        with patch.object(cli, "export_contact_sheet", lambda n, cid, html: 12):
            out["faces_sheet"] = self._cap(
                lambda: cli._cmd_faces_sheet(cfg, 3, self.root / "sheet.html"))
        with patch.object(cli, "rename_event", lambda n, eid, name: None):
            out["events_rename"] = self._cap(
                lambda: cli._cmd_events_rename(cfg, 2, self.data))
        with patch.object(cli, "add_manual_event", lambda n, name, a, b: 9):
            out["events_add"] = self._cap(
                lambda: cli._cmd_events_add(cfg, self.data, "2024-01-01", "2024-01-10"))

        report = SimpleNamespace(moved=10, skipped_in_place=2, failed=1, deleted=3)
        with patch.object(cli, "plan_and_sort", lambda *a, **k: report):
            out["sort_move"] = self._cap(lambda: self._sort(cfg, copy=False))
            out["sort_copy"] = self._cap(lambda: self._sort(cfg, copy=True))
        report = SimpleNamespace(moved=10, skipped_in_place=2, failed=1, deleted=0)
        with patch.object(cli, "plan_and_sort", lambda *a, **k: report):
            out["sort_no_deleted"] = self._cap(lambda: self._sort(cfg, copy=False))

        album = SimpleNamespace(album_name=self.data, transferred=8, failed=1,
                                blocked_multi=2)
        with patch.object(cli, "plan_album", lambda *a, **k: album):
            out["album"] = self._cap(lambda: self._album(cfg))
        album = SimpleNamespace(album_name=self.data, transferred=8, failed=1,
                                blocked_multi=0)
        with patch.object(cli, "plan_album", lambda *a, **k: album):
            out["album_no_blocked"] = self._cap(lambda: self._album(cfg))

        with patch.object(cli, "reset_index", lambda n, clear_geo=False: None):
            out["reset"] = self._cap(lambda: cli._cmd_reset(cfg, clear_geo=False))
            out["reset_geo"] = self._cap(lambda: cli._cmd_reset(cfg, clear_geo=True))
            out["reset_confirm"] = self._confirm_text(cfg, clear_geo=False)
            out["reset_confirm_geo"] = self._confirm_text(cfg, clear_geo=True)

        undone = SimpleNamespace(batch_id=4, undone=10, missing=1, failed=2,
                                 dirs_removed=3)
        with patch.object(cli, "undo_batch", lambda n, b, progress=None: undone):
            out["undo"] = self._cap(lambda: cli._cmd_undo(cfg, None))

        with patch.object(cli, "geo_cache_size", lambda n: 42), \
                patch.object(cli, "clear_geo_cache", lambda n: 17), \
                patch("sorta.imaging.preview_dir", lambda: self.previews), \
                patch("sorta.imaging.preview_cache_clear", lambda: None):
            out["cache_show"] = self._cap(
                lambda: cli._cmd_cache(cfg, clear=False, clear_geo=False))
            out["cache_clear_geo"] = self._cap(
                lambda: cli._cmd_cache(cfg, clear=False, clear_geo=True))
            out["cache_clear"] = self._cap(
                lambda: cli._cmd_cache(cfg, clear=True, clear_geo=False))
            out["doctor"] = self._doctor(cfg)

        out["stub"] = self._cap(cli._stub("step", "doc", self.lang))
        return out

    def _sort(self, cfg: str, *, copy: bool) -> None:
        cli._cmd_sort(cfg, "city", None, apply=True, copy=copy, where=[],
                      thumbnails=False, dedupe=False, delete_worse_dupes=False,
                      exclude=[])

    def _album(self, cfg: str) -> None:
        cli._cmd_album(cfg, "person", self.data, self.root / "album", copy=False,
                       move=False, where=[], name=None, apply=True)

    def _confirm_text(self, cfg: str, *, clear_geo: bool) -> str:
        seen: list[str] = []
        self._cap(lambda: cli._cmd_reset(cfg, clear_geo=clear_geo,
                                         confirm=seen.append))
        return seen[0]

    def _doctor(self, cfg: str) -> str:
        # F211: `doctor` opens by naming the interpreter and the `uv` it is running with,
        # so a shadowed PATH shows up at once rather than nine minutes later in a red
        # gate. Both are pinned here: this test is about the WORDS, and a real path would
        # turn it into a statement about whichever machine happened to run it.
        # F216: the same for the tier block below them — the machine running the suite
        # has whatever it has, so the three states are fixed here and it is the SENTENCES
        # that are under test.
        # F213: `sorta` and `exiftool` join them for the same reason and are pinned the
        # same way. The way out of a missing tier is pinned too — to the sentence an
        # INSTALLED copy gets (F230: the choice is by install kind now, and this case is
        # about the WORDS rather than about the machine the suite happens to run on). The cache mode
        # is fixed to a private one — the warning that follows an open cache has its own
        # cases in test_linux_install.
        health = SimpleNamespace(summary="health", available=True)
        found = {"uv": "uv.exe", "sorta": "sorta.exe", "exiftool": "exiftool.exe"}
        with patch.object(cli, "gpu_health", lambda **_kw: health), \
                patch.object(cli, "geo_data_health", lambda: health), \
                patch.object(cli, "tier_states", lambda: DOCTOR_TIERS), \
                patch.object(cli, "default_log_path", lambda: "run.log"), \
                patch.object(cli.sys, "executable", "python.exe"), \
                patch.object(cli.shutil, "which", found.get), \
                patch.object(cli, "_tier_hint_key",
                             lambda _kind=None: "cli.doctor.tier_hint.installed"), \
                patch.object(cli, "_directory_mode", lambda _path: 0o700), \
                patch("sorta.imaging.preview_cache_enabled", lambda: False):
            return self._cap(lambda: cli._cmd_doctor(cfg))


class _EnvCase(unittest.TestCase):
    lang = "en"
    data = "X"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.env = _Env(Path(self.tmp.name), self.lang, data=self.data)

    def tearDown(self):
        self.tmp.cleanup()


class TestRussianOutputIsUnchanged(_EnvCase):
    """Golden: `language: ru` prints byte for byte what it printed before F112."""

    lang = "ru"
    data = "Мама"

    def test_summaries(self):
        got = self.env.summaries()
        self.assertEqual(
            got["index"],
            "Готово: +2 новых, ~1 обновлено, 3 пропущено, 1 ошибок, "
            "4 дубликатов помечено")
        self.assertEqual(
            got["geo"],
            "Готово: 10 файлов — exact_gps 4, session_inferred 2, trip_inferred 1, "
            "path_inferred 2, unknown 1")
        self.assertEqual(
            got["landmarks"], "Места без GPS: просмотрено 5, определено 2\n  Мама: 2")
        self.assertEqual(
            got["faces"],
            "Детекция: 8 файлов, 12 лиц, 2 без лиц, 1 ошибок\n"
            "Кластеры: 3 (лиц в кластерах: 10, шум: 2, имён сохранено: 1)\n"
            "⚠ повреждённых эмбеддингов пропущено: 4")
        self.assertEqual(
            got["faces_clean"],
            "Детекция: 1 файлов, 1 лиц, 0 без лиц, 0 ошибок\n"
            "Кластеры: 1 (лиц в кластерах: 1, шум: 0, имён сохранено: 0)")
        self.assertEqual(
            got["events"],
            "События: 3 авто (20 файлов, имён сохранено: 1), 1 ручных (5 файлов)")
        self.assertEqual(
            got["junk"], "Классификация: 40/100 обработано (photo: 30, screenshot: 10)")
        self.assertEqual(
            got["junk_full"],
            "Классификация: 40/100 обработано (photo: 30); "
            "пропущено как уже обработанные: 7; "
            "VLM: 4/9 кандидатов переклассифицировано")
        self.assertEqual(
            got["refresh"],
            "Перечитано: 10 файлов, обновлено 4; вернулось координат: 2, "
            "дат съёмки: 3; без EXIF: 1, ошибок: 0")

    def test_commands(self):
        got = self.env.commands()
        self.assertEqual(got["stats_empty"], "Индекс пуст — запустите: sorta index\n")
        # The aligned block: the padding of every label is part of the ru template.
        self.assertEqual(
            got["stats"],
            "Файлов в индексе: 3 (+1 с ошибками)\n"
            "  с GPS:            2 (66%)\n"
            "  дата из exif     : 2 (66%)\n"
            "  дата из filename : 1 (33%)\n"
            "  дата из mtime    : 1 (33%)\n"
            "  дубликатов:       1\n"
            "Гео (places): 2\n"
            "  unknown         : 1 (50%)\n"
            "  exact_gps       : 1 (50%)\n"
            "Лица: 1 (кластеров: 1, именованных: 1)\n")
        self.assertEqual(got["dupes_exact"], "/c.jpg\n  -> дубликат /a.jpg\n\nВсего: 1\n")
        self.assertEqual(got["dupes_no_phash"],
                         "pHash ещё не посчитан — запустите: sorta phash\n")
        self.assertEqual(got["dupes_exact_none"], "Точных дубликатов не найдено\n")
        self.assertEqual(
            got["run"],
            "[этап 1/2] alpha\n  one\n[этап 2/2] beta\n  two\n"
            "\nАнализ завершён. Индекс наполнен; просмотрите план и запустите sort "
            "при необходимости.\n")
        self.assertEqual(
            got["run_no_source"],
            "не задан источник: укажите --src <каталог> или заполните "
            "'sources' в config.yaml")
        self.assertEqual(
            got["index_no_source"],
            "не задан источник: укажите каталог — sorta index <src_dir> — "
            "или заполните секцию 'sources' в config.yaml")
        self.assertEqual(
            got["excludes_no_source"],
            "--exclude-dir: не задан источник — укажите каталог позиционно "
            "или заполните 'sources' в config.yaml")
        self.assertEqual(
            got["excludes_saved"],
            f"Исключено из сканирования ({self.env.root / 'src'}): junk\n"
            f"Файл исключений: {self.env.root / 'excludes.yaml'}\n")
        self.assertEqual(got["phash"],
                         "pHash посчитан для 7 фото. Отчёт: sorta dupes --near\n")
        self.assertEqual(
            got["refresh_exif"],
            "Перечитано: 10 файлов, обновлено 4; вернулось координат: 2, "
            "дат съёмки: 3; без EXIF: 1, ошибок: 0\n"
            "Появились новые координаты — перезапустите: sorta geo (и sorta events)\n")
        self.assertEqual(got["faces_label"], "Кластер 3 назван: Мама\n")
        self.assertEqual(got["faces_merge"], "Слито: 7 -> 5\n")
        self.assertEqual(got["faces_sheet"],
                         f"Готово: 12 лиц -> {self.env.root / 'sheet.html'}\n")
        self.assertEqual(got["events_rename"], "Событие 2: Мама\n")
        self.assertEqual(got["events_add"],
                         "Ручное событие 9: Мама (2024-01-01..2024-01-10)\n")
        self.assertEqual(
            got["sort_move"],
            "Перемещено 10, на месте 2, ошибок 1, удалено дублей 3. Откат: sorta undo\n")
        self.assertEqual(
            got["sort_copy"],
            "Скопировано 10, на месте 2, ошибок 1, удалено дублей 3. Откат: sorta undo\n")
        self.assertEqual(got["sort_no_deleted"],
                         "Перемещено 10, на месте 2, ошибок 1. Откат: sorta undo\n")
        self.assertEqual(
            got["album"],
            "Альбом «Мама»: выгружено 8, ошибок 1, заблокировано (мульти) 2. "
            "Откат: sorta undo\n")
        self.assertEqual(got["album_no_blocked"],
                         "Альбом «Мама»: выгружено 8, ошибок 1. Откат: sorta undo\n")
        self.assertEqual(got["reset"],
                         "Индекс стёрт. Запустите `sorta index`/`sorta run` заново.\n")
        self.assertEqual(
            got["reset_geo"],
            "Индекс стёрт. Запустите `sorta index`/`sorta run` заново. "
            "Кэш геоданных очищен.\n")
        self.assertEqual(
            got["reset_confirm"],
            "Стереть весь индекс? Имена людей/событий и решения по дублям пропадут; "
            "фото и уже разложенные папки НЕ тронутся")
        self.assertEqual(
            got["reset_confirm_geo"],
            "Стереть весь индекс? Имена людей/событий и решения по дублям пропадут; "
            "фото и уже разложенные папки НЕ тронутся, "
            "кэш геоданных тоже будет очищен")
        self.assertEqual(
            got["undo"],
            "Откат батча 4: возвращено 10, отсутствовало 1, ошибок 2, "
            "убрано пустых каталогов 3\n")
        self.assertEqual(
            got["cache_show"],
            f"Кэш превью: {self.env.previews}\n"
            "  файлов: 0, размер: 0.00 ГБ\n"
            # F117: a size means little without the bound it is measured against, and
            # "not set" is a state rather than a limit of zero — the default prints the
            # config key that would set one.
            "  потолок: не задан (imaging.preview_cache_max_gb)\n"
            "Кэш геоданных (geo_cache): записей 42\n")
        self.assertEqual(got["cache_clear_geo"],
                         "Кэш геоданных очищен: удалено записей 17\n")
        self.assertEqual(got["cache_clear"],
                         f"Кэш превью удалён: {self.env.previews}\n")
        self.assertEqual(
            got["doctor"],
            f"Интерпретатор: python.exe\nМенеджер окружения uv: uv.exe\n"
            f"Команда sorta: sorta.exe\n"
            f"Чтение метаданных exiftool: exiftool.exe\n"
            f"Ярусы установки:\n"
            f"  Базовый ярус: на месте\n"
            f"  Лица: пакеты на месте, модели (buffalo_l, 400 МБ) скачаются при первом "
            f"запуске стадии\n"
            f"  Глубокий ярус (VLM): не установлен (нет: transformers)\n"
            f"  Доустановить ярус: sorta-setup (пункт «Sorta setup» в меню «Пуск»).\n"
            f"health\nhealth\nЛог прогона: run.log\n"
            f"Кэш превью: {self.env.previews} (ОТКЛЮЧЁН)\n")
        self.assertEqual(got["stub"], "'step' будет реализована в следующей фазе: doc\n2")

    def test_progress_and_phase_captions(self):
        self.assertEqual(cli._cluster_phase_labels("ru"), {
            "cluster_read": "кластеры: чтение эмбеддингов",
            "cluster_hdbscan": "кластеры: группировка лиц (без процента)",
            "cluster_inherit": "кластеры: перенос имён",
            "cluster_write": "кластеры: запись",
        })
        self.assertEqual(cli._junk_phase_labels("ru"), {
            "junk_clip": "junk: классификация CLIP",
            "junk_ocr": "junk: распознавание текста",
            "junk_vlm": "junk: глубокий анализ (VLM)",
            # F205: the other two passes that ask the model, each with a caption of its own
            "junk_pets_vlm": "junk: проверка животных (VLM)",
            "junk_rescue_vlm": "junk: поиск экранного (VLM)",
            "junk_write": "junk: запись вердиктов",
        })
        self.assertEqual(i18n.cli_text("cli.progress.index", "ru"), "index: сканирование")
        self.assertEqual(i18n.cli_text("cli.progress.geo", "ru"), "geo: места")
        self.assertEqual(i18n.cli_text("cli.progress.faces", "ru"), "faces: детекция")
        self.assertEqual(i18n.cli_text("cli.progress.faces_rescan", "ru"),
                         "faces: пересканирование")
        self.assertEqual(i18n.cli_text("cli.progress.landmarks", "ru"),
                         "landmarks: места без GPS")
        self.assertEqual(i18n.cli_text("cli.progress.phash", "ru"),
                         "phash: почти-дубликаты")
        self.assertEqual(i18n.cli_text("cli.progress.junk", "ru"), "junk: классификация")
        self.assertEqual(i18n.cli_text("cli.progress.events", "ru"),
                         "events: кластеризация")
        self.assertEqual(i18n.cli_text("cli.progress.refresh_exif", "ru"),
                         "refresh-exif: метаданные")


class TestEnglishHasNoCyrillic(_EnvCase):
    """`language: en` — the acceptance criterion of the feature: not one Russian
    letter left in the output. Every piece of user data fed in here is ASCII, so a
    Cyrillic character can only come from a string that stayed hard-coded."""

    lang = "en"

    def test_summaries(self):
        for name, text in self.env.summaries().items():
            self.assertIsNone(_CYRILLIC.search(text), f"{name}: {text!r}")

    def test_commands(self):
        for name, text in self.env.commands().items():
            self.assertIsNone(_CYRILLIC.search(text), f"{name}: {text!r}")

    def test_progress_and_phase_captions(self):
        for text in (*cli._cluster_phase_labels("en").values(),
                     *cli._junk_phase_labels("en").values()):
            self.assertIsNone(_CYRILLIC.search(text), text)

    def test_stats_reads_as_english(self):
        out = self.env.commands()["stats"]
        self.assertIn("Files in the index: 3 (+1 with errors)", out)
        self.assertIn("with GPS:", out)
        self.assertIn("date from exif", out)
        self.assertIn("duplicates:", out)


class TestJapaneseIsComplete(_EnvCase):
    """`language: ja` — same coverage as the UI has (ui._UI_STRINGS): nothing falls
    through to the English variant and no key leaks out raw."""

    lang = "ja"

    def _assert_localized(self, name: str, text: str) -> None:
        self.assertIsNone(_CYRILLIC.search(text), f"{name}: {text!r}")
        self.assertNotIn("cli.", text, f"{name} leaked a key: {text!r}")

    def test_summaries(self):
        for name, text in self.env.summaries().items():
            self._assert_localized(name, text)

    def test_commands(self):
        for name, text in self.env.commands().items():
            self._assert_localized(name, text)

    def test_japanese_differs_from_english(self):
        ja, en = self.env.summaries(), _Env(self.env.root, "en").summaries()
        for key in ja:
            self.assertNotEqual(ja[key], en[key], key)


class TestNumbersSubstituteInEveryLanguage(unittest.TestCase):
    """Counters are named format fields, so they survive the word order of each
    language — checked on the summaries that carry the most of them."""

    def test_counters_present_in_all_languages(self):
        for lang in _LANGS:
            index = cli._summarize_index(
                SimpleNamespace(added=11, updated=22, skipped=33, errors=44), 55, lang)
            for n in ("11", "22", "33", "44", "55"):
                self.assertIn(n, index, f"{lang}: {index!r}")
            faces = cli._summarize_faces(
                SimpleNamespace(files_processed=61, faces_found=72, no_face_files=83,
                                errors=94),
                SimpleNamespace(clusters=15, faces=72, noise=26, labels_kept=37,
                                malformed=48), lang)
            for n in ("61", "72", "83", "94", "15", "46", "26", "37", "48"):
                self.assertIn(n, faces, f"{lang}: {faces!r}")  # 46 = faces - noise
            junk = cli._summarize_junk(SimpleNamespace(
                total=900, processed=700, by_verdict={"photo": 500},
                skipped_incremental=13, vlm_candidates=17, vlm_applied=9), lang)
            for n in ("900", "700", "500", "13", "17", "9"):
                self.assertIn(n, junk, f"{lang}: {junk!r}")

    def test_float_field_formats_in_all_languages(self):
        for lang in _LANGS:
            self.assertIn("12.64", i18n.cli_text("cli.cache.preview_stats", lang,
                                                 files=3, size_gb=12.6431))
            # F117: same formatting contract for the ceiling line — two placeholders,
            # one of them rounded to whole percent.
            limit = i18n.cli_text("cli.cache.preview_limit", lang,
                                  limit_gb=40.0, percent=31.6)
            self.assertIn("40.00", limit)
            self.assertIn("32", limit)


class TestUnknownLanguageFallsBack(unittest.TestCase):
    """An unrecognised `language:` normalizes to the default instead of crashing —
    the config is hand-edited, and a typo must not take the CLI down."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _stats_with(self, language_line: str) -> str:
        cfg = self.root / "config.yaml"
        cfg.write_text(f'database: "{(self.root / "t.db").as_posix()}"\n'
                       f'{language_line}', encoding="utf-8")
        _seed(self.root / "t.db")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._cmd_stats(str(cfg))
        return buf.getvalue()

    def test_unknown_language_prints_the_default(self):
        out = self._stats_with("language: klingon\n")
        self.assertIn("Files in the index: 3", out)
        self.assertIsNone(_CYRILLIC.search(out), out)

    def test_missing_language_prints_the_default(self):
        self.assertIn("Files in the index: 3", self._stats_with(""))

    def test_lang_helper_normalizes(self):
        self.assertEqual(cli._lang(SimpleNamespace(language="RU")), "ru")
        self.assertEqual(cli._lang(SimpleNamespace(language="klingon")), "en")
        self.assertEqual(cli._lang(SimpleNamespace()), "en")

    def test_lang_of_survives_an_unreadable_config(self):
        # The pre-config checks (`--geo`, `--limit`, `--copy/--move`) must still be
        # able to say what is wrong when there is no config to read.
        self.assertEqual(cli._lang_of(str(self.root / "nope.yaml")), "en")
        broken = self.root / "broken.yaml"
        broken.write_text("language: [oops\n", encoding="utf-8")
        self.assertEqual(cli._lang_of(str(broken)), "en")

    def test_lang_of_reads_the_language(self):
        cfg = self.root / "ja.yaml"
        cfg.write_text("language: ja\n", encoding="utf-8")
        self.assertEqual(cli._lang_of(str(cfg)), "ja")


class TestBadParameterMessagesAreLocalized(unittest.TestCase):
    """The `typer.BadParameter` guards fire before the command loads the config, so
    they go through `_lang_of` — they must still speak the configured language.

    F114: the guards stayed in the typer shells (they answer with typer's own error),
    so each case reaches for the registered command of a built application.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.app = cli.build_app("en")  # the guards' language comes from --config

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, lang: str) -> str:
        path = self.root / f"{lang}.yaml"
        path.write_text(f'database: "{(self.root / "t.db").as_posix()}"\n'
                        f'language: {lang}\n', encoding="utf-8")
        return str(path)

    def test_geo_choice(self):
        import typer
        run = command_callback(self.app, "run")
        with self.assertRaises(typer.BadParameter) as ctx:
            run(by=None, dest=None, deep=None, geo="nope", faces=False,
                events=False, src=None, config=self._cfg("ru"))
        self.assertEqual(str(ctx.exception.message),
                         "--geo должен быть offline или online")
        with self.assertRaises(typer.BadParameter) as ctx:
            run(by=None, dest=None, deep=None, geo="nope", faces=False,
                events=False, src=None, config=self._cfg("en"))
        self.assertIsNone(_CYRILLIC.search(str(ctx.exception.message)))

    def test_faces_limit_guards(self):
        import typer
        faces_main = group_callback(self.app, "faces")
        ctx_obj = SimpleNamespace(invoked_subcommand=None)
        with self.assertRaises(typer.BadParameter) as ctx:
            faces_main(ctx_obj, rescan=False, limit=5, config=self._cfg("ru"))
        self.assertEqual(str(ctx.exception.message),
                         "--limit работает только вместе с --rescan")
        with self.assertRaises(typer.BadParameter) as ctx:
            faces_main(ctx_obj, rescan=True, limit=0, config=self._cfg("ru"))
        self.assertEqual(str(ctx.exception.message),
                         "--limit должен быть положительным числом")
        with self.assertRaises(typer.BadParameter) as ctx:
            faces_main(ctx_obj, rescan=True, limit=-1, config=self._cfg("ja"))
        self.assertIsNone(_CYRILLIC.search(str(ctx.exception.message)))

    def test_album_copy_and_move(self):
        import typer
        album = command_callback(self.app, "album")
        with self.assertRaises(typer.BadParameter) as ctx:
            album("person", "X", self.root, copy=True, move=True, where=[],
                  name=None, apply=False, config=self._cfg("ru"))
        self.assertEqual(str(ctx.exception.message),
                         "--copy и --move взаимоисключающи")
        with self.assertRaises(typer.BadParameter) as ctx:
            album("person", "X", self.root, copy=True, move=True, where=[],
                  name=None, apply=False, config=self._cfg("en"))
        self.assertIsNone(_CYRILLIC.search(str(ctx.exception.message)))

    def test_the_reset_shell_asks_with_typer(self):
        """`--yes` aside, the question is typer's: `_cmd_reset` only gets handed one."""
        reset = command_callback(self.app, "reset")
        with patch.object(cli, "_cmd_reset") as cmd, \
                patch("typer.confirm") as confirm:
            reset(yes=False, clear_geo=False, config=self._cfg("ru"))
            cmd.call_args.kwargs["confirm"]("question?")
        confirm.assert_called_once_with("question?", abort=True)
        with patch.object(cli, "_cmd_reset") as cmd:
            reset(yes=True, clear_geo=False, config=self._cfg("ru"))
        self.assertIsNone(cmd.call_args.kwargs["confirm"])


class TestEveryKeyTheCliAsksForExists(unittest.TestCase):
    """No `cli.*` literal in cli.py without an entry in the catalog — a typo in a key
    would otherwise print the key itself to the user and nothing would fail."""

    def test_no_unknown_keys_in_cli_module(self):
        source = Path(cli.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r'"(cli\.[a-z0-9_.]+)"', source))
        self.assertGreater(len(used), 50)  # the catalog is actually wired up
        for key in sorted(used):
            with self.subTest(key=key):
                # F230: a key of `install.INSTALL_ADVICE` is a BASE — the literal in the
                # source names a family of three, one per install kind, and it is
                # `install.advice_key` that turns it into the key that gets printed. All
                # three have to exist, which is a stronger check than the one above.
                if key in install.INSTALL_ADVICE:
                    for variant in install.advice_keys(key):
                        self.assertIn(variant, i18n._CLI_STRINGS, variant)
                    continue
                self.assertIn(key, i18n._CLI_STRINGS, key)

    def test_format_fields_are_named(self):
        # Requirement 5: substitutions go through NAMED fields — a positional `{}`
        # would pin the word order of one language onto all three.
        for key, entry in i18n._CLI_STRINGS.items():
            for lang, template in entry.items():
                for _, field, _, _ in string.Formatter().parse(template):
                    if field is not None:
                        self.assertTrue(field.isidentifier(),
                                        f"{key}/{lang}: positional field {field!r}")


if __name__ == "__main__":
    unittest.main()
