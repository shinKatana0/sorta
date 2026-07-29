"""F25: the i18n scaffold — language of service folders and country names (ru|en|ja).

A pure string layer, without FS/DB or side effects. Wiring to the consumers
(sorter/cli/config) is done by the calling modules.

Cities/districts are NOT localized here (reverse_geocoder — transliterated English
only, depends on backlog #4 for the geo hierarchy). F25 covers only countries and
our own service strings (layout folders, reason subfolders).
"""
from __future__ import annotations

from typing import Literal

Lang = Literal["ru", "en", "ja"]

_LANGS: tuple[Lang, ...] = ("ru", "en", "ja")
_DEFAULT_LANG: Lang = "en"


def normalize_lang(value: str | None) -> Lang:
    """Normalize an arbitrary config string into a supported Lang.

    An unknown/empty/invalid value → default en, never crashes.
    """
    if value is None:
        return _DEFAULT_LANG
    candidate = value.strip().lower()
    if candidate in _LANGS:
        return candidate  # type: ignore[return-value]
    return _DEFAULT_LANG


# The keys are the FOLDER SEGMENTS of the layout (user-visible folder names),
# not the internal reason codes from sorter.py/CSV — those must not be touched.
FOLDER_KEYS: tuple[str, ...] = (
    "unsorted",
    "documents",
    "duplicates",
    "shared",
    "junk",
    "no_place",
    "low_date",
    "downloaded",
    "not_personal",
    "no_event",
    "no_faces",
    "document",
    "products",
    "to_delete",
)

_FOLDERS: dict[str, dict[Lang, str]] = {
    "unsorted": {"ru": "_Неразобрано", "en": "_Unsorted", "ja": "_未分類"},
    "documents": {"ru": "_Документы", "en": "_Documents", "ja": "_書類"},
    "duplicates": {"ru": "_Дубликаты", "en": "_Duplicates", "ja": "_重複"},
    "shared": {"ru": "_Совместные", "en": "_Shared", "ja": "_共有"},
    "junk": {"ru": "мусор", "en": "junk", "ja": "ゴミ"},
    "no_place": {"ru": "без_места", "en": "no_place", "ja": "場所不明"},
    "low_date": {"ru": "без_даты", "en": "low_date", "ja": "日付不明"},
    # F78: neither junk nor a lost shot — forwarded/downloaded pictures the user may
    # well want to look through, so the name must not read as a verdict.
    "downloaded": {"ru": "скачанное", "en": "downloaded", "ja": "ダウンロード"},
    "not_personal": {"ru": "не_личное", "en": "not_personal", "ja": "非個人"},
    "no_event": {"ru": "без_события", "en": "no_event", "ja": "イベント不明"},
    "no_faces": {"ru": "без_лиц", "en": "no_faces", "ja": "顔なし"},
    "document": {"ru": "документ", "en": "document", "ja": "書類"},
    "products": {"ru": "_Товары", "en": "_Products", "ja": "_商品"},
    "to_delete": {"ru": "_удалить", "en": "_delete", "ja": "_削除"},
}


def folder(key: str, lang: Lang) -> str:
    """Return the localized folder/subfolder name for a key.

    An unknown key → the key itself (fallback, does not crash).
    """
    entry = _FOLDERS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry[_DEFAULT_LANG])


# Curated dictionary ISO-alpha2 -> {ru, en, ja}. Mandatory coverage of the
# collection's countries (RU/TH/ID/TR/AE) + a reasonable general set.
_COUNTRIES: dict[str, dict[Lang, str]] = {
    "ru": {"ru": "Россия", "en": "Russia", "ja": "ロシア"},
    "th": {"ru": "Таиланд", "en": "Thailand", "ja": "タイ"},
    "id": {"ru": "Индонезия", "en": "Indonesia", "ja": "インドネシア"},
    "tr": {"ru": "Турция", "en": "Turkey", "ja": "トルコ"},
    "ae": {"ru": "ОАЭ", "en": "United Arab Emirates", "ja": "アラブ首長国連邦"},
    "us": {"ru": "США", "en": "United States", "ja": "アメリカ合衆国"},
    "gb": {"ru": "Великобритания", "en": "United Kingdom", "ja": "イギリス"},
    "de": {"ru": "Германия", "en": "Germany", "ja": "ドイツ"},
    "fr": {"ru": "Франция", "en": "France", "ja": "フランス"},
    "it": {"ru": "Италия", "en": "Italy", "ja": "イタリア"},
    "es": {"ru": "Испания", "en": "Spain", "ja": "スペイン"},
    "jp": {"ru": "Япония", "en": "Japan", "ja": "日本"},
    "cn": {"ru": "Китай", "en": "China", "ja": "中国"},
    "ge": {"ru": "Грузия", "en": "Georgia", "ja": "ジョージア"},
    "am": {"ru": "Армения", "en": "Armenia", "ja": "アルメニア"},
    "az": {"ru": "Азербайджан", "en": "Azerbaijan", "ja": "アゼルバイジャン"},
    "kz": {"ru": "Казахстан", "en": "Kazakhstan", "ja": "カザフスタン"},
    "vn": {"ru": "Вьетнам", "en": "Vietnam", "ja": "ベトナム"},
    "kr": {"ru": "Южная Корея", "en": "South Korea", "ja": "韓国"},
    "in": {"ru": "Индия", "en": "India", "ja": "インド"},
    "gr": {"ru": "Греция", "en": "Greece", "ja": "ギリシャ"},
    "eg": {"ru": "Египет", "en": "Egypt", "ja": "エジプト"},
    "cy": {"ru": "Кипр", "en": "Cyprus", "ja": "キプロス"},
    "il": {"ru": "Израиль", "en": "Israel", "ja": "イスラエル"},
    "nl": {"ru": "Нидерланды", "en": "Netherlands", "ja": "オランダ"},
    "pt": {"ru": "Португалия", "en": "Portugal", "ja": "ポルトガル"},
    "ch": {"ru": "Швейцария", "en": "Switzerland", "ja": "スイス"},
    "at": {"ru": "Австрия", "en": "Austria", "ja": "オーストリア"},
    "cz": {"ru": "Чехия", "en": "Czechia", "ja": "チェコ"},
    "pl": {"ru": "Польша", "en": "Poland", "ja": "ポーランド"},
    "fi": {"ru": "Финляндия", "en": "Finland", "ja": "フィンランド"},
    "se": {"ru": "Швеция", "en": "Sweden", "ja": "スウェーデン"},
    "no": {"ru": "Норвегия", "en": "Norway", "ja": "ノルウェー"},
}


def country(cc: str, lang: Lang) -> str:
    """Return the localized country name for an ISO-alpha2 code.

    Code case is irrelevant. An unknown code → the code itself (fallback, does not crash).
    """
    entry = _COUNTRIES.get(cc.strip().lower())
    if entry is None:
        return cc
    return entry.get(lang, entry[_DEFAULT_LANG])


# The reverse of `country()`, built once from the same dictionary: casefolded name in
# ANY of the three languages -> ISO cc. One index for all languages on purpose — the
# caller (F85c: the country a folder name gives away) does not know which language the
# user named their folder in, and a country name is not ambiguous across these three.
_COUNTRY_BY_NAME: dict[str, str] = {
    name.casefold(): cc
    for cc, names in _COUNTRIES.items()
    for name in names.values()
}


def country_cc_by_name(name: str) -> str | None:
    """A country name in ru/en/ja -> ISO cc (upper case); None — not in the dictionary.

    Only the curated dictionary above, which is small and hand-checked; the bundled
    GeoNames base carries far more spellings and is asked separately (see
    geo._CountryFromPath). Case and surrounding whitespace are irrelevant.
    """
    cc = _COUNTRY_BY_NAME.get(name.strip().casefold())
    return cc.upper() if cc else None


# --- F112: the strings the command line prints ------------------------------
# The layout folders and the served UI (ui._UI_STRINGS) had been localized for a
# while, but the CLI spoke Russian whatever `language:` said. The `ru` variants below
# are the texts that used to be hard-coded in cli.py, word for word: a `ru` run has to
# stay byte-identical, this is a re-housing of the messages, not a rewrite.
#
# Keys are named after the command and the meaning (`cli.stats.files`,
# `cli.undo.done`), not after the order they are printed in — there will be a couple
# of hundred of them.
#
# Substitutions are NAMED format fields, never concatenation: word order differs
# between the three languages, and gluing fragments together is a guaranteed bad
# translation. The padding of the aligned `stats` block is baked into each language's
# template for the same reason — a label's width is a property of the language.
#
# NOT COVERED HERE: `--help` texts. They live inside `typer.Option(..., help=...)`
# decorators, which are evaluated at import time — before the config is read and the
# language is known. Localizing the help needs a different mechanism (deferred
# descriptors or an own command factory), which is a feature of its own, not a tail of
# this one. Also not covered: the summaries `sorta doctor` gets from diagnostics.py —
# they are produced by another module (see the note in cli.doctor).
_CLI_STRINGS: dict[str, dict[Lang, str]] = {
    # index
    "cli.index.done": {
        "ru": "Готово: +{added} новых, ~{updated} обновлено, {skipped} пропущено, "
              "{errors} ошибок, {dups} дубликатов помечено",
        "en": "Done: +{added} new, ~{updated} updated, {skipped} skipped, "
              "{errors} errors, {dups} duplicates marked",
        "ja": "完了: 新規 +{added}、更新 ~{updated}、スキップ {skipped}、"
              "エラー {errors}、重複マーク {dups}",
    },
    "cli.index.no_source": {
        "ru": "не задан источник: укажите каталог — sorta index <src_dir> — "
              "или заполните секцию 'sources' в config.yaml",
        "en": "no source given: pass a directory — sorta index <src_dir> — "
              "or fill in the 'sources' section of config.yaml",
        "ja": "ソースが指定されていません: ディレクトリを渡す（sorta index <src_dir>）か、"
              "config.yaml の 'sources' セクションを記入してください",
    },
    # refresh-exif
    "cli.refresh.done": {
        "ru": "Перечитано: {scanned} файлов, обновлено {updated}; "
              "вернулось координат: {gps}, дат съёмки: {dates}; "
              "без EXIF: {empty}, ошибок: {errors}",
        "en": "Re-read: {scanned} files, {updated} updated; "
              "coordinates recovered: {gps}, capture dates: {dates}; "
              "without EXIF: {empty}, errors: {errors}",
        "ja": "再読み込み: {scanned} ファイル、更新 {updated}。"
              "復元した座標: {gps}、撮影日: {dates}。"
              "EXIF なし: {empty}、エラー: {errors}",
    },
    "cli.refresh.rerun_geo": {
        "ru": "Появились новые координаты — перезапустите: sorta geo (и sorta events)",
        "en": "New coordinates appeared — re-run: sorta geo (and sorta events)",
        "ja": "新しい座標が見つかりました — 再実行してください: sorta geo（および sorta events）",
    },
    # index --exclude-dir
    "cli.excludes.no_source": {
        "ru": "--exclude-dir: не задан источник — укажите каталог позиционно "
              "или заполните 'sources' в config.yaml",
        "en": "--exclude-dir: no source given — pass a directory positionally "
              "or fill in 'sources' in config.yaml",
        "ja": "--exclude-dir: ソースが指定されていません — ディレクトリを位置引数で渡すか、"
              "config.yaml の 'sources' を記入してください",
    },
    "cli.excludes.saved": {
        "ru": "Исключено из сканирования ({root}): {values}",
        "en": "Excluded from scanning ({root}): {values}",
        "ja": "スキャンから除外しました（{root}）: {values}",
    },
    "cli.excludes.file": {
        "ru": "Файл исключений: {path}",
        "en": "Excludes file: {path}",
        "ja": "除外ファイル: {path}",
    },
    # geo
    "cli.geo.done": {
        "ru": "Готово: {total} файлов — exact_gps {exact_gps}, "
              "session_inferred {session_inferred}, trip_inferred {trip_inferred}, "
              "path_inferred {path_inferred}, unknown {unknown}",
        "en": "Done: {total} files — exact_gps {exact_gps}, "
              "session_inferred {session_inferred}, trip_inferred {trip_inferred}, "
              "path_inferred {path_inferred}, unknown {unknown}",
        "ja": "完了: {total} ファイル — exact_gps {exact_gps}、"
              "session_inferred {session_inferred}、trip_inferred {trip_inferred}、"
              "path_inferred {path_inferred}、unknown {unknown}",
    },
    # landmarks
    "cli.landmarks.done": {
        "ru": "Места без GPS: просмотрено {scanned}, определено {matched}",
        "en": "Places without GPS: {scanned} scanned, {matched} identified",
        "ja": "GPS なしの場所: 確認 {scanned}、判定 {matched}",
    },
    # faces
    "cli.faces.detected": {
        "ru": "Детекция: {files} файлов, {faces} лиц, {no_faces} без лиц, {errors} ошибок",
        "en": "Detection: {files} files, {faces} faces, {no_faces} without faces, "
              "{errors} errors",
        "ja": "検出: {files} ファイル、{faces} 個の顔、顔なし {no_faces}、エラー {errors}",
    },
    "cli.faces.clusters": {
        "ru": "Кластеры: {clusters} (лиц в кластерах: {clustered}, шум: {noise}, "
              "имён сохранено: {labels_kept})",
        "en": "Clusters: {clusters} (faces in clusters: {clustered}, noise: {noise}, "
              "names kept: {labels_kept})",
        "ja": "クラスタ: {clusters}（クラスタ内の顔: {clustered}、ノイズ: {noise}、"
              "保持した名前: {labels_kept}）",
    },
    "cli.faces.malformed": {
        "ru": "⚠ повреждённых эмбеддингов пропущено: {n}",
        "en": "⚠ malformed embeddings skipped: {n}",
        "ja": "⚠ 破損した埋め込みをスキップ: {n}",
    },
    "cli.faces.labeled": {
        "ru": "Кластер {cluster} назван: {name}",
        "en": "Cluster {cluster} named: {name}",
        "ja": "クラスタ {cluster} に名前を付けました: {name}",
    },
    "cli.faces.merged": {
        "ru": "Слито: {src} -> {dst}",
        "en": "Merged: {src} -> {dst}",
        "ja": "統合しました: {src} -> {dst}",
    },
    "cli.faces.sheet_done": {
        "ru": "Готово: {n} лиц -> {path}",
        "en": "Done: {n} faces -> {path}",
        "ja": "完了: {n} 個の顔 -> {path}",
    },
    "cli.faces.limit_needs_rescan": {
        "ru": "--limit работает только вместе с --rescan",
        "en": "--limit works only together with --rescan",
        "ja": "--limit は --rescan と一緒でのみ使えます",
    },
    "cli.faces.limit_positive": {
        "ru": "--limit должен быть положительным числом",
        "en": "--limit must be a positive number",
        "ja": "--limit は正の数でなければなりません",
    },
    # events
    "cli.events.done": {
        "ru": "События: {auto_events} авто ({auto_files} файлов, "
              "имён сохранено: {names_preserved}), "
              "{manual_events} ручных ({manual_files} файлов)",
        "en": "Events: {auto_events} automatic ({auto_files} files, "
              "names kept: {names_preserved}), "
              "{manual_events} manual ({manual_files} files)",
        "ja": "イベント: 自動 {auto_events}（{auto_files} ファイル、"
              "保持した名前: {names_preserved}）、"
              "手動 {manual_events}（{manual_files} ファイル）",
    },
    "cli.events.renamed": {
        "ru": "Событие {event_id}: {name}",
        "en": "Event {event_id}: {name}",
        "ja": "イベント {event_id}: {name}",
    },
    "cli.events.added": {
        "ru": "Ручное событие {event_id}: {name} ({date_from}..{date_to})",
        "en": "Manual event {event_id}: {name} ({date_from}..{date_to})",
        "ja": "手動イベント {event_id}: {name}（{date_from}..{date_to}）",
    },
    # junk
    "cli.junk.done": {
        "ru": "Классификация: {processed}/{total} обработано ({kinds})",
        "en": "Classification: {processed}/{total} processed ({kinds})",
        "ja": "分類: {processed}/{total} 処理済み（{kinds}）",
    },
    "cli.junk.skipped_incremental": {
        "ru": "; пропущено как уже обработанные: {n}",
        "en": "; skipped as already processed: {n}",
        "ja": "; 処理済みのためスキップ: {n}",
    },
    "cli.junk.vlm": {
        "ru": "; VLM: {applied}/{candidates} кандидатов переклассифицировано",
        "en": "; VLM: {applied}/{candidates} candidates reclassified",
        "ja": "; VLM: 候補 {candidates} 件中 {applied} 件を再分類",
    },
    # phash
    "cli.phash.done": {
        "ru": "pHash посчитан для {n} фото. Отчёт: sorta dupes --near",
        "en": "pHash computed for {n} photos. Report: sorta dupes --near",
        "ja": "{n} 枚の写真の pHash を計算しました。レポート: sorta dupes --near",
    },
    # stats — the padding of each label is part of its own language's template
    "cli.stats.empty": {
        "ru": "Индекс пуст — запустите: sorta index",
        "en": "The index is empty — run: sorta index",
        "ja": "インデックスが空です — 実行してください: sorta index",
    },
    "cli.stats.files": {
        "ru": "Файлов в индексе: {total} (+{errors} с ошибками)",
        "en": "Files in the index: {total} (+{errors} with errors)",
        "ja": "インデックス内のファイル: {total}（エラー +{errors}）",
    },
    "cli.stats.gps": {
        "ru": "  с GPS:            {n} ({pct}%)",
        "en": "  with GPS:           {n} ({pct}%)",
        "ja": "  GPS あり:           {n} ({pct}%)",
    },
    "cli.stats.date_source": {
        "ru": "  дата из {source:9}: {n} ({pct}%)",
        "en": "  date from {source:9}: {n} ({pct}%)",
        "ja": "  日付ソース {source:9}: {n} ({pct}%)",
    },
    "cli.stats.dupes": {
        "ru": "  дубликатов:       {n}",
        "en": "  duplicates:         {n}",
        "ja": "  重複:                {n}",
    },
    "cli.stats.geo_total": {
        "ru": "Гео (places): {n}",
        "en": "Geo (places): {n}",
        "ja": "位置情報 (places): {n}",
    },
    "cli.stats.geo_confidence": {
        # `confidence` is a code value from the DB (exact_gps, unknown, …) and is the
        # same in every language — only the padding column differs.
        "ru": "  {confidence:16}: {n} ({pct}%)",
        "en": "  {confidence:16}: {n} ({pct}%)",
        "ja": "  {confidence:16}: {n} ({pct}%)",
    },
    "cli.stats.faces": {
        "ru": "Лица: {faces} (кластеров: {clusters}, именованных: {named})",
        "en": "Faces: {faces} (clusters: {clusters}, named: {named})",
        "ja": "顔: {faces}（クラスタ: {clusters}、名前付き: {named}）",
    },
    # dupes
    "cli.dupes.no_phash": {
        "ru": "pHash ещё не посчитан — запустите: sorta phash",
        "en": "pHash has not been computed yet — run: sorta phash",
        "ja": "pHash がまだ計算されていません — 実行してください: sorta phash",
    },
    "cli.dupes.near_none": {
        "ru": "Почти-дубликатов не найдено",
        "en": "No near-duplicates found",
        "ja": "類似写真は見つかりませんでした",
    },
    "cli.dupes.near_group": {
        "ru": "Группа из {n} похожих:",
        "en": "A group of {n} similar:",
        "ja": "類似 {n} 件のグループ:",
    },
    "cli.dupes.near_item": {
        "ru": "  {path}  ({size} байт)",
        "en": "  {path}  ({size} bytes)",
        "ja": "  {path}  ({size} バイト)",
    },
    "cli.dupes.near_total": {
        "ru": "Групп: {n} (порог Хэмминга: {threshold})",
        "en": "Groups: {n} (Hamming threshold: {threshold})",
        "ja": "グループ: {n}（ハミング距離のしきい値: {threshold}）",
    },
    "cli.dupes.exact_none": {
        "ru": "Точных дубликатов не найдено",
        "en": "No exact duplicates found",
        "ja": "完全一致の重複は見つかりませんでした",
    },
    "cli.dupes.exact_item": {
        "ru": "{dup}\n  -> дубликат {canon}",
        "en": "{dup}\n  -> duplicate of {canon}",
        "ja": "{dup}\n  -> {canon} の重複",
    },
    "cli.dupes.exact_total": {
        "ru": "Всего: {n}",
        "en": "Total: {n}",
        "ja": "合計: {n}",
    },
    # run
    "cli.run.no_source": {
        "ru": "не задан источник: укажите --src <каталог> или заполните "
              "'sources' в config.yaml",
        "en": "no source given: pass --src <directory> or fill in "
              "'sources' in config.yaml",
        "ja": "ソースが指定されていません: --src <ディレクトリ> を指定するか、"
              "config.yaml の 'sources' を記入してください",
    },
    "cli.run.stage": {
        "ru": "[этап {index}/{total}] {name}",
        "en": "[stage {index}/{total}] {name}",
        "ja": "[ステージ {index}/{total}] {name}",
    },
    "cli.run.plan": {
        "ru": "[план] dry-run sort --by {by} -> {dest}",
        "en": "[plan] dry-run sort --by {by} -> {dest}",
        "ja": "[プラン] dry-run sort --by {by} -> {dest}",
    },
    "cli.run.finished": {
        "ru": "Анализ завершён. Индекс наполнен; просмотрите план и запустите sort "
              "при необходимости.",
        "en": "Analysis finished. The index is filled in; review the plan and run sort "
              "if needed.",
        "ja": "分析が完了しました。インデックスが作成されました。"
              "プランを確認し、必要であれば sort を実行してください。",
    },
    "cli.run.geo_choice": {
        "ru": "--geo должен быть offline или online",
        "en": "--geo must be offline or online",
        "ja": "--geo は offline か online でなければなりません",
    },
    # doctor
    "cli.doctor.log": {
        "ru": "Лог прогона: {path}",
        "en": "Run log: {path}",
        "ja": "実行ログ: {path}",
    },
    # cache (the preview-cache line is printed by `doctor` too — one text, one key)
    "cli.cache.preview_dir": {
        "ru": "Кэш превью: {path}",
        "en": "Preview cache: {path}",
        "ja": "プレビューキャッシュ: {path}",
    },
    "cli.cache.preview_disabled": {
        "ru": " (ОТКЛЮЧЁН)",
        "en": " (DISABLED)",
        "ja": "（無効）",
    },
    "cli.cache.preview_stats": {
        "ru": "  файлов: {files}, размер: {size_gb:.2f} ГБ",
        "en": "  files: {files}, size: {size_gb:.2f} GB",
        "ja": "  ファイル: {files}、サイズ: {size_gb:.2f} GB",
    },
    "cli.cache.preview_cleared": {
        "ru": "Кэш превью удалён: {path}",
        "en": "Preview cache removed: {path}",
        "ja": "プレビューキャッシュを削除しました: {path}",
    },
    "cli.cache.geo_cleared": {
        "ru": "Кэш геоданных очищен: удалено записей {n}",
        "en": "Geo cache cleared: {n} entries removed",
        "ja": "位置情報キャッシュを消去しました: {n} 件を削除",
    },
    "cli.cache.geo_size": {
        "ru": "Кэш геоданных (geo_cache): записей {n}",
        "en": "Geo cache (geo_cache): {n} entries",
        "ja": "位置情報キャッシュ (geo_cache): {n} 件",
    },
    # sort — moved/copied are separate templates, not a verb pasted into a sentence
    "cli.sort.moved": {
        "ru": "Перемещено {moved}, на месте {in_place}, ошибок {failed}{extra}. "
              "Откат: sorta undo",
        "en": "Moved {moved}, in place {in_place}, errors {failed}{extra}. "
              "Undo: sorta undo",
        "ja": "移動 {moved}、そのまま {in_place}、エラー {failed}{extra}。"
              "取り消し: sorta undo",
    },
    "cli.sort.copied": {
        "ru": "Скопировано {moved}, на месте {in_place}, ошибок {failed}{extra}. "
              "Откат: sorta undo",
        "en": "Copied {moved}, in place {in_place}, errors {failed}{extra}. "
              "Undo: sorta undo",
        "ja": "コピー {moved}、そのまま {in_place}、エラー {failed}{extra}。"
              "取り消し: sorta undo",
    },
    "cli.sort.deleted_dupes": {
        "ru": ", удалено дублей {n}",
        "en": ", duplicates deleted {n}",
        "ja": "、重複削除 {n}",
    },
    # album
    "cli.album.done": {
        "ru": "Альбом «{name}»: выгружено {transferred}, ошибок {failed}{extra}. "
              "Откат: sorta undo",
        "en": "Album “{name}”: {transferred} exported, {failed} errors{extra}. "
              "Undo: sorta undo",
        "ja": "アルバム「{name}」: 書き出し {transferred}、エラー {failed}{extra}。"
              "取り消し: sorta undo",
    },
    "cli.album.blocked_multi": {
        "ru": ", заблокировано (мульти) {n}",
        "en": ", blocked (multi) {n}",
        "ja": "、ブロック（マルチ）{n}",
    },
    "cli.album.copy_move_exclusive": {
        "ru": "--copy и --move взаимоисключающи",
        "en": "--copy and --move are mutually exclusive",
        "ja": "--copy と --move は同時に指定できません",
    },
    # reset
    "cli.reset.confirm": {
        "ru": "Стереть весь индекс? Имена людей/событий и решения по дублям "
              "пропадут; фото и уже разложенные папки НЕ тронутся{extra}",
        "en": "Erase the whole index? People/event names and duplicate decisions "
              "will be lost; photos and already sorted folders are NOT touched{extra}",
        "ja": "インデックスをすべて消去しますか？ 人物・イベントの名前と重複の判断は"
              "失われます。写真と整理済みのフォルダには手を触れません{extra}",
    },
    "cli.reset.confirm_geo": {
        "ru": ", кэш геоданных тоже будет очищен",
        "en": ", the geo cache will be cleared as well",
        "ja": "。位置情報キャッシュも消去されます",
    },
    "cli.reset.done": {
        "ru": "Индекс стёрт. Запустите `sorta index`/`sorta run` заново.{extra}",
        "en": "The index is erased. Run `sorta index`/`sorta run` again.{extra}",
        "ja": "インデックスを消去しました。`sorta index`/`sorta run` を"
              "もう一度実行してください。{extra}",
    },
    "cli.reset.done_geo": {
        "ru": " Кэш геоданных очищен.",
        "en": " The geo cache is cleared.",
        "ja": "位置情報キャッシュも消去しました。",
    },
    # undo
    "cli.undo.done": {
        "ru": "Откат батча {batch}: возвращено {undone}, отсутствовало {missing}, "
              "ошибок {failed}",
        "en": "Undo of batch {batch}: {undone} restored, {missing} missing, "
              "{failed} errors",
        "ja": "バッチ {batch} の取り消し: 復元 {undone}、欠落 {missing}、エラー {failed}",
    },
    # not-yet-implemented placeholder
    "cli.stub.next_phase": {
        "ru": "'{name}' будет реализована в следующей фазе: {doc}",
        "en": "'{name}' will be implemented in the next phase: {doc}",
        "ja": "'{name}' は次のフェーズで実装されます: {doc}",
    },
    # progress bar captions (a step's title and the phases inside it)
    "cli.progress.index": {
        "ru": "index: сканирование", "en": "index: scanning", "ja": "index: スキャン",
    },
    "cli.progress.refresh_exif": {
        "ru": "refresh-exif: метаданные", "en": "refresh-exif: metadata",
        "ja": "refresh-exif: メタデータ",
    },
    "cli.progress.geo": {
        "ru": "geo: места", "en": "geo: places", "ja": "geo: 場所",
    },
    "cli.progress.faces": {
        "ru": "faces: детекция", "en": "faces: detection", "ja": "faces: 検出",
    },
    "cli.progress.faces_rescan": {
        "ru": "faces: пересканирование", "en": "faces: rescan", "ja": "faces: 再スキャン",
    },
    "cli.progress.landmarks": {
        "ru": "landmarks: места без GPS", "en": "landmarks: places without GPS",
        "ja": "landmarks: GPS なしの場所",
    },
    "cli.progress.phash": {
        "ru": "phash: почти-дубликаты", "en": "phash: near-duplicates",
        "ja": "phash: 類似写真",
    },
    "cli.progress.junk": {
        "ru": "junk: классификация", "en": "junk: classification", "ja": "junk: 分類",
    },
    "cli.progress.events": {
        "ru": "events: кластеризация", "en": "events: clustering",
        "ja": "events: クラスタリング",
    },
    "cli.phase.cluster_read": {
        "ru": "кластеры: чтение эмбеддингов", "en": "clusters: reading embeddings",
        "ja": "クラスタ: 埋め込みの読み込み",
    },
    "cli.phase.cluster_cluster": {
        "ru": "кластеры: группировка лиц (без процента)",
        "en": "clusters: grouping faces (no percentage)",
        "ja": "クラスタ: 顔のグループ化（進捗率なし）",
    },
    "cli.phase.cluster_inherit": {
        "ru": "кластеры: перенос имён", "en": "clusters: carrying names over",
        "ja": "クラスタ: 名前の引き継ぎ",
    },
    "cli.phase.cluster_write": {
        "ru": "кластеры: запись", "en": "clusters: writing", "ja": "クラスタ: 書き込み",
    },
    "cli.phase.junk_clip": {
        "ru": "junk: классификация CLIP", "en": "junk: CLIP classification",
        "ja": "junk: CLIP 分類",
    },
    "cli.phase.junk_ocr": {
        "ru": "junk: распознавание текста", "en": "junk: text recognition",
        "ja": "junk: テキスト認識",
    },
    "cli.phase.junk_vlm": {
        "ru": "junk: глубокий анализ (VLM)", "en": "junk: deep analysis (VLM)",
        "ja": "junk: 詳細分析（VLM）",
    },
    "cli.phase.junk_write": {
        "ru": "junk: запись вердиктов", "en": "junk: writing verdicts",
        "ja": "junk: 判定の書き込み",
    },
}


def cli_text(key: str, lang: Lang, **fields: object) -> str:
    """Resolve and format a CLI string: exact language -> en -> the key itself.

    Same fallback chain as the served UI's `_t` (F33): a key that is missing a language
    still prints something readable instead of blowing up mid-command. `fields` are
    substituted by name; every language of a key carries the same set of fields (the
    test suite enforces it), so the substitution cannot depend on the language.
    """
    entry = _CLI_STRINGS.get(key)
    if entry is None:
        return key
    return (entry.get(lang) or entry[_DEFAULT_LANG]).format(**fields)
