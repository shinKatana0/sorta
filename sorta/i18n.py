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
    "animals",
    "screenshots",
    "memes",
    "blurred",
    "eyes_closed",
    "low_resolution",
    "people",
    "group_photos",
    "portraits",
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
    # F175: the DOWNLOADED-FILM folder and nothing else. `files.not_personal` comes from a
    # heuristic over the file NAME (`S01E05`, `1080p`, a rip group) and marks three files
    # out of 38 485 on the live collection — not to be confused with the «Служебные кадры»
    # slice, which is about what is IN the frame and holds some five thousand of them.
    "not_personal": {"ru": "не_личное", "en": "not_personal", "ja": "非個人"},
    "no_event": {"ru": "без_события", "en": "no_event", "ja": "イベント不明"},
    "no_faces": {"ru": "без_лиц", "en": "no_faces", "ja": "顔なし"},
    "document": {"ru": "документ", "en": "document", "ja": "書類"},
    "products": {"ru": "_Товары", "en": "_Products", "ja": "_商品"},
    "to_delete": {"ru": "_удалить", "en": "_delete", "ja": "_削除"},
    # F123: the default folder of the animal album. The slice has no selector to name
    # itself after (unlike a person or an event), so the name is a folder name like any
    # other the layout creates — and follows `language:` for the same reason.
    "animals": {"ru": "_Животные", "en": "_Animals", "ja": "_動物"},
    # F139: the default folder of the remaining slices, for the same reason as
    # `animals` — none of them has a selector to name itself after. `products` above is
    # reused by the product album rather than duplicated: the folder the layout files
    # that bucket into and the album a person gathers by hand are one name.
    "screenshots": {"ru": "_Скриншоты", "en": "_Screenshots", "ja": "_スクリーンショット"},
    "memes": {"ru": "_Мемы", "en": "_Memes", "ja": "_ミーム"},
    "blurred": {"ru": "_Размытые", "en": "_Blurred", "ja": "_ぼやけ"},
    "eyes_closed": {"ru": "_Закрытые_глаза", "en": "_Closed_eyes", "ja": "_目を閉じた"},
    # F150: named after the FACT (how many pixels the frame has) and never after a
    # judgement ("bad", "junk"). A frame of 640x480 can be the only surviving photograph
    # of somebody, sent ten years ago through a messenger, and a folder called otherwise
    # would be telling the person what to do with it.
    "low_resolution": {"ru": "_Низкое_разрешение", "en": "_Low_resolution",
                       "ja": "_低解像度"},
    # F152: the three face slices have no selector either — the collection holds exactly
    # one of each — so their album folders are named the same way the animal one is.
    # "People" here is the question "is there a face in this frame", not "who is it":
    # a named person's album is still called after the person.
    "people": {"ru": "_С людьми", "en": "_With people", "ja": "_人物あり"},
    "group_photos": {"ru": "_Групповые", "en": "_Group photos", "ja": "_集合写真"},
    "portraits": {"ru": "_Портреты", "en": "_Portraits", "ja": "_ポートレート"},
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

    Only the curated dictionary above; the bundled GeoNames base carries far more
    spellings and is asked separately (see geo._CountryFromPath).
    """
    cc = _COUNTRY_BY_NAME.get(name.strip().casefold())
    return cc.upper() if cc else None


# --- F112: the strings the command line prints ------------------------------
# The `ru` variants are the texts that used to be hard-coded in cli.py, word for word: a
# `ru` run has to stay byte-identical, this was a re-housing and not a rewrite.
#
# Keys are named after the command and the meaning (`cli.stats.files`), not after the
# order they print in — there are a couple of hundred of them.
#
# Substitutions are NAMED format fields, never concatenation: word order differs between
# the three languages. The padding of the aligned `stats` block is baked into each
# language's template for the same reason — a label's width is a property of the language.
#
# NOT COVERED HERE: the summaries `sorta doctor` gets from diagnostics.py.
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
    # F222: the stage is skipped and the file still configures it. SAID, and no more than
    # said — settings are not consent and nothing is switched on by them, or in six months
    # nobody could explain why the stage ran. Silence would be worse still: the owner of
    # such a file would learn about the change from a missing result.
    "cli.run.stage_skipped_configured": {
        "ru": "Этап «{stage}» пропущен: с этой версии он выключен по умолчанию, а в "
              "config.yaml остались его настройки ({keys}). {how}",
        "en": "The “{stage}” stage was skipped: it is off by default from this version "
              "on, and config.yaml still holds its settings ({keys}). {how}",
        "ja": "「{stage}」ステージはスキップされました: このバージョンから既定で無効に"
              "なりましたが、config.yaml にはその設定が残っています（{keys}）。{how}",
    },
    # Every way to switch that stage back on, in one sentence per stage — a flag, a key
    # and a checkbox are three different things and only the stage knows which it has.
    "cli.run.enable_landmarks": {
        "ru": "Включить: features.landmarks: true в config.yaml, флаг --landmarks у "
              "sorta run или галочка «Узнавать места по виду» на экране запуска.",
        "en": "To switch it on: features.landmarks: true in config.yaml, the --landmarks "
              "flag of sorta run, or the “Recognise places by sight” checkbox on the run "
              "screen.",
        "ja": "有効にするには: config.yaml の features.landmarks: true、sorta run の "
              "--landmarks フラグ、または実行画面の「見た目で場所を判定」チェック。",
    },
    # F237: said before the run starts, because after it there is nobody to say it — the
    # kernel kills the process without a traceback and the log stops mid-word. A warning
    # and not a refusal: memory moves, and somebody else's browser gives a gigabyte back a
    # minute later. The same sentence is the run screen's (`ui/strings.py` reads it).
    "cli.run.low_memory": {
        "ru": "Свободной памяти мало: {free}, а прогону с классификацией нужно около "
              "{needed} — система может убить процесс без сообщения. Чтобы обойтись "
              "меньшим, снимите «Глубокий анализ (VLM)» и классификацию кадров: "
              "раскладка по городам работает и без них.",
        "en": "Little free memory: {free}, and a run with the classification needs about "
              "{needed} — the system may kill the process with no message. To get by with "
              "less, switch off Deep analysis (VLM) and the frame classification: sorting "
              "into city folders works without them.",
        "ja": "空きメモリが少なめです: {free}。分類を含む実行には約 {needed} が必要で、"
              "システムがメッセージなしにプロセスを終了させることがあります。少ない"
              "メモリで済ませるには「詳細分析（VLM）」とコマの分類をオフにしてください: "
              "都市ごとの振り分けはそれらがなくても動作します。",
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
    # F211: which interpreter and which `uv` this run is using. A shadowed PATH is
    # otherwise found nine minutes later, in a red gate, and reads as somebody else's
    # mistake (F87, a system python with no ruff and no pytest).
    "cli.doctor.python": {
        "ru": "Интерпретатор: {path}",
        "en": "Interpreter: {path}",
        "ja": "インタプリタ: {path}",
    },
    "cli.doctor.uv": {
        "ru": "Менеджер окружения uv: {path}",
        "en": "Environment manager uv: {path}",
        "ja": "環境マネージャ uv: {path}",
    },
    # An installed copy carries `uv.exe` and the wizard installs tiers with it by absolute
    # path, so "not on PATH" was true and useless — and it stood one line above an
    # exiftool that names its own shipped copy.
    "cli.doctor.uv_bundled": {
        "ru": "Менеджер окружения uv: {path} (поставляется с программой)",
        "en": "Environment manager uv: {path} (ships with the program)",
        "ja": "環境マネージャ uv: {path}（本体に同梱）",
    },
    "cli.doctor.uv_missing": {
        "ru": "Менеджер окружения uv: не найден в PATH",
        "en": "Environment manager uv: not on PATH",
        "ja": "環境マネージャ uv: PATH にありません",
    },
    # F213: on Linux `uv tool install` IS the installer, so the ways it leaves a machine
    # half-installed are answered here, in words. `sorta: command not found` is the first:
    # uv writes into `~/.local/bin`, which a default shell profile keeps off PATH, and
    # uv's own warning scrolls past inside the install output.
    "cli.doctor.command": {
        "ru": "Команда sorta: {path}",
        "en": "Command sorta: {path}",
        "ja": "コマンド sorta: {path}",
    },
    "cli.doctor.command_missing": {
        "ru": "Команда sorta: не в PATH (команды этой установки лежат в {path})",
        "en": "Command sorta: not on PATH (this install keeps its commands in {path})",
        "ja": "コマンド sorta: PATH にありません（このインストールのコマンドは {path}）",
    },
    # F230: the fix differs per install path, so there are three keys and
    # `install.advice_key` picks one. `uv tool update-shell` repairs a `uv tool install`
    # and nothing else — in a checkout the commands were never meant to be on PATH.
    "cli.doctor.command_hint.tool": {
        "ru": "  Добавить этот каталог в PATH: uv tool update-shell, затем новый терминал.",
        "en": "  Add that directory to PATH: uv tool update-shell, then a new terminal.",
        "ja": "  その場所を PATH に追加: uv tool update-shell を実行し、端末を開き直します。",
    },
    "cli.doctor.command_hint.checkout": {
        "ru": "  В чекауте команды запускаются так: uv run sorta <команда> "
              "(или активируйте .venv).",
        "en": "  In a checkout the commands are run as: uv run sorta <command> "
              "(or activate .venv).",
        "ja": "  チェックアウトでは次のように実行します: uv run sorta <コマンド>"
              "（または .venv を有効化）。",
    },
    # The second one, and the quietest failure in the product: without exiftool the
    # reader falls back to Pillow, so HEIC/RAW/video dates, GPS and orientation are not
    # read at all. Nothing raises — the photographs simply have no dates to sort by.
    "cli.doctor.exiftool": {
        "ru": "Чтение метаданных exiftool: {path}",
        "en": "Metadata reader exiftool: {path}",
        "ja": "メタデータ読み取り exiftool: {path}",
    },
    "cli.doctor.exiftool_missing": {
        "ru": "Чтение метаданных exiftool: не найден в PATH — даты, GPS и ориентация "
              "HEIC/RAW/видео читаться не будут",
        "en": "Metadata reader exiftool: not on PATH — HEIC/RAW/video dates, GPS and "
              "orientation are not read",
        "ja": "メタデータ読み取り exiftool: PATH にありません — HEIC/RAW/動画の"
              "日付・GPS・向きは読み取られません",
    },
    "cli.doctor.exiftool_linux": {
        "ru": "  Установить: sudo apt install libimage-exiftool-perl "
              "(Fedora: sudo dnf install perl-Image-ExifTool).",
        "en": "  Install it: sudo apt install libimage-exiftool-perl "
              "(Fedora: sudo dnf install perl-Image-ExifTool).",
        "ja": "  インストール: sudo apt install libimage-exiftool-perl"
              "（Fedora: sudo dnf install perl-Image-ExifTool）。",
    },
    "cli.doctor.exiftool_windows": {
        "ru": "  Установить: winget install OliverBetz.ExifTool.",
        "en": "  Install it: winget install OliverBetz.ExifTool.",
        "ja": "  インストール: winget install OliverBetz.ExifTool。",
    },
    "cli.doctor.exiftool_macos": {
        "ru": "  Установить: brew install exiftool.",
        "en": "  Install it: brew install exiftool.",
        "ja": "  インストール: brew install exiftool。",
    },
    # F226: and the fourth answer, the one an installed copy gets. The installer carries
    # exiftool, so on that machine there is nothing to install and no `winget` line to
    # print — the path is named instead, because the whole defect was a product that
    # could not say which of its two exiftools it was using.
    "cli.doctor.exiftool_bundled": {
        "ru": "Чтение метаданных exiftool: {path} (поставляется с программой)",
        "en": "Metadata reader exiftool: {path} (ships with the program)",
        "ja": "メタデータ読み取り exiftool: {path}（本体に同梱）",
    },
    # F226: the same correction one line up. `uv tool update-shell` is the fix for an
    # install made by `uv tool install`; an installed copy was never on PATH and is not
    # meant to be, so it is told how it IS run instead of how to repair something that
    # was never broken.
    "cli.doctor.command_installed": {
        "ru": "Команда sorta: не в PATH — установленная копия ничего в PATH и не кладёт",
        "en": "Command sorta: not on PATH — an installed copy puts nothing there",
        "ja": "コマンド sorta: PATH にありません — インストール版は PATH に何も置きません",
    },
    "cli.doctor.command_hint.installed": {
        "ru": "  Запускать так: \"{path}\" -m sorta.cli <команда>, "
              "или ярлыки в меню «Пуск».",
        "en": "  Run it as: \"{path}\" -m sorta.cli <command>, "
              "or from the Start menu shortcuts.",
        "ja": "  実行方法: \"{path}\" -m sorta.cli <コマンド>、"
              "またはスタートメニューのショートカット。",
    },
    # F216: which tiers are actually on this machine. A tier is two halves that fail
    # differently — the packages in `{app}\lib` and the weights a stage fetches on first
    # use — so there are three answers and not two: the middle one is a tier whose
    # packages are in place and whose 400 MB have not been downloaded yet.
    "cli.doctor.tiers": {
        "ru": "Ярусы установки:",
        "en": "Installed tiers:",
        "ja": "インストール済みティア:",
    },
    "cli.doctor.tier_ready": {
        "ru": "  {name}: на месте",
        "en": "  {name}: in place",
        "ja": "  {name}: 利用可能",
    },
    "cli.doctor.tier_weights": {
        "ru": "  {name}: пакеты на месте, модели ({weights}, {size}) скачаются при "
              "первом запуске стадии",
        "en": "  {name}: packages in place, the models ({weights}, {size}) download on "
              "the first run of the stage",
        "ja": "  {name}: パッケージは配置済み、モデル（{weights}、{size}）は"
              "ステージの初回実行時にダウンロードされます",
    },
    "cli.doctor.tier_absent": {
        "ru": "  {name}: не установлен (нет: {missing})",
        "en": "  {name}: not installed (missing: {missing})",
        "ja": "  {name}: 未インストール（不足: {missing}）",
    },
    "cli.doctor.tier_hint.installed": {
        "ru": "  Доустановить ярус: sorta-setup (пункт «Sorta setup» в меню «Пуск»).",
        "en": "  To add a tier: sorta-setup (the Sorta setup item of the Start menu).",
        "ja": "  ティアの追加: sorta-setup（スタートメニューの「Sorta setup」）。",
    },
    # F213: the same way out for an install that came from `uv tool install`. The Start
    # menu belongs to the Windows installer and nothing else; naming it on a machine
    # that has no such entry sends a person looking for a menu item that is not there.
    "cli.doctor.tier_hint.tool": {
        "ru": "  Доустановить ярус: sorta-setup.",
        "en": "  To add a tier: run sorta-setup.",
        "ja": "  ティアの追加: sorta-setup を実行します。",
    },
    # F230: the third line, for a checkout. There a tier is an extra of `pyproject.toml`
    # and `uv sync` installs it; `sorta-setup` would `uv pip install` into a synced
    # environment, which the next `uv sync` silently undoes.
    "cli.doctor.tier_hint.checkout": {
        "ru": "  Доустановить ярус в чекауте: uv sync --extra <ярус> "
              "(например uv sync --extra gpu --extra dev).",
        "en": "  To add a tier in a checkout: uv sync --extra <tier> "
              "(for example uv sync --extra gpu --extra dev).",
        "ja": "  チェックアウトでティアを追加: uv sync --extra <ティア>"
              "（例: uv sync --extra gpu --extra dev）。",
    },
    # F222: a stage that goes to the network says so, and a stage that could not says
    # what it was fetching. Both sentences are built in `sorta/tiers.py`, from the one
    # table that knows which weights a stage raises — the CLI and the web app print the
    # same words because they call the same builder.
    "cli.download.started": {
        "ru": "Скачивается модель для стадии «{stage}»: {weights} (~{size}). "
              "Это происходит один раз, дальше она берётся с диска.",
        "en": "Downloading the model for the “{stage}” stage: {weights} (~{size}). "
              "This happens once; after that it is read from disk.",
        "ja": "「{stage}」ステージ用のモデルをダウンロードしています: {weights}"
              "（約 {size}）。これは初回のみで、以降はディスクから読み込まれます。",
    },
    # F225: ...and how far it has got, while it is getting there. The sentence above is
    # printed once; without this one the console then says nothing for the twenty minutes
    # the gigabytes take, which is what the run screen was reported as hanging for.
    "cli.download.progress": {
        "ru": "  …скачано {done} из {size}",
        "en": "  …{done} of {size} so far",
        "ja": "  …{size} 中 {done} まで取得",
    },
    # F229: the caption of the stage's progress bar while the weights arrive. That bar
    # counts FRAMES, and until the model is on disk not one can be processed, so its
    # number sits at zero for the whole download — the right number in the wrong unit,
    # which reads as a hang. No percentage: how much has arrived is the line above.
    "cli.download.waiting": {
        "ru": "жду модель — кадры пойдут, когда она скачается",
        "en": "waiting for the model — frames start once it is downloaded",
        "ja": "モデルを待機中 — ダウンロードが終わるとコマの処理が始まります",
    },
    # The one a person actually meets. What they used to get was
    # `<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>` — no stage, no model, no
    # size, nothing to do about it. The traceback stays in the log.
    # When the loaders were switched offline by US, the hub answers "check your
    # connection" on a machine whose connection is fine. Said after the reason, because
    # the reason is still the library's.
    "cli.download.offline_by_us": {
        "ru": "Загрузка моделей была отключена самой Sorta, потому что в кэше уже есть "
              "другие модели. Чтобы разрешить эту докачку: {variable}=1",
        "en": "Model downloads were switched off by Sorta itself, because the cache "
              "already holds other models. To allow this one: {variable}=1",
        "ja": "キャッシュに他のモデルがあるため、Sorta 自身がモデルのダウンロードを"
              "無効にしていました。今回だけ許可するには: {variable}=1",
    },
    "cli.download.failed": {
        "ru": "Стадия «{stage}» не смогла скачать модель {weights} (~{size}). "
              "Проверьте подключение к сети и запустите прогон снова; модель можно "
              "получить заранее через sorta-setup. Причина: {error}",
        "en": "The “{stage}” stage could not download the model {weights} (~{size}). "
              "Check the network connection and start the run again; the model can also "
              "be fetched in advance with sorta-setup. Reason: {error}",
        "ja": "「{stage}」ステージがモデル {weights}（約 {size}）をダウンロード"
              "できませんでした。ネットワーク接続を確認して再実行してください。"
              "モデルは sorta-setup で事前に取得することもできます。理由: {error}",
    },
    # The stages that can go to the network, named for those two sentences. Only those
    # four: a label nobody prints is a translation nobody checks.
    "cli.stage.landmarks": {
        "ru": "Определение мест по виду",
        "en": "Places recognised by sight",
        "ja": "見た目による場所の判定",
    },
    "cli.stage.classify": {
        "ru": "Классификация кадров",
        "en": "Frame classification",
        "ja": "コマの分類",
    },
    "cli.stage.junk": {
        "ru": "Служебные кадры и качество",
        "en": "Utility frames and quality",
        "ja": "実用目的のコマと品質",
    },
    "cli.stage.faces": {
        "ru": "Разбор по лицам",
        "en": "Face detection",
        "ja": "顔の検出",
    },
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
    # F213: the third case of a hand-made Linux install. F210 creates the cache 0700 and
    # deliberately does not repair a directory that already exists, so one made before
    # that rule stays readable by every other local account — with a decoded copy of the
    # whole collection inside it.
    "cli.doctor.cache_open": {
        "ru": "⚠ Кэш превью читается другими локальными пользователями (права {mode}) — "
              "закрыть: chmod 700 {path}",
        "en": "⚠ The preview cache is readable by other local accounts (mode {mode}) — "
              "close it: chmod 700 {path}",
        "ja": "⚠ プレビューキャッシュは他のローカルユーザーから読めます（権限 {mode}）— "
              "閉じる: chmod 700 {path}",
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
    # F117: printed right after the size, so the number above has something to be
    # measured against. Two texts, because "no ceiling" is a state, not a limit of 0.
    "cli.cache.preview_limit": {
        "ru": "  потолок: {limit_gb:.2f} ГБ (занято {percent:.0f}%)",
        "en": "  ceiling: {limit_gb:.2f} GB ({percent:.0f}% used)",
        "ja": "  上限: {limit_gb:.2f} GB（{percent:.0f}% 使用）",
    },
    "cli.cache.preview_no_limit": {
        "ru": "  потолок: не задан (imaging.preview_cache_max_gb)",
        "en": "  ceiling: none set (imaging.preview_cache_max_gb)",
        "ja": "  上限: 未設定 (imaging.preview_cache_max_gb)",
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
    # F224: the model weights — the gigabytes in caches that carry nobody's name. The
    # list is printed BEFORE the question in every mode, because «frees 1.9 GB» is an
    # answer and «clear the cache» is a riddle.
    "cli.cache.models_header": {
        "ru": "Скачанные модели (по каталогу ярусов):",
        "en": "Downloaded models (from the tier catalog):",
        "ja": "ダウンロード済みモデル（ティアカタログより）:",
    },
    "cli.cache.models_entry": {
        "ru": "  {weight}: {path} — {size_gb:.2f} ГБ",
        "en": "  {weight}: {path} — {size_gb:.2f} GB",
        "ja": "  {weight}: {path} — {size_gb:.2f} GB",
    },
    # A junction/symlink goes away as a link, so the target keeps its bytes and this
    # line says so before anybody agrees to anything.
    "cli.cache.models_link": {
        "ru": "  {weight}: {path} — ссылка; удалится сама ссылка, "
              "то, на что она указывает, останется",
        "en": "  {weight}: {path} — a link; the link itself is removed, "
              "whatever it points at stays",
        "ja": "  {weight}: {path} — リンクです。リンク自体は削除されますが、"
              "リンク先はそのまま残ります",
    },
    "cli.cache.models_behind_link": {
        "ru": "  {weight}: {path} — за ссылкой {link}; это чужое хранилище, "
              "не трогаем",
        "en": "  {weight}: {path} — behind the link {link}; that is somebody else's "
              "store, left alone",
        "ja": "  {weight}: {path} — リンク {link} の先にあります。"
              "他者の保管場所なので手を触れません",
    },
    "cli.cache.models_total": {
        "ru": "Освободится: {size_gb:.2f} ГБ",
        "en": "Would free: {size_gb:.2f} GB",
        "ja": "解放される容量: {size_gb:.2f} GB",
    },
    "cli.cache.models_none": {
        "ru": "Скачанных моделей на диске нет.",
        "en": "No downloaded models on this disk.",
        "ja": "このディスクにダウンロード済みモデルはありません。",
    },
    "cli.cache.models_confirm": {
        "ru": "Удалить эти модели и освободить {size_gb:.2f} ГБ? Фотографии, индекс и "
              "настройки не тронутся; веса скачаются заново при первом же прогоне, "
              "которому они понадобятся",
        "en": "Remove these models and free {size_gb:.2f} GB? Photographs, the index "
              "and the settings are not touched; the weights are downloaded again by "
              "the first run that needs them",
        "ja": "これらのモデルを削除して {size_gb:.2f} GB を解放しますか？ 写真・"
              "インデックス・設定には手を触れません。重みは必要になった最初の実行で"
              "再びダウンロードされます",
    },
    "cli.cache.models_cleared": {
        "ru": "Удалено моделей: {n}; освобождено {size_gb:.2f} ГБ",
        "en": "Models removed: {n}; {size_gb:.2f} GB freed",
        "ja": "削除したモデル: {n} 件、解放した容量 {size_gb:.2f} GB",
    },
    "cli.cache.models_kept": {
        "ru": "Оставлено (за ссылкой {link}): {path}",
        "en": "Left alone (behind the link {link}): {path}",
        "ja": "そのまま残しました（リンク {link} の先）: {path}",
    },
    "cli.cache.models_failed": {
        "ru": "Не удалось удалить {path}: {error}",
        "en": "Could not remove {path}: {error}",
        "ja": "{path} を削除できませんでした: {error}",
    },
    # F118: the plan summary and the warnings around it are printed by sorter.py, which
    # F112 never reached — it localized cli.py and left this file writing Russian
    # whatever `language:` said. The command echo itself (`sort --by city --apply`) is
    # deliberately NOT translated: it is the command the reader would type.
    "cli.sort.plan_counts": {
        "ru": "{files} файлов -> {dirs} каталогов",
        "en": "{files} files -> {dirs} folders",
        "ja": "{files} ファイル -> {dirs} フォルダ",
    },
    "cli.sort.plan_paths": {
        "ru": "; план: {csv}, {html}",
        "en": "; plan: {csv}, {html}",
        "ja": "; プラン: {csv}、{html}",
    },
    "cli.sort.plan_excluded": {
        "ru": "; исключено: {n}",
        "en": "; excluded: {n}",
        "ja": "; 除外: {n}",
    },
    "cli.sort.plan_manual": {
        "ru": "; ручные правки: перенесено {reassigned}, не трогать {excluded}",
        "en": "; manual edits: {reassigned} reassigned, {excluded} left alone",
        "ja": "; 手動修正: 移動 {reassigned}、そのまま {excluded}",
    },
    "cli.sort.warn_delete_dupes": {
        "ru": "ВНИМАНИЕ: --delete-worse-dupes БЕЗВОЗВРАТНО удаляет худшие "
              "почти-дубликаты (не подлежит откату через sorta undo)",
        "en": "WARNING: --delete-worse-dupes deletes the worse near-duplicates "
              "IRREVERSIBLY (sorta undo cannot bring them back)",
        "ja": "警告: --delete-worse-dupes は劣るほうの類似写真を**元に戻せない形で**"
              "削除します（sorta undo では復元できません）",
    },
    "cli.sort.warn_in_place": {
        "ru": "ВНИМАНИЕ: --dest не задан — реструктурируется ИСХОДНОЕ дерево "
              "в {path} (in-place раскладка)",
        "en": "WARNING: no --dest given — the SOURCE tree at {path} is the one being "
              "restructured (an in-place layout)",
        "ja": "警告: --dest が未指定です — {path} の**元のツリー**を組み替えます"
              "（その場での振り分け）",
    },
    "cli.geo.progress": {
        "ru": "geo: {done}/{total} файлов",
        "en": "geo: {done}/{total} files",
        "ja": "geo: {done}/{total} ファイル",
    },
    "cli.ui.serving": {
        "ru": "sorta ui: {url} (Ctrl+C для остановки)",
        "en": "sorta ui: {url} (Ctrl+C to stop)",
        "ja": "sorta ui: {url}（停止は Ctrl+C）",
    },
    # F207: the tray icon (`sorta-tray`) — two menu items, the address in the tooltip and
    # the question between one click and a five-hour run. Three WHOLE sentences and not
    # one with the operation pasted in: «идёт прогон» and «идёт раскладка» decline
    # differently, and glued fragments translate badly in all three languages.
    "cli.tray.open": {
        "ru": "Открыть",
        "en": "Open",
        "ja": "開く",
    },
    "cli.tray.quit": {
        "ru": "Выйти",
        "en": "Quit",
        "ja": "終了",
    },
    "cli.tray.tooltip": {
        "ru": "Sorta — фотоколлекция: {url}",
        "en": "Sorta — photo collection: {url}",
        "ja": "Sorta — 写真コレクション: {url}",
    },
    "cli.tray.serving": {
        "ru": "sorta: {url} (значок в трее: «Открыть» / «Выйти»)",
        "en": "sorta: {url} (tray icon: Open / Quit)",
        "ja": "sorta: {url}（トレイアイコン: 開く / 終了）",
    },
    "cli.tray.no_icon": {
        "ru": "sorta: значка в трее не будет ({reason}); программа работает как обычно",
        "en": "sorta: there will be no tray icon ({reason}); the program runs as usual",
        "ja": "sorta: トレイアイコンは表示されません（{reason}）。"
              "プログラムは通常どおり動作します",
    },
    "cli.tray.already_running": {
        "ru": "sorta уже запущена — открываю {url}",
        "en": "sorta is already running — opening {url}",
        "ja": "sorta は既に起動しています — {url} を開きます",
    },
    "cli.tray.port_busy": {
        "ru": "sorta: порт {port} занят другой программой — "
              "запустите с другим портом: --port",
        "en": "sorta: port {port} is held by another program — start it on a different "
              "one: --port",
        "ja": "sorta: ポート {port} は別のプログラムが使用しています — "
              "別のポートで起動してください: --port",
    },
    "cli.tray.quit_title": {
        "ru": "Выход из Sorta",
        "en": "Quit Sorta",
        "ja": "Sorta を終了",
    },
    "cli.tray.quit_process": {
        "ru": "Идёт прогон — он может считать часами. Прервать его и выйти?",
        "en": "A run is going — it can count for hours. Interrupt it and quit?",
        "ja": "処理が実行中です（数時間かかることがあります）。中断して終了しますか？",
    },
    "cli.tray.quit_sort": {
        "ru": "Идёт раскладка файлов. Прервать её и выйти?",
        "en": "A layout of the files is going. Interrupt it and quit?",
        "ja": "ファイルの振り分けが実行中です。中断して終了しますか？",
    },
    "cli.tray.quit_undo": {
        "ru": "Идёт откат перемещений. Прервать его и выйти?",
        "en": "A rollback of the moves is going. Interrupt it and quit?",
        "ja": "移動の取り消しが実行中です。中断して終了しますか？",
    },
    "cli.tray.quit_running": {
        "ru": "Сейчас идёт длительная операция. Прервать её и выйти?",
        "en": "A long operation is going. Interrupt it and quit?",
        "ja": "長時間の処理が実行中です。中断して終了しますか？",
    },
    "cli.tray.quit_failed": {
        "ru": "sorta: не удалось завершить программу (ответ {status})",
        "en": "sorta: could not close the program (answer {status})",
        "ja": "sorta: プログラムを終了できませんでした（応答 {status}）",
    },
    # F227: the only line of the window that goes up while the program is still starting.
    # It answers "did it hear me?" after a double-click that did nothing, without
    # promising a duration nobody has measured on that disk.
    "cli.tray.starting": {
        "ru": "Запускается — первый запуск дольше остальных…",
        "en": "Starting — the first launch takes longer than the rest…",
        "ja": "起動中です — 初回は時間がかかります…",
    },
    # F211: the first-run wizard (`sorta-setup`) — the one place the tiers are described
    # in words: what each costs, what it buys, and what stays unavailable after a no.
    # Refusing is a normal answer, so every "without" line has to be a fact about a
    # working program rather than a warning about a broken one.
    "cli.setup.title": {
        "ru": "Sorta — первая настройка",
        "en": "Sorta — first-run setup",
        "ja": "Sorta — 初回セットアップ",
    },
    "cli.setup.checking": {
        "ru": "Что уже есть на этой машине (sorta doctor):",
        "en": "What is already on this machine (sorta doctor):",
        "ja": "このマシンに既にあるもの（sorta doctor）:",
    },
    "cli.setup.exiftool_bundled": {
        "ru": "exiftool идёт в комплекте — даты, GPS и ориентация читаются из HEIC, RAW "
              "и видео сразу.",
        "en": "exiftool ships with the program — dates, GPS and orientation are read "
              "from HEIC, RAW and video right away.",
        "ja": "exiftool は同梱です — HEIC・RAW・動画から日付、GPS、向きをそのまま"
              "読み取ります。",
    },
    "cli.setup.exiftool_on_path": {
        "ru": "exiftool найден в системе — HEIC, RAW и видео читаются полностью.",
        "en": "exiftool was found on this machine — HEIC, RAW and video are read in full.",
        "ja": "exiftool がこのマシンで見つかりました — HEIC・RAW・動画を完全に"
              "読み取れます。",
    },
    # F230 took the `winget` command out of this line: the wizard also runs on Linux and
    # in a checkout, while `sorta doctor` — printed a few lines ABOVE it in the same
    # window — already names the command for the machine it is running on.
    "cli.setup.exiftool_absent": {
        "ru": "exiftool не найден: метаданные читает Pillow, а это только "
              "JPEG/PNG/TIFF/WEBP. У HEIC, RAW и видео не будет ни даты, ни GPS. "
              "Поставьте exiftool (команда для этой машины — в строке выше) и запустите "
              "настройку снова.",
        "en": "exiftool was not found: metadata is read by Pillow, which means "
              "JPEG/PNG/TIFF/WEBP only. HEIC, RAW and video will have neither a date nor "
              "GPS. Install exiftool (the command for this machine is in the line above) "
              "and run the setup again.",
        "ja": "exiftool が見つかりません: メタデータは Pillow が読み取るため、"
              "JPEG/PNG/TIFF/WEBP のみになります。HEIC・RAW・動画は日付も GPS も"
              "得られません。exiftool を入れてから（このマシン向けのコマンドは上の行に"
              "あります）セットアップをやり直してください。",
    },
    "cli.setup.base_ready": {
        "ru": "Базовый ярус уже установлен и работает без сети: индекс, EXIF, гео, "
              "дубликаты, раскладка по городам.",
        "en": "The base tier is installed and works with no network: index, EXIF, geo, "
              "duplicates, sorting by city.",
        "ja": "基本ティアはインストール済みで、ネットワークなしで動作します: "
              "インデックス、EXIF、位置情報、重複、都市別の振り分け。",
    },
    "cli.setup.tiers_intro": {
        "ru": "Дальше — необязательные ярусы. Отказ здесь нормальный ответ: то, что уже "
              "стоит, — работающая программа, а не обрубок.",
        "en": "The tiers below are optional. Refusing is a normal answer: what is already "
              "installed is a working program, not a stub.",
        "ja": "以下のティアは任意です。断るのも普通の答えです: 既にあるものだけで、"
              "削られた製品ではなく動く製品になります。",
    },
    "cli.setup.offer": {
        "ru": "{name}: {size} загрузки. {benefit}",
        "en": "{name}: {size} to download. {benefit}",
        "ja": "{name}: ダウンロード {size}。{benefit}",
    },
    "cli.setup.question": {
        "ru": "Доустановить? [y/N]",
        "en": "Add it? [y/N]",
        "ja": "追加しますか？ [y/N]",
    },
    # F223: the same question for the one tier whose default answer is yes — the tier the
    # layout itself needs. The brackets say which way Enter goes, and a person who does
    # not want it types the letter; the refusal is a normal answer here too, and the
    # `without` line of that tier states what it really costs rather than warning.
    "cli.setup.question_yes": {
        "ru": "Доустановить? [Y/n]",
        "en": "Add it? [Y/n]",
        "ja": "追加しますか？ [Y/n]",
    },
    # F223: one tier needing another, said out loud. The first dependency in this catalog:
    # search by words encodes the pictures with ViT-L-14 and the words with XLM-RoBERTa,
    # so half of what it needs lives in the tier above it.
    "cli.setup.requires": {
        "ru": "Ярус «{name}» работает только вместе с ярусом «{required}» — добавляю и его.",
        "en": "“{name}” does not work without “{required}”, so that one is added too.",
        "ja": "「{name}」は「{required}」なしでは動作しないため、そちらも追加します。",
    },
    "cli.setup.in_place": {
        "ru": "{name}: уже на месте — качать нечего.",
        "en": "{name}: already in place — nothing to download.",
        "ja": "{name}: 既に配置済みです — ダウンロードするものはありません。",
    },
    "cli.setup.already": {
        "ru": "Уже было на месте: {names}",
        "en": "Already in place: {names}",
        "ja": "既に配置済み: {names}",
    },
    # F225: the one download that is measured in half-hours. Said BEFORE it starts, and
    # separately from the line below, because a window that goes quiet for thirty minutes
    # is a window somebody closes — the deep tier is 7.0 GB against 1.6 for the next
    # biggest, and the difference is what a person has to be able to plan around.
    "cli.setup.weights_slow": {
        "ru": "Это большая загрузка ({size}) — она займёт десятки минут. Окно не зависло: "
              "ниже будет видно, сколько уже скачано.",
        "en": "This is a large download ({size}) — it takes tens of minutes. The window "
              "is not stuck: the lines below say how much has arrived.",
        "ja": "これは大きなダウンロードです（{size}）— 数十分かかります。ウィンドウは"
              "固まっていません: 以下に取得済みの量が表示されます。",
    },
    # F223: the weights of a tier somebody said yes to are fetched HERE, at the screen, so
    # that the gigabytes do not arrive in the middle of a run looking like a hang. The
    # progress line is not decoration: it is the difference between a download and a
    # window that has stopped answering.
    "cli.setup.weights_downloading": {
        "ru": "Скачиваю {weights} ({size}). Это один раз: дальше стадии берут их с диска.",
        "en": "Downloading {weights} ({size}). Once only: after this the stages read them "
              "off the disk.",
        "ja": "{weights}（{size}）をダウンロードしています。一度だけで、以降はステージが"
              "ディスクから読み込みます。",
    },
    "cli.setup.weights_progress": {
        "ru": "  …скачано {done} из {size}",
        "en": "  …{done} of {size} so far",
        "ja": "  …{size} 中 {done} まで",
    },
    "cli.setup.weights_ready": {
        "ru": "{weights} на месте ({size}) — в сеть за ними больше никто не полезет.",
        "en": "{weights} is in place ({size}) — nothing will go to the network for it "
              "again.",
        "ja": "{weights} を配置しました（{size}）— これ以降ネットワークへ取りに行くことは"
              "ありません。",
    },
    "cli.setup.weights_failed": {
        "ru": "Не удалось скачать {weights}: {error}. Установка цела и программа работает; "
              "стадия, которой они нужны, скачает их при первом прогоне и скажет об этом.",
        "en": "Could not download {weights}: {error}. The install is intact and the "
              "program works; the stage that needs them fetches them on its first run and "
              "says so.",
        "ja": "{weights} をダウンロードできませんでした: {error}。インストールは無傷で"
              "プログラムは動作します。必要とするステージが初回実行時に取得し、その旨を"
              "伝えます。",
    },
    # F223: the last line of a window that Windows is about to destroy. Without it the
    # summary — and, worse, any error — vanished in the same instant it was printed.
    "cli.setup.press_enter": {
        "ru": "Нажмите Enter, чтобы закрыть это окно.",
        "en": "Press Enter to close this window.",
        "ja": "Enter を押すとこのウィンドウを閉じます。",
    },
    "cli.setup.refused": {
        "ru": "Пропущено. {without}",
        "en": "Skipped. {without}",
        "ja": "スキップしました。{without}",
    },
    "cli.setup.installing": {
        "ru": "Устанавливаю {name}: {packages}",
        "en": "Installing {name}: {packages}",
        "ja": "{name} をインストールしています: {packages}",
    },
    "cli.setup.install_failed": {
        "ru": "Не удалось установить {name} (код {status}). Программа работает как "
              "раньше; запустите настройку ещё раз, чтобы повторить.",
        "en": "Could not install {name} (exit code {status}). The program works as "
              "before; run the setup again to retry.",
        "ja": "{name} をインストールできませんでした（終了コード {status}）。"
              "プログラムはこれまでどおり動作します。やり直すにはセットアップを"
              "再実行してください。",
    },
    "cli.setup.no_metadata": {
        "ru": "Не удалось узнать состав яруса {name} — нет метаданных пакета. Ярус не "
              "добавлен, остальное работает.",
        "en": "Could not read what the {name} tier consists of — the package metadata is "
              "missing. The tier was not added; everything else works.",
        "ja": "{name} ティアの構成を読み取れませんでした（パッケージのメタデータが"
              "ありません）。このティアは追加していませんが、他は動作します。",
    },
    # F225 removed `cli.setup.weights_later` ("the weights download on the first run of
    # the stage"): three tiers of four printed it instead of downloading anything. A tier
    # said yes to is fetched at the screen now, so the line describes nothing.
    "cli.setup.added": {
        "ru": "Выбрано: {names}",
        "en": "Chosen: {names}",
        "ja": "選択: {names}",
    },
    "cli.setup.skipped": {
        "ru": "Пропущено: {names}",
        "en": "Skipped: {names}",
        "ja": "スキップ: {names}",
    },
    "cli.setup.works_anyway": {
        "ru": "Ничего не выбрано — и это нормально: базовый ярус разбирает коллекцию "
              "по городам, находит дубликаты и читает даты.",
        "en": "Nothing was chosen, and that is fine: the base tier sorts the collection "
              "by city, finds duplicates and reads dates.",
        "ja": "何も選んでいませんが、それで問題ありません: 基本ティアだけで都市別の"
              "振り分け、重複の検出、日付の読み取りができます。",
    },
    # F230: three keys, because the wizard is run from all three installs and only one of
    # them has a Start menu. The line used to name it to everybody, including the developer
    # who started `uv run sorta-setup` in a checkout.
    "cli.setup.rerun.installed": {
        "ru": "Любой ярус можно доустановить позже: пункт «Sorta setup» в меню «Пуск» "
              "(или команда sorta-setup).",
        "en": "Any tier can be added later: the Sorta setup item of the Start menu (or "
              "the sorta-setup command).",
        "ja": "どのティアも後から追加できます: スタートメニューの「Sorta setup」"
              "（またはコマンド sorta-setup）。",
    },
    "cli.setup.rerun.tool": {
        "ru": "Любой ярус можно доустановить позже: запустите sorta-setup ещё раз.",
        "en": "Any tier can be added later: run sorta-setup again.",
        "ja": "どのティアも後から追加できます: sorta-setup をもう一度実行してください。",
    },
    "cli.setup.rerun.checkout": {
        "ru": "Это чекаут исходников: ярусы с пакетами ставятся через "
              "uv sync --extra <ярус>, а модели можно догрузить этим мастером.",
        "en": "This is a checkout of the sources: tiers that install packages come from "
              "uv sync --extra <tier>, and the model weights can be fetched by this "
              "wizard.",
        "ja": "これはソースのチェックアウトです: パッケージを含むティアは "
              "uv sync --extra <ティア> で入れ、モデルはこのウィザードで取得できます。",
    },
    "cli.setup.doctor_hint": {
        "ru": "`sorta doctor` показывает, чем установка оказалась на самом деле.",
        "en": "`sorta doctor` prints what the install actually became.",
        "ja": "`sorta doctor` は実際のインストール内容を表示します。",
    },
    # --- F230: the wizard asks about the acceleration tier KNOWING what card is in the
    # machine, so the three answers below are three different sentences — a card that
    # fits, a driver too old to load CUDA 13, and no card at all.
    "cli.setup.card_found": {
        "ru": "Видеокарта: {name} (драйвер {driver}) — ускорение на ней заработает.",
        "en": "Graphics card: {name} (driver {driver}) — acceleration will work on it.",
        "ja": "グラフィックカード: {name}（ドライバ {driver}）— "
              "アクセラレーションが利用できます。",
    },
    # The driver, said as part of the question rather than as a footnote. CUDA 13 wheels
    # do not import on an older driver at all, so 2.5 GB downloaded here would buy a
    # traceback; what is needed is a driver update, and that is what the line asks for.
    "cli.setup.card_driver_old": {
        "ru": "Видеокарта {name} есть, но драйвер {driver} старее нужного: для CUDA "
              "{cuda} нужен драйвер {needed} или новее. Ярус ускорения не предлагается — "
              "обновите драйвер NVIDIA и запустите настройку снова.",
        "en": "The card {name} is here, but driver {driver} is older than required: CUDA "
              "{cuda} needs driver {needed} or newer. The acceleration tier is not "
              "offered — update the NVIDIA driver and run the setup again.",
        "ja": "カード {name} はありますが、ドライバ {driver} は要件より古いです: CUDA "
              "{cuda} にはドライバ {needed} 以降が必要です。アクセラレーションティアは"
              "提供しません — NVIDIA ドライバを更新してからセットアップをやり直して"
              "ください。",
    },
    "cli.setup.card_absent": {
        "ru": "Видеокарта NVIDIA не найдена (nvidia-smi не ответил), поэтому ярус "
              "ускорения не предлагается: эти {size} на этой машине работать не будут. "
              "Всё считается на процессоре, и это рабочий вариант.",
        "en": "No NVIDIA card was found (nvidia-smi did not answer), so the acceleration "
              "tier is not offered: those {size} would not work on this machine. "
              "Everything is computed on the processor, which is a working setup.",
        "ja": "NVIDIA カードが見つかりませんでした（nvidia-smi が応答しません）。"
              "そのため、アクセラレーションティアは提供しません: この {size} は"
              "このマシンでは動作しません。すべて CPU で計算され、それでも動作します。",
    },
    # The price of a no, in the unit a person plans in. Megabytes are what the tier costs;
    # hours are what refusing it costs, and only one of the two was ever said.
    "cli.setup.card_refusal_cost": {
        "ru": "Цена отказа — время: без этого яруса модельные стадии считает процессор, "
              "и на коллекции в десятки тысяч кадров это часы вместо минут.",
        "en": "Refusing costs time: without this tier the model stages run on the "
              "processor, and on a collection of tens of thousands of frames that is "
              "hours instead of minutes.",
        "ja": "断ると時間がかかります: このティアがないとモデルのステージは CPU で"
              "実行され、数万枚のコレクションでは数分ではなく数時間になります。",
    },
    # F230: the profile was REPLACED, not added to (the only tier with `reinstall`), and
    # `onnxruntime` and `onnxruntime-gpu` unpack into the same directory (F76) — so which
    # profile actually won is a question only `sorta doctor` can answer, and the wizard
    # has to send the person there rather than declare success itself.
    "cli.setup.profile_changed": {
        "ru": "Профиль установки заменён на CUDA-сборки. Что получилось на самом деле, "
              "печатает `sorta doctor` — строка «install profile».",
        "en": "The install profile was replaced with the CUDA builds. What it actually "
              "became is printed by `sorta doctor` — the “install profile” line.",
        "ja": "インストールプロファイルを CUDA ビルドに置き換えました。実際の結果は "
              "`sorta doctor` の「install profile」の行に表示されます。",
    },
    # The way back, named. The acceleration tier is installed with `--reinstall` and takes
    # the working CPU profile with it; until F230 there was nothing in the catalog to
    # return to it, so a machine where the CUDA stack misbehaved had no way out but a
    # reinstall of the whole program.
    "cli.setup.cpu_back.installed": {
        "ru": "Вернуться на процессорный профиль можно в любой момент: "
              "sorta-setup --restore-cpu.",
        "en": "Going back to the CPU profile is possible at any time: "
              "sorta-setup --restore-cpu.",
        "ja": "いつでも CPU プロファイルに戻せます: sorta-setup --restore-cpu。",
    },
    # Word for word the same on a tool install: it is one command and it works there.
    "cli.setup.cpu_back.tool": {
        "ru": "Вернуться на процессорный профиль можно в любой момент: "
              "sorta-setup --restore-cpu.",
        "en": "Going back to the CPU profile is possible at any time: "
              "sorta-setup --restore-cpu.",
        "ja": "いつでも CPU プロファイルに戻せます: sorta-setup --restore-cpu。",
    },
    # ...and NOT the same in a checkout: there the profile is an extra of the project, and
    # `uv pip install` into a synced environment is undone by the next `uv sync`.
    "cli.setup.cpu_back.checkout": {
        "ru": "Вернуться на процессорный профиль в чекауте: "
              "uv sync --extra cpu --extra dev.",
        "en": "Going back to the CPU profile in a checkout: "
              "uv sync --extra cpu --extra dev.",
        "ja": "チェックアウトで CPU プロファイルに戻す: "
              "uv sync --extra cpu --extra dev。",
    },
    "cli.setup.restoring_cpu": {
        "ru": "Возвращаю процессорный профиль: {packages}",
        "en": "Restoring the CPU profile: {packages}",
        "ja": "CPU プロファイルを復元しています: {packages}",
    },
    "cli.setup.restored_cpu": {
        "ru": "Процессорный профиль вернулся. Проверьте `sorta doctor` — строка "
              "«install profile» скажет, какой профиль победил.",
        "en": "The CPU profile is back. Check `sorta doctor` — the “install profile” line "
              "says which profile won.",
        "ja": "CPU プロファイルに戻りました。`sorta doctor` の「install profile」の行で"
              "どちらのプロファイルになったか確認してください。",
    },
    "cli.setup.restore_cpu_failed": {
        "ru": "Не удалось вернуть процессорный профиль (код {status}). Установка осталась "
              "как была; попробуйте ещё раз или переустановите программу.",
        "en": "Could not restore the CPU profile (exit code {status}). The install is as "
              "it was; try again or reinstall the program.",
        "ja": "CPU プロファイルを復元できませんでした（終了コード {status}）。"
              "インストールはそのままです。もう一度試すか、再インストールしてください。",
    },
    "cli.setup.tier.cpu_profile.name": {
        "ru": "Процессорный профиль",
        "en": "CPU profile",
        "ja": "CPU プロファイル",
    },
    "cli.setup.size_mb": {
        "ru": "{mb} МБ",
        "en": "{mb} MB",
        "ja": "{mb} MB",
    },
    "cli.setup.size_gb": {
        "ru": "{gb} ГБ",
        "en": "{gb} GB",
        "ja": "{gb} GB",
    },
    "cli.setup.tier.base.name": {
        "ru": "Базовый ярус",
        "en": "Base tier",
        "ja": "基本ティア",
    },
    "cli.setup.tier.base.benefit": {
        "ru": "Индекс, EXIF, гео, дубликаты, раскладка по городам — без сети.",
        "en": "Index, EXIF, geo, duplicates, sorting by city — with no network.",
        "ja": "インデックス、EXIF、位置情報、重複、都市別の振り分け — ネットワーク不要。",
    },
    # F223: the tier named by what it DOES. These weights used to sit inside "Search by
    # words", where a person who did not want to search by words switched off the
    # classification without a word being said about it — and their run then fetched the
    # same 1.6 GB by itself, mid-run, on a stage they were never asked about.
    "cli.setup.tier.vision.name": {
        "ru": "Узнавание содержимого кадров",
        "en": "Recognising what is in a frame",
        "ja": "コマの内容の認識",
    },
    "cli.setup.tier.vision.benefit": {
        "ru": "Нужно раскладке: отделяет фотографии от скриншотов, документов и товарных "
              "карточек — иначе они поедут в папки городов вперемешку. Те же веса нужны "
              "стадии «Узнавать места по виду» и поиску словами. Скачивается сейчас, "
              "при вас.",
        "en": "The layout needs it: it tells photographs from screenshots, documents and "
              "product shots — without it they ride into the city folders among the "
              "pictures. The same weights are what the “Recognise places by sight” stage "
              "and search by words need. Downloaded now, while you are here.",
        "ja": "振り分けに必要です: 写真をスクリーンショット・書類・商品画像と区別します。"
              "これがないと、それらが写真に混ざったまま都市フォルダへ入ります。同じ重みは"
              "「見た目で場所を判定」ステージと言葉による検索にも使われます。ダウンロードは"
              "いま、この画面で行います。",
    },
    "cli.setup.tier.vision.without": {
        "ru": "Вердикты прогону нужны в любом случае: те же 1.6 ГБ скачает стадия "
              "классификации при первом запуске — посреди прогона и без этого экрана.",
        "en": "The verdicts are needed by a run in any case: the same 1.6 GB is fetched "
              "by the classification stage on its first run — in the middle of a run and "
              "away from this screen.",
        "ja": "判定は実行にどのみち必要です: 同じ 1.6 GB を分類ステージが初回実行時に"
              "取得します — 実行の途中、この画面のないところで。",
    },
    "cli.setup.tier.faces.name": {
        "ru": "Лица",
        "en": "Faces",
        "ja": "顔",
    },
    "cli.setup.tier.faces.benefit": {
        "ru": "Сортировка по людям: поиск лиц, кластеры, альбом на человека.",
        "en": "Sorting by people: face detection, clusters, an album per person.",
        "ja": "人物別の振り分け: 顔検出、クラスタ、人ごとのアルバム。",
    },
    "cli.setup.tier.faces.without": {
        "ru": "Разбор по людям останется недоступен; всё остальное работает.",
        "en": "Sorting by people stays unavailable; everything else works.",
        "ja": "人物別の振り分けは使えませんが、他はすべて動作します。",
    },
    "cli.setup.tier.search.name": {
        "ru": "Поиск словами",
        "en": "Search by words",
        "ja": "言葉による検索",
    },
    # F223: what is left of this tier once the model everybody needs was taken out of it —
    # the multilingual text tower, and nothing else. It says which tier it stands on,
    # because half of a search by words is done by the other one: the pictures are encoded
    # by ViT-L-14 and only the words by these weights.
    "cli.setup.tier.search.benefit": {
        "ru": "Запрос словами по коллекции и закреплённые срезы — на трёх языках. Стоит "
              "на ярусе «Узнавание содержимого кадров»: картинки кодирует он, слова — "
              "эти веса.",
        "en": "Ask the collection in words, and pin a query as a slice — in three "
              "languages. It stands on the “Recognising what is in a frame” tier: that "
              "one encodes the pictures, these weights encode the words.",
        "ja": "コレクションを言葉で検索し、クエリをスライスとして固定できます（3言語）。"
              "「コマの内容の認識」ティアの上に成り立ちます: 画像はそちらが、言葉はこの"
              "重みが符号化します。",
    },
    "cli.setup.tier.search.without": {
        "ru": "Поиск словами скажет, что индекса нет, вместо выдачи. Срезы по лицам и "
              "классам, вердикты и раскладка по городам работают как работали.",
        "en": "Search by words says it has no index instead of returning a ranking. The "
              "face and class slices, the verdicts and the sorting by city work exactly "
              "as before.",
        "ja": "言葉による検索は結果の代わりに索引がないと伝えます。顔やクラスのスライス、"
              "判定、都市別の振り分けはこれまでどおり動作します。",
    },
    "cli.setup.tier.gpu.name": {
        "ru": "Ускорение на NVIDIA (CUDA 13)",
        "en": "NVIDIA acceleration (CUDA 13)",
        "ja": "NVIDIA アクセラレーション（CUDA 13）",
    },
    "cli.setup.tier.gpu.benefit": {
        "ru": "Меняет процессорный профиль на CUDA-сборки torch и onnxruntime — нужна "
              "видеокарта NVIDIA с драйвером CUDA 13.",
        "en": "Replaces the CPU profile with the CUDA builds of torch and onnxruntime — "
              "needs an NVIDIA card with a CUDA 13 driver.",
        "ja": "CPU プロファイルを torch と onnxruntime の CUDA ビルドに置き換えます — "
              "CUDA 13 ドライバの NVIDIA カードが必要です。",
    },
    "cli.setup.tier.gpu.without": {
        "ru": "Стадии с моделями останутся на процессоре: они работают, но на большой "
              "коллекции это часы.",
        "en": "The model stages stay on the processor: they work, but on a large "
              "collection that means hours.",
        "ja": "モデルを使うステージは CPU のままです: 動作はしますが、大きな"
              "コレクションでは数時間かかります。",
    },
    "cli.setup.tier.deep.name": {
        "ru": "Глубокий ярус (VLM)",
        "en": "Deep tier (VLM)",
        "ja": "ディープティア（VLM）",
    },
    "cli.setup.tier.deep.benefit": {
        "ru": "Находит товары и спасает экранное; прогон становится вчетверо длиннее "
              "(67 минут против 272) и нужна карта на 24 ГБ.",
        "en": "Finds products and rescues on-screen frames; a run takes four times as "
              "long (67 minutes against 272) and it wants a 24 GB card.",
        "ja": "商品を見つけ、画面写りのコマを救います。実行時間は約4倍（67分に対し"
              "272分）になり、24 GB のカードが必要です。",
    },
    "cli.setup.tier.deep.without": {
        "ru": "Класс «товар» просто не появится, папка товаров останется пустой; "
              "остальные классы считает быстрый ярус.",
        "en": "The product class simply never appears and the products folder stays "
              "empty; the other classes come from the fast tier.",
        "ja": "「商品」クラスは現れず、商品フォルダは空のままです。他のクラスは"
              "高速ティアが判定します。",
    },
    "cli.album.empty": {
        "ru": "срез пуст, ничего не выгружено",
        "en": "the slice is empty, nothing exported",
        "ja": "対象が空のため、何も出力していません",
    },
    "cli.album.plan_counts": {
        "ru": "{files} файлов -> {dest}",
        "en": "{files} files -> {dest}",
        "ja": "{files} ファイル -> {dest}",
    },
    "cli.album.warn_move": {
        "ru": "ВНИМАНИЕ: --move изымает файлы альбома из общего пула сортировки "
              "(канон города/другие альбомы больше не увидят эти файлы)",
        "en": "WARNING: --move takes the album's files OUT of the shared pool (the city "
              "canon and any other album will no longer see them)",
        "ja": "警告: --move はアルバムのファイルを共有プールから**取り出します**"
              "（都市の正規構成や他のアルバムからは見えなくなります）",
    },
    "cli.album.warn_blocked_multi": {
        "ru": "ВНИМАНИЕ: {n} файл(ов) с 2+ названными людьми на кадре — move для них "
              "заблокирован (неясно, чей это альбом), используйте --link/--copy: "
              "{names}{more}",
        "en": "WARNING: {n} file(s) hold 2+ named people, so move is blocked for them "
              "(whose album would they go to?) — use --link/--copy: {names}{more}",
        "ja": "警告: 名前の付いた人物が 2 人以上写るファイルが {n} 件あり、move は"
              "ブロックされます（どちらのアルバムか決められません）。--link/--copy を"
              "使ってください: {names}{more}",
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
    # F127: only `animal` has nothing to select inside it. For a person and an event the
    # selector IS the subject, so its absence is an error and never "the whole slice".
    # F129: and for a query it is the words themselves, which makes it the subject too.
    # F139: the class and quality slices join `animal` — the collection holds exactly one
    # products bucket and exactly one blurred list, so there is nothing to select there.
    "cli.album.selector_required": {
        "ru": "альбому person/event/query нужен селектор: имя человека, имя или id "
              "события либо слова запроса. Срезы animal, product, screenshot, meme, "
              "blurred, eyes_closed и low_resolution собираются без селектора",
        "en": "a person/event/query album needs a selector: a person's name, an event's "
              "name or id, or the words to search for. The animal, product, screenshot, "
              "meme, blurred, eyes_closed and low_resolution slices are gathered "
              "without one",
        "ja": "person/event/query のアルバムにはセレクタが必要です: 人物の名前、"
              "イベントの名前か id、または検索する語。animal・product・screenshot・"
              "meme・blurred・eyes_closed・low_resolution のスライスはセレクタなしで"
              "作成できます",
    },
    # F129: search by words — the engine's two refusals and the sentence that closes a
    # result list. The hit lines themselves carry no words in any language (a rank, a
    # score, a path) and are printed directly, like the landmark names of a summary.
    "cli.search.empty_query": {
        "ru": 'пустой запрос: напишите слово — `sorta search "торт"`',
        "en": 'empty query: type a word — `sorta search "cake"`',
        "ja": '問い合わせが空です: 語を入力してください — `sorta search "ケーキ"`',
    },
    # F141: the key named here is `features.search_index` and not `store_embeddings`.
    # Search reads the multilingual index now, and a message that names the other toggle
    # sends a person to switch on something that will not make this sentence go away.
    "cli.search.no_embeddings": {
        "ru": "поисковый индекс не посчитан — запустите `sorta junk` "
              "(нужен features.search_index: true). Пустая выдача здесь читалась бы "
              "как «ничего не нашлось», а это другое",
        "en": "the search index has not been computed — run `sorta junk` (with "
              "features.search_index: true). An empty list here would read as "
              "“nothing matched”, which is a different thing",
        "ja": "検索インデックスがまだ計算されていません — `sorta junk` を実行してください"
              "（features.search_index: true が必要）。ここで空の一覧を返すと"
              "「該当なし」と読めてしまいますが、それとは別のことです",
    },
    "cli.search.other_model": {
        "ru": "в индексе {n} эмбеддингов, и все посчитаны другой моделью — сейчас "
              "настроена {model}. Векторы разных моделей несравнимы: запустите "
              "`sorta junk` заново, он их пересчитает",
        "en": "the index holds {n} embeddings and every one of them was computed by "
              "another model — the configured one is {model}. Vectors of different "
              "models are not comparable: run `sorta junk` again and it recomputes them",
        "ja": "インデックスには {n} 件の埋め込みがありますが、すべて別のモデルで"
              "計算されたものです（現在の設定は {model}）。異なるモデルのベクトルは"
              "比較できません。`sorta junk` を再実行すると再計算されます",
    },
    "cli.search.done": {
        "ru": "Найдено: {n} кадров по запросу «{query}». Это ранжирование, а не фильтр: "
              "порога «точно оно» здесь нет — смотрите, где оценки перестают быть "
              "про ваш запрос",
        "en": "Found: {n} frames for “{query}”. This is a ranking, not a filter: there "
              "is no “definitely it” threshold — look at where the scores stop being "
              "about your words",
        "ja": "「{query}」で {n} 件。これはフィルタではなくランキングです: "
              "「確実にそれ」という閾値はありません — スコアが問い合わせから"
              "離れる位置を見てください",
    },
    # F189: the OTHER kind of answer this command can give. A name is an exact selection,
    # so the sentence under it must not read like the one above: no threshold to look for,
    # no "where the scores stop" — the count is a count of somebody's photographs.
    "cli.search.person_done": {
        "ru": "Кадры человека: {name} — {n}. Это точный отбор по кластеру лиц, а не "
              "ранжирование: кадр либо в кластере этого человека, либо нет",
        "en": "Frames of a person: {name} — {n}. This is an exact selection by face "
              "cluster, not a ranking: a frame is either in this person's cluster or not",
        "ja": "人物のコマ: {name} — {n} 件。これは顔クラスタによる正確な抽出であり、"
              "ランキングではありません: そのコマがこの人物のクラスタに入っているかどうかです",
    },
    # A name can also be an ordinary word («Роза», «Марк»), and the other answer must not
    # disappear silently — the line says how to ask for it.
    "cli.search.person_words_hint": {
        "ru": "Это же слово можно поискать по картинке: "
              '`sorta search --words "{query}"`',
        "en": "The same word can be searched for as an image: "
              '`sorta search --words "{query}"`',
        "ja": "同じ語を画像として検索することもできます: "
              '`sorta search --words "{query}"`',
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
              "ошибок {failed}, убрано пустых каталогов {dirs}",
        "en": "Undo of batch {batch}: {undone} restored, {missing} missing, "
              "{failed} errors, {dirs} empty folders removed",
        "ja": "バッチ {batch} の取り消し: 復元 {undone}、欠落 {missing}、エラー {failed}、"
              "空フォルダー削除 {dirs}",
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
    # F165: the front half of the junk stage, run before faces — the verdicts alone.
    "cli.progress.classify": {
        "ru": "classify: вердикты", "en": "classify: verdicts", "ja": "classify: 判定",
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
    # F205: the two other passes that ask the same model. They reported under the caption
    # above until their measured price turned out to be a third of it, and a phase that is
    # named is a phase the run log can price. Each says WHICH question is being asked.
    "cli.phase.junk_pets_vlm": {
        "ru": "junk: проверка животных (VLM)", "en": "junk: animal check (VLM)",
        "ja": "junk: 動物の確認（VLM）",
    },
    "cli.phase.junk_rescue_vlm": {
        "ru": "junk: поиск экранного (VLM)",
        "en": "junk: screen-capture check (VLM)",
        "ja": "junk: 画面コマの確認（VLM）",
    },
    "cli.phase.junk_write": {
        "ru": "junk: запись вердиктов", "en": "junk: writing verdicts",
        "ja": "junk: 判定の書き込み",
    },

    # --- F114: the texts `--help` prints -------------------------------------
    # The `ru` variants are the strings that used to sit in the `typer.Option(...,
    # help=...)` decorators and the command docstrings of cli.py, word for word. They
    # could not be localized there: a decorator runs at import, long before anything has
    # read the config, which is why `cli.build_app` assembles the interface from these
    # keys once the language is known.
    #
    # The line breaks inside the multi-paragraph texts are meaningful — Typer keeps them
    # and only wraps what is too long — so the `ru` ones repeat the breaks they came with.
    "cli.help.app": {
        "ru": "Sorta v{version} — сортировка фотоколлекции",
        "en": "Sorta v{version} — sorting a photo collection",
        "ja": "Sorta v{version} — 写真コレクションの整理",
    },
    "cli.help.opt.config": {
        "ru": "Путь к config.yaml",
        "en": "Path to config.yaml",
        "ja": "config.yaml へのパス",
    },
    # F127: the frame-quality flag, offered by `junk` and by `run` — one override, one
    # text. It says which config key it replaces: a flag that only overrides for one
    # run is worth nothing if the reader cannot find the permanent setting behind it.
    # F186 retired the other two of the set with the question they overrode.
    "cli.help.opt.pets": {
        "ru": "Искать животных на этот прогон (features.pets): три дополнительных "
              "запроса внутри CLIP-вызова стадии junk, отдельной стадии не "
              "появляется; без флага — как в config.yaml",
        "en": "Look for animals on this run (features.pets): three extra prompts "
              "inside the CLIP call the junk stage makes anyway, not a stage of its "
              "own; without the flag — as in config.yaml",
        "ja": "この実行で動物を検出します (features.pets): junk ステージが行う "
              "CLIP 呼び出しに 3 つのプロンプトを追加するだけで、独立したステージには"
              "なりません。フラグなしの場合は config.yaml のとおり",
    },
    # index
    "cli.help.index": {
        "ru": "Сканировать источники, извлечь метаданные, пометить дубликаты.",
        "en": "Scan the sources, extract the metadata, mark the duplicates.",
        "ja": "ソースをスキャンし、メタデータを抽出し、重複をマークします。",
    },
    "cli.help.index.src": {
        "ru": "Каталог с фото (рекурсивно); переопределяет config sources",
        "en": "Photo directory (recursively); overrides config sources",
        "ja": "写真のディレクトリ（再帰的）。config の sources を上書きします",
    },
    "cli.help.index.refresh_exif": {
        "ru": "Перечитать метаданные уже проиндексированных файлов "
              "(вместо сканирования). Содержимое файлов не читается.",
        "en": "Re-read the metadata of the already indexed files (instead of "
              "scanning). File contents are not read.",
        "ja": "インデックス済みファイルのメタデータを読み直します（スキャンの代わり）。"
              "ファイルの中身は読みません。",
    },
    "cli.help.index.exclude_dir": {
        "ru": "Не сканировать эту папку источника (путь относительно корня). "
              "Можно повторять. Сохраняется в файл исключений.",
        "en": "Do not scan this source folder (path relative to the root). "
              "Repeatable. Saved into the excludes file.",
        "ja": "このソースフォルダをスキャンしません（ルートからの相対パス）。"
              "繰り返し指定できます。除外ファイルに保存されます。",
    },
    # stats
    "cli.help.stats": {
        "ru": "Покрытие индекса: GPS, источники дат, дубликаты.",
        "en": "Index coverage: GPS, date sources, duplicates.",
        "ja": "インデックスのカバレッジ: GPS、日付のソース、重複。",
    },
    # dupes
    "cli.help.dupes": {
        "ru": "Список точных дубликатов; с --near — группы почти-дубликатов.",
        "en": "List the exact duplicates; with --near — the near-duplicate groups.",
        "ja": "完全一致の重複を一覧表示します。--near を付けると類似写真のグループを表示します。",
    },
    "cli.help.dupes.near": {
        "ru": "Показать почти-дубликаты (pHash)",
        "en": "Show the near-duplicates (pHash)",
        "ja": "類似写真を表示します (pHash)",
    },
    # geo
    "cli.help.geo": {
        "ru": "Определить место каждого файла: GPS + наследование по сессиям.",
        "en": "Resolve the place of every file: GPS + inheritance across sessions.",
        "ja": "各ファイルの場所を判定します: GPS + セッション単位の継承。",
    },
    # landmarks
    "cli.help.landmarks": {
        "ru": "Места без GPS по известным достопримечательностям (CLIP). "
              "Запускать после geo.",
        "en": "Places without GPS, from well-known landmarks (CLIP). Run after geo.",
        "ja": "GPS のない場所を有名なランドマークから判定します (CLIP)。"
              "geo の後に実行してください。",
    },
    # phash
    "cli.help.phash": {
        "ru": "Посчитать pHash для почти-дубликатов (для `dupes --near`).",
        "en": "Compute the pHash for near-duplicates (for `dupes --near`).",
        "ja": "類似写真のための pHash を計算します（`dupes --near` 用）。",
    },
    # classify
    "cli.help.classify": {
        "ru": "Вердикты классификатора (screenshot|meme|document|product) — до лиц.\n"
              "\n"
              "Передняя половина стадии junk: только вердикты, без качества кадра и\n"
              "каскадов. Стадия `faces` пропускает всё, что здесь названо не фотографией;\n"
              "кадр без вердикта проходит детекцию лиц как обычно.",
        "en": "The classifier verdicts (screenshot|meme|document|product) — before faces.\n"
              "\n"
              "The front half of the junk stage: the verdicts alone, without the frame\n"
              "quality and the cascades. The `faces` stage skips whatever is called a\n"
              "non-photograph here; a frame with no verdict is detected as usual.",
        "ja": "分類器の判定 (screenshot|meme|document|product) — 顔検出の前に。\n"
              "\n"
              "junk ステージの前半です。判定のみを行い、フレーム品質やカスケードは\n"
              "実行しません。`faces` ステージはここで写真ではないとされたものを\n"
              "スキップします。判定のないフレームは通常どおり処理されます。",
    },
    # junk
    "cli.help.junk": {
        "ru": "Классифицировать фото/мусор (screenshot|meme|document) для сортировки.",
        "en": "Classify photos/junk (screenshot|meme|document) for sorting.",
        "ja": "写真とゴミを分類します (screenshot|meme|document)。整理に使います。",
    },
    # doctor
    "cli.help.doctor": {
        "ru": "Диагностика окружения: torch/onnxruntime, GPU, гео-база, лог-файл.",
        "en": "Environment diagnostics: torch/onnxruntime, GPU, geo data, log file.",
        "ja": "環境の診断: torch/onnxruntime、GPU、地理データ、ログファイル。",
    },
    # cache
    "cli.help.cache": {
        "ru": "Кэши: показать путь и размер, при --clear/--clear-geo — удалить.\n"
              "\n"
              "Кэш превью безопасно удалять в любой момент: он ленивый и пересоздаётся той\n"
              "стадией, которой первой понадобится кадр. Смысл команды — освободить место\n"
              "(порядка 150 КБ на фото) или заставить перегенерировать превью после смены\n"
              "настроек.\n"
              "\n"
              "Кэш геоданных (F93) — ответы онлайн-провайдера в таблице geo_cache. Он\n"
              "переживает и повторный прогон, и «Начать заново», поэтому --clear-geo —\n"
              "единственный способ переспросить провайдера, если он однажды ответил неверно.",
        "en": "Caches: show the path and the size, with --clear/--clear-geo — remove.\n"
              "\n"
              "The preview cache is safe to delete at any moment: it is lazy and is "
              "rebuilt by whichever stage needs a frame first. The point of the command "
              "is to free up space (some 150 KB per photo) or to force previews to be "
              "regenerated after a settings change.\n"
              "\n"
              "The geo cache (F93) holds the answers of the online provider, in the "
              "geo_cache table. It survives both a repeated run and “Start over”, so "
              "--clear-geo is the only way to ask the provider again once it has "
              "answered wrongly.",
        "ja": "キャッシュ: パスとサイズを表示します。--clear/--clear-geo で削除します。\n"
              "\n"
              "プレビューキャッシュはいつ削除しても安全です。遅延生成なので、フレームを"
              "最初に必要とするステージが作り直します。このコマンドの目的は、容量を空けること"
              "（写真 1 枚あたり約 150 KB）と、設定変更後にプレビューを作り直させることです。\n"
              "\n"
              "位置情報キャッシュ (F93) は geo_cache テーブルにあるオンラインプロバイダの"
              "応答です。再実行しても「最初からやり直す」を実行しても残るため、"
              "プロバイダが一度誤った応答を返した場合に問い合わせ直す唯一の方法が "
              "--clear-geo です。",
    },
    "cli.help.cache.clear": {
        "ru": "Удалить кэш превью (он пересоберётся сам)",
        "en": "Remove the preview cache (it rebuilds itself)",
        "ja": "プレビューキャッシュを削除します（自動で再生成されます）",
    },
    "cli.help.cache.clear_geo": {
        "ru": "Удалить кэш ответов онлайн-геокодера (F93): следующий `sorta geo` "
              "при provider: online снова сходит в сеть",
        "en": "Remove the cached answers of the online geocoder (F93): the next "
              "`sorta geo` with provider: online goes to the network again",
        "ja": "オンラインジオコーダの応答キャッシュ (F93) を削除します。"
              "provider: online の場合、次の `sorta geo` は再びネットワークに接続します",
    },
    # F224: the models are reached through `cache` and not through `reset`, because a
    # cache is what they are — derived files that come back by themselves on the next
    # run that needs them. `reset` erases the opposite of that (names of people, the
    # decisions about duplicates), which nothing can download again.
    "cli.help.cache.models": {
        "ru": "Показать скачанные веса моделей: пути и размеры, ничего не удаляя",
        "en": "Show the downloaded model weights — paths and sizes, deleting nothing",
        "ja": "ダウンロード済みモデルの重みを表示します（パスとサイズ、削除はしません）",
    },
    "cli.help.cache.clear_models": {
        "ru": "Удалить скачанные веса моделей после подтверждения: только модели из "
              "каталога ярусов, чужие файлы в тех же кэшах и цели ссылок не тронутся",
        "en": "Remove the downloaded model weights after a confirmation: only the "
              "models of the tier catalog — other files in the same caches and the "
              "targets of links are left alone",
        "ja": "確認のうえダウンロード済みモデルの重みを削除します。削除するのは"
              "ティアカタログのモデルだけで、同じキャッシュ内の他のファイルや"
              "リンク先には手を触れません",
    },
    "cli.help.cache.yes": {
        "ru": "Не спрашивать про --clear-models (согласие уже получено — так эту "
              "команду зовёт деинсталлятор Windows)",
        "en": "Do not ask about --clear-models (the consent has already been given — "
              "this is how the Windows uninstaller calls the command)",
        "ja": "--clear-models について確認しません（同意は取得済み — "
              "Windows のアンインストーラはこの形で呼び出します）",
    },
    "cli.help.cache.preview_max_gb": {
        "ru": "Потолок кэша превью в ГБ на этот прогон "
              "(imaging.preview_cache_max_gb); 0 — без потолка, как и в конфиге",
        "en": "Ceiling of the preview cache in GB for this run "
              "(imaging.preview_cache_max_gb); 0 means no ceiling, as in the config",
        "ja": "この実行でのプレビューキャッシュの上限 (GB) "
              "(imaging.preview_cache_max_gb)。0 は上限なしで、config と同じです",
    },
    # ui
    "cli.help.ui": {
        "ru": "Локальный веб-интерфейс: живой отчёт плана (пока режим city). "
              "Ctrl+C — стоп.",
        "en": "The local web interface: a live report of the plan (city mode so far). "
              "Ctrl+C stops it.",
        "ja": "ローカルのウェブインターフェース: プランのライブレポート（現状は city モード）。"
              "Ctrl+C で停止します。",
    },
    "cli.help.ui.port": {
        "ru": "Порт локального сервера (127.0.0.1)",
        "en": "Port of the local server (127.0.0.1)",
        "ja": "ローカルサーバのポート (127.0.0.1)",
    },
    # faces
    "cli.help.faces": {
        "ru": "Лица: детекция, кластеры, именование.",
        "en": "Faces: detection, clusters, naming.",
        "ja": "顔: 検出、クラスタ、名前付け。",
    },
    "cli.help.faces.rescan": {
        "ru": "Пересчитать лица заново: стереть строки faces и продетектировать "
              "все канонические фото (имена кластеров переносятся по файлам). "
              "Нужен после смены детектора; без флага шаг инкрементальный",
        "en": "Recompute the faces from scratch: erase the faces rows and detect over "
              "all canonical photos (cluster names are carried over by file). Needed "
              "after changing the detector; without the flag the step is incremental",
        "ja": "顔を最初から計算し直します: faces の行を削除し、すべての正本写真で検出します"
              "（クラスタ名はファイル単位で引き継がれます）。検出器を変更した後に必要です。"
              "フラグなしの場合、このステップは差分処理です",
    },
    "cli.help.faces.limit": {
        "ru": "Только с --rescan: пересчитать N случайных файлов, остальные не "
              "трогать (замер шага на живом пайплайне)",
        "en": "Only with --rescan: recompute N random files and leave the rest alone "
              "(timing the step on the live pipeline)",
        "ja": "--rescan と併用する場合のみ: ランダムな N 件のファイルだけを再計算し、"
              "残りには触れません（実パイプラインでのステップ計測用）",
    },
    "cli.help.faces.label": {
        "ru": 'Назвать кластер: sorta faces label 3 "Мама".',
        "en": 'Name a cluster: sorta faces label 3 "Mum".',
        "ja": 'クラスタに名前を付けます: sorta faces label 3 "母".',
    },
    "cli.help.faces.merge": {
        "ru": "Слить кластер src в dst (это один человек).",
        "en": "Merge cluster src into dst (they are one person).",
        "ja": "クラスタ src を dst に統合します（同一人物の場合）。",
    },
    "cli.help.faces.sheet": {
        "ru": "Экспорт контактного листа кластера в HTML.",
        "en": "Export a contact sheet of the cluster into HTML.",
        "ja": "クラスタのコンタクトシートを HTML に書き出します。",
    },
    # events
    "cli.help.events": {
        "ru": "События: автокластеризация, имена, ручные события.",
        "en": "Events: automatic clustering, names, manual events.",
        "ja": "イベント: 自動クラスタリング、名前、手動イベント。",
    },
    "cli.help.events.rename": {
        "ru": "Переименовать событие (имя переживает пересчёт).",
        "en": "Rename an event (the name survives a recompute).",
        "ja": "イベントの名前を変更します（名前は再計算後も残ります）。",
    },
    "cli.help.events.add": {
        "ru": 'Ручное событие на диапазон дат: events add "Конференция" '
              '2024-01-01 2024-01-10.',
        "en": 'A manual event over a date range: events add "Conference" '
              '2024-01-01 2024-01-10.',
        "ja": '日付範囲を指定する手動イベント: events add "会議" 2024-01-01 2024-01-10.',
    },
    # sort
    "cli.help.sort": {
        "ru": "Разложить файлы перемещением. По умолчанию — dry-run с планом "
              "(CSV+HTML).",
        "en": "Lay the files out by moving them. By default — a dry run with a plan "
              "(CSV+HTML).",
        "ja": "ファイルを移動して整理します。デフォルトはプラン付きの dry-run です"
              "（CSV+HTML）。",
    },
    "cli.help.sort.by": {
        "ru": "city | person | event",
        "en": "city | person | event",
        "ja": "city | person | event",
    },
    "cli.help.sort.dest": {
        "ru": "Каталог назначения; без него — in-place раскладка в корень источника "
              "(единственный sources)",
        "en": "Destination directory; without it — an in-place layout into the source "
              "root (the single sources entry)",
        "ja": "出力先のディレクトリ。指定しない場合はソースのルートに in-place で配置します"
              "（sources が 1 つのときのみ）",
    },
    "cli.help.sort.apply": {
        "ru": "Реально переместить (иначе dry-run)",
        "en": "Actually move (otherwise a dry run)",
        "ja": "実際に移動します（指定しない場合は dry-run）",
    },
    "cli.help.sort.copy": {
        "ru": "Копировать в новую структуру, оригиналы на месте (C16; иначе "
              "перемещение)",
        "en": "Copy into the new structure, leaving the originals in place (C16; "
              "otherwise a move)",
        "ja": "元のファイルを残したまま新しい構造にコピーします（C16。指定しない場合は移動）",
    },
    "cli.help.sort.where": {
        "ru": 'Фильтр, повторяемый: "country=DE", "year>=2020"',
        "en": 'Filter, repeatable: "country=DE", "year>=2020"',
        "ja": '繰り返し指定できるフィルタ: "country=DE"、"year>=2020"',
    },
    "cli.help.sort.thumbnails": {
        "ru": "Миниатюры в HTML-отчёте (медленно: декод всех фото)",
        "en": "Thumbnails in the HTML report (slow: every photo is decoded)",
        "ja": "HTML レポートにサムネイルを付けます（すべての写真をデコードするため遅い）",
    },
    "cli.help.sort.dedupe": {
        "ru": "Почти-дубли: лучший — по режиму, худшие — в _Duplicates "
              "(нужен sorta phash)",
        "en": "Near-duplicates: the best one goes by the mode, the worse ones into "
              "_Duplicates (needs sorta phash)",
        "ja": "類似写真: 最良の 1 枚は通常どおり、それ以外は _Duplicates に入れます"
              "（sorta phash が必要）",
    },
    "cli.help.sort.delete_worse_dupes": {
        "ru": "С --dedupe: БЕЗВОЗВРАТНО удалять худшие (не откатывается)",
        "en": "With --dedupe: delete the worse ones IRREVERSIBLY (no undo)",
        "ja": "--dedupe と併用: 劣る方を完全に削除します（取り消せません）",
    },
    "cli.help.sort.exclude": {
        "ru": "Не сортировать файлы из этого каталога (повторяемый); объединяется "
              "с sort.exclude_dirs",
        "en": "Do not sort the files from this directory (repeatable); merged with "
              "sort.exclude_dirs",
        "ja": "このディレクトリのファイルを整理しません（繰り返し指定可）。"
              "sort.exclude_dirs と統合されます",
    },
    # search (F129)
    "cli.help.search": {
        "ru": "Найти кадры словами: CLIP-ранжирование по сохранённым эмбеддингам.\n"
              "\n"
              "Печатает пути и оценки, лучшие сверху. Это ранжирование, а не фильтр:\n"
              "порога «точно оно» здесь нет. Одиночные предметы («торт», «снег», «море»)\n"
              "CLIP разбирает заметно лучше составных фраз.",
        "en": "Find frames by words: a CLIP ranking over the stored embeddings.\n"
              "\n"
              "Prints paths and scores, the best ones first. This is a ranking, not a\n"
              "filter: there is no “definitely it” threshold. Single subjects (“cake”,\n"
              "“snow”, “the sea”) work markedly better than compound phrases.",
        "ja": "語で写真を探します: 保存済みの埋め込みに対する CLIP のランキングです。\n"
              "\n"
              "パスとスコアを、上位から順に表示します。これはフィルタではなく"
              "ランキングで、\n「確実にそれ」という閾値はありません。単独の被写体"
              "（「ケーキ」「雪」「海」）は\n複合的な言い回しよりはるかにうまく扱えます。",
    },
    "cli.help.search.query": {
        "ru": "Что искать: слово или короткая фраза",
        "en": "What to look for: a word or a short phrase",
        "ja": "探すもの: 語または短い言い回し",
    },
    "cli.help.search.limit": {
        "ru": "Сколько кадров показать (по умолчанию features.search_limit) — "
              "это размер выборки, а не порог похожести",
        "en": "How many frames to show (features.search_limit by default) — a sample "
              "size, not a similarity threshold",
        "ja": "表示する枚数（既定は features.search_limit）— 類似度の閾値ではなく、"
              "取り出す標本の大きさです",
    },
    # F189: the way back to the ranking when the words are also somebody's name.
    "cli.help.search.words": {
        "ru": "Искать по словам, даже если строка совпала с именем человека "
              "(по умолчанию имя даёт точный отбор кадров этого человека)",
        "en": "Search by words even when the string is somebody's name (by default a "
              "name gives the exact selection of that person's frames)",
        "ja": "文字列が人物名と一致していても語として検索します"
              "（既定では名前はその人物のコマの正確な抽出になります）",
    },
    # album
    "cli.help.album": {
        "ru": "Выгрузить срез (человека/события/животных/запроса/класса/качества/лиц) "
              "в отдельную папку. По умолчанию — hardlink, dry-run.",
        "en": "Export a slice (a person/an event/the animals/a query/a class/a quality "
              "slice/the faces) into a separate folder. By default — hardlink, dry run.",
        "ja": "スライス（人物・イベント・動物・問い合わせ・分類・画質・顔）を別のフォルダに"
              "書き出します。デフォルトはハードリンクの dry-run です。",
    },
    "cli.help.album.kind": {
        "ru": "person | event | animal | query | product | screenshot | meme | "
              "blurred | eyes_closed | low_resolution | people | group | portrait",
        "en": "person | event | animal | query | product | screenshot | meme | "
              "blurred | eyes_closed | low_resolution | people | group | portrait",
        "ja": "person | event | animal | query | product | screenshot | meme | "
              "blurred | eyes_closed | low_resolution | people | group | portrait",
    },
    "cli.help.album.selector": {
        "ru": "имя человека / имя или id события / слова запроса; для animal, срезов "
              "класса и качества, people, group и portrait не нужен — такой срез в "
              "коллекции один",
        "en": "a person's name / an event's name or id / the words to search for; not "
              "needed for animal, a class/quality slice, people, group or portrait — "
              "the collection has a single slice of each",
        "ja": "人物の名前 / イベントの名前または id / 検索する語。animal・分類・画質・"
              "people・group・portrait では不要です — そのスライスはコレクションに"
              " 1 つだけです",
    },
    "cli.help.album.dest": {
        "ru": "Куда выгрузить альбом",
        "en": "Where to export the album",
        "ja": "アルバムの書き出し先",
    },
    "cli.help.album.copy": {
        "ru": "Копировать (иначе hardlink)",
        "en": "Copy (otherwise a hardlink)",
        "ja": "コピーします（指定しない場合はハードリンク）",
    },
    "cli.help.album.move": {
        "ru": "Изъять из пула (перемещение); иначе hardlink",
        "en": "Take out of the pool (a move); otherwise a hardlink",
        "ja": "プールから取り出します（移動）。指定しない場合はハードリンク",
    },
    "cli.help.album.where": {
        "ru": 'Доп. фильтр среза: "city=Барселона", "year>=2020"',
        "en": 'An extra filter on the slice: "city=Barcelona", "year>=2020"',
        "ja": 'スライスの追加フィルタ: "city=バルセロナ"、"year>=2020"',
    },
    "cli.help.album.name": {
        "ru": "Имя папки альбома (иначе имя человека/события)",
        "en": "Name of the album folder (otherwise the person's/event's name)",
        "ja": "アルバムフォルダの名前（指定しない場合は人物・イベントの名前）",
    },
    "cli.help.album.apply": {
        "ru": "Реально выгрузить (иначе dry-run)",
        "en": "Actually export (otherwise a dry run)",
        "ja": "実際に書き出します（指定しない場合は dry-run）",
    },
    # reset
    "cli.help.reset": {
        "ru": "Стереть индекс (БД) и начать с нуля. Фото и разложенные папки НЕ "
              "трогает.\n"
              "\n"
              "Внимание: пропадут имена людей/событий и решения по дублям. Кэш геоданных\n"
              "(F93) остаётся — названия точек на карте не зависят от того, какие файлы лежат\n"
              "у пользователя; стереть и его — `--clear-geo`.",
        "en": "Erase the index (the DB) and start from scratch. Photos and already "
              "sorted folders are NOT touched.\n"
              "\n"
              "Careful: people/event names and duplicate decisions will be lost. The "
              "geo cache (F93) stays — the names of the points on the map do not depend "
              "on which files a user happens to have; to erase it too — `--clear-geo`.",
        "ja": "インデックス（DB）を消去して最初からやり直します。"
              "写真と整理済みのフォルダには手を触れません。\n"
              "\n"
              "注意: 人物・イベントの名前と重複の判断は失われます。"
              "位置情報キャッシュ (F93) は残ります — 地図上の地点の名前は、"
              "利用者がどのファイルを持っているかに依存しないためです。"
              "これも消すには `--clear-geo` を使います。",
    },
    "cli.help.reset.yes": {
        "ru": "Без подтверждения",
        "en": "Without a confirmation",
        "ja": "確認なしで実行します",
    },
    "cli.help.reset.clear_geo": {
        "ru": "Заодно очистить кэш ответов онлайн-геокодера (F93); без флага он "
              "переживает сброс, и повторный прогон geo не стоит сети",
        "en": "Clear the cached answers of the online geocoder (F93) as well; without "
              "the flag it survives the reset and a repeated geo run costs no network",
        "ja": "オンラインジオコーダの応答キャッシュ (F93) も消去します。"
              "フラグなしの場合はリセット後も残るため、geo の再実行はネットワークを使いません",
    },
    # undo
    "cli.help.undo": {
        "ru": "Откатить перемещения последнего (или указанного) запуска sort по "
              "журналу.",
        "en": "Undo the moves of the last (or the given) sort run, from the journal.",
        "ja": "直近（または指定した）sort 実行の移動を、ジャーナルに従って取り消します。",
    },
    "cli.help.undo.batch": {
        "ru": "ID батча (по умолчанию последний)",
        "en": "Batch id (the last one by default)",
        "ja": "バッチの ID（デフォルトは直近のもの）",
    },
    # run
    "cli.help.run": {
        "ru": "Анализ одним прогоном: index -> geo -> landmarks -> classify -> junk "
              "(+faces/+events с флагами).\n"
              "\n"
              "Ничего не перемещает. С --by в конце строит dry-run план (в --dest либо\n"
              "in-place в корень источника, если --dest не задан).",
        "en": "The whole analysis in one run: index -> geo -> landmarks -> classify -> "
              "junk (+faces/+events behind flags).\n"
              "\n"
              "Moves nothing. With --by it builds a dry-run plan at the end (into "
              "--dest, or in place into the source root if --dest is not given).",
        "ja": "分析を 1 回の実行で: index -> geo -> landmarks -> classify -> junk"
              "（フラグで +faces/+events）。\n"
              "\n"
              "何も移動しません。--by を指定すると、最後に dry-run のプランを作成します"
              "（--dest に、--dest がなければソースのルートに in-place で）。",
    },
    "cli.help.run.by": {
        "ru": "city|person|event — построить dry-run план в конце",
        "en": "city|person|event — build a dry-run plan at the end",
        "ja": "city|person|event — 最後に dry-run のプランを作成します",
    },
    "cli.help.run.dest": {
        "ru": "Каталог назначения для плана с --by; без него — in-place",
        "en": "Destination directory for the --by plan; without it — in place",
        "ja": "--by のプランの出力先ディレクトリ。指定しない場合は in-place",
    },
    # F230: the requirement stays, the COMMAND moves out — this text used to end in
    # `uv sync --extra vlm` for everybody. `{how}` is filled from
    # `cli.help.run.deep_how.*` by the install the help is printed on.
    "cli.help.run.deep": {
        "ru": "Глубокий анализ VLM на этот прогон: медленнее, нужен глубокий ярус "
              "(VLM) — {how} (иначе откат на быстрый ярус); "
              "без флага — как в config.yaml (naming.vlm_enabled)",
        "en": "Deep VLM analysis for this run: slower, needs the deep tier (VLM) — {how} "
              "(otherwise it falls back to the fast tier); without the flag — as in "
              "config.yaml (naming.vlm_enabled)",
        "ja": "この実行で VLM による詳細分析を行います: 低速で、ディープティア（VLM）が"
              "必要です — {how}（ない場合は高速な階層にフォールバックします）。"
              "フラグなしの場合は config.yaml のとおり (naming.vlm_enabled)",
    },
    "cli.help.run.deep_how.checkout": {
        "ru": "поставьте его через uv sync --extra vlm",
        "en": "add it with uv sync --extra vlm",
        "ja": "uv sync --extra vlm で追加します",
    },
    "cli.help.run.deep_how.installed": {
        "ru": "добавьте его в мастере: пункт «Sorta setup» в меню «Пуск»",
        "en": "add it in the wizard: the Sorta setup item of the Start menu",
        "ja": "ウィザードで追加します: スタートメニューの「Sorta setup」",
    },
    "cli.help.run.deep_how.tool": {
        "ru": "добавьте его через sorta-setup",
        "en": "add it with sorta-setup",
        "ja": "sorta-setup で追加します",
    },
    "cli.help.run.geo": {
        "ru": "offline|online — online точнее для мест за границей, но "
              "отправляет GPS-координаты фото серверу геокодирования "
              "(Nominatim), сами фото не отправляются; без флага — как в "
              "config.yaml (geo.provider)",
        "en": "offline|online — online is more precise for places abroad, but it sends "
              "the GPS coordinates of the photos to a geocoding server (Nominatim); "
              "the photos themselves are not sent; without the flag — as in "
              "config.yaml (geo.provider)",
        "ja": "offline|online — online は国外の場所でより正確ですが、写真の GPS 座標を"
              "ジオコーディングサーバ (Nominatim) に送信します（写真自体は送信しません）。"
              "フラグなしの場合は config.yaml のとおり (geo.provider)",
    },
    "cli.help.run.faces": {
        "ru": "Разбор по лицам (детекция + кластеризация) — самый долгий "
              "шаг; по умолчанию выключен, доступен отдельно как `sorta "
              "faces`",
        "en": "The face pass (detection + clustering) — the longest step; off by "
              "default, available on its own as `sorta faces`",
        "ja": "顔の処理（検出 + クラスタリング）— 最も時間のかかるステップです。"
              "デフォルトは無効で、`sorta faces` として個別に実行できます",
    },
    "cli.help.run.events": {
        "ru": "Группировка в события по времени/месту; по умолчанию "
              "выключена, доступна отдельно как `sorta events`",
        "en": "Grouping into events by time/place; off by default, available on its "
              "own as `sorta events`",
        "ja": "時間・場所によるイベントへのグループ化。デフォルトは無効で、"
              "`sorta events` として個別に実行できます",
    },
    # F222: the third opt-in stage, and the only one with a config key behind it — hence
    # "the config decides" rather than "off by default" as the two above say.
    "cli.help.run.landmarks": {
        "ru": "Определять места по виду (CLIP по списку достопримечательностей); "
              "по умолчанию решает features.landmarks — выключено, 0,55% мест за "
              "1,6 ГБ весов; доступно отдельно как `sorta landmarks`",
        "en": "Recognise places by sight (CLIP over the landmark list); without the "
              "flag features.landmarks decides — off, 0.55% of the places for 1.6 GB "
              "of weights; available on its own as `sorta landmarks`",
        "ja": "見た目で場所を判定します（名所リストに対する CLIP）。フラグがなければ "
              "features.landmarks が決めます（既定は無効、場所の 0.55% に対し重み "
              "1.6 GB）。`sorta landmarks` として個別に実行できます",
    },
    "cli.help.run.src": {
        "ru": "Каталог-источник для этого прогона; переопределяет "
              "config sources (как позиционный аргумент у `index`)",
        "en": "Source directory for this run; overrides config sources (like the "
              "positional argument of `index`)",
        "ja": "この実行のソースディレクトリ。config の sources を上書きします"
              "（`index` の位置引数と同じ）",
    },
    # the argparse fallback (no typer): its one positional argument
    "cli.help.fallback.command": {
        "ru": "Команда",
        "en": "The command",
        "ja": "コマンド",
    },
}


# The `--help` half of the catalog: `build_app` reads exactly these, and the parity of
# the three languages over them is checked apart from the rest (tests/test_cli_help.py).
HELP_PREFIX = "cli.help."


def help_keys() -> tuple[str, ...]:
    """Every key of the catalog that a `--help` text is built from."""
    return tuple(sorted(k for k in _CLI_STRINGS if k.startswith(HELP_PREFIX)))


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


def stage_label(stage: str, lang: Lang) -> str:
    """A pipeline stage, named the way a person reads it — or its bare key (F222).

    Here rather than next to either caller because turning a key into a name in a
    language is what this module is for, and there are two callers: the sentences about a
    download (`sorta/tiers.py`) and the one about a stage that was skipped while its
    settings sat in the config (`sorta/config.py`). Not every stage has a name — only the
    ones some sentence prints — and a stage without one is called by its key rather than
    by nothing.
    """
    key = f"cli.stage.{stage}"
    named = cli_text(key, lang)
    return stage if named == key else named
