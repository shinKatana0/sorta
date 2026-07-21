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
