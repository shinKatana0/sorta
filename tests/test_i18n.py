import string

from sorta.i18n import _CLI_STRINGS, FOLDER_KEYS, cli_text, country, folder, normalize_lang

_LANGS = ("ru", "en", "ja")


class TestNormalizeLang:
    def test_known_values_pass_through(self) -> None:
        assert normalize_lang("ru") == "ru"
        assert normalize_lang("en") == "en"
        assert normalize_lang("ja") == "ja"

    def test_case_insensitive(self) -> None:
        assert normalize_lang("RU") == "ru"
        assert normalize_lang("EN") == "en"
        assert normalize_lang("Ja") == "ja"

    def test_whitespace_is_stripped(self) -> None:
        assert normalize_lang("  ru ") == "ru"
        assert normalize_lang("  en  ") == "en"

    def test_unknown_and_empty_default_to_en(self) -> None:
        assert normalize_lang("") == "en"
        assert normalize_lang("xx") == "en"
        assert normalize_lang(None) == "en"


class TestFolder:
    def test_each_key_has_three_distinct_nonempty_translations(self) -> None:
        for key in FOLDER_KEYS:
            values = {lang: folder(key, lang) for lang in ("ru", "en", "ja")}
            for lang, value in values.items():
                assert value, f"{key}/{lang} is empty"
            assert len(set(values.values())) == 3, f"{key}: translations are not distinct {values}"

    def test_unknown_key_returns_itself(self) -> None:
        assert folder("nonexistent_key", "ru") == "nonexistent_key"
        assert folder("nonexistent_key", "en") == "nonexistent_key"


class TestCountry:
    def test_ru_across_languages(self) -> None:
        assert country("ru", "ru") == "Россия"
        assert country("ru", "en") == "Russia"
        assert country("ru", "ja") == "ロシア"

    def test_required_collection_countries_covered(self) -> None:
        for cc in ("th", "id", "tr", "ae"):
            for lang in ("ru", "en", "ja"):
                value = country(cc, lang)
                assert value and value != cc

    def test_unknown_code_returns_itself(self) -> None:
        assert country("zz", "ru") == "zz"
        assert country("xx", "en") == "xx"

    def test_case_insensitive(self) -> None:
        assert country("RU", "ru") == country("ru", "ru")
        assert country("Th", "en") == country("th", "en")


class TestCliStrings:
    """F112: the catalog behind the command line. The parity rule is the same one the
    served UI lives by — a key without all three languages is a half-translated
    release, and it has to fail here rather than in someone's terminal."""

    def test_every_key_has_all_three_languages(self) -> None:
        for key, entry in _CLI_STRINGS.items():
            assert set(entry) == set(_LANGS), f"{key}: {sorted(entry)}"
            for lang in _LANGS:
                assert entry[lang].strip(), f"{key}/{lang} is empty"

    def test_every_language_of_a_key_takes_the_same_fields(self) -> None:
        # The caller passes ONE set of fields for all languages, so a placeholder that
        # exists only in, say, ja would raise KeyError for Japanese users only.
        def fields(template: str) -> set[str]:
            return {name for _, name, _, _ in string.Formatter().parse(template)
                    if name}

        for key, entry in _CLI_STRINGS.items():
            expected = fields(entry["en"])
            for lang in _LANGS:
                assert fields(entry[lang]) == expected, f"{key}/{lang}"

    def test_keys_are_namespaced_by_command(self) -> None:
        for key in _CLI_STRINGS:
            assert key.startswith("cli."), key
            assert len(key.split(".")) >= 3, f"{key}: needs a command segment"

    def test_substitution_is_by_name(self) -> None:
        assert cli_text("cli.undo.done", "ru", batch=4, undone=10, missing=1, failed=2) \
            == "Откат батча 4: возвращено 10, отсутствовало 1, ошибок 2"
        assert cli_text("cli.stats.files", "en", total=7, errors=2) \
            == "Files in the index: 7 (+2 with errors)"

    def test_unknown_key_returns_itself(self) -> None:
        assert cli_text("cli.nope.at_all", "ru") == "cli.nope.at_all"

    def test_missing_language_falls_back_to_english(self) -> None:
        _CLI_STRINGS["cli.f112.test_only"] = {"en": "only english"}  # type: ignore[dict-item]
        try:
            assert cli_text("cli.f112.test_only", "ja") == "only english"
        finally:
            del _CLI_STRINGS["cli.f112.test_only"]

    def test_no_key_is_identical_across_all_three(self) -> None:
        # A key whose three variants are the same string is almost always a variant
        # someone forgot to translate. The exceptions are texts that are pure code
        # (option names, DB confidence values) and genuinely have no words in them:
        # the DB confidence column, and the two F114 help texts that are nothing but
        # the list of values the option accepts ("city | person | event").
        untranslatable = {"cli.stats.geo_confidence", "cli.help.sort.by",
                          "cli.help.album.kind"}
        for key, entry in _CLI_STRINGS.items():
            if key in untranslatable:
                continue
            assert len(set(entry.values())) > 1, f"{key}: one text for three languages"


def test_folder_keys_catalog_is_complete() -> None:
    required = {
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
    }
    assert required <= set(FOLDER_KEYS)
    for key in FOLDER_KEYS:
        for lang in ("ru", "en", "ja"):
            assert folder(key, lang)
