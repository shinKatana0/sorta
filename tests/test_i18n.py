from sorta.i18n import FOLDER_KEYS, country, folder, normalize_lang


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
