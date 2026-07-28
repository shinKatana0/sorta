"""F99 (G3): the web app labels a frame with the same city name its folder gets.

The cards of the "Cities"/"Events" tabs show `geo` = country/city, and the city used to
come straight out of `places` — i.e. the English anchor geo writes there — while the
target folder next to it was already localized. One card then named two places.

The plan is built here through `sorter.plan_and_sort` (as `ui.PlanCache` does) and the
payload through `ui._plan_item_to_json`, without starting a server: what is being pinned
is the card's CONTENT, and the HTTP layer around it has its own tests.
"""
from __future__ import annotations

import unittest

from tests.test_sorter_city_names import _GID_KAPONG, _GID_SPB, CityNamesTestBase

from sorta import ui


class TestPlanCardCity(CityNamesTestBase):
    def _cards(self, lang: str, mode: str = "city") -> dict[str, dict]:
        return {item.src.name: ui._plan_item_to_json(item)
                for item in self.plan(lang, mode)}

    def test_card_shows_the_localized_city(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        card = self._cards("ru")["spb.jpg"]
        self.assertEqual(card["geo"], "RU/Санкт-Петербург")
        self.assertIn("Санкт-Петербург", card["target_rel"])

    def test_card_follows_the_interface_language(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        self.assertEqual(self._cards("ja")["spb.jpg"]["geo"], "RU/サンクトペテルブルク")
        self.assertEqual(self._cards("en")["spb.jpg"]["geo"], "RU/Saint Petersburg")

    def test_card_of_an_untranslated_city_keeps_the_english_name(self):
        self.add_file("kapong.jpg", country="TH", city="Kapong",
                      city_geonameid=_GID_KAPONG)
        self.assertEqual(self._cards("ru")["kapong.jpg"]["geo"], "TH/Kapong")

    def test_card_without_a_place_has_no_geo(self):
        self.add_file("noplace.jpg")
        self.assertIsNone(self._cards("ru")["noplace.jpg"]["geo"])

    def test_event_tab_cards_are_localized_too(self):
        # The "Events" tab serves the same plan items (mode='event') through the same
        # payload — the city on the card must not fall back to the anchor there.
        fid = self.add_file("ev.jpg", country="RU", city="Saint Petersburg",
                            city_geonameid=_GID_SPB)
        self.add_event(fid, "Поездка")
        card = self._cards("ru", mode="event")["ev.jpg"]
        self.assertEqual(card["geo"], "RU/Санкт-Петербург")
        self.assertEqual(card["target_rel"], "2022/Поездка/ev.jpg")


if __name__ == "__main__":
    unittest.main()
