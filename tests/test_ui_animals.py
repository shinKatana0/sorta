"""F123: animals in the UI — the "Process" checkbox, the "Animals" tab, the album.

Three things that must not blur into each other:

* the checkbox is a CONFIG OVERRIDE on the run (`features.pets`), the way `deep` is,
  and NOT a pipeline stage — the class of test that matters most here is the one that
  reads the composition of the run's stage list back and finds it unchanged;
* the tab appears by data presence (the F54 mechanism) and orders by confidence;
* the album is `POST /api/album` with `kind='animal'` and no selector.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

from sorta import ui

from tests.test_ui import UiServerTestBase
from tests.test_ui_process import ProcessTestBase, _poll_until


class AnimalsTestBase(UiServerTestBase):
    """A `frame_quality` fixture on top of the base U1 server."""

    def mark_animal(self, file_id: int, *, score: float | None = 0.9,
                    pet: str | None = "animal") -> None:
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, source,
                   updated_at)
               VALUES (?, 100.0, ?, ?, 'clip', '2026-01-01')""",
            (file_id, pet, score))
        self.conn.commit()

    def animals(self, query: str = "") -> dict:
        status, body, ctype = self.get("/api/animals" + query)
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)


class TestApiAnimals(AnimalsTestBase):
    def test_empty_when_nothing_carries_a_pet_verdict(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.mark_animal(fid, pet=None, score=0.3)
        self.start_server()
        data = self.animals()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["items"], [])

    def test_items_carry_the_score_the_card_shows(self):
        fid, p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid, score=0.83)
        self.start_server()
        item = self.animals()["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["name"], p.name)
        self.assertAlmostEqual(item["score"], 0.83)
        self.assertEqual(item["thumb_url"], f"/thumb/{fid}")
        self.assertFalse(item["video"])
        self.assertEqual(item["date"], "2022-05-01T10:00:00")

    def test_sorted_by_score_descending(self):
        low, _p, _c = self.add_photo_file("low.jpg")
        high, _p2, _c2 = self.add_photo_file("high.jpg")
        middle, _p3, _c3 = self.add_photo_file("mid.jpg")
        self.mark_animal(low, score=0.71)
        self.mark_animal(high, score=0.99)
        self.mark_animal(middle, score=0.85)
        self.start_server()
        data = self.animals()
        self.assertEqual([it["file_id"] for it in data["items"]], [high, middle, low])

    def test_equal_scores_keep_a_deterministic_order(self):
        # Without the id tiebreak a page boundary can show a frame twice or never.
        ids = []
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            fid, _p, _c = self.add_photo_file(name)
            self.mark_animal(fid, score=0.8)
            ids.append(fid)
        self.start_server()
        first = [it["file_id"] for it in self.animals()["items"]]
        second = [it["file_id"] for it in self.animals()["items"]]
        self.assertEqual(first, sorted(ids))
        self.assertEqual(first, second)

    def test_duplicates_and_read_errors_are_not_in_the_slice(self):
        canonical, _p, _c = self.add_photo_file("a.jpg")
        duplicate, _p2, _c2 = self.add_photo_file("b.jpg")
        broken, _p3, _c3 = self.add_photo_file("c.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'nope' WHERE id = ?", (broken,))
        self.conn.commit()
        for fid in (canonical, duplicate, broken):
            self.mark_animal(fid)
        self.start_server()
        data = self.animals()
        self.assertEqual(data["total"], 1)
        self.assertEqual([it["file_id"] for it in data["items"]], [canonical])

    def test_paging_walks_the_whole_slice_without_repeats(self):
        for i in range(5):
            fid, _p, _c = self.add_photo_file(f"a{i}.jpg")
            self.mark_animal(fid, score=0.9 - i / 100)
        self.start_server()
        page1 = self.animals("?offset=0&limit=2")
        page2 = self.animals("?offset=2&limit=2")
        page3 = self.animals("?offset=4&limit=2")
        self.assertEqual(page1["total"], 5)
        seen = [it["file_id"] for it in page1["items"] + page2["items"] + page3["items"]]
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(set(seen)), 5)

    def test_bad_offset_is_a_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/animals?offset=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class TestAnimalsTabVisibility(AnimalsTestBase):
    def test_hidden_while_no_frame_carries_a_verdict(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.mark_animal(fid, pet=None, score=0.2)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertFalse(json.loads(body)["animal"])

    def test_shown_once_a_verdict_exists(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.mark_animal(fid)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertTrue(json.loads(body)["animal"])

    def test_pin_absent_from_the_markup_by_default(self):
        # F133: "Animals" is a pinned slice, and the pin row is built from data — the
        # rule ("the slice exists exactly when there is something to show") is the same.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn('id="slice-pin-animal"', html)
        self.assertIn("animal: !!data.animal,", html)
        self.assertIn(
            'if (sliceVisibility.animal) pins.push({ key: "animal", '
            "label: I18N.tab_animal });", html)


class TestAnimalsOverviewCounter(AnimalsTestBase):
    def test_counted_over_the_same_population_as_the_tab(self):
        keep, _p, _c = self.add_photo_file("a.jpg")
        duplicate, _p2, _c2 = self.add_photo_file("b.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (keep, duplicate))
        self.conn.commit()
        self.mark_animal(keep)
        self.mark_animal(duplicate)
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertEqual(json.loads(body)["collection"]["animals"], 1)
        self.assertEqual(self.animals()["total"], 1)

    def test_zero_when_the_feature_never_ran(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertEqual(json.loads(body)["collection"]["animals"], 0)


class TestAnimalsTabHtml(AnimalsTestBase):
    def test_grid_paging_and_album_controls_are_in_the_markup(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-animal"', html)
        self.assertIn('id="animals-grid"', html)
        self.assertIn('id="animals-more-btn"', html)
        self.assertIn('id="animals-album"', html)
        self.assertIn('"/api/animals?offset="', html)
        self.assertIn('gatherAlbum("animal", ""', html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_the_grid_is_paged_rather_than_rendered_whole(self):
        # F70: 805 cards with previews must not land in the DOM at once.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("var ANIMALS_PAGE_SIZE = 200;", html)
        # F173: the paging is the shared pager's, not a fourth copy of the same loop.
        self.assertIn("var animalsPager = makePager(", html)
        self.assertIn("return animalsPager.load();", html)

    def test_i18n_ru_en_ja(self):
        self.start_server()
        for lang, expected in (("ru", "Животные"), ("en", "Animals"), ("ja", "動物")):
            _status, body, _ctype = self.get(f"/?lang={lang}")
            self.assertIn(expected, body.decode("utf-8"))

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("tab_animal", "animals_intro", "animals_empty", "animals_score_label",
                "slice_load_more", "slice_shown_label", "error_loading_animals",
                "overview_animals", "process_pets_label", "process_pets_hint")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())
        for lang in ("ru", "en", "ja"):
            self.assertIn("{score}", ui._UI_STRINGS["animals_score_label"][lang])
            self.assertIn("{shown}", ui._UI_STRINGS["slice_shown_label"][lang])
            self.assertIn("{total}", ui._UI_STRINGS["slice_shown_label"][lang])

    def test_the_intro_names_the_measured_precision(self):
        """F158: the caption promised "about 92%" — a number from the score-stratified
        F122 sample, which the 500-frame random re-measurement did not support and which
        the wider gate makes wronger still. It has to state what the shipped cascade was
        measured at (82%), in all three languages, or the slice oversells itself."""
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                intro = ui._UI_STRINGS["animals_intro"][lang]
                self.assertIn("82", intro)
                self.assertNotIn("92", intro)


class TestApiAlbumAnimal(AnimalsTestBase):
    def post(self, path: str, data: object) -> tuple[int, dict]:
        import urllib.error
        import urllib.request

        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_preview_counts_without_writing_anything(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "animal", "selector": "", "mode": "link", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "animal")
        self.assertEqual(body["count"], 1)
        self.assertFalse(body["applied"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_a_missing_selector_is_accepted_for_this_kind_only(self):
        fid, _p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        status, body = self.post("/api/album", {"kind": "animal", "mode": "link"})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        status, _body = self.post("/api/album", {"kind": "person", "mode": "link"})
        self.assertEqual(status, 400)

    def test_apply_link_creates_the_hardlink_and_an_album_animal_batch(self):
        fid, p, _c = self.add_photo_file("cat.jpg")
        self.mark_animal(fid)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "animal", "selector": "", "mode": "link", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        self.assertEqual(body["failed"], 0)
        self.assertTrue((Path(body["dest"]) / p.name).exists())
        batch = self.conn.execute(
            "SELECT mode, operation FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(batch["mode"], "album_animal")
        self.assertEqual(batch["operation"], "link")

    def test_bad_mode_is_still_a_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "animal", "selector": "", "mode": "teleport"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)


class TestProcessPetsOverride(ProcessTestBase):
    """The checkbox is `dataclasses.replace(cfg.features, ...)` on the run's config —
    the `deep` mechanism, not the `faces` one."""

    def _capture_junk_cfg(self) -> dict:
        """The config the junk stage of the run is actually handed."""
        captured: dict = {}

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                      verdicts_only=False, progress=None):
            # F165: the same function is the `classify` stage too — the config under
            # test is the one the half AFTER faces gets, so only that call is captured.
            if not verdicts_only:
                captured["cfg"] = cfg
            self.calls.append("classify" if verdicts_only else "junk")

        self.patch_fast_stages()
        self._patch("classify_junk", fake_junk)
        return captured

    def test_unchecked_leaves_the_run_config_alone(self):
        captured = self._capture_junk_cfg()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(captured["cfg"].features.pets)
        self.assertFalse(self.cfg.features.pets)

    def test_checked_reaches_the_junk_stage_of_this_run_only(self):
        captured = self._capture_junk_cfg()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "pets": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertTrue(captured["cfg"].features.pets)
        # the server cfg (config.yaml) is not mutated
        self.assertFalse(self.cfg.features.pets)

    def test_unchecked_forces_off_what_config_yaml_enables(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, pets=True)
        captured = self._capture_junk_cfg()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "pets": False})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(captured["cfg"].features.pets)
        self.assertTrue(self.cfg.features.pets)

    def test_the_other_feature_thresholds_survive_the_override(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, pet_threshold=0.55)
        captured = self._capture_junk_cfg()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir), "pets": True})
        _poll_until(self.status, lambda d: d["finished"])
        self.assertAlmostEqual(captured["cfg"].features.pet_threshold, 0.55)

    def test_pets_adds_no_stage_to_the_run(self):
        # The main way to get this feature wrong: a phantom stage in the list.
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "pets": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "classify", "junk",
             "phash"])
        # index/geo/landmarks/classify/junk/phash
        self.assertEqual(final["stage_total"], 6)
        self.assertNotIn("pets", ui._PIPELINE_STAGE_NAMES)
        self.assertNotIn("pets", ui._OPTIONAL_STAGES)

    def test_non_bool_pets_is_a_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "pets": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_defaults_start_from_config_yaml(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, pets=True)
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        self.assertTrue(json.loads(body)["pets"])


class TestRerunOptionalWithPets(ProcessTestBase):
    def test_pets_alone_re_runs_junk(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"pets": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(self.calls, ["junk"])
        self.assertEqual(final["stage_total"], 1)

    def test_pets_and_deep_together_are_one_junk_run(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process/rerun-optional", {"pets": True, "deep": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(self.calls, ["junk"])
        self.assertEqual(final["stage_total"], 1)

    def test_pets_with_faces_keeps_the_pipeline_order(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process/rerun-optional", {"faces": True, "pets": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(self.calls, ["faces", "junk"])
        self.assertEqual(final["stage_total"], 2)

    def test_pets_reaches_the_config_of_the_re_run(self):
        captured: dict = {}

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                      verdicts_only=False, progress=None):
            # F165: the same function is the `classify` stage too — the config under
            # test is the one the half AFTER faces gets, so only that call is captured.
            if not verdicts_only:
                captured["cfg"] = cfg
            self.calls.append("classify" if verdicts_only else "junk")

        self.patch_fast_stages()
        self._patch("classify_junk", fake_junk)
        self.start_server()
        self.post("/api/process/rerun-optional", {"pets": True})
        _poll_until(self.status, lambda d: d["finished"])
        self.assertTrue(captured["cfg"].features.pets)

    def test_all_four_false_is_still_a_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process/rerun-optional",
            {"faces": False, "events": False, "deep": False, "pets": False})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_non_bool_pets_is_a_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"pets": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)


class TestPetsCheckboxHtml(ProcessTestBase):
    def test_checkbox_sits_with_faces_and_events(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-pets-checkbox"', html)
        events_pos = html.index('id="process-events-checkbox"')
        pets_pos = html.index('id="process-pets-checkbox"')
        self.assertLess(events_pos, pets_pos)
        options_end = html.index('id="step-actions"')
        self.assertLess(pets_pos, options_end)  # in the run-options block

    def test_the_hint_says_it_is_almost_free(self):
        # Read as "one more long step" next to faces (17 min) and deep (hours), the
        # checkbox simply never gets ticked.
        self.start_server()
        for lang, expected in (("ru", "Почти бесплатно"),
                               ("en", "Almost free"),
                               ("ja", "ほぼ無料")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))

    def test_js_wires_the_checkbox_into_start_defaults_and_rerun(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn(
            'document.getElementById("process-pets-checkbox").checked = !!data.pets', html)
        self.assertIn("pets: pets,", html)
        self.assertIn('"process-pets-checkbox"', html)
        # and it is NOT a stage in the client's model of the run either
        self.assertIn('var ALL_PROCESS_STAGES = ["index", "geo", "landmarks", '
                      '"classify", "faces", "events",', html)
        self.assertNotIn('OPTIONAL_PROCESS_STAGES = { faces: true, events: true, pets',
                         html)


if __name__ == "__main__":
    unittest.main()
