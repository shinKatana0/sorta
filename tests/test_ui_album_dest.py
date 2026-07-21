"""F60: the album destination path in the UI — input + "Browse…" + default as in Cities.

_validate_album_payload parses the optional `dest`, /api/album accepts it and passes
it to `plan_album` (otherwise it falls back to `_album_dest`); the People/Events card
markup/JS carries a path field + a "Browse…" button (/api/browse) + a prefill via
GET /api/sort/suggest-dest; an i18n placeholder in 3 languages.
"""
from __future__ import annotations

import unittest
from unittest import mock

from sorta import ui
from sorta.ui import _validate_album_payload

from tests.test_ui_albums import AlbumsTestBase


class TestValidateAlbumPayloadDest(unittest.TestCase):
    def test_dest_string_is_parsed(self):
        parsed = _validate_album_payload(
            {"kind": "person", "selector": "Мама", "mode": "link", "dest": "C:\\Album"})
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[6], "C:\\Album")

    def test_dest_missing_is_none(self):
        parsed = _validate_album_payload(
            {"kind": "person", "selector": "Мама", "mode": "link"})
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed[6])

    def test_dest_empty_string_is_none(self):
        parsed = _validate_album_payload(
            {"kind": "person", "selector": "Мама", "mode": "link", "dest": "   "})
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed[6])

    def test_dest_not_string_returns_400(self):
        parsed = _validate_album_payload(
            {"kind": "person", "selector": "Мама", "mode": "link", "dest": 5})
        self.assertIsNone(parsed)


class TestHandleAlbumDest(AlbumsTestBase):
    def test_dest_passed_through_to_plan_album(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        with mock.patch.object(ui, "plan_album", wraps=ui.plan_album) as spy:
            status, _body = self.post(
                "/api/album",
                {"kind": "person", "selector": "Мама", "mode": "link", "apply": False,
                 "dest": str(self.root / "MyAlbum")})
            self.assertEqual(status, 200)
            _cfg, _conn, kind, selector, dest = spy.call_args[0]
            self.assertEqual(kind, "person")
            self.assertEqual(selector, "Мама")
            self.assertEqual(str(dest), str(self.root / "MyAlbum"))

    def test_missing_dest_falls_back_to_album_dest(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        expected = ui._album_dest(self.cfg, self.cfg.database)
        with mock.patch.object(ui, "plan_album", wraps=ui.plan_album) as spy:
            status, _body = self.post(
                "/api/album",
                {"kind": "person", "selector": "Мама", "mode": "link", "apply": False})
            self.assertEqual(status, 200)
            _cfg, _conn, _kind, _selector, dest = spy.call_args[0]
            self.assertEqual(str(dest), str(expected))


class TestAlbumDestMarkup(AlbumsTestBase):
    def test_person_and_event_cards_have_dest_input_and_browse(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("album-dest-input", html)
        self.assertIn("album-browse-btn", html)
        self.assertIn("appendAlbumDestControls", html)
        self.assertIn('"/api/browse"', html)
        self.assertIn('"/api/sort/suggest-dest"', html)
        self.assertIn("album_dest_placeholder", html)

    def test_gather_album_sends_dest_in_body(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("function gatherAlbum(kind, selector, mode, where, name, dest, statusEl)", html)
        self.assertIn("if (dest) body.dest = dest;", html)
        self.assertIn("destInput.value.trim() || null", html)

    def test_no_external_resources_u1(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


class TestAlbumDestI18n(AlbumsTestBase):
    def test_placeholder_ru_en_ja(self):
        for lang, text in (
            ("ru", "Путь назначения альбома"),
            ("en", "Album destination path"),
            ("ja", "アルバムの保存先パス"),
        ):
            self.cfg.raw = {"language": lang}
            self.start_server()
            _status, body, _ctype = self.get("/")
            html = body.decode("utf-8")
            self.assertIn(text, html, msg=f"lang={lang}")
            self.tearDown()
            self.setUp()


if __name__ == "__main__":
    unittest.main()
