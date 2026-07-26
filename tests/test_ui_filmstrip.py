"""F80: the lightbox filmstrip in the web app — `/frame/<id>/<i>` and the video mark.

Two things are being protected here. The endpoint must behave like its neighbours
(`/thumb`, `/preview`): a path only ever comes out of the DB by file_id, and anything
that does not resolve is a 404 rather than a traceback — the browser asks for frames
it cannot know exist, and uses the 404 to find the end of a short clip's strip.

And the photo path must be untouched: the same `/preview/<id>`, the same lightbox,
the same tile markup. The regression class at the bottom is the whole point of it.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
import urllib.parse
from pathlib import Path

from sorta import imaging, ui
from sorta.hashing import file_hash

from tests.test_ui import UiServerTestBase

HAVE_AV = importlib.util.find_spec("av") is not None
_JPEG_MAGIC = b"\xff\xd8"


class FilmstripUiTestBase(UiServerTestBase):
    def add_video_file(self, rel: str = "clip.mp4", *, real: bool = False,
                       country: str | None = None,
                       city: str | None = None) -> tuple[int, Path]:
        """A files row with media_type='video'.

        `real=False` writes a few bytes that no decoder will accept — enough for
        everything that only cares about the extension (the `video` flag in a payload,
        the 404s). `real=True` encodes a synthetic clip and needs PyAV.
        """
        self._n += 1
        path = self.src_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if real:
            from tests.test_imaging_filmstrip import make_gradient_video

            make_gradient_video(path)
        else:
            path.write_bytes(b"not a real container" * 20)
        digest, algo = file_hash(path)
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   taken_at, taken_at_source, taken_at_confidence, indexed_at)
               VALUES (?, ?, 0, 'mp4', 'video', ?, ?, '2022-05-01T10:00:00', 'exif',
                       'high', '2026-01-01')""",
            (str(path.resolve()), path.stat().st_size, digest, algo),
        )
        file_id = cur.lastrowid
        if country is not None or city is not None:
            self.conn.execute(
                """INSERT INTO places (file_id, country, region, city, confidence,
                       updated_at)
                   VALUES (?, ?, NULL, ?, 'exact_gps', '2026-01-01')""",
                (file_id, country, city))
        self.conn.commit()
        return file_id, path

    def plan_item(self, file_id: int, mode: str = "city") -> dict:
        """The plan row of one file, whichever target folder it landed in."""
        _status, body, _ctype = self.get(f"/api/plan?mode={mode}")
        for row in json.loads(body)["categories"]:
            _s, page, _c = self.get(
                f"/api/plan?mode={mode}&category=" + urllib.parse.quote(row["category"]))
            for item in json.loads(page)["items"]:
                if item["file_id"] == file_id:
                    return item
        raise AssertionError(f"file {file_id} is not in the {mode} plan")


class TestFrameEndpointGuards(FilmstripUiTestBase):
    """Everything that must be a 404 — the endpoint is reachable from a URL bar."""

    def test_unknown_file_id_is_404(self):
        self.start_server()
        self.assertEqual(self.get("/frame/999999/0")[0], 404)

    def test_non_numeric_id_is_404(self):
        self.start_server()
        self.assertEqual(self.get("/frame/abc/0")[0], 404)

    def test_non_numeric_index_is_404(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}/abc")[0], 404)

    def test_missing_index_is_404(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}")[0], 404)
        self.assertEqual(self.get(f"/frame/{fid}/")[0], 404)

    def test_negative_index_is_404(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}/-1")[0], 404)

    def test_a_traversal_attempt_does_not_reach_the_filesystem(self):
        self.start_server()
        self.assertEqual(self.get("/frame/..%2F..%2Fetc%2Fpasswd/0")[0], 404)

    def test_an_index_past_the_end_of_a_clip_is_404_not_500(self):
        fid, _p = self.add_video_file()
        self.start_server()
        for index in (1, 5, 99, 100000):
            self.assertEqual(self.get(f"/frame/{fid}/{index}")[0], 404, index)

    def test_an_undecodable_clip_is_404_not_500(self):
        fid, _p = self.add_video_file()  # bytes that are not a container
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}/0")[0], 404)


class TestFrameEndpointOnPhotos(FilmstripUiTestBase):
    def test_frame_zero_of_a_photo_is_the_lightbox_preview(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, frame, ctype = self.get(f"/frame/{fid}/0")
        _s, preview, _c = self.get(f"/preview/{fid}")

        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/jpeg")
        self.assertTrue(frame.startswith(_JPEG_MAGIC))
        self.assertEqual(frame, preview)

    def test_a_photo_has_no_second_frame(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}/1")[0], 404)


@unittest.skipUnless(HAVE_AV, "PyAV is not installed (it ships with the `vlm` extra)")
class TestFrameEndpointOnARealClip(FilmstripUiTestBase):
    def test_every_frame_of_the_strip_is_served_and_they_differ(self):
        fid, _p = self.add_video_file(real=True)
        self.start_server()
        bodies = []
        for index in range(imaging.video_frames()):
            status, body, ctype = self.get(f"/frame/{fid}/{index}")
            self.assertEqual(status, 200, index)
            self.assertEqual(ctype, "image/jpeg")
            self.assertTrue(body.startswith(_JPEG_MAGIC))
            bodies.append(body)
        self.assertEqual(len(set(bodies)), len(bodies))

    def test_frame_zero_is_the_same_image_as_the_tile_shows(self):
        """The cache guard, seen from the UI: the strip must not redraw the grid."""
        fid, _p = self.add_video_file(real=True)
        self.start_server()
        _s, frame_zero, _c = self.get(f"/frame/{fid}/0")
        _s, preview, _c = self.get(f"/preview/{fid}")
        self.assertEqual(frame_zero, preview)

    def test_the_first_index_past_the_strip_is_404(self):
        fid, _p = self.add_video_file(real=True)
        self.start_server()
        self.assertEqual(self.get(f"/frame/{fid}/{imaging.video_frames()}")[0], 404)


class TestVideoFlagInPayloads(FilmstripUiTestBase):
    def test_a_clip_is_marked_and_a_photo_is_not(self):
        photo_id, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        video_id, _v = self.add_video_file("clip.mp4", country="ru", city="Moscow")
        self.start_server()

        self.assertTrue(self.plan_item(video_id)["video"])
        self.assertFalse(self.plan_item(photo_id)["video"])

    def test_the_moves_manifest_marks_clips_too(self):
        fid, path = self.add_video_file()
        dst = str(self.root / "sorted" / "Moscow" / "2022" / "clip.mp4")
        cur = self.conn.execute(
            """INSERT INTO move_batches (mode, dest_root, started_at, operation)
               VALUES ('city', ?, '2026-01-01T00:00:00', 'move')""",
            (str(self.root / "sorted"),))
        self.conn.execute(
            """INSERT INTO moves (batch_id, file_id, src, dst, hash, status)
               VALUES (?, ?, ?, ?, 'deadbeef', 'done')""",
            (cur.lastrowid, fid, str(path), dst))
        self.conn.commit()
        self.start_server()

        _status, body, _ctype = self.get("/api/moves")
        self.assertTrue(json.loads(body)["moves"][0]["video"])


class TestLightboxMarkup(FilmstripUiTestBase):
    def test_the_frame_pager_is_in_the_page(self):
        self.start_server()
        html = self.get("/")[1].decode("utf-8")
        self.assertIn('id="lightbox-prev"', html)
        self.assertIn('id="lightbox-next"', html)
        self.assertIn('id="lightbox-dots"', html)
        self.assertIn('"/frame/"', html)

    def test_the_configured_frame_count_reaches_the_page(self):
        self.start_server()
        html = self.get("/")[1].decode("utf-8")
        self.assertIn(f"window.VIDEO_FRAMES = {imaging.video_frames()}", html)

    def test_the_new_captions_are_translated_in_all_three_languages(self):
        for key in ("video_badge", "video_open", "frame_prev", "frame_next", "frame_of"):
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    self.assertIn(lang, ui._UI_STRINGS[key])
                    self.assertTrue(ui._UI_STRINGS[key][lang].strip())

    def test_the_frame_counter_keeps_its_placeholders_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            caption = ui._UI_STRINGS["frame_of"][lang]
            self.assertIn("{n}", caption)
            self.assertIn("{all}", caption)

    def test_the_tile_badge_is_only_built_for_video(self):
        self.start_server()
        html = self.get("/")[1].decode("utf-8")
        # the wrapper exists, and it is created behind `if (!isVideo) return img;`
        self.assertIn("thumb-video-badge", html)
        self.assertIn("if (!isVideo) return img;", html)


class TestPhotoLightboxUnchanged(FilmstripUiTestBase):
    """The regression insurance of F80 — nothing about photos may have moved."""

    def test_the_photo_lightbox_still_opens_the_plain_preview(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, body, ctype = self.get(f"/preview/{fid}")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/jpeg")
        self.assertTrue(body.startswith(_JPEG_MAGIC))

    def test_the_photo_branch_of_the_lightbox_script_is_intact(self):
        self.start_server()
        html = self.get("/")[1].decode("utf-8")
        # a photo still goes to /preview/<id>, and arrows still page the sample list
        self.assertIn('lightboxImg.src = "/preview/" + lightboxSamples[index];', html)
        self.assertIn("lightboxSamples.length", html)

    def test_a_photo_tile_still_serves_the_same_thumbnail(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, body, _ctype = self.get(f"/thumb/{fid}")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(_JPEG_MAGIC))

    def test_the_frame_cache_key_separates_frames_of_one_file(self):
        """Two frames of one clip may never be served from one cache entry."""
        ui._thumb_cache_clear()
        fid, path = self.add_video_file()
        self.assertIsNone(ui._preview_bytes(fid, path, frame=0))
        self.assertIsNone(ui._preview_bytes(fid, path, frame=1))
        self.assertEqual(ui._preview_cache, {})


if __name__ == "__main__":
    unittest.main()
