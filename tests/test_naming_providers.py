"""F6: the local_vlm (ollama) and claude (Anthropic API) providers — HTTP mocked.

Real network traffic in tests is forbidden: urllib.request.urlopen is mocked.
"""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, _naming_from
from sorta.naming import ClaudeNamer, EventContext, LocalVLMNamer, naming_settings

CTX_DATES = {"started_at": "2023-05-01T10:00:00", "ended_at": "2023-05-01T18:00:00"}


class FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ProviderTestCase(unittest.TestCase):
    """Temporary images + urlopen interception."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.requests = []  # [(urllib.request.Request, timeout), ...]

    def tearDown(self):
        self.tmp.cleanup()

    def make_images(self, *names):
        paths = []
        for name in names:
            p = Path(self.tmp.name) / name
            p.write_bytes(b"fake image bytes: " + name.encode())
            paths.append(str(p))
        return tuple(paths)

    def settings(self, naming):
        cfg = Config(sources=[Path(self.tmp.name)], naming=_naming_from(naming))
        return naming_settings(cfg)

    def run_with_response(self, namer, ctx, payload):
        def fake_urlopen(req, timeout=None):
            self.requests.append((req, timeout))
            return FakeResponse(payload)

        with patch("sorta.naming.urllib.request.urlopen", side_effect=fake_urlopen):
            return namer.name(ctx)

    def sent_json(self, i=0):
        return json.loads(self.requests[i][0].data.decode("utf-8"))

    def sent_headers(self, i=0):
        return {k.lower(): v for k, v in self.requests[i][0].header_items()}


class TestLocalVLM(ProviderTestCase):
    def test_names_event_from_samples(self):
        namer = LocalVLMNamer(self.settings(
            {"provider": "local_vlm",
             "local_vlm": {"base_url": "http://gpu:11434", "model": "llava"}}))
        ctx = EventContext(**CTX_DATES, city="Paris",
                           sample_paths=self.make_images("a.jpg", "b.png"))
        name = self.run_with_response(namer, ctx, {"response": " Свадьба Ани \n..."})
        self.assertEqual(name, "2023-05-01 Свадьба Ани")
        req, _timeout = self.requests[0]
        self.assertEqual(req.full_url, "http://gpu:11434/api/generate")
        body = self.sent_json()
        self.assertEqual(body["model"], "llava")
        self.assertFalse(body["stream"])
        self.assertEqual(len(body["images"]), 2)  # both frames went to the model

    def test_max_samples_limit(self):
        namer = LocalVLMNamer(self.settings({"max_samples": 3}))
        paths = self.make_images(*[f"img_{i}.jpg" for i in range(10)])
        ctx = EventContext(**CTX_DATES, city=None, sample_paths=paths)
        self.run_with_response(namer, ctx, {"response": "Поход"})
        self.assertEqual(len(self.sent_json()["images"]), 3)

    def test_network_error_returns_none(self):
        namer = LocalVLMNamer(self.settings({}))
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        with patch("sorta.naming.urllib.request.urlopen", side_effect=OSError("нет сети")):
            self.assertIsNone(namer.name(ctx))

    def test_no_images_falls_back_to_template_without_http(self):
        namer = LocalVLMNamer(self.settings({}))
        ctx = EventContext(**CTX_DATES, city="Paris", sample_paths=())
        with patch("sorta.naming.urllib.request.urlopen") as urlopen:
            self.assertEqual(namer.name(ctx), "2023-05-01 Paris")
        urlopen.assert_not_called()


class TestClaude(ProviderTestCase):
    def claude(self, naming=None):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-123"}):
            return ClaudeNamer(self.settings(naming or {"provider": "claude"}))

    def claude_payload(self, text="День рождения"):
        return {"content": [{"type": "text", "text": text}]}

    def test_request_shape_and_name(self):
        namer = self.claude({"provider": "claude",
                             "claude": {"model": "claude-opus-4-8"}})
        jpg, png, heic = self.make_images("a.jpg", "b.png", "c.heic")
        ctx = EventContext(**CTX_DATES, city=None, sample_paths=(jpg, png, heic))
        name = self.run_with_response(namer, ctx, self.claude_payload("«Поход в горы»"))
        self.assertEqual(name, "2023-05-01 Поход в горы")  # quotes stripped

        req, _ = self.requests[0]
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
        headers = self.sent_headers()
        self.assertEqual(headers["x-api-key"], "sk-test-123")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        body = self.sent_json()
        self.assertEqual(body["model"], "claude-opus-4-8")
        blocks = body["messages"][0]["content"]
        images = [b for b in blocks if b["type"] == "image"]
        self.assertEqual(len(images), 2)  # heic is not supported by the API — skipped
        self.assertEqual(images[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(images[1]["source"]["media_type"], "image/png")
        self.assertEqual(
            base64.standard_b64decode(images[0]["source"]["data"]),
            Path(jpg).read_bytes())
        self.assertEqual(blocks[-1]["type"], "text")

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                ClaudeNamer(self.settings({"provider": "claude"}))

    def test_malformed_response_returns_none(self):
        namer = self.claude()
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        self.assertIsNone(self.run_with_response(namer, ctx, {"content": []}))

    def test_multiline_answer_sanitized(self):
        namer = self.claude()
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        name = self.run_with_response(
            namer, ctx, self.claude_payload('Пикник у озера.\nПояснение: на фото...'))
        self.assertEqual(name, "2023-05-01 Пикник у озера")

    def test_hostile_chars_stripped_for_folder_name(self):
        namer = self.claude()
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        name = self.run_with_response(namer, ctx, self.claude_payload('Утро/вечер: "тест"'))
        self.assertEqual(name, "2023-05-01 Утро вечер тест")


if __name__ == "__main__":
    unittest.main()
