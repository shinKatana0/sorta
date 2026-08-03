"""F6/F170: the local_vlm (ollama) provider — HTTP mocked — and the way out being shut.

Real network traffic in tests is forbidden: urllib.request.urlopen is mocked.

F170 removed the cloud provider, the one code path in the product that uploaded
photographs anywhere, and the guards below are what makes the removal a property rather
than a decision somebody remembers. They do not read the naming module; they read the
package and the guides as text, because the way this comes back is not a rewritten
provider but a helper, an example or a documentation line that quietly reopens the
address. The remaining provider that speaks HTTP at all — ollama — is tested here in the
same file so that "what leaves this process, and where to" stays one subject.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import (
    NAMING_PROVIDERS,
    REMOVED_NAMING_PROVIDERS,
    Config,
    _naming_from,
)
from sorta.naming import EventContext, LocalVLMNamer, naming_settings

CTX_DATES = {"started_at": "2023-05-01T10:00:00", "ended_at": "2023-05-01T18:00:00"}

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "sorta"
_GUIDES = tuple((_ROOT / "docs" / "guide").glob("user-guide.*.md"))


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

    def test_unsupported_formats_are_skipped(self):
        """HEIC/RAW are not sent as bytes: this API takes the handful it decodes."""
        namer = LocalVLMNamer(self.settings({}))
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg", "b.heic", "c.cr2"))
        self.run_with_response(namer, ctx, {"response": "Поход"})
        self.assertEqual(len(self.sent_json()["images"]), 1)

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

    def test_multiline_answer_sanitized(self):
        namer = LocalVLMNamer(self.settings({}))
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        name = self.run_with_response(
            namer, ctx, {"response": "Пикник у озера.\nПояснение: на фото..."})
        self.assertEqual(name, "2023-05-01 Пикник у озера")

    def test_hostile_chars_stripped_for_folder_name(self):
        namer = LocalVLMNamer(self.settings({}))
        ctx = EventContext(**CTX_DATES, city=None,
                           sample_paths=self.make_images("a.jpg"))
        name = self.run_with_response(namer, ctx, {"response": 'Утро/вечер: "тест"'})
        self.assertEqual(name, "2023-05-01 Утро вечер тест")


class TestNoCloudProviderInTheSources(unittest.TestCase):
    """F170: the package holds no address a photograph could be sent to.

    Every file of `sorta/` is read as bytes, data files included: the point is not to
    check the module that used to hold the provider — that one is easy to watch — but to
    make the whole shipped package the subject, so a helper, an example config or a
    bundled string cannot put the address back where nobody is looking.
    """

    FORBIDDEN = (b"api.anthropic.com", b"anthropic")

    def package_files(self):
        found = [p for p in _PACKAGE.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
        # A walk that found nothing would make the case below vacuously green. Byte-code
        # is skipped for the opposite reason: a stale .pyc is a copy of a source file
        # that no longer exists, and failing on one would say nothing about the package.
        self.assertGreater(len(found), 20)
        return found

    def test_no_file_of_the_package_names_the_vendor_api(self):
        for path in self.package_files():
            blob = path.read_bytes().lower()
            for needle in self.FORBIDDEN:
                with self.subTest(file=path.relative_to(_ROOT).as_posix(),
                                  needle=needle.decode()):
                    self.assertNotIn(needle, blob)

    def test_the_example_config_offers_only_local_providers(self):
        text = (_ROOT / "config.example.yaml").read_text(encoding="utf-8")
        self.assertNotIn("anthropic", text.lower())
        for provider in REMOVED_NAMING_PROVIDERS:
            self.assertNotIn(f"provider: {provider}", text)
        self.assertIn("provider: template", text)

    def test_the_guides_do_not_document_a_removed_provider(self):
        """All three languages, and the key names as well as the provider itself."""
        self.assertEqual(len(_GUIDES), 3)
        for path in _GUIDES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(guide=path.name):
                self.assertNotIn("anthropic", text.lower())
                for provider in REMOVED_NAMING_PROVIDERS:
                    self.assertNotIn(f"naming.provider: {provider}", text)
                    self.assertNotIn(f"naming.{provider}_", text)

    def test_the_guides_state_that_no_image_leaves_the_machine(self):
        """The removal is worth saying out loud — it is what a reader now relies on.

        Deleting the paragraph that described the cloud provider would satisfy the case
        above and leave the guides silent about the thing that changed. What a reader
        gets from this feature is the sentence, so the sentence is what is checked — one
        phrasing per language, and the wording of each is the guide's own business.
        """
        promised = re.compile(
            r"never sends your images|никогда не отправляет ваши изображения"
            r"|画像を外部へ送信することはありません")
        for path in _GUIDES:
            with self.subTest(guide=path.name):
                self.assertRegex(path.read_text(encoding="utf-8"), promised)


class TestRemovedProviderInAConfig(unittest.TestCase):
    """F170: somebody's working config.yaml still says `claude`. It has to start."""

    def test_the_removed_value_becomes_the_template_with_a_message(self):
        for provider in REMOVED_NAMING_PROVIDERS:
            with self.subTest(provider=provider):
                with self.assertLogs("sorta.config", level="WARNING") as logs:
                    naming = _naming_from({"provider": provider})
                self.assertEqual(naming.provider, "template")
                message = "\n".join(logs.output)
                self.assertIn(provider, message)
                for available in NAMING_PROVIDERS:
                    self.assertIn(available, message)

    def test_the_rest_of_the_section_survives_the_fallback(self):
        """A key of the removed provider is ignored, not a crash, and nothing else moves."""
        naming = _naming_from({
            "provider": "claude",
            "claude": {"model": "whatever", "api_key_env": "SOME_KEY"},
            "max_samples": 3,
            "local_vlm": {"model": "qwen2.5vl"},
        })
        self.assertEqual(naming.provider, "template")
        self.assertEqual(naming.max_samples, 3)
        self.assertEqual(naming.vlm_model, "qwen2.5vl")

    def test_a_living_provider_is_not_touched(self):
        for provider in NAMING_PROVIDERS:
            with self.subTest(provider=provider):
                self.assertEqual(
                    _naming_from({"provider": provider}).provider, provider)


if __name__ == "__main__":
    unittest.main()
