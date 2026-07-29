"""F95: the `vlm` provider — event names from the local Qwen2.5-VL, no network.

The model itself is never built here: `VlmNamer` takes a loader, and the fake one
counts what the real one would cost. What the tests pin down is the behaviour that
makes the provider safe to switch on — one call per event, one model per run, a
template name on every failure, and no document ever reaching a provider (not even
the cloud one).

F101 adds the split runtime (`SplitVlm`) the deep junk tier pipelines. It is tested
here as what it has to remain for this module: an ordinary describe().
"""
import tempfile
import threading
import types
import unittest
from pathlib import Path

from PIL import Image

from sorta import imaging
from sorta import naming as naming_module
from sorta.config import DEFAULT_VLM_MAX_EDGE, Config, VlmConfig, _naming_from
from sorta.db import connect
from sorta.naming import (
    DEFAULT_VLM_MODEL,
    EventContext,
    SplitVlm,
    TemplateNamer,
    ThreadLocalProcessors,
    VlmNamer,
    make_namer,
    name_events,
    naming_settings,
    processor_is_fast,
    reset_shared_vlm,
    shared_vlm,
)

CTX_DATES = {"started_at": "2023-05-01T10:00:00", "ended_at": "2023-05-03T18:00:00"}


class FakeVlm:
    """A loaded model: counts the builds and the calls, answers from a script."""

    def __init__(self, answers=("Поход в горы",)):
        self.answers = list(answers)
        self.calls = []  # [(n_frames, prompt), ...] — one entry per generate

    def __call__(self, frames, prompt, max_new_tokens):
        self.calls.append((len(frames), prompt))
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


class CountingLoader:
    """A stand-in for qwen_vlm: counts how many times the weights would be loaded."""

    def __init__(self, model=None, fails=False):
        self.model = model or FakeVlm()
        self.fails = fails
        self.builds = 0
        self.model_names = []

    def __call__(self, model_name):
        self.builds += 1
        self.model_names.append(model_name)
        if self.fails:
            raise RuntimeError("transformers не установлен")
        return self.model


def cfg_with(naming: dict | None = None, tmp: str = ".",
             vlm: VlmConfig | None = None) -> Config:
    return Config(sources=[Path(tmp)], database=Path(tmp) / "test.db",
                  naming=_naming_from(naming or {}),
                  vlm=vlm if vlm is not None else VlmConfig())


class VlmTestCase(unittest.TestCase):
    """Real JPEGs on disk: the namer decodes the frames itself (Unicode/HEIC-safe)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        reset_shared_vlm()  # the runtime cache is process-wide — no leaks between tests

    def tearDown(self):
        reset_shared_vlm()
        self.tmp.cleanup()

    def make_images(self, n, prefix="img"):
        paths = []
        for i in range(n):
            p = Path(self.tmp.name) / f"{prefix}_{i}.jpg"
            Image.new("RGB", (64, 48), (10 * i, 100, 200)).save(p, "JPEG")
            paths.append(str(p))
        return tuple(paths)

    def namer(self, naming=None, **kwargs):
        settings = naming_settings(cfg_with(naming or {"provider": "vlm"},
                                            tmp=self.tmp.name))
        return VlmNamer(settings, **kwargs)


class TestProviderSelection(unittest.TestCase):
    """Test 1: the provider comes from the config, and the default did not move."""

    def test_default_stays_template(self):
        self.assertIsInstance(make_namer(naming_settings(cfg_with())), TemplateNamer)

    def test_vlm_is_opt_in(self):
        namer = make_namer(naming_settings(cfg_with({"provider": "vlm"})))
        self.assertIsInstance(namer, VlmNamer)

    def test_unknown_provider_still_rejected(self):
        with self.assertRaises(ValueError):
            make_namer(naming_settings(cfg_with({"provider": "vlm-2"})))


class TestSharedRuntime(VlmTestCase):
    """Test 8: one model per run — for the naming stage and for junk alike."""

    def test_loader_called_once_for_many_events(self):
        loader = CountingLoader()
        namer = self.namer(loader=loader)
        paths = self.make_images(2)
        for day in ("01", "02", "03"):
            namer.name(EventContext(started_at=f"2023-05-{day}T10:00:00",
                                    ended_at=f"2023-05-{day}T18:00:00",
                                    city=None, sample_paths=paths))
        self.assertEqual(loader.builds, 1)
        self.assertEqual(len(loader.model.calls), 3)  # ...but one call per event

    def test_junk_and_naming_share_one_instance(self):
        loader = CountingLoader()
        first = shared_vlm(DEFAULT_VLM_MODEL, loader)
        second = shared_vlm(DEFAULT_VLM_MODEL, loader)
        self.assertIs(first, second)
        self.assertEqual(loader.builds, 1)

    def test_configured_model_is_the_junk_one(self):
        loader = CountingLoader()
        namer = self.namer({"provider": "vlm", "classify_vlm_model": "Qwen/other"},
                           loader=loader)
        namer.name(EventContext(**CTX_DATES, city=None,
                                sample_paths=self.make_images(1)))
        self.assertEqual(loader.model_names, ["Qwen/other"])

    def test_failed_build_is_not_retried_per_event(self):
        loader = CountingLoader(fails=True)
        namer = self.namer(loader=loader)
        paths = self.make_images(1)
        for _ in range(3):
            namer.name(EventContext(**CTX_DATES, city="Praha", sample_paths=paths))
        self.assertEqual(loader.builds, 1)


class TestOneCallPerEvent(VlmTestCase):
    """Test 2: 3-5 frames in a single call, never one call per file."""

    def test_single_call_with_all_frames(self):
        loader = CountingLoader()
        namer = self.namer(loader=loader)
        name = namer.name(EventContext(**CTX_DATES, city="Пхукет",
                                       sample_paths=self.make_images(3)))
        self.assertEqual(len(loader.model.calls), 1)
        self.assertEqual(loader.model.calls[0][0], 3)
        self.assertEqual(name, "2023-05-01..05-03 Пхукет Поход в горы")

    def test_frames_capped_by_max_samples(self):
        loader = CountingLoader()
        namer = self.namer({"provider": "vlm", "max_samples": 3}, loader=loader)
        namer.name(EventContext(**CTX_DATES, city=None,
                                sample_paths=self.make_images(12)))
        self.assertEqual(len(loader.model.calls), 1)
        self.assertEqual(loader.model.calls[0][0], 3)

    def test_prompt_asks_only_for_content(self):
        """The model is not asked for dates or places — those are known exactly."""
        loader = CountingLoader()
        namer = self.namer(loader=loader)
        namer.name(EventContext(**CTX_DATES, city="Пхукет",
                                sample_paths=self.make_images(1)))
        prompt = loader.model.calls[0][1]
        self.assertIn("без дат", prompt)
        self.assertNotIn("Пхукет", prompt)


class TestGracefulFallback(VlmTestCase):
    """Test 3 and 4: nothing about this provider may break the naming stage."""

    def ctx(self, paths=None):
        return EventContext(**CTX_DATES, city="Тайланд",
                            sample_paths=paths if paths is not None else ())

    def test_model_unavailable_falls_back_to_template(self):
        namer = self.namer(loader=CountingLoader(fails=True))
        self.assertEqual(namer.name(self.ctx(self.make_images(2))),
                         "2023-05-01..05-03 Тайланд")

    def test_generation_error_falls_back_to_template(self):
        class Boom(FakeVlm):
            def __call__(self, frames, prompt, max_new_tokens):
                raise RuntimeError("CUDA out of memory")

        namer = self.namer(loader=CountingLoader(model=Boom()))
        self.assertEqual(namer.name(self.ctx(self.make_images(1))),
                         "2023-05-01..05-03 Тайланд")

    def test_no_frames_falls_back_without_calling_the_model(self):
        loader = CountingLoader()
        namer = self.namer(loader=loader)
        self.assertEqual(namer.name(self.ctx()), "2023-05-01..05-03 Тайланд")
        self.assertEqual(loader.model.calls, [])

    def test_undecodable_frames_fall_back(self):
        broken = Path(self.tmp.name) / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        loader = CountingLoader()
        namer = self.namer(loader=loader)
        self.assertEqual(namer.name(self.ctx((str(broken), str(broken.parent / "gone.jpg")))),
                         "2023-05-01..05-03 Тайланд")
        self.assertEqual(loader.model.calls, [])

    def test_broken_dates_keep_the_current_name(self):
        """No date base — no name to build; the event keeps what it has (as template)."""
        namer = self.namer(loader=CountingLoader())
        self.assertIsNone(namer.name(EventContext(
            started_at="мусор", ended_at="мусор", city="Тайланд",
            sample_paths=self.make_images(1))))

    def test_garbage_answers(self):
        cases = [
            ("", "2023-05-01..05-03 Тайланд"),                       # empty
            ("   \n  ", "2023-05-01..05-03 Тайланд"),                # blank
            ('«Пляжный отдых»', "2023-05-01..05-03 Тайланд Пляжный отдых"),
            ("Свадьба в Праге.\nНа фото видно, что...",
             "2023-05-01..05-03 Тайланд Свадьба в Праге"),           # multiline
            ('Утро/вечер: "тест"', "2023-05-01..05-03 Тайланд Утро вечер тест"),
            ("x" * 200, "2023-05-01..05-03 Тайланд " + "x" * 80),    # length capped
        ]
        paths = self.make_images(1)
        for answer, expected in cases:
            with self.subTest(answer=answer[:20]):
                reset_shared_vlm()
                namer = self.namer(loader=CountingLoader(model=FakeVlm([answer])))
                self.assertEqual(namer.name(self.ctx(paths)), expected)


class NameEventsCase(VlmTestCase):
    """A DB with events and files — the stage seen end to end (no tests of its own)."""

    def setUp(self):
        super().setUp()
        self.cfg = cfg_with({"provider": "vlm"}, tmp=self.tmp.name)
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        super().tearDown()

    def add_file(self, taken_at, verdict=None, path=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', 'high', '2026-01-01')""",
            (path or f"/photos/img_{self._n}.jpg", taken_at))
        if verdict is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score,
                       updated_at, tier)
                   VALUES (?, ?, 'clip', 0.9, '2026-01-01', 'clip')""",
                (cur.lastrowid, verdict))
        self.conn.commit()
        return cur.lastrowid

    def add_event(self, started, ended, city=None, name="старое", manual=0,
                  file_ids=()):
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, place_city, name,
                   name_is_manual) VALUES (?,?,?,?,?)""",
            (started, ended, city, name, manual))
        for fid in file_ids:
            self.conn.execute(
                "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
                (cur.lastrowid, fid))
        self.conn.commit()
        return cur.lastrowid

    def event_name(self, event_id):
        return self.conn.execute(
            "SELECT name FROM events WHERE id = ?", (event_id,)).fetchone()["name"]


class TestNameEventsWithVlm(NameEventsCase):
    """The provider inside the stage: manual names, an empty media_class, one model."""

    def test_manual_name_is_never_overwritten(self):
        """Test 5: a name the user typed is untouchable, whatever the model says."""
        loader = CountingLoader()
        eid = self.add_event("2023-05-01T10:00:00", "2023-05-03T18:00:00", "Тайланд",
                             name="Свадьба Ани", manual=1,
                             file_ids=[self.add_file("2023-05-01T10:00:00")])
        stats = name_events(self.cfg, self.conn,
                            namer=self.namer(loader=loader))
        self.assertEqual(self.event_name(eid), "Свадьба Ани")
        self.assertEqual(stats.manual_kept, 1)
        self.assertEqual(loader.model.calls, [])  # not even shown to the model

    def test_empty_media_class_does_not_block_naming(self):
        """Test 7: junk is a later stage — on the first run there is nothing to filter by."""
        loader = CountingLoader()
        paths = self.make_images(2)
        eid = self.add_event(
            "2023-05-01T10:00:00", "2023-05-03T18:00:00", "Тайланд",
            file_ids=[self.add_file("2023-05-01T10:00:00", path=paths[0]),
                      self.add_file("2023-05-02T10:00:00", path=paths[1])])
        stats = name_events(self.cfg, self.conn, namer=self.namer(loader=loader))
        self.assertEqual(stats.renamed, 1)
        self.assertEqual(self.event_name(eid), "2023-05-01..05-03 Тайланд Поход в горы")
        self.assertEqual(loader.model.calls[0][0], 2)

    def test_names_are_written_for_every_auto_event(self):
        loader = CountingLoader()
        paths = self.make_images(2)
        first = self.add_event("2023-05-01T10:00:00", "2023-05-03T18:00:00", "Тайланд",
                               file_ids=[self.add_file("2023-05-01T10:00:00",
                                                       path=paths[0])])
        second = self.add_event("2023-06-01T10:00:00", "2023-06-01T18:00:00", "Прага",
                                file_ids=[self.add_file("2023-06-01T10:00:00",
                                                        path=paths[1])])
        stats = name_events(self.cfg, self.conn, namer=self.namer(loader=loader))
        self.assertEqual(stats.renamed, 2)
        self.assertEqual(loader.builds, 1)          # one model for the whole run
        self.assertEqual(len(loader.model.calls), 2)  # one call per event
        self.assertEqual(self.event_name(first),
                         "2023-05-01..05-03 Тайланд Поход в горы")
        self.assertEqual(self.event_name(second), "2023-06-01 Прага Поход в горы")


class TestVlmSection(VlmTestCase):
    """F102: the model and its input size come from the `vlm:` section, not from here.

    The namer and the deep junk tier run the SAME weights, so the size a frame is
    decoded to has to be the same number in both places — and the namer is the half of
    that pair nobody would think to check, because it makes one call per event and its
    cost never showed up in a profile.
    """

    def decodes_at(self, namer, paths):
        """The max_edge every frame of one naming call was decoded with."""
        seen: list[int] = []
        original = imaging.decode_rgb_preview

        def recording(path, mtime, size, max_edge):
            seen.append(max_edge)
            return original(path, mtime, size, max_edge=max_edge)

        imaging.decode_rgb_preview = recording
        try:
            namer.name(EventContext(**CTX_DATES, city="Пхукет", sample_paths=paths))
        finally:
            imaging.decode_rgb_preview = original
        return seen

    def test_frames_are_decoded_at_the_configured_size(self):
        namer = VlmNamer(naming_settings(cfg_with({"provider": "vlm"})),
                         loader=CountingLoader(), vlm=VlmConfig(max_edge=448))
        self.assertEqual(self.decodes_at(namer, self.make_images(2)), [448, 448])

    def test_without_a_section_the_896_that_shipped_is_used(self):
        namer = self.namer(loader=CountingLoader())
        self.assertEqual(self.decodes_at(namer, self.make_images(1)),
                         [DEFAULT_VLM_MAX_EDGE])

    def test_the_model_comes_from_the_section_when_there_is_one(self):
        loader = CountingLoader()
        namer = VlmNamer(naming_settings(cfg_with({"provider": "vlm"})), loader=loader,
                         vlm=VlmConfig(model="Qwen/from-section"))
        namer.name(EventContext(**CTX_DATES, city=None,
                                sample_paths=self.make_images(1)))
        self.assertEqual(loader.model_names, ["Qwen/from-section"])

    def test_the_stage_hands_the_section_to_the_provider(self):
        """make_namer is where name_events wires cfg.vlm into the provider."""
        cfg = cfg_with({"provider": "vlm"}, tmp=self.tmp.name,
                       vlm=VlmConfig(model="Qwen/wired", max_edge=672))
        asked: list[str] = []
        original = naming_module.shared_vlm

        def fake_shared(model_name, loader=None):
            asked.append(model_name)
            return FakeVlm()

        naming_module.shared_vlm = fake_shared
        self.addCleanup(setattr, naming_module, "shared_vlm", original)
        namer = make_namer(naming_settings(cfg), cfg.vlm)
        self.assertIsInstance(namer, VlmNamer)
        self.assertEqual(self.decodes_at(namer, self.make_images(1)), [672])
        self.assertEqual(asked, ["Qwen/wired"])


class TestSplitRuntime(VlmTestCase):
    """F101: the runtime hands out its halves — and stays a describe() for this module.

    The naming stage makes one call per EVENT (473 on a live collection, minutes), so
    it has nothing to overlap and must simply not notice the split. The junk stage,
    which makes one call per frame, is where the halves are used — see
    tests/test_junk_vlm_pipeline.py.
    """

    def runtime(self, answer="Поход в горы"):
        calls = {"prepare": [], "generate": []}

        def prepare(frames, prompt):
            calls["prepare"].append((len(frames), prompt))
            return ("inputs", len(frames))

        def generate(prepared, max_new_tokens):
            calls["generate"].append((prepared, max_new_tokens))
            return answer

        return SplitVlm(prepare=prepare, generate=generate), calls

    def test_calling_it_is_generate_of_prepare(self):
        runtime, calls = self.runtime()
        self.assertEqual(runtime([1, 2], "prompt", 8), "Поход в горы")
        self.assertEqual(calls["prepare"], [(2, "prompt")])
        self.assertEqual(calls["generate"], [(("inputs", 2), 8)])

    def test_the_namer_does_not_notice_the_split(self):
        runtime, calls = self.runtime()
        namer = self.namer(loader=lambda _model: runtime)
        name = namer.name(EventContext(**CTX_DATES, city="Пхукет",
                                       sample_paths=self.make_images(3)))
        self.assertEqual(name, "2023-05-01..05-03 Пхукет Поход в горы")
        self.assertEqual(len(calls["generate"]), 1)  # still ONE call per event
        self.assertEqual(calls["prepare"][0][0], 3)  # ...with all three frames in it


class TestThreadLocalProcessors(unittest.TestCase):
    """F101: a processor is mutable state — one per preparation thread, built once.

    The weights are shared (read-only, 20.5 GB); the processor is not — its tokenizer
    writes truncation/padding onto itself at every call. This is the F73 Reader rule
    and the F12.1 session rule applied to the third model in the project.
    """

    def processors(self, factory=None):
        return ThreadLocalProcessors("shared", factory)

    def test_without_a_factory_everyone_gets_the_shared_one(self):
        pool = self.processors()
        seen = []
        threads = [threading.Thread(target=lambda: seen.append(pool.get()))
                   for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(seen, ["shared"] * 3)
        self.assertEqual(pool.built, 0)

    def test_one_processor_per_thread_reused_across_calls(self):
        """One processor per thread that ran, shared with no other thread.

        The results used to be keyed by `threading.get_ident()`, which is only unique
        among *live* threads: on Linux a thread often finishes inside `start()` and the
        next one is handed the same ident, so all four entries collapsed into one and
        the test failed with "1 != 4" on a pool that was behaving correctly. The Thread
        object stays unique for the whole run, so counting threads is no longer a race —
        what is asserted is the invariant (as many processors as threads, none shared),
        not how the interpreter happened to schedule them.
        """
        built = []
        pool = self.processors(lambda: built.append(len(built)) or f"proc_{len(built)}")
        got: dict[threading.Thread, set] = {}

        def work():
            got[threading.current_thread()] = {pool.get() for _ in range(5)}

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(got), len(threads))                 # every thread reported
        self.assertEqual(pool.built, len(got))                   # not one per call
        self.assertTrue(all(len(v) == 1 for v in got.values()))  # stable per thread
        self.assertEqual(
            len({next(iter(v)) for v in got.values()}), len(got))  # never shared

    def test_the_calling_thread_gets_its_own_too(self):
        pool = self.processors(lambda: object())
        self.assertIs(pool.get(), pool.get())
        self.assertEqual(pool.built, 1)


class TestProcessorIsFast(unittest.TestCase):
    """F101, lever 1: `use_fast=True` is a request — the report must print the answer."""

    class Qwen2VLImageProcessorFast:
        pass

    class Qwen2VLImageProcessor:
        pass

    def test_fast_image_processor_is_recognized(self):
        self.assertTrue(processor_is_fast(
            types.SimpleNamespace(image_processor=self.Qwen2VLImageProcessorFast())))

    def test_slow_image_processor_is_recognized(self):
        self.assertFalse(processor_is_fast(
            types.SimpleNamespace(image_processor=self.Qwen2VLImageProcessor())))

    def test_processor_without_an_image_processor_answers_for_itself(self):
        self.assertTrue(processor_is_fast(self.Qwen2VLImageProcessorFast()))
        self.assertFalse(processor_is_fast(self.Qwen2VLImageProcessor()))


class TestSamplePathsExcludesJunk(NameEventsCase):
    """Test 6: at the _sample_paths level, so the cloud provider is covered too."""

    def sample_paths(self, verdicts):
        """One event holding one file per verdict; returns what a provider would see."""
        seen = []

        class Recorder:
            def name(self, ctx):
                seen.append(ctx.sample_paths)
                return None

        ids = [self.add_file(f"2023-05-01T1{i}:00:00", verdict=v)
               for i, v in enumerate(verdicts)]
        self.add_event("2023-05-01T10:00:00", "2023-05-03T18:00:00", "Тайланд",
                       file_ids=ids)
        name_events(self.cfg, self.conn, namer=Recorder())
        return seen[0]

    def test_documents_and_screenshots_never_reach_a_provider(self):
        paths = self.sample_paths(["photo", "document", "screenshot", "meme",
                                   "product", None])
        self.assertEqual([Path(p).name for p in paths], ["img_1.jpg", "img_6.jpg"])

    def test_event_made_only_of_documents_shows_nothing(self):
        """The case the filter exists for: a folder must not be named after a scan."""
        self.assertEqual(self.sample_paths(["document", "document"]), ())


if __name__ == "__main__":
    unittest.main()
