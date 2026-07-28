"""F105: the attention implementation and the batch — the two levers, without a GPU.

Both levers are the SAME MATHEMATICS done differently, so the bar the measurement holds
them to is that the verdicts match exactly. That leaves this file a precise job: pin
down the plumbing where a batch can go wrong silently and produce plausible WRONG
answers, because that is the failure mode a verdict comparison would then blame on the
kernel.

What is checked here:

* the requested attention implementation reaches transformers, in the dictionary form
  too (the visual tower has to be switchable without touching the language half) — and
  without a request the loader makes exactly the call that shipped;
* a batch of N frames comes back as N answers IN INPUT ORDER;
* the padding of a batch is on the LEFT and the attention mask comes from the processor
  (a right-padded batch does not fail, it answers out of padding);
* a batch of one is not a special case — it is literally the pre-F105 call;
* a frame that breaks the batch costs that frame only.

Neither transformers nor torch is installed in the gate's environment, so both are
faked at the import the loader does — which is also the only way to see WHAT was passed
to `from_pretrained`.
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from typing import Any

import numpy as np
from PIL import Image

from sorta import naming
from sorta.naming import (
    BatchVlm,
    SplitVlm,
    attention_kernels,
    attn_implementation,
    batched_describe,
    load_qwen,
    pad_generation_left,
    processor_pads_left,
    qwen_runtime,
)

PROMPT = "classify this"


def frame(mark: int) -> Image.Image:
    """A frame that says which one it is: the width IS the identity of the image.

    That is what makes an order test possible without a model — the fake processor
    carries the widths through as pixels and the fake model answers with them, so a
    shuffled batch produces visibly shuffled answers instead of quietly plausible ones.
    """
    return Image.new("RGB", (mark, 4), (0, 0, 0))


class FakeBatch(dict):
    """What a processor returns: a dict of tensors that knows how to move to a device."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)
        self.device: str | None = None

    def to(self, device: str) -> FakeBatch:
        self.device = device
        return self


class FakeProcessor:
    """Records every call, and answers with tensors shaped like the real ones."""

    PROMPT_TOKENS = 3

    def __init__(self) -> None:
        self.tokenizer = types.SimpleNamespace(padding_side="right")
        self.calls: list[dict[str, Any]] = []
        self.batches: list[FakeBatch] = []
        self.decoded: list[Any] = []

    def apply_chat_template(self, messages: Any, tokenize: bool,
                            add_generation_prompt: bool) -> str:
        content = messages[0]["content"]
        images = [c["image"] for c in content if c["type"] == "image"]
        text = next(c["text"] for c in content if c["type"] == "text")
        return f"[{len(images)} imgs] {text}"

    def __call__(self, text: list[str], images: list[Image.Image],
                 return_tensors: str, padding: bool | None = None,
                 padding_side: str | None = None) -> FakeBatch:
        self.calls.append({"text": list(text), "images": list(images),
                           "padding": padding, "padding_side": padding_side,
                           "return_tensors": return_tensors})
        rows = len(text)
        # The images arrive FLAT, as the real Qwen processor takes them, and are split
        # back over the sequences by the placeholder count of each one — the mark of a
        # sequence is the width of its first frame.
        widths, rest = [], list(images)
        for line in text:
            count = int(line.split(" ")[0].lstrip("["))
            widths.append([rest[0].width])
            rest = rest[count:]
        batch = FakeBatch({
            "input_ids": np.zeros((rows, self.PROMPT_TOKENS), dtype=int),
            # The mask the model must be given: built HERE, by the processor.
            "attention_mask": np.ones((rows, self.PROMPT_TOKENS), dtype=int),
            "pixel_values": np.array(widths, dtype=int),
        })
        self.batches.append(batch)
        return batch

    def batch_decode(self, gen_ids: Any, skip_special_tokens: bool = True) -> list[str]:
        self.decoded.append(gen_ids)
        return [f"answer-{int(row[0])} " for row in gen_ids]


class FakeModel:
    """A model whose answer is the frame it was shown — so order is observable."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = types.SimpleNamespace(
            _attn_implementation="sdpa",
            vision_config=types.SimpleNamespace(_attn_implementation="eager"))

    def eval(self) -> FakeModel:
        return self

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return np.concatenate([kwargs["input_ids"], kwargs["pixel_values"]], axis=1)


@contextlib.contextmanager
def fake_ml(model: Any = None, processor: Any = None, cuda: bool = False):
    """`import torch` / `from transformers import ...` answered with doubles.

    Restored on the way out — torch really is installed in the gate's environment, and
    a test that leaves a stub behind breaks whatever runs next.
    """
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda)
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]

    built = model if model is not None else FakeModel()
    loads: list[dict[str, Any]] = []
    processors: list[dict[str, Any]] = []

    class Qwen2_5_VLForConditionalGeneration:  # noqa: N801 — the transformers name
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> Any:
            loads.append({"model_name": model_name, **kwargs})
            return built

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> Any:
            processors.append({"model_name": model_name, **kwargs})
            return processor if processor is not None else FakeProcessor()

    transformers = types.ModuleType("transformers")
    transformers.Qwen2_5_VLForConditionalGeneration = (  # type: ignore[attr-defined]
        Qwen2_5_VLForConditionalGeneration)
    transformers.AutoProcessor = AutoProcessor  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in ("torch", "transformers")}
    sys.modules["torch"], sys.modules["transformers"] = torch, transformers
    try:
        yield types.SimpleNamespace(loads=loads, processors=processors, model=built)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestAttnImplementation(unittest.TestCase):
    """Test 1: the request the loader is given, in both shapes transformers takes."""

    def test_nothing_asked_is_nothing_passed(self):
        self.assertIsNone(attn_implementation())

    def test_one_name_for_the_whole_model(self):
        self.assertEqual(attn_implementation("sdpa"), "sdpa")

    def test_the_visual_tower_alone(self):
        """The lever of the feature: the tower moves, the language half is left alone."""
        self.assertEqual(attn_implementation(vision="sdpa"), {"vision_config": "sdpa"})

    def test_the_two_halves_separately(self):
        self.assertEqual(attn_implementation("eager", vision="sdpa"),
                         {"vision_config": "sdpa", "": "eager"})


class TestAttentionKernels(unittest.TestCase):
    """What the model got is READ, not assumed — a downgrade must be visible."""

    def test_both_halves_are_reported(self):
        self.assertEqual(attention_kernels(FakeModel()),
                         {"language": "sdpa", "vision": "eager"})

    def test_a_model_without_a_tower_says_so_instead_of_crashing(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager"))
        self.assertEqual(attention_kernels(model), {"language": "eager", "vision": "?"})


class TestLoadQwen(unittest.TestCase):
    """Test 1 and test 2: the request reaches transformers, the default is untouched."""

    def test_without_a_request_nothing_is_passed(self):
        """The regression: not passing the argument and passing None are not the same."""
        with fake_ml() as ml:
            model, processor, device = load_qwen("Qwen/test")
        self.assertNotIn("attn_implementation", ml.loads[0])
        self.assertEqual(ml.loads[0]["model_name"], "Qwen/test")
        self.assertEqual(device, "cpu")
        self.assertIs(model, ml.model)
        self.assertIsInstance(processor, FakeProcessor)
        self.assertEqual(ml.processors[0]["use_fast"], True)

    def test_a_name_reaches_from_pretrained(self):
        with fake_ml() as ml:
            load_qwen("Qwen/test", attn="sdpa")
        self.assertEqual(ml.loads[0]["attn_implementation"], "sdpa")

    def test_the_dictionary_form_reaches_from_pretrained(self):
        with fake_ml() as ml:
            load_qwen("Qwen/test", attn=attn_implementation(vision="sdpa"))
        self.assertEqual(ml.loads[0]["attn_implementation"], {"vision_config": "sdpa"})

    def test_the_slow_processor_can_still_be_asked_for(self):
        with fake_ml() as ml:
            load_qwen("Qwen/test", use_fast=False)
        self.assertEqual(ml.processors[0]["use_fast"], False)

    def test_cuda_picks_fp16(self):
        with fake_ml(cuda=True) as ml:
            _model, _processor, device = load_qwen("Qwen/test")
        self.assertEqual(device, "cuda")
        self.assertEqual(ml.loads[0]["torch_dtype"], "float16")


class RuntimeCase(unittest.TestCase):
    """A Qwen runtime over the fakes — the halves, batched and not."""

    def setUp(self):
        self.processor = FakeProcessor()
        self.model = FakeModel()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(fake_ml(model=self.model, processor=self.processor))
        self.addCleanup(self.stack.close)
        self.runtime = qwen_runtime(self.model, self.processor, "cuda")


class TestBatchedRuntime(RuntimeCase):
    """Tests 3 and 5: N answers in the input order, and a batch of one is the old path."""

    def test_the_runtime_offers_both_interfaces(self):
        self.assertIsInstance(self.runtime, BatchVlm)
        self.assertIsInstance(self.runtime, SplitVlm)  # ...and junk still recognizes it

    def test_n_frames_give_n_answers_in_input_order(self):
        marks = [11, 12, 13, 14]
        prepared = self.runtime.prepare_batch([[frame(m)] for m in marks], PROMPT)
        answers = self.runtime.generate_batch(prepared, 8)
        self.assertEqual(answers, [f"answer-{m}" for m in marks])

    def test_a_reordered_batch_gives_reordered_answers(self):
        """The failure this test exists for is answers that look fine but belong to
        another file: the same frames in another order must move the answers with them."""
        marks = [14, 11, 13, 12]
        answers = self.runtime.generate_batch(
            self.runtime.prepare_batch([[frame(m)] for m in marks], PROMPT), 8)
        self.assertEqual(answers, [f"answer-{m}" for m in marks])

    def test_a_group_may_hold_several_frames(self):
        prepared = self.runtime.prepare_batch(
            [[frame(11), frame(12)], [frame(13)]], PROMPT)
        self.assertEqual([call["text"] for call in self.processor.calls],
                         [[f"[2 imgs] {PROMPT}", f"[1 imgs] {PROMPT}"]])
        self.assertEqual(len(self.runtime.generate_batch(prepared, 8)), 2)

    def test_a_batch_of_one_is_the_call_that_shipped(self):
        """No padding argument reaches the processor when there is nothing to pad."""
        self.runtime.prepare([frame(11)], PROMPT)
        self.runtime.prepare_batch([[frame(11)]], PROMPT)
        single, batch_of_one = self.processor.calls
        self.assertEqual(single, batch_of_one)
        self.assertIsNone(single["padding"])
        self.assertIsNone(single["padding_side"])

    def test_generate_of_a_batch_of_one_is_the_unbatched_answer(self):
        prepared = self.runtime.prepare([frame(11)], PROMPT)
        self.assertEqual(self.runtime.generate(prepared, 8), "answer-11")
        self.assertEqual(self.runtime.generate_batch(prepared, 8), ["answer-11"])

    def test_calling_the_runtime_is_still_describe(self):
        self.assertEqual(self.runtime([frame(11)], PROMPT, 8), "answer-11")

    def test_the_answer_is_stripped(self):
        """batch_decode answers with a trailing space — the caller must not see it."""
        self.assertNotIn(" ", self.runtime([frame(11)], PROMPT, 8))


class TestBatchPadding(RuntimeCase):
    """The padding side and the attention mask: where a batch goes silently wrong."""

    def test_a_real_batch_is_padded_on_the_left(self):
        self.runtime.prepare_batch([[frame(11)], [frame(12)]], PROMPT)
        call = self.processor.calls[0]
        self.assertTrue(call["padding"])
        self.assertEqual(call["padding_side"], "left")

    def test_the_tokenizer_itself_is_set_left(self):
        """The argument may be ignored by an older processor; the tokenizer may not."""
        self.assertFalse(processor_pads_left(self.processor))
        self.runtime.prepare_batch([[frame(11)], [frame(12)]], PROMPT)
        self.assertTrue(processor_pads_left(self.processor))

    def test_the_mask_the_model_gets_is_the_processors_own(self):
        prepared = self.runtime.prepare_batch([[frame(11)], [frame(12)]], PROMPT)
        self.runtime.generate_batch(prepared, 8)
        given = self.model.calls[0]["attention_mask"]
        self.assertIs(given, self.processor.batches[0]["attention_mask"])

    def test_generation_is_greedy_and_moved_to_the_device(self):
        prepared = self.runtime.prepare_batch([[frame(11)], [frame(12)]], PROMPT)
        self.runtime.generate_batch(prepared, 8)
        self.assertEqual(self.model.calls[0]["do_sample"], False)
        self.assertEqual(self.model.calls[0]["max_new_tokens"], 8)
        self.assertEqual(prepared.device, "cuda")

    def test_only_the_generated_tail_is_decoded(self):
        """One offset for the whole batch is correct only because the padding is left."""
        self.runtime.generate_batch(
            self.runtime.prepare_batch([[frame(11)], [frame(12)]], PROMPT), 8)
        self.assertEqual(self.processor.decoded[0].shape,
                         (2, 1))  # the prompt tokens are cut off, one answer token left


class TestPadGenerationLeft(unittest.TestCase):
    def test_a_processor_without_a_tokenizer_is_not_a_crash(self):
        thing = types.SimpleNamespace()
        self.assertIs(pad_generation_left(thing), thing)

    def test_an_object_that_is_its_own_tokenizer_answers_for_itself(self):
        self.assertTrue(processor_pads_left(types.SimpleNamespace(padding_side="left")))
        self.assertFalse(processor_pads_left(types.SimpleNamespace()))


class FakeBatchVlm(BatchVlm):
    """A runtime whose batch can be told to fail — the rest answers by frame width."""


def batch_runtime(fail_prepare=(), fail_generate=(), answers=None) -> BatchVlm:
    """A BatchVlm over marks: `fail_*` are frame widths that poison their batch."""
    marks = lambda groups: [g[0].width for g in groups]  # noqa: E731

    def prepare_batch(groups, prompt):
        got = marks(groups)
        if any(m in fail_prepare for m in got):
            raise RuntimeError("процессор подавился кадром")
        return got

    def generate_batch(prepared, max_new_tokens):
        if any(m in fail_generate for m in prepared):
            raise RuntimeError("CUDA out of memory")
        if answers is not None:
            return list(answers)
        return [f"answer-{m}" for m in prepared]

    def prepare(frames, prompt):
        return prepare_batch([frames], prompt)

    def generate(prepared, max_new_tokens):
        return generate_batch(prepared, max_new_tokens)[0]

    return FakeBatchVlm(prepare=prepare, generate=generate,
                        prepare_batch=prepare_batch, generate_batch=generate_batch)


class TestBatchedDescribe(unittest.TestCase):
    """Test 4: one bad frame costs one frame — and the positions never shift."""

    def groups(self, *marks):
        return [[frame(m)] for m in marks]

    def test_answers_come_back_in_input_order(self):
        answers = batched_describe(batch_runtime(), self.groups(13, 11, 12), PROMPT, 8)
        self.assertEqual(answers, ["answer-13", "answer-11", "answer-12"])

    def test_a_frame_that_breaks_the_batch_costs_only_itself(self):
        runtime = batch_runtime(fail_prepare={12})
        answers = batched_describe(runtime, self.groups(11, 12, 13), PROMPT, 8)
        self.assertEqual(answers[0], "answer-11")
        self.assertEqual(answers[2], "answer-13")
        self.assertIsInstance(answers[1], RuntimeError)

    def test_a_batch_that_will_not_generate_is_retried_frame_by_frame(self):
        runtime = batch_runtime(fail_generate={13})
        answers = batched_describe(runtime, self.groups(11, 13), PROMPT, 8)
        self.assertEqual(answers[0], "answer-11")
        self.assertIsInstance(answers[1], RuntimeError)

    def test_a_model_that_answers_the_wrong_number_of_times_is_not_trusted(self):
        """Nothing can be said about which answer belongs to which frame — so the batch
        is thrown away and every frame is asked again, one at a time."""
        runtime = batch_runtime(answers=["answer-11"])
        answers = batched_describe(runtime, self.groups(11, 12), PROMPT, 8)
        self.assertEqual(answers, ["answer-11", "answer-11"])

    def test_a_group_without_frames_never_reaches_the_model(self):
        runtime = batch_runtime()
        answers = batched_describe(runtime, [[], [frame(12)], []], PROMPT, 8)
        self.assertIsInstance(answers[0], ValueError)
        self.assertEqual(answers[1], "answer-12")
        self.assertIsInstance(answers[2], ValueError)

    def test_nothing_to_do_is_not_a_call(self):
        self.assertEqual(batched_describe(batch_runtime(), [], PROMPT, 8), [])

    def test_a_runtime_without_the_batched_halves_still_answers(self):
        """A test double, an older runtime: the groups simply go one at a time."""
        calls = []

        def describe(frames, prompt, max_new_tokens):
            calls.append(len(frames))
            return f"answer-{frames[0].width}"

        answers = batched_describe(describe, self.groups(11, 12), PROMPT, 8)
        self.assertEqual(answers, ["answer-11", "answer-12"])
        self.assertEqual(calls, [1, 1])

    def test_a_plain_runtime_that_raises_loses_only_that_group(self):
        def describe(frames, prompt, max_new_tokens):
            if frames[0].width == 12:
                raise RuntimeError("нет VRAM")
            return "ok"

        answers = batched_describe(describe, self.groups(11, 12), PROMPT, 8)
        self.assertEqual(answers[0], "ok")
        self.assertIsInstance(answers[1], RuntimeError)


class TestTheProductPathDidNotMove(unittest.TestCase):
    """The brief's hard requirement: nothing about the shipped path changed (F105)."""

    def test_the_namer_still_makes_one_call_per_event(self):
        seen = []

        def describe(frames, prompt, max_new_tokens):
            seen.append(len(frames))
            return "Поход в горы"

        naming.reset_shared_vlm()
        self.addCleanup(naming.reset_shared_vlm)
        runtime = naming.shared_vlm("Qwen/test", lambda _name: describe)
        self.assertIs(runtime, describe)
        self.assertEqual(runtime([frame(11), frame(12)], PROMPT, 8), "Поход в горы")
        self.assertEqual(seen, [2])


if __name__ == "__main__":
    unittest.main()
