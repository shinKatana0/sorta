"""F250: a machine with no NVIDIA card is asked nothing about torch.

The owner's VM, 2026-08-24, after F246 had moved the torch import into a child process:

    startup step=environment elapsed=0.661
    startup step=gpu started         10:31:28.198
    startup step=gpu elapsed=222.576 10:34:52.425

Three minutes forty-two seconds of a launch, on a machine with no GPU in it, to learn
what `nvidia-smi` answers in milliseconds. All three problems `warn_if_gpu_mismatch`
names are about a card that is THERE, so the cheap question now goes first.

What is pinned here, in the order of the acceptance criteria:

1. no card — torch is not imported on the start path AND no child process is started to
   import it there either, asked of an observed import and of the calls to `launch.run`;
2. no card — no warning, which is a decision and not an accident (`TestTheDecision`);
3. a card — the three flags are computed and the warning arrives exactly as before;
4. `sorta doctor` still reads both stacks with no card in the machine;
5. the price: no child means no interpreter, which is the whole of the measurement;
6. no card and no `nvidia-smi` are one answer, because they are one probe.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import unittest
from typing import Any, Iterator
from unittest import mock

from sorta import diagnostics, launch, tray, ui

from tests.test_diagnostics import CUDA_PROVIDERS, fake_ort, fake_torch, patched
from tests.test_the_diagnostics_do_not_starve_the_server import (
    _completed,
    fresh_state,
    tripwire,
    without_torch,
)

LOGGER_NAME = "sorta.diagnostics"

# Two ways of having no card, which are the same answer (criterion 6).
NO_CARD = "no card"
NO_NVIDIA_SMI = "no nvidia-smi"
A_CARD = "NVIDIA GeForce RTX 5090, 581.15"

# The state that WOULD have been warned about: onnxruntime offers CUDA, torch is a CPU
# build. On a machine with a card it is the F63 mismatch; on one without, there is no
# CUDA to execute and nothing to repair.
_CPU_TORCH: dict[str, Any] = {
    "torch_version": "2.13.0+cpu",
    "torch_cuda_available": False,
    "torch_device_name": None,
    "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}
_NEITHER_STACK_ON_CUDA = {**_CPU_TORCH, "ort_providers": ["CPUExecutionProvider"]}


@contextlib.contextmanager
def machine(hardware: str, facts: dict[str, Any] | None = None) -> Iterator[mock.Mock]:
    """This machine's `nvidia-smi` and this machine's probe child, without either.

    Patched at `sorta.launch.run` (F228's single door) and not at the two callers: what
    has to be proven is that no child is STARTED, and a test that patched
    `probe_torch_facts` would be answering its own question.
    """
    def run(command: Any, **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        if tuple(command) == tuple(diagnostics._NVIDIA_SMI_CMD):
            if hardware == NO_NVIDIA_SMI:
                raise FileNotFoundError(2, "nvidia-smi")
            return _completed("" if hardware == NO_CARD else hardware + "\n")
        return _completed(json.dumps(facts if facts is not None else _CPU_TORCH))

    with mock.patch.object(launch, "run", side_effect=run) as patched:
        yield patched


def probes(run: mock.Mock) -> list[Any]:
    """The calls that started an interpreter — the seconds-to-minutes half."""
    return [call.args[0] for call in run.call_args_list
            if tuple(call.args[0]) != tuple(diagnostics._NVIDIA_SMI_CMD)]


def smi_calls(run: mock.Mock) -> list[Any]:
    return [call.args[0] for call in run.call_args_list
            if tuple(call.args[0]) == tuple(diagnostics._NVIDIA_SMI_CMD)]


# --- 1: nothing heavy is asked, in this process or in any other ------------------------


class TestACardLessMachinePaysForNothing(unittest.TestCase):
    """Criterion 1, the one the feature stands on."""

    def test_the_guard_imports_no_torch_and_starts_no_child(self):
        with tripwire() as wire, machine(NO_CARD) as run:
            self.assertFalse(diagnostics.warn_if_gpu_mismatch())
        self.assertEqual(wire.asked, [])
        self.assertEqual(probes(run), [],
                         "старт снова платит за интерпретатор ради вопроса без карты")
        # And the cheap question really was asked: a guard that quietly stopped probing
        # would pass both assertions above by doing nothing at all.
        self.assertEqual(len(smi_calls(run)), 1)

    def test_the_whole_start_up_starts_no_child_either(self):
        """The step that cost 222.6 s is `startup step=gpu`, so the proof has to be taken
        where the launch actually runs it and not only at the function under it."""
        state = fresh_state()
        with mock.patch.object(ui.common, "_startup_state", state):
            state.expect()
            with tripwire() as wire, machine(NO_CARD) as run, \
                    self.assertLogs("sorta.tray", level="INFO"):
                tray._finish_startup()
        self.assertEqual(wire.asked, [])
        self.assertEqual(probes(run), [])
        self.assertEqual([done["step"] for done in state.snapshot()["done"]],
                         [ui.STARTUP_ENVIRONMENT, ui.STARTUP_GPU, ui.STARTUP_GEO])

    def test_the_card_is_asked_about_once_per_call(self):
        """Two probes of one binary is how two answers start disagreeing (F230), and the
        second one would also be three more seconds of a launch."""
        with without_torch(), machine(A_CARD) as run:
            diagnostics.warn_if_gpu_mismatch()
        self.assertEqual(len(smi_calls(run)), 1)
        self.assertEqual(len(probes(run)), 1)


class TestTheDecision(unittest.TestCase):
    """Criterion 2. Dropping the F63 mismatch warning on a machine with no card is a
    product decision: today it could fire, tomorrow it will not. It is written down as a
    test so that the next reading of `mismatch` does not restore it as a bug fix."""

    def test_a_mismatch_without_a_card_is_not_worth_a_word(self):
        with without_torch(), machine(NO_CARD, _CPU_TORCH), \
                self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            self.assertFalse(diagnostics.warn_if_gpu_mismatch())

    def test_the_same_facts_with_a_card_do_warn(self):
        """The other half of the decision: what is dropped is the ABSENCE of the card,
        not the diagnosis."""
        with without_torch(), machine(A_CARD, _CPU_TORCH), \
                self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            self.assertTrue(diagnostics.warn_if_gpu_mismatch())
        self.assertIn("2.13.0+cpu", logs.records[0].getMessage())

    def test_a_health_handed_in_is_still_believed(self):
        """A caller with the answer already in its hands is not sent to nvidia-smi: the
        branch belongs to the launch that would have paid for the probe."""
        health = diagnostics.GpuHealth(
            torch_version="2.13.0+cpu", torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=("CUDAExecutionProvider",), gpu_present=False,
            install_kind="checkout")
        with mock.patch.object(launch, "run", side_effect=AssertionError("probed")):
            with self.assertLogs(LOGGER_NAME, level="WARNING"):
                self.assertTrue(diagnostics.warn_if_gpu_mismatch(health))


# --- 6: no card and no nvidia-smi are one answer ---------------------------------------


class TestNoBinaryReadsAsNoCard(unittest.TestCase):
    """Criterion 6. `nvidia_gpu_present` answers through `nvidia-smi`, so a machine with
    a card but without that binary on PATH is silent too — the accepted price of the
    wizard (F230) and the launch reading one probe."""

    def outcome(self, hardware: str) -> tuple[bool, list[Any]]:
        with without_torch(), machine(hardware, _CPU_TORCH) as run:
            with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
                warned = diagnostics.warn_if_gpu_mismatch()
        return warned, probes(run)

    def test_both_ways_of_having_no_card_behave_identically(self):
        self.assertEqual(self.outcome(NO_CARD), self.outcome(NO_NVIDIA_SMI))
        self.assertEqual(self.outcome(NO_NVIDIA_SMI), (False, []))

    def test_the_probe_itself_says_the_same_of_both(self):
        for hardware in (NO_CARD, NO_NVIDIA_SMI):
            with self.subTest(hardware=hardware), machine(hardware):
                self.assertFalse(diagnostics.nvidia_gpu_present())


# --- 3: a machine with a card is where it was -------------------------------------------


class TestACardChangesNothing(unittest.TestCase):
    """Criterion 3: the feature is about the machine that has no card."""

    def health(self, facts: dict[str, Any]) -> diagnostics.GpuHealth:
        with without_torch(), machine(A_CARD, facts):
            health = diagnostics.current_gpu_health()
        assert health is not None
        return health

    def test_the_f63_mismatch_is_still_computed(self):
        health = self.health(_CPU_TORCH)
        self.assertTrue(health.gpu_present)
        self.assertTrue(health.mismatch)
        self.assertTrue(health.torch_ignores_gpu)
        self.assertFalse(health.ort_ignores_gpu)
        self.assertTrue(health.degraded)

    def test_both_f76_cases_are_still_computed(self):
        health = self.health(_NEITHER_STACK_ON_CUDA)
        self.assertFalse(health.mismatch)
        self.assertTrue(health.torch_ignores_gpu)
        self.assertTrue(health.ort_ignores_gpu)
        self.assertEqual(len(health.problems), 2)

    def test_a_healthy_gpu_machine_is_still_silent(self):
        facts = {**_CPU_TORCH, "torch_version": "2.13.0+cu130",
                 "torch_cuda_available": True, "torch_device_name": "RTX 5090"}
        with without_torch(), machine(A_CARD, facts), \
                self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            self.assertFalse(diagnostics.warn_if_gpu_mismatch())

    def test_the_hardware_answer_is_not_asked_twice_of_the_child(self):
        """`current_gpu_health` now takes the card it was told about; passing None again
        would be the second nvidia-smi call this feature exists to avoid."""
        with without_torch(), machine(NO_CARD) as run:
            health = diagnostics.current_gpu_health(gpu_present=True)
        assert health is not None
        self.assertTrue(health.gpu_present)
        self.assertEqual(smi_calls(run), [])


# --- 4: `sorta doctor` is untouched -----------------------------------------------------


class TestTheDoctorStillAsksEverything(unittest.TestCase):
    """Criterion 4. `sorta doctor` answers "what is installed here", not "should this be
    warned about" — it is allowed the import and must keep paying for it."""

    def test_it_reads_torch_on_a_machine_with_no_card(self):
        with tripwire() as wire, machine(NO_CARD):
            health = diagnostics.gpu_health(install_kind="checkout")
        self.assertEqual(wire.asked[:1], ["torch"])
        self.assertIn("onnxruntime", wire.asked)
        self.assertFalse(health.gpu_present)

    def test_its_summary_still_names_both_stacks_in_both_cases(self):
        for hardware, present in ((NO_CARD, "no"), (A_CARD, "yes")):
            with self.subTest(hardware=hardware):
                with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)), \
                        machine(hardware):
                    summary = diagnostics.gpu_health(install_kind="checkout").summary
                self.assertIn("torch: 2.13.0+cpu", summary)
                self.assertIn("onnxruntime providers: CUDAExecutionProvider", summary)
                self.assertIn(f"NVIDIA GPU in the machine (nvidia-smi): {present}",
                              summary)
                # The F63 mismatch is still stated on a machine with no card: the doctor
                # reports what is installed, and only the WARNING was dropped.
                self.assertIn("mismatch: YES", summary)



class TestTheProbeIsGivenTimeToAnswer(unittest.TestCase):
    """The floor under `_NVIDIA_SMI_TIMEOUT_S`, and why it is not a style preference.

    At 3.0 s this machine's RTX 5090 read as NO CARD: measured 2026-08-24, the first
    `nvidia-smi` after boot takes 3738 ms and warm ones 1547-1636 ms, and the launch asks
    exactly once, cold. Since F250 that answer decides whether the GPU question is asked
    at all, and F230's wizard reads the same probe before offering 2.5 GB of CUDA wheels,
    so a false "no card" is wrong in two places at once.
    """

    def test_the_timeout_leaves_room_for_a_cold_call(self):
        self.assertGreaterEqual(diagnostics._NVIDIA_SMI_TIMEOUT_S, 10.0)

    def test_it_is_still_bounded(self):
        """A half-installed driver can hang on this call; the wait has to end."""
        self.assertLess(diagnostics._NVIDIA_SMI_TIMEOUT_S, 60.0)


if __name__ == "__main__":
    unittest.main()
