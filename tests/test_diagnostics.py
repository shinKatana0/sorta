"""F63/F76: GPU-health guard — torch/onnxruntime and nvidia-smi are mocked.

No real GPU, no real `nvidia-smi` call: the probe is injected everywhere, and
`gpu_health(gpu_present=...)` states the hardware instead of detecting it.
"""
import dataclasses
import json
import logging
import subprocess
import sys
import types
import unittest
from unittest import mock

from sorta.diagnostics import GpuHealth, gpu_health, nvidia_gpu_present, warn_if_gpu_mismatch

LOGGER_NAME = "sorta.diagnostics"


def fake_torch(
    version: str = "2.13.0+cpu",
    *,
    cuda_available: bool = False,
    device_name: str = "NVIDIA GeForce RTX 5090",
) -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.__version__ = version
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda index: device_name,
    )
    return mod


def fake_ort(providers: list[str]) -> types.ModuleType:
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: list(providers)
    return mod


def patched(torch_mod: object, ort_mod: object) -> mock._patch_dict:
    """Substitute both stacks in sys.modules (the lazy imports pick them up).

    A None value makes `import x` raise ImportError — that is the "not installed" case.
    """
    return mock.patch.dict(sys.modules, {"torch": torch_mod, "onnxruntime": ort_mod})


def smi(stdout: str = "NVIDIA GeForce RTX 5090\n", returncode: int = 0):
    """A canned `nvidia-smi` result to inject into the probe."""
    return lambda: subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_PROVIDERS = ["CPUExecutionProvider"]

FIX_ORT = "--force-reinstall --no-deps onnxruntime-gpu"


class TestGpuHealth(unittest.TestCase):
    def test_mismatch_when_ort_has_cuda_and_torch_is_cpu(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)):
            health = gpu_health(gpu_present=False)
        self.assertTrue(health.mismatch)
        self.assertTrue(health.ort_has_cuda)
        self.assertFalse(health.torch_cuda_available)
        self.assertEqual(health.torch_version, "2.13.0+cpu")
        self.assertIsNone(health.torch_device_name)
        self.assertEqual(health.ort_providers, tuple(CUDA_PROVIDERS))

    def test_healthy_gpu_venv_is_not_a_mismatch(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            health = gpu_health(gpu_present=True)
        self.assertFalse(health.mismatch)
        self.assertTrue(health.torch_cuda_available)
        self.assertEqual(health.torch_device_name, "NVIDIA GeForce RTX 5090")

    def test_pure_cpu_machine_is_not_a_mismatch(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CPU_PROVIDERS)):
            health = gpu_health(gpu_present=False)
        self.assertFalse(health.ort_has_cuda)
        self.assertFalse(health.mismatch)

    def test_tensorrt_provider_also_means_gpu_expected(self):
        providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        with patched(fake_torch("2.13.0+cpu"), fake_ort(providers)):
            health = gpu_health(gpu_present=False)
        self.assertTrue(health.ort_has_cuda)
        self.assertTrue(health.mismatch)

    def test_torch_not_installed_is_safe(self):
        with patched(None, fake_ort(CPU_PROVIDERS)):
            health = gpu_health(gpu_present=False)
        self.assertEqual(health.torch_version, "not installed")
        self.assertFalse(health.torch_cuda_available)
        self.assertIsNone(health.torch_device_name)
        self.assertFalse(health.mismatch)

    def test_torch_raising_is_safe(self):
        """A broken CUDA runtime raises from is_available() instead of returning False."""
        torch_mod = types.ModuleType("torch")
        torch_mod.__version__ = "2.13.0+cu130"

        def boom():
            raise RuntimeError("CUDA driver initialization failed")

        torch_mod.cuda = types.SimpleNamespace(is_available=boom)
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            health = gpu_health(gpu_present=False)
        self.assertEqual(health.torch_version, "2.13.0+cu130")
        self.assertFalse(health.torch_cuda_available)
        self.assertIsNone(health.torch_device_name)
        self.assertTrue(health.mismatch)  # ort is on CUDA, torch is not — still a signal

    def test_device_name_failure_does_not_break_collection(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        torch_mod.cuda.get_device_name = mock.Mock(side_effect=RuntimeError("no device"))
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            health = gpu_health(gpu_present=True)
        self.assertIsNone(health.torch_device_name)
        self.assertFalse(health.mismatch)

    def test_onnxruntime_not_installed_is_safe(self):
        with patched(fake_torch("2.13.0+cpu"), None):
            health = gpu_health(gpu_present=False)
        self.assertEqual(health.ort_providers, ())
        self.assertFalse(health.ort_has_cuda)
        self.assertFalse(health.mismatch)

    def test_summary_mentions_torch_version_and_providers(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)):
            summary = gpu_health(gpu_present=False).summary
        self.assertIn("2.13.0+cpu", summary)
        self.assertIn("CUDAExecutionProvider", summary)
        self.assertIn("CPUExecutionProvider", summary)
        self.assertIn("uv sync --extra gpu", summary)

    def test_summary_of_healthy_venv_reports_device_and_no_mismatch(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            summary = gpu_health(gpu_present=True).summary
        self.assertIn("2.13.0+cu130", summary)
        self.assertIn("NVIDIA GeForce RTX 5090", summary)
        self.assertIn("mismatch: no", summary)

    def test_probes_the_hardware_when_not_told(self):
        """Without an explicit gpu_present the nvidia-smi probe decides."""
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CPU_PROVIDERS)):
            with mock.patch("sorta.diagnostics.nvidia_gpu_present", return_value=True) as probe:
                health = gpu_health()
        probe.assert_called_once_with()
        self.assertTrue(health.gpu_present)
        self.assertTrue(health.degraded)


class TestCpuFallbackOnGpuMachine(unittest.TestCase):
    """F76: the machine HAS a GPU — a CPU-only stack is a problem, not a lifestyle."""

    def test_no_gpu_and_cpu_stack_is_no_problem(self):
        """Regression: a genuine CPU-only machine must not get a false alarm."""
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
            gpu_present=False,
        )
        self.assertFalse(health.degraded)
        self.assertFalse(health.torch_ignores_gpu)
        self.assertFalse(health.ort_ignores_gpu)
        self.assertEqual(health.problems, ())
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            self.assertFalse(warn_if_gpu_mismatch(health))

    def test_gpu_present_but_torch_is_cpu_only_is_a_problem(self):
        """The case that went unnoticed live: the whole [cpu] profile on an RTX box."""
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
            gpu_present=True,
        )
        self.assertTrue(health.degraded)
        self.assertTrue(health.torch_ignores_gpu)
        self.assertFalse(health.mismatch)  # the F63 definition still says nothing here
        self.assertIn("[gpu]", health.summary)
        self.assertIn("uv sync --extra gpu --extra dev", health.summary)
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            self.assertTrue(warn_if_gpu_mismatch(health))
        self.assertEqual(len(cm.records), 1)
        self.assertIn("[gpu]", cm.records[0].getMessage())

    def test_gpu_present_torch_on_cuda_but_ort_without_cuda_is_a_problem(self):
        """Faces would silently run on the CPU — the doctor must print the pip fix."""
        health = GpuHealth(
            torch_version="2.13.0+cu130",
            torch_cuda_available=True,
            torch_device_name="NVIDIA GeForce RTX 5090",
            ort_providers=("AzureExecutionProvider", "CPUExecutionProvider"),
            gpu_present=True,
        )
        self.assertTrue(health.degraded)
        self.assertTrue(health.ort_ignores_gpu)
        self.assertFalse(health.torch_ignores_gpu)
        self.assertFalse(health.mismatch)
        self.assertIn(FIX_ORT, health.summary)
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            self.assertTrue(warn_if_gpu_mismatch(health))
        self.assertEqual(len(cm.records), 1)
        self.assertIn(FIX_ORT, cm.records[0].getMessage())

    def test_gpu_present_and_everything_on_cuda_is_silent(self):
        health = GpuHealth(
            torch_version="2.13.0+cu130",
            torch_cuda_available=True,
            torch_device_name="NVIDIA GeForce RTX 5090",
            ort_providers=tuple(CUDA_PROVIDERS),
            gpu_present=True,
        )
        self.assertFalse(health.degraded)
        self.assertEqual(health.problems, ())
        self.assertIn("mismatch: no", health.summary)
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            self.assertFalse(warn_if_gpu_mismatch(health))

    def test_both_stacks_ignoring_the_gpu_report_one_torch_problem(self):
        """torch CPU-only is one diagnosis, not two — plus the separate ORT one."""
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
            gpu_present=True,
        )
        self.assertEqual(len(health.problems), 2)
        self.assertIn("[gpu]", health.problems[0])
        self.assertIn(FIX_ORT, health.problems[1])

    def test_mismatch_keeps_its_meaning_on_a_gpu_machine(self):
        """gpu_present must not silently redefine the old field (F63 semantics)."""
        for gpu_present in (False, True):
            with self.subTest(gpu_present=gpu_present):
                health = GpuHealth(
                    torch_version="2.13.0+cpu",
                    torch_cuda_available=False,
                    torch_device_name=None,
                    ort_providers=tuple(CUDA_PROVIDERS),
                    gpu_present=gpu_present,
                )
                self.assertTrue(health.mismatch)
                self.assertTrue(health.problems[0].startswith("torch is a CPU-only"))

    def test_gpu_present_defaults_to_false_for_pre_f76_callers(self):
        """Old construction sites keep the old verdict — only `mismatch` can fire."""
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
        )
        self.assertFalse(health.gpu_present)
        self.assertFalse(health.degraded)

    def test_asdict_stays_json_serialisable(self):
        """`sorta doctor` and the UI banner serialise this dataclass."""
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
            gpu_present=True,
        )
        payload = json.loads(json.dumps(dataclasses.asdict(health)))
        self.assertTrue(payload["gpu_present"])
        self.assertTrue(payload["degraded"])
        self.assertTrue(payload["torch_ignores_gpu"])
        self.assertEqual(payload["ort_providers"], list(CPU_PROVIDERS))
        self.assertEqual(len(payload["problems"]), 2)


class TestNvidiaGpuPresent(unittest.TestCase):
    """The probe: never raises, never blocks, never imports torch."""

    def test_gpu_name_in_output_means_present(self):
        self.assertTrue(nvidia_gpu_present(smi()))

    def test_empty_output_means_absent(self):
        self.assertFalse(nvidia_gpu_present(smi(stdout="  \n")))

    def test_missing_binary_means_absent(self):
        def boom():
            raise FileNotFoundError("nvidia-smi")

        self.assertFalse(nvidia_gpu_present(boom))

    def test_non_zero_exit_code_means_absent(self):
        self.assertFalse(nvidia_gpu_present(smi(stdout="no devices", returncode=9)))

    def test_timeout_means_absent(self):
        def boom():
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3.0)

        self.assertFalse(nvidia_gpu_present(boom))

    def test_any_other_exception_means_absent(self):
        def boom():
            raise OSError("WinError 5: access denied")

        self.assertFalse(nvidia_gpu_present(boom))

    def test_default_runner_calls_nvidia_smi_with_a_timeout(self):
        """The real binary is never launched — only the call itself is inspected."""
        completed = subprocess.CompletedProcess(["nvidia-smi"], 0, "RTX 5090\n", "")
        with mock.patch("subprocess.run", return_value=completed) as run:
            self.assertTrue(nvidia_gpu_present())
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], "nvidia-smi")
        self.assertIn("--query-gpu=name", args[0])
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 5)
        self.assertFalse(kwargs["check"])

    def test_does_not_import_torch(self):
        """Importing torch costs ~4.5 s; a hardware probe must not pay for it."""
        with mock.patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            self.assertTrue(nvidia_gpu_present(smi()))
            self.assertNotIn("torch", sys.modules)


class TestWarnIfGpuMismatch(unittest.TestCase):
    def test_warns_once_on_mismatch(self):
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CUDA_PROVIDERS),
        )
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            warned = warn_if_gpu_mismatch(health)
        self.assertTrue(warned)
        self.assertEqual(len(cm.records), 1)
        message = cm.records[0].getMessage()
        self.assertIn("2.13.0+cpu", message)
        self.assertIn("CUDAExecutionProvider", message)
        self.assertIn("uv sync --extra gpu --extra dev", message)

    def test_silent_when_no_mismatch(self):
        health = GpuHealth(
            torch_version="2.13.0+cu130",
            torch_cuda_available=True,
            torch_device_name="NVIDIA GeForce RTX 5090",
            ort_providers=tuple(CUDA_PROVIDERS),
        )
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            self.assertFalse(warn_if_gpu_mismatch(health))

    def test_silent_on_pure_cpu_machine(self):
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CPU_PROVIDERS),
        )
        with self.assertNoLogs(LOGGER_NAME, level=logging.WARNING):
            self.assertFalse(warn_if_gpu_mismatch(health))

    def test_collects_health_itself_when_not_given(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)):
            with mock.patch("sorta.diagnostics.nvidia_gpu_present", return_value=False):
                with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
                    warned = warn_if_gpu_mismatch()
        self.assertTrue(warned)
        self.assertEqual(len(cm.records), 1)

    def test_custom_logger_receives_the_warning(self):
        health = GpuHealth(
            torch_version="2.13.0+cpu",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=tuple(CUDA_PROVIDERS),
        )
        log = mock.Mock(spec=logging.Logger)
        self.assertTrue(warn_if_gpu_mismatch(health, log=log))
        log.warning.assert_called_once()

    def test_message_survives_missing_onnxruntime(self):
        """No providers at all — the %s placeholder must still render."""
        health = GpuHealth(
            torch_version="not installed",
            torch_cuda_available=False,
            torch_device_name=None,
            ort_providers=(),
            gpu_present=True,
        )
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            self.assertTrue(warn_if_gpu_mismatch(health))
        self.assertIn("providers: -", cm.records[0].getMessage())


if __name__ == "__main__":
    unittest.main()
