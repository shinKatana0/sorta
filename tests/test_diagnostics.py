"""F63: GPU-health guard — torch/onnxruntime are mocked, no real GPU is needed."""
import logging
import sys
import types
import unittest
from unittest import mock

from sorta.diagnostics import GpuHealth, gpu_health, warn_if_gpu_mismatch

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


CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_PROVIDERS = ["CPUExecutionProvider"]


class TestGpuHealth(unittest.TestCase):
    def test_mismatch_when_ort_has_cuda_and_torch_is_cpu(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)):
            health = gpu_health()
        self.assertTrue(health.mismatch)
        self.assertTrue(health.ort_has_cuda)
        self.assertFalse(health.torch_cuda_available)
        self.assertEqual(health.torch_version, "2.13.0+cpu")
        self.assertIsNone(health.torch_device_name)
        self.assertEqual(health.ort_providers, tuple(CUDA_PROVIDERS))

    def test_healthy_gpu_venv_is_not_a_mismatch(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            health = gpu_health()
        self.assertFalse(health.mismatch)
        self.assertTrue(health.torch_cuda_available)
        self.assertEqual(health.torch_device_name, "NVIDIA GeForce RTX 5090")

    def test_pure_cpu_machine_is_not_a_mismatch(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CPU_PROVIDERS)):
            health = gpu_health()
        self.assertFalse(health.ort_has_cuda)
        self.assertFalse(health.mismatch)

    def test_tensorrt_provider_also_means_gpu_expected(self):
        providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        with patched(fake_torch("2.13.0+cpu"), fake_ort(providers)):
            health = gpu_health()
        self.assertTrue(health.ort_has_cuda)
        self.assertTrue(health.mismatch)

    def test_torch_not_installed_is_safe(self):
        with patched(None, fake_ort(CPU_PROVIDERS)):
            health = gpu_health()
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
            health = gpu_health()
        self.assertEqual(health.torch_version, "2.13.0+cu130")
        self.assertFalse(health.torch_cuda_available)
        self.assertIsNone(health.torch_device_name)
        self.assertTrue(health.mismatch)  # ort is on CUDA, torch is not — still a signal

    def test_device_name_failure_does_not_break_collection(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        torch_mod.cuda.get_device_name = mock.Mock(side_effect=RuntimeError("no device"))
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            health = gpu_health()
        self.assertIsNone(health.torch_device_name)
        self.assertFalse(health.mismatch)

    def test_onnxruntime_not_installed_is_safe(self):
        with patched(fake_torch("2.13.0+cpu"), None):
            health = gpu_health()
        self.assertEqual(health.ort_providers, ())
        self.assertFalse(health.ort_has_cuda)
        self.assertFalse(health.mismatch)

    def test_summary_mentions_torch_version_and_providers(self):
        with patched(fake_torch("2.13.0+cpu"), fake_ort(CUDA_PROVIDERS)):
            summary = gpu_health().summary
        self.assertIn("2.13.0+cpu", summary)
        self.assertIn("CUDAExecutionProvider", summary)
        self.assertIn("CPUExecutionProvider", summary)
        self.assertIn("uv sync --extra gpu", summary)

    def test_summary_of_healthy_venv_reports_device_and_no_mismatch(self):
        torch_mod = fake_torch("2.13.0+cu130", cuda_available=True)
        with patched(torch_mod, fake_ort(CUDA_PROVIDERS)):
            summary = gpu_health().summary
        self.assertIn("2.13.0+cu130", summary)
        self.assertIn("NVIDIA GeForce RTX 5090", summary)
        self.assertIn("mismatch: no", summary)


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


if __name__ == "__main__":
    unittest.main()
