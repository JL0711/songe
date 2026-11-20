"""NVIDIA memory telemetry: NVML first, nvidia-smi as a patient fallback."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from songe.platform import common


def fake_nvml(free=8 * 2**30, fail=False):
    class NVMLError(Exception):
        pass

    def init():
        if fail:
            raise NVMLError("driver not loaded")

    return SimpleNamespace(
        NVMLError=NVMLError,
        nvmlInit=init,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=24 * 2**30, free=free),
        nvmlDeviceGetName=lambda h: b"NVIDIA GeForce RTX 4090",
        nvmlDeviceGetUUID=lambda h: "GPU-1",
    )


@pytest.fixture(autouse=True)
def fresh_nvml(monkeypatch):
    monkeypatch.setitem(common._NVML, "ready", None)
    monkeypatch.setattr(common.shutil, "which", lambda name: None)


def test_nvml_reads_memory_without_a_subprocess(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml())
    monkeypatch.setattr(
        common.subprocess, "check_output", lambda *a, **k: pytest.fail("no subprocess")
    )
    [gpu] = common.discrete_accelerators()
    assert (gpu.kind, gpu.name, gpu.memory_free) == (
        "cuda",
        "NVIDIA GeForce RTX 4090",
        8 * 2**30,
    )


def test_falls_back_to_nvidia_smi_and_retries_one_timeout(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(fail=True))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")
    calls = []

    def slow_then_ok(command, timeout, stderr):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout)
        return b"GPU-1, NVIDIA A100, 81920, 40960\n"

    monkeypatch.setattr(common.subprocess, "check_output", slow_then_ok)
    [gpu] = common.discrete_accelerators()
    assert gpu.memory_free == 40960 * 1024**2 and calls == [15, 15]


def test_unavailable_telemetry_still_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(fail=True))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")

    def always_slow(command, timeout, stderr):
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(common.subprocess, "check_output", always_slow)
    with pytest.raises(OSError, match="NVIDIA memory telemetry unavailable"):
        common.discrete_accelerators()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS footprint")
def test_macos_process_bytes_counts_physical_footprint():
    import os

    from songe.platform.darwin import phys_footprint

    # Footprint includes Metal buffers that RSS omits; it must be readable.
    assert phys_footprint(os.getpid()) > 1_000_000
