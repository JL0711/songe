import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from songe.memory import GuardPolicy, MemoryPressure, guard, guarded_run
from songe.platform import Accel, Memory


class Platform:
    name = "mock"

    def __init__(self, samples, devices=()):
        self.samples, self.devices, self.index = samples, devices, 0

    def memory(self):
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample

    def accelerators(self):
        return self.devices


def test_critical_pressure_before_start_never_launches(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("must not start")
    )
    with pytest.raises(MemoryPressure, match="available=123 bytes"):
        guarded_run(["unused"], platform=Platform([Memory(1000, 123, "critical")]))


def test_guard_critical_vram_bypasses_grace():
    callbacks = []
    guard(
        SimpleNamespace(returncode=None),
        Platform([Memory(1000, 900)], [Accel("cuda", "GPU", 100, 0)]),
        lambda m: callbacks.append("warn"),
        lambda m: callbacks.append("kill"),
    )
    assert callbacks == ["kill"]


def test_guard_allocation_rate_shortens_poll_and_warns(monkeypatch):
    from songe import memory

    samples = [Memory(10000, 9000 - 500 * i) for i in range(18)]
    platform = Platform(samples)
    intervals, callbacks = [], []
    ticks = iter(range(100))
    monkeypatch.setattr(memory.time, "monotonic", lambda: next(ticks))

    class Stop:
        def is_set(self):
            return len(intervals) >= 20 or bool(callbacks)

        def wait(self, interval):
            intervals.append(interval)

    guard(
        SimpleNamespace(returncode=None),
        platform,
        lambda m: callbacks.append("warn"),
        lambda m: callbacks.append("kill"),
        stop=Stop(),
    )
    assert min(intervals) < intervals[0]
    assert callbacks == ["warn"]  # the projection held for the grace period


def test_guard_brief_allocation_burst_does_not_stop(monkeypatch):
    from songe import memory

    # Prefill burst: fast drop, then flat well above the margin.
    samples = [Memory(1000, 900), Memory(1000, 400)] + [Memory(1000, 400)] * 4
    callbacks, polls = [], []
    ticks = iter(range(100))
    monkeypatch.setattr(memory.time, "monotonic", lambda: next(ticks))

    class Stop:
        def is_set(self):
            return len(polls) >= 6

        def wait(self, interval):
            polls.append(interval)

    guard(
        SimpleNamespace(returncode=None),
        Platform(samples),
        lambda m: callbacks.append("warn"),
        lambda m: callbacks.append("kill"),
        stop=Stop(),
    )
    assert callbacks == []


def test_guarded_run_normal_and_timeout_cleanup():
    platform = Platform([Memory(1000, 900)])
    result = guarded_run(
        [sys.executable, "-c", "print('done')"],
        platform=platform,
        stdout=subprocess.PIPE,
        timeout=10,
    )
    assert result.stdout.strip() == b"done"
    with pytest.raises(subprocess.TimeoutExpired):
        guarded_run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            platform=platform,
            timeout=0.02,
            policy=GuardPolicy(0.01, 0.01, 0.01),
        )


@pytest.mark.skipif(os.name == "nt", reason="Unix process group assertion")
def test_guard_reports_measured_memory_after_killing_child():
    platform = Platform([Memory(1000, 900)] * 4 + [Memory(1000, 17, "critical")])
    with pytest.raises(MemoryPressure, match="available=17 bytes") as stopped:
        guarded_run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            platform=platform,
            policy=GuardPolicy(0.01, 0.01, 0.01),
        )
    assert stopped.value.reason["rule"] == "critical_pressure"
    assert not any(
        t.is_alive() and t.name.endswith("(guard)") for t in threading.enumerate()
    )


def test_vram_is_checked_before_start_with_measured_numbers(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start")
    )
    platform = Platform([Memory(1000, 900)], [Accel("cuda", "fixture GPU", 500, 0)])
    with pytest.raises(
        MemoryPressure, match="VRAM fixture GPU: available=0 bytes, total=500 bytes"
    ):
        guarded_run(["unused"], platform=platform)
