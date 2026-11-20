import json
import sys

import pytest

from songe.schedule import Probes
from songe.sleep.probe import choose, probe_models, support_reason


def model(tmp_path, architecture):
    path = tmp_path / architecture
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": architecture}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"fixture")
    return path


def test_architecture_names_do_not_exclude_guarded_trials(tmp_path, monkeypatch):
    import songe.sleep.probe as module

    before = set(sys.modules)
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: object()
    )
    for architecture in ("lfm3", "qwen3_5", "granite", "future_architecture"):
        assert support_reason(model(tmp_path, architecture)) is None
    assert not any(m.startswith("mlx.core") for m in set(sys.modules) - before)


def test_default_prefers_trainable_agent_model_then_fastest_measured_fit():
    results = [
        {"model": "/models/lfm3", "passed": False},
        {
            "model": "/models/granite-3b",
            "passed": True,
            "step_seconds": 2,
            "generation_tps": 40,
        },
        {
            "model": "/models/lfm2-8b",
            "passed": True,
            "step_seconds": 1,
            "generation_tps": 80,
        },
    ]
    assert choose(results, "lfm3")["model"] == "/models/lfm2-8b"
    assert choose(results, "granite-3b")["model"] == "/models/granite-3b"
    with pytest.raises(ValueError):
        choose(results[:1])


def test_unavailable_hardware_never_launches_probe(tmp_path, monkeypatch):
    import songe.sleep.probe as module

    def forbidden(*a, **kw):
        raise AssertionError("No GPU process may be launched")

    monkeypatch.setattr(module, "invoke_probe", forbidden)
    result = probe_models([model(tmp_path, "granite")], tmp_path / "data", Probes())
    assert not result[0]["passed"]
    assert "Hardware" in result[0]["reason"]


def test_teacher_context_restores_mode_and_disables_adapters():
    from songe.sleep.distill import Backend

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    class Adapter:
        scale = 2.0

    backend = Backend.__new__(Backend)
    backend.model = Model()
    backend.wrappers = [Adapter()]
    with backend.base():
        assert not backend.model.training
        assert backend.wrappers[0].scale == 0
    assert backend.model.training
    assert backend.wrappers[0].scale == 2


def test_probe_timeout_uses_guarded_runner(tmp_path, monkeypatch):
    import subprocess

    from songe import memory
    from songe.sleep.probe import invoke_probe

    calls = []

    def timed_out(argv, **kw):
        calls.append((argv, kw))
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(memory, "guarded_run", timed_out)
    result = invoke_probe(tmp_path, tmp_path / "result.json", timeout=60)
    assert not result["passed"]
    assert result["retryable"]
    assert "60-second" in result["reason"]
    assert calls[0][0][2] == "songe.sleep.probe"


def test_probe_memory_stop_is_temporary_with_measured_numbers(tmp_path, monkeypatch):
    from songe import memory
    from songe.platform import Memory
    from songe.sleep.probe import invoke_probe

    def stopped(*args, **kwargs):
        raise memory.MemoryPressure(Memory(8000, 500, "warn"))

    monkeypatch.setattr(memory, "guarded_run", stopped)
    result = invoke_probe(tmp_path, tmp_path / "result.json")
    assert result["retryable"]
    assert "needs more free memory now" in result["reason"]
    assert result["available_bytes"] == 500
