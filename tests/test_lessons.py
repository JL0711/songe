"""Static, local practice: fake inference, tiny CPU contracts, no live services."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from songe.api import ReadOnlyStore, snapshot
from songe.models import registry
from songe.platform import Accel, Memory
from songe.sleep import worker
from songe.sleep.lessons import extract_lessons, groups, prune_lessons
from songe.sleep.worker import Journal
from songe.steps import digest
from tests.test_schedule import good
from tests.test_store import version
from tests.test_worker import Backend

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "code": {"type": "string"}},
                "required": ["file", "code"],
                "additionalProperties": False,
            },
        },
    }
]


def action(**args):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "write", "arguments": args},
            }
        ],
    }


def experience(index=1, *, split="train", suffix="py"):
    wrong = action(file=f"original-{index}.{suffix}", command="bad")
    right = action(file=f"original-{index}.{suffix}", code="print(1)")
    return {
        "id": str(index),
        "conversation": f"task-{index}",
        "split": split,
        "source_id": index,
        "observed_id": index + 100,
        "action": wrong,
        "context": [{"role": "user", "content": f"Write original-{index}.{suffix}"}],
        "tools": copy.deepcopy(TOOLS),
        "env_fingerprint": digest(TOOLS),
        "hindsight": [
            {"role": "tool", "content": '{"error":"missing required code"}'},
            right,
            {"role": "tool", "content": '{"exit_code":0}'},
        ],
        "evidence": [{"role": "tool", "content": '{"error":"missing required code"}'}],
        "signal": "later_success",
        "strength": 1.0,
        "answer_hint": right,
    }


class LocalBase(Backend):
    def __init__(self, config, check=lambda: None):
        super().__init__(
            replace(config, max_sequence=32000, hindsight_tokens=2048), check
        )
        self.calls = []

    def sample(self, prompt, seed=None):
        assert self.frozen, "Lesson generation must disable LoRA"
        self.check()
        data = json.loads(bytes(prompt))
        text = data["messages"][-1]["content"]
        self.calls.append(text)
        if text.startswith("Contrast"):
            answer = "Pass the program in the required code field when writing a file."
        else:
            answer = json.dumps(action(file="variant.py", code="print(42)"))
        return [1], answer, None


@pytest.fixture
def lesson_setup(store, config):
    steps = [experience(i) for i in range(1, 9)]
    store.sync_steps(steps)
    work = store.root / "unit-work"
    work.mkdir()
    backend = LocalBase(config)
    journal = Journal(work, backend.check, store, store.slice())
    lessons = extract_lessons(store, backend, journal, steps)
    assert len(lessons) == 1
    return backend, journal, lessons, steps


def test_grouping_contrasts_failures_and_successes_and_excludes_gate():
    steps = [
        experience(1),
        experience(2),
        experience(3, suffix="json"),
        experience(4, split="heldout"),
    ]
    no_success = experience(5)
    no_success["answer_hint"] = None
    no_success["conversation"] = "unresolved"
    no_success["hindsight"] = no_success["hindsight"][:1]
    result = groups(steps + [no_success])
    assert len(result) == 2
    python = next(g for g in result if g["file_types"] == ["py"])
    assert len(python["failures"]) == 3
    assert len(python["successes"]) == 2
    assert {e["step_id"] for e in python["evidence"]} == {"1", "2", "5"}
    assert all(e["conversation"] != "task-4" for g in result for e in g["evidence"])
    assert python["strength"] == 1
    changed_tool = experience(6)
    changed_tool["tools"][0]["function"]["name"] = "other"
    changed_tool["action"]["tool_calls"][0]["function"]["name"] = "other"
    changed_tool["answer_hint"]["tool_calls"][0]["function"]["name"] = "other"
    changed_tool["env_fingerprint"] = digest(changed_tool["tools"])
    assert len(groups(steps + [changed_tool])) == 3


def test_lessons_are_local_frozen_durable_and_idempotent(lesson_setup, store):
    backend, journal, lessons, steps = lesson_setup
    assert lessons[0]["text"].endswith(".")
    assert len(lessons[0]["evidence"]) == len(steps)
    assert not backend.frozen
    calls = len(backend.calls)
    assert extract_lessons(store, backend, journal, steps) == lessons
    assert len(backend.calls) == calls
    assert ReadOnlyStore(store.root).lessons() == lessons


@pytest.mark.parametrize("change", ["forget", "stale", "superseded"])
def test_evidence_retirement_cascades_to_lessons(lesson_setup, store, change):
    backend, journal, lessons, steps = lesson_setup
    assert store.lessons()
    if change == "forget":
        store.forget()
    elif change == "stale":
        with store.connect() as db:
            db.execute(
                "UPDATE steps SET retired_at=1,retired_reason='stale' WHERE id='1'"
            )
        prune_lessons(store)
    else:
        revised = steps[0] | {"observed_id": 999}
        store.sync_steps([revised])
    assert store.lessons() == []
    with store.connect() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM lessons WHERE retired_at IS NULL"
            ).fetchone()[0]
            == 0
        )


def test_pipeline_metrics_are_scoped_and_api_read_only(lesson_setup, store, config):
    backend, journal, lessons, steps = lesson_setup
    registry.migrate(config, store)
    registry.set_learning(store, "base-test", "all")
    registry.register(config, store, "other")
    primary = store.scoped("base-test")
    night = primary.night(None)
    primary.night(
        night,
        record={
            "consolidate_started": True,
            "consolidate_seconds": 7,
            "data_progress": {
                "phase": "train",
                "step": 2,
                "total_steps": 4,
                "eta_seconds": 5,
            },
        },
    )
    primary.add_version(version("base"))
    primary.swap("base", None)
    primary.add_version(version("learned", night) | {"kind": "peft-adapter"})
    primary.set(
        "wake_last",
        {
            "at": 10,
            "repeat_errors_delta": None,
            "drift": 0.01,
            "swapped": False,
            "expected": "base",
            "candidate": "learned",
        },
    )
    assert not primary.pipeline()["wake"]["last"]["swapped"]
    primary.swap("learned", "base")
    pipeline = primary.pipeline()
    assert set(pipeline) == {
        "day",
        "rewind",
        "consolidate",
        "wake",
        "versions",
        "next",
    }
    assert pipeline["next"]["stage"] in {"train", "test", "new", "ready"}
    assert pipeline["day"]["steps"] == 8
    assert pipeline["consolidate"] == {
        "runs": 1,
        "step": 2,
        "total": 4,
        "eta_seconds": 5,
        "last_seconds": 7,
    }
    assert pipeline["wake"]["last"] == {
        "at": 10,
        "repeat_errors_delta": None,
        "drift": 0.01,
        "swapped": True,
    }
    assert pipeline["versions"][-1]["current"]
    assert primary.pipeline()["rewind"]["signals"]["tool_error_fixed"] == 8
    assert store.scoped("other").pipeline()["consolidate"]["runs"] == 0
    assert store.scoped("other").pipeline()["day"]["steps"] == 0
    readonly = ReadOnlyStore(store.root)
    data = snapshot(config, readonly)
    assert (
        next(m for m in data["models"] if m["pipeline"]["day"]["steps"] == 8)[
            "pipeline"
        ]
        == pipeline
    )
    assert pipeline in data["pipeline"].values()
    assert readonly.scoped("base-test").pipeline() == pipeline


def test_off_and_new_windows_exclude_lesson_lineage(lesson_setup, store, config):
    backend, journal, lessons, steps = lesson_setup
    registry.migrate(config, store)
    registry.register(config, store, "other")
    other = store.scoped("other")
    assert other.lessons() == []
    registry.set_learning(store, "base-test", "all")
    extract_lessons(store.scoped("base-test"), backend, journal, steps)
    assert store.scoped("base-test").lessons()
    registry.set_learning(store, "base-test", "off")
    assert store.scoped("base-test").lessons() == []
    registry.set_learning(store, "base-test", "all")
    assert extract_lessons(store.scoped("base-test"), backend, journal, steps)


def test_worker_extracts_lessons_then_trains_real_steps_with_real_gate(
    config, store, monkeypatch
):
    events, training_pools, gate_ids = [], [], []

    class Model(LocalBase):
        scored, trained_units, initial_adapters = [], [], []
        comparisons = 0
        interrupt_training = interrupt_evaluation = False

        def sample(self, prompt, seed=None):
            text = json.loads(bytes(prompt))["messages"][-1]["content"]
            if text.startswith("Contrast"):
                events.append("lessons")
            else:
                self.check()
                return (
                    [1],
                    json.dumps(
                        {
                            "name": "write",
                            "arguments": {"file": "gate.py", "code": "print(1)"},
                        }
                    ),
                    None,
                )
            return super().sample(prompt, seed)

        def train(self, pool, anchors, work, progress):
            events.append("consolidate")
            training_pools.append(copy.deepcopy(pool))
            return super().train(pool, anchors, work, progress)

    config.upstream = "mlx"
    config.backend = "torch"
    config.force_cpu_training = config.calibration_available = True
    config.max_sequence, config.hindsight_tokens = 32000, 2048
    config.min_train, config.min_heldout = 3, 3
    config.slice_limit = 100
    train = [experience(i) for i in range(1, 9)]
    heldout = [experience(i, split="heldout") for i in range(101, 104)]
    store.sync_steps(train + heldout)
    monkeypatch.setattr(worker, "probe", lambda: good())
    monkeypatch.setattr(
        worker,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [Accel("cpu", "CPU stand-in")],
        ),
    )
    monkeypatch.setattr(worker, "choose", lambda *args: Model)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    anchors = worker.general_steps()[:2]
    monkeypatch.setattr(worker, "general_steps", lambda: anchors)
    from songe.sleep import evaluate

    original = evaluate.references

    def real_only(base, steps, **kwargs):
        gate_ids.extend(s["id"] for s in steps)
        assert all(s["split"] in {"heldout", "anchor"} for s in steps)
        return original(base, steps, **kwargs)

    monkeypatch.setattr(evaluate, "references", real_only)
    from songe.sleep import select as selection

    original_select = selection.select

    def select(steps):
        events.append("select")
        return original_select(steps)

    monkeypatch.setattr(selection, "select", select)
    assert worker.run(config, store)["accepted"]
    assert (
        events.index("lessons") < events.index("select") < events.index("consolidate")
    )
    assert len(training_pools) == 1
    assert all(s["split"] == "train" for s in training_pools[0])
    assert all(s["split"] == "train" for s in store.pool())
    assert len(gate_ids) == 5
    assert len(store.lessons()) == 1
    metrics = store.pipeline()
    assert metrics["consolidate"]["step"] == metrics["consolidate"]["total"]
    assert metrics["wake"]["last"]["swapped"] is False
