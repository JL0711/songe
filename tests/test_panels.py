"""Presentation accepts current, sparse and pre-pipeline API snapshots."""

from copy import deepcopy

import pytest

from songe.panels import has_data, panel_rows, pipeline, trainability


def test_pipeline_dashboard(pipeline_data):
    model = {"id": "one", "learning_level": "new", "pipeline": pipeline_data}
    before = deepcopy(model)
    rows = panel_rows(model)
    # Most important first; internal counts (examples, runs) stay out.
    assert [key for key, _ in rows] == ["result", "learned", "data", "versions"]
    rows = dict(rows)
    assert rows["result"].startswith("Repeat-error rate down 18 percentage points")
    # What it learns from, in plain words, without counts.
    assert (
        rows["learned"] == "fixed tool errors · your corrections · successful retries"
    )
    assert rows["data"] == "38 conversations · 6 lessons"
    assert pipeline(model)["rewind"]["signals"] == pipeline_data["rewind"]["signals"]
    assert (
        rows["versions"]
        == "v3 · In use · v2 · Available to restore · Original · Available to restore"
    )
    assert model == before


def test_empty_and_sparse_pipeline():
    model = {
        "id": "one",
        "learning_level": "new",
        "learning": {"conversations": 0},
        "pipeline": {"day": {}, "wake": {"last": {}}},
        "versions": [{"id": "base"}],
    }
    assert panel_rows(model) == []
    model.update(
        path="/fixture/model",
        format="mlx",
        support="unsupported",
        reason="embedding model",
    )
    assert panel_rows(model) == [
        ("problem", "This model is for embeddings · Choose a chat model for learning."),
    ]
    model["pipeline"] = {
        "day": {"conversations": 2, "steps": 0},
        "wake": {"last": {"repeat_errors_delta": 0, "drift": 0, "swapped": False}},
    }
    rows = dict(panel_rows(model))
    assert rows["data"] == "2 conversations"
    assert rows["result"].startswith("Kept previous version · did not pass checks")
    assert not has_data(
        {
            "day": {"conversations": 0},
            "rewind": {"signals": {"tool_error": 0}},
        }
    )


def test_memory_wait_is_shown_as_now_not_as_a_fix():
    model = {
        "id": "one",
        "learning_level": "new",
        "support": "pending",
        "reason": "calibration paused · memory · retry when idle",
    }
    assert panel_rows(model) == [("now", "Waiting for memory · retries automatically")]


def test_legacy_fallback_and_live_training_binding():
    model = {
        "id": "one",
        "files": [{"id": "two", "path": "/fixture/model", "format": "gguf"}],
        "learning": {
            "conversations": 2,
            "signals": {
                "tool_error_fixed": 1,
                "user_correction": 0,
                "retry_success": 0,
            },
            "last": {
                "created": 100,
                "repeat_errors_change": -0.18,
                "drift": 0.002,
                "status": "swapped",
            },
        },
    }
    work = {
        "model": "two",
        "phase": "train",
        "step": 41,
        "total_steps": 120,
        "data_progress": {"eta_seconds": 180},
    }
    rows = panel_rows(model, work)
    assert rows[0] == ("now", "Training · step 41 of 120 · about 3m left")
    rows = dict(rows)
    assert rows["data"] == "2 conversations"
    assert rows["learned"] == "fixed tool errors"
    assert "practice" not in rows
    assert "consolidate" not in pipeline(model, work | {"model": "other"})
    assert "consolidate" not in pipeline(model, work | {"phase": "gate"})
    model["pipeline"] = {
        "day": {"conversations": 38, "heldout": 12},
        "rewind": {"lessons": 6},
    }
    rows = dict(panel_rows(model))
    assert "now" not in rows
    assert rows["data"] == "38 conversations · 6 lessons"


@pytest.mark.parametrize(
    "model, expected",
    [
        ({"support": "trainable", "learning_level": "new"}, "learning"),
        ({"support": "pending"}, "not measured"),
        ({"support": "unsupported"}, "unsupported"),
        ({"support": "pending", "reason": "calibration paused · memory"}, "paused"),
        ({"support": "trainable", "learning_level": "off"}, "paused"),
    ],
)
def test_short_trainability(model, expected):
    assert trainability(model) == expected


def test_retained_versions_are_available_after_learning_window_reset():
    model = {
        "id": "one",
        "learning_level": "new",
        "learning": {"conversations": 0},
        "versions": [{"id": "base", "kind": "base", "current": True}],
    }
    assert panel_rows(model) == []
    model["versions"] = [{"id": "v2", "current": True}, {"id": "v1", "current": False}]
    assert panel_rows(model) == [
        ("versions", "v2 · In use · v1 · Available to restore")
    ]


def test_hashed_original_is_not_a_learned_version():
    model = {
        "id": "one",
        "learning_level": "off",
        "versions": [{"id": "base-4a9808bc", "current": True, "kind": "mlx-fused"}],
    }
    assert panel_rows(model) == []
    model["versions"].append({"id": "v1", "current": False, "kind": "adapter"})
    assert dict(panel_rows(model))["versions"] == (
        "Original · In use · v1 · Available to restore"
    )


def test_measuring_model_says_so_in_now():
    model = {"id": "one", "learning_level": "new"}
    snapshot = {
        "activity": {"training": False},
        "training": {"model": "one", "phase": "calibrate"},
    }
    assert panel_rows(model, snapshot=snapshot)[0] == (
        "now",
        "Measuring what this machine can train · first time only",
    )
    assert panel_rows(model, snapshot=snapshot | {"paused": True}) == []


def test_model_order_unsupported_last_and_turned_on_first():
    from songe.models.library import order_models

    models = [
        {"id": "a", "enabled": False},
        {"id": "embed", "enabled": False, "support": "unsupported"},
        {"id": "b", "enabled": False},
        {"id": "c", "enabled": True},
        {"id": "d", "enabled": False},
    ]
    ids = lambda order: [m["id"] for m in order_models(models, order)]  # noqa: E731
    # Enabled first, then library order, unsupported last.
    assert ids([]) == ["c", "a", "b", "d", "embed"]
    # Turned on most recently first; "b" stays on top after turning off.
    assert ids(["d", "b"]) == ["d", "b", "c", "a", "embed"]
    models[2]["enabled"] = False
    assert ids(["d", "b"])[:2] == ["d", "b"]


def test_dashboard_rows_fill_quiet_defaults():
    from songe.panels import STAGE_LABELS, dashboard_rows

    model = {"id": "one", "learning_level": "new"}
    rows = dashboard_rows(model)
    assert [key for key, *_ in rows] == [
        "now",
        "result",
        "data",
        "next",
        "versions",
    ]
    assert dict((k, v) for k, v, _ in rows)["next"] == (
        "First update after your first conversations"
    )
    assert all(empty for *_, empty in rows)
    assert set(STAGE_LABELS) >= {key for key, *_ in rows}
    model["versions"] = [{"id": "v3", "current": True}]
    assert (
        dict((k, v) for k, v, _ in dashboard_rows(model))["next"]
        == "update after more conversations"
    )
    model["learning_level"] = "off"
    assert dict((k, v) for k, v, _ in dashboard_rows(model))["next"] == (
        "Turn on New or All to learn"
    )
