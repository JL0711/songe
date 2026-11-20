"""Idle review: the loaded model reads user replies; phrases are a fallback."""

import json

import httpx
import pytest

from songe import review, steps
from songe.steps import Step, signal_for


def step(reply, *, step_id="s1"):
    return Step(
        step_id,
        "c" * 16,
        "train",
        [{"role": "user", "content": "recreate this page in html"}],
        [],
        {"role": "assistant", "content": "Done, boat.html written"},
        [{"role": "user", "content": reply}],
        1,
    )


def test_chinese_complaint_counts_once_reviewed(store, monkeypatch):
    monkeypatch.setattr(steps, "REVIEWED", {})
    item = step("这也太丑了，你去仔细看一下他们的前端")
    assert signal_for(item)[0] != "user_correction"  # no phrase matches
    store.sync_steps([item.dict()])
    key = steps.message_key(item.hindsight[-1])
    store.add_review(key, "user_correction", {"reply": "correction"}, ["s1"])
    with store.connect() as db:
        assert db.execute("SELECT signal FROM steps WHERE id='s1'").fetchone()[0] == (
            "user_correction"
        )
    steps.REVIEWED.clear()
    store.load_reviews()  # survives a restart
    assert steps.user_signal(item.hindsight[-1]) == "user_correction"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            '{"reply": "correction", "issue": "ugly", "lesson": "match the site"}',
            "correction",
        ),
        ('<think>hmm</think>\n{"reply": "approval"}', "approval"),
        ("no json here", None),
        ('{"reply": "maybe"}', None),
    ],
)
def test_verdict_parsing(text, expected):
    verdict = review.parse(text)
    assert (verdict or {}).get("reply") == expected


async def test_review_uses_the_given_model_and_stops_when_busy(store, monkeypatch):
    monkeypatch.setattr(steps, "REVIEWED", {})
    store.sync_steps(
        [step("不对，按钮要圆角").dict(), step("thanks", step_id="s2").dict()]
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"reply": "correction"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        review.httpx, "AsyncClient", lambda **k: real(transport=transport, **k)
    )
    done = await review.review(store, "http://127.0.0.1:1234/v1", "loaded-model")
    assert done == 2 and set(calls) == {"loaded-model"}
    assert review.pending(store, 10) == []
    # The user came back: nothing is read.
    store.sync_steps([step("again please", step_id="s3").dict()])
    assert await review.review(store, "u", "m", still_idle=lambda: False) == 0


def conversation(name, signal, n=4):
    return [
        {
            "id": f"{name}-{i}",
            "conversation": name,
            "split": "pending",
            "signal": signal,
            "context": [{"role": "user", "content": name}],
            "tools": [],
            "action": {"role": "assistant", "content": "x"},
            "hindsight": [{"role": "tool", "content": "ok"}],
            "source_id": i,
        }
        for i in range(n)
    ]


def test_tests_come_from_collected_data_across_kinds_and_never_move(store):
    items = [
        *[s for i in range(6) for s in conversation(f"err{i}", "tool_error")],
        *[s for i in range(4) for s in conversation(f"fix{i}", "user_correction")],
    ]
    store.sync_steps(items)
    assert store.feedback_counts() == (40, 10)
    store.assign_splits(minimum=2)
    held = {s["conversation"] for s in store.active_steps("heldout")}
    # 20% of ten conversations, one from each kind of feedback.
    assert len(held) == 2 and {c[:3] for c in held} == {"err", "fix"}
    store.sync_steps([s for i in range(5) for s in conversation(f"new{i}", "retry")])
    store.assign_splits(minimum=2)
    after = {s["conversation"] for s in store.active_steps("heldout")}
    assert held <= after and len(after) == 3  # settled tests stay tests
    trained = {s["conversation"] for s in store.active_steps("train")}
    assert not trained & after


def test_tool_calls_keep_the_machine_busy_until_a_final_answer():
    import time

    from songe.schedule import AGENT_FOLLOWUP_SECONDS, Scheduler

    worker = Scheduler.__new__(Scheduler)
    worker.last_traffic = time.monotonic() - 1000
    worker.response_finished("tool_calls")
    # Tools may run for minutes; the agent's next call is still part of it.
    assert worker.last_traffic - time.monotonic() > AGENT_FOLLOWUP_SECONDS - 5
    worker.last_traffic = time.monotonic() - 1000
    worker.response_finished("stop")
    assert abs(worker.last_traffic - time.monotonic()) < 1  # quiet time starts now
