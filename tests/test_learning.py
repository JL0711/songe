import json
from dataclasses import replace

import numpy as np
import pytest

from songe.sleep.distill import divergence
from songe.sleep.prompts import OversizedStep, UnusableStep, paired_prompts
from songe.sleep.select import select
from songe.sleep.worker import general_steps


class NumpyAPI:
    float32 = np.float32
    stop_gradient = staticmethod(lambda x: x)
    exp = staticmethod(np.exp)
    logaddexp = staticmethod(np.logaddexp)
    log = staticmethod(np.log)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    concatenate = staticmethod(np.concatenate)
    argpartition = staticmethod(np.argpartition)
    take_along_axis = staticmethod(np.take_along_axis)

    @staticmethod
    def logsumexp(x, axis=-1, keepdims=True):
        m = x.max(axis=axis, keepdims=True)
        return m + np.log(np.exp(x - m).sum(axis=axis, keepdims=keepdims))


def test_objective_direction_jsd_and_exact_tail_bucket():
    s = np.log(np.array([[0.8, 0.1, 0.1]]))
    t = np.log(np.array([[0.2, 0.3, 0.5]]))
    # Default is forward KL(teacher || student) over the full vocabulary.
    forward = (np.exp(t) * (t - s)).sum()
    reverse = (np.exp(s) * (s - t)).sum()
    assert divergence(NumpyAPI, s, t) == pytest.approx(forward, rel=1e-6)
    assert divergence(NumpyAPI, s, t, "reverse_kl") == pytest.approx(reverse, rel=1e-6)
    assert divergence(NumpyAPI, s, s) == pytest.approx(0, abs=1e-7)
    jsd = divergence(NumpyAPI, s, t, "jsd", 0.5)
    assert 0 < jsd < np.log(2)
    # top-k=1 keeps the teacher's top token and one exact teacher/student tail.
    expected = 0.5 * np.log(0.5 / 0.1) + 0.5 * np.log(0.5 / 0.9)
    assert divergence(NumpyAPI, s, t, top_k=1) == pytest.approx(expected, rel=1e-6)
    expected = 0.1 * np.log(0.1 / 0.5) + 0.9 * np.log(0.9 / 0.5)
    assert divergence(NumpyAPI, s, t, "reverse_kl", top_k=1) == pytest.approx(
        expected, rel=1e-6
    )
    eps = 1e-3
    perturbed = s.copy()
    perturbed[0, 0] += eps
    assert divergence(NumpyAPI, perturbed, t, "reverse_kl") > divergence(
        NumpyAPI, s, t, "reverse_kl"
    )
    # Forward KL grows when the student lowers a token the teacher supports.
    perturbed = s.copy()
    perturbed[0, 2] -= eps
    assert divergence(NumpyAPI, perturbed, t) > divergence(NumpyAPI, s, t)


def test_selection_trains_every_cleaned_train_step_without_scoring(config):
    config.slice_limit = 2
    steps = [
        {
            "id": str(i),
            "conversation": str(i),
            "source_id": i,
            "split": "train" if i < 5 else "heldout",
            "action": {"content": f"answer {i}"},
            "hindsight": [{"role": "user", "content": "Fix it"}] if i != 4 else [],
        }
        for i in range(10)
    ]
    steps[3]["retired_at"] = "2026-01-01"

    chosen, skipped = select(steps)
    # Held-out, retired and hindsight-free steps are never trained; no top-k cap.
    assert [s["id"] for s in chosen] == ["0", "1", "2"]
    assert skipped == 0
    assert all("surprise" not in s for s in chosen)
    assert select([]) == ([], 0)


class Tokenizer:
    def encode(self, text, **kw):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode(errors="ignore")

    def apply_chat_template(self, messages, tools=None, **kw):
        return list(json.dumps({"messages": messages, "tools": tools}).encode())


def test_oversized_prompt_preserves_all_context_for_later_memory(config):
    config = replace(config, max_sequence=900, response_tokens=64, hindsight_tokens=30)
    step = {
        "context": [
            {"role": "system", "content": "rules" * 80},
            {"role": "user", "content": "old" * 300},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
        ],
        "tools": [],
        "hindsight": [{"role": "user", "content": "correction"}],
    }
    with pytest.raises(OversizedStep):
        paired_prompts(Tokenizer(), step, config, check_length=True)
    # Length is not checked by default: the exact context is never cut.
    s, t = paired_prompts(Tokenizer(), step, config)
    assert (s, t) == paired_prompts(Tokenizer(), step, config, check_length=False)
    student, teacher = json.loads(bytes(s)), json.loads(bytes(t))
    assert teacher["messages"][:-1] == student["messages"]
    assert student["messages"][0]["role"] == "system"
    assert student["messages"][-1]["content"] == "recent question"
    assert student["messages"] == step["context"]
    assert max(len(s), len(t)) > config.max_sequence - config.response_tokens


def test_multimodal_and_oversize_tools_fail_closed(config):
    step = {
        "context": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example/image"}}
                ],
            }
        ],
        "hindsight": [],
        "tools": [],
    }
    with pytest.raises(UnusableStep):
        paired_prompts(Tokenizer(), step, config)
    step["context"][0]["content"] = "Hi"
    step["tools"] = [
        {"type": "function", "function": {"name": "used", "description": "x" * 10000}}
    ]
    step["action"] = {"tool_calls": [{"function": {"name": "used"}}]}
    with pytest.raises(UnusableStep):
        paired_prompts(Tokenizer(), step, config, check_length=True)


def test_150_original_diverse_anchor_prompts():
    anchors = general_steps()
    assert len(anchors) == 150
    assert len({s["id"] for s in anchors}) == 150
    assert all(s["split"] == "anchor" and not s["hindsight"] for s in anchors)


def test_checkpoints_are_spaced_not_written_every_step():
    import inspect

    from songe.backends import training

    source = inspect.getsource(training.train)
    # One at the start, on interruption and at the end; in between only when due.
    assert "if checkpoint_due():" in source
    assert training.CHECKPOINT_MIN_SECONDS >= 30
