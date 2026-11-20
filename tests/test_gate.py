import pytest

from songe.sleep.gate import decide


def metrics(kl=1, **changes):
    return [
        {
            "id": str(i),
            "conversation": str(i),
            "kl": kl,
            "drift": 0.01,
            "tool_valid": 1,
            "empty": 0,
            "repetition": 0,
            "tool_step": True,
            **changes,
        }
        for i in range(12)
    ]


def test_measured_improvement_passes():
    d = decide(metrics(), metrics(0.8), min_heldout=12, drift_limit=0.05)
    assert d.accepted
    assert d.heldout_change == pytest.approx(0.2)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"drift": 0.2}, "drift"),
        ({"tool_valid": 0}, "validity"),
        ({"empty": 1}, "Empty"),
        ({"repetition": 1}, "Looping"),
        ({"kl": float("nan")}, "finite"),
    ],
)
def test_reject_regressions(change, reason):
    after = [r | change for r in metrics(0.8)]
    assert reason in decide(metrics(), after, min_heldout=12, drift_limit=0.05).reason


def test_no_change_and_insufficient_independent_data_rejected():
    assert not decide(metrics(), metrics(), min_heldout=12, drift_limit=0.05).accepted
    assert not decide(
        metrics(conversation="one-thread"),
        metrics(0.8, conversation="one-thread"),
        min_heldout=12,
        drift_limit=0.05,
    ).accepted


def test_noisy_gain_fails_paired_cluster_bootstrap():
    after = metrics(0.99)
    after[0]["kl"] = 2
    assert not decide(metrics(), after, min_heldout=12, drift_limit=0.05).accepted


def test_more_repeated_phrases_in_code_is_not_a_regression():
    """Live: a coding adapter wrote ordinary code with more repeated phrases
    and was vetoed for "repetition"; only answers stuck in a loop count."""
    before = metrics(repetition=0.1)
    after = metrics(0.8, repetition=0.3)
    assert decide(before, after, min_heldout=12, drift_limit=0.05).accepted
    one = metrics(0.8)
    one[0]["repetition"] = 0.9  # a single looping answer is not enough
    assert decide(metrics(), one, min_heldout=12, drift_limit=0.05).accepted
