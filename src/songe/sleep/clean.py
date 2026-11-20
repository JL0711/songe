"""Dream, step 1: drop steps that carry no learning signal (DESIGN 3.2).

Only logged evidence decides. Input steps are never modified; the review note
that organises a kept step's hindsight is added when its teacher prompt is built.
"""

import itertools
import json

from .. import steps as step_signals

# Tool results that mean the call never ran: the agent or transport stopped it.
INTERRUPTED = {"aborted", "provider.transport"}


def interrupted(message):
    if message.get("role") != "tool":
        return False
    try:
        error = (json.loads(message.get("content") or "") or {}).get("error") or {}
    except (ValueError, AttributeError, TypeError):
        return False
    return isinstance(error, dict) and error.get("type") in INTERRUPTED


def empty_call(action):
    return any(
        (call.get("function") or {}).get("arguments") in (None, "", "{}", {})
        for call in action.get("tool_calls") or []
    )


def no_signal(step):
    """A reviewed user reply that moved on without judging the result, and
    nothing else came back: tool results after the action are evidence on
    their own, whatever the user said next."""
    hindsight = step.get("hindsight") or [{}]
    last = hindsight[-1]
    if last.get("role") != "user":
        return False
    if any(m.get("role") == "tool" for m in hindsight[:-1]):
        return False
    key = step_signals.message_key(last)
    note = step_signals.REVIEW_NOTES.get(step_signals.reply_key(step)) or {}
    return (
        key in step_signals.REVIEWED
        and step_signals.REVIEWED[key] is None
        and not note.get("issue")
    )


def action_key(step):
    action = step.get("action") or {}
    calls = [
        (
            (c.get("function") or {}).get("name"),
            (c.get("function") or {}).get("arguments"),
        )
        for c in action.get("tool_calls") or []
    ]
    return json.dumps([action.get("content") or "", calls], sort_keys=True, default=str)


def verdict(step):
    """Why cleaning drops a step, or None when it keeps it."""
    if not step.get("hindsight"):
        return "no outcome yet"
    if empty_call(step.get("action") or {}):
        return "tool call without arguments"
    if any(interrupted(m) for m in step["hindsight"]):
        return "tool call interrupted"
    if no_signal(step):
        return "your reply did not judge the result"
    return None


def useful(step):
    return verdict(step) is None


def clean(steps):
    """Keep steps with signal; a back-to-back run of the same action counts once.

    The run's second step represents it: its context already shows the first
    attempt, which is when the loop began."""
    ordered = sorted(
        (s for s in steps if useful(s)),
        key=lambda s: (s.get("conversation", ""), s.get("source_id", 0)),
    )
    kept = []
    for _, run in itertools.groupby(
        ordered, key=lambda s: (s.get("conversation", ""), action_key(s))
    ):
        run = list(run)
        kept.append(run[0] if len(run) == 1 else run[1] | {"repeated": len(run)})
    return kept
