"""Read-only, per-learner stage metrics from durable observations/checkpoints."""

import json

from .sleep.lessons import signal_counts


def next_update(store):
    """What this model's next update waits for, in examples."""
    from .config import Config

    # Thresholds are installation policy; reading them must not touch the store.
    config = Config.load(store.root)
    examples, conversations = store.feedback_counts()
    train, held, pending = store.experience_counts()
    # As the scheduler counts: examples not yet set aside for tests or training
    # are new too, or a first update would read "8 more" with plenty waiting.
    pending = max(pending, store.pending_learnable())
    if config.learning == "dreams":
        # The pool is kept practice; moments with feedback (filled by
        # held-out practice) test it; one of each is enough.
        from .practice.pool import (
            dream_minimums,
            heldout_units,
            practice_steps,
            untrained,
        )

        min_train, min_tests, min_new = dream_minimums(config)
        examples, pending = len(practice_steps(store)), len(untrained(store))
        tests = heldout_units(store)
        if examples < min_train:
            return {"stage": "train", "have": examples, "need": min_train}
        if tests < min_tests:
            return {"stage": "test", "have": tests, "need": min_tests}
        if pending < min_new:
            return {"stage": "new", "have": pending, "need": min_new}
        return {"stage": "ready"}
    if examples < config.min_train:
        return {"stage": "train", "have": examples, "need": config.min_train}
    if conversations < config.min_heldout + 1:
        return {"stage": "test", "have": conversations, "need": config.min_heldout + 1}
    if pending < config.min_new:
        return {"stage": "new", "have": pending, "need": config.min_new}
    return {"stage": "ready"}


def metrics(store):
    signals = signal_counts([])
    with store.connect() as db:
        eligible, params = store._learning_filter(db)
        where = "s.retired_at IS NULL AND " + eligible
        counts = db.execute(
            "SELECT COUNT(DISTINCT s.conversation),COUNT(*),COALESCE(SUM(s.split='heldout'),0) FROM steps s WHERE "
            + where,
            params,
        ).fetchone()
        # Each row is a compact summary, never a logged context or hindsight.
        if db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='step_signals'"
        ).fetchone():
            for row in db.execute(
                "SELECT m.signals FROM step_signals m JOIN steps s ON s.id=m.step_id WHERE "
                + where,
                params,
            ):
                for key, value in json.loads(row[0]).items():
                    signals[key] = signals.get(key, 0) + value
        else:
            for row in db.execute(
                "SELECT signal,COUNT(*) FROM steps s WHERE "
                + where
                + " GROUP BY signal",
                params,
            ):
                if row[0]:
                    signals[row[0]] = row[1]
    nights = store.nights()
    if not store.model_key:
        nights = [n for n in nights if n.get("model_id", "") == store.model_id]
    latest = nights[-1]["record"] if nights else {}
    progress = latest.get("data_progress", {})
    training = progress if progress.get("phase") == "train" else {}
    wake = store.get("wake_last")
    last = None
    if wake:
        last = {
            k: wake.get(k) for k in ("at", "repeat_errors_delta", "drift", "swapped")
        }
        # Only actual daytime observations can measure repeat errors. A KL
        # improvement or a synthetic verifier pass is never an error-rate delta.
        with store.connect() as db:
            rates = {}
            for version in (wake.get("expected"), wake.get("candidate")):
                row = db.execute(
                    "SELECT AVG(repeat_error_rate) FROM scoreboard WHERE model_version=?",
                    (version,),
                ).fetchone()
                rates[version] = row[0] if row else None
        before, after = (
            rates.get(wake.get("expected")),
            rates.get(wake.get("candidate")),
        )
        if before is not None and after is not None:
            last["repeat_errors_delta"] = after - before
    versions = store.versions()
    current = store.get("current")
    return {
        "next": next_update(store),
        "day": {
            "conversations": counts[0],
            "steps": counts[1],
            "heldout": counts[2],
        },
        "rewind": {"signals": signals, "lessons": len(store.lessons())},
        "consolidate": {
            "runs": sum(
                bool(n["record"].get("consolidate_started"))
                or bool(n["record"].get("iterations"))
                for n in nights
            ),
            "step": training.get("step", 0),
            "total": training.get("total_steps", 0),
            "eta_seconds": training.get("eta_seconds"),
            "last_seconds": next(
                (
                    n["record"]["consolidate_seconds"]
                    for n in reversed(nights)
                    if "consolidate_seconds" in n["record"]
                ),
                None,
            ),
        },
        "wake": {"last": last},
        "versions": [
            {
                "id": v["id"],
                "current": v["id"] == current,
                "kind": v.get("kind") or ("adapter" if v.get("adapter") else "base"),
            }
            for v in versions
        ],
    }
