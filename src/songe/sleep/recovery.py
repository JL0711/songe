"""One-time recovery of scoring consumed by legacy undersized calibration.

Only SQLite metadata is needed here: opening a store never loads a tokenizer,
model, or accelerator. Unknown lengths are measured in the next guarded gap.
"""

import json
import uuid

from ..config import Config

RECOVERY_KEY = "undersized_scoring_recovery_v1"


def recover_undersized_scoring(db, root):
    if db.execute("SELECT 1 FROM state WHERE key=?", (RECOVERY_KEY,)).fetchone():
        return

    def get(key, default=None):
        row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(key, value):
        db.execute(
            "INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value))
        )

    installation = Config.load(root)
    primary = get("primary_model")
    models = {
        "" if r["id"] == primary else r["id"]: (r["id"], json.loads(r["config"]))
        for r in db.execute("SELECT id,config FROM models")
    }
    models.setdefault("", (None, {}))
    recovered = {}
    for model_id, (model, values) in models.items():

        def key(name, model_id=model_id):
            return f"model:{model_id}:{name}" if model_id else name

        saved = get(key("calibration"), {}) or {}
        if not saved.get("rows"):
            continue
        budget = saved.get("budget", {})
        geometry = ("max_sequence", "response_tokens", "token_chunk")
        # Before f23, derive() persisted probe dimensions as learner policy.
        # Recover only an exact legacy budget match, using the installation's
        # configured limits; retain explicit subsequent per-model changes.
        restored = bool(
            model
            and budget.get("response_tokens") == 1
            and installation.response_tokens > 1
            and all(values.get(k) == budget.get(k) for k in geometry)
        )
        if restored:
            values.update({k: getattr(installation, k) for k in geometry})
            db.execute(
                "UPDATE models SET config=? WHERE id=?", (json.dumps(values), model)
            )
        response = values.get("response_tokens", installation.response_tokens)
        nights = [
            dict(r) | {"record": json.loads(r["record"])}
            for r in db.execute(
                "SELECT * FROM nights WHERE model_id=? ORDER BY id", (model_id,)
            )
        ]
        sizes = [s for n in nights for s in n["record"].get("slice_sizes", [])]
        if last_size := get(key("last_slice_size")):
            sizes.append(last_size)
        if not sizes:
            row = max(saved["rows"], key=lambda r: r["sequence"])
            sizes = [
                {
                    "max_sequence": row["sequence"],
                    "response_tokens": row.get("response_tokens", response),
                }
            ]
        unfit = get(key("unfit_samples"), {}) or {}
        lengths = {
            (s["id"], s["revision"]): s["required_sequence"]
            for s in [*saved.get("samples", []), *unfit.values()]
            if s.get("required_sequence") is not None
        }
        watermark = get(key("experience_watermark"), 0)
        invalid = set()
        for row in db.execute(
            "SELECT step_id,revision FROM scored_revisions WHERE model_id=?",
            (model_id,),
        ):
            step_id, revision = row["step_id"], row["revision"]
            required = lengths.get((step_id, revision))
            # Old stores did not tie a revision to its scoring geometry.
            # Retry uncertain untrained scores conservatively, once. A later
            # larger measurement cannot validate an earlier undersized score.
            if revision > watermark and any(
                size.get("response_tokens", response) < response
                or size.get("response_tokens") == 1
                or required is not None
                and required > size["max_sequence"]
                or required is None
                and size["max_sequence"]
                < values.get("max_sequence", installation.max_sequence)
                for size in sizes
            ):
                invalid.add((step_id, revision))
        if not invalid and not restored:
            continue
        db.executemany(
            "DELETE FROM scored_revisions WHERE step_id=? AND revision=? AND model_id=?",
            [(step_id, revision, model_id) for step_id, revision in invalid],
        )
        # Invalid capacity verdicts would otherwise consume the same revision
        # again before scoring. Calibration will reassess under current policy.
        put(
            key("unfit_samples"),
            {
                k: v
                for k, v in unfit.items()
                if not restored and (v["id"], v["revision"]) not in invalid
            },
        )
        if restored:
            # The old token lengths can include hindsight truncated at a probe
            # budget. Remeasure; retain completed rows as capacity evidence.
            saved.pop("progress", None)
            saved.pop("samples", None)
            put(key("calibration"), saved)
            put(key("calibration_progress"), None)
        put(key("candidate_generation"), str(uuid.uuid4()))
        for night in nights:
            record = night["record"]
            if (
                night["status"] in {"waiting", "interrupted"}
                and night["phase"] in {"select", "score", "rewind"}
                and not record.get("consolidate_started")
                and not record.get("steps_learned")
            ):
                record["reason"] = (
                    "Preparing examples after incomplete training-size measurement"
                )
                db.execute(
                    "UPDATE nights SET status='discarded',record=? WHERE id=?",
                    (json.dumps(record), night["id"]),
                )
        recovered[model_id] = {"requeued": len(invalid), "restored_policy": restored}
    if recovered:
        put("experience_revision", str(uuid.uuid4()))
    put(RECOVERY_KEY, recovered)
