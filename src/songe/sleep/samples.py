"""Admit intact samples, distinguishing a busy machine from an unfit sample."""

import json

from ..backends import Interrupted
from ..steps import digest, revision_id
from .prompts import UnusableStep, paired_prompts
from .sizing import cost_model, dimensions, predict


class SampleAdmission:
    def __init__(self, config, sizing, calibration, store, slice_id):
        self.config, self.sizing = config, sizing or {}
        self.calibration = calibration or {}
        self.store, self.slice_id = store, slice_id
        self.record = {"deferred_samples": [], "unfit_samples": []}
        # A new model, policy or machine measurement can make old samples fit.
        self.scope = digest(
            {
                "model": config.model,
                "max_sequence": config.max_sequence,
                "response_tokens": config.response_tokens,
                "hindsight_tokens": config.hindsight_tokens,
                "memory_fraction": config.memory_fraction,
                "calibration": {
                    k: self.calibration.get(k)
                    for k in ("model", "artifact_kind", "rows", "capacity_limits")
                },
                "capacity": self.calibration.get("memory_baseline", {}).get("total"),
                "devices": [
                    (a.get("kind"), a.get("memory_total"))
                    for a in self.calibration.get("memory_baseline", {}).get("vram", [])
                ],
                "environment": store.get("environment_fingerprint"),
                "context_policy": [
                    1,
                    config.teacher_chunk,
                ],
            }
        )
        self.unfit = store.get("unfit_samples", {})

    def key(self, step):
        return f"{self.scope}:{step['id']}:{revision_id(step)}"

    def known_unfit(self, step):
        return self.key(step) in self.unfit

    def never_fit(self, required):
        # Only the gradient window (context tail + response, at most
        # max_sequence) is backpropagated; longer context is read without
        # gradients. Length alone never makes an example unfit.
        required = min(required, self.config.max_sequence)
        ceiling = (self.calibration.get("progress") or {}).get("memory_ceiling")
        if ceiling is not None and required > ceiling:
            return "Required sequence exceeds measured machine memory capacity"
        rows = [
            r
            for r in self.calibration.get("rows", [])
            if "host_peak_bytes" in r and "device_peak_bytes" in r
        ]
        if not rows:
            return None  # No measurement supports a permanent memory verdict.
        if required > max(r["sequence"] for r in rows):
            return None  # An unfinished calibration is not a capacity bound.
        response = self.config.response_tokens
        if response > max(dimensions(r)[1] for r in rows):
            return None  # Recalibration, rather than a verdict about this sample.
        # Test the required context at the smallest measured chunk, using total
        # machine capacity rather than today's free memory. Never extrapolate.
        size = (
            max(required, min(r["sequence"] for r in rows)),
            max(response, min(dimensions(r)[1] for r in rows)),
            min(dimensions(r)[2] for r in rows),
            self.config.accumulation,
        )
        memory = self.sizing.get("memory") or self.calibration.get(
            "memory_baseline", {}
        )
        total = memory.get("total", 0)

        def peak(resource):
            model = cost_model(rows, resource)
            if model:
                return predict(model, size)
            # One geometry supports a verdict only at that exact geometry.
            matching = [r[resource] for r in rows if dimensions(r)[:3] == size[:3]]
            return max(matching, default=0)

        host = peak("host_peak_bytes")
        device = peak("device_peak_bytes")
        if total and host > total * self.config.memory_fraction:
            return (
                "Required sequence exceeds machine RAM at the smallest measured chunk"
            )
        devices = memory.get("vram", [])
        if devices and any(device > a["memory_total"] for a in devices):
            return (
                "Required sequence exceeds device memory at the smallest measured chunk"
            )
        return None

    def admit(self, steps, tokenizer, *, full_context=False):
        usable, deferred = [], False
        for step in steps:
            if not full_context and self.known_unfit(step):
                continue
            reason, required = None, None
            try:
                student, teacher = paired_prompts(
                    tokenizer,
                    step,
                    self.config,
                    check_length=False,
                )
                required = len(student) + self.config.response_tokens
                teacher_required = (
                    max(len(student), len(teacher)) + self.config.response_tokens
                )
                # Backpropagation covers only the context tail plus the
                # response (the gradient window, up to max_sequence); earlier
                # context is read without gradients (DESIGN 3.3). So a long
                # agent context is never unfit for its length; its full-context
                # forward is bounded by the measured teacher envelope below.
                # Gate forwards have their own measured envelope.
                reason = (
                    None
                    if full_context
                    else self.never_fit(
                        min(
                            required,
                            self.sizing.get("max_sequence", self.config.max_sequence),
                        )
                    )
                )
            except UnusableStep as exc:
                reason = str(exc)
            entry = {
                "id": step["id"],
                "revision": revision_id(step),
                "required_sequence": required,
                "teacher_sequence": teacher_required if required is not None else None,
            }
            if reason:
                entry["reason"] = reason
                self.unfit[self.key(step)] = entry
                self.store.mark_unfit(self.key(step), entry, step)
                self.record["unfit_samples"].append(entry)
            elif teacher_required > self.sizing.get(
                "max_teacher_sequence", float("inf")
            ) or self.config.response_tokens > self.sizing.get(
                "response_tokens", self.config.response_tokens
            ):
                entry["reason"] = "Intact sample exceeds the current slice capacity"
                if entry not in self.record["deferred_samples"]:
                    self.record["deferred_samples"].append(entry)
                deferred = True
            else:
                usable.append(step)
        # Persist before raising, including if the process is stopped later.
        with self.store.connect() as db:
            db.execute(
                "UPDATE slices SET record=? WHERE id=?",
                (json.dumps(self.record), self.slice_id),
            )
        if deferred and not usable:
            raise Interrupted("Waiting for memory: intact training samples deferred")
        return usable
