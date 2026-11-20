"""Practice: scenes written from signals, trained on instead of real tasks.

For each signal with a lesson, the learner's own model writes new situations
where that lesson matters, ending right before the choice it is about. The
frame (system prompt, tools) is the real one from the logged request; read,
grep, glob and file edits are computed from the scene's files; command output
is the model's own text. Nothing is ever executed. See docs/CONTRACTS.md,
Learning cycle.
"""
