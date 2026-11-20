"""Read: turn recorded sessions into learning signals.

In idle gaps the model being trained, already loaded on the model server,
reads each new session and says where feedback arrived (a user correction, a
rejected tool call, a failed result, an informative outcome), which steps were
responsible and what the lesson is. Rules find tool failures and checked
successes; the model handles user messages, credit assignment and lessons.
Signals feed practice (generated scenes trained on instead of real tasks);
confirmed successes form the drift set. See docs/CONTRACTS.md, Learning cycle.
"""
