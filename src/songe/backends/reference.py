"""Framework-free oracle for the distillation objective in sleep.distill.

Inputs are rows of vocabulary logits (ordinary Python sequences). The teacher's
unselected tokens form one exact tail bucket, including the reference's floor.
This deliberately favors clarity over speed; training uses the tensor version.
"""

import math

TAIL_FLOOR = 1e-12  # Same log(0) floor as the MLX reference.


def _logsumexp(row):
    maximum = max(row)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in row))


def divergence(student, teacher, kind="forward_kl", beta=0.5, top_k=0, clip=0.0):
    if kind not in {"forward_kl", "reverse_kl", "jsd", "gjsd"}:
        raise ValueError(f"Unknown divergence: {kind}")
    if not 0 < beta < 1 or top_k < 0:
        raise ValueError("beta must be in (0, 1) and top_k must be nonnegative")
    if not student or len(student) != len(teacher):
        raise ValueError("Expected equally sized, nonempty rows of logits")
    values = []
    for s_row, t_row in zip(student, teacher, strict=True):
        if not s_row or len(s_row) != len(t_row):
            raise ValueError("Vocabulary dimensions differ or are empty")
        s = [float(x) - _logsumexp(s_row) for x in s_row]
        t = [float(x) - _logsumexp(t_row) for x in t_row]
        if 0 < top_k < len(t):
            indices = sorted(range(len(t)), key=t.__getitem__)[-top_k:]
            s, t = [s[i] for i in indices], [t[i] for i in indices]
            s.append(math.log(max(1 - math.fsum(map(math.exp, s)), TAIL_FLOOR)))
            t.append(math.log(max(1 - math.fsum(map(math.exp, t)), TAIL_FLOOR)))
        if kind == "forward_kl":
            terms = (math.exp(b) * (b - a) for a, b in zip(s, t, strict=True))
            value = math.fsum(min(x, clip) if clip > 0 else x for x in terms)
        elif kind == "reverse_kl":
            value = math.fsum(math.exp(a) * (a - b) for a, b in zip(s, t, strict=True))
        else:
            value = 0.0
            for a, b in zip(s, t, strict=True):
                mix = _logsumexp([math.log(beta) + b, math.log1p(-beta) + a])
                value += beta * math.exp(b) * (b - mix)
                value += (1 - beta) * math.exp(a) * (a - mix)
        values.append(value)
    return math.fsum(values) / len(values)
