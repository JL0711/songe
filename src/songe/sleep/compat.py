"""Make architectures that mlx-lm can run also trainable.

Some mlx-lm models call GPU kernels that only implement the forward pass, so
gradients either fail or fall back to a path that is too slow to use. Each
patch below replaces exactly one such call with a differentiable equivalent
and is applied only to modules that are already loaded. All three were
verified numerically against the original on 2026-11-10 (see docs/evidence).

- Gated DeltaNet (Qwen3.5 / 3.6 / 3.8, Qwen3-Next, Ling, Kimi linear layers):
  in training mode mlx-lm uses a token-by-token Python loop. MLX >= 0.32.4
  ships ``mx.fast.gated_delta_update`` with a Metal gradient kernel.
- Attention sinks (gpt-oss, DeepSeek V4.1, MiMo V2 Flash): MLX's fused
  attention has no gradient when sinks are present. Compute it with plain ops.
- DeepSeek V4.1 sinkhorn: a forward-only kernel runs on the GPU even when
  training; its pure-ops fallback is differentiable.
"""

import sys

import mlx.core as mx

_applied = set()


def _gated_delta():
    module = sys.modules.get("mlx_lm.models.gated_delta")
    if module is None or not hasattr(mx.fast, "gated_delta_update"):
        return False
    if getattr(module.gated_delta_ops, "__songe__", False):
        return True

    def gated_delta_ops(q, k, v, g, beta, state=None, mask=None):
        y, final_state = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)[:2]
        return y, final_state

    gated_delta_ops.__songe__ = True
    module.gated_delta_ops = gated_delta_ops
    return True


def sdpa_with_sinks(queries, keys, values, scale, mask, sinks):
    """Softmax attention with one extra per-head "sink" logit in the
    denominator, written in differentiable MLX ops. Shapes follow
    mx.fast.scaled_dot_product_attention: (B, H, L, D) queries and
    (B, H_kv, S, D) keys and values."""
    heads, length = queries.shape[1], queries.shape[2]
    kv_heads, source = keys.shape[1], keys.shape[2]
    if heads != kv_heads:
        repeats = heads // kv_heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)
    scores = (queries.astype(mx.float32) * scale) @ keys.astype(mx.float32).swapaxes(
        -1, -2
    )
    floor = mx.array(mx.finfo(mx.float32).min, mx.float32)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"Unsupported attention mask: {mask}")
        rows = mx.arange(source - length, source)[:, None]
        cols = mx.arange(source)[None, :]
        scores = mx.where(rows >= cols, scores, floor)
    elif mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, floor)
        else:
            scores = scores + mask.astype(mx.float32)
    sink = mx.broadcast_to(
        sinks.astype(mx.float32).reshape(1, heads, 1, 1),
        (*scores.shape[:-1], 1),
    )
    weights = mx.softmax(mx.concatenate([sink, scores], axis=-1), axis=-1)[..., 1:]
    return (weights @ values.astype(mx.float32)).astype(queries.dtype)


def _sinks():
    patched = False
    for name, module in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models.") or module is None:
            continue
        original = getattr(module, "scaled_dot_product_attention", None)
        if original is None or getattr(original, "__songe__", False):
            continue

        def scaled_dot_product_attention(
            queries, keys, values, cache, scale, mask, sinks=None, _original=original
        ):
            if sinks is None or hasattr(cache, "bits"):
                return _original(queries, keys, values, cache, scale, mask, sinks)
            return sdpa_with_sinks(queries, keys, values, scale, mask, sinks)

        scaled_dot_product_attention.__songe__ = True
        module.scaled_dot_product_attention = scaled_dot_product_attention
        patched = True
    return patched


def _sinkhorn():
    module = sys.modules.get("mlx_lm.models.deepseek_v41")
    if module is None or getattr(module, "_sinkhorn_kernel", None) is None:
        return False
    module._sinkhorn_kernel = None
    return True


def apply():
    """Patch whatever the loaded model uses. Safe to call more than once."""
    for name, patch in (
        ("gated_delta", _gated_delta),
        ("attention_sinks", _sinks),
        ("sinkhorn", _sinkhorn),
    ):
        if patch():
            _applied.add(name)
    return sorted(_applied)
