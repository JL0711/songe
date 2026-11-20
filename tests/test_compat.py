import sys
import types

import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)

from songe.sleep import compat  # noqa: E402

mx.set_memory_limit(512 * 1024 * 1024)


def tensors(heads=4, kv_heads=4, length=12, source=12, dim=32, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, heads, length, dim))
    k = mx.random.normal((1, kv_heads, source, dim))
    v = mx.random.normal((1, kv_heads, source, dim))
    sinks = mx.random.normal((heads,))
    return q, k, v, sinks


@pytest.mark.parametrize("kv_heads", [4, 2])
@pytest.mark.parametrize("mask_kind", ["causal", "bool", "additive", "none"])
def test_sinks_attention_matches_fused_kernel(kv_heads, mask_kind):
    q, k, v, sinks = tensors(kv_heads=kv_heads)
    length = source = q.shape[2]
    causal = mx.arange(length)[:, None] >= mx.arange(source)[None, :]
    mask = {
        "causal": "causal",
        "bool": causal,
        "additive": mx.where(causal, 0.0, -1e9).astype(mx.float32),
        "none": None,
    }[mask_kind]
    expected = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.125, mask=mask, sinks=sinks
    )
    actual = compat.sdpa_with_sinks(q, k, v, 0.125, mask, sinks)
    assert mx.allclose(expected, actual, atol=1e-4).item()


def test_sinks_attention_is_differentiable():
    q, k, v, sinks = tensors(kv_heads=2)

    def loss(q, k, v, sinks):
        return compat.sdpa_with_sinks(q, k, v, 0.125, "causal", sinks).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3))(q, k, v, sinks)
    mx.eval(grads)
    assert all(mx.all(mx.isfinite(g)).item() for g in grads)
    assert grads[3].shape == sinks.shape


def test_patches_only_touch_loaded_modules(monkeypatch):
    calls = []

    def original(queries, keys, values, cache, scale, mask, sinks=None):
        calls.append(sinks is None)
        return queries

    fake = types.ModuleType("mlx_lm.models.fake_sinks")
    fake.scaled_dot_product_attention = original
    monkeypatch.setitem(sys.modules, "mlx_lm.models.fake_sinks", fake)
    compat.apply()
    q, k, v, sinks = tensors()
    # Without sinks the original fused kernel is still used.
    fake.scaled_dot_product_attention(q, k, v, None, 0.125, "causal")
    assert calls == [True]
    # With sinks the differentiable implementation takes over.
    out = fake.scaled_dot_product_attention(q, k, v, None, 0.125, "causal", sinks)
    assert calls == [True] and out.shape == q.shape
    # Applying twice does not wrap twice.
    patched = fake.scaled_dot_product_attention
    compat.apply()
    assert fake.scaled_dot_product_attention is patched


def test_gated_delta_patch_matches_reference():
    gated_delta = pytest.importorskip("mlx_lm.models.gated_delta")
    if not hasattr(mx.fast, "gated_delta_update"):
        pytest.skip("MLX without the gated delta primitive")
    reference = gated_delta.gated_delta_ops
    if getattr(reference, "__songe__", False):
        pytest.skip("already patched in this process")
    mx.random.seed(1)
    b, t, hk, hv, dk, dv = 1, 16, 2, 4, 128, 128
    q = mx.random.normal((b, t, hk, dk)) * 0.1
    k = mx.random.normal((b, t, hk, dk)) * 0.1
    v = mx.random.normal((b, t, hv, dv))
    g = mx.random.uniform(0.8, 1.0, (b, t, hv))
    beta = mx.random.uniform(0, 1, (b, t, hv))
    state = mx.zeros((b, hv, dv, dk), dtype=mx.float32)
    expected = reference(q, k, v, g, beta, state, None)
    compat.apply()
    actual = gated_delta.gated_delta_ops(q, k, v, g, beta, state, None)
    assert gated_delta.gated_delta_ops is not reference
    for e, a in zip(expected, actual, strict=True):
        assert mx.allclose(e, a, atol=1e-4).item()


def test_mlx_optimizer_accepts_float_schedule_past_warmup():
    pytest.importorskip("mlx.core")
    import math
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.optimizers as optim

    from songe.backends.mlx import Backend as MLXBackend

    backend = SimpleNamespace(mx=mx, optim=optim)
    optimizer = MLXBackend.optimizer(
        backend, lambda step: 1e-3 * (1 + math.cos(math.pi * 0.5)) / 2
    )
    params = {"w": mx.ones((2,))}
    optimizer.update(params, {"w": mx.ones((2,), dtype=mx.bfloat16)})
    mx.eval(params)
