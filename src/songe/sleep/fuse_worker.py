"""Official fuse/convert CLIs inside songe.memory's owned process group.

A LoRA fused back into a 4-bit base is re-quantized onto a coarse grid; the
rounding noise (KL ~1e-2 on Qwen3.5-0.8B) is ten times the learned change
(~1e-3), so the update would be lost. Below 8 bits, each adapted layer is
fused at full precision and stored at 8 bits (noise ~1e-3, the bf16 floor),
one layer at a time; untouched layers keep the base's own quantization.
"""

import sys

from ..schedule import probe

PRESERVING_BITS = 8


def report(progress, fraction):
    if progress:
        import json
        import os
        from pathlib import Path

        path = Path(progress)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({"fraction": round(fraction, 3)}))
        os.replace(temporary, path)


def high_precision_fuse(model_path, adapter_path, save_path, progress=None):
    """True when done here; False leaves the base's precision to mlx_lm.fuse."""
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load, save

    model, tokenizer, config = load(
        model_path, adapter_path=adapter_path, return_config=True
    )
    quantization = config.get("quantization") or {}
    bits = quantization.get("bits")
    if not isinstance(bits, int) or bits >= PRESERVING_BITS:
        return False
    group = quantization.get("group_size", 64)
    fused = []
    adapted = [(n, m) for n, m in model.named_modules() if hasattr(m, "fuse")]
    for index, (name, module) in enumerate(adapted):
        # Merging is most of the time; saving is the last tenth.
        report(progress, 0.9 * index / max(1, len(adapted)))
        layer = nn.QuantizedLinear.from_linear(
            module.fuse(dequantize=True), group_size=group, bits=PRESERVING_BITS
        )
        mx.eval(layer.parameters())
        fused.append((name, layer))
        quantization[name] = {"group_size": group, "bits": PRESERVING_BITS} | (
            {"mode": "affine"} if "mode" in quantization else {}
        )
    model.update_modules(tree_unflatten(fused))
    config["quantization"] = quantization
    if "quantization_config" in config:
        config["quantization_config"] = quantization
    report(progress, 0.9)
    save(Path(save_path), model_path, model, tokenizer, config, donate_model=False)
    report(progress, 1.0)
    return True


def main():
    hardware = probe()
    if not hardware.available or hardware.pressure == "critical":
        raise RuntimeError("Fusing requires measurable noncritical memory pressure")
    import mlx.core as mx
    from mlx_lm.fuse import main as operation

    # Allocation ceilings are advisory. The parent guard remains authoritative.
    mx.set_memory_limit(int(hardware.free_gb * 1e9))
    argv = sys.argv[1:]
    progress = None
    if "--songe-progress" in argv:
        at = argv.index("--songe-progress")
        progress = argv[at + 1]
        del argv[at : at + 2]
        sys.argv[1:] = argv
    if "--dequantize" not in argv:
        value = {
            flag: argv[argv.index(flag) + 1]
            for flag in ("--model", "--adapter-path", "--save-path")
            if flag in argv
        }
        if len(value) == 3 and high_precision_fuse(
            value["--model"], value["--adapter-path"], value["--save-path"], progress
        ):
            return
    operation()


if __name__ == "__main__":
    from songe.telemetry import diagnostic_main

    diagnostic_main(main)
