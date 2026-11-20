"""Explicit hardware/export smoke, outside default pytest and user libraries.

Run with `uv run --extra torch python -m tests.crossplatform_smoke --help`.
Downloads only tiny random public fixtures into HF_HOME. Artifacts/tools are
isolated below --work, and every heavy conversion is memory-guarded.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from songe.artifacts import export
from songe.artifacts.runtime import guarded_run
from songe.artifacts.tools import fetch
from songe.backends.torch import TorchBackend, free_memory, select_device
from songe.models import ModelRef, trainable_base
from songe.models.gguf import read_gguf
from tests.crossplatform_helpers import torch_config

FIXTURES = {
    "llama": (
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    ),
    "qwen2": (
        "yujiepan/qwen2.5-tiny-random",
        "0354728bb14944308c3134cf48d29302030a72c0",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "rocm", "mps", "xpu", "auto"]
    )
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument(
        "--four-bit",
        action="store_true",
        help="Exercise bnb with a deliberately constrained measured memory budget",
    )
    args = parser.parse_args()
    import psutil
    import torch
    from huggingface_hub import snapshot_download

    from songe.memory import CRITICAL, WARN, pressure_level

    torch.set_num_threads(1)
    device = select_device(torch, args.device)
    if args.device == "rocm" and not torch.version.hip:
        raise RuntimeError("--device rocm requires a ROCm torch wheel")
    if args.device == "cuda" and torch.version.hip:
        raise RuntimeError("--device cuda requires a CUDA torch wheel")

    def memory():
        level = pressure_level()
        return SimpleNamespace(
            available=psutil.virtual_memory().available,
            pressure="critical"
            if level >= CRITICAL
            else "warn"
            if level >= WARN
            else "normal",
        )

    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    platform = SimpleNamespace(
        name="windows" if sys.platform == "win32" else sys.platform,
        data_dir=lambda: root,
        memory=memory,
    )
    if not memory().pressure == "normal":
        raise MemoryError("Smoke requires normal memory pressure")
    results = []
    tools = fetch(platform) if args.artifacts else None
    for family, (repo, revision) in FIXTURES.items():
        cached = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.jinja",
                    "merges.txt",
                ],
            )
        )
        work = root / family
        if work.exists():
            raise FileExistsError(
                f"Choose a new --work directory; {work} already exists"
            )
        work.mkdir()
        source = work / "base"
        shutil.copytree(cached, source)
        # The legacy tiny fixture has an invalid negative pad id. Normalize
        # test-local metadata only; the downloaded snapshot stays untouched.
        for filename in ("config.json", "generation_config.json"):
            file = source / filename
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data.get("pad_token_id"), int) and data["pad_token_id"] < 0:
                data["pad_token_id"] = None
                file.write_text(json.dumps(data), encoding="utf-8")
        config = torch_config(source, args.device)
        config.response_tokens, config.max_sequence = 4, 512
        if args.four_bit:
            from transformers import AutoConfig, AutoModelForCausalLM

            with torch.device("meta"):
                empty = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(source, local_files_only=True)
                )
            bf16 = sum(p.numel() for p in empty.parameters()) * 2
            del empty
            # Reserve half of the bf16 weight budget, derived from live free
            # bytes. This exercises the 4-bit branch without exhausting VRAM.
            config.memory_fraction = (bf16 / 2) / free_memory(torch, device)
        backend = TorchBackend(config)
        if args.four_bit and not backend.quantized:
            raise AssertionError(
                "The constrained memory budget did not exercise 4-bit loading"
            )
        # These random checkpoints test architectures, not instruction tuning.
        # The old Llama fixture's alternating-turn template rejects the shared
        # MLX hindsight prompt (two user messages). Supply an explicit fixture
        # template consistently for both, without modifying the cached files.
        backend.tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        step = {
            "context": [{"role": "user", "content": "What is two plus two?"}],
            "hindsight": [{"role": "user", "content": "The correct answer is four."}],
        }
        with backend.base():
            frozen = backend.head(backend.hidden([1, 3])).clone()
        backend.optimizer(0.001)
        loss, grads, tokens = backend.train_step(step, seed=11)
        assert tokens > 0
        backend.apply(grads)
        with backend.base():
            torch.testing.assert_close(
                backend.head(backend.hidden([1, 3])), frozen, atol=0, rtol=0
            )
        adapter = work / "adapter"
        backend.save(adapter)
        assert any(value > 0 for value in backend.layer_deltas())
        export("peft-adapter", source, adapter, work / "peft", platform)
        row = {
            "family": family,
            "device": str(device),
            "rocm": bool(torch.version.hip),
            "four_bit": backend.quantized,
            "loss": loss,
            "free_before_load": backend.available_bytes,
            "deltas": backend.layer_deltas(),
        }
        del backend
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.artifacts:
            original = work / "original.gguf"
            guarded_run(
                [
                    sys.executable,
                    str(tools.convert_hf),
                    str(source),
                    "--outtype",
                    "bf16",
                    "--outfile",
                    str(original),
                ],
                platform=platform,
                env=tools.env(),
            )
            # The tiny fixture has narrow matrices that cannot use every K
            # quantization; Q8_0 still exercises the official quantizer on CPU.
            quantized = work / "original-q8.gguf"
            guarded_run(
                [str(tools.quantize), str(original), str(quantized), "Q8_0"],
                platform=platform,
                env=tools.env(),
            )
            ref = ModelRef(
                "llamacpp", family, "gguf", quantized, read_gguf(quantized).architecture
            )
            dense = trainable_base(ref, "torch", platform)
            assert list(dense.glob("*.safetensors"))
            export(
                "gguf-lora", source, adapter, work / "lora.gguf", platform, tools=tools
            )
            full = export(
                "gguf-full",
                dense,
                adapter,
                work / "full.gguf",
                platform,
                ref=ref,
                tools=tools,
            )
            assert read_gguf(full).quantization == read_gguf(quantized).quantization
            row["gguf"] = {
                "quantization": read_gguf(full).quantization,
                "dense_cache": str(dense),
            }
        results.append(row)
    (root / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    # All files are confined to --work and HF_HOME. No live services touched.


if __name__ == "__main__":
    main()
