# Linux GPU acceptance

The harness runs a real Songe night against an already-running loopback upstream: discover its model, park serving, prepare the trainable base, reuse the Mac harness's synthetic hindsight conversations, calibrate, select, train with Torch on CUDA/ROCm, apply the production gate, export, publish, make a socket request through the proxy, undo through `/api/return`, and request the base again. Rejection, CPU fallback, a wrong serving binding, or incomplete cleanup exits nonzero. Gate thresholds are unchanged; a small selected pool does not guarantee acceptance.

| Upstream | Export and serving evidence |
| --- | --- |
| Ollama | GGUF dequantization, full GGUF export preserving quantization, unique imported tags; verify imported weight SHA-256 and proxy version binding |
| llama.cpp | GGUF dequantization, GGUF LoRA export, supervised restart to register slots; verify the base path, active adapter path/scale, and proxy binding |
| vLLM (CUDA) | Local safetensors base, PEFT export, dynamic adapter load; verify the registered adapter path, distinct identifier, and proxy binding |

Use a disposable Linux GPU machine with this checkout and `pip install -e '.[cuda]'` or `'.[rocm]'`, plus pytest. For example, with Ollama already serving `qwen3:0.6b`:

```sh
python scripts/e2e_linux.py --upstream ollama --model qwen3:0.6b
python scripts/e2e_linux.py --upstream llamacpp --model YOUR_SERVER_ALIAS
python scripts/e2e_linux.py --upstream vllm --model Qwen/Qwen3-0.6B
```

`--url` selects another loopback endpoint; `--ollama-models-dir` identifies its readable library. llama.cpp must be locally identifiable: the harness records its invocation and creates a separate Songe-owned loopback server for parking and adapter registration. Cleanup stops only that test child; the discovered server keeps running. HOME, Songe data, conversion caches and exports live in a temporary directory. Only test tags/adapters are removed. `--output`, `--select-top`, `--lengths` and `--timeout` control diagnostics, selected data size, calibration lengths and the test-child watchdog. Scoring covers all new experience; training covers the selected pool once. The watchdog is a test harness limit, not a learning budget.

Each stage emits JSON with seconds and measured memory: sampled process-tree RSS or Torch peak allocated bytes, labelled by `memory_kind`. Gate rows contain measured held-out change, drift and per-scale verdicts; inapplicable metrics are `null`. Export time is also included in gate time because production gates full GGUF after export. Calibration, GPU identity, gate records and child logs remain under the output directory; model/work files are deleted.

For a **future paid run**, configure `runpodctl >=2.15` and its SSH key beforehand, choose a reviewed image digest satisfying current `pyproject.toml` Torch requirements, then run from the Mac:

```sh
bash scripts/runpod/run.sh 'NVIDIA GeForce RTX 4090' "$CUDA_IMAGE" \
  --max-seconds 7200 --max-hourly-cost 0.90
# AMD: use the actual AMD GPU id and a compatible ROCm image, with its own cap.
```

The launcher uses only configured `runpodctl` for provider access and transfers, including its supported legacy `exec python`; it never handles API keys. A temporary Git object database produces `git archive` of the current source, including uncommitted/new files, without a commit or index change. Source SHA-256 and receipts accompany results in `test-output/runpod/songe-test-<gpu>-<time>-<suffix>/`.

The bootstrap installs [Ollama](https://docs.ollama.com/linux), checksum-verified official llama.cpp platform tools, and CUDA-only vLLM in a separate environment. It preserves the image's Torch build: incompatible images fail setup. It downloads the small Qwen3 models, runs each acceptance variant sequentially, then runs CPU pytest and real GPU-marked gradient/update/checkpoint tests. ROCm records vLLM as skipped. The pinned llama.cpp release serves on CPU; Songe calibration/training and GPU-marked tests require the GPU. `/workspace/results/` contains logs, JUnit XML, package versions and exit status, copied back with checksum verification.

Cost is bounded operationally by one pod, a required hourly cap checked immediately after creation, the total lifetime measured from the create request, a separate local termination watchdog, and EXIT/INT/TERM cleanup. Setup, downloads and result copying share that lifetime; final time is reserved for deletion. No create retries or persistent volumes are used. The returned hourly price and total runtime are printed; allow separately for storage/tax. Keep the Mac awake and online: this installed CLI has no server TTL, so local deletion cannot guarantee a financial ceiling during a Mac/network/provider outage. A container timeout is additional protection, not confirmed pod termination. Unresolved attempts block the next launch; reconcile the retained receipt and use `runpodctl pod delete POD_ID` if needed.
