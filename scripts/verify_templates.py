"""CPU-only tool rendering checks. No model weights or Metal are needed."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import render_jinja_template

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "songe_probe",
            "description": "Read a local test path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
MESSAGES = [
    {"role": "user", "content": "Read the test path."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "songe_probe",
                    "arguments": {"path": "songe-sentinel"},
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "songe_probe",
        "content": "songe-result",
    },
    {"role": "user", "content": "What happened?"},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=Path("docs/evidence/template-verification.json")
    )
    p.add_argument(
        "--models", type=Path, default=Path("/Volumes/disk/local-llm/models")
    )
    args = p.parse_args()
    results = []
    for model in sorted(args.models.glob("*/*/tokenizer_config.json")):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model.parent), local_files_only=True, trust_remote_code=False
            )
            text = tokenizer.apply_chat_template(
                MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
            tokens = tokenizer.apply_chat_template(
                MESSAGES,
                tools=TOOLS,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
            assert (
                isinstance(tokens, list)
                and len(tokens) > 20
                and isinstance(tokens[0], int)
            )
            assert all(
                value in text
                for value in ("songe_probe", "songe-sentinel", "songe-result")
            )
            results.append(
                {
                    "model": str(model.parent),
                    "passed": True,
                    "tokens": len(tokens),
                    "scope": "installed tokenizer, tool schema, assistant call, tool result",
                }
            )
        except Exception as exc:
            results.append(
                {"model": str(model.parent), "passed": False, "error": str(exc)}
            )
    # Template-only probes for model families absent from this model library.
    # Download tiny public tokenizer metadata; never model weights.
    for repo in (
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    ):
        try:
            config = json.loads(
                Path(hf_hub_download(repo, "tokenizer_config.json")).read_text(
                    encoding="utf-8"
                )
            )
            template = config.get("chat_template")
            if not template:
                template = Path(hf_hub_download(repo, "chat_template.jinja")).read_text(
                    encoding="utf-8"
                )
            if isinstance(template, list):
                options = {v["name"]: v["template"] for v in template}
                template = options.get("tool_use", options.get("default"))
            text, _ = render_jinja_template(
                conversations=[MESSAGES],
                tools=TOOLS,
                chat_template=template,
                add_generation_prompt=True,
                bos_token=config.get("bos_token", ""),
                eos_token=config.get("eos_token", ""),
            )
            assert all(
                v in text[0] for v in ("songe_probe", "songe-sentinel", "songe-result")
            )
            results.append(
                {
                    "model": repo,
                    "passed": True,
                    "scope": "template metadata only; no inference or training",
                }
            )
        except Exception as exc:
            results.append({"model": repo, "passed": False, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
