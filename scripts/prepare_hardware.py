"""Prepare synthetic, read-only-observation traffic in a fresh hardware test store."""

import argparse
from pathlib import Path

from songe import NAME
from songe.config import Config
from songe.log import Store

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", required=True, type=Path, help="An isolated local model copy"
)
parser.add_argument("--data-dir", required=True, type=Path)
args = parser.parse_args()
if args.data_dir.exists():
    parser.error("data-dir must be new; existing experience is never overwritten")
config = Config(
    model=str(args.model.resolve()),
    upstream="mlx",
    public_model=f"{NAME}/hardware",
    rank=2,
    accumulation=1,
    response_tokens=32,
    hindsight_tokens=128,
    max_sequence=512,
    min_train=3,
    min_heldout=3,
    min_new=1,
)
config.save(args.data_dir)
store = Store(args.data_dir)
for index in range(40):
    context = [
        {
            "role": "user",
            "content": f"Example {index}: explain how to correct a Python function call.",
        }
    ]
    first = {"role": "assistant", "content": "Pass the command in the command keyword."}
    followup = context + [
        first,
        {
            "role": "user",
            "content": "Actually the function accepts code, not command. The successful call used code='print(4)'. Explain the corrected keyword.",
        },
    ]
    last = {"role": "assistant", "content": "Use the code keyword."}
    for messages, action in ((context, first), (followup, last)):
        store.record(
            "/v1/chat/completions",
            {"model": config.public_model, "messages": messages},
            {"choices": [{"message": action}]},
            binding="base",
        )
print(args.data_dir)
