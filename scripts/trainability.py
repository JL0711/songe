"""Run the same bounded candidate probes as init without writing agent configs."""

import argparse
import json
from pathlib import Path

from songe.schedule import probe
from songe.sleep.probe import probe_models
from songe.storage import private_dir
from songe.upstream.lmstudio import downloads_folder, local_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/Volumes/disk/songe-build-cache/trainability"),
    )
    p.add_argument("--model", type=Path)
    args = p.parse_args()
    models = [args.model.resolve()] if args.model else local_models(downloads_folder())
    result = probe_models(models, private_dir(args.output), probe())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
