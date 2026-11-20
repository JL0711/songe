"""Write an isolated calibration config without loading Metal or changing agents."""

import argparse
from pathlib import Path

from songe.config import Config
from songe.storage import private_dir
from songe.upstream.lmstudio import downloads_folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/Volumes/disk/songe-build-cache/calibration"),
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/Volumes/disk/local-llm/models/lmstudio-community/granite-4.2-3b-MLX-4bit"
        ),
    )
    args = p.parse_args()
    root = private_dir(args.output)
    Config(
        model=str(args.model.resolve()),
        public_model="songe/calibration",
        models_dir=str(downloads_folder()),
        max_sequence=2048,
    ).save(root)
    print(root / "config.toml")


if __name__ == "__main__":
    main()
