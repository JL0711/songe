"""Local model references and conversion entry point; no ML runtime on import."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    upstream: str
    serving_id: str
    format: str
    path: Path
    architecture: str


def trainable_base(ref, backend_kind, platform, **kwargs):
    from .convert import trainable_base as convert

    return convert(ref, backend_kind, platform, **kwargs)


__all__ = ["ModelRef", "trainable_base"]
