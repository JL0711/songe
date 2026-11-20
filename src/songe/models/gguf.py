"""Read GGUF metadata using the official memory-mapped reader, never weights."""

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GGUFInfo:
    architecture: str
    quantization: str
    parameters: int
    file_type: int
    embedding: bool = False


_CACHE: dict = {}
_LOCK = threading.Lock()


def read_gguf(path: Path) -> GGUFInfo:
    """Parsing a GGUF walks every metadata array (tokenizers hold ~10^5
    entries), so results are cached per file identity; a changed file is
    read again."""
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = _read_gguf(path)
        return _CACHE[key]


def _read_gguf(path: Path) -> GGUFInfo:
    try:
        from gguf import GGUFReader, LlamaFileType
    except ImportError as exc:
        raise RuntimeError(
            "GGUF inspection requires the gguf dependency; reinstall songe"
        ) from exc
    reader = GGUFReader(path, mode="r")
    architecture = reader.get_field("general.architecture")
    file_type = reader.get_field("general.file_type")
    if architecture is None or file_type is None:
        raise ValueError(
            f"GGUF lacks general.architecture or general.file_type: {path}"
        )
    value = int(file_type.contents())
    quantization = (
        LlamaFileType(value).name.removeprefix("MOSTLY_").removeprefix("ALL_")
    )
    parameters = sum(int(t.n_elements) for t in reader.tensors)
    name = str(architecture.contents())
    pooling = reader.get_field(f"{name}.pooling_type")
    return GGUFInfo(
        name, quantization, parameters, value, bool(pooling and pooling.contents())
    )
