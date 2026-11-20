"""Restore immutable exports from retained adapters, through guarded exporters."""

from pathlib import Path


def restore_artifact(store, config, version, platform, check=lambda: None):
    artifact = Path(version["artifact"])
    if (
        artifact.exists()
        or not version.get("adapter")
        or version.get("kind") not in {"mlx-fused", "gguf-full"}
    ):
        return artifact
    adapter = version.get("training_adapter") or version.get("adapter")
    if not adapter or not Path(adapter).exists():
        raise ValueError("Retained version has no adapter available for restoration")
    from ..artifacts.storage import (
        artifact_root,
        preflight,
        require_publish_volume,
        tensor_sizes,
    )
    from ..platform import Rooted
    from ..schedule import base_fingerprint
    from .fuse import export_candidate

    record = next(
        (n["record"] for n in store.nights() if n["id"] == version["night"]), {}
    )
    if record.get("base_fingerprint") and record[
        "base_fingerprint"
    ] != base_fingerprint(config):
        raise ValueError("Base weights changed since this version was trained")
    scratch = artifact_root(config, store)
    require_publish_volume(scratch, artifact)
    if version["kind"] == "gguf-full":
        dense, delta = tensor_sizes(config.model, config.rank)
        # Merged weights, converter scratch and final output coexist.
        preflight([(scratch, 3 * dense + delta)])
    else:
        from ..upstream.protection import weight_bytes

        # The same need sleep.fuse checks: an 8-bit fuse of a 4-bit base is at
        # most twice its size, plus the adapter.
        preflight(
            [(scratch, 2 * weight_bytes(config.model) + weight_bytes(Path(adapter)))]
        )
    check()
    import time

    from .fuse import PROGRESS

    # Only the serving export is kept; an older one is rebuilt here, and the
    # dashboard shows this progress meanwhile.
    (store.root / PROGRESS).unlink(missing_ok=True)
    store.set(
        "restoring",
        {"version": version["id"], "model": store.model_key, "started": time.time()},
    )
    try:
        export_candidate(
            config,
            version["kind"],
            Path(adapter),
            artifact,
            Rooted(platform, store.root),
            check,
            progress=store.root / PROGRESS,
        )
    finally:
        store.set("restoring", None)
        (store.root / PROGRESS).unlink(missing_ok=True)
    return artifact
