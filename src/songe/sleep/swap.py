import logging
from pathlib import Path

import httpx


async def install(
    store, upstream, config, version, expected, can_commit=lambda: True, generation=None
):
    """Prepare/load/healthcheck first. SQLite's current+previous transaction is
    the only commit point; a crash before it keeps the old pointer intact.
    """
    incumbent = store.version()
    from ..config import Config

    def consent():
        return (
            Config.load(store.root).replace_loaded_model
            if (store.root / "config.toml").exists()
            else config.replace_loaded_model
        )

    ticket = None
    if (
        consent()
        and incumbent
        and not incumbent.get("adapter")
        and version.get("night", 0) > 0
        and Path(incumbent["artifact"]).resolve()
        == Path(
            version.get("base") or config.model_ref.get("path") or config.model
        ).resolve()
        and incumbent["artifact"] != version["artifact"]
        and hasattr(upstream, "replacement_ticket")
    ):
        ticket = await upstream.replacement_ticket(incumbent)
    try:
        await upstream.load(version)
        payload = {
            "model": version["identifier"]
            if config.upstream != "mlx"
            else version["artifact"],
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "max_tokens": 8,
        }
        if config.upstream == "mlx":
            payload["adapters"] = version.get("adapter")
        if hasattr(upstream, "request_model"):
            payload["model"] = upstream.request_model(version)
        url = (
            upstream.request_url(version)
            if hasattr(upstream, "request_url")
            else config.upstream_url
        )
        async with httpx.AsyncClient(
            timeout=config.upstream_timeout, trust_env=False
        ) as client:
            response = await client.post(url + "/chat/completions", json=payload)
            response.raise_for_status()
            if not response.json().get("choices"):
                raise RuntimeError("Candidate did not return a completion")
        if not can_commit():
            if incumbent:
                await upstream.load(incumbent)
            return False
        store.add_version(version)
        store.swap(version["id"], expected, generation)
    except BaseException:
        # llama.cpp switches adapters globally; restore it as well as the
        # SQLite pointer on failed health checks or cancelled publication.
        current = store.version() or incumbent
        if current:
            await upstream.load(current)
        raise
    if ticket and consent():
        try:
            await upstream.release_replaced(ticket)
        except Exception:
            # Publication has already committed. Cleanup failure must never
            # roll it back or claim the user's instance as Songe-owned.
            logging.getLogger(__name__).warning(
                "Replaced base remains loaded", exc_info=True
            )
    return True


async def collect(store, upstream, in_flight):
    # Only the serving export stays; an older version keeps its adapter and is
    # fused again if the user returns to it (retention.restore_artifact).
    keep = {store.get("current"), "base"} | {
        k for k, count in in_flight.items() if count
    }
    keep.update(v["id"] for v in store.versions() if v["id"].startswith("base-"))
    stale = [r for r in store.versions() if r["id"] not in keep]
    for version in stale:
        artifact = Path(version["artifact"])
        adapters = [
            Path(p).resolve()
            for p in (version.get("training_adapter"), version.get("adapter"))
            if p
        ]
        # Adapter-only serving artifacts are small and retained too. Never
        # remove a directory containing the training adapter or base weights.
        if version.get("kind") not in {None, "mlx-fused", "gguf-full"}:
            continue
        if any(
            p == artifact.resolve() or p.is_relative_to(artifact.resolve())
            for p in adapters
        ):
            continue
        await upstream.unload(version)
        if hasattr(upstream, "prune_version"):
            await upstream.prune_version(version)
        if not artifact.exists():
            continue
        from ..artifacts.runtime import remove_owned
        from .cleanup import collect_derived

        retained = {
            Path(p).resolve()
            for v in store.versions()
            for k in ("adapter", "training_adapter")
            if (p := v.get(k))
        }
        collect_derived(store.root, artifact, [artifact], retained)
        remove_owned(artifact, store.get("installation_id"))
    from ..config import Config
    from .cleanup import collect_work

    config = (
        Config.for_model(store.root, store.model_key)
        if store.model_key
        else Config.load(store.root)
    )
    # The last few versions can be returned to (rebuilt from their adapters);
    # older ones are forgotten, or every update would stay on disk forever.
    older = sorted(
        (v for v in store.versions() if v["id"] not in keep),
        key=lambda v: v.get("night") or 0,
        reverse=True,
    )
    store.drop_versions(
        keep
        | {store.get("previous")}
        | {v["id"] for v in older[: config.kept_versions]}
    )
    collect_work(store, config, protected=keep)
