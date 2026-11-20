"""Purge external artifacts before destroying their durable ownership records."""

import json
import shutil
from pathlib import Path

import httpx

from ..artifacts.runtime import remove_owned


async def purge_imports(upstream, store, identifiers=None):
    from .ollama import _tag

    imports = store.get("runtime_imports", {})
    for identifier, version in list(imports.items()):
        if identifiers is not None and identifier not in identifiers:
            continue
        if version.get("installation_id") != store.get("installation_id"):
            raise RuntimeError(
                "External import ownership is unverifiable; recovery records preserved"
            )
        tags = (await upstream._request("GET", "/api/tags")).get("models", [])
        tag = next((m for m in tags if _tag(m["name"]) == _tag(identifier)), None)
        if tag is None and version.get("state") != "created":
            raise RuntimeError(
                "External publication is unconfirmed; recovery records preserved"
            )
        if tag is not None:
            if not await upstream.same_weights(version) or (
                version.get("manifest_digest")
                and tag.get("digest") != version["manifest_digest"]
            ):
                # The user replaced our tag. It is no longer ours to reclaim.
                imports.pop(identifier)
                store.set("runtime_imports", imports)
                continue
            running = [
                m
                for m in await upstream._running()
                if _tag(m.get("name", m.get("model"))) == _tag(identifier)
            ]
            if running and (
                not tag.get("digest")
                or any(m.get("digest") != tag["digest"] for m in running)
            ):
                raise RuntimeError(
                    "External runner identity changed; recovery records preserved"
                )
            if running:
                await upstream._warm(identifier, 0)
            try:
                await upstream._request(
                    "DELETE", "/api/delete", json={"model": identifier}
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
        remaining = (await upstream._request("GET", "/api/tags")).get("models", [])
        if any(_tag(m["name"]) == _tag(identifier) for m in remaining) or any(
            _tag(m.get("name", m.get("model"))) == _tag(identifier)
            for m in await upstream._running()
        ):
            raise RuntimeError(
                "External import cleanup incomplete; recovery records preserved"
            )
        imports.pop(identifier)
        store.set("runtime_imports", imports)


async def purge(config, store):
    from . import create

    if store.get("runtime_imports"):
        await purge_imports(create(config, store.root), store)
    claims = store.get("lmstudio_claims", {}).get("instances", {})
    if claims:
        upstream = create(config, store.root)
        upstream.owned = {}
        await upstream.recover()
        for version in list(upstream.owned.values()):
            await upstream.unload(version)
        if upstream.owned:
            raise RuntimeError(
                "External loads are still pending or unverifiable; recovery records preserved"
            )
    purge_library(config, store)


def purge_library(config, store):
    if not config.models_dir:
        return
    identity = store.get("installation_id")
    namespace = Path(config.models_dir) / config.publisher
    for marker in sorted(namespace.rglob("*songe_owned.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        artifact = (
            marker.parent
            if marker.name == "songe_owned.json"
            else marker.with_name(marker.name.removesuffix(".songe_owned.json"))
        )
        if (
            data.get("installation_id") == identity
            and not artifact.is_symlink()
            and artifact.resolve().is_relative_to(namespace.resolve())
        ):
            remove_owned(artifact, identity)
    # A crash during MLX fusion precedes the final ownership marker. The
    # staging journal and its originating work lease prove the same ownership.
    from ..artifacts.storage import artifact_roots

    roots = list(artifact_roots(store.root))
    for marker in (
        marker
        for directory in [namespace, *roots]
        for marker in sorted(directory.rglob("songe_fusing.json"))
    ):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        work = Path(data.get("adapter", "/")).parent
        lease = work / "songe_work.json"
        destination = Path(data.get("destination", "/"))
        if (
            not any(
                work.resolve().is_relative_to((r / "versions").resolve()) for r in roots
            )
            or not lease.is_file()
            or data.get("base") != config.model
            or not marker.parent.name.startswith("." + destination.name + "-")
        ):
            continue
        if json.loads(lease.read_text(encoding="utf-8")).get(
            "installation_id"
        ) == identity and any(
            marker.parent.resolve().is_relative_to(r.resolve())
            for r in [namespace, *roots]
        ):
            shutil.rmtree(marker.parent)
