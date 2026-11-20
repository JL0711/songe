"""Read-only verification of base weights on externally managed servers."""

from pathlib import Path

import httpx

from ..models.discovery import _cached_path


async def served_base(config, version, *, transport=None, url=None):
    if version.get("adapter") or version.get("kind") in {
        "gguf-lora",
        "peft-adapter",
        "mlx-adapter",
    }:
        return None
    url = (
        url or getattr(config, "server_source_url", "") or config.upstream_url
    ).rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    artifact = Path(version["artifact"]).resolve()
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=5, trust_env=False
        ) as client:
            if config.upstream == "llamacpp":
                response = await client.get(url.removesuffix("/v1") + "/props")
                response.raise_for_status()
                if Path(response.json()["model_path"]).resolve() != artifact:
                    return None
            response = await client.get(url + "/models")
            response.raise_for_status()
            for model in response.json()["data"]:
                if model.get("parent"):
                    continue
                path = _cached_path(model.get("root") or model["id"])
                if config.upstream == "llamacpp" or path.resolve() == artifact:
                    return {"model": model["id"], "url": url}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
        return None
    return None
