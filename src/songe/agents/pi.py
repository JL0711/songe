import copy

from .. import NAME
from ..models.names import exposed


def configure(data, config):
    data = copy.deepcopy(data)
    providers = data.setdefault("providers", {})
    existing = providers.get(NAME, {})
    from .opencode import context_for

    # Songe's own entries follow the served context; other fields survive.
    models = [
        m
        | (
            {
                "contextWindow": context_for(config, m["id"]),
                "maxTokens": min(4096, context_for(config, m["id"]) // 2),
            }
            if m.get("id") in exposed(config)
            else {}
        )
        for m in existing.get("models", [])
    ]
    present = {m["id"] for m in models}
    providers[NAME] = {
        **existing,
        "baseUrl": f"http://127.0.0.1:{config.port}/v1",
        "api": "openai-completions",
        "apiKey": "local",
        "models": models
        + [
            {
                "id": model,
                "name": model,
                "input": ["text"],
                "contextWindow": context_for(config, model),
                "maxTokens": min(4096, context_for(config, model) // 2),
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            }
            for model in exposed(config)
            if model not in present
        ],
    }
    return data
