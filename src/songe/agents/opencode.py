import copy

from .. import NAME
from ..models.names import exposed


def context_for(config, model):
    """The context the model server really serves this model with."""
    return int(
        (getattr(config, "served_contexts", None) or {}).get(model)
        or config.context_window
    )


def configure(data, config, default=None, v2=False):
    """Add Songe's models. default: True always, False never, None automatic."""
    data = copy.deepcopy(data)
    # The installed binary's version determines the schema. Existing v2 config
    # always takes precedence, even if the binary is currently unavailable.
    v2 = v2 or "providers" in data
    provider = {
        "name": "Songe",
        "models": {
            model: {
                "name": model,
                "limit": {
                    "context": context_for(config, model),
                    "output": min(4096, context_for(config, model) // 2),
                },
            }
            for model in exposed(config)
        },
    }
    if v2:
        provider.update(
            package="@opencode/ai/providers/openai-compatible",
            settings={
                "baseURL": f"http://127.0.0.1:{config.port}/v1",
                "apiKey": "local",
            },
        )
        providers = data.setdefault("providers", {})
    else:
        provider.update(
            npm="@ai-sdk/openai-compatible",
            options={
                "baseURL": f"http://127.0.0.1:{config.port}/v1",
                "apiKey": "local",
            },
        )
        providers = data.setdefault("provider", {})
    existing = providers.get(NAME, {})
    # Songe owns its own entries: current limits win, other fields survive.
    previous = existing.get("models", {})
    provider["models"] = {
        **previous,
        **{
            name: {**previous.get(name, {}), **entry}
            for name, entry in provider["models"].items()
        },
    }
    providers[NAME] = {**existing, **provider}
    models = exposed(config)
    # The model the user chose, not whichever sorts first in the library.
    chosen = f"{NAME}/" + (
        config.public_model if config.public_model in models else models[0]
    )
    if default is None:
        # Automatic: local models go through Songe so it can learn from them;
        # a cloud default is the user's choice and has nothing to learn.
        if not data.get("model") or local_model(data, data["model"]):
            # The model init chose already follows the one the agent last ran
            # (opencode's recent pick wins over its config default), so the
            # agent's default is the model that learns, not a stale default.
            data["model"] = chosen
    elif default:
        data["model"] = chosen
    return data


LOCAL_PROVIDERS = {"lmstudio", "ollama", "llamacpp", "llama.cpp", "mlx", "vllm"}


def local_model(data, reference):
    """A model served on this machine by something other than Songe."""
    provider = str(reference).partition("/")[0]
    if provider == NAME:
        return False
    if provider in LOCAL_PROVIDERS:
        return True
    settings = (data.get("providers") or data.get("provider") or {}).get(provider) or {}
    url = str(
        (settings.get("settings") or settings.get("options") or {}).get("baseURL") or ""
    )
    return any(host in url for host in ("127.0.0.1", "localhost", "[::1]"))
