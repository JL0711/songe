"""Stable public names, separate from learner keys and runtime identifiers."""

import re

from ..steps import digest


def public_name(model):
    if model.startswith("songe/"):
        return model
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.rsplit("/", 1)[-1]).strip("-")
    return f"songe/{name or 'local'}"


def catalog(config, store=None):
    from . import registry

    primary = config.model_ref.get("serving_id") or config.base_identifier
    result = {}
    if store:
        result.update(
            (row["id"], row["config"]["public_model"])
            for row in registry.entries(store)
        )
    for model in config.exposed_models:
        if model != config.public_model and model not in result.values():
            result.setdefault(model, public_name(model))
    if primary:
        result[primary] = config.public_model if config.model else public_name(primary)
    elif not result:
        result[config.public_model] = config.public_model
    # Publishers can ship different weights with the same folder name.
    claimed = {}
    for model in sorted(result, key=lambda model: model != primary):
        alias = result[model]
        if alias in claimed and claimed[alias] != model:
            result[model] = f"{alias}-{digest(model)[:8]}"
        claimed[result[model]] = model
    return result


def exposed(config):
    return list(dict.fromkeys(catalog(config).values()))
