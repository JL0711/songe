"""Turning learning on lists a model first; turning it off never moves it."""

from songe.models import registry


def test_set_learning_brings_model_to_front(config, store):
    for model in ("one", "two"):
        registry.register(config, store, model)
    registry.set_learning(store, "one", "new")
    registry.set_learning(store, "two", "all")
    assert store.get(registry.ORDER_KEY)[:2] == ["two", "one"]
    registry.set_learning(store, "two", "off")
    assert store.get(registry.ORDER_KEY)[:2] == ["two", "one"]
    registry.set_learning(store, "one", "all")
    assert store.get(registry.ORDER_KEY)[:2] == ["one", "two"]
