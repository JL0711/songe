import httpx
import pytest

from songe.sleep.swap import collect, install
from songe.storage import write_json
from tests.test_store import version


class Upstream:
    def __init__(self, fail=False):
        self.fail = fail
        self.loaded = []
        self.unloaded = []

    async def load(self, v):
        if self.fail:
            raise RuntimeError("failed load")
        self.loaded.append(v["id"])

    async def unload(self, v):
        self.unloaded.append(v["id"])


async def test_failed_load_does_not_commit(config, store):
    store.add_version(version("base"))
    store.swap("base", None)
    with pytest.raises(RuntimeError, match="failed load"):
        await install(store, Upstream(fail=True), config, version("night-1", 1), "base")
    assert store.get("current") == "base"
    assert store.get("previous") is None


async def test_healthcheck_before_atomic_commit_and_cas(config, store, monkeypatch):
    store.add_version(version("base"))
    store.swap("base", None)
    real = httpx.AsyncClient

    async def handler(request):
        assert store.get("current") == "base"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello"}}]}
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler))
    )
    await install(store, Upstream(), config, version("night-1", 1), "base")
    assert store.get("current") == "night-1"
    assert store.get("previous") == "base"


async def test_gc_keeps_only_current_inflight_and_unowned(tmp_path, store):
    for name in ["base", "old", "previous", "current", "busy", "unowned"]:
        artifact = tmp_path / name
        artifact.mkdir()
        if name != "unowned":
            write_json(
                artifact / "songe_owned.json",
                {"installation_id": store.get("installation_id")},
            )
        store.add_version(version(name) | {"artifact": str(artifact)})
    store.swap("previous", None)
    store.swap("current", "previous")
    upstream = Upstream()
    await collect(store, upstream, {"busy": 1})
    # An older version keeps its adapter and is fused again on return.
    assert not (tmp_path / "old").exists()
    assert not (tmp_path / "previous").exists()
    assert all(
        (tmp_path / name).is_dir() for name in ["base", "current", "busy", "unowned"]
    )
    assert store.get("current") == "current"


async def test_request_arriving_during_healthcheck_defers_pointer_commit(
    config, store, monkeypatch
):
    from songe.schedule import Scheduler

    store.add_version(version("base"))
    store.swap("base", None)
    scheduler = Scheduler(config, store, Upstream())
    revision = scheduler.traffic_revision
    real = httpx.AsyncClient

    async def handler(request):
        scheduler.traffic()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ready"}}]}
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler))
    )
    result = await install(
        store,
        scheduler.upstream,
        config,
        version("candidate", 1),
        "base",
        can_commit=lambda: scheduler.traffic_revision == revision,
    )
    assert result is False
    assert store.get("current") == "base"
    assert store.get("previous") is None
