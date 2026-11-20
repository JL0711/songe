"""Loopback benchmark against a fixed fake upstream, with recording enabled."""

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from songe.config import Config
from songe.log import Store
from songe.proxy import create_app
from tests.test_network import live_server


async def main():
    wire = b'{"model":"test-base","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}'

    async def answer(request):
        return Response(wire, media_type="application/json")

    upstream = Starlette(
        routes=[Route("/v1/chat/completions", answer, methods=["POST"])]
    )
    root = Path(
        tempfile.mkdtemp(
            prefix="proxy-benchmark-", dir="/Volumes/disk/songe-build-cache"
        )
    )
    store = Store(root)
    config = Config(
        model="/fixture", public_model="songe/benchmark", base_identifier="test-base"
    )

    class Passive:
        current_version = None
        reason = "Benchmark"

        def traffic(self):
            pass

        def acquire(self, *a):
            pass

        def release(self, *a):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

    with live_server(upstream) as direct:
        config.upstream_url = direct + "/v1"
        with live_server(create_app(config, store, scheduler=Passive())) as proxy:
            async with httpx.AsyncClient(trust_env=False) as client:
                timings = {}
                for name, url, model in (
                    ("direct", direct, "test-base"),
                    ("proxy", proxy, "songe/benchmark"),
                ):
                    values = []
                    for i in range(170):
                        start = time.perf_counter()
                        response = await client.post(
                            url + "/v1/chat/completions",
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": "hello"}],
                            },
                        )
                        assert response.content == wire
                        elapsed = (time.perf_counter() - start) * 1000
                        if i >= 20:
                            values.append(elapsed)
                    timings[name] = {
                        "p50_ms": statistics.median(values),
                        "p95_ms": sorted(values)[int(len(values) * 0.95)],
                        "requests": len(values),
                    }
    result = {
        "scope": "local HTTP, fake fixed upstream, recording enabled, 20 warmup + 150 measured per path",
        "timings": timings,
        "added_p50_ms": timings["proxy"]["p50_ms"] - timings["direct"]["p50_ms"],
        "recorded": len(store.exchanges()),
    }
    Path("docs/evidence/proxy-benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
