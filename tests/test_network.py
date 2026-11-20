import asyncio
import contextlib
import socket
import threading
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from songe.proxy import create_app


@contextlib.contextmanager
def live_server(app):
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="error",
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Test HTTP server failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("Test server did not stop")


async def test_real_http_streams_early_and_disconnect_cancels(config, store):
    disconnected = threading.Event()

    async def endpoint(request):
        async def chunks():
            try:
                yield b'data: {"choices":[]}\n\n'
                await asyncio.sleep(10)
                yield b"data: [DONE]\n\n"
            finally:
                disconnected.set()

        return StreamingResponse(chunks(), media_type="text/event-stream")

    upstream = Starlette(
        routes=[Route("/v1/chat/completions", endpoint, methods=["POST"])]
    )
    with live_server(upstream) as upstream_url:
        config.upstream_url = upstream_url + "/v1"
        proxy = create_app(config, store)
        with live_server(proxy) as proxy_url:
            async with httpx.AsyncClient(trust_env=False) as client:
                started = time.monotonic()
                async with client.stream(
                    "POST",
                    proxy_url + "/v1/chat/completions",
                    json={"model": "songe/test", "stream": True, "messages": []},
                ) as response:
                    first = await anext(response.aiter_raw())
                    assert first == b'data: {"choices":[]}\n\n'
                    assert time.monotonic() - started < 2
                assert await asyncio.to_thread(disconnected.wait, 2)
