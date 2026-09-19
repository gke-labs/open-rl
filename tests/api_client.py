"""An httpx client bound to the API server's ASGI app, for async tests.

Requests go through routing, body validation and the error handlers, so a test
exercises what the tinker client sees rather than a handler called as a function.
The lifespan is not run.
"""

from contextlib import contextmanager
from unittest.mock import patch

import httpx

from server import api_server
from server.api_runtime import ApiRuntime
from server.store import InMemoryStateStore, InMemoryStore


@contextmanager
def runtime_context(store=None, worker_manager=None, *, state=None):
  runtime = ApiRuntime(
    store if store is not None else InMemoryStore(),
    state if state is not None else InMemoryStateStore(),
    worker_manager,
    api_server.TMP_DIR,
  )
  with patch.object(api_server.app.state, "runtime", runtime, create=True):
    yield runtime


def asgi_client() -> httpx.AsyncClient:
  return httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://test")


async def post_json(client: httpx.AsyncClient, path: str, body: dict, **kwargs) -> dict:
  response = await client.post(f"/api/v1/{path}", json=body, **kwargs)
  response.raise_for_status()
  return response.json()
