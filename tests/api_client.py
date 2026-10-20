"""An httpx client bound to the API server's ASGI app, for async tests.

Requests go through routing, body validation and the error handlers, so a test
exercises what the tinker client sees rather than a handler called as a function.
The lifespan is not run.
"""

import httpx

from server import api_server


def asgi_client() -> httpx.AsyncClient:
  return httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://test")


# Every model must name its dtypes. Tests about something else get bfloat16;
# tests about dtypes post to the client directly.
DEFAULT_DTYPES = {"openrl.trainer_dtype": "bfloat16", "openrl.sampler_dtype": "bfloat16"}
MODEL_PATHS = ("create_model", "create_model_from_state")


def with_dtypes(path: str, body: dict) -> dict:
  """body with bfloat16 dtypes added to a model-creating request that names none."""
  if path not in MODEL_PATHS or not isinstance(body, dict):
    return body
  metadata = body.get("user_metadata") or {}
  if any(key in metadata for key in DEFAULT_DTYPES):
    return body
  return {**body, "user_metadata": {**DEFAULT_DTYPES, **metadata}}


async def post_json(client: httpx.AsyncClient, path: str, body: dict, **kwargs) -> dict:
  response = await client.post(f"/api/v1/{path}", json=with_dtypes(path, body), **kwargs)
  response.raise_for_status()
  return response.json()
