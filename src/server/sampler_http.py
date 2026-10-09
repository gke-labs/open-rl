"""The HTTP side of a routed sampler: what its set's llm-d router calls
instead of the queue, served from the sampler's own engine."""

import asyncio
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

ENGINE_POLL_SECONDS = 5


class Engine(Protocol):
  errored: bool


class Generator(Protocol):
  engine: Engine

  async def generate(self, request: dict[str, Any]) -> dict[str, Any]: ...


def failed(message: str) -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message}


def http_app(sampler: Generator) -> FastAPI:
  """The body is OpenAI shaped so the router can read the prompt and model;
  its `openrl` field is the request the queue would have carried, and the
  answer is the same result."""
  app = FastAPI()

  @app.post("/v1/completions")
  async def completions(request: Request) -> JSONResponse:
    if sampler.engine.errored:
      return JSONResponse(failed("vLLM engine is dead"), status_code=503)
    task = None
    try:
      body = await request.json()
      task = asyncio.create_task(sampler.generate(body["openrl"]))
      while not task.done():
        await asyncio.wait({task}, timeout=ENGINE_POLL_SECONDS)
        if sampler.engine.errored:
          raise RuntimeError("vLLM engine is dead")
        if await request.is_disconnected():
          raise RuntimeError("Sampling client disconnected")
      result = task.result()
      result["type"] = "sample"
    except Exception as exc:
      result = failed(f"vLLM Worker Error: {exc}")
    finally:
      if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return JSONResponse(result)

  @app.get("/health")
  async def health() -> Response:
    return Response(status_code=503 if sampler.engine.errored else 200)

  @app.get("/metrics")
  async def metrics() -> Response:
    # vLLM's metrics; prometheus_client comes with vLLM, which only samplers install.
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

  return app


async def serve_http(sampler: Generator, port: int) -> None:
  server = uvicorn.Server(uvicorn.Config(http_app(sampler), host="0.0.0.0", port=port, log_level="warning"))
  await server.serve()
