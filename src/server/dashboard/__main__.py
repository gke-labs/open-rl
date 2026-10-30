"""The dashboard as its own process, without the client API's queues or worker manager: `python -m server.dashboard`."""

import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse

from server.dashboard.router import router

app = FastAPI(title="Open-RL Inspection API")
app.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=1)  # a port-forward carries large JSON slowly
app.include_router(router)
app.get("/", include_in_schema=False)(lambda: RedirectResponse("/dashboard"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  # Flags only: Kubernetes injects OPEN_RL_DASHBOARD_PORT=tcp://... for a Service named open-rl-dashboard.
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  args = parser.parse_args()
  uvicorn.run(app, host=args.host, port=args.port)
