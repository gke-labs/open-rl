# The Harvey LAB driver for a cluster run. Sandboxes run on Agent Sandbox,
# so the image needs no Podman, only the client, LAB, and LAB's grading venv.
# Build from the repository root:
#   gcloud builds submit --config cloudbuild.yaml \
#     --substitutions=_IMAGE=<registry>/harvey-labs-client,_DOCKERFILE=examples/harvey_labs/Dockerfile,_TAG=<tag> .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

ARG LAB_REPO=https://github.com/ShubyM/harvey-labs
ARG LAB_REF=401bf7f0448044cb112556d26492652d56e8c317
RUN git clone --filter=blob:none "$LAB_REPO" /app/harvey_labs/harvey-labs \
    && git -C /app/harvey_labs/harvey-labs checkout "$LAB_REF"
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/harvey_labs/harvey-labs && uv sync --frozen

WORKDIR /app
COPY examples/pyproject.toml examples/uv.lock examples/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra agent-sandbox

COPY examples/common ./common
COPY examples/harvey_labs ./harvey_labs

ENV PYTHONPATH=/app
ENTRYPOINT ["/app/.venv/bin/python", "-m", "harvey_labs.train"]
