# Sandbox runtime for Text-to-SQL reward execution.
#
# The agent-sandbox reference runtime server (main.py, from
# kubernetes-sigs/agent-sandbox examples/python-runtime-sandbox) on :8888 plus
# /opt/run_sql.py, which executes one model-written query against a schema in
# an in-memory SQLite database. Nothing else is installed: sqlite3 is in the
# standard library and the sandbox has no network egress.
#
# Build (Cloud Build, from the repo root):
#   make cloud-build-sql-executor GCP_PROJECT=<project> CLOUD_IMAGE_TAG=<tag>
FROM python:3.12-slim

WORKDIR /app
# Pinned inline: the repo's .gitignore drops *.txt from the build context.
RUN pip install --no-cache-dir fastapi==0.138.0 uvicorn==0.49.0 python-multipart==0.0.32

COPY examples/text-to-sql/sandbox/main.py /opt/main.py
COPY examples/text-to-sql/sandbox/run_sql.py /opt/run_sql.py

# Commands and uploads are confined to SANDBOX_BASE_DIR; the template mounts
# an emptyDir there so the root filesystem can stay read-only.
ENV SANDBOX_BASE_DIR=/app/work \
    SANDBOX_EXEC_TIMEOUT_SECONDS=30 \
    PYTHONUNBUFFERED=1
RUN mkdir -p /app/work && chown -R 1000:1000 /app
USER 1000
EXPOSE 8888
CMD ["uvicorn", "--app-dir", "/opt", "main:app", "--host", "0.0.0.0", "--port", "8888", "--log-level", "warning"]
