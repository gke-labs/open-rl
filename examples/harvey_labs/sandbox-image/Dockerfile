# LAB's sandbox image with sandboxd as its entrypoint, for Agent Sandbox.
# BASE is LAB's image built from its sandbox/ directory; cloudbuild.yaml
# builds both at the pinned LAB revision.
ARG BASE
# sandboxd has no released image yet; this is the staging build of main.
FROM us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandboxd@sha256:b0d648fe11080b7c31ff211256a27cfcfe98f95deef20c298633c81bdfb37983 AS sandboxd

FROM ${BASE}
COPY --from=sandboxd /usr/local/bin/sandboxd /usr/local/bin/sandboxd
RUN chown -R 1000:1000 /workspace
# LibreOffice and pip want a writable home.
ENV HOME=/tmp
USER 1000
ENTRYPOINT ["/usr/local/bin/sandboxd", "--root-dir=/workspace"]
