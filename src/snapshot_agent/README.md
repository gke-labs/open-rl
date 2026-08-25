# Snapshot Agent

The snapshot agent is a small process-level GPU residency primitive.

It exposes four commands over a Unix socket:

- `REGISTER(pid)` records a worker process.
- `ACQUIRE(pid)` grants that process the right to touch CUDA.
- `RELEASE(pid)` checkpoints that process before another process can acquire CUDA.
- `UNREGISTER(pid)` removes the process registration.

Today every successful `RELEASE` checkpoints the process. This is simple and
conservative, but it is slow because even a single run pays checkpoint cost after
each acquire window.
