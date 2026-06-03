# Snapshot Agent

The snapshot agent is a small process-level GPU residency primitive.

It exposes four commands over a Unix socket:

- `REGISTER(run_id, pid)` records the process that owns a run.
- `ACQUIRE(run_id)` grants that process the right to touch CUDA.
- `RELEASE(run_id)` checkpoints that process before another run can acquire CUDA.
- `UNREGISTER(run_id)` removes the process registration.

Today every successful `RELEASE` checkpoints the process. This is simple and
conservative, but it is slow because even a single run pays checkpoint cost after
each acquire window.
