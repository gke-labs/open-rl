# Open-RL TODOs & Future Improvements

## 1. Do Not Cache Worker Pod Templates in Memory
- **Current Behavior:** `KubernetesFFTWorkerManager.__init__` reads and parses pod templates (`/etc/open-rl/trainer/trainer-worker-pod.yaml`) once at startup and stores them in memory (`self.trainer_template`).
- **Improvement:** Read and parse the template file dynamically from disk on every `render_pod()` invocation so live ConfigMap updates take effect immediately without requiring a rolling restart of the gateway deployment (`kubectl rollout restart deployment open-rl-gateway`).

## 2. Implement Reliable Queue Acknowledgment for Training Steps
- **Current Behavior:** The trainer requests processor pops training request items immediately from Redis upon picking up a step.
- **Improvement:** Ensure that during a trainer worker crash or OOM kill, the queue item is dequeued / acknowledged only **after** the full completion of the training step (specifically, post saving the updated model checkpoint weights back to shared persistent storage). If interrupted earlier, the item should remain unacknowledged so another pod instance can safely retry the training step.
