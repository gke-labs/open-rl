# Open-RL Agent Instructions

Welcome, Agent! This guide outlines the project structure, environments, and execution workflows for developing and testing the Open-RL framework.

---

## 0. Development Setup Scenario
In most scenarios, developers work on local machines (such as macOS or Linux laptops) that do **not** have local NVIDIA GPUs. Instead, they use a remote GCP VM with NVIDIA GPUs (such as `b7`) as the dev-test target.
Many Makefile targets that need to interact with the remote machine accept a `REMOTE_HOST=<host_name>` parameter (e.g. `make push-vm REMOTE_HOST=b7`).

---

## 1. Project Environments

Open-RL uses `uv` for environment isolation. There are two primary environments:

- **Server-Side Environment (`src/server`)**: Contains the gateway server and worker controllers.
- **Client/Examples Environment (`examples`)**: Contains recipes, client-side SDK compatibility checks, and E2E integration test scripts.

Always run tasks using the appropriate Makefile targets (such as `make server`, `make vllm`, or `make test`). If you must execute custom scripts, make sure to target the correct environment using the appropriate project flag (e.g., `uv --project examples ...` or `uv --project src/server ...`).

---

## 2. Fast Syntax Validation & Running Unit Tests

### Fast Python Compilation Check (`Tip`)
Whenever you add or modify Python files (`.py`), run `py_compile` on the modified files to catch syntax and indentation errors instantly before running slower test suites or container builds:
```bash
python3 -m py_compile path/to/file1.py path/to/file2.py
```

### Python Class Attribute Best Practice (`Tip`)
When defining or refactoring classes, **always explicitly initialize all instance attributes inside `__init__`** and use direct property access (`self.property`) rather than defensive `hasattr(self, ...)` or `getattr(self, ...)` checks. Avoid writing overly defensive code that obscures uninitialized attribute bugs:
- **Correct**: `self._my_flag = False` in `__init__`, then `if self._my_flag:` later.
- **Avoid**: `if getattr(self, "_my_flag", False):` without explicit initialization.

### Linter & Formatting Checks (`Rule`)
Before creating commits or opening pull requests, **always run linter and formatting checks across the repository from the root directory (`/open-rl`)**:
```bash
export PATH=$PATH:$HOME/.local/bin && make lint
```
To automatically fix import ordering (`isort / I001`) and formatting discrepancies (`e.g. before submitting pull requests`), run:
```bash
export PATH=$PATH:$HOME/.local/bin && make fmt
```
*Note: Keep `import ...` statements at the module top-level where possible and verify lines stay within the 150-character limit to ensure CI checks pass under `ruff check`.*

### Running the Standard Unit Test Suite
To run the standard unit test suite:
```bash
make test
```
*Note: This command runs package discovery inside the client/examples environment. Because the Makefile targets run `uv` under the hood, you must ensure that `uv` is in your `PATH` (typically installed at `~/.local/bin`). For example, prepend `export PATH=$PATH:$HOME/.local/bin` to your command.*

---

## 3. Running End-to-End (E2E) GPU Integration Tests

E2E tests boot up a client harness and run actual SFT/RL training workflows against the Open-RL backend.

### Option A: Running In-Cluster via Kubernetes (`Preferred`)
When testing against a Kubernetes GPU cluster (e.g., GKE), the preferred and most reliable way to execute E2E integration benchmarks is using the `make cluster-e2e` target. This deploys an in-cluster client job (`open-rl-e2e-client`) that communicates directly with `open-rl-gateway-service:8000`:

```bash
make cluster-e2e IMAGE_TAG=$(cat VERSION 2>/dev/null || echo latest) \
  E2E_SCENARIO=<scenario_name> \
  E2E_ARGS="<optional_key_val_args>"
```

**Examples:**
* Run a 5-step `Qwen2.5-0.5B-Instruct` RL test:
  ```bash
  make cluster-e2e IMAGE_TAG=0.1.45 E2E_SCENARIO=fft-gsm8k-rl E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=5"
  ```
* Run a 30-step `Qwen3-8B` RL benchmark with 192 batch size (`groups_per_batch=24 × group_size=8`):
  ```bash
  make cluster-e2e IMAGE_TAG=0.1.45 E2E_SCENARIO=fft-gsm8k-rl E2E_ARGS="base_model=Qwen/Qwen3-8B steps=30 group_size=8 groups_per_batch=24 max_tokens=512"
  ```

Before launching a new cluster run, always clean up any stale client jobs or previous dynamic worker pods:
```bash
kubectl delete job -l app=open-rl-e2e-client --ignore-not-found
kubectl delete pods -l timeslice.io/group=trainers --ignore-not-found
kubectl delete pods -l timeslice.io/group=samplers --ignore-not-found
```

### Option B: Local Port-Forward Execution (`Alternative / Local Dev`)
If running a client script on a local machine against a remote Kubernetes cluster, use port-forwarding:
```bash
pkill -9 -f port-forward 2>/dev/null; nohup sh -c 'while true; do kubectl port-forward svc/open-rl-gateway-service 8000:8000 >/dev/null 2>&1; sleep 1; done' >/dev/null 2>&1 &
make test e2e <scenario_name> BASE_URL=http://127.0.0.1:8000
```

### Supported Scenarios:
- **`tiny-lora`**: Minimal overfit test using LoRA (asserts that loss drops).
- **`tiny-fft`**: Minimal overfit test using Full Fine-Tuning (*requires running `redis-server`*).
- **`tiny-rl`**: Simple sample -> reward -> train policy update loop.
- **`lora-textsql`**: A trimmed version of a real Reinforcement Learning recipe for Text-to-SQL.
- **`fft-gsm8k`**: Full fine-tuning SFT training + vLLM evaluation on 100 math problems (*requires `redis-server`*).
- **`fft-gsm8k-rl`**: Reinforcement Learning recipe for GSM8K math problems (supports `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-8B`, etc.).
- **`fft-gsm8k-x2`**: Runs two concurrent `fft-gsm8k` jobs sharing a single GPU via the Accelerator Time-Slicer.

---

## 4. Syncing & Testing on Remote GPU Hosts (e.g., `b7`)

### Synchronization:
To push your current workspace to a remote test machine:
```bash
make push-vm REMOTE_HOST=<host_name>
```
To pull changes back:
```bash
make pull-vm REMOTE_HOST=<host_name>
```

### Running Tests on the Remote Machine:

**Option A: Direct SSH Execution (Simple)**
Run the command directly via SSH:
```bash
ssh <host_name> "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && <test_command>"
```

**Option B: Within a Tmux Session (Optional)**
If there is a persistent active tmux session (e.g., `work`) on the remote machine, you can run tests and monitor them without losing progress if you disconnect:
1. Send the test command to the tmux session:
   ```bash
   ssh <host_name> 'tmux send-keys -t work "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && <test_command>" C-m'
   ```
2. Monitor the pane output:
   ```bash
   ssh <host_name> "tmux capture-pane -t work -p"
   ```

---

## 5. Required System Dependencies on VM
If you encounter errors during E2E training or evaluation on a fresh GPU VM, ensure these system packages are installed:

- **`redis-server`**: Required by the Accelerator Time-Slicer for memory/state synchronization in FFT/time-slicing scenarios (`sudo apt-get install -y redis-server`).
- **`python3-dev`**: Required for compiling custom Triton runtime kernels during vLLM engine initialization (`sudo apt-get install -y python3-dev`).

---

## 6. Repeatable Kubernetes & Deployment Workflows

When debugging or executing distributed E2E benchmarks on Kubernetes (such as `fft-gsm8k-rl-x2`), always follow these standard lifecycle workflows:

### Rebuilding & Pushing Container Images After Code Changes
Kubernetes worker pods pull and execute Python code (`/app`) directly from the baked container images (`gcr.io/<project>/open-rl-server:<tag>`). Whenever you modify Python code under `src/`, always bump the version in the `VERSION` file (and update corresponding K8s manifests), then rebuild and push the images before running tests:
```bash
make build-images push-images IMAGE_TAG=$(cat VERSION 2>/dev/null || echo latest)
```

> [!IMPORTANT]
> **Never Overwrite Tags or Rely on `Always` Pull Policies**: All dynamic worker pod templates intentionally employ `imagePullPolicy: IfNotPresent`. Re-pulling the massive 24GB image on every deployment takes upwards of 5+ minutes, which heavily throttles dev iterations. Consequently, if you rebuild an image without bumping the version, Kubernetes nodes will silently reuse the cached, stale local image. You **must always increment the tag** (e.g. `0.3.13` -> `0.3.14`) to guarantee a fresh layer download.

### Cleaning Up Stale Worker Pods & Background Tasks
Aborting an E2E test harness (`make test e2e ...`) leaves background client tasks and active Kubernetes worker pods running. Always terminate stale client tasks and clean up worker pods cleanly by label before relaunching runs:
```bash
kubectl delete job -l app=open-rl-e2e-client --ignore-not-found
kubectl delete pods -l timeslice.io/group=trainers --ignore-not-found
kubectl delete pods -l timeslice.io/group=samplers --ignore-not-found
```

### Kustomize Deployment & Base Manifest Best Practices (`Tip`)
To prevent noisy container image tag diffs inside pull requests while ensuring clean deployments across Kubernetes environments, always follow this Kustomize pattern:
1. **Deploy via Kustomize (`-k`)**: Always apply directory manifests via `kubectl apply -k <directory>` rather than `kubectl apply -f <directory>`.
2. **Keep Base YAMLs at `:latest` or Placeholder**: In all base/rendered YAML template files (`04-gateway.yaml`, `05-worker-pod-template.yaml`, `07-accel-timeslicer-daemonset.yaml`, `09-sampler-pod-template.yaml`, `04-deployment.yaml`, etc.), keep container image tags permanently set to a static placeholder (`e.g. image: ghcr.io/gke-labs/open-rl/server:latest` or `/gateway:latest`). **Never bump image tags inside these base template files.**
3. **Single Source of Truth for Version Bumps**: When releasing or bumping image versions (`e.g. 0.1.75 -> 0.1.76`), modify **only**:
   - `VERSION` (`at repository root`)
   - The `newTag:` field inside the target environment's `kustomization.yaml` (`e.g. k8s/deploy/distributed-fft-timeslice/kustomization.yaml`)

### Applying Manifest Edits & Restarting the Gateway (Dev Mode)
Since this is a development-only mode, formal rolling updates (which include waiting for rollout status) are not necessary. Simply apply the manifests via Kustomize and delete the active gateway pod to trigger an immediate, fast recreation:
```bash
kubectl apply -k k8s/deploy/distributed-fft-timeslice/
kubectl delete pods -l app=open-rl-gateway
```

### Resetting the Time-Slicer DaemonSet (Dev Mode)
If worker pods crash or lose their TCP connection (`9753`) to the time-slicer daemon, clean up old worker pods and force-delete the daemonset pods directly to speed up recovery:
```bash
kubectl delete pods -l app=open-rl-accel-timeslicer --grace-period=0 --force
```

### Monitoring Live Training Progression inside Gateway Pod
When a cluster benchmark job (`make cluster-e2e ...`) is running, live step-by-step metrics (`metrics.jsonl`) are written to shared NFS storage (`/mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/...`). To inspect a clean progression table of live metrics (`Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time`) directly inside the Gateway pod:
```bash
kubectl exec deployment/open-rl-gateway -- python3 -c '
import json, os
metrics_path = "/mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/fft_gsm8k_rl/metrics.jsonl"
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print("Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time")
    print("-" * 80)
    for row in rows:
        if "env/all/correct" in row:
            step = row.get("progress/batch", "?")
            corr = row.get("env/all/correct", 0.0)
            rew = row.get("env/all/reward/total", 0.0)
            t_samp = row.get("time/sampling", 0.0)
            t_train = row.get("time/train_step", 0.0)
            t_save = row.get("time/save_checkpoint", 0.0)
            t_total = row.get("time/total", 0.0)
            print(f"{str(step):>4} | {corr:>7.2%}  | {rew:>6.4f} | {t_samp:>7.1f}s | {t_train:>9.1f}s | {t_save:>9.1f}s | {t_total:>14.1f}s")
'
```
*(For `fft-gsm8k-rl-x2` concurrent dual jobs, iterate over `fft_gsm8k_rl_job-a` and `fft_gsm8k_rl_job-b` directories).*

### Standard Benchmark Run Archive Convention (`runs/` Directory)
When an end-to-end benchmark campaign completes, always archive the results into the repository's `runs/` directory using standard `<date>_<scenario>_<details>` naming (`e.g. runs/2026-07-11_qwen8b_fft_rl_x2_192batch_30steps/`):
1. Save raw telemetry logs (`metrics.jsonl`) from the Gateway pod:
   ```bash
   kubectl exec deployment/open-rl-gateway -- cat /mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/fft_gsm8k_rl/metrics.jsonl > runs/<run_dir>/metrics.jsonl
   ```
2. Write a comprehensive markdown benchmark report (`benchmark_report.md`) inside `runs/<run_dir>/` documenting executive findings, full step-by-step progression tables, timing breakdown, and hardware/concurrency performance.
