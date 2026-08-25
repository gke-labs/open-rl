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

## 2. Running Unit Tests

To run the standard unit test suite:
```bash
make test
```
*Note: This command runs package discovery inside the client/examples environment. Because the Makefile targets run `uv` under the hood, you must ensure that `uv` is in your `PATH` (typically installed at `~/.local/bin`). For example, prepend `export PATH=$PATH:$HOME/.local/bin` to your command.*

---

## 3. Running End-to-End (E2E) GPU Integration Tests

E2E tests boot up a local backend and run actual SFT/RL training against it. Run them using the `gpu` extra environment (making sure `uv` is in your `PATH` first):
```bash
make test e2e <scenario_name>
```

### Supported Scenarios:
- **`tiny-lora`**: Minimal overfit test using LoRA (asserts that loss drops).
- **`tiny-fft`**: Minimal overfit test using Full Fine-Tuning (*requires running `redis-server`*).
- **`tiny-rl`**: Simple sample -> reward -> train policy update loop.
- **`lora-textsql`**: A trimmed version of a real Reinforcement Learning recipe for Text-to-SQL.
- **`fft-gsm8k`**: Full fine-tuning SFT training + vLLM evaluation on 100 math problems (*requires `redis-server`*).
- **`fft-gsm8k-x2`**: Runs two concurrent `fft-gsm8k` jobs sharing a single GPU via the Checkpoint/Restore Snapshot Agent.

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

- **`redis-server`**: Required by the Snapshot Agent for memory/state synchronization in FFT/time-slicing scenarios (`sudo apt-get install -y redis-server`).
- **`python3-dev`**: Required for compiling custom Triton runtime kernels during vLLM engine initialization (`sudo apt-get install -y python3-dev`).
