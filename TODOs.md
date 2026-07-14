# Open-RL Future Action Items (`TODOs.md`)

## 1. HuggingFace Hub Token (`HF_TOKEN`) Injection via ConfigMap / Secret

### Background & Problem Statement
When running models on Kubernetes that are either **gated** (`e.g., Google's Gemma 2 / Gemma 4 families, Meta's Llama 3 weights`) or **large unauthenticated checkpoints** (`e.g., 5.6+ GB safetensors via xet-core / xorbs CDN`), HuggingFace Hub enforces strict rate-limits and `403 Forbidden` throttling on unauthenticated requests (`user_id=public`). 

In multi-pod cluster setups (`make cluster-e2e`), this throttling causes `xet_client` to enter lengthy URL refresh/retry loops, increasing first-time model caching times from seconds to over 15+ minutes. Furthermore, gated models (`google/gemma-4-e2b`, `google/functiongemma-270m-it`) fail to download altogether without authentication.

### Proposed Architecture & Action Items

#### A. Cluster-Level Secret / ConfigMap Creation
1. Define a Kubernetes Secret (`or ConfigMap for non-sensitive environments`) inside the cluster namespaces (`default` / target E2E namespace):
   ```bash
   # Create via CLI or Kustomize secret generator
   kubectl create secret generic open-rl-hf-secret \
     --from-literal=HF_TOKEN=<your_huggingface_token> \
     --from-literal=HUGGING_FACE_HUB_TOKEN=<your_huggingface_token> \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

#### B. Dynamic Worker Pod Injection (`k8s_worker_manager.py`)
1. Update `render_pod()` inside `src/server/k8s_worker_manager.py` to optionally pull `open-rl-hf-secret` (or ConfigMap) into all dynamically spawned `Trainer` and `Sampler` worker containers via `envFrom`:
   ```python
   container.setdefault("envFrom", []).append({
       "secretRef": {
           "name": "open-rl-hf-secret",
           "optional": True,  # Ensures open models (e.g. Qwen) still run without error if the secret is absent
       }
   })
   ```
2. Alternatively, if `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is present in the Gateway pod's local environment, `render_pod()` should forward it explicitly down to the child worker containers using `set_env(container, "HF_TOKEN", os.getenv("HF_TOKEN", ""))`.

#### C. Static & Distributed Manifest Alignment (`k8s/deploy/`)
1. Update standard static deployment manifests (`k8s/deploy/single-process-gke/`, `k8s/deploy/distributed-fft-timeslice/`) to include an optional `envFrom` block targeting `open-rl-hf-secret` on Gateway and Worker deployments:
   ```yaml
   envFrom:
     - secretRef:
         name: open-rl-hf-secret
         optional: true
   ```

### Expected Impact
- **Zero Rate-Limiting**: Bypasses public `xorbs` CDN rate limits completely.
- **Fast Cold-Start Download**: Reduces first-time download times for multi-gigabyte models from ~15+ minutes to `<20 seconds` at full gigabit cluster bandwidth.
- **Universal Model Compatibility**: Unlocks immediate cluster E2E testing for all gated HuggingFace checkpoints (`Gemma 2, Gemma 4, Llama 3.1`).
