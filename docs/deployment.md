# Kubernetes Deployment (GKE)

The open-rl server is designed to run on a Kubernetes cluster with NVIDIA GPUs.

### 1. Build and Push the Image
The image is built using Docker BuildKit and pushed to Google Container Registry (GCR).

> [!TIP]
> You can change your GCP Project by editing the `GCP_PROJECT` variable in the `Makefile` (defaults to `cdrollouts-sunilarora`), or pass it via CLI: `make build-server-images GCP_PROJECT=my-project-id`.

```bash
make build-server-images
make push-server-images
```

### 2. Deploy to the Cluster

The architecture requires a `ReadWriteMany` network file system for model adapter synchronization. You have two options below:

#### Option A: Simple Shared File System (GKE Filestore/NFS)
*Recommended for simplicity and ease of management.*

Ensure the Cloud Filestore API is enabled on your GCP project:

```bash
gcloud services enable file.googleapis.com
```

You must also enable the GCP Filestore CSI driver addon on your GKE cluster:

```bash
gcloud container clusters update your-cluster-name \
    --location=your-compute-region \
    --update-addons=GcpFilestoreCsiDriver=ENABLED
```

#### Option B: High-Performance Parallel File System (Managed Lustre)
*Recommended for absolute highest throughput tensor caching across massive node counts.*

Ensure the Cloud Storage for Lustre API is enabled:

```bash
gcloud services enable lustre.googleapis.com
```

Configure **Private Services Access** for your VPC network (required for Lustre backend access):

```bash
# 1. Enable the Service Networking API
gcloud services enable servicenetworking.googleapis.com

# 2. Allocate an IP range for Google managed services
gcloud compute addresses create google-managed-services-default \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=20 \
    --description="Peering for Managed Lustre" \
    --network=default

# 3. Create the private connection
gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=google-managed-services-default \
    --network=default
```

Enable the Managed Lustre CSI driver addon on GKE:

```bash
gcloud container clusters update your-cluster-name \
    --location=your-compute-region \
    --update-addons=LustreCsiDriver=ENABLED
```

**Configuring OpenTelemetry (GKE Workload Identity)**:
To securely allow the GKE pods to export telemetry data (like Trainer GPU utilization) to Google Cloud Trace without granting node-level permissions, you must bind the `Cloud Trace Agent` role to the Kubernetes ServiceAccount (`open-rl-sa`) using Direct Workload Identity Federation:

```bash
PROJECT_ID="YOUR_PROJECT_ID"
PROJECT_NUM=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="principal://iam.googleapis.com/projects/${PROJECT_NUM}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/open-rl-sa" \
    --role="roles/cloudtrace.agent" \
    --condition=None
```

Apply the Kubernetes manifests. The deployment spins up a fully distributed, multi-node architecture utilizing your chosen shared file system for adapter synchronization:
1. **`open-rl-gateway`**: The PyTorch Training Gateway Deployment (Allocated to its dedicated L4 GPU node)
2. **`vllm-worker`**: The vLLM Inference Worker Deployment (Allocated to its dedicated L4 GPU node, horizontally scalable)
3. **`redis-broker`**: The Async Workload State Broker Deployment
4. **`open-rl-(shared|lustre)-pvc`**: A 1.2TB network file system mapped universally to the pods.

Depending on which storage option you chose above, **edit the `kustomization.yaml` within that directory to specify your GCP Project ID** (by default it uses `cdrollouts-sunilarora`), and then apply via `-k` (Kustomize):

```bash
# If using Option A (Filestore NFS):
kubectl apply -k server/kubernetes/distributed-shared/

# OR if using Option B (Managed Lustre):
kubectl apply -k server/kubernetes/distributed-lustre/

# Watch the distinct pods transition to Running status
kubectl get pods -l 'app in (open-rl-gateway, vllm, redis)' -w
```

### 4. Connect to the Server
The service is exposed internally as a `ClusterIP`. To connect your local SDK client to the GKE deployment, set up a secure port-forward to the PyTorch Gateway service:

```bash
kubectl port-forward svc/open-rl-gateway-service 8000:8000
```
Your SDK clients (e.g. `ServiceClient(base_url="http://localhost:8000")`) will now route traffic directly to the distributed GKE cluster.

> [!TIP]
> If a local process gets stuck on port 8000 from an old port-forward or server run, you can instantly terminate it with `make kill-server`.
