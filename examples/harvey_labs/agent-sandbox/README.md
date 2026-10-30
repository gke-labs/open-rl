# LAB sandboxes on Agent Sandbox

Each LAB episode runs its tools in its own sandbox: a pod from
[Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox) running LAB's
sandbox image with `sandboxd` as its entrypoint. The sandboxes live in their own
GKE cluster. The driver, the recipe running in the training cluster,
claims a sandbox per episode through the sandbox cluster's API, and talks to
`sandboxd` in it by pod IP. That works because both clusters are VPC-native on
the same network.

```
training cluster                         sandbox cluster
  harvey-labs driver ─claims──────────────▶ API server (private endpoint)
        │                                  SandboxWarmPool lab-sandbox
        └──files :8080 / exec :9090─────▶ sandbox pods (gVisor, no egress)
```

The values below are the ones this setup used; replace them with yours.
`10.112.0.0/14` is the training cluster's pod range, and the sandbox image is
built into your own registry in step 3.

## 1. The cluster

```sh
gcloud container clusters create open-rl-sandboxes --region us-central1 \
  --node-locations us-central1-a --network default --subnetwork default \
  --enable-ip-alias --release-channel None --no-enable-autoupgrade \
  --num-nodes 1 --machine-type e2-standard-4
# NetworkPolicy enforcement, so the template's no-egress policy applies.
gcloud container clusters update open-rl-sandboxes --region us-central1 --update-addons=NetworkPolicy=ENABLED
gcloud container clusters update open-rl-sandboxes --region us-central1 --enable-network-policy
# The sandbox nodes, isolated with gVisor.
gcloud container node-pools create gvisor --cluster open-rl-sandboxes --region us-central1 \
  --node-locations us-central1-a --machine-type n2-standard-32 --image-type COS_CONTAINERD \
  --sandbox type=gvisor --enable-autoscaling --num-nodes 2 --min-nodes 1 --max-nodes 8 \
  --no-enable-autoupgrade
```

Auto-upgrade is off because a node upgrade kills every sandbox on the node
mid-episode.

## 2. Reachability from the training cluster

The driver reaches `sandboxd` on the sandbox pods directly. Allow the training
cluster's pod range in, on those two ports only. The target tag is on the
sandbox cluster's nodes (`gcloud compute instances list --filter=name~open-rl-sandboxes --format="value(tags.items)"`).

```sh
gcloud compute firewall-rules create open-rl-sandboxes-from-training --network default \
  --direction INGRESS --source-ranges 10.112.0.0/14 \
  --target-tags gke-open-rl-sandboxes-<hash>-node --allow tcp:8080,tcp:9090
```

## 3. Agent Sandbox and the LAB sandboxes

```sh
gcloud builds submit --config examples/harvey_labs/sandbox-image/cloudbuild.yaml \
  --substitutions=_IMAGE=<registry>/lab-sandbox-asb,_TAG=v2 examples/harvey_labs/sandbox-image

kubectl --context <sandbox-cluster> apply --server-side \
  -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v1.0.5/sandbox.yaml \
  -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v1.0.5/extensions.yaml
kubectl --context <sandbox-cluster> apply -f examples/harvey_labs/agent-sandbox/sandboxes.yaml
```

`sandboxes.yaml` holds the namespace, the template (gVisor, non-root, no egress,
ingress only from the training pods), a warm pool of 120 and the driver's
ServiceAccount, scoped to claims in `lab-sandboxes`. Set its image to the one
you built. Keep the warm pool at least as large as the episodes in flight: a
step's rollouts plus its eval tasks.

## 4. The driver's access

The driver uses a kubeconfig for the sandbox cluster, mounted from a Secret in
the training cluster. It points at the sandbox cluster's private endpoint and
authenticates with the ServiceAccount's token.

```sh
ENDPOINT=$(gcloud container clusters describe open-rl-sandboxes --region us-central1 --format="value(privateClusterConfig.privateEndpoint)")
CA=$(gcloud container clusters describe open-rl-sandboxes --region us-central1 --format="value(masterAuth.clusterCaCertificate)")
TOKEN=$(kubectl --context <sandbox-cluster> -n lab-sandboxes get secret harvey-driver-token -o jsonpath='{.data.token}' | base64 -d)
cat > driver.kubeconfig <<EOF
apiVersion: v1
kind: Config
clusters: [{name: sandboxes, cluster: {server: "https://$ENDPOINT", certificate-authority-data: $CA}}]
users: [{name: harvey-driver, user: {token: $TOKEN}}]
contexts: [{name: sandboxes, context: {cluster: sandboxes, user: harvey-driver, namespace: lab-sandboxes}}]
current-context: sandboxes
EOF
kubectl --context <training-cluster> -n <job namespace> create secret generic sandbox-cluster-kubeconfig \
  --from-file=kubeconfig=driver.kubeconfig
```

Run the driver with `KUBECONFIG` pointing at it and
`automountServiceAccountToken: false`, so the sandbox client does not pick up
the training cluster's credentials.

## Running

Build the driver image from the repository root, set it, the judge endpoint
and `RUN` in `../cluster-job.yaml`, then apply it:

```sh
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE=<registry>/harvey-labs-client,_DOCKERFILE=examples/harvey_labs/Dockerfile,_TAG=<tag> .
kubectl apply -f examples/harvey_labs/cluster-job.yaml
```

Without `sandbox_warmpool`, the recipe runs sandboxes in local Podman as before.
