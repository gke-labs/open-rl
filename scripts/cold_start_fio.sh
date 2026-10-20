#!/usr/bin/env bash
# Read benchmark of the shared volume from a GPU node, with fio, against the
# real model files. Answers "bandwidth-bound or IOPS-bound" for the weights load.
#
#   scripts/cold_start_fio.sh <node-name> <out-dir> [model-dir-under-hub]
#   scripts/cold_start_fio.sh gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8 runs/<dir>/fio models--Qwen--Qwen3-8B
#
# Runs a pod on that node with the open-rl-shared-pvc mounted, installs fio,
# then: (1) sequential read of the largest safetensors file, 1 MiB blocks,
# O_DIRECT so the page cache does not answer; (2) 4 KiB random reads on the
# same file for 30 s, O_DIRECT; (3) the NFS client's own counters before and
# after, from /proc/self/mountstats. Output: fio JSON and mountstats in <out-dir>.
set -euo pipefail
NODE=$1; OUT=$2; MODEL_DIR=${3:-models--Qwen--Qwen3-8B}
NS=${OPENRL_NAMESPACE:-openrl-system}
POD=cold-start-fio
mkdir -p "$OUT"
kubectl -n "$NS" delete pod "$POD" --ignore-not-found --wait=true >/dev/null
cat <<EOF | kubectl -n "$NS" apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $POD
spec:
  restartPolicy: Never
  nodeName: $NODE
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: fio
    image: ubuntu:24.04
    command: ["bash", "-c", "apt-get update -qq >/dev/null && apt-get install -y -qq fio >/dev/null && sleep 3600"]
    volumeMounts:
    - name: shared
      mountPath: /mnt/shared
  volumes:
  - name: shared
    persistentVolumeClaim:
      claimName: open-rl-shared-pvc
EOF
kubectl -n "$NS" wait --for=condition=Ready pod/$POD --timeout=5m >/dev/null
for _ in $(seq 1 60); do kubectl -n "$NS" exec $POD -- which fio >/dev/null 2>&1 && break; sleep 3; done
FILE=$(kubectl -n "$NS" exec $POD -- bash -c "ls -S /mnt/shared/open-rl/huggingface/hub/$MODEL_DIR/snapshots/*/*.safetensors | head -1")
echo "file: $FILE" | tee "$OUT/file.txt"
kubectl -n "$NS" exec $POD -- bash -c "ls -l $FILE; grep -A30 ' /mnt/shared ' /proc/self/mountstats" > "$OUT/mountstats-before.txt"
kubectl -n "$NS" exec $POD -- fio --name=seqread --filename="$FILE" --rw=read --bs=1M --direct=1 --ioengine=psync --numjobs=1 --runtime=60 --time_based=0 --output-format=json > "$OUT/fio-seqread-1M-direct.json"
kubectl -n "$NS" exec $POD -- fio --name=randread --filename="$FILE" --rw=randread --bs=4k --direct=1 --ioengine=psync --numjobs=1 --runtime=30 --time_based=1 --output-format=json > "$OUT/fio-randread-4k-direct.json"
# The trainer's observed pattern: one synchronous ~256 KiB read at a time.
kubectl -n "$NS" exec $POD -- fio --name=seqread-256k-qd1 --filename="$FILE" --rw=read --bs=256k --direct=1 --ioengine=psync --numjobs=1 --runtime=30 --time_based=1 --output-format=json > "$OUT/fio-seqread-256k-qd1-direct.json"
# Headroom: four readers at once, 1 MiB each, different offsets of the same file.
kubectl -n "$NS" exec $POD -- fio --name=seqread-1M-4jobs --filename="$FILE" --rw=read --bs=1M --direct=1 --ioengine=psync --numjobs=4 --offset_increment=20% --runtime=30 --time_based=1 --group_reporting --output-format=json > "$OUT/fio-seqread-1M-4jobs-direct.json"
kubectl -n "$NS" exec $POD -- fio --name=seqread-buffered --filename="$FILE" --rw=read --bs=1M --direct=0 --ioengine=psync --numjobs=1 --runtime=60 --time_based=0 --output-format=json > "$OUT/fio-seqread-1M-buffered.json"
kubectl -n "$NS" exec $POD -- bash -c "grep -A30 ' /mnt/shared ' /proc/self/mountstats" > "$OUT/mountstats-after.txt"
kubectl -n "$NS" delete pod "$POD" --wait=false >/dev/null
python3 - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for name in ("fio-seqread-1M-direct", "fio-randread-4k-direct", "fio-seqread-256k-qd1-direct", "fio-seqread-1M-4jobs-direct", "fio-seqread-1M-buffered"):
  d = json.load(open(out / f"{name}.json"))["jobs"][0]["read"]
  print(f"{name:26} {d['bw_bytes']/2**20:8.1f} MiB/s  {d['iops']:9.1f} IOPS  mean lat {d['lat_ns']['mean']/1e6:7.2f} ms  read {d['io_bytes']/2**30:.2f} GiB in {d['runtime']/1000:.1f} s")
EOF
