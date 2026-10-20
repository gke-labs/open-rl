"""Render N in-cluster RL client Jobs for the fleet demo, from a short spec.

  python3 scripts/demo_jobs.py --image <client image> --run demo-01 \\
      --job lora:Qwen/Qwen3-0.6B:3 --job fft:google/gemma-4-e2b:2 --steps 3 | kubectl apply -f -

Each --job is kind:model:count. kind is lora or fft. Every Job runs the GSM8K
GRPO recipe (scripts/run_training_e2e.py, scenario lora-gsm8k-rl or
fft-gsm8k-rl) against the API server's in-cluster Service, so no port-forward
is involved. Jobs are named <run>-<kind>-<model slug>-<n> and labeled
openrl.io/demo-run=<run>, so one selector lists, watches or deletes them all.
Each Job writes its recipe output under
/mnt/shared/open-rl/runs/demo/<run>/<job>/, where scripts/demo_report.py
reads the step timings.
"""

import argparse
import re
import sys

TEMPLATE = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: openrl-demo
    openrl.io/demo-run: "{run}"
    openrl.io/demo-kind: "{kind}"
    openrl.io/demo-model: "{slug}"
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels:
        app: openrl-demo
        openrl.io/demo-run: "{run}"
    spec:
      restartPolicy: Never
      serviceAccountName: open-rl-sa
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: client
        image: {image}
        imagePullPolicy: IfNotPresent
        args:
        - "scenario={scenario}"
        - "uv_extra=cpu"
        - "base_url=http://open-rl-api-server-service:8000"
        - "base_model={model}"
        - "steps={steps}"
        - "group_size={group_size}"
        - "groups_per_batch={groups_per_batch}"
        - "max_tokens={max_tokens}"
        - "log_dir=/mnt/shared/open-rl/runs/demo/{run}/{name}"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        # openrl.* session settings, e.g. openrl.sampler_replicas=2; the SDK
        # sends TINKER_TAGS as the session's tags and the API server reads them.
        - name: TINKER_TAGS
          value: "{tags}"
        - name: HF_HOME
          value: "/hf"
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
          limits:
            memory: "8Gi"
        volumeMounts:
        - name: shared-storage
          mountPath: /mnt/shared
        - name: hf-cache
          mountPath: /hf
      # Each client gets its own Hugging Face cache on an emptyDir, seeded from the
      # shared one. Twelve clients locking one cache directory on NFS at the same
      # time lost a client to "Stale file handle" in filelock (drill 2). The seed
      # is the dataset plus the model's tokenizer and config files; weight files
      # (anything over 64 MiB) are skipped, the client never reads them.
      initContainers:
      - name: seed-hf-cache
        image: busybox:1.36
        command: ["sh", "-c"]
        args:
        - |
          set -e
          S=/mnt/shared/open-rl/huggingface; D=/hf; M={hub_dir}
          mkdir -p $D/hub $D/datasets
          [ -d $S/datasets ] && cp -a $S/datasets/. $D/datasets/
          [ -d $S/hub/datasets--openai--gsm8k ] && cp -a $S/hub/datasets--openai--gsm8k $D/hub/
          if [ -d $S/hub/$M ]; then
            cd $S/hub
            find $M -type d -exec mkdir -p $D/hub/{{}} \\;
            find $M \\( -type l -o \\( -type f -size -65536k \\) \\) -exec cp -a {{}} $D/hub/{{}} \\;
          fi
          du -sh $D
        volumeMounts:
        - name: shared-storage
          mountPath: /mnt/shared
        - name: hf-cache
          mountPath: /hf
      volumes:
      - name: shared-storage
        persistentVolumeClaim:
          claimName: open-rl-shared-pvc
      - name: hf-cache
        emptyDir: {{}}
"""


def slug(model: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:30]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--image", required=True)
  parser.add_argument("--run", required=True, help="name of this demo run; also the label value")
  parser.add_argument("--job", action="append", required=True, help="kind:model:count, kind in {lora, fft}")
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--group-size", type=int, default=8)
  parser.add_argument("--groups-per-batch", type=int, default=8)
  parser.add_argument("--max-tokens", type=int, default=512)
  parser.add_argument("--namespace", default="openrl-system")
  parser.add_argument("--lora-tags", default="", help="TINKER_TAGS for the LoRA jobs, e.g. openrl.sampler_replicas=2")
  parser.add_argument("--fft-tags", default="", help="TINKER_TAGS for the FFT jobs")
  args = parser.parse_args()
  out = []
  for spec in args.job:
    kind, model, count = spec.rsplit(":", 2)
    if kind not in ("lora", "fft"):
      sys.exit(f"kind must be lora or fft, got {kind!r}")
    for n in range(1, int(count) + 1):
      name = f"{args.run}-{kind}-{slug(model)}-{n:02d}"
      out.append(
        TEMPLATE.format(
          name=name,
          namespace=args.namespace,
          run=args.run,
          kind=kind,
          slug=slug(model),
          image=args.image,
          scenario=f"{kind}-gsm8k-rl",
          model=model,
          hub_dir="models--" + model.replace("/", "--"),
          steps=args.steps,
          group_size=args.group_size,
          groups_per_batch=args.groups_per_batch,
          max_tokens=args.max_tokens,
          tags=args.lora_tags if kind == "lora" else args.fft_tags,
        )
      )
  sys.stdout.write("".join(out))


if __name__ == "__main__":
  main()
