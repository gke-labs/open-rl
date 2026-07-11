#!/usr/bin/env python3
"""Run an end-to-end (E2E) training/RL benchmark job on the Kubernetes cluster.

Stamps the scenario, image, and extra arguments into k8s/eval/e2e-client-job.yaml,
applies it with kubectl, and streams the logs. Stdlib only - no uv needed:

  python3 scripts/run_cluster_e2e.py --scenario fft-gsm8k-rl-x2 \\
    --args "base_model=Qwen/Qwen3-8B steps=30 jitter_sec=5" --image gcr.io/cdrollouts-sunilarora/open-rl-client:latest

--print-only shows the rendered manifest and kubectl commands without running anything.
"""

import argparse
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "k8s" / "eval" / "e2e-client-job.yaml"
JOB = "open-rl-e2e-client"


def render_manifest(scenario: str, extra_args_str: str, image: str) -> str:
  manifest = MANIFEST.read_text(encoding="utf-8")
  for placeholder, value in [("E2E-IMAGE", image), ("E2E-SCENARIO", scenario)]:
    if placeholder not in manifest:
      raise RuntimeError(f"{MANIFEST} no longer contains the {placeholder} placeholder")
    manifest = manifest.replace(placeholder, value)

  extra_args = shlex.split(extra_args_str) if extra_args_str else []
  args_yaml = "\n".join(f'        - "{arg}"' for arg in extra_args) if extra_args else ""
  if '        - "E2E-EXTRA-ARGS"' not in manifest:
    raise RuntimeError(f"{MANIFEST} no longer contains the E2E-EXTRA-ARGS placeholder")
  manifest = manifest.replace('        - "E2E-EXTRA-ARGS"', args_yaml)
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scenario", required=True, help="E2E test scenario (e.g. fft-gsm8k-rl-x2).")
  parser.add_argument("--args", default="", help="Extra arguments to pass to run_training_e2e.py.")
  parser.add_argument("--image", required=True, help="Client container image to run.")
  parser.add_argument("--namespace", default="", help="Kubernetes namespace (defaults to the kubectl context's).")
  parser.add_argument("--no-follow", action="store_true", help="Launch the job but do not follow its logs.")
  parser.add_argument("--print-only", action="store_true", help="Print the kubectl commands and manifest; run nothing.")
  args = parser.parse_args()

  kubectl = ["kubectl"] + (["-n", args.namespace] if args.namespace else [])
  manifest = render_manifest(args.scenario, args.args, args.image)
  commands = [
    kubectl + ["delete", "job", JOB, "--ignore-not-found"],
    kubectl + ["apply", "-f", "<manifest>"],
    kubectl + ["wait", "--for=condition=Ready", "pod", "-l", f"job-name={JOB}", "--timeout=600s"],
    kubectl + ["logs", "-f", f"job/{JOB}"],
  ]

  if args.print_only:
    for command in commands:
      print("$ " + " ".join(command))
    print("\n# manifest applied at <manifest>:\n")
    print(manifest)
    return

  with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    f.write(manifest)
    manifest_path = f.name

  subprocess.run(commands[0], check=True)
  subprocess.run(kubectl + ["apply", "-f", manifest_path], check=True)
  if args.no_follow:
    print(f"[cluster-e2e] launched job/{JOB}; follow it with: {' '.join(kubectl)} logs -f job/{JOB}")
    return
  for _ in range(10):
    res = subprocess.run(commands[2])
    if res.returncode == 0:
      break
    time.sleep(2)
  else:
    raise RuntimeError(f"Timed out waiting for pod job-name={JOB}")
  subprocess.run(commands[3], check=True)
  done = subprocess.run(kubectl + ["wait", "--for=condition=Complete", f"job/{JOB}", "--timeout=30s"], capture_output=True, text=True)
  if done.returncode != 0:
    print(f"[cluster-e2e] job did not complete cleanly; inspect with: {' '.join(kubectl)} describe job/{JOB}")
    sys.exit(1)


if __name__ == "__main__":
  main()
