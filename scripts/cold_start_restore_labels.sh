#!/usr/bin/env bash
# Put the openrl.io role labels back to what cold_start_case.sh found.
#   scripts/cold_start_restore_labels.sh runs/<dir>/node-labels-before.json
set -euo pipefail
python3 - "$1" <<'EOF'
import json, subprocess, sys
for node, labels in json.load(open(sys.argv[1])).items():
  for role in ("trainer", "sampler"):
    key = f"openrl.io/{role}"
    if key in labels:
      subprocess.run(["kubectl", "label", "node", node, f"{key}={labels[key]}", "--overwrite"], check=True, capture_output=True)
    else:
      subprocess.run(["kubectl", "label", "node", node, f"{key}-"], check=False, capture_output=True)
  print(node, {k: v for k, v in labels.items() if k.startswith("openrl.io/")})
EOF
