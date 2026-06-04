"""Small CLI for launching autoresearch recipes."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import chz

from harness.utils import AgentPaths

ROOT = Path(__file__).parents[1]
IGNORED_RECIPE_DIRS = {"__pycache__", ".git", "gke"}
IGNORED_RECIPE_FILES = {".DS_Store", "kustomization.yaml"}


@chz.chz
class RunConfig:
  recipe_dir: Path
  session_name: str
  output_dir: Path = ROOT / ".runs"
  namespace: str = "default"
  context: str = ""
  apply: bool = True
  image: str = "ghcr.io/gke-labs/open-rl/client:latest"
  log_root: str = "/mnt/shared/open-rl/autoresearch"
  tinker_base_url: str = "http://open-rl-gateway-service:8000"
  attempt_timeout_minutes: float = 30
  agent_timeout_minutes: float = 10


@dataclass(frozen=True)
class Overlay:
  path: Path
  task_name: str
  session_name: str
  agent_id: str
  log_dir: Path
  sandbox_name: str
  ui_service_name: str


def session_id(value: str) -> str:
  value = slug(value, "session_name", 40)
  return f"session-{value}" if value.isdigit() else value


def slug(value: str, field: str, limit: int) -> str:
  normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  if not normalized:
    raise SystemExit(f"{field} is required")
  if len(normalized) <= limit:
    return normalized
  suffix = hashlib.sha1(normalized.encode()).hexdigest()[:8]
  return f"{normalized[: limit - 9].rstrip('-')}-{suffix}"


def task_id(recipe_name: str, session: str) -> str:
  return slug(f"{recipe_name}-{session}", "task", 39)


def recipe_files(recipe_dir: Path) -> list[Path]:
  files = []
  for path in sorted(recipe_dir.iterdir()):
    if path.is_dir() and path.name not in IGNORED_RECIPE_DIRS:
      raise SystemExit(f"recipe directories are not supported yet: {path}")
    if path.is_file() and path.name not in IGNORED_RECIPE_FILES:
      files.append(path)
  required = {"program.md", "autoresearch.toml"}
  missing = sorted(required - {path.name for path in files})
  if missing:
    raise SystemExit("recipe is missing required files: " + ", ".join(missing))
  return files


def write_overlay(args: RunConfig) -> Overlay:
  recipe_dir = args.recipe_dir.resolve()
  session = session_id(args.session_name)
  task_name = task_id(recipe_dir.name, session)
  agent = session
  recipe_name = recipe_dir.name
  overlay_name = task_name
  overlay = (args.output_dir / overlay_name).resolve()
  try:
    overlay.relative_to(ROOT.resolve())
  except ValueError:
    raise SystemExit(f"output_dir must be under {ROOT} so kubectl apply -k can load the base") from None
  recipe_copy = overlay / "recipe"
  image_name, image_tag = args.image, "latest"
  if ":" in args.image.rsplit("/", 1)[-1]:
    image_name, image_tag = args.image.rsplit(":", 1)

  if overlay.exists():
    shutil.rmtree(overlay)
  recipe_copy.mkdir(parents=True)
  for source in recipe_files(recipe_dir):
    shutil.copy2(source, recipe_copy / source.name)

  recipe_config_map = f"autoresearch-recipe-{task_name}"
  run_config_map = f"autoresearch-run-{task_name}"
  prefix = "open-rl-"
  base_ref = os.path.relpath((ROOT / "k8s/base").resolve(), overlay)
  file_lines = "\n".join(f"  - {path.name}=recipe/{path.name}" for path in sorted(recipe_copy.iterdir()))
  shared_literals = f"  - LOG_ROOT={args.log_root.rstrip('/')}"
  run_literals = "\n".join(
    f"  - {key}={value}"
    for key, value in {
      "RECIPE": f"recipes/{recipe_name}/autoresearch.toml",
      "LOG_ROOT": args.log_root.rstrip("/"),
      "RUN_NAME": task_name,
      "TINKER_BASE_URL": args.tinker_base_url,
      "ATTEMPT_TIMEOUT_MINUTES": args.attempt_timeout_minutes,
      "AGENT_TIMEOUT_MINUTES": args.agent_timeout_minutes,
      "AGENT_FLAGS": "--include-directories /mnt/shared --yolo --output-format stream-json",
      "ENABLE_GCP_TRACE": "1",
    }.items()
  )

  (overlay / "kustomization.yaml").write_text(
    f"""resources:
- {base_ref}

namePrefix: {prefix}

labels:
- includeSelectors: false
  pairs:
    app.kubernetes.io/part-of: open-rl-autoresearch
    autoresearch.open-rl.dev/task: {task_name}
    open-rl/session: {session}
    open-rl/agent: {agent}

generatorOptions:
  disableNameSuffixHash: true

configMapGenerator:
- name: autoresearch-settings
  behavior: merge
  literals:
{shared_literals}
- name: {run_config_map}
  literals:
{run_literals}
- name: {recipe_config_map}
  files:
{file_lines}

images:
- name: ghcr.io/gke-labs/open-rl/client
  newName: {image_name}
  newTag: {image_tag}

patches:
- target:
    kind: Sandbox
    name: autoresearch-agent-1
  patch: |-
    - op: replace
      path: /metadata/name
      value: autoresearch-{task_name}
    - op: replace
      path: /metadata/labels/open-rl~1agent
      value: "{agent}"
    - op: add
      path: /metadata/labels/open-rl~1session
      value: "{session}"
    - op: replace
      path: /spec/podTemplate/spec/containers/0/env/0
      value:
        name: AGENT_ID
        value: "{agent}"
    - op: replace
      path: /spec/podTemplate/spec/containers/0/envFrom/0/configMapRef/name
      value: {run_config_map}
    - op: add
      path: /spec/podTemplate/spec/volumes/-
      value:
        name: recipe-files
        configMap:
          name: {recipe_config_map}
    - op: add
      path: /spec/podTemplate/spec/containers/0/volumeMounts/-
      value:
        name: recipe-files
        mountPath: /app/autoresearch/recipes/{recipe_name}
        readOnly: true
""",
    encoding="utf-8",
  )
  return Overlay(
    path=overlay,
    task_name=task_name,
    session_name=session,
    agent_id=agent,
    log_dir=AgentPaths.from_run(Path(args.log_root.rstrip("/")), task_name, agent).root,
    sandbox_name=f"{prefix}autoresearch-{task_name}",
    ui_service_name=f"{prefix}autoresearch-ui",
  )


def kubectl_args(args: RunConfig, overlay: Path) -> list[str]:
  command = ["kubectl"]
  if args.context:
    command.extend(["--context", args.context])
  if args.namespace:
    command.extend(["--namespace", args.namespace])
  command.extend(["apply", "-k", str(overlay)])
  return command


def run(args: RunConfig) -> None:
  overlay = write_overlay(args)
  print(f"wrote autoresearch overlay: {overlay.path}")
  print(f"run: {overlay.task_name}")
  print(f"session: {overlay.session_name}")
  print(f"logs: {overlay.log_dir}")
  print(f"kubernetes sandbox: {overlay.sandbox_name}")
  if args.apply:
    subprocess.run(kubectl_args(args, overlay.path), check=True)
    print(f"UI: kubectl port-forward svc/{overlay.ui_service_name} 8080:8080")
  else:
    print("render: kubectl kustomize " + str(overlay.path))
    print("apply:  " + " ".join(kubectl_args(args, overlay.path)))


def main(argv: list[str] | None = None) -> None:
  argv = list(sys.argv[1:] if argv is None else argv)
  if argv and argv[0] == "run":
    argv.pop(0)
  if argv and "=" not in argv[0]:
    argv[0] = f"recipe_dir={argv[0]}"
  args = chz.entrypoint(RunConfig, argv=argv, allow_hyphens=True)
  run(args)


if __name__ == "__main__":
  main()
