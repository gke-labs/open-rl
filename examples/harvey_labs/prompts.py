"""System prompt, skills, and conversation prefix assembly for LAB episodes."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from gemma4_renderer import register_gemma4_tool_renderer
from reward import ARTIFACT_EXTENSIONS
from tasks import LabTask
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer

OUTPUT_FILE_RE = re.compile(rf"`([^`]+\.(?:{'|'.join(ARTIFACT_EXTENSIONS)}))`", re.IGNORECASE)


def default_skills(lab_root: Path) -> list[str]:
  return sorted(path.parent.name for path in (lab_root / "harness" / "skills").glob("*/SKILL.md"))


def lab_system_prompt(lab_root: Path) -> str:
  prompt = (lab_root / "harness" / "system_prompt.md").read_text(encoding="utf-8")
  for skill_name in default_skills(lab_root):
    skill_path = lab_root / "harness" / "skills" / skill_name / "SKILL.md"
    prompt += f"\n\n## Skill: {skill_name}\n\n{skill_path.read_text(encoding='utf-8')}"
  return prompt


def copy_skill_scripts(lab_root: Path, workspace_dir: Path) -> None:
  for skill_name in default_skills(lab_root):
    scripts_dir = lab_root / "harness" / "skills" / skill_name / "scripts"
    if scripts_dir.exists():
      shutil.copytree(
        scripts_dir,
        workspace_dir / "skills" / skill_name / "scripts",
        dirs_exist_ok=True,
      )


def initial_messages(
  task: LabTask,
  renderer: Renderer,
  system_prompt: str,
  tool_specs: list[dict[str, Any]],
) -> list[Message]:
  return renderer.create_conversation_prefix_with_tools(
    tools=tool_specs,
    system_prompt=system_prompt + artifact_path_prompt(task),
  ) + [{"role": "user", "content": task.instructions}]


def artifact_path_prompt(task: LabTask) -> str:
  """Restate the sandbox's asymmetric write/bash paths for requested outputs."""
  output_names = list(dict.fromkeys(OUTPUT_FILE_RE.findall(task.instructions)))
  if not output_names:
    return ""
  output_paths = ", ".join(f"`/workspace/output/{name}`" for name in output_names)
  docx_names = [name for name in output_names if name.lower().endswith(".docx")]
  docx_example = ""
  if docx_names:
    docx_example = (
      "\nFor a new DOCX authored from Markdown, call `write` with a bare name such as "
      "`draft.md`, then run `python skills/docx/scripts/generate_from_md.py "
      f"/workspace/output/draft.md /workspace/output/{docx_names[0]}` and validate it "
      f"with `python skills/docx/scripts/validate.py /workspace/output/{docx_names[0]}`."
    )
  return (
    "\n\n## Required output-path contract\n\n"
    f"The required deliverables are {output_paths}. "
    "For the `write` tool, use a bare relative `file_path` without an `output/` prefix; "
    "the tool already stores it under `/workspace/output`. For `bash` and skill scripts, "
    "use absolute `/workspace/output/...` paths. Create every command input before using "
    "it, and confirm and validate all requested deliverables before stopping." + docx_example
  )


def lab_renderer(model_name: str, renderer_name: str | None) -> Renderer:
  register_gemma4_tool_renderer()
  tokenizer = tokenizer_utils.get_tokenizer(model_name)
  resolved_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
  renderer = get_renderer(resolved_name, tokenizer, model_name=model_name)
  if resolved_name.startswith("qwen3") and hasattr(renderer, "strip_thinking_from_history"):
    # Multi-turn RL needs each observation to extend the preceding one. The
    # Qwen default removes earlier thinking blocks when history is re-rendered,
    # which breaks sampler prefix affinity and makes trajectory_to_data emit a
    # separate cumulative datum for nearly every turn.
    renderer.strip_thinking_from_history = False
  if not renderer.has_extension_property:
    raise ValueError(f"Harvey LAB multi-turn RL requires a prefix-extending renderer; {resolved_name!r} does not preserve history")
  return renderer
