"""LAB tool adapters for tinker-cookbook tool environments."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult


def bounded_tool_result(
  result: str,
  *,
  tool_name: str,
  arguments: str | dict[str, Any],
  tokenizer: Any,
  max_tokens: int,
) -> str:
  """Keep one oversized observation from exhausting the whole trajectory."""
  tokens = tokenizer.encode(result, add_special_tokens=False)
  if len(tokens) <= max_tokens:
    return result

  decoded = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False)
  last_newline = decoded.rfind("\n")
  line_aligned = last_newline >= 0
  if line_aligned:
    decoded = decoded[: last_newline + 1]

  detail = "Run a narrower command or search to retrieve the remaining content."
  if tool_name == "read" and line_aligned:
    if isinstance(arguments, str):
      try:
        arguments = json.loads(arguments)
      except json.JSONDecodeError:
        arguments = {}
    offset = int(arguments.get("offset") or 0)
    next_offset = offset + decoded.count("\n")
    detail = f"Continue the same `read` with `offset={next_offset}` and a finite `limit`, or use `grep` to retrieve only relevant passages."

  return decoded.rstrip() + f"\n\n[Observation truncated at {max_tokens:,} tokens. {detail}]"


@dataclass(frozen=True)
class LabTool:
  spec: dict[str, Any]
  executor: Any
  tokenizer: Any
  max_result_tokens: int

  @property
  def name(self) -> str:
    return str(self.spec["name"])

  @property
  def description(self) -> str:
    return str(self.spec.get("description", ""))

  @property
  def parameters_schema(self) -> dict[str, Any]:
    return dict(self.spec.get("parameters", {"type": "object", "properties": {}}))

  def to_spec(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": self.parameters_schema,
    }

  async def run(self, input: ToolInput) -> ToolResult:
    # The executor shells into podman synchronously (up to command_timeout);
    # off the event loop so one env's slow tool call can't stall the group.
    result = await asyncio.to_thread(self._execute_bounded, input)
    return simple_tool_result(result, call_id=input.call_id or "", name=self.name)

  def _execute_bounded(self, input: ToolInput) -> str:
    result = self.executor.execute(self.name, input.arguments)
    return bounded_tool_result(
      result,
      tool_name=self.name,
      arguments=input.arguments,
      tokenizer=self.tokenizer,
      max_tokens=self.max_result_tokens,
    )
