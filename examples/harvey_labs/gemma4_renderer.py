"""Gemma 4 renderer backed by the checkpoint's native chat template."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.renderers.base import (
  Message,
  ParseTermination,
  RenderContext,
  RenderedMessage,
  TextPart,
  ThinkingPart,
  ToolCall,
  ToolSpec,
)

# chat_template.jinja is pinned from Gemma 4 discussion #36 at HF revision
# 4e34fcbc4c9a95b92d6a8a97c2faed16dd783f91.
GEMMA4_CHAT_TEMPLATE_SHA256 = "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"

_TOOL_CALL = re.compile(
  r"<\|tool_call>(?P<body>call:(?P<name>\w+)\{.*?\})"
  r"(?:<tool_call\|>|<eos>|(?=<\|tool_response>)|$)",
  re.DOTALL,
)


def _valid_tool_calls(parsed: Any) -> list[dict[str, Any]]:
  calls = parsed.get("tool_calls") if isinstance(parsed, Mapping) else None
  if not isinstance(calls, list):
    return []
  return [
    call
    for call in calls
    if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping) and isinstance(call["function"].get("name"), str)
  ]


@cache
def gemma4_chat_template() -> str:
  """Return the pinned canonical Gemma 4 tool-calling template."""
  template = Path(__file__).with_name("chat_template.jinja").read_text(encoding="utf-8")
  digest = sha256(template.encode()).hexdigest()
  if digest != GEMMA4_CHAT_TEMPLATE_SHA256:
    raise ValueError(f"Gemma 4 chat template hash mismatch: {digest}")
  return template


class Gemma4ToolRenderer(renderers.Renderer):
  """Render and parse Gemma 4's native function-calling protocol."""

  def __init__(self, tokenizer: Any, *, enable_thinking: bool = True):
    super().__init__(tokenizer)
    self.chat_template = gemma4_chat_template()
    self.enable_thinking = enable_thinking
    self.tool_names: set[str] = set()

  @property
  def has_extension_property(self) -> bool:
    return True

  @property
  def bos_tokens(self) -> list[int]:
    return self.tokenizer.encode("<bos>", add_special_tokens=False)

  @property
  def end_message_tokens(self) -> set[int]:
    return {
      self.tokenizer.encode("<turn|>", add_special_tokens=False)[0],
      self.tokenizer.encode("<|tool_response>", add_special_tokens=False)[0],
    }

  def get_stop_sequences(self) -> list[int]:
    return sorted(self.end_message_tokens)

  def template_tokens(
    self,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
  ) -> list[int]:
    rendered = self.tokenizer.apply_chat_template(
      messages,
      chat_template=self.chat_template,
      tokenize=True,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
    )
    if isinstance(rendered, Mapping):
      rendered = rendered["input_ids"]
    return list(rendered)

  def build_generation_prompt(
    self,
    messages: list[Message],
    role: str = "assistant",
    prefill: str | None = None,
  ) -> tinker.ModelInput:
    if role != "assistant":
      raise ValueError(f"Gemma 4 only supports assistant generation, got {role!r}")
    prompt = self.template_tokens(
      [self.to_openai_message(message) for message in messages],
      add_generation_prompt=True,
    )
    if prefill:
      prompt.extend(self.tokenizer.encode(prefill, add_special_tokens=False))
    return tinker.ModelInput.from_ints(tokens=prompt)

  def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    """Render one non-tool message for cookbook masking utilities.

    Live LAB rollouts use ``build_generation_prompt`` so the full conversation is
    always formatted by the pinned canonical Hugging Face template.
    """
    if message["role"] == "tool":
      raise NotImplementedError("Gemma 4 tool responses require their preceding tool call")

    rendered = self.template_tokens(
      [self.to_openai_message(message)],
      add_generation_prompt=False,
      enable_thinking=False,
    )
    bos_len = len(self.bos_tokens)
    rendered = rendered[bos_len:]
    role = "model" if message["role"] in ("assistant", "model") else message["role"]
    header = self.tokenizer.encode(f"<|turn>{role}\n", add_special_tokens=False)
    if rendered[: len(header)] != header:
      raise ValueError(f"Unexpected Gemma 4 template output for role {message['role']!r}")
    return RenderedMessage(
      header=tinker.EncodedTextChunk(tokens=header),
      output=[tinker.EncodedTextChunk(tokens=rendered[len(header) :])],
    )

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    eos_token_id = self.tokenizer.eos_token_id
    ended_with_eos = eos_token_id is not None and bool(response) and response[-1] == eos_token_id
    if ended_with_eos:
      response = response[:-1]
    decoded = self.tokenizer.decode(response, skip_special_tokens=False)
    parse_input = decoded
    if self.enable_thinking and "<channel|>" in decoded and not decoded.startswith("<|channel>"):
      # After a tool response the canonical template opens the thought channel
      # in the prompt. Sampling returns only its continuation, so reconstruct
      # that opener before handing the response to Gemma's native parser.
      parse_input = "<|channel>thought\n" + decoded
    try:
      parsed = self.tokenizer.parse_response(parse_input)
    except Exception:
      parsed = {"role": "assistant"}

    tool_calls = _valid_tool_calls(parsed)
    if "<|tool_call>" in parse_input and not tool_calls:
      # Gemma sometimes closes thought, emits final text, then opens a tool
      # call. That channel order is outside the response schema even when the
      # call block itself is complete and valid. Parse complete call blocks in
      # isolation so useful actions are not discarded with their prose.
      tool_call_input = "".join(
        f"<|tool_call>{match.group('body')}<tool_call|>" for match in _TOOL_CALL.finditer(parse_input) if match.group("name") in self.tool_names
      )
      try:
        tool_call_parse = self.tokenizer.parse_response(tool_call_input)
      except Exception:
        tool_call_parse = {}
      tool_calls = _valid_tool_calls(tool_call_parse)

    parsed_tool_calls = []
    for tool_call in tool_calls:
      function = tool_call["function"]
      parsed_tool_calls.append(
        ToolCall(
          id=tool_call.get("id"),
          function=ToolCall.FunctionBody(
            name=function["name"],
            arguments=json.dumps(function.get("arguments", {}), separators=(",", ":")),
          ),
        )
      )

    parts: list[TextPart | ThinkingPart] = []
    if thinking := parsed.get("thinking"):
      parts.append(ThinkingPart(type="thinking", thinking=thinking))
    if content := parsed.get("content"):
      parts.append(TextPart(type="text", text=content))

    message = Message(role="assistant", content=parts if parts else "")
    if parsed_tool_calls:
      message["tool_calls"] = parsed_tool_calls

    malformed_tool_call = "<|tool_call>" in decoded and not parsed_tool_calls
    if malformed_tool_call:
      return message, ParseTermination.MALFORMED
    termination = ParseTermination.EOS if ended_with_eos else ParseTermination.STOP_SEQUENCE
    return message, termination

  def create_conversation_prefix_with_tools(
    self,
    tools: list[ToolSpec],
    system_prompt: str = "",
  ) -> list[Message]:
    self.tool_names = {str(tool["name"]) for tool in tools}
    native_tools = [{"type": "function", "function": tool} for tool in tools]
    rendered = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": system_prompt}],
      tools=native_tools,
      chat_template=self.chat_template,
      tokenize=False,
      add_generation_prompt=False,
      # Tools are embedded into the returned system message. The final full
      # conversation render injects the single thinking marker.
      enable_thinking=False,
    )
    prefix = "<bos><|turn>system\n"
    suffix = "<turn|>\n"
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
      raise ValueError("Unexpected Gemma 4 system/tool template")
    return [Message(role="system", content=rendered[len(prefix) : -len(suffix)])]

  def to_openai_message(self, message: Message) -> dict[str, Any]:
    result = super().to_openai_message(message)
    if result["role"] == "model":
      result["role"] = "assistant"
    for tool_call in result.get("tool_calls", []):
      arguments = tool_call["function"].get("arguments")
      if isinstance(arguments, str):
        tool_call["function"]["arguments"] = json.loads(arguments)
    return result


def register_gemma4_tool_renderer(name: str = "gemma4") -> None:
  renderers.register_renderer(name, lambda tokenizer, img_proc=None: Gemma4ToolRenderer(tokenizer))
