"""Wrapper around tinker_cookbook.recipes.math_rl.train that registers Gemma renderers (gemma4, gemma2)."""

import asyncio

import chz
import tinker
import tinker_cookbook.renderers as renderers
from common.tinker_utils import patch_tinker_default_headers
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main
from tinker_cookbook.renderers.base import Message, ParseTermination, RenderContext, RenderedMessage, parse_response_for_stop_token


class Gemma4Renderer(renderers.Renderer):
  _turn_start = "<|turn>"
  _turn_end = "<turn|>"

  @property
  def _bos_tokens(self) -> list[int]:
    return self.tokenizer.encode("<bos>", add_special_tokens=False)

  @property
  def _end_message_token(self) -> int:
    return self.tokenizer.encode(self._turn_end, add_special_tokens=False)[0]

  def get_stop_sequences(self) -> list[int]:
    return [self._end_message_token]

  def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    maybe_newline = "\n" if ctx.idx > 0 else ""
    role_map = {"assistant": "model", "model": "model", "user": "user", "system": "user"}
    role = role_map.get(message["role"], "user")
    header_str = f"{maybe_newline}{self._turn_start}{role}\n"
    content = message["content"]
    output_content = str(content) + self._turn_end if content != "" else ""
    header = tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(header_str, add_special_tokens=False))
    output = [tinker.types.EncodedTextChunk(tokens=self.tokenizer.encode(output_content, add_special_tokens=False))] if output_content else []
    return RenderedMessage(header=header, output=output)

  def parse_response(self, response: list[int]):
    # A reply that ends on <eos> terminated cleanly, as in the harvey_labs renderer.
    if response and response[-1] == self.tokenizer.eos_token_id:
      return Message(role="assistant", content=self.tokenizer.decode(response[:-1])), ParseTermination.EOS
    return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


class Gemma2Renderer(Gemma4Renderer):
  # gemma-2 chat markup; <end_of_turn> is a single special token in the
  # gemma-2 vocab, so _end_message_token is its exact id.
  _turn_start = "<start_of_turn>"
  _turn_end = "<end_of_turn>"


class LenientFormatMathEnv(math_env.MathEnv):
  # Grade format on any clean termination (STOP_SEQUENCE or EOS), as
  # EnvFromMessageEnv does; the math datasets build MathEnv with the strict default.
  def __init__(self, *args, **kwargs):
    kwargs.setdefault("require_stop_sequence_for_format", False)
    super().__init__(*args, **kwargs)


math_env.MathEnv = LenientFormatMathEnv

# Register into SDK global lookup dictionary at script startup
renderers.register_renderer("gemma4", lambda tokenizer, img_proc=None: Gemma4Renderer(tokenizer))
renderers.register_renderer("gemma2", lambda tokenizer, img_proc=None: Gemma2Renderer(tokenizer))

if __name__ == "__main__":
  patch_tinker_default_headers()
  cli_config = chz.entrypoint(CLIConfig)
  asyncio.run(cli_main(cli_config))
