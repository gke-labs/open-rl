"""Unit tests for the Gemma 4 renderer and format grading in train_gemma.py.

Run from examples/: uv run python -m unittest recipes.math_rl.test_train_gemma
"""

import asyncio
import unittest

from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.renderers.base import ParseTermination

from recipes.math_rl import train_gemma

SPECIAL_TOKENS = {"<bos>": 1, "<eos>": 2, "<|turn>": 3, "<turn|>": 4}


class StubTokenizer:
  """Special tokens get fixed ids; every other character is its own token."""

  eos_token_id = SPECIAL_TOKENS["<eos>"]

  def __init__(self):
    self._id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    ids, i = [], 0
    while i < len(text):
      special = next((s for s in SPECIAL_TOKENS if text.startswith(s, i)), None)
      if special:
        ids.append(SPECIAL_TOKENS[special])
        i += len(special)
      else:
        ids.append(1000 + ord(text[i]))
        i += 1
    return ids

  def decode(self, ids: list[int]) -> str:
    return "".join(self._id_to_special.get(t) or chr(t - 1000) for t in ids)


class Gemma4ParseResponseTest(unittest.TestCase):
  def setUp(self):
    self.tokenizer = StubTokenizer()
    self.renderer = train_gemma.Gemma4Renderer(self.tokenizer)

  def test_eos_finish_is_clean_and_stripped(self):
    message, termination = self.renderer.parse_response(self.tokenizer.encode("The answer is 4.<eos>"))
    self.assertEqual(termination, ParseTermination.EOS)
    self.assertEqual(message["content"], "The answer is 4.")

  def test_end_of_turn_finish_unchanged(self):
    message, termination = self.renderer.parse_response(self.tokenizer.encode("The answer is 4.<turn|>"))
    self.assertEqual(termination, ParseTermination.STOP_SEQUENCE)
    self.assertEqual(message["content"], "The answer is 4.")

  def test_no_end_token_is_malformed(self):
    message, termination = self.renderer.parse_response(self.tokenizer.encode("The answer is"))
    self.assertEqual(termination, ParseTermination.MALFORMED)
    self.assertEqual(message["content"], "The answer is")


class LenientFormatMathEnvTest(unittest.TestCase):
  def _step(self, reply: str):
    tokenizer = StubTokenizer()
    env = math_env.MathEnv("What is 2 + 2?", "4", train_gemma.Gemma4Renderer(tokenizer))
    return asyncio.run(env.step(tokenizer.encode(reply)))

  def test_datasets_build_the_lenient_env(self):
    self.assertIs(math_env.MathEnv, train_gemma.LenientFormatMathEnv)

  def test_eos_finish_gets_format_credit(self):
    result = self._step("\\boxed{4}<eos>")
    self.assertEqual(result.metrics["format"], 1.0)
    self.assertEqual(result.reward, 1.0)

  def test_truncated_reply_gets_no_format_credit(self):
    result = self._step("\\boxed{4}")
    self.assertEqual(result.metrics["format"], 0.0)


if __name__ == "__main__":
  unittest.main()
