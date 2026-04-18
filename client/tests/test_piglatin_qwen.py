"""End-to-end: pig-latin SFT against Qwen3-0.6B.

Non-gated model kept small for CI speed. Skips the baseline eval and runs
5 steps — enough to see loss drop hard and exact-match hit ~15-20%.
"""

import unittest

from piglatin_sft import PRESETS, run_training

from tests._server_fixture import CLIENT_DIR, PIGLATIN_EVAL_EXAMPLES, OpenRlServerCase


class TestPigLatinQwen(OpenRlServerCase):
  BASE_MODEL = "Qwen/Qwen3-0.6B"
  PORT = 9010

  def test_sft_improves(self) -> None:
    config = (
      PRESETS["qwen"]
      .clone()
      .apply(
        {
          "base_url": self.BASE_URL,
          "steps": 5,
          "assert_improvement": False,
          "skip_before_eval": True,
          "plot_path": str(CLIENT_DIR / "artifacts" / "test_qwen_metrics.png"),
          "custom_examples": PIGLATIN_EVAL_EXAMPLES,
        },
        layer_name="ci_override",
      )
      .make()
    )

    m = run_training(config)
    print(f"Training metrics: {m}")

    self.assertGreater(m["loss_drop"], 0.0, f"Loss did not drop: {m['loss_drop']:.3f}")
    self.assertGreater(m["after_exact"], 0.0, f"After training, exact match still 0: {m['after_exact']:.3f}")


if __name__ == "__main__":
  unittest.main()
