"""End-to-end: pig-latin SFT against a local Open-RL server running Gemma 3 1B.

Requires HF_TOKEN (Gemma 3 is gated). Skipped automatically otherwise.
Asserts real improvement: exact-match goes up, similarity gain meets the
preset's threshold, and the final exact-match clears a 20% floor.
"""

import unittest

from piglatin_sft import PRESETS, run_training

from tests._server_fixture import CLIENT_DIR, PIGLATIN_EVAL_EXAMPLES, OpenRlServerCase


class TestPigLatinGemma(OpenRlServerCase):
  BASE_MODEL = "google/gemma-3-1b-it"
  PORT = 9011
  REQUIRE_HF_TOKEN = True

  def test_sft_improves(self) -> None:
    config = (
      PRESETS["gemma"]
      .clone()
      .apply(
        {
          "base_url": self.BASE_URL,
          "steps": 25,
          "assert_improvement": False,
          "plot_path": str(CLIENT_DIR / "artifacts" / "test_gemma_metrics.png"),
          "custom_examples": PIGLATIN_EVAL_EXAMPLES,
        },
        layer_name="test_override",
      )
      .make()
    )

    m = run_training(config)
    print(f"Training metrics: {m}")

    self.assertGreater(m["after_exact"], m["before_exact"], f"Exact match didn't improve: {m['before_exact']:.2f} -> {m['after_exact']:.2f}")
    sim_gain = m["after_sim"] - m["before_sim"]
    self.assertGreaterEqual(sim_gain, config.min_similarity_gain, f"Similarity gain insufficient: {m['before_sim']:.2f} -> {m['after_sim']:.2f}")
    self.assertGreaterEqual(m["after_exact"], 0.20, f"Expected >= 20% exact match, got {m['after_exact']:.0%}")


if __name__ == "__main__":
  unittest.main()
