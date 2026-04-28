"""End-to-end: pig-latin SFT against Qwen3-0.6B.

Non-gated model kept small for CI speed. Skips the baseline eval and runs
5 steps — enough to see loss drop hard and exact-match hit ~15-20%.
"""

import unittest

from piglatin_sft import PRESETS, run_training

from tests._server_fixture import REPO_ROOT, OpenRlServerCase

CLIENT_DIR = REPO_ROOT / "examples" / "sft" / "pig-latin"

PIGLATIN_EVAL_EXAMPLES = [
  ("banana", "anana-bay"),
  ("quantum", "uantum-qay"),
  ("donut", "onut-day"),
  ("pickle", "ickle-pay"),
  ("space", "ace-spay"),
  ("rubber", "ubber-ray"),
  ("coding", "oding-cay"),
  ("hello", "ello-hay"),
  ("machine", "achine-may"),
  ("artificial", "rtificial-aay"),
  ("data", "ata-day"),
]


class TestPigLatinQwen(OpenRlServerCase):
  BASE_MODEL = "Qwen/Qwen3-0.6B"
  PORT = 9010

  def test_sft_improves(self) -> None:
    CI_CONFIG = {
      "base_url": self.BASE_URL,
      "steps": 5,
      "assert_improvement": False,
      "skip_before_eval": True,
      "plot_path": str(CLIENT_DIR / "artifacts" / "test_qwen_metrics.png"),
      "custom_examples": PIGLATIN_EVAL_EXAMPLES,
    }

    config = PRESETS["qwen"].clone().apply(CI_CONFIG, layer_name="ci_override").make()

    metrics = run_training(config)
    print(f"Training metrics: {metrics}")

    self.assertGreater(metrics["loss_drop"], 0.0, f"Loss did not drop: {metrics['loss_drop']:.3f}")
    self.assertGreater(metrics["after_exact"], 0.0, f"After training, exact match still 0: {metrics['after_exact']:.3f}")


if __name__ == "__main__":
  unittest.main()
