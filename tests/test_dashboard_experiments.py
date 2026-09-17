import json
import tempfile
import unittest
from pathlib import Path

from server.dashboard import experiments


class ExperimentsScanTest(unittest.TestCase):
  def test_scan_reads_config_and_charted_series_per_run(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir) / "runs"
      run = root / "sweep-1" / "open-rl-tmp" / "gsm8k_rl_mega_small-r1-a"
      run.mkdir(parents=True)
      (run / "iteration_000000").mkdir()
      with open(run / "config.json", "w") as f:
        json.dump({"model_name": "Qwen/Qwen3-0.6B", "lora_rank": 1, "max_steps": 40, "base_url": "secret"}, f)
      with open(run / "metrics.jsonl", "w") as f:
        for step, reward in enumerate((0.1, 0.4, 0.7)):
          f.write(json.dumps({"step": step, "env/all/reward/total": reward, "env/all/correct": reward, "time/total": 3.0}) + "\n")
        f.write("not json\n")

      found = experiments.scan(root)

    self.assertEqual(len(found), 1)
    entry = found[0]
    self.assertEqual(entry["sweep"], "sweep-1")
    self.assertEqual(entry["name"], "gsm8k_rl_mega_small-r1-a")
    self.assertEqual(entry["config"], {"model_name": "Qwen/Qwen3-0.6B", "lora_rank": 1, "max_steps": 40})
    self.assertEqual(entry["step"], 2)
    self.assertEqual(entry["series"]["reward"], [[0, 0.1], [1, 0.4], [2, 0.7]])
    self.assertEqual(entry["last"]["correct"], 0.7)
    self.assertNotIn("lr", entry["series"])

  def test_missing_root_is_empty_not_an_error(self) -> None:
    self.assertEqual(experiments.scan(Path(tempfile.gettempdir()) / "does-not-exist"), [])
