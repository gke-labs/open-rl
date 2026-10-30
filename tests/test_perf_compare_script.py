import argparse
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "perf_compare.py"
spec = importlib.util.spec_from_file_location("perf_compare", SCRIPT)
perf_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_compare)

LOG = """\
2026-10-27T00:40:19.676897111Z                                  Step 0
2026-10-27T00:40:19.676964113Z │ env/all/reward/total │ -0.082812  │
2026-10-27T00:40:19.677078284Z │ time/save_checkpoint │ 15.667198  │
2026-10-27T00:40:19.677075651Z │ time/sampling        │ 51.990044  │
2026-10-27T00:40:19.677083575Z │ time/total           │ 117.589038 │
2026-10-27T00:40:19.677117539Z │ time/train_step      │ 49.804913  │
2026-10-27T00:40:19.677126533Z └──────────────────────┴────────────┘
2026-10-27T00:41:19.500000000Z │ time/sampling        │ 20.0       │
2026-10-27T00:41:19.600000000Z │ time/total           │ 60.0       │
2026-10-27T00:41:19.700000000Z └──────────────────────┴────────────┘
"""


def record(version, run, ends, peak):
  """A run of two shared Qwen LoRA jobs, each finishing step 1 at 100 s and step 2 at `ends`."""
  jobs = {
    f"qwen-lora-{i}": {
      "group": "qwen-lora",
      "model": "Qwen/Qwen3-0.6B-Base",
      "shared_lora": True,
      "ok": True,
      "done_s": end,
      "steps": [{"t": 100.0, "time/total": 100.0}, {"t": end, "time/total": end - 100, "time/sampling": 5.0}],
      "model_ids": [],
    }
    for i, end in enumerate(ends, start=1)
  }
  workers = {
    "orw-lora-x-0-trainer": {
      "workload": "lora-x-0-trainer",
      "role": "trainer",
      "model_id": "Qwen/Qwen3-0.6B-Base",
      "peak_mib": peak,
      "gpu": "NVIDIA H100",
    }
  }
  return {"run": run, "version": version, "images": {"api-server": "a", "worker": "w"}, "steps": 2, "valid": True, "jobs": jobs, "workers": workers}


class PerfCompareScriptTest(unittest.TestCase):
  def test_parse_version_requires_api_server_and_worker(self) -> None:
    self.assertEqual(perf_compare.parse_version("api-server=a:1,worker=w:1"), {"api-server": "a:1", "worker": "w:1"})
    with self.assertRaises(argparse.ArgumentTypeError):
      perf_compare.parse_version("worker=w:1")
    with self.assertRaises(argparse.ArgumentTypeError):
      perf_compare.parse_version("api-server=a:1,worker=w:1,client=c:1")

  def test_default_workload_is_23_jobs_on_two_shared_lora_runtimes(self) -> None:
    jobs = perf_compare.load_workload(perf_compare.WORKLOAD)
    self.assertEqual(len(jobs), 23)
    self.assertEqual(len({j.name for j in jobs}), 23)
    self.assertEqual({j.model for j in jobs if j.shared_lora}, {"google/gemma-4-E2B", "Qwen/Qwen3-0.6B-Base"})
    self.assertFalse(next(j for j in jobs if j.group == "gemma-fft-x").shared_lora)

  def test_parse_client_log_reads_each_step_table(self) -> None:
    steps = perf_compare.parse_client_log(LOG)
    self.assertEqual(len(steps), 2)
    self.assertAlmostEqual(steps[0]["time/total"], 117.589038)
    self.assertAlmostEqual(steps[0]["time/save_checkpoint"], 15.667198)
    self.assertAlmostEqual(steps[1]["t"] - steps[0]["t"], 59.922883, places=4)

  def test_run_order_alternates(self) -> None:
    self.assertEqual(perf_compare.run_order(2), [("b1", "before"), ("a1", "after"), ("a2", "after"), ("b2", "before")])

  def test_report_compares_totals_steps_and_memory(self) -> None:
    records = [record("before", "b1", [300, 400], 20000), record("after", "a1", [200, 250], 30000)]
    report = perf_compare.build_report(records, {"qwen-lora": "Qwen LoRA (2 jobs)"})
    self.assertIn("Launch until the last job finishes step 2: **400 → 250 (-38%)** seconds.", report)
    self.assertIn("| Qwen LoRA (2 jobs) | 100 → 100 (+0%) | 350 → 225 (-36%) |", report)
    self.assertIn("| Qwen LoRA (2 jobs) | 250 → 125 (-50%) | 5.0 → 5.0 | - | - |", report)
    self.assertIn("| Qwen LoRA (2 jobs) | H100 | 20 → 29 (+50%) | - |", report)

  def test_describe_groups_names_model_training_and_sharing(self) -> None:
    labels = perf_compare.describe_groups(perf_compare.load_workload(perf_compare.WORKLOAD))
    self.assertEqual(labels["gemma-fft-x"], "gemma-4-E2B FFT, exclusive (1 job)")
    self.assertEqual(labels["qwen-fft-ts"], "Qwen3-0.6B-Base FFT, time-sliced (2 jobs)")
    self.assertEqual(labels["gemma-lora"], "gemma-4-E2B LoRA, shared runtime (6 jobs)")

  def test_job_manifest_names_run_and_volume(self) -> None:
    job = perf_compare.Job("qwen-lora-1", "qwen-lora", "lora-gsm8k-rl", "Qwen/Qwen3-0.6B-Base", "openrl.recipe=rl")
    manifest = perf_compare.job_manifest(job, "b1", 3, "open-rl-shared-zonal", "openrl-system")
    pod = manifest["spec"]["template"]["spec"]
    self.assertEqual(manifest["metadata"]["name"], "perf-b1-qwen-lora-1")
    self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "open-rl-shared-zonal")
    self.assertIn("steps=3", pod["containers"][0]["args"])
    self.assertIn({"name": "TINKER_TAGS", "value": "openrl.recipe=rl"}, pod["containers"][0]["env"])


if __name__ == "__main__":
  unittest.main()
