import unittest

from server.dashboard import snapshot

STATE = {
  "available": True,
  "pods": [
    {
      "name": "orw-lora-trainer",
      "uid": "p1",
      "phase": "Running",
      "node": "n1",
      "worker": "lora-qwen-0-trainer",
      "owner_uids": ["w1"],
      "restarts": 0,
      "created_at": "t",
      "problem": None,
      "containers": [],
      "events": [],
    },
    {
      "name": "orw-fft-trainer",
      "uid": "p2",
      "phase": "Pending",
      "node": None,
      "worker": "fft-r2-trainer",
      "owner_uids": [],
      "restarts": 0,
      "created_at": "t",
      "problem": "Pending",
      "containers": [],
      "events": [],
    },
  ],
  "scheduler": {
    "workloads": [
      {
        "uid": "w1",
        "name": "lora-qwen-0-trainer",
        "model_id": "Qwen/Qwen3-8B",
        "training_kind": "lora",
        "role": "trainer",
        "node_name": "n1",
        "claim_name": "c1",
        "pod_name": "orw-lora-trainer",
        "device_count": 1,
        "phase": "Running",
      },
      {
        "uid": "w2",
        "name": "fft-r2-trainer",
        "model_id": "r2",
        "training_kind": "fft",
        "role": "trainer",
        "node_name": None,
        "claim_name": None,
        "pod_name": "orw-fft-trainer",
        "device_count": 0,
        "phase": "Pending",
      },
    ]
  },
  "devices": {"claims": {"c1": ["gpu.nvidia.com/n1/gpu-0"]}},
}
METADATA = [
  {"model_id": "r1", "base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 2, "total_steps_completed": 3},
  {"model_id": "r3", "base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 1},
  {"model_id": "r2", "base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", "status": "active", "created_at": 3},
]


class SnapshotJoinTest(unittest.TestCase):
  def test_lora_runs_share_the_runtime_worker_and_fft_runs_own_theirs(self) -> None:
    runs = {r["run_id"]: r for r in snapshot.join_runs(METADATA, STATE)}
    self.assertEqual(runs["r1"]["display_status"], "Running")
    self.assertEqual([p["name"] for p in runs["r1"]["pods"]], ["orw-lora-trainer"])
    self.assertEqual(runs["r1"]["pods"][0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])
    self.assertEqual(sorted(runs["r1"]["runtime_run_ids"]), ["r1", "r3"])
    self.assertEqual(runs["r2"]["display_status"], "Needs attention")
    self.assertEqual([w["uid"] for w in runs["r2"]["workloads"]], ["w2"])

  def test_only_placed_workloads_become_placements_and_carry_their_runs(self) -> None:
    runs = snapshot.join_runs(METADATA, STATE)
    placements = snapshot.placements_of(STATE, runs)
    self.assertEqual(len(placements), 1)
    self.assertEqual(placements[0]["id"], "w1")
    self.assertEqual(sorted(placements[0]["run_ids"]), ["r1", "r3"])
    self.assertEqual(placements[0]["label"], "Qwen/Qwen3-8B")
    self.assertEqual(placements[0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])
