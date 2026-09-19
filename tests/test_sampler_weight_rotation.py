import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from server import training_requests_processor as trp
from server.store import InMemoryStateStore, InMemoryStore
from tests.test_fft_batch_failure import SlicerStub
from training.commands import SaveWeightsForSampler


class RecordingWorker:
  def __init__(self):
    self.saves = []

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"):
    os.makedirs(state_path, exist_ok=True)
    os.utime(state_path, (len(self.saves), len(self.saves)))
    self.saves.append(os.path.basename(state_path))
    return {"path": state_path}


class SamplerWeightRotationTest(unittest.TestCase):
  def test_only_the_newest_versions_stay_on_the_volume(self) -> None:
    with (
      tempfile.TemporaryDirectory() as tmp,
      patch.dict(os.environ, {"REDIS_URL": "redis://test", "OPEN_RL_TMP_DIR": tmp}),
      patch.object(trp, "SAMPLER_VERSIONS_KEPT", 3),
    ):
      worker = RecordingWorker()
      proc = trp.FFTTrainingRequestsProcessor(InMemoryStore(), InMemoryStateStore(), worker, "run-a", SlicerStub())
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 6):
        asyncio.run(
          proc.save_weights_for_sampler(
            SaveWeightsForSampler(request_id=str(step), model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}"), "run-a"
          )
        )
        time.sleep(0.01)
      self.assertEqual(worker.saves, [f"sampler-{s}" for s in range(1, 6)])
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4", "sampler-5"])


if __name__ == "__main__":
  unittest.main()
