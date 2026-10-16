import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from server import training_requests_processor as trp
from server.model_metadata import TrainingModelMetadata, persist_model_metadata
from server.sampling_store import SamplingStore
from server.store import InMemoryStateStore, InMemoryStore, RedisStore
from tests.redis_server import RedisTestCase
from tests.test_fft_batch_failure import SlicerStub
from training import commands


class RecordingWorker(trp.FFTTrainingWorker):
  def __init__(self):
    super().__init__()
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
      proc = trp.TrainingRequestsProcessor(InMemoryStore(), worker, "run-a", time_slicer=SlicerStub())
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 6):
        command = commands.SaveWeightsForSampler(request_id=f"r{step}", model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}")
        asyncio.run(proc.dispatch_operation(command))
        time.sleep(0.01)
      self.assertEqual(worker.saves, [f"sampler-{s}" for s in range(1, 6)])
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4", "sampler-5"])


class PinnedVersionTest(RedisTestCase):
  async def test_a_version_a_pending_request_names_outlives_retention(self) -> None:
    tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": self.url, "OPEN_RL_TMP_DIR": tmp}))
    self.enterContext(patch.object(trp, "SAMPLER_VERSIONS_KEPT", 3))
    state, meta = InMemoryStateStore(), TrainingModelMetadata(base_model="base", fine_tuning_type="full")
    await persist_model_metadata(state, "run-a", meta)
    self.enterContext(patch.object(trp, "get_state_store", return_value=state))
    versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
    pinned = {"request_id": "waiting", "pool": meta.pool("run-a"), "revision": os.path.join(versions, "sampler-1"), "idempotency_key": None}
    await SamplingStore(self.redis).accept(pinned)
    proc = trp.TrainingRequestsProcessor(RedisStore(self.redis), RecordingWorker(), "run-a", time_slicer=SlicerStub())
    for step in range(1, 6):
      await proc.dispatch_operation(
        commands.SaveWeightsForSampler(request_id=f"r{step}", model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}")
      )
      await asyncio.sleep(0.01)
    self.assertEqual(sorted(os.listdir(versions)), ["sampler-1", "sampler-3", "sampler-4", "sampler-5"])


if __name__ == "__main__":
  unittest.main()
