import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from server import gateway
from server.store import InMemoryStore


class GetInfoTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-a"}))

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_prefers_the_models_own_base_model(self) -> None:
    meta = json.dumps({"base_model": "google/gemma-4-e2b", "fine_tuning_type": "full"})
    asyncio.run(gateway.store.set_value("open_rl:model_meta:model-g", meta))
    with patch.dict(os.environ, {"BASE_MODEL": "Qwen/Qwen2.5-0.5B"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-g"}))
      via_sampler_ref = asyncio.run(gateway.get_info({"model_id": "tinker://model-g/sampler_weights/sampler-1"}))
      unknown = asyncio.run(gateway.get_info({"model_id": "model-unknown"}))

    # The client loads its tokenizer from this name, so it must be the job's model.
    self.assertEqual(info["model_name"], "google/gemma-4-e2b")
    self.assertEqual(info["model_data"]["tokenizer_id"], "google/gemma-4-e2b")
    self.assertEqual(via_sampler_ref["model_name"], "google/gemma-4-e2b")
    self.assertEqual(unknown["model_name"], "Qwen/Qwen2.5-0.5B")

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_info({"model_id": "model-a"}))
    self.assertEqual(response.status_code, 404)

  def test_create_model_requires_base_model_payload(self) -> None:
    response = asyncio.run(gateway.create_model({}))
    self.assertEqual(response.status_code, 400)

  def test_create_model_accepts_base_model_payload(self) -> None:
    created = asyncio.run(gateway.create_model({"base_model": "my-model"}))
    model_id = created["request_id"]
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["model_id"], model_id)
    self.assertEqual(queued[0]["op"], "create_model")
    self.assertEqual(queued[0]["base_model"], "my-model")
    meta = json.loads(gateway.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "my-model")


class SaveSeqIdZeroTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_the_first_saves_zero_seq_id_is_kept(self) -> None:
    # The client's counter is 0-based; 0 must not fall back to a timestamp id.
    asyncio.run(gateway.save_weights_for_sampler({"model_id": "job-a", "sampling_session_seq_id": 0}))
    asyncio.run(gateway.save_weights({"model_id": "job-a", "seq_id": 0}))
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["sampling_session_id"], "tinker://job-a/sampler_weights/sampler-0")
    self.assertTrue(queued[1]["state_path"].endswith("job-a-samp-0"))


class GatewayPathTest(unittest.TestCase):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    old_tmp_dir = gateway.TMP_DIR
    with tempfile.TemporaryDirectory() as tmp_dir:
      gateway.TMP_DIR = tmp_dir
      self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)

      self.assertEqual(
        gateway.checkpoint_state_path("job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        gateway.checkpoint_state_path("job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_a_tinker_path_names_the_model_that_saved_it(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      self.assertEqual(gateway.tinker_state_path(state_dir), "tinker://job-a/weights/step-5")
      # A resuming job passes the dead job's path under its own model id.
      self.assertEqual(gateway.checkpoint_state_path("job-b", "tinker://job-a/weights/step-5"), state_dir)
      self.assertEqual(gateway.tinker_state_path("/elsewhere/final"), "/elsewhere/final")
      # Only weights paths are checkpoints. A sampler path is refused, not resolved under the caller.
      self.assertIsNone(gateway.tinker_checkpoint_dir("tinker://job-a/sampler_weights/sampler-3"))
      refused = asyncio.run(gateway.load_weights({"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"}))
      self.assertEqual(refused.status_code, 400)

  def test_save_state_keeps_the_optimizer_and_answers_with_a_tinker_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      asyncio.run(gateway.save_weights({"model_id": "job-a", "path": "step-5"}))
      queued = asyncio.run(gateway.store.get_requests())
      self.assertEqual(queued[0]["op"], "save_state")
      self.assertTrue(queued[0]["include_optimizer"])
      saved = gateway.translate_future_result({"type": "state_saved", "path": queued[0]["state_path"]})
    self.assertEqual(saved, {"type": "save_weights", "path": "tinker://job-a/weights/step-5"})

  def test_weights_info_reads_the_checkpoint_on_disk(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      os.makedirs(os.path.join(state_dir, "job-a"))
      with open(os.path.join(state_dir, "metadata.json"), "w") as f:
        json.dump({"base_model": "google/gemma-4-e2b", "model_id": "job-a", "has_optimizer": True}, f)
      with open(os.path.join(state_dir, "job-a", "adapter_config.json"), "w") as f:
        json.dump({"r": 8}, f)
      info = asyncio.run(gateway.weights_info({"tinker_path": "tinker://job-a/weights/step-5"}))
      missing = asyncio.run(gateway.weights_info({"tinker_path": "tinker://job-a/weights/never"}))
    self.assertEqual(info["base_model"], "google/gemma-4-e2b")
    self.assertTrue(info["is_lora"])
    self.assertEqual(info["lora_rank"], 8)
    self.assertEqual(missing.status_code, 404)

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(gateway.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


if __name__ == "__main__":
  unittest.main()
