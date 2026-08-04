import json
import unittest
from unittest.mock import patch

from server import gateway
from server.worker_manager import LocalWorkerManager


class StoreStub:
  def __init__(self):
    self.forwarded_requests = []
    self.futures = {}
    self.kv_store = {}

  async def put_request(self, req_data: dict, active_set_id: str | None = None) -> None:
    self.forwarded_requests.append(req_data)

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result

  async def set_value(self, key: str, value: str) -> None:
    self.kv_store[key] = value

  async def get_value(self, key: str) -> str | None:
    return self.kv_store.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.kv_store.get(key)

  async def get_model_metadata(self, model_id: str) -> dict | None:
    val = self.kv_store.get(f"open_rl:model_meta:{model_id}")
    if val:
      try:
        return json.loads(val)
      except Exception:
        return None
    return None


class WorkerManagerStub:
  def __init__(self, error: Exception | None = None):
    self.error = error
    self.launched_model_ids = []
    self.launched_trainer_model_ids = []
    self.launched_sampler_model_ids = []
    self.shutdown_model_ids = []

  def launch(self, model_id: str) -> None:
    self.launched_model_ids.append(model_id)
    if self.error is not None:
      raise self.error

  def launch_trainer(self, model_id: str) -> None:
    self.launched_trainer_model_ids.append(model_id)
    self.launch(model_id)

  def launch_sampler(self, model_id: str) -> None:
    self.launched_sampler_model_ids.append(model_id)
    self.launch(model_id)

  def shutdown(self, model_id: str) -> None:
    self.shutdown_model_ids.append(model_id)

  def shutdown_all(self) -> None:
    pass


class GatewayInlineWorkerLaunchTest(unittest.IsolatedAsyncioTestCase):
  """create_model in FFT mode launches the model's worker directly, then
  enqueues onto its per-model queue — there is no separate launch queue."""

  def setUp(self) -> None:
    self.store = StoreStub()
    self.worker_manager = WorkerManagerStub()
    self.old_store = gateway.store
    self.old_manager = gateway.worker_manager
    gateway.store = self.store
    gateway.worker_manager = self.worker_manager
    self.addCleanup(self._restore)

  def _restore(self) -> None:
    gateway.store = self.old_store
    gateway.worker_manager = self.old_manager

  async def test_create_model_launches_worker_then_enqueues(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    request = self.store.forwarded_requests[0]
    self.assertEqual(request["op"], "create_model")
    self.assertEqual(request["model_id"], model_id)
    self.assertEqual(request["payload"], {})
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "base-model")

  async def test_create_model_failed_launch_fails_future_and_enqueues_nothing(self) -> None:
    self.worker_manager.error = RuntimeError("boom")

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.gateway.traceback.print_exc"):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(self.store.forwarded_requests, [])
    self.assertEqual(self.store.futures[model_id], {"type": "RequestFailedResponse", "error_message": "boom"})

  async def test_create_model_from_state_launches_worker_then_enqueues(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model_from_state(
        {
          "state_path": "/tmp/checkpoint",
          "base_model": "restored-base",
          "full_config": {"weight_sync_strategy": "delta"},
          "restore_optimizer": True,
        }
      )

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    req_forwarded = self.store.forwarded_requests[0]
    self.assertEqual(req_forwarded["op"], "create_model_from_state")
    self.assertEqual(req_forwarded["payload"]["state_path"], "/tmp/checkpoint")
    self.assertTrue(req_forwarded["payload"]["restore_optimizer"])

    # Assert canonical metadata persistence:
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "restored-base")
    self.assertEqual(meta["fine_tuning_type"], "restored")
    self.assertEqual(meta["full_config"]["weight_sync_strategy"], "delta")

    # Assert no dual-key writing:
    self.assertIsNone(self.store.get_value_sync(f"open_rl:model_base:{model_id}"))

  async def test_ensure_sampler_launched_delegates_to_worker_manager_with_model_id(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}):
      self.store.kv_store["open_rl:model_meta:model-x"] = json.dumps(
        {
          "base_model": "base-vllm",
          "weight_sync_strategy": "delta",
          "fine_tuning_type": "full",
        }
      )
      await gateway.ensure_sampler_launched("model-x")

    self.assertEqual(self.worker_manager.launched_sampler_model_ids, ["model-x"])

  async def test_create_model_launches_trainer_when_worker_manager_present(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "false"}):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)


class GatewayLifespanTest(unittest.IsolatedAsyncioTestCase):
  async def test_lifespan_full_mode_requires_redis(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      async with gateway.lifespan(gateway.app):
        pass


class LocalWorkerManagerTest(unittest.IsolatedAsyncioTestCase):
  async def test_requires_redis(self) -> None:
    with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      LocalWorkerManager()

  async def test_local_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.launch("Model_A.1")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  async def test_local_sampler_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.launch_sampler("Model_A.1")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_MODEL_ID"], "Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "sampler-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "samplers")

  async def test_launch_fetches_metadata_from_store(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "base-model-a",
        "weight_sync_config": {"strategy": "delta"},
        "fine_tuning_type": "full",
      }
    )

    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.store.get_store", return_value=s),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.launch_trainer("Model_A.1")
      _, kwargs = popen.call_args
      self.assertEqual(kwargs["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")

      manager.launch_sampler("Model_A.1")
      _, kwargs_s = popen.call_args
      self.assertEqual(kwargs_s["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs_s["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")


class GatewayMetadataExtractionTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = StoreStub()
    self.old_store = gateway.store
    gateway.store = self.store
    self.addCleanup(self._restore)

  def _restore(self) -> None:
    gateway.store = self.old_store

  async def test_extract_and_persist_metadata_from_headers(self) -> None:
    import json

    from fastapi import Request

    scope = {
      "type": "http",
      "headers": [
        (b"x-open-rl-weight-sync-strategy", b"delta"),
        (b"x-open-rl-fine-tuning-type", b"lora"),
      ],
    }
    request = Request(scope)
    model_id = await gateway._extract_and_persist_model_metadata(
      {"base_model": "Qwen/Qwen2.5-0.5B"},
      request,
      default_fine_tuning_type="full",
    )

    meta_val = self.store.kv_store.get(f"open_rl:model_meta:{model_id}")
    self.assertIsNotNone(meta_val)
    meta_dict = json.loads(meta_val)
    self.assertEqual(meta_dict["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(meta_dict["fine_tuning_type"], "lora")
    self.assertEqual(meta_dict["weight_sync_config"]["strategy"], "delta")


class GatewayFutureTranslationTest(unittest.TestCase):
  def test_create_model_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_created",
          "model_id": "model-a",
          "base_model": "base-model",
          "fine_tuning_type": "full",
        }
      ),
      {
        "type": "create_model",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 16,
      },
    )

  def test_create_model_from_state_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_loaded_from_state",
          "model_id": "model-a",
          "base_model": "base-model",
          "fine_tuning_type": "full",
        }
      ),
      {
        "type": "create_model_from_state",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 16,
      },
    )

  def test_lora_create_model_result_translates_rank_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_created",
          "model_id": "model-a",
          "base_model": "base-model",
          "rank": 4,
          "fine_tuning_type": "lora",
        }
      ),
      {
        "type": "create_model",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 4,
      },
    )

  def test_internal_future_result_types_translate_to_tinker_types(self) -> None:
    cases = [
      ("forward_backward_completed", "forward_backward"),
      ("optim_step_completed", "optim_step"),
      ("sample_completed", "sample"),
      ("state_saved", "save_weights"),
      ("weights_loaded", "load_weights"),
      ("sampler_weights_saved", "save_weights_for_sampler"),
      ("weights_saved", "save_weights"),
    ]

    for internal_type, public_type in cases:
      with self.subTest(internal_type=internal_type):
        self.assertEqual(
          gateway.translate_future_result({"type": internal_type, "path": "/tmp/x"}),
          {"type": public_type, "path": "/tmp/x"},
        )


class LocalWorkerManagerSamplerLaunchTest(unittest.TestCase):
  def setUp(self) -> None:
    from pathlib import Path

    with patch.dict("os.environ", {"REDIS_URL": "redis://127.0.0.1:6379"}):
      self.manager = LocalWorkerManager(project_dir=Path("/tmp"))
    self.store = StoreStub()

  @patch("server.worker_manager._fetch_metadata_from_store")
  @patch("subprocess.Popen")
  def test_launch_sampler_lora_uses_base_model_and_reuses_process(self, mock_popen, mock_fetch) -> None:
    from server.model_metadata import TrainingModelMetadata

    mock_proc = unittest.mock.MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_fetch.return_value = TrainingModelMetadata(
      base_model="Qwen/Qwen2.5-0.5B",
      created_at=100.0,
      fine_tuning_type="lora",
    )

    # Launch for first LoRA model ID
    self.manager.launch_sampler("model-lora-1")
    self.assertIn("Qwen/Qwen2.5-0.5B", self.manager.sampler_processes)
    self.assertEqual(mock_popen.call_count, 1)

    cmd_args = mock_popen.call_args[0][0]
    self.assertIn("server.lora_sampler", cmd_args)
    self.assertIn("Qwen/Qwen2.5-0.5B", cmd_args)

    # Launch for second LoRA model ID sharing the same base model
    self.manager.launch_sampler("model-lora-2")
    # Should reuse existing process and NOT call popen again!
    self.assertEqual(mock_popen.call_count, 1)

  @patch("server.worker_manager._fetch_metadata_from_store")
  @patch("subprocess.Popen")
  def test_launch_sampler_fft_uses_model_id(self, mock_popen, mock_fetch) -> None:
    from server.model_metadata import TrainingModelMetadata

    mock_proc = unittest.mock.MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    mock_fetch.return_value = TrainingModelMetadata(
      base_model="Qwen/Qwen2.5-0.5B",
      created_at=100.0,
      fine_tuning_type="full",
    )

    self.manager.launch_sampler("model-fft-1")
    self.assertIn("model-fft-1", self.manager.sampler_processes)
    self.assertEqual(mock_popen.call_count, 1)

    cmd_args = mock_popen.call_args[0][0]
    self.assertIn("server.vllm_sampler", cmd_args)
    self.assertIn("model-fft-1", cmd_args)


if __name__ == "__main__":
  unittest.main()
