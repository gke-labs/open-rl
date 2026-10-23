import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server import api_server
from server.session_registry import SessionRegistry
from server.worker_manager import LocalWorkerManager
from tests.api_client import asgi_client, post_json


class StoreStub:
  def __init__(self):
    self.forwarded_requests = []
    self.futures = {}
    self.kv_store = {}

  async def put_request(self, req_data: dict, active_set_id: str | None = None) -> None:
    self.forwarded_requests.append(req_data)

  async def set_result(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result

  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    self.kv_store[key] = value

  async def add_to_set(self, key: str, member: str) -> None:
    pass

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

  def ensure(self, model_id: str, role: str) -> None:
    self.launched_model_ids.append(model_id)
    (self.launched_trainer_model_ids if role == "trainer" else self.launched_sampler_model_ids).append(model_id)
    if self.error is not None:
      raise self.error

  def release(self, model_id: str) -> None:
    self.shutdown_model_ids.append(model_id)

  def close(self) -> None:
    pass


class ApiServerInlineWorkerLaunchTest(unittest.IsolatedAsyncioTestCase):
  """create_model in FFT mode launches the model's worker directly, then
  enqueues onto its per-model queue — there is no separate launch queue."""

  def setUp(self) -> None:
    self.store = StoreStub()
    self.worker_manager = WorkerManagerStub()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", self.store))
    self.enterContext(patch.object(api_server, "worker_manager", self.worker_manager))
    self.enterContext(patch.object(api_server, "session_registry", SessionRegistry(self.store)))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.store))

  async def asyncSetUp(self) -> None:
    self.client = await self.enterAsyncContext(asgi_client())
    self.session_id = (await self.post("create_session", {}))["session_id"]

  async def post(self, path: str, body: dict) -> dict:
    return await post_json(self.client, path, body)

  async def test_create_model_launches_worker_then_enqueues(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await self.post("create_model", {"base_model": "base-model", "session_id": self.session_id})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    request = self.store.forwarded_requests[0]
    self.assertEqual(request["op"], "create_model")
    self.assertEqual(request["model_id"], model_id)
    self.assertEqual(request["payload"]["base_model"], "base-model")
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "base-model")

  async def test_create_model_failed_launch_fails_future_and_enqueues_nothing(self) -> None:
    self.worker_manager.error = RuntimeError("boom")

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.api_server.traceback.print_exc"):
      result = await self.post("create_model", {"base_model": "base-model", "session_id": self.session_id})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(self.store.forwarded_requests, [])
    self.assertEqual(self.store.futures[model_id], {"type": "RequestFailedResponse", "error_message": "boom"})

  async def test_create_model_from_state_launches_worker_then_enqueues(self) -> None:
    self.enterContext(patch.object(api_server, "checkpoint_info", return_value={"base_model": "restored-base", "is_lora": True}))
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await self.post(
        "create_model_from_state",
        {
          "session_id": self.session_id,
          "state_path": "/tmp/checkpoint",
          "base_model": "restored-base",
          "restore_optimizer": True,
        },
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
    self.assertEqual(meta["fine_tuning_type"], "lora")
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
      await api_server.bind_session(self.session_id, "model-x")
      await api_server.ensure_sampler_launched("model-x")

    self.assertEqual(self.worker_manager.launched_sampler_model_ids, ["model-x"])

  async def test_create_model_launches_trainer_when_worker_manager_present(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "false"}):
      result = await self.post("create_model", {"base_model": "base-model", "session_id": self.session_id})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)


class ApiServerLifespanTest(unittest.IsolatedAsyncioTestCase):
  async def test_lifespan_full_mode_requires_redis(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      async with api_server.lifespan(api_server.app):
        pass


class CreateWorkerManagerTest(unittest.TestCase):
  def test_none_mode_disables_worker_manager(self) -> None:
    from server.worker_manager import create_worker_manager

    with patch.dict("os.environ", {"OPEN_RL_WORKER_MANAGER": "none", "REDIS_URL": "redis://localhost:6379"}, clear=True):
      self.assertIsNone(create_worker_manager())


class LocalWorkerManagerTest(unittest.IsolatedAsyncioTestCase):
  async def test_requires_redis(self) -> None:
    with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      LocalWorkerManager()

  async def test_local_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_ENABLE_FFT": "true"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.ensure("Model_A.1", "trainer")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  async def test_a_local_fft_sampler_on_the_trainers_gpu_serves_its_pool_itself_in_the_trainers_claim(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      LocalWorkerManager().ensure("fft-job", "sampler")
    ((command, kwargs),) = [(c[0][0], c[1]) for c in popen.call_args_list]
    self.assertEqual(command[command.index("-m") + 1 :], ["server.vllm_sampler", "--pool", "sampler-fft-job"])
    self.assertEqual((kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"]), ("sampler-fft-job", "trainers"))

  async def test_launch_fetches_metadata_from_store(self) -> None:
    import json

    from server.store import InMemoryStateStore

    s = InMemoryStateStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "base-model-a",
        "weight_sync_config": {"strategy": "delta"},
        "fine_tuning_type": "full",
      }
    )

    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.worker_manager.get_state_store", return_value=s),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.ensure("Model_A.1", "trainer")
      _, kwargs = popen.call_args
      self.assertEqual(kwargs["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")

      manager.ensure("Model_A.1", "sampler")
      _, kwargs_s = popen.call_args
      self.assertEqual(kwargs_s["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs_s["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")


class ApiServerMetadataExtractionTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = StoreStub()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", self.store))

  async def test_extract_and_persist_metadata_from_settings(self) -> None:
    import json

    user_metadata = {"openrl.fine_tuning_type": "full", "openrl.weight_sync": "full", "openrl.fft_seed": "7"}
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      model_id, _ = await api_server._extract_and_persist_model_metadata(
        api_server.CreateModelRequest(base_model="Qwen/Qwen2.5-0.5B", user_metadata=user_metadata)
      )

    meta_dict = json.loads(self.store.kv_store.get(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta_dict["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(meta_dict["fine_tuning_type"], "full")
    self.assertEqual(meta_dict["weight_sync_config"]["strategy"], "full")
    self.assertEqual(meta_dict["full_config"], {"seed": 7, "cpu_offload": True, "weight_sync_strategy": "full"})


class ApiServerFutureTranslationTest(unittest.TestCase):
  def test_create_model_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      api_server.translate_future_result(
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
      api_server.translate_future_result(
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
      api_server.translate_future_result(
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
          api_server.translate_future_result({"type": internal_type, "path": "/tmp/x"}),
          {"type": public_type, "path": "/tmp/x"},
        )


class LocalWorkerManagerSamplerLaunchTest(unittest.TestCase):
  def setUp(self) -> None:

    with patch.dict("os.environ", {"REDIS_URL": "redis://127.0.0.1:6379"}):
      self.manager = LocalWorkerManager(project_dir=Path("/tmp"))
    self.store = StoreStub()
    # A LoRA launch creates its snapshot directory under OPEN_RL_TMP_DIR.
    self.enterContext(patch.dict("os.environ", {"OPEN_RL_TMP_DIR": self.enterContext(tempfile.TemporaryDirectory())}))

  @patch("server.worker_manager.metadata_for")
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

    self.manager.ensure("model-lora-1", "sampler")
    self.assertEqual(set(self.manager.processes), {("sampler", "Qwen/Qwen2.5-0.5B")})
    ((command, serve_kwargs),) = [(c[0][0], c[1]) for c in mock_popen.call_args_list]
    self.assertEqual((command[-2], command[-1].startswith("sampler-qwen-qwen2-5-0-5b-")), ("--pool", True))
    self.assertEqual((serve_kwargs["env"]["BASE_MODEL"], serve_kwargs["env"]["OPEN_RL_FINE_TUNING_TYPE"]), ("Qwen/Qwen2.5-0.5B", "lora"))
    # Triton kernels live under OPEN_RL_TMP_DIR; vLLM's own compile cache stays where vLLM puts it.
    tmp = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    self.assertEqual(serve_kwargs["env"]["TRITON_CACHE_DIR"], f"{tmp}/cache/triton")
    self.assertEqual(serve_kwargs["env"].get("VLLM_CACHE_ROOT"), os.environ.get("VLLM_CACHE_ROOT"))

    # Launch for second LoRA model ID sharing the same base model
    self.manager.ensure("model-lora-2", "sampler")
    # Should reuse existing process and NOT call popen again!
    self.assertEqual(mock_popen.call_count, 1)

  @patch("server.worker_manager.metadata_for")
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
      exclusive=True,
    )

    self.manager.ensure("model-fft-1", "sampler")
    self.assertEqual(set(self.manager.processes), {("sampler", "model-fft-1")})
    ((command, kwargs),) = [(c[0][0], c[1]) for c in mock_popen.call_args_list]
    self.assertEqual(command[-2:], ["--pool", "sampler-model-fft-1"])
    self.assertEqual((kwargs["env"]["BASE_MODEL"], kwargs["env"]["OPEN_RL_FINE_TUNING_TYPE"]), ("Qwen/Qwen2.5-0.5B", "full"))


if __name__ == "__main__":
  unittest.main()
