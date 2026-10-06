import json
import unittest
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

  async def set_future(self, req_id: str, result: dict) -> None:
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
          "full_config": {"weight_sync_strategy": "delta"},
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

  async def test_local_sampler_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = LocalWorkerManager()
      manager.ensure("Model_A.1", "sampler")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_MODEL_ID"], "Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "sampler-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "samplers")

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


class LocalWorkerManagerCommandTest(unittest.TestCase):
  """The exact command and env the local manager launches each worker with."""

  BASE_ENV = {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm", "TRAINER_TPU_VISIBLE_CHIPS": "1"}

  def launch(self, role: str, env: dict[str, str], **prefs) -> tuple[list[str], dict[str, str]]:
    import tempfile
    from pathlib import Path

    from server.model_metadata import TrainingModelMetadata

    meta = TrainingModelMetadata(base_model="Qwen/Qwen3-0.6B", fine_tuning_type="lora", **prefs)
    with (
      tempfile.TemporaryDirectory() as tmp,
      patch.dict("os.environ", {**env, "OPEN_RL_TMP_DIR": tmp}, clear=True),
      patch("server.worker_manager.metadata_for", return_value=meta),
      patch("server.worker_manager.shutil.which", return_value="/usr/bin/uv"),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      LocalWorkerManager(project_dir=Path("/repo")).ensure("model-1", role)
    args, kwargs = popen.call_args
    self.assertEqual(kwargs["cwd"], Path("/repo"))
    launched_env = dict(kwargs["env"])
    launched_env.pop("OPEN_RL_TMP_DIR")
    return args[0], launched_env

  def test_gpu_trainer_command_and_env_are_unchanged(self) -> None:
    command, env = self.launch("trainer", {**self.BASE_ENV, "TRAINER_CUDA_VISIBLE_DEVICES": "0"})

    self.assertEqual(
      command,
      [
        "uv", "run", "--extra", "gpu", "python", "-u", "-m", "server.training_requests_processor",
        "--model-id", "Qwen/Qwen3-0.6B", "--active-tenant-set-id", "Qwen/Qwen3-0.6B-1",
      ],
    )  # fmt: skip
    self.assertEqual(
      env,
      {
        **self.BASE_ENV,
        "TRAINER_CUDA_VISIBLE_DEVICES": "0",
        "BASE_MODEL": "Qwen/Qwen3-0.6B",
        "OPEN_RL_BASE_MODEL": "Qwen/Qwen3-0.6B",
        "OPEN_RL_ENABLE_FFT": "false",
        "OPEN_RL_FINE_TUNING_TYPE": "lora",
        "OPEN_RL_ACCELERATOR_MEMORY": "5487067136",
        "OPEN_RL_WEIGHT_SYNC_STRATEGY": "delta",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OPEN_RL_TIME_SLICE_JOB_ID": "trainer-Qwen/Qwen3-0.6B",
        "OPEN_RL_TIME_SLICE_GROUP": "trainers",
        "CUDA_VISIBLE_DEVICES": "0",
      },
    )

  def test_gpu_sampler_command_and_env_are_unchanged(self) -> None:
    command, env = self.launch("sampler", {**self.BASE_ENV, "SAMPLER_CUDA_VISIBLE_DEVICES": "1"})

    self.assertEqual(
      command,
      ["uv", "run", "--extra", "gpu", "--extra", "vllm", "python", "-u", "-m", "server.vllm_sampler", "--model-id", "Qwen/Qwen3-0.6B"],
    )
    self.assertEqual(
      env,
      {
        **self.BASE_ENV,
        "SAMPLER_CUDA_VISIBLE_DEVICES": "1",
        "BASE_MODEL": "Qwen/Qwen3-0.6B",
        "OPEN_RL_BASE_MODEL": "Qwen/Qwen3-0.6B",
        "OPEN_RL_ENABLE_FFT": "false",
        "OPEN_RL_FINE_TUNING_TYPE": "lora",
        "OPEN_RL_ACCELERATOR_MEMORY": "10542424064",
        "OPEN_RL_WEIGHT_SYNC_STRATEGY": "delta",
        "OPEN_RL_MODEL_ID": "Qwen/Qwen3-0.6B",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "OPEN_RL_TIME_SLICE_JOB_ID": "sampler-Qwen/Qwen3-0.6B",
        "OPEN_RL_TIME_SLICE_GROUP": "samplers",
        "CUDA_VISIBLE_DEVICES": "1",
      },
    )

  def test_tpu_trainer_runs_in_its_own_env_on_its_chip(self) -> None:
    command, env = self.launch("trainer", {**self.BASE_ENV, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}, trainer_accel_prefs=["tpu", "gpu"])

    self.assertEqual(command[:4], ["uv", "run", "--extra", "tpu"])
    self.assertIn("server.training_requests_processor", command)
    self.assertEqual(env["UV_PROJECT_ENVIRONMENT"], "/repo/.venv-tpu-trainer")
    self.assertEqual(env["OPEN_RL_DEVICE"], "tpu")
    self.assertEqual(env["TORCH_DEVICE_BACKEND_AUTOLOAD"], "1")
    self.assertEqual(env["TPU_VISIBLE_CHIPS"], "1")
    self.assertEqual(env["TPU_PROCESS_BOUNDS"], "1,1,1")
    self.assertEqual(env["TPU_CHIPS_PER_PROCESS_BOUNDS"], "1,1,1")

  def test_tpu_trainer_refuses_a_multi_chip_pin(self) -> None:
    with self.assertRaisesRegex(ValueError, "one chip"):
      self.launch("trainer", {**self.BASE_ENV, "TRAINER_TPU_VISIBLE_CHIPS": "0,1"}, trainer_accel_prefs=["tpu"])

  def test_tpu_sampler_is_refused_until_tpu_samplers_exist(self) -> None:
    with self.assertRaisesRegex(NotImplementedError, "TPU samplers"):
      self.launch("sampler", self.BASE_ENV, sampler_accel_prefs=["tpu"])


class ApiServerMetadataExtractionTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = StoreStub()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", self.store))

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
    model_id, _ = await api_server._extract_and_persist_model_metadata(
      api_server.CreateModelRequest(base_model="Qwen/Qwen2.5-0.5B"),
      request,
      default_fine_tuning_type="full",
    )

    meta_val = self.store.kv_store.get(f"open_rl:model_meta:{model_id}")
    self.assertIsNotNone(meta_val)
    meta_dict = json.loads(meta_val)
    self.assertEqual(meta_dict["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(meta_dict["fine_tuning_type"], "lora")
    self.assertEqual(meta_dict["weight_sync_config"]["strategy"], "delta")


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
    from pathlib import Path

    with patch.dict("os.environ", {"REDIS_URL": "redis://127.0.0.1:6379"}):
      self.manager = LocalWorkerManager(project_dir=Path("/tmp"))
    self.store = StoreStub()

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

    # Launch for first LoRA model ID
    self.manager.ensure("model-lora-1", "sampler")
    self.assertIn(("sampler", "Qwen/Qwen2.5-0.5B"), self.manager.processes)
    self.assertEqual(mock_popen.call_count, 1)

    cmd_args = mock_popen.call_args[0][0]
    self.assertIn("server.vllm_sampler", cmd_args)
    self.assertIn("Qwen/Qwen2.5-0.5B", cmd_args)

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
    )

    self.manager.ensure("model-fft-1", "sampler")
    self.assertIn(("sampler", "model-fft-1"), self.manager.processes)
    self.assertEqual(mock_popen.call_count, 1)

    cmd_args = mock_popen.call_args[0][0]
    self.assertIn("server.vllm_sampler", cmd_args)
    self.assertIn("model-fft-1", cmd_args)


if __name__ == "__main__":
  unittest.main()
