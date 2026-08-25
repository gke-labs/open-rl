import unittest
from unittest.mock import patch

from redis.exceptions import TimeoutError as RedisTimeoutError

from server import gateway
from server.store import InMemoryStore, RedisStore
from server.worker_launch_processor import WorkerLaunchProcessor


class StoreStub:
  def __init__(self):
    self.forwarded_requests = []
    self.futures = {}

  async def put_request(self, req_data: dict) -> None:
    self.forwarded_requests.append(req_data)

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result


class WorkerManagerStub:
  def __init__(self, error: Exception | None = None):
    self.error = error
    self.launched_model_ids = []

  def launch(self, model_id: str) -> None:
    self.launched_model_ids.append(model_id)
    if self.error is not None:
      raise self.error


class WorkerLaunchProcessorTest(unittest.IsolatedAsyncioTestCase):
  async def test_process_request_launches_worker_then_forwards_request(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub()
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {
      "request_id": "model-a",
      "model_id": "model-a",
      "op": "create_model",
      "payload": {"base_model": "base-model"},
    }

    await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, ["model-a"])
    self.assertEqual(len(store.forwarded_requests), 1)
    self.assertEqual(store.forwarded_requests[0]["op"], "create_model")
    self.assertEqual(store.forwarded_requests[0]["payload"]["base_model"], "base-model")
    self.assertEqual(store.futures, {})

  async def test_process_request_sets_future_failure_when_launch_fails(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub(RuntimeError("boom"))
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {
      "request_id": "model-a",
      "model_id": "model-a",
      "op": "create_model",
      "payload": {"base_model": "base-model"},
    }

    with patch("server.worker_launch_processor.traceback.print_exc"):
      await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, ["model-a"])
    self.assertEqual(store.forwarded_requests, [])
    self.assertEqual(
      store.futures["model-a"],
      {"type": "RequestFailedResponse", "error_message": "boom"},
    )

  async def test_process_request_sets_future_failure_without_model_id(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub()
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {
      "request_id": "model-a",
      "op": "create_model",
      "payload": {"base_model": "base-model"},
    }

    with patch("server.worker_launch_processor.traceback.print_exc"):
      await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, [])
    self.assertEqual(store.forwarded_requests, [])
    self.assertEqual(store.futures["model-a"]["type"], "RequestFailedResponse")

  async def test_process_request_forwards_create_model_from_state_request(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub()
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {
      "request_id": "model-a",
      "model_id": "model-a",
      "op": "create_model_from_state",
      "payload": {"state_path": "/tmp/checkpoint"},
    }

    await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, ["model-a"])
    self.assertEqual(store.forwarded_requests, [request])


class WorkerLaunchQueueTest(unittest.IsolatedAsyncioTestCase):
  async def test_in_memory_worker_launch_queue_is_unsupported(self) -> None:
    store = InMemoryStore()

    with self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await store.put_worker_launch_request({"model_id": "model-a"})


class GatewayWorkerLaunchQueueTest(unittest.IsolatedAsyncioTestCase):
  async def test_lifespan_full_mode_requires_redis(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      async with gateway.lifespan(gateway.app):
        pass

  async def test_create_model_full_mode_uses_worker_launch_queue(self) -> None:
    store = StoreStub()
    store.worker_launch_requests = []

    async def put_worker_launch_request(req_data: dict) -> None:
      store.worker_launch_requests.append(req_data)

    store.put_worker_launch_request = put_worker_launch_request
    old_store = gateway.store
    gateway.store = store

    try:
      with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
        result = await gateway.create_model({"base_model": "base-model"})
    finally:
      gateway.store = old_store

    self.assertEqual(result["request_id"], store.worker_launch_requests[0]["model_id"])
    self.assertEqual(store.worker_launch_requests[0]["op"], "create_model")
    self.assertEqual(store.worker_launch_requests[0]["payload"]["base_model"], "base-model")
    self.assertEqual(store.forwarded_requests, [])


class GatewayFutureTranslationTest(unittest.TestCase):
  def test_create_model_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_created",
          "model_id": "model-a",
          "base_model": "base-model",
          "training_kind": "full",
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
          "training_kind": "full",
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
          "training_kind": "lora",
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


class RedisStoreTimeoutTest(unittest.IsolatedAsyncioTestCase):
  async def test_worker_launch_queue_timeout_returns_empty_batch(self) -> None:
    store = RedisStore("redis://unused")

    class RedisStub:
      async def blpop(self, *_args, **_kwargs):
        raise RedisTimeoutError("idle timeout")

    store.redis = RedisStub()

    self.assertEqual(await store.get_worker_launch_requests(), [])


if __name__ == "__main__":
  unittest.main()
