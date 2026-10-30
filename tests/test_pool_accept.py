import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import redis

from server import api_server
from server.lora_snapshots import freeze_adapter, snapshot_name
from server.model_metadata import TrainingModelMetadata, persist_model_metadata
from server.sampling_store import SamplingStore, stream_key
from server.scheduler_worker_manager import SchedulerWorkerManager
from server.session_registry import SessionRegistry
from server.store import RedisStateStore, RedisStore
from server.worker_manager import default_metadata, owner_of
from tests.api_client import asgi_client
from tests.redis_server import RedisTestCase
from tests.test_scheduler_worker_manager import FakeApi, FakeCustomObjectsApi


class PoolAcceptTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.tmp, self.base = self.enterContext(tempfile.TemporaryDirectory()), f"org/m-{uuid.uuid4().hex[:8]}"
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp, "SAMPLING_BACKEND": "vllm"}))
    self.state, self.store = RedisStateStore(self.redis, redis.Redis.from_url(self.url, decode_responses=True)), RedisStore(self.redis)
    patched = [("store", self.store), ("state", self.state), ("TMP_DIR", self.tmp), ("worker_manager", Mock()), ("SAVED_WEIGHTS_WAIT_SECONDS", 0.3)]
    for name, value in patched:
      self.enterContext(patch.object(api_server, name, value))
    self.client = await self.enterAsyncContext(asgi_client())

  async def sample(self, seq_id: int = 0, session: str | None = None, **create) -> dict:
    session = session or (await self.client.post("/api/v1/create_sampling_session", json=create)).json()["sampling_session_id"]
    body = {"sampling_session_id": session, "seq_id": seq_id, "prompt": {"chunks": [{"tokens": [1]}]}}
    response = await self.client.post("/api/v1/asample", json=body)
    return {"session": session, "status": response.status_code, **response.json()}

  async def pending(self, pool: str, field: str) -> list:
    return [r[field] for r in await SamplingStore(self.redis).pending(pool)]

  async def test_a_retried_seq_id_is_one_request_and_another_session_gets_its_own(self) -> None:
    first = await self.sample(7, base_model=self.base)
    retry, other = await self.sample(7, first["session"]), await self.sample(7, base_model=self.base)
    self.assertEqual(retry["request_id"], first["request_id"])
    self.assertEqual(await self.pending(default_metadata(self.base).pool(self.base), "request_id"), [first["request_id"], other["request_id"]])

  async def test_fft_samples_need_saved_weights_and_name_their_immutable_revision(self) -> None:
    model_id = uuid.uuid4().hex
    await persist_model_metadata(self.state, model_id, TrainingModelMetadata(base_model=self.base, fine_tuning_type="full"))
    saved = Path(self.tmp, "sampler_full", model_id, "sampler_weights", "sampler-1")
    saved.mkdir(parents=True)
    (saved / "metadata.json").write_text("{}")
    missing = await self.sample(model_path=f"tinker://{model_id}/sampler_weights/sampler-9")
    self.assertEqual((missing["status"], missing["error"]), (400, f"No saved sampler weights for tinker://{model_id}/sampler_weights/sampler-9"))
    await self.sample(model_path=f"tinker://{model_id}")
    self.assertEqual(await self.pending(f"sampler-{model_id}", "revision"), [str(saved)])

  async def test_weights_saved_while_the_sample_waits_are_accepted(self) -> None:
    # A save from the trainer's node can show up on NFS a moment after the client hears it finished.
    self.enterContext(patch.object(api_server, "SAVED_WEIGHTS_WAIT_SECONDS", 5.0))
    model_id = uuid.uuid4().hex
    await persist_model_metadata(self.state, model_id, TrainingModelMetadata(base_model=self.base, fine_tuning_type="full"))
    saved = Path(self.tmp, "sampler_full", model_id, "sampler_weights", "sampler-1")

    def finish_save() -> None:
      saved.mkdir(parents=True)
      (saved / "metadata.json").write_text("{}")

    asyncio.get_running_loop().call_later(0.5, finish_save)
    accepted = await self.sample(model_path=f"tinker://{model_id}/sampler_weights/sampler-1")
    self.assertEqual(accepted["status"], 200)
    self.assertEqual(await self.pending(f"sampler-{model_id}", "revision"), [str(saved)])

  async def test_lora_samples_need_a_frozen_snapshot(self) -> None:
    model_id = uuid.uuid4().hex
    meta = TrainingModelMetadata(base_model=self.base, fine_tuning_type="lora")
    await persist_model_metadata(self.state, model_id, meta)
    ref = f"tinker://{model_id}/sampler_weights/sampler-1"
    refused = await self.sample(model_path=ref)
    self.assertEqual((refused["status"], refused["error"]), (400, f"No saved LoRA sampler weights for {ref}"))
    Path(self.tmp, "peft", model_id, model_id).mkdir(parents=True)
    Path(self.tmp, "peft", model_id, model_id, "adapter_config.json").write_text("{}")
    freeze_adapter(model_id, self.base, ref)
    self.assertEqual((await self.sample(1, refused["session"]))["status"], 200)
    self.assertEqual(await self.pending(meta.pool(model_id), "model"), [snapshot_name(ref)])

  async def test_pool_names_are_valid_kubernetes_names(self) -> None:
    pools = {default_metadata("Qwen/Qwen3.5-9B").pool("Qwen/Qwen3.5-9B"), TrainingModelMetadata(base_model="Qwen/Qwen3-8B").pool("job")}
    self.assertTrue(all(len(p) <= 63 and p == p.lower() and "/" not in p and "." not in p for p in pools), pools)

  def on_kubernetes(self, workloads: FakeCustomObjectsApi | None = None) -> None:
    """The API server with the scheduler's worker manager over fake Kubernetes APIs."""
    self.workloads, self.apps = workloads or FakeCustomObjectsApi(), FakeApi()
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": self.url}))
    self.manager = SchedulerWorkerManager(custom_api=self.workloads, apps_api=self.apps, networking_api=FakeApi())
    self.enterContext(patch.object(api_server, "worker_manager", self.manager))
    self.enterContext(patch.object(api_server, "session_registry", SessionRegistry(self.state)))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))

  async def sampling_session(self, live: bool, **create) -> str:
    """A sampling session on a pool the scheduler launched, opened by a client session that is still live or has ended."""
    client_session = (await self.client.post("/api/v1/create_session", json={})).json()["session_id"]
    body = {"session_id": client_session, **create}
    sampling = (await self.client.post("/api/v1/create_sampling_session", json=body)).json()["sampling_session_id"]
    self.assertTrue(self.launched())
    if not live:
      await self.state.delete_values(f"open_rl:session:{client_session}")
    return sampling

  async def sampling_on_kubernetes(self, live: bool) -> str:
    self.on_kubernetes()
    return await self.sampling_session(live, base_model=self.base)

  def launched(self) -> set:
    return set(self.workloads.existing) | set(self.apps.objects)

  def delete_pool(self) -> None:
    # By hand, by a node pool going away, or by the reaper. A pass also serves other tests' leftover requests in the
    # shared Redis, so checks look only at this test's pool.
    self.workloads.existing.clear()
    self.apps.objects.clear()

  async def answer(self, pool: str, count: int = 1) -> None:
    sampling = SamplingStore(self.redis)
    for r in (await sampling.pending(pool))[:count]:
      await sampling.set_result(r, {"type": "sample", "sequences": []})

  async def test_the_reaper_deletes_a_released_pools_stream_once_nothing_is_pending_in_it(self) -> None:
    pool = default_metadata(self.base).pool(self.base)
    session = await self.sampling_on_kubernetes(live=False)
    await self.sample(0, session)
    await self.sample(1, session)
    await self.answer(pool)
    await api_server.reap_owner(owner_of(self.base))
    self.assertEqual(len(await self.pending(pool, "request_id")), 1)
    # The pool comes back for the next session, and once its last request is answered the reaper deletes the stream.
    session = await self.sampling_session(live=False, base_model=self.base)
    await self.answer(pool)
    await api_server.reap_owner(owner_of(self.base))
    self.assertFalse(await self.redis.exists(stream_key(pool)))

  async def test_deleting_an_fft_job_deletes_its_pools_empty_stream(self) -> None:
    self.on_kubernetes()
    model_id = uuid.uuid4().hex
    meta = TrainingModelMetadata(base_model=self.base, fine_tuning_type="full")
    await persist_model_metadata(self.state, model_id, meta)
    record = {"request_id": str(uuid.uuid4()), "pool": meta.pool(model_id), "job": model_id, "revision": "", "model": self.base}
    await SamplingStore(self.redis).accept({**record, "idempotency_key": None, "prompt_token_ids": [1]})
    await self.answer(meta.pool(model_id))
    self.assertTrue(await self.redis.exists(stream_key(meta.pool(model_id))))
    self.assertEqual((await self.client.post("/api/v1/delete_model", json={"model_id": model_id})).status_code, 200)
    self.assertFalse(await self.redis.exists(stream_key(meta.pool(model_id))))

  async def test_the_reconcile_deletes_a_stream_whose_last_request_was_answered_after_its_pool_went_away(self) -> None:
    # As the deadline sweep, or a sampler's late result, does once the reaper has already released the pool.
    model_id = uuid.uuid4().hex
    meta = TrainingModelMetadata(base_model=self.base, fine_tuning_type="full")
    record = {"request_id": str(uuid.uuid4()), "pool": meta.pool(model_id), "job": model_id, "revision": "", "model": self.base}
    await SamplingStore(self.redis).accept({**record, "idempotency_key": None, "prompt_token_ids": [1]})
    await self.answer(meta.pool(model_id))
    await api_server.reconcile_pools()
    self.assertFalse(await self.redis.exists(stream_key(meta.pool(model_id))))

  async def test_a_pool_deleted_under_a_live_jobs_pending_requests_comes_back_without_another_sample(self) -> None:
    session = await self.sampling_on_kubernetes(live=True)
    launched = self.launched()
    await self.sample(0, session)
    self.delete_pool()
    await api_server.reconcile_pools()
    self.assertLessEqual(launched, self.launched())


class PoolsNeedRedisTest(unittest.IsolatedAsyncioTestCase):
  def env(self, **values: str):
    unset = ("REDIS_URL", "BASE_MODEL", "SAMPLING_BACKEND", "OPEN_RL_ENABLE_FFT", "OPEN_RL_WORKER_MANAGER")
    return patch.dict(os.environ, {**{k: v for k, v in os.environ.items() if k not in unset}, **values}, clear=True)

  async def test_a_server_that_samples_through_pools_will_not_start_without_redis(self) -> None:
    with self.env(), self.assertRaisesRegex(RuntimeError, "Sampling pools need REDIS_URL"):
      async with api_server.lifespan(api_server.app):
        pass

  def test_torch_sampling_in_one_process_and_pools_with_redis_pass_the_check(self) -> None:
    for values in ({"BASE_MODEL": "org/m"}, {"REDIS_URL": "redis://127.0.0.1:6379/0"}):
      with self.subTest(**values), self.env(**values):
        api_server.check_sampler_backend()
