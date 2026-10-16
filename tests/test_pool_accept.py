import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import redis

from server import api_server
from server.lora_snapshots import freeze_adapter, snapshot_name
from server.model_metadata import TrainingModelMetadata, persist_model_metadata
from server.sampling_store import SamplingStore
from server.store import RedisStateStore, RedisStore
from server.worker_manager import default_metadata
from tests.api_client import asgi_client
from tests.redis_server import RedisTestCase


class PoolAcceptTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.tmp, self.base = self.enterContext(tempfile.TemporaryDirectory()), f"org/m-{uuid.uuid4().hex[:8]}"
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp, "SAMPLING_BACKEND": "vllm"}))
    self.state, self.store = RedisStateStore(self.redis, redis.Redis.from_url(self.url, decode_responses=True)), RedisStore(self.redis)
    for name, value in [("store", self.store), ("state", self.state), ("TMP_DIR", self.tmp), ("worker_manager", Mock())]:
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
