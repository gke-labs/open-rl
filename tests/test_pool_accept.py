import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import redis
from fastapi import HTTPException

from server import api_server
from server.lora_snapshots import freeze_adapter, snapshot_name
from server.model_metadata import TrainingModelMetadata, persist_model_metadata
from server.sampling_store import SamplingStore
from server.store import RedisStateStore, RedisStore
from tests.redis_server import RedisTestCase


class PoolAcceptTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.tmp, self.base, self.pool = self.enterContext(tempfile.TemporaryDirectory()), f"org/m-{uuid.uuid4().hex[:8]}", f"pool-{uuid.uuid4().hex}"
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp, "OPEN_RL_SAMPLER_POOLS": json.dumps({self.base: self.pool})}))
    self.state = RedisStateStore(self.redis, redis.Redis.from_url(self.url, decode_responses=True))
    for name, value in [("store", RedisStore(self.redis)), ("state", self.state), ("TMP_DIR", self.tmp)]:
      self.enterContext(patch.object(api_server, name, value))

  async def accept(self, model_id: str | None = None, session: str | None = None, seq_id: int = 0) -> str:
    req = api_server.AsampleRequest(model_id=model_id, sampling_session_id=session, seq_id=seq_id, prompt={"chunks": [{"tokens": [1]}]})
    return await api_server.accept_pool_sample(req, [1])

  async def pending(self, pool: str, field: str) -> list:
    return [r[field] for r in await SamplingStore(self.redis).pending(pool)]

  async def test_a_retried_seq_id_is_one_request_and_another_session_gets_its_own(self) -> None:
    session, other = f"{self.base}@{uuid.uuid4().hex}", f"{self.base}@{uuid.uuid4().hex}"
    first = await self.accept(session=session, seq_id=7)
    retry, second = await self.accept(session=session, seq_id=7), await self.accept(session=other, seq_id=7)
    self.assertEqual(retry, first)
    self.assertEqual(await self.pending(self.pool, "request_id"), [first, second])

  async def test_fft_samples_need_saved_weights_and_name_their_immutable_revision(self) -> None:
    model_id = uuid.uuid4().hex
    await persist_model_metadata(self.state, model_id, TrainingModelMetadata(base_model=self.base, fine_tuning_type="full"))
    saved = Path(self.tmp, "sampler_full", model_id, "sampler_weights", "sampler-1")
    saved.mkdir(parents=True)
    (saved / "metadata.json").write_text("{}")
    with self.assertRaises(HTTPException) as missing:
      await self.accept(f"tinker://{model_id}/sampler_weights/sampler-9")
    self.assertEqual(
      (missing.exception.status_code, missing.exception.detail), (400, f"No saved sampler weights for tinker://{model_id}/sampler_weights/sampler-9")
    )
    await self.accept(f"tinker://{model_id}")
    self.assertEqual(await self.pending(f"sampler-{model_id}", "revision"), [str(saved)])

  async def test_lora_samples_need_a_frozen_snapshot(self) -> None:
    model_id = uuid.uuid4().hex
    await persist_model_metadata(self.state, model_id, TrainingModelMetadata(base_model=self.base, fine_tuning_type="lora"))
    ref = f"tinker://{model_id}/sampler_weights/sampler-1"
    with self.assertRaises(HTTPException) as refused:
      await self.accept(ref)
    self.assertEqual((refused.exception.status_code, refused.exception.detail), (400, f"No saved LoRA sampler weights for {ref}"))
    Path(self.tmp, "peft", model_id, model_id).mkdir(parents=True)
    Path(self.tmp, "peft", model_id, model_id, "adapter_config.json").write_text("{}")
    freeze_adapter(model_id, self.base, ref)
    await self.accept(ref)
    self.assertEqual(await self.pending(self.pool, "model"), [snapshot_name(ref)])
