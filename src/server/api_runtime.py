"""Application-owned training submission, sessions, and background task lifetime."""

import asyncio
import json
import os
import traceback
import uuid
from collections import defaultdict
from typing import Any

from opentelemetry import propagate

from server.model_metadata import TrainingModelMetadata
from server.session_registry import SessionRegistry
from server.store import RequestStore
from server.worker_manager import WorkerManager, owner_of
from training import commands


class ApiRuntime:
  def __init__(self, store: RequestStore, worker_manager: WorkerManager | None, tmp_dir: str):
    self.store = store
    self.worker_manager = worker_manager
    self.sessions = SessionRegistry(store)
    self.checkpoint_root = os.path.join(tmp_dir, "checkpoints")
    # Serialize attachment and teardown of an owner within this API process.
    self.owner_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    self.tasks: list[asyncio.Task[Any]] = []

  async def submit(self, command: commands.TrainingCommand) -> str:
    """Start trainers for create commands, then route the work by request ID."""
    request_id = command.request_id
    try:
      if self.worker_manager is not None and isinstance(command, commands.CreateModel | commands.CreateModelFromState):
        await asyncio.to_thread(self.worker_manager.ensure, command.model_id, "trainer")
      meta = await self.store.get_model_metadata(command.model_id)
      active_set_id = f"{meta['base_model']}-1" if meta and meta.get("fine_tuning_type") == "lora" and meta.get("base_model") else None
      carrier: dict[str, str] = {}
      propagate.inject(carrier)
      await self.store.put_request(commands.wire(command.model_copy(update={"trace_context": carrier})), active_set_id=active_set_id)
    except Exception as exc:
      traceback.print_exc()
      await self.store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
      return request_id
    print(f"[API_SERVER] enqueued op={command.op} request_id={request_id} model_id={command.model_id} active_set={active_set_id}")
    return request_id

  async def persist_model_metadata(self, metadata: TrainingModelMetadata) -> str:
    model_id = str(uuid.uuid4())
    await self.store.set_value(f"open_rl:model_meta:{model_id}", json.dumps(metadata.to_dict()))
    return model_id

  async def bind_session(self, session_id: str | None, model_id: str) -> None:
    if self.worker_manager is not None and session_id:
      owner = await asyncio.to_thread(owner_of, model_id)
      async with self.owner_locks[owner]:
        await self.sessions.attach(session_id, owner)

  async def reap_owner(self, owner: str) -> None:
    if self.worker_manager is None:
      return
    async with self.owner_locks[owner]:
      if await self.sessions.in_use(owner):
        return
      print(f"[API_SERVER] No live session uses {owner}; tearing its workers down")
      for model in await asyncio.to_thread(self.worker_manager.release_owner, owner):
        await self.store.delete_values(f"open_rl:sampler_ready:{model}")
      await self.sessions.forget(owner)

  async def reap_dead_sessions(self) -> None:
    while True:
      await asyncio.sleep(30)
      try:
        owners = await self.sessions.owners()
      except Exception:
        traceback.print_exc()
        continue
      for owner in owners:
        try:
          await self.reap_owner(owner)
        except Exception:
          traceback.print_exc()

  def start_reaper(self) -> None:
    if self.worker_manager is not None:
      self.tasks.append(asyncio.create_task(self.reap_dead_sessions()))

  async def start_local_training(self, base_model: str | None) -> None:
    from server import training_requests_processor

    worker = training_requests_processor.LoraTrainingWorker()
    if base_model:
      await asyncio.to_thread(worker.load_base_model, base_model)
    self.tasks.append(asyncio.create_task(training_requests_processor.run_training_requests_processor(worker)))

  async def close(self) -> None:
    for task in self.tasks:
      task.cancel()
    try:
      await asyncio.gather(*self.tasks, return_exceptions=True)
    finally:
      self.tasks.clear()
      if self.worker_manager is not None:
        self.worker_manager.close()
        self.worker_manager = None
