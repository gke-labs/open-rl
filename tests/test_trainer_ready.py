"""The trainer announces readiness to the API server the way the sampler
does: a TTL key in the state store, set once a model is loaded, refreshed
while the loop runs, deleted on exit. The model view turns it into `ready`."""

import asyncio
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStateStore, InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor


class Worker:
  def __init__(self):
    self.loaded = False

  def is_ready(self) -> bool:
    return self.loaded


class TrainerReadyTest(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://test"}))
    self.enterContext(patch("server.training_requests_processor.is_primary", return_value=True))
    self.worker = Worker()
    self.state = InMemoryStateStore()
    self.proc = TrainingRequestsProcessor(InMemoryStore(), self.worker, "google/gemma-4-e2b")
    self.proc.state = self.state

  def key(self):
    return asyncio.run(self.state.get_value("open_rl:trainer_ready:google/gemma-4-e2b"))

  def test_announces_only_once_a_model_is_loaded_and_withdraws_on_exit(self):
    asyncio.run(self.proc.announce_readiness())
    self.assertIsNone(self.key(), "no model yet: the view must keep saying starting")
    self.worker.loaded = True
    asyncio.run(self.proc.announce_readiness())
    self.assertEqual(self.key(), "1")
    asyncio.run(self.proc.withdraw_readiness())
    self.assertIsNone(self.key())

  def test_refreshes_at_most_once_a_minute(self):
    self.worker.loaded = True
    asyncio.run(self.proc.announce_readiness())
    first = self.proc.ready_announced_at
    asyncio.run(self.proc.announce_readiness())
    self.assertEqual(self.proc.ready_announced_at, first)

  def test_only_the_primary_rank_announces(self):
    self.worker.loaded = True
    with patch("server.training_requests_processor.is_primary", return_value=False):
      asyncio.run(self.proc.announce_readiness())
    self.assertIsNone(self.key())


if __name__ == "__main__":
  unittest.main()
