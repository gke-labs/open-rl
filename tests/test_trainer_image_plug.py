import asyncio
import os
import tempfile
import unittest
from collections import Counter
from typing import Any
from unittest.mock import patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor, build_worker
from training import commands
from training.types import Datum


class CountingTrainer:
  """The smallest trainer the processor can drive: it counts calls and
  returns well-formed results without a model."""

  def __init__(self) -> None:
    self.calls: Counter = Counter()

  def load_base_model(self, base_model: str) -> None:
    self.calls["load_base_model"] += 1

  def create_model(self, base_model: str, model_id: str, config: Any) -> None:
    self.calls["create_model"] += 1

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: Any = None, model_id: Any = None, forward_only: bool = False) -> dict:
    self.calls["forward_backward"] += 1
    outputs = [{"logprobs": {"data": [0.0] * len(d.model_input), "dtype": "float32", "shape": [len(d.model_input)]}} for d in data]
    return {"loss_fn_outputs": outputs, "metrics": {"loss:mean": 0.0}}

  def optim_step(self, adam_params: dict, model_id: str) -> dict:
    self.calls["optim_step"] += 1
    return {"metrics": {"steps": float(self.calls["optim_step"])}}

  def save_for_sampler(self, model_id: str, alias: Any, ref: Any) -> None:
    self.calls["save_for_sampler"] += 1

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict:
    self.calls["save_state"] += 1
    return {"path": state_path}

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict:
    self.calls["load_from_state"] += 1
    return {}

  def delete_model(self, model_id: str) -> None:
    self.calls["delete_model"] += 1


class TrainerImagePlugTest(unittest.TestCase):
  """A trainer image names its class in OPEN_RL_TRAINER_BACKEND as <module>:<Class>;
  the request processor then drives it like any built-in trainer."""

  def test_the_processor_drives_the_named_trainer(self) -> None:
    tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": "tests.test_trainer_image_plug:CountingTrainer"}))
    worker = build_worker(is_lora=True)
    self.assertIsInstance(worker, CountingTrainer)
    processor = TrainingRequestsProcessor(InMemoryStore(), worker)

    datum = Datum(model_input=[1, 2, 3], loss_fn_inputs={"target_tokens": {"data": [2, 3, 4]}, "weights": {"data": [1.0, 1.0, 1.0]}})
    loop = [
      commands.CreateModel(request_id="c", model_id="job", base_model="Qwen/Qwen3-0.6B"),
      *[commands.ForwardBackward(request_id=f"f{i}", model_id="job", data=[datum]) for i in range(2)],
      commands.OptimStep(request_id="o", model_id="job"),
      commands.SaveState(request_id="s", model_id="job", state_path=os.path.join(tmp, "state")),
      commands.DeleteModel(request_id="d", model_id="job"),
    ]
    results = dict(asyncio.run(processor.handle_request(commands.wire(c))) for c in loop)

    self.assertEqual(results["f0"]["loss_fn_outputs"][0]["logprobs"]["shape"], [3])
    self.assertEqual(results["o"]["metrics"], {"steps": 1.0})
    self.assertEqual(results["s"]["path"], os.path.join(tmp, "state"))
    self.assertEqual(dict(worker.calls), {"create_model": 1, "forward_backward": 2, "optim_step": 1, "save_state": 1, "delete_model": 1})

  def test_without_a_class_the_built_in_trainers_are_used(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": ""}):
      self.assertEqual(type(build_worker(is_lora=True)).__name__, "LoraTrainingWorker")


if __name__ == "__main__":
  unittest.main()
