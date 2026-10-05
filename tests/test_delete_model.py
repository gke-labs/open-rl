import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server import training_requests_processor as trp
from server.store import InMemoryStore
from training import commands
from training.types import Datum, LoraConfig, TensorData


def tiny_lora_worker() -> trp.LoraTrainingWorker:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
  )
  worker = trp.LoraTrainingWorker()
  worker.device = torch.device("cpu")
  # Already loaded, so create_model skips the hub download.
  worker.base_model = LlamaForCausalLM(config)
  worker.base_model_name = "tiny"
  return worker


class DeleteModelTest(unittest.IsolatedAsyncioTestCase):
  """Deleting one job on a shared LoRA trainer frees its adapter and leaves the others training."""

  async def run_commands(self, processor: trp.TrainingRequestsProcessor, *steps: commands.Command) -> list[dict]:
    results = []
    for step in steps:
      await processor.process_request(commands.wire(step))
      result = await processor.store.get_future(step.request_id, timeout=5)
      self.assertNotEqual(result.get("type"), "RequestFailedResponse", f"{step.request_id}: {result}")
      results.append(result)
    return results

  def train(self, model_id: str, step: int) -> list[commands.Command]:
    row = [3, 5, 8, 13, 21, 34]
    datum = Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * 5)})
    return [
      commands.ForwardBackward(request_id=f"fb-{model_id}-{step}", model_id=model_id, data=[datum]),
      commands.OptimStep(request_id=f"optim-{model_id}-{step}", model_id=model_id, adam_params={"learning_rate": 1e-2}),
    ]

  def create(self, model_id: str) -> commands.Command:
    return commands.CreateModel(request_id=f"create-{model_id}", model_id=model_id, base_model="tiny", lora_config=LoraConfig(rank=2, seed=1))

  async def test_the_other_jobs_keep_training(self) -> None:
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
      worker = tiny_lora_worker()
      processor = trp.TrainingRequestsProcessor(InMemoryStore(), worker)
      await self.run_commands(processor, self.create("a"), self.create("b"), *self.train("b", 0), *self.train("a", 0))

      # a is the active adapter when it goes.
      (deleted,) = await self.run_commands(processor, commands.DeleteModel(request_id="delete-a", model_id="a"))
      self.assertEqual(deleted["type"], "model_deleted")
      self.assertNotIn("a", worker.adapter_states)
      self.assertNotIn("a", worker.peft_model.peft_config)
      self.assertFalse(any(".a." in name for name, _ in worker.peft_model.named_parameters()))

      # b still trains, the last adapter can go, and a new job starts on the same model.
      await self.run_commands(processor, *self.train("b", 1), commands.DeleteModel(request_id="delete-b", model_id="b"))
      self.assertEqual(worker.adapter_states, {})
      await self.run_commands(processor, self.create("c"), *self.train("c", 0))

  async def test_deleting_an_unknown_job_is_a_no_op(self) -> None:
    worker = tiny_lora_worker()
    processor = trp.TrainingRequestsProcessor(InMemoryStore(), worker)
    await self.run_commands(processor, commands.DeleteModel(request_id="delete", model_id="never-created"))


if __name__ == "__main__":
  unittest.main()
