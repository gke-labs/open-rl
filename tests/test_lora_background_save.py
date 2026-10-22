import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from server import training_requests_processor as trp
from server.store import InMemoryStore
from training import commands
from training import lora_trainer_worker as lora_module
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


def adapter_files(directory: str) -> tuple[dict[str, torch.Tensor], dict]:
  tensors = load_file(os.path.join(directory, "adapter_model.safetensors"))
  with open(os.path.join(directory, "adapter_config.json")) as f:
    return tensors, json.load(f)


class LoraBackgroundSaveTest(unittest.IsolatedAsyncioTestCase):
  """A sampler save is copied between requests, written in the background, and answered once on disk."""

  async def asyncSetUp(self) -> None:
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp}))
    self.worker = tiny_lora_worker()
    self.processor = trp.TrainingRequestsProcessor(InMemoryStore(), self.worker)
    for job in ("job-one", "job-two"):
      await self.run_steps(commands.CreateModel(request_id=f"create-{job}", model_id=job, base_model="tiny", lora_config=LoraConfig(rank=2, seed=1)))
      await self.run_steps(*self.train(job, 0))

  async def run_steps(self, *steps: commands.Command) -> None:
    for step in steps:
      await self.processor.process_request(commands.wire(step))
      result = await self.processor.store.get_result(step.request_id, timeout=5)
      self.assertNotIn(result.get("type"), ("RequestFailedResponse", "try_again"), f"{step.request_id}: {result}")

  def train(self, model_id: str, step: int) -> list[commands.Command]:
    row = [3, 5, 8, 13, 21, 34]
    datum = Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * 5)})
    return [
      commands.ForwardBackward(request_id=f"fb-{model_id}-{step}", model_id=model_id, data=[datum]),
      commands.OptimStep(request_id=f"optim-{model_id}-{step}", model_id=model_id, adam_params={"learning_rate": 1e-2}),
    ]

  async def test_writes_what_save_pretrained_writes(self) -> None:
    await self.run_steps(commands.SaveWeightsForSampler(request_id="save", model_id="job-one", alias="step-1"))

    with tempfile.TemporaryDirectory() as reference:
      self.worker.peft_model.save_pretrained(reference, selected_adapters=["job-one"])
      expected_tensors, expected_config = adapter_files(os.path.join(reference, "job-one"))
    tensors, config = adapter_files(os.path.join(self.tmp, "peft", "job-one", "job-one"))

    self.assertEqual(sorted(tensors), sorted(expected_tensors))
    for name, tensor in expected_tensors.items():
      self.assertTrue(torch.equal(tensors[name], tensor), name)
    self.assertEqual(config, expected_config)
    with open(os.path.join(self.tmp, "peft", "job-one", "metadata.json")) as f:
      self.assertEqual(json.load(f)["alias"], "step-1")

  async def test_the_answer_waits_for_the_write_while_the_next_request_runs(self) -> None:
    release = threading.Event()
    write = lora_module.write_adapter

    def slow_write(*args, **kwargs):
      release.wait(timeout=10)
      write(*args, **kwargs)

    with patch.object(lora_module, "write_adapter", slow_write):
      before = {name: tensor.clone() for name, tensor in self.worker.peft_model.named_parameters() if ".job-one." in name}
      await self.processor.process_request(commands.wire(commands.SaveWeightsForSampler(request_id="save", model_id="job-one")))

      # Unanswered while the write is blocked; the other job's training goes ahead.
      self.assertEqual((await self.processor.store.get_result("save", timeout=0.2))["type"], "try_again")
      await self.run_steps(*self.train("job-two", 1), *self.train("job-one", 1))

      release.set()
      self.assertEqual((await self.processor.store.get_result("save", timeout=5))["type"], "sampler_weights_saved")

    # The file holds the adapter as it was when the save was requested, not after the later step.
    tensors, _ = adapter_files(os.path.join(self.tmp, "peft", "job-one", "job-one"))
    lora_b = next(name for name in before if "lora_B" in name)
    saved = next(tensor for name, tensor in tensors.items() if name == lora_b.replace(".job-one", ""))
    self.assertTrue(torch.equal(saved, before[lora_b]))
    self.assertFalse(torch.equal(saved, dict(self.worker.peft_model.named_parameters())[lora_b].detach()))


if __name__ == "__main__":
  unittest.main()
