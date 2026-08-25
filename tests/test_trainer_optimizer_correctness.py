import asyncio
import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import torch


def _load_trainer_modules():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  with patch.dict(sys.modules, stubs):
    for module_name in list(sys.modules):
      if module_name == "training" or module_name.startswith("training."):
        del sys.modules[module_name]
    from training import fft_trainer_worker, lora_trainer_worker, losses, trainer_worker

  return trainer_worker, lora_trainer_worker, fft_trainer_worker, losses


def _load_training_requests_processor_module():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  env = {
    "OPEN_RL_ENABLE_FFT": "true",
    "REDIS_URL": "redis://localhost:6379",
  }
  with patch.dict(sys.modules, stubs), patch.dict(os.environ, env):
    for module_name in list(sys.modules):
      if module_name == "server.training_requests_processor":
        del sys.modules[module_name]
    training_requests_processor = importlib.import_module("server.training_requests_processor")
  return training_requests_processor


trainer_worker_module, lora_trainer_worker_module, fft_trainer_worker_module, losses_module = _load_trainer_modules()
training_requests_processor_module = _load_training_requests_processor_module()
BaseTrainerWorker = trainer_worker_module.BaseTrainerWorker
FFTTrainingWorker = fft_trainer_worker_module.FFTTrainingWorker
LoraTrainingWorker = lora_trainer_worker_module.LoraTrainingWorker


class _PeftModelStub:
  def __init__(self, adapter_params):
    self.adapter_params = adapter_params
    self.active_adapter = None

  def set_adapter(self, adapter_id):
    self.active_adapter = adapter_id
    for param in self.parameters():
      param.requires_grad_(False)
    for param in self.adapter_params[adapter_id]:
      param.requires_grad_(True)

  def parameters(self):
    for params in self.adapter_params.values():
      yield from params

  def save_pretrained(self, *_args, **_kwargs):
    return None


class _TokenizerStub:
  pad_token_id = 0


class _LogitModelStub:
  def __init__(self, vocab_size: int = 17):
    self.vocab_size = vocab_size
    self.calls = []

  def train(self):
    return None

  def __call__(self, input_tensor, attention_mask=None, **_kwargs):
    if attention_mask is not None:
      self.calls.append((input_tensor.detach().clone(), attention_mask.detach().clone()))
    vocab = torch.arange(self.vocab_size, dtype=torch.float32, device=input_tensor.device).view(1, 1, -1)
    positions = torch.arange(input_tensor.shape[1], dtype=torch.float32, device=input_tensor.device).view(1, -1, 1)
    logits = torch.cos(input_tensor.float().unsqueeze(-1) * 0.11 + positions * 0.07 + vocab * 0.13)
    logits.requires_grad_()
    return types.SimpleNamespace(logits=logits)


class _FullModelStub:
  def __init__(self, params):
    self.params = params

  def train(self):
    return None

  def parameters(self):
    yield from self.params


class _RecordingFullWorker(training_requests_processor_module.FFTTrainingWorker):
  def __init__(self):
    self.base_model_name = None
    self.loaded_base_models = []
    self.created_models = []
    self.saved_states = []

  def load_base_model(self, base_model_name):
    self.base_model_name = base_model_name
    self.loaded_base_models.append(base_model_name)

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))

  def forward_backward(self, data, loss_fn, loss_config=None, model_id=None):
    return {"model_id": model_id, "loss_fn": loss_fn, "loss_config": loss_config, "data": data}

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"):
    self.saved_states.append((model_id, state_path, include_optimizer, kind))
    return {"path": state_path}


class _RecordingLoraWorker(training_requests_processor_module.LoraTrainingWorker):
  def __init__(self):
    self.loaded_base_models = []
    self.created_models = []

  def load_base_model(self, base_model_name):
    self.loaded_base_models.append(base_model_name)

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))


class _FutureStoreStub:
  def __init__(self):
    self.results = {}

  async def set_future(self, req_id, result):
    self.results[req_id] = result


class _TrainingRequestsStoreStub(_FutureStoreStub):
  def __init__(self, batches):
    super().__init__()
    self.batches = list(batches)
    self.queried_model_ids = []

  async def get_requests_for_model(self, model_id):
    self.queried_model_ids.append(model_id)
    if self.batches:
      return self.batches.pop(0)
    raise asyncio.CancelledError()


class _SnapshotClientStub:
  def __init__(self):
    self.events = []

  async def register(self, pid):
    self.events.append(("register", pid))
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, pid):
    self.events.append(("acquire", pid))
    try:
      yield
    finally:
      self.events.append(("release", pid))

  async def unregister(self, pid):
    self.events.append(("unregister", pid))
    return {"ok": True}

  async def close(self):
    self.events.append(("close",))


def _datum(model_input, target_tokens, *, weights=None, logprobs=None, advantages=None):
  loss_fn_inputs = {"target_tokens": trainer_worker_module.TensorData(data=target_tokens)}
  if weights is not None:
    loss_fn_inputs["weights"] = trainer_worker_module.TensorData(data=weights)
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = trainer_worker_module.TensorData(data=logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = trainer_worker_module.TensorData(data=advantages)
  return trainer_worker_module.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


class TestTrainerOptimizerCorrectness(unittest.TestCase):
  def test_lora_create_model_loads_base_then_creates_adapter(self) -> None:
    worker = LoraTrainingWorker()
    config = lora_trainer_worker_module.LoraConfig(rank=2, seed=123)
    calls = []

    worker.load_base_model = lambda base_model_name: calls.append(("load", base_model_name))
    worker.create_adapter = lambda model_id, adapter_config: calls.append(("adapter", model_id, adapter_config))

    worker.create_model("base-model", "adapter-a", config)

    self.assertEqual(calls[0], ("load", "base-model"))
    self.assertEqual(calls[1][0], "adapter")
    self.assertEqual(calls[1][1], "adapter-a")
    self.assertIs(calls[1][2], config)

  def test_save_adapter_selects_adapter_it_saves(self) -> None:
    adapter_a_param = torch.nn.Parameter(torch.tensor([1.0]))
    adapter_b_param = torch.nn.Parameter(torch.tensor([1.0]))
    worker = LoraTrainingWorker()
    worker.peft_model = _PeftModelStub(
      {
        "adapter-a": [adapter_a_param],
        "adapter-b": [adapter_b_param],
      }
    )
    worker.peft_model.set_adapter("adapter-b")

    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp_dir}):
      worker.save_adapter("adapter-a")
      self.assertTrue(os.path.exists(os.path.join(tmp_dir, "peft", "adapter-a", "metadata.json")))

    self.assertEqual(worker.peft_model.active_adapter, "adapter-a")

  def test_fft_create_model_loads_base_then_prepares_model(self) -> None:
    worker = FFTTrainingWorker()
    config = fft_trainer_worker_module.FFTConfig(seed=123)
    calls = []

    worker.load_base_model = lambda base_model_name: calls.append(("load", base_model_name))
    worker.prepare_model_for_training = lambda: calls.append(("prepare", None))

    worker.create_model("base-model", "model-a", config)

    self.assertEqual(calls, [("load", "base-model"), ("prepare", None)])

  def test_optim_step_only_updates_active_adapter_params(self) -> None:
    active_param = torch.nn.Parameter(torch.tensor([1.0]))
    other_param = torch.nn.Parameter(torch.tensor([1.0]))
    active_param.grad = torch.tensor([1.0])
    other_param.grad = torch.tensor([10.0])

    worker = LoraTrainingWorker()
    worker.peft_model = _PeftModelStub(
      {
        "adapter-a": [active_param],
        "adapter-b": [other_param],
      }
    )
    worker.adapter_states = {
      "adapter-a": {"trainable_params": lora_trainer_worker_module.active_adapter_parameters(worker.peft_model, "adapter-a"), "optimizer": None}
    }
    worker.save_adapter = lambda *_args, **_kwargs: None

    result = worker.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      },
      "adapter-a",
    )

    self.assertEqual(worker.peft_model.active_adapter, "adapter-a")
    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(active_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(other_param.detach(), torch.tensor([1.0])))
    if active_param.grad is not None:
      self.assertTrue(torch.allclose(active_param.grad, torch.zeros_like(active_param.grad)))
    self.assertIsNotNone(other_param.grad)

  def test_fft_optim_step_updates_full_model_trainable_params(self) -> None:
    trainable_param = torch.nn.Parameter(torch.tensor([1.0]))
    frozen_param = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    trainable_param.grad = torch.tensor([1.0])
    frozen_param.grad = torch.tensor([10.0])

    worker = FFTTrainingWorker()
    worker.model = _FullModelStub([trainable_param, frozen_param])
    worker.trainable_params = fft_trainer_worker_module.trainable_model_parameters(worker.model)

    result = worker.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      }
    )

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(trainable_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(frozen_param.detach(), torch.tensor([1.0])))
    if trainable_param.grad is not None:
      self.assertTrue(torch.allclose(trainable_param.grad, torch.zeros_like(trainable_param.grad)))
    self.assertIsNotNone(frozen_param.grad)


class TestTrainingRequestsProcessorFullMode(unittest.IsolatedAsyncioTestCase):
  async def test_importing_training_requests_processor_does_not_create_worker(self) -> None:
    self.assertFalse(hasattr(training_requests_processor_module, "worker"))

  async def test_lora_processor_create_model_uses_worker_create_model(self) -> None:
    worker = _RecordingLoraWorker()
    store = _FutureStoreStub()
    processor = training_requests_processor_module.LoraTrainingRequestsProcessor(store, worker)

    await processor.process_request(
      {
        "request_id": "req-a",
        "model_id": "adapter-a",
        "op": "create_model",
        "payload": {
          "base_model": "base-model",
          "lora_config": {"seed": 123, "rank": 2},
        },
      },
      "adapter-a",
    )

    self.assertEqual(worker.loaded_base_models, [])
    base_model, model_id, config = worker.created_models[0]
    self.assertEqual(base_model, "base-model")
    self.assertEqual(model_id, "adapter-a")
    self.assertEqual(config.seed, 123)
    self.assertEqual(config.rank, 2)
    result = store.results["req-a"]
    self.assertEqual(result["model_id"], "adapter-a")
    self.assertEqual(result["rank"], 2)
    self.assertEqual(result["training_kind"], "lora")
    self.assertEqual(result["type"], "model_created")

  def test_parse_datum_flattens_chunked_model_input(self) -> None:
    datum = training_requests_processor_module.parse_datum(
      {
        "model_input": {"chunks": [{"tokens": [1, 2]}, {"tokens": [3]}]},
        "loss_fn_inputs": {
          "target_tokens": [2, 3, 4],
          "weights": {"data": [1.0, 0.5, 0.25]},
        },
      }
    )

    self.assertEqual(datum.model_input, [1, 2, 3])
    self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [2, 3, 4])
    self.assertEqual(datum.loss_fn_inputs["weights"].data, [1.0, 0.5, 0.25])

  async def test_full_processor_create_model_uses_model_worker(self) -> None:
    worker = _RecordingFullWorker()
    store = _FutureStoreStub()
    snapshot_client = _SnapshotClientStub()

    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
      processor = training_requests_processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", snapshot_client=snapshot_client)
      await processor.process_request(
        {
          "request_id": "req-a",
          "model_id": "model-a",
          "op": "create_model",
          "payload": {
            "base_model": "base-model",
            "full_config": {"seed": 123, "rank": 8},
          },
        },
        "model-a",
      )

    self.assertEqual(worker.loaded_base_models, [])
    base_model, model_id, config = worker.created_models[0]
    self.assertEqual(base_model, "base-model")
    self.assertEqual(model_id, "model-a")
    self.assertEqual(config.seed, 123)
    result = store.results["req-a"]
    self.assertEqual(result["model_id"], "model-a")
    self.assertEqual(result["base_model"], "base-model")
    self.assertEqual(result["training_kind"], "full")
    self.assertEqual(result["type"], "model_created")

  async def test_full_processor_saves_sampler_checkpoint_as_full_state(self) -> None:
    worker = _RecordingFullWorker()
    store = _FutureStoreStub()
    snapshot_client = _SnapshotClientStub()

    with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": "/tmp/open-rl-test", "REDIS_URL": "redis://localhost:6379"}):
      processor = training_requests_processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", snapshot_client=snapshot_client)
      await processor.process_request(
        {
          "request_id": "req-a",
          "model_id": "model-a",
          "op": "save_weights_for_sampler",
          "payload": {
            "path": "tinker://model-a/sampler_weights/final",
            "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7",
          },
        },
        "model-a",
      )

    self.assertEqual(
      worker.saved_states,
      [("model-a", "/tmp/open-rl-test/sampler_full/model-a/sampler_weights/final", False, "sampler")],
    )
    self.assertEqual(
      store.results["req-a"],
      {
        "path": "tinker://model-a/sampler_weights/final",
        "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7",
        "type": "sampler_weights_saved",
      },
    )

  async def test_full_processor_requires_redis(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await training_requests_processor_module.run_training_requests_processor(_RecordingFullWorker(), "model-a")

  async def test_full_processor_uses_default_snapshot_socket(self) -> None:
    store = _TrainingRequestsStoreStub([])
    snapshot_client = _SnapshotClientStub()

    with (
      patch.dict(
        os.environ,
        {
          "OPEN_RL_ENABLE_FFT": "true",
          "REDIS_URL": "redis://localhost:6379",
        },
        clear=True,
      ),
      patch.object(training_requests_processor_module, "get_store", return_value=store),
      patch.object(training_requests_processor_module, "create_snapshot_agent_client", return_value=snapshot_client) as create_snapshot_agent_client,
    ):
      await training_requests_processor_module.run_training_requests_processor(_RecordingFullWorker(), "model-a")

    create_snapshot_agent_client.assert_called_once_with("/tmp/open-rl/snapshot-agent.sock")
    self.assertEqual([event[0] for event in snapshot_client.events], ["register", "unregister", "close"])

  async def test_full_processor_uses_injected_snapshot_client(self) -> None:
    worker = _RecordingFullWorker()
    store = _TrainingRequestsStoreStub(
      [
        [
          {
            "request_id": "req-a",
            "model_id": "model-a",
            "op": "create_model",
            "payload": {
              "base_model": "base-model",
              "full_config": {"seed": 123},
            },
          }
        ]
      ]
    )
    snapshot_client = _SnapshotClientStub()

    with (
      patch.dict(
        os.environ,
        {
          "OPEN_RL_ENABLE_FFT": "true",
          "REDIS_URL": "redis://localhost:6379",
        },
      ),
      patch.object(training_requests_processor_module, "get_store", return_value=store),
    ):
      await training_requests_processor_module.run_training_requests_processor(worker, "model-a", snapshot_client=snapshot_client)

    self.assertEqual(store.queried_model_ids, ["model-a", "model-a"])
    self.assertEqual([event[0] for event in snapshot_client.events], ["register", "acquire", "release", "unregister", "close"])
    for event in snapshot_client.events:
      if len(event) == 2:
        self.assertEqual(event[1], os.getpid())
    self.assertEqual(worker.created_models[0][0], "base-model")
    self.assertEqual(store.results["req-a"]["model_id"], "model-a")


class TestTrainerPaddedBatchingMath(unittest.TestCase):
  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

  def _data(self):
    return [
      _datum(
        [3, 4, 5, 6],
        [1, 2, 3, 4],
        weights=[1.0, 0.5, 0.25, 2.0],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        advantages=[1.0, -0.5, 2.0, 0.25],
      ),
      _datum(
        [7, 8],
        [2, 3],
        logprobs=[-0.7, -0.8],
        advantages=[0.75, 1.25],
      ),
      _datum(
        [9, 10, 11],
        [5, 6, 7, 8],
        weights=[0.2, 0.4, 0.6, 0.8],
        logprobs=[-0.9, -1.0, -1.1, -1.2],
        advantages=[-1.0, 0.3, 0.9, 1.7],
      ),
    ]

  def training_tensors(self, worker, model, data):
    input_ids, attention_mask, input_lengths = worker.pad_model_inputs(data)
    target_token_ids, weights, lengths = worker.pad_targets_and_weights(data, input_lengths)
    logprobs = worker.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)
    old_logprobs = worker.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in data], lengths, torch.float32)
    advantages = worker.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages, batch_lengths = self.training_tensors(worker, model, data)
    single_results = [self.training_tensors(worker, model, [datum]) for datum in data]

    for row, (single_logprobs, single_weights, single_old_logprobs, single_advantages, single_lengths) in enumerate(single_results):
      length = batch_lengths[row]
      self.assertEqual(length, single_lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :length], single_logprobs[0, :length])
      torch.testing.assert_close(batch_weights[row, :length], single_weights[0, :length])
      torch.testing.assert_close(batch_weights[row, length:], torch.zeros_like(batch_weights[row, length:]))
      torch.testing.assert_close(batch_old_logprobs[row, :length], single_old_logprobs[0, :length])
      torch.testing.assert_close(batch_old_logprobs[row, length:], torch.zeros_like(batch_old_logprobs[row, length:]))
      torch.testing.assert_close(batch_advantages[row, :length], single_advantages[0, :length])
      torch.testing.assert_close(batch_advantages[row, length:], torch.zeros_like(batch_advantages[row, length:]))

    def single_sum(fn):
      losses = [fn(logprobs, weights, old_logprobs, advantages).sum() for logprobs, weights, old_logprobs, advantages, _lengths in single_results]
      return torch.stack(losses).sum()

    torch.testing.assert_close(
      losses_module.cross_entropy_loss(batch_logprobs, batch_weights).sum(),
      single_sum(lambda logprobs, weights, _old_logprobs, _advantages: losses_module.cross_entropy_loss(logprobs, weights)),
    )
    torch.testing.assert_close(
      losses_module.importance_sampling_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.importance_sampling_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
        )
      ),
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses_module.ppo_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
        ppo_config,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.ppo_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
          ppo_config,
        )
      ),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    worker = self._worker()
    data = self._data()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = worker.make_training_batches(data)

    seen = [idx for batch in batches for idx, _datum in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(datum.model_input) for _idx, datum in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_padded_batches_preserve_client_output_shape(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = worker.forward_backward(model, data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(model.calls), 0)
    self.assertTrue(any(call[0].shape[0] > 1 for call in model.calls))
    for datum, output in zip(data, result["loss_fn_outputs"], strict=True):
      logprobs = output["logprobs"]
      self.assertEqual(logprobs["shape"], [min(len(datum.model_input), len(datum.loss_fn_inputs["target_tokens"].data))])

  def test_fft_forward_backward_uses_single_process_model(self) -> None:
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    worker.model = _LogitModelStub()
    data = self._data()

    result = worker.forward_backward(data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(worker.model.calls), 0)


if __name__ == "__main__":
  unittest.main()
