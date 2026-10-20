"""Precision is declared by the client, checked against what each trainer and the
sampler support, carried to the workers, and never silently changed."""

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from server import api_server
from server.estimator import footprint
from server.model_metadata import TrainingModelMetadata
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import worker_env
from tests.test_session_lifecycle import RuntimeManager
from training.trainer_worker import BaseTrainerWorker
from training.types import SUPPORTED_DTYPES, check_dtypes, deployment_dtype

HAS_VLLM = importlib.util.find_spec("vllm") is not None


def dtypes(trainer: str | None = "bfloat16", sampler: str | None = "bfloat16") -> dict[str, str]:
  out = {}
  if trainer is not None:
    out["openrl.trainer_dtype"] = trainer
  if sampler is not None:
    out["openrl.sampler_dtype"] = sampler
  return out


class SupportMatrixTest(unittest.TestCase):
  def test_supported_pairs_pass(self) -> None:
    for (backend, kind), supported in SUPPORTED_DTYPES.items():
      for dtype in supported:
        with self.subTest(backend=backend, kind=kind, dtype=dtype):
          check_dtypes(backend, kind, dtype, dtype)

  def test_both_are_required(self) -> None:
    for trainer, sampler, missing in ((None, "bfloat16", "openrl.trainer_dtype"), ("bfloat16", None, "openrl.sampler_dtype")):
      with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, f"{missing} must be set"):
        check_dtypes("pytorch", "full", trainer, sampler)

  def test_cross_precision_full_is_supported_for_vllm(self) -> None:
    check_dtypes("pytorch", "full", "float32", "bfloat16")
    check_dtypes("pytorch", "full", "bfloat16", "float32")
    with self.assertRaisesRegex(ValueError, "in-process torch sampler shares the trainer model"):
      check_dtypes("pytorch", "full", "float32", "bfloat16", vllm_sampler=False)

  def test_fp32_lora_is_refused_for_vllm(self) -> None:
    with self.assertRaisesRegex(ValueError, "vLLM's LoRA kernels support fp16/bf16 only"):
      check_dtypes("pytorch", "lora", "float32", "float32")

  def test_fp32_lora_is_fine_with_the_torch_sampler(self) -> None:
    check_dtypes("pytorch", "lora", "float32", "float32", vllm_sampler=False)
    with self.assertRaisesRegex(ValueError, "automodel trainer trains in bfloat16 only"):
      check_dtypes("automodel", "lora", "float32", "float32", vllm_sampler=False)

  def test_automodel_is_bf16_only(self) -> None:
    with self.assertRaisesRegex(ValueError, "automodel trainer trains in bfloat16 only"):
      check_dtypes("automodel", "lora", "float32", "float32")

  def test_a_trainer_image_owns_its_check_but_not_the_samplers(self) -> None:
    check_dtypes("ghcr.io/org/trainer:1", "full", "float32", "float32")
    with self.assertRaisesRegex(ValueError, "vLLM's LoRA kernels"):
      check_dtypes("ghcr.io/org/trainer:1", "lora", "float32", "float32")

  def test_deployment_dtype(self) -> None:
    self.assertEqual(deployment_dtype({}), "bfloat16")
    self.assertEqual(deployment_dtype({"OPEN_RL_DTYPE": "float32"}), "float32")
    with self.assertRaisesRegex(ValueError, "OPEN_RL_DTYPE"):
      deployment_dtype({"OPEN_RL_DTYPE": "float16"})


class ApiPrecisionTest(unittest.TestCase):
  """Through the ASGI app, with a worker manager, so per-model dtypes are allowed."""

  def setUp(self) -> None:
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=api_server.state))
    self.enterContext(patch.object(api_server, "worker_manager", RuntimeManager()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}))
    self.client = TestClient(api_server.app)

  def post(self, path: str, body: dict):
    return self.client.post(f"/api/v1/{path}", json=body)

  def create(self, **user_metadata):
    return self.post("create_model", {"base_model": "m", "user_metadata": user_metadata})

  def metadata(self, model_id: str) -> dict:
    return json.loads(api_server.state.get_value_sync(f"open_rl:model_meta:{model_id}"))

  def queued(self) -> list[dict]:
    return [request["payload"] for request in asyncio.run(api_server.store.get_requests())]

  def test_missing_dtypes_are_a_400(self) -> None:
    response = self.create()
    self.assertEqual(response.status_code, 400)
    self.assertIn("openrl.trainer_dtype and openrl.sampler_dtype must be set", response.json()["error"])

  def test_cross_precision_full_is_accepted_and_stored(self) -> None:
    response = self.create(**{"openrl.fine_tuning_type": "full"}, **dtypes("float32", "bfloat16"))
    self.assertEqual(response.status_code, 200, response.text)
    meta = self.metadata(response.json()["request_id"])
    self.assertEqual((meta["trainer_dtype"], meta["sampler_dtype"]), ("float32", "bfloat16"))
    self.assertEqual(self.queued()[0]["dtype"], "float32")

  def test_unknown_dtype_is_a_400(self) -> None:
    response = self.create(**dtypes("float16", "float16"))
    self.assertEqual(response.status_code, 400)
    self.assertIn("openrl.trainer_dtype", response.json()["error"])

  def test_fp32_lora_is_a_400(self) -> None:
    response = self.create(**dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 400)
    self.assertIn("vLLM's LoRA kernels", response.json()["error"])

  def test_fp32_automodel_is_a_400(self) -> None:
    response = self.create(**{"openrl.trainer_backend": "automodel"}, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 400)
    self.assertIn("bfloat16 only", response.json()["error"])

  def test_fp32_full_is_stored_and_sent_to_the_trainer(self) -> None:
    response = self.create(**{"openrl.fine_tuning_type": "full"}, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 200, response.text)
    meta = self.metadata(response.json()["request_id"])
    self.assertEqual((meta["trainer_dtype"], meta["sampler_dtype"]), ("float32", "float32"))
    self.assertEqual(self.queued()[0]["dtype"], "float32")

  def test_a_session_can_set_the_dtypes(self) -> None:
    session_id = self.post("create_session", {"tags": ["openrl.trainer_dtype=bfloat16", "openrl.sampler_dtype=bfloat16"]}).json()["session_id"]
    response = self.post("create_model", {"base_model": "m", "session_id": session_id})
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(self.metadata(response.json()["request_id"])["trainer_dtype"], "bfloat16")

  def test_a_session_without_dtypes_is_accepted(self) -> None:
    # Only a model must name them; the session may leave them to create_model.
    self.assertEqual(self.post("create_session", {"tags": []}).status_code, 200)

  def write_checkpoint(self, directory: str, dtype: str | None) -> None:
    saved = {"base_model": "checkpoint-base"}
    if dtype is not None:
      saved["dtype"] = dtype
    with open(os.path.join(directory, "metadata.json"), "w") as f:
      json.dump(saved, f)

  def restore(self, directory: str, **user_metadata):
    return self.post("create_model_from_state", {"state_path": directory, "user_metadata": user_metadata})

  def test_restore_requires_the_dtypes(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      self.write_checkpoint(directory, "bfloat16")
      response = self.restore(directory)
    self.assertEqual(response.status_code, 400)
    self.assertIn("must be set", response.json()["error"])

  def test_restore_must_match_the_checkpoint(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      self.write_checkpoint(directory, "bfloat16")
      response = self.restore(directory, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 400)
    self.assertIn("saved in bfloat16", response.json()["error"])

  def test_restore_with_the_checkpoint_dtype(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      self.write_checkpoint(directory, "float32")
      response = self.restore(directory, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(self.queued()[0]["dtype"], "float32")

  def test_restore_of_a_checkpoint_without_a_dtype_trusts_the_client(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      self.write_checkpoint(directory, None)
      response = self.restore(directory, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 200, response.text)


class StaticRuntimePrecisionTest(ApiPrecisionTest):
  """Without a worker manager one set of workers serves every model, in OPEN_RL_DTYPE."""

  def setUp(self) -> None:
    super().setUp()
    self.enterContext(patch.object(api_server, "worker_manager", None))

  def test_another_dtype_than_the_deployment_is_a_400(self) -> None:
    response = self.create(**{"openrl.fine_tuning_type": "full"}, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 400)
    self.assertIn("OPEN_RL_DTYPE", response.json()["error"])

  def test_the_deployment_dtype_is_accepted(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_DTYPE": "float32"}):
      response = self.create(**{"openrl.fine_tuning_type": "full"}, **dtypes("float32", "float32"))
    self.assertEqual(response.status_code, 200, response.text)

  # The inherited per-model fp32 cases do not apply to a bf16 static runtime,
  # and a static runtime refuses automodel before looking at its dtype.
  test_cross_precision_full_is_accepted_and_stored = None
  test_fp32_automodel_is_a_400 = None
  test_fp32_full_is_stored_and_sent_to_the_trainer = None
  test_restore_with_the_checkpoint_dtype = None
  test_restore_of_a_checkpoint_without_a_dtype_trusts_the_client = None


class WorkerEnvTest(unittest.TestCase):
  def meta(self, **fields) -> TrainingModelMetadata:
    return TrainingModelMetadata(base_model="Qwen/Qwen3-0.6B", fine_tuning_type="full", **fields)

  def test_each_role_gets_its_dtype(self) -> None:
    meta = self.meta(trainer_dtype="float32", sampler_dtype="float32")
    self.assertEqual(worker_env(meta, meta.base_model, "r", False, "trainer")["OPEN_RL_TRAINER_DTYPE"], "float32")
    self.assertEqual(worker_env(meta, meta.base_model, "r", False, "sampler")["OPEN_RL_SAMPLER_DTYPE"], "float32")
    self.assertNotIn("OPEN_RL_SAMPLER_DTYPE", worker_env(meta, meta.base_model, "r", False, "trainer"))

  def test_a_record_without_dtypes_runs_in_the_deployment_dtype(self) -> None:
    meta = self.meta()
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("OPEN_RL_DTYPE", None)
      self.assertEqual(worker_env(meta, meta.base_model, "r", False, "trainer")["OPEN_RL_TRAINER_DTYPE"], "bfloat16")


class EstimatorPrecisionTest(unittest.TestCase):
  def test_fp32_doubles_the_per_param_bytes(self) -> None:
    bf16 = footprint("Qwen/Qwen3-4B", "full", "trainer", "bfloat16").accelerator_bytes
    fp32 = footprint("Qwen/Qwen3-4B", "full", "trainer", "float32").accelerator_bytes
    reserve = 4 * 1024**3
    self.assertEqual(fp32 - reserve, 2 * (bf16 - reserve))
    self.assertGreater(
      footprint("Qwen/Qwen3-4B", "full", "sampler", "float32").accelerator_bytes,
      footprint("Qwen/Qwen3-4B", "full", "sampler", "bfloat16").accelerator_bytes,
    )

  def test_the_default_is_unchanged(self) -> None:
    self.assertEqual(footprint("Qwen/Qwen3-4B", "lora", "sampler"), footprint("Qwen/Qwen3-4B", "lora", "sampler", "bfloat16"))


class TrainerWorkerPrecisionTest(unittest.TestCase):
  def worker(self, dtype: str) -> BaseTrainerWorker:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_DTYPE": dtype}):
      return BaseTrainerWorker()

  def test_the_declared_dtype_is_used(self) -> None:
    self.assertEqual(self.worker("float32").model_dtype(), torch.float32)

  def test_bf16_on_a_gpu_without_it_raises_instead_of_falling_back(self) -> None:
    worker = self.worker("bfloat16")
    worker.device = torch.device("cuda", 0)
    with (
      patch("torch.cuda.is_bf16_supported", return_value=False),
      patch("torch.cuda.get_device_name", return_value="Tesla V100"),
      self.assertRaisesRegex(RuntimeError, "does not support bfloat16"),
    ):
      worker.model_dtype()

  def test_a_model_in_another_dtype_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "runs in bfloat16"):
      self.worker("bfloat16").check_command_dtype("float32")
    self.worker("bfloat16").check_command_dtype(None)  # an older server

  def test_a_bad_env_value_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "OPEN_RL_TRAINER_DTYPE"):
      self.worker("fp32")


@unittest.skipUnless(HAS_VLLM, "needs vllm")
class SamplerPrecisionTest(unittest.TestCase):
  def kwargs(self, env: dict[str, str], fft: bool) -> dict:
    from server.vllm_sampler import engine_kwargs_from_env

    with patch.dict(os.environ, {"BASE_MODEL": "m", **env}):
      return engine_kwargs_from_env(fft)

  def test_dtype_is_passed_explicitly(self) -> None:
    self.assertEqual(self.kwargs({"OPEN_RL_SAMPLER_DTYPE": "float32"}, fft=True)["dtype"], "float32")
    self.assertEqual(self.kwargs({"OPEN_RL_SAMPLER_DTYPE": "bfloat16"}, fft=False)["dtype"], "bfloat16")

  def test_fp32_lora_sampler_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "bfloat16 only"):
      self.kwargs({"OPEN_RL_SAMPLER_DTYPE": "float32"}, fft=False)

  def test_cross_precision_weights_are_accepted_and_invalid_dtypes_refused(self) -> None:
    from server.delta_weight_transfer_engine import check_weight_dtype

    check_weight_dtype({}, torch.bfloat16)  # an older trainer
    check_weight_dtype({"dtype": "bfloat16"}, torch.bfloat16)
    check_weight_dtype({"dtype": "float32"}, torch.bfloat16)
    check_weight_dtype({"dtype": "bfloat16"}, torch.float32)
    with self.assertRaisesRegex(ValueError, "Unsupported checkpoint dtype"):
      check_weight_dtype({"dtype": "float16"}, torch.bfloat16)


if __name__ == "__main__":
  unittest.main()
