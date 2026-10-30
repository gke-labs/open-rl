import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from server import api_server
from server.estimator import footprint
from server.model_metadata import TrainingModelMetadata
from server.scheduler_worker_manager import GROUP, PLURAL, VERSION, SchedulerWorkerManager
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import owner_id
from tests.api_client import asgi_client, post_json


class ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class FakeCustomObjectsApi:
  def __init__(self):
    self.created: list[dict[str, Any]] = []
    self.deleted: list[str] = []
    self.existing: dict[str, dict] = {}
    self.deleting: set[str] = set()

  def create_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict) -> dict:
    assert (group, version, plural) == (GROUP, VERSION, PLURAL)
    name = body["metadata"]["name"]
    if name in self.existing:
      raise ApiError(409)
    self.existing[name] = body
    self.created.append(body)
    return body

  def get_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict:
    if name not in self.existing:
      raise ApiError(404)
    metadata = dict(self.existing[name]["metadata"])
    if name in self.deleting:
      metadata["deletionTimestamp"] = "2026-09-08T00:00:00Z"
    return {"metadata": metadata}

  def delete_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict:
    if name not in self.existing:
      raise ApiError(404)
    del self.existing[name]
    self.deleted.append(name)
    return {}

  def list_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, label_selector: str = "") -> dict:
    key, _, value = label_selector.partition("=")
    items = [body for name, body in sorted(self.existing.items()) if not key or body["metadata"]["labels"].get(key) == value]
    return {"items": items}


def gpu_golden(role: str, tmp_dir: str) -> str:
  """A GPU LoRA worker's Workload for Qwen/Qwen3-0.6B as JSON, captured before TPU workers existed."""
  with open(Path(__file__).with_name("gpu_lora_workloads.json")) as f:
    return json.dumps(json.load(f)[role]).replace("$OPEN_RL_TMP_DIR", tmp_dir)


class FakeApi:
  def __init__(self):
    self.objects: dict[tuple[str, str], dict] = {}

  def __getattr__(self, method: str):
    verb, kind = method.split("_namespaced_")

    def call(*args, **kwargs):
      if verb == "list":
        key, _, value = kwargs["label_selector"].partition("=")
        found = [name for (k, name), body in self.objects.items() if k == kind and body["metadata"].get("labels", {}).get(key) == value]
        return SimpleNamespace(items=[SimpleNamespace(metadata=SimpleNamespace(name=name)) for name in found])
      if verb in ("delete", "read") and (kind, args[0]) not in self.objects:
        raise ApiError(404)
      if verb == "delete":
        return self.objects.pop((kind, args[0]))
      if verb == "read":
        return SimpleNamespace(metadata=SimpleNamespace(deletion_timestamp=None))
      if (kind, args[1]["metadata"]["name"]) in self.objects:
        raise ApiError(409)
      self.objects[(kind, args[1]["metadata"]["name"])] = args[1]

    return call


class HistoryStore(InMemoryStateStore):
  def __init__(self):
    super().__init__()
    self.pushed = threading.Event()

  def push_recent_sync(self, *args) -> None:
    super().push_recent_sync(*args)
    self.pushed.set()


class SchedulerWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    # The API server Deployment sets OPEN_RL_TMP_DIR to the shared volume every worker pod mounts. A temp dir stands in for it.
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_TMP_DIR": self.tmp}))
    self.api, self.k8s = FakeCustomObjectsApi(), FakeApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api, apps_api=self.k8s, networking_api=self.k8s)

  def store_with(self, model_id: str, meta: dict) -> InMemoryStore:
    s = InMemoryStateStore()
    s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
    return s

  def test_lora_trainer_and_sampler_share_an_owner(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "sampler")

    trainer, sampler = self.api.created
    owner = owner_id("Qwen/Qwen2.5-0.5B")
    self.assertEqual(trainer["metadata"]["name"], f"lora-{owner}-0-trainer")
    self.assertEqual(sampler["metadata"]["name"], f"lora-{owner}-0-sampler")
    # Same owner: one turn for the pair, and they hold the devices together.
    self.assertEqual(trainer["spec"]["ownerID"], owner)
    self.assertEqual(trainer["spec"]["ownerID"], sampler["spec"]["ownerID"])
    self.assertEqual(trainer["spec"]["trainingKind"], "lora")
    self.assertTrue(trainer["spec"]["exclusive"])
    self.assertEqual(trainer["spec"]["accelerator"], {"mode": "SingleGPU", "memory": footprint("Qwen/Qwen2.5-0.5B", "lora", "trainer").accelerator})
    t_container = trainer["spec"]["template"]["spec"]["containers"][0]
    s_container = sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["command"][-1], "server.training_requests_processor")
    self.assertEqual(s_container["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_serve"])
    self.assertIn("--active-tenant-set-id", t_container["args"])

  def test_an_exclusive_fft_worker_is_placed_alone_and_never_time_sliced(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full", "exclusive": True})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")
      self.manager.ensure("Model_A.1", "sampler")

    for worker in self.api.created:
      self.assertTrue(worker["spec"]["exclusive"])
      env = {e["name"]: e.get("value") for e in worker["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual(env["OPEN_RL_TIME_SLICING"], "off")

  def test_fft_worker_is_its_own_owner(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")

    (worker,) = self.api.created
    self.assertEqual(worker["spec"]["role"], "trainer")
    # An FFT job is its own owner: its trainer (and sampler, if launched)
    # share one turn, and no other job ever matches this ID.
    self.assertEqual(worker["metadata"]["name"], f"fft-{owner_id('Model_A.1')}-trainer")
    self.assertEqual(worker["spec"]["ownerID"], owner_id("Model_A.1"))
    self.assertEqual(worker["spec"]["trainingKind"], "fft")
    self.assertFalse(worker["spec"]["exclusive"])
    self.assertEqual(worker["spec"]["accelerator"]["memory"], footprint("Qwen/Qwen3-8B", "full", "trainer").accelerator)
    container = worker["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(env["OPEN_RL_ENABLE_FFT"], "true")
    self.assertNotIn("OPEN_RL_TIME_SLICING", env)
    self.assertEqual(env["OPEN_RL_FINE_TUNING_TYPE"], "full")
    self.assertEqual(env["OPEN_RL_WORKLOAD_ID"], worker["metadata"]["name"])

  def test_each_worker_is_sized_for_its_roles_first_accelerator(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_accel_prefs": ["tpu", "gpu"], "sampler_accel_prefs": ["gpu"]}
    s = self.store_with("job-tpu", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_TPU_TRAINER_IMAGE": "tpu-trainer:1"}):
      self.manager.ensure("job-tpu", "trainer")
      self.manager.ensure("job-tpu", "sampler")

    trainer, sampler = self.api.created
    for worker, accelerator in ((trainer, "tpu"), (sampler, "gpu")):
      fp = footprint("Qwen/Qwen3-0.6B", "lora", worker["spec"]["role"], accelerator=accelerator)
      container = worker["spec"]["template"]["spec"]["containers"][0]
      env = {e["name"]: e.get("value") for e in container["env"]}
      self.assertEqual(container["resources"], fp.resources, accelerator)
      self.assertEqual(env["OPEN_RL_ACCELERATOR_MEMORY"], str(fp.accelerator_bytes))
    self.assertNotEqual(trainer["spec"]["template"]["spec"]["containers"][0]["resources"], footprint("Qwen/Qwen3-0.6B", "lora", "trainer").resources)

  def test_a_default_model_is_sized_for_gpu(self) -> None:
    s = self.store_with("job-gpu", {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-gpu", "trainer")

    (worker,) = self.api.created
    gpu = footprint("Qwen/Qwen3-0.6B", "lora", "trainer", accelerator="gpu")
    self.assertEqual(worker["spec"]["template"]["spec"]["containers"][0]["resources"], gpu.resources)

  def test_gpu_workloads_are_unchanged(self) -> None:
    s = self.store_with("job", {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job", "trainer")
      self.manager.ensure("job", "sampler")

    trainer, sampler = self.api.created
    # Compared as JSON so key order counts too.
    self.assertEqual(json.dumps(trainer), gpu_golden("trainer", self.tmp))
    self.assertEqual(json.dumps(sampler), gpu_golden("sampler", self.tmp))

  def tpu_workloads(self, env: dict[str, str] | None = None) -> tuple[dict, dict]:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_accel_prefs": ["tpu"], "sampler_accel_prefs": ["tpu", "gpu"]}
    s = self.store_with("job-tpu", meta)
    images = {"OPEN_RL_TPU_TRAINER_IMAGE": "tpu-trainer:1", "OPEN_RL_TPU_SAMPLER_IMAGE": "tpu-sampler:1"}
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {**images, **(env or {})}):
      self.manager.ensure("job-tpu", "trainer")
      self.manager.ensure("job-tpu", "sampler")
    trainer, sampler = self.api.created
    return trainer, sampler

  def test_a_tpu_trainer_pod(self) -> None:
    trainer, _ = self.tpu_workloads()
    fp = footprint("Qwen/Qwen3-0.6B", "lora", "trainer", accelerator="tpu")
    self.assertEqual(trainer["spec"]["accelerator"], {"type": "TPU", "mode": "SingleGPU", "memory": fp.accelerator})
    template_spec = trainer["spec"]["template"]["spec"]
    container = template_spec["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(container["image"], "tpu-trainer:1")
    self.assertEqual(container["command"], ["uv", "run", "python", "-u", "-m", "server.training_requests_processor"])
    self.assertEqual(env["OPEN_RL_DEVICE"], "tpu")
    self.assertEqual(container["resources"], fp.resources)
    self.assertEqual(template_spec["tolerations"], [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}])
    self.assertEqual([v["name"] for v in template_spec["volumes"]], ["shared-storage"])
    self.assertEqual(container["volumeMounts"], [{"name": "shared-storage", "mountPath": "/mnt/shared"}])
    self.assertNotIn("envFrom", container)

  def test_a_tpu_sampler_pod(self) -> None:
    _, sampler = self.tpu_workloads()
    fp = footprint("Qwen/Qwen3-0.6B", "lora", "sampler", accelerator="tpu")
    self.assertEqual(sampler["spec"]["accelerator"], {"type": "TPU", "mode": "SingleGPU", "memory": fp.accelerator})
    template_spec = sampler["spec"]["template"]["spec"]
    container = template_spec["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(container["image"], "tpu-sampler:1")
    # A shared LoRA sampler is a routed pool replica, as on GPU.
    self.assertEqual(container["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_serve"])
    # OPEN_RL_DEVICE picks the trainer's device; the sampler image carries vLLM for TPU.
    self.assertNotIn("OPEN_RL_DEVICE", env)
    self.assertEqual(container["resources"], fp.resources)
    self.assertEqual(template_spec["tolerations"], [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}])
    self.assertIn({"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}, template_spec["volumes"])
    self.assertIn({"name": "dshm", "mountPath": "/dev/shm"}, container["volumeMounts"])
    self.assertIn({"name": "shared-storage", "mountPath": "/mnt/shared"}, container["volumeMounts"])

  def test_tpu_placement_stays_out_of_the_template(self) -> None:
    for worker in self.tpu_workloads():
      template_spec = worker["spec"]["template"]["spec"]
      for key in ("nodeSelector", "nodeName", "affinity", "resourceClaims"):
        self.assertNotIn(key, template_spec)

  def test_each_accelerator_reads_its_own_worker_configmap(self) -> None:
    configmaps = {"OPEN_RL_GPU_WORKER_ENV_CONFIGMAP": "gpu-env", "OPEN_RL_TPU_WORKER_ENV_CONFIGMAP": "tpu-env"}
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_accel_prefs": ["tpu"], "sampler_accel_prefs": ["gpu"]}
    s = self.store_with("job-mixed", meta)
    with (
      patch("server.worker_manager.get_state_store", return_value=s),
      patch.dict(os.environ, {"OPEN_RL_TPU_TRAINER_IMAGE": "tpu-trainer:1", **configmaps}),
    ):
      self.manager.ensure("job-mixed", "trainer")
      self.manager.ensure("job-mixed", "sampler")

    trainer, sampler = self.api.created
    self.assertEqual(trainer["spec"]["template"]["spec"]["containers"][0]["envFrom"], [{"configMapRef": {"name": "tpu-env"}}])
    self.assertEqual(sampler["spec"]["template"]["spec"]["containers"][0]["envFrom"], [{"configMapRef": {"name": "gpu-env"}}])

  def test_no_worker_configmap_when_unset(self) -> None:
    trainer, sampler = self.tpu_workloads({"OPEN_RL_GPU_WORKER_ENV_CONFIGMAP": "gpu-env"})
    for worker in (trainer, sampler):
      self.assertNotIn("envFrom", worker["spec"]["template"]["spec"]["containers"][0])

  def test_a_tpu_worker_without_its_image_fails_naming_the_variable(self) -> None:
    s = self.store_with("job-tpu", {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "sampler_accel_prefs": ["tpu"]})
    env = {k: v for k, v in os.environ.items() if k not in {"OPEN_RL_TPU_TRAINER_IMAGE", "OPEN_RL_TPU_SAMPLER_IMAGE"}}
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, env, clear=True):
      self.manager.ensure("job-tpu", "trainer")
      with self.assertRaisesRegex(RuntimeError, "OPEN_RL_TPU_SAMPLER_IMAGE"):
        self.manager.ensure("job-tpu", "sampler")
    # The GPU trainer still went out; the TPU sampler never did.
    self.assertEqual([w["spec"]["role"] for w in self.api.created], ["trainer"])

  def test_an_fft_tpu_sampler_pod_uses_the_fft_sampler_image(self) -> None:
    meta = {
      "base_model": "Qwen/Qwen3-0.6B",
      "fine_tuning_type": "full",
      "exclusive": True,
      "trainer_accel_prefs": ["tpu"],
      "sampler_accel_prefs": ["tpu"],
    }
    s = self.store_with("job-fft", meta)
    images = {
      "OPEN_RL_TPU_TRAINER_IMAGE": "tpu-trainer:1",
      "OPEN_RL_TPU_SAMPLER_IMAGE": "tpu-sampler:1",
      "OPEN_RL_TPU_FFT_SAMPLER_IMAGE": "tpu-fft-sampler:1",
    }
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, images):
      self.manager.ensure("job-fft", "trainer")
      self.manager.ensure("job-fft", "sampler")
    trainer, sampler = self.api.created
    self.assertEqual(trainer["spec"]["template"]["spec"]["containers"][0]["image"], "tpu-trainer:1")
    self.assertEqual(sampler["spec"]["template"]["spec"]["containers"][0]["image"], "tpu-fft-sampler:1")

  def test_an_fft_tpu_sampler_without_its_image_fails_naming_the_variable(self) -> None:
    s = self.store_with("job-fft", {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", "exclusive": True, "sampler_accel_prefs": ["tpu"]})
    env = {k: v for k, v in os.environ.items() if k != "OPEN_RL_TPU_FFT_SAMPLER_IMAGE"}
    env["OPEN_RL_TPU_SAMPLER_IMAGE"] = "tpu-sampler:1"
    with (
      patch("server.worker_manager.get_state_store", return_value=s),
      patch.dict(os.environ, env, clear=True),
      self.assertRaisesRegex(RuntimeError, "OPEN_RL_TPU_FFT_SAMPLER_IMAGE"),
    ):
      self.manager.ensure("job-fft", "sampler")

  def test_mutable_worker_images_use_the_requested_pull_policy(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with (
      patch("server.worker_manager.get_state_store", return_value=s),
      patch.dict(os.environ, {"OPEN_RL_WORKER_IMAGE": "localhost:5001/open-rl-server:kind-dev", "OPEN_RL_WORKER_IMAGE_PULL_POLICY": "Always"}),
    ):
      self.manager.ensure("job-lora-1", "trainer")

    container = self.api.created[0]["spec"]["template"]["spec"]["containers"][0]
    # Reusing kind-dev must fetch the rebuilt worker rather than its cached predecessor.
    self.assertEqual(container["imagePullPolicy"], "Always")

  def test_workers_get_the_deployment_settings(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    settings = {"VLLM_MAX_MODEL_LEN": "131072", "VLLM_ENABLE_MULTIMODAL": "1", "OPEN_RL_TRAIN_TOKEN_BUDGET": "131072", "MAX_JOBS": "4"}
    with patch.dict(os.environ, settings), patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")
      self.manager.ensure("job-lora-1", "trainer")

    self.assertEqual(len(self.api.created), 2)
    for pod in self.api.created:
      env = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual({k: env.get(k) for k in settings}, settings)

  def test_samplers_keep_the_vllm_compile_cache_on_the_shared_volume(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")
      self.manager.ensure("job-lora-1", "trainer")

    envs = {}
    for pod in self.api.created:
      envs[pod["spec"]["role"]] = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
    self.assertEqual(envs["sampler"]["VLLM_CACHE_ROOT"], f"{self.tmp}/vllm-cache")
    self.assertEqual(envs["sampler"]["TRITON_CACHE_DIR"], f"{self.tmp}/vllm-cache/triton")
    self.assertNotIn("VLLM_CACHE_ROOT", envs["trainer"])

  def test_a_bare_base_model_sampler_on_an_fft_deployment_is_fft_with_a_pool_apart_from_the_shared_lora_pool(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")
      self.manager.ensure("Qwen/Qwen2.5-0.5B", "sampler")
    self.assertEqual([w["spec"]["trainingKind"] for w in self.api.created], ["lora", "fft"])
    shared, bare = self.api.created
    self.assertNotEqual(bare["spec"]["ownerID"], shared["spec"]["ownerID"])
    pools = [w["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/instance"] for w in (shared, bare)]
    self.assertNotEqual(pools[0], pools[1])

  def test_both_sampler_shapes_get_every_engine_setting_the_api_server_has(self) -> None:
    # A replica and an engine in the sampler's process compute their settings on their own pods, from these values.
    settings = {"VLLM_MAX_LORAS": "16", "VLLM_MAX_NUM_SEQS": "512", "OPEN_RL_ENABLE_FFT": "true"}
    for model_id, meta in (("job-lora-1", {"fine_tuning_type": "lora"}), ("job-fft-1", {"fine_tuning_type": "full"})):
      s = self.store_with(model_id, {"base_model": "Qwen/Qwen2.5-0.5B", **meta})
      with patch.dict(os.environ, settings), patch("server.worker_manager.get_state_store", return_value=s):
        self.manager.ensure(model_id, "sampler")
    samplers = [pod for pod in self.api.created if pod["spec"]["role"] == "sampler"]
    commands = {pod["spec"]["template"]["spec"]["containers"][0]["command"][-1] for pod in samplers}
    self.assertEqual(commands, {"server.vllm_serve", "server.vllm_sampler"})
    for pod in samplers:
      env = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual((env.get("VLLM_MAX_LORAS"), env.get("VLLM_MAX_NUM_SEQS")), ("16", "512"))

  def test_launch_is_idempotent(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "trainer")
    self.assertEqual(len(self.api.created), 1)

  def test_gemma_4_samplers_load_the_text_only_model(self) -> None:
    for model_id, tuning in (("job-lora-1", "lora"), ("job-fft-1", "full")):
      s = self.store_with(model_id, {"base_model": "google/gemma-4-E2B", "fine_tuning_type": tuning})
      with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.worker_manager.get_state_store", return_value=s):
        self.manager.ensure(model_id, "sampler")

    self.assertEqual(len(self.api.created), 2)
    for pod in self.api.created:
      env = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
      self.assertEqual(env.get("VLLM_ARCHITECTURE_OVERRIDE"), "Gemma4ForCausalLM")

  def test_multimodal_gemma_4_samplers_keep_the_full_model(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "google/gemma-4-E2B", "fine_tuning_type": "lora"})
    with patch.dict(os.environ, {"VLLM_ENABLE_MULTIMODAL": "1"}), patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")

    (pod,) = self.api.created
    env = {e["name"]: e.get("value") for e in pod["spec"]["template"]["spec"]["containers"][0]["env"]}
    self.assertNotIn("VLLM_ARCHITECTURE_OVERRIDE", env)
    self.assertEqual(env.get("VLLM_ENABLE_MULTIMODAL"), "1")

  def test_placement_knowledge_stays_out_of_the_template(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")

    (worker,) = self.api.created
    template_spec = worker["spec"]["template"]["spec"]
    env_names = {e["name"] for e in template_spec["containers"][0]["env"]}
    # The group is the claim name: placement's output, stamped by the
    # controller. Node selection and claims likewise never appear here --
    # the controller rejects a template that carries them.
    self.assertNotIn("OPEN_RL_TIME_SLICE_GROUP", env_names)
    self.assertNotIn("nodeSelector", template_spec)
    self.assertNotIn("nodeName", template_spec)
    self.assertNotIn("affinity", template_spec)
    self.assertNotIn("resourceClaims", template_spec)

  def test_release_deletes_an_fft_jobs_workloads_and_tolerates_absence(self) -> None:
    s = self.store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")
      self.manager.release("Model_A.1")
      self.manager.release("Model_A.1")

    self.assertEqual(self.api.deleted, [f"fft-{owner_id('Model_A.1')}-trainer"])

  def test_release_leaves_a_shared_lora_runtime_alone(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.release("job-lora-1")

    self.assertEqual(self.api.deleted, [])

  def test_exclusive_lora_models_get_runtimes_of_their_own(self) -> None:
    s = InMemoryStateStore()
    for model_id in ("job-a", "job-b"):
      meta = {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora", "exclusive": True}
      s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-a", "trainer")
      self.manager.ensure("job-b", "trainer")
      self.manager.release("job-a")

    self.assertEqual([w["metadata"]["name"] for w in self.api.created], ["lora-job-a-0-trainer", "lora-job-b-0-trainer"])
    self.assertEqual(self.api.deleted, ["lora-job-a-0-trainer"])

  def test_automodel_jobs_share_a_runtime_apart_from_pytorch(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel"}
    s = self.store_with("job-am-1", meta)
    s.kv_store["open_rl:model_meta:job-am-2"] = json.dumps({**meta, "lora_config": {"rank": 32}})
    s.kv_store["open_rl:model_meta:job-pt"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-am-1", "trainer")
      self.manager.ensure("job-am-1", "sampler")
      self.manager.ensure("job-am-2", "trainer")
      self.manager.ensure("job-pt", "trainer")
      self.manager.release("job-am-1")

    # The second Automodel job's create found the first one's trainer, so only
    # three workloads exist.
    am_trainer, am_sampler, pt_trainer = self.api.created
    self.assertEqual(am_trainer["metadata"]["name"], f"lora-{owner_id('automodel-Qwen/Qwen3-0.6B')}-0-trainer")
    self.assertEqual(pt_trainer["metadata"]["name"], f"lora-{owner_id('Qwen/Qwen3-0.6B')}-0-trainer")
    runtime = am_trainer["spec"]["modelID"]
    self.assertEqual(am_sampler["spec"]["ownerID"], am_trainer["spec"]["ownerID"])
    t_container = am_trainer["spec"]["template"]["spec"]["containers"][0]
    s_container = am_sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["image"], "am:1")
    self.assertEqual(t_container["command"], ["python", "-u", "-m", "server.training_requests_processor"])
    self.assertEqual(t_container["args"], ["--model-id", runtime, "--active-tenant-set-id", f"{runtime}-1"])
    self.assertEqual({e["name"]: e.get("value") for e in t_container["env"]}["OPEN_RL_TRAINER_BACKEND"], "automodel")
    self.assertNotEqual(s_container["image"], "am:1")
    self.assertEqual(s_container["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_serve"])
    # Like any shared LoRA runtime, deleting one job leaves it up.
    self.assertEqual(self.api.deleted, [])

  def test_an_exclusive_automodel_job_gets_its_own_automodel_trainer(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "exclusive": True}
    s = self.store_with("job-am", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-am", "trainer")
      self.manager.release("job-am")

    (trainer,) = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-job-am-0-trainer")
    self.assertEqual(trainer["spec"]["template"]["spec"]["containers"][0]["image"], "am:1")
    self.assertEqual(self.api.deleted, ["lora-job-am-0-trainer"])

  def test_a_multi_gpu_automodel_trainer_is_one_torchrun_group(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "trainer_gpus": 4}
    s = self.store_with("job-dp", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-dp", "trainer")
      self.manager.ensure("job-dp", "sampler")

    trainer, sampler = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-job-dp-0-trainer")
    self.assertTrue(trainer["spec"]["exclusive"])
    one = footprint("Qwen/Qwen3-0.6B", "lora", "trainer")
    self.assertEqual(trainer["spec"]["accelerator"], {"mode": "MultiGPU", "devices": 4, "memory": one.accelerator})
    pod = trainer["spec"]["template"]["spec"]
    container = pod["containers"][0]
    self.assertEqual(container["command"][:6], ["python", "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"])
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(env["OPEN_RL_CONTROL_BACKEND"], "cpu:gloo,cuda:nccl")
    self.assertEqual(env["OPEN_RL_TIME_SLICING"], "off")
    self.assertNotIn("OPEN_RL_AUTOMODEL_CP", env)
    self.assertIn({"name": "dshm", "mountPath": "/dev/shm"}, container["volumeMounts"])
    self.assertEqual(container["resources"]["requests"]["memory"], f"{-(-one.host_request_bytes * 4 // 2**30)}Gi")
    # The sampler is the usual single-GPU one, alone with this job.
    self.assertEqual(sampler["metadata"]["name"], "lora-job-dp-0-sampler")
    self.assertEqual(sampler["spec"]["accelerator"]["mode"], "SingleGPU")
    self.assertTrue(sampler["spec"]["exclusive"])

  def test_a_context_parallel_trainer_gets_its_cp_size(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "automodel", "trainer_gpus": 4, "trainer_cp": 4}
    s = self.store_with("job-cp", meta)
    with patch("server.worker_manager.get_state_store", return_value=s), patch.dict(os.environ, {"OPEN_RL_AUTOMODEL_IMAGE": "am:1"}):
      self.manager.ensure("job-cp", "trainer")

    (trainer,) = self.api.created
    env = {e["name"]: e.get("value") for e in trainer["spec"]["template"]["spec"]["containers"][0]["env"]}
    self.assertEqual(env["OPEN_RL_AUTOMODEL_CP"], "4")

  def test_a_job_that_names_an_image_runs_its_trainer_from_it(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "ghcr.io/org/trainer:1"}
    s = self.store_with("job-img", meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-img", "trainer")
      self.manager.ensure("job-img", "sampler")

    trainer, sampler = self.api.created
    self.assertRegex(trainer["metadata"]["name"], r"^lora-image-[0-9a-f]{10}-qwen-qwen3-0-6b-[0-9a-f]{8}-0-trainer$")
    t_container = trainer["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["image"], "ghcr.io/org/trainer:1")
    self.assertEqual(t_container["command"], ["python", "-u", "-m", "server.training_requests_processor"])
    # The image sets its own OPEN_RL_TRAINER_BACKEND.
    self.assertNotIn("OPEN_RL_TRAINER_BACKEND", {e["name"] for e in t_container["env"]})
    self.assertNotEqual(sampler["spec"]["template"]["spec"]["containers"][0]["image"], "ghcr.io/org/trainer:1")

  def test_a_job_image_can_run_a_tpu_trainer(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "trainer_backend": "ghcr.io/org/trainer:1", "trainer_accel_prefs": ["tpu"]}
    s = self.store_with("job-img-tpu", meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-img-tpu", "trainer")

    (trainer,) = self.api.created
    self.assertEqual(trainer["spec"]["accelerator"]["type"], "TPU")
    template_spec = trainer["spec"]["template"]["spec"]
    container = template_spec["containers"][0]
    self.assertEqual(container["image"], "ghcr.io/org/trainer:1")
    self.assertEqual({e["name"]: e.get("value") for e in container["env"]}["OPEN_RL_DEVICE"], "tpu")
    self.assertEqual(template_spec["tolerations"][0]["key"], "google.com/tpu")

  def test_release_owner_deletes_a_shared_lora_pair_and_nothing_else(self) -> None:
    s = self.store_with("adapter", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    s.kv_store["open_rl:model_meta:other"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("adapter", "trainer")
      self.manager.ensure("adapter", "sampler")
      self.manager.ensure("other", "trainer")
    owner = owner_id("Qwen/Qwen2.5-0.5B")
    self.assertEqual(self.manager.release_owner(owner), {"Qwen/Qwen2.5-0.5B"})
    self.assertEqual(self.manager.release_owner(owner), set())

    self.assertEqual(self.api.deleted, [f"lora-{owner}-0-sampler", f"lora-{owner}-0-trainer"])
    self.assertEqual(list(self.api.existing), [f"lora-{owner_id('Qwen/Qwen3-0.6B')}-0-trainer"])

  def test_release_owner_finds_workloads_by_their_spec_not_their_labels(self) -> None:
    # A workload from before this API server carries only the managed-by label.
    self.api.existing["fft-old-trainer"] = {
      "metadata": {"name": "fft-old-trainer", "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
      "spec": {"ownerID": "old", "modelID": "old"},
    }
    self.assertEqual(self.manager.release_owner("old"), {"old"})
    self.assertEqual(self.api.deleted, ["fft-old-trainer"])

  def place_lora_pair_and_other(self) -> str:
    s = self.store_with("adapter", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    s.kv_store["open_rl:model_meta:other"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("adapter", "trainer")
      self.manager.ensure("adapter", "sampler")
      self.manager.ensure("other", "trainer")
    for name, body in self.api.existing.items():
      body["status"] = {"nodeName": "n1", "podName": f"orw-{name}", "gpuUUIDs": [f"GPU-{name}"], "placedAt": "2026-10-29T12:00:00Z"}
    return owner_id("Qwen/Qwen2.5-0.5B")

  def test_release_records_the_final_placement_of_each_released_workload(self) -> None:
    owner, store = self.place_lora_pair_and_other(), HistoryStore()
    with patch("server.scheduler_worker_manager.get_state_store", return_value=store):
      self.manager.release_owner(owner)
    self.assertTrue(store.pushed.wait(5))

    self.assertEqual(list(store.lists), ["open_rl:placements:Qwen/Qwen2.5-0.5B"])
    records = [json.loads(r) for r in store.lists["open_rl:placements:Qwen/Qwen2.5-0.5B"]]
    self.assertEqual(
      sorted((r["name"], r["role"], r["node"], r["gpu_uuids"][0]) for r in records),
      [
        (f"lora-{owner}-0-sampler", "sampler", "n1", f"GPU-lora-{owner}-0-sampler"),
        (f"lora-{owner}-0-trainer", "trainer", "n1", f"GPU-lora-{owner}-0-trainer"),
      ],
    )
    self.assertTrue(all(r["training_kind"] == "lora" and r["released_at"] > r["placed_at"] for r in records), records)

  def test_a_failing_or_hanging_history_store_does_not_hold_up_the_release(self) -> None:
    unblock = threading.Event()
    self.addCleanup(unblock.set)
    for push in (RuntimeError("redis down"), lambda *args: unblock.wait(5)):
      self.api.deleted.clear()
      owner = self.place_lora_pair_and_other()
      store = SimpleNamespace(push_recent_sync=Mock(side_effect=push))
      with patch("server.scheduler_worker_manager.get_state_store", return_value=store):
        started = time.monotonic()
        self.assertEqual(self.manager.release_owner(owner), {"Qwen/Qwen2.5-0.5B"})
      self.assertLess(time.monotonic() - started, 1)
      self.assertEqual(len(self.api.deleted), 2)

  def test_ensure_waits_for_a_terminating_workload_before_recreating_it(self) -> None:
    s = self.store_with("adapter", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("adapter", "trainer")
      name = self.api.created[0]["metadata"]["name"]
      self.api.deleting.add(name)

      def finalizer_finishes(_seconds: float) -> None:
        self.api.deleting.discard(name)
        self.api.existing.pop(name, None)

      with patch("server.scheduler_worker_manager.time.sleep", side_effect=finalizer_finishes) as sleep:
        self.manager.ensure("adapter", "trainer")

    self.assertEqual(sleep.call_count, 1)
    self.assertEqual(len(self.api.created), 2)

  def test_create_worker_manager_selects_scheduler_mode(self) -> None:
    from server.worker_manager import create_worker_manager

    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_MANAGER": "scheduler"}, clear=False),
      patch("server.scheduler_worker_manager.SchedulerWorkerManager") as manager_cls,
    ):
      create_worker_manager()
      manager_cls.assert_called_once()

  def test_sampler_replicas_are_single_gpu_workloads_released_together(self) -> None:
    meta = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 3}
    s = self.store_with("job-sr", meta)
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-sr", "trainer")
      self.manager.ensure("job-sr", "sampler")
      self.manager.release("job-sr")

    names = [w["metadata"]["name"] for w in self.api.created]
    self.assertEqual(names, ["lora-job-sr-0-trainer", "lora-job-sr-0-sampler", "lora-job-sr-1-sampler", "lora-job-sr-2-sampler"])
    for sampler in self.api.created[1:]:
      self.assertEqual(sampler["spec"]["accelerator"]["mode"], "SingleGPU")
      self.assertEqual(sampler["spec"]["ownerID"], "job-sr")
    self.assertEqual(sorted(self.api.deleted), sorted(names))

  def test_a_pool_of_one_replica_serves_itself_with_no_router(self) -> None:
    s = self.store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora", "exclusive": True})
    with patch("server.worker_manager.get_state_store", return_value=s):
      self.manager.ensure("job-lora-1", "sampler")
    (sampler,) = self.api.created
    container = sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual((container["command"][-1], container["args"]), ("server.vllm_sampler", ["--pool", "sampler-job-lora-1"]))
    self.assertEqual({e["name"]: e.get("value") for e in container["env"]}["OPEN_RL_TIME_SLICING"], "off")
    self.assertEqual(self.k8s.objects, {})

  def test_a_shared_lora_pool_is_routed_and_grows_to_the_most_replicas_any_job_asks_for(self) -> None:
    for first, second in ((1, 2), (2, 1)):
      with self.subTest(first=first, second=second):
        api, k8s = FakeCustomObjectsApi(), FakeApi()
        manager = SchedulerWorkerManager(custom_api=api, apps_api=k8s, networking_api=k8s)
        s = InMemoryStateStore()
        for model_id, replicas in (("job-a", first), ("job-b", second)):
          meta = {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora", "sampler_replicas": replicas}
          s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
        with patch("server.worker_manager.get_state_store", return_value=s):
          manager.ensure("job-a", "sampler")
          manager.ensure("job-b", "sampler")
        commands = {w["metadata"]["name"]: w["spec"]["template"]["spec"]["containers"][0]["command"] for w in api.created}
        serve = ["uv", "run", "python", "-u", "-m", "server.vllm_serve"]
        owner = owner_id("Qwen/Qwen2.5-0.5B")
        self.assertEqual(commands, {f"lora-{owner}-0-sampler": serve, f"lora-{owner}-1-sampler": serve})
        pools = {w["metadata"]["labels"]["app.kubernetes.io/instance"] for w in api.created}
        self.assertEqual([name for kind, name in k8s.objects if kind == "deployment"], list(pools))

  def test_base_models_that_clean_to_one_name_get_workers_of_their_own(self) -> None:
    long = "org/" + "m" * 70
    for first, second in (("Qwen/Qwen3-0.6B", "qwen/qwen3-0.6b"), (f"{long}-a", f"{long}-b")):
      with self.subTest(first=first, second=second):
        api = FakeCustomObjectsApi()
        manager = SchedulerWorkerManager(custom_api=api, apps_api=FakeApi(), networking_api=FakeApi())
        s = InMemoryStateStore()
        for model_id, base_model in (("job-a", first), ("job-b", second)):
          s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps({"base_model": base_model, "fine_tuning_type": "lora"})
        with patch("server.worker_manager.get_state_store", return_value=s):
          for model_id in ("job-a", "job-b"):
            manager.ensure(model_id, "trainer")
            manager.ensure(model_id, "sampler")
        self.assertEqual(len({w["spec"]["ownerID"] for w in api.created}), 2)
        served = {w["metadata"]["labels"]["app.kubernetes.io/instance"] for w in api.created if w["spec"]["role"] == "sampler"}
        self.assertEqual(served, {TrainingModelMetadata(base_model=b).pool("job") for b in (first, second)})


class MixedSamplingSessionTest(unittest.IsolatedAsyncioTestCase):
  async def test_lora_and_fft_sessions_launch_their_own_sampler_types(self) -> None:
    store = InMemoryStore()
    state = InMemoryStateStore()
    for model_id, kind in (("lora-a", "lora"), ("lora-b", "lora"), ("fft-a", "full")):
      await state.set_value(f"open_rl:model_meta:{model_id}", json.dumps({"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": kind}))
    api, tmp = FakeCustomObjectsApi(), self.enterContext(tempfile.TemporaryDirectory())
    env = {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm", "OPEN_RL_TMP_DIR": tmp}
    with (
      patch.dict(os.environ, env),
      patch("server.worker_manager.get_state_store", return_value=state),
      patch.object(api_server, "store", store),
      patch.object(api_server, "state", state),
      patch.object(api_server, "get_store", return_value=store),
      patch.object(api_server, "worker_manager", SchedulerWorkerManager(custom_api=api, apps_api=FakeApi(), networking_api=FakeApi())),
    ):
      async with asgi_client() as client:
        for model_id in ("lora-a", "fft-a", "lora-b"):
          await post_json(client, "create_sampling_session", {"model_path": f"tinker://{model_id}/sampler_weights/checkpoint"})

    self.assertEqual(len(api.created), 2, "LoRA sessions should reuse one sampler while FFT gets its own")
    lora, fft = api.created
    self.assertEqual(lora["spec"]["trainingKind"], "lora")
    self.assertEqual(lora["spec"]["template"]["spec"]["containers"][0]["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_serve"])
    self.assertEqual(fft["spec"]["trainingKind"], "fft")
    self.assertEqual(fft["metadata"]["name"], "fft-fft-a-sampler")


class SchedulerModeTpuModelTest(unittest.IsolatedAsyncioTestCase):
  async def create_tpu_model(self, env: dict[str, str]) -> tuple[FakeCustomObjectsApi, InMemoryStore, str]:
    store = InMemoryStore()
    state = InMemoryStateStore()
    api = FakeCustomObjectsApi()
    with (
      patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_TMP_DIR": tempfile.mkdtemp(), **env}),
      patch("server.worker_manager.get_state_store", return_value=state),
      patch.object(api_server, "store", store),
      patch.object(api_server, "state", state),
      patch.object(api_server, "get_store", return_value=store),
      patch.object(api_server, "worker_manager", SchedulerWorkerManager(custom_api=api)),
    ):
      async with asgi_client() as client:
        user_metadata = {"openrl.trainer_accel_prefs": "tpu", "openrl.sampler_accel_prefs": "tpu"}
        created = await post_json(client, "create_model", {"base_model": "Qwen/Qwen3-0.6B", "user_metadata": user_metadata})
    return api, store, created["request_id"]

  async def test_a_tpu_model_gets_a_tpu_trainer(self) -> None:
    api, _, _ = await self.create_tpu_model({"OPEN_RL_TPU_TRAINER_IMAGE": "tpu-trainer:1"})
    (trainer,) = api.created
    self.assertEqual(trainer["spec"]["accelerator"]["type"], "TPU")
    self.assertEqual(trainer["spec"]["template"]["spec"]["containers"][0]["image"], "tpu-trainer:1")

  async def test_a_missing_tpu_image_fails_the_request_by_name(self) -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPEN_RL_TPU_TRAINER_IMAGE"}
    with patch.dict(os.environ, env, clear=True):
      api, store, request_id = await self.create_tpu_model({})
    self.assertEqual(api.created, [])
    future = store.futures_store[request_id]
    self.assertEqual(future["type"], "RequestFailedResponse")
    self.assertIn("OPEN_RL_TPU_TRAINER_IMAGE", future["error_message"])


if __name__ == "__main__":
  unittest.main()
