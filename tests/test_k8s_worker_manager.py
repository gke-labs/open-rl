import os
import tempfile
import time
import types
import unittest
from typing import Any
from unittest.mock import patch

from server.k8s_worker_manager import KubernetesWorkerManager, sanitize_job_id
from server.worker_manager import LocalWorkerManager, create_worker_manager

POD_TEMPLATE = """\
apiVersion: v1
kind: Pod
spec:
  restartPolicy: OnFailure
  containers:
  - name: trainer-worker
    image: example/server:latest
    command: ["python", "-m", "server.training_requests_processor"]
    env:
    - name: REDIS_URL
      value: "redis://redis-service:6379"
  resourceClaims:
  - name: trainer-gpu
    resourceClaimName: open-rl-trainer-gpu-1
"""


class _ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class _FakeCustomObjectsApi:
  def __init__(self, api_client: Any = None):
    self.api_client = api_client

  def list_namespaced_custom_object(
    self,
    group: str,
    version: str,
    namespace: str,
    plural: str,
    label_selector: str = "",
  ) -> dict[str, Any]:
    parts = dict(pair.split("=") for pair in label_selector.split(",") if "=" in pair)
    workload_type = parts.get("open-rl.io/workload-type", "")
    role = parts.get("open-rl.io/role", "")
    custom_objects = getattr(self.api_client, "custom_objects", None)
    if custom_objects is not None:
      names = custom_objects.get((workload_type, role), [])
    elif workload_type == "lora":
      names = ["open-rl-lora-trainer-gpu-1"] if role == "trainer" else ["open-rl-lora-sampler-gpu-1"]
    else:
      names = ["open-rl-trainer-gpu-1"] if role == "trainer" else ["open-rl-sampler-gpu-1"]
    return {"items": [{"metadata": {"name": name}} for name in names]}

  def list_cluster_custom_object(self, group: str, version: str, plural: str) -> dict[str, Any]:
    return {
      "items": [
        {
          "spec": {
            "driver": "gpu.nvidia.com",
            "devices": [
              {
                "name": "gpu-0",
                "attributes": {"productName": {"string": "NVIDIA L4"}},
                "capacity": {"memory": {"value": "23034Mi"}},
              },
              {
                "name": "gpu-0",
                "attributes": {"productName": {"string": "NVIDIA H100 80GB HBM3"}},
                "capacity": {"memory": {"value": "81559Mi"}},
              },
            ],
          }
        }
      ]
    }

  def create_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict) -> dict[str, Any]:
    name = body.get("metadata", {}).get("name", "created-claim")
    return {"metadata": {"name": name}}

  def delete_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict[str, Any]:
    return {"status": "Success"}


class _FakeCoreApi:
  def __init__(self, pod_phases: dict[str, str] | None = None, custom_objects: dict[tuple[str, str], list[str]] | None = None):
    self.pod_phases = pod_phases or {}
    self.created: list[tuple[str, dict]] = []
    self.deleted: list[str] = []
    self.create_error: Exception | None = None
    self.api_client = types.SimpleNamespace(custom_objects=custom_objects)

  def read_namespaced_pod(self, name: str, namespace: str):
    if name not in self.pod_phases:
      raise _ApiError(404)
    return types.SimpleNamespace(status=types.SimpleNamespace(phase=self.pod_phases[name]))

  def create_namespaced_pod(self, namespace: str, body: dict):
    if self.create_error is not None:
      raise self.create_error
    self.created.append((namespace, body))

  def delete_namespaced_pod(self, name: str, namespace: str):
    self.deleted.append(name)
    self.pod_phases.pop(name, None)


class KubernetesWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    self._custom_api_patcher = patch("server.k8s_worker_manager.client.CustomObjectsApi", _FakeCustomObjectsApi)
    self._custom_api_patcher.start()
    self.addCleanup(self._custom_api_patcher.stop)
    self.template_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    self.template_file.write(POD_TEMPLATE)
    self.template_file.close()
    self.addCleanup(os.unlink, self.template_file.name)
    self.env = {
      "REDIS_URL": "redis://redis-service:6379",
      "OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name,
      "OPEN_RL_WORKER_NAMESPACE": "training",
      "OPEN_RL_ENABLE_FFT": "1",
    }

  def _manager(self, core_api: _FakeCoreApi) -> KubernetesWorkerManager:
    with patch.dict(os.environ, self.env, clear=True):
      return KubernetesWorkerManager(core_api=core_api)

  def test_launch_stamps_name_labels_args_and_job_id_env(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch("Model_A.1")

    self.assertEqual(len(api.created), 1)
    namespace, body = api.created[0]
    self.assertEqual(namespace, "training")
    self.assertEqual(body["metadata"]["name"], "open-rl-trainer-model-a-1-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-trainer-worker",
        "accel-timeslicer": "true",
        "open-rl.io/workload-type": "full",
        "open-rl.io/role": "trainer",
        "open-rl.io/assigned-claim": "open-rl-trainer-gpu-1",
        "open-rl.io/bound-base-model": "model-a-1",
        "timeslice.io/group": "trainers",
        "timeslice.io/job-id": "trainer-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["args"], ["--model-id", "Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "trainer-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "trainers"}, container["env"])

  def test_launch_replaces_stale_job_id_env_from_template(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "stale-job"})
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "stale-group"})

    manager.launch("Model_A.1")

    container = api.created[0][1]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    self.assertEqual(env["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-model-a-1")
    self.assertEqual(env["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  def test_launch_sampler_stamps_sampler_identity(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch_sampler("Model_A.1")

    self.assertEqual(len(api.created), 1)
    _, body = api.created[0]
    self.assertEqual(body["metadata"]["name"], "open-rl-sampler-model-a-1-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-sampler-worker",
        "accel-timeslicer": "true",
        "open-rl.io/workload-type": "full",
        "open-rl.io/role": "sampler",
        "open-rl.io/assigned-claim": "open-rl-sampler-gpu-1",
        "open-rl.io/bound-base-model": "model-a-1",
        "timeslice.io/group": "samplers",
        "timeslice.io/job-id": "sampler-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["command"], ["uv", "run", "python", "-u", "-m", "server.vllm_sampler"])
    self.assertEqual(container["args"], ["--model-id", "Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "sampler-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "samplers"}, container["env"])

  def test_launch_is_idempotent_while_pod_is_live(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a-1": "Running"})
    self._manager(api).launch("model-a")

    self.assertEqual(api.created, [])
    self.assertEqual(api.deleted, [])

  def test_launch_replaces_terminal_pod(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a-1": "Failed"})
    self._manager(api).launch("model-a")

    self.assertEqual(api.deleted, ["open-rl-trainer-model-a-1"])
    self.assertEqual(len(api.created), 1)

  def test_launch_tolerates_conflict_on_create(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(409)
    self._manager(api).launch("model-a")  # must not raise

  def test_launch_raises_on_other_api_errors(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(403)
    with self.assertRaises(_ApiError):
      self._manager(api).launch("model-a")

  def test_launch_queries_model_metadata_for_pod_env(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "gemma-4-k8s",
        "weight_sync_config": {"strategy": "full"},
        "fine_tuning_type": "full",
      }
    )
    api = _FakeCoreApi()

    with patch("server.store.get_store", return_value=s):
      self._manager(api).launch("Model_A.1")
      self._manager(api).launch_sampler("Model_A.1")

    trainer_container = api.created[0][1]["spec"]["containers"][0]
    trainer_env = {item["name"]: item["value"] for item in trainer_container["env"] if "value" in item}
    self.assertEqual(trainer_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(trainer_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

    sampler_container = api.created[1][1]["spec"]["containers"][0]
    sampler_env = {item["name"]: item["value"] for item in sampler_container["env"] if "value" in item}
    self.assertEqual(sampler_env.get("BASE_MODEL"), "gemma-4-k8s")
    self.assertEqual(sampler_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

  def test_render_lora_pod_and_base_model_sharing(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:job-lora-1"] = json.dumps(
      {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "fine_tuning_type": "lora",
      }
    )
    s.kv_store["open_rl:model_meta:job-lora-2"] = json.dumps(
      {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "fine_tuning_type": "lora",
      }
    )
    api = _FakeCoreApi()
    manager = self._manager(api)

    with patch("server.store.get_store", return_value=s):
      manager.launch("job-lora-1")
      manager.launch_sampler("job-lora-1")

      # First call creates trainer & sampler pods targeting base_model 'qwen-qwen2-5-0-5b-1'
      self.assertEqual(len(api.created), 2)
      trainer_pod = api.created[0][1]
      sampler_pod = api.created[1][1]

      self.assertEqual(trainer_pod["metadata"]["name"], "open-rl-trainer-qwen-qwen2-5-0-5b-1")
      self.assertEqual(sampler_pod["metadata"]["name"], "open-rl-sampler-qwen-qwen2-5-0-5b-1")
      self.assertEqual(trainer_pod["metadata"]["labels"].get("accel-timeslicer"), "false")
      self.assertEqual(sampler_pod["metadata"]["labels"].get("accel-timeslicer"), "false")
      self.assertNotIn("timeslice.io/group", trainer_pod["metadata"]["labels"])

      sampler_container = sampler_pod["spec"]["containers"][0]
      self.assertEqual(sampler_container["command"], ["uv", "run", "python", "-u", "-m", "server.lora_sampler"])

      # Simulate live running status for existing base-model pods
      api.pod_phases["open-rl-trainer-qwen-qwen2-5-0-5b-1"] = "Running"
      api.pod_phases["open-rl-sampler-qwen-qwen2-5-0-5b-1"] = "Running"

      # Second call for job-lora-2 targeting the SAME base model reuses the running pods!
      manager.launch("job-lora-2")
      manager.launch_sampler("job-lora-2")
      self.assertEqual(len(api.created), 2)  # No new pods created!

  def test_lora_vs_fft_claims_without_node_selector_overrides(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:lora-1"] = json.dumps({"base_model": "qwen-base", "fine_tuning_type": "lora"})
    s.kv_store["open_rl:model_meta:fft-1"] = json.dumps({"base_model": "qwen-base", "fine_tuning_type": "full"})
    api = _FakeCoreApi()
    manager = self._manager(api)
    with patch("server.store.get_store", return_value=s):
      manager.launch_trainer("lora-1")
      manager.launch_trainer("fft-1")
      self.assertEqual(len(api.created), 2)
      lora_pod = api.created[0][1]
      fft_pod = api.created[1][1]
      self.assertNotIn("cloud.google.com/gke-accelerator", lora_pod["spec"].get("nodeSelector", {}))
      self.assertEqual(lora_pod["spec"]["resourceClaims"][0]["resourceClaimName"], "open-rl-lora-trainer-gpu-1")
      self.assertNotIn("cloud.google.com/gke-accelerator", fft_pod["spec"].get("nodeSelector", {}))
      self.assertEqual(fft_pod["spec"]["resourceClaims"][0]["resourceClaimName"], "open-rl-trainer-gpu-1")

  def test_label_based_claim_discovery_and_mutual_exclusion(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)

    # 1. Test error when no labeled claims are discovered and dynamic creation fails
    with (
      patch.object(manager, "_discover_eligible_claims", return_value=[]),
      patch.object(manager, "_create_managed_claim", side_effect=RuntimeError("No DRA claims available")),
      self.assertRaisesRegex(RuntimeError, "No DRA claims available"),
    ):
      manager.resolve_claim("lora", "trainer", "Qwen3-0.6B")

    # 2. Test LoRA mutual exclusion with discovered claims and active pod locks
    lora_claims = ["lora-gpu-1", "lora-gpu-2"]
    with patch.object(manager, "_discover_eligible_claims", return_value=lora_claims):
      # Initially no locks -> selects index 0
      self.assertEqual(manager.resolve_claim("lora", "trainer", "Qwen3-0.6B"), "lora-gpu-1")

      # Simulate lora-gpu-1 locked to qwen3-0-6b
      usage = {"lora-gpu-1": {"models": {"qwen3-0-6b"}, "workers": 1}}
      with patch.object(manager, "_get_claim_usage", return_value=usage):
        # Same base model -> affinity reuses lora-gpu-1
        self.assertEqual(manager.resolve_claim("lora", "trainer", "Qwen3-0.6B"), "lora-gpu-1")
        # Different base model -> mutual exclusion skips lora-gpu-1 and selects lora-gpu-2
        self.assertEqual(manager.resolve_claim("lora", "trainer", "Qwen3-8B"), "lora-gpu-2")

    # 3. Test FFT workload (no mutual exclusion)
    fft_claims = ["fft-gpu-1", "fft-gpu-2"]
    with (
      patch.object(manager, "_discover_eligible_claims", return_value=fft_claims),
      patch.object(manager, "_get_claim_usage", return_value={"fft-gpu-1": {"models": {"qwen3-0-6b"}, "workers": 1}}),
    ):
      # FFT trainer allows up to max_workers_per_claim (2) -> fft-gpu-1 has 1 worker (1/2), so reuses fft-gpu-1
      self.assertEqual(manager.resolve_claim("full", "trainer", "Qwen3-8B"), "fft-gpu-1")

  def test_claim_capacity_counts_workers_not_distinct_models(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)
    claims = ["fft-gpu-1", "fft-gpu-2"]

    # Two workers of the SAME base model fill a claim with max_workers_per_claim=2.
    # Counting distinct models here would see 1/2 and keep packing onto fft-gpu-1.
    saturated = {"fft-gpu-1": {"models": {"qwen3-8b"}, "workers": 2}}
    with (
      patch.object(manager, "_discover_eligible_claims", return_value=claims),
      patch.object(manager, "_get_claim_usage", return_value=saturated),
    ):
      self.assertEqual(manager.resolve_claim("full", "trainer", "Qwen3-8B"), "fft-gpu-2")

    # Base-model affinity must respect the same worker ceiling.
    with (
      patch.object(manager, "_discover_eligible_claims", return_value=["fft-gpu-1"]),
      patch.object(manager, "_get_claim_usage", return_value=saturated),
      patch.object(manager, "_create_managed_claim", return_value="fft-gpu-new") as create,
    ):
      self.assertEqual(manager.resolve_claim("full", "trainer", "Qwen3-8B"), "fft-gpu-new")
      create.assert_called_once()

  def test_claim_usage_counts_each_pod_separately(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)

    def _pod(name: str, claim: str, model: str, phase: str = "Running") -> dict[str, Any]:
      return {
        "metadata": {"name": name, "labels": {"open-rl.io/assigned-claim": claim, "open-rl.io/bound-base-model": model}},
        "status": {"phase": phase},
      }

    pods = {
      "items": [
        _pod("w1", "claim-a", "qwen3-8b"),
        _pod("w2", "claim-a", "qwen3-8b"),
        _pod("w3", "claim-a", "qwen3-0-6b"),
        _pod("w4", "claim-a", "qwen3-8b", phase="Succeeded"),
      ]
    }
    with patch.object(api, "list_namespaced_pod", return_value=pods, create=True):
      usage = manager._get_claim_usage()

    # Terminal pods are excluded; the two live qwen3-8b pods count twice.
    self.assertEqual(usage["claim-a"]["workers"], 3)
    self.assertEqual(usage["claim-a"]["models"], {"qwen3-8b", "qwen3-0-6b"})

  def test_dynamic_claim_is_released_when_pod_create_fails(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(500)
    manager = self._manager(api)

    with (
      patch.object(manager, "_discover_eligible_claims", return_value=[]),
      patch.object(manager, "_delete_managed_claim") as delete_claim,
      self.assertRaises(_ApiError),
    ):
      manager.launch("model-a")

    # The claim provisioned while rendering the pod must not outlive the failure.
    delete_claim.assert_called_once()
    self.assertTrue(delete_claim.call_args[0][0].startswith("open-rl-managed-full-trainer-"))

  def test_dynamic_claim_is_released_when_pod_create_conflicts(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(409)
    manager = self._manager(api)

    with (
      patch.object(manager, "_discover_eligible_claims", return_value=[]),
      patch.object(manager, "_delete_managed_claim") as delete_claim,
    ):
      manager.launch("model-a")  # 409 is tolerated, but the claim still leaks without cleanup

    delete_claim.assert_called_once()

  def test_successful_launch_keeps_its_dynamic_claim(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)

    with (
      patch.object(manager, "_discover_eligible_claims", return_value=[]),
      patch.object(manager, "_delete_managed_claim") as delete_claim,
    ):
      manager.launch("model-a")

    delete_claim.assert_not_called()
    self.assertEqual(manager._claims_created_this_launch, [])
    claim = api.created[0][1]["spec"]["resourceClaims"][0]["resourceClaimName"]
    self.assertTrue(claim.startswith("open-rl-managed-full-trainer-"))

  def test_concurrent_launches_provision_a_single_claim(self) -> None:
    import threading

    api = _FakeCoreApi()
    manager = self._manager(api)
    created_claims: list[str] = []
    barrier = threading.Barrier(4)

    real_create = manager._create_managed_claim

    def slow_create(workload_type: str, role: str, memory_tier: str) -> str:
      # Widen the resolve -> create window so an unsynchronized launch path would
      # reliably interleave and double-provision.
      time.sleep(0.05)
      name = real_create(workload_type, role, memory_tier)
      created_claims.append(name)
      return name

    def launch() -> None:
      barrier.wait()
      manager.launch("model-a")

    with (
      patch.object(manager, "_discover_eligible_claims", side_effect=lambda *a, **k: list(created_claims)),
      patch.object(manager, "_get_claim_usage", side_effect=lambda: {c: {"models": {"model-a"}, "workers": 1} for c in created_claims}),
      patch.object(manager, "_create_managed_claim", side_effect=slow_create),
    ):
      threads = [threading.Thread(target=launch) for _ in range(4)]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

    self.assertEqual(len(created_claims), 1, f"expected one dynamic claim, got {created_claims}")

  def _managed_claim(self, name: str, age_seconds: float) -> dict[str, Any]:
    import datetime

    created = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=age_seconds)
    return {"metadata": {"name": name, "creationTimestamp": created.strftime("%Y-%m-%dT%H:%M:%SZ")}}

  def _reconcile(self, manager: KubernetesWorkerManager, claims: list[dict], usage: dict, **kwargs) -> list[str]:
    """Run a reconcile pass over `claims`, returning the names it deleted."""
    with (
      patch.object(_FakeCustomObjectsApi, "list_namespaced_custom_object", return_value={"items": claims}),
      patch.object(manager, "_get_claim_usage", return_value=usage),
      patch.object(manager, "_delete_managed_claim") as delete_claim,
    ):
      deleted = manager.reconcile_managed_claims(**kwargs)
    # The returned names must be exactly the claims it actually issued deletes for.
    self.assertEqual(deleted, [call.args[0] for call in delete_claim.call_args_list])
    return deleted

  def test_reconcile_deletes_unreferenced_claims(self) -> None:
    manager = self._manager(_FakeCoreApi())
    claims = [self._managed_claim("claim-idle", age_seconds=600), self._managed_claim("claim-busy", age_seconds=600)]
    usage = {"claim-busy": {"models": {"qwen3-8b"}, "workers": 1}}

    deleted = self._reconcile(manager, claims, usage)

    self.assertEqual(deleted, ["claim-idle"])

  def test_reconcile_skips_claims_inside_the_grace_period(self) -> None:
    manager = self._manager(_FakeCoreApi())
    # A claim provisioned seconds ago whose Pod has not registered yet is
    # indistinguishable from an abandoned one; another replica may be mid-launch.
    claims = [self._managed_claim("claim-just-created", age_seconds=5)]

    self.assertEqual(self._reconcile(manager, claims, {}, min_age_seconds=120), [])
    # Past the grace period the same claim is collected.
    self.assertEqual(self._reconcile(manager, claims, {}, min_age_seconds=1), ["claim-just-created"])

  def test_reconcile_grace_period_is_configurable_by_env(self) -> None:
    manager = self._manager(_FakeCoreApi())
    claims = [self._managed_claim("claim-idle", age_seconds=60)]

    with patch.dict(os.environ, {**self.env, "OPEN_RL_CLAIM_RECONCILE_MIN_AGE_SECONDS": "600"}, clear=True):
      self.assertEqual(self._reconcile(manager, claims, {}), [])
    with patch.dict(os.environ, {**self.env, "OPEN_RL_CLAIM_RECONCILE_MIN_AGE_SECONDS": "10"}, clear=True):
      self.assertEqual(self._reconcile(manager, claims, {}), ["claim-idle"])

  def test_reconcile_excludes_launches_in_the_same_process(self) -> None:
    import threading

    manager = self._manager(_FakeCoreApi())
    order: list[str] = []
    launch_holds_lock = threading.Event()
    reconcile_attempted = threading.Event()

    def hold_lock() -> None:
      with manager._launch_lock:
        launch_holds_lock.set()
        reconcile_attempted.wait(timeout=2)
        time.sleep(0.05)
        order.append("launch-done")

    holder = threading.Thread(target=hold_lock)
    holder.start()
    self.assertTrue(launch_holds_lock.wait(timeout=2))

    def reconcile() -> None:
      reconcile_attempted.set()
      self._reconcile(manager, [self._managed_claim("claim-idle", age_seconds=600)], {})
      order.append("reconcile-done")

    reconciler = threading.Thread(target=reconcile)
    reconciler.start()
    holder.join(timeout=5)
    reconciler.join(timeout=5)

    # Reconciliation cannot inspect claims while a launch is provisioning one.
    self.assertEqual(order, ["launch-done", "reconcile-done"])

  def test_claim_age_tolerates_missing_or_bad_timestamps(self) -> None:
    self.assertEqual(KubernetesWorkerManager._claim_age_seconds({}), float("inf"))
    self.assertEqual(KubernetesWorkerManager._claim_age_seconds({"creationTimestamp": "not-a-date"}), float("inf"))
    self.assertLess(KubernetesWorkerManager._claim_age_seconds(self._managed_claim("c", 30)["metadata"]) - 30, 5)

  def test_requires_template_and_redis(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True), self.assertRaisesRegex(RuntimeError, "POD_TEMPLATE"):
      KubernetesWorkerManager(core_api=_FakeCoreApi())
    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name}, clear=True),
      self.assertRaisesRegex(RuntimeError, "REDIS_URL"),
    ):
      KubernetesWorkerManager(core_api=_FakeCoreApi())

  def test_sanitize_job_id(self) -> None:
    self.assertEqual(sanitize_job_id("Model_A.1"), "model-a-1")
    self.assertEqual(sanitize_job_id("a" * 80), "a" * 63)
    with self.assertRaises(ValueError):
      sanitize_job_id("___")


class CreateWorkerManagerTest(unittest.TestCase):
  def test_default_launcher_is_subprocess(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True):
      manager = create_worker_manager()
    self.assertIsInstance(manager, LocalWorkerManager)

  def test_kubernetes_launcher_is_selected_by_env(self) -> None:
    env = {"REDIS_URL": "redis://r:6379", "OPEN_RL_WORKER_MANAGER": "kubernetes"}
    with (
      patch.dict(os.environ, env, clear=True),
      patch("server.k8s_worker_manager.KubernetesWorkerManager") as manager_cls,
    ):
      manager = create_worker_manager()
    self.assertIs(manager, manager_cls.return_value)


if __name__ == "__main__":
  unittest.main()

if __name__ == "__main__":
  unittest.main()
