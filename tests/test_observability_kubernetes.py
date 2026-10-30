import asyncio
import unittest
from unittest import mock

from server.observability.kubernetes import gpu_capacity, kubernetes, problem


class NodeTest(unittest.TestCase):
  def test_a_dra_node_counts_the_gpus_in_its_resource_slices(self):
    node = {"metadata": {"name": "n1"}, "status": {"capacity": {"cpu": "200"}}}

    def slice_on(on: str, devices: int, generation: int = 2, driver: str = "gpu.nvidia.com") -> dict:
      return {"spec": {"driver": driver, "nodeName": on, "pool": {"name": on, "generation": generation}, "devices": [{}] * devices}}

    slices = [slice_on("n1", 8), slice_on("n1", 8, generation=1), slice_on("n2", 4), slice_on("n1", 4, driver="tpu.google.com")]
    self.assertEqual((gpu_capacity(node, []), gpu_capacity(node, slices)), (0, 8))


class ProblemTest(unittest.TestCase):
  def test_a_pod_status_names_what_keeps_it_from_running(self):
    crashing = {"phase": "Running", "containerStatuses": [{"state": {"waiting": {"reason": "CrashLoopBackOff", "message": "back-off 5m0s"}}}]}
    pending = {"phase": "Pending", "conditions": [{"type": "PodScheduled", "status": "False", "message": "0/3 nodes are available"}]}
    completed = {"phase": "Succeeded", "containerStatuses": [{"state": {"terminated": {"reason": "Completed", "exitCode": 0}}}]}

    self.assertEqual(problem(crashing), "CrashLoopBackOff: back-off 5m0s")
    self.assertEqual(problem(pending), "Unschedulable: 0/3 nodes are available")
    self.assertIsNone(problem(completed))


class KubernetesTest(unittest.TestCase):
  def test_a_cluster_without_a_client_is_reported_as_unavailable(self):
    with mock.patch("server.observability.kubernetes.kubernetes_api", side_effect=RuntimeError("no kubeconfig")):
      cluster = asyncio.run(kubernetes())

    self.assertFalse(cluster["available"])
    self.assertEqual(cluster["errors"], ["Kubernetes unavailable: no kubeconfig"])


if __name__ == "__main__":
  unittest.main()
