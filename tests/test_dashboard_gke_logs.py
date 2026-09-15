import unittest
from unittest.mock import patch

from server.dashboard import gke

CONFIG = {"project": "proj", "location": "us-central1", "cluster": "c1", "mode": "auto", "enabled": True, "configured": True}
SOURCE = {"pod": "orw-a", "node": "n1", "role": "trainer", "created_at": "2026-09-10T10:00:00+00:00", "until": "2026-09-10T11:00:00+00:00"}


def entry(pod="orw-a", at="2026-09-10T10:30:00Z", **extra):
  return {
    "timestamp": at,
    "insertId": "i1",
    "logName": "projects/proj/logs/stdout",
    "severity": "INFO",
    "resource": {"labels": {"pod_name": pod, "container_name": "worker"}},
    **extra,
  }


class GkeLogHelpersTest(unittest.TestCase):
  def setUp(self) -> None:
    for module, name, value in ((gke, "configuration", lambda: CONFIG), (gke.cluster, "namespace", lambda: "openrl-system")):
      patcher = patch.object(module, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_filter_scopes_to_cluster_sources_window_and_search(self) -> None:
    text = gke.logs_filter([SOURCE], "2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00", "UNKNOWN", "oom")
    self.assertIn('resource.type="k8s_container"', text)
    self.assertIn('resource.labels.cluster_name="c1"', text)
    self.assertIn('resource.labels.namespace_name="openrl-system"', text)
    self.assertIn('resource.labels.pod_name="orw-a"', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)

  def test_entry_becomes_a_record_only_inside_a_source_lifetime(self) -> None:
    record = gke.entry_record(entry(textPayload="hello"), [SOURCE], "run-1")
    self.assertEqual((record["pod"], record["container"], record["role"], record["message"]), ("orw-a", "worker", "trainer", "hello"))
    self.assertIsNone(gke.entry_record(entry(at="2026-09-10T12:00:00Z", textPayload="late"), [SOURCE], "run-1"))
    self.assertIsNone(gke.entry_record(entry(pod="someone-else", textPayload="x"), [SOURCE], "run-1"))

  def test_json_payload_is_rendered_when_there_is_no_text(self) -> None:
    record = gke.entry_record(entry(jsonPayload={"message": "m", "rank": 2}), [SOURCE], "run-1")
    self.assertEqual(record["rank"], 2)
    self.assertIn('"message": "m"', record["message"])

  def test_pod_sources_cover_live_pods_and_recorded_placements(self) -> None:
    run = {"run_id": "run-1", "pods": [{"name": "orw-live", "node": "n1", "role": "sampler", "created_at": "2026-09-10T10:00:00Z"}]}
    past = [
      {"pod": "orw-gone", "node": "n2", "role": "trainer", "run_ids": ["run-1"], "first_seen": 1789000000.0, "last_seen": 1789003600.0},
      {"pod": "orw-other", "node": "n3", "role": "trainer", "run_ids": ["run-2"], "first_seen": 1789000000.0, "last_seen": 1789003600.0},
    ]
    sources = {s["pod"]: s for s in gke.pod_sources(run, past, "2026-09-10T11:00:00Z")}
    self.assertEqual(set(sources), {"orw-live", "orw-gone"})
    self.assertEqual(sources["orw-live"]["until"], "2026-09-10T11:00:00.000000+00:00")
    self.assertEqual(sources["orw-gone"]["role"], "trainer")
    self.assertGreater(sources["orw-gone"]["until"], sources["orw-gone"]["created_at"])
