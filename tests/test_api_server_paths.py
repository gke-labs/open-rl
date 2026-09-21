import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import api_server
from server.store import InMemoryStateStore, InMemoryStore


class ApiServerTest(unittest.TestCase):
  """Requests go through the ASGI app, so routing, body validation and the
  error handlers are the ones the tinker client sees."""

  def setUp(self) -> None:
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.client = TestClient(api_server.app)

  def post(self, path: str, body: dict, **kwargs):
    return self.client.post(f"/api/v1/{path}", json=body, **kwargs)

  def post_bytes(self, path: str, body: bytes, media_type: str):
    return self.client.post(f"/api/v1/{path}", content=body, headers={"Content-Type": media_type})

  def queued(self) -> list[dict]:
    return asyncio.run(api_server.store.get_requests())


class GetInfoTest(ApiServerTest):
  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = self.post("get_info", {"model_id": "model-a"}).json()

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_prefers_the_models_own_base_model(self) -> None:
    meta = json.dumps({"base_model": "google/gemma-4-e2b", "fine_tuning_type": "full"})
    asyncio.run(api_server.state.set_value("open_rl:model_meta:model-g", meta))
    with patch.dict(os.environ, {"BASE_MODEL": "Qwen/Qwen2.5-0.5B"}, clear=True):
      info = self.post("get_info", {"model_id": "model-g"}).json()
      via_sampler_ref = self.post("get_info", {"model_id": "tinker://model-g/sampler_weights/sampler-1"}).json()
      unknown = self.post("get_info", {"model_id": "model-unknown"}).json()

    # The client loads its tokenizer from this name, so it must be the job's model.
    self.assertEqual(info["model_name"], "google/gemma-4-e2b")
    self.assertEqual(info["model_data"]["tokenizer_id"], "google/gemma-4-e2b")
    self.assertEqual(via_sampler_ref["model_name"], "google/gemma-4-e2b")
    self.assertEqual(unknown["model_name"], "Qwen/Qwen2.5-0.5B")

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = self.post("get_info", {"model_id": "model-a"})
    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.json(), {"error": "No base model is configured"})

  def test_create_model_requires_base_model_payload(self) -> None:
    response = self.post("create_model", {})
    self.assertEqual(response.status_code, 422)
    self.assertEqual(response.json()["detail"][0]["loc"], ["body", "base_model"])

  def test_a_missing_model_id_is_a_422_that_names_it(self) -> None:
    for path in ("delete_model", "optim_step", "save_weights", "save_weights_for_sampler", "load_weights"):
      with self.subTest(path=path):
        response = self.post(path, {})
        self.assertEqual(response.status_code, 422)
        self.assertIn(["body", "model_id"], [error["loc"] for error in response.json()["detail"]])

  def test_create_model_accepts_base_model_payload(self) -> None:
    model_id = self.post("create_model", {"base_model": "my-model"}).json()["request_id"]
    queued = self.queued()
    self.assertEqual(queued[0]["model_id"], model_id)
    self.assertEqual(queued[0]["payload"]["base_model"], "my-model")
    meta = json.loads(api_server.state.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "my-model")


class ErrorShapeTest(ApiServerTest):
  """Every refused request answers {"error": ...}, whichever layer refused it."""

  def test_an_unknown_route_answers_the_shared_error_shape(self) -> None:
    response = self.client.get("/api/v1/no_such_endpoint")
    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.json(), {"error": "Not Found"})

  def test_a_body_of_the_wrong_shape_is_a_422_that_names_the_field(self) -> None:
    response = self.post("get_info", {"model_id": ["not", "a", "string"]})
    self.assertEqual(response.status_code, 422)
    body = response.json()
    self.assertEqual(body["error"], "invalid request")
    self.assertEqual(body["detail"][0]["loc"], ["body", "model_id"])
    self.assertEqual(body["detail"][0]["input"], ["not", "a", "string"])

  def test_a_binary_body_on_a_json_route_is_a_422_not_a_500(self) -> None:
    response = self.post_bytes("optim_step", b"\x8a\xff", "application/x-protobuf")
    self.assertEqual(response.status_code, 422)
    self.assertEqual(response.json()["detail"][0]["input"], "<2 bytes>")

  def test_http_errors_keep_their_headers_and_body_rules(self) -> None:
    from fastapi import HTTPException

    throttled = asyncio.run(api_server.http_error(None, HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "3"})))
    self.assertEqual((throttled.status_code, throttled.headers["retry-after"]), (429, "3"))
    self.assertEqual(json.loads(throttled.body), {"error": "slow down"})
    unchanged = asyncio.run(api_server.http_error(None, HTTPException(status_code=304)))
    self.assertEqual((unchanged.status_code, unchanged.body), (304, b""))


class SaveSeqIdZeroTest(ApiServerTest):
  def test_the_first_saves_zero_seq_id_is_kept(self) -> None:
    # The client's counter is 0-based; 0 must not fall back to a timestamp id.
    self.post("save_weights_for_sampler", {"model_id": "job-a", "sampling_session_seq_id": 0})
    self.post("save_weights", {"model_id": "job-a", "seq_id": 0})
    queued = self.queued()
    self.assertEqual(queued[0]["payload"]["sampling_session_id"], "tinker://job-a/sampler_weights/sampler-0")
    self.assertTrue(queued[1]["payload"]["state_path"].endswith("job-a-samp-0"))


class ApiServerPathTest(ApiServerTest):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    old_tmp_dir = api_server.TMP_DIR
    with tempfile.TemporaryDirectory() as tmp_dir:
      api_server.TMP_DIR = tmp_dir
      self.addCleanup(setattr, api_server, "TMP_DIR", old_tmp_dir)

      self.assertEqual(
        api_server.checkpoint_state_path("job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        api_server.checkpoint_state_path("job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_a_tinker_path_names_the_model_that_saved_it(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(api_server, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      self.assertEqual(api_server.tinker_state_path(state_dir), "tinker://job-a/weights/step-5")
      # A resuming job passes the dead job's path under its own model id.
      self.assertEqual(api_server.checkpoint_state_path("job-b", "tinker://job-a/weights/step-5"), state_dir)
      self.assertEqual(api_server.tinker_state_path("/elsewhere/final"), "/elsewhere/final")
      # Only weights paths are checkpoints. A sampler path is refused, not resolved under the caller.
      self.assertIsNone(api_server.tinker_checkpoint_dir("tinker://job-a/sampler_weights/sampler-3"))
      refused = self.post("load_weights", {"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"})
      self.assertEqual(refused.status_code, 400)
      self.assertIn("is not a tinker://<model>/weights/<name> path", refused.json()["error"])

  def test_save_state_keeps_the_optimizer_and_answers_with_a_tinker_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(api_server, "TMP_DIR", tmp_dir):
      self.post("save_weights", {"model_id": "job-a", "path": "step-5"})
      queued = self.queued()
      self.assertEqual(queued[0]["op"], "save_state")
      self.assertTrue(queued[0]["payload"]["include_optimizer"])
      saved = api_server.translate_future_result({"type": "state_saved", "path": queued[0]["payload"]["state_path"]})
    self.assertEqual(saved, {"type": "save_weights", "path": "tinker://job-a/weights/step-5"})

  def test_weights_info_reads_the_checkpoint_on_disk(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(api_server, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      os.makedirs(os.path.join(state_dir, "job-a"))
      with open(os.path.join(state_dir, "metadata.json"), "w") as f:
        json.dump({"base_model": "google/gemma-4-e2b", "model_id": "job-a", "has_optimizer": True}, f)
      with open(os.path.join(state_dir, "job-a", "adapter_config.json"), "w") as f:
        json.dump({"r": 8}, f)
      info = self.post("weights_info", {"tinker_path": "tinker://job-a/weights/step-5"}).json()
      missing = self.post("weights_info", {"tinker_path": "tinker://job-a/weights/never"})
    self.assertEqual(info["base_model"], "google/gemma-4-e2b")
    self.assertTrue(info["is_lora"])
    self.assertEqual(info["lora_rank"], 8)
    self.assertEqual(missing.status_code, 404)
    self.assertEqual(missing.json(), {"error": "No checkpoint at tinker://job-a/weights/never"})

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(api_server.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


class ProtobufWireTest(unittest.TestCase):
  """Tinker SDK >= 0.25 sends forward_backward as protobuf and only reads
  forward_backward and sample results as protobuf."""

  FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "fwdbwd_request_tinker_0.29.0.pb")

  def setUp(self) -> None:
    from fastapi.testclient import TestClient

    patcher = patch.object(api_server, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.client = TestClient(api_server.app)

  def _queued(self) -> list[dict]:
    return asyncio.run(api_server.store.get_requests())

  def test_protobuf_and_json_forward_backward_queue_the_same_request(self) -> None:
    with open(self.FIXTURE, "rb") as fh:
      body = fh.read()
    proto = self.client.post("/api/v1/forward_backward", content=body, headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(proto.status_code, 200, proto.text)
    from_proto = self._queued()[0]

    with open(self.FIXTURE[:-3] + ".json") as fh:
      json_body = json.load(fh)
    as_json = self.client.post("/api/v1/forward_backward", json=json_body)
    self.assertEqual(as_json.status_code, 200, as_json.text)
    from_json = self._queued()[0]

    self.assertEqual(from_proto["op"], "forward_backward")
    self.assertEqual(from_proto["model_id"], "model-abc")
    self.assertEqual(from_proto["payload"], from_json["payload"])
    self.assertEqual(from_proto["payload"]["loss_fn"], "importance_sampling")
    self.assertEqual(from_proto["payload"]["loss_config"], {"clip_range": 0.2, "kl_coeff": 0.01, "mode": "token"})

  def test_forward_only_reaches_the_worker_from_both_routes(self) -> None:
    from server.proto import tinker_public_pb2 as pb

    msg = pb.ForwardBackwardRequest(model_id="model-abc", seq_id=1, loss_fn="cross_entropy", forward_only=True)
    response = self.client.post("/api/v1/forward_backward", content=msg.SerializeToString(), headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(response.status_code, 200, response.text)
    queued = self._queued()[0]
    self.assertEqual(queued["op"], "forward_backward")
    self.assertTrue(queued["payload"]["forward_only"])

    legacy = self.client.post("/api/v1/forward", json={"model_id": "model-abc", "forward_input": {"data": [], "loss_fn": "cross_entropy"}})
    self.assertEqual(legacy.status_code, 200, legacy.text)
    self.assertTrue(self._queued()[0]["payload"]["forward_only"])

    train = self.client.post(
      "/api/v1/forward_backward", json={"model_id": "model-abc", "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"}}
    )
    self.assertEqual(train.status_code, 200, train.text)
    self.assertFalse(self._queued()[0]["payload"]["forward_only"])

  def test_bad_bodies_are_client_errors_not_500s(self) -> None:
    garbage = self.client.post("/api/v1/forward_backward", content=b"\xff\xfe not proto", headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(garbage.status_code, 400, garbage.text)
    compressed = self.client.post(
      "/api/v1/forward_backward", content=b"x", headers={"Content-Type": "application/x-protobuf", "Content-Encoding": "zstd"}
    )
    self.assertEqual(compressed.status_code, 415, compressed.text)
    other = self.client.post("/api/v1/forward_backward", content=b"x", headers={"Content-Type": "text/plain"})
    self.assertEqual(other.status_code, 415, other.text)
    # A non-JSON body on a plain `req: dict` route used to crash FastAPI's 422
    # handler while it JSON-encoded the raw bytes, turning it into a 500.
    binary_to_dict_route = self.client.post("/api/v1/optim_step", content=b"\x8a\xff", headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(binary_to_dict_route.status_code, 422, binary_to_dict_route.text)
    self.assertNotIn("\x8a", binary_to_dict_route.text)

  def test_retrieve_future_answers_protobuf_only_when_asked_and_only_for_proto_types(self) -> None:
    from server.proto import tinker_public_pb2 as pb

    asyncio.run(
      api_server.store.set_future(
        "samp-1", {"type": "sample_completed", "sequences": [{"tokens": [1, 2], "logprobs": [-0.5, -1.0], "stop_reason": "stop"}]}
      )
    )
    asyncio.run(api_server.store.set_future("optim-1", {"type": "optim_step_completed", "metrics": {"grad_norm:mean": 0.0}}))

    as_json = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"})
    self.assertEqual(as_json.status_code, 200)
    self.assertTrue(as_json.headers["content-type"].startswith("application/json"))
    self.assertEqual(as_json.json()["type"], "sample")

    as_proto = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"}, headers={"Accept": "application/x-protobuf"})
    self.assertEqual(as_proto.status_code, 200)
    self.assertEqual(as_proto.headers["content-type"], "application/x-protobuf")
    msg = pb.SampleResponse()
    msg.ParseFromString(as_proto.content)
    self.assertEqual(msg.sequences[0].stop_reason, pb.STOP_REASON_STOP)

    optim = self.client.post("/api/v1/retrieve_future", json={"request_id": "optim-1"}, headers={"Accept": "application/x-protobuf"})
    self.assertTrue(optim.headers["content-type"].startswith("application/json"))
    self.assertEqual(optim.json()["type"], "optim_step")


class SampleSequenceIdsTest(ApiServerTest):
  """Tinker SDK >= 0.25 asserts that every asample promise carries one
  sequence id per requested sample."""

  def test_asample_promise_carries_one_id_per_sample(self) -> None:
    with patch.object(api_server, "get_sampler_backend", return_value="torch"):
      promise = self.post("asample", {"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1, 2]}]}, "num_samples": 3}).json()
    self.assertEqual(len(promise["sample_sequence_ids"]), 3)
    self.assertEqual(len(set(promise["sample_sequence_ids"])), 3)
    self.assertTrue(all(sid.startswith(promise["request_id"]) for sid in promise["sample_sequence_ids"]))

  def test_asample_defaults_to_a_single_sample(self) -> None:
    with patch.object(api_server, "get_sampler_backend", return_value="torch"):
      promise = self.post("asample", {"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1]}]}}).json()
    self.assertEqual(len(promise["sample_sequence_ids"]), 1)


if __name__ == "__main__":
  unittest.main()


class InputBoundaryTest(ApiServerTest):
  def test_invalid_training_datum_is_a_validation_error(self) -> None:
    response = self.post(
      "forward_backward",
      {"model_id": "m", "forward_backward_input": {"data": [{"model_input": {"chunks": [{"tokens": ["bad"]}]}, "loss_fn_inputs": {}}]}},
    )
    self.assertEqual(response.status_code, 422)
    self.assertEqual(api_server.store.queues, {})

  def test_null_configs_and_sampling_defaults_preserve_zero(self) -> None:
    response = self.post("create_model", {"base_model": "base", "lora_config": None, "full_config": None})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(self.queued()[0]["payload"]["lora_config"]["rank"], 16)
    for value in (None, 0):
      with self.subTest(value=value), patch.object(api_server, "get_sampler_backend", return_value="torch"):
        response = self.post(
          "asample", {"model_id": "base", "prompt": {"chunks": [{"tokens": [1]}]}, "sampling_params": {"temperature": value, "max_tokens": value}}
        )
        self.assertEqual(response.status_code, 200)
        queued = self.queued()[0]["payload"]
        self.assertEqual(queued["temperature"], 1.0 if value is None else 0)
        self.assertEqual(queued["max_tokens"], 20 if value is None else 0)


class RestoreRoutingTest(ApiServerTest):
  def test_restore_uses_checkpoint_identity(self):
    for kind in ("lora", "full"):
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}):
        with open(os.path.join(directory, "metadata.json"), "w") as f:
          json.dump({"base_model": "checkpoint-base"}, f)
        if kind == "lora":
          with open(os.path.join(directory, "adapter_config.json"), "w") as f:
            json.dump({"r": 8}, f)
        response = self.post("create_model_from_state", {"state_path": directory})
        self.assertEqual(response.status_code, 200)
        model_id = response.json()["request_id"]
        metadata = json.loads(api_server.state.get_value_sync(f"open_rl:model_meta:{model_id}"))
        self.assertEqual((metadata["base_model"], metadata["fine_tuning_type"]), ("checkpoint-base", kind))
        self.assertEqual(self.queued()[0]["payload"]["fine_tuning_type"], kind)
