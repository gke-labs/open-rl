import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider

from server import api_server
from server.store import InMemoryStore
from tests.api_client import runtime_context
from training import commands


class ApiServerTest(unittest.TestCase):
  """Requests go through the ASGI app, so routing, body validation and the
  error handlers are the ones the tinker client sees."""

  def setUp(self) -> None:
    self.runtime = self.enterContext(runtime_context())
    self.client = TestClient(api_server.app)

  def post(self, path: str, body: dict, **kwargs):
    return self.client.post(f"/api/v1/{path}", json=body, **kwargs)

  def post_bytes(self, path: str, body: bytes, media_type: str):
    return self.client.post(f"/api/v1/{path}", content=body, headers={"Content-Type": media_type})

  def queued(self) -> list[dict]:
    return asyncio.run(self.runtime.store.get_requests())


class GetInfoTest(ApiServerTest):
  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = self.post("get_info", {"model_id": "model-a"}).json()

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_prefers_the_models_own_base_model(self) -> None:
    meta = json.dumps({"base_model": "google/gemma-4-e2b", "fine_tuning_type": "full"})
    asyncio.run(self.runtime.store.set_value("open_rl:model_meta:model-g", meta))
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
    self.assertEqual(queued[0]["op"], "create_model")
    self.assertEqual(queued[0]["base_model"], "my-model")
    meta = json.loads(self.runtime.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
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
    self.assertEqual(queued[0]["sampling_session_id"], "tinker://job-a/sampler_weights/sampler-0")
    self.assertTrue(queued[1]["state_path"].endswith("job-a-samp-0"))


class ApiServerPathTest(ApiServerTest):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      self.runtime.checkpoint_root = os.path.join(tmp_dir, "checkpoints")

      self.assertEqual(
        api_server.checkpoint_path(self.runtime.checkpoint_root, "job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        api_server.checkpoint_path(self.runtime.checkpoint_root, "job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_a_tinker_path_names_the_model_that_saved_it(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(self.runtime, "checkpoint_root", os.path.join(tmp_dir, "checkpoints")):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      self.assertEqual(api_server.checkpoint_uri(self.runtime.checkpoint_root, state_dir), "tinker://job-a/weights/step-5")
      # A resuming job passes the dead job's path under its own model id.
      self.assertEqual(api_server.checkpoint_path(self.runtime.checkpoint_root, "job-b", "tinker://job-a/weights/step-5"), state_dir)
      self.assertEqual(api_server.checkpoint_uri(self.runtime.checkpoint_root, "/elsewhere/final"), "/elsewhere/final")
      # Only weights paths are checkpoints. A sampler path is refused, not resolved under the caller.
      self.assertIsNone(api_server.checkpoint_from_uri(self.runtime.checkpoint_root, "tinker://job-a/sampler_weights/sampler-3"))
      refused = self.post("load_weights", {"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"})
      self.assertEqual(refused.status_code, 400)
      self.assertIn("is not a tinker://<model>/weights/<name> path", refused.json()["error"])
      refused_save = self.post("save_weights", {"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"})
      self.assertEqual(refused_save.status_code, 400)

  def test_save_state_keeps_the_optimizer_and_answers_with_a_tinker_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(self.runtime, "checkpoint_root", os.path.join(tmp_dir, "checkpoints")):
      self.post("save_weights", {"model_id": "job-a", "path": "step-5"})
      queued = self.queued()
      self.assertEqual(queued[0]["op"], "save_state")
      self.assertTrue(queued[0]["include_optimizer"])
      saved = api_server.translate_future_result({"type": "state_saved", "path": queued[0]["state_path"]}, self.runtime.checkpoint_root)
    self.assertEqual(saved, {"type": "save_weights", "path": "tinker://job-a/weights/step-5"})

  def test_weights_info_reads_the_checkpoint_on_disk(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(self.runtime, "checkpoint_root", os.path.join(tmp_dir, "checkpoints")):
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
    self.assertEqual(api_server.checkpoint_path(self.runtime.checkpoint_root, "job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


class ProtobufWireTest(unittest.TestCase):
  """Tinker SDK >= 0.25 sends forward_backward as protobuf and only reads
  forward_backward and sample results as protobuf."""

  FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "fwdbwd_request_tinker_0.29.0.pb")

  def setUp(self) -> None:
    from fastapi.testclient import TestClient

    self.runtime = self.enterContext(runtime_context())
    self.client = TestClient(api_server.app)

  def _queued(self) -> list[dict]:
    return asyncio.run(self.runtime.store.get_requests())

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
    self.assertEqual(
      {k: v for k, v in from_proto.items() if k not in ("request_id", "trace_context")},
      {k: v for k, v in from_json.items() if k not in ("request_id", "trace_context")},
    )
    self.assertEqual(from_proto["loss_fn"], "importance_sampling")
    self.assertEqual(from_proto["loss_config"], {"clip_range": 0.2, "kl_coeff": 0.01, "mode": "token"})

  def test_forward_only_reaches_the_worker_from_both_routes(self) -> None:
    from server.proto import tinker_public_pb2 as pb

    msg = pb.ForwardBackwardRequest(model_id="model-abc", seq_id=1, loss_fn="cross_entropy", forward_only=True)
    response = self.client.post("/api/v1/forward_backward", content=msg.SerializeToString(), headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(response.status_code, 200, response.text)
    queued = self._queued()[0]
    self.assertEqual(queued["op"], "forward_backward")
    self.assertTrue(queued["forward_only"])

    legacy = self.client.post("/api/v1/forward", json={"model_id": "model-abc", "forward_input": {"data": [], "loss_fn": "cross_entropy"}})
    self.assertEqual(legacy.status_code, 200, legacy.text)
    self.assertTrue(self._queued()[0]["forward_only"])

    train = self.client.post(
      "/api/v1/forward_backward", json={"model_id": "model-abc", "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"}}
    )
    self.assertEqual(train.status_code, 200, train.text)
    self.assertFalse(self._queued()[0]["forward_only"])

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
      self.runtime.store.set_future(
        "samp-1", {"type": "sample_completed", "sequences": [{"tokens": [1, 2], "logprobs": [-0.5, -1.0], "stop_reason": "stop"}]}
      )
    )
    asyncio.run(self.runtime.store.set_future("optim-1", {"type": "optim_step_completed", "metrics": {"grad_norm:mean": 0.0}}))

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

  def test_vllm_submission_carries_http_trace_without_future_registration(self) -> None:
    trace_id = "1234567890abcdef1234567890abcdef"
    with patch.object(api_server, "get_sampler_backend", return_value="vllm"):
      response = self.post(
        "asample",
        {"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1, 2]}]}, "num_samples": 2},
        headers={"traceparent": f"00-{trace_id}-1234567890abcdef-01"},
      )
    self.assertEqual(response.status_code, 200)
    promise = response.json()
    queued = asyncio.run(self.runtime.store.get_sampling_requests_for_model("job-a"))[0]
    self.assertEqual(queued["request_id"], promise["request_id"])
    self.assertNotIn(promise["request_id"], self.runtime.store.futures_store)
    self.assertEqual(queued["prompt_token_ids"], [1, 2])
    self.assertEqual(len(promise["sample_sequence_ids"]), 2)
    context = trace.get_current_span(propagate.extract(queued["trace_context"])).get_span_context()
    self.assertEqual(context.trace_id, int(trace_id, 16))


class QueueTraceContextTest(unittest.IsolatedAsyncioTestCase):
  async def test_submissions_capture_current_context_without_mutation_or_leakage(self) -> None:
    provider = TracerProvider()
    self.addCleanup(provider.shutdown)
    tracer = provider.get_tracer(__name__)
    store = InMemoryStore()
    self.runtime = self.enterContext(runtime_context(store))

    async def submit(queue, request_id):
      if queue == "training":
        command = commands.OptimStep(request_id=request_id, model_id="model", trace_context={"old": "context"})
        returned_id = await self.runtime.submit(command)
        self.assertEqual(command.trace_context, {"old": "context"})
        raw = (await store.get_requests())[0]
        commands.parse_command(raw)
      else:
        request = {"request_id": request_id, "model_id": "model", "trace_context": {"old": "context"}}
        returned_id = await api_server.enqueue_sampling(self.runtime, request)
        self.assertEqual(request["trace_context"], {"old": "context"})
        raw = (await store.get_sampling_requests_for_model("model"))[0]
      self.assertEqual(returned_id, request_id)
      self.assertNotIn(request_id, store.futures_store)
      return raw["trace_context"]

    for queue in ("training", "sampling"):
      for index in range(2):
        with tracer.start_as_current_span(f"request-{index}") as parent:
          carrier = await submit(queue, f"{queue}-{index}")
          extracted = trace.get_current_span(propagate.extract(carrier)).get_span_context()
          self.assertEqual(extracted.trace_id, parent.get_span_context().trace_id)
          self.assertEqual(extracted.span_id, parent.get_span_context().span_id)
      self.assertEqual(await submit(queue, f"{queue}-untraced"), {})


if __name__ == "__main__":
  unittest.main()
