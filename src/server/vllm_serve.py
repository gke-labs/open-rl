"""Starts `vllm serve` for a routed pool replica, with the engine settings computed on this pod for its GPU."""

import json
import os

from server.vllm_options import serve_args


def main() -> None:
  base_model = os.environ["BASE_MODEL"]
  # The pool's EPP subscribes to these events at llm-d's default port and topic prefix; the topic names the pod and the served model.
  topic = f"kv@{os.environ['POD_IP']}:8000@{base_model}"
  kv_events = {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": "tcp://*:5557", "topic": topic}
  args = serve_args(base_model, os.environ.get("OPEN_RL_FINE_TUNING_TYPE") == "lora")
  os.execvp("vllm", ["vllm", "serve", *args, "--kv-events-config", json.dumps(kv_events)])


if __name__ == "__main__":
  main()
