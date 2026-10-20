# Cold start timeline: after-C1-8b-h100-cold

t=0 is the client's `create_model` call at 2026-10-15T05:04:58.139+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen3-8b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 05:04:58.000 | -0.1 |  | Workload creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 05:04:58.742 | 0.6 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 05:04:58.742 | 0.6 |  | workloads.tsv poller: status.podName set |
| trainer.pod_created | 05:04:59.000 | 0.9 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 05:04:59.000 | 0.9 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 05:04:59.000 | 0.9 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 05:04:59.000 | 0.9 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.workload_reports_node | 05:05:00.362 | 2.2 |  | workloads.tsv poller: status.nodeName set |
| trainer.event_Pulling | 05:05:02.000 | 3.9 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" |
| trainer.pod_Ready | 05:06:29.000 | 90.9 |  | pod condition Ready |
| trainer.pod_ContainersReady | 05:06:29.000 | 90.9 |  | pod condition ContainersReady |
| trainer.container_started | 05:06:29.000 | 90.9 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 05:06:29.000 | 90.9 | 86.476 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" in 1m26.476s (1m26.47 |
| trainer.event_Created | 05:06:29.000 | 90.9 |  | event: Container created |
| trainer.event_Started | 05:06:29.000 | 90.9 |  | event: Container started |
| trainer.condition_Placed_True | 05:06:30.000 | 91.9 |  | Workload condition Placed |
| trainer.process_first_log | 05:06:33.850 | 95.7 |  | container log, first line |
| trainer.load_start | 05:06:34.074 | 95.9 |  | container log: Loading base model Qwen/Qwen3-8B to cuda:0... |
| trainer.ready_key | 05:06:47.000 | 108.9 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen3-8B |
| trainer.load_done | 05:06:47.108 | 109.0 |  | container log: Successfully loaded. |
| trainer.processor_started | 05:06:47.110 | 109.0 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen3-8B active_set=Qwen/Qwen3-8B-1). |

## sampler: `orw-lora-qwen-qwen3-8b-0-sampler` on `gke-open-rl-dra-h100-sampler-a721f127-0853`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_first_seen_by_poller | 05:06:56.988 | 118.8 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 05:06:56.988 | 118.8 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 05:06:56.988 | 118.8 |  | workloads.tsv poller: status.nodeName set |
| sampler.workload_created | 05:06:57.000 | 118.9 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 05:06:58.000 | 119.9 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 05:06:58.000 | 119.9 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 05:06:58.000 | 119.9 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 05:06:58.000 | 119.9 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-sampler to gke-open-rl-dra-h100-sampler-a721f127-0853 |
| sampler.event_Pulling | 05:07:01.000 | 122.9 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" |
| sampler.condition_Placed_True | 05:08:28.000 | 209.9 |  | Workload condition Placed |
| sampler.pod_Ready | 05:08:28.000 | 209.9 |  | pod condition Ready |
| sampler.pod_ContainersReady | 05:08:28.000 | 209.9 |  | pod condition ContainersReady |
| sampler.container_started | 05:08:28.000 | 209.9 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 05:08:28.000 | 209.9 | 86.459 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" in 1m26.459s (1m26.45 |
| sampler.event_Created | 05:08:28.000 | 209.9 |  | event: Container created |
| sampler.event_Started | 05:08:28.000 | 209.9 |  | event: Container started |
| sampler.process_first_log | 05:08:35.508 | 217.4 |  | container log, first line |
| sampler.engine_init_start | 05:08:52.969 | 234.8 |  | container log: (EngineCore pid=169) INFO 10-15 05:08:52 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 05:08:54.079 | 235.9 |  | container log: (EngineCore pid=169) INFO 10-15 05:08:54 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 05:08:55.068 | 236.9 |  | container log: (EngineCore pid=169) INFO 10-15 05:08:55 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 05:09:08.096 | 250.0 | 13.03 | container log: (EngineCore pid=169) INFO 10-15 05:09:08 [default_loader.py:430] Loading weights took 13.03 seconds |
| sampler.model_loaded | 05:09:08.762 | 250.6 | 14.956827 | container log: (EngineCore pid=169) INFO 10-15 05:09:08 [model_runner.py:428] Model loading took 18.17 GiB memory and 14.956827 seconds |
| sampler.compile_done | 05:09:45.574 | 287.4 | 35.82 | container log: (EngineCore pid=169) INFO 10-15 05:09:45 [monitor.py:53] torch.compile took 35.82 s in total |
| sampler.graphs_captured | 05:10:06.072 | 307.9 | 17.0 | container log: (EngineCore pid=169) INFO 10-15 05:10:06 [model_runner.py:1066] Graph capturing finished in 17 secs, took 0.78 GiB |
| sampler.kv_cache_sized | 05:10:06.701 | 308.6 |  | container log: (EngineCore pid=169) INFO 10-15 05:10:06 [gpu_worker.py:640] Available KV cache memory: 50.17 GiB |
| sampler.engine_init_done | 05:10:25.431 | 327.3 | 76.66 | container log: (EngineCore pid=169) INFO 10-15 05:10:25 [core.py:372] init engine (profile, create kv cache, warmup model) took 76.66 s |
| sampler.ready_key | 05:10:26.000 | 327.9 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen3-8B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 05:04:58.138 | -0.0 |
| client.create_model.start | 05:04:58.139 | 0.0 |
| client.create_model.done | 05:06:49.594 | 111.5 |
| client.forward_backward.start | 05:06:52.784 | 114.6 |
| client.forward_backward.done | 05:06:55.595 | 117.5 |
| client.optim_step.done | 05:06:57.617 | 119.5 |
| client.save_weights_and_get_sampling_client.start | 05:06:57.618 | 119.5 |
| client.save_weights_and_get_sampling_client.done | 05:06:59.533 | 121.4 |
| client.sample.start | 05:06:59.534 | 121.4 |
| client.sample.done | 05:10:27.431 | 329.3 |
| client.end | 05:10:27.434 | 329.3 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.9 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 90.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 5.1 |
| weights load | trainer.load_start | trainer.load_done | 13.0 |
| ready key | trainer.load_done | trainer.ready_key | -0.1 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 111.5 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 2.8 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | 0.4 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 90.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 7.5 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 17.5 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.1 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 14.0 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 37.5 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 39.9 |
| ready key | sampler.engine_init_done | sampler.ready_key | 0.6 |
| client: first sample blocked | client.sample.start | client.sample.done | 207.9 |

