# Cold start timeline: after-B2-0.5b-l4-cached

t=0 is the client's `create_model` call at 2026-10-15T05:01:15.634+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen2-5-0-5b-0-trainer` on `gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.ready_key | 05:01:12.000 | -3.6 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.workload_created | 05:01:16.000 | 0.4 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 05:01:16.000 | 0.4 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 05:01:16.000 | 0.4 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 05:01:16.000 | 0.4 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 05:01:16.000 | 0.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-trainer to gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8 |
| trainer.workload_first_seen_by_poller | 05:01:16.282 | 0.6 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 05:01:16.282 | 0.6 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 05:01:17.948 | 2.3 |  | workloads.tsv poller: status.nodeName set |
| trainer.container_started | 05:01:20.000 | 4.4 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 05:01:20.000 | 4.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| trainer.event_Created | 05:01:20.000 | 4.4 |  | event: Container created |
| trainer.event_Started | 05:01:20.000 | 4.4 |  | event: Container started |
| trainer.pod_Ready | 05:01:21.000 | 5.4 |  | pod condition Ready |
| trainer.pod_ContainersReady | 05:01:21.000 | 5.4 |  | pod condition ContainersReady |
| trainer.condition_Placed_True | 05:01:22.000 | 6.4 |  | Workload condition Placed |
| trainer.process_first_log | 05:01:27.847 | 12.2 |  | container log, first line |
| trainer.load_start | 05:01:27.989 | 12.4 |  | container log: Loading base model Qwen/Qwen2.5-0.5B to cuda:0... |
| trainer.load_done | 05:01:30.531 | 14.9 |  | container log: Successfully loaded. |
| trainer.processor_started | 05:01:30.533 | 14.9 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen2.5-0.5B active_set=Qwen/Qwen2.5-0.5B-1). |

## sampler: `orw-lora-qwen-qwen2-5-0-5b-0-sampler` on `gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 05:01:38.000 | 22.4 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 05:01:38.000 | 22.4 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 05:01:38.000 | 22.4 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 05:01:38.000 | 22.4 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 05:01:38.000 | 22.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-sampler to gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208 |
| sampler.workload_first_seen_by_poller | 05:01:38.028 | 22.4 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 05:01:38.028 | 22.4 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 05:01:39.789 | 24.2 |  | workloads.tsv poller: status.nodeName set |
| sampler.condition_Placed_True | 05:01:42.000 | 26.4 |  | Workload condition Placed |
| sampler.pod_Ready | 05:01:42.000 | 26.4 |  | pod condition Ready |
| sampler.pod_ContainersReady | 05:01:42.000 | 26.4 |  | pod condition ContainersReady |
| sampler.container_started | 05:01:42.000 | 26.4 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 05:01:42.000 | 26.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| sampler.event_Created | 05:01:42.000 | 26.4 |  | event: Container created |
| sampler.event_Started | 05:01:42.000 | 26.4 |  | event: Container started |
| sampler.process_first_log | 05:02:05.308 | 49.7 |  | container log, first line |
| sampler.engine_init_start | 05:02:21.575 | 65.9 |  | container log: (EngineCore pid=161) INFO 10-15 05:02:21 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 05:02:23.325 | 67.7 |  | container log: (EngineCore pid=161) INFO 10-15 05:02:23 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 05:02:24.679 | 69.0 |  | container log: (EngineCore pid=161) INFO 10-15 05:02:24 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 05:02:25.907 | 70.3 | 1.23 | container log: (EngineCore pid=161) INFO 10-15 05:02:25 [default_loader.py:430] Loading weights took 1.23 seconds |
| sampler.model_loaded | 05:02:26.718 | 71.1 | 3.717496 | container log: (EngineCore pid=161) INFO 10-15 05:02:26 [model_runner.py:428] Model loading took 1.47 GiB memory and 3.717496 seconds |
| sampler.compile_done | 05:02:57.251 | 101.6 | 28.32 | container log: (EngineCore pid=161) INFO 10-15 05:02:57 [monitor.py:53] torch.compile took 28.32 s in total |
| sampler.graphs_captured | 05:03:12.777 | 117.1 | 11.0 | container log: (EngineCore pid=161) INFO 10-15 05:03:12 [model_runner.py:1066] Graph capturing finished in 11 secs, took 0.23 GiB |
| sampler.kv_cache_sized | 05:03:13.327 | 117.7 |  | container log: (EngineCore pid=161) INFO 10-15 05:03:13 [gpu_worker.py:640] Available KV cache memory: 17.33 GiB |
| sampler.engine_init_done | 05:03:24.097 | 128.5 | 57.37 | container log: (EngineCore pid=161) INFO 10-15 05:03:24 [core.py:372] init engine (profile, create kv cache, warmup model) took 57.37 s |
| sampler.ready_key | 05:03:26.000 | 130.4 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen2.5-0.5B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 05:01:15.634 | 0.0 |
| client.create_model.start | 05:01:15.634 | 0.0 |
| client.create_model.done | 05:01:31.419 | 15.8 |
| client.forward_backward.start | 05:01:34.271 | 18.6 |
| client.forward_backward.done | 05:01:37.484 | 21.9 |
| client.optim_step.done | 05:01:38.241 | 22.6 |
| client.save_weights_and_get_sampling_client.start | 05:01:38.245 | 22.6 |
| client.save_weights_and_get_sampling_client.done | 05:01:38.818 | 23.2 |
| client.sample.start | 05:01:38.819 | 23.2 |
| client.sample.done | 05:03:26.180 | 130.5 |
| client.end | 05:03:26.181 | 130.5 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.4 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 4.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 8.0 |
| weights load | trainer.load_start | trainer.load_done | 2.5 |
| ready key | trainer.load_done | trainer.ready_key | -18.5 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 15.8 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.2 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.2 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 4.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 23.3 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 16.3 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.8 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 2.6 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 31.3 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 26.8 |
| ready key | sampler.engine_init_done | sampler.ready_key | 1.9 |
| client: first sample blocked | client.sample.start | client.sample.done | 107.4 |

