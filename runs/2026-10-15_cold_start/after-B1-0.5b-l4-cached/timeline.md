# Cold start timeline: after-B1-0.5b-l4-cached

t=0 is the client's `create_model` call at 2026-10-15T04:57:21.804+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen2-5-0-5b-0-trainer` on `gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.ready_key | 04:57:18.000 | -3.8 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.workload_created | 04:57:22.000 | 0.2 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 04:57:22.000 | 0.2 |  | pod creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 04:57:22.498 | 0.7 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 04:57:22.498 | 0.7 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 04:57:22.498 | 0.7 |  | workloads.tsv poller: status.nodeName set |
| trainer.pod_Initialized | 04:57:23.000 | 1.2 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 04:57:23.000 | 1.2 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 04:57:23.000 | 1.2 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-trainer to gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8 |
| trainer.pod_Ready | 04:57:26.000 | 4.2 |  | pod condition Ready |
| trainer.pod_ContainersReady | 04:57:26.000 | 4.2 |  | pod condition ContainersReady |
| trainer.container_started | 04:57:26.000 | 4.2 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 04:57:26.000 | 4.2 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| trainer.event_Created | 04:57:26.000 | 4.2 |  | event: Container created |
| trainer.event_Started | 04:57:26.000 | 4.2 |  | event: Container started |
| trainer.condition_Placed_True | 04:57:27.000 | 5.2 |  | Workload condition Placed |
| trainer.process_first_log | 04:57:33.659 | 11.9 |  | container log, first line |
| trainer.load_start | 04:57:33.802 | 12.0 |  | container log: Loading base model Qwen/Qwen2.5-0.5B to cuda:0... |
| trainer.load_done | 04:57:36.346 | 14.5 |  | container log: Successfully loaded. |
| trainer.processor_started | 04:57:36.349 | 14.5 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen2.5-0.5B active_set=Qwen/Qwen2.5-0.5B-1). |

## sampler: `orw-lora-qwen-qwen2-5-0-5b-0-sampler` on `gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 04:57:44.000 | 22.2 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 04:57:44.000 | 22.2 |  | pod creationTimestamp (apiserver) |
| sampler.event_FailedScheduling | 04:57:44.000 | 22.2 |  | event: 0/11 nodes are available: 1 node(s) had untolerated taint(s), 10 node(s) didn't match Pod's node affinity/selector. still not schedulable, p |
| sampler.event_NotTriggerScaleUp | 04:57:44.000 | 22.2 |  | event: Pod didn't trigger scale-up: 1 node(s) had untolerated taint(s) |
| sampler.workload_first_seen_by_poller | 04:57:44.625 | 22.8 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 04:57:44.625 | 22.8 |  | workloads.tsv poller: status.podName set |
| sampler.pod_Initialized | 04:58:13.000 | 51.2 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 04:58:13.000 | 51.2 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 04:58:13.000 | 51.2 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-sampler to gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208 |
| sampler.workload_reports_node | 04:58:13.579 | 51.8 |  | workloads.tsv poller: status.nodeName set |
| sampler.container_started | 04:58:16.000 | 54.2 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 04:58:16.000 | 54.2 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| sampler.event_Created | 04:58:16.000 | 54.2 |  | event: Container created |
| sampler.event_Started | 04:58:16.000 | 54.2 |  | event: Container started |
| sampler.condition_Placed_True | 04:58:17.000 | 55.2 |  | Workload condition Placed |
| sampler.pod_Ready | 04:58:17.000 | 55.2 |  | pod condition Ready |
| sampler.pod_ContainersReady | 04:58:17.000 | 55.2 |  | pod condition ContainersReady |
| sampler.process_first_log | 04:58:27.584 | 65.8 |  | container log, first line |
| sampler.engine_init_start | 04:58:54.733 | 92.9 |  | container log: (EngineCore pid=161) INFO 10-15 04:58:54 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 04:58:56.402 | 94.6 |  | container log: (EngineCore pid=161) INFO 10-15 04:58:56 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 04:58:57.753 | 95.9 |  | container log: (EngineCore pid=161) INFO 10-15 04:58:57 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 04:58:58.859 | 97.1 | 1.11 | container log: (EngineCore pid=161) INFO 10-15 04:58:58 [default_loader.py:430] Loading weights took 1.11 seconds |
| sampler.model_loaded | 04:58:59.653 | 97.8 | 3.562917 | container log: (EngineCore pid=161) INFO 10-15 04:58:59 [model_runner.py:428] Model loading took 1.47 GiB memory and 3.562917 seconds |
| sampler.compile_done | 04:59:30.263 | 128.5 | 28.39 | container log: (EngineCore pid=161) INFO 10-15 04:59:30 [monitor.py:53] torch.compile took 28.39 s in total |
| sampler.graphs_captured | 04:59:45.735 | 143.9 | 11.0 | container log: (EngineCore pid=161) INFO 10-15 04:59:45 [model_runner.py:1066] Graph capturing finished in 11 secs, took 0.23 GiB |
| sampler.kv_cache_sized | 04:59:46.273 | 144.5 |  | container log: (EngineCore pid=161) INFO 10-15 04:59:46 [gpu_worker.py:640] Available KV cache memory: 17.33 GiB |
| sampler.engine_init_done | 04:59:57.142 | 155.3 | 57.48 | container log: (EngineCore pid=161) INFO 10-15 04:59:57 [core.py:372] init engine (profile, create kv cache, warmup model) took 57.48 s |
| sampler.ready_key | 04:59:58.000 | 156.2 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen2.5-0.5B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 04:57:21.804 | 0.0 |
| client.create_model.start | 04:57:21.804 | 0.0 |
| client.create_model.done | 04:57:37.299 | 15.5 |
| client.forward_backward.start | 04:57:40.066 | 18.3 |
| client.forward_backward.done | 04:57:43.279 | 21.5 |
| client.optim_step.done | 04:57:44.032 | 22.2 |
| client.save_weights_and_get_sampling_client.start | 04:57:44.033 | 22.2 |
| client.save_weights_and_get_sampling_client.done | 04:57:44.610 | 22.8 |
| client.sample.start | 04:57:44.612 | 22.8 |
| client.sample.done | 04:59:59.246 | 157.4 |
| client.end | 04:59:59.247 | 157.4 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.2 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 1.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 3.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 7.8 |
| weights load | trainer.load_start | trainer.load_done | 2.5 |
| ready key | trainer.load_done | trainer.ready_key | -18.3 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 15.5 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.2 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.0 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 29.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 3.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 11.6 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 27.1 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.7 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 2.5 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 31.4 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 26.9 |
| ready key | sampler.engine_init_done | sampler.ready_key | 0.9 |
| client: first sample blocked | client.sample.start | client.sample.done | 134.6 |

