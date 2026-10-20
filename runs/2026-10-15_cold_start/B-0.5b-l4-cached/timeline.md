# Cold start timeline: B-0.5b-l4-cached

t=0 is the client's `create_model` call at 2026-10-15T02:59:11.609+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen2-5-0-5b-0-trainer` on `gke-open-rl-dra-l4-trainer-2x-67d40c4c-sr2c`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_first_seen_by_poller | 02:59:11.918 | 0.3 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_created | 02:59:12.000 | 0.4 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 02:59:13.000 | 1.4 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 02:59:13.000 | 1.4 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 02:59:13.000 | 1.4 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 02:59:13.000 | 1.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-trainer to gke-open-rl-dra-l4-trainer-2x-67d40c4c-sr2c |
| trainer.workload_reports_pod | 02:59:13.554 | 1.9 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 02:59:13.554 | 1.9 |  | workloads.tsv poller: status.nodeName set |
| trainer.container_started | 02:59:17.000 | 5.4 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 02:59:17.000 | 5.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" already present on machine and can  |
| trainer.event_Created | 02:59:17.000 | 5.4 |  | event: Container created |
| trainer.event_Started | 02:59:17.000 | 5.4 |  | event: Container started |
| trainer.condition_Placed_True | 02:59:18.000 | 6.4 |  | Workload condition Placed |
| trainer.pod_Ready | 02:59:18.000 | 6.4 |  | pod condition Ready |
| trainer.pod_ContainersReady | 02:59:18.000 | 6.4 |  | pod condition ContainersReady |
| trainer.process_first_log | 02:59:23.844 | 12.2 |  | container log, first line |
| trainer.load_start | 02:59:23.989 | 12.4 |  | container log: Loading base model Qwen/Qwen2.5-0.5B to cuda:0... |
| trainer.ready_key | 02:59:28.000 | 16.4 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.load_done | 02:59:28.232 | 16.6 |  | container log: Successfully loaded. |
| trainer.processor_started | 02:59:28.234 | 16.6 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen2.5-0.5B active_set=Qwen/Qwen2.5-0.5B-1). |

## sampler: `orw-lora-qwen-qwen2-5-0-5b-0-sampler` on `gke-open-rl-dra-l4-sampler-2x-1ad5171e-bs96`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_first_seen_by_poller | 02:59:35.958 | 24.3 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_created | 02:59:36.000 | 24.4 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 02:59:36.000 | 24.4 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 02:59:36.000 | 24.4 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 02:59:36.000 | 24.4 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 02:59:36.000 | 24.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-sampler to gke-open-rl-dra-l4-sampler-2x-1ad5171e-bs96 |
| sampler.workload_reports_pod | 02:59:37.643 | 26.0 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 02:59:37.643 | 26.0 |  | workloads.tsv poller: status.nodeName set |
| sampler.condition_Placed_True | 02:59:40.000 | 28.4 |  | Workload condition Placed |
| sampler.pod_Ready | 02:59:40.000 | 28.4 |  | pod condition Ready |
| sampler.pod_ContainersReady | 02:59:40.000 | 28.4 |  | pod condition ContainersReady |
| sampler.container_started | 02:59:40.000 | 28.4 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 02:59:40.000 | 28.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" already present on machine and can  |
| sampler.event_Created | 02:59:40.000 | 28.4 |  | event: Container created |
| sampler.event_Started | 02:59:40.000 | 28.4 |  | event: Container started |
| sampler.process_first_log | 02:59:51.481 | 39.9 |  | container log, first line |
| sampler.engine_init_start | 03:00:18.656 | 67.0 |  | container log: (EngineCore pid=161) INFO 10-15 03:00:18 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 03:00:20.368 | 68.8 |  | container log: (EngineCore pid=161) INFO 10-15 03:00:20 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 03:00:21.715 | 70.1 |  | container log: (EngineCore pid=161) INFO 10-15 03:00:21 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 03:00:23.366 | 71.8 | 1.65 | container log: (EngineCore pid=161) INFO 10-15 03:00:23 [default_loader.py:430] Loading weights took 1.65 seconds |
| sampler.model_loaded | 03:00:24.163 | 72.6 | 4.105582 | container log: (EngineCore pid=161) INFO 10-15 03:00:24 [model_runner.py:428] Model loading took 1.47 GiB memory and 4.105582 seconds |
| sampler.compile_done | 03:00:54.727 | 103.1 | 28.38 | container log: (EngineCore pid=161) INFO 10-15 03:00:54 [monitor.py:53] torch.compile took 28.38 s in total |
| sampler.graphs_captured | 03:01:10.141 | 118.5 | 11.0 | container log: (EngineCore pid=161) INFO 10-15 03:01:10 [model_runner.py:1066] Graph capturing finished in 11 secs, took 0.23 GiB |
| sampler.kv_cache_sized | 03:01:10.683 | 119.1 |  | container log: (EngineCore pid=161) INFO 10-15 03:01:10 [gpu_worker.py:640] Available KV cache memory: 17.33 GiB |
| sampler.engine_init_done | 03:01:21.459 | 129.9 | 57.29 | container log: (EngineCore pid=161) INFO 10-15 03:01:21 [core.py:372] init engine (profile, create kv cache, warmup model) took 57.29 s |
| sampler.ready_key | 03:01:23.000 | 131.4 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen2.5-0.5B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 02:59:11.608 | -0.0 |
| client.create_model.start | 02:59:11.609 | 0.0 |
| client.create_model.done | 02:59:29.165 | 17.6 |
| client.forward_backward.start | 02:59:32.362 | 20.8 |
| client.forward_backward.done | 02:59:35.480 | 23.9 |
| client.optim_step.done | 02:59:36.339 | 24.7 |
| client.save_weights_and_get_sampling_client.start | 02:59:36.340 | 24.7 |
| client.save_weights_and_get_sampling_client.done | 02:59:36.946 | 25.3 |
| client.sample.start | 02:59:36.947 | 25.3 |
| client.sample.done | 03:01:23.638 | 132.0 |
| client.end | 03:01:23.639 | 132.0 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 1.4 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 4.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 7.0 |
| weights load | trainer.load_start | trainer.load_done | 4.2 |
| ready key | trainer.load_done | trainer.ready_key | -0.2 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 17.6 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.1 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.3 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 4.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 11.5 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 27.2 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.7 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 3.0 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 31.4 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 26.7 |
| ready key | sampler.engine_init_done | sampler.ready_key | 1.5 |
| client: first sample blocked | client.sample.start | client.sample.done | 106.7 |

