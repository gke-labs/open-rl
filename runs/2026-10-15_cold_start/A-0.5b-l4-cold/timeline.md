# Cold start timeline: A-0.5b-l4-cold

t=0 is the client's `create_model` call at 2026-10-15T03:04:15.892+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen2-5-0-5b-0-trainer` on `gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 03:04:16.000 | 0.1 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 03:04:16.000 | 0.1 |  | pod creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 03:04:16.578 | 0.7 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 03:04:16.578 | 0.7 |  | workloads.tsv poller: status.podName set |
| trainer.pod_Initialized | 03:04:17.000 | 1.1 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 03:04:17.000 | 1.1 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 03:04:17.000 | 1.1 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-trainer to gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8 |
| trainer.workload_reports_node | 03:04:18.224 | 2.3 |  | workloads.tsv poller: status.nodeName set |
| trainer.event_Pulling | 03:04:21.000 | 5.1 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" |
| trainer.event_Pulled | 03:07:09.000 | 173.1 | 168.59 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" in 2m48.59s (2m48.59s inc |
| trainer.event_Created | 03:07:09.000 | 173.1 |  | event: Container created |
| trainer.container_started | 03:07:10.000 | 174.1 |  | containerStatuses.state.running.startedAt |
| trainer.event_Started | 03:07:10.000 | 174.1 |  | event: Container started |
| trainer.condition_Placed_True | 03:07:11.000 | 175.1 |  | Workload condition Placed |
| trainer.pod_Ready | 03:07:11.000 | 175.1 |  | pod condition Ready |
| trainer.pod_ContainersReady | 03:07:11.000 | 175.1 |  | pod condition ContainersReady |
| trainer.process_first_log | 03:07:17.253 | 181.4 |  | container log, first line |
| trainer.load_start | 03:07:17.410 | 181.5 |  | container log: Loading base model Qwen/Qwen2.5-0.5B to cuda:0... |
| trainer.ready_key | 03:07:21.000 | 185.1 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.load_done | 03:07:21.686 | 185.8 |  | container log: Successfully loaded. |
| trainer.processor_started | 03:07:21.689 | 185.8 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen2.5-0.5B active_set=Qwen/Qwen2.5-0.5B-1). |

## sampler: `orw-lora-qwen-qwen2-5-0-5b-0-sampler` on `gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_first_seen_by_poller | 03:07:28.976 | 193.1 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_created | 03:07:29.000 | 193.1 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 03:07:29.000 | 193.1 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 03:07:30.000 | 194.1 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 03:07:30.000 | 194.1 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 03:07:30.000 | 194.1 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-sampler to gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208 |
| sampler.workload_reports_pod | 03:07:30.816 | 194.9 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 03:07:30.816 | 194.9 |  | workloads.tsv poller: status.nodeName set |
| sampler.event_Pulling | 03:07:33.000 | 197.1 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" |
| sampler.container_started | 03:10:22.000 | 366.1 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 03:10:22.000 | 366.1 | 168.566 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" in 2m48.566s (2m48.566s i |
| sampler.event_Created | 03:10:22.000 | 366.1 |  | event: Container created |
| sampler.event_Started | 03:10:22.000 | 366.1 |  | event: Container started |
| sampler.condition_Placed_True | 03:10:23.000 | 367.1 |  | Workload condition Placed |
| sampler.pod_Ready | 03:10:23.000 | 367.1 |  | pod condition Ready |
| sampler.pod_ContainersReady | 03:10:23.000 | 367.1 |  | pod condition ContainersReady |
| sampler.process_first_log | 03:10:33.926 | 378.0 |  | container log, first line |
| sampler.engine_init_start | 03:11:01.668 | 405.8 |  | container log: (EngineCore pid=161) INFO 10-15 03:11:01 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 03:11:03.371 | 407.5 |  | container log: (EngineCore pid=161) INFO 10-15 03:11:03 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 03:11:04.731 | 408.8 |  | container log: (EngineCore pid=161) INFO 10-15 03:11:04 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 03:11:06.644 | 410.8 | 1.92 | container log: (EngineCore pid=161) INFO 10-15 03:11:06 [default_loader.py:430] Loading weights took 1.92 seconds |
| sampler.model_loaded | 03:11:07.445 | 411.6 | 4.387871 | container log: (EngineCore pid=161) INFO 10-15 03:11:07 [model_runner.py:428] Model loading took 1.47 GiB memory and 4.387871 seconds |
| sampler.compile_done | 03:11:38.154 | 442.3 | 28.48 | container log: (EngineCore pid=161) INFO 10-15 03:11:38 [monitor.py:53] torch.compile took 28.48 s in total |
| sampler.graphs_captured | 03:11:53.539 | 457.6 | 11.0 | container log: (EngineCore pid=161) INFO 10-15 03:11:53 [model_runner.py:1066] Graph capturing finished in 11 secs, took 0.23 GiB |
| sampler.kv_cache_sized | 03:11:54.078 | 458.2 |  | container log: (EngineCore pid=161) INFO 10-15 03:11:54 [gpu_worker.py:640] Available KV cache memory: 17.33 GiB |
| sampler.engine_init_done | 03:12:04.781 | 468.9 | 57.33 | container log: (EngineCore pid=161) INFO 10-15 03:12:04 [core.py:372] init engine (profile, create kv cache, warmup model) took 57.33 s |
| sampler.ready_key | 03:12:06.000 | 470.1 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen2.5-0.5B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 03:04:15.891 | -0.0 |
| client.create_model.start | 03:04:15.892 | 0.0 |
| client.create_model.done | 03:07:22.617 | 186.7 |
| client.forward_backward.start | 03:07:25.417 | 189.5 |
| client.forward_backward.done | 03:07:28.837 | 192.9 |
| client.optim_step.done | 03:07:29.600 | 193.7 |
| client.save_weights_and_get_sampling_client.start | 03:07:29.600 | 193.7 |
| client.save_weights_and_get_sampling_client.done | 03:07:30.173 | 194.3 |
| client.sample.start | 03:07:30.174 | 194.3 |
| client.sample.done | 03:12:07.151 | 471.3 |
| client.end | 03:12:07.154 | 471.3 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.1 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 1.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 172.0 |
| container create | trainer.event_Pulled | trainer.container_started | 1.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 7.4 |
| weights load | trainer.load_start | trainer.load_done | 4.3 |
| ready key | trainer.load_done | trainer.ready_key | -0.7 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 186.7 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.4 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.6 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 1.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 172.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 11.9 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 27.7 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.7 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 3.3 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 31.5 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 26.6 |
| ready key | sampler.engine_init_done | sampler.ready_key | 1.2 |
| client: first sample blocked | client.sample.start | client.sample.done | 277.0 |

