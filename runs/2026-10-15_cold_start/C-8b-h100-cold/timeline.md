# Cold start timeline: C-8b-h100-cold

t=0 is the client's `create_model` call at 2026-10-15T03:14:13.683+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen3-8b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 03:14:14.000 | 0.3 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 03:14:14.000 | 0.3 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 03:14:14.000 | 0.3 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 03:14:14.000 | 0.3 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 03:14:14.000 | 0.3 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.workload_first_seen_by_poller | 03:14:14.472 | 0.8 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 03:14:14.472 | 0.8 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 03:14:14.472 | 0.8 |  | workloads.tsv poller: status.nodeName set |
| trainer.event_Pulling | 03:14:19.000 | 5.3 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" |
| trainer.container_started | 03:15:51.000 | 97.3 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 03:15:51.000 | 97.3 | 92.497 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" in 1m32.497s (1m32.497s i |
| trainer.event_Created | 03:15:51.000 | 97.3 |  | event: Container created |
| trainer.event_Started | 03:15:51.000 | 97.3 |  | event: Container started |
| trainer.condition_Placed_True | 03:15:52.000 | 98.3 |  | Workload condition Placed |
| trainer.pod_Ready | 03:15:52.000 | 98.3 |  | pod condition Ready |
| trainer.pod_ContainersReady | 03:15:52.000 | 98.3 |  | pod condition ContainersReady |
| trainer.process_first_log | 03:15:56.012 | 102.3 |  | container log, first line |
| trainer.load_start | 03:15:56.267 | 102.6 |  | container log: Loading base model Qwen/Qwen3-8B to cuda:0... |
| trainer.ready_key | 03:30:59.000 | 1005.3 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen3-8B |
| trainer.load_done | 03:30:59.760 | 1006.1 |  | container log: Successfully loaded. |
| trainer.processor_started | 03:30:59.761 | 1006.1 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen3-8B active_set=Qwen/Qwen3-8B-1). |

## sampler: `orw-lora-qwen-qwen3-8b-0-sampler` on `gke-open-rl-dra-h100-sampler-a721f127-0853`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 03:31:10.000 | 1016.3 |  | Workload creationTimestamp (apiserver) |
| sampler.workload_first_seen_by_poller | 03:31:10.295 | 1016.6 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.pod_created | 03:31:11.000 | 1017.3 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 03:31:11.000 | 1017.3 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 03:31:11.000 | 1017.3 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 03:31:11.000 | 1017.3 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-sampler to gke-open-rl-dra-h100-sampler-a721f127-0853 |
| sampler.workload_reports_pod | 03:31:11.907 | 1018.2 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 03:31:11.907 | 1018.2 |  | workloads.tsv poller: status.nodeName set |
| sampler.event_Pulling | 03:31:15.000 | 1021.3 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" |
| sampler.container_started | 03:32:50.000 | 1116.3 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 03:32:50.000 | 1116.3 | 95.306 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" in 1m35.306s (1m35.306s i |
| sampler.event_Created | 03:32:50.000 | 1116.3 |  | event: Container created |
| sampler.event_Started | 03:32:50.000 | 1116.3 |  | event: Container started |
| sampler.condition_Placed_True | 03:32:51.000 | 1117.3 |  | Workload condition Placed |
| sampler.pod_Ready | 03:32:51.000 | 1117.3 |  | pod condition Ready |
| sampler.pod_ContainersReady | 03:32:51.000 | 1117.3 |  | pod condition ContainersReady |
| sampler.process_first_log | 03:32:57.768 | 1124.1 |  | container log, first line |
| sampler.engine_init_start | 03:33:16.117 | 1142.4 |  | container log: (EngineCore pid=168) INFO 10-15 03:33:16 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 03:33:17.209 | 1143.5 |  | container log: (EngineCore pid=168) INFO 10-15 03:33:17 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 03:33:18.247 | 1144.6 |  | container log: (EngineCore pid=168) INFO 10-15 03:33:18 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 03:35:00.792 | 1247.1 | 102.55 | container log: (EngineCore pid=168) INFO 10-15 03:35:00 [default_loader.py:430] Loading weights took 102.55 seconds |
| sampler.model_loaded | 03:35:01.444 | 1247.8 | 104.493682 | container log: (EngineCore pid=168) INFO 10-15 03:35:01 [model_runner.py:428] Model loading took 18.17 GiB memory and 104.493682 second |
| sampler.compile_done | 03:35:37.706 | 1284.0 | 35.25 | container log: (EngineCore pid=168) INFO 10-15 03:35:37 [monitor.py:53] torch.compile took 35.25 s in total |
| sampler.graphs_captured | 03:35:57.941 | 1304.3 | 16.0 | container log: (EngineCore pid=168) INFO 10-15 03:35:57 [model_runner.py:1066] Graph capturing finished in 16 secs, took 0.78 GiB |
| sampler.kv_cache_sized | 03:35:58.562 | 1304.9 |  | container log: (EngineCore pid=168) INFO 10-15 03:35:58 [gpu_worker.py:640] Available KV cache memory: 50.17 GiB |
| sampler.engine_init_done | 03:36:17.004 | 1323.3 | 75.55 | container log: (EngineCore pid=168) INFO 10-15 03:36:17 [core.py:372] init engine (profile, create kv cache, warmup model) took 75.55 s |
| sampler.ready_key | 03:36:18.000 | 1324.3 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen3-8B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 03:14:13.682 | -0.0 |
| client.create_model.start | 03:14:13.683 | 0.0 |
| client.create_model.done | 03:31:02.295 | 1008.6 |
| client.forward_backward.start | 03:31:05.464 | 1011.8 |
| client.forward_backward.done | 03:31:08.365 | 1014.7 |
| client.optim_step.done | 03:31:10.461 | 1016.8 |
| client.save_weights_and_get_sampling_client.start | 03:31:10.462 | 1016.8 |
| client.save_weights_and_get_sampling_client.done | 03:31:12.490 | 1018.8 |
| client.sample.start | 03:31:12.490 | 1018.8 |
| client.sample.done | 03:36:43.884 | 1350.2 |
| client.end | 03:36:43.886 | 1350.2 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.3 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 97.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 5.3 |
| weights load | trainer.load_start | trainer.load_done | 903.5 |
| ready key | trainer.load_done | trainer.ready_key | -0.8 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 1008.6 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 2.9 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | 0.5 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 99.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 7.8 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 18.3 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.1 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 103.6 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 36.9 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 39.3 |
| ready key | sampler.engine_init_done | sampler.ready_key | 1.0 |
| client: first sample blocked | client.sample.start | client.sample.done | 331.4 |

