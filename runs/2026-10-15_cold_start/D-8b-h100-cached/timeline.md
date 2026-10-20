# Cold start timeline: D-8b-h100-cached

t=0 is the client's `create_model` call at 2026-10-15T04:28:47.756+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen3-8b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 04:28:48.000 | 0.2 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 04:28:48.000 | 0.2 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 04:28:48.000 | 0.2 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 04:28:48.000 | 0.2 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 04:28:48.000 | 0.2 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.workload_first_seen_by_poller | 04:28:48.508 | 0.8 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 04:28:48.508 | 0.8 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 04:28:48.508 | 0.8 |  | workloads.tsv poller: status.nodeName set |
| trainer.container_started | 04:28:52.000 | 4.2 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 04:28:52.000 | 4.2 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" already present on machine and can  |
| trainer.event_Created | 04:28:52.000 | 4.2 |  | event: Container created |
| trainer.event_Started | 04:28:52.000 | 4.2 |  | event: Container started |
| trainer.condition_Placed_True | 04:28:53.000 | 5.2 |  | Workload condition Placed |
| trainer.pod_Ready | 04:28:53.000 | 5.2 |  | pod condition Ready |
| trainer.pod_ContainersReady | 04:28:53.000 | 5.2 |  | pod condition ContainersReady |
| trainer.process_first_log | 04:28:57.210 | 9.5 |  | container log, first line |
| trainer.load_start | 04:28:57.431 | 9.7 |  | container log: Loading base model Qwen/Qwen3-8B to cuda:0... |
| trainer.ready_key | 04:29:37.000 | 49.2 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen3-8B |
| trainer.load_done | 04:29:37.342 | 49.6 |  | container log: Successfully loaded. |
| trainer.processor_started | 04:29:37.343 | 49.6 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen3-8B active_set=Qwen/Qwen3-8B-1). |

## sampler: `orw-lora-qwen-qwen3-8b-0-sampler` on `gke-open-rl-dra-h100-sampler-a721f127-0853`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 04:29:47.000 | 59.2 |  | Workload creationTimestamp (apiserver) |
| sampler.workload_first_seen_by_poller | 04:29:47.960 | 60.2 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 04:29:47.960 | 60.2 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 04:29:47.960 | 60.2 |  | workloads.tsv poller: status.nodeName set |
| sampler.pod_created | 04:29:48.000 | 60.2 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 04:29:48.000 | 60.2 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 04:29:48.000 | 60.2 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 04:29:48.000 | 60.2 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-sampler to gke-open-rl-dra-h100-sampler-a721f127-0853 |
| sampler.event_Pulled | 04:29:51.000 | 63.2 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:agent3-9f8dc13" already present on machine and can  |
| sampler.event_Created | 04:29:51.000 | 63.2 |  | event: Container created |
| sampler.condition_Placed_True | 04:29:52.000 | 64.2 |  | Workload condition Placed |
| sampler.pod_Ready | 04:29:52.000 | 64.2 |  | pod condition Ready |
| sampler.pod_ContainersReady | 04:29:52.000 | 64.2 |  | pod condition ContainersReady |
| sampler.container_started | 04:29:52.000 | 64.2 |  | containerStatuses.state.running.startedAt |
| sampler.event_Started | 04:29:52.000 | 64.2 |  | event: Container started |
| sampler.process_first_log | 04:29:59.120 | 71.4 |  | container log, first line |
| sampler.engine_init_start | 04:30:16.823 | 89.1 |  | container log: (EngineCore pid=169) INFO 10-15 04:30:16 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 04:30:17.937 | 90.2 |  | container log: (EngineCore pid=169) INFO 10-15 04:30:17 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 04:30:18.941 | 91.2 |  | container log: (EngineCore pid=169) INFO 10-15 04:30:18 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 04:30:31.464 | 103.7 | 12.53 | container log: (EngineCore pid=169) INFO 10-15 04:30:31 [default_loader.py:430] Loading weights took 12.53 seconds |
| sampler.model_loaded | 04:30:32.103 | 104.3 | 14.439416 | container log: (EngineCore pid=169) INFO 10-15 04:30:32 [model_runner.py:428] Model loading took 18.17 GiB memory and 14.439416 seconds |
| sampler.compile_done | 04:31:08.896 | 141.1 | 35.76 | container log: (EngineCore pid=169) INFO 10-15 04:31:08 [monitor.py:53] torch.compile took 35.76 s in total |
| sampler.graphs_captured | 04:31:29.315 | 161.6 | 16.0 | container log: (EngineCore pid=169) INFO 10-15 04:31:29 [model_runner.py:1066] Graph capturing finished in 16 secs, took 0.78 GiB |
| sampler.kv_cache_sized | 04:31:29.941 | 162.2 |  | container log: (EngineCore pid=169) INFO 10-15 04:31:29 [gpu_worker.py:640] Available KV cache memory: 50.17 GiB |
| sampler.engine_init_done | 04:31:48.438 | 180.7 | 76.33 | container log: (EngineCore pid=169) INFO 10-15 04:31:48 [core.py:372] init engine (profile, create kv cache, warmup model) took 76.33 s |
| sampler.ready_key | 04:31:49.000 | 181.2 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen3-8B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 04:28:47.756 | 0.0 |
| client.create_model.start | 04:28:47.756 | 0.0 |
| client.create_model.done | 04:29:39.925 | 52.2 |
| client.forward_backward.start | 04:29:42.710 | 55.0 |
| client.forward_backward.done | 04:29:45.623 | 57.9 |
| client.optim_step.done | 04:29:47.732 | 60.0 |
| client.save_weights_and_get_sampling_client.start | 04:29:47.733 | 60.0 |
| client.save_weights_and_get_sampling_client.done | 04:29:49.654 | 61.9 |
| client.sample.start | 04:29:49.657 | 61.9 |
| client.sample.done | 04:31:50.315 | 182.6 |
| client.end | 04:31:50.316 | 182.6 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.2 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 4.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 5.4 |
| weights load | trainer.load_start | trainer.load_done | 39.9 |
| ready key | trainer.load_done | trainer.ready_key | -0.3 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 52.2 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 2.9 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | 0.3 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 3.0 |
| container create | sampler.event_Pulled | sampler.container_started | 1.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 7.1 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 17.7 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.1 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 13.5 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 37.4 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 39.5 |
| ready key | sampler.engine_init_done | sampler.ready_key | 0.6 |
| client: first sample blocked | client.sample.start | client.sample.done | 120.7 |

