# Cold start timeline: after-A1-0.5b-l4-cold

t=0 is the client's `create_model` call at 2026-10-15T04:47:51.111+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen2-5-0-5b-0-trainer` on `gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 04:47:51.000 | -0.1 |  | Workload creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 04:47:51.664 | 0.6 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 04:47:51.664 | 0.6 |  | workloads.tsv poller: status.podName set |
| trainer.pod_created | 04:47:52.000 | 0.9 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 04:47:52.000 | 0.9 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 04:47:52.000 | 0.9 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 04:47:52.000 | 0.9 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-trainer to gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8 |
| trainer.workload_reports_node | 04:47:53.304 | 2.2 |  | workloads.tsv poller: status.nodeName set |
| trainer.event_Pulling | 04:47:56.000 | 4.9 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" |
| trainer.event_Pulled | 04:50:44.000 | 172.9 | 168.27100000000002 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" in 2m48.271s (2m48.27 |
| trainer.event_Created | 04:50:44.000 | 172.9 |  | event: Container created |
| trainer.condition_Placed_True | 04:50:45.000 | 173.9 |  | Workload condition Placed |
| trainer.pod_Ready | 04:50:45.000 | 173.9 |  | pod condition Ready |
| trainer.pod_ContainersReady | 04:50:45.000 | 173.9 |  | pod condition ContainersReady |
| trainer.container_started | 04:50:45.000 | 173.9 |  | containerStatuses.state.running.startedAt |
| trainer.event_Started | 04:50:45.000 | 173.9 |  | event: Container started |
| trainer.process_first_log | 04:50:52.077 | 181.0 |  | container log, first line |
| trainer.load_start | 04:50:52.221 | 181.1 |  | container log: Loading base model Qwen/Qwen2.5-0.5B to cuda:0... |
| trainer.load_done | 04:51:07.261 | 196.2 |  | container log: Successfully loaded. |
| trainer.processor_started | 04:51:07.264 | 196.2 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen2.5-0.5B active_set=Qwen/Qwen2.5-0.5B-1). |
| trainer.ready_key | 04:51:08.000 | 196.9 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |

## sampler: `orw-lora-qwen-qwen2-5-0-5b-0-sampler` on `gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 04:51:15.000 | 203.9 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 04:51:15.000 | 203.9 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 04:51:15.000 | 203.9 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 04:51:15.000 | 203.9 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 04:51:15.000 | 203.9 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen2-5-0-5b-0-sampler to gke-open-rl-dra-l4-sampler-2x-ab5fdc9d-2208 |
| sampler.workload_first_seen_by_poller | 04:51:15.069 | 204.0 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 04:51:15.069 | 204.0 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 04:51:15.069 | 204.0 |  | workloads.tsv poller: status.nodeName set |
| sampler.event_Pulling | 04:51:19.000 | 207.9 |  | event: Pulling image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" |
| sampler.event_Pulled | 04:54:21.000 | 389.9 | 182.5 | event: Successfully pulled image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" in 3m2.5s (3m2.5s inc |
| sampler.event_Created | 04:54:21.000 | 389.9 |  | event: Container created |
| sampler.condition_Placed_True | 04:54:22.000 | 390.9 |  | Workload condition Placed |
| sampler.pod_Ready | 04:54:22.000 | 390.9 |  | pod condition Ready |
| sampler.pod_ContainersReady | 04:54:22.000 | 390.9 |  | pod condition ContainersReady |
| sampler.container_started | 04:54:22.000 | 390.9 |  | containerStatuses.state.running.startedAt |
| sampler.event_Started | 04:54:22.000 | 390.9 |  | event: Container started |
| sampler.process_first_log | 04:54:45.242 | 414.1 |  | container log, first line |
| sampler.engine_init_start | 04:55:00.403 | 429.3 |  | container log: (EngineCore pid=161) INFO 10-15 04:55:00 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 04:55:02.093 | 431.0 |  | container log: (EngineCore pid=161) INFO 10-15 04:55:02 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 04:55:03.439 | 432.3 |  | container log: (EngineCore pid=161) INFO 10-15 04:55:03 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 04:55:04.519 | 433.4 | 1.08 | container log: (EngineCore pid=161) INFO 10-15 04:55:04 [default_loader.py:430] Loading weights took 1.08 seconds |
| sampler.model_loaded | 04:55:05.323 | 434.2 | 3.543823 | container log: (EngineCore pid=161) INFO 10-15 04:55:05 [model_runner.py:428] Model loading took 1.47 GiB memory and 3.543823 seconds |
| sampler.compile_done | 04:55:36.056 | 464.9 | 28.53 | container log: (EngineCore pid=161) INFO 10-15 04:55:36 [monitor.py:53] torch.compile took 28.53 s in total |
| sampler.graphs_captured | 04:55:51.460 | 480.3 | 11.0 | container log: (EngineCore pid=161) INFO 10-15 04:55:51 [model_runner.py:1066] Graph capturing finished in 11 secs, took 0.23 GiB |
| sampler.kv_cache_sized | 04:55:52.012 | 480.9 |  | container log: (EngineCore pid=161) INFO 10-15 04:55:52 [gpu_worker.py:640] Available KV cache memory: 17.33 GiB |
| sampler.engine_init_done | 04:56:02.777 | 491.7 | 57.45 | container log: (EngineCore pid=161) INFO 10-15 04:56:02 [core.py:372] init engine (profile, create kv cache, warmup model) took 57.45 s |
| sampler.ready_key | 04:56:04.000 | 492.9 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen2.5-0.5B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 04:47:51.110 | -0.0 |
| client.create_model.start | 04:47:51.111 | 0.0 |
| client.create_model.done | 04:51:08.186 | 197.1 |
| client.forward_backward.start | 04:51:10.937 | 199.8 |
| client.forward_backward.done | 04:51:14.170 | 203.1 |
| client.optim_step.done | 04:51:15.072 | 204.0 |
| client.save_weights_and_get_sampling_client.start | 04:51:15.072 | 204.0 |
| client.save_weights_and_get_sampling_client.done | 04:51:15.661 | 204.6 |
| client.sample.start | 04:51:15.662 | 204.6 |
| client.sample.done | 04:56:04.835 | 493.7 |
| client.end | 04:56:04.838 | 493.7 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.9 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 172.0 |
| container create | trainer.event_Pulled | trainer.container_started | 1.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 7.2 |
| weights load | trainer.load_start | trainer.load_done | 15.0 |
| ready key | trainer.load_done | trainer.ready_key | 0.7 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 197.1 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.2 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.1 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 186.0 |
| container create | sampler.event_Pulled | sampler.container_started | 1.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 23.2 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 15.2 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.7 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 2.4 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 31.5 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 26.7 |
| ready key | sampler.engine_init_done | sampler.ready_key | 1.2 |
| client: first sample blocked | client.sample.start | client.sample.done | 289.2 |

