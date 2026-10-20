# Cold start timeline: after-D1-8b-h100-cached

t=0 is the client's `create_model` call at 2026-10-15T05:11:45.157+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen3-8b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.ready_key | 05:11:41.000 | -4.2 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.workload_created | 05:11:45.000 | -0.2 |  | Workload creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 05:11:45.752 | 0.6 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 05:11:45.752 | 0.6 |  | workloads.tsv poller: status.podName set |
| trainer.pod_created | 05:11:46.000 | 0.8 |  | pod creationTimestamp (apiserver) |
| trainer.pod_Initialized | 05:11:46.000 | 0.8 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 05:11:46.000 | 0.8 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 05:11:46.000 | 0.8 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.workload_reports_node | 05:11:47.402 | 2.2 |  | workloads.tsv poller: status.nodeName set |
| trainer.container_started | 05:11:49.000 | 3.8 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 05:11:49.000 | 3.8 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| trainer.event_Created | 05:11:49.000 | 3.8 |  | event: Container created |
| trainer.event_Started | 05:11:49.000 | 3.8 |  | event: Container started |
| trainer.condition_Placed_True | 05:11:50.000 | 4.8 |  | Workload condition Placed |
| trainer.pod_Ready | 05:11:50.000 | 4.8 |  | pod condition Ready |
| trainer.pod_ContainersReady | 05:11:50.000 | 4.8 |  | pod condition ContainersReady |
| trainer.process_first_log | 05:11:54.291 | 9.1 |  | container log, first line |
| trainer.load_start | 05:11:54.511 | 9.4 |  | container log: Loading base model Qwen/Qwen3-8B to cuda:0... |
| trainer.load_done | 05:12:08.031 | 22.9 |  | container log: Successfully loaded. |
| trainer.processor_started | 05:12:08.033 | 22.9 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen3-8B active_set=Qwen/Qwen3-8B-1). |

## sampler: `orw-lora-qwen-qwen3-8b-0-sampler` on `gke-open-rl-dra-h100-sampler-a721f127-0853`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 05:12:18.000 | 32.8 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 05:12:18.000 | 32.8 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 05:12:18.000 | 32.8 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 05:12:18.000 | 32.8 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 05:12:18.000 | 32.8 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-sampler to gke-open-rl-dra-h100-sampler-a721f127-0853 |
| sampler.workload_first_seen_by_poller | 05:12:18.877 | 33.7 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 05:12:18.877 | 33.7 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 05:12:18.877 | 33.7 |  | workloads.tsv poller: status.nodeName set |
| sampler.container_started | 05:12:22.000 | 36.8 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 05:12:22.000 | 36.8 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| sampler.event_Created | 05:12:22.000 | 36.8 |  | event: Container created |
| sampler.event_Started | 05:12:22.000 | 36.8 |  | event: Container started |
| sampler.condition_Placed_True | 05:12:23.000 | 37.8 |  | Workload condition Placed |
| sampler.pod_Ready | 05:12:23.000 | 37.8 |  | pod condition Ready |
| sampler.pod_ContainersReady | 05:12:23.000 | 37.8 |  | pod condition ContainersReady |
| sampler.process_first_log | 05:12:37.408 | 52.3 |  | container log, first line |
| sampler.engine_init_start | 05:12:46.943 | 61.8 |  | container log: (EngineCore pid=169) INFO 10-15 05:12:46 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 05:12:47.997 | 62.8 |  | container log: (EngineCore pid=169) INFO 10-15 05:12:47 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 05:12:48.986 | 63.8 |  | container log: (EngineCore pid=169) INFO 10-15 05:12:48 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 05:13:00.893 | 75.7 | 11.91 | container log: (EngineCore pid=169) INFO 10-15 05:13:00 [default_loader.py:430] Loading weights took 11.91 seconds |
| sampler.model_loaded | 05:13:01.578 | 76.4 | 13.84558 | container log: (EngineCore pid=169) INFO 10-15 05:13:01 [model_runner.py:428] Model loading took 18.17 GiB memory and 13.845580 seconds |
| sampler.compile_done | 05:13:38.517 | 113.4 | 35.91 | container log: (EngineCore pid=169) INFO 10-15 05:13:38 [monitor.py:53] torch.compile took 35.91 s in total |
| sampler.graphs_captured | 05:13:58.672 | 133.5 | 16.0 | container log: (EngineCore pid=169) INFO 10-15 05:13:58 [model_runner.py:1066] Graph capturing finished in 16 secs, took 0.78 GiB |
| sampler.kv_cache_sized | 05:13:59.319 | 134.2 |  | container log: (EngineCore pid=169) INFO 10-15 05:13:59 [gpu_worker.py:640] Available KV cache memory: 50.17 GiB |
| sampler.engine_init_done | 05:14:17.795 | 152.6 | 76.21 | container log: (EngineCore pid=169) INFO 10-15 05:14:17 [core.py:372] init engine (profile, create kv cache, warmup model) took 76.21 s |
| sampler.ready_key | 05:14:18.000 | 152.8 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen3-8B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 05:11:45.156 | -0.0 |
| client.create_model.start | 05:11:45.157 | 0.0 |
| client.create_model.done | 05:12:10.447 | 25.3 |
| client.forward_backward.start | 05:12:13.323 | 28.2 |
| client.forward_backward.done | 05:12:16.232 | 31.1 |
| client.optim_step.done | 05:12:18.352 | 33.2 |
| client.save_weights_and_get_sampling_client.start | 05:12:18.353 | 33.2 |
| client.save_weights_and_get_sampling_client.done | 05:12:20.273 | 35.1 |
| client.sample.start | 05:12:20.275 | 35.1 |
| client.sample.done | 05:14:19.688 | 154.5 |
| client.end | 05:14:19.689 | 154.5 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.8 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 3.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 5.5 |
| weights load | trainer.load_start | trainer.load_done | 13.5 |
| ready key | trainer.load_done | trainer.ready_key | -27.0 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 25.3 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 2.9 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | -0.4 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 4.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 15.4 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 9.5 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.1 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 12.9 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 37.6 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 39.3 |
| ready key | sampler.engine_init_done | sampler.ready_key | 0.2 |
| client: first sample blocked | client.sample.start | client.sample.done | 119.4 |

