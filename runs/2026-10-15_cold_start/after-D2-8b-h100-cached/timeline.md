# Cold start timeline: after-D2-8b-h100-cached

t=0 is the client's `create_model` call at 2026-10-15T05:15:36.633+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-qwen-qwen3-8b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.ready_key | 05:15:34.000 | -2.6 |  | redis poller: open_rl:trainer_ready:Qwen/Qwen2.5-0.5B |
| trainer.workload_created | 05:15:37.000 | 0.4 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 05:15:37.000 | 0.4 |  | pod creationTimestamp (apiserver) |
| trainer.pod_PodScheduled | 05:15:37.000 | 0.4 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 05:15:37.000 | 0.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.workload_first_seen_by_poller | 05:15:37.332 | 0.7 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 05:15:37.332 | 0.7 |  | workloads.tsv poller: status.podName set |
| trainer.pod_Initialized | 05:15:38.000 | 1.4 |  | pod condition Initialized |
| trainer.workload_reports_node | 05:15:38.981 | 2.3 |  | workloads.tsv poller: status.nodeName set |
| trainer.container_started | 05:15:41.000 | 4.4 |  | containerStatuses.state.running.startedAt |
| trainer.event_Pulled | 05:15:41.000 | 4.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| trainer.event_Created | 05:15:41.000 | 4.4 |  | event: Container created |
| trainer.event_Started | 05:15:41.000 | 4.4 |  | event: Container started |
| trainer.condition_Placed_True | 05:15:42.000 | 5.4 |  | Workload condition Placed |
| trainer.pod_Ready | 05:15:42.000 | 5.4 |  | pod condition Ready |
| trainer.pod_ContainersReady | 05:15:42.000 | 5.4 |  | pod condition ContainersReady |
| trainer.process_first_log | 05:15:45.886 | 9.3 |  | container log, first line |
| trainer.load_start | 05:15:46.108 | 9.5 |  | container log: Loading base model Qwen/Qwen3-8B to cuda:0... |
| trainer.load_done | 05:15:58.258 | 21.6 |  | container log: Successfully loaded. |
| trainer.processor_started | 05:15:58.259 | 21.6 |  | container log: [WORKER] Training requests processor started (model=Qwen/Qwen3-8B active_set=Qwen/Qwen3-8B-1). |

## sampler: `orw-lora-qwen-qwen3-8b-0-sampler` on `gke-open-rl-dra-h100-sampler-a721f127-0853`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| sampler.workload_created | 05:16:09.000 | 32.4 |  | Workload creationTimestamp (apiserver) |
| sampler.pod_created | 05:16:09.000 | 32.4 |  | pod creationTimestamp (apiserver) |
| sampler.pod_Initialized | 05:16:09.000 | 32.4 |  | pod condition Initialized |
| sampler.pod_PodScheduled | 05:16:09.000 | 32.4 |  | pod condition PodScheduled |
| sampler.event_Scheduled | 05:16:09.000 | 32.4 |  | event: Successfully assigned openrl-system/orw-lora-qwen-qwen3-8b-0-sampler to gke-open-rl-dra-h100-sampler-a721f127-0853 |
| sampler.workload_first_seen_by_poller | 05:16:09.599 | 33.0 |  | workloads.tsv poller (laptop clock, 1 s) |
| sampler.workload_reports_pod | 05:16:09.599 | 33.0 |  | workloads.tsv poller: status.podName set |
| sampler.workload_reports_node | 05:16:09.599 | 33.0 |  | workloads.tsv poller: status.nodeName set |
| sampler.condition_Placed_True | 05:16:13.000 | 36.4 |  | Workload condition Placed |
| sampler.pod_Ready | 05:16:13.000 | 36.4 |  | pod condition Ready |
| sampler.pod_ContainersReady | 05:16:13.000 | 36.4 |  | pod condition ContainersReady |
| sampler.container_started | 05:16:13.000 | 36.4 |  | containerStatuses.state.running.startedAt |
| sampler.event_Pulled | 05:16:13.000 | 36.4 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| sampler.event_Created | 05:16:13.000 | 36.4 |  | event: Container created |
| sampler.event_Started | 05:16:13.000 | 36.4 |  | event: Container started |
| sampler.process_first_log | 05:16:28.490 | 51.9 |  | container log, first line |
| sampler.engine_init_start | 05:16:38.151 | 61.5 |  | container log: (EngineCore pid=169) INFO 10-15 05:16:38 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='Qwen/Q |
| sampler.load_start | 05:16:39.244 | 62.6 |  | container log: (EngineCore pid=169) INFO 10-15 05:16:39 [model_runner.py:396] Loading model from scratch... |
| sampler.prefetch_start | 05:16:40.237 | 63.6 |  | container log: (EngineCore pid=169) INFO 10-15 05:16:40 [weight_utils.py:857] Prefetching checkpoint files into page cache started (in  |
| sampler.weights_loaded | 05:16:52.391 | 75.8 | 12.16 | container log: (EngineCore pid=169) INFO 10-15 05:16:52 [default_loader.py:430] Loading weights took 12.16 seconds |
| sampler.model_loaded | 05:16:53.020 | 76.4 | 14.053276 | container log: (EngineCore pid=169) INFO 10-15 05:16:53 [model_runner.py:428] Model loading took 18.17 GiB memory and 14.053276 seconds |
| sampler.compile_done | 05:17:29.578 | 112.9 | 35.52 | container log: (EngineCore pid=169) INFO 10-15 05:17:29 [monitor.py:53] torch.compile took 35.52 s in total |
| sampler.graphs_captured | 05:17:49.866 | 133.2 | 16.0 | container log: (EngineCore pid=169) INFO 10-15 05:17:49 [model_runner.py:1066] Graph capturing finished in 16 secs, took 0.78 GiB |
| sampler.kv_cache_sized | 05:17:50.489 | 133.9 |  | container log: (EngineCore pid=169) INFO 10-15 05:17:50 [gpu_worker.py:640] Available KV cache memory: 50.17 GiB |
| sampler.engine_init_done | 05:18:09.053 | 152.4 | 76.03 | container log: (EngineCore pid=169) INFO 10-15 05:18:09 [core.py:372] init engine (profile, create kv cache, warmup model) took 76.03 s |
| sampler.ready_key | 05:18:10.000 | 153.4 |  | redis poller: open_rl:sampler_ready:Qwen/Qwen3-8B |

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 05:15:36.633 | 0.0 |
| client.create_model.start | 05:15:36.633 | 0.0 |
| client.create_model.done | 05:16:00.765 | 24.1 |
| client.forward_backward.start | 05:16:03.882 | 27.2 |
| client.forward_backward.done | 05:16:06.825 | 30.2 |
| client.optim_step.done | 05:16:08.923 | 32.3 |
| client.save_weights_and_get_sampling_client.start | 05:16:08.924 | 32.3 |
| client.save_weights_and_get_sampling_client.done | 05:16:11.071 | 34.4 |
| client.sample.start | 05:16:11.076 | 34.4 |
| client.sample.done | 05:18:10.965 | 154.3 |
| client.end | 05:18:10.966 | 154.3 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.4 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 4.0 |
| container create | trainer.event_Pulled | trainer.container_started | 0.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 5.1 |
| weights load | trainer.load_start | trainer.load_done | 12.1 |
| ready key | trainer.load_done | trainer.ready_key | -24.3 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 24.1 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 2.9 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | 0.1 |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | 0.0 |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | 4.0 |
| container create | sampler.event_Pulled | sampler.container_started | 0.0 |
| process start to first log | sampler.container_started | sampler.process_first_log | 15.5 |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | 9.7 |
| engine init until load start | sampler.engine_init_start | sampler.load_start | 1.1 |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | 13.1 |
| torch.compile | sampler.weights_loaded | sampler.compile_done | 37.2 |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | 39.5 |
| ready key | sampler.engine_init_done | sampler.ready_key | 0.9 |
| client: first sample blocked | client.sample.start | client.sample.done | 119.9 |

