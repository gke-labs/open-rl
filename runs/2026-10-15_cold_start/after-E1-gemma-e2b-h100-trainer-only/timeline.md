# Cold start timeline: after-E1-gemma-e2b-h100-trainer-only

t=0 is the client's `create_model` call at 2026-10-15T05:19:16.956+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer: `orw-lora-google-gemma-4-e2b-0-trainer` on `gke-open-rl-dra-h100-trainer-702095b4-1g3a`

| Mark | Clock (UTC) | t+ s | Self-reported s | Source |
|---|---|---|---|---|
| trainer.workload_created | 05:19:17.000 | 0.0 |  | Workload creationTimestamp (apiserver) |
| trainer.pod_created | 05:19:17.000 | 0.0 |  | pod creationTimestamp (apiserver) |
| trainer.workload_first_seen_by_poller | 05:19:17.844 | 0.9 |  | workloads.tsv poller (laptop clock, 1 s) |
| trainer.workload_reports_pod | 05:19:17.844 | 0.9 |  | workloads.tsv poller: status.podName set |
| trainer.workload_reports_node | 05:19:17.844 | 0.9 |  | workloads.tsv poller: status.nodeName set |
| trainer.pod_Initialized | 05:19:18.000 | 1.0 |  | pod condition Initialized |
| trainer.pod_PodScheduled | 05:19:18.000 | 1.0 |  | pod condition PodScheduled |
| trainer.event_Scheduled | 05:19:18.000 | 1.0 |  | event: Successfully assigned openrl-system/orw-lora-google-gemma-4-e2b-0-trainer to gke-open-rl-dra-h100-trainer-702095b4-1g3a |
| trainer.event_Pulled | 05:19:21.000 | 4.0 | None | event: Container image "us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-server:coldstart1-dca985a" already present on machine and  |
| trainer.event_Created | 05:19:21.000 | 4.0 |  | event: Container created |
| trainer.condition_Placed_True | 05:19:22.000 | 5.0 |  | Workload condition Placed |
| trainer.pod_Ready | 05:19:22.000 | 5.0 |  | pod condition Ready |
| trainer.pod_ContainersReady | 05:19:22.000 | 5.0 |  | pod condition ContainersReady |
| trainer.container_started | 05:19:22.000 | 5.0 |  | containerStatuses.state.running.startedAt |
| trainer.event_Started | 05:19:22.000 | 5.0 |  | event: Container started |
| trainer.process_first_log | 05:19:26.404 | 9.4 |  | container log, first line |
| trainer.load_start | 05:19:26.629 | 9.7 |  | container log: Loading base model google/gemma-4-e2b to cuda:0... |
| trainer.ready_key | 05:20:59.000 | 102.0 |  | redis poller: open_rl:trainer_ready:google/gemma-4-e2b |
| trainer.load_done | 05:20:59.148 | 102.2 |  | container log: Successfully loaded. |
| trainer.processor_started | 05:20:59.149 | 102.2 |  | container log: [WORKER] Training requests processor started (model=google/gemma-4-e2b active_set=google/gemma-4-e2b-1). |

## sampler

no worker observed

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 05:19:16.955 | -0.0 |
| client.create_model.start | 05:19:16.956 | 0.0 |
| client.create_model.done | 05:21:00.658 | 103.7 |
| client.forward_backward.start | 05:21:04.495 | 107.5 |
| client.forward_backward.done | 05:21:08.031 | 111.1 |
| client.optim_step.done | 05:21:09.190 | 112.2 |
| client.end | 05:21:09.196 | 112.2 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | 0.0 |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | 1.0 |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | 3.0 |
| container create | trainer.event_Pulled | trainer.container_started | 1.0 |
| process start to load start (imports) | trainer.container_started | trainer.load_start | 4.6 |
| weights load | trainer.load_start | trainer.load_done | 92.5 |
| ready key | trainer.load_done | trainer.ready_key | -0.1 |
| client: create_model blocked | client.create_model.start | client.create_model.done | 103.7 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | 3.5 |

### sampler

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.save_weights_and_get_sampling_client.start | sampler.pod_created | unmeasured |
| pod scheduled | sampler.pod_created | sampler.pod_PodScheduled | unmeasured |
| image pull (0 if cached) | sampler.pod_PodScheduled | sampler.event_Pulled | unmeasured |
| container create | sampler.event_Pulled | sampler.container_started | unmeasured |
| process start to first log | sampler.container_started | sampler.process_first_log | unmeasured |
| imports until engine init | sampler.process_first_log | sampler.engine_init_start | unmeasured |
| engine init until load start | sampler.engine_init_start | sampler.load_start | unmeasured |
| weights load (vLLM prefetch into page cache + load) | sampler.load_start | sampler.weights_loaded | unmeasured |
| torch.compile | sampler.weights_loaded | sampler.compile_done | unmeasured |
| profiling, KV cache, graph capture | sampler.compile_done | sampler.engine_init_done | unmeasured |
| ready key | sampler.engine_init_done | sampler.ready_key | unmeasured |
| client: first sample blocked | client.sample.start | client.sample.done | unmeasured |

