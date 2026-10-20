# Cold start timeline: D-8b-h100-cached

t=0 is the client's `create_model` call at 2026-10-15T03:39:03.686+00:00 UTC. Offsets are seconds after t=0; a negative offset is before it.

## trainer

no worker observed

## sampler

no worker observed

## client

| Phase | Clock (UTC) | t+ s |
|---|---|---|
| client.start | 03:39:03.686 | 0.0 |
| client.create_model.start | 03:39:03.686 | 0.0 |
| client.create_model.done | 03:48:18.935 | 555.2 |
| client.forward_backward.start | 03:48:21.731 | 558.0 |

## Critical path (derived from the marks above)

### trainer

| Phase | From | To | Seconds |
|---|---|---|---|
| placement: Workload to pod | client.create_model.start | trainer.pod_created | unmeasured |
| pod scheduled | trainer.pod_created | trainer.pod_PodScheduled | unmeasured |
| image pull (0 if cached) | trainer.pod_PodScheduled | trainer.event_Pulled | unmeasured |
| container create | trainer.event_Pulled | trainer.container_started | unmeasured |
| process start to load start (imports) | trainer.container_started | trainer.load_start | unmeasured |
| weights load | trainer.load_start | trainer.load_done | unmeasured |
| ready key | trainer.load_done | trainer.ready_key | unmeasured |
| client: create_model blocked | client.create_model.start | client.create_model.done | 555.2 |
| client: first forward_backward | client.forward_backward.start | client.forward_backward.done | unmeasured |

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

