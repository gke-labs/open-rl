# Using the tinker-cookbook skills against OpenRL

OpenRL serves the Tinker API, so Thinking Machines' tinker-cookbook skills are
the reference for how to train: methodology, SFT, RL, DPO, distillation,
hyperparameters, evaluation, logging, checkpoints, and SDK-level debugging.
This skill does not repeat any of it.

| Skill | Use it for |
|---|---|
| `research` | planning and running experiments: recipes, model choice, hyperparameters, evaluation, checkpoints, reading `metrics.jsonl` and rollout logs |
| `debug` | the SDK's background thread and GIL contention, renderer and tokenization problems, outputs that differ from vLLM, the error decoder |
| `inkling` | not applicable: Inkling models are hosted only and do not run on OpenRL |

They are directories under `skills/` in
github.com/thinking-machines-lab/tinker-cookbook, in the same `SKILL.md`
plus `references/` format as this skill. Install them the way your agent
installs skills: copy `skills/research` and `skills/debug` from a clone of
the cookbook into the directory your agent reads skills from, or point the
agent at their `SKILL.md` files. Claude Code users can instead run
`/plugin marketplace add thinking-machines-lab/tinker-cookbook` and
`/plugin install tinker@tinker-skills`, after which the skills answer to
`/tinker:research` and `/tinker:debug`.

The cookbook skills assume the hosted Tinker service. Where they do, this is
what is true on OpenRL instead.

| The cookbook skill says | On OpenRL |
|---|---|
| `TINKER_API_KEY` authenticates you | There is no auth. Set it to any string because the SDK refuses to start without one. The skills never mention `TINKER_BASE_URL`; set it (or `base_url=`) to the OpenRL endpoint, see `connect.md` |
| choose from the model lineup; prefer MoE because cost scales with active parameters | Any Hugging Face model id the cluster can load; the first use downloads it there. `get_server_capabilities` -> `openrl.models.examples` names the ones the examples use. No Inkling. The cost is your cluster's GPU time, not tokens |
| `tinker run list`; `tinker checkpoint list`, `info`, `publish`, `set-ttl`, `delete`; `rest.list_user_checkpoints()` | Not implemented. `tinker://` paths work as described: `save_state` returns `tinker://<model_id>/weights/<name>`, `save_weights_for_sampler` returns `tinker://<model_id>/sampler_weights/<name>`, and both load through `load_state`, `create_training_client_from_state` and `create_sampling_client`. Keep the paths your run printed, or read `checkpoints.jsonl` in the log directory; there is nothing to list. `GET /api/v1/list_adapters` lists saved LoRA adapter directories |
| a recipe resumes by listing checkpoints | Pass the `tinker://` path explicitly, or set `save_every=0` unless you have verified resume on your server |
| `weights.download(tinker_path=...)`, `tinker checkpoint download`, `push-hf` | Not implemented. The files are on the API server's `OPEN_RL_TMP_DIR`, the shared volume on Kubernetes; copy them out of the api-server pod or ask the platform owner for the mount. `weights.build_hf_model` and `build_lora_adapter` then work on the copied adapter directory |
| HTTP 402 is billing, 429 is the rate limit | Neither happens. A 400 that names an `openrl.*` setting is OpenRL's, see `runbook.md` |
| "Is Tinker down?": run the smoke test | `GET /api/v1/healthz`, then `get_server_capabilities` and read the `openrl` key. No status page; the platform owner has the API server log |
| `Session heartbeat failed` means the SDK thread is blocked (GIL contention) | On a laptop it usually means the port-forward dropped. Check `healthz` through the client's URL first; if that is fine, follow the `debug` skill. The server releases the workers 120 s after the last heartbeat, see `lifecycle.md` |
| a slow step is GIL contention, a service issue, or a large batch | On OpenRL the first step of a run waits for workers to start (minutes on a cold node) and LoRA jobs on one base model share a trainer. Read the model view (`connect.md`) before profiling the client |
| the newest SDK | The repository's examples pin `tinker==0.29.0` and `tinker-cookbook[math-rl]==0.5.7`. Servers from v0.0.4 on speak the protobuf wire that SDKs >= 0.25 use. Methods the cookbook skills mention that are newer than the pin may not exist on your server; `docs/tinker-client-compatibility.md` in the repository is the generated list |

## Supported Tinker surface

From `docs/tinker-client-compatibility.md`, generated against the server by
calling every public SDK method:

- `ServiceClient`: `create_lora_training_client`, `create_sampling_client`,
  `create_training_client_from_state` (with and without optimizer), `get_server_capabilities`.
- `TrainingClient`: `forward`, `forward_backward`, `optim_step`, `save_state`, `load_state`
  (with and without optimizer), `save_weights_for_sampler`,
  `save_weights_and_get_sampling_client`, `create_sampling_client`, `get_info`, `get_tokenizer`.
- `SamplingClient`: `sample`, `compute_logprobs`, `get_base_model`, `get_tokenizer`.

Not supported: the `RestClient` checkpoint listing, deletion, archive and publish calls, and
the session, billing and audit APIs.

Full fine-tuning is a setting, not a different client: `openrl.fine_tuning_type=full` on the
session or the model. The old `x-open-rl-fine-tuning-type` header is refused with a 400
that names the setting.
