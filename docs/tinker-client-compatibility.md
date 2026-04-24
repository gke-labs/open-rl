# Tinker Client Compatibility

Generated from `tinker==0.18.1` by
`examples/tests/tinker_client_compat.py`.

The test discovers public Tinker client methods with `dir()` and `inspect`,
starts the real Open-RL FastAPI gateway in single-process mode with a tiny
local model fixture, lets the SDK fetch server bootstrap config, calls each
discovered method
with small fixture arguments, and records whether the call succeeds before
the probe timeout.

- Supported methods: 36
- Unsupported methods: 43

## Supported Methods

### RestClient

- `get_telemetry`

### SamplingClient

- `compute_logprobs`
- `compute_logprobs_async`
- `get_telemetry`
- `on_queue_state_change`
- `sample`
- `sample_async`

### ServiceClient

- `create_lora_training_client`
- `create_lora_training_client_async`
- `create_rest_client`
- `create_sampling_client`
- `create_sampling_client_async`
- `get_server_capabilities`
- `get_server_capabilities_async`
- `get_telemetry`

### TrainingClient

- `create_sampling_client`
- `create_sampling_client_async`
- `forward_backward`
- `forward_backward_async`
- `get_info`
- `get_info_async`
- `get_telemetry`
- `get_tokenizer`
- `load_state`
- `load_state_async`
- `load_state_with_optimizer`
- `load_state_with_optimizer_async`
- `optim_step`
- `optim_step_async`
- `save_state`
- `save_state_async`
- `save_weights_and_get_sampling_client`
- `save_weights_and_get_sampling_client_async`
- `save_weights_and_get_sampling_client_submit`
- `save_weights_for_sampler`
- `save_weights_for_sampler_async`

## Unsupported Methods

### RestClient

- `delete_checkpoint`
- `delete_checkpoint_async`
- `delete_checkpoint_from_tinker_path`
- `delete_checkpoint_from_tinker_path_async`
- `get_checkpoint_archive_url`
- `get_checkpoint_archive_url_async`
- `get_checkpoint_archive_url_from_tinker_path`
- `get_checkpoint_archive_url_from_tinker_path_async`
- `get_sampler`
- `get_sampler_async`
- `get_session`
- `get_session_async`
- `get_training_run`
- `get_training_run_async`
- `get_training_run_by_tinker_path`
- `get_training_run_by_tinker_path_async`
- `get_weights_info_by_tinker_path`
- `list_checkpoints`
- `list_checkpoints_async`
- `list_sessions`
- `list_sessions_async`
- `list_training_runs`
- `list_training_runs_async`
- `list_user_checkpoints`
- `list_user_checkpoints_async`
- `publish_checkpoint_from_tinker_path`
- `publish_checkpoint_from_tinker_path_async`
- `set_checkpoint_ttl_from_tinker_path`
- `set_checkpoint_ttl_from_tinker_path_async`
- `unpublish_checkpoint_from_tinker_path`
- `unpublish_checkpoint_from_tinker_path_async`

### SamplingClient

- `create`
- `get_base_model`
- `get_base_model_async`
- `get_tokenizer`

### ServiceClient

- `create_training_client_from_state`
- `create_training_client_from_state_async`
- `create_training_client_from_state_with_optimizer`
- `create_training_client_from_state_with_optimizer_async`

### TrainingClient

- `forward`
- `forward_async`
- `forward_backward_custom`
- `forward_backward_custom_async`
