# 1. Scope & MVP Endpoints
The Tinker API supports a lot of endpoints, from creating LoRA training clients to sampling and tracking completions.
Should the MVP strict implement only the core SFT endpoints (e.g. `create_lora_training_client`, `forward_backward`, `optim_step`, `save_weights_and_get_sampling_client`, `sample`), or are there other parts of the Tinker surface (like tracking runs or metrics) we must have on day one?

# 2. Server Architecture
You mentioned using OSS libraries (vLLM for inference, Huggingface TRL for training).
Should our API server act as a single monolithic Python process that directly imports and manages vLLM/TRL states internally, or will it orchestrate separate backend processes/containers for training and inference?

# 3. State Management
Tinker supports training updates with multiple `forward_backward` calls followed by `optim_step`. This inherently requires the backend string optimizer state and model gradients in memory between API calls.
For our MVP (single user, single model), is it acceptable to simply hold the PyTorch state in the API server's RAM/VRAM globally, or do we need some robust multi-session isolation?

# 4. LoRA Specifics
We're building an SFT MVP first. Does the SFT process under the hood strictly use LoRA (as Tinker’s `create_lora_training_client` implies)? Is there a specific configuration for LoRA (e.g., specific modules) or QLoRA that you want it to default to?

# 5. Client Library
Will you be using an existing `tinker` Python library (meaning we must exactly mimic the wire protocol and API schema of Thinker), or are we also building a custom `cute-rl` Python client that implements the Tinker-like interfaces described in the docs?
