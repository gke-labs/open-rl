# Custom trainer images

OpenRL ships with built-in trainers. If you want to train with your own code
instead, put that code in a container image and tell OpenRL to use it.

## Using your image

Pass the image when you start your training script:

```bash
TINKER_TAGS=openrl.trainer_backend=ghcr.io/acme/my-trainer:1.0 python my_rl_loop.py
```

Your script does not change. It still talks to the OpenRL API with the Tinker
SDK. OpenRL starts your image as the trainer for that job, and every training
call your script makes (`forward_backward`, `optim_step`, saving weights, and so
on) is handed to your code.

This needs OpenRL running on Kubernetes with the scheduler, since that is what
starts trainer pods.

## Building your image

Your image needs two things:

1. **OpenRL's server code.** The easiest way is to start from the OpenRL server
   image.
2. **The name of your trainer class**, in the environment variable
   `OPEN_RL_TRAINER_BACKEND`, written as `module:Class`.

```dockerfile
FROM ghcr.io/gke-labs/open-rl/server:latest
COPY my_trainer.py /app/my_trainer.py
ENV OPEN_RL_TRAINER_BACKEND=my_trainer:MyTrainer
```

When the job starts, OpenRL runs its own small program inside your container.
That program receives the job's training calls and passes each one to your
class.

## Writing the trainer class

Your class needs one method per training operation:

```python
class MyTrainer:
  def load_base_model(self, base_model): ...
  def create_model(self, base_model, model_id, config): ...
  def forward_backward(self, data, loss_fn, loss_config=None, model_id=None, forward_only=False): ...
  def optim_step(self, adam_params, model_id): ...
  def save_for_sampler(self, model_id, alias, ref): ...
  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"): ...
  def load_from_state(self, model_id, state_path, restore_optimizer=False): ...
  def delete_model(self, model_id): ...
```

What each one does:

- **`load_base_model`** loads the base model's weights. It runs once, when the
  trainer starts.
- **`create_model`** sets up a model to train, such as a new LoRA adapter.
- **`forward_backward`** runs a batch through the model and computes gradients.
  Return the per-token log probabilities for each example in the batch, like
  `{"loss_fn_outputs": [{"logprobs": {"data": [...], "dtype": "float32", "shape": [n]}}, ...], "metrics": {...}}`.
- **`optim_step`** applies the gradients. Return `{"metrics": {...}}`.
- **`save_for_sampler`** writes the current weights where the sampler can load
  them. For LoRA, save the adapter to
  `$OPEN_RL_TMP_DIR/peft/<model_id>/<model_id>`.
- **`save_state`** saves a checkpoint to `state_path` and returns
  `{"path": state_path}`.
- **`load_from_state`** loads a checkpoint, so a job can resume.
- **`delete_model`** frees the model when the job is done.

A few setups need more. Full fine-tuning on shared GPUs also calls `wake_up()`
and `sleep()` and reads a `cpu_offload` attribute. Sampling with
`SAMPLING_BACKEND=torch` also calls `generate(...)`.

## What your container gets

- Your job's GPUs. Set `openrl.trainer_gpus=N` for more than one; OpenRL then
  starts one process per GPU with `torchrun`.
- The shared volume at `/mnt/shared`, where weights and checkpoints go.
- `BASE_MODEL`, `OPEN_RL_TMP_DIR`, `HF_HOME` and `REDIS_URL` in the environment.

## Testing

`tests/test_trainer_image_plug.py` runs a small trainer through the same code
that will call yours. Copy it to check your class before building an image.
