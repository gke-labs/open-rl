import asyncio
import os
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.train import Config as TrainConfig
from tinker_cookbook.supervised.train import main as train
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")


@chz.chz
class GSM8KDataset(SupervisedDatasetBuilder):
  model_name: str
  batch_size: int = 16
  max_length: int = 640
  seed: int = 0

  def __call__(self):
    tokenizer = get_tokenizer(self.model_name)
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=self.seed)

    def make_datum(row: dict) -> tinker.Datum:
      prompt = tokenizer.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
      completion = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
      tokens = (prompt + completion)[: self.max_length]
      weights = ([0] * len(prompt) + [1] * len(completion))[: self.max_length]
      return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=cast(Any, {"target_tokens": tokens[1:], "weights": [float(w) for w in weights[1:]]}),
      )

    return SupervisedDatasetFromHFDataset(dataset, self.batch_size, map_fn=make_datum), None


@chz.chz
class Config:
  base_model: str = "Qwen/Qwen2.5-0.5B"
  base_url: str = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:9003"))
  log_path: str = str(Path(__file__).with_name("artifacts") / "gsm8k_sft")
  epochs: int = 1
  batch: int = 16
  lr: float = 2e-5
  rank: int = 32
  max_len: int = 640
  seed: int = 0
  max_steps: int | None = None
  save_every: int = 0
  behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"


def main(config: Config) -> None:
  cli_utils.check_log_dir(config.log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
  asyncio.run(
    train(
      TrainConfig(
        log_path=config.log_path,
        model_name=config.base_model,
        dataset_builder=GSM8KDataset(
          model_name=config.base_model,
          batch_size=config.batch,
          max_length=config.max_len,
          seed=config.seed,
        ),
        learning_rate=config.lr,
        lr_schedule="cosine",
        num_epochs=config.epochs,
        lora_rank=config.rank,
        base_url=config.base_url,
        save_every=config.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        max_steps=config.max_steps,
      )
    )
  )
  checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
  if checkpoint is None:
    checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="state_path")
  if checkpoint is not None:
    path = checkpoint.sampler_path or checkpoint.state_path
    if path and path.startswith("tinker://"):
      path = str(Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")) / "sampler_full" / path.removeprefix("tinker://"))
    if path:
      print(f"eval_model_path={path}")


if __name__ == "__main__":
  chz.nested_entrypoint(main, allow_hyphens=True)
