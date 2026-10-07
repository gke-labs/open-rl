"""Weight updates for vllm-torchtpu's TPUWorker, mixed in through vLLM's worker_extension_cls.

TPUWorker has none of the GPU worker's weight-transfer methods, nor
reset_encoder_cache, which pause_generation(clear_cache=True) calls. These
mirror vLLM's GPU worker for one model with no draft. Only the TPU sampler
sets this: vLLM refuses an extension whose names the worker already defines.
"""

from typing import Any

from vllm.config import set_current_vllm_config
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory


class TPUWeightTransferExtension:
  _open_rl_transfer_engine: WeightTransferEngine | None = None

  def reset_encoder_cache(self) -> None:
    self.model_runner.reset_encoder_cache()

  def _open_rl_engine(self) -> WeightTransferEngine:
    # Created on first use: TPUWorker builds no engine at load time and drops self.device.
    if self._open_rl_transfer_engine is None:
      config = self.vllm_config.weight_transfer_config
      if config is None:
        raise RuntimeError("Weight transfer not configured. Please set weight_transfer_config to enable weight transfer.")
      model = self.model_runner.get_model()
      device = next(model.parameters()).device
      self._open_rl_transfer_engine = WeightTransferEngineFactory.create_engine(config, self.vllm_config, device, model)
    return self._open_rl_transfer_engine

  def start_weight_update(self) -> None:
    with set_current_vllm_config(self.vllm_config):
      self._open_rl_engine().start_weight_update()

  def update_weights(self, update_info: dict[str, Any]) -> None:
    with set_current_vllm_config(self.vllm_config):
      self._open_rl_engine().update_weights(update_info)

  def finish_weight_update(self) -> None:
    with set_current_vllm_config(self.vllm_config):
      self._open_rl_engine().finish_weight_update()
