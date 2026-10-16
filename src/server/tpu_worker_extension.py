"""Weight updates for the TPU platform's worker, mixed in through vLLM's worker_extension_cls.

The TPU worker has none of the GPU worker's weight-transfer methods, nor
reset_encoder_cache, which the sampler calls after each update. These mirror
vLLM's GPU worker for one model with no draft model.
"""

from typing import Any

from vllm.config import set_current_vllm_config
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

from server.tpu_weights import FLIPPED_ATTR, flipped_modules


class TPUWeightTransferExtension:
  _open_rl_transfer_engine: WeightTransferEngine | None = None

  def reset_encoder_cache(self) -> None:
    self.model_runner.reset_encoder_cache()

  def _open_rl_engine(self) -> WeightTransferEngine:
    # Created on first use: the TPU worker builds no engine at load time and keeps no self.device.
    if self._open_rl_transfer_engine is None:
      config = self.vllm_config.weight_transfer_config
      if config is None:
        raise RuntimeError("Weight transfer not configured. Please set weight_transfer_config to enable weight transfer.")
      model = self.model_runner.get_model()
      # A full reload relies on the platform marking its transposed layers; no marks means the flag was renamed.
      if not flipped_modules(model):
        raise RuntimeError(f"The TPU platform marked no layers with {FLIPPED_ATTR}, so a full reload would serve untransposed weights")
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
