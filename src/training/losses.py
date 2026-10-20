"""Training loss functions and cross-precision off-policy RL corrections.

When the trainer and sampler run in different floating-point precisions (for
example, a ``float32`` full-fine-tuning trainer paired with a ``bfloat16`` vLLM
sampler), the sampler policy ``pi_sampler`` and trainer policy ``pi_trainer``
diverge even at identical step weights. Three independent toggles in
``loss_fn_config`` correct for this off-policy mismatch on ``importance_sampling``
and ``ppo`` losses:

1. **Top-p Replay** (``loss_fn_config["top_p"]``, plus optional ``"temperature"``):
   Re-applies top-p nucleus truncation to the trainer's vocabulary logits before
   ``log_softmax`` so ``pi_trainer`` normalizes over the truncated nucleus
   support, matching ``SamplingParams(top_p=...)`` on the sampler.
2. **Masked Importance Sampling (MIS)** (``loss_fn_config["mis_rho_min"]`` and/or
   ``loss_fn_config["mis_rho_max"]``):
   Multiplies each token's objective by a detached binary indicator
   ``1[mis_rho_min <= rho_t <= mis_rho_max]`` where ``rho_t = pi_trainer(y_t) /
   pi_sampler(y_t)``. Unlike PPO clipping, tokens whose importance ratio falls
   outside ``[mis_rho_min, mis_rho_max]`` receive zero gradient instead of being
   clamped to the boundary.
3. **Score Centering (SC)** (``loss_fn_config["score_centering"] = 1``):
   Subtracts the expected trainer score under the sampler's top-K token
   distribution, ``sum_{j=1}^K pi_sampler(v_{t,j}) * grad log pi_trainer(v_{t,j})``,
   from the sampled token's score ``grad log pi_trainer(y_t)`` using raw
   unnormalized top-K sampler probabilities ``exp(topk_logprobs)``. Requires
   ``topk_token_ids`` and ``topk_logprobs`` (from ``sample(..., topk_sample_logprobs=K)``)
   in ``datum.loss_fn_inputs``.
"""

from dataclasses import dataclass
from typing import Any

import torch

MASK_LOGPROB_THRESHOLD = -9999.0


@dataclass(frozen=True)
class OffPolicyConfig:
  """Parsed off-policy RL correction toggles from ``loss_fn_config``."""

  top_p: float = 1.0
  temperature: float = 1.0
  mis_rho_min: float | None = None
  mis_rho_max: float | None = None
  score_centering: bool = False

  @property
  def uses_logit_replay(self) -> bool:
    return self.top_p < 1.0 or self.temperature != 1.0

  @property
  def uses_mis(self) -> bool:
    return self.mis_rho_min is not None or self.mis_rho_max is not None


def _parse_bool_toggle(value: Any, name: str) -> bool:
  if value in (None, False, 0, 0.0):
    return False
  if value in (True, 1, 1.0):
    return True
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in ("", "0", "false"):
      return False
    if lowered in ("1", "true"):
      return True
  raise ValueError(f"loss_fn_config[{name!r}] must be a boolean (0/1 or 'true'/'false'), got {value!r}")


def parse_off_policy_config(loss_config: dict[str, Any] | None) -> OffPolicyConfig:
  """Validate and extract the three independent off-policy correction toggles."""
  if not loss_config:
    return OffPolicyConfig()

  top_p_raw = loss_config.get("top_p")
  top_p = 1.0 if top_p_raw is None else float(top_p_raw)
  if not (0.0 < top_p <= 1.0):
    raise ValueError(f"loss_fn_config['top_p'] must be in (0.0, 1.0], got {top_p}")

  temp_raw = loss_config.get("temperature")
  temperature = 1.0 if temp_raw is None else float(temp_raw)
  if temperature <= 0.0:
    raise ValueError(f"loss_fn_config['temperature'] must be > 0, got {temperature}")

  rho_min_raw = loss_config.get("mis_rho_min")
  mis_rho_min = None if rho_min_raw is None else float(rho_min_raw)
  if mis_rho_min is not None and mis_rho_min < 0.0:
    raise ValueError(f"loss_fn_config['mis_rho_min'] must be >= 0, got {mis_rho_min}")

  rho_max_raw = loss_config.get("mis_rho_max")
  mis_rho_max = None if rho_max_raw is None else float(rho_max_raw)
  if mis_rho_max is not None and mis_rho_max <= 0.0:
    raise ValueError(f"loss_fn_config['mis_rho_max'] must be > 0, got {mis_rho_max}")

  if mis_rho_min is not None and mis_rho_max is not None and mis_rho_min > mis_rho_max:
    raise ValueError(f"loss_fn_config['mis_rho_min'] ({mis_rho_min}) must be <= loss_fn_config['mis_rho_max'] ({mis_rho_max})")

  score_centering = _parse_bool_toggle(loss_config.get("score_centering"), "score_centering")

  return OffPolicyConfig(
    top_p=top_p,
    temperature=temperature,
    mis_rho_min=mis_rho_min,
    mis_rho_max=mis_rho_max,
    score_centering=score_centering,
  )


def apply_top_p(logits: torch.Tensor, top_p: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
  """Toggle 1 (Top-p Replay): scale logits by temperature and mask tokens outside the top-p nucleus to -inf.

  Uses the same ascending-sort cumulative probability threshold (``probs_sum <= 1 - top_p``)
  as vLLM's ``apply_top_k_top_p_pytorch`` while keeping at least the top-1 token at every position.
  """
  if temperature <= 0.0:
    raise ValueError(f"temperature must be > 0, got {temperature}")
  if not (0.0 < top_p <= 1.0):
    raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")
  if temperature != 1.0:
    logits = logits / temperature
  if top_p >= 1.0:
    return logits

  logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
  probs_sort = logits_sort.softmax(dim=-1)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  top_p_mask = probs_sum <= (1.0 - top_p)
  top_p_mask[..., -1] = False
  logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))
  return torch.empty_like(logits).scatter(dim=-1, index=logits_idx, src=logits_sort)


def compute_mis_mask(
  ratio: torch.Tensor,
  rho_min: float | None = None,
  rho_max: float | None = None,
) -> torch.Tensor:
  """Toggle 2 (Masked Importance Sampling): detached binary mask ``1[rho_min <= ratio <= rho_max]``."""
  mask = torch.ones_like(ratio)
  detached = ratio.detach()
  if rho_min is not None:
    mask = mask * (detached >= rho_min).to(mask.dtype)
  if rho_max is not None:
    mask = mask * (detached <= rho_max).to(mask.dtype)
  return mask


def compute_score_center(
  trainer_topk_logprobs: torch.Tensor,
  sampler_topk_logprobs: torch.Tensor,
) -> torch.Tensor:
  """Toggle 3 (Score Centering): ``S_t = sum_{j=1}^K stop_grad(exp(sampler_topk_lp_{t,j})) * trainer_topk_lp_{t,j}``.

  Uses raw unnormalized sampler probabilities ``exp(sampler_topk_logprobs)`` over the
  top-K tokens at each position, masking out sentinel (``-99999.0``) or non-finite entries.
  """
  valid = (sampler_topk_logprobs > MASK_LOGPROB_THRESHOLD) & torch.isfinite(trainer_topk_logprobs)
  safe_trainer_lp = torch.where(valid, trainer_topk_logprobs, torch.zeros_like(trainer_topk_logprobs))
  sampler_probs = torch.where(valid, torch.exp(sampler_topk_logprobs), torch.zeros_like(sampler_topk_logprobs))
  return (sampler_probs.detach() * safe_trainer_lp).sum(dim=-1)


def apply_score_centering(
  ratio: torch.Tensor,
  trainer_topk_logprobs: torch.Tensor,
  sampler_topk_logprobs: torch.Tensor,
) -> torch.Tensor:
  """Return a surrogate ratio with identical forward value ``ratio`` and score-centered gradient.

  ``grad (ratio - stop_grad(ratio) * (S_t - stop_grad(S_t))) = ratio * (grad log pi_trainer(y_t) - grad S_t)``.
  """
  score_center = compute_score_center(trainer_topk_logprobs, sampler_topk_logprobs)
  return ratio - ratio.detach() * (score_center - score_center.detach())


def cross_entropy_loss(target_logprobs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return -target_logprobs * weights


def importance_sampling_loss(
  target_logprobs: torch.Tensor,
  weights: torch.Tensor,
  old_logprobs: torch.Tensor,
  advantages: torch.Tensor,
  loss_config: dict[str, Any] | None = None,
  trainer_topk_logprobs: torch.Tensor | None = None,
  sampler_topk_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
  cfg = parse_off_policy_config(loss_config)
  ratio = policy_ratio(target_logprobs, old_logprobs)
  effective_ratio = ratio
  if cfg.score_centering:
    if trainer_topk_logprobs is None or sampler_topk_logprobs is None:
      raise ValueError("loss_fn_config['score_centering'] requires topk_token_ids and topk_logprobs")
    effective_ratio = apply_score_centering(ratio, trainer_topk_logprobs, sampler_topk_logprobs)
  if cfg.uses_mis:
    mis_mask = compute_mis_mask(ratio, cfg.mis_rho_min, cfg.mis_rho_max)
    effective_ratio = effective_ratio * mis_mask
  elementwise_loss = -(effective_ratio * advantages) * weights
  return torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)


def ppo_loss(
  target_logprobs: torch.Tensor,
  weights: torch.Tensor,
  old_logprobs: torch.Tensor,
  advantages: torch.Tensor,
  loss_config: dict[str, Any] | None,
  trainer_topk_logprobs: torch.Tensor | None = None,
  sampler_topk_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
  cfg = parse_off_policy_config(loss_config)
  diff = torch.clamp(target_logprobs - old_logprobs, min=-20.0, max=20.0)
  ratio = torch.exp(diff)
  effective_ratio = ratio
  if cfg.score_centering:
    if trainer_topk_logprobs is None or sampler_topk_logprobs is None:
      raise ValueError("loss_fn_config['score_centering'] requires topk_token_ids and topk_logprobs")
    effective_ratio = apply_score_centering(ratio, trainer_topk_logprobs, sampler_topk_logprobs)
  epsilon = loss_config.get("clip_range", 0.2) if loss_config else 0.2
  surr1 = effective_ratio * advantages
  surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
  elementwise_objective = torch.min(surr1, surr2)
  kl_coeff = loss_config.get("kl_coeff", 0.0) if loss_config else 0.0
  if kl_coeff > 0:
    kl = (ratio - 1) - diff
    elementwise_objective = elementwise_objective - kl_coeff * kl
  if cfg.uses_mis:
    mis_mask = compute_mis_mask(ratio, cfg.mis_rho_min, cfg.mis_rho_max)
    elementwise_objective = elementwise_objective * mis_mask
  return -(elementwise_objective * weights)


def policy_ratio(target_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
  return torch.exp(torch.clamp(target_logprobs - ref_logprobs, min=-20.0, max=20.0))
