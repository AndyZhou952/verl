from typing import Any, Optional

import torch

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.diffusion import diffusion_algos
from verl.trainer.ppo import core_algos

FLOW_GRPO_ADV_ESTIMATOR = diffusion_algos.FLOW_GRPO_ADV_ESTIMATOR


def compute_response_mask(data: DataProto):
    """Compute the valid-step mask for diffusion latents.

    For diffusion models, every denoising timestep is a valid optimization step,
    so the mask is all-ones with shape [batch, num_timesteps].
    """
    all_latents = data.batch["all_latents"]
    b, t, _, _ = all_latents.shape
    return torch.ones((b, t), dtype=torch.int32)


def _build_diffusion_advantage_kwargs(
    data: DataProto,
    config: Optional[AlgoConfig] = None,
) -> dict[str, Any]:
    """Build diffusion-facing advantage kwargs from diffusion batch fields."""
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    adv_kwargs = {
        "sample_level_rewards": data.batch["sample_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "config": config,
    }
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    if "reward_baselines" in data.batch:
        adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
    return adv_kwargs


def _build_registry_advantage_kwargs(diffusion_adv_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate diffusion kwargs to the shared registry contract."""
    registry_kwargs = dict(diffusion_adv_kwargs)
    registry_kwargs["token_level_rewards"] = registry_kwargs.pop("sample_level_rewards")
    return registry_kwargs


def compute_advantage(
    data: DataProto,
    adv_estimator: str,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute diffusion advantages using the shared estimator registry."""
    diffusion_adv_kwargs = _build_diffusion_advantage_kwargs(data, config=config)
    if adv_estimator == FLOW_GRPO_ADV_ESTIMATOR:
        advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
            **diffusion_adv_kwargs,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            global_std=global_std,
        )
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        advantages, returns = adv_estimator_fn(**_build_registry_advantage_kwargs(diffusion_adv_kwargs))

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data
