# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch

from verl import DataProto
from verl.trainer.config import DiffusionAlgoConfig
from verl.trainer.diffusion import diffusion_algos

DIFFUSION_ADV_ESTIMATOR_REGISTRY = diffusion_algos.DIFFUSION_ADV_ESTIMATOR_REGISTRY
FLOW_GRPO_ADV_ESTIMATOR = diffusion_algos.FLOW_GRPO_ADV_ESTIMATOR
register_diffusion_adv_est = diffusion_algos.register_diffusion_adv_est
get_diffusion_adv_estimator_fn = diffusion_algos.get_diffusion_adv_estimator_fn


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
    config: Optional[DiffusionAlgoConfig] = None,
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


def compute_advantage(
    data: DataProto,
    adv_estimator: str,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DiffusionAlgoConfig] = None,
) -> DataProto:
    """Compute diffusion advantages using the diffusion-local registry."""
    diffusion_adv_kwargs = _build_diffusion_advantage_kwargs(data, config=config)
    adv_estimator_fn = get_diffusion_adv_estimator_fn(adv_estimator)
    adv_kwargs = dict(diffusion_adv_kwargs)
    if adv_estimator == FLOW_GRPO_ADV_ESTIMATOR:
        adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
        adv_kwargs["global_std"] = global_std
    advantages, returns = adv_estimator_fn(**adv_kwargs)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data
