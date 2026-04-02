import os

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.diffusion import diffusion_algos
from verl.trainer.diffusion.advantage import (
    _build_diffusion_advantage_kwargs,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator


def _make_diffusion_batch(
    batch_size: int = 4,
    num_steps: int = 6,
    include_reward_baselines: bool = False,
) -> DataProto:
    tensors = {
        "all_latents": torch.randn(batch_size, num_steps, 3, 3),
        "sample_level_rewards": torch.randn(batch_size, 1),
    }
    if include_reward_baselines:
        tensors["reward_baselines"] = torch.randn(batch_size)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={"uid": np.asarray([f"uid-{idx // 2}" for idx in range(batch_size)], dtype=object)},
    )


def test_compute_response_mask_returns_valid_step_mask() -> None:
    data = _make_diffusion_batch(batch_size=3, num_steps=5)

    valid_step_mask = compute_response_mask(data)

    assert valid_step_mask.shape == (3, 5)
    assert valid_step_mask.dtype == torch.int32
    assert torch.equal(valid_step_mask, torch.ones((3, 5), dtype=torch.int32))


def test_build_diffusion_advantage_kwargs_maps_diffusion_batch_fields() -> None:
    data = _make_diffusion_batch(include_reward_baselines=True)
    data.batch["response_mask"] = compute_response_mask(data)
    config = {"name": "diffusion-config"}

    adv_kwargs = _build_diffusion_advantage_kwargs(data, config=config)

    assert torch.equal(adv_kwargs["token_level_rewards"], data.batch["sample_level_rewards"])
    assert torch.equal(adv_kwargs["response_mask"], data.batch["response_mask"])
    assert adv_kwargs["index"] is data.non_tensor_batch["uid"]
    assert torch.equal(adv_kwargs["reward_baselines"], data.batch["reward_baselines"])
    assert adv_kwargs["config"] is config


def test_compute_advantage_uses_diffusion_module_for_flow_grpo(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _make_diffusion_batch()

    def fake_flow_grpo(**kwargs):
        assert torch.equal(kwargs["token_level_rewards"], data.batch["sample_level_rewards"])
        assert kwargs["index"] is data.non_tensor_batch["uid"]
        response_mask = kwargs["response_mask"]
        advantages = torch.full(response_mask.shape, 2.0)
        returns = torch.full(response_mask.shape, 3.0)
        return advantages, returns

    monkeypatch.setattr(diffusion_algos, "compute_flow_grpo_outcome_advantage", fake_flow_grpo)

    result = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.FLOW_GRPO,
        norm_adv_by_std_in_grpo=False,
        global_std=False,
    )

    assert torch.equal(result.batch["advantages"], torch.full((4, 6), 2.0))
    assert torch.equal(result.batch["returns"], torch.full((4, 6), 3.0))


def test_compute_advantage_dispatches_generic_estimator_with_diffusion_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _make_diffusion_batch(include_reward_baselines=True)
    config = {"alpha": 0.1}

    def fake_estimator(**kwargs):
        assert torch.equal(kwargs["token_level_rewards"], data.batch["sample_level_rewards"])
        assert torch.equal(kwargs["response_mask"], compute_response_mask(data))
        assert kwargs["index"] is data.non_tensor_batch["uid"]
        assert torch.equal(kwargs["reward_baselines"], data.batch["reward_baselines"])
        assert kwargs["config"] is config
        response_mask = kwargs["response_mask"]
        return torch.ones_like(response_mask, dtype=torch.float32), torch.zeros_like(response_mask, dtype=torch.float32)

    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "diffusion_test_estimator", fake_estimator)

    result = compute_advantage(
        data,
        adv_estimator="diffusion_test_estimator",
        config=config,
    )

    assert result.batch["advantages"].shape == (4, 6)
    assert result.batch["returns"].shape == (4, 6)


def test_flow_grpo_estimator_registered_from_diffusion_module() -> None:
    assert core_algos.get_adv_estimator_fn(AdvantageEstimator.FLOW_GRPO) is diffusion_algos.compute_flow_grpo_outcome_advantage


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    batch_size = 8
    steps = 10
    token_level_rewards = torch.randn((batch_size, 1), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)


def test_compute_policy_loss_flow_grpo() -> None:
    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import FSDPActorConfig

    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "clip_ratio=0.0001",
                "clip_ratio_high=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPActorConfig = omega_conf_to_dataclass(cfg)

    for step in range(steps):
        pg_loss, pg_metrics = diffusion_algos.compute_policy_loss_flow_grpo(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            response_mask=response_mask[:, step],
            loss_agg_mode="token-mean",
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics
