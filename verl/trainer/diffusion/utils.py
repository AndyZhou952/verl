def need_diffusion_reference_policy(config) -> bool:
    """Return whether diffusion training needs a separate reference policy."""

    return bool(config.actor_rollout_ref.actor.get("use_kl_loss", False))
