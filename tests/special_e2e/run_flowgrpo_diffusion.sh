#!/usr/bin/env bash
# End-to-end test for RayFlowGRPOTrainer (diffusion / FlowGRPO pipeline).
#
# Exercises the full training loop: dataset loading -> vllm_omni agent-loop
# rollout -> VisualRewardManager (jpeg_compressibility) -> flow_grpo advantage
# -> FSDP LoRA actor update -> weight sync.
#
# Requirements:
#   - vllm-omni installed
#   - diffusers >= 0.37.0
#   - Tiny Qwen-Image model at ~/models/tiny-random/Qwen-Image
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Configurable env vars (CI can override)
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

# Tokenizer encodes 34 chat-template prefix tokens before the actual prompt.
TOKENIZER_MAX_LEN=1024
PROMPT_TEMPLATE_OFFSET=34
MAX_PROMPT_LENGTH=$((TOKENIZER_MAX_LEN + PROMPT_TEMPLATE_OFFSET))

# ---------------------------------------------------------------------------
# Generate synthetic test data (idempotent; skipped if files already exist)
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size 32 \
        --val_size 8
fi

# ---------------------------------------------------------------------------
# Batch-size arithmetic (keep small for CI)
# ---------------------------------------------------------------------------
n_resp_per_prompt=2
micro_bsz_per_gpu=2
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

# ---------------------------------------------------------------------------
# Run the dedicated FlowGRPO trainer
# ---------------------------------------------------------------------------
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_flowgrpo \
    --config-name=diffusion_trainer \
    algorithm.adv_estimator=flow_grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.max_length=${MAX_PROMPT_LENGTH} \
    +data.apply_chat_template_kwargs.padding=True \
    +data.apply_chat_template_kwargs.truncation=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
    actor_rollout_ref.model.external_lib=examples.flowgrpo_trainer.diffusers.qwen_image \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.extra_configs.true_cfg_scale=4.0 \
    +actor_rollout_ref.model.extra_configs.max_sequence_length=${MAX_PROMPT_LENGTH} \
    +actor_rollout_ref.model.extra_configs.noise_level=1.0 \
    +actor_rollout_ref.model.extra_configs.sde_window_size=2 \
    "+actor_rollout_ref.model.extra_configs.sde_window_range=[0,5]" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.num_inference_steps=4 \
    actor_rollout_ref.rollout.height=256 \
    actor_rollout_ref.rollout.width=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.max_model_len=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.val_kwargs.num_inference_steps=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=examples.flowgrpo_trainer.vllm_omni.pipeline_qwenimage.QwenImagePipelineWithLogProb \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.reward_manager.name=visual \
    reward.reward_model.enable=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-diffusion-e2e \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS}

echo "FlowGRPO diffusion e2e test passed (training completed successfully)."
