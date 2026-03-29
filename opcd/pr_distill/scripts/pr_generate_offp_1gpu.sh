#!/bin/bash
# PR-CD Step 1: Generate off-policy teacher data using hinted prompts
# Teacher model sees question + GLM-5 hint, generates responses + logprobs
# Single GPU A100 80GB
set -x

MODEL_PATH="${1:-Qwen/Qwen3-1.7B}"
DATA_DIR="${DATA_DIR:-/tmp/pr_distill_data}"
EXP_NAME="${EXP_NAME:-pr-offp-gen-$(basename $MODEL_PATH)}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 4096))  # larger for hinted prompts
PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1

if [ -n "$WANDB_API_KEY" ]; then
    wandb login ${WANDB_API_KEY}
fi

python3 -m verl.trainer.main_ppo \
    data.prompt_key=content \
    data.train_files=${DATA_DIR}/teacher_train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=full \
    actor_rollout_ref.actor.kl_topk=256 \
    actor_rollout_ref.actor.kl_renorm_topk=False \
    actor_rollout_ref.actor.profile_kl=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.stage=consolidate \
    trainer.experience_path="" \
    trainer.val_before_train=False \
    trainer.on_policy_merge=False \
    trainer.generate_off_policy=True \
    trainer.off_policy_save_dir=/tmp/${EXP_NAME}/off_policy_data \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT:-pr-distill} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=10000000000 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_sleep_hack=True \
    trainer.default_local_dir=/tmp/${EXP_NAME}
