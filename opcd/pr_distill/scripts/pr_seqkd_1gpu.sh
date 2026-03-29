#!/bin/bash
# SeqKD from GLM-5: SFT student on GLM-5 responses (baseline)
# Single GPU A100 80GB
set -x

MODEL_PATH="${1:-Qwen/Qwen3-1.7B}"
DATA_DIR="${DATA_DIR:-/tmp/pr_distill_data}"
EXP_NAME="${EXP_NAME:-pr-seqkd-$(basename $MODEL_PATH)}"
MAX_LENGTH="${MAX_LENGTH:-9216}"

export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export HYDRA_FULL_ERROR=1

if [ -n "$WANDB_API_KEY" ]; then
    wandb login ${WANDB_API_KEY}
fi

python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/seqkd_train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=${MAX_LENGTH} \
    data.truncation=right \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=${MODEL_PATH} \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    model.fsdp_config.cpu_offload=True \
    optim.lr=5e-6 \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=/tmp/${EXP_NAME} \
    trainer.project_name=${WANDB_PROJECT:-pr-distill} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.save_freq=500 \
    trainer.test_freq=200 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null
