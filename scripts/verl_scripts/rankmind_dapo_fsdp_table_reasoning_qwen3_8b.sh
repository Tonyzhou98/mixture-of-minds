#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=rankmind_qwen3_8b_rl_dapo
#SBATCH --output=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_8b_rl_dapo.stdout
#SBATCH --error=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_8b_rl_dapo.stderr


set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
project_name="rankmind"
algorithm=grpo
rollout_n=8
# for mean@K computation
k_max=8
#experiment_name=${model}-${algorithm}-${data}-n${n}
experiment_name="rankmind_qwen3_8b_dapo"
ROOT_DIR=/your/path/to/your_fs/verl
CHECKPOINT_PATH=/your/path/to/your_fs/checkpoints
MODEL_PATH=/your/path/to/your_fs/Qwen/Qwen3-8B
DATA_PATH=/your/path/to/your_fs/RankMind/datasets/train_data
REWARD_PATH=/your/path/to/your_fs/RankMind/reward_functions/table_reasoning_reward.py

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=False
overlong_buffer_len=4096
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))


mkdir -p logs/${project_name}
rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files=$DATA_PATH/table_instruct_scout_failed_train.parquet \
    data.val_files=$DATA_PATH/table_instruct_scout_failed_test.parquet \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=10 2>&1 | tee logs/${project_name}/${experiment_name}.log



# Path to experiment directory
base_dir="$CHECKPOINT_PATH/${project_name}/${experiment_name}"

# Get the global_step_* directory with the largest number
latest_step_dir=$(find "$base_dir" -maxdepth 1 -type d -name "global_step_*" \
  | awk -F'_' '{ print $0, $(NF) }' \
  | sort -k2 -n \
  | tail -n 1 \
  | awk '{ print $1 }')

# If no match is found
if [[ -z "$latest_step_dir" ]]; then
  echo "No global_step_* directory found in $base_dir"
  exit 1
fi

echo "Latest step: $latest_step_dir"

# Run the merge command
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$latest_step_dir/actor" \
    --target_dir "$latest_step_dir/actor/huggingface"