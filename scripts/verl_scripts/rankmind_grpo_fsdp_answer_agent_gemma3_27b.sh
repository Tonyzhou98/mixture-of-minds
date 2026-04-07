#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/code/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=rankmind_gemma3_27b_rl_grpo_answer_agent
#SBATCH --output=/your/path/to/your_fs/code/verl/slurm/rankmind_gemma3_27b_rl_grpo_answer_agent.stdout
#SBATCH --error=/your/path/to/your_fs/code/verl/slurm/rankmind_gemma3_27b_rl_grpo_answer_agent.stderr


set -x

project_name="rankmind"
algorithm=grpo
rollout_n=8
k_max=8
loss_agg_mode="token-mean"
experiment_name="rankmind_gemma3_27b_grpo_answer_agent"
ROOT_DIR=/your/path/to/your_fs/code/verl
CHECKPOINT_PATH=/checkpoints/your_username
MODEL_PATH=/your/path/to/your_fs/models/gemma3_27b_textonly
DATA_PATH=/your/path/to/your_fs/code/RankMind/datasets/train_data
REWARD_PATH=/your/path/to/your_fs/code/RankMind/reward_functions/table_reasoning_reward.py


mkdir -p logs/${project_name}
rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_PATH/gemma3_27b_textonly_answer_grpo_data_train.parquet \
    data.val_files=$DATA_PATH/gemma3_27b_textonly_answer_grpo_data_test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score_batch_multiturn_nonreasoning \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=20 2>&1 | tee logs/${project_name}/${experiment_name}.log


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