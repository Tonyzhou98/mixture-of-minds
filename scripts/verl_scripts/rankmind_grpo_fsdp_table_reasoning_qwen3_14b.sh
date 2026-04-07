#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=rankmind_qwen3_14b_rl_grpo_megatron
#SBATCH --output=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_14b_rl_grpo.stdout
#SBATCH --error=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_14b_rl_grpo.stderr


set -x

project_name="rankmind"
algorithm=grpo
rollout_n=8
k_max=8
loss_agg_mode="token-mean"
experiment_name="rankmind_qwen3_14b_grpo_megatron_qwen3_failed_data_15epochs"
ROOT_DIR=/your/path/to/your_fs/verl
CHECKPOINT_PATH=/your/path/to/your_fs/checkpoints
HF_MODEL_PATH=/your/path/to/your_fs/Qwen/Qwen3-14B
DATA_PATH=/your/path/to/your_fs/RankMind/datasets/train_data
REWARD_PATH=/your/path/to/your_fs/RankMind/reward_functions/table_reasoning_reward.py
DIST_CKPT_PATH=/your/path/to/your_fs/Qwen/Qwen3-14B-mcore

# python3 -m scripts.converter_hf_to_mcore --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
# echo "HF to mcore done"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping


mkdir -p logs/${project_name}
rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=$algorithm \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_PATH/TableInstruct_qwen3_failed_train.parquet \
    data.val_files=$DATA_PATH/TableInstruct_qwen3_failed_test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.max_critic_ckpt_to_keep=10 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=15 2>&1 | tee logs/${project_name}/${experiment_name}.log


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
    --backend megatron \
    --local_dir "$latest_step_dir/actor" \
    --target_dir "$latest_step_dir/actor/huggingface"