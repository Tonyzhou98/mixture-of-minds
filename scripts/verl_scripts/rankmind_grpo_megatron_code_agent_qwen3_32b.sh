#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/verl/
#SBATCH --nodes 2 
#SBATCH --tasks-per-node 8 
#SBATCH --cpus-per-task 24 
#SBATCH --gpus-per-node 8
#SBATCH --mem 500G
#SBATCH --job-name=rankmind_qwen3_32b_rl_grpo_megatron_code_agent
#SBATCH --output=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_32b_rl_grpo_megatron_code_agent.stdout
#SBATCH --error=/your/path/to/your_fs/verl/slurm/rankmind_qwen3_32b_rl_grpo_megatron_code_agent.stderr

set -x

project_name="rankmind"
algorithm=grpo
rollout_n=8
k_max=8
loss_agg_mode="token-mean"
experiment_name="rankmind_qwen3_32b_grpo_megatron_code_agent"
ROOT_DIR=/your/path/to/your_fs/verl
CHECKPOINT_PATH=/your/path/to/your_fs/checkpoints
HF_MODEL_PATH=/your/path/to/your_fs/Qwen/Qwen3-32B
DATA_PATH=/your/path/to/your_fs/RankMind/datasets/train_data
REWARD_PATH=/your/path/to/your_fs/RankMind/reward_functions/table_reasoning_reward.py
DIST_CKPT_PATH=/your/path/to/your_fs/Qwen/Qwen3-32B-mcore

# python3 -m scripts.converter_hf_to_mcore --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
# echo "HF to mcore done"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping


mkdir -p logs/${project_name}
# rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 处理IPv6或多个IP的情况
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head
echo "IP Head: $ip_head"

# -----------start Ray Head ----------
echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &

sleep 20
# -----------start Ray Worker ----------
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address $ip_head \
      --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &
  sleep 5
done
sleep 30
for i in {1..20}; do
  worker_cnt=$(ray status | grep GPU | grep -o "[0-9.]\+/[0-9.]\+ GPU" | head -n 1 | cut -d/ -f2)
  if [[ "$worker_cnt" == "16.0 GPU" ]]; then
    echo "All workers connected!"
    break
  fi

  echo "current GPUs: ($worker_cnt) Waiting for workers... ($i)"
  sleep 5
done

ray status

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    export NCCL_SOCKET_IFNAME=${NETWORK_INTERFACE:-eth0}
    export GLOO_SOCKET_IFNAME=${NETWORK_INTERFACE:-eth0}
    python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=$algorithm \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_PATH/qwen3_32b_code_grpo_data_train.parquet \
    data.val_files=$DATA_PATH/qwen3_32b_code_grpo_data_test.parquet \
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
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score_batch_pandas \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=2 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=10 2>&1
  "
wait


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

torchrun --nproc_per_node 8 --nnodes 1 -m verl.model_merger merge \
    --backend megatron \
    --local_dir "$latest_step_dir/actor" \
    --target_dir "$latest_step_dir/actor/huggingface"