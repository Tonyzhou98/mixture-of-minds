#!/bin/bash
set -x -e

# Detect the correct network interface for NCCL
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# export CUDA_LAUNCH_BLOCKING=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"
export NNODES=2
export num_gpus=8
export WANDB_DISABLED=true
export full_batch_size=128
export batch_size=1
export gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus*$NNODES)]
export CPUS_PER_TASK=32

# Use a fixed port for better stability
export MASTER_PORT=29500

## slurm
export JOB_NAME=rankmind_llama3_70b_code_gpt_o4_sft
export output_dir=/checkpoints/your_username/rankmind/llama3_70b_code_gpt_o4_sft
export train_dataset=code_gpt_o4_mini_rollout_train
export eval_dataset=code_gpt_o4_mini_rollout_test
export model_name_or_path=/your/path/to/your_fs/Llama/Llama-3.3-70B-Instruct/

cd /your/path/to/your_fs/LLaMA-Factory

srun --job-name=${JOB_NAME} \
    --gres=gpu:${num_gpus} \
    --nodes=${NNODES} \
    --ntasks-per-node=1 \
    --mem 500G \
    --cpus-per-task=${CPUS_PER_TASK} \
    bash -c 'torchrun \
    --nnodes $NNODES \
    --nproc_per_node ${num_gpus:-1} \
    --node_rank="${SLURM_NODEID}" \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port=$MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_endpoint $(scontrol show hostname $SLURM_NODELIST | head -n1):$MASTER_PORT \
    /your/path/to/your_fs/LLaMA-Factory/src/train.py \
    --deepspeed /your/path/to/your_fs/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --model_name_or_path $model_name_or_path \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset $train_dataset \
    --template llama3 \
    --cutoff_len 8192 \
    --overwrite_cache true \
    --preprocessing_num_workers 64 \
    --output_dir $output_dir \
    --num_train_epochs 10 \
    --logging_steps 1 \
    --save_steps 20 \
    --save_only_model false \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --eval_dataset $eval_dataset \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --flash_attn fa2 \
    --report_to none'