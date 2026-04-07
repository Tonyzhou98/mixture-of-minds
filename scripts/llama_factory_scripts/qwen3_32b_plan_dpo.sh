#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/LLaMA-Factory/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=rankmind_qwen3_32b_plan_dpo
#SBATCH --output=/your/path/to/your_fs/LLaMA-Factory/slurm/rankmind_qwen3_32b_plan_dpo.stdout
#SBATCH --error=/your/path/to/your_fs/LLaMA-Factory/slurm/rankmind_qwen3_32b_plan_dpo.stderr

llamafactory-cli train /your/path/to/your_fs/RankMind/scripts/llama_factory_scripts/qwen3_32b_plan_dpo.yaml