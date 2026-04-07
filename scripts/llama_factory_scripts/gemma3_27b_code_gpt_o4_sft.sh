#!/bin/bash

#SBATCH --chdir=/your/path/to/your_fs/LLaMA-Factory/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=rankmind_gemma3_27b_code_gpt_o4_sft
#SBATCH --output=/your/path/to/your_fs/LLaMA-Factory/slurm/rankmind_gemma3_27b_code_gpt_o4_sft.stdout
#SBATCH --error=/your/path/to/your_fs/LLaMA-Factory/slurm/rankmind_gemma3_27b_code_gpt_o4_sft.stderr

llamafactory-cli train /your/path/to/your_fs/RankMind/scripts/llama_factory_scripts/gemma3_27b_code_gpt_o4_sft.yaml