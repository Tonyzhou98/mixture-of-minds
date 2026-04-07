#!/bin/bash
#
#SBATCH --qos=genai_interns_low
#SBATCH --chdir=/your_fs/your_username/RankAgent-Benchmark/
#SBATCH --gres=gpu:0
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=tablebench_deepseek_v2_chat_0628
#SBATCH --mem 128G
#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_deepseek_v2_chat_0628.stdout
#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_deepseek_v2_chat_0628.stderr

python tablebench_eval.py --config-path=/your_fs/your_username/RankAgent-Benchmark/configs/3rd_party_benchmarks/tablebench --config-name=deepseek_v2_chat_0628
