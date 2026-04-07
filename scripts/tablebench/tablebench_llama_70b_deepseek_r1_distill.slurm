#!/bin/bash
#
#SBATCH --qos=genai_interns_low
#SBATCH --chdir=/your_fs/your_username/RankAgent-Benchmark/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=tablebench_llama_70b_deepseek_r1_distill
#SBATCH --mem 128G
#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_llama_70b_deepseek_r1_distill.stdout
#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_llama_70b_deepseek_r1_distill.stderr

python tablebench_eval.py --config-path=/your_fs/your_username/RankAgent-Benchmark/configs/3rd_party_benchmarks/tablebench --config-name=llama_70b_deepseek_r1_distill
