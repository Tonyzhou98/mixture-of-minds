#!/bin/bash
#
#SBATCH --qos=genai_interns_low
#SBATCH --chdir=/your_fs/your_username/RankAgent-Benchmark/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=tablebench_qwen2p5_3b_instruct
#SBATCH --mem 128G
#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_qwen2p5_3b_instruct.stdout
#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_qwen2p5_3b_instruct.stderr

python tablebench_eval.py --config-path=/your_fs/your_username/RankAgent-Benchmark/configs/3rd_party_benchmarks/tablebench --config-name=qwen2p5_3b_instruct
