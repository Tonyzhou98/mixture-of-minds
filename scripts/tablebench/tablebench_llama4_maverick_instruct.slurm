#!/bin/bash
#
#SBATCH --chdir=/your_fs/your_username/RankAgent-Benchmark/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=tablebench_llama4_maverick_instruct
#SBATCH --mem 128G
#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_llama4_maverick_instruct.stdout
#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_llama4_maverick_instruct.stderr

python tablebench_eval.py --config-path=/your_fs/your_username/RankAgent-Benchmark/configs/3rd_party_benchmarks/tablebench --config-name=llama4_maverick_instruct
