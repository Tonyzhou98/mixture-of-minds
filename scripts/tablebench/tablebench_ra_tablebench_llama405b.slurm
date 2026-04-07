#!/bin/bash
#
#SBATCH --chdir=/your_fs/your_username/RankAgent-Benchmark/
#SBATCH --gres=gpu:0
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=tablebench_ra_tablebench_llama405b
#SBATCH --mem 128G
#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_ra_tablebench_llama405b.stdout
#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/tablebench/tablebench_ra_tablebench_llama405b.stderr

python rankagent_tablebench.py --config-path=/your_fs/your_username/RankAgent-Benchmark/configs/3rd_party_benchmarks/tablebench --config-name=ra_tablebench_llama405b
