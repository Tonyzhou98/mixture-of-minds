#!/bin/bash
#
#SBATCH --chdir=/your/path/to/your_fs/RankMind/scripts/
#SBATCH --gres=gpu:8
#SBATCH --mem 200G
#SBATCH -c 64
#SBATCH --job-name=mcts_data_gen_qwen3_32b
#SBATCH --output=/your/path/to/your_fs/RankMind/scripts/verl_scripts/outputs/slurm/tablebench/mcts_data_gen_qwen3_32b_mcts.stdout
#SBATCH --error=/your/path/to/your_fs/RankMind/scripts/verl_scripts/outputs/slurm/tablebench/mcts_data_gen_qwen3_32b_mcts.stderr

python generate_rl_data.py --plan_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --code_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --answer_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --output_fname_prefix /your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_32b_mcts_rollout_tableinstruct/qwen3_32b_mcts_rollout_tableinstruct --batch_size 512