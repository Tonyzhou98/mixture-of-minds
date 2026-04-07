#!/bin/bash
#
#SBATCH --chdir=/your/path/to/your_fs/RankMind/scripts/
#SBATCH --gres=gpu:3
#SBATCH --mem 200G
#SBATCH -c 64
#SBATCH --job-name=mcts_data_gen_qwen3_8b_code_update
#SBATCH --output=/your/path/to/your_fs/RankMind/scripts/verl_scripts/outputs/slurm/tablebench/mcts_data_gen_qwen3_8b_code_update.stdout
#SBATCH --error=/your/path/to/your_fs/RankMind/scripts/verl_scripts/outputs/slurm/tablebench/mcts_data_gen_qwen3_8b_code_update.stderr

# python generate_rl_data.py --plan_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --code_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --answer_model_path /your/path/to/your_fs/Qwen/Qwen3-32B --output_fname_prefix /your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_32b_mcts_rollout_tableinstruct/qwen3_32b_mcts_rollout_tableinstruct
python generate_rl_data.py  --batch_size 256 --num_plans 1 --num_codes_per_plan 4 --num_answers_per_code 8 --plan_model_path /your/path/to/your_fs/checkpoints/rankmind/rankmind_qwen3_8b_grpo_plan_agent/global_step_100/actor/huggingface  --code_model_path /your/path/to/your_fs/checkpoints/rankmind/rankmind_qwen3_8b_grpo_code_agent/global_step_60/actor/huggingface --answer_model_path /your/path/to/your_fs/Qwen/Qwen3-8B --output_fname_prefix /your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_code_update_mcts_rollout_tableinstruct/qwen3_8b_code_update_mcts_rollout_tableinstruct
