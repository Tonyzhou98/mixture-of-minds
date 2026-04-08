# Mixture of Minds

[![Paper](https://img.shields.io/badge/Paper-arXiv:2510.20176-B31B1B.svg)](https://arxiv.org/abs/2510.20176)
**Accepted by ACL 2026 Main Conference**

Mixture of Minds is a multi-agent reinforcement learning pipeline that dynamically aligns expert language models (Planner, Coder, and Answerer) to handle complex, multi-step structural reasoning tasks like TableBench and FinQA.

Traditional reinforcement learning struggles with credit assignment across long, multi-step code-execution paths. This pipeline resolves that by iteratively freezing upstream experts, fanning out independent multi-agent rollouts, and training each agent role using Group Relative Policy Optimization (GRPO) to specialize without regressions.

## Pipeline Architecture & Example Workflows

The framework is built around three distinct training phases. Each phase relies on a sequence of 1) Data Generation (Rollouts), 2) Ground Truth Verification (Filtering), and 3) GRPO Training.

### 1. Planner Agent
**Goal:** Train a base model to emit highly accurate architectural plans.

**A. Generate Rollout Data**
Use the base models to explore multiple possible plans via MCTS branching (`--num_plans 8`).
```bash
srun --gres=gpu:8 --mem 128G -c 64 python scripts/generate_rl_data.py \
  --batch_size 256 \
  --num_plans 8 \
  --num_codes_per_plan 4 \
  --num_answers_per_code 1 \
  --plan_model_path /your/path/to/models/gemma3_12b_textonly \
  --code_model_path /your/path/to/models/gemma3_12b_textonly \
  --answer_model_path /your/path/to/models/gemma3_12b_textonly \
  --output_fname_prefix ./datasets/rl_datasets/gemma3_12b_mcts_rollout \
  --input_fname ./datasets/train_data/TableInstruct.jsonl
```

**B. Filter High-Quality Plans**
Evaluate the success of the rollouts and convert the single best plan for each task into Verl GRPO format.
```bash
python datasets/plan_agent_grpo_data_generation.py \
  --directory ./datasets/rl_datasets/gemma3_12b_mcts_rollout/ \
  --output ./datasets/train_data/gemma3_12b_plan_grpo_data.parquet
```

**C. Train Planner (GRPO)**
```bash
sbatch scripts/verl_scripts/rankmind_grpo_fsdp_plan_agent_gemma3_12b.sh
```

---

### 2. Coder Agent
**Goal:** Train a base model to execute flawless Python code using the newly minted Planner model.

**A. Generate Rollout Data**
Rely on the Planner for 1 robust plan (`--num_plans 1`), then fan out the Coder model generation (`--num_codes_per_plan 8`).
```bash
srun --gres=gpu:8 --mem 128G -c 64 python scripts/generate_rl_data.py \
  --batch_size 128 \
  --num_plans 1 \
  --num_codes_per_plan 8 \
  --num_answers_per_code 1 \
  --plan_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_plan_agent/actor/huggingface \
  --code_model_path /your/path/to/models/gemma3_12b_textonly \
  --answer_model_path /your/path/to/models/gemma3_12b_textonly \
  --output_fname_prefix ./datasets/rl_datasets/gemma3_12b_plan_update \
  --input_fname ./datasets/train_data/TableInstruct.jsonl
```

**B. Filter Bug-Free Code**
Isolate the python scripts matching the generated plan that execute without throwing exceptions.
```bash
python datasets/code_agent_grpo_data_generation.py \
  --directory ./datasets/rl_datasets/gemma3_12b_plan_update/ \
  --output ./datasets/train_data/gemma3_12b_code_grpo_data.parquet
```

**C. Train Coder (GRPO)**
```bash
sbatch scripts/verl_scripts/rankmind_grpo_fsdp_code_agent_gemma3_12b.sh
```

---

### 3. Answer Agent
**Goal:** Train a base model to interpret terminal execution traces and extract standard final answers perfectly.

**A. Generate Rollout Data**
With Planner and Coder acting optimally, fan out Answer generation (`--num_answers_per_code 8`).
```bash
srun --gres=gpu:8 --mem 128G -c 64 python scripts/generate_rl_data.py \
  --batch_size 128 \
  --num_plans 1 \
  --num_codes_per_plan 4 \
  --num_answers_per_code 8 \
  --plan_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_plan_agent/actor/huggingface \
  --code_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_code_agent/actor/huggingface \
  --answer_model_path /your/path/to/models/gemma3_12b_textonly \
  --output_fname_prefix ./datasets/rl_datasets/gemma3_12b_plan_coder_update \
  --input_fname ./datasets/train_data/TableInstruct.jsonl
```

**B. Filter Verifiable Extractions**
```bash
python datasets/answer_agent_grpo_data_generation.py \
  --directory ./datasets/rl_datasets/gemma3_12b_plan_coder_update/ \
  --output ./datasets/train_data/gemma3_12b_answer_grpo_data.parquet
```

**C. Train Answer Agent (GRPO)**
```bash
sbatch scripts/verl_scripts/rankmind_grpo_fsdp_answer_agent_gemma3_12b.sh
```

---

## Evaluation Examples

### Multi-Agent Pipeline Zero-Shot (TableBench)
Use `run_tablebench_verl_ckpt.py` to evaluate your specialized multi-agent framework fully cohesively:
```bash
srun --gres=gpu:6 --mem 128G -c 64 python scripts/run_tablebench_verl_ckpt.py \
  --jsonl_path /your/path/to/TableBench_DP.jsonl \
  --plan_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_plan_agent/actor/huggingface \
  --code_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_code_agent/actor/huggingface \
  --answer_model_path /your/path/to/checkpoints/rankmind_gemma3_12b_grpo_answer_agent/actor/huggingface \
  --output_dir ./outputs/tablebench/gemma_3_12b_multi_agent \
  --output_file TableBench_gemma_3_12b_multi_agent.jsonl \
  --n_generations 1
```

### Base Model Evaluation (FinQA)
You can optionally bypass the multi-agent framework and test a single foundational model (e.g., Qwen3-32B) directly using `--generation_framework model_only`:
```bash
srun --gres=gpu:2 --mem 128G -c 64 python scripts/run_tablebench_verl_ckpt.py \
  --plan_model_path /your/path/to/models/Qwen/Qwen3-32B \
  --code_model_path x \
  --answer_model_path x \
  --jsonl_path /your/path/to/datasets/finqa/finqa_test.jsonl \
  --output_dir ./outputs/finqa/qwen3_32b_base/ \
  --output_file finqa_qwen3_32b.jsonl \
  --n_generations 1 \
  --generation_framework model_only
```

## Repository Structure

```
mixture-of-minds/
├── datasets/            # Scripts for selecting the best multi-agent rollouts and mapping them to Parquet logic
├── reward_functions/    # The customized GRPO trace / execution reward models (BLEU, Error Trace parsing)
├── scripts/             # Inference frameworks, eval scripts (e.g., run_tablebench_verl_ckpt.py, generate_rl_data.py)
└── scripts/verl_scripts # Scalable SLURM templates for FSDP2 / Megatron distributed RL PPO runs
```

## Datasets

The following datasets are included in the `datasets/` folder of this repository:

- **TableInstruct** — training data for rollout generation: `datasets/train_data/TableInstruct.jsonl`
- **TableBench** — evaluation benchmark: `datasets/tablebench/` (multiple splits: `TableBench_DP.jsonl`, `TableBench_PoT.jsonl`, `TableBench_TCoT.jsonl`, `TableBench_SCoT.jsonl`)
- **FinQA** — financial QA evaluation benchmark: `datasets/finqa/finqa_test.jsonl`

---

## Getting Started

### Prerequisites
You will need a cluster with GPUs to scale. Most SLURM templates (`scripts/verl_scripts`) are written for large 8-GPU nodes (using FSDP or Megatron backend configurations for parallel `vLLM` actor rollouts).

### Adjusting the Configs
Before running SLURM scripts, update the standard baseline variables inside `scripts/verl_scripts/` to match your local paths:
*   `ROOT_DIR` -> Your `verl` root working dir
*   `CHECKPOINT_PATH` -> Your distributed model output target
*   `MODEL_PATH` -> HuggingFace foundation model paths
*   `DATA_PATH` -> Your customized parquet datasets path

*Ensure that cluster interface networking binds (like `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` in the Megatron configurations) match your cluster layout (defaults to `eth0`).*
