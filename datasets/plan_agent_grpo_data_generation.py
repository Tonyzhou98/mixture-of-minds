#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate GRPO data.
Selects the best plan for each ID where average correctness > 0.
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import os
import re
import sys
import datasets
import random
import numpy as np

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)
from agents import cot_prompts as cp

def read_jsonl_file(directory: Path) -> list:
    data = []
    
    # Find all files ending with .jsonl
    jsonl_files = list(directory.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for file_path in jsonl_files:
        print(f"  - {file_path.name}")
    
    # Read all data
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    print(f"\nTotal records read: {len(data)}")
    
    if data:
        print(f"Keys in data[0]: {list(data[0].keys())}")
    else:
        print("No data found")
    
    return data


def analyze_plan_performance(data: list) -> list:
    """
    Analyze plan performance by grouping by id and plan_idx.
    For each (id, plan_idx), calculate average correctness across all code_idx (0-3).
    """
    # Group data by (id, plan_idx)
    plan_groups = defaultdict(list)
    
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        code_idx = item.get('code_idx')
        is_correct = item.get('is_correct')
        
        if id_val is not None and plan_idx is not None and code_idx is not None and is_correct is not None:
            plan_groups[(id_val, plan_idx)].append({
                'code_idx': code_idx,
                'is_correct': is_correct
            })
    
    # Calculate average correctness for each plan
    plan_performance = []
    
    for (id_val, plan_idx), code_results in plan_groups.items():
        if len(code_results) > 0:
            # Calculate average correctness
            total_correct = sum(result['is_correct'] for result in code_results)
            avg_correctness = total_correct / len(code_results)
            
            plan_performance.append({
                'id': id_val,
                'plan_idx': plan_idx,
                'avg_correctness': avg_correctness,
                'num_code_rollouts': len(code_results)
            })
    
    # Sort by id, then by plan_idx
    plan_performance.sort(key=lambda x: (x['id'], x['plan_idx']))
    
    return plan_performance


def select_best_plans(plan_performance: list, data: list) -> list:
    """
    Select the best plan for each ID where average correctness > 0.
    Returns list of best plans with their data.
    """
    # Group plans by ID
    id_plans = defaultdict(list)
    for plan in plan_performance:
        id_plans[plan['id']].append(plan)
    
    # Create a mapping from (id, plan_idx) to the original data
    data_map = {}
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        if id_val is not None and plan_idx is not None:
            data_map[(id_val, plan_idx)] = item
    
    best_plans = []
    skipped_ids = []
    
    for id_val, plans in id_plans.items():
        # Filter plans with avg_correctness > 0.1
        valid_plans = [p for p in plans if p['avg_correctness'] > 0.1]
        
        if valid_plans:
            # Select the plan with highest average correctness
            best_plan = max(valid_plans, key=lambda p: p['avg_correctness'])
            best_plan_data = data_map.get((id_val, best_plan['plan_idx']))
            
            if best_plan_data:
                # Check if planner_raw_response contains <plan> tags
                planner_response = best_plan_data.get('planner_raw_response', '')
                plan_match = re.search(r'<plan>(.*?)</plan>', planner_response, re.DOTALL)
                
                if plan_match:
                    best_plans.append({
                        'id': id_val,
                        'plan_idx': best_plan['plan_idx'],
                        'avg_correctness': best_plan['avg_correctness'],
                        'num_code_rollouts': best_plan['num_code_rollouts'],
                        'data': best_plan_data
                    })
                else:
                    # No <plan> tags found
                    skipped_ids.append({
                        'id': id_val,
                        'reason': 'No <plan> tags in planner_raw_response',
                        'max_correctness': best_plan['avg_correctness'],
                        'planner_response_preview': planner_response[:200] + '...' if len(planner_response) > 200 else planner_response
                    })
        else:
            # No plans with correctness > 0.1
            skipped_ids.append({
                'id': id_val,
                'reason': 'No plans with correctness > 0.1',
                'all_plans': plans,
                'max_correctness': max(p['avg_correctness'] for p in plans) if plans else 0
            })
    
    return best_plans, skipped_ids


def convert_to_verl_format(best_plans: list, data_source: str) -> list:
    """
    Convert best plans to Verl format for training.
    Each plan becomes a dictionary with:
    - data_source: source of the data
    - prompt: list with system and user messages
    - ability: task type
    - reward_model: ground truth for evaluation
    - extra_info: additional metadata
    """
    verl_data = []
    
    for i, plan in enumerate(best_plans):
        data = plan['data']
        
        instruction = data.get('instruction', '')
        planner_response = data.get('planner_raw_response', '')
        
        # Extract content between <plan> and </plan> tags
        if "<plan>" not in planner_response or "</plan>" not in planner_response:
            print(f"No <plan> tags found in planner_response for id {plan['id']}")
            continue
        
        plan_match = re.search(r'<plan>(.*?)</plan>', planner_response, re.DOTALL)
        plan_content = plan_match.group(1).strip()
        
        verl_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": cp.PROMPTS["COT_AGENT_PLANNER_SYSTEM_PROMPT"]
                },
                {
                    "role": "user",
                    "content": cp.PROMPTS["COT_AGENT_PLANNER_USER_PROMPT"].format(
                        instruction=instruction
                    )
                }
            ],
            "ability": "planning",
            "reward_model": {"style": "rule", "ground_truth": plan_content},
            "extra_info": {
                "split": "train",
                "index": plan['id'],
                "answer": plan_content,
                "question": instruction,
                "plan_idx": plan['plan_idx'],
                "avg_correctness": plan['avg_correctness'],
                "num_code_rollouts": plan['num_code_rollouts'],
                "planner_response": planner_response,
            },
        }

        if i == 0:
            print(verl_item)
        
        verl_data.append(verl_item)
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and generate GRPO data"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_rl_data/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_8b_plan_grpo_data.parquet",
        type=str,
        help="Output file path to save GRPO dataset in Verl format",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    if data:
        plan_performance = analyze_plan_performance(data)
        
        # Select best plans
        print("\n" + "="*50)
        print("SELECTING BEST PLANS FOR GRPO")
        print("="*50)
        
        best_plans, skipped_ids = select_best_plans(plan_performance, data)
        
        print(f"Best plans selected: {len(best_plans)}")
        print(f"IDs skipped (no plans with correctness > 0): {len(skipped_ids)}")
        
        if best_plans:
            # Show statistics
            correctness_values = [plan['avg_correctness'] for plan in best_plans]
            print(f"\nBest Plan Statistics:")
            print(f"  Average correctness: {sum(correctness_values)/len(correctness_values):.3f}")
            print(f"  Min correctness: {min(correctness_values):.3f}")
            print(f"  Max correctness: {max(correctness_values):.3f}")
            print(f"  Plans with correctness > 0.5: {sum(1 for c in correctness_values if c > 0.5)}")
            print(f"  Plans with correctness > 1.0: {sum(1 for c in correctness_values if c > 1.0)}")
            
            print(f"\nFirst 5 best plans:")
            for i, plan in enumerate(best_plans[:5]):
                print(f"  {i+1}. ID: {plan['id']}, Plan: {plan['plan_idx']}, "
                      f"Correctness: {plan['avg_correctness']:.3f}, "
                      f"Code rollouts: {plan['num_code_rollouts']}")
        
        # Show examples of skipped IDs
        if skipped_ids:
            print(f"\nExamples of Skipped IDs (first 3):")
            for i, skipped in enumerate(skipped_ids[:3]):
                print(f"\n  Skipped ID {i+1}: {skipped['id']}")
                print(f"    Max correctness: {skipped['max_correctness']:.3f}")
                # print(f"    Number of plans: {len(skipped['all_plans'])}")
                # print(f"    All correctness values: {[p['avg_correctness'] for p in skipped['all_plans']]}")
        
        # Convert to Verl format and save if output path provided
        if args.output and best_plans:
            print(f"\nConverting to Verl format...")
            verl_data = convert_to_verl_format(best_plans, args.directory)
            
            # Convert to HuggingFace dataset
            verl_dataset = datasets.Dataset.from_list(verl_data)
            
            # Set random seed for reproducibility
            random.seed(42)
            np.random.seed(42)
            
            # Shuffle and split the dataset into train and test (50 samples for test)
            total_size = len(verl_dataset)
            indices = np.arange(total_size)
            np.random.shuffle(indices)
            test_size = min(50, total_size)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            test_dataset = verl_dataset.select(test_indices.tolist())
            train_dataset = verl_dataset.select(train_indices.tolist())
            
            # Save to parquet
            parquet_base = args.output.replace(".parquet", "")
            train_dataset.to_parquet(parquet_base + "_train.parquet")
            test_dataset.to_parquet(parquet_base + "_test.parquet")

            # save to jsonl
            train_dataset.to_json(parquet_base + "_train.jsonl", orient="records", lines=True)
            test_dataset.to_json(parquet_base + "_test.jsonl", orient="records", lines=True)
            
            print(f"Saved {len(train_dataset)} training examples to {parquet_base}_train.parquet")
            print(f"Saved {len(test_dataset)} evaluation examples to {parquet_base}_test.parquet")
            print(f"Format: Verl compatible parquet")


if __name__ == "__main__":
    main() 