#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate GRPO data for code agent.
Selects the correct code for each ID where execution_result has no errors and is_correct > 0.7.
If multiple codes satisfy the criteria for one ID, keeps the one with highest is_correct.
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
from evaluation.TableBench.eval.table_bench_custom_eval import *
from evaluation.TableBench.metrics.custom_em_metric import *


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


def analyze_code_performance(data: list) -> list:
    """
    Analyze code performance by grouping by id, plan_idx, and code_idx.
    For each (id, plan_idx, code_idx), check execution_result and is_correct.
    """
    # Group data by (id, plan_idx, code_idx)
    code_groups = defaultdict(list)
    
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        code_idx = item.get('code_idx')
        is_correct = item.get('is_correct')
        execution_result = item.get('execution_result')
        generated_code = item.get('generated_code')
        
        if (id_val is not None and plan_idx is not None and 
            code_idx is not None and is_correct is not None and 
            execution_result is not None and generated_code is not None):
            code_groups[(id_val, plan_idx, code_idx)].append({
                'is_correct': is_correct,
                'execution_result': execution_result,
                'generated_code': generated_code
            })
    
    # Analyze each code generation
    code_performance = []
    
    for (id_val, plan_idx, code_idx), code_results in code_groups.items():
        if len(code_results) > 0:
            # Get the first result (should be only one per code generation)
            result = code_results[0]
            
            # Check if execution_result has no errors
            execution_result = result['execution_result']
            has_no_errors = not any(error_indicator in execution_result.lower() 
                                  for error_indicator in ['error', 'exception', 'traceback', 'failed'])
            
            code_performance.append({
                'id': id_val,
                'plan_idx': plan_idx,
                'code_idx': code_idx,
                'is_correct': result['is_correct'],
                'has_no_errors': has_no_errors,
                'execution_result': execution_result,
                'generated_code': result['generated_code']
            })
    
    # Sort by id, then by plan_idx, then by code_idx
    code_performance.sort(key=lambda x: (x['id'], x['plan_idx'], x['code_idx']))
    
    return code_performance


def select_correct_codes(code_performance: list, data: list) -> list:
    """
    Select correct codes for each ID where:
    1. execution_result has no errors
    2. is_correct > 0.7 (except for CausalAnalysis, DescriptiveAnalysis, AnomalyDetection)
    For CausalAnalysis, DescriptiveAnalysis, AnomalyDetection: select max is_correct with no errors
    If multiple codes satisfy criteria for one ID, keep the one with highest is_correct.
    Returns list of correct codes with their data.
    """
    # Group codes by ID
    id_codes = defaultdict(list)
    for code in code_performance:
        id_codes[code['id']].append(code)
    
    # Create a mapping from (id, plan_idx, code_idx) to the original data
    data_map = {}
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        code_idx = item.get('code_idx')
        if id_val is not None and plan_idx is not None and code_idx is not None:
            data_map[(id_val, plan_idx, code_idx)] = item
    
    correct_codes = []
    skipped_ids = []
    
    for id_val, codes in id_codes.items():
        # Get the qsubtype for this ID from the first code's data
        qsubtype = None
        if codes and len(codes) > 0:
            first_code_data = data_map.get((id_val, codes[0]['plan_idx'], codes[0]['code_idx']))
            if first_code_data:
                qsubtype = first_code_data.get('qsubtype')
        
        # Check if this is an exception qsubtype
        exception_qsubtypes = ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']
        is_exception = qsubtype in exception_qsubtypes
        
        if is_exception:
            # For exception qsubtypes: select max is_correct with no errors
            valid_codes = [c for c in codes if c['has_no_errors'] and c['is_correct'] >= 0.3]
        else:
            # For regular qsubtypes: filter codes that meet our criteria
            valid_codes = [c for c in codes if c['has_no_errors'] and c['is_correct'] >= 0.5]
        
        if valid_codes:
            # Select the code with highest correctness
            best_code = max(valid_codes, key=lambda c: c['is_correct'])
            best_code_data = data_map.get((id_val, best_code['plan_idx'], best_code['code_idx']))
            
            if best_code_data:
                correct_codes.append({
                    'id': id_val,
                    'plan_idx': best_code['plan_idx'],
                    'code_idx': best_code['code_idx'],
                    'is_correct': best_code['is_correct'],
                    'has_no_errors': best_code['has_no_errors'],
                    'qsubtype': qsubtype,
                    'data': best_code_data
                })
            else:
                skipped_ids.append({
                    'id': id_val,
                    'reason': 'Data not found in original dataset',
                    'best_code': best_code
                })
        else:
            # No codes meet the criteria
            max_correctness = max(c['is_correct'] for c in codes) if codes else 0
            has_any_no_errors = any(c['has_no_errors'] for c in codes) if codes else False
            
            if is_exception:
                reason = f'No codes with no errors for exception qsubtype: {qsubtype}'
            else:
                reason = 'No codes with no errors and correctness > 0.7'
            
            skipped_ids.append({
                'id': id_val,
                'reason': reason,
                'max_correctness': max_correctness,
                'has_any_no_errors': has_any_no_errors,
                'qsubtype': qsubtype,
                'all_codes': codes
            })
    
    return correct_codes, skipped_ids


def convert_to_verl_format(correct_codes: list, data_source: str) -> list:
    """
    Convert correct codes to Verl format for training.
    Each code becomes a dictionary with:
    - data_source: source of the data
    - prompt: list with system and user messages
    - ability: task type
    - reward_model: ground truth for evaluation
    - extra_info: additional metadata
    """
    verl_data = []
    
    for i, code in enumerate(correct_codes):
        data = code['data']
        
        instruction = data.get('instruction', '')
        generated_plan = data.get('generated_plan', '')
        generated_code = data.get('generated_code', '')
        
        verl_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": cp.PROMPTS["COT_AGENT_EXECUTOR_SYSTEM_PROMPT"]
                },
                {
                    "role": "user",
                    "content": cp.PROMPTS["COT_AGENT_EXECUTOR_USER_PROMPT"].format(
                        instruction=instruction,
                        plan=generated_plan
                    )
                }
            ],
            "ability": "coding",
            "reward_model": {"style": "rule", "ground_truth": generated_code},
            "extra_info": {
                "split": "train",
                "index": code['id'],
                "answer": generated_code,
                "gt_answer": data.get('answer', ''),
                "question": instruction,
                "plan_idx": code['plan_idx'],
                "code_idx": code['code_idx'],
                "is_correct": code['is_correct'],
                "has_no_errors": code['has_no_errors'],
                "execution_result": data.get('execution_result', ''),
                "generated_plan": generated_plan,
            },
        }

        if i == 0:
            print(verl_item)
        
        verl_data.append(verl_item)
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and generate GRPO data for code agent"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_rl_data/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_8b_code_grpo_data.parquet",
        type=str,
        help="Output file path to save GRPO dataset in Verl format",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    if data:
        code_performance = analyze_code_performance(data)
        
        # Select correct codes
        print("\n" + "="*50)
        print("SELECTING CORRECT CODES FOR GRPO")
        print("="*50)
        
        correct_codes, skipped_ids = select_correct_codes(code_performance, data)
        
        print(f"Correct codes selected: {len(correct_codes)}")
        print(f"IDs skipped: {len(skipped_ids)}")
        
        if correct_codes:
            # Show statistics
            correctness_values = [code['is_correct'] for code in correct_codes]
            print(f"\nCorrect Code Statistics:")
            print(f"  Average correctness: {sum(correctness_values)/len(correctness_values):.3f}")
            print(f"  Min correctness: {min(correctness_values):.3f}")
            print(f"  Max correctness: {max(correctness_values):.3f}")
            print(f"  Codes with correctness > 0.8: {sum(1 for c in correctness_values if c > 0.8)}")
            print(f"  Codes with correctness > 0.9: {sum(1 for c in correctness_values if c > 0.9)}")
            
            print(f"\nFirst 5 correct codes:")
            for i, code in enumerate(correct_codes[:5]):
                print(f"  {i+1}. ID: {code['id']}, Plan: {code['plan_idx']}, Code: {code['code_idx']}, "
                      f"Correctness: {code['is_correct']:.3f}, No Errors: {code['has_no_errors']}")
        
        # Show examples of skipped IDs
        if skipped_ids:
            print(f"\nExamples of Skipped IDs (first 3):")
            for i, skipped in enumerate(skipped_ids[:3]):
                print(f"\n  Skipped ID {i+1}: {skipped['id']}")
                print(f"    Reason: {skipped['reason']}")
                if 'max_correctness' in skipped:
                    print(f"    Max correctness: {skipped['max_correctness']:.3f}")
                if 'has_any_no_errors' in skipped:
                    print(f"    Has any no errors: {skipped['has_any_no_errors']}")
        
        # Convert to Verl format and save if output path provided
        if args.output and correct_codes:
            print(f"\nConverting to Verl format...")
            verl_data = convert_to_verl_format(correct_codes, args.directory)
            
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
            
            print(f"Saved {len(train_dataset)} training examples to {parquet_base}_train.parquet")
            print(f"Saved {len(test_dataset)} evaluation examples to {parquet_base}_test.parquet")
            print(f"Format: Verl compatible parquet")


if __name__ == "__main__":
    main() 
