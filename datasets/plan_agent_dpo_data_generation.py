#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate DPO data.
"""

import json
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import os
import sys
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


def count_valid_dpo_ids(plan_performance: list, data: list) -> dict:
    """
    Count how many IDs have valid DPO conditions:
    - At least one plan with higher correctness (positive example)
    - At least one plan with lower correctness (negative example)
    - Simply requires min_correctness < max_correctness
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
    
    valid_ids = []
    invalid_ids = []
    
    for id_val, plans in id_plans.items():
        correctness_values = [p['avg_correctness'] for p in plans]
        max_correctness = max(correctness_values)
        min_correctness = min(correctness_values)
        
        # Check if this ID has valid DPO conditions
        has_different_correctness = min_correctness < max_correctness
        
        if has_different_correctness:
            valid_ids.append({
                'id': id_val,
                'plans': plans,
                'max_correctness': max_correctness,
                'min_correctness': min_correctness,
                'num_plans': len(plans)
            })
        else:
            # Get some example data for this invalid ID
            example_plans = []
            for plan in plans[:3]:  # Show first 3 plans
                plan_data = data_map.get((id_val, plan['plan_idx']))
                if plan_data:
                    example_plans.append({
                        'plan_idx': plan['plan_idx'],
                        'avg_correctness': plan['avg_correctness'],
                        'instruction': plan_data.get('instruction', 'N/A')[:100] + '...' if len(plan_data.get('instruction', '')) > 100 else plan_data.get('instruction', 'N/A'),
                        'generated_plan': plan_data.get('generated_plan', 'N/A')[:100] + '...' if len(plan_data.get('generated_plan', '')) > 100 else plan_data.get('generated_plan', 'N/A'),
                        'generated_answer': plan_data.get('generated_answer', 'N/A')[:100] + '...' if len(plan_data.get('generated_answer', '')) > 100 else plan_data.get('generated_answer', 'N/A'),
                        'answer': plan_data.get('answer', 'N/A')[:100] + '...' if len(plan_data.get('answer', '')) > 100 else plan_data.get('answer', 'N/A')
                    })
            
            invalid_ids.append({
                'id': id_val,
                'reason': f"max_correctness={max_correctness}, min_correctness={min_correctness} (all plans have same correctness)",
                'all_correctness': correctness_values,
                'example_plans': example_plans
            })
    
    return {
        'valid_ids': valid_ids,
        'invalid_ids': invalid_ids,
        'total_valid': len(valid_ids),
        'total_invalid': len(invalid_ids)
    }


def construct_dpo_dataset(plan_performance: list, data: list) -> list:
    """
    Construct DPO dataset with multiple pairs per ID when there are significant differences.
    Creates pairs for:
    1. Best vs Worst plans
    2. 2nd best vs 2nd worst (if gap > 0.5)
    3. Any plan with 0 correctness vs any plan with > 0 correctness
    
    Only includes pairs with gap >= 0.1
    """
    # First get valid IDs
    valid_info = count_valid_dpo_ids(plan_performance, data)
    valid_ids = valid_info['valid_ids']
    
    print(f"\nDPO Dataset Construction:")
    print(f"  Valid IDs for DPO: {valid_info['total_valid']}")
    print(f"  Invalid IDs: {valid_info['total_invalid']}")
    
    # Create a mapping from (id, plan_idx) to the original data
    data_map = {}
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        if id_val is not None and plan_idx is not None:
            data_map[(id_val, plan_idx)] = item
    
    dpo_pairs = []
    
    for valid_id_info in valid_ids:
        id_val = valid_id_info['id']
        plans = valid_id_info['plans']
        
        # Sort plans by correctness (best to worst)
        sorted_plans = sorted(plans, key=lambda p: p['avg_correctness'], reverse=True)
        
        # Strategy 1: Best vs Worst
        if len(sorted_plans) >= 2:
            best_plan = sorted_plans[0]
            worst_plan = sorted_plans[-1]
            
            gap = best_plan['avg_correctness'] - worst_plan['avg_correctness']
            if gap >= 0.1:  # Only include if gap >= 0.1
                positive_data = data_map.get((id_val, best_plan['plan_idx']))
                negative_data = data_map.get((id_val, worst_plan['plan_idx']))
                
                if positive_data and negative_data:
                    dpo_pair = {
                        'id': id_val,
                        'positive_plan_idx': best_plan['plan_idx'],
                        'negative_plan_idx': worst_plan['plan_idx'],
                        'positive_correctness': best_plan['avg_correctness'],
                        'negative_correctness': worst_plan['avg_correctness'],
                        'positive_data': positive_data,
                        'negative_data': negative_data,
                        'pair_type': 'best_vs_worst'
                    }
                    dpo_pairs.append(dpo_pair)
        
        # Strategy 2: 2nd best vs 2nd worst (if gap > 0.5)
        if len(sorted_plans) >= 4:
            second_best = sorted_plans[1]
            second_worst = sorted_plans[-2]
            
            gap = second_best['avg_correctness'] - second_worst['avg_correctness']
            if gap > 0.5:  # This already ensures gap >= 0.1
                positive_data = data_map.get((id_val, second_best['plan_idx']))
                negative_data = data_map.get((id_val, second_worst['plan_idx']))
                
                if positive_data and negative_data:
                    dpo_pair = {
                        'id': id_val,
                        'positive_plan_idx': second_best['plan_idx'],
                        'negative_plan_idx': second_worst['plan_idx'],
                        'positive_correctness': second_best['avg_correctness'],
                        'negative_correctness': second_worst['avg_correctness'],
                        'positive_data': positive_data,
                        'negative_data': negative_data,
                        'pair_type': 'second_best_vs_second_worst'
                    }
                    dpo_pairs.append(dpo_pair)
        
        # Strategy 3: Any plan with 0 correctness vs any plan with > 0 correctness
        zero_plans = [p for p in sorted_plans if p['avg_correctness'] == 0]
        non_zero_plans = [p for p in sorted_plans if p['avg_correctness'] > 0]
        
        if zero_plans and non_zero_plans:
            # Create pairs: best non-zero vs each zero plan
            best_non_zero = max(non_zero_plans, key=lambda p: p['avg_correctness'])
            
            for zero_plan in zero_plans:
                gap = best_non_zero['avg_correctness'] - zero_plan['avg_correctness']
                if gap >= 0.1:  # Only include if gap >= 0.1
                    positive_data = data_map.get((id_val, best_non_zero['plan_idx']))
                    negative_data = data_map.get((id_val, zero_plan['plan_idx']))
                    
                    if positive_data and negative_data:
                        dpo_pair = {
                            'id': id_val,
                            'positive_plan_idx': best_non_zero['plan_idx'],
                            'negative_plan_idx': zero_plan['plan_idx'],
                            'positive_correctness': best_non_zero['avg_correctness'],
                            'negative_correctness': zero_plan['avg_correctness'],
                            'positive_data': positive_data,
                            'negative_data': negative_data,
                            'pair_type': 'non_zero_vs_zero'
                        }
                        dpo_pairs.append(dpo_pair)
    
    return dpo_pairs


def convert_to_llama_factory_format(dpo_pairs: list) -> list:
    """
    Convert DPO pairs to LLaMA-Factory format.
    Each pair becomes a dictionary with:
    - conversations: list with system and human messages
    - chosen: positive example response
    - rejected: negative example response
    """
    llama_factory_data = []
    
    for pair in dpo_pairs:
        positive_data = pair['positive_data']
        negative_data = pair['negative_data']
        
        conversations = [
            {
                "from": "system",
                "value": cp.PROMPTS["COT_AGENT_PLANNER_SYSTEM_PROMPT"]
            },
            {
                "from": "human",
                "value": cp.PROMPTS["COT_AGENT_PLANNER_USER_PROMPT"].format(
                    instruction=positive_data.get('instruction', '')
                )
            }
        ]
        
        # Create chosen (positive) response with only the planner raw response
        chosen = {
            "from": "gpt",
            "value": positive_data.get('planner_raw_response', '')
        }
        
        # Create rejected (negative) response with only the planner raw response
        rejected = {
            "from": "gpt", 
            "value": negative_data.get('planner_raw_response', '')
        }
        
        llama_factory_item = {
            "conversations": conversations,
            "chosen": chosen,
            "rejected": rejected
        }
        
        llama_factory_data.append(llama_factory_item)
    
    return llama_factory_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and analyze plan performance"
    )
    # optional argument
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/code/RankMind/datasets/rl_datasets",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_32b_plan_dpo_data.json",
        type=str,
        help="Output file path to save DPO dataset in LLaMA-Factory format",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    if data:
        plan_performance = analyze_plan_performance(data)
        
        # Count valid DPO IDs
        print("\n" + "="*50)
        print("COUNTING VALID DPO IDs")
        print("="*50)
        
        valid_info = count_valid_dpo_ids(plan_performance, data)
        
        print(f"Valid IDs for DPO: {valid_info['total_valid']}")
        print(f"Invalid IDs: {valid_info['total_invalid']}")
        
        if valid_info['valid_ids']:
            print(f"\nFirst 5 valid IDs:")
            for i, valid_id in enumerate(valid_info['valid_ids'][:5]):
                print(f"  {i+1}. ID: {valid_id['id']}, "
                      f"Plans: {valid_id['num_plans']}, "
                      f"Max Correctness: {valid_id['max_correctness']:.3f}, "
                      f"Min Correctness: {valid_id['min_correctness']:.3f}")
        
        # Print examples of invalid IDs
        if valid_info['invalid_ids']:
            print(f"\nExamples of Invalid IDs (first 3):")
            for i, invalid_id in enumerate(valid_info['invalid_ids'][:3]):
                print(f"\n  Invalid ID {i+1}: {invalid_id['id']}")
                print(f"    Reason: {invalid_id['reason']}")
                print(f"    All correctness values: {invalid_id['all_correctness']}")
                print(f"    Example plans:")
                
                for j, plan in enumerate(invalid_id['example_plans']):
                    print(f"      Plan {j+1} (idx: {plan['plan_idx']}, correctness: {plan['avg_correctness']:.3f}):")
                    print(f"        Instruction: {plan['instruction']}")
                    print(f"        Generated Plan: {plan['generated_plan']}")
                    print(f"        Generated Answer: {plan['generated_answer']}")
                    print(f"        Correct Answer: {plan['answer']}")
                    print()
        
        # Construct DPO dataset
        print("\n" + "="*50)
        print("CONSTRUCTING DPO DATASET")
        print("="*50)
        
        dpo_pairs = construct_dpo_dataset(plan_performance, data)
        
        print(f"Total DPO pairs created: {len(dpo_pairs)}")
        
        if dpo_pairs:
            # Count different types of pairs
            pair_types = {}
            for pair in dpo_pairs:
                pair_type = pair['pair_type']
                pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
            
            print(f"\nDPO Pair Types:")
            for pair_type, count in pair_types.items():
                print(f"  {pair_type}: {count} pairs")
            
            print(f"\nFirst 5 DPO pairs:")
            for i, pair in enumerate(dpo_pairs[:5]):
                print(f"  {i+1}. ID: {pair['id']}, Type: {pair['pair_type']}, "
                      f"Positive Plan: {pair['positive_plan_idx']} "
                      f"(correctness: {pair['positive_correctness']:.3f}), "
                      f"Negative Plan: {pair['negative_plan_idx']} "
                      f"(correctness: {pair['negative_correctness']:.3f})")
            
            # Show some statistics about correctness gaps
            gaps = [pair['positive_correctness'] - pair['negative_correctness'] for pair in dpo_pairs]
            if gaps:
                print(f"\nCorrectness Gap Statistics:")
                print(f"  Average gap: {sum(gaps)/len(gaps):.3f}")
                print(f"  Min gap: {min(gaps):.3f}")
                print(f"  Max gap: {max(gaps):.3f}")
                print(f"  Pairs with gap > 0.5: {sum(1 for gap in gaps if gap > 0.5)}")
                print(f"  Pairs with gap > 1.0: {sum(1 for gap in gaps if gap > 1.0)}")
            
            # Convert to LLaMA-Factory format and save if output path provided
            if args.output:
                print(f"\nConverting to LLaMA-Factory format...")
                llama_factory_data = convert_to_llama_factory_format(dpo_pairs)
                
                output_path = Path(args.output)
                eval_path = Path(args.output.replace(".json", "_eval.json"))
                
                # Split dataset: 100 for eval, remaining for train
                if len(llama_factory_data) >= 100:
                    eval_data = llama_factory_data[:100]
                    train_data = llama_factory_data[100:]
                    
                    # Save training dataset
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(train_data, f, indent=2, ensure_ascii=False)
                    
                    # Save evaluation dataset
                    with open(eval_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Saved {len(train_data)} training pairs to {output_path}")
                    print(f"Saved {len(eval_data)} evaluation pairs to {eval_path}")
                    print(f"Format: LLaMA-Factory compatible JSON")
                else:
                    # If less than 100 pairs, save all as training data
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(llama_factory_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Warning: Only {len(llama_factory_data)} pairs available, saved all as training data to {output_path}")
                    print(f"Format: LLaMA-Factory compatible JSON")


if __name__ == "__main__":
    main()
