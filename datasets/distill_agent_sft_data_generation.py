#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and filter data based on qsubtype and is_correct.
"""

import json
import argparse
import random
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
    """
    Read all JSONL files from the specified directory.
    """
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


def filter_data(data: list) -> tuple[list, dict]:
    """
    Filter data based on qsubtype and is_correct criteria:
    - For qsubtype in ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']: keep all data
    - For other qsubtypes: only keep data where is_correct == 1
    """
    # Define the qsubtypes that should keep all data
    keep_all_qsubtypes = {'CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection'}
    
    filtered_data = []
    filtering_stats = {
        'total_records': len(data),
        'kept_records': 0,
        'filtered_out_records': 0,
        'qsubtype_counts': defaultdict(int),
        'qsubtype_filtered_counts': defaultdict(int),
        'is_correct_counts': defaultdict(int)
    }
    
    for item in data:
        qsubtype = item.get('qsubtype', 'Unknown')
        is_correct = item.get('is_correct', None)
        
        # Count qsubtypes
        filtering_stats['qsubtype_counts'][qsubtype] += 1
        
        # Count is_correct values
        if is_correct is not None:
            filtering_stats['is_correct_counts'][str(is_correct)] += 1
        
        # Apply filtering logic
        should_keep = False
        
        if qsubtype in keep_all_qsubtypes:
            # Keep all data for these qsubtypes
            if is_correct >= 0.4 and "Error: " not in item['execution_result']:
                should_keep = True
        else:
            # For other qsubtypes, only keep if is_correct == 1
            if is_correct == 1 and "Error: " not in item['execution_result']:
                should_keep = True
            else:
                filtering_stats['qsubtype_filtered_counts'][qsubtype] += 1
        
        if should_keep:
            filtered_data.append(item)
            filtering_stats['kept_records'] += 1
        else:
            filtering_stats['filtered_out_records'] += 1
    
    return filtered_data, filtering_stats


# Function removed - not saving for now


def print_filtering_summary(filtering_stats: dict) -> None:
    """
    Print a summary of the filtering results.
    """
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    
    print(f"Total records processed: {filtering_stats['total_records']}")
    print(f"Records kept: {filtering_stats['kept_records']}")
    print(f"Records filtered out: {filtering_stats['filtered_out_records']}")
    print(f"Filtering rate: {filtering_stats['filtered_out_records'] / filtering_stats['total_records'] * 100:.2f}%")
    
    print(f"\nQSubtype distribution (before filtering):")
    for qsubtype, count in sorted(filtering_stats['qsubtype_counts'].items()):
        print(f"  {qsubtype}: {count}")
    
    print(f"\nQSubtype filtering impact:")
    for qsubtype, count in sorted(filtering_stats['qsubtype_filtered_counts'].items()):
        print(f"  {qsubtype}: {count} records filtered out")
    
    # print(f"\nIs_correct distribution (before filtering):")
    # for is_correct, count in sorted(filtering_stats['is_correct_counts'].items()):
    #     print(f"  is_correct={is_correct}: {count}")


def construct_planner_dataset(filtered_data: list) -> list:
    """
    Construct dataset for fine-tuning the PLANNER agent.
    Format: system_prompt + user_prompt -> planner_raw_response
    """
    planner_dataset = []
    
    for item in filtered_data:
        # Check if we have the required fields
        if 'instruction' in item and 'planner_raw_response' in item:
            # Filter out responses with "<think>None</think>"
            if "<think>None</think>" in item['planner_raw_response']:
                continue
                
            # Create user prompt
            user_prompt = cp.PROMPTS["COT_AGENT_PLANNER_USER_PROMPT"].format(
                instruction=item['instruction']
            )
            
            planner_sample = {
                'instruction': cp.PROMPTS["COT_AGENT_PLANNER_SYSTEM_PROMPT"],
                'input': user_prompt,
                'output': item['planner_raw_response']
            }
            planner_dataset.append(planner_sample)
    
    print(f"Planner dataset: {len(planner_dataset)} samples created")
    return planner_dataset


def construct_executor_dataset(filtered_data: list) -> list:
    """
    Construct dataset for fine-tuning the EXECUTOR agent.
    Format: system_prompt + user_prompt -> executor_raw_response
    """
    executor_dataset = []
    
    for item in filtered_data:
        # Check if we have therequired fields
        if ('instruction' in item and 'generated_plan' in item and 
            'executor_raw_response' in item):
            
            # Filter out responses with "<think>None</think>"
            if "<think>None</think>" in item['executor_raw_response']:
                continue
                
            # Create user prompt
            user_prompt = cp.PROMPTS["COT_AGENT_EXECUTOR_USER_PROMPT"].format(
                instruction=item['instruction'],
                plan=item['generated_plan']
            )
            
            executor_sample = {
                'instruction': cp.PROMPTS["COT_AGENT_EXECUTOR_SYSTEM_PROMPT"],
                'input': user_prompt,
                'output': item['executor_raw_response']
            }
            executor_dataset.append(executor_sample)
    
    print(f"Executor dataset: {len(executor_dataset)} samples created")
    return executor_dataset


def construct_answerer_dataset(filtered_data: list) -> list:
    """
    Construct dataset for fine-tuning the ANSWERER agent.
    Format: system_prompt + user_prompt -> answer_raw_response
    """
    answerer_dataset = []
    
    for item in filtered_data:
        # Check if we have the required fields
        if ('instruction' in item and 'generated_plan' in item and 
            'execution_result' in item and 'answer_raw_response' in item):
            
            # Filter out responses with "<think>None</think>"
            if "<think>None</think>" in item['answer_raw_response']:
                continue
                
            # Use the exact format from rl_data_collector.py
            instruction = f"{item.get('format_instruction', '')}\n\n{item['instruction']}"
            user_prompt = cp.PROMPTS["COT_AGENT_ANSWERER_USER_PROMPT"].format(
                instruction=instruction,
                plan=item['generated_plan'],
                code_output=item['execution_result']
            )
            
            answerer_sample = {
                'instruction': cp.PROMPTS["COT_AGENT_ANSWERER_SYSTEM_PROMPT"],
                'input': user_prompt,
                'output': item['answer_raw_response']
            }
            answerer_dataset.append(answerer_sample)
    
    print(f"Answerer dataset: {len(answerer_dataset)} samples created")
    return answerer_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and filter data based on qsubtype and is_correct"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/gpt_o4_mini_rollout/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/sft_data/gpt_o4_mini_rollout.json",
        type=str,
        help="Base directory and filename for output datasets (without extension)",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    
    # Check if directory exists
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        return
    
    # Read data
    print(f"Reading JSONL files from: {directory}")
    data = read_jsonl_file(directory)
    
    if not data:
        print("No data found. Exiting.")
        return
    
    # Filter data
    print("\nFiltering data based on qsubtype and is_correct criteria...")
    filtered_data, filtering_stats = filter_data(data)
    
    # Print summary
    print_filtering_summary(filtering_stats)
    
    # Not saving data for now - just show the filtered results
    print(f"\nFiltered data ready: {len(filtered_data)} records")
    
    # Construct knowledge distillation datasets for fine-tuning
    print("\n" + "="*60)
    print("CONSTRUCTING KNOWLEDGE DISTILLATION DATASETS")
    print("="*60)
    
    # Create datasets for each agent
    planner_dataset = construct_planner_dataset(filtered_data)
    executor_dataset = construct_executor_dataset(filtered_data)
    answerer_dataset = construct_answerer_dataset(filtered_data)
    
    print(f"\nDataset Summary:")
    print(f"  Planner Dataset: {len(planner_dataset)} samples")
    print(f"  Executor Dataset: {len(executor_dataset)} samples") 
    print(f"  Answerer Dataset: {len(answerer_dataset)} samples")
    
    # Show sample from each dataset
    if planner_dataset:
        print(f"\nPlanner Dataset Sample:")
        print(f"  Instruction (System): {planner_dataset[0]['instruction']}")
        print(f"  Input (User): {planner_dataset[0]['input']}")
        print(f"  Output: {planner_dataset[0]['output']}")
    
    if executor_dataset:
        print(f"\nExecutor Dataset Sample:")
        print(f"  Instruction (System): {executor_dataset[0]['instruction']}")
        print(f"  Input (User): {executor_dataset[0]['input']}")
        print(f"  Output: {executor_dataset[0]['output']}")
    
    if answerer_dataset:
        print(f"\nAnswerer Dataset Sample:")
        print(f"  Instruction (System): {answerer_dataset[0]['instruction']}")
        print(f"  Input (User): {answerer_dataset[0]['input']}")
        print(f"  Output: {answerer_dataset[0]['output']}")
    
    # Save datasets to JSON files
    if args.output:
        output_base = Path(args.output)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        
        # Save full datasets
        planner_output = args.output.replace('.json', '_train_plan.json')
        with open(planner_output, 'w', encoding='utf-8') as f:
            json.dump(planner_dataset, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full planner dataset to: {planner_output}")
        
        executor_output = args.output.replace('.json', '_train_code.json')
        with open(executor_output, 'w', encoding='utf-8') as f:
            json.dump(executor_dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved full executor dataset to: {executor_output}")
        
        answerer_output = args.output.replace('.json', '_train_answer.json')
        with open(answerer_output, 'w', encoding='utf-8') as f:
            json.dump(answerer_dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved full answerer dataset to: {answerer_output}")
        
        # Create and save validation datasets (50 random samples each)
        print(f"\nCreating validation datasets (50 random samples each)...")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Sample 50 from planner dataset
        if len(planner_dataset) >= 20:
            planner_test = random.sample(planner_dataset, 20)
            planner_test_output = args.output.replace('.json', '_test_plan.json')
            with open(planner_test_output, 'w', encoding='utf-8') as f:
                json.dump(planner_test, f, indent=2, ensure_ascii=False)
            print(f"Saved planner validation dataset to: {planner_test_output}")
        else:
            print(f"Warning: Planner dataset has only {len(planner_dataset)} samples, cannot create validation set")
        
        # Sample 50 from executor dataset
        if len(executor_dataset) >= 20:
            executor_test = random.sample(executor_dataset, 20)
            executor_test_output = args.output.replace('.json', '_test_code.json')
            with open(executor_test_output, 'w', encoding='utf-8') as f:
                json.dump(executor_test, f, indent=2, ensure_ascii=False)
            print(f"Saved executor validation dataset to: {executor_test_output}")
        else:
            print(f"Warning: Executor dataset has only {len(executor_dataset)} samples, cannot create validation set")
        
        # Sample 50 from answerer dataset
        if len(answerer_dataset) >= 20:
            answerer_test = random.sample(answerer_dataset, 20)
            answerer_test_output = args.output.replace('.json', '_test_answer.json')
            with open(answerer_test_output, 'w', encoding='utf-8') as f:
                json.dump(answerer_test, f, indent=2, ensure_ascii=False)
            print(f"Saved answerer validation dataset to: {answerer_test_output}")
        else:
            print(f"Warning: Answerer dataset has only {len(answerer_dataset)} samples, cannot create validation set")
        
        print(f"\nAll datasets saved successfully!")
        print(f"Full datasets:")
        print(f"  Planner: {len(planner_dataset)} samples -> {planner_output}")
        print(f"  Executor: {len(executor_dataset)} samples -> {executor_output}")
        print(f"  Answerer: {len(answerer_dataset)} samples -> {answerer_output}")
        print(f"\nValidation datasets:")
        print(f"  Planner: 20 samples -> {planner_test_output}")
        print(f"  Executor: 20 samples -> {executor_test_output}")
        print(f"  Answerer: 20 samples -> {answerer_test_output}")


if __name__ == "__main__":
    main()
