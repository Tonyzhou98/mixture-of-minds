#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate GRPO data for answer agent.
Selects the correct answers for each ID where execution_result has no errors and is_correct == 1.
For each ID, we have one plan, multiple codes, and multiple answers per code.
The aim is to find proper input data for GRPO training of the answer agent.
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


def load_format_instructions(file_path: str) -> dict:
    """Load format instructions from TableInstruct_qwen3_failed.jsonl file"""
    format_instructions = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if 'id' in item and 'format_instruction' in item:
                        format_instructions[item['id']] = item['format_instruction']
        
        print(f"Loaded {len(format_instructions)} format instructions from {file_path}")
        return format_instructions
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using original format_instructions.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading format instructions from {file_path}: {e}")
        return {}


def analyze_answer_performance(data: list) -> list:
    """
    Analyze answer performance by grouping by id, plan_idx, code_idx, and answer_idx.
    For each (id, plan_idx, code_idx, answer_idx), check execution_result and is_correct.
    """
    # Group data by (id, plan_idx, code_idx, answer_idx)
    answer_groups = defaultdict(list)
    
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        code_idx = item.get('code_idx')
        answer_idx = item.get('answer_idx')
        is_correct = item.get('is_correct')
        execution_result = item.get('execution_result')
        generated_answer = item.get('generated_answer')
        generated_code = item.get('generated_code')
        generated_plan = item.get('generated_plan')
        
        if (id_val is not None and plan_idx is not None and 
            code_idx is not None and answer_idx is not None and 
            is_correct is not None and execution_result is not None and 
            generated_answer is not None and generated_code is not None and
            generated_plan is not None):
            answer_groups[(id_val, plan_idx, code_idx, answer_idx)].append({
                'is_correct': is_correct,
                'execution_result': execution_result,
                'generated_answer': generated_answer,
                'generated_code': generated_code,
                'generated_plan': generated_plan
            })
    
    # Analyze each answer generation
    answer_performance = []
    
    for (id_val, plan_idx, code_idx, answer_idx), answer_results in answer_groups.items():
        if len(answer_results) > 0:
            # Get the first result (should be only one per answer generation)
            result = answer_results[0]
            
            # Check if execution_result has no errors
            execution_result = result['execution_result']
            has_no_errors = not any(error_indicator in execution_result.lower() 
                                  for error_indicator in ['error', 'exception', 'traceback', 'failed'])
            
            answer_performance.append({
                'id': id_val,
                'plan_idx': plan_idx,
                'code_idx': code_idx,
                'answer_idx': answer_idx,
                'is_correct': result['is_correct'],
                'has_no_errors': has_no_errors,
                'execution_result': execution_result,
                'generated_answer': result['generated_answer'],
                'generated_code': result['generated_code'],
                'generated_plan': result['generated_plan']
            })
    
    # Sort by id, then by plan_idx, then by code_idx, then by answer_idx
    answer_performance.sort(key=lambda x: (x['id'], x['plan_idx'], x['code_idx'], x['answer_idx']))
    
    return answer_performance


def select_correct_answers(answer_performance: list, data: list) -> list:
    """
    Select correct answers for each ID where:
    1. For special qsubtypes (CausalAnalysis, DescriptiveAnalysis, AnomalyDetection): is_correct >= 0.3
    2. For others: is_correct >= 0.7
    3. Each code appears only once in the training data
    4. Execution errors are allowed if correctness meets the threshold
    Returns list of correct answers with their data.
    """
    # Group answers by ID
    id_answers = defaultdict(list)
    for answer in answer_performance:
        id_answers[answer['id']].append(answer)
    
    # Create a mapping from (id, plan_idx, code_idx, answer_idx) to the original data
    data_map = {}
    for item in data:
        id_val = item.get('id')
        plan_idx = item.get('plan_idx')
        code_idx = item.get('code_idx')
        answer_idx = item.get('answer_idx')
        if id_val is not None and plan_idx is not None and code_idx is not None and answer_idx is not None:
            data_map[(id_val, plan_idx, code_idx, answer_idx)] = item
    
    correct_answers = []
    skipped_ids = []
    used_codes = set()  # Track used (id, plan_idx, code_idx) combinations
    
    for id_val, answers in id_answers.items():
        # Get the qsubtype for this ID from the first answer's data
        qsubtype = None
        if answers and len(answers) > 0:
            first_answer_data = data_map.get((id_val, answers[0]['plan_idx'], answers[0]['code_idx'], answers[0]['answer_idx']))
            if first_answer_data:
                qsubtype = first_answer_data.get('qsubtype')
                qtype = first_answer_data.get('qtype')
        
        # Check if this is an exception qsubtype
        exception_qsubtypes = ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']
        is_exception = qsubtype in exception_qsubtypes
        
        if is_exception:
            # For exception qsubtypes: filter answers with is_correct > 0.3 (allow errors)
            valid_answers = [a for a in answers if a['is_correct'] >= 0.3]
        else:
            # For regular qsubtypes: filter answers with is_correct >= 0.7 (allow errors)
            valid_answers = [a for a in answers if a['is_correct'] >= 0.7]
        
        if valid_answers:
            # Group valid answers by code (plan_idx, code_idx) to ensure each code appears only once
            code_groups = defaultdict(list)
            for answer in valid_answers:
                code_key = (answer['plan_idx'], answer['code_idx'])
                code_groups[code_key].append(answer)
            
            # Collect all valid answers for this ID, sorted by correctness
            id_valid_answers = []
            for code_key, code_answers in code_groups.items():
                plan_idx, code_idx = code_key
                
                # Check if this code has already been used
                global_code_key = (id_val, plan_idx, code_idx)
                if global_code_key in used_codes:
                    continue
                
                # Select the answer with highest correctness for this code
                best_answer = max(code_answers, key=lambda a: a['is_correct'])
                best_answer_data = data_map.get((id_val, best_answer['plan_idx'], best_answer['code_idx'], best_answer['answer_idx']))
                
                if best_answer_data:
                    id_valid_answers.append({
                        'id': id_val,
                        'plan_idx': best_answer['plan_idx'],
                        'code_idx': best_answer['code_idx'],
                        'answer_idx': best_answer['answer_idx'],
                        'is_correct': best_answer['is_correct'],
                        'has_no_errors': best_answer['has_no_errors'],
                        'qsubtype': qsubtype,
                        'qtype': qtype,
                        'data': best_answer_data,
                        'global_code_key': global_code_key
                    })
            
            # Sort by correctness (descending) and take at most 2 answers per ID
            id_valid_answers.sort(key=lambda x: x['is_correct'], reverse=True)
            selected_answers = id_valid_answers[:2]
            
            # Add selected answers to the final list
            for answer in selected_answers:
                correct_answers.append({
                    'id': answer['id'],
                    'plan_idx': answer['plan_idx'],
                    'code_idx': answer['code_idx'],
                    'answer_idx': answer['answer_idx'],
                    'is_correct': answer['is_correct'],
                    'has_no_errors': answer['has_no_errors'],
                    'qsubtype': answer['qsubtype'],
                    'qtype': answer['qtype'],
                    'data': answer['data']
                })
                used_codes.add(answer['global_code_key'])
        else:
            # No answers meet the criteria
            max_correctness = max(a['is_correct'] for a in answers) if answers else 0
            has_any_no_errors = any(a['has_no_errors'] for a in answers) if answers else False
            
            if is_exception:
                reason = f'No answers with correctness >= 0.2 for exception qsubtype: {qsubtype}'
            else:
                reason = 'No answers with correctness >= 0.5'
            
            skipped_ids.append({
                'id': id_val,
                'reason': reason,
                'max_correctness': max_correctness,
                'has_any_no_errors': has_any_no_errors,
                'qsubtype': qsubtype,
                'all_answers': answers
            })
    
    return correct_answers, skipped_ids


def convert_to_verl_format(correct_answers: list, data_source: str, format_instructions: dict) -> list:
    """
    Convert correct answers to Verl format for training.
    Each answer becomes a dictionary with:
    - data_source: source of the data
    - prompt: list with system and user messages
    - ability: task type
    - reward_model: ground truth for evaluation
    - extra_info: additional metadata
    """
    verl_data = []
    
    for i, answer in enumerate(correct_answers):
        data = answer['data']
        
        instruction = data.get('instruction', '')
        original_format_instruction = data.get('format_instruction', '')
        generated_plan = data.get('generated_plan', '')
        generated_code = data.get('generated_code', '')
        execution_result = data.get('execution_result', '')
        gt_answer = data.get('answer', '')
        
        # Use format_instruction from TableInstruct_qwen3_failed.jsonl if available
        id_val = answer['id']
        format_instruction = format_instructions.get(id_val, original_format_instruction)
        
        # Format instruction for answer agent
        formatted_instruction = f"{format_instruction}\n\n{instruction}"
        
        verl_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": cp.PROMPTS["COT_AGENT_ANSWERER_SYSTEM_PROMPT"]
                },
                {
                    "role": "user",
                    "content": cp.PROMPTS["COT_AGENT_ANSWERER_USER_PROMPT"].format(
                        instruction=formatted_instruction,
                        plan=generated_plan,
                        code_output=execution_result
                    )
                }
            ],
            "ability": "answering",
            "reward_model": {"style": "rule", "ground_truth": gt_answer},
            "extra_info": {
                "split": "train",
                "index": answer['id'],
                "answer": gt_answer,
                "question": instruction,
                "plan_idx": answer['plan_idx'],
                "code_idx": answer['code_idx'],
                "answer_idx": answer['answer_idx'],
                "is_correct": answer['is_correct'],
                "has_no_errors": answer['has_no_errors'],
                "qsubtype": answer['qsubtype'],
                "qtype": answer['qtype'],
                "execution_result": execution_result,
                "generated_plan": generated_plan,
                "generated_code": generated_code,
                "format_instruction_source": "TableInstruct_qwen3_failed.jsonl" if id_val in format_instructions else "original",
            },
        }

        if i == 0:
            print(verl_item)
        
        verl_data.append(verl_item)
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and generate GRPO data for answer agent"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_rl_data/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_8b_answer_grpo_data.parquet",
        type=str,
        help="Output file path to save GRPO dataset in Verl format",
    )
    parser.add_argument(
        "--format_instructions_file",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/TableInstruct_qwen3_failed.jsonl",
        type=str,
        help="Path to TableInstruct_qwen3_failed.jsonl file for format instructions",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    # Load format instructions
    format_instructions = load_format_instructions(args.format_instructions_file)
    
    if data:
        answer_performance = analyze_answer_performance(data)
        
        # Select correct answers
        print("\n" + "="*50)
        print("SELECTING CORRECT ANSWERS FOR GRPO")
        print("="*50)
        
        correct_answers, skipped_ids = select_correct_answers(answer_performance, data)
        
        print(f"Correct answers selected: {len(correct_answers)}")
        print(f"IDs skipped: {len(skipped_ids)}")
        
        # Analyze skipped IDs
        if skipped_ids:
            print(f"\nSkipped IDs Analysis:")
            skip_reasons = defaultdict(int)
            skip_qsubtypes = defaultdict(int)
            skip_error_counts = defaultdict(int)
            
            for skipped in skipped_ids:
                reason = skipped['reason']
                skip_reasons[reason] += 1
                
                if 'qsubtype' in skipped and skipped['qsubtype']:
                    skip_qsubtypes[skipped['qsubtype']] += 1
                
                if 'max_correctness' in skipped:
                    max_corr = skipped['max_correctness']
                    if max_corr < 0.3:
                        skip_error_counts['very_low_correctness'] += 1
                    elif max_corr < 0.7:
                        skip_error_counts['low_correctness'] += 1
                    else:
                        skip_error_counts['high_correctness_but_errors'] += 1
                
                # Also check if there are any answers with good correctness but errors
                if 'all_answers' in skipped:
                    good_answers_with_errors = [a for a in skipped['all_answers'] 
                                              if a['is_correct'] >= 0.7 and not a['has_no_errors']]
                    if good_answers_with_errors:
                        skip_error_counts['good_answers_with_errors'] += 1
            
            print(f"  Skip reasons:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason}: {count}")
            
            print(f"  Skip qsubtypes:")
            for qsubtype, count in sorted(skip_qsubtypes.items(), key=lambda x: x[1], reverse=True):
                print(f"    {qsubtype}: {count}")
            
            print(f"  Skip error patterns:")
            for pattern, count in sorted(skip_error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {pattern}: {count}")
        
        if correct_answers:
            # Show statistics
            correctness_values = [answer['is_correct'] for answer in correct_answers]
            qsubtypes = [answer.get('qsubtype', 'Unknown') for answer in correct_answers]
            
            # Count answers per ID
            id_counts = defaultdict(int)
            for answer in correct_answers:
                id_counts[answer['id']] += 1
            
            print(f"\nCorrect Answer Statistics:")
            print(f"  Average correctness: {sum(correctness_values)/len(correctness_values):.3f}")
            print(f"  Min correctness: {min(correctness_values):.3f}")
            print(f"  Max correctness: {max(correctness_values):.3f}")
            # Count answers with errors vs no errors
            answers_with_errors = [answer for answer in correct_answers if not answer['has_no_errors']]
            answers_without_errors = [answer for answer in correct_answers if answer['has_no_errors']]
            
            print(f"  Answers with correctness >= 0.7: {sum(1 for c in correctness_values if c >= 0.7)}")
            print(f"  Answers with correctness == 1: {sum(1 for c in correctness_values if c == 1)}")
            print(f"  Answers with correctness > 0.3: {sum(1 for c in correctness_values if c > 0.3)}")
            print(f"  Answers with execution errors: {len(answers_with_errors)} ({len(answers_with_errors)/len(correctness_values)*100:.1f}%)")
            print(f"  Answers without execution errors: {len(answers_without_errors)} ({len(answers_without_errors)/len(correctness_values)*100:.1f}%)")
            print(f"  Total unique IDs: {len(id_counts)}")
            print(f"  IDs with 1 answer: {sum(1 for count in id_counts.values() if count == 1)}")
            print(f"  IDs with 2 answers: {sum(1 for count in id_counts.values() if count == 2)}")
            
            # Show qsubtype distribution
            qsubtype_counts = defaultdict(int)
            for qsubtype in qsubtypes:
                qsubtype_counts[qsubtype] += 1
            print(f"\nQSubtype Distribution:")
            for qsubtype, count in sorted(qsubtype_counts.items()):
                print(f"  {qsubtype}: {count}")
            
            print(f"\nFirst 5 correct answers:")
            for i, answer in enumerate(correct_answers[:5]):
                print(f"  {i+1}. ID: {answer['id']}, Plan: {answer['plan_idx']}, Code: {answer['code_idx']}, "
                      f"Answer: {answer['answer_idx']}, Correctness: {answer['is_correct']:.3f}, "
                      f"No Errors: {answer['has_no_errors']}, QSubtype: {answer.get('qsubtype', 'Unknown')}")
        
        # Show examples of skipped IDs
        if skipped_ids:
            print(f"\nExamples of Skipped IDs (first 5):")
            for i, skipped in enumerate(skipped_ids[:5]):
                print(f"\n  Skipped ID {i+1}: {skipped['id']}")
                print(f"    Reason: {skipped['reason']}")
                if 'max_correctness' in skipped:
                    print(f"    Max correctness: {skipped['max_correctness']:.3f}")
                if 'has_any_no_errors' in skipped:
                    print(f"    Has any no errors: {skipped['has_any_no_errors']}")
                
                # Show some example answers and their codes
                if 'all_answers' in skipped and len(skipped['all_answers']) > 0:
                    print(f"    Total answers for this ID: {len(skipped['all_answers'])}")
                    
                    # Show top 3 answers by correctness
                    sorted_answers = sorted(skipped['all_answers'], key=lambda x: x['is_correct'], reverse=True)
                    for j, answer in enumerate(sorted_answers[:3]):
                        print(f"      Answer {j+1}:")
                        print(f"        Correctness: {answer['is_correct']:.3f}")
                        print(f"        Has no errors: {answer['has_no_errors']}")
                        print(f"        Plan: {answer['plan_idx']}, Code: {answer['code_idx']}, Answer: {answer['answer_idx']}")
                        
                        # Show a snippet of the generated code
                        if 'generated_code' in answer:
                            code = answer['generated_code']
                            if len(code) > 200:
                                code = code[:200] + "..."
                            print(f"        Code snippet: {code}")
                        
                        # Show a snippet of the generated answer
                        if 'generated_answer' in answer:
                            ans = answer['generated_answer']
                            if len(ans) > 100:
                                ans = ans[:100] + "..."
                            print(f"        Answer snippet: {ans}")
                        
                        # Show execution result if there are errors
                        if not answer['has_no_errors'] and 'execution_result' in answer:
                            exec_result = answer['execution_result']
                            if len(exec_result) > 150:
                                exec_result = exec_result[:150] + "..."
                            print(f"        Execution result: {exec_result}")
                        print()
        
        # Convert to Verl format and save if output path provided
        if args.output and correct_answers:
            print(f"\nConverting to Verl format...")
            verl_data = convert_to_verl_format(correct_answers, args.directory, format_instructions)
            
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