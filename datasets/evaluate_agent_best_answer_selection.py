 #!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate GRPO data for evaluate agent.
Creates answer selection tasks for each ID where we have multiple answers, and asks the evaluator to select the best answer(s).
For each ID, we present all available answers and use the answer(s) with highest is_correct as ground truth.
The aim is to find proper input data for GRPO training of the evaluate agent for best answer selection.
"""

import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
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

# Prompts are now imported from cot_prompts.py


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


def create_answer_selection_tasks(answer_performance: list, data: list, format_instructions: dict, target_qsubtypes: list) -> list:
    """
    Create answer selection tasks for evaluation training.
    For each ID, create a task with all available answers and determine which is/are best based on is_correct.
    Returns list of answer selection tasks with their evaluation data.
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
    
    answer_selection_tasks = []
    skipped_ids = []
    
    for id_val, answers in id_answers.items():
        # Filter answers to ensure we have at least one with is_correct > 0
        valid_answers = [a for a in answers if a['is_correct'] > 0]
        
        if len(valid_answers) < 2:
            # Skip if we don't have at least 2 valid answers
            max_correctness = max(a['is_correct'] for a in answers) if answers else 0
            skipped_ids.append({
                'id': id_val,
                'reason': f'Not enough valid answers with is_correct > 0 (need >= 2, got {len(valid_answers)})',
                'max_correctness': max_correctness,
                'total_answers': len(answers),
                'valid_answers': len(valid_answers)
            })
            continue
        
        # If we have more than 8 valid answers, randomly select 8
        if len(valid_answers) > 8:
            # Set random seed for reproducibility based on ID
            random.seed(hash(id_val) % (2**32))
            valid_answers = random.sample(valid_answers, 8)
            # Reset random seed to avoid affecting other operations
            random.seed(42)
        
        # Get the question context for this ID to determine qsubtype
        question_context = None
        qsubtype = None
        qtype = None
        format_instruction = None
        if valid_answers and len(valid_answers) > 0:
            first_answer_data = data_map.get((id_val, valid_answers[0]['plan_idx'], valid_answers[0]['code_idx'], valid_answers[0]['answer_idx']))
            if first_answer_data:
                instruction = first_answer_data.get('instruction', '')
                original_format_instruction = first_answer_data.get('format_instruction', '')
                # Use format_instruction from TableInstruct_qwen3_failed.jsonl if available
                format_instruction = format_instructions.get(id_val, original_format_instruction)
                # Format instruction for answer agent
                question_context = f"{format_instruction}\n\n{instruction}"
                qsubtype = first_answer_data.get('qsubtype')
                qtype = first_answer_data.get('qtype')
        
        # Only process questions with specific qsubtypes
        if qsubtype not in target_qsubtypes:
            skipped_ids.append({
                'id': id_val,
                'reason': f'QSubtype {qsubtype} not in target list {target_qsubtypes}',
                'max_correctness': max(a['is_correct'] for a in valid_answers) if valid_answers else 0,
                'total_answers': len(answers),
                'valid_answers': len(valid_answers),
                'qsubtype': qsubtype
            })
            continue
        
        # Check if all answers have the same normalized content
        normalized_answers = []
        for answer in valid_answers:
            answer_data = data_map.get((id_val, answer['plan_idx'], answer['code_idx'], answer['answer_idx']))
            if answer_data and 'generated_answer' in answer_data:
                normalized_answers.append(answer_data['generated_answer'])
        
        # Skip if all normalized answers are the same
        if len(set(normalized_answers)) <= 1:
            skipped_ids.append({
                'id': id_val,
                'reason': f'All answers have the same normalized content (normalized: {normalized_answers[0] if normalized_answers else "N/A"})',
                'max_correctness': max(a['is_correct'] for a in valid_answers) if valid_answers else 0,
                'total_answers': len(answers),
                'valid_answers': len(valid_answers),
                'normalized_answers': normalized_answers,
                'qsubtype': qsubtype
            })
            continue
        

        
        # Find the highest is_correct value
        max_correctness = max(a['is_correct'] for a in valid_answers)
        
        # Find all answers with the highest is_correct
        best_answers = [a for a in valid_answers if a['is_correct'] == max_correctness]
        
        # Create answer options with letters
        answer_options = []
        ground_truth_letters = []
        
        for i, answer in enumerate(valid_answers):
            letter = chr(65 + i)  # A, B, C, D, etc.
            
            # Get the original data for this answer
            answer_data = data_map.get((id_val, answer['plan_idx'], answer['code_idx'], answer['answer_idx']))
            
            if answer_data:
                answer_options.append(f"{letter}. {answer_data['generated_answer']}")
                
                # Check if this answer is one of the best answers
                if answer['is_correct'] == max_correctness:
                    ground_truth_letters.append(letter)
        
        # Create the ground truth string (e.g., "A", "AB", "ABC")
        ground_truth = "".join(sorted(ground_truth_letters))
        
        # Create the answer selection task
        answer_selection_tasks.append({
            'id': id_val,
            'question_context': question_context,
            'qsubtype': qsubtype,
            'qtype': qtype,
            'answer_options': "\n\n".join(answer_options),
            'ground_truth': ground_truth,
            'max_correctness': max_correctness,
            'total_answers': len(valid_answers),
            'best_answers_count': len(best_answers),
            'all_answers': valid_answers,
            'answer_data_map': {chr(65 + i): data_map.get((id_val, a['plan_idx'], a['code_idx'], a['answer_idx'])) 
                               for i, a in enumerate(valid_answers)}
        })
    
    return answer_selection_tasks, skipped_ids


def convert_to_verl_format(answer_selection_tasks: list, data_source: str, format_instructions: dict) -> list:
    """
    Convert answer selection tasks to Verl format for training.
    Each task becomes a dictionary with:
    - data_source: source of the data
    - prompt: list with system and user messages
    - ability: task type
    - reward_model: ground truth for evaluation
    - extra_info: additional metadata
    """
    verl_data = []
    
    for i, task in enumerate(answer_selection_tasks):
        question_context = task['question_context']
        answer_options = task['answer_options']
        ground_truth = task['ground_truth']
        
        verl_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": cp.PROMPTS["BEST_ANSWER_SELECTION_SYSTEM_PROMPT"]
                },
                {
                    "role": "user",
                    "content": cp.PROMPTS["BEST_ANSWER_SELECTION_USER_PROMPT"].format(
                        instruction=question_context,
                        answer_options=answer_options
                    )
                }
            ],
            "ability": "evaluating",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": "train",
                "index": task['id'],
                "question": question_context,
                "ground_truth": ground_truth,
                "max_correctness": task['max_correctness'],
                "total_answers": task['total_answers'],
                "best_answers_count": task['best_answers_count'],
                "qsubtype": task['qsubtype'],
                "qtype": task['qtype'],
            },
        }

        if i == 0:
            print(verl_item)
        
        verl_data.append(verl_item)
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and generate GRPO data for evaluate agent (best answer selection)"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_rl_data/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_8b_evaluate_best_answer_selection_grpo_data.parquet",
        type=str,
        help="Output file path to save GRPO dataset in Verl format (will append qsubtype info)",
    )
    parser.add_argument(
        "--format_instructions_file",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/TableInstruct_qwen3_failed.jsonl",
        type=str,
        help="Path to TableInstruct_qwen3_failed.jsonl file for format instructions",
    )
    parser.add_argument(
        "--target_qsubtypes",
        default="CausalAnalysis,DescriptiveAnalysis,AnomalyDetection",
        type=str,
        help="Comma-separated list of target qsubtypes to process",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    # Parse target qsubtypes
    target_qsubtypes = [q.strip() for q in args.target_qsubtypes.split(',')]
    print(f"Target qsubtypes: {target_qsubtypes}")
    
    # Load format instructions
    format_instructions = load_format_instructions(args.format_instructions_file)
    
    if data:
        answer_performance = analyze_answer_performance(data)
        
        # Create answer selection tasks
        print("\n" + "="*50)
        print("CREATING ANSWER SELECTION TASKS FOR EVALUATION TRAINING")
        print("="*50)
        print("Note: If more than 8 valid answers exist, only 8 will be randomly selected")
        print("="*50)
        
        answer_selection_tasks, skipped_ids = create_answer_selection_tasks(answer_performance, data, format_instructions, target_qsubtypes)
        
        print(f"Answer selection tasks created: {len(answer_selection_tasks)}")
        print(f"IDs skipped: {len(skipped_ids)}")
        print(f"Target qsubtypes: {', '.join(target_qsubtypes)}")
        
        # Count how many were skipped due to qsubtype filtering
        qsubtype_filtered_count = sum(1 for s in skipped_ids if 'QSubtype' in s.get('reason', ''))
        if qsubtype_filtered_count > 0:
            print(f"  - Skipped due to qsubtype filtering: {qsubtype_filtered_count}")
            print(f"    (Only processing qsubtypes: {', '.join(target_qsubtypes)})")
        
        # Count how many were skipped due to identical normalized answers
        identical_normalized_count = sum(1 for s in skipped_ids if 'normalized_answers' in s)
        if identical_normalized_count > 0:
            print(f"  - Skipped due to identical normalized answers: {identical_normalized_count}")
        
        # Analyze skipped IDs
        if skipped_ids:
            print(f"\nSkipped IDs Analysis:")
            skip_reasons = defaultdict(int)
            
            for skipped in skipped_ids:
                reason = skipped['reason']
                skip_reasons[reason] += 1
            
            print(f"  Skip reasons:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason}: {count}")
            
            # Show examples of normalized answer duplicates
            normalized_duplicates = [s for s in skipped_ids if 'normalized_answers' in s]
            if normalized_duplicates:
                print(f"\nExamples of questions with identical normalized answers (first 3):")
                for i, skipped in enumerate(normalized_duplicates[:3]):
                    print(f"  {i+1}. ID: {skipped['id']}")
                    print(f"     QSubtype: {skipped.get('qsubtype', 'Unknown')}")
                    print(f"     Normalized answer: {skipped['normalized_answers'][0] if skipped['normalized_answers'] else 'N/A'}")
                    print(f"     Total answers: {skipped['total_answers']}")
                    print(f"     Valid answers: {skipped['valid_answers']}")
                
                # Show qsubtype distribution for normalized duplicates
                qsubtype_counts = defaultdict(int)
                for skipped in normalized_duplicates:
                    qsubtype = skipped.get('qsubtype', 'Unknown')
                    qsubtype_counts[qsubtype] += 1
                print(f"\nQSubtype distribution for questions with identical normalized answers:")
                for qsubtype, count in sorted(qsubtype_counts.items()):
                    print(f"  {qsubtype}: {count}")
        
        if answer_selection_tasks:
            # Show statistics
            max_correctness_values = [task['max_correctness'] for task in answer_selection_tasks]
            best_answers_counts = [task['best_answers_count'] for task in answer_selection_tasks]
            total_answers_counts = [task['total_answers'] for task in answer_selection_tasks]
            qsubtypes = [task.get('qsubtype', 'Unknown') for task in answer_selection_tasks]
            
            # Count tasks per ID
            id_counts = defaultdict(int)
            for task in answer_selection_tasks:
                id_counts[task['id']] += 1
            
            print(f"\nAnswer Selection Task Statistics:")
            print(f"  Average max correctness: {sum(max_correctness_values)/len(max_correctness_values):.3f}")
            print(f"  Min max correctness: {min(max_correctness_values):.3f}")
            print(f"  Max correctness: {max(max_correctness_values):.3f}")
            print(f"  Average best answers count: {sum(best_answers_counts)/len(best_answers_counts):.1f}")
            print(f"  Average total answers per task: {sum(total_answers_counts)/len(total_answers_counts):.1f}")
            print(f"  Note: Answers limited to maximum of 8 per task for consistency")
            
            # Count ground truth patterns
            ground_truth_patterns = defaultdict(int)
            for task in answer_selection_tasks:
                ground_truth_patterns[task['ground_truth']] += 1
            
            print(f"\nGround Truth Pattern Distribution:")
            for pattern, count in sorted(ground_truth_patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"  {pattern}: {count} ({count/len(answer_selection_tasks)*100:.1f}%)")
            
            # Show balance analysis
            print(f"\nBalance Analysis:")
            single_best = sum(1 for pattern in ground_truth_patterns.keys() if len(pattern) == 1)
            multiple_best = sum(1 for pattern in ground_truth_patterns.keys() if len(pattern) > 1)
            
            print(f"  Single best answer tasks: {single_best} ({single_best/len(answer_selection_tasks)*100:.1f}%)")
            print(f"  Multiple best answers tasks: {multiple_best} ({multiple_best/len(answer_selection_tasks)*100:.1f}%)")
            
            # Show qsubtype distribution
            qsubtype_counts = defaultdict(int)
            for qsubtype in qsubtypes:
                qsubtype_counts[qsubtype] += 1
            print(f"\nQSubtype Distribution (Target qsubtypes only):")
            for qsubtype, count in sorted(qsubtype_counts.items()):
                print(f"  {qsubtype}: {count}")
            
            # Verify we only have target qsubtypes
            non_target_qsubtypes = [q for q in qsubtypes if q not in target_qsubtypes]
            if non_target_qsubtypes:
                print(f"  Warning: Found non-target qsubtypes: {set(non_target_qsubtypes)}")
            else:
                print(f"  ✓ All tasks are from target qsubtypes")
            
            print(f"\nFirst 5 answer selection tasks:")
            for i, task in enumerate(answer_selection_tasks[:5]):
                print(f"  {i+1}. ID: {task['id']}, Ground Truth: {task['ground_truth']}, "
                      f"Max Correctness: {task['max_correctness']:.3f}, "
                      f"Total Answers: {task['total_answers']}, "
                      f"Best Answers: {task['best_answers_count']}, "
                      f"QSubtype: {task.get('qsubtype', 'Unknown')}")
        
        # Show examples of skipped IDs
        if skipped_ids:
            print(f"\nExamples of Skipped IDs (first 5):")
            for i, skipped in enumerate(skipped_ids[:5]):
                print(f"\n  Skipped ID {i+1}: {skipped['id']}")
                print(f"    Reason: {skipped['reason']}")
                if 'max_correctness' in skipped:
                    print(f"    Max correctness: {skipped['max_correctness']:.3f}")
                if 'total_answers' in skipped:
                    print(f"    Total answers: {skipped['total_answers']}")
                    print(f"    Valid answers: {skipped['valid_answers']}")
        
        # Convert to Verl format and save if output path provided
        if args.output and answer_selection_tasks:
            print(f"\nConverting to Verl format...")
            verl_data = convert_to_verl_format(answer_selection_tasks, args.directory, format_instructions)
            
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
            
            # Save to parquet with qsubtype info
            parquet_base = args.output.replace(".parquet", "")
            qsubtype_suffix = "_" + "_".join(target_qsubtypes).lower().replace("analysis", "anal").replace("detection", "detect")
            train_dataset.to_parquet(parquet_base + qsubtype_suffix + "_train.parquet")
            test_dataset.to_parquet(parquet_base + qsubtype_suffix + "_test.parquet")
            
            print(f"Saved {len(train_dataset)} training examples to {parquet_base + qsubtype_suffix}_train.parquet")
            print(f"Saved {len(test_dataset)} evaluation examples to {parquet_base + qsubtype_suffix}_test.parquet")
            print(f"Format: Verl compatible parquet")
            print(f"QSubtypes: {', '.join(target_qsubtypes)}")


if __name__ == "__main__":
    main()