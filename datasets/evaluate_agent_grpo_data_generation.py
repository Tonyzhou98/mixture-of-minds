#!/usr/bin/env python3
"""
Script to read all JSONL files from a directory and generate GRPO data for evaluate agent.
Creates answer pairs for each ID where we have multiple answers, and asks the evaluator to judge which is better.
For each ID, we create pairs of answers and use is_correct as ground truth for which answer is better.
The aim is to find proper input data for GRPO training of the evaluate agent.
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
import itertools

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


def create_answer_pairs(answer_performance: list, data: list, format_instructions: dict, max_samples_per_id: int = 4) -> list:
    """
    Create answer pairs for evaluation training.
    For each ID, create pairs of answers and determine which is better based on is_correct.
    Returns list of answer pairs with their evaluation data.
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
    
    answer_pairs = []
    skipped_ids = []
    
    for id_val, answers in id_answers.items():
        # Get the question context for this ID from the first answer's data
        question_context = None
        qsubtype = None
        qtype = None
        format_instruction = None
        if answers and len(answers) > 0:
            first_answer_data = data_map.get((id_val, answers[0]['plan_idx'], answers[0]['code_idx'], answers[0]['answer_idx']))
            if first_answer_data:
                instruction = first_answer_data.get('instruction', '')
                original_format_instruction = first_answer_data.get('format_instruction', '')
                # Use format_instruction from TableInstruct_qwen3_failed.jsonl if available
                format_instruction = format_instructions.get(id_val, original_format_instruction)
                # Format instruction for answer agent (same as in answer_agent_grpo_data_generation.py)
                question_context = f"{format_instruction}\n\n{instruction}"
                qsubtype = first_answer_data.get('qsubtype')
                qtype = first_answer_data.get('qtype')
        
        # Filter answers based on qsubtype requirements
        if qsubtype in ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']:
            # For analysis qsubtypes, use is_correct threshold to ensure meaningful differences
            valid_answers = [a for a in answers]
        else:
            valid_answers = []
        # else:
            
        #     # Check which answers match ground truth
        #     correct_answers = []
        #     incorrect_answers = []
            
        #     for a in answers:
        #         # Get the ground truth answer from data_map
        #         answer_data = data_map.get((id_val, a['plan_idx'], a['code_idx'], a['answer_idx']))
        #         if answer_data and 'answer' in answer_data:
        #             gt_answer = answer_data['answer']
        #             if normalize_answer(a['generated_answer']) == normalize_answer(gt_answer):
        #                 correct_answers.append(a)
        #             else:
        #                 incorrect_answers.append(a)
        #         else:
        #             # If we can't find ground truth, skip this answer
        #             continue
            
        #     # Only proceed if we have both correct and incorrect answers
        #     if len(correct_answers) > 0 and len(incorrect_answers) > 0:
        #         valid_answers = correct_answers + incorrect_answers
        #     else:
        #         valid_answers = []
        
        if len(valid_answers) >= 2:
            # Create pairs of answers
            pairs_created = 0
            max_pairs = min(max_samples_per_id, len(valid_answers) * (len(valid_answers) - 1) // 2)
            
            # Sort answers by correctness for better pairing
            valid_answers.sort(key=lambda x: x['is_correct'], reverse=True)
            
            # Create pairs ensuring diversity based on qsubtype
            for i in range(len(valid_answers)):
                if pairs_created >= max_pairs:
                    break
                    
                for j in range(i + 1, len(valid_answers)):
                    if pairs_created >= max_pairs:
                        break
                    
                    answer_a = valid_answers[i]
                    answer_b = valid_answers[j]
                    
                    # Randomly assign which answer becomes A vs B to ensure balance
                    if random.random() < 0.5:
                        # Swap A and B randomly
                        answer_a, answer_b = answer_b, answer_a
                        answer_a_data = data_map.get((id_val, answer_a['plan_idx'], answer_a['code_idx'], answer_a['answer_idx']))
                        answer_b_data = data_map.get((id_val, answer_b['plan_idx'], answer_b['code_idx'], answer_b['answer_idx']))
                    else:
                        answer_a_data = data_map.get((id_val, answer_a['plan_idx'], answer_a['code_idx'], answer_a['answer_idx']))
                        answer_b_data = data_map.get((id_val, answer_b['plan_idx'], answer_b['code_idx'], answer_b['answer_idx']))
                    
                    # Different pairing logic based on qsubtype
                    if qsubtype in ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']:
                        # For these qsubtypes, use is_correct difference > 0.1
                        correctness_diff = abs(answer_a['is_correct'] - answer_b['is_correct'])
                        
                        if correctness_diff < 0.1:
                            # Skip pairs with too small correctness difference
                            continue
                        
                        if answer_a['is_correct'] > answer_b['is_correct']:
                            better_answer = 'A'
                            worse_answer = 'B'
                        elif answer_b['is_correct'] > answer_a['is_correct']:
                            better_answer = 'B'
                            worse_answer = 'A'
                        else:
                            # If correctness is equal, skip this pair
                            continue
                    else:
                        # For other qsubtypes, ensure chosen answer matches ground truth and rejected doesn't
                        # Check if answers match ground truth
                        if not answer_a_data or not answer_b_data or 'answer' not in answer_a_data or 'answer' not in answer_b_data:
                            # Skip if we can't find ground truth for either answer
                            continue
                        
                        answer_a_matches = normalize_answer(answer_a['generated_answer']) == normalize_answer(answer_a_data['answer'])
                        answer_b_matches = normalize_answer(answer_b['generated_answer']) == normalize_answer(answer_b_data['answer'])
                        
                        # Skip if both match or both don't match (we need one correct, one incorrect)
                        if answer_a_matches == answer_b_matches:
                            continue
                        
                        # Determine which is better based on correctness
                        if answer_a_matches and not answer_b_matches:
                            better_answer = 'A'
                            worse_answer = 'B'
                        elif answer_b_matches and not answer_a_matches:
                            better_answer = 'B'
                            worse_answer = 'A'
                        else:
                            # This shouldn't happen given the check above, but just in case
                            continue
                    
                    if answer_a_data and answer_b_data:
                        answer_pairs.append({
                            'id': id_val,
                            'question_context': question_context,
                            'qsubtype': qsubtype,
                            'qtype': qtype,
                            'answer_a': answer_a_data['answer_raw_response'],
                            'answer_b': answer_b_data['answer_raw_response'],
                            'answer_a_correctness': answer_a['is_correct'],
                            'answer_b_correctness': answer_b['is_correct'],
                            'answer_a_has_no_errors': answer_a['has_no_errors'],
                            'answer_b_has_no_errors': answer_b['has_no_errors'],
                            'better_answer': better_answer,
                            'worse_answer': worse_answer,
                            'answer_a_data': answer_a_data,
                            'answer_b_data': answer_b_data,
                            'answer_a_plan': answer_a['generated_plan'],
                            'answer_b_plan': answer_b['generated_plan'],
                            'answer_a_code': answer_a['generated_code'],
                            'answer_b_code': answer_b['generated_code'],
                            'answer_a_execution': answer_a['execution_result'],
                            'answer_b_execution': answer_b['execution_result'],
                            'ground_truth_answer': answer_a_data.get('answer', '')  # Add ground truth answer
                        })
                        pairs_created += 1
        else:
            # Not enough valid answers for comparison
            max_correctness = max(a['is_correct'] for a in answers) if answers else 0
            skipped_ids.append({
                'id': id_val,
                'reason': f'Not enough valid answers (need >= 2, got {len(valid_answers)})',
                'max_correctness': max_correctness,
                'total_answers': len(answers),
                'valid_answers': len(valid_answers),
                'qsubtype': qsubtype
            })
    
    return answer_pairs, skipped_ids


def convert_to_verl_format(answer_pairs: list, data_source: str, format_instructions: dict) -> list:
    """
    Convert answer pairs to Verl format for training.
    Each pair becomes a dictionary with:
    - data_source: source of the data
    - prompt: list with system and user messages
    - ability: task type
    - reward_model: ground truth for evaluation
    - extra_info: additional metadata
    """
    verl_data = []
    
    for i, pair in enumerate(answer_pairs):
        question_context = pair['question_context']  # This now includes format_instruction + instruction
        answer_a = pair['answer_a']
        answer_b = pair['answer_b']
        better_answer = pair['better_answer']
        
        # Create the ground truth response
        if better_answer == 'A':
            ground_truth = "Answer A"
        else:
            ground_truth = "Answer B"
        
        verl_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": cp.PROMPTS["COT_AGENT_EVALUATOR_SYSTEM_PROMPT"]
                },
                {
                    "role": "user",
                    "content": cp.PROMPTS["COT_AGENT_EVALUATOR_USER_PROMPT"].format(
                        instruction=question_context,
                        answer_a=answer_a,
                        answer_b=answer_b
                    )
                }
            ],
            "ability": "evaluating",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": "train",
                "index": pair['id'],
                "question": question_context,
                "ground_truth_answer": pair.get('ground_truth_answer', ''),  # Add ground truth answer
                "answer_a_correctness": pair['answer_a_correctness'],
                "answer_b_correctness": pair['answer_b_correctness'],
                "answer_a_has_no_errors": pair['answer_a_has_no_errors'],
                "answer_b_has_no_errors": pair['answer_b_has_no_errors'],
                "better_answer": better_answer,
                "worse_answer": pair['worse_answer'],
                "qsubtype": pair['qsubtype'],
                "qtype": pair['qtype'],
                "answer_a_plan": pair['answer_a_plan'],
                "answer_b_plan": pair['answer_b_plan'],
                "answer_a_code": pair['answer_a_code'],
                "answer_b_code": pair['answer_b_code'],
                "answer_a_execution": pair['answer_a_execution'],
                "answer_b_execution": pair['answer_b_execution']
            },
        }

        if i == 0:
            print(verl_item)
        
        verl_data.append(verl_item)
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and generate GRPO data for evaluate agent"
    )
    parser.add_argument(
        "--directory",
        default="/your/path/to/your_fs/RankMind/datasets/rl_datasets/qwen3_8b_rl_data/",
        type=str,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/qwen3_8b_evaluate_grpo_data.parquet",
        type=str,
        help="Output file path to save GRPO dataset in Verl format",
    )
    parser.add_argument(
        "--format_instructions_file",
        default="/your/path/to/your_fs/RankMind/datasets/train_data/TableInstruct_qwen3_failed.jsonl",
        type=str,
        help="Path to TableInstruct_qwen3_failed.jsonl file for format instructions",
    )
    parser.add_argument(
        "--max_samples_per_id",
        default=3,
        type=int,
        help="Maximum number of answer pairs to create per ID",
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    data = read_jsonl_file(directory)
    
    # Load format instructions
    format_instructions = load_format_instructions(args.format_instructions_file)
    
    if data:
        answer_performance = analyze_answer_performance(data)
        
        # Create answer pairs
        print("\n" + "="*50)
        print("CREATING ANSWER PAIRS FOR EVALUATION TRAINING")
        print("="*50)
        
        answer_pairs, skipped_ids = create_answer_pairs(answer_performance, data, format_instructions, args.max_samples_per_id)
        
        print(f"Answer pairs created: {len(answer_pairs)}")
        print(f"IDs skipped: {len(skipped_ids)}")
        
        # Analyze skipped IDs
        if skipped_ids:
            print(f"\nSkipped IDs Analysis:")
            skip_reasons = defaultdict(int)
            skip_qsubtypes = defaultdict(int)
            
            for skipped in skipped_ids:
                reason = skipped['reason']
                skip_reasons[reason] += 1
                
                if 'qsubtype' in skipped and skipped['qsubtype']:
                    skip_qsubtypes[skipped['qsubtype']] += 1
            
            print(f"  Skip reasons:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason}: {count}")
            
            print(f"  Skip qsubtypes:")
            for qsubtype, count in sorted(skip_qsubtypes.items(), key=lambda x: x[1], reverse=True):
                print(f"    {qsubtype}: {count}")
        
        if answer_pairs:
            # Show statistics
            correctness_diffs = [abs(pair['answer_a_correctness'] - pair['answer_b_correctness']) for pair in answer_pairs]
            better_answer_counts = defaultdict(int)
            qsubtypes = [pair.get('qsubtype', 'Unknown') for pair in answer_pairs]
            
            # Count pairs per ID
            id_counts = defaultdict(int)
            for pair in answer_pairs:
                id_counts[pair['id']] += 1
            
            print(f"\nAnswer Pair Statistics:")
            print(f"  Average correctness difference: {sum(correctness_diffs)/len(correctness_diffs):.3f}")
            print(f"  Min correctness difference: {min(correctness_diffs):.3f}")
            print(f"  Max correctness difference: {max(correctness_diffs):.3f}")
            
            # Count better answers
            for pair in answer_pairs:
                better_answer_counts[pair['better_answer']] += 1
            
            print(f"  Better answer distribution:")
            for answer, count in sorted(better_answer_counts.items()):
                print(f"    Answer {answer}: {count} ({count/len(answer_pairs)*100:.1f}%)")
            
            # Enhanced balance analysis
            print(f"\nBalance Analysis:")
            total_pairs = len(answer_pairs)
            answer_a_count = better_answer_counts.get('A', 0)
            answer_b_count = better_answer_counts.get('B', 0)
            
            print(f"  Total pairs: {total_pairs}")
            print(f"  Answer A is better: {answer_a_count} ({answer_a_count/total_pairs*100:.1f}%)")
            print(f"  Answer B is better: {answer_b_count} ({answer_b_count/total_pairs*100:.1f}%)")
            
            # Check if data is balanced
            balance_ratio = min(answer_a_count, answer_b_count) / max(answer_a_count, answer_b_count) if max(answer_a_count, answer_b_count) > 0 else 0
            print(f"  Balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced, 0.0 = completely imbalanced)")
            
            if balance_ratio >= 0.8:
                print(f"✅ Data is well balanced!")
            elif balance_ratio >= 0.6:
                print(f"⚠️  Data is moderately balanced")
            else:
                print(f"❌ Data is imbalanced - consider balancing strategies")
            
            # Show balance by qsubtype
            print(f"\nBalance by QSubtype:")
            qsubtype_balance = defaultdict(lambda: {'A': 0, 'B': 0})
            for pair in answer_pairs:
                qsubtype = pair.get('qsubtype', 'Unknown')
                better = pair['better_answer']
                qsubtype_balance[qsubtype][better] += 1
            
            for qsubtype, counts in sorted(qsubtype_balance.items()):
                total_qsubtype = counts['A'] + counts['B']
                if total_qsubtype > 0:
                    a_pct = counts['A'] / total_qsubtype * 100
                    b_pct = counts['B'] / total_qsubtype * 100
                    balance = min(counts['A'], counts['B']) / max(counts['A'], counts['B']) if max(counts['A'], counts['B']) > 0 else 0
                    print(f"  {qsubtype}: A={counts['A']} ({a_pct:.1f}%), B={counts['B']} ({b_pct:.1f}%), Balance={balance:.3f}")
            
            print(f"  Total unique IDs: {len(id_counts)}")
            print(f"  IDs with 1 pair: {sum(1 for count in id_counts.values() if count == 1)}")
            print(f"  IDs with 2 pairs: {sum(1 for count in id_counts.values() if count == 2)}")
            print(f"  IDs with 3 pairs: {sum(1 for count in id_counts.values() if count == 3)}")
            print(f"  IDs with 4 pairs: {sum(1 for count in id_counts.values() if count == 4)}")
            
            # Show qsubtype distribution
            qsubtype_counts = defaultdict(int)
            for qsubtype in qsubtypes:
                qsubtype_counts[qsubtype] += 1
            print(f"\nQSubtype Distribution:")
            for qsubtype, count in sorted(qsubtype_counts.items()):
                print(f"  {qsubtype}: {count}")
            
            print(f"\nFirst 5 answer pairs:")
            for i, pair in enumerate(answer_pairs[:5]):
                print(f"  {i+1}. ID: {pair['id']}, Better: Answer {pair['better_answer']}, "
                      f"Correctness A: {pair['answer_a_correctness']:.3f}, "
                      f"Correctness B: {pair['answer_b_correctness']:.3f}, "
                      f"QSubtype: {pair.get('qsubtype', 'Unknown')}")
        
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
        if args.output and answer_pairs:
            print(f"\nConverting to Verl format...")
            verl_data = convert_to_verl_format(answer_pairs, args.directory, format_instructions)
            
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