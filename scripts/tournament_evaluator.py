import random
import os
import sys
from collections import defaultdict
from vllm import SamplingParams
import tqdm

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)

from evaluation.TableBench.eval.table_bench_custom_eval import *
from evaluation.TableBench.metrics.custom_em_metric import *
from agents import cot_prompts as cp


def run_tournament_evaluation(evaluation_tournaments, args, df, tokenizer, eval_llm):
    """
    Run tournament-style evaluation for multiple generations per idx.
    
    Args:
        evaluation_tournaments: List of tournament structures
        args: Command line arguments
        df: Original DataFrame with instructions
        tokenizer: Tokenizer for prompt formatting
    
    Returns:
        data: List of selected generations (one per idx)
    """
    data = []
    
    if not evaluation_tournaments:
        return data
    
    # Process evaluation tournaments in batches
    eval_batch_size = args.batch_size
    
    # Main progress bar for tournaments
    tournament_pbar = tqdm.tqdm(
        total=len(evaluation_tournaments),
        desc="Processing tournaments",
        unit="tournament"
    )
    
    for i in range(0, len(evaluation_tournaments), eval_batch_size):
        batch = evaluation_tournaments[i:i+eval_batch_size]
        
        # Process each tournament
        for tournament in batch:
            print(f"Running tournament for idx {tournament['idx']} with {len(tournament['generations'])} generations")
            
            # Start with the generations for this tournament
            current_generations = tournament['generations'].copy()
            
            # Run tournament rounds until only one generation remains
            round_num = 1
            while len(current_generations) > 1:
                print(f"Round {round_num}: {len(current_generations)} generations remaining")
                
                # Handle odd number of generations by skipping the last one
                skipped_generation = None
                if len(current_generations) % 2 == 1:
                    skipped_generation = current_generations.pop()
                    print(f"Odd number detected, skipping last generation: {skipped_generation['generated_answer']}")
                
                # Create pairs for this round
                round_pairs = []
                for j in range(0, len(current_generations), 2):
                    if j + 1 < len(current_generations):
                        # Create a pair
                        gen_a = current_generations[j]
                        gen_b = current_generations[j + 1]
                        round_pairs.append({
                            'gen_a': gen_a,
                            'gen_b': gen_b,
                            'answer_a': gen_a['answer_raw_response'],
                            'answer_b': gen_b['answer_raw_response']
                        })
                
                # Initialize winners list
                winners = []
                
                # Evaluate all pairs in this round
                if round_pairs:
                    eval_prompts = []
                    valid_pairs = []
                    
                    for pair in round_pairs:
                        # Combine instruction and format_instruction
                        full_instruction =  f"{tournament['format_instruction']}\n\n{tournament['instruction']}"
                        
                        # Create the evaluation prompt using the imported COT evaluator format
                        system_prompt = cp.PROMPTS["COT_AGENT_EVALUATOR_SYSTEM_PROMPT"]
                        
                        user_prompt = cp.PROMPTS["COT_AGENT_EVALUATOR_USER_PROMPT"].format(
                            instruction=full_instruction,
                            answer_a=pair['answer_a'],
                            answer_b=pair['answer_b']
                        )
                        
                        # Use tokenizer.apply_chat_template like in rl_data_collector.py
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        
                        full_prompt = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        eval_prompts.append(full_prompt)
                        valid_pairs.append(pair)
                    
                    # Generate evaluations for this round
                    if eval_prompts:
                        eval_sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
                        eval_outputs = eval_llm.generate(eval_prompts, eval_sampling_params)
                        
                        # Process evaluation results for this round
                        for pair, eval_output in zip(valid_pairs, eval_outputs):
                            eval_text = eval_output.outputs[0].text.strip()
                            
                            # Extract judgment from evaluation
                            print(f"\n--- Round {round_num} Evaluation ---")
                            print(f"Answer A (is_correct: {pair['gen_a']['is_correct']}): {pair['gen_a']['generated_answer']}")
                            print(f"Answer B (is_correct: {pair['gen_b']['is_correct']}): {pair['gen_b']['generated_answer']}")
                            print(f'Ground truth: {pair["gen_a"]["answer"]}')
                            # print(f"LLM Judgment: {eval_text}")
                            
                            if "Judgment: Answer A" in eval_text:
                                winners.append(pair['gen_a'])
                                print(f"Winner: Answer A (is_correct: {pair['gen_a']['is_correct']}), Answer B (is_correct: {pair['gen_b']['is_correct']})")
                            elif "Judgment: Answer B" in eval_text:
                                winners.append(pair['gen_b'])
                                print(f"Winner: Answer B (is_correct: {pair['gen_b']['is_correct']}), Answer A (is_correct: {pair['gen_a']['is_correct']})")
                            else:
                                # Fallback: randomly select one
                                winner = random.choice([pair['gen_a'], pair['gen_b']])
                                winners.append(winner)
                                print(f"Fallback winner (random): {'Answer A' if winner == pair['gen_a'] else 'Answer B'} (is_correct: {winner['is_correct']})")
                            print("--- End Round Evaluation ---\n")
                
                # Add the skipped generation back for the next round
                if skipped_generation is not None:
                    winners.append(skipped_generation)
                    print(f"Adding skipped generation back for next round: {skipped_generation['generated_answer']}")
                
                # Update current_generations for next round
                current_generations = winners
                print(f"Round {round_num} complete. {len(current_generations)} generations advancing to next round.")
                
                round_num += 1
            
            # Tournament complete - we have the winner
            if current_generations:
                winner = current_generations[0]
                print(f"Tournament winner for idx {tournament['idx']}: {winner['generated_answer']} (is_correct: {winner['is_correct']})")
                print(f"Ground truth: {winner['answer']}")
                print("---- End of Tournament ----")
                data.append(winner)
            else:
                # Fallback: use the first generation if tournament failed
                data.append(tournament['generations'][0])
            
            # Update main progress bar
            tournament_pbar.update(1)
    
    # Close the main progress bar
    tournament_pbar.close()
    
    return data


def create_evaluation_tournaments(generations_by_idx):
    """
    Create tournament structures for evaluation.
    
    Args:
        generations_by_idx: Dictionary mapping idx to list of generations
    
    Returns:
        evaluation_tournaments: List of tournament structures
    """
    evaluation_tournaments = []
    
    # Add progress bar for tournament creation
    creation_pbar = tqdm.tqdm(
        total=len(generations_by_idx),
        desc="Creating tournament structures",
        unit="idx"
    )
    
    for idx_val, generations in generations_by_idx.items():
        first_gen = generations[0]

        if first_gen['qsubtype'] not in ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']:
            creation_pbar.update(1)
            continue
        
        # Group by generated_answer to find different answers
        answers_by_content = defaultdict(list)
        for g in generations:
            answer = normalize_answer(g['generated_answer'])
            answers_by_content[answer].append(g)
        
        if len(answers_by_content) <= 1:
            # All generations have the same answer, no tournament needed
            creation_pbar.update(1)
            continue
        
        # Create tournament structure for this idx
        answer_list = list(answers_by_content.keys())
        generations_list = [answers_by_content[answer][0] for answer in answer_list]  # Get first gen for each answer
        
        
        evaluation_tournaments.append({
            'idx': idx_val,
            'instruction': first_gen['instruction'],
            'format_instruction': first_gen['format_instruction'],
            'generations': generations_list
        })
        
        creation_pbar.update(1)
    
    # Close the creation progress bar
    creation_pbar.close()
    
    return evaluation_tournaments 
    

# def run_tournament_evaluation(evaluation_tournaments, args, df, tokenizer, eval_llm):
#     """
#     Run single inference evaluation for multiple generations per idx.
    
#     Args:
#         evaluation_tournaments: List of evaluation structures
#         args: Command line arguments
#         df: Original DataFrame with instructions
#         tokenizer: Tokenizer for prompt formatting
    
#     Returns:
#         data: List of selected generations (one per idx)
#     """
#     data = []
    
#     if not evaluation_tournaments:
#         return data
    
#     # Process evaluation tournaments in batches
#     eval_batch_size = args.batch_size
    
#     # Main progress bar for evaluations
#     evaluation_pbar = tqdm.tqdm(
#         total=len(evaluation_tournaments),
#         desc="Processing evaluations",
#         unit="evaluation"
#     )
    
#     for i in range(0, len(evaluation_tournaments), eval_batch_size):
#         batch = evaluation_tournaments[i:i+eval_batch_size]
        
#         # Process each evaluation
#         for evaluation in batch:
#             print(f"Running evaluation for idx {evaluation['idx']} with {len(evaluation['generations'])} generations")
            
#             # Create answer options with letters
#             answer_options = []
#             generation_map = {}
            
#             for j, generation in enumerate(evaluation['generations']):
#                 letter = chr(65 + j)  # A, B, C, D, etc.
#                 answer_options.append(f"{letter}. {generation['generated_answer']}")
#                 generation_map[letter] = generation
            
#             # Combine instruction and format_instruction
#             full_instruction = f"{evaluation['format_instruction']}\n\n{evaluation['instruction']}"
            
#             # Create the evaluation prompt using the imported COT evaluator format
#             system_prompt = cp.PROMPTS["BEST_ANSWER_SELECTION_SYSTEM_PROMPT"]
            
#             user_prompt = cp.PROMPTS["BEST_ANSWER_SELECTION_USER_PROMPT"].format(
#                 instruction=full_instruction,
#                 answer_options="\n\n".join(answer_options)
#             )
            
#             # Use tokenizer.apply_chat_template like in rl_data_collector.py
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]
            
#             full_prompt = tokenizer.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=True
#             )
            
#             # Generate evaluation
#             eval_sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
#             eval_outputs = eval_llm.generate([full_prompt], eval_sampling_params)
            
#             # Process evaluation result
#             eval_text = eval_outputs[0].outputs[0].text.strip()
            
#             print(f"\n--- Evaluation Result ---")
#             print(f"Answer options:")
#             for option in answer_options:
#                 print(f"  {option}")
#             print(f'Ground truth: {evaluation["generations"][0]["answer"]}')
#             print(f"LLM Judgment: {eval_text}")
            
#             # Extract judgment from evaluation - look for single letter selection
#             selected_answer = None
#             for letter in generation_map.keys():
#                 if f"Judgment: {letter}" in eval_text:
#                     selected_answer = generation_map[letter]
#                     break
            
#             if selected_answer is None:
#                 # Fallback: randomly select one
#                 selected_answer = random.choice(evaluation['generations'])
#                 print(f"Fallback winner (random): {selected_answer['generated_answer']} (is_correct: {selected_answer['is_correct']})")
#             else:
#                 print(f"Selected answer: {selected_answer['generated_answer']} (is_correct: {selected_answer['is_correct']})")
#                 print(f"Maximum BLEU score: {max([g['is_correct'] for g in evaluation['generations']])}")
            
#             print("--- End Evaluation ---\n")
            
#             # Add the selected generation to results
#             data.append(selected_answer)
            
#             # Update main progress bar
#             evaluation_pbar.update(1)
    
#     # Close the main progress bar
#     evaluation_pbar.close()
    
#     return data


# def create_evaluation_tournaments(generations_by_idx):
#     """
#     Create evaluation structures for single inference evaluation.
    
#     Args:
#         generations_by_idx: Dictionary mapping idx to list of generations
    
#     Returns:
#         evaluation_tournaments: List of evaluation structures
#     """
#     evaluation_tournaments = []
    
#     # Add progress bar for evaluation creation
#     creation_pbar = tqdm.tqdm(
#         total=len(generations_by_idx),
#         desc="Creating evaluation structures",
#         unit="idx"
#     )
    
#     for idx_val, generations in generations_by_idx.items():
#         first_gen = generations[0]

#         if first_gen['qsubtype'] not in ['CausalAnalysis', 'DescriptiveAnalysis', 'AnomalyDetection']:
#             creation_pbar.update(1)
#             continue
        
#         # Group by generated_answer to find different answers
#         answers_by_content = defaultdict(list)
#         for g in generations:
#             answer = normalize_answer(g['generated_answer'])
#             answers_by_content[answer].append(g)
        
#         # Create evaluation structure for this idx
#         answer_list = list(answers_by_content.keys())
#         generations_list = [answers_by_content[answer][0] for answer in answer_list]  # Get first gen for each answer
        
        
#         evaluation_tournaments.append({
#             'idx': idx_val,
#             'instruction': first_gen['instruction'],
#             'format_instruction': first_gen['format_instruction'],
#             'generations': generations_list
#         })
        
#         creation_pbar.update(1)
    
#     # Close the creation progress bar
#     creation_pbar.close()
    
#     return evaluation_tournaments 