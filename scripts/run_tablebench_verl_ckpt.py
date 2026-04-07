import json
import asyncio
import io
import os
import sys
import torch
import argparse
import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from collections import defaultdict, Counter
from pathlib import Path
from transformers import AutoTokenizer
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)
from agents.rl_data_collector import RLDataCollector
from agents import cot_prompts as cp
from tournament_evaluator import run_tournament_evaluation, create_evaluation_tournaments
from evaluation.TableBench.eval.table_bench_custom_eval import *
from evaluation.TableBench.metrics.custom_em_metric import *

# sample command for running this script:
# srun --gres=gpu:3 --mem 128G -c 64 python3 run_tablebench_verl_ckpt.py --plan_model_path /your/path/to/your_fs/checkpoints/rankmind/rankmind_qwen3_8b_grpo_plan_agent/global_step_100/actor/huggingface --code_model_path /your/path/to/your_fs/Qwen/Qwen3-8B --answer_model_path /your/path/to/your_fs/Qwen/Qwen3-8B --output_dir /your/path/to/your_fs/RankMind/outputs/tablebench/qwen3_8b_ra_plan_grpo/ --output_file TableBench_qwen3_8b.jsonl --n_generations 1 --problem_type All


if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()


def self_consistency_selection(all_generations):
    generations_by_idx = defaultdict(list)
    for d in all_generations:
        idx_val = d.get('idx')
        if idx_val is not None:
            generations_by_idx[idx_val].append(d)

    # Step 3: For each idx, select the generation with the most frequent 'generated_answer'
    data = []
    for idx_val, generations in generations_by_idx.items():
        # Count occurrences of each generated_answer
        answer_counter = Counter()
        for g in generations:
            answer = normalize_answer(g.get('generated_answer', ''))
            answer_counter[answer] += 1
        if not answer_counter:
            # fallback: just take the first
            data.append(generations[0])
            continue
        # Find the most common answer(s)
        most_common_answer, _ = answer_counter.most_common(1)[0]
        # Find the first generation with the most common answer
        for g in generations:
            if normalize_answer(g.get('generated_answer', '')) == most_common_answer:
                data.append(g)
                break
    return data


def model_only_inference(all_prompts, tokenizer, model_path, sampling_params, args, data):

    # llm = sgl.Engine(model_path=model_path, tp_size=tensor_parallel_size, dp_size=int(torch.cuda.device_count() / tensor_parallel_size))
    llm = LLM(model=model_path, tensor_parallel_size=int(torch.cuda.device_count()), gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
    print("Initialized model. Using {} GPUs".format(torch.cuda.device_count()))

    cur_output_path = os.path.join(
        args.output_dir,
        args.output_file,
    )
    prompt_templates = []

    if "Qwen/" not in model_path and "Llama/" not in model_path:
        # not base model
        system_message = """Your response MUST be structured with two parts:
1.  **Thinking:** A detailed, step-by-step reasoning process enclosed in <think> tags. Decompose the problem, identify the necessary information from the table, perform any calculations, and explain your logic.
2.  **Answer:** The final, concise answer to the user's question, enclosed in <answer> tags.

Example structure:
<think> reasoning process here </think> <answer> [The final answer derived from your thinking] </answer>"""
    else:
        # base model
        system_message = "You are a helpful assistant."
    
    print(system_message)

    for i, prompt in enumerate(all_prompts):

        if system_message != "You are a helpful assistant.":
            question_steps = prompt.split("\n")
            processed_question_steps = []
            if "qwen3_failed" in model_path and "``python" not in prompt:
                for q in question_steps:
                    if (
                        "Read the table below in JSON format:" in q
                        or "[TABLE] " in q
                        or "{'columns': " in q
                        or "Question: " in q
                    ):
                        processed_question_steps.append(q)
                processed_question_steps = processed_question_steps[: -1] + [""] + [processed_question_steps[-1]]
                prompt = "\n".join(processed_question_steps)

            else:
                for q in question_steps:
                    if (
                        "Ensure the final answer" in q
                        or "Give the final answer" in q
                        or not q
                    ):
                        continue
                    else:
                        processed_question_steps.append(q)
                prompt = "\n".join(processed_question_steps)

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_templates.append(prompt)

    predictions = []
    batch_size = args.batch_size

    for i, d in enumerate(data):
        d['idx'] = i
    data = data * args.n_generations
    prompt_templates = prompt_templates * args.n_generations
    
    print(f"Generating for {len(prompt_templates)} rows")
    print(f"Using {args.n_generations} generations per row")
    
    for i in range(0, len(prompt_templates), batch_size):
        batch = prompt_templates[i:i+batch_size]
        outputs = llm.generate(batch, sampling_params)
        for j, gen in enumerate(outputs):
            # print(i+j)
            # print(gen['text'])
            # print("Answer: ", data[i+j]['answer'])
            # print("=" * 50)
            predictions.append(gen.outputs[0].text.strip())
        
        print("Example prediction:")
        print(predictions[-1])
    
    if args.n_generations > 1:
        for i, prediction in enumerate(predictions):
            data[i]['prediction'] = [prediction]
            parsed_answer = prediction.split('</think>')[-1].strip()
            parsed_answer = parsed_answer.split('\n')[-1]
            if 'Final Answer:' in parsed_answer:
                parsed_answer = parsed_answer.split('Final Answer:')[-1].strip()
            data[i]['generated_answer'] = parsed_answer
        
        data = self_consistency_selection(data)
    else:
        for i, prediction in enumerate(predictions):
            data[i]['prediction'] = [prediction]
    
    print(f"Generated {len(data)} rows")
    
    with open(cur_output_path, "w", encoding="utf-8") as f:
        for i in range(len(data)):
            f.write(json.dumps(data[i]) + "\n")

    with open(cur_output_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(cur_output_path.replace("jsonl", "json"), "w") as f:
        json.dump(data, f, indent=2)


def ra_framework_inference(tokenizer, plan_model_path, code_model_path, answer_model_path, sampling_params, args):

    tensor_parallel_size = args.tp_size

    if plan_model_path == code_model_path == answer_model_path:

        if "gpt" in plan_model_path.lower():
            if args.n_generations == 1:
                sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
            else:
                sampling_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
            llm = LLM(model=plan_model_path, tensor_parallel_size=int(torch.cuda.device_count()), gpu_memory_utilization=0.95)
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=llm, code_model=llm, answer_model=llm, single_gen_params=sampling_params, framework_type="vllm", model_name="gpt-oss")
            print("Initialized GPT-OSS VLLM model. Using {} GPUs".format(torch.cuda.device_count()) + f" for tensor parallel {int(torch.cuda.device_count())} and for data parallel 1")
        else:
            # if the three models are the same, use the same model for all three
            # llm = sgl.Engine(model_path=plan_model_path, tp_size=tensor_parallel_size, dp_size=int(torch.cuda.device_count() / tensor_parallel_size), trust_remote_code=True)
            if args.n_generations == 1:
                sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
            else:
                sampling_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
            llm = LLM(model=plan_model_path, tensor_parallel_size=int(torch.cuda.device_count()), gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=llm, code_model=llm, answer_model=llm, single_gen_params=sampling_params, framework_type="vllm")
            print("Initialized VLLM model. Using {} GPUs".format(torch.cuda.device_count()) + f" for tensor parallel {tensor_parallel_size} and for data parallel " + str(int(torch.cuda.device_count() / tensor_parallel_size)))
    else:
        if args.n_generations == 1:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
        else:
            sampling_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
        

        if tensor_parallel_size != 0:
            if tensor_parallel_size == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                plan_model = LLM(model=plan_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                code_model = LLM(model=code_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                os.environ["CUDA_VISIBLE_DEVICES"] = "2"
                answer_model = LLM(model=answer_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                if args.evaluate_model_path is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
                    eval_model = LLM(model=args.evaluate_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                else:
                    eval_model = None
            elif tensor_parallel_size == 2:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
                plan_model = LLM(model=plan_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
                code_model = LLM(model=code_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
                answer_model = LLM(model=answer_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                if args.evaluate_model_path is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
                    eval_model = LLM(model=args.evaluate_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                else:
                    eval_model = None
            elif tensor_parallel_size == 4:
                if plan_model_path != code_model_path and code_model_path != answer_model_path:
                    raise ValueError("No Enough GPUs and terminate for now")
                elif plan_model_path != code_model_path and code_model_path == answer_model_path:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                    plan_model = LLM(model=plan_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
                    code_model = LLM(model=code_model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
                    # share the same model for answer model
                    answer_model = code_model
                else:
                    raise ValueError("No Enough GPUs and no such case")

            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_model, code_model=code_model, answer_model=answer_model, single_gen_params=sampling_params, framework_type="vllm")
            print("Initialized three VLLM models. Using {} GPUs".format(torch.cuda.device_count()) + f" for tensor parallel {tensor_parallel_size} and for data parallel 1")
        else:
            from openai import OpenAI
            plan_client = OpenAI(base_url=f"http://{plan_model_path}/v1", api_key="dummy")
            code_client = OpenAI(base_url=f"http://{code_model_path}/v1", api_key="dummy")
            answer_client = OpenAI(base_url=f"http://{answer_model_path}/v1", api_key="dummy")
            gen_params = {"temperature": sampling_params.temperature, "max_tokens": sampling_params.max_tokens, "top_p": sampling_params.top_p}
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_client, code_model=code_client, answer_model=answer_client, single_gen_params=gen_params, framework_type="openai")
            print(f"Initialized three OpenAI models with port {plan_model_path}, {code_model_path}, {answer_model_path}")
        

    cur_output_path = os.path.join(
        args.output_dir,
        args.output_file,
    )

    all_generations = []
    # if args.output_dir does not exist, create one and if so, empty it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        # read all files ending with .jsonl in the directory and only generate new data and keep the old data
        directory = Path(args.output_dir)
        jsonl_files = list(directory.glob("*.jsonl"))
        
        print(f"Found {len(jsonl_files)} JSONL files:")
        for file_path in jsonl_files:
            print(f"  - {file_path.name}")

        # Read existing generations into a list
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        all_generations.append(d)

    print(f"Found {len(all_generations)} generations")

    df = pd.read_json(args.jsonl_path, lines=True)
    print(f"Loaded {len(df)} rows")

    if "finqa" in args.jsonl_path:
        df["instruction"] = "## Context Before Table:" + df["prompt"].str.split("## Context Before Table:").str[-1]
        df['qtype'] = "NumericalReasoning"

    def extract_tablebench_format_instruction(instruction):
        # use the sentence betweeen [Answer Format] and "Give the final answer" as the format instruction
        format_instruction = "[Answer Format]\n" + instruction.split("[Answer Format]\n")[-1].split("Give the final answer")[0]
        return format_instruction
    
    def extract_finqa_format_instruction(instruction):
        # use the sentence betweeen [Answer Format] and "Give the final answer" as the format instruction
        format_instruction = "[Answer Format]\n" + instruction.split("[Answer Format]\n")[-1]
        return format_instruction

    def clean_tablebench_instruction(instruction):
        sentences = instruction.split("\n")
        processed_question_steps = []
        for q in sentences:
            if (
                "Read the table below in JSON format:" in q
                or "[TABLE] " in q
                or "{'columns': " in q
                or "Question: " in q
            ):
                processed_question_steps.append(q)
        processed_question_steps = processed_question_steps[: -1] + [""] + [processed_question_steps[-1]]
        instruction = "\n".join(processed_question_steps)
        return instruction
    
    def clean_finqa_instruction(instruction):
        instruction = instruction.split('\n\n## Answer Requirements:')[0]
        return instruction

    # Apply the function to the 'instruction' column and create a new 'format_instruction' column

    if "TableBench" in args.jsonl_path:
        df['format_instruction'] = df['instruction'].apply(extract_tablebench_format_instruction)
        df['instruction'] = df['instruction'].apply(clean_tablebench_instruction)
    else:
        df['format_instruction'] = df['instruction'].apply(extract_finqa_format_instruction)
        df['instruction'] = df['instruction'].apply(clean_finqa_instruction)

    # print(df)

    if args.problem_type != "All":
        df = df[df['qtype'] == args.problem_type]
    
    print(f"Loaded {len(df)} rows for {args.problem_type}")

    # only generate for not existing ids
    df = df[~df['id'].isin([g['id'] for g in all_generations])]
    print(f"Generating for {len(df)} rows")

    start_idx = len(list(set([g['id'] for g in all_generations])))
    print(f"Start generating from {start_idx}")

    batch_size = args.batch_size
    for idx in tqdm.tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[idx:idx+batch_size]
        output_fname = cur_output_path.replace(".jsonl", f"_{start_idx + idx}_bsize{batch_size}.jsonl")
        print(f"Generating for {len(batch_df)} rows and saving to {output_fname}")
        data_collector.process_data(
            batch_df, output_fname,
            num_plans=args.n_generations, 
            num_codes_per_plan=1, 
            code_iterations=args.code_iterations
        )


    directory = Path(args.output_dir)
    jsonl_files = list(directory.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for file_path in jsonl_files:
        print(f"  - {file_path.name}")


    # After generating new data, we need to aggregate the generations by idx and select the generation with the most frequent 'generated_answer'
    # Step 1: Read all generations into a list
    all_generations = []
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    all_generations.append(d)

    # if args.evaluate_model_path is None, we use the self-consistency selection
    if args.evaluate_model_path is None or args.n_generations == 1: 
        # Step 2: Group generations by 'idx'
        data = self_consistency_selection(all_generations)
        
    else:
        # If the evaluate model path is provided, we use it to evaluate the generations.
        # We extract the generations with different answers, pair them with the original data,
        # and use the evaluation model to select the best generation for each idx.
        
        # Step 1: Group generations by 'idx'
        generations_by_idx = defaultdict(list)
        for d in all_generations:
            idx_val = d.get('idx')
            if idx_val is not None:
                generations_by_idx[idx_val].append(d)
        
        # # Step 2: Create tournament structures for evaluation
        evaluation_tournaments = create_evaluation_tournaments(generations_by_idx)
        print(f"Created {len(evaluation_tournaments)} evaluation tournaments")

        data = []
        
        # # Step 3: Run tournament-style evaluation for each idx
        if evaluation_tournaments:
            tournament_winners = run_tournament_evaluation(evaluation_tournaments, args, df, tokenizer, eval_model)
            data.extend(tournament_winners)
        
        # Step 4: For idxs that didn't have evaluation tournaments, do the self-consistency selection
        evaluated_idxs = set(d['idx'] for d in data)
        subset_generations = [g for g in all_generations if g['idx'] not in evaluated_idxs]
        data.extend(self_consistency_selection(subset_generations))

    
    print(f"Loaded {len(data)} rows")

    for d in data:
        d['prediction'] = [d['generated_answer']]
    
    with open(cur_output_path.replace("jsonl", "json"), "w") as f:
        json.dump(data, f, indent=2)


    # if args.generation_part == "all":
    #     all_generations = []
    #     # if args.output_dir does not exist, create one and if so, empty it
    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir)
    #     else:
    #         # read all files ending with .jsonl in the directory and only generate new data and keep the old data
    #         directory = Path(args.output_dir)
    #         jsonl_files = list(directory.glob("*.jsonl"))
            
    #         print(f"Found {len(jsonl_files)} JSONL files:")
    #         for file_path in jsonl_files:
    #             print(f"  - {file_path.name}")

    #         # Read existing generations into a list
    #         for file_path in jsonl_files:
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 for line in f:
    #                     line = line.strip()
    #                     if line:
    #                         d = json.loads(line)
    #                         all_generations.append(d)

    #     print(f"Found {len(all_generations)} generations")

    #     df = pd.read_json(args.jsonl_path, lines=True)
    #     print(f"Loaded {len(df)} rows")

    #     # Apply the function to the 'instruction' column and create a new 'format_instruction' column
    #     df['format_instruction'] = df['instruction'].apply(extract_format_instruction)
    #     df['instruction'] = df['instruction'].apply(clean_instruction)

    #     # print(df)

    #     if args.problem_type != "All":
    #         df = df[df['qtype'] == args.problem_type]
        
    #     print(f"Loaded {len(df)} rows for {args.problem_type}")

    #     # only generate for not existing ids
    #     df = df[~df['id'].isin([g['id'] for g in all_generations])]
    #     print(f"Generating for {len(df)} rows")

    #     start_idx = len(list(set([g['id'] for g in all_generations])))
    #     print(f"Start generating from {start_idx}")

    #     llm = sgl.Engine(model_path=args.model_path, tp_size=tensor_parallel_size, dp_size=int(torch.cuda.device_count() / tensor_parallel_size))
    #     data_collector = RLDataCollector(tokenizer=tokenizer, model=llm, single_gen_params=sampling_params, framework_type="sglang")
    #     print("Initialized SGLang model. Using {} GPUs".format(torch.cuda.device_count()) + f" for tensor parallel {tensor_parallel_size} and for data parallel " + str(int(torch.cuda.device_count() / tensor_parallel_size)))
        
    #     batch_size = args.batch_size
    #     for idx in tqdm.tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    #         batch_df = df.iloc[idx:idx+batch_size]
    #         output_fname = cur_output_path.replace(".jsonl", f"_{start_idx + idx}_bsize{batch_size}.jsonl")
    #         print(f"Generating for {len(batch_df)} rows and saving to {output_fname}")
    #         data_collector.process_data(
    #             batch_df, output_fname,
    #             num_plans=1, num_codes_per_plan=args.n_generations,
    #         )
        
    #     self_consistency_selection(args, cur_output_path)

    # else:
    #     sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
    #     llm = LLM(model=args.model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.95)
    #     data_collector = RLDataCollector(tokenizer=tokenizer, model=llm, single_gen_params=sampling_params, framework_type="vllm")
    #     print("Initialized VLLM model. Using {} GPUs".format(torch.cuda.device_count()) + f" for tensor parallel {tensor_parallel_size} and for data parallel 1")

    #     df = pd.read_json(args.jsonl_path, lines=True)

    #     if args.generation_part == "plan":
    #         data_collector.generate_plans_only(df, cur_output_path.replace(".jsonl", "_plans.jsonl"), num_plans=1)
    #     elif args.generation_part == "code":
    #         data_collector.generate_codes_only(df, plans_file=cur_output_path.replace(".jsonl", "_plans.jsonl"), output_file=cur_output_path.replace(".jsonl", "_codes.jsonl"), num_codes_per_plan=args.n_generations)
    #     elif args.generation_part == "answer":
    #         data_collector.generate_answers_only(df, codes_file=cur_output_path.replace(".jsonl", "_codes.jsonl"), output_file=cur_output_path.replace(".jsonl", "_answers.jsonl"))
    #         self_consistency_selection(args, cur_output_path, suffix="*_answers.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run TableBench with a specific model and data using vLLM.")
    parser.add_argument('--plan_model_path', type=str, required=True, help='Path to the plan model checkpoint directory')
    parser.add_argument('--code_model_path', type=str, required=True, help='Path to the code model checkpoint directory')
    parser.add_argument('--answer_model_path', type=str, required=True, help='Path to the answer model checkpoint directory')
    parser.add_argument('--evaluate_model_path', type=str, required=False, default=None, help='Path to the evaluate model checkpoint directory')
    parser.add_argument('--jsonl_path', type=str, default="/your/path/to/your_fs/RankMind/datasets/tablebench/TableBench_DP.jsonl", help='Path to the input JSONL file')
    parser.add_argument('--output_dir', type=str, default="/your/path/to/your_fs/RankMind/outputs/tablebench/qwen3_8b_grpo", help='Directory to save the output files')
    parser.add_argument('--output_file', type=str, default="TableBench_qwen3_8b_grpo_table_instruct_predictions.jsonl", help='Output JSONL file name')
    parser.add_argument('--problem_type', type=str, default="All", help='which type of data to evaluate')
    parser.add_argument('--generation_framework', type=str, default="rankagent", choices=["rankagent", "model_only"], help='generation with model only or rankagent framework')
    parser.add_argument('--n_generations', type=int, default=1, help='Number of code generations to use for self-consistency')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generating data')
    parser.add_argument('--tp_size', type=int, default=2, help='Tensor parallel size')
    parser.add_argument('--code_iterations', type=int, default=1, help='Code generation max iterations')
    parser.add_argument('--tokenizer_path', type=str, default="/your/path/to/your_fs/Llama/DeepSeek-R1-Distill-Llama-70B", help='Tokenizer path')
    args = parser.parse_args()

    if args.evaluate_model_path is not None:
        args.output_dir = os.path.join(args.output_dir, f"n_generations_{args.n_generations}_evaluate")
    else:
        if args.code_iterations > 1:
            args.output_dir = os.path.join(args.output_dir, f"n_generations_{args.n_generations}_code_iterations_{args.code_iterations}")
        else:
            args.output_dir = os.path.join(args.output_dir, f"n_generations_{args.n_generations}")

    # if plan_model_path does not exist, use the tokenizer_path to load the tokenizer
    if os.path.exists(args.plan_model_path):
        tokenizer = AutoTokenizer.from_pretrained(args.plan_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)


    if args.n_generations == 1:
        # sampling_params = {"temperature": 0.0, "max_new_tokens": 8192, "stop_token_ids": [tokenizer.eos_token_id]}
        sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, top_p=1)
    else:
        # sampling_params = {"temperature": 1.0, "max_new_tokens": 8192, "stop_token_ids": [tokenizer.eos_token_id]}
        sampling_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)

    data = []
    with open(args.jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
        
    if "finqa" in args.jsonl_path:
        for d in data:
            d['instruction'] = "## Context Before Table:" +d['prompt'].split("## Context Before Table:")[-1]
            d['qtype'] = "NumericalReasoning"

    all_prompts = []
    for d in data:
        prompt = d['instruction']
        all_prompts.append(prompt)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    

    if args.generation_framework == "rankagent":
        ra_framework_inference(tokenizer, args.plan_model_path, args.code_model_path, args.answer_model_path, sampling_params, args)
    elif args.generation_framework == "model_only":
        model_only_inference(all_prompts, tokenizer, args.plan_model_path, sampling_params, args, data)
    else:
        raise ValueError(f"Invalid generation framework: {args.generation_framework}")


if __name__ == "__main__":
    main()