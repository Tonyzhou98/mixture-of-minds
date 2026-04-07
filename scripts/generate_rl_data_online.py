import logging
import pandas as pd
import json
from tqdm import tqdm
from openai import OpenAI
import argparse
from transformers import AutoTokenizer
import os
import sys
# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)
from agents.rl_data_collector import RLDataCollector
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate RL data with specified model path.")
    parser.add_argument("--tokenizer_path", type=str, default="/your/path/to/your_fs/Llama/Llama-3_3-Nemotron-Super-49B-v1_5", help="Path to the tokenizer")
    parser.add_argument("--plan_model_port", type=str, default="a100-st2-p4de24xlarge-89:8000", help="Port of the plan model")
    parser.add_argument("--code_model_port", type=str, default="a100-st2-p4de24xlarge-89:8000", help="Port of the code model")
    parser.add_argument("--answer_model_port", type=str, default="a100-st2-p4de24xlarge-89:8000", help="Port of the answer model")
    parser.add_argument("--num_plans", type=int, default=8, help="Number of plans to generate")
    parser.add_argument("--num_codes_per_plan", type=int, default=4, help="Number of codes to generate per plan")
    parser.add_argument("--num_answers_per_code", type=int, default=1, help="Number of answers to generate per code")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=1000000, help="End index")
    parser.add_argument(
        "--output_fname_prefix",
        type=str,
        default="/your/path/to/your_fs/code/RankMind/datasets/rl_datasets/qwen3_32b_rl_data",
        help="Prefix for the output file name"
    )
    parser.add_argument(
        "--input_fname",
        type=str,
        default="/your/path/to/your_fs/code/RankMind/datasets/train_data/TableInstruct_qwen3_failed.jsonl",
        help="Path to the input file"
    )
    args = parser.parse_args()

    output_fname_prefix = args.output_fname_prefix
    input_fname = args.input_fname
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    plan_client = OpenAI(base_url=f"http://{args.plan_model_port}/v1", api_key="dummy")

    code_client = OpenAI(base_url=f"http://{args.code_model_port}/v1", api_key="dummy")

    answer_client = OpenAI(base_url=f"http://{args.answer_model_port}/v1", api_key="dummy")

    gen_params = {"temperature": 1.0, "max_tokens": 8192, "top_p": 1}
    print(f"Using temperature {gen_params['temperature']} for generation")
    data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_client, code_model=code_client, answer_model=answer_client, single_gen_params=gen_params, framework_type="openai")

    # Load data
    df = pd.read_json(input_fname, lines=True)
    logging.info(f"Loaded {len(df)} rows")

    # if the directory of output_fname_prefix does not exist, create it
    if not os.path.exists(os.path.dirname(output_fname_prefix)):
        os.makedirs(os.path.dirname(output_fname_prefix))
        print(f"Created directory {os.path.dirname(output_fname_prefix)}")
    
    # all_generations = []
    # # read all files with this output_fname_prefix and append to all_generations
    # for file in os.listdir(os.path.dirname(output_fname_prefix)):
    #     if file.startswith(os.path.basename(output_fname_prefix)):
    #         # read jsonl file
    #         with open(os.path.join(os.path.dirname(output_fname_prefix), file), 'r') as f:
    #             for line in f:
    #                 all_generations.append(json.loads(line))
    
    # # count the number of questions in all_generations
    # num_questions = len(set([d['idx'] for d in all_generations]))
    # print(f"Number of questions in all_generations: {num_questions}")

    # start from the previous idx
    batch_size = args.batch_size
    # print(f"Starting from idx {num_questions}")
    for idx in tqdm(range(args.start_idx, min(args.end_idx, len(df)), batch_size), desc="Processing batches"):
        batch_df = df.iloc[idx:idx+batch_size]
        output_fname = f"{output_fname_prefix}_{idx}_bsize{batch_size}.jsonl"
        data_collector.process_data(
            batch_df, output_fname,
            num_plans=args.num_plans, num_codes_per_plan=args.num_codes_per_plan, num_answers_per_code=args.num_answers_per_code
        )
