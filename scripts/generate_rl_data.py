import asyncio
import io
import logging
import pandas as pd
import json
import torch
from transformers import (
    AutoTokenizer
)
from vllm import LLM
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
# import sglang as sgl
# from sglang.test.test_utils import is_in_ci
import argparse

import os
import sys
# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)
from agents.rl_data_collector import RLDataCollector
from inference.model_list import (
    CLOSED_SOURCE_MODELS,
)
logger = logging.getLogger(__name__)


# if is_in_ci():
#     import patch
# else:
#     import nest_asyncio

#     nest_asyncio.apply()

# model = LLM(
#     model=model_path,
#     tensor_parallel_size=2, # https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html
#     data_parallel_size=4,
#     generation_config="vllm",
#     download_dir="/your/path/to/your_fs/.cache/huggingface",
#     gpu_memory_utilization=0.95,
#     distributed_executor_backend="mp",
#     trust_remote_code=True
# )


# gen_params = SamplingParams(
#     temperature=1,  # Add some randomness for diverse generations
#     top_p=0.95,
#     max_tokens=16384,
#     stop_token_ids=[tokenizer.eos_token_id]
# )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate RL data with specified model path.")
    parser.add_argument("--plan_model_path", type=str, default="/your/path/to/your_fs/models/Qwen/Qwen3-32B", help="Path to the model checkpoint directory")
    parser.add_argument("--code_model_path", type=str, default="/your/path/to/your_fs/models/Qwen/Qwen3-32B", help="Path to the model checkpoint directory")
    parser.add_argument("--answer_model_path", type=str, default="/your/path/to/your_fs/models/Qwen/Qwen3-32B", help="Path to the model checkpoint directory")
    parser.add_argument("--num_plans", type=int, default=8, help="Number of plans to generate")
    parser.add_argument("--num_codes_per_plan", type=int, default=4, help="Number of codes to generate per plan")
    parser.add_argument("--num_answers_per_code", type=int, default=1, help="Number of answers to generate per code")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size")
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
        default="/your/path/to/your_fs/code/RankMind/datasets/train_data/TableInstruct.jsonl",
        help="Path to the input file"
    )
    args = parser.parse_args()

    output_fname_prefix = args.output_fname_prefix
    input_fname = args.input_fname

    tp_size = args.tp_size

    # if args.plan_model_path == args.code_model_path == args.answer_model_path:
    #     model_path = args.plan_model_path
    #     model = sgl.Engine(model_path=model_path, tp_size=tp_size, dp_size=int(torch.cuda.device_count() / tp_size))
    #     gen_params = {"temperature": 1, "max_new_tokens": 8192, "stop_token_ids": [tokenizer.eos_token_id]}

    #     data_collector = RLDataCollector(tokenizer, model, model, model, gen_params)
    # else:
    #     if tp_size == 1:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #         plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #         code_model = LLM(model=args.code_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #         answer_model = LLM(model=args.answer_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    #         plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    #         code_model = LLM(model=args.code_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    #         answer_model = LLM(model=args.answer_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95)
        
    #     os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    #     gen_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
    #     print(f"Using temperature {gen_params.temperature} for generation")
    #     data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_model, code_model=code_model, answer_model=answer_model, single_gen_params=gen_params, framework_type="vllm")

    if args.plan_model_path in CLOSED_SOURCE_MODELS:
        gen_params = {"temperature": 1.0, "top_p": 1}
        data_collector = RLDataCollector(tokenizer=None, plan_model=args.plan_model_path, code_model=args.code_model_path, answer_model=args.answer_model_path, single_gen_params=gen_params, framework_type="openrouter")
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plan_model_path)
        tokenizer.pad_token = tokenizer.eos_token

        if args.plan_model_path == args.code_model_path == args.answer_model_path:
            plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=int(torch.cuda.device_count()), data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
            gen_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_model, code_model=plan_model, answer_model=plan_model, single_gen_params=gen_params, framework_type="vllm")
        else:
            if tp_size == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                code_model = LLM(model=args.code_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = "2"
                answer_model = LLM(model=args.answer_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
            elif tp_size == 2:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
                plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
                code_model = LLM(model=args.code_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
                answer_model = LLM(model=args.answer_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
            elif tp_size == 4:
                if args.plan_model_path != args.code_model_path and args.code_model_path != args.answer_model_path:
                    raise ValueError("No Enough GPUs and terminate for now")
                elif args.plan_model_path != args.code_model_path and args.code_model_path == args.answer_model_path:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                    plan_model = LLM(model=args.plan_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
                    code_model = LLM(model=args.code_model_path, tensor_parallel_size=tp_size, data_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True, enforce_eager=True)
                    answer_model = code_model
                else:
                    raise ValueError("No Enough GPUs and no such case")
            else:
                raise ValueError("No Enough GPUs and no such case")
                    
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            gen_params = SamplingParams(temperature=1.0, max_tokens=8192, top_p=1)
            print(f"Using temperature {gen_params.temperature} for generation")
            data_collector = RLDataCollector(tokenizer=tokenizer, plan_model=plan_model, code_model=code_model, answer_model=answer_model, single_gen_params=gen_params, framework_type="vllm")

    # Load data
    df = pd.read_json(input_fname, lines=True)
    logging.info(f"Loaded {len(df)} rows")

    # if the directory of output_fname_prefix does not exist, create it
    if not os.path.exists(os.path.dirname(output_fname_prefix)):
        os.makedirs(os.path.dirname(output_fname_prefix))
        print(f"Created directory {os.path.dirname(output_fname_prefix)}")
    
    # all_generations = []
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

    end_idx = min(args.end_idx, len(df))
    print(f"Starting from idx {args.start_idx} to {end_idx}")
    # for idx in tqdm(range(num_questions, len(df), batch_size), desc="Processing batches"):
    for idx in tqdm(range(args.start_idx, end_idx, batch_size), desc="Processing batches"):
        batch_df = df.iloc[idx:idx+batch_size]
        output_fname = f"{output_fname_prefix}_{idx}_bsize{batch_size}.jsonl"
        data_collector.process_data(
            batch_df, output_fname,
            num_plans=args.num_plans, num_codes_per_plan=args.num_codes_per_plan, num_answers_per_code=args.num_answers_per_code
        )
