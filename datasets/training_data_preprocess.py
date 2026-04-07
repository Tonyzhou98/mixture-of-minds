"""
Preprocess the Rankmind training dataset to parquet format
"""

import argparse
import datasets
import random
import numpy as np

# how to use this script
# python training_data_preprocess.py --data_path ./train_data/table_instruct_scout_failed.jsonl
# python training_data_preprocess.py --data_path ./train_data/TableInstruct_qwen3_failed.jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./train_data")
    parser.add_argument("--data_path", default="./train_data/table_instruct_scout_failed.jsonl")

    args = parser.parse_args()

    # Load the JSONL file as a HuggingFace dataset
    train_dataset = datasets.load_dataset(
        "json",
        data_files=args.data_path,
        split="train"
    )

    system_message = """Your response MUST be structured with two parts:
1.  **Thinking:** A detailed, step-by-step reasoning process enclosed in <think> tags. Decompose the problem, identify the necessary information from the table, perform any calculations, and explain your logic.
2.  **Answer:** The final, concise answer to the user's question, enclosed in <answer> tags.

Example structure:
<think> reasoning process here </think> <answer> [The final answer derived from your thinking] </answer>"""

    # add a row to each data item that represents a unique id
    def make_map_fn_scout(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw

            answer_raw = example.pop("solution")
            # Use the solution directly as ground truth
            solution = answer_raw
            data = {
                "data_source": args.data_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "table_analysis",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    def make_map_fn_qwen3(split):
        def process_fn(example, idx):

            question_raw = example.pop("instruction")

            question = question_raw

            answer_raw = example.pop("answer")

            solution = answer_raw

            data = {
                "data_source": args.data_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "table_analysis",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Shuffle and split the dataset into train and test (50 samples for test)
    total_size = len(train_dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    test_size = min(50, total_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    test_dataset = train_dataset.select(test_indices.tolist())
    train_dataset = train_dataset.select(train_indices.tolist())

    if "scout" in args.data_path:
        train_dataset = train_dataset.map(function=make_map_fn_scout("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn_scout("test"), with_indices=True)
    elif "qwen3" in args.data_path:
        # only keep qtype == 'FactChecking' and qtype == 'NumericalReasoning'
        train_dataset = train_dataset.filter(lambda x: x["qtype"] in ["FactChecking", "NumericalReasoning"])
        test_dataset = test_dataset.filter(lambda x: x["qtype"] in ["FactChecking", "NumericalReasoning"])
        print(f"train_dataset size: {len(train_dataset)}")
        print(f"test_dataset size: {len(test_dataset)}")
        
        train_dataset = train_dataset.map(function=make_map_fn_qwen3("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn_qwen3("test"), with_indices=True)
    else:
        raise ValueError(f"Unknown dataset: {args.data_path}")


    # Save to parquet
    parquet_base = args.data_path.replace(".jsonl", "")
    train_dataset.to_parquet(parquet_base + "_train.parquet")
    test_dataset.to_parquet(parquet_base + "_test.parquet")
