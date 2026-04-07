"""
Preprocess the Rankmind training dataset to parquet format
"""

import argparse
import datasets
import random
import numpy as np

# how to use this script
# python training_data_tool_multiturn.py --data_path ./train_data/TableInstruct_qwen3_failed.jsonl


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

    system_message = """You are an expert data analyst specializing in tabular data analysis. Your task is to answer questions about structured tables using Python and the pandas library.
Reasoning step by step about how you can write the code to answer the question then write the complete code with necessary imports and inputs to answer this question.

Put your code as the argument of `code_interpreter` tool with the format:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": {the_code_you_write}, "executes": "True"}}
</tool_call>

Here is an example of the code you should write:
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "import pandas as pd\nimport numpy as np\ndata={the table mentioned in the question}\ndf = pd.DataFrame(data[\'data\'], columns=data[\'columns\'])\ndf.columns = ["Season", "Team", "League", "Regular_Header", "GP_Regular", "G_Regular", "A_Regular", "Pts_Regular", "PIM_Regular", "Playoff_Header", "GP_Playoff", "G_Playoff", "A_Playoff", "Pts_Playoff", "PIM_Playoff"]\ndf = df[~df[\'Season\'].str.contains(\'totals\')]\ndf.replace(\'—\', np.nan, inplace=True)\ndf.apply(pd.to_numeric, errors=\'coerce\')\n\n# Calculate correlation between regular season GP and Pts\ncorrelation = df[[\'GP_Regular\', \'Pts_Regular\']].corr().iloc[0,1]\nprint(result)\n", "executes": "True"}}
</tool_call>

You should use the `code_interpreter` tool to execute the code, before generating the final answer at least once and refine your answer if necessary.
Only after getting the result from `code_interpreter` tool, you canput your final answer within <answer> and </answer> tags. {answer_format}
"""


    def make_map_fn_qwen3(split):
        def process_fn(example, idx):

            question_raw = example.pop("instruction")

            question = question_raw

            answer_raw = example.pop("answer")

            solution = answer_raw

            answer_format = example.pop("format_instruction")

            data = {
                "data_source": args.data_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_message.replace("{answer_format}", answer_format),
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
                    "qtype": example.pop("qtype"),
                    "qsubtype": example.pop("qsubtype"),
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "code_interpreter": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
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


    train_dataset = train_dataset.filter(lambda x: x["qtype"] in ["FactChecking", "NumericalReasoning", "DataAnalysis"])
    test_dataset = test_dataset.filter(lambda x: x["qtype"] in ["FactChecking", "NumericalReasoning", "DataAnalysis"])
    print(f"train_dataset size: {len(train_dataset)}")
    print(f"test_dataset size: {len(test_dataset)}")
    
    train_dataset = train_dataset.map(function=make_map_fn_qwen3("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn_qwen3("test"), with_indices=True)


    # Save to parquet
    parquet_base = args.data_path.replace(".jsonl", "")
    train_dataset.to_parquet(parquet_base + "_multiturn_tool_train.parquet")
    test_dataset.to_parquet(parquet_base + "_multiturn_tool_test.parquet")

    # save to jsonl
    train_dataset.to_json(parquet_base + "_multiturn_tool_train.jsonl", orient="records", lines=True)
    test_dataset.to_json(parquet_base + "_multiturn_tool_test.jsonl", orient="records", lines=True)
