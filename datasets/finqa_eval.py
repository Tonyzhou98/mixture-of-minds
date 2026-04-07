import os
import json
import sys
import argparse
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify import parse, verify


# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)

from evaluation.TableBench.eval.table_bench_custom_eval import *
from evaluation.TableBench.metrics.custom_em_metric import *

# round the number to 1 decimal place
def round_number(number):
    return round(number, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Read all JSONL files from a directory and analyze plan performance"
    )
    # optional argument
    parser.add_argument(
        "--file_path",
        default="/your/path/to/your_fs/RankMind/outputs/finqa/qwen3_8b_base/n_generations_1/finqa_qwen3_8b.json",
        type=str,
    )
    args = parser.parse_args()


    # load json file
    with open(args.file_path, 'r') as f:
        data = json.load(f)

    total_number = 0
    correct_number = 0
    for d in data:
        if d['answer']:
            total_number += 1
            prediction = d['prediction'][0]
            answer = d['answer']
            if "Final Answer: " in prediction:
                prediction = prediction.split("Final Answer: ")[-1].strip()
            
            if "<answer>" in prediction:
                prediction = prediction.split("<answer>")[1].strip()
            if "</answer>" in prediction:
                prediction = prediction.split("</answer>")[0].strip()
            
            prediction = prediction.replace("%", "").replace("$", "")
            answer = answer.replace("%", "").replace("$", "")

            try:
                prediction = round_number(float(prediction))
                answer = round_number(float(answer))
            except:
                continue
            
            correct = compute_em([str(answer)], [str(prediction)])
            correct_number += correct

            # print(f"Prediction: {prediction}, Answer: {answer}, Score: {correct}")
            # print("--------------------------------")
            # if total_number == 50:
            #     break
            
    
    print(f"Accuracy: {correct_number / total_number}")

if __name__ == "__main__":
    main()