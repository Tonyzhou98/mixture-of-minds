"""
Reward function for table reasoning tasks
"""

import os
import sys
import re
import ast
import traceback
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add this directory to the beginning of sys.path
sys.path.append(current_dir)

from evaluation.TableBench.eval.table_bench_custom_eval import *
from evaluation.TableBench.metrics.custom_em_metric import *
from agents.tool import PythonREPLTool


def extract_python_code(text):
    """Extract Python code from text"""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def execute_code_safe(code: str) -> str:
    """Execute code safely and return output"""
    try:
        python_repl = PythonREPLTool()
        # Execute the code
        result = python_repl.execute(code)
        return str(result)
    except Exception as e:
        return f"Error during execution: {str(e)}\n{traceback.format_exc()}"


def extract_index_from_slice(slice_node):
    """Safely extract index from slice across Python versions."""
    if isinstance(slice_node, ast.Constant):  # Python 3.8+
        return slice_node.value
    elif isinstance(slice_node, ast.Index):  # Python <3.9
        return getattr(slice_node.value, 'value', None)
    elif hasattr(slice_node, 'value'):
        return getattr(slice_node.value, 'value', None)
    return None


def extract_operations(code: str):
    semantics = set()

    try:
        tree = ast.parse(code)
    except Exception as e:
        print("AST parsing failed:", e)
        return sorted(semantics)

    for node in ast.walk(tree):
        # --- DataFrame creation ---
        try:
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'DataFrame':
                    semantics.add('create_dataframe')
                    for kw in node.keywords:
                        if kw.arg == 'columns':
                            semantics.add('assign_column_names')
                if hasattr(node.func, 'attr') and node.func.attr == 'to_numeric':
                    semantics.add('convert_to_numeric')
                    for kw in node.keywords:
                        if kw.arg == 'errors' and getattr(kw.value, 'value', None) == 'coerce':
                            semantics.add('handle_nulls')
        except Exception as e:
            print("DataFrame block error:", e)

        # --- Column selection ---
        try:
            if isinstance(node, ast.Subscript):
                index_val = extract_index_from_slice(node.slice)
                if isinstance(index_val, str):
                    semantics.add('select_column')
                elif isinstance(index_val, list):
                    semantics.add('select_columns')
        except Exception as e:
            print("Column selection block error:", e)

        # --- Filtering and comparisons ---
        try:
            if isinstance(node, ast.Compare):
                is_lhs_column = isinstance(node.left, ast.Subscript)
                for comp in node.comparators:
                    if is_lhs_column:
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, (str, int, float)):
                            semantics.add('filter_row_by_value')
                        elif isinstance(comp, ast.Str):  # Python < 3.8
                            semantics.add('filter_row_by_value')
                    if isinstance(comp, ast.Subscript):
                        semantics.add('compare_columns')
                for op in node.ops:
                    if isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.NotEq)):
                        if is_lhs_column:
                            semantics.add('filter_row_by_condition')
        except Exception as e:
            print("Comparison block error:", e)

        # --- Method chains (chained calls) ---
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr == 'sort_values':
                    semantics.add('sort_by_column')
                elif attr == 'groupby':
                    semantics.add('groupby')
                elif attr in ['sum', 'mean', 'count']:
                    semantics.add(f'aggregate_{attr}')
                elif attr == 'astype':
                    semantics.add('change_dtype')
                elif attr == 'replace':
                    semantics.add('replace_values')
                    semantics.add('string_cleanup')
                elif attr == 'dropna':
                    semantics.add('handle_nulls')
                elif attr == 'corr':
                    semantics.add('compute_correlation')
        except Exception as e:
            print("Method chain block error:", e)

        # --- Math operations ---
        try:
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Sub):
                    semantics.add('compute_difference')
                elif isinstance(node.op, ast.Add):
                    semantics.add('add_columns')
        except Exception as e:
            print("Math operation block error:", e)

        # --- Top/bottom rows ---
        try:
            if isinstance(node, ast.Subscript):
                index_val = extract_index_from_slice(node.slice)
                if index_val == 0:
                    semantics.add('select_top_row')
                elif index_val == -1:
                    semantics.add('select_bottom_row')
        except Exception as e:
            print("Row selection block error:", e)

        # --- Type casting ---
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['float', 'int', 'str']:
                    semantics.add(f'convert_to_{node.func.id}')
        except Exception as e:
            print("Type casting block error:", e)

        # --- Print and output ---
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'print':
                    semantics.add('print_result')
            if isinstance(node, ast.JoinedStr):
                semantics.add('format_output')
        except Exception as e:
            print("Output block error:", e)

        # --- Answer assignment ---
        try:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in ['answer', 'result', 'summary']:
                        semantics.add('assign_answer_text')
        except Exception as e:
            print("Assignment block error:", e)

    return sorted(semantics)


def compute_rouge(reference_list, prediction_list):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference, prediction in zip(reference_list, prediction_list):
        scores = scorer.score(reference, prediction)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    rouge_l_score = result['rougeL'].mid.fmeasure # or precision/recall as needed
    return rouge_l_score


def compute_bleu(references, hypotheses):
    smoothie = SmoothingFunction().method4
    scores = []

    for ref, hyp in zip(references, hypotheses):
        # Tokenize each sentence into word lists
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()

        # BLEU expects list of references (each ref is a list of tokens)
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        scores.append(score)

    avg_bleu = sum(scores) / len(scores) if scores else 0.0
    return avg_bleu


def compute_score_batch_nonreasoning(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0
        # answer reward: 1.0
        pattern = r"Final Answer: (.+)"
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
        else:
            answer = content.strip()

        reward += compute_em([ground_truth], [answer])

        rewards.append(reward)

    return rewards

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # format reward: 1.0
        content = solution_str
        format_pattern = re.compile(
            r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL
        )
        match = format_pattern.search(content)
        if match:
            # make sure </think> only appear once
            if content.count("</think>") == 1:
                reward += 1.0

        # answer reward: 1.0
        pattern = r"Final Answer: (.+)"
        content = content.replace("</answer>", "").strip()
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
        elif "<answer>" in content:
            answer = content.split("<answer>")[1].strip()
        else:
            answer = ""

        reward += compute_em([ground_truth], [answer])

        rewards.append(reward)

    return rewards


def compute_score_batch_nonreasoning_bleu(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0
        ground_truth = ground_truth.strip()
        # extract plan content
        if "<plan>" in solution_str and "</plan>" in solution_str:
            reward += 0.1

        content = solution_str.replace("</plan>", "").strip()

        if "</think>" in content:
            plan_content = content.split("</think>")[-1].strip()
        elif "<plan>" in content:
            plan_content = content.split("<plan>")[-1].strip()
        else:
            plan_content = content.strip()

        reward += compute_bleu([ground_truth], [plan_content]) * 0.9
        
        rewards.append(reward)

    return rewards


def compute_score_batch_bleu(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0
        ground_truth = ground_truth.strip()
        # format reward: 1.0
        content = solution_str
        format_pattern = re.compile(
            r"^<think>.*?</think>\s*<plan>.*?</plan>$", re.DOTALL
        )
        match = format_pattern.search(content)
        if match:
            # make sure </think> only appear once
            if content.count("</think>") == 1:
                reward += 0.1

            # extract plan content
            content = content.replace("</plan>", "").strip()
            plan_content = content.split("<plan>")[-1].strip()
            reward += compute_bleu([ground_truth], [plan_content]) * 0.9

        rewards.append(reward)

    return rewards


def compute_score_batch_evaluate(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # answer reward: 1.0
        if "Judgment: Answer A" in solution_str and ground_truth == "Answer A":
            reward += 1.0
        elif "Judgment: Answer B" in solution_str and ground_truth == "Answer B":
            reward += 1.0
        else:
            reward += 0.0

        rewards.append(reward)

    return rewards


def compute_score_batch_best_answer(data_sources, solution_strs, ground_truths, extra_infos):

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        if "Judgment: " in solution_str:
            solution_str = solution_str.split("Judgment: ")[1].strip()
        
            if solution_str in ground_truth:
                reward += 1.0
            else:
                reward += 0.0

        rewards.append(reward)

    return rewards


def compute_score_evaluate(data_source, solution_str, ground_truth, extra_info):
    reward = 0.0

    # answer reward: 1.0
    if "Judgment: Answer A" in solution_str and ground_truth == "Answer A":
        reward += 1.0
    elif "Judgment: Answer B" in solution_str and ground_truth == "Answer B":
        reward += 1.0
    else:
        reward += 0.0

    return reward



def compute_score_batch_pandas(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute rewards for code generation tasks involving pandas/numpy operations.

    Rewards:
    - Format reward: Output python code in ```python\s*\n(.*?)```
    - Execution reward: Whether the code can be executed without error
    - Operation reward: ROUGE score between extracted pandas/numpy operation sequences
    """

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # 1. Format reward: output python code in python"""<code>"""
        code_block = extract_python_code(solution_str)
        if code_block:
            reward += 0.1  # assign 0.1 for correct format

        # 2. Execution reward: whether the code can be executable or not
        exec_reward = 0.0
        if code_block:
            try:
                code_output = execute_code_safe(code_block)
                if code_output and "Error" not in code_output:
                    exec_reward = 0.2  # assign 0.2 if executable
            except Exception:
                exec_reward = 0.0
        reward += exec_reward

        # 3. Operation reward: f1 score between operation sequences
        gt_code_block = ground_truth
        ops_pred = set(extract_operations(code_block))
        ops_gt = set(extract_operations(gt_code_block))
        match = len(ops_gt & ops_pred)
        if match > 0:
            precision = match / len(ops_pred)
            recall = match / len(ops_gt)
            f1 = 2 * precision * recall / (precision + recall)
            reward += 0.2 * f1  # scale operation reward
        
        # === 4. Output Match Reward (BLEU) ===

        if code_block:
            output_reward = 0.0
            try:
                pred_out = code_output
                gt_out = execute_code_safe(gt_code_block)
                pred_out = normalize_answer(pred_out)
                gt_out = normalize_answer(gt_out)
                if pred_out.strip() and gt_out.strip():
                    smoothie = SmoothingFunction().method4
                    pred_tokens = pred_out.strip().split()
                    gt_tokens = gt_out.strip().split()
                    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
                    output_reward = 0.5 * bleu  # scale BLEU
            except Exception:
                output_reward = 0.0
            reward += output_reward

        rewards.append(reward)

    return rewards


def compute_score_batch_pandas_output_match(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute rewards for code generation tasks involving pandas/numpy operations.

    Rewards:
    - Format reward: Output python code in ```python\s*\n(.*?)```
    - Execution reward: Whether the code can be executed without error
    - Operation reward: ROUGE score between extracted pandas/numpy operation sequences
    """

    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # 1. Format reward: output python code in python"""<code>"""
        code_block = extract_python_code(solution_str)
        if code_block:
            reward += 0.1  # assign 0.1 for correct format

        # 2. Execution reward: whether the code can be executable or not
        exec_reward = 0.0
        if code_block:
            try:
                code_output = execute_code_safe(code_block)
                if code_output and "Error" not in code_output:
                    exec_reward = 0.2  # assign 0.2 if executable
            except Exception:
                exec_reward = 0.0
        reward += exec_reward

        # 3. Operation reward: f1 score between operation sequences
        gt_code_block = ground_truth
        ops_pred = set(extract_operations(code_block))
        ops_gt = set(extract_operations(gt_code_block))
        match = len(ops_gt & ops_pred)
        if match > 0:
            precision = match / len(ops_pred)
            recall = match / len(ops_gt)
            f1 = 2 * precision * recall / (precision + recall)
            reward += 0.2 * f1  # scale operation reward
        
        # === 4. Output Match Reward (BLEU) ===

        if code_block:
            output_reward = 0.0
            try:
                pred_out = code_output
                gt_out = execute_code_safe(gt_code_block)
                pred_out = normalize_answer(pred_out)
                gt_out = normalize_answer(gt_out)
                if pred_out.strip() and gt_out.strip():
                    smoothie = SmoothingFunction().method4
                    pred_tokens = pred_out.strip().split()
                    gt_tokens = gt_out.strip().split()
                    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
                    output_reward = 0.3 * bleu  # scale BLEU
                
                if normalize_answer(extra_info["gt_answer"]) in normalize_answer(pred_out):
                    output_reward += 0.2
                    
            except Exception:
                output_reward = 0.0
            
            reward += output_reward

        rewards.append(reward)

    return rewards


def compute_score_batch_multiturn_nonreasoning(data_sources, solution_strs, ground_truths, extra_infos):

    # multiturn with tool reward only consider the final answer
    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # answer reward: 1.0
        if "Final Answer: " in solution_str:
            solution_str = solution_str.split("Final Answer: ")[-1].strip()
            solution_str = "Final Answer: " + solution_str
        
        pattern = r"Final Answer: (.+)"
        content = solution_str.strip()
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
        else:
            answer = content

        if extra_info["qtype"] == "FactChecking" or extra_info["qtype"] == "NumericalReasoning":
            reward += compute_em([ground_truth], [answer])
        elif extra_info["qtype"] == "DataAnalysis":
            if extra_info["qsubtype"] == "ImpactAnalysis":
                reward += compute_em([ground_truth], [answer])
            elif extra_info["qsubtype"] in [
                "CorrelationAnalysis",
                "TrendForecasting",
                "StatisticalAnalysis",
            ]:
                reward += compute_em_with_tolerance([ground_truth], [answer], 10)
            else:
                reward += compute_rouge([ground_truth], [answer])
        else:
            raise ValueError(f"Unknown qtype: {extra_info['qtype']}")
        rewards.append(reward)

    return rewards



def compute_score_batch_multiturn(data_sources, solution_strs, ground_truths, extra_infos):

    # multiturn with tool reward only consider the final answer
    rewards = []

    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = 0.0

        # answer reward: 1.0
        if "Final Answer: " in solution_str:
            solution_str = solution_str.split("Final Answer: ")[-1].strip()
            solution_str = "Final Answer: " + solution_str

        pattern = r"Final Answer: (.+)"
        content = solution_str.replace("</answer>", "").strip()
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
        elif "<answer>" in content:
            answer = content.split("<answer>")[1].strip()
        else:
            answer = ""

        if extra_info["qtype"] == "FactChecking" or extra_info["qtype"] == "NumericalReasoning":
            reward += compute_em([ground_truth], [answer])
        elif extra_info["qtype"] == "DataAnalysis":
            if extra_info["qsubtype"] == "ImpactAnalysis":
                reward += compute_em([ground_truth], [answer])
            elif extra_info["qsubtype"] in [
                "CorrelationAnalysis",
                "TrendForecasting",
                "StatisticalAnalysis",
            ]:
                reward += compute_em_with_tolerance([ground_truth], [answer], 10)
            else:
                reward += compute_rouge([ground_truth], [answer])
        else:
            raise ValueError(f"Unknown qtype: {extra_info['qtype']}")
        rewards.append(reward)

    return rewards


def compute_score(data_source, solution_str, ground_truth, extra_info):

    reward = 0.0

    # format reward: 1.0
    content = solution_str
    format_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL
    )
    match = format_pattern.search(content)
    if match:
        # make sure </think> only appear once
        if content.count("</think>") == 1:
            reward += 1.0

    # answer reward: 1.0
    pattern = r"Final Answer: (.+)"
    content = content.replace("</answer>", "").strip()
    match = re.search(pattern, content, re.IGNORECASE)

    if match:
        answer = match.group(1).strip()
    elif "<answer>" in content:
        answer = content.split("<answer>")[1].strip()
    else:
        answer = ""

    reward += compute_em([ground_truth], [answer])

    return reward