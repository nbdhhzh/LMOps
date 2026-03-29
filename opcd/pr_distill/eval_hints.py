"""
Evaluate model outputs against DAPO-Math-17K ground truth.

Evaluation logic aligned with OPCD (math_dapo.py): minerva + last_boxed.

Supports:
  - Evaluating hint/response columns in parquet files
  - Evaluating checkpoint json files (from prepare_hints.py)
  - Comparing multiple models/methods side by side
  - Per-category breakdown (if data_source column exists)

Usage:
    # Evaluate a single parquet file
    python eval_hints.py --input data.parquet --columns hint

    # Evaluate checkpoint file
    python eval_hints.py --input data.parquet --checkpoint hints.checkpoint.json --checkpoint_column kimi

    # Compare multiple columns in same parquet
    python eval_hints.py --input data.parquet --columns glm5_hint kimi_hint

    # Compare parquet files from different models
    python eval_hints.py --input data.parquet --columns hint \
        --extra model_b:other_data.parquet:hint

    # Export wrong answers for analysis
    python eval_hints.py --input data.parquet --columns hint --export_errors errors.json
"""

import argparse
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd


# --------------- OPCD answer matching (from math_dapo.py) ---------------

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer):
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i, right, depth = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            depth += 1
        if string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return string[idx:right + 1] if right is not None else None


def remove_boxed(s):
    left = "\\boxed{"
    assert s[:len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left):-1]


def is_correct_minerva(solution_str, gt, answer_pattern=r"(?i)Answer\s*:\s*([^\n]+)"):
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    gt_norm = normalize_final_answer(gt)
    return (pred == gt_norm), pred


def is_correct_strict_box(pred, gt):
    pred = pred[-100:]
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify_answer(solution_str, gt):
    solution_str = str(solution_str)[-300:]
    correct, pred = is_correct_minerva(solution_str, gt)
    if correct:
        return True, pred, "minerva"
    correct_strict, pred_strict = is_correct_strict_box(solution_str, gt)
    if correct_strict == 1:
        return True, pred_strict, "boxed"
    return False, pred, "fail"


# --------------- Evaluation ---------------

def evaluate_column(df, col, gt_col="solution"):
    results = []
    for idx, row in df.iterrows():
        response = row.get(col, "")
        gt = str(row[gt_col])
        if not response or (isinstance(response, float) and np.isnan(response)):
            results.append({
                "idx": idx, "correct": False, "pred": "[EMPTY]",
                "method": "empty", "gt": gt, "response_len": 0,
            })
            continue
        response = str(response)
        correct, pred, method = verify_answer(response, gt)
        results.append({
            "idx": idx, "correct": correct, "pred": pred,
            "method": method, "gt": gt, "response_len": len(response),
            "tail": response[-100:],
        })
    return results


def print_summary(name, results, df=None, category_col="data_source"):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    empty = sum(1 for r in results if r["method"] == "empty")
    minerva_ok = sum(1 for r in results if r["correct"] and r["method"] == "minerva")
    boxed_ok = sum(1 for r in results if r["correct"] and r["method"] == "boxed")
    avg_len = sum(r["response_len"] for r in results) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:   {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"  Empty:      {empty}/{total}")
    print(f"  By minerva: {minerva_ok}  |  By boxed: {boxed_ok}")
    print(f"  Avg length: {avg_len:.0f} chars")

    if df is not None and category_col in df.columns:
        print(f"\n  Per-category breakdown ({category_col}):")
        print(f"  {'Category':30s} | {'Acc':>12s} | {'Count':>6s}")
        print(f"  {'-'*55}")
        categories = df[category_col].unique()
        for cat in sorted(categories, key=str):
            cat_indices = df[df[category_col] == cat].index
            cat_results = [r for r in results if r["idx"] in cat_indices]
            cat_correct = sum(1 for r in cat_results if r["correct"])
            cat_total = len(cat_results)
            pct = 100 * cat_correct / max(cat_total, 1)
            print(f"  {str(cat):30s} | {cat_correct:>4d}/{cat_total:<4d} ({pct:5.1f}%) | {cat_total:>6d}")

    wrong = [r for r in results if not r["correct"] and r["method"] != "empty"]
    if wrong:
        print(f"\n  Wrong answers (showing first 10):")
        for r in wrong[:10]:
            print(f"    idx={r['idx']}: gt={r['gt']}, pred={r['pred']}, "
                  f"tail=...{r.get('tail','')[-60:]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs against DAPO ground truth")
    parser.add_argument("--input", required=True, help="Input parquet file with ground truth")
    parser.add_argument("--columns", nargs="+", default=[], help="Column names to evaluate in input parquet")
    parser.add_argument("--checkpoint", nargs="*", default=[], help="Checkpoint JSON files (from prepare_hints.py)")
    parser.add_argument("--checkpoint_names", nargs="*", default=[], help="Display names for checkpoint files")
    parser.add_argument("--gt_col", default="solution", help="Ground truth column name")
    parser.add_argument("--export_errors", default=None, help="Export wrong answers to JSON file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")

    all_results = {}

    for col in args.columns:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found in parquet. Skipping.")
            continue
        results = evaluate_column(df, col, args.gt_col)
        all_results[col] = results
        print_summary(f"Column: {col}", results, df)

    for i, ckpt_path in enumerate(args.checkpoint):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint '{ckpt_path}' not found. Skipping.")
            continue
        with open(ckpt_path, "r") as f:
            hints = json.load(f)
        name = args.checkpoint_names[i] if i < len(args.checkpoint_names) else os.path.basename(ckpt_path)

        temp_col = f"__ckpt_{i}"
        df[temp_col] = ""
        for str_idx, hint in hints.items():
            int_idx = int(str_idx)
            if int_idx in df.index:
                df.at[int_idx, temp_col] = hint
        results = evaluate_column(df, temp_col, args.gt_col)
        all_results[name] = results
        print_summary(f"Checkpoint: {name}", results, df)
        df.drop(columns=[temp_col], inplace=True)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Method':30s} | {'Accuracy':>12s} | {'Avg Len':>8s}")
        print(f"  {'-'*55}")
        for name, results in all_results.items():
            total = len(results)
            correct = sum(1 for r in results if r["correct"])
            avg_len = sum(r["response_len"] for r in results) / max(total, 1)
            pct = 100 * correct / max(total, 1)
            print(f"  {name:30s} | {correct:>4d}/{total:<4d} ({pct:5.1f}%) | {avg_len:>7.0f}")

    if args.export_errors and all_results:
        errors = {}
        for name, results in all_results.items():
            errors[name] = [r for r in results if not r["correct"]]
        with open(args.export_errors, "w") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\nExported errors to {args.export_errors}")


if __name__ == "__main__":
    main()
