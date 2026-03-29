"""
Build datasets for Posterior-Refined Distillation experiments.

Reads DAPO-Math-17K with strong-model hints and produces:
  1. student_train.parquet  - student view (original prompts, for all experiments)
  2. teacher_train.parquet  - teacher view (prompts with hints, for off-policy generation)
  3. seqkd_train.parquet    - SeqKD format (original prompts + hint as teacher_response)
  4. test.parquet            - eval data (original DAPO test set)

Usage:
    python build_pr_dataset.py \
        --train_input /tmp/pr_distill_data/dapo_train_kimi_hints.parquet \
        --test_input /tmp/pr_distill_data/dapo_test_kimi_hints.parquet \
        --output_dir /tmp/pr_distill_data/datasets \
        [--hint_column hint]
"""

import argparse
import ast
import json
import os
from copy import deepcopy

import pandas as pd


HINT_TEMPLATE = """{problem}

Here is a reference solution for your consideration:
{hint}

Please solve the problem step by step. You may use, correct, or ignore the reference solution above."""


import numpy as np


def parse_content(content):
    if isinstance(content, np.ndarray):
        content = content.tolist()
    if isinstance(content, str):
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [{"role": "user", "content": content}]
    if isinstance(content, list):
        return content
    return [{"role": "user", "content": str(content)}]


def get_user_content(messages):
    for msg in messages:
        if isinstance(msg, np.ndarray):
            msg = msg.tolist()
        if not isinstance(msg, dict):
            try:
                msg = dict(msg)
            except (TypeError, ValueError):
                continue
        if msg.get("role") == "user":
            return msg["content"]
    return str(messages)


def build_hinted_messages(messages, hint):
    hinted = deepcopy(messages)
    for msg in hinted:
        if isinstance(msg, dict) and msg.get("role") == "user":
            msg["content"] = HINT_TEMPLATE.format(problem=msg["content"], hint=hint)
            break
    return hinted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", required=True,
                        help="DAPO train parquet with hint column")
    parser.add_argument("--test_input", required=True,
                        help="DAPO test parquet")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for all datasets")
    parser.add_argument("--prompt_key", default="content",
                        help="Key for prompt messages in the parquet")
    parser.add_argument("--hint_column", default="hint",
                        help="Column name containing hints (default: hint)")
    parser.add_argument("--max_hint_tokens", type=int, default=None,
                        help="Truncate hints exceeding this token count (approx chars/3.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hint_col = args.hint_column

    df_train = pd.read_parquet(args.train_input)
    print(f"Loaded train: {len(df_train)} samples")
    print(f"Train columns: {list(df_train.columns)}")

    assert hint_col in df_train.columns, f"Missing '{hint_col}' column. Run prepare_hints.py first."

    has_hint = df_train[hint_col].apply(lambda x: bool(x) if isinstance(x, str) else False)
    print(f"Samples with hints: {has_hint.sum()} / {len(df_train)}")

    df_valid = df_train[has_hint].reset_index(drop=True)
    print(f"Using {len(df_valid)} samples with valid hints")

    if args.max_hint_tokens:
        max_chars = int(args.max_hint_tokens * 3.5)
        long_hints = df_valid[hint_col].str.len() > max_chars
        n_truncated = long_hints.sum()
        if n_truncated > 0:
            df_valid.loc[long_hints, hint_col] = df_valid.loc[long_hints, hint_col].str[:max_chars]
            print(f"Truncated {n_truncated} hints to ~{args.max_hint_tokens} tokens ({max_chars} chars)")

    student_rows = []
    teacher_rows = []
    seqkd_rows = []

    for idx, row in df_valid.iterrows():
        content = row[args.prompt_key]
        messages = parse_content(content)
        hint = row[hint_col]

        base_fields = {}
        for col in df_valid.columns:
            if col not in [args.prompt_key, hint_col]:
                base_fields[col] = row[col]

        student_row = {**base_fields, args.prompt_key: messages, "hint": hint}
        student_rows.append(student_row)

        hinted_messages = build_hinted_messages(messages, hint)
        teacher_row = {**base_fields, args.prompt_key: hinted_messages}
        teacher_rows.append(teacher_row)

        problem_text = get_user_content(messages)
        seqkd_row = {**base_fields, "question": problem_text, "answer": hint}
        seqkd_rows.append(seqkd_row)

    df_student = pd.DataFrame(student_rows)
    df_teacher = pd.DataFrame(teacher_rows)
    df_seqkd = pd.DataFrame(seqkd_rows)

    student_path = os.path.join(args.output_dir, "student_train.parquet")
    teacher_path = os.path.join(args.output_dir, "teacher_train.parquet")
    seqkd_path = os.path.join(args.output_dir, "seqkd_train.parquet")

    df_student.to_parquet(student_path, index=False)
    df_teacher.to_parquet(teacher_path, index=False)
    df_seqkd.to_parquet(seqkd_path, index=False)

    print(f"Saved student_train.parquet: {len(df_student)} rows")
    print(f"Saved teacher_train.parquet: {len(df_teacher)} rows")
    print(f"Saved seqkd_train.parquet:   {len(df_seqkd)} rows")

    if args.test_input and os.path.exists(args.test_input):
        df_test = pd.read_parquet(args.test_input)
        test_path = os.path.join(args.output_dir, "test.parquet")
        df_test.to_parquet(test_path, index=False)
        print(f"Copied test.parquet: {len(df_test)} rows")

    print("\nDataset summary:")
    if len(df_student) > 0:
        sample = df_student.iloc[0]
        user_content = get_user_content(sample[args.prompt_key])
        print(f"  Student prompt (first 200 chars): {user_content[:200]}...")
        teacher_sample = df_teacher.iloc[0]
        teacher_content = get_user_content(teacher_sample[args.prompt_key])
        print(f"  Teacher prompt (first 200 chars): {teacher_content[:200]}...")
        print(f"  Hint (first 200 chars): {str(sample['hint'])[:200]}...")
    else:
        print("  WARNING: No valid samples with hints found!")
    print(f"\nAll files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
