"""
Generate per-problem hints for DAPO-Math-17K via Verdent proxy.

Prompt aligned with OPCD: no system prompt, raw DAPO content as user message.
Uses streaming to avoid CloudFront 120s gateway timeout on long responses.

Usage:
    python prepare_hints.py \
        --input /tmp/pr_distill_data/dapo_test_raw.parquet \
        --output /tmp/pr_distill_data/dapo_test_hints.parquet \
        --token YOUR_VERDENT_TOKEN \
        [--model kimi-k2.5] \
        [--max_workers 10] \
        [--resume]
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd


API_URL = "https://fenji-llm-proxy.verdent.ai/llm/stream"
CHANNEL = "internal_proxy_usage"


def call_model(client, token, model, problem_content, max_tokens=16384, max_retries=5, disable_thinking=True):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "channel": CHANNEL,
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": problem_content}],
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}

    for attempt in range(max_retries):
        try:
            text_parts = []
            usage = {}
            with client.stream("POST", API_URL, headers=headers, json=payload, timeout=600) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode()[:200]
                    if resp.status_code == 429:
                        wait = min(2 ** attempt * 5, 60)
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    elif resp.status_code >= 500:
                        wait = 2 ** attempt
                        print(f"  Server error {resp.status_code}, retry {attempt+1}/{max_retries} after {wait}s")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  HTTP error {resp.status_code}: {body}")
                        return None, {}

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str.strip() == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_parts.append(delta.get("text", ""))
                    if "usage" in evt:
                        usage = evt["usage"]

            result = "".join(text_parts)
            if result:
                return result, usage
            if not text_parts:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Empty response, retry {attempt+1}/{max_retries} after {wait}s")
                    time.sleep(wait)
                    continue
                return "", usage
            return result, usage

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Error: {e}, retry {attempt+1}/{max_retries} after {wait}s")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None, {}
    return None, {}


def extract_problem_text(content_field):
    if hasattr(content_field, "tolist"):
        content_field = content_field.tolist()
    if isinstance(content_field, list):
        for msg in content_field:
            if hasattr(msg, "tolist"):
                msg = msg.tolist()
            if not isinstance(msg, dict):
                try:
                    msg = dict(msg)
                except (TypeError, ValueError):
                    continue
            if msg.get("role") == "user":
                return msg["content"]
        return str(content_field)
    if isinstance(content_field, str):
        try:
            import ast
            parsed = ast.literal_eval(content_field)
            if isinstance(parsed, list):
                return extract_problem_text(parsed)
        except (ValueError, SyntaxError):
            pass
    return str(content_field)


def main():
    parser = argparse.ArgumentParser(description="Generate hints using strong model via Verdent proxy")
    parser.add_argument("--input", required=True, help="Input parquet file (DAPO-Math-17K)")
    parser.add_argument("--output", required=True, help="Output parquet with hint column")
    parser.add_argument("--token", default=None, help="Verdent API token (or set VERDENT_TOKEN env)")
    parser.add_argument("--model", default="kimi-k2.5", help="Model name (kimi-k2.5, glm-5, etc.)")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max output tokens per problem")
    parser.add_argument("--max_workers", type=int, default=10, help="Parallel API calls")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per API call")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable model thinking (default: disabled)")
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (for testing)")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every N samples")
    parser.add_argument("--hint_column", default="hint", help="Output column name for hints")
    args = parser.parse_args()

    token = args.token or os.environ.get("VERDENT_TOKEN")
    if not token:
        parser.error("Provide --token or set VERDENT_TOKEN environment variable")

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")
    print(f"Columns: {list(df.columns)}")
    print(f"Model: {args.model} (thinking={'enabled' if args.enable_thinking else 'disabled'})")
    print(f"Max tokens: {args.max_tokens}, Workers: {args.max_workers}, Stream: True")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} samples")

    checkpoint_path = args.output + ".checkpoint.json"
    existing_hints = {}
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            existing_hints = json.load(f)
        print(f"Resumed {len(existing_hints)} existing hints from checkpoint")

    problems = []
    for idx in df.index:
        if str(idx) in existing_hints:
            continue
        row = df.loc[idx]
        content = row.get("content", row.get("prompt", ""))
        problem_text = extract_problem_text(content)
        problems.append((idx, problem_text))

    print(f"Need to generate hints for {len(problems)} samples")

    if problems:
        completed = 0
        total = len(problems)
        total_input_tokens = 0
        total_output_tokens = 0
        disable_thinking = not args.enable_thinking

        with httpx.Client() as client:
            def process_one(item):
                idx, problem_text = item
                hint, usage = call_model(client, token, args.model, problem_text,
                                         args.max_tokens, args.max_retries, disable_thinking)
                return idx, hint, usage

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_one, item): item for item in problems}
                for future in as_completed(futures):
                    idx, hint, usage = future.result()
                    existing_hints[str(idx)] = hint if hint else ""
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    completed += 1

                    if completed % 10 == 0:
                        success = sum(1 for v in existing_hints.values() if v)
                        print(f"  [{completed}/{total}] Success: {success}/{len(existing_hints)} | "
                              f"Tokens: {total_input_tokens} in / {total_output_tokens} out")

                    if completed % args.checkpoint_every == 0:
                        with open(checkpoint_path, "w") as f:
                            json.dump(existing_hints, f, ensure_ascii=False)
                        print(f"  Checkpoint saved ({len(existing_hints)} hints)")

            failed = [(idx, text) for idx, text in problems if not existing_hints.get(str(idx))]
            if failed:
                print(f"\nRetrying {len(failed)} failed items...")
                with httpx.Client() as retry_client:
                    def retry_one(item):
                        idx, problem_text = item
                        hint, usage = call_model(retry_client, token, args.model, problem_text,
                                                 args.max_tokens, args.max_retries, disable_thinking)
                        return idx, hint, usage
                    with ThreadPoolExecutor(max_workers=args.max_workers) as retry_executor:
                        retry_futures = {retry_executor.submit(retry_one, item): item for item in failed}
                        for future in as_completed(retry_futures):
                            idx, hint, usage = future.result()
                            if hint:
                                existing_hints[str(idx)] = hint
                                total_input_tokens += usage.get("input_tokens", 0)
                                total_output_tokens += usage.get("output_tokens", 0)
                    still_failed = sum(1 for idx, _ in failed if not existing_hints.get(str(idx)))
                    print(f"  After retry: {len(failed) - still_failed}/{len(failed)} recovered, {still_failed} still failed")

        with open(checkpoint_path, "w") as f:
            json.dump(existing_hints, f, ensure_ascii=False)
        print(f"\nFinal stats: {total_input_tokens} input tokens, {total_output_tokens} output tokens")

    hints_list = []
    for idx in df.index:
        hint = existing_hints.get(str(idx), "")
        hints_list.append(hint)

    df[args.hint_column] = hints_list

    success_count = sum(1 for h in hints_list if h)
    print(f"Total: {len(df)}, With hints: {success_count}, Missing: {len(df) - success_count}")

    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Cleaned up checkpoint file")


if __name__ == "__main__":
    main()
