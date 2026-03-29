"""
Test models on DAPO-Math-17K problems.
Prompt and evaluation aligned with OPCD's original implementation:
  - No system prompt; raw DAPO content as-is
  - Answer matching: minerva (Answer: ...) + last \\boxed{} from math_dapo.py
  - Thinking disabled for all models
"""
import httpx, time, json, sys, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed

VERDENT_TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DAPO_PATH = os.environ.get("DAPO_PATH", "/tmp/pr_distill_data/dapo_test_raw.parquet")

VERDENT_URL = "https://fenji-llm-proxy.verdent.ai/llm/stream"

MODELS = [
    {"name": "kimi-k2.5",    "backend": "verdent", "extra": {"thinking": {"type": "disabled"}}},
]

MAX_TOKENS = 8192
NUM_PROBLEMS = 20
SEED = 42


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
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


def is_correct_strict_box(pred, gt):
    pred = pred[-100:]
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify_answer(solution_str, gt):
    solution_str = solution_str[-300:]
    correct, pred = is_correct_minerva(solution_str, gt)
    if correct:
        return True, pred
    correct_strict, pred_strict = is_correct_strict_box(solution_str, gt)
    if correct_strict == 1:
        return True, pred_strict
    return False, pred


# --------------- Data loading ---------------

def load_problems():
    import pandas as pd, ast, numpy as np
    df = pd.read_parquet(DAPO_PATH)
    sampled = df.sample(n=NUM_PROBLEMS, random_state=SEED)
    problems = []
    for _, row in sampled.iterrows():
        content = row["content"]
        if isinstance(content, str):
            try:
                content = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                pass
        if hasattr(content, "tolist"):
            content = content.tolist()
        if isinstance(content, list):
            for msg in content:
                if hasattr(msg, "tolist"):
                    msg = msg.tolist()
                if not isinstance(msg, dict):
                    try:
                        msg = dict(msg)
                    except (TypeError, ValueError):
                        continue
                if msg.get("role") == "user":
                    content = msg["content"]
                    break
        problems.append({
            "label": f"Q{len(problems)+1}",
            "prompt": str(content),
            "answer": str(row.get("solution", "?")),
        })
    return problems


# --------------- API calls ---------------

def call_verdent(model_cfg, prompt):
    payload = {
        "channel": "internal_proxy_usage",
        "model": model_cfg["name"],
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if "extra" in model_cfg:
        payload.update(model_cfg["extra"])
    start = time.time()
    with httpx.Client() as c:
        resp = c.post(VERDENT_URL, headers={"Content-Type": "application/json", "Authorization": f"Bearer {VERDENT_TOKEN}"}, json=payload, timeout=180)
    elapsed = time.time() - start
    if resp.status_code != 200:
        return {"time": elapsed, "error": f"HTTP {resp.status_code}"}
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return {"time": elapsed, "text": text, "out_tokens": usage.get("output_tokens", "?"), "stop": data.get("stop_reason", "?")}


def call_openrouter(model_cfg, prompt):
    start = time.time()
    with httpx.Client() as c:
        resp = c.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                      json={"model": model_cfg["name"], "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}, timeout=300)
    elapsed = time.time() - start
    if resp.status_code != 200:
        return {"time": elapsed, "error": f"HTTP {resp.status_code}"}
    data = resp.json()
    ch = data["choices"][0]
    content = ch["message"].get("content", "") or ""
    usage = data.get("usage", {})
    return {"time": elapsed, "text": content, "out_tokens": usage.get("completion_tokens", "?"), "stop": ch.get("finish_reason", "?")}


def run_one(model_cfg, prompt):
    try:
        if model_cfg["backend"] == "verdent":
            return call_verdent(model_cfg, prompt)
        return call_openrouter(model_cfg, prompt)
    except Exception as e:
        return {"time": 0, "error": str(e)[:80]}


# --------------- Main ---------------

def main():
    print("Loading DAPO-Math-17K problems...", flush=True)
    problems = load_problems()
    for p in problems:
        print(f"  {p['label']} (gt={p['answer']}): {p['prompt'][:80]}...", flush=True)

    results = {m["name"]: {} for m in MODELS}
    tasks = [(m, p) for m in MODELS for p in problems]
    print(f"\nRunning {len(tasks)} calls in parallel...\n", flush=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=max(len(MODELS) * 2, 10)) as ex:
        for m, p in tasks:
            f = ex.submit(run_one, m, p["prompt"])
            futures[f] = (m["name"], p["label"])
        done = 0
        for f in as_completed(futures):
            name, label = futures[f]
            r = f.result()
            results[name][label] = r
            done += 1
            if "error" in r:
                print(f"  [{done}/{len(tasks)}] {name:25s} {label} -> ERROR: {r['error']}", flush=True)
            else:
                print(f"  [{done}/{len(tasks)}] {name:25s} {label} -> {r['time']:5.1f}s  {r['out_tokens']} tok", flush=True)

    print(f"\n{'='*80}")
    print("RESULTS (eval: OPCD math_dapo.py minerva + last_boxed)")
    print(f"{'='*80}")
    labels = [p["label"] for p in problems]
    gt_map = {p["label"]: p["answer"] for p in problems}

    print(f"\n--- Per-question detail ---")
    print(f"{'Q':>4s} | {'GT':>8s} | ", end="")
    for m in MODELS:
        print(f"{m['name']:>20s} | ", end="")
    print()
    print("-" * (18 + 23 * len(MODELS)))

    model_stats = {m["name"]: {"times": [], "correct": 0, "total": 0, "tokens": []} for m in MODELS}
    for p in problems:
        l = p["label"]
        row = f"{l:>4s} | {gt_map[l]:>8s} | "
        for m in MODELS:
            name = m["name"]
            r = results[name].get(l, {})
            if "error" in r:
                row += f"{'ERR':>20s} | "
            else:
                text = r.get("text", "")
                ok, pred = verify_answer(text, gt_map[l])
                tok = r["out_tokens"]
                mark = "Y" if ok else "N"
                tail = text[-100:].replace("\n", "\\n")
                row += f"{r['time']:5.1f}s {str(tok):>5s}t {mark:>1s} ({pred[:8]})"
                row += f"\n       {'':>8s}   tail: {tail}\n"
                model_stats[name]["times"].append(r["time"])
                model_stats[name]["tokens"].append(int(tok) if str(tok).isdigit() else 0)
                model_stats[name]["total"] += 1
                if ok:
                    model_stats[name]["correct"] += 1
        print(row, flush=True)

    print(f"\n--- Summary ---")
    print(f"{'Model':25s} | {'Accuracy':>10s} | {'Avg Time':>10s} | {'Avg Tokens':>10s} | {'Total Time':>10s}")
    print("-" * 78)
    for m in MODELS:
        s = model_stats[m["name"]]
        if s["total"] > 0:
            acc = f"{s['correct']}/{s['total']}"
            avg_t = f"{sum(s['times'])/len(s['times']):.1f}s"
            avg_tok = f"{sum(s['tokens'])//len(s['tokens'])}"
            tot_t = f"{sum(s['times']):.0f}s"
        else:
            acc = avg_t = avg_tok = tot_t = "N/A"
        print(f"{m['name']:25s} | {acc:>10s} | {avg_t:>10s} | {avg_tok:>10s} | {tot_t:>10s}")
    print()


if __name__ == "__main__":
    main()
