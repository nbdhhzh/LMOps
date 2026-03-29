import json
import sys
import time
import httpx

TOKEN = sys.argv[1]
API_URL = "https://fenji-llm-proxy.verdent.ai/llm/stream"
PROBLEM = "Let S be the set of all integers from 1 to 2^{10}. Find the number of subsets of S such that the sum of elements is divisible by 4."

for model in ["glm-5", "kimi-k2.5"]:
    print(f"=== {model} ===")
    start = time.time()
    resp = httpx.post(API_URL, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }, json={
        "channel": "internal_proxy_usage",
        "model": model,
        "stream": False,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": PROBLEM}],
    }, timeout=180)
    elapsed = time.time() - start
    data = resp.json()

    thinking_len = sum(len(b.get("thinking", "")) for b in data.get("content", []) if b.get("type") == "thinking")
    text_len = sum(len(b.get("text", "")) for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})

    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Thinking: {thinking_len} chars")
    print(f"  Text:     {text_len} chars")
    print(f"  Tokens:   {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out")
    print(f"  Stop:     {data.get('stop_reason')}")
    for b in data.get("content", []):
        if b.get("type") == "text":
            print(f"  Preview:  {b['text'][:200]}...")
            break
    print()
