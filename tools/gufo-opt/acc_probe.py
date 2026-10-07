#!/usr/bin/env python3
"""Serve-side MTP acceptance + decode speed over 10 fixed prompts (greedy, thinking off, 256 tok).
Reads gufo /metrics deltas. usage: acc_probe.py URL NAME OUTJSON"""
import json, re, sys, time, urllib.request
url, name, out = sys.argv[1:4]
P = ["Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and doctests.",
     "Explain in five sentences why the sky is blue.",
     "用五句话解释什么是区块链，以及它的优缺点。",
     "Write a bash script that finds the 10 largest files under a directory and prints their sizes in MB.",
     "List the first 40 prime numbers separated by commas.",
     "Translate to French and German: The meeting has been moved to Thursday afternoon because the client is travelling.",
     "What is 1729 * 37? Show the multiplication step by step.",
     "Write a JSON array describing three books with title, author, year and three tags each.",
     "Summarize the plot of Romeo and Juliet in one paragraph.",
     "Write a SQL schema for customers and orders, then a query returning the top 5 customers by total order amount."]
def metrics():
    t = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^([a-z_:]+) ([0-9.e+-]+)$", t, re.M)}
m0 = metrics(); t0 = time.time(); toks = 0
for p in P:
    body = {"model": "flash-next", "messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0,
            "presence_penalty": 0, "frequency_penalty": 0, "reasoning_effort": "none", "stream": False}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=600))
    toks += r["usage"]["completion_tokens"]
m1 = metrics(); d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
drafted = d.get("llamacpp:spec_decode_num_draft_tokens_total", 0) or d.get("halogen:draft_tokens_total", 0); acc = d.get("llamacpp:spec_decode_num_accepted_tokens_total", 0) or d.get("halogen:draft_tokens_accepted_total", 0)
res = {"name": name, "tokens": toks, "wall_s": round(time.time() - t0, 1),
       "decode_tps": round(d.get("llamacpp:tokens_predicted_total", 0) / max(d.get("llamacpp:tokens_predicted_seconds_total", 1e-9), 1e-9), 1),
       "drafts": int(d.get("llamacpp:spec_decode_num_drafts_total", 0)), "drafted": int(drafted), "accepted": int(acc),
       "accept_rate": round(acc / drafted, 3) if drafted else None}
json.dump(res, open(out, "w")); print(json.dumps(res))
