#!/usr/bin/env python3
"""KalaiNova data budget counter."""
import argparse
import json
import os
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

LOCAL_FILES = [
    "coding_data.txt",
    "coding_data_v2.txt",
    "instruction_data_multilang.txt",
    "grok_flutter_qa.txt",
    "flutter_synthetic_qa.jsonl",
    "flutter_training_data.jsonl",
]


def count_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return len(ENC.encode(f.read()))


def count_jsonl(path):
    total, bad_lines = 0, 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            text = " ".join(v for v in obj.values() if isinstance(v, str))
            total += len(ENC.encode(text))
    if bad_lines:
        print(f"    ! {bad_lines} malformed line(s) skipped")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()

    print("Local files:")
    results = {}
    for fname in LOCAL_FILES:
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            print(f"  {fname:<32} not found, skipping")
            continue
        n = count_jsonl(path) if fname.endswith(".jsonl") else count_txt(path)
        results[fname] = n
        print(f"  {fname:<32} {n:>12,} tokens")

    total = sum(results.values())
    print(f"\nTOTAL: {total:,} tokens")
    if total:
        print(f"Chinchilla floor (~20 tokens/param, single pass): ~{total // 20:,} params")


if __name__ == "__main__":
    main()
