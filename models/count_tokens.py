#!/usr/bin/env python3
"""
KalaiNova data budget counter.

Counts real token totals (cl100k_base — same encoding modal_train.py /
train_v2.py / train_bpe.py already use) across the local data files
cataloged in the technical doc, so the from-scratch model can be sized
to the actual data budget instead of guessed at 25-150M params.

Usage:
    pip install tiktoken --break-system-packages   # if not already installed
    python count_tokens.py --dir /path/to/kalainova/data

    # also try to pull + count the 4 Hugging Face datasets (needs internet
    # + `pip install datasets`); dataset slugs below are copied from the
    # doc's Section 5 table and are NOT verified against the real download
    # scripts, so check them against download_multilang.py /
    # download_more_data.py if any fail to load:
    python count_tokens.py --dir /path/to/kalainova/data --include-hf

Add/remove filenames in LOCAL_FILES if your layout differs — it will
just skip anything it can't find and tell you so.
"""

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

# (hub_id, config_name) — unverified, copied from the doc's table.
HF_DATASETS = [
    ("NoirZangetsu/Flutter-Code-with-Questions", None),
    ("sahil2801/CodeAlpaca-20k", None),
    ("iamtarun/python_code_instructions_18k", None),
    ("mlabonne/Evol-Instruct-Python-26k", None),
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


def count_local(data_dir):
    results = {}
    for fname in LOCAL_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  {fname:<32} not found, skipping")
            continue
        n = count_jsonl(path) if fname.endswith(".jsonl") else count_txt(path)
        results[fname] = n
        print(f"  {fname:<32} {n:>12,} tokens")
    return results


def count_hf():
    try:
        from datasets import load_dataset
    except ImportError:
        print("  `datasets` not installed — skipping HF sources")
        print("  (pip install datasets --break-system-packages)")
        return {}

    results = {}
    for hub_id, config in HF_DATASETS:
        try:
            ds = load_dataset(hub_id, config, split="train")
        except Exception as e:
            print(f"  {hub_id:<45} FAILED: {e}")
            continue
        total = sum(
            len(ENC.encode(" ".join(v for v in row.values() if isinstance(v, str))))
            for row in ds
        )
        results[hub_id] = total
        print(f"  {hub_id:<45} {total:>12,} tokens")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding the local data files")
    ap.add_argument("--include-hf", action="store_true", help="also fetch + count the 4 HF datasets")
    args = ap.parse_args()

    print("Local files:")
    local_results = count_local(args.dir)

    hf_results = {}
    if args.include_hf:
        print("\nHF datasets:")
        hf_results = count_hf()

    total = sum(local_results.values()) + sum(hf_results.values())
    print(f"\nTOTAL: {total:,} tokens")
    if total:
        print(f"Chinchilla floor (~20 tokens/param, single pass): ~{total // 20:,} params")
        print("Multiple epochs can stretch this somewhat, but returns drop off fast "
              "past a handful of passes — treat that number as a ceiling to design "
              "toward, not one to multiply by 5-10x.")


if __name__ == "__main__":
    main()