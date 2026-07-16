"""
Builds flutter_finetune.npz for finetune.py (stage 2), from the curated
Flutter Q&A sources:
    flutter_training_data.jsonl   (157 examples)
    flutter_synthetic_qa.jsonl    (731 examples)
    grok_flutter_qa.txt           (~175 examples, markdown Q/A blocks)

Each example is formatted as "{question}\n{answer}<|endoftext|>" -- the
same convention testqa.py already uses at inference time (it feeds the
raw question text and expects the answer to follow immediately), so
prompting after fine-tuning matches what the model actually saw during
training.

Labels mask everything except the answer + stop token with -100 (ignored
by model.py's cross_entropy), so the loss only ever grades the answer --
the exact masking regression the technical doc flagged in
fine_tune_kalainova.py, fixed here for the from-scratch track.
"""

import numpy as np
from tokenizers import Tokenizer

from data_quality import (
    clean_qa_pairs,
    iter_question_answer_jsonl,
    iter_question_answer_markdown,
)

TOKENIZER_PATH = "kalainova_tokenizer/tokenizer.json"
OUT_PATH = "flutter_finetune.npz"
MAX_LEN = 1024   # matches pretrain.py's CONTEXT_LENGTH / model.py's max_seq_len

SOURCES = [
    ("flutter_training_data.jsonl", iter_question_answer_jsonl),
    ("flutter_synthetic_qa.jsonl", iter_question_answer_jsonl),
    ("grok_flutter_qa.txt", iter_question_answer_markdown),
]


def build():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = tokenizer.token_to_id("<pad>")

    all_pairs = []
    for path, reader in SOURCES:
        pairs = list(reader(path))
        print(f"  {path:<32} {len(pairs):>5} raw pairs")
        all_pairs.extend(pairs)

    pairs = list(clean_qa_pairs(all_pairs))
    print(f"After quality filter + dedup: {len(pairs)} pairs (from {len(all_pairs)} raw)")

    input_ids_list, labels_list = [], []
    dropped_too_long = 0
    for question, answer in pairs:
        prompt_ids = tokenizer.encode(question + "\n").ids
        answer_ids = tokenizer.encode(answer).ids + [eot_id]

        if len(prompt_ids) >= MAX_LEN:
            dropped_too_long += 1
            continue

        ids = (prompt_ids + answer_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]

        pad_len = MAX_LEN - len(ids)
        ids = ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        input_ids_list.append(ids)
        labels_list.append(labels)

    if dropped_too_long:
        print(f"Dropped {dropped_too_long} example(s) whose question alone exceeded MAX_LEN={MAX_LEN}")

    input_ids = np.array(input_ids_list, dtype=np.int32)
    labels = np.array(labels_list, dtype=np.int32)
    print(f"Final tensor: {input_ids.shape}")

    supervised_tokens = int((labels != -100).sum())
    print(f"Supervised (answer) tokens: {supervised_tokens:,} "
          f"({supervised_tokens / labels.size:.1%} of all positions)")

    np.savez(OUT_PATH, input_ids=input_ids, labels=labels)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build()
