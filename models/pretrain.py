"""
KalaiNova pretraining -- stage 1 of 2.

Trains model.py's KalaiNovaModel (RoPE, GQA, weight-tied, 24k-vocab
tokenizer) from scratch on the local raw-text corpus, producing
checkpoints_pretrain/pretrain_best.pt for finetune.py to load.

This replaces train.py / train_v2.py / train_bpe.py / modal_train.py,
which all trained a DIFFERENT, degraded inline model with no positional
encoding whatsoever (see KalaiNova_Technical_Documentation.docx section
3.2/9) -- the single biggest reason those checkpoints produced
topically-tinted but structurally wrong/incoherent output. This script
is the only from-scratch trainer that actually uses the fixed model.

Honest data-budget note (see count_tokens.py): the local corpus here is
~10M tokens total, not the ~904M this model size was originally
budgeted for in model.py's docstring. Chinchilla ratio would suggest a
much smaller model at this scale; this script instead trains model.py's
existing 52M-param default for many epochs with a held-out validation
set and early-stops on val loss via best-checkpoint saving, which is a
more realistic way to spend a small corpus than a single big model
matched to a data budget that isn't actually present on disk. If you
have a larger corpus (e.g. more Dart-specific text) drop it next to
coding_data.txt and add it to RAW_SOURCES below -- more real tokens is
the one lever that actually raises the ceiling here.
"""

import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from data_quality import (
    clean_qa_pairs,
    iter_question_answer_markdown,
    iter_question_answer_txt,
)
from model import KalaiNovaModel

# ---------------- config ----------------
TOKENIZER_PATH = "kalainova_tokenizer/tokenizer.json"
CHECKPOINT_DIR = "checkpoints_pretrain"
CONTEXT_LENGTH = 1024     # matches model.py's max_seq_len -- the full RoPE
                          # cache range is already precomputed for this
BATCH_SIZE = 8
MAX_STEPS = 6000
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
VAL_TOKENS_FRACTION = 0.02   # held-out contiguous slice, not shuffled in
EVAL_EVERY = 200
EVAL_BATCHES = 20
LOG_EVERY = 20
SEED = 1337
# -----------------------------------------

RAW_SOURCES_PLAIN = ["coding_data.txt"]          # already clean, used verbatim
RAW_SOURCES_QA_TXT = ["instruction_data_multilang.txt"]   # filtered (see data_quality.py)
RAW_SOURCES_QA_MD = ["grok_flutter_qa.txt"]               # filtered (see data_quality.py)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

use_amp = device == "cuda"
amp_dtype = torch.bfloat16


def build_token_stream(tokenizer):
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    all_ids = []

    for path in RAW_SOURCES_PLAIN:
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(eot_id)
        print(f"  {path:<32} {len(ids):>12,} tokens")

    for path in RAW_SOURCES_QA_TXT:
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        pairs = list(clean_qa_pairs(iter_question_answer_txt(path)))
        n_tok = 0
        for q, a in pairs:
            ids = tokenizer.encode(f"{q}\n{a}").ids
            all_ids.extend(ids)
            all_ids.append(eot_id)
            n_tok += len(ids) + 1
        print(f"  {path:<32} {n_tok:>12,} tokens  (after cleaning: {len(pairs)} pairs)")

    for path in RAW_SOURCES_QA_MD:
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        pairs = list(clean_qa_pairs(iter_question_answer_markdown(path)))
        n_tok = 0
        for q, a in pairs:
            ids = tokenizer.encode(f"{q}\n{a}").ids
            all_ids.extend(ids)
            all_ids.append(eot_id)
            n_tok += len(ids) + 1
        print(f"  {path:<32} {n_tok:>12,} tokens  (after cleaning: {len(pairs)} pairs)")

    return np.array(all_ids, dtype=np.int64)


def get_batch(tokens):
    ix = np.random.randint(0, len(tokens) - CONTEXT_LENGTH - 1, size=BATCH_SIZE)
    x = torch.from_numpy(np.stack([tokens[i : i + CONTEXT_LENGTH] for i in ix])).to(device)
    y = torch.from_numpy(np.stack([tokens[i + 1 : i + CONTEXT_LENGTH + 1] for i in ix])).to(device)
    return x, y


@torch.no_grad()
def estimate_val_loss(model, val_tokens, n_batches=EVAL_BATCHES):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_tokens)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    progress = min(progress, 1.0)
    return float(0.5 * LR * (1 + np.cos(np.pi * progress)))


def main():
    print("Building token stream from local sources:")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokens = build_token_stream(tokenizer)
    print(f"Total tokens: {len(tokens):,}")

    n_val = max(CONTEXT_LENGTH * (EVAL_BATCHES + 1), int(len(tokens) * VAL_TOKENS_FRACTION))
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]
    print(f"Train tokens: {len(train_tokens):,}  |  Val tokens: {len(val_tokens):,}")

    model = KalaiNovaModel().to(device)
    print(f"Parameters: {model.num_params()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    resume_path = os.path.join(CHECKPOINT_DIR, "pretrain_last.pt")
    start_step = 0
    best_val = float("inf")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from step {ckpt['step']} (best_val={best_val:.4f})")

    model.train()
    t0 = time.time()

    for step in range(start_step, MAX_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_tokens)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(f"step {step:5d}/{MAX_STEPS} | train loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")

        if (step + 1) % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            val_loss = estimate_val_loss(model, val_tokens)
            print(f"  === step {step + 1} | val loss {val_loss:.4f} ===")

            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "best_val": best_val},
                resume_path,
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "step": step, "val_loss": val_loss},
                    os.path.join(CHECKPOINT_DIR, "pretrain_best.pt"),
                )
                print(f"  saved new best: {CHECKPOINT_DIR}/pretrain_best.pt")

    print("Pretraining done. Best checkpoint: checkpoints_pretrain/pretrain_best.pt")


if __name__ == "__main__":
    main()
