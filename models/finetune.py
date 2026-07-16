"""
KalaiNova fine-tuning — stage 2 of 2.

Loads a pretrained checkpoint (from pretrain.py) and continues training
on the curated Flutter Q&A set (flutter_finetune.npz), using the
answer-only masked loss already baked into that file's labels array
(question tokens = -100, answer tokens = real ids) -- the model.py
forward() already handles ignore_index=-100 correctly, so no extra
masking logic is needed here.

Much shorter run than pretraining: the Q&A set is ~1,000 examples, not
1B tokens, so this does several epochs over it instead of ~1 pass.
Learning rate is 10x lower than pretraining, to adapt the model's
behavior (answer questions) without overwriting what it just learned
about Dart syntax and structure.
"""

import os
import time

import numpy as np
import torch

from model import KalaiNovaModel

# ---------------- config ----------------
PRETRAIN_CKPT = "checkpoints_pretrain/pretrain_best.pt"
FINETUNE_DATA = "flutter_finetune.npz"
CHECKPOINT_DIR = "checkpoints_finetune"
BATCH_SIZE = 8
NUM_EPOCHS = 8
LR = 3e-5              # 10x lower than pretrain's 3e-4 -- adapting, not relearning
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 20
VAL_FRACTION = 0.1      # bigger fraction than pretrain since this dataset is tiny
LOG_EVERY = 10
SEED = 1337
# -----------------------------------------

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

data = np.load(FINETUNE_DATA)
input_ids_all = data["input_ids"].astype(np.int64)
labels_all = data["labels"].astype(np.int64)
n = len(input_ids_all)
n_val = max(1, int(n * VAL_FRACTION))

rng = np.random.RandomState(SEED)
perm = rng.permutation(n)
val_idx, train_idx = perm[:n_val], perm[n_val:]
train_x, train_y = input_ids_all[train_idx], labels_all[train_idx]
val_x, val_y = input_ids_all[val_idx], labels_all[val_idx]
print(f"Total examples: {n}  |  train: {len(train_idx)}  |  val: {len(val_idx)}")

STEPS_PER_EPOCH = max(1, len(train_idx) // BATCH_SIZE)
MAX_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
print(f"Steps/epoch: {STEPS_PER_EPOCH}  |  total steps: {MAX_STEPS} ({NUM_EPOCHS} epochs)")


def get_batch(x_pool, y_pool):
    ix = np.random.randint(0, len(x_pool), size=BATCH_SIZE)
    x = torch.from_numpy(x_pool[ix]).to(device)
    y = torch.from_numpy(y_pool[ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_val_loss(model, n_batches=10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_x, val_y)
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
    model = KalaiNovaModel().to(device)

    print(f"Loading pretrained checkpoint: {PRETRAIN_CKPT}")
    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from pretrain step {ckpt['step']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.train()
    t0 = time.time()
    best_val = float("inf")

    for step in range(MAX_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_x, train_y)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            epoch = step // STEPS_PER_EPOCH
            print(f"step {step:5d} (epoch {epoch}) | train loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")

        if (step + 1) % STEPS_PER_EPOCH == 0:
            val_loss = estimate_val_loss(model)
            epoch = (step + 1) // STEPS_PER_EPOCH
            print(f"  === epoch {epoch} done | val loss {val_loss:.4f} ===")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "step": step, "val_loss": val_loss},
                    os.path.join(CHECKPOINT_DIR, "finetune_best.pt"),
                )
                print(f"  saved new best: {CHECKPOINT_DIR}/finetune_best.pt")

    torch.save(
        {"model": model.state_dict(), "step": MAX_STEPS},
        os.path.join(CHECKPOINT_DIR, "finetune_final.pt"),
    )
    print("Fine-tuning done. Best checkpoint: checkpoints_finetune/finetune_best.pt")


if __name__ == "__main__":
    main()