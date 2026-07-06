"""
KalaiNova Coder - Training Loop
Trains the model on coding data
Run on MacBook M4 with Metal acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import time
import os
import json
import math


# ─────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Metal (M4 GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


# ─────────────────────────────────────────
# MODEL (same as test.py)
# ─────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size: int = 64000
    context_length: int = 512
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 4
    ffn_hidden_size: int = 4096
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up   = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = config.num_heads
        self.d = config.hidden_size // config.num_heads
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.n2 = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class KalaiNovaCoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, targets=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────
# SIMPLE TOKENIZER (character level for now)
# Later: replace with proper BPE tokenizer
# ─────────────────────────────────────────

class SimpleTokenizer:
    """
    Character-level tokenizer for initial testing.
    Replace with tiktoken/sentencepiece for real training.
    """
    def __init__(self):
        # Common code characters
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                     "!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t `~\\")
        self.char2id = {c: i+1 for i, c in enumerate(chars)}
        self.id2char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1
        self.pad_id = 0

    def encode(self, text):
        return [self.char2id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join([self.id2char.get(i, '') for i in ids])


# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, data: list, context_length: int, tokenizer: SimpleTokenizer):
        self.context_length = context_length
        self.tokenizer = tokenizer

        # Tokenize all data
        all_tokens = []
        for text in data:
            all_tokens.extend(tokenizer.encode(text))

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Dataset: {len(self.tokens)} tokens total")

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ─────────────────────────────────────────
# SAMPLE CODING DATA (for initial testing)
# Replace with real dataset later
# ─────────────────────────────────────────

SAMPLE_CODE_DATA = [
    """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    """
class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements
""",
    """
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

const processItems = (items) => {
    return items
        .filter(item => item.active)
        .map(item => ({ ...item, processed: true }))
        .sort((a, b) => a.name.localeCompare(b.name));
};
""",
    """
fn merge_sort(arr: &mut Vec<i32>) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    let mid = len / 2;
    let mut left = arr[..mid].to_vec();
    let mut right = arr[mid..].to_vec();
    merge_sort(&mut left);
    merge_sort(&mut right);
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            arr[k] = left[i];
            i += 1;
        } else {
            arr[k] = right[j];
            j += 1;
        }
        k += 1;
    }
}
""",
    """
import 'package:flutter/material.dart';

class CounterWidget extends StatefulWidget {
  const CounterWidget({super.key});

  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text('Count: $_count', style: Theme.of(context).textTheme.headlineMedium),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('Increment'),
        ),
      ],
    );
  }
}
""",
]


# ─────────────────────────────────────────
# LEARNING RATE SCHEDULER (cosine warmup)
# ─────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────

@dataclass
class TrainConfig:
    batch_size: int = 4
    context_length: int = 256
    max_steps: int = 1000
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 500
    checkpoint_dir: str = "./checkpoints"


# ─────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────

def train():
    device = get_device()

    # Configs
    model_config = ModelConfig(context_length=256)
    train_config = TrainConfig()

    # Tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Update model vocab to match tokenizer
    model_config.vocab_size = max(tokenizer.vocab_size, 128)

    # Model
    model = KalaiNovaCoder(model_config).to(device)
    print(f"Model parameters: {model.count_params() / 1e6:.1f}M")

    # Dataset
    dataset = CodeDataset(SAMPLE_CODE_DATA, train_config.context_length, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.max_lr,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )

    # Checkpoint dir
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    # Training
    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    model.train()
    step = 0
    data_iter = iter(dataloader)
    losses = []
    start_time = time.time()

    while step < train_config.max_steps:
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Learning rate schedule
        lr = get_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.max_lr,
            train_config.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        optimizer.zero_grad()
        logits, loss = model(x, y)

        # Backward
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Update
        optimizer.step()

        losses.append(loss.item())
        step += 1

        # Log
        if step % train_config.eval_every == 0:
            avg_loss = sum(losses[-train_config.eval_every:]) / train_config.eval_every
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{train_config.max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Save checkpoint
        if step % train_config.save_every == 0:
            ckpt_path = f"{train_config.checkpoint_dir}/step_{step}.pt"
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model_config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")

    # Save final model
    final_path = f"{train_config.checkpoint_dir}/final_model.pt"
    torch.save({
        'model_state': model.state_dict(),
        'config': model_config,
        'tokenizer_vocab': tokenizer.char2id,
    }, final_path)
    print(f"Saved final model: {final_path}")

    return model, tokenizer


# ─────────────────────────────────────────
# QUICK GENERATION TEST
# ─────────────────────────────────────────

def generate(model, tokenizer, prompt, device, max_tokens=100):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(x)
            next_token = logits[0, -1, :].argmax().item()
            x = torch.cat([x, torch.tensor([[next_token]]).to(device)], dim=1)
            if x.shape[1] >= model.config.context_length:
                break

    return tokenizer.decode(x[0].tolist())


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

print("KalaiNova Coder - Training")
print("=" * 50)

model, tokenizer = train()

device = get_device()
print("\nGeneration test:")
result = generate(model, tokenizer, "def ", device, max_tokens=50)
print(result)
