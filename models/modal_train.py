import modal

app = modal.App("kalainova-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "tiktoken",
        "datasets",
        "numpy",
    )
)

volume = modal.Volume.from_name("kalainova-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=72000,
    volumes={"/checkpoints": volume},
)
def full():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    import tiktoken
    import time
    import os
    import json as jsonlib
    from dataclasses import dataclass
    from datasets import load_dataset

    IGNORE_INDEX = -100

    @dataclass
    class ModelConfig:
        vocab_size: int = 100277
        context_length: int = 512
        hidden_size: int = 1536
        num_layers: int = 24
        num_heads: int = 16
        ffn_hidden_size: int = 6144
        norm_eps: float = 1e-5

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    class SwiGLU(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.gate = nn.Linear(c.hidden_size, c.ffn_hidden_size, bias=False)
            self.up = nn.Linear(c.hidden_size, c.ffn_hidden_size, bias=False)
            self.down = nn.Linear(c.ffn_hidden_size, c.hidden_size, bias=False)
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Attention(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.h = c.num_heads
            self.d = c.hidden_size // c.num_heads
            self.q = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
            self.k = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
            self.v = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
            self.o = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        def forward(self, x):
            B, T, C = x.shape
            q = self.q(x).view(B, T, self.h, self.d).transpose(1, 2)
            k = self.k(x).view(B, T, self.h, self.d).transpose(1, 2)
            v = self.v(x).view(B, T, self.h, self.d).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(out.transpose(1, 2).contiguous().view(B, T, C))

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.n1 = RMSNorm(c.hidden_size)
            self.attn = Attention(c)
            self.n2 = RMSNorm(c.hidden_size)
            self.ffn = SwiGLU(c)
        def forward(self, x):
            x = x + self.attn(self.n1(x))
            x = x + self.ffn(self.n2(x))
            return x

    class KalaiNovaAssistant(nn.Module):
        def __init__(self, c, use_grad_checkpoint=True):
            super().__init__()
            self.config = c
            self.use_grad_checkpoint = use_grad_checkpoint
            self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
            self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
            self.norm = RMSNorm(c.hidden_size)
            self.head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
            self.head.weight = self.embed.weight
        def forward(self, x, targets=None):
            x = self.embed(x)
            for layer in self.layers:
                if self.use_grad_checkpoint and self.training:
                    x = grad_checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            logits = self.head(self.norm(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=IGNORE_INDEX,
                )
            return logits, loss
        def count_params(self):
            return sum(p.numel() for p in self.parameters())

    print("=" * 60)
    print("KalaiNova Assistant - Proper Training on Modal A100")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nDownloading datasets...")
    all_qa = []

    try:
        ds1 = load_dataset('NoirZangetsu/Flutter-Code-with-Questions-Dataset-English', split='train')
        for item in ds1:
            q = (item.get('questions') or '').strip()
            a = (item.get('code') or '').strip()
            if q and a and len(a) > 20:
                all_qa.append(f"### Question\n{q}\n\n### Answer\n{a}\n\n")
        print(f"Flutter/Dart: {len([x for x in all_qa if '### Question' in x])} examples")
    except Exception as e:
        print(f"Flutter error: {e}")

    try:
        ds2 = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
        for item in ds2:
            q = (item.get('instruction') or '').strip()
            a = (item.get('output') or '').strip()
            if q and a and len(a) > 20:
                all_qa.append(f"### Question\n{q}\n\n### Answer\n{a}\n\n")
        print(f"General coding (JS/Python/etc): {len(ds2)} examples")
    except Exception as e:
        print(f"CodeAlpaca error: {e}")

    try:
        ds3 = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
        for item in ds3:
            q = (item.get('instruction') or '').strip()
            a = (item.get('output') or '').strip()
            if q and a and len(a) > 20:
                all_qa.append(f"### Question\n{q}\n\n### Answer\n{a}\n\n")
        print(f"Python: {len(ds3)} examples")
    except Exception as e:
        print(f"Python error: {e}")

    # Load synthetic Grok-generated data (Flutter + general code + business + general knowledge)
    for fpath in ["/checkpoints/flutter_synthetic_qa.jsonl"]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = jsonlib.loads(line)
                    q, a = obj.get("question", "").strip(), obj.get("answer", "").strip()
                    if q and a and len(a) > 20:
                        all_qa.append(f"### Question\n{q}\n\n### Answer\n{a}\n\n")
                        count += 1
                print(f"{fpath}: {count} examples")
        except FileNotFoundError:
            print(f"{fpath} not found, skipping")

    print(f"\nTotal Q&A pairs: {len(all_qa)}")

    enc = tiktoken.get_encoding("cl100k_base")
    stop_tokens = enc.encode("\n\n### Question")

    print("Tokenizing per example (question masked, answer + stop token supervised)...")
    examples = []
    context = 512
    for text in all_qa:
        if "### Answer\n" not in text:
            continue
        q_part, a_part = text.split("### Answer\n", 1)
        q_ids = enc.encode(q_part + "### Answer\n")
        a_ids = enc.encode(a_part) + stop_tokens
        seq = q_ids + a_ids
        label_seq = [IGNORE_INDEX] * len(q_ids) + a_ids
        if len(seq) > context + 1:
            seq = seq[:context + 1]
            label_seq = label_seq[:context + 1]
        pad_len = (context + 1) - len(seq)
        seq = seq + [0] * pad_len
        label_seq = label_seq + [IGNORE_INDEX] * pad_len
        input_ids = seq[:-1]      # positions 0..context-1
        labels = label_seq[1:]    # shifted: labels[t] = token at t+1
        examples.append((input_ids, labels))

    print(f"Total training examples: {len(examples):,}")
    supervised_tokens = sum(1 for _, labels in examples for t in labels if t != IGNORE_INDEX)
    print(f"Total supervised (answer) tokens: {supervised_tokens:,}")

    device = torch.device("cuda")
    config = ModelConfig()
    model = KalaiNovaAssistant(config, use_grad_checkpoint=True).to(device)
    param_count = model.count_params()
    print(f"\nModel parameters: {param_count/1e6:.1f}M")
    print(f"Context length: {config.context_length}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.num_layers}")
    print(f"Gradient checkpointing: ON (saves ~60% activation memory)")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    def get_lr(step, warmup_steps=500, max_steps=10000, max_lr=3e-4, min_lr=3e-5):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        import math
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    os.makedirs("/checkpoints", exist_ok=True)
    batch = 8
    grad_accum = 4
    max_steps = 10000

    print(f"\nTraining {max_steps} steps")
    print(f"Batch: {batch}, Gradient accumulation: {grad_accum}")
    print(f"Effective batch size: {batch * grad_accum}")
    print("=" * 60)

    model.train()
    start = time.time()
    optimizer.zero_grad()

    ids_tensor = torch.tensor([e[0] for e in examples], dtype=torch.long)
    labels_tensor = torch.tensor([e[1] for e in examples], dtype=torch.long)
    n_examples = len(examples)

    for step in range(1, max_steps + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss_accum = 0.0
        for micro_step in range(grad_accum):
            idx = torch.randint(0, n_examples, (batch,))
            x = ids_tensor[idx].to(device)
            y = labels_tensor[idx].to(device)

            with torch.cuda.amp.autocast():
                _, loss = model(x, y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = step / elapsed
            remaining = (max_steps - step) / steps_per_sec
            print(f"Step {step:5d}/{max_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | {steps_per_sec:.2f} steps/s | ETA: {remaining/3600:.1f}h")

        if step % 2000 == 0:
            path = f"/checkpoints/kalainova_step_{step}.pt"
            torch.save({'model_state': model.state_dict(), 'step': step, 'config': vars(config)}, path)
            volume.commit()
            print(f"✅ Checkpoint saved: {path}")

    final_path = "/checkpoints/kalainova_final.pt"
    torch.save({'model_state': model.state_dict(), 'step': max_steps, 'config': vars(config)}, final_path)
    volume.commit()
    print(f"\n✅ Final model saved: {final_path}")

    def generate(prompt, max_new_tokens=150, temperature=0.7, top_k=40):
        model.eval()
        ids = enc.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(x)
                logits = logits[0, -1, :] / temperature
                top_vals, top_idx = torch.topk(logits, top_k)
                probs = F.softmax(top_vals, dim=-1)
                next_tok = top_idx[torch.multinomial(probs, 1)].item()
                generated.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]]).to(device)], dim=1)
                if len(generated) >= len(stop_tokens):
                    if generated[-len(stop_tokens):] == stop_tokens:
                        break
        full = enc.decode(ids + generated)
        answer = full.split("### Answer\n")[-1]
        answer = answer.split("### Question")[0].strip()
        return answer

    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)

    tests = [
        "### Question\nHow do I create a StatefulWidget in Flutter?\n\n### Answer\n",
        "### Question\nHow do I reverse a list in Python?\n\n### Answer\n",
        "### Question\nWhat is MSME registration?\n\n### Answer\n",
        "### Question\nWrite a JavaScript function to check if a string is a palindrome.\n\n### Answer\n",
    ]

    for q in tests:
        question = q.split("### Question\n")[1].split("\n\n### Answer")[0]
        print(f"\nQ: {question}")
        print(f"A: {generate(q)}")

    print("\nTraining complete!")
    return "Done!"


@app.local_entrypoint()
def main():
    print("Launching KalaiNova full training on Modal A100...")
    result = full.remote()
    print(result)