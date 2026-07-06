import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken, time, os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 100277
    context_length: int = 256
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
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

class KalaiNovaCoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size)
        self.head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x, targets=None):
        x = self.embed(x)
        for l in self.layers:
            x = l(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


print("=" * 50)
print("KalaiNova Assistant - Extended Training v2")
print("=" * 50)

# Load bigger dataset
print("Loading coding_data_v2.txt...")
with open("coding_data_v2.txt", "r") as f:
    text = f.read()
print(f"Characters: {len(text):,}")

enc = tiktoken.get_encoding("cl100k_base")
print("Tokenizing...")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Tokens: {len(tokens):,}")

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

config = ModelConfig()
model = KalaiNovaCoder(config).to(device)
print(f"Parameters: {model.count_params()/1e6:.1f}M")

# Load previous checkpoint to continue training instead of starting over
checkpoint_path = "checkpoints/v2_step_8000.pt"
if os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Resumed from step {ckpt.get('step', 'unknown')}")
else:
    print("No checkpoint found, starting fresh.")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.1, betas=(0.9, 0.95))

os.makedirs("checkpoints", exist_ok=True)
context, batch, max_steps = 256, 8, 2000

print(f"\nTraining {max_steps} steps on {len(tokens):,} tokens...")
print("=" * 50)

model.train()
start = time.time()

for step in range(1, max_steps + 1):
    idx = torch.randint(0, len(tokens) - context - 1, (batch,))
    x = torch.stack([tokens[i:i+context] for i in idx]).to(device)
    y = torch.stack([tokens[i+1:i+context+1] for i in idx]).to(device)

    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        elapsed = time.time() - start
        print(f"Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

    if step % 2000 == 0:
        torch.save({
            "model_state": model.state_dict(),
            "step": step,
        }, f"checkpoints/v2_step_{step}.pt")
        print(f"Checkpoint saved: v2_step_{step}.pt")

print("\nTraining complete!")

# Quick generation test
def generate_with_sampling(model, x, max_new_tokens, temperature=0.7, top_k=40):
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = F.softmax(top_vals, dim=-1)
        next_tok = top_idx[torch.multinomial(probs, 1)].item()
        x = torch.cat([x, torch.tensor([[next_tok]]).to(x.device)], dim=1)
    return x

model.eval()
prompt = "def calculate("
ids = enc.encode(prompt)
x = torch.tensor([ids], dtype=torch.long).to(device)
with torch.no_grad():
    x = generate_with_sampling(model, x, max_new_tokens=80)
print("\nGeneration test:")
print(enc.decode(x[0].tolist()))