import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import time
import os
import math

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Metal (M4 GPU)")
        return torch.device("mps")
    return torch.device("cpu")

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
        B,T,C = x.shape
        q = self.q(x).view(B,T,self.h,self.d).transpose(1,2)
        k = self.k(x).view(B,T,self.h,self.d).transpose(1,2)
        v = self.v(x).view(B,T,self.h,self.d).transpose(1,2)
        out = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(out.transpose(1,2).contiguous().view(B,T,C))

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
    def forward(self, x, targets=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Load and tokenize data
print("Loading coding_data.txt...")
with open("coding_data.txt", "r") as f:
    text = f.read()

print(f"Characters: {len(text):,}")
enc = tiktoken.get_encoding("cl100k_base")
print("Tokenizing with BPE...")
tokens = enc.encode(text)
print(f"BPE Tokens: {len(tokens):,}")
tokens = torch.tensor(tokens, dtype=torch.long)

device = get_device()
config = ModelConfig()
model = KalaiNovaCoder(config).to(device)
print(f"Parameters: {model.count_params()/1e6:.1f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
os.makedirs("checkpoints", exist_ok=True)

context = 256
batch = 4
max_steps = 3000

print(f"\nTraining {max_steps} steps with BPE tokenizer...")
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

    if step % 200 == 0:
        elapsed = time.time() - start
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    if step % 1000 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'config': config,
        }, f"checkpoints/bpe_step_{step}.pt")
        print(f"Checkpoint saved: bpe_step_{step}.pt")

print("\nTraining complete!")

# Generation test
model.eval()
prompt = "def calculate("
ids = enc.encode(prompt)
x = torch.tensor([ids], dtype=torch.long).to(device)

with torch.no_grad():
    for _ in range(100):
        logits, _ = model(x)
        next_tok = logits[0, -1, :].argmax().item()
        x = torch.cat([x, torch.tensor([[next_tok]]).to(device)], dim=1)

result = enc.decode(x[0].tolist())
print(f"\nGeneration test:\n{result}")
