"""
KalaiNova Coder - Distributed Training
Coordinator Node - Run on MacBook M4
Shares gradients with HP laptop via Hivemind
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import hivemind
import time
import os

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Metal (M4 GPU)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

print("=" * 50)
print("KalaiNova Coder - Distributed Training")
print("Role: COORDINATOR (MacBook M4)")
print("=" * 50)

# Start Hivemind DHT
print("\nStarting Hivemind DHT node...")
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/59640"],
    start=True
)
print("DHT started!")
print("Node ID:", dht.peer_id)
for addr in dht.get_visible_maddrs():
    print("Address:", addr)

# Load data
print("\nLoading coding data...")
with open("coding_data.txt", "r") as f:
    text = f.read()

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
print(f"Tokens: {len(tokens):,}")
tokens = torch.tensor(tokens, dtype=torch.long)

# Setup
device = get_device()
config = ModelConfig()
model = KalaiNovaCoder(config).to(device)
print(f"Parameters: {model.count_params()/1e6:.1f}M")

# Hivemind Optimizer - this syncs gradients between devices
optimizer = hivemind.optim.CollaborativeOptimizer(
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1),
    dht=dht,
    prefix="kalainova_coder",
    target_batch_size=512,  # sync after this many samples across all peers
    batch_size_per_step=4,
    verbose=True,
)

os.makedirs("checkpoints", exist_ok=True)

context = 256
batch = 4
max_steps = 5000

print(f"\nDistributed Training Started!")
print("Waiting for HP laptop to connect...")
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
    optimizer.step()  # Hivemind syncs here automatically

    if step % 100 == 0:
        elapsed = time.time() - start
        peers = len(dht.get_visible_maddrs())
        print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | "
              f"Peers: {peers} | Time: {elapsed:.1f}s")

    if step % 1000 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'config': config,
            'step': step,
        }, f"checkpoints/dist_step_{step}.pt")
        print(f"Checkpoint saved: dist_step_{step}.pt")

print("\nDistributed Training Complete!")