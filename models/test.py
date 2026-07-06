import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

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
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

config = ModelConfig()
model = KalaiNovaCoder(config)
params = model.count_params()
print("=" * 40)
print("KalaiNova Coder - Model Test")
print("=" * 40)
print(f"Parameters: {params/1e6:.1f}M")
print(f"Layers: {config.num_layers}")
print(f"Hidden: {config.hidden_size}")
x = torch.randint(0, config.vocab_size, (2, 128))
out = model(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print("Model works!")
