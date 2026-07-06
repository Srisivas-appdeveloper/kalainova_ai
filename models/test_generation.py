import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken
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
        return logits, None
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
config = ModelConfig()
model = KalaiNovaCoder(config).to(device)

print('Loading checkpoint...')
ckpt = torch.load('checkpoints/v2_step_2000.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print('Loaded! Step:', ckpt.get('step', 'unknown'))

enc = tiktoken.get_encoding('cl100k_base')
prompts = ['def calculate(', 'class ', 'import ', 'def fibonacci(']

def generate_with_sampling(model, x, max_new_tokens, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = F.softmax(top_vals, dim=-1)
        next_tok = top_idx[torch.multinomial(probs, 1)].item()
        x = torch.cat([x, torch.tensor([[next_tok]]).to(x.device)], dim=1)
    return x

for prompt in prompts:
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        x = generate_with_sampling(model, x, max_new_tokens=60, temperature=0.7, top_k=40)
    result = enc.decode(x[0].tolist())
    print(f'--- Prompt: {prompt!r} ---')
    print(result)
    print()