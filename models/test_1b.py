import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass

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

# Load model
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')
config = ModelConfig()
model = KalaiNovaAssistant(config).to(device)
ckpt = torch.load('checkpoints/kalainova_clean_final.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Model loaded! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

enc = tiktoken.get_encoding('cl100k_base')
stop_tokens = enc.encode('\n\n### Question')

def generate(prompt, max_new_tokens=200, temperature=0.7, top_k=40):
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
    answer = full.split('### Answer\n')[-1]
    answer = answer.split('### Question')[0].strip()
    return answer

print('\n' + '='*50)
print('KalaiNova Assistant 1B - Local Test')
print('='*50)

tests = [
    'How do I create a StatefulWidget in Flutter?',
    'How do I reverse a list in Python?',
    'How do I use setState in Flutter?',
    'Write a JavaScript function to check if a number is even.',
    'What is the difference between StatelessWidget and StatefulWidget in Flutter?',
]

for q in tests:
    prompt = f'### Question\n{q}\n\n### Answer\n'
    print(f'\nQ: {q}')
    print(f'A: {generate(prompt)}')
    print()


# ============== QUICK EVALUATION ==============
print("\n" + "="*60)
print("QUICK EVALUATION")
print("="*60)

tests = [
    "How do I reverse a list in Python? Write clean code and explain it.",
    "Explain how to use setState in Flutter with a simple counter example.",
    "Write a JavaScript function to check if a number is even.",
    "What is the difference between StatelessWidget and StatefulWidget in Flutter?"
]

for q in tests:
    prompt = f"### Question\n{q}\n\n### Answer\n"
    print(f"Q: {q}")
    answer = generate(prompt)   # your existing generate function
    print(f"A: {answer}")
    print("-" * 80)