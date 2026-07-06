"""
KalaiNova model — right-sized architecture (vocab=24,000, ~52M params).

Implements the same ingredients model.py aspired to (RMSNorm, SwiGLU,
true GQA, RoPE, weight-tied lm_head) but at the scale the real data
budget (904M Dart pretrain + 40M curated fine-tune tokens) supports,
using the custom 24k tokenizer trained on the Dart corpus instead of
the 100,277-token tiktoken vocab.

Fixes vs. the models that were actually trained (train_v2.py /
train_bpe.py / modal_train.py): those ran a stripped inline attention
with NO positional encoding at all. This one applies RoPE to every
query/key before attention, so the model can represent token order and
distance — the root cause of the wrong/repeating output.

Also fixes the nonstandard top-p sampling flagged in the audit doc
(this uses the standard "smallest set whose cumulative prob >= top_p"
definition).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, max_seq_len, theta=500000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim) -- rotates pairs of dims
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden, ffn_hidden):
        super().__init__()
        self.gate = nn.Linear(hidden, ffn_hidden, bias=False)
        self.up = nn.Linear(hidden, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, ffn_hidden):
        super().__init__()
        self.attn_norm = RMSNorm(hidden)
        self.attn = GQAAttention(hidden, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class KalaiNovaModel(nn.Module):
    """Default config = the confirmed spec: vocab 24,000 / hidden 576 /
    10 layers / 8 heads / 2 kv-heads / SwiGLU ffn=3x -> ~52M params."""

    def __init__(
        self,
        vocab_size=24000,
        hidden=576,
        n_layers=10,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=3.0,
        max_seq_len=1024,
        rope_theta=500000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        ffn_hidden = int(hidden * ffn_mult)

        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList(
            [Block(hidden, n_heads, n_kv_heads, ffn_hidden) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        head_dim = hidden // n_heads
        cos, sin = precompute_rope(head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # nn.Module.parameters() already de-duplicates by object identity,
        # so the tied embedding/lm_head weight is counted once automatically.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """Standard nucleus sampling: keeps the smallest token set whose
        cumulative probability >= top_p (the doc flagged the original
        top-p implementation as nonstandard/edge-case-prone)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_sorted = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, next_sorted)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


if __name__ == "__main__":
    model = KalaiNovaModel()
    n = model.num_params()
    print(f"Total params (weight-tied, counted once): {n:,}")

    # forward pass + loss check
    B, T = 2, 64
    idx = torch.randint(0, 24000, (B, T))
    targets = torch.randint(0, 24000, (B, T))
    targets[:, :10] = -100  # simulate masked question tokens
    logits, loss = model(idx, targets)
    print(f"logits shape: {tuple(logits.shape)}  loss: {loss.item():.4f}")

    # generation smoke test
    prompt = torch.randint(0, 24000, (1, 8))
    out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_p=0.9)
    print(f"generated shape: {tuple(out.shape)} (prompt 8 + 20 new = 28 expected)")