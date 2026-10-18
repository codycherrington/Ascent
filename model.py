"""Minimal GPT for Ascent: decoder-only transformer with tied embeddings.

Pre-norm blocks (LayerNorm, no bias), causal attention via
F.scaled_dot_product_attention, 4x GELU MLP, GPT-2-style init.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.proj.is_residual = True
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.proj.is_residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def param_counts(self):
        """Return (total, embedding, transformer) unique parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return total, emb, total - emb

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def generate_stream(model, ids, max_new_tokens, temperature=0.8, top_k=40, seed=None, eos_id=None):
    """Yield new token ids one at a time (see generate() for the arguments).

    temperature == 0 means greedy. top_k always keeps at least the top token.
    Stops after yielding eos_id. Closing the generator early restores train/eval mode.
    """
    device = next(model.parameters()).device
    if not torch.is_tensor(ids):
        ids = torch.tensor([ids], dtype=torch.long)
    ids = ids.to(device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            ctx = ids[:, -model.cfg.block_size:]
            logits, _ = model(ctx)
            logits = logits[0, -1].float().cpu()
            if temperature == 0:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = max(1, min(top_k, logits.numel()))
                    kth = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=gen))
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
            yield nxt
            if eos_id is not None and nxt == eos_id:
                break
    finally:
        model.train(was_training)


def generate(model, ids, max_new_tokens, temperature=0.8, top_k=40, seed=None, eos_id=None):
    """Extend ids (1D list or LongTensor of shape (1, T)) and return the new token ids."""
    return list(generate_stream(model, ids, max_new_tokens, temperature, top_k, seed, eos_id))


if __name__ == "__main__":
    m = GPT(GPTConfig())
    total, emb, tr = m.param_counts()
    print(f"params: {total:,} total | {emb:,} embedding (tok+pos) | {tr:,} transformer")
