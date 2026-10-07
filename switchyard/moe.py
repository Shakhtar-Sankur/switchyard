"""The mixture-of-experts layer: route each token to its top-k experts, run the experts,
and combine their outputs weighted by the router.

Three ways to run the experts, all computing the same function:
- `experts_reference`: a loop over experts, as Hugging Face transformers does it;
- `experts_grouped`: tokens sorted by expert once, each expert a contiguous slice
  (the layout the CUDA kernels and expert parallelism use);
- the expert-parallel runtime in `ep.py`, which sends each slice to the GPU that owns it.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def route(h, router, top_k, norm_topk_prob):
    """h [N, H] -> (indices [N, k] int64, weights [N, k] in h's dtype). Softmax in float32
    over all experts, then the top k (as OLMoE and transformers do)."""
    logits = F.linear(h, router)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    w, idx = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return idx, w.to(h.dtype)


def expert_mlp(x, gate_up, down):
    """SwiGLU expert: x [n, H], gate_up [2I, H], down [H, I]."""
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


def experts_reference(h, idx, w, gate_up, down):
    out = torch.zeros_like(h)
    for e in range(gate_up.shape[0]):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        y = expert_mlp(h[tok], gate_up[e], down[e]) * w[tok, slot, None]
        out.index_add_(0, tok, y.to(out.dtype))
    return out


@dataclass
class Plan:
    """Tokens sorted by expert. The i-th row of the sorted batch is token `token[i]`'s
    `slot[i]`-th choice; expert e owns rows offsets[e]:offsets[e+1]."""
    token: torch.Tensor    # [N*k] int64
    slot: torch.Tensor     # [N*k] int64
    weight: torch.Tensor   # [N*k] router weight of that (token, choice)
    counts: torch.Tensor   # [E] int64
    offsets: torch.Tensor  # [E+1] int64


def plan(idx, w, num_experts):
    N, k = idx.shape
    flat = idx.reshape(-1)
    order = torch.sort(flat, stable=True).indices  # by expert, then token, then slot
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=idx.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return Plan(order // k, order % k, w.reshape(-1)[order], counts, offsets)


def run_sorted(x_sorted, counts, gate_up, down):
    """Each expert's contiguous slice of x_sorted through that expert."""
    out = torch.empty_like(x_sorted)
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            out[start:start + n] = expert_mlp(x_sorted[start:start + n], gate_up[e], down[e])
        start += n
    return out


def combine(y_sorted, p, n_tokens):
    """Weighted sum back to token order. Contributions are added in expert order for
    every token, as in the reference."""
    out = torch.zeros(n_tokens, y_sorted.shape[1], dtype=y_sorted.dtype, device=y_sorted.device)
    out.index_add_(0, p.token, y_sorted * p.weight[:, None])
    return out


def experts_grouped(h, idx, w, gate_up, down):
    p = plan(idx, w, gate_up.shape[0])
    y = run_sorted(h[p.token], p.counts, gate_up, down)
    return combine(y, p, h.shape[0])
