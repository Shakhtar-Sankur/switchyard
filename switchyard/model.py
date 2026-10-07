"""An OLMoE decoder in plain PyTorch, with a KV cache for generation.

The mixture-of-experts layers call `self.moe(layer_index, h)`, so the same model runs
with the reference loop, the grouped kernels, or experts spread over several GPUs."""

import math

import torch
import torch.nn.functional as F

from . import moe as moe_lib


def rmsnorm(x, w, eps):
    # As transformers does it: normalize in float32, cast back, then scale.
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class Cache:
    """Keys and values of every layer: [B, kv_heads, capacity, head_dim]. Batches are
    left-padded; `valid` marks the positions that hold real tokens."""

    def __init__(self, config, batch, capacity, dtype, device):
        shape = (batch, config.kv_heads, capacity, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.layers)]
        self.valid = torch.zeros(batch, capacity, dtype=torch.bool, device=device)
        self.next_pos = torch.zeros(batch, dtype=torch.int64, device=device)  # rotary position of the next real token
        self.length = 0


class Model:
    def __init__(self, config, weights, moe=None):
        self.c = config
        self.w = weights
        self.moe = moe or self.local_moe
        dev = weights["embed"].device
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.int64, device=dev).float() / config.head_dim))
        self.inv_freq = inv

    # The experts of one layer on this process, with a choice of implementation.
    experts_impl = staticmethod(moe_lib.experts_grouped)

    def local_moe(self, i, h):
        L = self.w["layers"][i]
        idx, wts = moe_lib.route(h, L["router"], self.c.top_k, self.c.norm_topk_prob)
        return self.experts_impl(h, idx, wts, L["gate_up"], L["down"])

    def _rope(self, positions, dtype):
        freqs = positions[..., None].float() * self.inv_freq  # [B, T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]

    def _attention(self, L, x, cos, sin, cache, li, start, mask):
        c = self.c
        B, T, _ = x.shape
        q = rmsnorm(F.linear(x, L["wq"]), L["q_norm"], c.rms_eps)
        k = rmsnorm(F.linear(x, L["wk"]), L["k_norm"], c.rms_eps)
        v = F.linear(x, L["wv"])
        if c.clip_qkv is not None:
            q, k, v = (t.clamp(-c.clip_qkv, c.clip_qkv) for t in (q, k, v))
        q = q.view(B, T, c.heads, c.head_dim).transpose(1, 2)
        k = k.view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        v = v.view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            cache.k[li][:, :, start:start + T] = k
            cache.v[li][:, :, start:start + T] = v
            k, v = cache.k[li][:, :, :start + T], cache.v[li][:, :, :start + T]
        if c.kv_heads != c.heads:
            k = k.repeat_interleave(c.heads // c.kv_heads, dim=1)
            v = v.repeat_interleave(c.heads // c.kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(c.head_dim)
        scores = scores.masked_fill(~mask[:, None], float("-inf"))
        p = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        a = torch.matmul(p, v).transpose(1, 2).reshape(B, T, c.heads * c.head_dim)
        return F.linear(a, L["wo"])

    # forward() in three parts, so that several models (one per GPU) can step through the
    # layers together and share their MoE layers (see ep.py / serve.py).
    def begin(self, tokens, cache=None, valid=None):
        c, W = self.c, self.w
        B, T = tokens.shape
        dev = tokens.device
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        start = cache.length if cache is not None else 0
        base = cache.next_pos if cache is not None else torch.zeros(B, dtype=torch.int64, device=dev)
        positions = (base[:, None] + torch.cumsum(valid.long(), 1) - 1).clamp(min=0)
        if cache is not None:
            cache.valid[:, start:start + T] = valid
            keys_valid = cache.valid[:, :start + T]
        else:
            keys_valid = valid
        causal = torch.ones(T, start + T, dtype=torch.bool, device=dev).tril(start)
        mask = causal[None] & keys_valid[:, None, :]
        # A padding query has no valid key; let it attend to itself so no row is all -inf
        # (its output is never used).
        diag = torch.zeros(T, start + T, dtype=torch.bool, device=dev)
        diag[torch.arange(T, device=dev), start + torch.arange(T, device=dev)] = True
        mask = mask | (~mask.any(-1, keepdim=True) & diag[None])
        cos, sin = self._rope(positions, W["embed"].dtype)
        state = dict(B=B, T=T, cache=cache, start=start, mask=mask, cos=cos, sin=sin, base=base, valid=valid)
        return F.embedding(tokens, W["embed"]), state

    def attention_block(self, i, x, st):
        """x + attention(norm(x)); returns it and the normalized input of the MoE layer [B*T, H]."""
        L, c = self.w["layers"][i], self.c
        x = x + self._attention(L, rmsnorm(x, L["attn_norm"], c.rms_eps), st["cos"], st["sin"], st["cache"], i,
                                st["start"], st["mask"])
        return x, rmsnorm(x, L["mlp_norm"], c.rms_eps).reshape(st["B"] * st["T"], -1)

    def finish(self, x, st):
        cache = st["cache"]
        if cache is not None:
            cache.length = st["start"] + st["T"]
            cache.next_pos = st["base"] + st["valid"].long().sum(1)
        W = self.w
        return F.linear(rmsnorm(x, W["final_norm"], self.c.rms_eps), W["lm_head"])

    @torch.no_grad()
    def forward(self, tokens, cache=None, valid=None):
        """tokens [B, T]; valid [B, T] marks real tokens (False = left padding). With a
        cache, the tokens are appended after what it already holds."""
        x, st = self.begin(tokens, cache, valid)
        for i in range(self.c.layers):
            x, h = self.attention_block(i, x, st)
            x = x + self.moe(i, h).view(st["B"], st["T"], -1)
        return self.finish(x, st)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, eos=None):
        """Greedy decoding of a batch of token lists (left-padded internally). Returns the
        generated tokens of each prompt (stopping at eos)."""
        dev, dtype = self.w["embed"].device, self.w["embed"].dtype
        B, P = len(prompts), max(len(p) for p in prompts)
        tokens = torch.zeros(B, P, dtype=torch.int64, device=dev)
        valid = torch.zeros(B, P, dtype=torch.bool, device=dev)
        for b, p in enumerate(prompts):
            tokens[b, P - len(p):] = torch.tensor(p, device=dev)
            valid[b, P - len(p):] = True
        cache = Cache(self.c, B, P + max_new_tokens, dtype, dev)
        logits = self.forward(tokens, cache, valid)[:, -1]
        out = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    out[b].append(int(nxt[b]))
                    done[b] = eos is not None and int(nxt[b]) == eos
            if all(done):
                break
            logits = self.forward(nxt[:, None], cache)[:, -1]
        return out
