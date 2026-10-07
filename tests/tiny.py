"""Tiny random OLMoE checkpoints written by Hugging Face transformers, for tests."""

import os
import tempfile

import torch

_CACHE = {}


def checkpoint(experts=8, top_k=2, heads=4, kv_heads=4, norm_topk_prob=False, clip_qkv=None, layers=2, seed=0):
    key = (experts, top_k, heads, kv_heads, norm_topk_prob, clip_qkv, layers, seed)
    if key in _CACHE:
        return _CACHE[key]
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(seed)
    cfg = OlmoeConfig(vocab_size=131, hidden_size=64, intermediate_size=32, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, num_experts=experts,
                      num_experts_per_tok=top_k, norm_topk_prob=norm_topk_prob, clip_qkv=clip_qkv,
                      max_position_embeddings=256, eos_token_id=1, pad_token_id=0, bos_token_id=2)
    m = OlmoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif "gate.weight" in name:      # the router: spread enough to pick distinct experts
                p.normal_(0, 1.0)
            else:
                p.normal_(0, 0.08)
    path = tempfile.mkdtemp(prefix="switchyard-tiny-")
    m.save_pretrained(path)
    _CACHE[key] = (path, m)
    return path, m


def prompts(n=4, vocab=131, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, vocab, (int(torch.randint(6, 24, (1,), generator=g)),), generator=g).tolist() for _ in range(n)]
