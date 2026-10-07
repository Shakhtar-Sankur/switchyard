"""A mixture-of-experts decoder's shape, read from a Hugging Face config.json (OLMoE)."""

import json
import os
from dataclasses import dataclass


@dataclass
class Config:
    vocab: int
    hidden: int
    intermediate: int          # per expert
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    norm_topk_prob: bool
    rms_eps: float
    rope_theta: float
    clip_qkv: float | None
    tie_embeddings: bool
    max_positions: int

    @classmethod
    def load(cls, path):
        c = json.load(open(os.path.join(path, "config.json")))
        heads = c["num_attention_heads"]
        rope = c.get("rope_parameters") or {}
        return cls(
            vocab=c["vocab_size"],
            hidden=c["hidden_size"],
            intermediate=c["intermediate_size"],
            layers=c["num_hidden_layers"],
            heads=heads,
            kv_heads=c.get("num_key_value_heads") or heads,
            head_dim=c.get("head_dim") or c["hidden_size"] // heads,
            experts=c["num_experts"],
            top_k=c["num_experts_per_tok"],
            norm_topk_prob=bool(c.get("norm_topk_prob", False)),
            rms_eps=c.get("rms_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", c.get("rope_theta", 10000.0)),
            clip_qkv=c.get("clip_qkv"),
            tie_embeddings=bool(c.get("tie_word_embeddings", False)),
            max_positions=c.get("max_position_embeddings", 4096),
        )
