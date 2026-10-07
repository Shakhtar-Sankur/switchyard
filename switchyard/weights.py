"""Loads an OLMoE checkpoint (safetensors, one file or sharded) into plain tensors.

Experts are stacked per layer: gate_up [E, 2I, H] (gate rows first, then up) and
down [E, H, I]. `experts` selects which experts to load, so that an expert-parallel
rank holds only its own share of the expert weights; everything else (attention,
norms, router, embeddings) is loaded on every rank."""

import json
import os

import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, path):
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            self.where = json.load(open(index))["weight_map"]
        else:
            with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}
        self._open = {}

    def get(self, name):
        file = self.where[name]
        if file not in self._open:
            self._open[file] = safe_open(os.path.join(self.path, file), "pt")
        return self._open[file].get_tensor(name)

    def has(self, name):
        return name in self.where


def load(path, config, dtype=torch.float32, device="cpu", experts=None):
    """Returns {"embed", "final_norm", "lm_head", "layers": [dict per layer]}, with the
    experts in `experts` (default: all) stacked in that order."""
    ck = Checkpoint(path)
    experts = list(range(config.experts)) if experts is None else list(experts)

    def t(name):
        return ck.get(name).to(device=device, dtype=dtype)

    out = {"embed": t("model.embed_tokens.weight"), "final_norm": t("model.norm.weight")}
    out["lm_head"] = out["embed"] if config.tie_embeddings or not ck.has("lm_head.weight") else t("lm_head.weight")
    layers = []
    for i in range(config.layers):
        p = f"model.layers.{i}."
        L = {
            "attn_norm": t(p + "input_layernorm.weight"),
            "mlp_norm": t(p + "post_attention_layernorm.weight"),
            "wq": t(p + "self_attn.q_proj.weight"),
            "wk": t(p + "self_attn.k_proj.weight"),
            "wv": t(p + "self_attn.v_proj.weight"),
            "wo": t(p + "self_attn.o_proj.weight"),
            "q_norm": t(p + "self_attn.q_norm.weight"),
            "k_norm": t(p + "self_attn.k_norm.weight"),
            "router": t(p + "mlp.gate.weight"),
        }
        e = p + "mlp.experts."
        L["gate_up"] = torch.stack([torch.cat([t(f"{e}{j}.gate_proj.weight"), t(f"{e}{j}.up_proj.weight")]) for j in experts])
        L["down"] = torch.stack([t(f"{e}{j}.down_proj.weight") for j in experts])
        layers.append(L)
    out["layers"] = layers
    out["experts"] = experts
    return out
