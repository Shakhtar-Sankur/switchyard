"""M0: the reference MoE model computes what Hugging Face transformers computes."""

import pytest
import torch

from switchyard import moe, weights
from switchyard.config import Config
from switchyard.model import Model
from tiny import checkpoint, prompts

SHAPES = {
    "plain": {},
    "norm_topk": {"norm_topk_prob": True},
    "gqa": {"heads": 4, "kv_heads": 2},
    "clip_qkv": {"clip_qkv": 0.3},
    "top4_of_16": {"experts": 16, "top_k": 4},
}


def ours(path, **kw):
    c = Config.load(path)
    return Model(c, weights.load(path, c), **kw)


@pytest.mark.parametrize("shape", SHAPES)
def test_logits_match_transformers(shape):
    path, hf = checkpoint(**SHAPES[shape])
    m = ours(path)
    for p in prompts(3):
        t = torch.tensor([p])
        with torch.no_grad():
            want = hf(t).logits
        got = m.forward(t)
        assert (got - want).abs().max().item() < 1e-5


def test_grouped_experts_equal_the_reference_loop():
    torch.manual_seed(0)
    E, H, I, N, k = 8, 16, 12, 40, 3
    h = torch.randn(N, H)
    router = torch.randn(E, H)
    gu, dn = torch.randn(E, 2 * I, H) * 0.3, torch.randn(E, H, I) * 0.3
    idx, w = moe.route(h, router, k, False)
    a = moe.experts_reference(h, idx, w, gu, dn)
    b = moe.experts_grouped(h, idx, w, gu, dn)
    assert torch.equal(a, b)  # same products, added in the same (expert) order


def test_plan_sorts_tokens_by_expert():
    idx = torch.tensor([[2, 0], [0, 1], [2, 1]])
    w = torch.tensor([[.6, .4], [.7, .3], [.5, .5]])
    p = moe.plan(idx, w, 4)
    assert p.counts.tolist() == [2, 2, 2, 0]
    assert p.offsets.tolist() == [0, 2, 4, 6, 6]
    assert p.token.tolist() == [0, 1, 1, 2, 0, 2]
    assert p.slot.tolist() == [1, 0, 1, 1, 0, 0]
    assert torch.allclose(p.weight, torch.tensor([.4, .7, .3, .5, .6, .5]))


@pytest.mark.parametrize("shape", ["plain", "gqa"])
def test_decoding_with_the_cache_equals_a_full_forward(shape):
    path, _ = checkpoint(**SHAPES[shape])
    m = ours(path)
    p = prompts(1, seed=3)[0]
    full = m.forward(torch.tensor([p + [5, 6, 7]]))
    from switchyard.model import Cache
    cache = Cache(m.c, 1, 32, torch.float32, "cpu")
    out = [m.forward(torch.tensor([p]), cache)]
    for t in (5, 6, 7):
        out.append(m.forward(torch.tensor([[t]]), cache))
    assert (torch.cat(out, 1) - full).abs().max().item() < 1e-5


def test_greedy_generation_matches_transformers_and_batching_changes_nothing():
    path, hf = checkpoint()
    m = ours(path)
    ps = prompts(4, seed=5)
    batched = m.generate(ps, 10)
    for p, got in zip(ps, batched):
        alone = m.generate([p], 10)[0]
        with torch.no_grad():
            want = hf.generate(torch.tensor([p]), max_new_tokens=10, do_sample=False, eos_token_id=None)[0, len(p):].tolist()
        assert got == alone == want


def test_loading_a_subset_of_experts():
    path, _ = checkpoint()
    c = Config.load(path)
    full = weights.load(path, c)
    part = weights.load(path, c, experts=[4, 5, 6, 7])
    for L, P in zip(full["layers"], part["layers"]):
        assert torch.equal(L["gate_up"][4:], P["gate_up"]) and torch.equal(L["down"][4:], P["down"])
        assert torch.equal(L["router"], P["router"])
