"""M2/M3 on the CPU: the expert-parallel protocol (route, sort, dispatch, experts, return,
combine) with several ranks gives exactly the single-device result."""

import pytest
import torch

from switchyard import moe
from switchyard.ep import ExpertParallel, TorchOps
from switchyard.model import Model
from switchyard.config import Config
from switchyard import weights
from switchyard.serve import ParallelModel
from tiny import checkpoint, prompts


def single(h, router, gu, dn, k, norm):
    ops = TorchOps()
    idx, w = ops.route(h, router, k, norm)
    counts, offsets, token, wsorted, pos_of = ops.sort(idx, w, gu.shape[0])
    y = ops.ffn(h, token, offsets, counts, wsorted, gu, dn, idx.numel())
    return ops.combine(y, pos_of, idx, h.shape[0])


@pytest.mark.parametrize("R", [1, 2, 4])
@pytest.mark.parametrize("sizes", [(5, 9, 1, 13), (0, 3, 7, 2), (40, 40, 40, 40)])
def test_expert_parallel_layer_is_bit_identical_to_one_device(R, sizes):
    torch.manual_seed(0)
    E, H, I, k = 16, 24, 12, 4
    router = torch.randn(E, H)
    gu, dn = torch.randn(E, 2 * I, H) * 0.3, torch.randn(E, H, I) * 0.3
    hs = [torch.randn(n, H) for n in sizes[:R]]
    ep = ExpertParallel(["cpu"] * R, E, k, False)
    per = E // R
    outs = ep.layer(hs, [router] * R, [gu[r * per:(r + 1) * per] for r in range(R)],
                    [dn[r * per:(r + 1) * per] for r in range(R)])
    for h, o in zip(hs, outs):
        assert torch.equal(o, single(h, router, gu, dn, k, False))
        if h.shape[0]:
            idx, w = moe.route(h, router, k, False)
            assert torch.allclose(o, moe.experts_reference(h, idx, w, gu, dn), atol=1e-5)
    if R > 1 and sum(sizes[:R]) > 4:
        assert ep.stats["dispatched_rows"] > 0  # tokens really crossed ranks


def test_a_model_served_over_two_ranks_generates_what_one_model_does():
    path, _ = checkpoint(experts=16, top_k=4)
    c = Config.load(path)
    one = Model(c, weights.load(path, c))
    two = ParallelModel(path, ["cpu", "cpu"], dtype=torch.float32)
    ps = prompts(5, seed=11)
    assert two.generate(ps, 8) == one.generate(ps, 8)


def test_logits_over_four_ranks_match_one_model():
    path, _ = checkpoint(experts=16, top_k=4)
    c = Config.load(path)
    one = Model(c, weights.load(path, c))
    four = ParallelModel(path, ["cpu"] * 4, dtype=torch.float32)
    ps = [p[:6] for p in prompts(4, seed=2)]
    toks = [torch.tensor([p]) for p in ps]
    for t, got in zip(toks, four.forward(toks)):
        assert (got - one.forward(t)).abs().max().item() < 1e-5
