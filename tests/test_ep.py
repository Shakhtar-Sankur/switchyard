"""M2/M3 on the CPU: the expert-parallel protocol (route, sort, dispatch, experts, return,
combine) with several ranks gives exactly the single-device result."""

import math

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
    threaded = ep.layer_threaded(hs, [router] * R, [gu[r * per:(r + 1) * per] for r in range(R)],
                                 [dn[r * per:(r + 1) * per] for r in range(R)])
    assert all(torch.equal(a, b) for a, b in zip(outs, threaded))


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


def test_threaded_and_single_threaded_forward_agree():
    path, _ = checkpoint(experts=16, top_k=4)
    two = ParallelModel(path, ["cpu", "cpu"], dtype=torch.float32)
    toks = [torch.tensor([p[:7]]) for p in prompts(2, seed=4)]
    a = two.forward(toks, threaded=True)
    b = two.forward(toks, threaded=False)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_a_failing_rank_does_not_hang_the_others():
    ep = ExpertParallel(["cpu", "cpu"], 4, 2, False)

    def fn(r):
        if r == 1:
            raise ValueError("boom")
        ep.barrier.wait()

    with pytest.raises(Exception):
        ep.run_ranks(fn)
    assert ep.run_ranks(lambda r: r) == [0, 1]  # usable again


def _dropped_reference(h, router, gu, dn, k, cf):
    idx, w = moe.route(h, router, k, False)
    E, M = router.shape[0], idx.numel()
    cap = math.ceil(cf * M / E)
    w = w.clone()
    seen = [0] * E
    for t in range(idx.shape[0]):
        for s in range(k):
            e = int(idx[t, s])
            if seen[e] >= cap:
                w[t, s] = 0
            seen[e] += 1
    return moe.experts_reference(h, idx, w, gu, dn), sum(max(0, c - cap) for c in seen)


@pytest.mark.parametrize("cf", [0.5, 1.0, 1.25, 100.0])
def test_capacity_drops_the_rows_past_each_experts_capacity_in_token_order(cf):
    torch.manual_seed(1)
    E, H, I, k, R = 16, 24, 12, 4, 2
    router = torch.randn(E, H) * 2  # skewed enough to overflow some experts
    gu, dn = torch.randn(E, 2 * I, H) * 0.3, torch.randn(E, H, I) * 0.3
    hs = [torch.randn(n, H) for n in (30, 17)]
    per = E // R
    ep = ExpertParallel(["cpu"] * R, E, k, False)
    ep.capacity_factor = cf
    outs = ep.layer(hs, [router] * R, [gu[r * per:(r + 1) * per] for r in range(R)],
                    [dn[r * per:(r + 1) * per] for r in range(R)])
    dropped = 0
    for h, o in zip(hs, outs):
        want, d = _dropped_reference(h, router, gu, dn, k, cf)
        dropped += d
        assert torch.allclose(o, want, atol=1e-5)
    assert ep.stats["dropped_rows"] == dropped
    assert ep.stats["routed_rows"] == sum(h.shape[0] for h in hs) * k
    assert (dropped == 0) == (cf == 100.0)


def test_trace_counts_every_routed_row():
    path, _ = checkpoint(experts=16, top_k=4)
    two = ParallelModel(path, ["cpu", "cpu"], dtype=torch.float32)
    two.ep.trace = []
    toks = [torch.tensor([p[:9]]) for p in prompts(2, seed=6)]
    two.forward(toks, threaded=False)
    c = Config.load(path)
    assert len(two.ep.trace) == c.layers * 2
    for r, counts in two.ep.trace:
        assert int(counts.sum()) == 9 * 4


def test_a_permuted_placement_computes_the_same_model():
    from switchyard import balance
    path, _ = checkpoint(experts=16, top_k=4)
    c = Config.load(path)
    one = Model(c, weights.load(path, c))
    g = torch.Generator().manual_seed(0)
    placement = [torch.randperm(16, generator=g).tolist() for _ in range(c.layers)]
    two = ParallelModel(path, ["cpu", "cpu"], dtype=torch.float32, placement=placement)
    toks = [torch.tensor([p[:8]]) for p in prompts(2, seed=8)]
    for t, got in zip(toks, two.forward(toks)):
        assert (got - one.forward(t)).abs().max().item() < 1e-5


def test_greedy_placement_balances_a_skewed_load():
    from switchyard import balance
    load = torch.tensor([100.0, 90, 80, 70] + [10.0] * 12)
    assert balance.imbalance(balance.rank_loads(load, balance.contiguous(16), 2)) > 1.3
    p = balance.greedy(load, 2)
    assert sorted(p) == list(range(16))
    assert balance.imbalance(balance.rank_loads(load, p, 2)) < 1.03
