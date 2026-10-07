"""M1 on a GPU: the CUDA kernels against the PyTorch reference (skipped without CUDA)."""

import pytest
import torch

from switchyard import moe

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def setup(N, E=64, k=8, H=256, I=128, seed=0, skew=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(N, H, device="cuda", generator=g).half()
    router = (torch.randn(E, H, device="cuda", generator=g) * 0.2)
    router[: max(1, E // 8)] += skew  # a few popular experts
    gu = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * H ** -0.5).half()
    dn = (torch.randn(E, H, I, device="cuda", generator=g) * I ** -0.5).half()
    return h, router.half(), gu, dn


@cuda
@pytest.mark.parametrize("E,k", [(8, 2), (64, 8), (60, 4), (128, 8)])
def test_route_matches_softmax_topk(E, k):
    from switchyard import kernels
    h, router, _, _ = setup(300, E=E, k=k)
    for norm in (False, True):
        idx, w = kernels.route(h, router, k, norm)
        ridx, rw = moe.route(h, router, k, norm)
        assert torch.equal(idx.long(), ridx)
        assert (w.float() - rw.float()).abs().max().item() < 1e-3


@cuda
@pytest.mark.parametrize("N", [1, 7, 64, 333, 2048])
@pytest.mark.parametrize("skew", [0.0, 2.0])
def test_moe_layer_matches_the_reference(N, skew):
    from switchyard import kernels
    h, router, gu, dn = setup(N, skew=skew)
    ridx, rw = moe.route(h, router, 8, False)
    want = moe.experts_reference(h.float(), ridx, rw.float(), gu.float(), dn.float())  # fp32 reference
    got = kernels.moe_layer(h, router, gu, dn, 8, False).float()
    half = moe.experts_reference(h, ridx, rw, gu, dn).float()                         # fp16, as transformers runs it
    err, err_half = (got - want).abs().max().item(), (half - want).abs().max().item()
    assert err <= max(2 * err_half, 2e-3), (err, err_half)


@cuda
def test_sort_is_stable_and_deterministic():
    from switchyard import kernels
    h, router, _, _ = setup(999)
    idx, w = kernels.route(h, router, 8, False)
    a = kernels.ext().sort_by_expert(idx, w, 64)
    b = kernels.ext().sort_by_expert(idx, w, 64)
    p = moe.plan(idx.long(), w, 64)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(a[2].long(), p.token)
    assert torch.equal(a[0].long(), p.counts)


@cuda
def test_empty_experts_and_a_single_token():
    from switchyard import kernels
    h, router, gu, dn = setup(1, E=64, k=8)
    out = kernels.moe_layer(h, router, gu, dn, 8, False)
    ridx, rw = moe.route(h, router, 8, False)
    want = moe.experts_reference(h.float(), ridx, rw.float(), gu.float(), dn.float())
    assert (out.float() - want).abs().max().item() < 2e-3
