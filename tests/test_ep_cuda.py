"""M2 on GPUs: the expert-parallel layer with the CUDA kernels and peer-to-peer dispatch is
bit-identical to the same kernels on one GPU (skipped without CUDA; two-GPU cases need two)."""

import pytest
import torch

from switchyard.ep import CudaOps, ExpertParallel
from switchyard.serve import ParallelModel
from tiny import checkpoint, prompts

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
two = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")


def weights(E=64, H=256, I=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    router = (torch.randn(E, H, device="cuda", generator=g) * 0.2).half()
    gu = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * H ** -0.5).half()
    dn = (torch.randn(E, H, I, device="cuda", generator=g) * I ** -0.5).half()
    return router, gu, dn


def run(devices, sizes, mode, E=64, k=8):
    from switchyard import kernels
    router, gu, dn = weights(E)
    R, per = len(devices), E // len(devices)
    g = torch.Generator(device="cuda").manual_seed(1)
    hs0 = [torch.randn(n, router.shape[1], device="cuda", generator=g).half() for n in sizes]
    hs = [h.to(d) for h, d in zip(hs0, devices)]
    ep = ExpertParallel(devices, E, k, False, ops=CudaOps(mode))
    outs = ep.layer(hs, [router.to(d) for d in devices], [gu[r * per:(r + 1) * per].to(devices[r]) for r in range(R)],
                    [dn[r * per:(r + 1) * per].to(devices[r]) for r in range(R)])
    threaded = ep.layer_threaded(hs, [router.to(d) for d in devices],
                                 [gu[r * per:(r + 1) * per].to(devices[r]) for r in range(R)],
                                 [dn[r * per:(r + 1) * per].to(devices[r]) for r in range(R)])
    for d in devices:
        torch.cuda.synchronize(d)
    for h, o, t in zip(hs0, outs, threaded):
        want = kernels.moe_layer(h, router, gu, dn, k, False, mode)
        assert torch.equal(o.to("cuda:0"), want)
        assert torch.equal(t.to("cuda:0"), want)
    return ep


@cuda
@pytest.mark.parametrize("mode", ["gemv", "gemm"])
@pytest.mark.parametrize("sizes", [(1, 1), (7, 30), (0, 5), (200, 200)])
def test_two_ranks_on_one_gpu(mode, sizes):
    run(["cuda:0", "cuda:0"], sizes, mode)


@two
@pytest.mark.parametrize("mode", ["gemv", "gemm"])
@pytest.mark.parametrize("sizes", [(1, 1), (7, 30), (0, 5), (200, 200), (1000, 3)])
def test_two_gpus_peer_to_peer(mode, sizes):
    ep = run(["cuda:0", "cuda:1"], sizes, mode)
    assert ep.peer
    if sum(sizes) > 2:
        assert ep.stats["dispatched_rows"] > 0


def _served(devices):
    path, _ = checkpoint(experts=16, top_k=4, layers=6)
    pm = ParallelModel(path, devices, dtype=torch.float16)
    return pm, prompts(8, seed=5)


def _threaded_generate_matches(devices):
    # Many layers and decode steps, every GPU's thread allocating and freeing while the other
    # sends to it: the case a one-layer test is too short to race in.
    pm, ps = _served(devices)
    pm.threaded = False
    want = pm.generate(ps, 24)
    pm.threaded = True
    for _ in range(3):
        assert pm.generate(ps, 24) == want


@cuda
def test_threaded_serving_on_one_gpu_matches_one_thread():
    _threaded_generate_matches(["cuda:0", "cuda:0"])


@two
def test_threaded_serving_on_two_gpus_matches_one_thread():
    _threaded_generate_matches(["cuda:0", "cuda:1"])


@cuda
@pytest.mark.parametrize("threaded", [False, True])
def test_the_layer_waits_for_its_input(threaded):
    # h is written on the caller's stream after a long GPU sleep; a layer that does not order
    # its stream after the caller's reads the old contents (zeros).
    from switchyard import kernels
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else ["cuda:0", "cuda:0"]
    router, gu, dn = weights(64)
    g = torch.Generator(device="cuda").manual_seed(3)
    real = [torch.randn(16, router.shape[1], device="cuda", generator=g).half() for _ in devices]
    torch.cuda.synchronize()
    hs = [torch.zeros(16, router.shape[1], dtype=torch.half, device=d) for d in devices]
    ep = ExpertParallel(devices, 64, 8, False, ops=CudaOps("gemv"))
    args = ([router.to(d) for d in devices], [gu[r * 32:(r + 1) * 32].to(d) for r, d in enumerate(devices)],
            [dn[r * 32:(r + 1) * 32].to(d) for r, d in enumerate(devices)])
    for d in devices:
        torch.cuda.synchronize(d)
    for h, x, d in zip(hs, real, devices):
        with torch.cuda.device(d):
            torch.cuda._sleep(200_000_000)  # ~0.1 s at T4 clocks
            h.copy_(x)
    outs = ep.layer_threaded(hs, *args) if threaded else ep.layer(hs, *args)
    for d in devices:
        torch.cuda.synchronize(d)
    for x, o in zip(real, outs):
        assert torch.equal(o.to("cuda:0"), kernels.moe_layer(x, router, gu, dn, 8, False, "gemv"))
