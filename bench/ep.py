"""One OLMoE MoE layer (64 experts, top 8, 2048 -> 1024) with its experts split over two GPUs,
N tokens on each GPU. Compared with:
  one_gpu   the whole layer (all 64 experts) on one GPU, for 2N tokens: no communication;
  nccl      the same expert parallelism with PyTorch's all_to_all on NCCL as the transport
            (bench/ep_nccl.py, two processes), same kernels for everything else.
Prints one JSON line per N."""

import json
import sys

import torch

from switchyard import kernels
from switchyard.ep import ExpertParallel


def timed(fn, devices, reps=30, warmup=5):
    import time
    for _ in range(warmup):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        for d in devices:
            torch.cuda.synchronize(d)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main(sizes=(1, 4, 16, 64, 256, 1024), E=64, k=8, H=2048, I=1024):
    devs = ["cuda:0", "cuda:1"]
    g = torch.Generator(device="cuda").manual_seed(0)
    router = (torch.randn(E, H, device="cuda", generator=g) * 0.02).half()
    gu = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * H ** -0.5).half()
    dn = (torch.randn(E, H, I, device="cuda", generator=g) * I ** -0.5).half()
    per = E // 2
    routers = [router.to(d) for d in devs]
    gus = [gu[r * per:(r + 1) * per].to(devs[r]) for r in range(2)]
    dns = [dn[r * per:(r + 1) * per].to(devs[r]) for r in range(2)]
    ep = ExpertParallel(devs, E, k, False)
    print(json.dumps({"bench": "ep_layer", "gpus": 2, "peer_access": ep.peer, "experts": E, "top_k": k}), flush=True)
    for N in sizes:
        hs = [torch.randn(N, H, device=d, generator=None).half() for d in devs]
        both = torch.cat([hs[0], hs[1].to("cuda:0")])
        rec = {"tokens_per_gpu": N,
               "switchyard_ep_ms": timed(lambda: ep.layer(hs, routers, gus, dns), devs),
               "one_gpu_all_experts_ms": timed(lambda: kernels.moe_layer(both, router, gu, dn, k, False), ["cuda:0"])}
        ep.stats = {"dispatched_rows": 0, "local_rows": 0}
        ep.layer(hs, routers, gus, dns)
        rec["rows_sent_to_peer"] = ep.stats["dispatched_rows"]
        rec["rows_kept_local"] = ep.stats["local_rows"]
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(*(tuple(int(x) for x in sys.argv[1].split(",")),) if len(sys.argv) > 1 else ())
