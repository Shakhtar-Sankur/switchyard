"""One OLMoE-1B-7B MoE layer (64 experts, top 8, hidden 2048, expert width 1024) on one GPU:
the per-expert loop transformers uses, the same loop over tokens sorted by expert, and
switchyard's kernels. Prints one JSON line per batch size."""

import json
import sys

import torch

from switchyard import kernels, moe


def timed(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def main(sizes=(1, 8, 32, 128, 512, 2048, 4096), E=64, k=8, H=2048, I=1024):
    g = torch.Generator(device="cuda").manual_seed(0)
    router = (torch.randn(E, H, device="cuda", generator=g) * 0.02).half()
    gu = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * H ** -0.5).half()
    dn = (torch.randn(E, H, I, device="cuda", generator=g) * I ** -0.5).half()
    print(json.dumps({"bench": "moe_layer", "gpu": torch.cuda.get_device_name(), "experts": E, "top_k": k,
                      "hidden": H, "expert_width": I}), flush=True)
    for N in sizes:
        h = torch.randn(N, H, device="cuda", generator=g).half()
        idx, w = moe.route(h, router, k, False)
        ref = lambda: moe.experts_reference(h, idx, w, gu, dn)
        grp = lambda: moe.experts_grouped(h, idx, w, gu, dn)
        ours = lambda: kernels.moe_layer(h, router, gu, dn, k, False)
        gemv = lambda: kernels.moe_layer(h, router, gu, dn, k, False, "gemv")
        gemm = lambda: kernels.moe_layer(h, router, gu, dn, k, False, "gemm")
        rec = {"tokens": N, "loop_ms": timed(ref), "sorted_loop_ms": timed(grp), "switchyard_ms": timed(ours),
               "gemv_ms": timed(gemv), "gemm_ms": timed(gemm)}
        want = moe.experts_reference(h.float(), idx, w.float(), gu.float(), dn.float())
        rec["max_err_switchyard"] = (ours().float() - want).abs().max().item()
        rec["max_err_fp16_loop"] = (ref().float() - want).abs().max().item()
        rec["speedup_vs_loop"] = rec["loop_ms"] / rec["switchyard_ms"]
        rec["speedup_vs_sorted_loop"] = rec["sorted_loop_ms"] / rec["switchyard_ms"]
        rec["max_rows_per_expert"] = int(torch.bincount(idx.flatten(), minlength=E).max())
        # Bytes of expert weights touched (each used expert read once) over the kernel time.
        used = int((torch.bincount(idx.flatten(), minlength=E) > 0).sum())
        rec["experts_used"] = used
        rec["weight_GBps"] = used * 3 * H * I * 2 / (rec["switchyard_ms"] / 1e3) / 1e9
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(*(tuple(int(x) for x in sys.argv[1].split(",")),) if len(sys.argv) > 1 else ())
