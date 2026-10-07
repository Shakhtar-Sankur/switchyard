"""The baseline transport: the same expert-parallel MoE layer as switchyard.ep, but tokens
travel with torch.distributed.all_to_all_single on NCCL (two processes, one per GPU), the way
most PyTorch MoE code does it. Same routing, sorting, expert and combine kernels.
Prints one JSON line per N (tokens per GPU), from rank 0."""

import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def layer(h, router, gu, dn, k, E, R, rank):
    from switchyard import kernels
    ext = kernels.ext()
    per = E // R
    idx, w = kernels.route(h, router, k, False)
    counts, offsets, token, wsorted, pos_of = ext.sort_by_expert(idx, w, E)
    off = offsets.tolist()
    send = [off[(d + 1) * per] - off[d * per] for d in range(R)]
    send_t = torch.tensor(send, device=h.device)
    recv_t = torch.empty_like(send_t)
    dist.all_to_all_single(recv_t, send_t)                       # how many rows each rank sends
    recv = recv_t.tolist()
    x = h[token.long()]                                          # rows in sorted order
    xr = torch.empty(sum(recv), h.shape[1], dtype=h.dtype, device=h.device)
    dist.all_to_all_single(xr, x, recv, send)
    wr = torch.empty(sum(recv), dtype=wsorted.dtype, device=h.device)
    dist.all_to_all_single(wr, wsorted, recv, send)
    cnt = counts.view(R, per).contiguous()
    cr = torch.empty_like(cnt)
    dist.all_to_all_single(cr, cnt)                              # per-expert counts of what arrives
    ys = []
    start = 0
    for s in range(R):                                           # each source's rows, by local expert
        n = recv[s]
        c = cr[s]
        o = torch.zeros(per + 1, dtype=torch.int32, device=h.device)
        o[1:] = torch.cumsum(c, 0)
        mode = kernels.pick(int(c.max()) if n else 0)
        ys.append(kernels.ffn(xr[start:start + n], None, o, c, wr[start:start + n], gu, dn, n, mode))
        start += n
    yr = torch.cat(ys) if ys else xr
    y = torch.empty(x.shape[0], h.shape[1], dtype=h.dtype, device=h.device)
    dist.all_to_all_single(y, yr, send, recv)                    # results home, in sorted order
    return ext.combine(y, pos_of, idx, h.shape[0])


def worker(rank, R, sizes, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=R)
    E, k, H, I = 64, 8, 2048, 1024
    g = torch.Generator(device="cuda").manual_seed(0)
    router = (torch.randn(E, H, device="cuda", generator=g) * 0.02).half()
    gu = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * H ** -0.5).half()
    dn = (torch.randn(E, H, I, device="cuda", generator=g) * I ** -0.5).half()
    per = E // R
    gu, dn = gu[rank * per:(rank + 1) * per].contiguous(), dn[rank * per:(rank + 1) * per].contiguous()
    for N in sizes:
        h = torch.randn(N, H, device="cuda").half()
        fn = lambda: layer(h, router, gu, dn, k, E, R, rank)
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            dist.barrier()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        if rank == 0:
            print(json.dumps({"tokens_per_gpu": N, "nccl_all_to_all_ms": ts[len(ts) // 2]}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    sizes = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1, 4, 16, 64, 256, 1024)
    mp.spawn(worker, args=(2, sizes, 29511), nprocs=2, join=True)
