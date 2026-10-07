"""The CUDA kernels (csrc/moe_kernels.cu), built on first use with PyTorch's extension
loader, and the MoE layer assembled from them."""

import os

import torch
import torch.nn.functional as F

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        _EXT = load(name="switchyard_moe", sources=[os.path.join(here, "csrc", "moe_kernels.cu")],
                    extra_cuda_cflags=["-O3"], verbose=os.environ.get("SWITCHYARD_VERBOSE_BUILD") == "1")
    return _EXT


def tile_offsets(counts, block=64):
    tiles = (counts + block - 1) // block
    off = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    off[1:] = torch.cumsum(tiles, 0)
    return off


def route(h, router, top_k, norm_topk_prob):
    """Same as moe.route, fused: returns (indices int32 [N, k], weights fp16 [N, k])."""
    return ext().route(F.linear(h, router).contiguous(), top_k, norm_topk_prob)


# Measured on a T4 with OLMoE's shapes (bench/kernels.py): the GEMV kernel wins while no
# expert has more than 16 rows (decode), the tensor-core GEMM up to about 48, and cuBLAS's
# GEMM, one call per expert on the sorted rows, beyond that (prefill).
GEMV_MAX_ROWS = 16
GEMM_MAX_ROWS = 48


def pick(max_rows):
    if max_rows <= GEMV_MAX_ROWS:
        return "gemv"
    return "gemm" if max_rows <= GEMM_MAX_ROWS else "cublas"


def ffn(x, a_rows, offsets, counts, wsorted, gate_up, down, rows, mode):
    """The experts over rows already sorted by expert: row i is x[a_rows[i]] (or x[i] when
    a_rows is None); expert e owns rows offsets[e]:offsets[e+1]. Returns each row's expert
    output scaled by its router weight, [rows, H]."""
    E = ext()
    if rows == 0:
        return torch.empty(0, down.shape[1], dtype=x.dtype, device=x.device)
    if mode == "gemv":
        act = E.grouped_gemv(x.contiguous(), a_rows, gate_up, offsets, rows, True, None)
        return E.grouped_gemv(act, None, down, offsets, rows, False, wsorted)
    if mode == "gemm":
        toff = tile_offsets(counts)
        total = int(toff[-1])
        act = E.grouped_gemm(x.contiguous(), a_rows, gate_up, offsets, toff, total, rows, True, None)
        return E.grouped_gemm(act, None, down, offsets, toff, total, rows, False, wsorted)
    # cublas: one GEMM pair per expert on its contiguous slice
    xs = x[a_rows.long()] if a_rows is not None else x
    y = torch.empty(rows, down.shape[1], dtype=x.dtype, device=x.device)
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            gate, up = F.linear(xs[start:start + n], gate_up[e]).chunk(2, dim=-1)
            y[start:start + n] = F.linear(F.silu(gate) * up, down[e]) * wsorted[start:start + n, None]
        start += n
    return y


def experts(h, idx, w, gate_up, down, mode="auto"):
    """The experts of one MoE layer on this GPU, for routing (idx, w) computed already:
    sort by expert, SwiGLU projection reading rows of h in place, down projection scaled by
    the router weights, combine. idx int32 [N, k] in 0..E-1 local expert ids.
    mode: "gemv" (streams weights; few rows per expert), "gemm" (tensor-core tiles),
    "cublas" (cuBLAS per expert), or "auto" (by the largest expert's row count)."""
    E = ext()
    N = h.shape[0]
    counts, offsets, token, wsorted, pos_of = E.sort_by_expert(idx.contiguous(), w.contiguous(), gate_up.shape[0])
    if mode == "auto":
        mode = pick(int(counts.max()))
    y = ffn(h, token, offsets, counts, wsorted, gate_up, down, idx.numel(), mode)
    return E.combine(y, pos_of, idx.contiguous(), N)


def moe_layer(h, router, gate_up, down, top_k, norm_topk_prob, mode="auto"):
    idx, w = route(h, router, top_k, norm_topk_prob)
    return experts(h, idx, w, gate_up, down, mode)
