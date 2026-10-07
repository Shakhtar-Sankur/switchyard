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


# Below this many rows per expert the layer streams weights (decode): use the GEMV kernel.
GEMV_MAX_ROWS = 8


def experts(h, idx, w, gate_up, down, mode="auto"):
    """The experts of one MoE layer on this GPU, for routing (idx, w) computed already:
    sort by expert, SwiGLU projection reading rows of h in place, down projection scaled by
    the router weights, combine. idx int32 [N, k] in 0..E-1 local expert ids.
    mode: "gemv" (streams weights; for few rows per expert), "gemm" (tensor-core tiles),
    or "auto" (gemv when no expert has more than GEMV_MAX_ROWS rows)."""
    E = ext()
    N = h.shape[0]
    counts, offsets, token, wsorted, pos_of = E.sort_by_expert(idx.contiguous(), w.contiguous(), gate_up.shape[0])
    rows = idx.numel()
    if mode == "auto":
        mode = "gemv" if int(counts.max()) <= GEMV_MAX_ROWS else "gemm"
    if mode == "gemv":
        act = E.grouped_gemv(h.contiguous(), token, gate_up, offsets, rows, True, None)
        y = E.grouped_gemv(act, None, down, offsets, rows, False, wsorted)
    else:
        toff = tile_offsets(counts)
        total = int(toff[-1])
        act = E.grouped_gemm(h.contiguous(), token, gate_up, offsets, toff, total, rows, True, None)
        y = E.grouped_gemm(act, None, down, offsets, toff, total, rows, False, wsorted)
    return E.combine(y, pos_of, idx.contiguous(), N)


def moe_layer(h, router, gate_up, down, top_k, norm_topk_prob, mode="auto"):
    idx, w = route(h, router, top_k, norm_topk_prob)
    return experts(h, idx, w, gate_up, down, mode)
