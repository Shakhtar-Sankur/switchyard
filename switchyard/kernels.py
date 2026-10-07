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


def experts(h, idx, w, gate_up, down):
    """The experts of one MoE layer on this GPU, for routing (idx, w) computed already:
    sort by expert, grouped SwiGLU GEMM reading rows of h in place, grouped down GEMM scaled
    by the router weights, combine. idx int32 [N, k] in 0..E-1 local expert ids."""
    E = ext()
    N = h.shape[0]
    counts, offsets, token, wsorted, pos_of = E.sort_by_expert(idx.contiguous(), w.contiguous(), gate_up.shape[0])
    toff = tile_offsets(counts)
    total = int(toff[-1])
    rows = idx.numel()
    act = E.grouped_gemm(h.contiguous(), token, gate_up, offsets, toff, total, rows, True, None)
    y = E.grouped_gemm(act, None, down, offsets, toff, total, rows, False, wsorted)
    return E.combine(y, pos_of, idx.contiguous(), N)


def moe_layer(h, router, gate_up, down, top_k, norm_topk_prob):
    idx, w = route(h, router, top_k, norm_topk_prob)
    return experts(h, idx, w, gate_up, down)
