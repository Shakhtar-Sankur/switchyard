"""Who is off, FlashInfer or the test's float16 reference? Runs vLLM's
test_flashinfer_prefill_with_paged_kv cases on the GPU, capturing FlashInfer's output and the
test's reference, and recomputes the reference from the same inputs in float64 (the reference
rounds attention scores to float32 with .float(), still far finer than float16).
Run from a vLLM checkout with tests importable (see kaggle_vllm_turing_fp64.sh)."""
import json
import sys

import torch

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("test_flashinfer", "tests/kernels/attention/test_flashinfer.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

captured = {}
original_ref = T.ref_paged_attn


def recording_ref(**kw):
    """The test's reference as is, plus the same computation in float64 on a compact copy of the
    cache: only the blocks the block tables use (the test's cache has 32,768 blocks, about 126
    of which are read), so the float64 copy fits next to the test's tensors."""
    bt = kw["block_tables"]
    used = torch.unique(bt.flatten().long())
    small = dict(kw)
    small["query"] = kw["query"].clone().double()  # the reference scales its query in place
    small["key_cache"] = kw["key_cache"][used].double()
    small["value_cache"] = kw["value_cache"][used].double()
    small["block_tables"] = torch.searchsorted(used, bt.long()).to(bt.dtype)
    captured["f64"] = original_ref(**small)
    out = original_ref(**kw)
    captured["ref"] = out
    return out


def recording_close(actual, expected, atol, rtol):
    captured.update(out=actual, atol=atol, rtol=rtol)


T.ref_paged_attn = recording_ref
T.torch.testing.assert_close = recording_close


def over(x, truth, atol, rtol):
    return int((torch.abs(x.double() - truth) > atol + rtol * torch.abs(truth)).sum())


SEQ_LENS = [(1, 1328), (5, 18), (129, 463)]  # (query tokens, kv tokens) per sequence, as in the test


def per_sequence(err):
    """Max error in each sequence's rows of the [tokens, heads, dim] output."""
    out, start = [], 0
    for q, _ in SEQ_LENS:
        out.append(round(float(err[start:start + q].max()), 5))
        start += q
    return out


def sdpa_fp16(query, key_cache, value_cache, block_tables, scale, soft_cap, sliding_window, **_):
    """PyTorch's own float16 attention on the same inputs (no soft cap: only cases without one)."""
    outs, start = [], 0
    _, block_size, kv_heads, head = key_cache.shape
    for i, (ql, kl) in enumerate(SEQ_LENS):
        nb = (kl + block_size - 1) // block_size
        idx = block_tables[i, :nb].long()
        k = key_cache[idx].reshape(-1, kv_heads, head)[:kl].transpose(0, 1)
        v = value_cache[idx].reshape(-1, kv_heads, head)[:kl].transpose(0, 1)
        q = query[start:start + ql].transpose(0, 1)
        rep = q.shape[0] // kv_heads
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        pos_q = torch.arange(kl - ql, kl, device=q.device)[:, None]
        pos_k = torch.arange(kl, device=q.device)[None, :]
        allowed = pos_k <= pos_q
        if sliding_window is not None:
            allowed &= pos_k > pos_q - sliding_window
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scale)
        outs.append(o.transpose(0, 1))
        start += ql
    return torch.cat(outs)


inputs = {}
original_ref_for_inputs = recording_ref


def recording_ref_with_inputs(**kw):
    inputs.clear()
    inputs.update(kw)
    inputs["query"] = kw["query"].clone()  # the reference scales its query in place; the caches are only read
    return original_ref_for_inputs(**kw)


T.ref_paged_attn = recording_ref_with_inputs

CASES = [(None, None, 32, 128), (None, None, 16, 128), (64, None, 32, 128), (None, None, 32, 256)]
for sliding_window, soft_cap, block_size, head_size in CASES:
    captured.clear()
    T.test_flashinfer_prefill_with_paged_kv(
        seq_lens=SEQ_LENS, num_heads=(32, 8), head_size=head_size,
        dtype=torch.float16, block_size=block_size, soft_cap=soft_cap, sliding_window=sliding_window)
    t = captured["f64"]
    out, ref = captured["out"].double(), captured["ref"].double()
    sd = sdpa_fp16(**inputs).double()
    worst = torch.argmax(torch.abs(out - t))
    print(json.dumps({
        "case": f"window={sliding_window} page={block_size} head={head_size}",
        "max_abs_output": round(float(torch.abs(t).max()), 3),
        "flashinfer_err_per_sequence": per_sequence(torch.abs(out - t)),
        "fp16_reference_err_per_sequence": per_sequence(torch.abs(ref - t)),
        "torch_sdpa_fp16_err_per_sequence": per_sequence(torch.abs(sd - t)),
        "flashinfer_worst_element": [int(x) for x in torch.unravel_index(worst, out.shape)],
        "true_value_there": round(float(t.flatten()[worst]), 4),
        "flashinfer_value_there": round(float(out.flatten()[worst]), 4)}), flush=True)
    del captured["out"], captured["ref"], captured["f64"]
    torch.cuda.empty_cache()
