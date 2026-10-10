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


CASES = [(None, None, 32, 128), (64, 30.0, 32, 128), (None, None, 16, 128),
         (None, 30.0, 32, 256), (64, None, 32, 256), (64, 30.0, 32, 256), (None, None, 32, 256)]
for repeat in range(3):  # the same process, so state left by earlier cases is present
    for sliding_window, soft_cap, block_size, head_size in CASES:
        captured.clear()
        T.test_flashinfer_prefill_with_paged_kv(
            seq_lens=[(1, 1328), (5, 18), (129, 463)], num_heads=(32, 8), head_size=head_size,
            dtype=torch.float16, block_size=block_size, soft_cap=soft_cap, sliding_window=sliding_window)
        t, a, r = captured["f64"], captured["atol"], captured["rtol"]
        out, ref = captured["out"].double(), captured["ref"].double()
        print(json.dumps({
            "repeat": repeat,
            "case": f"window={sliding_window} soft_cap={soft_cap} page={block_size} head={head_size}",
            "flashinfer_vs_fp64_max": round(float(torch.max(torch.abs(out - t))), 5),
            "fp16_reference_vs_fp64_max": round(float(torch.max(torch.abs(ref - t))), 5),
            "flashinfer_vs_fp16_reference_max": round(float(torch.max(torch.abs(out - ref))), 5),
            "flashinfer_outside_tolerance_vs_fp64": over(captured["out"], t, a, r),
            "fp16_reference_outside_tolerance_vs_fp64": over(captured["ref"], t, a, r),
            "flashinfer_outside_tolerance_vs_fp16_reference": over(captured["out"], ref, a, r),
            "elements": captured["out"].numel()}), flush=True)
        del captured["out"], captured["ref"], captured["f64"]
        torch.cuda.empty_cache()
