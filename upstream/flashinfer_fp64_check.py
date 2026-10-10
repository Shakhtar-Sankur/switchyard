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
    clone = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}  # ref scales query in place
    out = original_ref(**kw)
    f64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in clone.items()}
    captured["ref"] = out
    captured["f64"] = original_ref(**f64)
    return out


def recording_close(actual, expected, atol, rtol):
    captured.update(out=actual, atol=atol, rtol=rtol)


T.ref_paged_attn = recording_ref
T.torch.testing.assert_close = recording_close


def over(x, truth, atol, rtol):
    return int((torch.abs(x.double() - truth) > atol + rtol * torch.abs(truth)).sum())


for sliding_window, soft_cap, block_size, head_size in [(None, None, 32, 128), (64, 30.0, 32, 128),
                                                         (None, None, 16, 128), (None, 30.0, 32, 256)]:
    captured.clear()
    T.test_flashinfer_prefill_with_paged_kv(
        seq_lens=[(1, 1328), (5, 18), (129, 463)], num_heads=(32, 8), head_size=head_size,
        dtype=torch.float16, block_size=block_size, soft_cap=soft_cap, sliding_window=sliding_window)
    t, a, r = captured["f64"], captured["atol"], captured["rtol"]
    print(json.dumps({
        "case": f"window={sliding_window} soft_cap={soft_cap} page={block_size} head={head_size}",
        "flashinfer_vs_fp64_max": round(float(torch.max(torch.abs(captured["out"].double() - t))), 5),
        "fp16_reference_vs_fp64_max": round(float(torch.max(torch.abs(captured["ref"].double() - t))), 5),
        "flashinfer_vs_fp16_reference_max": round(float(torch.max(torch.abs(captured["out"].double() - captured["ref"].double()))), 5),
        "flashinfer_outside_tolerance_vs_fp64": over(captured["out"], t, a, r),
        "fp16_reference_outside_tolerance_vs_fp64": over(captured["ref"], t, a, r),
        "elements": captured["out"].numel()}), flush=True)
