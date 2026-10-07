"""Does FlashInfer's paged attention work on SM75 (Tesla T4) again?

vLLM excludes FlashInfer on SM75 (vllm/v1/attention/backends/flashinfer.py,
supports_compute_capability: floor raised to SM80) because of flashinfer-ai/flashinfer#3620:
the paged prefill kernel failed to launch on Turing's 64 KiB shared memory. The fix
(flashinfer-ai/flashinfer#3526) merged on 2026-07-22. This runs the kernels vLLM uses on the
T4, with the FlashInfer version vLLM pins, against an fp32 reference:

  prefill (BatchPrefillWithPagedKVCacheWrapper) and decode (BatchDecodeWithPagedKVCacheWrapper,
  with and without tensor cores), head_dim 64 / 128 / 256, MHA and GQA, fp16 and fp8 (e4m3)
  KV cache, ragged batches with partial last pages, prefill appended after a cached prefix.

Prints one JSON line per case and a summary. Usage on Kaggle (GPU T4 x2):
  pip install -q flashinfer-python==0.7.0.post1 --extra-index-url https://flashinfer.ai/whl/
  python flashinfer_sm75_check.py
"""

import json
import math
import time
import traceback

import torch

DEV = "cuda"


def reference(q, kv_pages, page_idx, kv_len, qo_len, num_kv_heads, k_scale=1.0, v_scale=1.0):
    """One request: q [qo_len, Hq, D]; keys/values gathered from its pages. The queries are the
    last qo_len positions of the kv_len-long sequence (causal, bottom-right aligned)."""
    k = kv_pages[page_idx, 0].float().reshape(-1, num_kv_heads, q.shape[-1])[:kv_len] * k_scale
    v = kv_pages[page_idx, 1].float().reshape(-1, num_kv_heads, q.shape[-1])[:kv_len] * v_scale
    rep = q.shape[1] // num_kv_heads
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(q.shape[-1])
    pos_q = torch.arange(kv_len - qo_len, kv_len, device=q.device)[:, None]
    pos_k = torch.arange(kv_len, device=q.device)[None, :]
    s = s.masked_fill(pos_k > pos_q, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)


def make_case(kv_lens, qo_lens, Hq, Hkv, D, page, kv_dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pages_per = [(L + page - 1) // page for L in kv_lens]
    n_pages = sum(pages_per) + 3
    perm = torch.randperm(n_pages, generator=g, device=DEV).int()  # pages scattered in memory
    indices, indptr, last = [], [0], []
    at = 0
    for L, n in zip(kv_lens, pages_per):
        indices.append(perm[at:at + n])
        at += n
        indptr.append(indptr[-1] + n)
        last.append(L - (n - 1) * page)
    kv = torch.randn(n_pages, 2, page, Hkv, D, generator=g, device=DEV) * 0.5
    k_scale = v_scale = 1.0
    if kv_dtype == torch.float8_e4m3fn:
        k_scale = v_scale = 0.05            # store x / scale, as vLLM's fp8 KV cache does
        kv = (kv / k_scale).to(kv_dtype)
    else:
        kv = kv.to(kv_dtype)
    q = torch.randn(sum(qo_lens), Hq, D, generator=g, device=DEV).half()
    return dict(kv=kv, q=q, indices=indices, indptr=torch.tensor(indptr, dtype=torch.int32, device=DEV),
                last=torch.tensor(last, dtype=torch.int32, device=DEV), k_scale=k_scale, v_scale=v_scale)


def check(out, c, kv_lens, qo_lens, Hkv):
    worst, worst_rel = 0.0, 0.0
    start = 0
    for i, (L, Lq) in enumerate(zip(kv_lens, qo_lens)):
        want = reference(c["q"][start:start + Lq], c["kv"], c["indices"][i].long(), L, Lq, Hkv,
                         c["k_scale"], c["v_scale"])
        got = out[start:start + Lq].float()
        if not torch.isfinite(got).all():
            return float("inf"), float("inf")
        err = (got - want).abs().max().item()
        worst = max(worst, err)
        worst_rel = max(worst_rel, err / max(want.abs().max().item(), 1e-6))
        start += Lq
    return worst, worst_rel


def run_prefill(fi, kv_lens, qo_lens, Hq, Hkv, D, page, kv_dtype):
    c = make_case(kv_lens, qo_lens, Hq, Hkv, D, page, kv_dtype)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
    w = fi.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    w.plan(qo_indptr, c["indptr"], torch.cat(c["indices"]), c["last"], Hq, Hkv, D, page, causal=True,
           q_data_type=torch.float16, kv_data_type=kv_dtype, o_data_type=torch.float16)
    out = w.run(c["q"], c["kv"], k_scale=c["k_scale"], v_scale=c["v_scale"])
    torch.cuda.synchronize()
    return check(out, c, kv_lens, qo_lens, Hkv)


def run_decode(fi, kv_lens, Hq, Hkv, D, page, kv_dtype, tensor_cores):
    qo_lens = [1] * len(kv_lens)
    c = make_case(kv_lens, qo_lens, Hq, Hkv, D, page, kv_dtype, seed=1)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
    w = fi.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tensor_cores)
    w.plan(c["indptr"], torch.cat(c["indices"]), c["last"], Hq, Hkv, D, page,
           q_data_type=torch.float16, kv_data_type=kv_dtype, o_data_type=torch.float16)
    out = w.run(c["q"], c["kv"], k_scale=c["k_scale"], v_scale=c["v_scale"])
    torch.cuda.synchronize()
    return check(out, c, kv_lens, qo_lens, Hkv)


def main():
    import flashinfer as fi
    cap = torch.cuda.get_device_capability()
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "capability": f"sm{cap[0]}{cap[1]}",
                      "flashinfer": fi.__version__, "torch": torch.__version__}), flush=True)
    page = 16
    kv_lens = [17, 100, 513, 1000]      # ragged, partial last pages
    full = list(kv_lens)                 # prefill of whole prompts
    appended = [5, 37, 128, 1]           # prefill appended after a cached prefix
    results = []
    for D in (64, 128, 256):
        for Hq, Hkv in ((16, 16), (16, 4)):
            for kv_dtype in (torch.float16, torch.float8_e4m3fn):
                tol = 2e-2 if kv_dtype == torch.float16 else 5e-2   # relative to the output's max
                cases = [("prefill", lambda: run_prefill(fi, kv_lens, full, Hq, Hkv, D, page, kv_dtype)),
                         ("prefill_after_prefix", lambda: run_prefill(fi, kv_lens, appended, Hq, Hkv, D, page, kv_dtype)),
                         ("decode", lambda: run_decode(fi, kv_lens, Hq, Hkv, D, page, kv_dtype, False)),
                         ("decode_tensor_cores", lambda: run_decode(fi, kv_lens, Hq, Hkv, D, page, kv_dtype, True))]
                for name, fn in cases:
                    rec = {"kernel": name, "head_dim": D, "heads": f"{Hq}/{Hkv}",
                           "kv": "fp8_e4m3" if kv_dtype == torch.float8_e4m3fn else "fp16"}
                    t = time.time()
                    try:
                        err, rel = fn()
                        rec.update(ok=bool(rel <= tol), max_abs_err=round(err, 5), max_rel_err=round(rel, 5))
                    except Exception as e:  # launch failure is the bug being checked for
                        msg = str(e)
                        # the line that says why (a build's linker/compiler error, a CUDA error)
                        why = next((l for l in msg.splitlines() if "error" in l.lower() and "ninja" not in l.lower()),
                                   msg.splitlines()[0] if msg else "")
                        rec.update(ok=False, error=f"{type(e).__name__}: {why[:300]}")
                        if not any("error" in r for r in results):  # the first failure in full
                            traceback.print_exc()
                    rec["seconds"] = round(time.time() - t, 1)
                    results.append(rec)
                    print(json.dumps(rec), flush=True)
    passed = sum(r["ok"] for r in results)
    print(json.dumps({"summary": f"{passed}/{len(results)} cases pass",
                      "failed": [f'{r["kernel"]} d{r["head_dim"]} {r["heads"]} {r["kv"]}' for r in results if not r["ok"]]}),
          flush=True)


if __name__ == "__main__":
    main()
