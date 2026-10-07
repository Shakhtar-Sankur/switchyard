"""OLMoE-1B-7B (6.9B parameters, 64 experts per layer, 13.8 GB in fp16) on two T4s.

1. Hugging Face transformers, fp16, device_map="auto" (layers split over the two GPUs; how the
   model is normally run when it does not fit on one): greedy output and decode speed.
2. switchyard: experts split over the GPUs (expert parallelism), each GPU serving its own
   requests; same prompts: agreement with transformers and decode speed.
Prints JSON lines."""

import gc
import json
import os
import sys
import time

import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Photosynthesis is the process by which",
    "In 1969, Neil Armstrong",
    "The three laws of motion are",
    "A mixture-of-experts model routes each token to",
    "Water boils at",
    "The quick brown fox",
]


def hf_run(path, tok, new):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto")
    except TypeError:  # transformers 4.x
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    m.eval()
    out, firsts = [], []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
        with torch.no_grad():
            firsts.append(m(ids).logits[0, -1].float().cpu())
            g = m.generate(ids, max_new_tokens=new, do_sample=False)
        out.append(g[0, ids.shape[1]:].tolist())
    speeds = {}
    for B in (1, 8):
        ids = tok(PROMPTS[:B], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            m.generate(**ids, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
            t = time.perf_counter()
            m.generate(**ids, max_new_tokens=new, do_sample=False, min_new_tokens=new)
            torch.cuda.synchronize()
        speeds[B] = B * new / (time.perf_counter() - t)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out, firsts, speeds


def ours_run(path, tok, new):
    from switchyard.serve import ParallelModel
    pm = ParallelModel(path, ["cuda:0", "cuda:1"], dtype=torch.float16)
    ids = [tok(p).input_ids for p in PROMPTS]
    out = pm.generate(ids, new)
    firsts = []
    for i in range(0, len(ids), 2):  # two prompts at a time, one on each GPU
        logits = pm.forward([torch.tensor([ids[i]], device="cuda:0"), torch.tensor([ids[i + 1]], device="cuda:1")])
        firsts += [logits[0][0, -1].float().cpu(), logits[1][0, -1].float().cpu()]
    speeds = {}
    for per_gpu in (1, 4, 16, 32):
        batch = [ids[i % len(ids)] for i in range(2 * per_gpu)]
        pm.generate(batch, 4)
        for d in ("cuda:0", "cuda:1"):
            torch.cuda.synchronize(d)
        t = time.perf_counter()
        pm.generate(batch, new)
        for d in ("cuda:0", "cuda:1"):
            torch.cuda.synchronize(d)
        speeds[2 * per_gpu] = 2 * per_gpu * new / (time.perf_counter() - t)
    mem = [torch.cuda.max_memory_allocated(d) / 2**30 for d in (0, 1)]
    return out, firsts, speeds, mem


def main(path, new=32):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    hf_out, hf_first, hf_speed = hf_run(path, tok, new)
    print(json.dumps({"engine": "transformers device_map=auto fp16", "decode_tok_per_s": hf_speed}), flush=True)
    our_out, our_first, our_speed, mem = ours_run(path, tok, new)
    agree = []
    for a, b in zip(hf_out, our_out):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        agree.append(n)
    rec = {"engine": "switchyard expert-parallel fp16, 2 GPUs", "decode_tok_per_s": our_speed,
           "peak_GiB_per_gpu": mem, "tokens_identical_before_first_difference": agree, "of": new,
           "identical_outputs": sum(a == b for a, b in zip(hf_out, our_out)), "prompts": len(PROMPTS)}
    if our_first:
        d = [(a - b).abs().max().item() for a, b in zip(our_first, hf_first)]
        same_top = [int(a.argmax() == b.argmax()) for a, b in zip(our_first, hf_first)]
        rec.update(first_token_logit_max_abs_diff=max(d), first_token_same_argmax=f"{sum(same_top)}/{len(same_top)}")
    print(json.dumps(rec), flush=True)
    print(json.dumps({"sample": PROMPTS[0] + tok.decode(our_out[0])}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
