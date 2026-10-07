"""Diagnostic: OLMoE-1B-7B over two GPUs with each GPU driven by its own thread, against the
same model driven from one thread. Run with CUDA_LAUNCH_BLOCKING=1 so that a failing kernel is
reported at the call that launched it."""

import json
import sys
import traceback

import torch

from switchyard.serve import ParallelModel


def main(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    pm = ParallelModel(path, ["cuda:0", "cuda:1"], dtype=torch.float16)
    ids = [tok(p).input_ids for p in ["The capital of France is", "def fibonacci(n):", "Water boils at", "The quick brown fox"]]
    pm.threaded = False
    want = pm.generate(ids, 8)
    pm.threaded = True
    try:
        got = pm.generate(ids, 8)
        print(json.dumps({"threaded_generate_ok": True, "same_tokens_as_one_thread": got == want}), flush=True)
    except Exception:
        print(json.dumps({"threaded_generate_ok": False}), flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1])
