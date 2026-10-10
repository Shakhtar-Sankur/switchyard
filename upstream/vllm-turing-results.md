# vLLM's FlashInfer kernel tests on a Tesla T4 (SM75): results

Kaggle, one Tesla T4 (compute capability 7.5, driver 580.178.04), torch 2.11.0+cu128,
flashinfer-python 0.7.0.post1, vLLM main at `193922d` (9 Oct 2026). Scripts:
[`kaggle_vllm_turing_tests.sh`](kaggle_vllm_turing_tests.sh), [`kaggle_vllm_turing_diag.sh`](kaggle_vllm_turing_diag.sh).

## 1. As on main (bfloat16)

The first decode test crashes the process: `torch.AcceleratorError: CUDA error: unspecified launch
failure`. Turing has no bfloat16, and after the crash every later test in the process would fail too.

## 2. Float16 below SM80 (the whole file, as CI runs it, `-k 'not num_heads2'`)

**108 passed, 52 skipped, 6 failed** in 12 minutes. All six failures are
`test_flashinfer_prefill_with_paged_kv` with head size 128 and block size 32 (soft cap None or
30, sliding window None or 64), plus two with head size 256 and soft cap 30.

## 3. The six failures, one test per process, twice each

- Head size 128: fails identically on both runs: **4 of 552,960 elements** exceed the tolerance,
  greatest absolute difference **0.0545 against 0.05 allowed**, always at index (1, 10, 29).
  Deterministic, and just over the line.
- Head size 256, soft cap 30: **passes** on both runs alone, though it failed in the full-file run.
  The test's random inputs depend on what ran before it in the same process, so these two are
  borderline cases that depend on test order.
- Controls (head size 128 with block 16, head size 256 without soft cap): pass.

## What this means for a pull request

Running the tests in float16 below SM80 turns a crash into 108 passing tests, but on its own it
would leave 6 tolerance failures on Turing. A pull request needs a decision on those first:
a float16 tolerance justified by measurement (float16 keeps more mantissa bits than bfloat16, so
the excess is the kernel's accumulation on SM75, not the input type), or a fix in FlashInfer's
SM75 path. Not opened yet; vLLM pull request #60955 (causal prefill tests) is still under review.
