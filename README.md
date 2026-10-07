# switchyard

Mixture-of-experts inference from scratch: expert-parallel serving of
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924) (6.9B parameters, 64 experts
per layer, top-8 routing) over two T4 GPUs, with my own CUDA kernels for routing, sorting and
grouped expert matmuls, and my own peer-to-peer token dispatch between the GPUs instead of NCCL.

```
  GPU 0: attention for its requests            GPU 1: attention for its requests
        │                                              │
        ▼ route (top-8 of 64) + stable sort by expert  ▼
  ┌──── rows for experts 0-31 stay ────┐        ┌──── rows for experts 32-63 stay ───┐
  │  rows for 32-63 ── written straight into GPU 1's memory (peer-to-peer kernel) ──▶│
  │◀── rows for 0-31 ──────────── from GPU 1 ───────────────────────────────────────│
  ▼                                                                                 ▼
  grouped expert GEMM / GEMV over its 32 experts           grouped expert GEMM / GEMV
  results copied back to the sender, at the place its sort gave them
  ▼                                                                                 ▼
  combine: each token's 8 expert outputs summed in expert order (bit-identical to one GPU)
```

**Headline results (2× Tesla T4, PCIe, measured on Kaggle; raw output in `results/t4`):**

- OLMoE-1B-7B decodes at **112 tokens/s at batch 8, 2.2× Hugging Face transformers** on the
  same two GPUs (52 tokens/s), and 823 tokens/s at batch 64, using 7.6 GiB per GPU. Greedy
  output is token-for-token identical to transformers on 7 of 8 prompts over 32 tokens.
- One MoE layer at decode batch sizes is **5.4–7× faster than a per-expert PyTorch loop** (as in transformers' reference OLMoE code) and
  1.9–2.3× faster than a cuBLAS loop over sorted rows; the grouped GEMV streams expert
  weights at 233 GB/s (73% of the T4's 320 GB/s).
- The expert-parallel layer, with my own peer-to-peer dispatch, is **bit-identical** to the
  single-GPU layer and 1.25× faster than the same layer on NCCL's all_to_all at 256 tokens
  per GPU (level at 16–64, slower at 1 and 1,024).
- On real text the busiest expert gets **4×** the rows of the average one, but over 32
  experts per GPU this averages out to a 6% imbalance between the GPUs; and capacity-limited
  routing, which drops what overflows, is not an option for this model: a capacity factor of
  2.0 still drops 9.6% of expert choices and raises WikiText-2 perplexity from 12.9 to 20.2.

What did not work is written down as carefully as what did (see M2 and M4).

## Plan

| | Milestone | State |
|---|---|---|
| M0 | OLMoE in plain PyTorch (QK-norm, softmax top-8 routing, KV cache, left padding), checked against transformers | done |
| M1 | CUDA kernels: fused routing, stable sort by expert without atomics, grouped GEMM on tensor cores with fused SwiGLU, grouped GEMV for decode, deterministic combine | done |
| M2 | Expert parallelism over GPUs: route, sort, dispatch peer-to-peer, experts, return, combine; bit-identical to one GPU | done |
| M3 | Serve OLMoE-1B-7B on two T4s: attention data-parallel, experts expert-parallel, one thread per GPU | done: 112 tok/s at batch 8 (2.2× transformers) |
| M4 | Load balance on real text: expert skew, GPU imbalance, placement, capacity vs dropless | done: measured; placement does not help on 2 GPUs, dropping hurts badly |
| M5 | Write-up | done: this README, raw results in `results/t4` |

## M0: the model

`switchyard/model.py` is OLMoE in ~160 lines of PyTorch: RMSNorm with QK-norm, rotary
embeddings, a KV cache with left padding for batched generation, and a hook
(`self.moe(layer, h)`) where the mixture-of-experts layer goes, so the same model runs with
the reference loop, the CUDA kernels, or experts spread over GPUs. Weights load straight
from the safetensors checkpoint (`weights.py`), each GPU reading only its own experts.

Tests (`tests/test_model.py`) build tiny random OLMoE checkpoints with transformers and
require the same logits in five shapes (plain, grouped-query attention, normalized top-k
weights, QKV clipping, top-4 of 16 experts) and the same greedy generations. On the real model in fp16 (M3): first-token
logits within 0.28 of transformers (both fp16), same top token for 8 of 8 prompts, and the
same 32 greedy tokens for 7 of 8 (the eighth agrees for 16, then takes a different
near-tie).

## M1: kernels (`switchyard/csrc/moe_kernels.cu`)

| Kernel | What it does |
|---|---|
| `route` | one warp per token: softmax over the 64 router logits in fp32 and top-k by warp shuffles |
| `sort_by_expert` | counts per expert, a block-wide prefix scan, then a stable scatter: rows grouped by expert, tokens in order within an expert, no atomics, so it is deterministic |
| `grouped_gemm` | all experts' matmuls in one launch: 64×64 tiles over each expert's rows, tensor cores (WMMA, fp16 in, fp32 accumulate) on sm_75, register prefetch of the next K-slice, SwiGLU (and router-weight scaling for the down projection) fused into the epilogue; rows are gathered from the unsorted input inside the kernel, never copied |
| `grouped_gemv` | the same for decode, where an expert sees a few rows: the kernel is a weight stream, each warp reading 4 columns of an expert's weights once for all of its rows |
| `combine` | each token's k outputs summed in fp32 in expert order, so the result does not depend on how rows were split or scheduled |

`kernels.pick` chooses by the largest per-expert row count, with thresholds measured on the
T4: GEMV up to 16 rows, tensor-core GEMM up to 48, cuBLAS per expert beyond that (prefill).

One OLMoE MoE layer (64 experts, top 8, 2048 → 1024 → 2048) on one T4, fp16:

| Tokens | expert loop (as transformers writes it) | cuBLAS loop over sorted rows | switchyard | vs loop | vs sorted |
|---|---|---|---|---|---|
| 1 | 4.88 ms | 1.31 ms | **0.69 ms** | 7.1× | 1.9× |
| 8 | 11.99 ms | 5.12 ms | **2.21 ms** (233 GB/s of weights) | 5.4× | 2.3× |
| 32 | 16.44 ms | 7.26 ms | **3.61 ms** | 4.6× | 2.0× |
| 128 | 17.18 ms | 7.91 ms | **6.46 ms** | 2.7× | 1.2× |
| 512 | 17.16 ms | 8.50 ms | 9.39 ms | 1.8× | **0.90×** |
| 2048 | 20.41 ms | 16.80 ms | 15.53 ms | 1.3× | 1.08× |
| 4096 | 30.08 ms | 28.18 ms | 25.58 ms | 1.2× | 1.10× |

Decode is where the kernels win: they read each used expert's weights once. Around 512
tokens my tensor-core GEMM loses to cuBLAS and the cuBLAS path is chosen; its extra cost
there is the sort and combine. Errors against an fp32 reference are below those of
the same loop in fp16 at every size.

## M2: expert parallelism (`switchyard/ep.py`)

Each GPU owns 32 of every layer's 64 experts and runs attention for its own requests (data-
parallel attention, as DeepSeek serves its MoE models). One MoE layer, on every GPU at once:

1. route its tokens and sort the (token, expert) rows by expert; the rows for the other GPU
   are now one contiguous segment;
2. **dispatch**: a kernel gathers that segment's hidden states and writes them straight into
   the other GPU's memory over PCIe peer-to-peer (no staging copy), with the per-expert
   offsets and router weights; a CUDA event tells the receiver when they have landed;
3. run its experts over its own rows (read in place) and over the rows it received;
4. **return**: copy each result segment back into the sender's buffer at the place the
   sender's sort gave it;
5. combine, in expert order.

The arithmetic for each row is the one a single GPU does and the combine adds in the same
order, so the outputs are **bit-identical** to the single-GPU layer: tested on the CPU with
1, 2 and 4 ranks (PyTorch reference operations), and on GPUs with the CUDA kernels for two
ranks on one GPU and on two GPUs (`tests/test_ep*.py`). After the sort each GPU reads its
per-expert offsets to the host once; everything else (kernel choice, grid sizes) uses that
copy, so no later step waits for a GPU.

Two GPUs, N tokens on each; NCCL runs the same expert parallelism with
`torch.distributed.all_to_all_single` as the transport and the same kernels for everything
else (`bench/ep_nccl.py`):

| Tokens per GPU | switchyard (P2P) | NCCL all_to_all | one GPU, all 64 experts, 2N tokens |
|---|---|---|---|
| 1 | 1.69 ms | **1.59 ms** | 1.00 ms |
| 16 | **4.07 ms** | 4.12 ms | 4.19 ms |
| 64 | **7.14 ms** | 7.26 ms | 6.57 ms |
| 256 | **8.79 ms** | 11.03 ms | 9.31 ms |
| 1024 | 20.83 ms | **19.01 ms** | 16.87 ms |

Two honest conclusions. My dispatch beats NCCL only between 16 and 256 tokens per GPU. And
on PCIe T4s splitting the experts is never faster than keeping all 64 on one GPU: the
transfers cost more than the halved expert work saves. Expert parallelism here buys
**memory**: OLMoE-1B-7B is 13.8 GB in fp16 and does not fit one 15 GB T4 with room for a KV
cache; split, it takes 7.6 GiB per GPU. With NVLink, or more GPUs and larger batches, the
trade changes; I have not measured that.

## M3: serving OLMoE-1B-7B on two T4s (`switchyard/serve.py`)

Every GPU holds the attention weights and its half of the experts, serves its own share of
the requests, and is driven by its own Python thread (the kernels release the GIL), meeting
the other GPU at every MoE layer. Greedy decoding, 32 new tokens, fp16 (`bench/serve.py`):

| Batch | transformers, `device_map="auto"` | switchyard, one thread | switchyard, one thread per GPU |
|---|---|---|---|
| 1 / 2 | 27 tok/s (batch 1) | 27 tok/s | 29 tok/s (batch 2) |
| 8 | 52 tok/s | 88 tok/s | **112 tok/s** |
| 32 | | 332 tok/s | **452 tok/s** |
| 64 | | 580 tok/s | **823 tok/s** |

transformers with `device_map="auto"` splits the layers over the two GPUs, which then work
one after the other; it is the usual way to run this model when it does not fit one GPU, not a
tuned serving engine. I have not compared against vLLM or SGLang.

Getting one thread per GPU right took two bugs worth recording, both invisible with
`CUDA_LAUNCH_BLOCKING=1`, under which the threaded model already matched:

- **A wait that waited on itself.** Before routing, the MoE stream was told to wait for
  "the current stream", but inside the MoE stream's context the current stream *is* the MoE
  stream: a no-op. The router could read the hidden states before attention had written them,
  producing garbage expert indices and an illegal memory access. With one thread the host
  was usually slow enough to hide it. A test now delays the input on the GPU by ~0.1 s and
  checks the layer still waits for it.
- **A shared allocator pool.** The buffers a GPU sends to its peer came from a pool the
  peer's own thread was freeing temporaries into while its kernels still read them; the
  caching allocator could hand such a block to the sender, which overwrote it from another
  stream. Each (sender, receiver) pair now allocates from a pool of its own, reused only
  after the receiver has signalled it is done.

## M4: load balance on real text (`bench/balance.py`)

OLMoE-1B-7B on WikiText-2 (test set), 256-token sequences, 2,048 tokens per GPU per pass;
placements computed on 32 calibration sequences and evaluated on 32 others.

**Experts are skewed.** In a pass, the busiest expert of a layer gets on average **4.0×** the
rows of the mean expert (worst 5.5×), and the busiest 8 of 64 experts take 32% of all rows
(12.5% if uniform). No expert went unused.

**The GPUs are not, and placement does not help.** The rows each GPU computes, as max / mean
over the two GPUs (1.0 = perfect):

| Expert placement | mean over passes and layers | worst |
|---|---|---|
| experts 0-31 / 32-63 (index order) | 1.059 | 1.183 |
| greedy placement from the calibration text | 1.071 | 1.223 |
| greedy placement on the evaluation text itself (best case for a fixed placement) | 1.023 | 1.050 |

Thirty-two experts per GPU average the skew away to 6%, and which experts are busy changes
from batch to batch, so a placement learned on other text is no better than index order.
Reloading the model with the calibrated placement measured the same: 583 ms per pass against
575 ms (perplexity unchanged, 12.848 vs 12.853). Rebalancing (as in DeepSeek's EPLB, which
also replicates hot experts) pays off with many GPUs, few experts per GPU and large batches;
at this scale the best possible fixed placement would save at most ~3% of the MoE time.

**Dropping tokens is not an option for this model.** Capacity-limited routing (GShard style,
per GPU batch: each expert takes at most c × tokens × 8 / 64 rows; the rest of a token's
choices are dropped and the token keeps its other experts and the residual) against
dropless:

| Capacity factor | expert choices dropped | WikiText-2 perplexity |
|---|---|---|
| dropless (switchyard) | 0% | **12.85** |
| 2.0 | 9.6% | 20.19 |
| 1.5 | 15.9% | 29.42 |
| 1.25 | 21.1% | 42.88 |
| 1.0 | 28.6% | 92.14 |

With a 4× hottest expert, even twice the average room overflows. OLMoE was trained without
dropping, and serving it with a capacity limit changes the model; this is why the kernels are
built around variable-size groups (sorted rows, per-expert offsets) rather than fixed-capacity
buffers. (The dropped rows are still computed in this measurement; it measures quality, not
speed.)

## Limitations

- Two T4s over PCIe only; no NVLink, no more than two GPUs measured.
- Inference only: no backward pass through the expert-parallel layer.
- The control flow is Python (one thread per GPU); no CUDA graphs, so small-batch decode
  is partly launch-bound.
- Greedy decoding with a static batch; no continuous batching or paged KV cache here (see
  [relay](https://github.com/Shakhtar-Sankur/relay) for those).

## Reproduce

Kaggle notebook, accelerator "GPU T4 x2", internet on:

```
!curl -sL https://raw.githubusercontent.com/Shakhtar-Sankur/switchyard/main/scripts/kaggle.sh | bash
```

`RUN=kernels`, `ep`, `serve` or `balance` runs one part. It builds the extension, runs the
79 tests (31 run without a GPU: `python -m pytest tests`), and prints the benchmarks as JSON
lines.

```
switchyard/
  model.py, config.py, weights.py   OLMoE in PyTorch; checkpoint loading (per-GPU expert slices)
  moe.py                            reference routing and experts (the loop transformers runs; sorted)
  csrc/moe_kernels.cu, kernels.py   CUDA kernels and their Python side
  ep.py                             expert parallelism (dispatch, compute, return, combine)
  serve.py                          the model over several GPUs, one thread per GPU
  balance.py                        expert load, GPU imbalance, greedy placement
bench/      kernels.py, ep.py, ep_nccl.py, serve.py, balance.py
tests/      against transformers, the reference layer, and one GPU
results/t4/ raw benchmark output
```
