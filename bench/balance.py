"""M4: expert load of OLMoE-1B-7B on real text (WikiText-2 test set), on two T4s.

1. Route 2 x 32 sequences of 256 tokens (calibration, then evaluation; disjoint) and record,
   for every forward pass and layer, how many rows each expert receives.
2. Load balance across the two GPUs: rows each GPU computes per layer with the experts split
   in index order (0-31 | 32-63), with a placement computed from the calibration text
   (greedy, equal expert counts), and with one computed on the evaluation text itself (the
   best a static placement can do). Then reload the model with the calibrated placement and
   time the forward passes against the index-order split.
3. Capacity-limited routing (GShard-style, per GPU batch) against dropless: the share of
   (token, expert) choices dropped and the evaluation perplexity, for capacity factors
   1.0 to 2.0.
Prints JSON lines."""

import gc
import json
import sys
import time

import torch
import torch.nn.functional as F

from switchyard import balance
from switchyard.serve import ParallelModel

DEVICES = ["cuda:0", "cuda:1"]


def wikitext(tok, n, T):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    ids = tok("".join(pq.read_table(f).column("text").to_pylist())).input_ids
    assert len(ids) >= n * T, len(ids)
    return [ids[i * T:(i + 1) * T] for i in range(n)]


def batches(seqs, B):
    for i in range(0, len(seqs), 2 * B):
        yield [torch.tensor(seqs[i:i + B], device=DEVICES[0]), torch.tensor(seqs[i + B:i + 2 * B], device=DEVICES[1])]


def sync():
    for d in DEVICES:
        torch.cuda.synchronize(d)


def run(pm, seqs, B, threaded, trace=False):
    """Returns (summed next-token NLL, predicted tokens, loads [batches, layers, E] or None)."""
    pm.ep.trace = [] if trace else None
    nll, n = 0.0, 0
    for toks in batches(seqs, B):
        for t, logits in zip(toks, pm.forward(toks, threaded=threaded)):
            nll += F.cross_entropy(logits[:, :-1].float().flatten(0, 1), t[:, 1:].flatten(), reduction="sum").item()
            n += t[:, 1:].numel()
    loads = None
    if trace:
        R, L = len(DEVICES), pm.c.layers
        tr = [c.double().cpu() for _, c in pm.ep.trace]
        per_batch = len(tr) // (L * R)
        loads = torch.stack([torch.stack([sum(tr[(b * L + i) * R + r] for r in range(R)) for i in range(L)])
                             for b in range(per_batch)])
        pm.ep.trace = None
    return nll, n, loads


def timed(pm, seqs, B):
    run(pm, seqs[:4 * B], B, True)  # warm up
    sync()
    t = time.perf_counter()
    for toks in batches(seqs, B):
        pm.forward(toks, threaded=True)
    sync()
    return (time.perf_counter() - t) / (len(seqs) // (2 * B))


def summary(loads, placements):
    """loads [batches, layers, E]; placements[i]: the placement of layer i. Mean and worst
    max/mean over (batch, layer) of the rows per GPU."""
    vals = torch.tensor([[balance.imbalance(balance.rank_loads(loads[b, i], placements[i], len(DEVICES)))
                          for i in range(loads.shape[1])] for b in range(loads.shape[0])])
    return {"mean": round(vals.mean().item(), 4), "worst": round(vals.max().item(), 4),
            "per_layer_mean": [round(v, 3) for v in vals.mean(0).tolist()]}


def main(path, n=64, T=256, B=8):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    seqs = wikitext(tok, n, T)
    calib, ev = seqs[:n // 2], seqs[n // 2:]
    print(json.dumps({"bench": "balance", "text": "WikiText-2 test", "seq_len": T, "calibration_seqs": len(calib),
                      "evaluation_seqs": len(ev), "tokens_per_gpu_per_pass": B * T}), flush=True)

    pm = ParallelModel(path, DEVICES, dtype=torch.float16)
    E, L = pm.c.experts, pm.c.layers
    _, _, cal_loads = run(pm, calib, B, False, trace=True)
    nll, cnt, ev_loads = run(pm, ev, B, False, trace=True)
    base_ppl = torch.tensor(nll / cnt).exp().item()

    # 1. expert skew
    mean_rows = ev_loads.sum(-1, keepdim=True) / E
    hot = ev_loads / mean_rows                       # each expert's rows / the mean expert's
    total = ev_loads.sum(0)                          # [L, E] over the whole evaluation text
    share_top8 = (total.sort(-1, descending=True).values[:, :8].sum(-1) / total.sum(-1))
    print(json.dumps({
        "expert_rows_max_over_mean": {"mean_over_batches_and_layers": round(hot.max(-1).values.mean().item(), 3),
                                      "worst": round(hot.max().item(), 3),
                                      "per_layer": [round(v, 2) for v in hot.max(-1).values.mean(0).tolist()]},
        "experts_with_no_rows_per_batch_layer": round((ev_loads == 0).double().sum(-1).mean().item(), 3),
        f"share_of_rows_on_busiest_8_of_{E}_experts": {"mean": round(share_top8.mean().item(), 3),
                                                     "max": round(share_top8.max().item(), 3)},
    }), flush=True)

    # 2. placement
    contiguous = [balance.contiguous(E)] * L
    calibrated = [balance.greedy(cal_loads.sum(0)[i], len(DEVICES)) for i in range(L)]
    oracle = [balance.greedy(total[i], len(DEVICES)) for i in range(L)]
    print(json.dumps({"rows_per_gpu_max_over_mean": {
        "experts_in_index_order": summary(ev_loads, contiguous),
        "placed_from_calibration_text": summary(ev_loads, calibrated),
        "placed_on_evaluation_text_itself": summary(ev_loads, oracle)}}), flush=True)

    t_contig = timed(pm, ev, B)

    # 3. capacity
    caps = []
    for cf in (1.0, 1.25, 1.5, 2.0):
        pm.ep.capacity_factor = cf
        pm.ep.stats.update(dropped_rows=0, routed_rows=0)
        nll, cnt, _ = run(pm, ev, B, False)
        caps.append({"capacity_factor": cf, "dropped_share": round(pm.ep.stats["dropped_rows"] / pm.ep.stats["routed_rows"], 4),
                     "perplexity": round(torch.tensor(nll / cnt).exp().item(), 3)})
    pm.ep.capacity_factor = None
    print(json.dumps({"dropless_perplexity": round(base_ppl, 3), "capacity": caps}), flush=True)

    del pm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    placed = ParallelModel(path, DEVICES, dtype=torch.float16, placement=calibrated)
    nll2, cnt2, placed_loads = run(placed, ev, B, False, trace=True)
    t_placed = timed(placed, ev, B)
    print(json.dumps({
        "forward_ms_per_pass": {"experts_in_index_order": round(t_contig * 1e3, 1),
                                "placed_from_calibration_text": round(t_placed * 1e3, 1)},
        "speedup": round(t_contig / t_placed, 3),
        "measured_rows_per_gpu_max_over_mean_with_placement": summary(placed_loads, contiguous)["mean"],
        "perplexity_with_placement": round(torch.tensor(nll2 / cnt2).exp().item(), 3),
    }), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
