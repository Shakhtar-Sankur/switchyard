"""Expert load and placement.

`loads` are per-expert row counts of one layer (how many (token, choice) rows each expert
received over some text). With expert parallelism, the rank that holds the busiest experts
does the most work while the others wait for it at the combine, so where the experts go
matters as much as how many there are."""

import torch


def rank_loads(load, placement, ranks):
    """Rows each rank computes for one layer: load [E], placement a permutation of the experts
    (rank r holds placement[r*E/R:(r+1)*E/R])."""
    per = len(placement) // ranks
    load = torch.as_tensor(load, dtype=torch.float64)
    return torch.stack([load[list(placement[r * per:(r + 1) * per])].sum() for r in range(ranks)])


def imbalance(rows):
    """max / mean: 1.0 is perfect balance; the slowest rank sets the layer's time."""
    rows = torch.as_tensor(rows, dtype=torch.float64)
    return (rows.max() / rows.mean()).item() if rows.sum() > 0 else 1.0


def greedy(load, ranks):
    """Longest-processing-time-first with equal expert counts: the busiest expert goes to the
    least loaded rank that still has room. Returns a placement (permutation of the experts)."""
    E = len(load)
    per = E // ranks
    order = sorted(range(E), key=lambda e: -float(load[e]))
    bins, total = [[] for _ in range(ranks)], [0.0] * ranks
    for e in order:
        r = min((r for r in range(ranks) if len(bins[r]) < per), key=lambda r: total[r])
        bins[r].append(e)
        total[r] += float(load[e])
    return [e for b in bins for e in sorted(b)]


def contiguous(E):
    return list(range(E))
