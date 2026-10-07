"""An MoE model served over several GPUs: attention data-parallel (every GPU holds the
attention weights and serves its own requests), experts expert-parallel (every GPU holds
1/R of each layer's experts). One Python process drives all GPUs, each on its own stream."""

import torch

from . import weights as weights_lib
from .config import Config
from .ep import ExpertParallel
from .model import Cache, Model


class ParallelModel:
    def __init__(self, path, devices, dtype=torch.float16, ops=None, config=None):
        self.c = config or Config.load(path)
        self.devices = [torch.device(d) for d in devices]
        R = len(self.devices)
        per = self.c.experts // R
        self.models = []
        for r, dev in enumerate(self.devices):
            w = weights_lib.load(path, self.c, dtype=dtype, device=dev, experts=range(r * per, (r + 1) * per))
            self.models.append(Model(self.c, w))
        self.ep = ExpertParallel(self.devices, self.c.experts, self.c.top_k, self.c.norm_topk_prob, ops)

    @torch.no_grad()
    def forward(self, tokens, caches=None, valids=None):
        """tokens[r]: [B_r, T] for rank r (every rank the same T). Returns logits per rank."""
        R = len(self.models)
        caches = caches or [None] * R
        valids = valids or [None] * R
        xs, sts = [], []
        for r, m in enumerate(self.models):
            with torch.cuda.device(self.devices[r]) if self.devices[r].type == "cuda" else _null():
                x, st = m.begin(tokens[r], caches[r], valids[r])
            xs.append(x)
            sts.append(st)
        for i in range(self.c.layers):
            hs = []
            for r, m in enumerate(self.models):
                with torch.cuda.device(self.devices[r]) if self.devices[r].type == "cuda" else _null():
                    xs[r], h = m.attention_block(i, xs[r], sts[r])
                hs.append(h)
            Ls = [m.w["layers"][i] for m in self.models]
            outs = self.ep.layer(hs, [L["router"] for L in Ls], [L["gate_up"] for L in Ls], [L["down"] for L in Ls])
            for r in range(R):
                xs[r] = xs[r] + outs[r].view(sts[r]["B"], sts[r]["T"], -1)
        return [m.finish(xs[r], sts[r]) for r, m in enumerate(self.models)]

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens):
        """Greedy decoding; prompts are dealt to the ranks round-robin. Returns the generated
        tokens per prompt, in the order given."""
        R = len(self.models)
        share = [list(range(r, len(prompts), R)) for r in range(R)]
        assert all(share), "need at least one prompt per GPU"
        P = max(len(p) for p in prompts)
        toks, valids, caches = [], [], []
        for r, ids in enumerate(share):
            dev = self.devices[r]
            t = torch.zeros(len(ids), P, dtype=torch.int64, device=dev)
            v = torch.zeros(len(ids), P, dtype=torch.bool, device=dev)
            for b, i in enumerate(ids):
                t[b, P - len(prompts[i]):] = torch.tensor(prompts[i], device=dev)
                v[b, P - len(prompts[i]):] = True
            toks.append(t)
            valids.append(v)
            caches.append(Cache(self.c, len(ids), P + max_new_tokens, self.models[r].w["embed"].dtype, dev))
        logits = self.forward(toks, caches, valids)
        out = [[] for _ in prompts]
        for _ in range(max_new_tokens):
            nxt = [l[:, -1].argmax(-1) for l in logits]
            for r, ids in enumerate(share):
                for b, i in enumerate(ids):
                    out[i].append(int(nxt[r][b]))
            logits = self.forward([n[:, None] for n in nxt], caches)
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
