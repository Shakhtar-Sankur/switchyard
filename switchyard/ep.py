"""Expert parallelism: every MoE layer's experts split over R GPUs (rank r owns experts
[r*E/R, (r+1)*E/R)), each GPU running attention for its own batch of tokens (data-parallel
attention, as DeepSeek serves its MoE models).

One MoE layer, for every rank r at once:
  1. route its tokens; sort the (token, choice) rows by global expert, so the rows bound for
     rank d are one contiguous segment;
  2. dispatch: a kernel on r writes the hidden states of each segment straight into rank d's
     memory (peer to peer, no staging), with the segment's per-expert offsets and weights;
  3. every rank runs its experts over its own segment (rows read in place) and over each
     segment it received;
  4. return: each result segment is copied back into the sender's buffer, at the place the
     sender's sort gave it, so
  5. the sender's combine kernel sums every token's k results exactly as on one GPU.
The arithmetic per row is the same as on a single GPU and the combine adds in the same
order, so the output is bit-identical to the single-GPU layer.

The orchestration only calls a small set of operations (`Ops`): CUDA kernels on GPUs, plain
PyTorch on the CPU, so the same code is tested without GPUs."""

import torch
import torch.nn.functional as F

from . import moe


class TorchOps:
    """Reference operations in PyTorch (CPU tests)."""

    def route(self, h, router, k, norm):
        idx, w = moe.route(h, router, k, norm)
        return idx.int(), w

    def sort(self, idx, w, E):
        p = moe.plan(idx.long(), w, E)
        pos_of = torch.empty(idx.numel(), dtype=torch.int32)
        pos_of[p.token * idx.shape[1] + p.slot] = torch.arange(idx.numel(), dtype=torch.int32)
        return p.counts.int(), p.offsets.int(), p.token.int(), p.weight, pos_of

    def ffn(self, x, a_rows, offsets, counts, wsorted, gate_up, down, rows):
        xs = x[a_rows.long()] if a_rows is not None else x
        return moe.run_sorted(xs, counts.long(), gate_up, down) * wsorted[:, None]

    def combine(self, y, pos_of, idx, N):
        # rows sorted by expert: adding them in row order adds each token's k results in
        # expert order, as the reference does
        token = torch.empty(idx.numel(), dtype=torch.int64)
        token[pos_of.long()] = torch.arange(idx.numel()) // idx.shape[1]
        out = torch.zeros(N, y.shape[1], dtype=y.dtype)
        out.index_add_(0, token, y)
        return out

    def gather_to(self, src, index, dst):
        dst.copy_(src[index.long()])

    def copy_to(self, dst, src):
        dst.copy_(src)

    def host(self, t):
        return t.tolist()


class CudaOps:
    """switchyard's kernels (kernels.py / csrc/moe_kernels.cu)."""

    def __init__(self, mode="auto"):
        from . import kernels
        self.k, self.mode = kernels, mode

    def route(self, h, router, k, norm):
        return self.k.route(h, router, k, norm)

    def sort(self, idx, w, E):
        return self.k.ext().sort_by_expert(idx, w, E)

    def ffn(self, x, a_rows, offsets, counts, wsorted, gate_up, down, rows):
        mode = self.mode if self.mode != "auto" else self.k.pick(int(counts.max()) if counts.numel() else 0)
        return self.k.ffn(x, a_rows, offsets, counts, wsorted, gate_up, down, rows, mode)

    def combine(self, y, pos_of, idx, N):
        return self.k.ext().combine(y, pos_of, idx, N)

    def gather_to(self, src, index, dst):
        self.k.ext().gather_rows(src, index, dst)

    def copy_to(self, dst, src):
        self.k.ext().copy_async(dst, src)

    def host(self, t):
        return t.tolist()


class _NoStream:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ExpertParallel:
    """Runs MoE layers whose experts are split over `devices` (one rank per device; ranks may
    share a device, which tests the protocol on one GPU or on the CPU)."""

    def __init__(self, devices, num_experts, top_k, norm_topk_prob, ops=None):
        self.devices = [torch.device(d) for d in devices]
        self.R = len(self.devices)
        assert num_experts % self.R == 0, "experts must split evenly over the ranks"
        self.E, self.per = num_experts, num_experts // self.R
        self.k, self.norm = top_k, norm_topk_prob
        self.cuda = self.devices[0].type == "cuda"
        self.ops = ops or (CudaOps() if self.cuda else TorchOps())
        if self.cuda:
            from . import kernels
            self.streams = [torch.cuda.Stream(d) for d in self.devices]
            idx = sorted({d.index for d in self.devices})
            self.peer = all(kernels.ext().enable_peer_access(a, b) for a in idx for b in idx if a != b)
        self.stats = {"dispatched_rows": 0, "local_rows": 0}

    def owner(self, expert):
        return expert // self.per

    # --- small helpers that hide CPU vs CUDA ---
    def _on(self, r):
        if not self.cuda:
            return _NoStream()
        return _Both(torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]))

    def _event(self, r):
        if not self.cuda:
            return None
        e = torch.cuda.Event()
        e.record(self.streams[r])
        return e

    def _wait(self, r, event):
        if event is not None:
            self.streams[r].wait_event(event)

    def layer(self, hs, routers, gate_ups, downs):
        """hs[r]: [N_r, H] hidden states of rank r's tokens; routers[r]: the layer's router (on
        every rank); gate_ups[r], downs[r]: rank r's experts of this layer. Returns the MoE
        output for every rank's tokens."""
        R, ops = self.R, self.ops
        if self.cuda:  # the streams start after whatever produced hs
            for r in range(R):
                self.streams[r].wait_stream(torch.cuda.current_stream(self.devices[r]))
        st = []
        for r in range(R):
            with self._on(r):
                idx, w = ops.route(hs[r], routers[r], self.k, self.norm)
                counts, offsets, token, wsorted, pos_of = ops.sort(idx, w, self.E)
                y = torch.empty(idx.numel(), hs[r].shape[1], dtype=hs[r].dtype, device=hs[r].device)
                st.append(dict(idx=idx, counts=counts, offsets=offsets, token=token, wsorted=wsorted,
                               pos_of=pos_of, y=y))
        for r in range(R):  # segment bounds per destination (one host read per rank, on its stream)
            with self._on(r):
                off = ops.host(st[r]["offsets"])
            st[r]["seg"] = [(off[d * self.per], off[(d + 1) * self.per]) for d in range(R)]

        # 2. dispatch: rank r writes the rows bound for rank d into d's memory
        inbox = [[None] * R for _ in range(R)]  # inbox[d][r]: what d received from r
        for r in range(R):
            with self._on(r):
                for d in range(R):
                    a, b = st[r]["seg"][d]
                    if d == r or b == a:
                        continue
                    dev = self.devices[d]
                    x = torch.empty(b - a, hs[r].shape[1], dtype=hs[r].dtype, device=dev)
                    ops.gather_to(hs[r], st[r]["token"][a:b], x)
                    lo = st[r]["offsets"][d * self.per:(d + 1) * self.per + 1]
                    offs = torch.empty_like(lo, device=dev)
                    ops.copy_to(offs, (lo - a).to(lo.dtype).contiguous())
                    wts = torch.empty(b - a, dtype=st[r]["wsorted"].dtype, device=dev)
                    ops.copy_to(wts, st[r]["wsorted"][a:b].contiguous())
                    cnt = torch.empty(self.per, dtype=st[r]["counts"].dtype, device=dev)
                    ops.copy_to(cnt, st[r]["counts"][d * self.per:(d + 1) * self.per].contiguous())
                    inbox[d][r] = dict(x=x, offsets=offs, counts=cnt, w=wts, rows=b - a, done=self._event(r))
                    self.stats["dispatched_rows"] += b - a

        # 3. experts: own segment in place, then each received segment; 4. results go home
        back = [[] for _ in range(R)]
        for d in range(R):
            with self._on(d):
                a, b = st[d]["seg"][d]
                if b > a:
                    lo = st[d]["offsets"][d * self.per:(d + 1) * self.per + 1]
                    offs = (lo - a).to(lo.dtype).contiguous()
                    cnt = st[d]["counts"][d * self.per:(d + 1) * self.per]
                    st[d]["y"][a:b] = ops.ffn(hs[d], st[d]["token"][a:b].contiguous(), offs, cnt,
                                              st[d]["wsorted"][a:b].contiguous(), gate_ups[d], downs[d], b - a)
                    self.stats["local_rows"] += b - a
                for r in range(R):
                    m = inbox[d][r]
                    if m is None:
                        continue
                    self._wait(d, m["done"])
                    yr = ops.ffn(m["x"], None, m["offsets"], m["counts"], m["w"], gate_ups[d], downs[d], m["rows"])
                    a2, b2 = st[r]["seg"][d]
                    ops.copy_to(st[r]["y"][a2:b2], yr)
                    back[r].append(self._event(d))
                    m["keep"] = yr  # alive until the copy has run

        # 5. combine at home
        outs = []
        for r in range(R):
            with self._on(r):
                for e in back[r]:
                    self._wait(r, e)
                outs.append(ops.combine(st[r]["y"], st[r]["pos_of"], st[r]["idx"], hs[r].shape[0]))
        if self.cuda:  # the caller's streams continue after the layer
            for r in range(R):
                torch.cuda.current_stream(self.devices[r]).wait_stream(self.streams[r])
        self._keep = inbox  # buffers stay referenced until the next layer
        return outs


class _Both:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __enter__(self):
        self.a.__enter__()
        self.b.__enter__()
        return self

    def __exit__(self, *e):
        self.b.__exit__(*e)
        self.a.__exit__(*e)
        return False
