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

import threading
from concurrent.futures import ThreadPoolExecutor

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

    def ffn(self, x, a_rows, offsets, counts, wsorted, gate_up, down, rows, host_counts=None):
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

    def ffn(self, x, a_rows, offsets, counts, wsorted, gate_up, down, rows, host_counts=None):
        hc = host_counts if host_counts is not None else counts.tolist()
        mode = self.mode if self.mode != "auto" else self.k.pick(max(hc, default=0))
        return self.k.ffn(x, a_rows, offsets, counts, wsorted, gate_up, down, rows, mode, hc)

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
            # mail[r][d]: a stream on device d used only to allocate what rank r sends to rank d
            # (see phase_send); no kernel runs on it.
            self.mail = [[torch.cuda.Stream(d) for d in self.devices] for _ in self.devices]
            idx = sorted({d.index for d in self.devices})
            self.peer = all(kernels.ext().enable_peer_access(a, b) for a in idx for b in idx if a != b)
        self.stats = {"dispatched_rows": 0, "local_rows": 0}
        self._pool = None
        self.barrier = threading.Barrier(self.R)

    def owner(self, expert):
        return expert // self.per

    # --- small helpers that hide CPU vs CUDA ---
    def _on(self, r):
        if not self.cuda:
            return _NoStream()
        return _Both(torch.cuda.device(self.devices[r]), torch.cuda.stream(self.streams[r]))

    def _alloc(self, r, d):
        """Allocations on device d for what rank r sends there, from a pool of their own."""
        if not self.cuda:
            return _NoStream()
        return torch.cuda.stream(self.mail[r][d])

    def _event(self, r):
        if not self.cuda:
            return None
        e = torch.cuda.Event()
        e.record(self.streams[r])
        return e

    def _wait(self, r, event):
        if event is not None:
            self.streams[r].wait_event(event)

    # One MoE layer runs in three phases per rank. `layer()` runs them for all ranks from one
    # thread; serve.py runs each rank in its own thread with a barrier between the phases,
    # so that the GPUs are driven in parallel. After its sort a rank reads its per-expert
    # offsets once; everything later (kernel choice, grid sizes) uses that host copy, so no
    # phase waits for a GPU.
    def new_state(self):
        R = self.R
        return dict(st=[None] * R, inbox=[[None] * R for _ in range(R)], back=[[] for _ in range(R)])

    def phase_send(self, sh, r, h, router):
        """Route rank r's tokens, sort them by expert, and send every other rank its rows."""
        ops, R = self.ops, self.R
        with self._on(r):
            if self.cuda:
                self.streams[r].wait_stream(torch.cuda.current_stream(self.devices[r]))
            idx, w = ops.route(h, router, self.k, self.norm)
            counts, offsets, token, wsorted, pos_of = ops.sort(idx, w, self.E)
            y = torch.empty(idx.numel(), h.shape[1], dtype=h.dtype, device=h.device)
            off = ops.host(offsets)  # the one host read of the layer, on this rank's stream
            hc = [off[e + 1] - off[e] for e in range(self.E)]
            seg = [(off[d * self.per], off[(d + 1) * self.per]) for d in range(R)]
            st = dict(idx=idx, offsets=offsets, counts=counts, token=token, wsorted=wsorted, pos_of=pos_of,
                      y=y, seg=seg, hc=hc)
            sh["st"][r] = st
            sh["back"][r] = []
            for d in range(R):
                a, b = seg[d]
                if d == r or b == a:
                    sh["inbox"][d][r] = None
                    continue
                dev = self.devices[d]
                # The buffers on device d come from a pool only rank r allocates from. The
                # caching allocator reuses a freed block at once for the next allocation on the
                # same stream, which is safe only for kernels on that stream; rank r writes from
                # its own stream. In a pool of rank d's (its default or MoE stream) this rank
                # could be handed a block that rank d's thread has just freed while d's kernels
                # still read it (an offsets array, say), and overwrite it. A block of this pool
                # is reused only after rank r has waited for d to finish with it (the events
                # d sends back with the results).
                with self._alloc(r, d):
                    x = torch.empty(b - a, h.shape[1], dtype=h.dtype, device=dev)
                    lo = offsets[d * self.per:(d + 1) * self.per + 1]
                    offs = torch.empty(self.per + 1, dtype=lo.dtype, device=dev)
                    wts = torch.empty(b - a, dtype=wsorted.dtype, device=dev)
                    cnt = torch.empty(self.per, dtype=counts.dtype, device=dev)
                ops.gather_to(h, token[a:b], x)  # straight into rank d's memory
                ops.copy_to(offs, (lo - a).to(lo.dtype).contiguous())
                ops.copy_to(wts, wsorted[a:b].contiguous())
                ops.copy_to(cnt, counts[d * self.per:(d + 1) * self.per].contiguous())
                sh["inbox"][d][r] = dict(x=x, offsets=offs, counts=cnt, w=wts, rows=b - a,
                                         hc=hc[d * self.per:(d + 1) * self.per], done=self._event(r))
                self.stats["dispatched_rows"] += b - a

    def phase_compute(self, sh, d, h, gate_up, down):
        """Rank d's experts over its own rows and over every segment it received; each result
        segment is copied back into its sender's buffer."""
        ops, st = self.ops, sh["st"]
        with self._on(d):
            a, b = st[d]["seg"][d]
            if b > a:
                lo = st[d]["offsets"][d * self.per:(d + 1) * self.per + 1]
                offs = (lo - a).to(lo.dtype).contiguous()
                cnt = st[d]["counts"][d * self.per:(d + 1) * self.per]
                hc = st[d]["hc"][d * self.per:(d + 1) * self.per]
                st[d]["y"][a:b] = ops.ffn(h, st[d]["token"][a:b].contiguous(), offs, cnt,
                                          st[d]["wsorted"][a:b].contiguous(), gate_up, down, b - a, hc)
                self.stats["local_rows"] += b - a
            for r in range(self.R):
                m = sh["inbox"][d][r]
                if m is None:
                    continue
                self._wait(d, m["done"])
                yr = ops.ffn(m["x"], None, m["offsets"], m["counts"], m["w"], gate_up, down, m["rows"], m["hc"])
                a2, b2 = st[r]["seg"][d]
                ops.copy_to(st[r]["y"][a2:b2], yr)  # results home, into the sender's memory
                m["keep"] = yr
                sh["back"][r].append(self._event(d))

    def phase_combine(self, sh, r, n_tokens):
        st = sh["st"][r]
        with self._on(r):
            for e in sh["back"][r]:
                self._wait(r, e)
            out = self.ops.combine(st["y"], st["pos_of"], st["idx"], n_tokens)
        if self.cuda:
            torch.cuda.current_stream(self.devices[r]).wait_stream(self.streams[r])
        return out

    def layer(self, hs, routers, gate_ups, downs):
        """hs[r]: [N_r, H] hidden states of rank r's tokens; routers[r]: the layer's router (on
        every rank); gate_ups[r], downs[r]: rank r's experts of this layer. Returns the MoE
        output for every rank's tokens. All ranks from this thread."""
        sh = self.new_state()
        for r in range(self.R):
            self.phase_send(sh, r, hs[r], routers[r])
        for d in range(self.R):
            self.phase_compute(sh, d, hs[d], gate_ups[d], downs[d])
        outs = [self.phase_combine(sh, r, hs[r].shape[0]) for r in range(self.R)]
        self._keep = sh  # buffers stay referenced until the next layer
        return outs


    # --- one thread per rank ---
    def layer_rank(self, sh, r, h, router, gate_up, down):
        """Rank r's part of one layer, called from rank r's thread (every rank with the same
        shared state `sh`); the barriers order the phases across ranks."""
        self.phase_send(sh, r, h, router)
        self.barrier.wait()
        self.phase_compute(sh, r, h, gate_up, down)
        self.barrier.wait()
        return self.phase_combine(sh, r, h.shape[0])

    def run_ranks(self, fn):
        """fn(r) in one thread per rank (a persistent pool); returns the results in rank order.
        If one rank fails the barrier is broken so the others stop instead of waiting forever."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.R, thread_name_prefix="switchyard-rank")

        def guarded(r):
            try:
                if self.cuda:
                    torch.cuda.set_device(self.devices[r])
                return fn(r)
            except BaseException:
                self.barrier.abort()
                raise

        futures = [self._pool.submit(guarded, r) for r in range(self.R)]
        errors = [f.exception() for f in futures]  # waits for every rank
        if self.barrier.broken:
            self.barrier.reset()
        # raise the rank that failed first, not the others' BrokenBarrierError
        real = [e for e in errors if e is not None and not isinstance(e, threading.BrokenBarrierError)]
        if real or any(errors):
            raise (real or [e for e in errors if e is not None])[0]
        return [f.result() for f in futures]

    def layer_threaded(self, hs, routers, gate_ups, downs):
        """Same as layer(), every rank driven by its own thread."""
        sh = self.new_state()
        outs = self.run_ranks(lambda r: self.layer_rank(sh, r, hs[r], routers[r], gate_ups[r], downs[r]))
        self._keep = sh
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
