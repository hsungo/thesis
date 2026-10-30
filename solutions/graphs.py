"""CUDA graphs for the uniform steps of EESDEngine (``EngineConfig.graphs``) - removes kernel-launch overhead.

Every graph runs all B cache rows 0..B-1 with the same number of tokens per row (T, padded) and reads the
same number of cache positions (L, padded to 64), so its shapes and addresses are fixed:
    layers(start, end, T, L)   embed or gather the inputs from static index buffers, run decoder layers
                               [start, end) as one packed bucket (contiguous rows: no KV gather), optionally
                               store the outputs into exit_h; the output goes to a buffer shared per T
    head_draft(T=1)            final norm + LM head + argmax of the B rows, scattered into seq_buf
    head_verify(T)             LM head + argmax of the B*T verify tokens
    accept(T)                  the greedy acceptance of EESDEngine._deep (first mismatch per row, bonus token)
A row (or token) that does not take part is padded: its position is the scratch position max_seq-1 (never
read: real queries attend to positions <= their own) and its outputs go to the scratch row B of seq_buf /
exit_h.  The arithmetic per real token is that of solutions/packed.py.  Steps that do not have this shape
(M3 chunks, M2 probes, separate prefill) run eagerly.  Graphs are captured lazily and shared one memory pool.
"""

from __future__ import annotations

import numpy as np
import torch

from solutions.packed import Bucket, PackedBatch, packed_layers

L_STEP = 64


def round_up(x: int, m: int = L_STEP) -> int:
    return (x + m - 1) // m * m


class GraphRunner:
    def __init__(self, model, B: int, seq_buf: torch.Tensor, exit_h: torch.Tensor, d_max: int):
        self.m, self.B, self.dev = model, B, model.device
        self.seq_flat = seq_buf.view(-1)                 # [(B+1) * Dm1], row B = scratch
        self.exh_flat = exit_h.view(-1, exit_h.shape[-1])
        self.Dm1 = d_max + 1
        self.S = B * self.Dm1                            # scratch index in seq_flat / exh_flat
        self.max_seq = model.cache.max_seq
        self.scratch_pos = self.max_seq - 1
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict = {}
        self.inputs: dict = {}                           # (name, n) -> static long buffer
        self.hbuf: dict = {}                             # T -> [B*T, dim] layer output shared by graphs of that T
        self.ybuf: dict = {}                             # T -> [B*T] verifier argmax
        self.acc: dict = {}                              # T -> [B+1] accepted drafts per row (graph output)
        self.n_captured = 0

    # ------------------------------------------------------------------ buffers
    def buf(self, name: str, n: int, fill: int = 0) -> torch.Tensor:
        key = (name, n)
        if key not in self.inputs:
            self.inputs[key] = torch.full((n,), fill, dtype=torch.long, device=self.dev)
        return self.inputs[key]

    def set(self, name: str, arr) -> None:
        a = np.asarray(arr, dtype=np.int64)
        self.buf(name, len(a)).copy_(torch.from_numpy(a).pin_memory(), non_blocking=True)

    def out(self, T: int) -> torch.Tensor:
        if T not in self.hbuf:
            self.hbuf[T] = torch.zeros(self.B * T, self.m.cfg.dim, dtype=self.m.dtype, device=self.dev)
        return self.hbuf[T]

    # ------------------------------------------------------------------ bodies
    def _packed(self, pos: torch.Tensor, T: int, L: int) -> PackedBatch:
        m, B, G = self.m, self.B, self.m.group
        rows = torch.arange(B, device=self.dev).repeat_interleave(T)
        cos = m.rope_cos[pos].to(m.dtype).unsqueeze(1)
        sin = m.rope_sin[pos].to(m.dtype).unsqueeze(1)
        keys = torch.arange(L, device=self.dev)
        msk = keys.view(1, 1, L) <= pos.view(B, T, 1)
        mask = msk.unsqueeze(1).unsqueeze(1).expand(B, 1, G, T, L).reshape(B, 1, G * T, L)
        pb = PackedBatch([], np.arange(B + 1) * T, pos, rows, cos, sin, n_tokens=B * T)
        pb.buckets.append(Bucket(0, B * T, B, T, L, None, 0, mask))
        return pb

    def _layers_body(self, start: int, end: int, T: int, L: int, src: str, store_exit: bool) -> None:
        n = self.B * T
        idx, pos = self.buf(f"{src}_idx", n), self.buf(f"pos{T}", n)
        if src == "seq":                                  # token ids from seq_buf (shallow / AR)
            x = self.m.embed(self.seq_flat[idx])
        else:                                             # layer-k hidden states from exit_h (deep)
            x = self.exh_flat[idx]
        h = packed_layers(self.m, x, self._packed(pos, T, L), start, end)
        if store_exit:
            self.exh_flat[self.buf("exit_dst", n)] = h
        self.out(T).copy_(h)

    def _head_draft_body(self) -> None:
        y = self.m.head(self.out(1)).argmax(-1)
        self.seq_flat[self.buf("drf_dst", self.B)] = y

    def _head_verify_body(self, T: int) -> None:
        self.ybuf[T].copy_(self.m.head(self.out(T)).argmax(-1))

    def _accept_body(self, T: int) -> None:
        B, n = self.B, self.B * T
        Yv = self.ybuf[T]
        dsrc, jt, bt = self.buf(f"dsrc{T}", n), self.buf(f"j{T}", n), self.buf(f"blk{T}", n)
        dr = torch.where(dsrc >= 0, self.seq_flat[dsrc.clamp(min=0)], torch.full_like(dsrc, -1))
        big = torch.full((B + 1,), T + 10, dtype=torch.long, device=self.dev)
        acc = big.scatter_reduce(0, bt, torch.where(Yv != dr, jt, torch.full_like(jt, T + 10)), reduce="amin",
                                 include_self=True)
        starts = torch.arange(B + 1, device=self.dev) * T
        last = Yv[(starts + acc).clamp(max=n - 1)]
        self.seq_flat[self.buf(f"col0_{T}", B + 1)] = last
        self.acc[T] = acc

    # ------------------------------------------------------------------ capture / replay
    def _graph(self, key, body):
        g = self.graphs.get(key)
        if g is None:
            s = torch.cuda.Stream(self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()
            torch.cuda.current_stream(self.dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.graphs[key] = g
            self.n_captured += 1
        return g

    def layers(self, start: int, end: int, T: int, L: int, src: str, store_exit: bool) -> torch.Tensor:
        self._graph(("layers", start, end, T, L, src, store_exit),
                    lambda: self._layers_body(start, end, T, L, src, store_exit)).replay()
        return self.out(T)

    def head_draft(self) -> None:
        self._graph(("head_draft",), self._head_draft_body).replay()

    def head_verify(self, T: int) -> torch.Tensor:
        self._graph(("head_verify", T), lambda: self._head_verify_body(T)).replay()
        return self.ybuf[T]

    def accept(self, T: int) -> torch.Tensor:
        self._graph(("accept", T), lambda: self._accept_body(T)).replay()
        return self.acc[T]

    def precapture(self, segments, deep, Ts, max_L: int) -> None:
        """Capture ahead of the run so no capture happens inside a measured window.
        segments: (start, end, src, store_exit) layer ranges with T = 1; deep: (start, end) of the verify
        segment, Ts: its token counts per row."""
        Ls = range(L_STEP, min(round_up(max_L), self.max_seq) + 1, L_STEP)
        for (a, b, src, st) in segments:
            for L in Ls:
                self.layers(a, b, 1, L, src, st)
        self.head_draft()
        for T in (Ts if deep else ()):
            for L in Ls:
                self.layers(deep[0], deep[1], T, L, "exh", False)
            self.head_verify(T)
            self.accept(T)

    def init_buffers(self, Ts) -> None:
        """Scratch-filled inputs so that warm-up / capture runs touch only scratch rows and positions."""
        B, S, sp = self.B, self.S, self.scratch_pos
        for T in sorted(set(Ts) | {1}):
            n = B * T
            self.buf(f"pos{T}", n).fill_(sp)
            for name in ("seq_idx", "exh_idx"):
                self.buf(name, n).fill_(S)
            self.buf(f"dsrc{T}", n).fill_(-1)
            self.buf(f"j{T}", n).zero_()
            self.buf(f"blk{T}", n).fill_(B)
            self.buf(f"col0_{T}", B + 1).fill_(S)
            self.out(T)
            self.ybuf.setdefault(T, torch.zeros(n, dtype=torch.long, device=self.dev))
        self.buf("exit_dst", B).fill_(S)
        self.buf("drf_dst", B).fill_(S)
