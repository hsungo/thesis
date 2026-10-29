"""EESD engine: batched early-exit self-speculation with packed (ragged) shallow/deep steps.

Step types (spec mode)
    shallow step   every decoding slot runs its current input token (last committed token if d=0,
                   else its newest draft) through layers [0,k) and the shared exit head -> one draft.
    deep step      the verifying slots run their newest draft through [0,k) and then all d_i+1
                   tokens (inputs = cached layer-k hidden states) through [k,L) + head; greedy
                   acceptance; slots that do not verify draft in the same step.
    Prefill chunks (M3) can ride on either segment.  All tokens of a segment are packed into one
    axis (qo_indptr = cumsum(T_i)) - see solutions/packed.py.

Rollback: a request's KV "length pointer" is its committed position `pos`; rejected drafts leave
stale KV beyond it that is never read (queries read positions <= their own) and is overwritten
by the next round - nothing is moved.

Host state (NumPy, one entry per slot): state (free / prefilling / decoding), pos, d, generated
count; alpha lives in the scheduler.  GPU state: seq_buf[B, d_max+1] (column 0 = last committed
token, columns 1.. = drafts) and exit_h[B, d_max+1, dim] (layer-k hidden states).  The host only
reads the GPU after a deep step (accepted counts, verified tokens, probe counts, first tokens of
finished prefills) - shallow steps and the scheduler never synchronise.

Modes: ``mode="spec"`` (sched x exit x prefill switches) and ``mode="ar"`` (all layers, 1 token).
Drivers: ``run(prompts, "cyclic")`` (ASPIRE cyclic replay: warm-up completions dropped, run until
the decode throughput converges), ``"static"`` (every prompt once; losslessness tests) and
``run_poisson`` (requests arrive by a Poisson process in wall-clock time; B varies over time).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from common.timing import Interval
from solutions.exit_select import ExitController
from solutions.packed import Block, build_packed, packed_layers, upload
from solutions.prefill import M3Planner, split_separate
from solutions.schedulers import make_scheduler

FREE, PREFILL, DECODE = 0, 1, 2


@dataclass
class EngineConfig:
    mode: str = "spec"              # spec | ar
    sched: str = "fixed"            # fixed | aspire | m1
    gamma: int = 4
    k: int = 3
    exit: str = "fixed"             # fixed | m2
    prefill: str = "separate"       # separate | m3
    d_max: int = 16
    sla_steps: int = 32
    initial_draft: int = 3
    omega: float = 0.8
    alpha0: float = 0.9
    probe_every: int = 20
    k_cands: tuple = (2, 3, 4, 5, 6, 7, 8)
    switch_margin: float = 0.05
    max_wait: int = 3
    knee_safety: float = 0.9
    prefill_chunk: int = 512
    bucketing: str = "exact"        # exact | pad (attention bucketing, packed GEMMs either way)
    m3_chunk_cap: Optional[int] = None   # tests: force small chunks
    m2_force: Optional[str] = None       # tests: "cycle"

    def label(self) -> str:
        if self.mode == "ar":
            return "ar"
        s = f"{self.sched}" + (f"-g{self.gamma}" if self.sched == "fixed" else "")
        return f"{s}-{'k' + str(self.k) if self.exit == 'fixed' else 'm2'}-{self.prefill}"


@dataclass
class Request:
    pid: str
    prompt: list
    out: list = field(default_factory=list)
    finish: Optional[str] = None
    rec_assign: int = 0          # index into engine.records when the slot was assigned
    rec_first: int = -1          # record that produced the first generated token
    rec_finish: int = -1
    wall_arrival: float = 0.0
    wall_first: float = 0.0
    measured: bool = False
    pf_sh: int = 0               # prompt tokens through the shallow segment (M3)
    pf_dp: int = 0               # prompt tokens through all layers (M3)
    piggybacked: int = 0         # prompt tokens prefilled by piggybacking (M3)


@dataclass
class Rec:
    kind: str                    # shallow | deep | probe | ar | prefill
    iv: Interval
    measured: bool
    n_sh: int = 0
    n_dp: int = 0
    n_chunk: int = 0
    plan_ms: float = 0.0
    k: int = 0
    _ms: Optional[float] = None

    @property
    def ms(self) -> float:
        if self._ms is None:
            self._ms = self.iv.ms
        return self._ms


class EESDEngine:
    def __init__(self, model, B: int, max_prompt: int, max_new: int, eos_ids, cfg: EngineConfig, cost_model,
                 stop_for: Optional[Callable] = None):
        if cfg.mode == "ar" and cfg.prefill != "separate":
            raise ValueError("AR mode supports --prefill separate only")
        self.m, self.dev, self.B, self.cfg, self.cm = model, model.device, B, cfg, cost_model
        self.max_new, self.eos = max_new, set(eos_ids)
        self.stop_for = stop_for or (lambda pid: None)
        self.k = cfg.k
        self.L = model.n_layers
        if not 1 <= self.k < self.L:
            raise ValueError("exit layer must be in [1, n_layers)")
        Dm = cfg.d_max
        model.allocate_cache(B, max_prompt + max_new + Dm + 40)
        self.seq_buf = torch.zeros(B, Dm + 1, dtype=torch.long, device=self.dev)
        self.exit_h = torch.zeros(B, Dm + 1, model.cfg.dim, dtype=model.dtype, device=self.dev)
        self.state = np.zeros(B, dtype=np.int64)
        self.pos = np.zeros(B, dtype=np.int64)
        self.d = np.zeros(B, dtype=np.int64)
        self.gen = np.zeros(B, dtype=np.int64)
        self.req: list[Optional[Request]] = [None] * B
        self.pending: list[list] = [[] for _ in range(B)]   # M3: (pos0, c, hidden at layer k) awaiting deep
        self.steps_since_deep = 0
        self.records: list[Rec] = []
        self.sched = make_scheduler(cfg.sched, B, cost_model, self.k, gamma=cfg.gamma, d_max=Dm,
                                    sla_steps=cfg.sla_steps, initial_draft=cfg.initial_draft, omega=cfg.omega,
                                    alpha0=cfg.alpha0) if cfg.mode == "spec" else None
        self.exitc = ExitController(cost_model, self.k, [c for c in cfg.k_cands if 1 <= c < self.L], cfg.probe_every,
                                    cfg.switch_margin, cfg.omega, Dm, cfg.m2_force) if cfg.exit == "m2" else None
        self.m3 = M3Planner(cost_model, cfg.max_wait, cfg.m3_chunk_cap, cfg.prefill_chunk, cfg.knee_safety) if cfg.prefill == "m3" else None
        self.measuring = False
        self.stats = {"steps": 0, "deep_steps": 0, "probe_steps": 0, "deep_tokens": 0, "verifiers": 0, "drafted": 0,
                      "accepted": 0, "committed": 0, "plan_ms": 0.0, "plan_calls": 0, "prefill_tokens": 0,
                      "piggyback_tokens": 0, "separate_prefills": 0, "m3_fallbacks": 0, "dropped_drafts": 0}
        self.round_len = cfg.gamma   # last observed steps per round (M3 deep-size estimate)
        self.fallback_reasons: dict = {}

    # ------------------------------------------------------------------ helpers
    def _rec(self, kind, iv, **kw) -> Rec:
        r = Rec(kind, iv, self.measuring, k=self.k, **kw)
        self.records.append(r)
        return r

    def _commit(self, slot: int, toks) -> int:
        r = self.req[slot]
        n = 0
        for t in toks:
            if r.finish is not None:
                break
            r.out.append(int(t))
            n += 1
            if int(t) in self.eos:
                r.finish = "eos"
            elif len(r.out) >= self.max_new:
                r.finish = "length"
        stop = self.stop_for(r.pid)
        if n and stop and r.finish is None:
            cut = stop(r.out, n)
            if cut is not None:
                n -= len(r.out) - cut
                del r.out[cut:]
                r.finish = "stop"
        self.gen[slot] = len(r.out)
        return n

    def _first_token(self, slot: int, y0: int) -> int:
        r = self.req[slot]
        r.rec_first = len(self.records) - 1
        r.wall_first = time.perf_counter()
        self.state[slot] = DECODE
        self.pos[slot] = len(r.prompt)
        self.d[slot] = 0
        if self.sched is not None:
            self.sched.reset(slot)
        return self._commit(slot, [y0])

    # ------------------------------------------------------------------ assignment / prefill
    def assign(self, slot: int, req: Request) -> int:
        """Put a request in a free slot.  Separate prefill runs now (the batch pauses); M3 queues it."""
        req.rec_assign = len(self.records)
        req.measured = self.measuring
        self.req[slot] = req
        self.pending[slot] = []
        self.d[slot] = 0
        if self.m3 is not None and (self.state == DECODE).any():
            dec = self.state == DECODE
            n_dec = int(dec.sum())
            ok, why = self.m3.admit(len(req.prompt), n_dec, n_dec * (self.round_len + 1), self.k,
                                    float((self.pos + self.d)[dec].sum()))
            if ok:
                self.state[slot] = PREFILL
                req.pf_sh = req.pf_dp = 0
                return 0
            self.stats["m3_fallbacks"] += 1
            self.fallback_reasons[why] = self.fallback_reasons.get(why, 0) + 1
        return self._prefill_separate(slot)

    @torch.inference_mode()
    def _prefill_separate(self, slot: int) -> int:
        m, r = self.m, self.req[slot]
        iv = Interval(self.dev).start()
        h = None
        for s, c in split_separate(len(r.prompt), self.cfg.prefill_chunk):
            pb = build_packed(m, [Block(slot, s, c)], self.cfg.bucketing)
            x = m.embed(upload(r.prompt[s:s + c], self.dev).unsqueeze(0))[0]
            h = packed_layers(m, x, pb, 0, self.L)
        y0 = m.head(h[-1:]).argmax(-1)
        self.seq_buf[slot, 0] = y0[0]
        iv.stop()
        self._rec("prefill", iv, n_chunk=len(r.prompt))
        self.stats["separate_prefills"] += 1
        if self.measuring:
            self.stats["prefill_tokens"] += len(r.prompt)
        return self._first_token(slot, int(y0.item()))

    # ------------------------------------------------------------------ one step
    @torch.inference_mode()
    def step(self) -> tuple[int, list]:
        """Run one step; returns (committed tokens, slots that finished)."""
        if self.cfg.mode == "ar":
            return self._ar_step()
        return self._spec_step()

    def _ar_step(self):
        m = self.m
        dec = np.flatnonzero(self.state == DECODE)
        if len(dec) == 0:
            return 0, []
        pb = build_packed(m, [Block(int(i), int(self.pos[i]), 1) for i in dec], self.cfg.bucketing)
        slots = [b.row for b in pb.blocks]
        iv = Interval(self.dev).start()
        x = m.embed(self.seq_buf[upload(slots, self.dev), 0])
        h = packed_layers(m, x, pb, 0, self.L)
        y = m.head(h).argmax(-1)
        self.seq_buf[upload(slots, self.dev), 0] = y
        iv.stop()
        self._rec("ar", iv, n_sh=len(slots), n_dp=len(slots))
        y_h = y.tolist()
        committed, done = 0, []
        for s, t in zip(slots, y_h):
            self.pos[s] += 1
            committed += self._commit(s, [t])
            if self.req[s].finish:
                done.append(s)
        self._account(committed, deep=True, n_deep=len(slots), verifiers=len(slots))
        return committed, done

    def _spec_step(self):
        m, cfg, k, L = self.m, self.cfg, self.k, self.L
        Dm1 = cfg.d_max + 1
        t0 = time.perf_counter()
        dec = self.state == DECODE
        d = self.d.copy()
        ctx = np.where(dec, self.pos + d, 0)
        V = self.sched.decide(d, ctx, dec, self.steps_since_deep) if dec.any() else np.zeros(self.B, dtype=bool)
        V = (V | (dec & (d >= cfg.d_max))) & dec & (d >= 1)
        # M3 progress guarantee: with no decoding slot left, prefilling requests run their chunks
        # through both segments in the same step (otherwise no deep step would ever come)
        flush = (not dec.any()) and bool((self.state == PREFILL).any())
        deep = bool(V.any()) or flush
        Dr = dec & ~V                                   # draft this step
        probe = deep and self.exitc is not None and self.exitc.want_probe()
        # ---- M3: chunks riding on this step's shallow segment
        chunks = []
        pf = np.flatnonzero(self.state == PREFILL)
        if len(pf):
            n_sh = int(dec.sum()) + (int((d[V] + 1).sum() - V.sum()) if probe else 0)
            n_dp_est = int((d[V] + 1).sum()) if deep else int(dec.sum()) * (max(self.round_len, 1) + 1)
            pend = sum(c for i in range(self.B) for (_, c, _) in self.pending[i])
            C_sh = float(ctx[dec].sum())
            for i in pf:
                r = self.req[i]
                rem = len(r.prompt) - r.pf_sh
                if rem <= 0:
                    continue
                c = min(rem, self.m3.budget(0 if flush else n_sh, 0 if flush else n_dp_est, 0 if flush else pend))
                if flush and c <= 0:
                    c = min(rem, self.cfg.prefill_chunk)
                if c <= 0:
                    continue
                chunks.append((int(i), r.pf_sh, c))
                r.pf_sh += c
                n_sh += c
                pend += c
        plan_ms = 1000 * (time.perf_counter() - t0)

        # ---- shallow segment blocks
        blocks, src_seq = [], {}
        for i in np.flatnonzero(dec):
            i = int(i)
            if probe and V[i]:
                blocks.append(Block(i, int(self.pos[i]), int(d[i]) + 1, ("full", i)))
            else:
                blocks.append(Block(i, int(self.pos[i] + d[i]), 1, ("sh", i)))
        for (i, p0, c) in chunks:
            blocks.append(Block(i, p0, c, ("chunk", i)))
        if not blocks:
            return 0, []
        pb = build_packed(m, blocks, cfg.bucketing)
        N = pb.n_tokens
        seq_idx = np.full(N, -1, dtype=np.int64)
        prompt_ids = np.zeros(N, dtype=np.int64)
        exit_src, exit_dst, drf_src, drf_dst = [], [], [], []
        chunk_slices = []
        for bi, b in enumerate(pb.blocks):
            s = int(pb.offsets[bi])
            kind, i = b.tag
            if kind == "sh":
                seq_idx[s] = i * Dm1 + d[i]
                exit_src.append(s)
                exit_dst.append(i * Dm1 + d[i])
                if Dr[i]:
                    drf_src.append(s)
                    drf_dst.append(i * Dm1 + d[i] + 1)
            elif kind == "full":
                seq_idx[s:s + b.T] = i * Dm1 + np.arange(b.T)
                exit_src.extend(range(s, s + b.T))
                exit_dst.extend(i * Dm1 + np.arange(b.T))
            else:
                prompt_ids[s:s + b.T] = self.req[i].prompt[b.pos0:b.pos0 + b.T]
                chunk_slices.append((i, b.pos0, b.T, s))
        iv = Interval(self.dev).start()
        sidx = upload(seq_idx, self.dev)
        ids = torch.where(sidx >= 0, self.seq_buf.view(-1)[sidx.clamp(min=0)], upload(prompt_ids, self.dev))
        cap_sh = {} if probe else None
        h = packed_layers(m, m.embed(ids), pb, 0, k, cap_sh, tuple(c for c in (self.exitc.cands if probe else ()) if c < k))
        exh = self.exit_h.view(-1, self.exit_h.shape[-1])
        exh[upload(exit_dst, self.dev)] = h[upload(exit_src, self.dev)]
        if drf_src:
            self.seq_buf.view(-1)[upload(drf_dst, self.dev)] = m.head(h[upload(drf_src, self.dev)]).argmax(-1)
        for (i, p0, c, s) in chunk_slices:
            self.pending[i].append((p0, c, h[s:s + c]))
        readback = None
        n_v_tok = 0
        if deep:
            readback, n_v_tok, vinfo, cinfo = self._deep(V, d, probe, h, pb, cap_sh)
        iv.stop()
        kind = "probe" if probe else ("deep" if deep else "shallow")
        self._rec(kind, iv, n_sh=N, n_dp=n_v_tok, n_chunk=sum(c for (_, _, c) in chunks), plan_ms=plan_ms)
        # ---- host bookkeeping (no GPU access until here; a deep step synchronises once in .cpu())
        self.d[Dr] += 1
        if self.measuring:
            self.stats["piggyback_tokens"] += sum(c for (_, _, c) in chunks)
        for (i, _, c) in chunks:
            self.req[i].piggybacked += c
        committed, done = 0, []
        if not deep:
            self.steps_since_deep += 1
            self._account(0, deep=False, plan_ms=plan_ms)
            return 0, []
        rb = readback.tolist()
        vslots, vstarts, vT = vinfo
        nv = len(vslots)
        acc = np.array(rb[:nv], dtype=np.int64)
        Y = rb[nv:nv + n_v_tok]
        accepted = np.zeros(self.B, dtype=np.int64)
        for j, i in enumerate(vslots):
            a = int(acc[j])
            accepted[i] = a
            toks = Y[vstarts[j]:vstarts[j] + a + 1]
            self.pos[i] += a + 1
            self.d[i] = 0
            committed += self._commit(i, toks)
            if self.measuring and self.req[i].finish is None:
                self.stats["drafted"] += int(d[i])
                self.stats["accepted"] += a
            if self.req[i].finish:
                done.append(i)
        off = nv + n_v_tok
        for j, i in enumerate(cinfo):
            committed += self._first_token(i, int(rb[off + j]))
            if self.req[i].finish:
                done.append(i)
        off += len(cinfo)
        self.sched.observe(V, accepted, d)
        self.round_len = self.steps_since_deep
        self.steps_since_deep = 0
        t2 = time.perf_counter()
        if self.exitc is not None:
            if probe:
                nc = len(self.exitc.cands)
                self.exitc.observe_probe(rb[off:off + nc], int(rb[off + nc]))
            self.exitc.after_deep_step()
            dec_now = self.state == DECODE
            new = self.exitc.maybe_switch(int(dec_now.sum()), float((self.pos + self.d)[dec_now].sum()),
                                          can_switch=not (self.state == PREFILL).any())
            if new is not None:
                self._switch_exit(new)
        plan_ms += 1000 * (time.perf_counter() - t2)          # M2 decision is planning time too
        self._account(committed, deep=True, n_deep=n_v_tok, verifiers=nv, plan_ms=plan_ms, probe=probe)
        return committed, done

    def _deep(self, V, d, probe, h_sh, pb_sh, cap_sh):
        """Deep segment of a deep step (inside the step's timing interval).  Returns the packed
        int64 tensor to read back: [accepted per verifier | Y of verify tokens | first tokens of
        finished prefills | probe matches per candidate, probe total]."""
        m, cfg, k, L = self.m, self.cfg, self.k, self.L
        Dm1 = cfg.d_max + 1
        blocks = []
        for i in np.flatnonzero(V):
            blocks.append(Block(int(i), int(self.pos[i]), int(d[i]) + 1, ("v", int(i))))
        for i in range(self.B):
            for (p0, c, hk) in self.pending[i]:
                blocks.append(Block(i, p0, c, ("cd", i, hk)))
        pb = build_packed(m, blocks, cfg.bucketing)
        pieces, exit_rows = [], []
        vslots, vstarts, vT, vtok_packed = [], [], [], []
        cfin, cfin_idx = [], []
        flat_v, j_loc, blk_of, dsrc = [], [], [], []
        for bi, b in enumerate(pb.blocks):
            s = int(pb.offsets[bi])
            if b.tag[0] == "v":
                i = b.tag[1]
                idx = i * Dm1 + np.arange(b.T)
                pieces.append(("exit", idx))
                vstarts.append(len(vtok_packed))
                vslots.append(i)
                vT.append(b.T)
                vtok_packed.extend(range(s, s + b.T))
                j = np.arange(b.T)
                j_loc.extend(j)
                blk_of.extend([len(vslots) - 1] * b.T)
                dsrc.extend(np.where(j < b.T - 1, i * Dm1 + j + 1, -1))
            else:
                i, hk = b.tag[1], b.tag[2]
                pieces.append(("h", hk))
                if b.pos0 + b.T >= len(self.req[i].prompt):
                    cfin.append(i)
                    cfin_idx.append(s + b.T - 1)
        exh = self.exit_h.view(-1, self.exit_h.shape[-1])
        xs = [exh[upload(p[1], self.dev)] if p[0] == "exit" else p[1] for p in pieces]
        x = torch.cat(xs) if len(xs) > 1 else xs[0]
        cands = self.exitc.cands if probe else ()
        cap_dp = {} if probe else None
        hd = packed_layers(m, x, pb, k, L, cap_dp, tuple(c for c in cands if c > k))
        out = []
        nvt = len(vtok_packed)
        if nvt:
            vt = upload(vtok_packed, self.dev)
            Yv = m.head(hd[vt]).argmax(-1)
            dsrc_t = upload(dsrc, self.dev)
            dr = torch.where(dsrc_t >= 0, self.seq_buf.view(-1)[dsrc_t.clamp(min=0)], torch.full_like(dsrc_t, -1))
            jt = upload(j_loc, self.dev)
            bt = upload(blk_of, self.dev)
            nb = len(vslots)
            big = torch.full((nb,), cfg.d_max + 10, dtype=torch.long, device=self.dev)
            first_mis = big.scatter_reduce(0, bt, torch.where(Yv != dr, jt, torch.full_like(jt, cfg.d_max + 10)),
                                           reduce="amin", include_self=True)
            acc = first_mis                                           # last token always "mismatches" (dr = -1)
            starts = upload(vstarts, self.dev)
            last_tok = Yv[starts + acc]
            self.seq_buf[upload(vslots, self.dev), 0] = last_tok
            out += [acc, Yv]
        if cfin:
            y0 = m.head(hd[upload(cfin_idx, self.dev)]).argmax(-1)
            self.seq_buf[upload(cfin, self.dev), 0] = y0
            out.append(y0)
        if probe and nvt:
            out.append(self._probe_counts(V, d, vslots, vT, vtok_packed, Yv, acc, x, hd, cap_dp, h_sh, pb_sh, cap_sh, bt, jt))
        # pending chunks are consumed by this deep step
        for i in range(self.B):
            for (p0, c, _) in self.pending[i]:
                self.req[i].pf_dp = p0 + c
            self.pending[i] = []
        rb = torch.cat([o.to(torch.long).reshape(-1) for o in out]).cpu() if out else torch.zeros(0, dtype=torch.long)
        return rb, nvt, (vslots, vstarts, vT), cfin

    def _probe_counts(self, V, d, vslots, vT, vtok_packed, Yv, acc, x_k, hd, cap_dp, h_sh, pb_sh, cap_sh, bt, jt):
        """Agreement of every candidate exit with Y on the positions with a correct prefix (j <= acc)."""
        m = self.m
        Dm1 = self.cfg.d_max + 1
        # map verify tokens -> their row in the shallow segment (the probe ran all d+1 tokens there)
        sh_pos = {}
        for bi, b in enumerate(pb_sh.blocks):
            if b.tag[0] == "full":
                sh_pos[b.tag[1]] = int(pb_sh.offsets[bi])
        sh_idx = np.concatenate([sh_pos[i] + np.arange(T) for i, T in zip(vslots, vT)])
        sh_t = upload(sh_idx, self.dev)
        vt = upload(vtok_packed, self.dev)
        hs = []
        for c in self.exitc.cands:
            if c < self.k:
                hs.append(cap_sh[c][sh_t])
            elif c == self.k:
                hs.append(x_k[vt])
            else:
                hs.append(cap_dp[c][vt])
        pred = m.head(torch.cat(hs)).argmax(-1).view(len(hs), -1)
        valid = jt <= acc[bt]
        matches = ((pred == Yv.unsqueeze(0)) & valid.unsqueeze(0)).sum(-1)
        return torch.cat([matches, valid.sum().view(1)])

    def _switch_exit(self, new_k: int) -> None:
        dropped = int(self.d[self.state == DECODE].sum())
        self.stats["dropped_drafts"] += dropped
        self.d[:] = 0
        self.k = new_k
        self.sched.set_exit(new_k, self.exitc.alpha.get(new_k))

    def _account(self, committed, deep, n_deep=0, verifiers=0, plan_ms=0.0, probe=False):
        if not self.measuring:
            return
        st = self.stats
        st["steps"] += 1
        st["committed"] += committed
        st["plan_ms"] += plan_ms
        st["plan_calls"] += 1
        if deep:
            st["deep_steps"] += 1
            st["deep_tokens"] += n_deep
            st["verifiers"] += verifiers
            st["probe_steps"] += int(probe)

    # ------------------------------------------------------------------ drivers
    def run(self, prompts: list, mode: str = "cyclic", warmup_samples: int = 8, warmup_per_slot: float = 1.0,
            min_eval: int = 16, min_eval_per_slot: float = 2.0, max_eval: int = 256, converge_threshold: float = 0.01,
            converge_window: int = 4, keep_outputs: bool = False, max_steps: Optional[int] = None) -> dict:
        B, n_pool = self.B, len(prompts)
        warmup_n = 0 if mode == "static" else max(warmup_samples, math.ceil(warmup_per_slot * B))
        min_eval_n = max(min_eval, math.ceil(min_eval_per_slot * B))
        max_eval_n = max(max_eval, min_eval_n)
        self.measuring = warmup_n == 0
        cursor, total_done, evaluated, stable, anchor, stop_reason = 0, 0, 0, 0, None, None
        completions = []
        rec_measure_start = 0
        tok_measured = 0

        def make(p):
            return Request(p.pid, list(p.tokens))

        for slot in range(B):
            if mode == "static" and cursor >= n_pool:
                break
            self.assign(slot, make(prompts[cursor % n_pool]))
            cursor += 1
        pre_done = [i for i in range(B) if self.req[i] is not None and self.req[i].finish]
        steps = 0
        while True:
            if not pre_done and not ((self.state == DECODE) | (self.state == PREFILL)).any():
                stop_reason = "pool_exhausted"
                break
            if pre_done:
                committed, done = 0, pre_done
                pre_done = []
            else:
                committed, done = self.step()
                steps += 1
            if self.measuring:
                tok_measured += committed
            queue = list(done)
            while queue:
                slot = queue.pop(0)
                r = self.req[slot]
                if r is None:
                    continue
                r.rec_finish = len(self.records) - 1
                completions.append(self._completion(r, keep_outputs))
                total_done += 1
                self.state[slot] = FREE
                self.req[slot] = None
                if not self.measuring:
                    if total_done >= warmup_n:
                        self.measuring = True
                        rec_measure_start = len(self.records)
                elif mode == "cyclic":
                    evaluated += 1
                    if evaluated >= min_eval_n:
                        thr = self._throughput(rec_measure_start, tok_measured)
                        if anchor is None:
                            anchor = thr
                        elif abs(thr - anchor) / max(anchor, 1e-9) < converge_threshold:
                            stable += 1
                        else:
                            stable, anchor = 0, thr
                        if stable >= converge_window:
                            stop_reason = "converged"
                    if stop_reason is None and evaluated >= max_eval_n:
                        stop_reason = "max_eval_samples"
                if stop_reason is None and (mode == "cyclic" or cursor < n_pool):
                    self.assign(slot, make(prompts[cursor % n_pool]))
                    cursor += 1
                    if self.req[slot].finish:          # finished at its first token
                        queue.append(slot)
            if stop_reason is not None:
                break
            if max_steps is not None and steps >= max_steps:
                stop_reason = "max_steps"
                break
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        return self._result(completions, rec_measure_start, tok_measured, stop_reason)

    def run_poisson(self, prompts: list, rate_per_s: float, n_requests: int, warm_requests: int = 8, seed: int = 0,
                    keep_outputs: bool = False) -> dict:
        """Open loop: requests arrive by a Poisson process in wall-clock time; free slots take them FIFO."""
        rng = np.random.default_rng(seed)
        B = self.B
        queue, completions, cursor = [], [], 0
        t0 = time.perf_counter()
        next_arr = t0 + rng.exponential(1 / rate_per_s)
        rec_measure_start, tok_measured = 0, 0
        self.measuring = warm_requests == 0
        total_done = 0
        t_meas = t0
        while total_done < n_requests + warm_requests:
            now = time.perf_counter()
            while next_arr <= now:
                p = prompts[cursor % len(prompts)]
                cursor += 1
                queue.append(Request(p.pid, list(p.tokens), wall_arrival=next_arr))
                next_arr += rng.exponential(1 / rate_per_s)
            instant = []
            for slot in np.flatnonzero(self.state == FREE):
                if not queue:
                    break
                self.assign(int(slot), queue.pop(0))
                if self.req[int(slot)].finish:
                    instant.append(int(slot))
            if not instant and not ((self.state == DECODE) | (self.state == PREFILL)).any():
                time.sleep(max(0.0, min(next_arr - time.perf_counter(), 0.01)))
                continue
            committed, done = (0, instant) if instant else self.step()
            if self.measuring:
                tok_measured += committed
            for slot in done:
                r = self.req[slot]
                r.rec_finish = len(self.records) - 1
                completions.append(self._completion(r, keep_outputs))
                total_done += 1
                self.state[slot] = FREE
                self.req[slot] = None
                if not self.measuring and total_done >= warm_requests:
                    self.measuring = True
                    rec_measure_start = len(self.records)
                    t_meas = time.perf_counter()
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        res = self._result(completions, rec_measure_start, tok_measured, "n_requests")
        wall = time.perf_counter() - t_meas
        res["metrics"]["wall_throughput_tok_s"] = tok_measured / wall if wall > 0 else None
        res["metrics"]["poisson_rate_per_s"] = rate_per_s
        res["metrics"]["queue_left"] = len(queue)
        return res

    # ------------------------------------------------------------------ results
    def _throughput(self, start: int, tokens: int) -> float:
        ms = sum(r.ms for r in self.records[start:] if r.kind != "prefill")
        return tokens / (ms / 1000) if ms > 0 else 0.0

    def _completion(self, r: Request, keep: bool) -> dict:
        c = {"pid": r.pid, "prompt_len": len(r.prompt), "generated": len(r.out), "finish_reason": r.finish,
             "measured": r.measured, "rec_assign": r.rec_assign, "rec_first": r.rec_first, "rec_finish": r.rec_finish,
             "piggybacked": r.piggybacked, "wall_ttft_s": (r.wall_first - r.wall_arrival) if r.wall_arrival else None}
        if keep:
            c["tokens"] = list(r.out)
        return c

    def _result(self, completions, start, tokens, stop_reason) -> dict:
        recs = self.records
        ms = np.array([r.ms for r in recs])
        cum = np.concatenate([[0.0], np.cumsum(ms)])
        meas = np.array([r.measured for r in recs], dtype=bool)
        kinds = np.array([r.kind for r in recs])
        decode_ms = float(ms[meas & (kinds != "prefill")].sum())
        prefill_ms = float(ms[meas & (kinds == "prefill")].sum())
        ttft, tpot = [], []
        for c in completions:
            if not c["measured"] or c["rec_first"] < 0:
                continue
            ttft.append(cum[c["rec_first"] + 1] - cum[c["rec_assign"]])
            if c["generated"] > 1 and c["rec_finish"] > c["rec_first"]:
                seg = slice(c["rec_first"] + 1, c["rec_finish"] + 1)
                dm = ms[seg][kinds[seg] != "prefill"].sum()
                tpot.append(dm / (c["generated"] - 1))
        st = dict(self.stats)
        nsteps = max(st["steps"], 1)
        piggy_ms = float(sum(r.ms for r in recs if r.measured and r.n_chunk))   # steps that carried chunks
        metrics = {
            "config": self.cfg.label(), "B": self.B, "stop_reason": stop_reason,
            "measured_tokens": tokens, "decode_ms": decode_ms, "prefill_ms": prefill_ms,
            "throughput_tok_s": tokens / (decode_ms / 1000) if decode_ms else None,
            "e2e_throughput_tok_s": tokens / ((decode_ms + prefill_ms) / 1000) if decode_ms + prefill_ms else None,
            "prefill_time_share": prefill_ms / (decode_ms + prefill_ms) if decode_ms + prefill_ms else None,
            "prefill_tokens_separate": st["prefill_tokens"], "prefill_tokens_piggyback": st["piggyback_tokens"],
            "piggyback_share_of_prefill_tokens": st["piggyback_tokens"] / max(st["piggyback_tokens"] + st["prefill_tokens"], 1),
            "steps_with_piggyback_ms": piggy_ms,
            "deep_steps_per_token": st["deep_steps"] / max(st["committed"], 1),
            "deep_exec_fraction": st["deep_steps"] / nsteps,
            "tokens_per_deep_step": st["deep_tokens"] / max(st["deep_steps"], 1),
            "verifiers_per_deep_step": st["verifiers"] / max(st["deep_steps"], 1),
            "acceptance_rate": st["accepted"] / max(st["drafted"], 1),
            "plan_ms_per_step": st["plan_ms"] / max(st["plan_calls"], 1),
            "ttft_ms_mean": float(np.mean(ttft)) if ttft else None,
            "ttft_ms_p99": float(np.percentile(ttft, 99)) if ttft else None,
            "tpot_ms_mean": float(np.mean(tpot)) if tpot else None,
            "tpot_ms_p99": float(np.percentile(tpot, 99)) if tpot else None,
            "k_final": self.k, "k_timeline": self.exitc.timeline if self.exitc else [(0, self.k)],
            "k_alpha": self.exitc.alpha if self.exitc else None,
            "probe_steps": st["probe_steps"], "dropped_drafts_at_switch": st["dropped_drafts"],
            "m3_fallbacks": st["m3_fallbacks"], "m3_fallback_reasons": dict(self.fallback_reasons),
            "separate_prefills": st["separate_prefills"],
            "evaluated_completions": sum(1 for c in completions if c["measured"]),
            "steps": st["steps"], "deep_steps": st["deep_steps"],
        }
        return {"metrics": metrics, "engine_config": asdict(self.cfg), "completions": completions}
