"""Trace-driven simulator of *asynchronous* batched early-exit speculation.

Each step, every slot either drafts one token (shallow layers only) or verifies
(shallow for its last draft + deep for d+1 tokens) — S1's depth-segmented mixed
forward.  Acceptance is exact (greedy agreement traces, plan sec.3.1); step time
comes from the O5 layer cost model.  The batch is refilled cyclically.

Used by O6 today; use it to prototype S2/S3 schedulers before touching the real
system (a new policy only needs the `common.policies.Policy` interface):

    sim = AsyncTraceSimulator(traces, exit_layer, B, cost_model, policy)
    summary = sim.run(steps=4000, warmup=500)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.cost_model import LayerCostModel
from common.policies import BatchView, Policy
from common.traces import TraceSet, run_lengths


@dataclass
class SimStep:
    time_ms: float
    deep: bool
    n_verify: int
    n_deep_tokens: int
    committed: int
    round_lengths: list   # d+1 of every request that verified this step


class AsyncTraceSimulator:
    def __init__(self, traceset: TraceSet, exit_layer: int, batch_size: int, cost_model: LayerCostModel,
                 policy: Policy, seed: int = 0):
        col = traceset.col(exit_layer)
        self.traces = traceset.traces
        self.runs = [run_lengths(t.agree[:, col]) for t in self.traces]
        self.prompt = np.array([t.prompt_len for t in self.traces])
        self.length = np.array([t.length for t in self.traces])
        self.k, self.B, self.cm, self.policy = exit_layer, batch_size, cost_model, policy
        self.order = np.random.default_rng(seed).permutation(len(self.traces))
        self.cursor = 0
        self.tr = np.zeros(batch_size, dtype=np.int64)
        self.c = np.ones(batch_size, dtype=np.int64)   # committed tokens (y_0 from prefill)
        self.d = np.zeros(batch_size, dtype=np.int64)  # drafts in the current round
        self.active = np.ones(batch_size, dtype=bool)
        for i in range(batch_size):
            self._refill(i)

    def _refill(self, i: int) -> None:
        self.tr[i] = self.order[self.cursor % len(self.order)]
        self.cursor += 1
        self.c[i], self.d[i] = 1, 0
        self.policy.reset_slots(np.array([i]))

    def seq_lens(self) -> np.ndarray:
        return self.prompt[self.tr] + self.c + self.d

    def step(self) -> SimStep:
        ctx = self.seq_lens()
        V = self.policy.decide(BatchView(self.d.copy(), ctx, self.active)).verify
        n_deep = int((self.d[V] + 1).sum())
        n_head = int((~V).sum()) + n_deep
        t = float(self.cm.two_segment(self.k, self.B, float(ctx.sum()), n_deep, float(ctx[V].sum()), n_head,
                                      float(ctx.max()), float(ctx[V].max()) if V.any() else 0.0))
        accepted = np.zeros(self.B, dtype=np.int64)
        drafts = self.d.copy()
        committed, rounds, refill = 0, [], []
        for i in np.flatnonzero(V):
            tid, c, d = self.tr[i], int(self.c[i]), int(self.d[i])
            a = min(int(self.runs[tid][c]), d)
            accepted[i] = a
            committed += min(a + 1, int(self.length[tid]) - c)
            rounds.append(d + 1)
            self.c[i] = c + a + 1
            self.d[i] = 0
            if self.c[i] >= self.length[tid]:
                refill.append(i)
        self.d[~V] += 1
        self.policy.observe_verify(V, accepted, drafts)
        for i in refill:
            self._refill(i)
        return SimStep(t, n_deep > 0, int(V.sum()), n_deep, committed, rounds)

    def run(self, steps: int, warmup: int = 0) -> dict:
        recs = [self.step() for _ in range(warmup + steps)][warmup:]
        deep = np.array([r.deep for r in recs])
        nv = np.array([r.n_verify for r in recs])
        rl = np.concatenate([np.asarray(r.round_lengths) for r in recs if r.round_lengths] or [np.ones(1)])
        L = float(rl.mean())
        total_ms = sum(r.time_ms for r in recs)
        deep_fixed_ms = float(deep.sum()) * (self.cm.n_layers - self.k) * self.cm.bw
        return {
            "steps": len(recs),
            "deep_exec_fraction": float(deep.mean()),
            "theory_independent": 1 - (1 - 1 / L) ** self.B,
            "mean_steps_per_round_L": L,
            "mean_deep_batch_when_run": float(nv[deep].mean()) if deep.any() else 0.0,
            "mean_deep_batch_fraction_of_B": float(nv[deep].mean() / self.B) if deep.any() else 0.0,
            "tokens_per_ms": sum(r.committed for r in recs) / total_ms,
            "mean_step_ms": total_ms / len(recs),
            "deep_fixed_time_share": deep_fixed_ms / total_ms,
            "mean_verify_per_step": float(nv.mean()),
            "deep_batch_hist": np.bincount(nv[deep], minlength=self.B + 1).tolist() if deep.any() else [],
        }
