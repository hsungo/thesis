"""Mechanism 2 (``--exit m2``): batch-shared, dynamically selected exit layer.

LayerSkip's early exits all share the final RMSNorm + LM head (`LayerSkipLlama.head`), so any
layer's hidden state can be turned into a draft with the same head; that is what makes a
runtime switch of k possible without extra weights.

Every ``probe_every``-th deep step is a *probe*: the verifiers' blocks run all layers on all
d+1 tokens (the shallow layers are recomputed for the probe instead of reusing the cached
exit states), the hidden state entering every candidate exit layer k' is captured, the shared
head is applied, and its argmax is compared with the full model's output Y on the positions
whose prefix is correct (j <= accepted).  That gives a per-token acceptance estimate alpha(k')
for every candidate (EMA over probes).  The extra work of the probe is part of the measured
step time.

After a deep step, with no half-done prefill, the controller computes for every candidate
    R*(k) = max_s  B E(alpha(k), s) / (s t_sh(B,k) + t_deep(B(s+1), k))        (cost model)
and switches when R*(k_best) > (1 + margin) R*(k_current).  Pending drafts (made with the old
exit) are dropped at the switch - drafts are only guesses, so this is lossless.
"""

from __future__ import annotations

import numpy as np


class ExitController:
    def __init__(self, cost_model, k0: int, cands, probe_every: int = 20, margin: float = 0.05,
                 omega: float = 0.8, d_max: int = 16, force: str | None = None):
        self.cm, self.k = cost_model, k0
        self.cands = sorted(set(int(c) for c in cands) | {k0})
        self.probe_every, self.margin, self.omega, self.d_max = probe_every, margin, omega, d_max
        self.alpha = {c: None for c in self.cands}
        self.deep_steps = 0
        self.probes = 0
        self.timeline = [(0, k0)]          # (deep step index, k)
        self.force = force                 # tests: "cycle" switches to the next candidate at every probe
        self.last_rates = {}

    def want_probe(self) -> bool:
        return (self.deep_steps + 1) % self.probe_every == 0

    def after_deep_step(self) -> None:
        self.deep_steps += 1

    def observe_probe(self, matches, total: int) -> None:
        if total <= 0:
            return
        self.probes += 1
        for c, m in zip(self.cands, matches):
            a = float(m) / total
            self.alpha[c] = a if self.alpha[c] is None else self.omega * self.alpha[c] + (1 - self.omega) * a

    def maybe_switch(self, B: int, C: float, can_switch: bool) -> int | None:
        """Called right after a deep step; returns the new k or None."""
        if not can_switch or self.probes == 0:
            return None
        if self.force == "cycle":
            new = self.cands[(self.cands.index(self.k) + 1) % len(self.cands)]
            if self.deep_steps % self.probe_every:
                return None
        else:
            known = {c: a for c, a in self.alpha.items() if a is not None}
            if self.k not in known or B <= 0:
                return None
            Cm = C / max(B, 1)
            self.last_rates = {c: self.cm.best_rate(a, B, c, B * Cm, self.d_max) for c, a in known.items()}
            new = max(self.last_rates, key=self.last_rates.get)
            if new == self.k or self.last_rates[new] <= (1 + self.margin) * self.last_rates[self.k]:
                return None
        self.k = new
        self.timeline.append((self.deep_steps, new))
        return new
