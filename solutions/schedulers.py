"""When to run the deep layers (``--sched``).  Pure CPU / NumPy, no GPU access.

Every scheduler sees the decoding slots' state before a step and returns the verifying set V
(deep step iff V is non-empty); the remaining decoding slots draft one token in the same step.

    fixed   synchronous baseline: a deep step every gamma shallow steps; it verifies every slot
            with drafts (a slot that joined mid-round verifies with fewer drafts).
    aspire  ASPIRE: per-request target draft length from an EMA acceptance estimate and the
            homogeneous-batch cost ratio t_sh / t_deep; a slot verifies on its own when it
            reaches its target (independent scheduling, observation O6).
    m1      mechanism 1 (batch-level trigger): after every step compare
                R(s)   = sum_i E_i(d_i)   / (s t_sh     + t_deep(sum_i (d_i+1)))
                R(s+1) = sum_i E_i(d_i+1) / ((s+1) t_sh + t_deep(sum_i (d_i+2)))
            E_i(d) = (1 - a_i^(d+1)) / (1 - a_i); draft once more if R(s+1) > R(s), else run the
            deep step.  Forced when any d_i >= d_max or after --sla-steps steps without a deep step.
            Every slot with d_i >= 1 verifies; slots with d_i = 0 draft in the same step.

The acceptance estimate is ASPIRE's: a_i <- omega a_i + (1 - omega) a_hat, a_hat = acc / min(acc+1, d).
"""

from __future__ import annotations

import numpy as np

from common.policies import expected_token_targets


def expected_tokens(alpha, d) -> np.ndarray:
    a = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1 - 1e-6)
    d = np.asarray(d, dtype=np.float64)
    return (1 - a ** (d + 1)) / (1 - a)


class _EMA:
    def __init__(self, B: int, omega: float, alpha0: float):
        self.omega, self.alpha0 = omega, alpha0
        self.alpha = np.full(B, alpha0)
        self.fresh = np.ones(B, dtype=bool)

    def observe(self, verify, accepted, drafts) -> None:
        valid = verify & (drafts > 0)
        if valid.any():
            a = accepted[valid].astype(np.float64)
            d = drafts[valid].astype(np.float64)
            self.alpha[valid] = self.omega * self.alpha[valid] + (1 - self.omega) * (a / np.minimum(a + 1.0, d))
            self.fresh[valid] = False

    def reset(self, slot: int, alpha: float | None = None) -> None:
        self.alpha[slot] = self.alpha0 if alpha is None else alpha
        self.fresh[slot] = True


class Scheduler:
    name = "base"

    def __init__(self, B: int, cost_model, k: int, d_max: int = 16, omega: float = 0.8, alpha0: float = 0.9):
        self.B, self.cm, self.k, self.d_max = B, cost_model, k, d_max
        self.ema = _EMA(B, omega, alpha0)

    def decide(self, d: np.ndarray, ctx: np.ndarray, dec: np.ndarray, steps_since_deep: int) -> np.ndarray:
        raise NotImplementedError

    def observe(self, verify, accepted, drafts) -> None:
        self.ema.observe(verify, accepted, drafts)

    def reset(self, slot: int) -> None:
        self.ema.reset(slot)

    def set_exit(self, k: int, alpha: float | None = None) -> None:
        """Exit layer changed (M2): costs change; the per-request acceptance belongs to the old k."""
        self.k = k
        if alpha is not None:
            self.ema.alpha[:] = alpha


class FixedScheduler(Scheduler):
    name = "fixed"

    def __init__(self, B, cost_model, k, gamma: int, **kw):
        super().__init__(B, cost_model, k, **kw)
        if not 1 <= gamma <= self.d_max:
            raise ValueError("gamma must be in [1, d_max]")
        self.gamma = gamma

    def decide(self, d, ctx, dec, steps_since_deep):
        if steps_since_deep >= self.gamma:
            return dec & (d >= 1)
        return dec & (d >= self.d_max)


class AspireScheduler(Scheduler):
    name = "aspire"

    def __init__(self, B, cost_model, k, initial_draft: int = 3, min_draft: int = 1, **kw):
        super().__init__(B, cost_model, k, **kw)
        self.initial_draft, self.min_draft = initial_draft, min_draft

    def decide(self, d, ctx, dec, steps_since_deep):
        n = int(dec.sum())
        if n == 0:
            return np.zeros_like(dec)
        C = float(ctx[dec].sum())
        t_sh = float(self.cm.t_sh(n, self.k, C))
        # homogeneous-batch ratio c_i = t_draft / t_verify(d_i) (ASPIRE LatencyModel.homogeneous_ratio)
        t_v = self.cm.t_deep(n * (d + 1), self.k, C, n)
        ratio = np.clip(t_sh / np.maximum(t_v, 1e-9), 1e-4, 1.0)
        tgt = expected_token_targets(self.ema.alpha, ratio, self.min_draft, self.d_max)
        minimum = np.where(self.ema.fresh, max(self.min_draft, self.initial_draft), self.min_draft)
        tgt = np.minimum(np.maximum(tgt, minimum), self.d_max)
        return dec & (d >= 1) & ((d >= tgt) | (d >= self.d_max))


class M1Scheduler(Scheduler):
    name = "m1"

    def __init__(self, B, cost_model, k, sla_steps: int = 32, **kw):
        super().__init__(B, cost_model, k, **kw)
        self.sla_steps = sla_steps

    def rates(self, d, ctx, dec, s):
        """(R(s), R(s+1)) for the current state (exposed for tests / logging)."""
        dd = d[dec]
        al = self.ema.alpha[dec]
        n = len(dd)
        C = float(ctx[dec].sum())
        t_sh = float(self.cm.t_sh(n, self.k, C))
        ver = dd >= 1
        e_now = float(expected_tokens(al[ver], dd[ver]).sum())
        t_now = s * t_sh + float(self.cm.t_deep(int((dd[ver] + 1).sum()), self.k, C, int(ver.sum())))
        e_next = float(expected_tokens(al, dd + 1).sum())
        t_next = (s + 1) * t_sh + float(self.cm.t_deep(int((dd + 2).sum()), self.k, C, n))
        return e_now / max(t_now, 1e-9), e_next / max(t_next, 1e-9)

    def decide(self, d, ctx, dec, steps_since_deep):
        ver = dec & (d >= 1)
        if not ver.any():
            return ver
        if (d[dec] >= self.d_max).any() or steps_since_deep >= self.sla_steps:
            return ver
        r_now, r_next = self.rates(d, ctx, dec, steps_since_deep)
        return ver if r_next <= r_now else np.zeros_like(ver)


def make_scheduler(name: str, B: int, cost_model, k: int, *, gamma: int = 4, d_max: int = 16, sla_steps: int = 32,
                   initial_draft: int = 3, omega: float = 0.8, alpha0: float = 0.9) -> Scheduler:
    kw = dict(d_max=d_max, omega=omega, alpha0=alpha0)
    if name == "fixed":
        return FixedScheduler(B, cost_model, k, gamma, **kw)
    if name == "aspire":
        return AspireScheduler(B, cost_model, k, initial_draft=initial_draft, **kw)
    if name == "m1":
        return M1Scheduler(B, cost_model, k, sla_steps=sla_steps, **kw)
    raise ValueError(f"unknown scheduler {name!r}")
