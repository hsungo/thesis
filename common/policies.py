"""Speculation scheduling policies: decide, every step, which requests verify.

The interface mirrors ASPIRE (`aspire/policies/base.py`, `core.py`) so the same
policy object can drive both the trace simulator (`common/simulator.py`, used by O6)
and, later, the real asynchronous engine of S1/S2/S3:

    decision = policy.decide(BatchView(...))          # once per step, sees the whole batch
    policy.observe_verify(verify_mask, accepted, drafts)  # after verification
    policy.reset_slots(slot_ids)                       # slot refilled with a new request

A batch-wide policy (e.g. S3 verification coalescing, which needs to know how many
requests would join a deep pass) fits the same interface because `decide` sees every
slot at once.  Write new policies here and register them in `POLICIES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass
class BatchView:
    draft_counts: IntArray   # drafts already produced in the current round, per slot
    seq_lens: IntArray       # context length per slot (prompt + committed + drafts)
    active: BoolArray        # slot holds a live request


@dataclass
class PolicyDecision:
    verify: BoolArray        # slots that verify this step (the rest draft one token)
    targets: IntArray        # current target draft length per slot (for logging)
    metrics: dict = field(default_factory=dict)


class Policy(Protocol):
    max_draft: int

    def decide(self, batch: BatchView) -> PolicyDecision: ...

    def observe_verify(self, verify_mask: BoolArray, accepted: IntArray, draft_counts: IntArray) -> None: ...

    def reset_slots(self, slot_ids: NDArray[np.integer]) -> None: ...


# ----------------------------------------------------------------------------- helpers

def expected_token_targets(alphas, cost_ratios, min_draft: int, max_draft: int) -> IntArray:
    """ASPIRE's per-request target: the largest gamma whose marginal gain is positive in
    E[tokens]/E[time] = ((1 - a^(g+1)) / (1 - a)) / (1 + c g)   (Leviathan et al.)."""
    alpha = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 1.0)
    cost = np.broadcast_to(np.asarray(cost_ratios, dtype=np.float64), alpha.shape)
    if max_draft < 1:
        return np.full(alpha.shape, min_draft, dtype=np.int64)
    gam = np.arange(1, max_draft + 1, dtype=np.int64)[:, None]
    lhs = alpha[None, :] ** gam * ((1 - alpha)[None, :] * cost[None, :] * (gam - 1) + cost[None, :] + (1 - alpha)[None, :])
    hold = lhs > cost[None, :]
    any_hold = hold.any(axis=0)
    last = (max_draft - 1) - np.argmax(hold[::-1], axis=0)
    t = np.where(any_hold, last + 1, min_draft)
    t = np.where(alpha >= 1 - 1e-7, max_draft, t)
    return np.clip(t, min_draft, max_draft).astype(np.int64)


def best_gamma_expected(alpha: float, c: float, gmax: int) -> int:
    """argmax_g ((1-a^(g+1))/(1-a)) / (1 + c g), brute force (reference for tests)."""
    g = np.arange(0, gmax + 1)
    a = min(max(alpha, 1e-6), 1 - 1e-6)
    val = (1 - a ** (g + 1)) / (1 - a) / (1 + c * g)
    return int(g[np.argmax(val)])


class LayerCostRatio:
    """c_i = T_draft / T_verify(d_i) under the homogeneous-batch approximation
    (ASPIRE `LatencyModel.homogeneous_ratio`), with the per-layer cost model."""

    def __init__(self, cost_model, exit_layer: int, batch_size: int):
        self.cm, self.k, self.B = cost_model, exit_layer, batch_size

    def __call__(self, draft_counts: IntArray, seq_lens: IntArray) -> NDArray[np.float64]:
        d = np.asarray(draft_counts, dtype=np.float64)
        ctx = np.asarray(seq_lens, dtype=np.float64)
        draft = self.cm.sync_draft(self.k, self.B, ctx)
        verify = self.cm.sync_verify(self.k, d, self.B, ctx)
        return np.clip(draft / np.maximum(verify, 1e-9), 1e-4, 1.0)


# ----------------------------------------------------------------------------- policies

class FixedDraftPolicy:
    """Every request verifies after `gamma` drafts.  With all slots starting together
    (and refilled at verify boundaries) this is the synchronous batch."""

    def __init__(self, batch_size: int, gamma: int):
        self.max_draft = gamma
        self.targets = np.full(batch_size, gamma, dtype=np.int64)

    def decide(self, batch: BatchView) -> PolicyDecision:
        return PolicyDecision((batch.draft_counts >= self.targets) & batch.active, self.targets.copy())

    def observe_verify(self, verify_mask, accepted, draft_counts) -> None:
        return None

    def reset_slots(self, slot_ids) -> None:
        return None


class FeedbackPolicy:
    """ASPIRE-FSM: full acceptance -> target+1, any rejection -> target-1."""

    def __init__(self, batch_size: int, min_draft: int, max_draft: int, initial_draft: int):
        self.min_draft, self.max_draft, self.initial_draft = min_draft, max_draft, initial_draft
        self.targets = np.full(batch_size, initial_draft, dtype=np.int64)

    def decide(self, batch: BatchView) -> PolicyDecision:
        verify = (batch.draft_counts >= self.targets) & batch.active
        return PolicyDecision(verify, self.targets.copy(), {"target_mean": float(self.targets.mean())})

    def observe_verify(self, verify_mask, accepted, draft_counts) -> None:
        ids = np.flatnonzero(verify_mask)
        if ids.size == 0:
            return
        cur = self.targets[ids]
        full = accepted[ids] >= cur
        self.targets[ids] = np.where(full, np.minimum(cur + 1, self.max_draft), np.maximum(cur - 1, self.min_draft))

    def reset_slots(self, slot_ids) -> None:
        self.targets[slot_ids] = self.initial_draft


class AspirePolicy:
    """ASPIRE latency-aware scheduler: EMA acceptance estimate + cost ratio -> target;
    a request verifies when its draft count reaches the target (or max_draft); a new
    request drafts at least `initial_draft` tokens before its first verification."""

    def __init__(self, batch_size: int, min_draft: int, max_draft: int, initial_draft: int,
                 cost_ratio: Callable[[IntArray, IntArray], NDArray[np.float64]],
                 omega: float = 0.8, alpha_init: float = 0.9):
        self.min_draft, self.max_draft, self.initial_draft = min_draft, max_draft, initial_draft
        self.cost_ratio = cost_ratio
        self.omega, self.alpha_init = omega, alpha_init
        self.alpha = np.full(batch_size, alpha_init)
        self.fresh = np.ones(batch_size, dtype=bool)
        self.targets = np.full(batch_size, initial_draft, dtype=np.int64)

    def decide(self, batch: BatchView) -> PolicyDecision:
        d = batch.draft_counts
        minimum = np.where(self.fresh, max(self.min_draft, self.initial_draft), self.min_draft)
        ratio = self.cost_ratio(d, batch.seq_lens)
        tgt = expected_token_targets(self.alpha, ratio, self.min_draft, self.max_draft)
        self.targets = np.minimum(np.maximum(tgt, minimum), self.max_draft)
        verify = (((d >= self.targets) & (d >= minimum)) | (d >= self.max_draft)) & batch.active
        return PolicyDecision(verify, self.targets.copy(), {"alpha_mean": float(self.alpha.mean())})

    def observe_verify(self, verify_mask, accepted, draft_counts) -> None:
        valid = verify_mask & (draft_counts > 0)
        if not valid.any():
            return
        a = accepted[valid].astype(np.float64)
        d = draft_counts[valid].astype(np.float64)
        inst = a / np.minimum(a + 1.0, d)
        self.alpha[valid] = self.omega * self.alpha[valid] + (1 - self.omega) * inst
        self.fresh[valid] = False

    def reset_slots(self, slot_ids) -> None:
        self.alpha[slot_ids] = self.alpha_init
        self.fresh[slot_ids] = True
        self.targets[slot_ids] = self.initial_draft


def make_policy(name: str, batch_size: int, *, cost_model=None, exit_layer: int | None = None,
                min_draft: int = 1, max_draft: int = 16, initial_draft: int = 3, fixed_gamma: int = 4,
                omega: float = 0.8, alpha_init: float = 0.9) -> Policy:
    if name == "fixed":
        return FixedDraftPolicy(batch_size, fixed_gamma)
    if name == "ar":   # gamma = 0: every step verifies one token through all layers = autoregressive
        return FixedDraftPolicy(batch_size, 0)
    if name == "fsm":
        return FeedbackPolicy(batch_size, min_draft, max_draft, initial_draft)
    if name == "aspire":
        if cost_model is None or exit_layer is None:
            raise ValueError("the aspire policy needs a cost model and an exit layer")
        return AspirePolicy(batch_size, min_draft, max_draft, initial_draft,
                            LayerCostRatio(cost_model, exit_layer, batch_size), omega, alpha_init)
    raise ValueError(f"unknown policy {name!r}; known: {POLICIES}")


POLICIES = ("aspire", "fsm", "fixed", "ar")
