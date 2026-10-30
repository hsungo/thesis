"""Prefill policies (``--prefill``).

separate   the whole batch pauses while the new prompt is prefilled in chunks of
           ``--prefill-chunk`` tokens through all layers (as ASPIRE / vLLM chunked prefill).
m3         mechanism 3, depth-split piggybacking: a chunk of the pending prompt rides on the
           shallow segment of a step (layers [0,k), its layer-k hidden state is kept) and on the
           deep segment of the next deep step (layers [k,L)).  KV is written to every layer as the
           chunk passes through it, and later chunks attend to earlier ones (causal over positions).
           The final chunk's deep pass yields the first generated token; the request then starts
           drafting with d = 0.

           chunk size  c = min(remaining, knee*s - tokens already in the shallow segment,
                               knee*s - tokens expected in the next deep segment)
           knee = measured knee of the cost model (calibrate), s = --knee-safety (default 0.9)
           piggyback only if the whole prompt is cheaper that way (n = ceil(P / c) chunks, one per round):
               n * dt(c) + n * t_round / (n_dec + 1)  <  t_prefill(P)
           dt(c) = marginal cost of the chunk in both segments, including the extra attention bucket;
           n * t_round / (n_dec + 1) = the slot produces nothing for n rounds, as batch time;
           t_prefill(P) = one separate pass over the whole prompt, which stalls the whole batch.
           fall back to `separate` when c <= 0 (no room below the knee), when the prompt would need more
           than ``--max-wait`` deep steps (a TTFT cap), or when piggybacking is not cheaper.  The decision is
           taken when the request is assigned to a slot (`admit`); afterwards its chunks ride on every step
           that has room.

`M3Planner` is pure CPU (cost model queries only).
"""

from __future__ import annotations

import math


class M3Planner:
    def __init__(self, cost_model, max_wait: int = 8, chunk_cap: int | None = None, no_knee_chunk: int = 512,
                 knee_safety: float = 0.9, force: bool = False):
        self.cm, self.max_wait, self.chunk_cap = cost_model, max_wait, chunk_cap
        self.force = force                      # tests: skip the cost check so the piggyback path is exercised
        self.no_knee_chunk = no_knee_chunk      # a model without a knee (linear) never limits the chunk: cap it
        self.knee_safety = knee_safety          # fill the step only up to knee * safety (--knee-safety)

    def knee(self) -> float:
        return self.cm.knee() * self.knee_safety

    def budget(self, n_sh: int, n_deep_est: int, pending_deep: int) -> int:
        knee = self.knee()
        if not math.isfinite(knee):
            knee = self.no_knee_chunk + max(n_sh, n_deep_est + pending_deep)
        c = int(min(knee - n_sh, knee - n_deep_est - pending_deep))
        if self.chunk_cap is not None:
            c = min(c, self.chunk_cap)
        return max(c, 0)

    def admit(self, prompt_len: int, n_sh: int, n_deep_est: int, k: int, C: float) -> tuple[bool, str]:
        """At assignment: piggyback the prompt (True) or prefill it separately now (False), and why.
        Uses the chunk size the current step mix allows; C = cached positions read by the decode tokens."""
        c = min(self.budget(n_sh, n_deep_est, 0), prompt_len)
        if c <= 0:
            return False, "no_room_below_knee"
        n = math.ceil(prompt_len / c)
        if n > self.max_wait:
            return False, "exceeds_max_wait"
        if not self.force and not self.worth_it(prompt_len, c, k, n_sh, C, n_deep_est, C):
            return False, "not_cheaper_than_separate"
        return True, "piggyback"

    def worth_it(self, prompt_len: int, c: int, k: int, n_sh: int, C_sh: float, n_dp: int, C_dp: float) -> bool:
        """Whole-prompt comparison: n chunks of marginal cost dt(c) (both segments, one more attention bucket)
        plus the n rounds the slot sits idle, as batch time, vs one separate pass over the prompt."""
        cm, L = self.cm, self.cm.dims.n_layers
        n = math.ceil(prompt_len / c)
        C_chunk = float(prompt_len) / 2 + c / 2          # mean cached positions a chunk reads (pos0 + c)
        dt = (float(cm.t_segment(k, n_sh + c, C_sh + C_chunk, 2)) - float(cm.t_segment(k, n_sh, C_sh))
              + float(cm.t_segment(L - k, n_dp + c, C_dp + C_chunk, 2)) - float(cm.t_segment(L - k, n_dp, C_dp)))
        n_dec = max(n_sh, 1)
        round_len = max(n_dp / n_dec - 1, 1.0)            # n_dp = n_dec * (round_len + 1)
        t_round = round_len * float(cm.t_sh(n_dec, k, C_sh)) + float(cm.t_deep(n_dp, k, C_sh, n_dec, 1, C_dp))
        return n * dt + n * t_round / (n_dec + 1) < float(cm.t_prefill(prompt_len, prompt_len / 2))


def chunk_kv_positions(pos0: int, c: int) -> float:
    """Cached positions read by a causal chunk [pos0, pos0+c) in one layer (sum over its queries / c * c):
    the attention reads the prefix once per query block - pos0 + c positions."""
    return float(pos0 + c)


def split_separate(prompt_len: int, chunk: int) -> list[tuple[int, int]]:
    """(start, length) chunks for a separate chunked prefill."""
    return [(s, min(chunk, prompt_len - s)) for s in range(0, prompt_len, chunk)] or [(0, 0)]


