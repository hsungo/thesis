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
           piggyback only if  dt(c) < t_prefill(c)   (marginal cost in both segments vs a separate pass)
           fall back to `separate` when c <= 0 (no room below the knee), when the prompt would need more
           than ``--max-wait`` deep steps, or when dt(c) >= t_prefill(c).  The decision is taken when the
           request is assigned to a slot (`admit`); afterwards its chunks ride on every step that has room.

`M3Planner` is pure CPU (cost model queries only).
"""

from __future__ import annotations

import math


class M3Planner:
    def __init__(self, cost_model, max_wait: int = 3, chunk_cap: int | None = None, no_knee_chunk: int = 512,
                 knee_safety: float = 0.9):
        self.cm, self.max_wait, self.chunk_cap = cost_model, max_wait, chunk_cap
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
        if math.ceil(prompt_len / c) > self.max_wait:
            return False, "exceeds_max_wait"
        if not self.worth_it(c, k, n_sh, C, n_deep_est, C, float(c)):
            return False, "not_cheaper_than_separate"
        return True, "piggyback"

    def worth_it(self, c: int, k: int, n_sh: int, C_sh: float, n_dp: int, C_dp: float, C_chunk: float) -> bool:
        """dt(c) < t_prefill(c): marginal time of carrying c tokens in both segments vs a separate pass."""
        L = self.cm.dims.n_layers
        dt = (float(self.cm.t_segment(k, n_sh + c, C_sh + C_chunk)) - float(self.cm.t_segment(k, n_sh, C_sh))
              + float(self.cm.t_segment(L - k, n_dp + c, C_dp + C_chunk)) - float(self.cm.t_segment(L - k, n_dp, C_dp)))
        return dt < float(self.cm.t_prefill(c, C_chunk))


def chunk_kv_positions(pos0: int, c: int) -> float:
    """Cached positions read by a causal chunk [pos0, pos0+c) in one layer (sum over its queries / c * c):
    the attention reads the prefix once per query block - pos0 + c positions."""
    return float(pos0 + c)


def split_separate(prompt_len: int, chunk: int) -> list[tuple[int, int]]:
    """(start, length) chunks for a separate chunked prefill."""
    return [(s, min(chunk, prompt_len - s)) for s in range(0, prompt_len, chunk)] or [(0, 0)]


