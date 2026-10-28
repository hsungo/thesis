"""Batched decoding engines on top of `LayerSkipLlama`.

* `ARBatchEngine`       – plain autoregressive decoding, one token per request per step
                          (evaluation baseline; also used by O1-O3 trace collection).
* `SyncSpecEngine`      – synchronous batched LayerSkip self-speculation: every request
                          drafts `gamma` tokens with the first `exit_layer` layers, then the
                          whole batch verifies together (plan P1, observation O4).
                          Verification reuses the shallow KV and the cached exit-layer hidden
                          states (LayerSkip "cache reuse"), so shallow layers only process the
                          last draft token.

Both share `DecodeEngine.run`, a cyclic-replay driver modelled on ASPIRE
(`aspire/engine/cyclic.py`): the batch is kept full by refilling finished slots
with the next prompt of the pool, the first `warmup_samples` completions are
ignored, and the run stops when cumulative decode throughput has converged or
`max_eval_samples` completions were measured.  Throughput only counts decode
forward time (prefill is excluded), as in ASPIRE.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from common.model import KVCache, LayerSkipLlama
from common.timing import Interval


# ----------------------------------------------------------------------------- records

@dataclass
class StepRecord:
    kind: str               # "ar" | "draft" | "verify"
    forward_ms: float
    batch: int              # requests in the call
    shallow_tokens: int     # tokens through layers [0, k)
    deep_tokens: int        # tokens through layers [k, N)   (== shallow for AR)
    kv_read: int            # KV positions read per layer (batch * padded length)
    measured: bool          # counted in throughput (after warm-up)


@dataclass
class SlotState:
    prompt_idx: int = -1
    pid: str = ""
    prompt_len: int = 0
    pos: int = 0            # position of `last_token` (its KV is not written yet)
    last_token: int = 0
    output: list = field(default_factory=list)
    finish_reason: Optional[str] = None
    idle: bool = True
    t_assigned: float = 0.0
    decode_ms: float = 0.0  # forward time spent while this request was in the batch


@dataclass
class RunConfig:
    mode: str = "cyclic"             # cyclic | static
    warmup_samples: int = 8
    max_eval_samples: int = 256
    converge: bool = True
    converge_threshold: float = 0.01
    converge_window: int = 4
    converge_min_samples: int = 16
    max_steps: Optional[int] = None  # hard cap on model steps (draft+verify each count 1)
    keep_outputs: bool = False
    # Steady state needs every slot to turn over: with a fixed small warm-up / minimum the
    # measured window at large B is the start-up transient, dominated by requests that hit
    # EOS early (observed: B=92 "converged" after 20 completions of ~30 tokens).  The
    # effective thresholds are therefore max(fixed value, factor * batch size).
    warmup_per_slot: float = 1.0
    min_eval_per_slot: float = 2.0


def sample_tokens(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits [..., V] -> token ids [...].  temperature == 0 -> greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    shape = logits.shape[:-1]
    x = logits.reshape(-1, logits.shape[-1]).float() / temperature
    if top_k and top_k > 0:
        kth = torch.topk(x, min(top_k, x.shape[-1]), dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p < 1.0:
        sx, si = torch.sort(x, descending=True, dim=-1)
        cum = sx.softmax(-1).cumsum(-1)
        remove = cum - sx.softmax(-1) > top_p
        sx = sx.masked_fill(remove, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(-1, si, sx)
    probs = x.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).view(shape)


# ----------------------------------------------------------------------------- base

class DecodeEngine:
    name = "base"

    def __init__(self, model: LayerSkipLlama, batch_size: int, max_prompt_len: int, max_new_tokens: int,
                 eos_ids: tuple[int, ...], *, ignore_eos: bool = False, spec_slack: int = 32,
                 temperature: float = 0.0, top_k: int = 50, top_p: float = 0.9, seed: int = 0):
        self.model = model
        self.device = model.device
        self.B = batch_size
        self.max_new = max_new_tokens
        self.eos = set(eos_ids)
        self.ignore_eos = ignore_eos
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.gen = torch.Generator(device=self.device).manual_seed(seed) if temperature > 0 else None
        self.max_seq = max_prompt_len + max_new_tokens + spec_slack + 2
        model.allocate_cache(batch_size, self.max_seq)
        self.slots = [SlotState() for _ in range(batch_size)]
        self.records: list[StepRecord] = []
        self.probe = None  # optional observer (see common/traces.py: ExitProbe)
        self.stop_checker = None  # optional common.data.StopChecker (task stop strings)

    # ---------------------------------------------------------------- slot helpers
    def _choose(self, logits: torch.Tensor) -> torch.Tensor:
        return sample_tokens(logits, self.temperature, self.top_k, self.top_p, self.gen)

    def _assign(self, slot: int, prompt_idx: int, prompt) -> None:
        s = self.slots[slot]
        tokens = prompt.tokens
        if len(tokens) + self.max_new + 2 > self.model.cache.max_seq:
            raise ValueError(f"prompt of {len(tokens)} tokens does not fit max_seq={self.model.cache.max_seq}")
        logits, hidden = self.model.prefill(tokens, slot, capture_exits=self.probe is not None)
        first = int(self._choose(logits.unsqueeze(0))[0])
        s.prompt_idx, s.pid, s.prompt_len = prompt_idx, prompt.pid, len(tokens)
        s.pos, s.last_token = len(tokens), first
        s.output, s.finish_reason, s.idle = [], None, False
        s.t_assigned, s.decode_ms = time.perf_counter(), 0.0
        if self.probe is not None:
            self.probe.on_prefill(slot, s, hidden, logits, first)
        self._commit(slot, [first])

    def _commit(self, slot: int, tokens: list[int]) -> int:
        """Append tokens to the slot output, stop at EOS / max_new.  Returns the number
        of useful tokens appended (what throughput counts)."""
        s = self.slots[slot]
        n = 0
        for t in tokens:
            if s.finish_reason is not None:
                break
            s.output.append(int(t))
            n += 1
            if not self.ignore_eos and int(t) in self.eos:
                s.finish_reason = "eos"
            elif len(s.output) >= self.max_new:
                s.finish_reason = "length"
        if n and self.stop_checker and not self.ignore_eos:
            cut = self.stop_checker(s.output, n)
            if cut is not None:
                n -= len(s.output) - cut
                del s.output[cut:]
                s.finish_reason = "stop"
        return n

    def _active(self) -> list[int]:
        return [i for i, s in enumerate(self.slots) if not s.idle]

    # ---------------------------------------------------------------- one iteration
    def _iteration(self, measured: bool) -> int:
        """Run one scheduling iteration; return number of useful tokens committed."""
        raise NotImplementedError

    def extra_metrics(self) -> dict:
        return {}

    # ---------------------------------------------------------------- driver
    @torch.inference_mode()
    def run(self, prompts: list, cfg: RunConfig) -> dict:
        self.records = []
        n_pool = len(prompts)
        cursor = 0
        completions: list[dict] = []
        total_completed = 0
        warmup_n = max(cfg.warmup_samples, math.ceil(cfg.warmup_per_slot * self.B)) if cfg.warmup_samples else 0
        min_eval_n = max(cfg.converge_min_samples, math.ceil(cfg.min_eval_per_slot * self.B))
        max_eval_n = max(cfg.max_eval_samples, min_eval_n) if cfg.converge else cfg.max_eval_samples
        warm = cfg.mode == "static" or warmup_n == 0
        evaluated = 0
        anchor = None
        stable = 0
        stop_reason = None
        measured_tokens = 0
        measured_ms = 0.0
        wall_t0 = time.perf_counter() if warm else None
        wall_measured = 0.0

        # initial fill
        for slot in range(self.B):
            if cfg.mode == "static" and cursor >= n_pool:
                break
            self._assign(slot, cursor % n_pool, prompts[cursor % n_pool])
            cursor += 1

        steps = 0
        while True:
            if not self._active():
                stop_reason = "pool_exhausted"
                break
            n_rec_before = len(self.records)
            committed = self._iteration(measured=warm)
            new_recs = self.records[n_rec_before:]
            step_ms = sum(r.forward_ms for r in new_recs)
            steps += len(new_recs)
            for i in self._active():
                self.slots[i].decode_ms += step_ms
            if warm:
                measured_tokens += committed
                measured_ms += step_ms

            # completions (requests finishing in the step where warm-up ends were not
            # measured, so they do not count as evaluated - same as ASPIRE's was_warmup)
            was_warm = warm
            for slot, s in enumerate(self.slots):
                if s.idle or s.finish_reason is None:
                    continue
                if stop_reason is not None:
                    break
                comp = {
                    "pid": s.pid, "prompt_idx": s.prompt_idx, "prompt_len": s.prompt_len,
                    "generated": len(s.output), "finish_reason": s.finish_reason,
                    "decode_ms": s.decode_ms, "measured": was_warm,
                }
                if cfg.keep_outputs:
                    comp["tokens"] = list(s.output)
                if self.probe is not None:
                    self.probe.on_finish(slot, s)
                completions.append(comp)
                total_completed += 1
                s.idle = True
                if not was_warm:
                    if not warm and total_completed >= warmup_n:
                        warm = True
                        wall_t0 = time.perf_counter()
                elif cfg.mode == "cyclic":
                    evaluated += 1
                    if cfg.converge and evaluated >= min_eval_n and measured_ms > 0:
                        thr = measured_tokens / (measured_ms / 1000)
                        if anchor is None:
                            anchor = thr
                        elif abs(thr - anchor) / anchor < cfg.converge_threshold:
                            stable += 1
                        else:
                            stable, anchor = 0, thr
                        if stable >= cfg.converge_window:
                            stop_reason = "converged"
                    if stop_reason is None and evaluated >= max_eval_n:
                        stop_reason = "max_eval_samples"
                # refill
                if cfg.mode == "cyclic" or cursor < n_pool:
                    self._assign(slot, cursor % n_pool, prompts[cursor % n_pool])
                    cursor += 1
            if stop_reason is not None:
                break
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                stop_reason = "max_steps"
                break
        if wall_t0 is not None:
            wall_measured = time.perf_counter() - wall_t0

        rec = [r for r in self.records if r.measured]
        per_kind = {}
        for kind in sorted({r.kind for r in rec}):
            ms = [r.forward_ms for r in rec if r.kind == kind]
            per_kind[kind] = {"steps": len(ms), "mean_ms": float(np.mean(ms)), "p50_ms": float(np.median(ms))}
        measured_completions = [c for c in completions if c["measured"]]
        tpot = [c["decode_ms"] / max(c["generated"] - 1, 1) for c in measured_completions]
        metrics = {
            "stop_reason": stop_reason,
            "warmup_samples_effective": warmup_n,
            "min_eval_samples_effective": min_eval_n,
            "total_completions": total_completed,
            "evaluated_completions": evaluated if cfg.mode == "cyclic" else total_completed,
            "measured_tokens": measured_tokens,
            "measured_forward_s": measured_ms / 1000,
            "measured_wall_s": wall_measured,
            "throughput_tok_s": measured_tokens / (measured_ms / 1000) if measured_ms else None,
            "wall_throughput_tok_s": measured_tokens / wall_measured if wall_measured else None,
            "mean_tpot_ms": float(np.mean(tpot)) if tpot else None,
            "steps": per_kind,
            **self.extra_metrics(),
        }
        return {
            "metrics": metrics,
            "completions": completions,
            "steps": [asdict(r) for r in self.records],
        }


# ----------------------------------------------------------------------------- AR

class ARBatchEngine(DecodeEngine):
    name = "autoregressive"

    def _iteration(self, measured: bool) -> int:
        m = self.model
        dev = self.device
        tokens = torch.tensor([[s.last_token] for s in self.slots], dtype=torch.long, device=dev)
        pos_host = [s.pos for s in self.slots]
        pos = torch.tensor(pos_host, dtype=torch.long, device=dev).unsqueeze(1)
        cap = [] if self.probe is not None else None

        timer = Interval(dev).start()
        ctx = m.make_ctx(pos, 0, max(pos_host))
        h = m.forward_layers(m.embed(tokens), 0, m.n_layers, ctx, capture=cap)
        logits = m.head(h[:, -1])
        nxt = self._choose(logits)
        timer.stop()

        nxt_host = nxt.tolist()
        self.records.append(StepRecord("ar", timer.ms, self.B, self.B, self.B, ctx.kv_read, measured))
        if self.probe is not None:
            self.probe.on_step(self, [c[:, -1] for c in cap], logits, nxt_host)
        committed = 0
        for i, s in enumerate(self.slots):
            if s.idle:
                continue
            s.pos += 1
            s.last_token = nxt_host[i]
            committed += self._commit(i, [nxt_host[i]])
        return committed


# ----------------------------------------------------------------------------- sync LayerSkip

class SyncSpecEngine(DecodeEngine):
    """Synchronous batched LayerSkip: all requests draft `gamma` tokens, then verify."""

    name = "sync_layerskip"

    def __init__(self, model: LayerSkipLlama, batch_size: int, max_prompt_len: int, max_new_tokens: int,
                 eos_ids, *, exit_layer: int, gamma: int, **kw):
        if kw.get("temperature", 0.0) > 0:
            raise NotImplementedError("SyncSpecEngine implements greedy verification only")
        if not 1 <= exit_layer < model.n_layers:
            raise ValueError(f"exit_layer must be in [1, {model.n_layers - 1}]")
        super().__init__(model, batch_size, max_prompt_len, max_new_tokens, eos_ids,
                         spec_slack=gamma + 32, **kw)
        self.k = exit_layer
        self.gamma = gamma
        self.exit_h = torch.zeros(batch_size, max(gamma, 1), model.cfg.dim, dtype=model.dtype, device=self.device)
        self.stats = {"rounds": 0, "request_rounds": 0, "drafted": 0, "accepted": 0,
                      "rejected": 0, "overdraft_steps": 0, "full_accept_rounds": 0}
        self.accept_hist = np.zeros(gamma + 1, dtype=np.int64)

    def _iteration(self, measured: bool) -> int:
        m, dev, k, g, B = self.model, self.device, self.k, self.gamma, self.B
        P_host = [s.pos for s in self.slots]
        P = torch.tensor(P_host, dtype=torch.long, device=dev).unsqueeze(1)          # [B,1]
        inp = torch.tensor([[s.last_token] for s in self.slots], dtype=torch.long, device=dev)
        maxP = max(P_host)
        drafts = torch.empty(B, g, dtype=torch.long, device=dev)
        timers = []

        # ---- draft: gamma steps through layers [0, k)
        for j in range(g):
            t = Interval(dev).start()
            ctx = m.make_ctx(P + j, 0, maxP + j)
            h = m.forward_layers(m.embed(inp), 0, k, ctx)
            self.exit_h[:, j] = h[:, 0]
            d = m.head(h[:, -1]).argmax(-1)
            t.stop()
            drafts[:, j] = d
            inp = d.unsqueeze(1)
            timers.append(("draft", t, B, B, 0, ctx.kv_read))

        # ---- verify: shallow only for the last (not yet processed) token, deep for gamma+1 tokens
        t = Interval(dev).start()
        ctx_s = m.make_ctx(P + g, 0, maxP + g)
        h_last = m.forward_layers(m.embed(inp), 0, k, ctx_s)
        deep_in = torch.cat([self.exit_h[:, :g], h_last], dim=1) if g > 0 else h_last
        pos_d = P + torch.arange(g + 1, device=dev).unsqueeze(0)
        ctx_d = m.make_ctx(pos_d, 0, maxP + g)
        h = m.forward_layers(deep_in, k, m.n_layers, ctx_d)
        Y = m.head(h).argmax(-1)                                                   # [B, g+1]
        if g > 0:
            match = (drafts == Y[:, :g]).int()
            acc = match.cumprod(dim=1).sum(dim=1)
        else:
            acc = torch.zeros(B, dtype=torch.long, device=dev)
        t.stop()
        timers.append(("verify", t, B, B, B * (g + 1), ctx_d.kv_read))

        acc_host = acc.tolist()
        Y_host = Y.tolist()
        D_host = drafts.tolist()
        for kind, tm, b, st, dt, kv in timers:
            self.records.append(StepRecord(kind, tm.ms, b, st, dt, kv, measured))

        committed = 0
        self.stats["rounds"] += 1
        for i, s in enumerate(self.slots):
            if s.idle:
                continue
            a = acc_host[i]
            new = D_host[i][:a] + [Y_host[i][a]]
            s.pos = P_host[i] + a + 1
            s.last_token = Y_host[i][a]
            committed += self._commit(i, new)
            # the round in which a request finishes is truncated by EOS / max_new: keep it
            # out of the acceptance statistics (the trace simulator does the same)
            if measured and s.finish_reason is None:
                st = self.stats
                st["request_rounds"] += 1
                st["drafted"] += g
                st["accepted"] += a
                st["rejected"] += g - a
                st["overdraft_steps"] += max(g - a - 1, 0)
                st["full_accept_rounds"] += int(a == g)
                self.accept_hist[a] += 1
        return committed

    def extra_metrics(self) -> dict:
        st = dict(self.stats)
        rr = max(st["request_rounds"], 1)
        st.update(
            exit_layer=self.k,
            gamma=self.gamma,
            acceptance_rate=st["accepted"] / max(st["drafted"], 1),
            rejected_draft_ratio=st["rejected"] / max(st["drafted"], 1),
            overdraft_step_ratio=st["overdraft_steps"] / max(st["drafted"], 1),
            full_accept_round_ratio=st["full_accept_rounds"] / rr,
            tokens_per_verify=(st["accepted"] + st["request_rounds"]) / rr,
            accept_hist=self.accept_hist.tolist(),
        )
        return {"spec": st}


def add_run_args(parser) -> None:
    g = parser.add_argument_group("run (cyclic replay, as ASPIRE)")
    g.add_argument("--mode", default="cyclic", choices=["cyclic", "static"])
    g.add_argument("--warmup-samples", type=int, default=8)
    g.add_argument("--max-eval-samples", type=int, default=256)
    g.add_argument("--no-converge", action="store_true")
    g.add_argument("--converge-threshold", type=float, default=0.01)
    g.add_argument("--converge-window", type=int, default=4)
    g.add_argument("--converge-min-samples", type=int, default=16)
    g.add_argument("--max-steps", type=int, default=None, help="hard cap on model forward steps")
    g.add_argument("--warmup-per-slot", type=float, default=1.0,
                   help="effective warm-up = max(--warmup-samples, this * batch size) completions")
    g.add_argument("--min-eval-per-slot", type=float, default=2.0,
                   help="effective min evaluated completions = max(--converge-min-samples, this * batch size)")
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--ignore-eos", action="store_true", help="always generate max-new-tokens (fixed-length runs)")


def run_config_from_args(args, keep_outputs: bool = False) -> RunConfig:
    return RunConfig(
        mode=args.mode, warmup_samples=args.warmup_samples, max_eval_samples=args.max_eval_samples,
        converge=not args.no_converge, converge_threshold=args.converge_threshold,
        converge_window=args.converge_window, converge_min_samples=args.converge_min_samples,
        max_steps=args.max_steps, keep_outputs=keep_outputs,
        warmup_per_slot=args.warmup_per_slot, min_eval_per_slot=args.min_eval_per_slot,
    )
