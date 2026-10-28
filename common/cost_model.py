"""Per-layer, batch-aware step-time model (plan sec.S2, fitted by observation O5).

    tau_t = beta0 + head(n_head) + sum_l 1[n_{t,l} > 0] * layer(n_{t,l}, KV_{t,l}, L_{t,l})
    layer(n, KV, L) = b_w + b_mlp * max(n - n_knee, 0) + b_attn * KV + b_len * L
    head(n)         = h_w + h_tok * max(n - h_knee, 0)

* b_w     per-layer fixed cost (weight loading + kernel launches), paid iff the layer runs
* b_mlp   per-token cost of one layer (all GEMMs, norms, RoPE) once the GEMMs are compute-bound
* n_knee  tokens per layer at which the GEMMs turn compute-bound.  Below the knee a layer
          costs the same for 1 or n_knee tokens (weight-load bound).  On an RTX 4090 the knee
          is ~100 tokens (436 MB of layer weights at ~1 TB/s vs ~80 TFLOPS), i.e. exactly the
          B*(gamma+1) range of a batched verify step, so a purely linear model (n_knee = 0,
          the plan's original formula) mis-prices verification; O5 reports both fits.
* b_attn  per-KV-position cost of one layer (attention read, B*L positions)
* b_len   per padded-sequence-position cost of one layer, independent of the batch (mask
          construction + SDPA over the padded length in this eager implementation)
* h_w/h_tok/h_knee  final norm + LM head (the LM head of a 128k vocab is ~2 layers of
          weights, so it is kept separate from beta0)

The plan's formula folds the LM head into beta0; keeping it explicit matters because
draft steps and verify steps both run it, but with different token counts.  With
n_knee = h_knee = b_len = 0 the model is the plan's linear model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def nnls_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with non-negative coefficients (tiny active-set loop, no scipy)."""
    active = list(range(X.shape[1]))
    while True:
        beta = np.zeros(X.shape[1])
        if active:
            sol, *_ = np.linalg.lstsq(X[:, active], y, rcond=None)
            beta[active] = sol
        neg = [j for j in active if beta[j] < 0]
        if not neg:
            return beta
        worst = min(neg, key=lambda j: beta[j])
        active.remove(worst)


@dataclass
class LayerCostModel:
    n_layers: int
    beta0: float
    bw: float
    bmlp: float
    battn: float
    head_w: float
    head_tok: float
    meta: dict = field(default_factory=dict)
    knee: float = 0.0
    head_knee: float = 0.0
    blen: float = 0.0

    # ------------------------------------------------------------------ primitives
    def layer(self, n_tokens, kv, length=0.0) -> np.ndarray:
        """One layer on n tokens reading kv KV positions; `length` = padded sequence length."""
        n = np.asarray(n_tokens, dtype=np.float64)
        t = (self.bw + self.bmlp * np.maximum(n - self.knee, 0.0) + self.battn * np.asarray(kv, dtype=np.float64)
             + self.blen * np.asarray(length, dtype=np.float64))
        return np.where(n > 0, t, 0.0)

    def head(self, n_tokens) -> np.ndarray:
        n = np.asarray(n_tokens, dtype=np.float64)
        return np.where(n > 0, self.head_w + self.head_tok * np.maximum(n - self.head_knee, 0.0), 0.0)

    def step_layers(self, n_per_layer: Sequence[float], kv_per_layer: Sequence[float], n_head: float,
                    len_per_layer: Sequence[float] | float = 0.0) -> float:
        """General step time given per-layer token counts and KV reads."""
        return float(self.beta0 + self.head(n_head) + np.sum(self.layer(n_per_layer, kv_per_layer, len_per_layer)))

    def two_segment(self, k: int, n_shallow, kv_shallow, n_deep, kv_deep, n_head,
                    len_shallow=0.0, len_deep=0.0) -> np.ndarray:
        """Step with a shallow segment [0,k) and a deep segment [k,N) (plan S1)."""
        return (self.beta0 + self.head(n_head) + k * self.layer(n_shallow, kv_shallow, len_shallow)
                + (self.n_layers - k) * self.layer(n_deep, kv_deep, len_deep))

    # ------------------------------------------------------------------ sync-batch helpers (homogeneous batch)
    def sync_draft(self, k: int, B: int, ctx) -> np.ndarray:
        """One draft step: B requests x 1 token through layers [0,k) + head."""
        ctx = np.asarray(ctx, dtype=np.float64)
        return self.beta0 + self.head(B) + k * self.layer(B, B * ctx, ctx)

    def sync_verify(self, k: int, gamma, B: int, ctx) -> np.ndarray:
        """One verify step with KV reuse: shallow 1 token/request, deep gamma+1 tokens/request."""
        ctx = np.asarray(ctx, dtype=np.float64)
        g = np.asarray(gamma, dtype=np.float64)
        return (self.beta0 + self.head(B * (g + 1)) + k * self.layer(B, B * ctx, ctx)
                + (self.n_layers - k) * self.layer(B * (g + 1), B * ctx, ctx))

    def ar_step(self, B: int, ctx) -> np.ndarray:
        return self.sync_verify(self.n_layers, 0, B, ctx)

    def fixed_fraction(self, B: int, ctx) -> np.ndarray:
        """Share of an AR step spent on per-layer + head fixed costs (O5 figure a)."""
        fixed = self.n_layers * self.bw + self.head_w
        return fixed / self.ar_step(B, ctx)

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LayerCostModel":
        d = json.loads(Path(path).read_text())
        d.pop("alternatives", None)
        return cls(**d)

    @classmethod
    def layer_count(cls, n_layers: int, head_equiv_layers: float = 2.0) -> "LayerCostModel":
        """Fallback when no profile exists: cost = number of layers run (weight-bound),
        LM head worth `head_equiv_layers` layers.  Only for quick looks at O3/O6."""
        return cls(n_layers, 0.0, 1.0, 0.0, 0.0, head_equiv_layers, 0.0, {"source": "layer-count fallback"})

    @classmethod
    def load_or_fallback(cls, path: str | None, n_layers: int) -> "LayerCostModel":
        if path:
            m = cls.load(path)
            if m.n_layers != n_layers:
                raise ValueError(f"cost model has {m.n_layers} layers, traces have {n_layers}")
            return m
        print("[cost] no --cost-model given: using the layer-count fallback (run O5 profile_layers.py first!)")
        return cls.layer_count(n_layers)

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _hinge_fit(n: np.ndarray, extra: np.ndarray, y: np.ndarray, knees: Sequence[float], relative: bool = False):
        """NNLS of y ~ [1, max(n - knee, 0), extra...] for every knee; returns the best
        (sse, knee, coefficients).  relative=True minimises squared *relative* error, so the
        many cheap draft-sized steps are not drowned by a few large compute-bound ones."""
        best = None
        w = 1.0 / np.maximum(y, 1e-9) if relative else np.ones_like(y)
        for kn in knees:
            X = np.c_[np.ones_like(n), np.maximum(n - kn, 0.0), extra]
            beta = nnls_lstsq(X * w[:, None], y * w)
            sse = float(np.sum(((X @ beta - y) * w) ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(kn), beta)
        return best

    @classmethod
    def fit(cls, layer_rows: Iterable[dict], head_rows: Iterable[dict], step_rows: Iterable[dict],
            n_layers: int, meta: dict | None = None, form: str = "hinge",
            knee_grid: Sequence[float] | None = None) -> "LayerCostModel":
        """layer_rows: {n, kv, ms[, B]}; head_rows: {n, ms}; step_rows: {ms, layers_ms, head_ms}.

        form = "linear": the plan's model (no knee, no length term), ordinary least squares;
               "hinge":  knee + padded-length term (L = kv / B, needs B in the rows), fitted on
                         relative error."""
        lr = list(layer_rows)
        n = np.array([float(r["n"]) for r in lr])
        kv = np.array([float(r["kv"]) for r in lr])
        y = np.array([float(r["ms"]) for r in lr])
        hr = list(head_rows)
        nh = np.array([float(r["n"]) for r in hr])
        yh = np.array([float(r["ms"]) for r in hr])
        if form == "linear":
            knees, hknees = [0.0], [0.0]
            extra = kv[:, None]
        elif form == "hinge":
            top = float(np.percentile(n, 90)) if len(n) else 0.0
            knees = list(knee_grid) if knee_grid is not None else list(np.arange(0.0, max(top, 4.0), 4.0))
            hknees = knees
            length = kv / np.array([float(r.get("B", 1)) for r in lr])
            extra = np.c_[kv, length]
        else:
            raise ValueError(f"unknown cost model form {form!r}")
        rel = form == "hinge"
        _, knee, b = cls._hinge_fit(n, extra, y, knees, rel)
        bw, bmlp, battn = b[0], b[1], b[2]
        blen = b[3] if form == "hinge" else 0.0
        _, hknee, bh = cls._hinge_fit(nh, np.zeros((len(nh), 0)), yh, hknees, rel)
        hw, ht = bh[0], bh[1]
        model = cls(n_layers, 0.0, float(bw), float(bmlp), float(battn), float(hw), float(ht), {},
                    knee=knee, head_knee=hknee, blen=float(blen))
        resid = [r["ms"] - r["layers_ms"] - r["head_ms"] for r in step_rows]
        model.beta0 = float(max(np.median(resid), 0.0)) if resid else 0.0
        length = kv / np.array([float(r.get("B", 1)) for r in lr])
        pred = model.layer(n, kv, length)
        model.meta = {
            **(meta or {}), "form": form,
            "layer_fit_r2": float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
            "layer_fit_mape": float(np.mean(np.abs(pred - y) / np.maximum(y, 1e-9))),
            "layer_fit_max_ape": float(np.max(np.abs(pred - y) / np.maximum(y, 1e-9))),
            "layer_samples": len(lr),
            "head_samples": len(hr),
        }
        return model
