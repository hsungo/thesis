"""Portable per-layer cost model for the EESD engine (plan S2).

    t_layer(n, C) = max(W / BW, 2 * P * n / F) + KV(C) / BW + n_op * t_op          (roofline)

    W      bytes of one decoder layer's weights            (from config.json)
    P      parameters of one decoder layer                 (from config.json)
    n      tokens through the layer in this step
    KV(C)  bytes of K and V read by attention for C cached positions (summed over requests)
    BW     effective weight-streaming bandwidth (large GEMV)          \\
    F      effective bf16 GEMM throughput of the model's own layer      |  calibrate(), once per GPU
           shapes at decode sizes (F_eff; the 8192^3 peak is kept as     |
           the "roofline-peak" variant)                                  |
    t_op   time of one eager op (empty-kernel loop, launch-bound)      |
    n_op   ops issued per layer by this implementation (traced)       /

The LM head (final norm + lm_head, shared by every exit) uses the same form with the head
weights.  GEMM terms are tabulated for n = 0..n_max at start-up; at run time only the KV term
is added.  Provided queries (all in ms):

    t_sh(B, k, C)                 shallow step: B tokens through layers [0,k) + head on B tokens
    t_deep(n, k, C, n_req)        deep step: n_req latest drafts through [0,k), n tokens through [k,L), head on n
    t_prefill(n, C)               separate prefill chunk: n tokens through all L layers + head on 1 token
    t_segment(n_layers, n, C)     generic segment (used by the M3 marginal-cost test)
    knee(k)                       tokens per layer where a layer turns compute-bound: the hinge knee measured by
                                  calibrate (knee_measured); F*W / (2*P*BW) only if no measurement exists

`LinearCostModel` (ablation, ASPIRE-style 3 parameters) prices a layer as a + b*n + c*KV positions,
least-squares fitted on measured layer times; it has the same interface.

Calibration (GPU) writes a JSON with both models' parameters:
    python -m solutions.calibrate --model layerskip-llama3-8B
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ------------------------------------------------------------------------------ model dimensions

@dataclass
class ModelDims:
    n_layers: int
    dim: int
    vocab: int
    layer_params: int          # P
    head_params: int           # final norm + lm_head
    kv_bytes_per_pos: int      # K+V bytes per cached position per layer
    dtype_bytes: int = 2

    @property
    def layer_bytes(self) -> int:   # W
        return self.layer_params * self.dtype_bytes

    @property
    def head_bytes(self) -> int:
        return self.head_params * self.dtype_bytes

    @classmethod
    def from_config(cls, cfg, dtype_bytes: int = 2) -> "ModelDims":
        """cfg: common.model.ModelConfig (or a path to a model folder with config.json)."""
        if not hasattr(cfg, "n_layers"):
            from common.model import ModelConfig
            cfg = ModelConfig.from_hf_dir(cfg)
        return cls(cfg.n_layers, cfg.dim, cfg.vocab_size, cfg.layer_param_count, cfg.vocab_size * cfg.dim + cfg.dim,
                   2 * cfg.n_kv_heads * cfg.head_dim * dtype_bytes, dtype_bytes)


# ------------------------------------------------------------------------------ GPU parameters

@dataclass
class GPUParams:
    bw: float                  # bytes / ms
    flops: float               # flop / ms
    t_op: float                # ms per eager op
    n_op_layer: float          # ops per layer with one attention bucket
    n_op_bucket: float         # extra ops per additional attention bucket
    n_op_head: float           # ops of norm + lm_head + argmax
    n_op_segment: float        # ops to build a packed segment's metadata (per segment)
    meta: dict = field(default_factory=dict)
    flops_eff: Optional[float] = None       # flop / ms the model's own layer GEMMs reach for n <= 512 (calibrate)
    knee_measured: Optional[float] = None   # tokens: hinge fit of one measured decoder layer over n = 1..512


# ------------------------------------------------------------------------------ models

class _Base:
    dims: ModelDims

    # ---- primitives implemented by subclasses: layer_gemm_ms(n) [array ok], head_ms(n), kv_ms(C), ops_ms(...)
    def t_segment(self, n_layers: int, n, C, buckets: int = 1):
        """n_layers layers on n tokens reading C cached positions (summed over requests)."""
        n = np.asarray(n)
        t = n_layers * (self.layer_gemm_ms(n) + self.kv_ms(C) + self.layer_ops_ms(buckets)) + self.segment_ops_ms()
        return np.where(n > 0, t, 0.0) if n_layers else np.zeros_like(n, dtype=np.float64)

    def t_sh(self, B, k: int, C):
        return self.t_segment(k, B, C) + self.head_ms(B)

    def t_deep(self, n, k: int, C, n_req, buckets: int = 1):
        """Deep step: latest draft of every verifier through [0,k), all n tokens through [k,L), head on n."""
        return (self.t_segment(k, n_req, C) + self.t_segment(self.dims.n_layers - k, n, C, buckets) + self.head_ms(n))

    def t_prefill(self, n, C):
        return self.t_segment(self.dims.n_layers, n, C) + self.head_ms(1)

    def t_ar(self, B, C):
        return self.t_segment(self.dims.n_layers, B, C) + self.head_ms(B)

    def best_rate(self, alpha: float, B: int, k: int, C: float, d_max: int = 16) -> float:
        """max_s B * E(alpha, s) / (s * t_sh + t_deep)  (synchronous round, used by M2)."""
        a = min(max(alpha, 0.0), 1 - 1e-6)
        s = np.arange(d_max + 1)
        tok = B * (1 - a ** (s + 1)) / (1 - a)
        t = s * self.t_sh(B, k, C) + self.t_deep(B * (s + 1), k, C, B)
        return float(np.max(tok / t))


class RooflineCostModel(_Base):
    """variant="measured" (default): compute term uses F_eff (the model's own GEMM shapes at decode sizes)
    and knee() returns the measured knee; variant="peak": the original form (8192^3 GEMM peak F, analytic
    knee F*W/(2*P*BW)), kept for the MAPE comparison in calibrate --validate.  Calibration files without
    the measured fields fall back to the peak values."""
    kind = "roofline"

    def __init__(self, gpu: GPUParams, dims: ModelDims, n_max: int = 16384, variant: str = "measured"):
        self.gpu, self.dims, self.variant = gpu, dims, variant
        use_eff = variant == "measured" and gpu.flops_eff
        self.F = float(gpu.flops_eff) if use_eff else float(gpu.flops)
        n = np.arange(n_max + 1, dtype=np.float64)
        self._layer_tab = np.maximum(dims.layer_bytes / gpu.bw, 2.0 * dims.layer_params * n / self.F)
        self._head_tab = np.maximum(dims.head_bytes / gpu.bw, 2.0 * dims.head_params * n / self.F)
        self._n_max = n_max

    def _lookup(self, tab, n, w_bytes, params):
        """Table lookup for n <= n_max (start-up precomputation), exact formula beyond."""
        n = np.asarray(n)
        idx = np.clip(n, 0, self._n_max).astype(np.int64)
        exact = np.maximum(w_bytes / self.gpu.bw, 2.0 * params * n / self.F)
        return np.where(n <= self._n_max, tab[idx], exact)

    def layer_gemm_ms(self, n):
        return self._lookup(self._layer_tab, n, self.dims.layer_bytes, self.dims.layer_params)

    def head_ms(self, n):
        n = np.asarray(n)
        return np.where(n > 0, self._lookup(self._head_tab, n, self.dims.head_bytes, self.dims.head_params)
                        + self.gpu.n_op_head * self.gpu.t_op, 0.0)

    def kv_ms(self, C):
        return np.asarray(C, dtype=np.float64) * self.dims.kv_bytes_per_pos / self.gpu.bw

    def layer_ops_ms(self, buckets: int = 1):
        return (self.gpu.n_op_layer + self.gpu.n_op_bucket * max(buckets - 1, 0)) * self.gpu.t_op

    def segment_ops_ms(self):
        return self.gpu.n_op_segment * self.gpu.t_op

    def knee(self, k: int | None = None) -> float:
        """Tokens per layer where a layer turns compute-bound (same for every k: all layers have one shape).
        measured variant: the hinge knee fitted on measured layer times; otherwise F*W / (2*P*BW)."""
        if self.variant == "measured" and self.gpu.knee_measured:
            return float(self.gpu.knee_measured)
        return self.F * self.dims.layer_bytes / (2.0 * self.dims.layer_params * self.gpu.bw)


class LinearCostModel(_Base):
    """ASPIRE-style linear latency model: t_layer = a + b * n + c * KV positions (no knee)."""
    kind = "linear"

    def __init__(self, a: float, b: float, c: float, head_a: float, head_b: float, dims: ModelDims,
                 seg_ms: float = 0.0):
        self.a, self.b, self.c, self.head_a, self.head_b, self.dims, self.seg = a, b, c, head_a, head_b, dims, seg_ms

    def layer_gemm_ms(self, n):
        return self.a + self.b * np.asarray(n, dtype=np.float64)

    def head_ms(self, n):
        n = np.asarray(n, dtype=np.float64)
        return np.where(n > 0, self.head_a + self.head_b * n, 0.0)

    def kv_ms(self, C):
        return self.c * np.asarray(C, dtype=np.float64)

    def layer_ops_ms(self, buckets: int = 1):
        return 0.0

    def segment_ops_ms(self):
        return self.seg

    def knee(self, k: int | None = None) -> float:
        return float("inf")   # no knee in a linear model


# ------------------------------------------------------------------------------ IO

def load_cost_model(path: str | Path, cfg, kind: str = "roofline"):
    """path: calibration JSON written by solutions/calibrate.py; cfg: ModelConfig of the served model."""
    d = json.loads(Path(path).read_text())
    dims = ModelDims.from_config(cfg)
    if kind in ("roofline", "roofline-peak"):
        return RooflineCostModel(GPUParams(**d["gpu"]), dims, variant="measured" if kind == "roofline" else "peak")
    if kind == "linear":
        lin = d.get("linear")
        if lin is None:
            raise ValueError(f"{path} has no linear fit (run solutions/calibrate.py with the model loaded)")
        if lin.get("model_name") and lin.get("n_layers") != dims.n_layers:
            raise ValueError("linear fit belongs to a different model")
        return LinearCostModel(lin["a"], lin["b"], lin["c"], lin["head_a"], lin["head_b"], dims, lin.get("seg_ms", 0.0))
    raise ValueError(kind)


def synthetic_cost_model(cfg, kind: str = "roofline"):
    """Deterministic made-up GPU (RTX-4090-like) for CPU tests and dry runs - never used for results."""
    dims = ModelDims.from_config(cfg)
    gpu = GPUParams(bw=1.0e9, flops=1.6e11, t_op=0.003, n_op_layer=52, n_op_bucket=12, n_op_head=11, n_op_segment=20,
                    meta={"source": "synthetic (tests only)"}, flops_eff=1.0e11, knee_measured=100.0)
    if kind in ("roofline", "roofline-peak"):
        return RooflineCostModel(gpu, dims, variant="measured" if kind == "roofline" else "peak")
    return LinearCostModel(0.6, 0.002, 4e-6, 1.2, 0.007, dims)


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
