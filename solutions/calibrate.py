"""Calibrate the portable cost model on this GPU and report its prediction error (GPU, real measurements).

    python -m solutions.calibrate --model layerskip-llama3-8B            # writes results/eesd/calibration/<model>.json
    python -m solutions.calibrate --model layerskip-llama3-8B --validate # + measured steps -> MAPE table

What is measured
    BW      large GEMV (n=1) through a 28672 x 8192 bf16 weight (weight-streaming bandwidth)
    F       8192^3 bf16 GEMM
    t_op    a loop of tiny eager kernels (launch-bound time per op)
    n_op    kernels issued by one packed decoder layer (1 and 2 attention buckets), by the head and by
            the packed-metadata build, counted with the CUDA profiler on the loaded model
    linear  ASPIRE-style 3-parameter fit (a + b n + c KV) on measured per-layer times of the real
            packed layer over a (tokens, context) grid, and a 2-parameter head fit
Validation (``--validate``) times real shallow steps, deep steps and prefill chunks of the engine's
packed code path on a grid and reports MAPE / max APE of the roofline and linear predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_model_args, load_model_from_args  # noqa: E402
from common.utils import env_info, seed_everything  # noqa: E402
from solutions.cost_model import GPUParams, LinearCostModel, ModelDims, RooflineCostModel, save_json  # noqa: E402
from solutions.packed import Block, build_packed, packed_layer, packed_layers  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eesd"


def _t(fn, reps=20, warm=5) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return float(np.median(ts))


def micro(dev) -> dict:
    W = torch.randn(28672, 8192, device=dev, dtype=torch.bfloat16)
    x = torch.randn(1, 8192, device=dev, dtype=torch.bfloat16)
    bw = W.numel() * 2 / _t(lambda: F.linear(x, W))                   # bytes / ms
    del W
    A = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    Bm = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    flops = 2 * 8192 ** 3 / _t(lambda: A @ Bm, reps=10)                # flop / ms
    del A, Bm
    s = torch.zeros(8, device=dev)

    def loop():
        for _ in range(1000):
            s.add_(1.0)
    t_op = _t(loop, reps=10) / 1000
    torch.cuda.empty_cache()
    return {"bw": bw, "flops": flops, "t_op": t_op}


def count_kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def trace_ops(m) -> dict:
    m.allocate_cache(4, 512)
    one = build_packed(m, [Block(0, 100, 1), Block(1, 100, 1)])
    two = build_packed(m, [Block(0, 100, 1), Block(1, 100, 3)])
    x1 = torch.randn(one.n_tokens, m.cfg.dim, device=m.device, dtype=m.dtype)
    x2 = torch.randn(two.n_tokens, m.cfg.dim, device=m.device, dtype=m.dtype)
    with torch.inference_mode():
        n1 = count_kernels(lambda: packed_layer(m, 0, x1, one))
        n2 = count_kernels(lambda: packed_layer(m, 0, x2, two))
        nh = count_kernels(lambda: m.head(x1).argmax(-1))
        ns = count_kernels(lambda: build_packed(m, [Block(0, 100, 1), Block(1, 100, 1)]))
    return {"n_op_layer": n1, "n_op_bucket": max(n2 - n1, 0), "n_op_head": nh, "n_op_segment": ns}


def layer_grid(m, Bs=(1, 4, 16, 32, 64), Ts=(1, 4, 8, 17), ctxs=(256, 1024, 2048)):
    rows, heads = [], []
    with torch.inference_mode():
        for C in ctxs:
            for B in Bs:
                try:
                    m.allocate_cache(B, C + 64)
                except torch.OutOfMemoryError:
                    m.free_cache()
                    continue
                for T in Ts:
                    pb = build_packed(m, [Block(b, C - T, T) for b in range(B)])
                    x = torch.randn(pb.n_tokens, m.cfg.dim, device=m.device, dtype=m.dtype)
                    ms = _t(lambda: packed_layer(m, 1, x, pb), reps=10, warm=3)
                    rows.append({"n": B * T, "kv": pb.kv_read, "ms": ms, "B": B, "T": T, "ctx": C})
                    if C == ctxs[0]:
                        heads.append({"n": B * T, "ms": _t(lambda: m.head(x).argmax(-1), reps=10, warm=3)})
                m.free_cache()
    return rows, heads


def fit_linear(rows, heads) -> dict:
    X = np.array([[1.0, r["n"], r["kv"]] for r in rows])
    y = np.array([r["ms"] for r in rows])
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    Xh = np.array([[1.0, h["n"]] for h in heads])
    ha, hb = np.linalg.lstsq(Xh, np.array([h["ms"] for h in heads]), rcond=None)[0]
    return {"a": float(a), "b": float(b), "c": float(c), "head_a": float(ha), "head_b": float(hb)}


def validate(m, cms: dict, k: int, out_csv: Path) -> dict:
    """Measured engine-path steps vs predictions."""
    L = m.n_layers
    rows = []
    with torch.inference_mode():
        for C in (256, 1024, 2048):
            for B in (1, 4, 8, 16, 32):
                try:
                    m.allocate_cache(B, C + 2048 + 64)
                except torch.OutOfMemoryError:
                    m.free_cache()
                    continue
                # shallow step: B decode tokens through [0,k) + head
                pb = build_packed(m, [Block(b, C, 1) for b in range(B)])
                x = torch.randn(B, m.cfg.dim, device=m.device, dtype=m.dtype)
                ms = _t(lambda: m.head(packed_layers(m, x, pb, 0, k)).argmax(-1), reps=10, warm=3)
                rows.append({"kind": "shallow", "B": B, "ctx": C, "T": 1, **{n: float(c.t_sh(B, k, pb.kv_read)) for n, c in cms.items()}, "ms": ms})
                for T in (2, 4, 6, 9, 17):                     # deep step: latest draft [0,k), T tokens [k,L), head
                    pbs = build_packed(m, [Block(b, C + T - 1, 1) for b in range(B)])
                    pbd = build_packed(m, [Block(b, C, T) for b in range(B)])
                    xd = torch.randn(B * T, m.cfg.dim, device=m.device, dtype=m.dtype)

                    def deep():
                        packed_layers(m, x, pbs, 0, k)
                        m.head(packed_layers(m, xd, pbd, k, L)).argmax(-1)
                    ms = _t(deep, reps=10, warm=3)
                    rows.append({"kind": "deep", "B": B, "ctx": C, "T": T,
                                 **{n: float(c.t_deep(B * T, k, pbd.kv_read, B)) for n, c in cms.items()}, "ms": ms})
                for n_tok in (64, 256, 1024):                  # separate prefill chunk on row 0
                    if B != 1:
                        break
                    pbp = build_packed(m, [Block(0, C, n_tok)])
                    xp = torch.randn(n_tok, m.cfg.dim, device=m.device, dtype=m.dtype)
                    ms = _t(lambda: m.head(packed_layers(m, xp, pbp, 0, L)[-1:]).argmax(-1), reps=5, warm=2)
                    rows.append({"kind": "prefill", "B": 1, "ctx": C, "T": n_tok,
                                 **{n: float(c.t_prefill(n_tok, pbp.kv_read)) for n, c in cms.items()}, "ms": ms})
                m.free_cache()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for name in cms:
        for kind in ("shallow", "deep", "prefill", "all"):
            rr = [r for r in rows if kind == "all" or r["kind"] == kind]
            if rr:
                ape = np.array([abs(r[name] - r["ms"]) / r["ms"] for r in rr])
                summ[f"{name}_{kind}"] = {"n": len(rr), "mape": float(ape.mean()), "max_ape": float(ape.max())}
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap)
    ap.add_argument("--k", type=int, default=3, help="exit layer used for the validation steps")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--skip-linear", action="store_true", help="GPU constants only (no per-model layer grid)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    model = load_model_from_args(args)
    if model.device.type != "cuda":
        raise SystemExit("calibration needs a CUDA GPU")
    name = Path(args.model).name
    out = Path(args.out) if args.out else RESULTS / "calibration" / f"{name}.json"
    mic = micro(model.device)
    ops = trace_ops(model)
    gpu = GPUParams(**mic, **ops, meta={"host": socket.gethostname(), "env": env_info(model.device)})
    res = {"model": name, "gpu": gpu, "model_dims": ModelDims.from_config(model.cfg).__dict__}
    if not args.skip_linear:
        rows, heads = layer_grid(model)
        res["linear"] = {**fit_linear(rows, heads), "n_layers": model.n_layers, "model_name": name}
        res["linear_grid"] = rows
    print(json.dumps({k: v for k, v in res.items() if k != "linear_grid"}, indent=2, default=str))
    if args.validate:
        dims = ModelDims.from_config(model.cfg)
        cms = {"roofline": RooflineCostModel(gpu, dims)}
        if "linear" in res:
            lin = res["linear"]
            cms["linear"] = LinearCostModel(lin["a"], lin["b"], lin["c"], lin["head_a"], lin["head_b"], dims)
        res["validation"] = validate(model, cms, args.k, out.with_name(out.stem + "_validation.csv"))
        for kname, v in res["validation"].items():
            print(f"[calibrate] {kname:20s} n={v['n']:3d}  MAPE {100 * v['mape']:5.1f}%  max {100 * v['max_ape']:5.1f}%")
    save_json(out, res)
    print(f"[calibrate] wrote {out}")


if __name__ == "__main__":
    main()
