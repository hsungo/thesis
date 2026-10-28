"""Measurements and ablations of the EESD engine (real GPU runs only).  Resumable: one JSON per run.

    python -m solutions.run_ablation --preset phase0              # stage 0: AR / fixed / aspire baselines
    python -m solutions.run_ablation --preset phase1              # + M1 (and M1 with the linear cost model)
    python -m solutions.run_ablation --preset ksweep              # fixed k = 2..8 (upper bound for M2)
    python -m solutions.run_ablation --preset phase2              # + M2
    python -m solutions.run_ablation --preset prefill-share       # prefill time share (upper bound for M3)
    python -m solutions.run_ablation --preset phase3              # + M3
    python -m solutions.run_ablation --preset full                # complete ablation (3 models x 4 tasks x 3 lengths x B x configs x 3 reps)
    python -m solutions.run_ablation --preset poisson             # dynamic mode: Poisson arrivals
    python -m solutions.run_ablation --summarize --out results/eesd/ablation   # CSV + tables only (no GPU)
    ... --dry-run                                                 # list the runs, no GPU

Every option of a preset can be overridden (--models, --datasets, --lengths, --batch-sizes, --configs, --reps, ...).

Configurations (``--configs``):
    ar                       autoregressive
    fixed-g<g>               synchronous, gamma g, exit k (``--k``), separate prefill
    fixed-g<g>-k<k>          same with exit layer k (k sweep)
    aspire                   ASPIRE per-request scheduling
    m1                       +M1      m1-linear   M1 with the linear (ASPIRE 3-parameter) cost model
    m2                       +M2 (fixed gamma = --base-gamma)
    m3                       +M3 (fixed gamma = --base-gamma)
    m1+m2  m1+m3  m2+m3  all
Metrics (per run JSON and CSV): decode-only throughput (ASPIRE protocol, prefill excluded), end-to-end
throughput (prefill included), speedup vs AR (same model / task / length / B / repetition), deep steps per
token, tokens per deep step, acceptance, exit layer timeline, planning time, TTFT, p99 TPOT, prefill share.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eesd"
DATASETS = ("cnn_dm_summarization", "xsum_summarization", "human_eval", "gsm8k")
MODELS = ("layerskip-llama3-1B", "layerskip-llama2-7B", "layerskip-llama3-8B")
BASE = ["ar", "fixed-g2", "fixed-g3", "fixed-g4", "fixed-g6", "fixed-g8", "aspire"]
MECH = ["m1", "m2", "m3", "m1+m2", "m1+m3", "m2+m3", "all", "m1-linear"]

PRESETS = {
    "smoke": dict(models=["layerskip-llama3-1B"], datasets=["gsm8k"], lengths=[256], batch_sizes=[4],
                  configs=["ar", "fixed-g4", "aspire", "m1", "m2", "m3", "all"], reps=1, max_eval=32),
    "phase0": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512],
                   batch_sizes=[1, 4, 8, 16, 32], configs=BASE, reps=1),
    "phase1": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512],
                   batch_sizes=[1, 4, 8, 16, 32], configs=["ar", "fixed-g4", "aspire", "m1", "m1-linear"], reps=1),
    "ksweep": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512], batch_sizes=[1, 4, 8, 16, 32],
                   configs=["ar"] + [f"fixed-g{g}-k{k}" for k in range(2, 9) for g in (3, 4)], reps=1),
    "phase2": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512],
                   batch_sizes=[1, 4, 8, 16, 32], configs=["ar", "fixed-g4", "m1", "m2", "m1+m2"], reps=1),
    "prefill-share": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[256, 512, 1024],
                          batch_sizes=[1, 4, 8, 16, 32], configs=["ar", "fixed-g4", "m1"], reps=1),
    "phase3": dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512],
                   batch_sizes=[1, 4, 8, 16, 32], configs=["ar", "fixed-g4", "m1", "m3", "m1+m3"], reps=1),
    "full": dict(models=list(MODELS), datasets=list(DATASETS), lengths=[256, 512, 1024], batch_sizes=[1, 4, 8, 16, 32],
                 configs=BASE + MECH, reps=3),
    "poisson": dict(models=["layerskip-llama3-8B"], datasets=["cnn_dm_summarization", "gsm8k"], lengths=[512],
                    batch_sizes=[32], configs=["ar", "fixed-g4", "aspire", "m1", "m3", "all"], reps=3,
                    poisson_loads=[0.5, 0.8, 1.0]),
}


# ------------------------------------------------------------------------------ configs

def engine_config(name: str, args):
    from solutions.engine import EngineConfig
    base = dict(k=args.k, d_max=args.d_max, sla_steps=args.sla_steps, probe_every=args.probe_every,
                k_cands=tuple(args.k_cands), switch_margin=args.switch_margin, max_wait=args.max_wait,
                prefill_chunk=args.prefill_chunk, bucketing=args.bucketing, gamma=args.base_gamma)
    cost = "roofline"
    if name == "ar":
        return EngineConfig(mode="ar", **base), cost
    if name.startswith("fixed-g"):
        parts = name[len("fixed-g"):].split("-k")
        base["gamma"] = int(parts[0])
        if len(parts) > 1:
            base["k"] = int(parts[1])
        return EngineConfig(sched="fixed", **base), cost
    if name == "aspire":
        return EngineConfig(sched="aspire", **base), cost
    if name == "m1-linear":
        return EngineConfig(sched="m1", **base), "linear"
    mech = {"m1": ("m1", "fixed", "separate"), "m2": ("fixed", "m2", "separate"), "m3": ("fixed", "fixed", "m3"),
            "m1+m2": ("m1", "m2", "separate"), "m1+m3": ("m1", "fixed", "m3"), "m2+m3": ("fixed", "m2", "m3"),
            "all": ("m1", "m2", "m3")}
    if name not in mech:
        raise ValueError(f"unknown config {name!r}")
    s, e, p = mech[name]
    return EngineConfig(sched=s, exit=e, prefill=p, **base), cost


# ------------------------------------------------------------------------------ workloads

def load_workload(tok, dataset: str, length: int, n: int, mode: str, seed: int):
    from common.data import DEFAULT_N_SHOT, Prompt, load_prompts
    kw = dict(seed=seed, n_shot=DEFAULT_N_SHOT.get(dataset, 0))
    try:
        if mode == "exact":
            ps = load_prompts(dataset, tok, n, truncate_to=length, **kw)
        else:   # cap: left-truncate longer prompts, keep shorter ones as they are
            ps = load_prompts(dataset, tok, n, **kw)
            ps = [p if len(p.tokens) <= length else Prompt(p.pid, p.tokens[:1] + p.tokens[len(p.tokens) - length + 1:],
                                                            p.reference) for p in ps]
    except ValueError:
        return []
    return [Prompt(f"{dataset}/{p.pid}", p.tokens, p.reference) for p in ps]


def run_dir(out: Path, model: str, dataset: str, length: int, B: int) -> Path:
    return out / "runs" / model / f"{dataset}_L{length}" / f"B{B}"


# ------------------------------------------------------------------------------ main loop

def run(args) -> None:
    import torch
    from common.cli import load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.memory import resolve_batch_sizes
    from common.utils import env_info, resolve_model_path, seed_everything
    from solutions.cost_model import load_cost_model
    from solutions.engine import EESDEngine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = list(iter_plan(args))
    if args.dry_run:
        for p in plan:
            print(p)
        print(f"{len(plan)} runs (before memory checks / skips)")
        return
    for model_name in args.models:
        calib = Path(args.calibration_dir) / f"{model_name}.json"
        if not calib.exists():
            raise SystemExit(f"missing {calib}: run  python -m solutions.calibrate --model {model_name}  first")
        seed_everything(args.seed)
        tok = load_tokenizer(resolve_model_path(model_name))
        stops = {ds: StopChecker(tok, STOP_STRINGS.get(ds, ())) for ds in DATASETS}
        mx_len = max(args.lengths)
        args.model, args.device, args.dtype = model_name, args.device, args.dtype
        model = load_model_from_args(args, max_seq=mx_len + args.max_new_tokens + args.d_max + 64)
        eos = eos_ids(tok, model.cfg)
        cms = {"roofline": load_cost_model(calib, model.cfg, "roofline")}
        try:
            cms["linear"] = load_cost_model(calib, model.cfg, "linear")
        except ValueError:
            pass
        env = env_info(model.device)
        for ds in args.datasets:
            for L in args.lengths:
                pool = load_workload(tok, ds, L, args.prompts_per_workload, args.length_mode, args.seed)
                if len(pool) < args.min_prompts:
                    note = {"model": model_name, "dataset": ds, "length": L, "prompts": len(pool),
                            "reason": f"fewer than {args.min_prompts} prompts of length {L} ({args.length_mode})"}
                    print(f"[ablation] skip {note}", flush=True)
                    (out / "skipped").mkdir(exist_ok=True)
                    (out / "skipped" / f"{model_name}_{ds}_L{L}.json").write_text(json.dumps(note))
                    continue
                max_p = max(len(p.tokens) for p in pool)
                seq = max_p + args.max_new_tokens + args.d_max + 40
                Bs, skipped = resolve_batch_sizes(",".join(map(str, args.batch_sizes)), model, seq, max_p, args.d_max + 1)
                for B in skipped:
                    print(f"[ablation] skip B={B} {model_name} {ds} L={L}: KV cache does not fit", flush=True)
                    (out / "skipped").mkdir(exist_ok=True)
                    (out / "skipped" / f"{model_name}_{ds}_L{L}_B{B}.json").write_text(json.dumps(
                        {"model": model_name, "dataset": ds, "length": L, "B": B, "reason": "KV cache does not fit"}))
                for B in Bs:
                    d = run_dir(out, model_name, ds, L, B)
                    d.mkdir(parents=True, exist_ok=True)
                    for rep in range(args.reps):
                        rng = np.random.default_rng(args.seed + 1000 * rep)
                        prompts = [pool[i] for i in rng.permutation(len(pool))]
                        for cname in args.configs:
                            f = d / f"{cname}_rep{rep}.json"
                            if f.exists():
                                continue
                            cfg, cost = engine_config(cname, args)
                            if cfg.k >= model.n_layers or cost not in cms:
                                continue
                            t0 = time.time()
                            try:
                                eng = EESDEngine(model, B, max_p, args.max_new_tokens, eos, cfg, cms[cost],
                                                 stop_for=lambda pid: stops.get(pid.split("/")[0]))
                                if args.poisson_loads:
                                    res = run_poisson_series(eng, prompts, args, max_p, B, rep)
                                else:
                                    res = eng.run(prompts, "cyclic", min_eval_per_slot=args.min_eval_per_slot,
                                                  max_eval=args.max_eval, keep_outputs=args.keep_outputs)
                            except torch.OutOfMemoryError:
                                model.free_cache()
                                torch.cuda.empty_cache()
                                (out / "skipped").mkdir(exist_ok=True)
                                (out / "skipped" / f"{model_name}_{ds}_L{L}_B{B}_{cname}.json").write_text(
                                    json.dumps({"reason": "OOM during run"}))
                                print(f"[ablation] OOM {model_name} {ds} L={L} B={B} {cname}", flush=True)
                                continue
                            res["run"] = {"model": model_name, "dataset": ds, "length": L, "B": B, "config": cname,
                                          "cost_model": cost, "rep": rep, "seed": args.seed + 1000 * rep,
                                          "prompts": len(pool), "max_new_tokens": args.max_new_tokens,
                                          "host": socket.gethostname(), "env": env, "argv": sys.argv,
                                          "calibration": str(calib), "wall_s": time.time() - t0}
                            f.write_text(json.dumps(res, default=str))
                            m = res["metrics"]
                            print(f"[ablation] {model_name} {ds} L={L} B={B:2d} {cname:12s} rep{rep}  "
                                  f"{m['throughput_tok_s'] or 0:8.1f} tok/s  e2e {m['e2e_throughput_tok_s'] or 0:8.1f}  "
                                  f"deep/tok {m['deep_steps_per_token']:.3f}  k={m['k_final']}  ({time.time() - t0:.0f}s)",
                                  flush=True)
                            model.free_cache()
        del model
        torch.cuda.empty_cache()
    summarize(out)


def run_poisson_series(eng, prompts, args, max_p, B, rep):
    """Poisson mode: the offered load is a multiple of this config's own saturated rate measured by a
    short closed-loop run (requests/s = load * saturated tok/s / mean output length)."""
    sat = eng.run(prompts, "cyclic", min_eval_per_slot=1.0, max_eval=max(2 * B, 16))
    gen = [c["generated"] for c in sat["completions"] if c["measured"]] or [128]
    rate_sat = (sat["metrics"]["e2e_throughput_tok_s"] or 1.0) / float(np.mean(gen))
    series = {"saturated": sat["metrics"]}
    from solutions.engine import EESDEngine
    for load in args.poisson_loads:
        e2 = EESDEngine(eng.m, B, max_p, args.max_new_tokens, eng.eos, eng.cfg, eng.cm, eng.stop_for)
        r = e2.run_poisson(prompts, load * rate_sat, args.poisson_requests, warm_requests=B, seed=rep)
        series[f"load{load}"] = r["metrics"]
    return {"metrics": series["saturated"], "poisson": series, "engine_config": sat["engine_config"], "completions": []}


def iter_plan(args):
    for m in args.models:
        for ds in args.datasets:
            for L in args.lengths:
                for B in args.batch_sizes:
                    for rep in range(args.reps):
                        for c in args.configs:
                            yield (m, ds, L, B, rep, c)


# ------------------------------------------------------------------------------ summary

def summarize(out: Path) -> None:
    rows, outputs = [], {}
    for f in sorted((out / "runs").rglob("*.json")):
        d = json.loads(f.read_text())
        r, m = d["run"], d["metrics"]
        toks = {c["pid"]: c["tokens"] for c in d.get("completions", []) if "tokens" in c}
        if toks:
            outputs[(r["model"], r["dataset"], r["length"], r["B"], r["rep"], r["config"])] = toks
        rows.append({**{k: r[k] for k in ("model", "dataset", "length", "B", "config", "cost_model", "rep")},
                     **{k: m.get(k) for k in ("throughput_tok_s", "e2e_throughput_tok_s", "deep_steps_per_token",
                                              "tokens_per_deep_step", "verifiers_per_deep_step", "acceptance_rate",
                                              "k_final", "plan_ms_per_step", "ttft_ms_mean", "ttft_ms_p99", "tpot_ms_mean",
                                              "tpot_ms_p99", "prefill_time_share", "piggyback_share_of_prefill_tokens",
                                              "m3_fallbacks", "probe_steps", "evaluated_completions", "stop_reason")},
                     "k_switches": max(len(m.get("k_timeline") or []) - 1, 0)})
        if "poisson" in d:
            for lab, pm in d["poisson"].items():
                if lab != "saturated":
                    rows[-1][f"{lab}_wall_tok_s"] = pm.get("wall_throughput_tok_s")
                    rows[-1][f"{lab}_ttft_ms_p99"] = pm.get("ttft_ms_p99")
    if not rows:
        print("[ablation] no runs yet")
        return
    key = lambda r: (r["model"], r["dataset"], r["length"], r["B"], r["rep"])  # noqa: E731
    ar = {key(r): r for r in rows if r["config"] == "ar"}
    for r in rows:
        a = ar.get(key(r))
        r["speedup_vs_ar"] = r["throughput_tok_s"] / a["throughput_tok_s"] if a and a["throughput_tok_s"] and r["throughput_tok_s"] else None
        r["e2e_speedup_vs_ar"] = (r["e2e_throughput_tok_s"] / a["e2e_throughput_tok_s"]
                                  if a and a["e2e_throughput_tok_s"] and r["e2e_throughput_tok_s"] else None)
    for r in rows:                         # bf16 token agreement with AR (runs made with --keep-outputs)
        mine = outputs.get(key(r) + (r["config"],))
        ref = outputs.get(key(r) + ("ar",))
        if mine and ref:
            ids = [p for p in mine if p in ref]
            r["identical_to_ar"] = sum(mine[p] == ref[p] for p in ids) / max(len(ids), 1)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # mean / std over repetitions
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["dataset"], r["length"], r["B"], r["config"]), []).append(r)
    agg = []
    metrics = ["throughput_tok_s", "e2e_throughput_tok_s", "speedup_vs_ar", "e2e_speedup_vs_ar", "deep_steps_per_token",
               "tokens_per_deep_step", "acceptance_rate", "plan_ms_per_step", "ttft_ms_mean", "tpot_ms_p99",
               "prefill_time_share", "piggyback_share_of_prefill_tokens", "k_switches", "identical_to_ar"]
    for g, rr in sorted(groups.items()):
        row = dict(zip(("model", "dataset", "length", "B", "config"), g), reps=len(rr))
        for mname in metrics:
            v = [x[mname] for x in rr if isinstance(x.get(mname), (int, float))]
            row[f"{mname}_mean"] = float(np.mean(v)) if v else None
            row[f"{mname}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else (0.0 if v else None)
        row["k_final"] = ",".join(str(x["k_final"]) for x in rr)
        agg.append(row)
    keys = list(dict.fromkeys(k for r in agg for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(agg)
    write_tables(out, agg)
    print(f"[ablation] {len(rows)} runs -> {out / 'runs.csv'}, {out / 'summary.csv'}, {out / 'SUMMARY.md'}")


def _fmt(m, s, f="{:.2f}"):
    if m is None:
        return "—"
    return f.format(m) + (f" ± {f.format(s)}" if s else "")


def write_tables(out: Path, agg: list) -> None:
    md = ["# EESD measurements (real GPU runs)", "",
          "Speedup = decode-only throughput (prefill excluded, ASPIRE protocol) / AR of the same model, task, prompt length, "
          "B and repetition; mean ± std over repetitions.  e2e includes prefill time.", ""]
    for (model, ds, L) in sorted({(r["model"], r["dataset"], r["length"]) for r in agg}):
        rr = [r for r in agg if (r["model"], r["dataset"], r["length"]) == (model, ds, L)]
        Bs = sorted({r["B"] for r in rr})
        cfgs = list(dict.fromkeys(r["config"] for r in rr))
        md += [f"## {model} · {ds} · prompt {L}", "",
               "| config | " + " | ".join(f"B={b} speedup" for b in Bs) + " | " + " | ".join(f"B={b} e2e" for b in Bs)
               + " | deep/token (B=max) | k | plan ms/step |",
               "|" + "---|" * (2 * len(Bs) + 4)]
        for c in cfgs:
            cells, cells2 = [], []
            for b in Bs:
                x = next((r for r in rr if r["config"] == c and r["B"] == b), None)
                cells.append(_fmt(x["speedup_vs_ar_mean"], x["speedup_vs_ar_std"]) if x else "—")
                cells2.append(_fmt(x["e2e_speedup_vs_ar_mean"], x["e2e_speedup_vs_ar_std"]) if x else "—")
            last = next((r for r in sorted(rr, key=lambda r: -r["B"]) if r["config"] == c), None)
            md.append(f"| {c} | " + " | ".join(cells) + " | " + " | ".join(cells2) + " | "
                      + (_fmt(last["deep_steps_per_token_mean"], None, "{:.4f}") if last else "—") + " | "
                      + (last["k_final"] if last else "—") + " | "
                      + (_fmt(last["plan_ms_per_step_mean"], None, "{:.3f}") if last else "—") + " |")
        md.append("")
    (out / "SUMMARY.md").write_text("\n".join(md))


# ------------------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--lengths", nargs="+", type=int)
    ap.add_argument("--batch-sizes", nargs="+", type=int)
    ap.add_argument("--configs", nargs="+")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--poisson-loads", nargs="+", type=float, default=None)
    ap.add_argument("--poisson-requests", type=int, default=96)
    ap.add_argument("--length-mode", choices=["exact", "cap"], default="exact",
                    help="exact: left-truncate to exactly L, drop shorter prompts; cap: truncate longer ones only")
    ap.add_argument("--prompts-per-workload", type=int, default=64)
    ap.add_argument("--min-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--base-gamma", type=int, default=4, help="gamma of the fixed scheduler under +M2 / +M3")
    ap.add_argument("--d-max", type=int, default=16)
    ap.add_argument("--sla-steps", type=int, default=32)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--k-cands", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--switch-margin", type=float, default=0.05)
    ap.add_argument("--max-wait", type=int, default=3)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--bucketing", choices=["exact", "pad"], default="exact")
    ap.add_argument("--min-eval-per-slot", type=float, default=2.0)
    ap.add_argument("--max-eval", type=int, default=256)
    ap.add_argument("--keep-outputs", action="store_true", help="store generated tokens (token agreement with AR)")
    ap.add_argument("--calibration-dir", default=str(RESULTS / "calibration"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    preset = PRESETS.get(args.preset, {})
    for k, v in preset.items():
        if getattr(args, k, None) in (None, []):
            setattr(args, k, v)
    defaults = dict(models=["layerskip-llama3-8B"], datasets=list(DATASETS), lengths=[512], batch_sizes=[1, 4, 8, 16, 32],
                    configs=BASE, reps=1)
    for k, v in defaults.items():
        if getattr(args, k, None) in (None, []):
            setattr(args, k, v)
    if preset.get("max_eval") and args.max_eval == 256:
        args.max_eval = preset["max_eval"]
    args.out = args.out or str(RESULTS / (args.preset or "ablation"))
    if args.summarize:
        summarize(Path(args.out))
        return
    run(args)


if __name__ == "__main__":
    main()
