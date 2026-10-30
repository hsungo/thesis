"""Time budget and waste of real 1B runs (A, B, D, E-tail).  All times measured on this GPU.

    python observation/v3/budget.py run      # cyclic runs: tasks x B x configs x {eager, graphs} x reps
    python observation/v3/budget.py tails    # fixed-g4 B=16 with draft tokens (E: rejected tails), not timed
    python observation/v3/budget.py analyze  # CSVs + stacked bar charts

Every run: EngineConfig(segment_timing=True, log_rounds=True) - CUDA events between the segments of every step
and a host-side log of every step (token counts, attention lengths, drafters, verifications).  While a run
executes, nvidia-smi samples the SM clock, memory clock, temperature and power every 500 ms.
After each run the KV-read slope is measured on the same engine: one segment (same rows, same T) at L = 64 and
at the largest L of the run, 50 times each (median); attention/KV time of a step = max(slope, 0) * (L_step - 64)
(a launch-bound eager segment can measure a slope of ~0 or slightly < 0: the raw slope is kept in the run JSON).
Raw: results/observe-4060/runs/<task>/B<B>/<config>_<mode>_rep<r>.json (not in git).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "observe-4060"
MODEL = "layerskip-llama3-1B"
CONFIGS = ["ar", "fixed-g2", "fixed-g3", "fixed-g4", "fixed-g6", "fixed-g8", "m1"]
TASKS = ["cnn_dm_summarization", "gsm8k"]


# ------------------------------------------------------------------------------ GPU telemetry

class Telemetry:
    def __init__(self):
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                                   "--format=csv,noheader,nounits", "-lms", "500"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def stop(self) -> dict:
        self.p.terminate()
        out, _ = self.p.communicate(timeout=10)
        rows = []
        for line in out.splitlines():
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass
        if not rows:
            return {}
        a = np.array(rows)
        return {"samples": len(a), "sm_clock_mhz_median": float(np.median(a[:, 0])), "sm_clock_mhz_min": float(a[:, 0].min()),
                "mem_clock_mhz_median": float(np.median(a[:, 1])), "temp_c_median": float(np.median(a[:, 2])),
                "temp_c_max": float(a[:, 2].max()), "power_w_median": float(np.median(a[:, 3]))}


# ------------------------------------------------------------------------------ KV slope

def _time(fn, dev, n=50, warm=5) -> float:
    import torch
    ts = []
    for i in range(warm + n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        if i >= warm:
            ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def kv_slopes(eng, log: list) -> dict:
    """ms per cache position (per segment kind and T) from two measured L points, same engine and rows."""
    import torch
    from solutions.graphs import round_up
    from solutions.packed import Block, build_packed, packed_layers
    m, B, k, L, dev = eng.m, eng.B, eng.k, eng.L, eng.dev
    meas = [e for e in log if e["measured"]]
    segs = {}
    for e in meas:
        if e["kind"] == "ar":
            segs.setdefault(("ar", 1), []).append(e["L_sh"])
        else:
            segs.setdefault(("sh", 1), []).append(e["L_sh"])
            if e["kind"] != "shallow" and e["verifiers"]:
                T = (e["pad_dp"] // B) if e["graph"] else max(v["d"] for v in e["verifiers"]) + 1
                segs.setdefault(("dp", T), []).append(e["L_dp"])
    out = {}
    lo = 64
    with torch.inference_mode():
        for (seg, T), Ls in segs.items():
            hi = int(max(Ls))
            if hi <= lo:
                continue
            a, b = {"ar": (0, L), "sh": (0, k), "dp": (k, L)}[seg]
            if eng.gr is not None:
                n_rows = B                                    # graphs always run every row
            elif seg == "dp":
                n_rows = int(np.median([len(e["verifiers"]) for e in meas if e["kind"] != "shallow" and e.get("verifiers")]))
            else:
                n_rows = int(np.median([e["B_dec"] for e in meas]))
            t = {}
            for Lx in (lo, hi):
                if eng.gr is not None:
                    gr = eng.gr
                    gr.init_buffers([T])                     # scratch rows/positions: every row reads Lx keys
                    src = "exh" if seg == "dp" else "seq"
                    t[Lx] = _time(lambda: gr.layers(a, b, T, Lx, src, seg == "sh"), dev)
                else:
                    pos0 = Lx - T
                    pb = build_packed(m, [Block(i, pos0, T) for i in range(n_rows)], "exact")
                    x = torch.randn(n_rows * T, m.cfg.dim, dtype=m.dtype, device=dev)
                    t[Lx] = _time(lambda: packed_layers(m, x, pb, a, b), dev)
            out[f"{seg}:{T}"] = {"slope_ms_per_pos": (t[hi] - t[lo]) / (hi - lo), "t_lo": t[lo], "t_hi": t[hi],
                                 "L_lo": lo, "L_hi": hi, "rows": n_rows}
    return out


# ------------------------------------------------------------------------------ run

def cmd_run(args) -> None:
    import torch
    from common.cli import load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.utils import env_info, git_commit, resolve_model_path, seed_everything
    from solutions.cost_model import load_cost_model
    from solutions.engine import EESDEngine
    from solutions.feasibility import engine_args
    from solutions.run_ablation import engine_config, load_workload
    seed_everything(42)
    tok = load_tokenizer(resolve_model_path(MODEL))
    model = load_model_from_args(SimpleNamespace(model=MODEL, device=None, dtype="bf16"), max_seq=512 + 256 + 16 + 64)
    eos = eos_ids(tok, model.cfg)
    calib = ROOT / "results" / "eesd" / "calibration" / f"{MODEL}.json"
    cm = load_cost_model(calib, model.cfg, "roofline")
    env = env_info(model.device)
    pools = {ds: load_workload(tok, ds, 512, 64, "exact", 42) for ds in args.tasks}
    stops = {ds: StopChecker(tok, STOP_STRINGS.get(ds, ())) for ds in args.tasks}
    plan = [(rep, ds, B, c, g) for rep in range(args.reps) for ds in args.tasks for B in args.batch_sizes
            for c in args.configs for g in args.modes]
    print(f"[budget] {len(plan)} runs", flush=True)
    for i, (rep, ds, B, cname, mode) in enumerate(plan):
        f = Path(args.out) / "runs" / ds / f"B{B}" / f"{cname}_{mode}_rep{rep}.json"
        if f.exists():
            continue
        pool = pools[ds]
        prompts = [pool[j] for j in np.random.default_rng(42 + 1000 * rep).permutation(len(pool))]
        cfg, _ = engine_config(cname, engine_args(3))
        cfg = dataclasses.replace(cfg, log_rounds=True, log_drafts=False, segment_timing=True, graphs=mode == "graph")
        t0 = time.time()
        eng = EESDEngine(model, B, max(len(p.tokens) for p in pool), 256, eos, cfg, cm,
                         stop_for=lambda pid, ds=ds: stops[ds])
        tel = Telemetry()
        res = eng.run(prompts, "cyclic", max_eval=args.max_eval)
        res["telemetry"] = tel.stop()
        res["kv_slopes"] = kv_slopes(eng, res["round_log"])
        res["run"] = {"model": MODEL, "dataset": ds, "B": B, "config": cname, "mode": mode, "rep": rep,
                      "env": env, "git_commit": git_commit(), "argv": sys.argv, "calibration": str(calib),
                      "wall_s": time.time() - t0}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, default=str))
        model.free_cache()
        del eng
        torch.cuda.empty_cache()
        mtr = res["metrics"]
        print(f"[budget] {i + 1}/{len(plan)} {ds[:5]} B={B:2d} {cname:9s} {mode:5s} rep{rep}: {mtr['wall_throughput_tok_s']:7.1f} tok/s "
              f"wall, graph steps {mtr['graph_step_share']:.0%}, {res['telemetry'].get('sm_clock_mhz_median', 0):.0f} MHz "
              f"{res['telemetry'].get('temp_c_max', 0):.0f} C ({time.time() - t0:.0f}s)", flush=True)


def cmd_tails(args) -> None:
    """fixed-g4, B = 16, draft tokens logged (E: rejected tails).  Wall clock not used."""
    from solutions.feasibility import cmd_rounds
    a = SimpleNamespace(models=[MODEL], tasks=args.tail_tasks, out=str(Path(args.out) / "tails"), expect_gpu="",
                        allow_other_gpu=True, k=3, length=512, max_new_tokens=256, prompts_per_workload=64, seed=42,
                        round_batch_sizes=[16], configs=["fixed-g4"], max_eval=args.max_eval)
    cal = Path(a.out) / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    src = ROOT / "results" / "eesd" / "calibration" / f"{MODEL}.json"
    (cal / src.name).write_text(src.read_text())
    cmd_rounds(a)


# ------------------------------------------------------------------------------ analyze

COMP = ["sh_layers", "sh_attn_kv", "sh_head", "dp_layers", "dp_attn_kv", "dp_head", "accept_readback",
        "metadata_upload", "host_other", "prefill"]
COMP_ZH = {"sh_layers": "淺層 k 層（不含 KV 讀取）", "sh_attn_kv": "淺層 attention/KV 讀取", "sh_head": "淺層 LM head",
           "dp_layers": "深層 L−k 層（AR：全部 L 層；不含 KV 讀取）", "dp_attn_kv": "深層 attention/KV 讀取",
           "dp_head": "深層 LM head（AR：LM head）",
           "accept_readback": "接受判斷與讀回", "metadata_upload": "metadata 與上傳", "host_other": "host 開銷（其他）",
           "prefill": "prefill"}


def budget_of_run(res: dict) -> dict:
    """ms per committed token of every component over the measured window (sums to wall ms per token)."""
    m, log = res["metrics"], res["round_log"]
    tok = m["measured_tokens"]
    slopes = res["kv_slopes"]
    B = res["run"]["B"]
    c = {k: 0.0 for k in COMP}
    up_gpu = 0.0

    def slope(key):
        return max(slopes.get(key, {}).get("slope_ms_per_pos", 0.0), 0.0)
    for e in log:
        if not e["measured"]:
            continue
        sg = e.get("seg_ms", {})
        if e["kind"] == "ar":
            kv = slope("ar:1") * max(e["L_sh"] - 64, 0)
            c["dp_attn_kv"] += kv
            c["dp_layers"] += sg.get("layers", 0.0) - kv
            c["dp_head"] += sg.get("head", 0.0)
            up_gpu += sg.get("upload", 0.0)
            continue
        kv = slope("sh:1") * max(e["L_sh"] - 64, 0)
        c["sh_attn_kv"] += kv
        c["sh_layers"] += sg.get("sh_layers", 0.0) - kv
        c["sh_head"] += sg.get("sh_head", 0.0)
        up_gpu += sg.get("upload", 0.0) + sg.get("dp_meta", 0.0)
        if e["kind"] != "shallow" and e["verifiers"]:
            T = (e["pad_dp"] // B) if e["graph"] else max(v["d"] for v in e["verifiers"]) + 1
            kvd = slope(f"dp:{T}") * max(e["L_dp"] - 64, 0)
            c["dp_attn_kv"] += kvd
            c["dp_layers"] += sg.get("dp_layers", 0.0) - kvd
            c["dp_head"] += sg.get("dp_head", 0.0)
            c["accept_readback"] += sg.get("accept", 0.0)
    gpu = m["decode_ms"] + m["prefill_ms"]
    host = 1000 * m["wall_s"] - gpu
    meta_host = m["meta_host_ms"]
    c["metadata_upload"] = up_gpu + min(meta_host, max(host, 0.0))
    c["host_other"] = max(host - meta_host, 0.0)
    c["prefill"] = m["prefill_ms"]
    out = {k: v / tok for k, v in c.items()}
    out["wall_ms_per_token"] = 1000 * m["wall_s"] / tok
    out["gpu_ms_per_token"] = gpu / tok
    out["unattributed_gpu_ms_per_token"] = (m["decode_ms"] - sum(c[k] for k in COMP[:7]) - up_gpu) / tok
    out["wall_tok_s"] = m["wall_throughput_tok_s"]
    out["plan_host_ms_per_token"] = m["plan_host_ms"] / tok
    return out


def waste_of_run(res: dict) -> dict:
    """B: shallow time on drafts later rejected, rounds stopped with every draft accepted, rejected share of deep
    tokens (and of deep time), shallow steps per round."""
    log = [e for e in res["round_log"]]
    pending = {}                         # uid -> list of (step index, cost ms of producing that draft)
    sh_total = sh_wasted = 0.0
    dp_tok = dp_rej = 0
    dp_ms = dp_rej_ms = 0.0
    rounds = full = 0
    d_sum = 0
    for e in log:
        if e["kind"] == "ar":
            continue
        sg = e.get("seg_ms", {})
        sh_ms = sg.get("sh_layers", 0.0) + sg.get("sh_head", 0.0)
        n_tok = max(e["n_sh"], 1)
        per = sh_ms / n_tok
        if e["measured"]:
            sh_total += sh_ms
        for uid, j in e["drafters"]:
            pending.setdefault(uid, []).append((j, per, e["measured"]))
        if e["kind"] == "shallow":
            continue
        dpm = sg.get("dp_layers", 0.0) + sg.get("dp_head", 0.0)
        ntok_dp = max(e["n_dp"], 1)
        for v in e["verifiers"]:
            d, a = v["d"], v["a"]
            items = pending.pop(v["uid"], [])
            # the verifier's newest draft also ran through [0, k) in this step's shallow segment
            items.append((d, per, e["measured"]))
            for (j, cost, meas) in items:
                if meas and j > a:
                    sh_wasted += cost
            if e["measured"]:
                rounds += 1
                full += int(a == d)
                d_sum += d
                dp_tok += d + 1
                dp_rej += d - a
                dp_ms += dpm * (d + 1) / ntok_dp
                dp_rej_ms += dpm * (d - a) / ntok_dp
    deep_steps = [e for e in log if e["measured"] and e["kind"] not in ("shallow", "ar")]
    sh_between = len([e for e in log if e["measured"] and e["kind"] == "shallow"]) / max(len(deep_steps), 1)
    return {"shallow_time_on_rejected_drafts": sh_wasted / max(sh_total, 1e-9), "full_accept_round_share": full / max(rounds, 1),
            "rejected_share_of_deep_tokens": dp_rej / max(dp_tok, 1), "rejected_share_of_deep_time": dp_rej_ms / max(dp_ms, 1e-9),
            "shallow_steps_per_round_slot": d_sum / max(rounds, 1), "shallow_steps_between_deep_steps": sh_between,
            "rounds": rounds}


def slack_of_run(res: dict, knee: float) -> list:
    rows = []
    for seg, key in (("shallow_segment", "n_sh"), ("deep_segment", "n_dp")):
        n = np.array([e[key] for e in res["round_log"] if e["measured"] and e["kind"] != "ar"
                      and (seg == "shallow_segment" or e["kind"] != "shallow")], dtype=float)
        if len(n):
            rows.append({"segment": seg, "steps": len(n), "n_median": pct(n, 50), "n_p10": pct(n, 10), "n_p90": pct(n, 90),
                         "slack_median": knee - pct(n, 50), "slack_p10": knee - pct(n, 90), "slack_p90": knee - pct(n, 10),
                         "share_over_knee": float((n > knee).mean())})
    return rows


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def med_iqr(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def cmd_analyze(args) -> None:
    from solutions.feasibility import extension_from_log, write_csv
    out = Path(args.out)
    knee = json.loads((ROOT / "results" / "eesd" / "calibration" / f"{MODEL}.json").read_text())["gpu"]["knee_measured"]
    runs = {}
    for f in sorted((out / "runs").glob("*/B*/*.json")):
        res = json.loads(f.read_text())
        r = res["run"]
        runs.setdefault((r["dataset"], r["B"], r["config"], r["mode"]), []).append(res)
    rows_all, rows_sum, waste, slack, tele = [], [], [], [], []
    for (ds, B, cname, mode), rs in sorted(runs.items()):
        bud = [budget_of_run(r) for r in rs]
        for r, b in zip(rs, bud):
            rows_all.append({"task": ds, "B": B, "config": cname, "mode": mode, "rep": r["run"]["rep"], **b,
                             **{f"tel_{k}": v for k, v in r.get("telemetry", {}).items()}})
            tele.append(r.get("telemetry", {}))
        row = {"task": ds, "B": B, "config": cname, "mode": mode, "reps": len(rs)}
        for key in COMP + ["wall_ms_per_token", "gpu_ms_per_token", "unattributed_gpu_ms_per_token", "wall_tok_s"]:
            m_, q1, q3 = med_iqr([b[key] for b in bud])
            row[key], row[f"{key}_q1"], row[f"{key}_q3"] = m_, q1, q3
        rows_sum.append(row)
        if cname != "ar":
            ws = [waste_of_run(r) for r in rs]
            wrow = {"task": ds, "B": B, "config": cname, "mode": mode}
            for key in ws[0]:
                wrow[key] = med_iqr([w[key] for w in ws])[0]
            waste.append(wrow)
            for srow in slack_of_run(rs[0], knee):
                slack.append({"task": ds, "B": B, "config": cname, "mode": mode, "knee_measured": knee, **srow})
    # best gamma per (task, B, mode) by median wall throughput
    for (ds, B, mode) in sorted({(r["task"], r["B"], r["mode"]) for r in rows_sum}):
        fx = [r for r in rows_sum if (r["task"], r["B"], r["mode"]) == (ds, B, mode) and r["config"].startswith("fixed")]
        if fx:
            best = max(fx, key=lambda r: r["wall_tok_s"])
            for r in fx:
                r["best_gamma"] = r is best
    write_csv(out / "A_budget_runs.csv", rows_all)
    write_csv(out / "A_budget.csv", rows_sum)
    write_csv(out / "B_waste_1B.csv", waste)
    write_csv(out / "D_slack.csv", slack)
    # E: rejected tails from the tail runs
    tails = []
    for f in sorted((out / "tails" / "runs" / MODEL).glob("*/B16_fixed-g4.json")):
        run = json.loads(f.read_text())
        t = extension_from_log(run, [4, 8, 16])
        tails.append({"model": MODEL, "task": f.parent.name, "source": "rejected_tail", "rounds": t["rounds"],
                      "tail_rounds_share": t["tails"] / max(t["rounds"], 1), "tail_alignment_mean": t["tail_m"] / max(t["tails"], 1),
                      "extra_per_round": t["B_extra"] / max(t["rounds"], 1), "ngram_extra_per_round_G8_log": t[("A", 8)] / max(t["rounds"], 1)})
    write_csv(out / "E_tail.csv", tails)
    plot_budget(rows_sum, out)


def plot_budget(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#4C78A8", "#9ECAE9", "#F58518", "#54A24B", "#B5D99C", "#E45756", "#FF9DA6", "#B279A2", "#9D755D", "#BAB0AC"]
    for mode in ("eager", "graph"):
        tasks = sorted({r["task"] for r in rows})
        fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), sharey=False, squeeze=False)
        for ax, ds in zip(axes[0], tasks):
            sel = [r for r in rows if r["task"] == ds and r["mode"] == mode]
            Bs = sorted({r["B"] for r in sel})
            labels, bars = [], []
            for B in Bs:
                for name in ("ar", "best-γ", "m1"):
                    if name == "best-γ":
                        cand = [r for r in sel if r["B"] == B and r.get("best_gamma")]
                    else:
                        cand = [r for r in sel if r["B"] == B and r["config"] == name]
                    if cand:
                        r = cand[0]
                        g = r["config"].replace("fixed-g", "γ") if name == "best-γ" else name.upper()
                        labels.append(f"B{B}\n{g}")
                        bars.append([r[c] for c in COMP])
            if not bars:
                continue
            bars = np.array(bars)
            x = np.arange(len(bars))
            bottom = np.zeros(len(bars))
            for ci, c in enumerate(COMP):
                ax.bar(x, bars[:, ci], bottom=bottom, color=colors[ci], label=COMP_ZH[c], width=0.75)
                bottom += bars[:, ci]
            ax.set_xticks(x, labels, fontsize=7)
            ax.set_ylabel("ms / 確認 token（牆鐘）")
            ax.set_title(f"{ds} · {'CUDA graph' if mode == 'graph' else '無 graph'}")
        axes[0][-1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / f"A_budget_{mode}.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tails", "analyze"])
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--tail-tasks", nargs="+", default=["cnn_dm_summarization", "xsum_summarization", "human_eval", "gsm8k"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--modes", nargs="+", default=["eager", "graph"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-eval", type=int, default=64)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {"run": cmd_run, "tails": cmd_tails, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
