"""Losslessness on a real checkpoint (GPU): every switch combination vs AR, greedy, static mode.

    python -m solutions.check_lossless --model layerskip-llama3-1B --dtype fp32     # must be 100% identical
    python -m solutions.check_lossless --model layerskip-llama3-8B --dtype bf16     # reports the token agreement rate

fp32 removes the batch-shape dependence of bf16 GEMMs, so every combination must reproduce AR token
for token.  In bf16 the AR baseline itself changes with the batch shape; the script therefore also
reports AR(B) vs AR(1) as the floor.  Output: results/eesd/lossless/<model>_<dtype>.json + .md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eesd"


def agreement(out: dict, ref: dict) -> tuple[float, float]:
    ids = [p for p in ref if p in out]
    same = [out[p] == ref[p] for p in ids]
    pref = []
    for p in ids:
        a, b = out[p], ref[p]
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        pref.append(i / max(len(b), 1))
    return (sum(same) / max(len(same), 1), sum(pref) / max(len(pref), 1))


def main() -> None:
    from common.cli import add_model_args, load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.utils import resolve_model_path, seed_everything
    from solutions.cost_model import load_cost_model
    from solutions.engine import EESDEngine, EngineConfig
    from solutions.run_ablation import load_workload

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "cnn_dm_summarization"])
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--probe-every", type=int, default=4, help="small so that M2 probes / switches happen")
    ap.add_argument("--calibration-dir", default=str(RESULTS / "calibration"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    name = Path(args.model).name
    tok = load_tokenizer(resolve_model_path(args.model))
    prompts = [p for ds in args.datasets for p in load_workload(tok, ds, args.length, args.prompts, "cap", 42)]
    stops = {ds: StopChecker(tok, STOP_STRINGS.get(ds, ())) for ds in args.datasets}
    model = load_model_from_args(args, max_seq=args.length + args.max_new_tokens + 64)
    eos = eos_ids(tok, model.cfg)
    calib = Path(args.calibration_dir) / f"{name}.json"
    cms = {"roofline": load_cost_model(calib, model.cfg, "roofline")}
    try:
        cms["linear"] = load_cost_model(calib, model.cfg, "linear")
    except ValueError:
        pass
    maxp = max(len(p.tokens) for p in prompts)
    stop_for = lambda pid: stops.get(pid.split("/")[0])  # noqa: E731

    def run(cfg, cost, B):
        eng = EESDEngine(model, B, maxp, args.max_new_tokens, eos, cfg, cms[cost], stop_for)
        res = eng.run(prompts, "static", keep_outputs=True)
        model.free_cache()
        return {c["pid"]: c["tokens"] for c in res["completions"]}, res["metrics"]

    ref, _ = run(EngineConfig(mode="ar"), "roofline", 1)
    rows = []
    ar_b, _ = run(EngineConfig(mode="ar"), "roofline", args.batch_size)
    s, p = agreement(ar_b, ref)
    rows.append({"config": f"ar B={args.batch_size} (floor)", "identical": s, "prefix_agreement": p})
    for sched, ex, pf, cost in itertools.product(["fixed", "aspire", "m1"], ["fixed", "m2"], ["separate", "m3"],
                                                 list(cms)):
        cfg = EngineConfig(sched=sched, exit=ex, prefill=pf, k=args.k, probe_every=args.probe_every)
        out, m = run(cfg, cost, args.batch_size)
        s, p = agreement(out, ref)
        rows.append({"config": f"{cfg.label()} [{cost}]", "identical": s, "prefix_agreement": p,
                     "k_switches": max(len(m["k_timeline"]) - 1, 0), "piggybacked": m["prefill_tokens_piggyback"]})
        print(f"[lossless] {rows[-1]['config']:40s} identical {100 * s:5.1f}%  prefix {100 * p:5.1f}%", flush=True)
    out_dir = RESULTS / "lossless"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{args.dtype}"
    (out_dir / f"{tag}.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    md = [f"# Losslessness vs AR(B=1): {name}, {args.dtype}, {len(prompts)} prompts, max_new {args.max_new_tokens}", "",
          "| config | identical outputs | mean prefix agreement | k switches | piggybacked prompt tokens |",
          "|---|---|---|---|---|"]
    md += [f"| {r['config']} | {100 * r['identical']:.1f}% | {100 * r['prefix_agreement']:.1f}% | {r.get('k_switches', '')} | "
           f"{r.get('piggybacked', '')} |" for r in rows]
    (out_dir / f"{tag}.md").write_text("\n".join(md) + "\n")
    bad = [r for r in rows[1:] if r["identical"] < 1.0]
    print(f"[lossless] {len(rows) - 1} combinations, {len(bad)} not identical to AR -> {out_dir / tag}.md")
    if args.dtype == "fp32" and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
