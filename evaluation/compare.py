"""Aggregate benchmark JSONs into one table with speedup vs AR.

Runs are matched to the AR run with the same model, dataset, prompt-length setting,
generation length and batch size (plan sec.6.5: "相對 AR 的加速比").

    python evaluation/compare.py evaluation/results/cnn_dm [more dirs ...] --out evaluation/results/table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setting_key(cfg: dict) -> tuple:
    return (cfg.get("model"), cfg.get("dataset"), cfg.get("n_shot"), cfg.get("truncate_to"),
            cfg.get("min_prompt_tokens"), cfg.get("max_prompt_tokens"), cfg.get("max_new_tokens"),
            cfg.get("ignore_eos"), cfg.get("temperature"), cfg.get("batch_size"))


def load_runs(dirs: list[str]) -> list[dict]:
    runs = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                r = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if "method" in r and "metrics" in r and "config" in r:
                r["_path"] = str(p)
                runs.append(r)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args()

    runs = load_runs(args.dirs)
    ar = {}
    for r in runs:
        if r["method"] == "autoregressive":
            ar.setdefault(setting_key(r["config"]), []).append(r["metrics"]["throughput_tok_s"])
    rows = []
    for r in runs:
        cfg, m = r["config"], r["metrics"]
        ref = ar.get(setting_key(cfg))
        thr = m["throughput_tok_s"] or 0.0
        ref_thr = sum(ref) / len(ref) if ref else None
        spec = m.get("spec", {})
        rows.append({
            "dataset": cfg.get("dataset"), "bucket": cfg.get("truncate_to") or "natural",
            "B": cfg.get("batch_size"), "method": r["method"],
            "params": json.dumps(r.get("method_params", {}), sort_keys=True),
            "throughput_tok_s": round(thr, 2),
            "speedup_vs_ar": round(thr / ref_thr, 3) if ref_thr else "",
            "acceptance": round(spec["acceptance_rate"], 3) if spec else "",
            "tok_per_verify": round(spec["tokens_per_verify"], 2) if spec else "",
            "tpot_ms": round(m.get("mean_tpot_ms") or 0.0, 2),
            "completions": m.get("evaluated_completions"),
            "rep": cfg.get("repetition"), "file": r["_path"],
        })
    rows.sort(key=lambda x: (str(x["dataset"]), str(x["bucket"]), x["B"] or 0, x["method"], x["params"]))
    if not rows:
        raise SystemExit("no benchmark JSON found")
    cols = ["dataset", "bucket", "B", "method", "params", "throughput_tok_s", "speedup_vs_ar",
            "acceptance", "tok_per_verify", "tpot_ms"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for x in rows:
        print("| " + " | ".join(str(x[c]) for c in cols) + " |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
