"""Check that a speculative method reproduces AR outputs token by token (plan sec.6.5).

Both files come from `benchmark.py --mode static --save-outputs` (greedy, same dataset
settings).  Requests are matched by prompt id.

    python evaluation/check_lossless.py --reference results/static/autoregressive-B16-outputs.jsonl \
        --candidate results/static/sync_layerskip-k8-g4-B16-outputs.jsonl

Exact equality is expected in fp32.  In bf16, a different batch composition / kernel
shape can flip the argmax at a near-tie; the report therefore also gives the first
divergence position and the share of requests whose first N tokens agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["pid"])] = r["tokens"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--prefix", type=int, default=64, help="also report agreement of the first N tokens")
    args = ap.parse_args()
    ref, cand = load(args.reference), load(args.candidate)
    common = sorted(set(ref) & set(cand))
    if not common:
        print("no common prompt ids")
        return 2
    exact, first_div, prefix_ok = 0, [], 0
    for pid in common:
        a, b = ref[pid], cand[pid]
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), None)
        if div is None and len(a) == len(b):
            exact += 1
        else:
            first_div.append(div if div is not None else n)
        if a[:args.prefix] == b[:args.prefix]:
            prefix_ok += 1
    print(f"requests compared   : {len(common)} (missing in candidate: {len(set(ref) - set(cand))})")
    print(f"identical outputs   : {exact}/{len(common)} ({100 * exact / len(common):.1f}%)")
    print(f"first {args.prefix} tokens equal: {prefix_ok}/{len(common)}")
    if first_div:
        print(f"first divergence    : median {int(np.median(first_div))}, min {min(first_div)} (token index)")
    return 0 if exact == len(common) else 1


if __name__ == "__main__":
    sys.exit(main())
