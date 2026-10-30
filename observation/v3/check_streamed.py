"""The streamed path (observation/v3/streamed.py) must give the same tokens as the normal path.

    python observation/v3/check_streamed.py --model layerskip-llama3-1B
    -> results/observe-4060/check_streamed_<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.engine import ARBatchEngine, RunConfig
    from common.model import LayerSkipLlama
    from common.utils import env_info, resolve_model_path
    from observation.v3.streamed import StreamedLlama
    from solutions.run_ablation import load_workload

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="layerskip-llama3-1B")
    ap.add_argument("--datasets", nargs="+", default=["cnn_dm_summarization", "gsm8k"])
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "results" / "observe-4060"))
    args = ap.parse_args()
    path = resolve_model_path(args.model)
    tok = load_tokenizer(path)
    max_seq = 512 + args.max_new_tokens + 64
    outs = {}
    for kind in ("normal", "streamed"):
        m = (LayerSkipLlama.from_pretrained(path, max_seq=max_seq) if kind == "normal"
             else StreamedLlama.load(path, max_seq=max_seq))
        for ds in args.datasets:
            pool = load_workload(tok, ds, 512, args.prompts, "exact", 42)
            eng = ARBatchEngine(m, args.batch_size, 512, args.max_new_tokens, eos_ids(tok, m.cfg))
            eng.stop_checker = StopChecker(tok, STOP_STRINGS.get(ds, ()))
            res = eng.run(pool, RunConfig(mode="static", keep_outputs=True))
            outs[(kind, ds)] = {c["pid"]: c["tokens"] for c in res["completions"]}
            m.free_cache()
        del m
        torch.cuda.empty_cache()
    report = {"model": args.model, "env": env_info(torch.device("cuda")), "datasets": {}}
    for ds in args.datasets:
        a, b = outs[("normal", ds)], outs[("streamed", ds)]
        same = sum(a[p] == b.get(p) for p in a)
        toks = sum(len(v) for v in a.values())
        agree = sum(sum(x == y for x, y in zip(a[p], b.get(p, []))) for p in a)
        report["datasets"][ds] = {"requests": len(a), "identical_requests": same, "tokens": toks, "agreeing_tokens": agree}
        print(f"[check_streamed] {ds}: {same}/{len(a)} requests identical, {agree}/{toks} tokens", flush=True)
    out = Path(args.out) / f"check_streamed_{Path(args.model).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
