"""AR reference outputs for the m1-oracle diagnostic (GPU).

For every (model, task, prompt length) of an ablation this runs plain AR decoding once over the same
prompt pool that run_ablation uses and records, at every generated position, the AR token and whether
the exit-layer-k draft (shared head on the layer-k hidden state) equals it.  With greedy decoding the
round that starts with c committed tokens and d drafts commits min(run[c], d) + 1 tokens, run[c] =
number of consecutive agreeing positions from c - that is the "true acceptance" m1-oracle uses.

    python -m solutions.make_ref --model layerskip-llama3-8B --datasets cnn_dm_summarization gsm8k --lengths 512
    -> results/eesd/ref/<model>/<dataset>_L<length>.npz   (exit layers 2..8 stored, any of them usable)

Use the same --max-new-tokens / --prompts-per-workload / --length-mode / --seed as the ablation run.

Feasibility analysis (solutions/feasibility.py) additionally needs, per position, the top-64 tokens and logits of
exit layer k and the top-32 tokens of the full model, the prompt tokens, and token frequencies from prompts outside
the evaluation pool:
    python -m solutions.make_ref --model ... --topk-layer 3 --freq-prompts 64 --length-mode auto --out results/feasibility/ref
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eesd"


def ref_path(root: Path | str, model: str, dataset: str, length: int) -> Path:
    return Path(root) / Path(model).name / f"{dataset}_L{length}.npz"


def load_ref(path: Path | str, k: int) -> dict:
    """pid -> (AR tokens as a list, run-length array of exit-k agreement)."""
    from common.traces import load_traces, run_lengths
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing AR reference {p}: create it with  python -m solutions.make_ref ...")
    ts = load_traces(p, min_len=1)
    if k not in ts.exit_layers:
        raise ValueError(f"{p} has exit layers {ts.exit_layers}, not {k}")
    col = ts.col(k)
    return {t.pid: (t.tokens.tolist(), run_lengths(t.agree[:, col])) for t in ts.traces}


def length_mode(mode: str, dataset: str) -> str:
    return ("cap" if dataset == "human_eval" else "exact") if mode == "auto" else mode


def heldout_freq(tok, ds: str, L: int, mode: str, args, vocab: int):
    """Token counts over prompts (+ their reference answers) that follow the evaluation pool in the
    deterministic shuffle order, i.e. not in it.  Returns (counts [vocab], number of prompts)."""
    from solutions.run_ablation import load_workload
    n = args.prompts_per_workload
    pool = load_workload(tok, ds, L, n + args.freq_prompts, mode, args.seed)
    held = pool[n:]
    counts = np.zeros(vocab, dtype=np.int64)
    for p in held:
        ids = list(p.tokens) + tok(p.reference, add_special_tokens=False)["input_ids"]
        np.add.at(counts, np.asarray([i for i in ids if i < vocab], dtype=np.int64), 1)
    return counts, len(held)


def main() -> None:
    from common.cli import add_model_args, load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.engine import ARBatchEngine, RunConfig
    from common.traces import ExitProbe
    from common.utils import resolve_model_path, seed_everything
    from solutions.run_ablation import load_workload

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap)
    ap.add_argument("--datasets", nargs="+", default=["cnn_dm_summarization", "xsum_summarization", "human_eval", "gsm8k"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[512])
    ap.add_argument("--exit-layers", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompts-per-workload", type=int, default=64)
    ap.add_argument("--length-mode", choices=["exact", "cap", "auto"], default="exact",
                    help="auto = cap for human_eval (short prompts), exact otherwise")
    ap.add_argument("--topk-layer", type=int, default=None, help="also store the top-k of this exit layer and of the full model")
    ap.add_argument("--topk-exit", type=int, default=64)
    ap.add_argument("--topk-full", type=int, default=32)
    ap.add_argument("--freq-prompts", type=int, default=0,
                    help="count token frequencies over this many prompts (+ references) that follow the evaluation pool")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(RESULTS / "ref"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    tok = load_tokenizer(resolve_model_path(args.model))
    model = load_model_from_args(args, max_seq=max(args.lengths) + args.max_new_tokens + 64)
    layers = [k for k in args.exit_layers if 1 <= k < model.n_layers] + [model.n_layers]
    for ds in args.datasets:
        mode = length_mode(args.length_mode, ds)
        for L in args.lengths:
            pool = load_workload(tok, ds, L, args.prompts_per_workload, mode, args.seed)
            if not pool:
                print(f"[make_ref] {ds} L={L}: no prompts, skipped", flush=True)
                continue
            eng = ARBatchEngine(model, min(args.batch_size, len(pool)), max(len(p.tokens) for p in pool),
                                args.max_new_tokens, eos_ids(tok, model.cfg))
            eng.stop_checker = StopChecker(tok, STOP_STRINGS.get(ds, ()))
            probe = ExitProbe(model, layers, topk_layer=args.topk_layer, topk_exit=args.topk_exit, topk_full=args.topk_full)
            eng.probe = probe
            eng.run(pool, RunConfig(mode="static"))
            out = ref_path(args.out, args.model, ds, L)
            by_pid = {p.pid: p.tokens for p in pool}
            ptoks = [by_pid[r["pid"]] for r in probe.finished]
            extra = {"prompt_tokens": np.concatenate([np.asarray(t, dtype=np.int32) for t in ptoks]),
                     "prompt_offsets": np.cumsum([0] + [len(t) for t in ptoks]).astype(np.int64)}
            n_freq = 0
            if args.freq_prompts:
                extra["heldout_freq"], n_freq = heldout_freq(tok, ds, L, mode, args, model.cfg.vocab_size)
            probe.save(out, {"model": Path(args.model).name, "n_layers": model.n_layers, "dataset": ds, "length": L,
                             "max_new_tokens": args.max_new_tokens, "length_mode": mode,
                             "prompts_per_workload": args.prompts_per_workload, "seed": args.seed,
                             "heldout_prompts": n_freq}, extra)
            model.free_cache()
            print(f"[make_ref] {ds} L={L}: {len(probe.finished)} requests -> {out}", flush=True)


if __name__ == "__main__":
    main()
