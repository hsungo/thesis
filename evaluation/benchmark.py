"""Benchmark any registered decoding method (common/methods.py) with one protocol.

Protocol (ASPIRE `benchmark.py`): cyclic replay keeps the batch full, the first
--warmup-samples completions are dropped, the run stops when cumulative decode
throughput is stable (+-1% over --converge-window completions) or after
--max-eval-samples; throughput counts decode forward time only (prefill excluded).
`--mode static` generates every prompt exactly once (with --save-outputs: keeps the
token ids for evaluation/check_lossless.py).

Output: one JSON per (batch size, repetition)
    <output-dir>/<method><tag>-B<B>-rep<r>.json
and with --save-outputs  <output-dir>/<method><tag>-B<B>-outputs.jsonl
Aggregate many runs with evaluation/compare.py.

    python evaluation/benchmark.py --method autoregressive --model layerskip-llama3-8B \
        --dataset cnn_dm_summarization --batch-sizes 1,4,8,16,max --output-dir evaluation/results/cnn_dm
    python evaluation/benchmark.py --method sync_layerskip --exit-layer 8 --gamma 4 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_model_args, load_model_from_args  # noqa: E402
from common.data import StopChecker, add_data_args, eos_ids, load_tokenizer, prompts_from_args, stop_strings_from_args  # noqa: E402
from common.engine import add_run_args, run_config_from_args  # noqa: E402
from common.memory import resolve_batch_sizes  # noqa: E402
from common.methods import METHODS, load_plugins  # noqa: E402
from common.utils import env_info, resolve_model_path, seed_everything, sha256_json, write_json  # noqa: E402


def build_parser(default_method: str | None = None) -> argparse.ArgumentParser:
    load_plugins()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--method", default=default_method, required=default_method is None, choices=sorted(METHODS))
    add_model_args(ap)
    add_data_args(ap, default_limit=128)
    add_run_args(ap)
    g = ap.add_argument_group("benchmark")
    g.add_argument("--batch-sizes", default="16",
                   help="comma list; 'max' = largest batch that fits; sizes that do not fit are skipped")
    g.add_argument("--repetitions", type=int, default=1)
    g.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--save-outputs", action="store_true", help="store generated token ids (.jsonl)")
    g.add_argument("--save-steps", action="store_true", help="store per-step timing records in the JSON")
    g.add_argument("--output-dir", default="evaluation/results/run")
    for spec in METHODS.values():
        spec.add_args(ap.add_argument_group(f"method: {spec.name}"))
    return ap


def run(args) -> list[Path]:
    seed_everything(args.seed)
    spec = METHODS[args.method]
    if args.temperature > 0 and not spec.supports_sampling:
        raise SystemExit(f"method {spec.name} implements greedy decoding only (use --temperature 0)")
    out = Path(args.output_dir)
    tok = load_tokenizer(resolve_model_path(args.model))
    prompts = prompts_from_args(args, tok)
    lens = [len(p.tokens) for p in prompts]
    print(f"[bench] {spec.name}: {len(prompts)} prompts from {args.dataset}, tokens min/mean/max = "
          f"{min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}")
    max_seq = max(lens) + args.max_new_tokens + spec.spec_slack + 64
    model = load_model_from_args(args, max_seq=max_seq)
    eos = eos_ids(tok, model.cfg)
    run_cfg = run_config_from_args(args, keep_outputs=args.save_outputs)
    batch_sizes, skipped = resolve_batch_sizes(args.batch_sizes, model, max_seq, max(lens), spec.step_tokens(args))
    if skipped:
        print(f"[bench] skipping batch sizes {skipped}: KV cache for {max_seq} tokens/request does not fit "
              f"(lower --max-new-tokens / --max-prompt-tokens or use a smaller model)")
    dataset_meta = {
        "name": args.dataset, "limit": args.limit, "n_shot": args.n_shot, "stop": list(stop_strings_from_args(args)), "seed": args.seed,
        "truncate_to": args.truncate_to, "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens, "prompt_count": len(prompts),
        "prompt_tokens": lens, "token_sha256": sha256_json([p.tokens for p in prompts]),
    }
    tag = spec.tag(args)
    written = []
    for B in batch_sizes:
        for rep in range(args.repetitions):
            engine = spec.build(model, args, B, max(lens), eos)
            engine.stop_checker = StopChecker(tok, stop_strings_from_args(args))
            res = engine.run(prompts, run_cfg)
            m, comps = res["metrics"], res["completions"]
            payload = {
                "method": spec.name,
                "method_params": spec.params(args),
                "config": {**vars(args), "batch_size": B, "repetition": rep, "eos_ids": list(eos)},
                "dataset": dataset_meta,
                "env": env_info(model.device),
                "metrics": m,
                "outputs": {
                    "generated_tokens": [c["generated"] for c in comps],
                    "finish_reasons": [c["finish_reason"] for c in comps],
                    "token_sha256": sha256_json([c.get("tokens") for c in comps]) if args.save_outputs else None,
                },
                "completions": [{k: v for k, v in c.items() if k != "tokens"} for c in comps],
            }
            if args.save_steps:
                payload["steps"] = res["steps"]
            path = write_json(out / f"{spec.name}{tag}-B{B}-rep{rep:02d}.json", payload)
            written.append(path)
            if args.save_outputs:
                with open(out / f"{spec.name}{tag}-B{B}-outputs.jsonl", "w", encoding="utf-8") as f:
                    for c in comps:
                        f.write(json.dumps({"pid": c["pid"], "prompt_idx": c["prompt_idx"], "tokens": c["tokens"],
                                            "text": tok.decode(c["tokens"])}) + "\n")
            extra = ""
            if "spec" in m:
                extra = f"  acc {m['spec']['acceptance_rate']:.3f}  tok/verify {m['spec']['tokens_per_verify']:.2f}"
            print(f"[bench] {spec.name}{tag} B={B:3d} rep={rep}  {m['throughput_tok_s'] or 0:9.1f} tok/s (forward)  "
                  f"{m['wall_throughput_tok_s'] or 0:9.1f} tok/s (wall)  TPOT {m['mean_tpot_ms'] or 0:.2f} ms{extra}  "
                  f"({m['evaluated_completions']} completions, {m['stop_reason']}) -> {path}")
            model.free_cache()
    return written


def main(argv=None, default_method: str | None = None) -> None:
    run(build_parser(default_method).parse_args(argv))


if __name__ == "__main__":
    main()
