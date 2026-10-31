"""Upper-bound ("cheat") experiments on the real 1B engine with CUDA graphs.  Each one removes one cost while
keeping everything else real; outputs are not required to be correct.  These are upper bounds, not methods.

  head      the shallow LM head is a GEMM over 1024 rows of the real lm_head (its argmax is computed and
            discarded).  The draft written is replayed from the round log of the matching budget run (same task,
            B, config, repetition, CUDA graphs): accepted drafts = the committed tokens, the first rejected draft =
            committed token + 1 (never accepted).  With graphs every shape is fixed and the head does not touch
            any hidden state, so the trajectory (commits, acceptance, schedule) is that of the real run; the run
            stops no later than the recorded one (max_eval = its evaluated completions).
  prefill   perfect prefix cache: every prompt of the pool is prefilled once before the run and its KV kept;
            every prefill of the run only copies it into the slot.
  oracle    m1-oracle: verification timing with the true acceptance (solutions/schedulers.py).

    python observation/v3/cheats.py --cheat head prefill oracle --configs fixed-g4 m1
Output: results/observe-4060/cheats/<task>/B<B>/<cheat>_<config>_rep<r>.json and G_cheats.csv (analyze).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from observation.v3.budget import MODEL, OUT, Telemetry  # noqa: E402


def replay_table(rec: dict, prompts: list) -> dict:
    """uid -> (prompt length, generated index -> token, round start pos -> accepted) from a budget run's log.
    uid u is the u-th assignment of the cyclic driver: prompt prompts[u % len(prompts)]."""
    tab = {}
    for e in rec["round_log"]:
        for v in e.get("verifiers", ()):
            u = v["uid"]
            plen = len(prompts[u % len(prompts)].tokens)
            t = tab.setdefault(u, (plen, {}, {}))
            n_out = v["pos"] - plen + 1
            for i, tok in enumerate(v["committed"]):
                t[1][n_out + i] = tok
            t[2][v["pos"]] = v["a"]
    return tab


def make_runner_cls(tab: dict, vocab: int, tiny: int = 1024):
    """GraphRunner whose shallow head is a tiny GEMM and whose drafts are replayed from a recorded run."""
    import torch
    import torch.nn.functional as F
    from common.model import rms_norm
    from solutions.graphs import GraphRunner

    class CheatHead(GraphRunner):
        def init_buffers(self, Ts) -> None:
            super().init_buffers(Ts)
            self.W_tiny = self.m.lm_head_w[:tiny].clone()
            self.sink = torch.zeros(self.B, dtype=torch.long, device=self.dev)
            self.buf("ref_draft", self.B)
            self.last_drf = None

        def set(self, name, arr) -> None:
            if name == "drf_dst":
                self.last_drf = np.asarray(arr, dtype=np.int64)
            super().set(name, arr)

        def _head_draft_body(self) -> None:
            h = self.out(1)
            self.sink.copy_(F.linear(rms_norm(h, self.m.norm_w, self.m.cfg.norm_eps), self.W_tiny).argmax(-1))
            self.seq_flat[self.buf("drf_dst", self.B)] = self.buf("ref_draft", self.B)

        def head_draft(self) -> None:
            eng = getattr(self, "engine", None)
            if eng is not None and self.last_drf is not None:
                drafts = np.zeros(self.B, dtype=np.int64)
                for i in np.flatnonzero(self.last_drf != self.S):
                    r = eng.req[i]
                    j = int(self.last_drf[i] - i * self.Dm1)                 # draft index d + 1
                    P = int(eng.pos[i])
                    plen, stream, accs = tab.get(r.uid, (len(r.prompt), {}, {}))
                    n_out = P - plen + 1
                    a = accs.get(P)
                    if a is None:
                        tok = None
                    elif j <= a:
                        tok = stream.get(n_out + j - 1)
                    else:
                        b = stream.get(n_out + a)
                        tok = None if b is None else (b + 1) % vocab
                    self.n_lookups += 1
                    self.n_missing += int(tok is None)
                    drafts[i] = 0 if tok is None else tok
                super().set("ref_draft", drafts)
            super().head_draft()

    CheatHead.n_lookups = 0
    CheatHead.n_missing = 0
    return CheatHead


def make_engine(kind: str, model, B, max_p, eos, cfg, cm, stop, tab=None, oracle_ref=None, pool=None):
    import solutions.graphs as G
    from solutions.engine import EESDEngine
    if kind == "head":
        cls = make_runner_cls(tab, model.cfg.vocab_size)
        orig = G.GraphRunner
        G.GraphRunner = cls                    # the engine imports GraphRunner from solutions.graphs at init
        try:
            eng = EESDEngine(model, B, max_p, 256, eos, cfg, cm, stop_for=stop)
        finally:
            G.GraphRunner = orig
        eng.gr.engine = eng
        return eng
    eng = EESDEngine(model, B, max_p, 256, eos, cfg, cm, stop_for=stop, ref=oracle_ref)
    if kind == "prefill":
        install_prefix_cache(eng, pool)
    return eng


def install_prefix_cache(eng, pool) -> None:
    """Prefill every prompt of the pool once (before the run, slot 0) and keep its KV; during the run a separate
    prefill only copies the KV into the slot."""
    import torch
    from common.timing import Interval
    from solutions.engine import FREE, Request
    store = {}
    orig = eng._prefill_separate
    c = eng.m.cache
    for p in pool:
        eng.req[0] = Request(p.pid, list(p.tokens))
        orig(0)
        P = len(p.tokens)
        store[p.pid] = ([c.k[i][0, :, :P].clone() for i in range(eng.L)],
                        [c.v[i][0, :, :P].clone() for i in range(eng.L)], int(eng.seq_buf[0, 0]))
    eng.req[0] = None
    eng.state[:] = FREE
    eng.records.clear()
    eng.stats["separate_prefills"] = 0

    @torch.inference_mode()
    def prefill(slot: int) -> int:
        r = eng.req[slot]
        P = len(r.prompt)
        ks, vs, y0 = store[r.pid]
        iv = Interval(eng.dev).start()
        for i in range(eng.L):
            c.k[i][slot, :, :P].copy_(ks[i])
            c.v[i][slot, :, :P].copy_(vs[i])
        eng.seq_buf[slot, 0] = y0
        iv.stop()
        eng._rec("prefill", iv, n_chunk=P)
        eng.stats["separate_prefills"] += 1
        if eng.measuring:
            eng.stats["prefill_tokens"] += P
        return eng._first_token(slot, y0)

    eng._prefill_separate = prefill


def best_gamma(out: Path) -> dict:
    """(task, B) -> best fixed-gamma config of the budget runs with CUDA graphs (median wall throughput)."""
    import csv
    return {(r["task"], int(r["B"])): r["config"] for r in csv.DictReader(open(out / "A_budget.csv", encoding="utf-8"))
            if r["mode"] == "graph" and r.get("best_gamma") == "True"}


def cmd_run(args) -> None:
    import torch
    from common.cli import load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.utils import env_info, git_commit, resolve_model_path, seed_everything
    from observation.v3.budget import kv_slopes
    from solutions.cost_model import load_cost_model
    from solutions.feasibility import engine_args
    from solutions.make_ref import load_ref, ref_path
    from solutions.run_ablation import engine_config, load_workload
    seed_everything(42)
    tok = load_tokenizer(resolve_model_path(MODEL))
    model = load_model_from_args(SimpleNamespace(model=MODEL, device=None, dtype="bf16"), max_seq=512 + 256 + 16 + 64)
    eos = eos_ids(tok, model.cfg)
    calib = ROOT / "results" / "eesd" / "calibration" / f"{MODEL}.json"
    cm = load_cost_model(calib, model.cfg, "roofline")
    env = env_info(model.device)
    refdir = Path(args.out) / "ref"
    bg = best_gamma(Path(args.out))
    plan = [(rep, ds, B, ch, c) for rep in range(args.reps) for ds in args.tasks for B in args.batch_sizes
            for ch in args.cheat for c in (["m1"] if ch == "oracle" else [bg[(ds, B)], "m1"])]
    print(f"[cheats] {len(plan)} runs", flush=True)
    for n, (rep, ds, B, ch, cname) in enumerate(plan):
        f = Path(args.out) / "cheats" / ds / f"B{B}" / f"{ch}_{cname}_rep{rep}.json"
        if f.exists():
            continue
        pool = load_workload(tok, ds, 512, 64, "exact", 42)
        prompts = [pool[j] for j in np.random.default_rng(42 + 1000 * rep).permutation(len(pool))]
        stop = StopChecker(tok, STOP_STRINGS.get(ds, ()))
        base = "m1-oracle" if ch == "oracle" else cname
        cfg, _ = engine_config(base, engine_args(3))
        cfg = dataclasses.replace(cfg, log_rounds=True, segment_timing=True, graphs=True)
        rp = ref_path(refdir, MODEL, ds, 512)
        rec = Path(args.out) / "runs" / ds / f"B{B}" / f"{cname}_graph_rep{rep}.json"    # budget run, same cell
        recd = json.loads(rec.read_text())
        max_eval = recd["metrics"]["evaluated_completions"] if ch == "head" else args.max_eval
        t0 = time.time()
        eng = make_engine(ch, model, B, max(len(p.tokens) for p in pool), eos, cfg, cm, lambda pid, s=stop: s,
                          tab=replay_table(recd, prompts) if ch == "head" else None,
                          oracle_ref=load_ref(rp, 3) if ch == "oracle" else None, pool=pool)
        del recd
        tel = Telemetry()
        res = eng.run(prompts, "cyclic", max_eval=max_eval)
        res["telemetry"] = tel.stop()
        res["kv_slopes"] = kv_slopes(eng, res["round_log"])
        if ch == "head":
            res["metrics"]["ref_lookups"] = eng.gr.n_lookups
            res["metrics"]["ref_missing"] = eng.gr.n_missing
        res["run"] = {"model": MODEL, "dataset": ds, "B": B, "config": cname, "cheat": ch, "rep": rep,
                      "mode": "graph", "env": env, "git_commit": git_commit(),
                      "argv": sys.argv, "wall_s": time.time() - t0}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, default=str))
        model.free_cache()
        del eng
        torch.cuda.empty_cache()
        m = res["metrics"]
        print(f"[cheats] {n + 1}/{len(plan)} {ds[:5]} B={B:2d} {ch:7s} {cname:9s} rep{rep}: {m['wall_throughput_tok_s']:7.1f} tok/s "
              f"wall, acceptance {m['acceptance_rate']:.3f} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--cheat", nargs="+", default=["head", "prefill", "oracle"])
    ap.add_argument("--tasks", nargs="+", default=["cnn_dm_summarization", "gsm8k"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-eval", type=int, default=64)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
