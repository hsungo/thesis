"""Feasibility measurements for two candidate mechanisms (measurement only - neither is implemented).

new M2  batch-union draft vocabulary: the exit head scores only the tokens relevant to the batch (V);
        verification keeps the full vocabulary.
new M3  verify free guesses in deep-step slack: after the drafts, append n-gram guesses or the rejected
        draft tail and verify them in the same deep step.

Sections (outputs in --out, default results/feasibility/; one CSV per item + SUMMARY.md):
  1  s1_step_timing.csv       shallow / deep step split into CUDA-event segments + wall clock   (GPU, measured)
     s1_lmhead_subvocab.csv   LM head on V_sub rows vs full; index_select rebuild of the rows      (GPU, measured)
  2  s2_coverage.csv          batch-union vocabulary: |V|, alpha_V vs alpha                        (exact, from AR outputs)
  3  s3_guess_extension.csv   extra tokens confirmed by n-gram guesses / rejected tails           (exact, from outputs)
  4  s4_deep_slack.csv        knee_measured - tokens per deep step                                (logged real runs)
Sections 2-4 are feasibility analyses: acceptance lengths computed exactly (greedy) from real outputs.

    python -m solutions.feasibility all --models layerskip-llama3-8B layerskip-llama3-1B     # everything
    python -m solutions.feasibility timing  --models ...      # sec.1
    python -m solutions.feasibility rounds  --models ...      # fixed-g4 / m1 cyclic runs with --log-rounds
    python -m solutions.feasibility analyze --models ...      # sec.2-4 (CPU)
    python -m solutions.feasibility summary                   # SUMMARY.md from the CSVs

`all` also runs, when missing, solutions/calibrate (into <out>/calibration, per GPU) and solutions/make_ref with
--topk-layer k --freq-prompts 64 --length-mode auto (into <out>/ref).  It refuses to write results unless the GPU
name contains --expect-gpu (default "4090"); --allow-other-gpu (with a different --out) is for pipeline checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feasibility"
TASKS = ("cnn_dm_summarization", "xsum_summarization", "human_eval", "gsm8k")
MODELS = ("layerskip-llama3-8B", "layerskip-llama3-1B")
SEG_SH = ("meta", "layers_k", "head")
SEG_DP = ("meta", "layers_k", "deep_meta", "layers_deep", "head", "accept")


# ============================================================================== helpers

def write_csv(path: Path, rows: list[dict], replace_key: tuple = ()) -> None:
    """Write rows; with replace_key, keep existing rows whose key differs (per-model reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if replace_key and path.exists():
        new_keys = {tuple(str(r[k]) for k in replace_key) for r in rows}
        old = [r for r in read_csv(path) if tuple(r[k] for k in replace_key) not in new_keys]
    rows = old + rows
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def check_gpu(args) -> str:
    name = gpu_name()
    if args.expect_gpu and args.expect_gpu not in name and not args.allow_other_gpu:
        raise SystemExit(f"GPU is {name!r}, expected one containing {args.expect_gpu!r}. Results must come from that "
                         f"GPU; for a pipeline check use --allow-other-gpu with a different --out.")
    if args.allow_other_gpu and Path(args.out).resolve() == OUT.resolve() and args.expect_gpu not in name:
        raise SystemExit(f"--allow-other-gpu on {name!r}: choose another --out than {OUT}")
    return name


def calib_path(out: Path, model: str) -> Path:
    return Path(out) / "calibration" / f"{Path(model).name}.json"


def load_calibration(out: Path, model: str, cfg=None, check: bool = True):
    """(knee_measured, cost model or None).  For GPU runs the calibration must come from the GPU in use."""
    p = calib_path(out, model)
    if not p.exists():
        raise SystemExit(f"missing {p}: python -m solutions.calibrate --model {model} --out {p}")
    d = json.loads(p.read_text())
    calib_gpu = d["gpu"].get("meta", {}).get("env", {}).get("gpu")
    try:
        import torch
        if check and torch.cuda.is_available() and calib_gpu and calib_gpu != gpu_name():
            raise SystemExit(f"{p} was measured on {calib_gpu}, this GPU is {gpu_name()}: recalibrate")
    except ImportError:
        pass
    cm = None
    if cfg is not None:
        from solutions.cost_model import load_cost_model
        cm = load_cost_model(p, cfg, "roofline")
    return float(d["gpu"]["knee_measured"]), cm


def engine_args(k: int) -> SimpleNamespace:
    """run_ablation defaults (the configs are built exactly as in the ablation)."""
    return SimpleNamespace(k=k, d_max=16, sla_steps=32, probe_every=20, k_cands=[2, 3, 4, 5, 6, 7, 8],
                           switch_margin=0.05, max_wait=8, knee_safety=0.9, prefill_chunk=512, bucketing="exact",
                           base_gamma=4, log_rounds=True)


def pct(a, q) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ============================================================================== sec.1 timing (GPU)

class Marks:
    """Segment boundaries: a CUDA event (device time) and perf_counter (host issue time) per mark."""

    def __init__(self, dev, n: int):
        import torch
        self.cuda = dev.type == "cuda"
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if self.cuda else []
        self.host, self.i = [], 0

    def mark(self) -> None:
        if self.cuda:
            self.ev[self.i].record()
        self.host.append(time.perf_counter())
        self.i += 1

    def segments(self) -> tuple[list, list]:
        host = [(b - a) * 1000 for a, b in zip(self.host, self.host[1:])]
        if not self.cuda:
            return host, host
        return [a.elapsed_time(b) for a, b in zip(self.ev, self.ev[1:])], host


def _measure(fn, dev, n_marks: int, warmup: int, iters: int) -> dict:
    """Median over iters of every segment (device + host) and of the whole step's wall clock, each step
    synchronised at both ends."""
    import torch
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    gpu, host, wall = [], [], []
    for it in range(warmup + iters):
        mk = Marks(dev, n_marks)
        sync()
        t0 = time.perf_counter()
        fn(mk)
        sync()
        w = (time.perf_counter() - t0) * 1000
        if it >= warmup:
            g, h = mk.segments()
            gpu.append(g)
            host.append(h)
            wall.append(w)
    return {"gpu": np.median(gpu, 0), "host": np.median(host, 0), "wall": float(np.median(wall)),
            "span": float(np.median(np.sum(gpu, 1)))}


def shallow_step(m, B: int, C: int, k: int, buf: dict, mk: Marks) -> None:
    """One shallow step of EESDEngine with every slot drafting (T = 1 at position C), same calls."""
    from solutions.packed import Block, build_packed, packed_layers, upload
    dev, Dm1, dim = m.device, buf["Dm1"], m.cfg.dim
    mk.mark()
    pb = build_packed(m, [Block(i, C, 1, ("sh", i)) for i in range(B)], "exact")
    col = np.arange(B, dtype=np.int64) * Dm1
    sidx, src, dst = upload(col, dev), upload(np.arange(B), dev), upload(col + 1, dev)
    mk.mark()
    h = packed_layers(m, m.embed(buf["seq"].view(-1)[sidx]), pb, 0, k)
    buf["exh"].view(-1, dim)[sidx] = h[src]
    mk.mark()
    buf["seq"].view(-1)[dst] = m.head(h[src]).argmax(-1)
    mk.mark()


def deep_step(m, B: int, C: int, k: int, gamma: int, G: int, buf: dict, mk: Marks) -> None:
    """One deep step of a synchronous round (every slot verifies d = gamma drafts), same calls as
    EESDEngine._spec_step/_deep.  G > 0 appends G guess tokens per slot after the drafts: they run the
    shallow segment with the newest draft and the deep segment with the verify block."""
    import torch
    from solutions.packed import Block, build_packed, packed_layers, upload
    dev, Dm1, dim, L = m.device, buf["Dm1"], m.cfg.dim, m.n_layers
    T = gamma + 1 + G
    mk.mark()
    pb = build_packed(m, [Block(i, C + gamma, 1 + G, ("sh", i)) for i in range(B)], "exact")
    cols = (np.arange(B)[:, None] * Dm1 + gamma + np.arange(1 + G)[None, :]).reshape(-1)
    sidx = upload(cols, dev)
    mk.mark()
    h = packed_layers(m, m.embed(buf["seq"].view(-1)[sidx]), pb, 0, k)
    buf["exh"].view(-1, dim)[sidx] = h
    mk.mark()
    pbd = build_packed(m, [Block(i, C, T, ("v", i)) for i in range(B)], "exact")
    vcols = (np.arange(B)[:, None] * Dm1 + np.arange(T)[None, :]).reshape(-1)
    j = np.tile(np.arange(T), B)
    vidx, jt, bt = upload(vcols, dev), upload(j, dev), upload(np.repeat(np.arange(B), T), dev)
    dsrc = upload(np.where(j < T - 1, vcols + 1, -1), dev)
    starts = upload(np.arange(B) * T, dev)
    x = buf["exh"].view(-1, dim)[vidx]
    mk.mark()
    hd = packed_layers(m, x, pbd, k, L)
    mk.mark()
    Yv = m.head(hd).argmax(-1)
    mk.mark()
    dr = torch.where(dsrc >= 0, buf["seq"].view(-1)[dsrc.clamp(min=0)], torch.full_like(dsrc, -1))
    big = torch.full((B,), T + 10, dtype=torch.long, device=dev)
    acc = big.scatter_reduce(0, bt, torch.where(Yv != dr, jt, torch.full_like(jt, T + 10)), reduce="amin",
                             include_self=True)
    buf["seq"][:, 0] = Yv[starts + acc]
    torch.cat([acc, Yv]).cpu()
    mk.mark()


def time_model(model, model_name: str, gpu: str, args) -> tuple[list, list]:
    import torch
    import torch.nn.functional as F
    from common.model import rms_norm
    dev, dim = model.device, model.cfg.dim
    gmax = max(args.guess)
    Dm1 = args.gamma + 1 + gmax + 1
    steps, heads = [], []
    # process-level warm-up (cuBLAS handles, allocator, kernel selection) before the first measured point:
    # the per-point warm-up alone left the first point ~40% slow
    B0, C0 = max(args.batch_sizes), args.contexts[0]
    model.allocate_cache(B0, C0 + Dm1 + 64)
    buf = {"Dm1": Dm1, "seq": torch.zeros(B0, Dm1, dtype=torch.long, device=dev),
           "exh": torch.zeros(B0, Dm1, dim, dtype=model.dtype, device=dev)}
    with torch.inference_mode():
        for B in args.batch_sizes:
            bb = {"Dm1": Dm1, "seq": buf["seq"][:B], "exh": buf["exh"][:B]}
            for _ in range(args.warmup):
                shallow_step(model, B, C0, args.k, bb, Marks(dev, 4))
                deep_step(model, B, C0, args.k, args.gamma, 0, bb, Marks(dev, 7))
    model.free_cache()
    for C in args.contexts:
        for B in args.batch_sizes:
            try:
                model.allocate_cache(B, C + Dm1 + 64)
                buf = {"Dm1": Dm1, "seq": torch.randint(0, model.cfg.vocab_size, (B, Dm1), device=dev),
                       "exh": torch.zeros(B, Dm1, dim, dtype=model.dtype, device=dev)}
                with torch.inference_mode():
                    r = _measure(lambda mk: shallow_step(model, B, C, args.k, buf, mk), dev, 4, args.warmup, args.iters)
                    row = {"model": model_name, "gpu": gpu, "step": "shallow", "B": B, "C": C, "G": 0,
                           "tokens_sh": B, "tokens_deep": 0}
                    for n, g, h in zip(SEG_SH, r["gpu"], r["host"]):
                        row[f"{n}_gpu_ms"], row[f"{n}_host_ms"] = float(g), float(h)
                    row.update(gpu_ms=r["span"], wall_ms=r["wall"], host_overhead_ms=r["wall"] - r["span"],
                               head_share_gpu=row["head_gpu_ms"] / r["span"], head_share_wall=row["head_gpu_ms"] / r["wall"])
                    steps.append(row)
                    for G in [0] + list(args.guess):
                        r = _measure(lambda mk: deep_step(model, B, C, args.k, args.gamma, G, buf, mk), dev, 7,
                                     args.warmup, args.iters)
                        row = {"model": model_name, "gpu": gpu, "step": "deep", "B": B, "C": C, "G": G,
                               "tokens_sh": B * (1 + G), "tokens_deep": B * (args.gamma + 1 + G)}
                        for n, g, h in zip(SEG_DP, r["gpu"], r["host"]):
                            row[f"{n}_gpu_ms"], row[f"{n}_host_ms"] = float(g), float(h)
                        row.update(gpu_ms=r["span"], wall_ms=r["wall"], host_overhead_ms=r["wall"] - r["span"])
                        steps.append(row)
                    if C == args.contexts[0]:          # the head does not depend on the context
                        heads += time_heads(model, model_name, gpu, B, args, F, rms_norm)
            except torch.OutOfMemoryError:
                print(f"[timing] {model_name} B={B} C={C}: OOM, skipped", flush=True)
            finally:
                model.free_cache()
            print(f"[timing] {model_name} B={B:2d} C={C}: done", flush=True)
    return steps, heads


def time_heads(model, model_name, gpu, B, args, F, rms_norm) -> list:
    """norm + LM head + argmax on B rows: full vocabulary vs V_sub rows (+ mapping the argmax back to a
    token id), and the index_select that rebuilds the V_sub rows from the full lm_head."""
    import torch
    dev, eps = model.device, model.cfg.norm_eps
    W, V = model.lm_head_w, model.cfg.vocab_size
    h = torch.randn(B, model.cfg.dim, dtype=model.dtype, device=dev)
    rng = np.random.default_rng(0)

    def timed(fn):
        def run(mk):
            mk.mark()
            fn()
            mk.mark()
        return float(_measure(run, dev, 2, args.warmup, args.iters)["gpu"][0])

    full = timed(lambda: model.head(h).argmax(-1))
    rows = []
    for vs in args.vsub:
        idx = torch.as_tensor(np.sort(rng.choice(V, min(vs, V), replace=False)), device=dev)
        Ws = W.index_select(0, idx)
        sub = timed(lambda: idx[F.linear(rms_norm(h, model.norm_w, eps), Ws).argmax(-1)])
        rebuild = timed(lambda: W.index_select(0, idx))
        row = {"model": model_name, "gpu": gpu, "B": B, "V_sub": vs, "full_head_ms": full, "sub_head_ms": sub,
               "saved_ms": full - sub, "rebuild_ms": rebuild}
        for N in (1, 8, 32):
            row[f"net_saved_ms_rebuild_every_{N}"] = full - sub - rebuild / N
        rows.append(row)
        del Ws
    return rows


def cmd_timing(args) -> None:
    import torch
    from common.cli import load_model_from_args
    gpu = check_gpu(args)
    out = Path(args.out)
    for name in args.models:
        model = load_model_from_args(SimpleNamespace(model=name, device=None, dtype="bf16"),
                                     max_seq=max(args.contexts) + 128)
        steps, heads = time_model(model, name, gpu, args)
        write_csv(out / "s1_step_timing.csv", steps, ("model", "gpu", "step", "B", "C", "G"))
        write_csv(out / "s1_lmhead_subvocab.csv", heads, ("model", "gpu", "B", "V_sub"))
        del model
        torch.cuda.empty_cache()


# ============================================================================== logged runs (GPU)

def cmd_rounds(args) -> None:
    import dataclasses
    import torch
    from common.cli import load_model_from_args
    from common.data import STOP_STRINGS, StopChecker, eos_ids, load_tokenizer
    from common.utils import env_info, resolve_model_path, seed_everything
    from solutions.engine import EESDEngine
    from solutions.make_ref import length_mode
    from solutions.run_ablation import engine_config, load_workload
    gpu = check_gpu(args)
    out = Path(args.out)
    for name in args.models:
        seed_everything(args.seed)
        tok = load_tokenizer(resolve_model_path(name))
        model = load_model_from_args(SimpleNamespace(model=name, device=None, dtype="bf16"),
                                     max_seq=args.length + args.max_new_tokens + 16 + 64)
        _, cm = load_calibration(out, name, model.cfg)
        eos = eos_ids(tok, model.cfg)
        env = env_info(model.device)
        for ds in args.tasks:
            stop = StopChecker(tok, STOP_STRINGS.get(ds, ()))
            pool = load_workload(tok, ds, args.length, args.prompts_per_workload, length_mode("auto", ds), args.seed)
            prompts = [pool[i] for i in np.random.default_rng(args.seed).permutation(len(pool))]
            max_p = max(len(p.tokens) for p in pool)
            for B in args.round_batch_sizes:
                for cname in args.configs:
                    f = out / "runs" / Path(name).name / ds / f"B{B}_{cname}.json"
                    if f.exists():
                        continue
                    cfg, _ = engine_config(cname, engine_args(args.k))
                    cfg = dataclasses.replace(cfg, log_rounds=True)
                    t0 = time.time()
                    try:
                        eng = EESDEngine(model, B, max_p, args.max_new_tokens, eos, cfg, cm, stop_for=lambda pid: stop)
                        res = eng.run(prompts, "cyclic", max_eval=args.max_eval, keep_outputs=True)
                    except torch.OutOfMemoryError:
                        print(f"[rounds] {name} {ds} B={B} {cname}: OOM, skipped", flush=True)
                        model.free_cache()
                        continue
                    res["prompts"] = {p.pid: list(p.tokens) for p in pool}
                    res["run"] = {"model": name, "dataset": ds, "length": args.length, "B": B, "config": cname,
                                  "gpu": gpu, "env": env, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                                  "wall_s": time.time() - t0}
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(res, default=str))
                    model.free_cache()
                    m = res["metrics"]
                    print(f"[rounds] {name} {ds} B={B:2d} {cname:9s} {m['wall_throughput_tok_s']:.0f} tok/s wall, "
                          f"{len(res['round_log'])} deep steps ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


# ============================================================================== sec.2 coverage (CPU)

def load_ref(path: Path) -> tuple[list[dict], np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    k = int(z["topk_layer"])
    col = z["exit_layers"].tolist().index(k)
    off, poff = z["offsets"], z["prompt_offsets"]
    reqs = []
    for r in range(len(off) - 1):
        a, b = int(off[r]), int(off[r + 1])
        reqs.append({"pid": meta["pids"][r], "prompt": z["prompt_tokens"][poff[r]:poff[r + 1]], "y": z["tokens"][a:b],
                     "am": z["exit_argmax"][a:b, col], "tid": z["exit_topk_ids"][a:b], "tlg": z["exit_topk_logits"][a:b],
                     "full": z["full_topk_ids"][a:b]})
    meta["k"] = k
    return reqs, z["heldout_freq"], meta


def _pad(reqs: list, idx, key: str, T: int, fill=0):
    first = reqs[idx[0]][key]
    out = np.full((len(idx), T) + first.shape[1:], fill, dtype=first.dtype)
    for j, i in enumerate(idx):
        a = reqs[i][key]
        out[j, :len(a)] = a
    return out


def coverage_batch(reqs: list, idx, R: int, K: int, top_f: np.ndarray, vocab: int) -> dict:
    """Batch-union vocabulary at every generated position t >= 1 (positions aligned across the batch):
    V_t = prompt tokens of the batch | full-model top-K of every request at t-R .. t-1 | F most frequent tokens.
    alpha_V: exit-k argmax restricted to V equals the AR token.  Only exit-k's top-64 (ids, logits) is
    known: the position is 'undetermined' when that cannot decide it - no top-64 token in V while the AR
    token is in V, or the AR token within one bf16 ulp of the best in-V logit together with another in-V
    token (a GEMM over a row subset may round differently), also tokens outside the top-64."""
    lens = np.array([len(reqs[i]["y"]) for i in idx])
    T = int(lens.max())
    Y, AM = _pad(reqs, idx, "y", T), _pad(reqs, idx, "am", T)
    TID, TLG, FULL = _pad(reqs, idx, "tid", T), _pad(reqs, idx, "tlg", T), _pad(reqs, idx, "full", T)
    base = np.zeros(vocab, dtype=bool)
    for i in idx:
        base[reqs[i]["prompt"]] = True
    base[top_f] = True
    base_size = int(base.sum())
    last = np.full(vocab, -1, dtype=np.int64)       # last position whose top-K contained the token (non-base)
    cnt = np.zeros(T + 1, dtype=np.int64)           # cnt[p] = non-base tokens whose last == p
    sizes, n_pos, match, undet, orig = [], 0, 0, 0, 0
    for t in range(1, T):
        ids = np.unique(FULL[lens > t - 1, t - 1, :K])
        ids = ids[~base[ids]]
        old = last[ids]
        np.add.at(cnt, old[old >= 0], -1)
        last[ids] = t - 1
        cnt[t - 1] += len(ids)
        act = lens > t
        if not act.any():
            continue
        lo = max(t - R, 0)
        sizes.append(base_size + int(cnt[lo:t].sum()))
        tid, tlg, y = TID[act, t], TLG[act, t], Y[act, t]
        in_top = base[tid] | (last[tid] >= lo)
        y_in = base[y] | (last[y] >= lo)
        any_in = in_top.any(1)
        best = tlg[np.arange(len(y)), in_top.argmax(1)]
        lo_logit = best - np.exp2(np.floor(np.log2(np.maximum(np.abs(best), 1e-30))) - 7)   # best - 1 bf16 ulp
        cand = in_top & (tlg >= lo_logit[:, None])
        y_top = tid == y[:, None]
        y_cand = (cand & y_top).any(1)
        edge = lo_logit <= tlg[:, -1]                # a token outside the top-64 may be a candidate too
        ok = any_in & y_cand & (cand.sum(1) == 1) & ~edge
        und = (~any_in & y_in) | (any_in & y_cand & ~ok) | (any_in & edge & y_in & ~y_top.any(1))
        n_pos += int(act.sum())
        match += int(ok.sum())
        undet += int(und.sum())
        orig += int((AM[act, t] == y).sum())
    return {"sizes": sizes, "n_pos": n_pos, "match": match, "undet": undet, "orig": orig}


def analyze_coverage(ref_dir: Path, model: str, tasks, args) -> list[dict]:
    rows = []
    for ds in tasks:
        p = ref_dir / Path(model).name / f"{ds}_L{args.length}.npz"
        if not p.exists():
            print(f"[coverage] missing {p}", flush=True)
            continue
        reqs, freq, meta = load_ref(p)
        vocab = len(freq)
        order = np.lexsort((np.arange(vocab), -freq))
        for B in args.cov_batch_sizes:
            if B > len(reqs):
                continue
            rng = np.random.default_rng(args.seed + B)
            batches = [rng.choice(len(reqs), B, replace=False) for _ in range(args.cov_batches)]
            for F in args.cov_F:
                top_f = order[:F][freq[order[:F]] > 0]
                for R in args.cov_R:
                    for K in args.cov_K:
                        acc = {"sizes": [], "n_pos": 0, "match": 0, "undet": 0, "orig": 0}
                        for bi in batches:
                            r = coverage_batch(reqs, bi, R, K, top_f, vocab)
                            acc["sizes"] += r["sizes"]
                            for key in ("n_pos", "match", "undet", "orig"):
                                acc[key] += r[key]
                        s = np.array(acc["sizes"])
                        a, n = acc["orig"] / acc["n_pos"], acc["n_pos"]
                        rows.append({"model": Path(model).name, "task": ds, "k": meta["k"], "B": B, "R": R, "K": K,
                                     "F": F, "batches": len(batches), "positions": n,
                                     "V_mean": float(s.mean()), "V_p50": pct(s, 50), "V_p90": pct(s, 90),
                                     "V_max": int(s.max()), "share_V_le_16k": float((s <= 16384).mean()),
                                     "alpha": a, "alpha_V_lower": acc["match"] / n,
                                     "alpha_V_upper": (acc["match"] + acc["undet"]) / n,
                                     "undetermined_share": acc["undet"] / n,
                                     "ratio_lower": acc["match"] / n / a, "ratio_upper": (acc["match"] + acc["undet"]) / n / a})
                        print(f"[coverage] {model} {ds} B={B} R={R} K={K} F={F}: |V| p90 {pct(s, 90):.0f}, "
                              f"alpha {a:.3f} -> {acc['match'] / n:.3f}..{(acc['match'] + acc['undet']) / n:.3f}", flush=True)
    return rows


# ============================================================================== sec.3 guess extension (CPU)

def ngram_guess(ctx: np.ndarray, G: int, nmax: int = 4) -> np.ndarray:
    """Prompt-lookup guess: the longest suffix n-gram (n = nmax..1) of ctx that occurred earlier; return the
    (up to G) tokens that followed its most recent earlier occurrence."""
    from numpy.lib.stride_tricks import sliding_window_view
    Lc = len(ctx)
    for n in range(min(nmax, Lc - 1), 0, -1):
        win = sliding_window_view(ctx[:Lc - 1], n)          # occurrences starting at j <= Lc-1-n
        hits = np.flatnonzero((win == ctx[Lc - n:]).all(1))
        if len(hits):
            j = int(hits[-1])
            return ctx[j + n:j + n + G]
    return ctx[:0]


def match_len(guess, target) -> int:
    n = min(len(guess), len(target))
    if n == 0:
        return 0
    neq = np.flatnonzero(np.asarray(guess[:n]) != np.asarray(target[:n]))
    return int(neq[0]) if len(neq) else n


def ngram_from_ref(reqs: list, gamma: int, Gs) -> dict:
    """Source A on the AR reference: synchronous gamma rounds (accepted = min(run, gamma), advance accepted + 1);
    in a fully accepted round the guess follows the gamma-th draft and confirms g extra tokens."""
    from common.traces import run_lengths, simulate_rounds
    tot = {"rounds": 0, "full": 0, "committed": 0, **{G: 0 for G in Gs}}
    gmax = max(Gs)
    for r in reqs:
        y = r["y"].astype(np.int64)
        starts, acc, committed = simulate_rounds(run_lengths(r["am"] == r["y"]), gamma)
        tot["rounds"] += len(starts)
        tot["committed"] += int(committed.sum())
        prompt = r["prompt"].astype(np.int64)
        for c, a in zip(starts, acc):
            if a < gamma:
                continue
            tot["full"] += 1
            ctx = np.concatenate([prompt, y[:c + gamma]])
            guess = ngram_guess(ctx, gmax)
            tgt = y[c + gamma:]
            for G in Gs:
                tot[G] += match_len(guess[:G], tgt)
    return tot


def rounds_by_request(run: dict) -> tuple[dict, dict, dict]:
    """uid -> list of (deep-step index, verifier record); uid -> output tokens; uid -> prompt tokens."""
    comp = {c["uid"]: c for c in run["completions"]}
    per = {}
    for si, st in enumerate(run["round_log"]):
        for v in st["verifiers"]:
            if v["uid"] in comp:
                per.setdefault(v["uid"], []).append((si, v))
    outs = {u: np.asarray(c["tokens"], dtype=np.int64) for u, c in comp.items()}
    prompts = {u: np.asarray(run["prompts"][c["pid"]], dtype=np.int64) for u, c in comp.items()}
    return per, outs, prompts


def extension_from_log(run: dict, Gs, exclude=frozenset()) -> dict:
    """On a logged fixed-gamma run: source A (n-gram after a fully accepted round) and source B (the rejected
    tail of round r verified as a second branch in round r+1: it confirms m tokens, the round's own drafts
    a_{r+1} + 1, so it adds max(0, m - a_{r+1})).  Also the raw alignment m of every tail.  Extra tokens are
    attributed to the deep step that would carry the guesses (A: same round, B: next round)."""
    per, outs, prompts = rounds_by_request(run)
    gmax = max(Gs)
    n_steps = len(run["round_log"])
    stepA = {G: np.zeros(n_steps) for G in Gs}
    stepB = np.zeros(n_steps)
    tot = {"rounds": 0, "full": 0, "committed": 0, "tails": 0, "tail_m": 0, "tail_m_pos": 0, "B_extra": 0,
           **{("A", G): 0 for G in Gs}}
    for uid, rr in per.items():
        if uid in exclude:
            continue
        out, prompt = outs[uid], prompts[uid]
        for idx, (si, v) in enumerate(rr):
            tot["rounds"] += 1
            tot["committed"] += len(v["committed"])
            n_out = v["pos"] - len(prompt) + 1                  # tokens committed before this round
            d, a = v["d"], v["a"]
            if a == d and d > 0:
                tot["full"] += 1
                guess = ngram_guess(np.concatenate([prompt, out[:n_out + d]]), gmax)
                for G in Gs:
                    g = match_len(guess[:G], out[n_out + d:])
                    tot[("A", G)] += g
                    stepA[G][si] += g
            elif a < d - 1:                                     # a tail exists after the rejected draft
                tail = v["drafts"][a + 1:]
                m = match_len(tail, out[n_out + a + 1:])
                tot["tails"] += 1
                tot["tail_m"] += m
                tot["tail_m_pos"] += int(m > 0)
                if idx + 1 < len(rr):
                    sj, nv = rr[idx + 1]
                    extra = max(0, m - nv["a"])
                    tot["B_extra"] += extra
                    stepB[sj] += extra
    meas = np.array([st["measured"] for st in run["round_log"]], dtype=bool)
    tot["deep_steps"] = int(meas.sum())
    tot["stepA"] = {G: float(stepA[G][meas].mean()) if meas.any() else 0.0 for G in Gs}
    tot["stepB"] = float(stepB[meas].mean()) if meas.any() else 0.0
    return tot


REPETITIVE = 0.5     # self_repetition(8-gram) above this = degenerate loop (robustness column, common/traces.py)


def analyze_extension(ref_dir: Path, out: Path, model: str, tasks, args) -> list[dict]:
    """Main numbers over all requests; *_excl_repetitive repeats them without requests whose output is a
    degenerate loop (n-gram guesses are trivially right there)."""
    from common.traces import self_repetition
    rows = []
    name = Path(model).name
    for ds in tasks:
        p = ref_dir / name / f"{ds}_L{args.length}.npz"
        if p.exists():
            reqs, _, _ = load_ref(p)
            keep = [r for r in reqs if self_repetition(r["y"]) <= REPETITIVE]
            t = ngram_from_ref(reqs, args.gamma, args.guess)
            tx = ngram_from_ref(keep, args.gamma, args.guess)
            for G in args.guess:
                rows.append({"model": name, "task": ds, "source": "A_ngram", "data": "AR reference (sync gamma split)",
                             "G": G, "rounds": t["rounds"], "full_accept_share": t["full"] / t["rounds"],
                             "committed_per_round": t["committed"] / t["rounds"],
                             "extra_per_round": t[G] / t["rounds"], "extra_per_full_round": t[G] / max(t["full"], 1),
                             "relative_gain": t[G] / t["committed"], "B": "", "extra_per_deep_step": "",
                             "repetitive_share": 1 - len(keep) / len(reqs),
                             "extra_per_round_excl_repetitive": tx[G] / max(tx["rounds"], 1)})
        f = out / "runs" / name / ds / f"B{args.tail_batch}_fixed-g{args.gamma}.json"
        if not f.exists():
            print(f"[extension] missing {f}", flush=True)
            continue
        run = json.loads(f.read_text())
        t = extension_from_log(run, args.guess)
        rep_uids = frozenset(c["uid"] for c in run["completions"] if self_repetition(c["tokens"]) > REPETITIVE)
        tx = extension_from_log(run, args.guess, rep_uids)
        base = {"model": name, "task": ds, "B": args.tail_batch, "rounds": t["rounds"],
                "full_accept_share": t["full"] / t["rounds"], "committed_per_round": t["committed"] / t["rounds"],
                "repetitive_share": len(rep_uids) / max(len(run["completions"]), 1)}
        xr = max(tx["rounds"], 1)
        for G in args.guess:
            rows.append({**base, "source": "A_ngram", "data": f"logged fixed-g{args.gamma} run", "G": G,
                         "extra_per_round": t[("A", G)] / t["rounds"], "extra_per_full_round": t[("A", G)] / max(t["full"], 1),
                         "relative_gain": t[("A", G)] / t["committed"], "extra_per_deep_step": t["stepA"][G],
                         "extra_per_round_excl_repetitive": tx[("A", G)] / xr})
            rows.append({**base, "source": "B_tail", "data": f"logged fixed-g{args.gamma} run", "G": G,
                         "extra_per_round": t["B_extra"] / t["rounds"], "relative_gain": t["B_extra"] / t["committed"],
                         "extra_per_deep_step": t["stepB"], "tail_rounds_share": t["tails"] / t["rounds"],
                         "extra_per_round_excl_repetitive": tx["B_extra"] / xr,
                         "tail_alignment_mean": t["tail_m"] / max(t["tails"], 1),
                         "tail_alignment_pos_share": t["tail_m_pos"] / max(t["tails"], 1)})
            rows.append({**base, "source": "A+B", "data": f"logged fixed-g{args.gamma} run", "G": G,
                         "extra_per_round": (t[("A", G)] + t["B_extra"]) / t["rounds"],
                         "relative_gain": (t[("A", G)] + t["B_extra"]) / t["committed"],
                         "extra_per_deep_step": t["stepA"][G] + t["stepB"],
                         "extra_per_round_excl_repetitive": (tx[("A", G)] + tx["B_extra"]) / xr})
        print(f"[extension] {name} {ds}: full-accept {t['full'] / t['rounds']:.2f}, A(G=8) "
              f"{t[('A', 8)] / t['rounds'] if 8 in args.guess else float('nan'):.3f}/round, B {t['B_extra'] / t['rounds']:.3f}/round",
              flush=True)
    return rows


# ============================================================================== sec.4 deep-step slack (CPU)

def analyze_slack(out: Path, model: str, tasks, args) -> list[dict]:
    name = Path(model).name
    knee, _ = load_calibration(out, name, check=False)
    rows = []
    for ds in tasks:
        for cname in args.configs:
            for B in args.round_batch_sizes:
                f = out / "runs" / name / ds / f"B{B}_{cname}.json"
                if not f.exists():
                    continue
                run = json.loads(f.read_text())
                st = [s for s in run["round_log"] if s["measured"]]
                n = np.array([s["n_dp"] for s in st], dtype=float)
                nv = np.array([max(len(s["verifiers"]), 1) for s in st], dtype=float)
                slack = knee - n
                ps = slack / nv
                row = {"model": name, "task": ds, "config": cname, "B": B, "knee_measured": knee, "deep_steps": len(st),
                       "n_median": pct(n, 50), "verifiers_median": pct(nv, 50),
                       "slack_median": pct(slack, 50), "slack_p10": pct(slack, 10), "slack_p90": pct(slack, 90),
                       "slack_per_verifier_median": pct(ps, 50), "slack_per_verifier_p10": pct(ps, 10),
                       "slack_per_verifier_p90": pct(ps, 90), "share_over_knee": float((slack < 0).mean())}
                for G in args.guess:
                    row[f"share_fits_G{G}_per_verifier"] = float((slack >= G * nv).mean())
                rows.append(row)
    return rows


def cmd_analyze(args) -> None:
    out = Path(args.out)
    ref_dir = Path(args.ref_dir) if args.ref_dir else out / "ref"
    cov, ext, slack = [], [], []
    for model in args.models:
        cov += analyze_coverage(ref_dir, model, args.tasks, args)
        ext += analyze_extension(ref_dir, out, model, args.tasks, args)
        slack += analyze_slack(out, model, args.tasks, args)
    write_csv(out / "s2_coverage.csv", cov, ("model", "task", "B", "R", "K", "F"))
    write_csv(out / "s3_guess_extension.csv", ext, ("model", "task", "source", "data", "G"))
    write_csv(out / "s4_deep_slack.csv", slack, ("model", "task", "config", "B"))


# ============================================================================== summary

def _f(x, fmt="{:.3f}"):
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return "–"


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def cmd_summary(args) -> None:
    out = Path(args.out)
    steps = read_csv(out / "s1_step_timing.csv")
    heads = read_csv(out / "s1_lmhead_subvocab.csv")
    cov = read_csv(out / "s2_coverage.csv")
    ext = read_csv(out / "s3_guess_extension.csv")
    slack = read_csv(out / "s4_deep_slack.csv")
    main = Path(args.models[0]).name
    gpus = sorted({r["gpu"] for r in steps}) or ["(no timing data)"]
    L = []
    w = L.append
    w("# 新 M2 / 新 M3 可行性量測")
    w("")
    w(f"GPU: {', '.join(gpus)} · 主模型 {main}" + (f"，對照 {', '.join(Path(m).name for m in args.models[1:])}" if args.models[1:] else "")
      + f" · k = {args.k}, bf16, greedy · prompt {args.length}（human_eval 用 cap）, max_new_tokens {args.max_new_tokens}")
    w("")
    w("第 1 節是實機計時（CUDA event 分段 + perf_counter 牆鐘，warmup 5、量 20 取中位數）。第 2、3、4 節是**可行性分析**："
      "接受長度從實機產生的輸出精確計算（greedy 下是確定的），第 4 節的 token 數來自實機執行的記錄。吞吐量上限都是**估算**。")
    w("")

    # ---------------- criteria
    def sh(model, B, C):
        return next((r for r in steps if r["model"] == model and r["step"] == "shallow" and int(r["B"]) == B and int(r["C"]) == C), None)

    def dp(model, B, C, G):
        return next((r for r in steps if r["model"] == model and r["step"] == "deep" and int(r["B"]) == B
                     and int(r["C"]) == C and int(r["G"]) == G), None)

    Bs_small = [B for B in sorted({int(r["B"]) for r in steps if r["model"] == main}) if B <= 16]
    Cs = sorted({int(r["C"]) for r in steps if r["model"] == main})
    shares = [(B, C, float(sh(main, B, C)["head_share_wall"]), float(sh(main, B, C)["head_share_gpu"]))
              for B in Bs_small for C in Cs if sh(main, B, C)]
    m2a = bool(shares) and all(s >= 0.25 for _, _, s, _ in shares)
    min_share = min(shares, key=lambda x: x[2]) if shares else None

    # M2 (b): per B, a setting (R, K, F) that gives max |V| <= 16k and alpha_V (lower bound) >= 0.98 alpha on every task
    cm = [r for r in cov if r["model"] == main]
    tasks_cov = sorted({r["task"] for r in cm})
    m2b_by_B = {}
    for B in sorted({int(r["B"]) for r in cm}):
        best = None
        for key in sorted({(r["R"], r["K"], r["F"]) for r in cm if int(r["B"]) == B}):
            rs = [r for r in cm if int(r["B"]) == B and (r["R"], r["K"], r["F"]) == key]
            if len(rs) < len(tasks_cov):
                continue
            ok = all(int(r["V_max"]) <= 16384 and float(r["ratio_lower"]) >= 0.98 for r in rs)
            worst = min(float(r["ratio_lower"]) for r in rs)
            vmax = max(int(r["V_max"]) for r in rs)
            cand = (ok, worst, -vmax, key)
            if best is None or cand > best:
                best = cand
        m2b_by_B[B] = best
    m2b = bool(m2b_by_B) and all(m2b_by_B.get(B) and m2b_by_B[B][0] for B in m2b_by_B if B <= 16)

    # M3 (a): extra tokens per round >= 0.3 on >= 2 tasks, for some source and G
    em = [r for r in ext if r["model"] == main]
    m3a_hits = []
    for src, data in sorted({(r["source"], r["data"]) for r in em}):
        for G in sorted({int(r["G"]) for r in em}):
            rs = [r for r in em if r["source"] == src and r["data"] == data and int(r["G"]) == G]
            good = [r["task"] for r in rs if float(r["extra_per_round"]) >= 0.3]
            m3a_hits.append((len(good), src, data, G, good, rs))
    m3a_ok = [h for h in m3a_hits if h[0] >= 2]
    m3a = bool(m3a_ok)
    chosen = (min(m3a_ok, key=lambda h: (h[3], -h[0])) if m3a_ok
              else max(m3a_hits, key=lambda h: (h[0], -h[3]), default=None))
    G_star = chosen[3] if chosen else 4
    # M3 (b): median slack per verifying slot >= G*, fixed-g4 at B = tail batch (where (a) was measured)
    sm = [r for r in slack if r["model"] == main]
    sb = [r for r in sm if r["config"] == f"fixed-g{args.gamma}" and int(r["B"]) == args.tail_batch]
    m3b = bool(sb) and all(float(r["slack_per_verifier_median"]) >= G_star for r in sb)

    w("## 判斷標準")
    w("")
    w("**新 M2 值得做**：" + _mark(m2a and m2b))
    w("")
    if min_share:
        w(f"- {_mark(m2a)} {main} 在 B ≤ 16 時 LM head ≥ 淺層 step 的 25%：最低 {min_share[2]:.1%}（B = {min_share[0]}, C = {min_share[1]}；"
          f"head 的 GPU 時間 ÷ 整步牆鐘）。各點：" + "、".join(f"B{B}/C{C} {s:.0%}" for B, C, s, _ in shares) + "。")
    else:
        w("- ✗ 沒有第 1 節的數據")
    for B, best in m2b_by_B.items():
        if best is None:
            continue
        ok, worst, nvmax, (R, K, F) = best
        w(f"- {_mark(ok)} B = {B}：最佳設定 R = {R}, K = {K}, F = {F}，四任務最大 |V| = {-nvmax}，最差 α_V/α = {worst:.3f}"
          f"（α_V 取下界：無法判定的位置算不一致）{'' if B <= 16 else '（B > 16，不列入判定）'}")
    w(f"- 判定：B ≤ 16 的每個 B 都要有一組設定，在四個任務都讓最大 |V| ≤ 16384 且 α_V ≥ 0.98 α → {_mark(m2b)}")
    w("")
    w("**新 M3 值得做**：" + _mark(m3a and m3b))
    w("")
    if m3a_hits:
        best = max(m3a_hits, key=lambda h: (h[0], -h[3]))
        for h in sorted([h for h in m3a_hits if h[3] == G_star], key=lambda h: -h[0]):
            vals = "、".join(f"{r['task'].split('_')[0]} {float(r['extra_per_round']):.3f}" for r in h[5])
            w(f"- {_mark(h[0] >= 2)} 來源 {h[1]}（{h[2]}）G = {h[3]}：{h[0]} 個任務 ≥ 0.3（每輪平均多確認：{vals}）")
        w(f"- 判定 (a)：至少兩個任務 ≥ 0.3 → {_mark(m3a)}" + (f"；最小可行 G* = {G_star}（上面只列 G*，其餘 G 見第 3 節）"
                                                           if m3a else f"（最多 {best[0]} 個任務）"))
        if chosen:
            xs = [(r["task"].split("_")[0], r.get("extra_per_round_excl_repetitive"), r.get("repetitive_share")) for r in chosen[5]]
            n_x = sum(1 for _, x, _ in xs if x not in (None, "") and float(x) >= 0.3)
            w(f"- 穩健性（不改變判定）：排除輸出為重複迴圈的請求（8-gram 自我重複 > {REPETITIVE}）後，同一來源與 G 有 {n_x} 個任務 ≥ 0.3："
              + "、".join(f"{t} {_f(x)}（重複請求 {_f(sh, '{:.0%}')}）" for t, x, sh in xs))
    if sb:
        vals = "、".join(f"{r['task'].split('_')[0]} {float(r['slack_per_verifier_median']):.1f}" for r in sb)
        w(f"- {_mark(m3b)} (b) fixed-g{args.gamma}、B = {args.tail_batch} 的每個驗證 slot 空位中位數 ≥ G = {G_star}：{vals} "
          f"（knee_measured = {_f(sb[0]['knee_measured'], '{:.0f}')}；每個 slot 都要放 G 個猜測，所以用「空位 ÷ 驗證 slot 數」）。"
          f"整步空位中位數：" + "、".join(f"{r['task'].split('_')[0]} {float(r['slack_median']):.0f}" for r in sb))
    else:
        w("- ✗ 沒有第 4 節的數據")
    w("")

    # ---------------- throughput upper bounds (estimates)
    w("### 吞吐量上限（估算，用第 1 節實測時間）")
    w("")
    w("同步 γ = 4 一輪 = 4 個淺層 step + 1 個深層 step（牆鐘）。")
    w("")
    w("| B | C | 一輪 ms | M2：V_sub = 16k 每步省 ms（每 8 步重建） | M2 上限 | M3：深層加 G* 個/slot 多花 ms | M3 估算 | M3 上限（猜測免費） |")
    w("|---|---|---|---|---|---|---|---|")
    hm = {(int(r["B"]), int(r["V_sub"])): r for r in heads if r["model"] == main}
    rel = float(np.mean([float(r["relative_gain"]) for r in chosen[5]])) if chosen else 0.0
    for C in Cs:
        for B in sorted({int(r["B"]) for r in steps if r["model"] == main}):
            s, d0, dg = sh(main, B, C), dp(main, B, C, 0), dp(main, B, C, G_star)
            if not (s and d0):
                continue
            T = 4 * float(s["wall_ms"]) + float(d0["wall_ms"])
            hr = hm.get((B, 16384))
            sv = float(hr["net_saved_ms_rebuild_every_8"]) if hr else float("nan")
            m2 = T / (T - 4 * sv) if hr else float("nan")
            dt = float(dg["wall_ms"]) - float(d0["wall_ms"]) if dg else float("nan")
            m3 = (1 + rel) * T / (T + max(dt, 0.0)) if dg else float("nan")   # dt < 0 is noise
            w(f"| {B} | {C} | {T:.2f} | {sv:.3f} | {m2:.3f}× | {dt:.3f} | {m3:.3f}× | {1 + rel:.3f}× |")
    w("")
    w(f"M2 上限假設 α 不變（實際 α_V ≤ α）。M3 用各任務平均的相對增益 {rel:.1%}"
      f"（每輪多確認 ÷ 每輪確認；來源 {chosen[1] if chosen else '–'}、{chosen[2] if chosen else ''}、G = {G_star}）；"
      "「估算」扣掉實測的深層 step 多花時間（負值是量測雜訊，估算時當 0），「上限」假設猜測不花時間。")
    w("")

    # ---------------- section tables
    w("## 1. 淺層與深層 step 的時間拆解（實機）")
    w("")
    for model in args.models:
        name = Path(model).name
        rs = [r for r in steps if r["model"] == name]
        if not rs:
            continue
        w(f"**{name}**（GPU 時間 = CUDA event；host = 該段的 CPU 發出時間；牆鐘含開頭結尾同步）")
        w("")
        w("| step | B | C | meta | k 層 | LM head / L−k 層 | 合計 GPU | 牆鐘 | host 開銷 | head 占比（牆鐘 / GPU） |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            if r["step"] == "shallow":
                w(f"| 淺層 | {r['B']} | {r['C']} | {_f(r['meta_gpu_ms'])} | {_f(r['layers_k_gpu_ms'])} | head {_f(r['head_gpu_ms'])} | "
                  f"{_f(r['gpu_ms'])} | {_f(r['wall_ms'])} | {_f(r['host_overhead_ms'])} | {float(r['head_share_wall']):.0%} / {float(r['head_share_gpu']):.0%} |")
            elif r["G"] == "0":
                w(f"| 深層 | {r['B']} | {r['C']} | {_f(float(r['meta_gpu_ms']) + float(r['deep_meta_gpu_ms']))} | {_f(r['layers_k_gpu_ms'])} | "
                  f"L−k {_f(r['layers_deep_gpu_ms'])}, head {_f(r['head_gpu_ms'])}, 接受 {_f(r['accept_gpu_ms'])} | {_f(r['gpu_ms'])} | "
                  f"{_f(r['wall_ms'])} | {_f(r['host_overhead_ms'])} | |")
        w("")
        hr = [r for r in heads if r["model"] == name]
        if hr:
            w(f"縮小詞彙的 LM head（{name}，n = B 列；省下 = 完整 − 子集；重建 = index_select）")
            w("")
            w("| B | V_sub | 完整 ms | 子集 ms | 每步省 ms | 重建 ms | 每 8 步重建的淨省 | 每 32 步 |")
            w("|---|---|---|---|---|---|---|---|")
            for r in hr:
                w(f"| {r['B']} | {r['V_sub']} | {_f(r['full_head_ms'])} | {_f(r['sub_head_ms'])} | {_f(r['saved_ms'])} | "
                  f"{_f(r['rebuild_ms'])} | {_f(r['net_saved_ms_rebuild_every_8'])} | {_f(r['net_saved_ms_rebuild_every_32'])} |")
            w("")
        gr = [r for r in rs if r["step"] == "deep"]
        if gr:
            w("深層 step 加 G 個猜測 / slot 的牆鐘（ms）")
            w("")
            Gs = sorted({int(r["G"]) for r in gr})
            w("| B | C | " + " | ".join(f"G = {G}" for G in Gs) + " |")
            w("|---|---|" + "---|" * len(Gs))
            for C in sorted({int(r["C"]) for r in gr}):
                for B in sorted({int(r["B"]) for r in gr}):
                    vals = [dp(name, B, C, G) for G in Gs]
                    if vals[0]:
                        w(f"| {B} | {C} | " + " | ".join(_f(v["wall_ms"]) if v else "–" for v in vals) + " |")
            w("")

    w("## 2. 動態詞彙的覆蓋率（可行性分析）")
    w("")
    w("每個位置 t 的 V = 批次 prompt token ∪ 每個請求前 R 個位置的完整模型 top-K ∪ 前 F 個高頻詞（另 64 個不在評估集的 prompt + 參考答案統計）。"
      "批次內位置以生成序號對齊；α 是同一批位置上 exit-k argmax = AR token 的比例。α_V 給下界 / 上界（無法判定算不一致 / 算一致）。")
    w("")
    for model in args.models:
        name = Path(model).name
        rs = [r for r in cov if r["model"] == name]
        if not rs:
            continue
        w(f"**{name}**（每個 B 只列 α_V/α 下界最高的 3 組，以及 |V| 最小、比值 ≥ 0.98 的一組；完整表在 s2_coverage.csv）")
        w("")
        w("| 任務 | B | R | K | F | |V| 平均 / p90 / 最大 | α | α_V 下界–上界 | 無法判定 | α_V/α 下界 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for ds in tasks_cov:
            for B in sorted({int(r["B"]) for r in rs}):
                sel = [r for r in rs if r["task"] == ds and int(r["B"]) == B]
                top = sorted(sel, key=lambda r: -float(r["ratio_lower"]))[:3]
                small = sorted([r for r in sel if float(r["ratio_lower"]) >= 0.98], key=lambda r: float(r["V_max"]))[:1]
                for r in top + [r for r in small if r not in top]:
                    w(f"| {ds.split('_')[0]} | {B} | {r['R']} | {r['K']} | {r['F']} | {float(r['V_mean']):.0f} / {float(r['V_p90']):.0f} / {r['V_max']} | "
                      f"{_f(r['alpha'])} | {_f(r['alpha_V_lower'])}–{_f(r['alpha_V_upper'])} | {float(r['undetermined_share']):.2%} | {_f(r['ratio_lower'])} |")
        w("")

    w("## 3. 猜測延伸（可行性分析）")
    w("")
    w(f"來源 A：AR 參考輸出上模擬同步 γ = {args.gamma}（接受數 = min(exit-k 連續一致長度, γ)，前進接受數 + 1），草稿全接受的輪次在第 γ 個草稿後接 n-gram 猜測"
      f"（n = 4..1 取最長、最近一次出現），g = 與 AR 輸出接下來連續一致的長度。同一算法也套在 fixed-g{args.gamma} B = {args.tail_batch} 的實機記錄上，"
      "以得到每個深層 step 的數字。來源 B：被拒位置之後的草稿尾巴，與下一輪起確認的 token 對齊；當作第二條分支驗證時多確認 max(0, m − a_下一輪) 個"
      "（尾巴長度 ≤ γ − 1，與 G 無關）。")
    w("")
    for model in args.models:
        name = Path(model).name
        rs = [r for r in ext if r["model"] == name]
        if not rs:
            continue
        w(f"**{name}**")
        w("")
        w("| 任務 | 來源 | 資料 | G | 全接受輪次比例 | 每輪確認 | 每輪多確認 | 排除重複請求後 | 相對增益 | 每深層 step 多確認 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            w(f"| {r['task'].split('_')[0]} | {r['source']} | {r['data']} | {r['G']} | {float(r['full_accept_share']):.1%} | "
              f"{_f(r['committed_per_round'], '{:.2f}')} | {_f(r['extra_per_round'])} | {_f(r.get('extra_per_round_excl_repetitive'))} | "
              f"{float(r['relative_gain']):.1%} | {_f(r['extra_per_deep_step'], '{:.2f}')} |")
        tails = [r for r in rs if r["source"] == "B_tail" and r["G"] == str(args.guess[0])]
        if tails:
            w("")
            w("來源 B 的原始對齊長度 m（尾巴與後續確認 token 連續吻合數）：" + "、".join(
                f"{r['task'].split('_')[0]} 平均 {_f(r['tail_alignment_mean'], '{:.2f}')}（m ≥ 1 占 {float(r['tail_alignment_pos_share']):.0%}，"
                f"有尾巴的輪次 {float(r['tail_rounds_share']):.0%}）" for r in tails))
        w("")

    w("## 4. 深層 step 的空位（實機記錄）")
    w("")
    w("空位 = knee_measured − 該深層 step 的 token 數 n（只看量測區間）。每 slot 空位 = 空位 ÷ 驗證 slot 數。")
    w("")
    for model in args.models:
        name = Path(model).name
        rs = [r for r in slack if r["model"] == name]
        if not rs:
            continue
        Gs = args.guess
        w(f"**{name}**（knee_measured = {_f(rs[0]['knee_measured'], '{:.0f}')}）")
        w("")
        w("| 任務 | config | B | n 中位數 | 空位 中位數 (p10, p90) | 每 slot 空位 中位數 (p10, p90) | 超過膝點 | " +
          " | ".join(f"放得下 G={G}/slot" for G in Gs) + " |")
        w("|---|---|---|---|---|---|---|" + "---|" * len(Gs))
        for r in rs:
            w(f"| {r['task'].split('_')[0]} | {r['config']} | {r['B']} | {_f(r['n_median'], '{:.0f}')} | "
              f"{_f(r['slack_median'], '{:.0f}')} ({_f(r['slack_p10'], '{:.0f}')}, {_f(r['slack_p90'], '{:.0f}')}) | "
              f"{_f(r['slack_per_verifier_median'], '{:.1f}')} ({_f(r['slack_per_verifier_p10'], '{:.1f}')}, {_f(r['slack_per_verifier_p90'], '{:.1f}')}) | "
              f"{float(r['share_over_knee']):.0%} | " + " | ".join(f"{float(r[f'share_fits_G{G}_per_verifier']):.0%}" for G in Gs) + " |")
        w("")
    w("## 注意事項")
    w("")
    w("- 第 1 節每次量一步並在前後同步，所以牆鐘包含 kernel launch 的等待；連續執行時 CPU 與 GPU 會重疊，實際每步時間可能較短。")
    w("- 第 2 節的 V 每個位置重建（最細的粒度）；第 1 節的重建成本以「每 N 步一次」攤提。前 R 個位置的完整模型 top-K 在實際系統中要驗證後才拿得到，"
      "本分析未扣除這個延遲（最近 γ 個位置可能尚未驗證），所以是樂觀的。")
    w("- 第 2 節批次內的請求以生成序號對齊；已結束的請求不再加入新的 top-K（實際系統會補位新的請求）。")
    w("- 第 3 節來源 A 的回合切分依規格不因猜測而改變（猜測成功時實際會前進更多，輪數略少）。")
    w("- bf16 下引擎的輸出與 AR 參考可能有少數位置不同；來源 B 和第 4 節用引擎自己的輸出當真值。")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[summary] wrote {out / 'SUMMARY.md'}", flush=True)


# ============================================================================== all

def _py(*a) -> None:
    cmd = [sys.executable, "-m", *map(str, a)]
    print("[all] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_model(name: str) -> None:
    """Download facebook/<name> into ./models/<name> when it is missing (gated repos need `hf auth login`
    and an accepted licence on the model page)."""
    from common.utils import MODELS_DIR
    if Path(name).exists() or (MODELS_DIR / name / "config.json").exists():
        return
    from huggingface_hub import snapshot_download
    repo = "facebook/" + {"layerskip-llama3-1B": "layerskip-llama3.2-1B"}.get(name, name)
    print(f"[all] downloading {repo} -> {MODELS_DIR / name}", flush=True)
    try:
        snapshot_download(repo, local_dir=str(MODELS_DIR / name),
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not download {repo} ({e}); log in with `hf auth login`, accept the licence on "
                         f"https://huggingface.co/{repo}, or put the checkpoint in {MODELS_DIR / name}")


def cmd_all(args) -> None:
    check_gpu(args)
    out = Path(args.out)
    ref_dir = Path(args.ref_dir) if args.ref_dir else out / "ref"
    for name in args.models:
        ensure_model(name)
    for name in args.models:
        p = calib_path(out, name)
        stale = p.exists() and json.loads(p.read_text())["gpu"].get("meta", {}).get("env", {}).get("gpu") != gpu_name()
        if not p.exists() or stale:
            _py("solutions.calibrate", "--model", name, "--k", args.k, "--out", p)
        missing = [ds for ds in args.tasks if not (ref_dir / Path(name).name / f"{ds}_L{args.length}.npz").exists()]
        if missing:
            _py("solutions.make_ref", "--model", name, "--datasets", *missing, "--lengths", args.length,
                "--max-new-tokens", args.max_new_tokens, "--prompts-per-workload", args.prompts_per_workload,
                "--length-mode", "auto", "--seed", args.seed, "--topk-layer", args.k, "--topk-exit", 64,
                "--topk-full", 32, "--freq-prompts", 64, "--exit-layers", args.k, "--out", ref_dir)
    if not args.skip_timing:
        cmd_timing(args)
    cmd_rounds(args)
    cmd_analyze(args)
    cmd_summary(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["all", "timing", "rounds", "analyze", "summary"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="first one is the main model (criteria)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ref-dir", default=None, help="default <out>/ref")
    ap.add_argument("--expect-gpu", default="4090")
    ap.add_argument("--allow-other-gpu", action="store_true", help="pipeline check on another GPU (needs another --out)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompts-per-workload", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # sec.1
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--vsub", nargs="+", type=int, default=[1024, 4096, 16384, 32768])
    ap.add_argument("--skip-timing", action="store_true")
    # sec.2
    ap.add_argument("--cov-batch-sizes", nargs="+", type=int, default=[4, 16, 32])
    ap.add_argument("--cov-batches", type=int, default=50)
    ap.add_argument("--cov-R", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--cov-K", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--cov-F", nargs="+", type=int, default=[0, 1024, 4096])
    # sec.3 / 4
    ap.add_argument("--guess", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--configs", nargs="+", default=["fixed-g4", "m1"], help="logged runs (sec.3 source B uses fixed-g<gamma>)")
    ap.add_argument("--round-batch-sizes", nargs="+", type=int, default=[4, 8, 16, 32], dest="round_batch_sizes")
    ap.add_argument("--tail-batch", type=int, default=16, help="B of the run used for source B")
    ap.add_argument("--max-eval", type=int, default=256)
    args = ap.parse_args()
    {"all": cmd_all, "timing": cmd_timing, "rounds": cmd_rounds, "analyze": cmd_analyze, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
