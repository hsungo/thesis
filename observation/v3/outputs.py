"""Output-based observations (exact, from real AR outputs of make_ref; no timing):

  C  acceptance structure      C_position / C_calibration / C_token_type / C_requests / C_autocorr / C_exit_layers
  E  free guesses              E_guess.csv          (n-gram over prompt + generated text, other requests of the batch)
  F  batch-union vocabulary    F_coverage.csv
  B  waste, reference side     B_waste_ref.csv      (synchronous gamma rounds simulated on the AR outputs)

    python observation/v3/outputs.py --models layerskip-llama3-8B layerskip-llama3-1B
Input: results/observe-4060/ref/<model>/<task>_L512.npz (solutions/make_ref.py --topk-layers 2..8).
"Accepted" = the exit-k argmax equals the AR token at that position (greedy: what verification accepts when
the prefix is right).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.traces import run_lengths, self_repetition, simulate_rounds  # noqa: E402
from solutions.feasibility import (analyze_coverage, match_len, ngram_guess, pct,  # noqa: E402
                                   write_csv)

OUT = ROOT / "results" / "observe-4060"
TASKS = ("cnn_dm_summarization", "xsum_summarization", "human_eval", "gsm8k")


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    off, poff = d["offsets"], d["prompt_offsets"]
    d["reqs"] = [(int(off[r]), int(off[r + 1]), d["prompt_tokens"][poff[r]:poff[r + 1]]) for r in range(len(off) - 1)]
    return d


def agree(d: dict, k: int) -> np.ndarray:
    return d["exit_argmax"][:, d["exit_layers"].tolist().index(k)] == d["tokens"]


def gen_mask(d: dict) -> np.ndarray:
    """Positions with a draft: every generated token after the first (y_0 comes from the prefill)."""
    m = np.ones(len(d["tokens"]), dtype=bool)
    m[[a for a, _, _ in d["reqs"]]] = False
    return m


# ------------------------------------------------------------------------------ C

def c_position(d, k, gmax=8) -> list:
    """P(draft j accepted | drafts 1..j-1 accepted) in synchronous rounds of gamma = gmax."""
    ag = agree(d, k)
    reached, acc = np.zeros(gmax + 1), np.zeros(gmax + 1)
    for a, b, _ in d["reqs"]:
        run = run_lengths(ag[a:b])
        starts, accs, _ = simulate_rounds(run, gmax)
        for c, n in zip(starts, accs):
            T = b - a
            for j in range(1, gmax + 1):
                if c + j - 1 >= T:
                    break
                reached[j] += 1
                if n >= j:
                    acc[j] += 1
                else:
                    break
    return [{"position": j, "reached": int(reached[j]), "accept_rate": acc[j] / max(reached[j], 1)} for j in range(1, gmax + 1)]


def c_calibration(d, k, bins=10) -> tuple[list, dict]:
    tl = d["topk_layers"].tolist().index(k)
    m = gen_mask(d)
    ag = agree(d, k)[m]
    p1 = np.exp(d["exit_topk_logits"][m, tl, 0] - d["exit_lse"][m, tl])
    ent = d["exit_entropy"][m, tl]
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p1 >= lo) & ((p1 < hi) | (hi == 1))
        if s.any():
            rows.append({"signal": "top1_prob", "bin_lo": lo, "bin_hi": hi, "n": int(s.sum()), "share": s.mean(),
                         "mean_signal": float(p1[s].mean()), "accept_rate": float(ag[s].mean())})
            ece += s.mean() * abs(p1[s].mean() - ag[s].mean())
    qs = np.quantile(ent, np.linspace(0, 1, bins + 1))
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (ent >= lo) & ((ent < hi) | (hi == qs[-1]))
        if s.any():
            rows.append({"signal": "entropy", "bin_lo": float(lo), "bin_hi": float(hi), "n": int(s.sum()), "share": s.mean(),
                         "mean_signal": float(ent[s].mean()), "accept_rate": float(ag[s].mean())})
    from numpy import corrcoef
    stats = {"ece_top1": ece, "corr_top1_accept": float(corrcoef(p1, ag)[0, 1]),
             "corr_entropy_accept": float(corrcoef(ent, ag)[0, 1])}
    # how well a confidence threshold separates: accept rate above / below p1 = 0.5
    hi = p1 >= 0.5
    stats.update(share_p1_ge_05=float(hi.mean()), accept_p1_ge_05=float(ag[hi].mean()) if hi.any() else float("nan"),
                 accept_p1_lt_05=float(ag[~hi].mean()) if (~hi).any() else float("nan"))
    return rows, stats


def token_classes(tok, vocab: int) -> dict:
    import string
    punct = set(string.punctuation) | set("，。！？：；、「」（）—…")
    cls = np.zeros(vocab, dtype=np.int8)        # 0 word, 1 punctuation, 2 digit, 3 whitespace/newline
    for i in range(vocab):
        t = tok.decode([i])
        st = t.strip()
        if not st:
            cls[i] = 3
        elif any(ch.isdigit() for ch in st):
            cls[i] = 2
        elif all(ch in punct for ch in st):
            cls[i] = 1
    return cls


def c_token_type(d, k, cls) -> list:
    ag = agree(d, k)
    y = d["tokens"]
    m = gen_mask(d)
    inp = np.zeros(len(y), dtype=bool)
    for a, b, pr in d["reqs"]:
        inp[a:b] = np.isin(y[a:b], pr)
    rows = []
    names = {0: "word", 1: "punctuation", 2: "digit", 3: "whitespace"}
    for c, nm in names.items():
        s = m & (cls[y] == c)
        rows.append({"group": "type", "value": nm, "n": int(s.sum()), "share": s.sum() / m.sum(),
                     "accept_rate": float(ag[s].mean()) if s.any() else float("nan")})
    for v, nm in ((True, "in_prompt"), (False, "not_in_prompt")):
        s = m & (inp == v)
        rows.append({"group": "prompt", "value": nm, "n": int(s.sum()), "share": s.sum() / m.sum(),
                     "accept_rate": float(ag[s].mean()) if s.any() else float("nan")})
    return rows


def c_requests(d, k) -> tuple[dict, list]:
    ag = agree(d, k)
    per, ac = [], {}
    for a, b, _ in d["reqs"]:
        x = ag[a + 1:b].astype(float)
        if len(x) < 8:
            continue
        per.append(x.mean())
        xc = x - x.mean()
        den = (xc * xc).sum()
        for lag in range(1, 17):
            if len(x) > lag and den > 0:
                num, cnt = ac.get(lag, (0.0, 0.0))
                ac[lag] = (num + (xc[:-lag] * xc[lag:]).sum() / den * len(x), cnt + len(x))
    per = np.array(per)
    summary = {"requests": len(per), "mean": per.mean(), "std": per.std(), "p10": pct(per, 10), "p25": pct(per, 25),
               "p50": pct(per, 50), "p75": pct(per, 75), "p90": pct(per, 90), "min": per.min(), "max": per.max()}
    auto = [{"lag": lag, "autocorr": num / cnt} for lag, (num, cnt) in sorted(ac.items())]
    return summary, auto


def c_exit_layers(d, gamma=4) -> list:
    rows = []
    m = gen_mask(d)
    for k in d["exit_layers"].tolist()[:-1]:
        ag = agree(d, k)
        tok, rounds = 0, 0
        for a, b, _ in d["reqs"]:
            _, _, com = simulate_rounds(run_lengths(ag[a:b]), gamma)
            tok += com.sum()
            rounds += len(com)
        rows.append({"k": k, "alpha": float(ag[m].mean()), f"tokens_per_round_g{gamma}": tok / max(rounds, 1)})
    return rows


# ------------------------------------------------------------------------------ B (reference side)

def b_waste_ref(d, k, gammas=(2, 3, 4, 6, 8)) -> list:
    """Synchronous rounds of gamma drafts on the AR outputs: wasted drafts, rounds stopped although every draft
    was right, rejected share of deep-segment tokens, shallow steps per round."""
    ag = agree(d, k)
    rows = []
    for g in gammas:
        drafts = acc = full = rounds = deep_tok = 0
        for a, b, _ in d["reqs"]:
            starts, accs, com = simulate_rounds(run_lengths(ag[a:b]), g)
            nd = np.minimum(g, (b - a) - starts)             # drafts that exist before the trace ends
            drafts += nd.sum()
            acc += accs.sum()
            full += int((accs == nd).sum())
            rounds += len(starts)
            deep_tok += (nd + 1).sum()
        rows.append({"gamma": g, "rounds": rounds, "rejected_draft_share": 1 - acc / drafts,
                     "full_accept_round_share": full / rounds, "rejected_share_of_deep_tokens": (drafts - acc) / deep_tok,
                     "shallow_steps_per_round": drafts / rounds, "tokens_per_round": (acc + rounds) / rounds})
    return rows


# ------------------------------------------------------------------------------ E

def e_ngram(d, k, gamma, Gs) -> dict:
    ag = agree(d, k)
    y = d["tokens"].astype(np.int64)
    tot = {"rounds": 0, "full": 0, "committed": 0, **{G: 0 for G in Gs}, **{("x", G): 0 for G in Gs}, "x_rounds": 0}
    for a, b, pr in d["reqs"]:
        yy = y[a:b]
        starts, accs, com = simulate_rounds(run_lengths(ag[a:b]), gamma)
        rep = self_repetition(yy) > 0.5
        tot["rounds"] += len(starts)
        tot["committed"] += int(com.sum())
        if not rep:
            tot["x_rounds"] += len(starts)
        for c, n in zip(starts, accs):
            if n < gamma:
                continue
            tot["full"] += 1
            guess = ngram_guess(np.concatenate([pr.astype(np.int64), yy[:c + gamma]]), max(Gs))
            for G in Gs:
                g = match_len(guess[:G], yy[c + gamma:])
                tot[G] += g
                if not rep:
                    tot[("x", G)] += g
    return tot


def corpus_guess(suffix_src: np.ndarray, corpus: np.ndarray, G: int, nmax: int = 4) -> np.ndarray:
    """Longest suffix n-gram (n = nmax..1) of suffix_src that occurs in corpus (other requests' outputs, -1
    separated); the tokens after its last occurrence, up to G, cut at a separator."""
    from numpy.lib.stride_tricks import sliding_window_view
    for n in range(min(nmax, len(suffix_src)), 0, -1):
        if len(corpus) <= n:
            continue
        win = sliding_window_view(corpus[:-1], n)
        hits = np.flatnonzero((win == suffix_src[-n:]).all(1))
        if len(hits):
            j = int(hits[-1])
            cont = corpus[j + n:j + n + G]
            cut = np.flatnonzero(cont < 0)
            return cont[:cut[0]] if len(cut) else cont
    return corpus[:0]


def e_cross(d, k, gamma, Gs, B=16, batches=20, seed=0) -> dict:
    """Guess from the other requests of a batch: their generated tokens up to the same generation index."""
    ag = agree(d, k)
    y = d["tokens"].astype(np.int64)
    R = len(d["reqs"])
    rng = np.random.default_rng(seed)
    tot = {"rounds": 0, "full": 0, **{G: 0 for G in Gs}}
    for _ in range(batches):
        idx = rng.choice(R, min(B, R), replace=False)
        outs = [y[d["reqs"][i][0]:d["reqs"][i][1]] for i in idx]
        for qi, i in enumerate(idx):
            a, b, pr = d["reqs"][i]
            starts, accs, _ = simulate_rounds(run_lengths(ag[a:b]), gamma)
            tot["rounds"] += len(starts)
            for c, n in zip(starts, accs):
                if n < gamma:
                    continue
                tot["full"] += 1
                t = c + gamma
                corpus = np.concatenate([np.append(o[:t], -1) for j, o in enumerate(outs) if j != qi])
                ctx = np.concatenate([pr.astype(np.int64), outs[qi][:t]])
                guess = corpus_guess(ctx, corpus, max(Gs))
                for G in Gs:
                    tot[G] += match_len(guess[:G], outs[qi][t:])
    return tot


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["layerskip-llama3-8B", "layerskip-llama3-1B"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--guess", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip", nargs="*", default=[], help="sections to skip (C E F B)")
    args = ap.parse_args()
    out = Path(args.out)
    from common.data import load_tokenizer
    from common.utils import resolve_model_path
    rows = {n: [] for n in ("C_position", "C_calibration", "C_calibration_stats", "C_token_type", "C_requests",
                            "C_autocorr", "C_exit_layers", "E_guess", "F_coverage", "B_waste_ref")}
    for model in args.models:
        name = Path(model).name
        tok = load_tokenizer(resolve_model_path(model))
        cls = None
        for ds in args.tasks:
            p = out / "ref" / name / f"{ds}_L512.npz"
            if not p.exists():
                print(f"[outputs] missing {p}", flush=True)
                continue
            d = load(p)
            key = {"model": name, "task": ds}
            if "C" not in args.skip:
                if cls is None:
                    cls = token_classes(tok, len(d["heldout_freq"]))
                rows["C_position"] += [{**key, "k": args.k, **r} for r in c_position(d, args.k)]
                cal, st = c_calibration(d, args.k)
                rows["C_calibration"] += [{**key, "k": args.k, **r} for r in cal]
                rows["C_calibration_stats"].append({**key, "k": args.k, **st})
                rows["C_token_type"] += [{**key, "k": args.k, **r} for r in c_token_type(d, args.k, cls)]
                summ, auto = c_requests(d, args.k)
                rows["C_requests"].append({**key, "k": args.k, **summ})
                rows["C_autocorr"] += [{**key, "k": args.k, **r} for r in auto]
                rows["C_exit_layers"] += [{**key, **r} for r in c_exit_layers(d, args.gamma)]
            if "B" not in args.skip:
                rows["B_waste_ref"] += [{**key, "k": args.k, **r} for r in b_waste_ref(d, args.k)]
            if "E" not in args.skip:
                t = e_ngram(d, args.k, args.gamma, args.guess)
                x = e_cross(d, args.k, args.gamma, args.guess)
                for G in args.guess:
                    rows["E_guess"].append({**key, "source": "ngram", "G": G, "rounds": t["rounds"],
                                            "full_accept_share": t["full"] / t["rounds"],
                                            "committed_per_round": t["committed"] / t["rounds"],
                                            "extra_per_round": t[G] / t["rounds"],
                                            "extra_per_round_excl_repetitive": t[("x", G)] / max(t["x_rounds"], 1)})
                    rows["E_guess"].append({**key, "source": "cross_request_B16", "G": G, "rounds": x["rounds"],
                                            "full_accept_share": x["full"] / x["rounds"],
                                            "extra_per_round": x[G] / x["rounds"]})
            print(f"[outputs] {name} {ds} done", flush=True)
        if "F" not in args.skip:
            a = SimpleNamespace(length=512, cov_batch_sizes=[4, 16, 32], cov_batches=50, cov_R=[16, 64], cov_K=[8, 32],
                                cov_F=[0, 1024, 4096], seed=42)
            rows["F_coverage"] += analyze_coverage(out / "ref", model, args.tasks, a)
    for n, rs in rows.items():
        if rs:
            write_csv(out / f"{n}.csv", rs, tuple(k for k in ("model", "task", "k", "position", "signal", "bin_lo", "group",
                                                                "value", "lag", "gamma", "source", "G", "B", "R", "K", "F")
                                                  if k in rs[0]))


if __name__ == "__main__":
    main()
