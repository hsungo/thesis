"""solutions/ (EESD engine): losslessness of every switch combination + unit tests.

All on CPU / fp32 with the tiny random checkpoint of conftest.py (no GPU, no downloads).
The cost model is `synthetic_cost_model` (made-up GPU constants): it only drives decisions here.
"""

import itertools

import numpy as np
import pytest
import torch

from conftest import random_prompts

MAX_NEW = 40


def _outputs(res):
    out = {}
    for c in res["completions"]:
        out.setdefault(c["pid"], c["tokens"])
    return out


@pytest.fixture(scope="module")
def ar_ref(tiny_model):
    from common.engine import ARBatchEngine, RunConfig

    prompts = random_prompts(7, seed=3)
    eng = ARBatchEngine(tiny_model, 2, 20, MAX_NEW, (2,), ignore_eos=True)
    return prompts, _outputs(eng.run(prompts, RunConfig(mode="static", keep_outputs=True)))


def _engine(model, cfg, B=3, cost="roofline"):
    from solutions.cost_model import synthetic_cost_model
    from solutions.engine import EESDEngine

    return EESDEngine(model, B, 24, MAX_NEW, (), cfg, synthetic_cost_model(model.cfg, cost))


def _cfg(**kw):
    from solutions.engine import EngineConfig

    base = dict(k=2, d_max=6, gamma=3, probe_every=2, k_cands=(1, 2, 3, 4, 5), max_wait=100, m3_chunk_cap=3)
    base.update(kw)
    return EngineConfig(**base)


# ------------------------------------------------------------------------------ losslessness

COMBOS = list(itertools.product(["fixed", "aspire", "m1"], ["fixed", "m2"], ["separate", "m3"], ["roofline", "linear"]))


@pytest.mark.parametrize("sched,exit_,prefill,cost", COMBOS)
def test_every_combination_is_lossless(tiny_model, ar_ref, sched, exit_, prefill, cost):
    prompts, ref = ar_ref
    cfg = _cfg(sched=sched, exit=exit_, prefill=prefill, m2_force="cycle" if exit_ == "m2" else None)
    res = _engine(tiny_model, cfg, cost=cost).run(prompts, "static", keep_outputs=True)
    out = _outputs(res)
    assert set(out) == set(ref)
    assert out == ref


def test_ar_mode_is_lossless(tiny_model, ar_ref):
    prompts, ref = ar_ref
    res = _engine(tiny_model, _cfg(mode="ar")).run(prompts, "static", keep_outputs=True)
    assert _outputs(res) == ref


@pytest.mark.parametrize("sched", ["fixed", "aspire", "m1"])
def test_pad_bucketing_and_cyclic_driver_are_lossless(tiny_model, ar_ref, sched):
    prompts, ref = ar_ref
    res = _engine(tiny_model, _cfg(sched=sched, bucketing="pad", prefill="m3")).run(
        prompts, "cyclic", warmup_samples=2, warmup_per_slot=0, min_eval=4, min_eval_per_slot=0, max_eval=10,
        keep_outputs=True)
    out = res["completions"]
    assert len(out) >= 10
    assert all(c["tokens"] == ref[c["pid"]] for c in out)


def test_exit_switch_keeps_outputs(tiny_model, ar_ref):
    prompts, ref = ar_ref
    res = _engine(tiny_model, _cfg(sched="m1", exit="m2", m2_force="cycle")).run(prompts, "static", keep_outputs=True)
    ks = {k for _, k in res["metrics"]["k_timeline"]}
    assert len(ks) >= 3, "the forced cycle must actually switch the exit layer"
    assert _outputs(res) == ref


def test_m2_probe_estimates_are_probabilities(tiny_model, ar_ref):
    prompts, _ = ar_ref
    res = _engine(tiny_model, _cfg(sched="fixed", exit="m2")).run(prompts, "static")
    alpha = res["metrics"]["k_alpha"]
    assert res["metrics"]["probe_steps"] > 0
    assert all(a is None or 0.0 <= a <= 1.0 for a in alpha.values())


# ------------------------------------------------------------------------------ unit tests

def test_packed_segmentation(tiny_model):
    from solutions.packed import Block, build_packed, qo_indptr

    tiny_model.allocate_cache(4, 128)
    blocks = [Block(2, 10, 3, "a"), Block(0, 5, 1, "b"), Block(1, 7, 2, "c"), Block(3, 9, 1, "d")]
    pb = build_packed(tiny_model, blocks, "exact")
    assert pb.n_tokens == 7
    assert list(pb.offsets) == list(qo_indptr([b.T for b in pb.blocks]))
    # bucket-major by T, rows sorted inside a bucket; every block is contiguous in the packed axis
    assert [b.T for b in pb.blocks] == [1, 1, 2, 3]
    assert [bk.T for bk in pb.buckets] == [1, 2, 3]
    pos = pb.pos.tolist()
    for i, b in enumerate(pb.blocks):
        assert pos[pb.offsets[i]:pb.offsets[i + 1]] == list(range(b.pos0, b.pos0 + b.T))
    assert pb.buckets[0].rows is not None                # rows 0 and 3: not contiguous -> gathered
    pad = build_packed(tiny_model, blocks, "pad")
    assert len(pad.buckets) == 1 and pad.buckets[0].T == 3
    idx = pad.buckets[0].pad_index.view(-1).tolist()
    assert sorted(i for i in idx if i >= 0) == list(range(7))


def test_packed_forward_matches_layer_by_layer(tiny_model):
    """A packed call over ragged blocks == running every block alone with the model's own layer()."""
    from solutions.packed import Block, build_packed, packed_layers

    m = tiny_model
    m.allocate_cache(3, 64)
    g = torch.Generator().manual_seed(0)
    for row in range(3):                                   # some cached prefix in every row
        m.prefill(torch.randint(3, 60, (8,), generator=g).tolist(), row)
    blocks = [Block(0, 8, 1), Block(1, 8, 4), Block(2, 8, 2)]
    x = torch.randn(7, m.cfg.dim, generator=g)
    pb = build_packed(m, blocks, "exact")
    order = [blocks.index(b) for b in pb.blocks]
    starts = np.concatenate([[0], np.cumsum([b.T for b in blocks])])
    xp = torch.cat([x[starts[i]:starts[i + 1]] for i in order])
    yp = packed_layers(m, xp, pb, 0, m.n_layers)
    for j, i in enumerate(order):
        b = blocks[i]
        pos = torch.arange(b.pos0, b.pos0 + b.T).unsqueeze(0)
        ctx = m.make_ctx(pos, b0=b.row, max_pos=b.pos0 + b.T - 1)
        ref = m.forward_layers(x[starts[i]:starts[i + 1]].unsqueeze(0), 0, m.n_layers, ctx)[0]
        got = yp[pb.offsets[j]:pb.offsets[j + 1]]
        assert torch.allclose(got, ref, atol=1e-5), (i, (got - ref).abs().max())


def test_rollback_moves_no_kv(tiny_model, ar_ref):
    """Rejected drafts are rolled back by the position pointer only: a step never rewrites cached
    positions below a slot's committed position, and pos always = prompt_len + generated - 1."""
    from solutions.engine import DECODE, Request

    prompts, _ = ar_ref
    eng = _engine(tiny_model, _cfg(sched="m1"), B=2)
    for s in range(2):
        eng.assign(s, Request(prompts[s].pid, list(prompts[s].tokens)))
    m = tiny_model
    saw_reject = False
    for _ in range(25):
        live = [s for s in range(2) if eng.state[s] == DECODE and eng.req[s].finish is None]
        if not live:
            break
        before = {s: int(eng.pos[s]) for s in live}
        snap = {s: [m.cache.k[l][s, :, :before[s]].clone() for l in range(m.n_layers)] for s in live}
        drafted = eng.d.copy()
        eng.step()
        for s in live:
            for l in range(m.n_layers):
                assert torch.equal(m.cache.k[l][s, :, :before[s]], snap[s][l])
            r = eng.req[s]
            assert eng.pos[s] == len(r.prompt) + len(r.out) - 1
            if drafted[s] and eng.pos[s] - before[s] < drafted[s] + 1 and eng.d[s] == 0:
                saw_reject = True
    assert saw_reject, "the test needs at least one round with a rejected draft"


def test_piggyback_prefill_writes_the_same_kv_as_a_normal_prefill(tiny_model, ar_ref):
    from solutions.engine import DECODE, PREFILL, Request

    prompts, _ = ar_ref
    m = tiny_model
    eng = _engine(m, _cfg(sched="m1", prefill="m3", m3_chunk_cap=3), B=3)
    eng.assign(0, Request(prompts[0].pid, list(prompts[0].tokens)))       # decoding slot (separate prefill)
    p = prompts[4]
    eng.assign(1, Request(p.pid, list(p.tokens)))
    assert eng.state[1] == PREFILL, "the request must be piggybacked"
    n_chunks = 0
    while eng.state[1] == PREFILL:
        eng.step()
        n_chunks += 1
        assert n_chunks < 50
    assert eng.req[1].piggybacked == len(p.tokens) and len(p.tokens) > 3
    logits, _ = m.prefill(list(p.tokens), 2)                              # reference into the free row 2
    n = len(p.tokens)
    for l in range(m.n_layers):
        assert torch.allclose(m.cache.k[l][1, :, :n], m.cache.k[l][2, :, :n], atol=1e-5)
        assert torch.allclose(m.cache.v[l][1, :, :n], m.cache.v[l][2, :, :n], atol=1e-5)
    assert eng.req[1].out[0] == int(logits.argmax())
    assert eng.state[1] == DECODE


def test_shallow_steps_do_not_synchronise(tiny_model, ar_ref, monkeypatch):
    """No device->host read during shallow steps; exactly one per deep step (spec, M1)."""
    from solutions.engine import Request

    prompts, _ = ar_ref
    eng = _engine(tiny_model, _cfg(sched="m1"), B=2)
    for s in range(2):
        eng.assign(s, Request(prompts[s].pid, list(prompts[s].tokens)))
    calls = []
    orig_cpu, orig_item = torch.Tensor.cpu, torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: (calls.append("cpu"), orig_cpu(self, *a, **k))[1])
    monkeypatch.setattr(torch.Tensor, "item", lambda self: (calls.append("item"), orig_item(self))[1])
    kinds = []
    for _ in range(12):
        calls.clear()
        n_rec = len(eng.records)
        eng.step()
        kind = eng.records[n_rec].kind
        kinds.append(kind)
        n_cpu = calls.count("cpu") + calls.count("item")
        if kind == "shallow":
            assert n_cpu == 0
        else:
            assert n_cpu == 1
    assert "shallow" in kinds and "deep" in kinds


def test_m1_rule_direction():
    from common.model import ModelConfig
    from solutions.cost_model import synthetic_cost_model
    from solutions.schedulers import M1Scheduler

    cfg = ModelConfig(vocab_size=128256, dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
                      intermediate_size=14336, norm_eps=1e-5, rope_theta=5e5, max_position_embeddings=8192)
    cm = synthetic_cost_model(cfg)
    B = 16
    d, ctx, dec = np.full(B, 2), np.full(B, 1000), np.ones(B, dtype=bool)
    s = M1Scheduler(B, cm, 3, d_max=16)
    s.ema.alpha[:] = 0.98
    assert not s.decide(d, ctx, dec, 2).any(), "high acceptance: keep drafting"
    s.ema.alpha[:] = 0.05
    assert s.decide(d, ctx, dec, 2).all(), "low acceptance: verify now"
    d2 = d.copy()
    d2[0] = 16
    s.ema.alpha[:] = 0.98
    assert s.decide(d2, ctx, dec, 2).all(), "d_max forces verification"
    d3 = d.copy()
    d3[1] = 0
    v = s.decide(d3, ctx, dec, 40)                         # sla forces it; d=0 slots do not verify
    assert v[0] and not v[1]


def test_cost_model_formula_and_io(tmp_path):
    from common.model import ModelConfig
    from solutions.cost_model import GPUParams, ModelDims, RooflineCostModel, load_cost_model, save_json

    cfg = ModelConfig(vocab_size=1000, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
                      intermediate_size=128, norm_eps=1e-5, rope_theta=1e4, max_position_embeddings=512)
    dims = ModelDims.from_config(cfg)
    gpu = GPUParams(bw=1e6, flops=1e8, t_op=0.01, n_op_layer=50, n_op_bucket=10, n_op_head=8, n_op_segment=5)
    cm = RooflineCostModel(gpu, dims, n_max=64)
    W, P = dims.layer_bytes, dims.layer_params
    for n in (1, 10, 64, 200):                              # 200 > n_max: linear extrapolation
        assert np.isclose(cm.layer_gemm_ms(n), max(W / gpu.bw, 2 * P * n / gpu.flops))
    t = cm.t_segment(3, 10, 500, buckets=2)
    manual = 3 * (max(W / gpu.bw, 2 * P * 10 / gpu.flops) + 500 * dims.kv_bytes_per_pos / gpu.bw + 60 * 0.01) + 0.05
    assert np.isclose(t, manual)
    assert np.isclose(cm.knee(), gpu.flops * W / (2 * P * gpu.bw))
    save_json(tmp_path / "c.json", {"gpu": gpu, "linear": {"a": 1, "b": 2, "c": 3, "head_a": 4, "head_b": 5,
                                                          "n_layers": 4}})
    r = load_cost_model(tmp_path / "c.json", cfg, "roofline")
    assert np.isclose(r.t_prefill(32, 32), cm.t_prefill(32, 32))
    lin = load_cost_model(tmp_path / "c.json", cfg, "linear")
    assert np.isclose(lin.layer_gemm_ms(10), 1 + 2 * 10)


# ------------------------------------------------------------------------------ ablation driver (no GPU)

def test_ablation_config_names():
    import argparse

    from solutions.run_ablation import BASE, MECH, engine_config

    args = argparse.Namespace(k=3, d_max=16, sla_steps=32, probe_every=20, k_cands=[2, 3, 4], switch_margin=0.05,
                              max_wait=3, prefill_chunk=512, bucketing="exact", base_gamma=4)
    seen = {}
    for name in BASE + MECH + ["fixed-g3-k6"]:
        cfg, cost = engine_config(name, args)
        seen[name] = (cfg.mode, cfg.sched, cfg.exit, cfg.prefill, cfg.gamma, cfg.k, cost)
    assert seen["ar"][0] == "ar"
    assert seen["fixed-g6"][1:5] == ("fixed", "fixed", "separate", 6)
    assert seen["fixed-g3-k6"][4:6] == (3, 6)
    assert seen["all"][1:4] == ("m1", "m2", "m3")
    assert seen["m2+m3"][1:4] == ("fixed", "m2", "m3")
    assert seen["m1-linear"][1] == "m1" and seen["m1-linear"][6] == "linear"


def test_ablation_summary_pairs_with_ar(tmp_path):
    import json

    from solutions.run_ablation import summarize

    for rep in range(2):
        for cfg, thr in (("ar", 100.0 + rep), ("m1", 180.0 + 2 * rep)):
            d = tmp_path / "runs" / "M" / "gsm8k_L256" / "B4"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{cfg}_rep{rep}.json").write_text(json.dumps({
                "run": {"model": "M", "dataset": "gsm8k", "length": 256, "B": 4, "config": cfg, "cost_model": "roofline",
                        "rep": rep},
                "metrics": {"throughput_tok_s": thr, "e2e_throughput_tok_s": thr / 2, "deep_steps_per_token": 0.1,
                            "k_final": 3, "k_timeline": [[0, 3]]}}))
    summarize(tmp_path)
    import csv
    agg = {r["config"]: r for r in csv.DictReader(open(tmp_path / "summary.csv"))}
    assert abs(float(agg["m1"]["speedup_vs_ar_mean"]) - np.mean([180 / 100, 182 / 101])) < 1e-9
    assert float(agg["m1"]["speedup_vs_ar_std"]) > 0
    assert (tmp_path / "SUMMARY.md").exists()
