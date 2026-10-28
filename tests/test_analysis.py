"""Pure-numpy pieces: traces, cost model, policies, simulator, memory planner, data truncation."""

import numpy as np
import pytest


# ----------------------------------------------------------------------------- traces

def test_run_lengths_brute_force():
    from common.traces import run_lengths

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.random(int(rng.integers(1, 40))) < 0.7
        ref = []
        for t in range(len(a)):
            n = 0
            while t + n < len(a) and a[t + n]:
                n += 1
            ref.append(n)
        assert run_lengths(a).tolist() == ref


def test_simulate_rounds_covers_sequence():
    from common.traces import run_lengths, simulate_rounds

    rng = np.random.default_rng(1)
    for g in [0, 1, 3, 8]:
        a = rng.random(100) < 0.8
        starts, acc, committed = simulate_rounds(run_lengths(a), g)
        assert committed.sum() == len(a) - 1           # every token after the prefill one is committed once
        assert (acc <= g).all() and starts[0] == 1
        assert (np.diff(starts) == acc[:-1] + 1).all()


# ----------------------------------------------------------------------------- cost model

def test_cost_model_fit_recovers_coefficients():
    from common.cost_model import LayerCostModel

    rng = np.random.default_rng(0)
    bw, bmlp, battn, hw, ht = 0.3, 0.002, 1e-5, 0.8, 0.004
    lr = [{"n": n, "kv": kv, "ms": bw + bmlp * n + battn * kv} for n in [1, 4, 16, 64, 256] for kv in [100, 1000, 30000]]
    hr = [{"n": n, "ms": hw + ht * n} for n in [1, 8, 64, 512]]
    sr = [{"ms": 1.0 + 5.0, "layers_ms": 5.0, "head_ms": 0.0}]
    cm = LayerCostModel.fit(lr, hr, sr, n_layers=32, form="linear")
    assert np.allclose([cm.bw, cm.bmlp, cm.battn, cm.head_w, cm.head_tok, cm.beta0], [bw, bmlp, battn, hw, ht, 1.0], rtol=1e-6)
    # AR step == verify with gamma 0 for any k;  deep layers skipped when no token goes through them
    assert np.isclose(cm.sync_verify(5, 0, 8, 100), cm.ar_step(8, 100))
    assert np.isclose(cm.two_segment(5, 8, 800, 0, 0, 8), cm.sync_draft(5, 8, 100))


def test_hinge_fit_recovers_knee():
    """GEMMs are flat (weight-load bound) up to the knee, then linear in tokens."""
    from common.cost_model import LayerCostModel

    bw, bmlp, knee, battn, blen = 0.6, 0.003, 96.0, 3e-6, 3e-5
    lr = [{"B": B, "n": B * T, "kv": B * L, "ms": bw + bmlp * max(B * T - knee, 0) + battn * B * L + blen * L}
          for B in [1, 4, 16, 64] for T in [1, 4, 17] for L in [128, 1024, 4096]]
    hr = [{"n": n, "ms": 1.0 + 0.006 * max(n - 64, 0)} for n in [1, 16, 64, 128, 512, 1024]]
    cm = LayerCostModel.fit(lr, hr, [], n_layers=32, form="hinge")
    assert cm.knee == knee and cm.head_knee == 64
    assert np.allclose([cm.bw, cm.bmlp, cm.battn, cm.blen], [bw, bmlp, battn, blen], rtol=1e-4)
    # a verify step below the knee costs the same as a draft step through the same layers
    assert np.isclose(cm.layer(8, 8 * 512, 512), cm.layer(64, 8 * 512, 512))


def test_nnls_never_negative():
    from common.cost_model import nnls_lstsq

    X = np.c_[np.ones(20), np.arange(20), np.arange(20) ** 2]
    y = 5 - 0.1 * np.arange(20)
    assert (nnls_lstsq(X, y) >= 0).all()


# ----------------------------------------------------------------------------- policies / simulator

def test_expected_token_targets_is_argmax():
    from common.policies import best_gamma_expected, expected_token_targets

    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.05, 0.98, 300)
    cost = rng.uniform(0.02, 0.9, 300)
    got = expected_token_targets(alpha, cost, 0, 16)
    ref = [best_gamma_expected(a, c, 16) for a, c in zip(alpha, cost)]
    assert got.tolist() == ref


def test_fixed_policy_is_synchronous(tmp_path):
    from common.cost_model import LayerCostModel
    from common.policies import make_policy
    from common.simulator import AsyncTraceSimulator
    from common.traces import Trace, TraceSet

    rng = np.random.default_rng(0)
    traces = [Trace(f"r{i}", 50, np.zeros(200, dtype=np.int32), rng.random((200, 3)) < 0.7) for i in range(10)]
    ts = TraceSet([2, 4, 8], 8, traces, {})
    cm = LayerCostModel.layer_count(8)
    for g in [2, 4]:
        res = AsyncTraceSimulator(ts, 4, 8, cm, make_policy("fixed", 8, fixed_gamma=g)).run(300, 30)
        assert res["deep_exec_fraction"] == pytest.approx(1 / (g + 1), abs=0.01)
        assert res["mean_deep_batch_fraction_of_B"] == pytest.approx(1.0)
    asp = AsyncTraceSimulator(ts, 4, 8, cm, make_policy("aspire", 8, cost_model=cm, exit_layer=4)).run(300, 30)
    assert 0 < asp["deep_exec_fraction"] <= 1 and asp["tokens_per_ms"] > 0


# ----------------------------------------------------------------------------- memory / data

def test_plan_table_8b_on_24gb():
    from common.memory import plan_table
    from common.model import ModelConfig

    cfg = ModelConfig(vocab_size=128256, dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
                      intermediate_size=14336, norm_eps=1e-5, rope_theta=5e5, max_position_embeddings=8192)
    rows = plan_table(cfg, 24 * 2**30, [1024, 2048, 4096])
    assert 45 < rows[0]["max_batch"] < 75          # ~7.5 GiB of KV at 128 KiB/token
    assert rows[0]["max_batch"] > rows[1]["max_batch"] > rows[2]["max_batch"]


class _FakeTok:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1] + [ord(c) for c in text]}


def test_truncate_keeps_bos_and_tail(tmp_path):
    import json

    from common.data import load_prompts

    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": s}) for s in ["abcdefghij", "xyz"]))
    out = load_prompts("custom_jsonl", _FakeTok(), 10, data_path=str(p), truncate_to=5, shuffle=False)
    assert len(out) == 1                                      # 'xyz' (4 tokens) is too short
    assert out[0].tokens == [1] + [ord(c) for c in "ghij"]    # BOS + last 4 tokens
