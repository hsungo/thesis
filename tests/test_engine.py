"""Engines: speculation must be lossless, and the trace simulator must reproduce the engine."""

import pytest

from conftest import random_prompts

MAX_NEW = 40


def _outputs(res):
    return {c["pid"]: c["tokens"] for c in res["completions"]}


@pytest.fixture(scope="module")
def ar_reference(tiny_model):
    from common.engine import ARBatchEngine, RunConfig
    from common.traces import ExitProbe

    prompts = random_prompts(5)
    eng = ARBatchEngine(tiny_model, 2, 20, MAX_NEW, (2,), ignore_eos=True)
    probe = ExitProbe(tiny_model, list(range(1, tiny_model.n_layers + 1)))
    eng.probe = probe
    res = eng.run(prompts, RunConfig(mode="static", keep_outputs=True))
    return prompts, _outputs(res), probe


@pytest.mark.parametrize("k,gamma,B", [(1, 3, 2), (3, 5, 3), (5, 2, 1), (2, 0, 2), (4, 8, 5)])
def test_sync_layerskip_is_lossless(tiny_model, ar_reference, k, gamma, B):
    from common.engine import RunConfig, SyncSpecEngine

    prompts, ref, _ = ar_reference
    eng = SyncSpecEngine(tiny_model, B, 20, MAX_NEW, (2,), ignore_eos=True, exit_layer=k, gamma=gamma)
    res = eng.run(prompts, RunConfig(mode="static", keep_outputs=True))
    assert _outputs(res) == ref


@pytest.mark.parametrize("k,gamma", [(1, 3), (3, 5), (5, 4)])
def test_simulator_matches_engine(tiny_model, ar_reference, tmp_path, k, gamma):
    """Plan sec.3.1: accepted drafts replayed from one AR trace == real speculative run."""
    from common.engine import RunConfig, SyncSpecEngine
    from common.traces import load_traces, run_lengths, simulate_rounds

    prompts, _, probe = ar_reference
    path = probe.save(tmp_path / "t.npz", {"n_layers": tiny_model.n_layers})
    ts = load_traces(path)
    sim = sum(int(simulate_rounds(run_lengths(t.agree[:, ts.col(k)]), gamma)[1][:-1].sum()) for t in ts.traces)
    eng = SyncSpecEngine(tiny_model, 2, 20, MAX_NEW, (2,), ignore_eos=True, exit_layer=k, gamma=gamma)
    res = eng.run(prompts, RunConfig(mode="static"))
    assert res["metrics"]["spec"]["accepted"] == sim
    assert sim > 0  # the tiny model must actually exercise acceptance


def test_full_layer_column_equals_tokens(ar_reference):
    _, _, probe = ar_reference
    for r in probe.finished:
        assert (r["exit_argmax"][:, -1] == r["tokens"]).all()


def test_cyclic_run_converges_and_reports(tiny_model):
    from common.engine import ARBatchEngine, RunConfig

    eng = ARBatchEngine(tiny_model, 3, 20, 12, (2,), ignore_eos=True)
    res = eng.run(random_prompts(4), RunConfig(mode="cyclic", warmup_samples=2, max_eval_samples=6,
                                                converge=False))
    m = res["metrics"]
    assert m["stop_reason"] == "max_eval_samples"
    assert m["evaluated_completions"] == 6
    assert m["measured_tokens"] > 0 and m["throughput_tok_s"] > 0
    assert all(c["generated"] == 12 for c in res["completions"])


def test_eos_stops_generation(tiny_model, ar_reference):
    from common.engine import ARBatchEngine, RunConfig, SyncSpecEngine

    prompts, ref, _ = ar_reference
    # pick a token that appears in some reference output and treat it as EOS
    eos = ref["p0"][5]
    ar = _outputs(ARBatchEngine(tiny_model, 2, 20, MAX_NEW, (eos,)).run(prompts, RunConfig(mode="static", keep_outputs=True)))
    sp = _outputs(SyncSpecEngine(tiny_model, 2, 20, MAX_NEW, (eos,), exit_layer=3, gamma=4)
                  .run(prompts, RunConfig(mode="static", keep_outputs=True)))
    assert ar == sp
    assert ar["p0"][-1] == eos and len(ar["p0"]) <= 6


class _ToyTokenizer:
    """decode(ids) = one character per id; ids in `nl` decode to a newline."""

    def __init__(self, nl):
        self.nl = set(nl)

    def decode(self, ids, skip_special_tokens=False):
        return "".join("\n" if i in self.nl else chr(65 + i % 26) for i in ids)


def test_stop_checker_cuts_at_first_stop():
    from common.data import StopChecker

    sc = StopChecker(_ToyTokenizer({9}), ("\nB",))       # "\n" then id 1 ("B")
    out = [3, 4, 9, 5, 9, 1, 7, 9, 1]
    assert sc(out[:4], 4) is None
    assert sc(out[:6], 2) == 6                            # the token completing the first "\nB"
    assert sc(out[:5], 1) is None                         # "\n" alone is not the stop yet
    assert sc(out[:7], 7) == 6                            # several committed tokens at once
    assert not StopChecker(_ToyTokenizer(set()), ())


@pytest.mark.parametrize("k,gamma", [(2, 3), (4, 6)])
def test_stop_strings_keep_speculation_lossless(tiny_model, ar_reference, k, gamma):
    """A stop string that completes inside a multi-token commit must truncate the speculative
    output exactly where the AR run stops."""
    from common.data import StopChecker
    from common.engine import ARBatchEngine, RunConfig, SyncSpecEngine

    prompts, ref, _ = ar_reference
    nl = {ref["p0"][7], ref["p2"][11]}
    sc = StopChecker(_ToyTokenizer(nl), ("\n",))
    ar = ARBatchEngine(tiny_model, 2, 20, MAX_NEW, (2,), ignore_eos=False)
    ar.stop_checker = sc
    sp = SyncSpecEngine(tiny_model, 2, 20, MAX_NEW, (2,), exit_layer=k, gamma=gamma)
    sp.stop_checker = sc
    cfg = RunConfig(mode="static", keep_outputs=True)
    a, s = ar.run(prompts, cfg), sp.run(prompts, cfg)
    assert _outputs(a) == _outputs(s)
    assert len(_outputs(a)["p0"]) <= 8
    assert any(c["finish_reason"] == "stop" for c in a["completions"])
