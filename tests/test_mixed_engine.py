"""observation/v2/mixed_engine.py: the two-segment engine must be lossless (token-identical to
AR, fp32 CPU) for any verify schedule, including random partial verification sets."""

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import random_prompts

_V2 = Path(__file__).resolve().parents[1] / "observation" / "v2"
if not (_V2 / "mixed_engine.py").exists():   # observation/ is stored separately, not in this repo
    pytest.skip("observation/v2 is not in this checkout", allow_module_level=True)
sys.path.insert(0, str(_V2))

MAX_NEW = 40


def _outputs(res):
    return {c["pid"]: c["tokens"] for c in res["completions"]}


@pytest.fixture(scope="module")
def ar_ref(tiny_model):
    from common.engine import ARBatchEngine, RunConfig

    prompts = random_prompts(7, seed=3)
    eng = ARBatchEngine(tiny_model, 2, 20, MAX_NEW, (2,), ignore_eos=True)
    return prompts, _outputs(eng.run(prompts, RunConfig(mode="static", keep_outputs=True)))


class RandomPolicy:
    """Verify each slot with probability p (adversarial partial sets)."""

    def __init__(self, B, p, seed=0, max_draft=6):
        self.rng, self.p, self.max_draft = np.random.default_rng(seed), p, max_draft

    def decide(self, batch):
        from common.policies import PolicyDecision
        v = (self.rng.random(len(batch.active)) < self.p) & batch.active
        return PolicyDecision(v, batch.draft_counts.copy())

    def observe_verify(self, *a):
        pass

    def reset_slots(self, ids):
        pass


@pytest.mark.parametrize("k,B,policy", [(1, 2, "fixed3"), (2, 3, "syncfixed2"), (2, 3, "s3"), (3, 3, "fixed0"), (2, 4, "fsm"), (3, 3, "aspire"),
                                        (2, 3, "rand0.3"), (4, 5, "rand0.6"), (1, 1, "rand0.5")])
def test_mixed_engine_is_lossless(tiny_model, ar_ref, k, B, policy):
    from common.cost_model import LayerCostModel
    from common.engine import RunConfig
    from common.policies import make_policy
    from mixed_engine import MixedEngine

    prompts, ref = ar_ref
    if policy == "s3" or policy.startswith("syncfixed"):
        from simlib import SyncRulePolicy
        pol = SyncRulePolicy(B, "s3" if policy == "s3" else "fixed", LayerCostModel.layer_count(tiny_model.n_layers), k,
                             gamma=int(policy[9:]) if policy != "s3" else 4, d_max=6)
    elif policy.startswith("rand"):
        pol = RandomPolicy(B, float(policy[4:]))
    elif policy.startswith("fixed"):
        pol = make_policy("fixed", B, fixed_gamma=int(policy[5:]))
    else:
        pol = make_policy(policy, B, cost_model=LayerCostModel.layer_count(tiny_model.n_layers), exit_layer=k,
                          max_draft=6)
    eng = MixedEngine(tiny_model, B, 20, MAX_NEW, (2,), ignore_eos=True, exit_layer=k, policy=pol, d_max=6)
    res = eng.run(prompts, RunConfig(mode="static", keep_outputs=True))
    assert _outputs(res) == ref


@pytest.mark.parametrize("mode,B,gamma,k", [("immediate", 2, 3, 2), ("aligned_unfused", 3, 2, 1),
                                            ("aligned_fused", 2, 3, 2), ("aligned_fused", 4, 4, 3)])
def test_arrival_engine_is_lossless(tiny_model, ar_ref, mode, B, gamma, k):
    """B6 engine (separate or fused prefill, newcomers joining a running round) == AR."""
    from b6_real_arrivals import drive

    prompts, ref = ar_ref
    res = drive(tiny_model, prompts, B, gamma, mode, (), {}, k, rate=0.5, n_requests=len(prompts),
                warm_requests=0, seed=0, max_new=MAX_NEW)
    assert res["outputs"] and all(res["outputs"][pid] == ref[pid] for pid in res["outputs"])
