"""Checks on the real LayerSkip checkpoints in ./models (slow; run with `pytest -m slow`).
On a GPU server also run them with `--device cuda` semantics via EESD_TEST_DEVICE=cuda."""

import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "layerskip-llama3-1B"
DEVICE = os.environ.get("EESD_TEST_DEVICE", "cpu")

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not MODEL.exists(), reason="checkpoint not downloaded")]


def test_1b_matches_hf_and_is_lossless():
    from transformers import AutoTokenizer, LlamaForCausalLM

    from common.data import Prompt
    from common.engine import ARBatchEngine, RunConfig, SyncSpecEngine
    from common.model import LayerSkipLlama

    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = LayerSkipLlama.from_pretrained(MODEL, device=DEVICE, dtype=dtype, max_seq=1024)
    text = "The capital of France is Paris. The capital of Germany is"
    ids = tok(text)["input_ids"]
    m.allocate_cache(1, 256)
    with torch.inference_mode():
        logits, _ = m.prefill(ids, 0)
        if DEVICE == "cpu":
            hf = LlamaForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
            ref = hf(torch.tensor([ids])).logits[0, -1]
            assert torch.allclose(logits, ref, atol=1e-3)
    assert tok.decode([int(logits.argmax())]).strip() == "Berlin"

    prompts = [Prompt(f"p{i}", tok(t)["input_ids"]) for i, t in enumerate([
        "Question: Tom has 3 apples and buys 5 more. How many apples does he have?\nAnswer:",
        "def fibonacci(n):\n", "The history of the Roman Empire begins with"])]
    cfg = RunConfig(mode="static", keep_outputs=True)
    ar = ARBatchEngine(m, 3, 40, 24, (128001,), ignore_eos=True).run(prompts, cfg)
    sp = SyncSpecEngine(m, 3, 40, 24, (128001,), ignore_eos=True, exit_layer=6, gamma=4).run(prompts, cfg)
    same = [a["tokens"] == b["tokens"] for a, b in zip(sorted(ar["completions"], key=lambda c: c["pid"]),
                                                       sorted(sp["completions"], key=lambda c: c["pid"]))]
    if DEVICE == "cpu":
        assert all(same)
    else:  # bf16: tolerate rare near-tie flips, but most requests must match exactly
        assert sum(same) >= len(same) - 1
