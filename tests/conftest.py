"""Shared fixtures: tiny random Llama checkpoints written by transformers itself, so the
tests exercise the real loading path without any downloaded model or GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LLAMA3_ROPE = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


def make_tiny_checkpoint(path: Path, *, layers: int = 6, vocab: int = 64, tie: bool = False,
                         rope: dict | None = None, seed: int = 0) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=tie, bos_token_id=1, eos_token_id=2,
        rope_parameters=rope or {"rope_type": "default", "rope_theta": 10000.0},
    )
    LlamaForCausalLM(cfg).eval().save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    return make_tiny_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tiny_model(tiny_dir):
    from common.model import LayerSkipLlama

    return LayerSkipLlama.from_pretrained(tiny_dir, device="cpu", dtype=torch.float32, max_seq=512)


def random_prompts(n: int, vocab: int = 64, seed: int = 0, min_len: int = 5, max_len: int = 20):
    from common.data import Prompt

    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        L = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        toks = [1] + torch.randint(3, vocab, (L - 1,), generator=g).tolist()
        out.append(Prompt(pid=f"p{i}", tokens=toks))
    return out
