"""GPU memory planning: how many requests fit in the KV cache.

The KV cache is contiguous: `B x max_seq` slots are reserved up front, where
max_seq = longest prompt + max_new_tokens + speculation slack.  On a 24 GB RTX 4090
the 8B LayerSkip model leaves ~7.5 GiB after its 15 GiB of weights, i.e. ~60k cached
tokens (128 KiB / token): B~23 at 2.6k tokens per request, B=64 only below ~900.
The 1B model (32 KiB / token, 2.5 GiB weights) fits ~650k tokens.
Scripts accept `--batch-sizes ...,max` / `--batch-size auto` and skip batch sizes
that do not fit instead of crashing.
"""

from __future__ import annotations

from typing import Optional

import torch

from common.model import KVCache, LayerSkipLlama, ModelConfig

GiB = 2**30


def weight_bytes(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    return cfg.param_count * dtype_bytes


def activation_reserve_bytes(cfg: ModelConfig, prefill_len: int, step_tokens: int, dtype_bytes: int = 2) -> int:
    """Rough upper bound of transient memory: prefill of one prompt (MLP intermediates,
    repeated K/V for causal SDPA) or one decode step (logits, fp32 sampling), plus
    a fixed margin for the CUDA context / allocator fragmentation."""
    prefill = prefill_len * (4 * cfg.dim + 3 * cfg.intermediate_size) * dtype_bytes
    prefill += prefill_len * cfg.n_heads * cfg.head_dim * 2 * dtype_bytes
    step = step_tokens * cfg.vocab_size * (dtype_bytes + 4)
    return int(1.5 * max(prefill, step)) + 512 * 2**20


def free_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def max_batch(model: LayerSkipLlama, max_seq: int, max_prompt: int, step_tokens_per_request: int = 1,
              reserve_bytes: Optional[int] = None, cpu_default: int = 8) -> int:
    """Largest batch whose KV cache (B x max_seq) + activations fit in free memory.
    Frees the model's current cache first."""
    model.free_cache()
    free = free_bytes(model.device)
    if free is None:
        return cpu_default
    max_seq = (max_seq + 63) // 64 * 64  # allocate_cache rounds up the same way
    per_req = KVCache.bytes_needed(model.cfg, 1, max_seq, model.dtype.itemsize)
    # reserve grows with B (decode logits), so search downward from the KV-only bound
    b = max(1, free // per_req)
    while b > 1:
        res = reserve_bytes if reserve_bytes is not None else activation_reserve_bytes(
            model.cfg, max_prompt, b * step_tokens_per_request)
        if b * per_req + res <= free:
            break
        b = int(b * 0.95) if b > 20 else b - 1
    return int(max(b, 1))


def fits(model: LayerSkipLlama, batch: int, max_seq: int, max_prompt: int, step_tokens_per_request: int = 1) -> bool:
    if model.device.type != "cuda":
        return True
    return batch <= max_batch(model, max_seq, max_prompt, step_tokens_per_request)


def resolve_batch_sizes(spec: str, model: LayerSkipLlama, max_seq: int, max_prompt: int,
                        step_tokens_per_request: int = 1) -> tuple[list[int], list[int]]:
    """'1,4,16,max' -> (sizes that fit, sizes skipped).  'auto' and 'max' both mean
    the largest batch that fits."""
    want: list[int] = []
    for part in str(spec).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("max", "auto"):
            want.append(max_batch(model, max_seq, max_prompt, step_tokens_per_request))
        else:
            want.append(int(part))
    want = sorted(set(want))
    if model.device.type != "cuda":
        return want, []
    limit = max_batch(model, max_seq, max_prompt, step_tokens_per_request)
    return [b for b in want if b <= limit], [b for b in want if b > limit]


def plan_table(cfg: ModelConfig, gpu_bytes: int, seqs: list[int], dtype_bytes: int = 2) -> list[dict]:
    """Max batch per sequence length for a GPU of `gpu_bytes` (used by scripts/check_env.py)."""
    rows = []
    free = gpu_bytes - weight_bytes(cfg, dtype_bytes) - 600 * 2**20  # CUDA context
    for s in seqs:
        per = KVCache.bytes_needed(cfg, 1, s, dtype_bytes)
        res = activation_reserve_bytes(cfg, min(s, 4096), 64)
        rows.append({"max_seq": s, "max_batch": max(0, int((free - res) // per)),
                     "kv_tokens_total": max(0, int((free - res) // (per // s)))})
    return rows
