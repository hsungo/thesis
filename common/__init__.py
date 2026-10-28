"""Shared building blocks for the batched early-exit self-speculative decoding thesis.

Everything here is pure PyTorch (no FlashInfer / no HF modeling code), so the same
code runs on Linux servers and on Windows.  Modules:

    model       LayerSkip-Llama with an explicit per-layer forward and a static KV cache
    engine      batched decoding engines (autoregressive, synchronous LayerSkip) + cyclic driver
    data        prompt datasets (prompt formats follow LayerSkip / DEL)
    cost_model  per-layer, batch-aware step-time model (plan sec.S2 / O5)
    traces      agreement traces from one AR pass and the round simulator (plan sec.3.1)
    timing      CUDA-event / wall-clock timers
    utils       IO helpers
"""
