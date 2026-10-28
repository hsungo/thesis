"""Synthetic-step timing harness (O5 per-layer profile, O7 shallow-reuse measurement).

A "step" is a list of segments.  Each segment runs layers [start, end) for
`batch` requests (cache rows 0..batch-1) with `tokens` query tokens per request
placed at positions ctx .. ctx+tokens-1.  The KV cache content is irrelevant
for timing, so it is never filled; only the attended length matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common.model import KVCache, LayerSkipLlama
from common.timing import Interval, LayerTimer, sync


@dataclass
class Segment:
    start: int
    end: int
    batch: int
    tokens: int          # query tokens per request
    pos_offset: int = 0  # first query position = ctx + pos_offset


class StepProfiler:
    def __init__(self, model: LayerSkipLlama, warmup: int = 3, reps: int = 10):
        self.m = model
        self.dev = model.device
        self.warmup = warmup
        self.reps = reps
        self._cache_shape = None

    def ensure_cache(self, batch: int, seq: int) -> bool:
        """(Re)allocate the cache if needed.  Returns False if it does not fit in memory."""
        seq = int(np.ceil(seq / 64) * 64)
        if self._cache_shape and self._cache_shape[0] >= batch and self._cache_shape[1] >= seq:
            return True
        need = KVCache.bytes_needed(self.m.cfg, batch, seq)
        self._cache_shape = None
        if self.dev.type == "cuda":
            self.m.free_cache()
            free, _ = torch.cuda.mem_get_info(self.dev)
            if need > 0.85 * free:
                return False
        try:
            self.m.allocate_cache(batch, seq)
        except torch.OutOfMemoryError:
            self.m.free_cache()
            return False
        self._cache_shape = (batch, seq)
        return True

    @torch.inference_mode()
    def time_step(self, segments: list[Segment], ctx: int, head_tokens: int, per_layer: bool = True) -> dict:
        """Median timings over `reps` runs: {'total_ms', 'head_ms', 'layers_ms': {i: ms},
        'segment_ms': [ms per segment], 'kv': [kv read per layer per segment]}."""
        m, dev = self.m, self.dev
        max_b = max(s.batch for s in segments)
        max_pos = ctx + max(s.pos_offset + s.tokens for s in segments)
        if not self.ensure_cache(max_b, max_pos + 1):
            raise torch.OutOfMemoryError(f"KV cache for batch={max_b} seq={max_pos} does not fit")
        gen = torch.Generator(device=dev).manual_seed(0)
        inputs, ctxs = [], []
        for s in segments:
            tok = torch.randint(0, m.cfg.vocab_size, (s.batch, s.tokens), device=dev, generator=gen)
            pos = (ctx + s.pos_offset + torch.arange(s.tokens, device=dev)).unsqueeze(0).expand(s.batch, s.tokens)
            inputs.append(m.embed(tok))
            ctxs.append(m.make_ctx(pos.contiguous(), 0, ctx + s.pos_offset + s.tokens - 1))
        head_in = torch.randn(head_tokens, m.cfg.dim, device=dev, dtype=m.dtype, generator=gen) if head_tokens else None

        totals, heads, segs, layers = [], [], [], []
        for rep in range(self.warmup + self.reps):
            lt = LayerTimer(dev) if per_layer else None
            seg_iv = [Interval(dev) for _ in segments]
            tot = Interval(dev).start()
            for s, x, c, iv in zip(segments, inputs, ctxs, seg_iv):
                iv.start()
                m.forward_layers(x, s.start, s.end, c, layer_hook=lt)
                iv.stop()
            hiv = Interval(dev).start()
            if head_in is not None:
                m.head(head_in).argmax(-1)
            hiv.stop()
            tot.stop()
            sync(dev)
            if rep >= self.warmup:
                totals.append(tot.ms)
                heads.append(hiv.ms)
                segs.append([iv.ms for iv in seg_iv])
                if lt is not None:
                    layers.append(lt.results())
        out = {
            "total_ms": float(np.median(totals)),
            "head_ms": float(np.median(heads)),
            "segment_ms": np.median(np.asarray(segs), axis=0).tolist(),
            "kv": [c.kv_read for c in ctxs],
        }
        if layers:
            keys = layers[0].keys()
            out["layers_ms"] = {int(i): float(np.median([d[i] for d in layers])) for i in keys}
        return out
