"""Layer-streamed LayerSkipLlama: generate outputs of a model that does not fit in GPU memory (8B on 8 GB).

Only for output-based analyses - its step times are dominated by PCIe copies and are never reported.

Weights of every decoder layer live in pinned host memory; two GPU buffers hold the current and the next
layer.  Layer i+1 (wrapping to 0) is copied on a side stream while layer i computes, so one forward pass
costs about one host->device copy of all layers; a batch of requests shares it.  The embedding table stays
on the host (row gather, exact); final norm and LM head stay on the GPU.  The arithmetic is the same code
on the same shapes as LayerSkipLlama, so outputs are token-identical (checked on 1B: check_streamed.py).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from common.model import LayerSkipLlama, LayerWeights

FIELDS = ("wqkv", "wo", "w13", "w2", "attn_norm", "ffn_norm")


class StreamedLlama(LayerSkipLlama):
    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", dtype=torch.bfloat16, max_seq=None) -> "StreamedLlama":
        cpu = LayerSkipLlama.from_pretrained(path, device="cpu", dtype=dtype, max_seq=max_seq)
        dev = torch.device(device)
        m = cls(cpu.cfg, cpu.embed_w, [], cpu.norm_w.to(dev), cpu.lm_head_w.to(dev), dev, dtype,
                max_seq or cpu.cfg.max_position_embeddings)
        m.host_layers = [LayerWeights(**{f: getattr(lw, f).pin_memory() for f in FIELDS}) for lw in cpu.layers]
        del cpu
        m.slots = [LayerWeights(**{f: torch.empty_like(getattr(m.host_layers[0], f), device=dev) for f in FIELDS})
                   for _ in range(2)]
        m.layers = [None] * len(m.host_layers)
        m.copy_stream = torch.cuda.Stream(dev)
        m.ready = [torch.cuda.Event() for _ in range(2)]
        m.free = [torch.cuda.Event() for _ in range(2)]
        m.loaded = [-1, -1]                 # layer held by each GPU slot
        for e in m.free:
            e.record()
        m._fetch(0)
        return m

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    def _fetch(self, i: int) -> None:
        s = i % 2
        if self.loaded[s] == i:
            return
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.free[s])          # the slot's previous layer has finished
            for f in FIELDS:
                getattr(self.slots[s], f).copy_(getattr(self.host_layers[i], f), non_blocking=True)
            self.ready[s].record(self.copy_stream)
        self.loaded[s] = i

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens.cpu(), self.embed_w).to(self.device, non_blocking=True)

    def layer(self, i: int, h: torch.Tensor, ctx) -> torch.Tensor:
        s = i % 2
        self._fetch(i)
        torch.cuda.current_stream(self.device).wait_event(self.ready[s])
        self.layers[i] = self.slots[s]
        out = super().layer(i, h, ctx)
        self.layers[i] = None
        self.free[s].record()
        self._fetch((i + 1) % self.n_layers)                 # prefetch the next layer into the other slot
        return out
