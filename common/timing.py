"""Timers that use CUDA events on GPU and perf_counter on CPU."""

from __future__ import annotations

import time
from typing import Optional

import torch


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Interval:
    """One measured interval.  `start()` / `stop()` enqueue CUDA events; `ms` synchronizes."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        if self.cuda:
            self._e0 = torch.cuda.Event(enable_timing=True)
            self._e1 = torch.cuda.Event(enable_timing=True)

    def start(self) -> "Interval":
        if self.cuda:
            self._e0.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def stop(self) -> "Interval":
        if self.cuda:
            self._e1.record()
        else:
            self._t1 = time.perf_counter()
        return self

    @property
    def ms(self) -> float:
        if self.cuda:
            self._e1.synchronize()
            return float(self._e0.elapsed_time(self._e1))
        return (self._t1 - self._t0) * 1000.0


class LayerTimer:
    """Per-layer timing hook for `LayerSkipLlama.forward_layers(layer_hook=...)`.

    On GPU it records one event before and after each layer (events are cheap and
    do not synchronize); `results()` synchronizes once at the end."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.records: dict[int, list] = {}

    def __call__(self, layer: int, when: str) -> None:
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        else:
            ev = time.perf_counter()
        self.records.setdefault(layer, [None, None])[0 if when == "pre" else 1] = ev

    def results(self) -> dict[int, float]:
        out = {}
        for layer, (a, b) in self.records.items():
            if self.cuda:
                b.synchronize()
                out[layer] = float(a.elapsed_time(b))
            else:
                out[layer] = (b - a) * 1000.0
        return out
