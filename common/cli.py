"""Argument helpers shared by the observation / evaluation scripts."""

from __future__ import annotations

import torch

from common.model import LayerSkipLlama
from common.utils import DTYPES, pick_device, resolve_model_path


def add_model_args(parser, default_model: str = "layerskip-llama3-8B") -> None:
    g = parser.add_argument_group("model")
    g.add_argument("--model", default=default_model,
                   help="folder under ./models (layerskip-llama3-8B, layerskip-llama3-1B) or a path")
    g.add_argument("--device", default=None, help="default: cuda if available else cpu")
    g.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))


def load_model_from_args(args, max_seq: int | None = None) -> LayerSkipLlama:
    device = pick_device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        print("[model] CPU run: switching dtype to fp32 (bf16 matmuls are very slow on CPU)")
        dtype = torch.float32
    path = resolve_model_path(args.model)
    print(f"[model] loading {path} on {device} ({dtype})")
    model = LayerSkipLlama.from_pretrained(path, device=str(device), dtype=dtype, max_seq=max_seq)
    print(f"[model] {model.cfg.n_layers} layers, dim {model.cfg.dim}, vocab {model.cfg.vocab_size}")
    return model
