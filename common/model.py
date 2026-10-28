"""LayerSkip-Llama in plain PyTorch with an explicit, layer-by-layer forward.

Why not HF `LlamaForCausalLM`: early-exit self-speculation needs to (a) run an
arbitrary layer range [start, end) on an arbitrary subset of requests, (b) reuse
the shallow-layer KV written during drafting, and (c) time each layer on its own.
DEL's `forward_early / forward_remainder` do this on top of the legacy HF cache API
that no longer exists in transformers>=4.50, and ASPIRE's gpt-fast model needs a
patched FlashInfer (Linux only).  This file keeps the same math as HF Llama
(rotate-half RoPE incl. llama3 scaling, fp32 RMSNorm, SwiGLU) with gpt-fast-style
fused weights (wqkv, w13) so that every layer is exactly 4 weight GEMMs.

Conventions
-----------
* exit layer ``k`` means "run layers 0..k-1, then final norm + LM head"
  (same as LayerSkip / DEL ``exit_layer``); ``k == n_layers`` is the full model.
* KV cache: per layer one ``[max_batch, n_kv_heads, max_seq, head_dim]`` tensor for
  K and one for V.  Request ``i`` of a call lives in batch row ``b0 + i``.
* Every call carries explicit positions ``pos [b, T]``.  A query at position p
  attends to cache positions ``<= p`` of the *same layer*.  Stale entries beyond
  the committed length are therefore never read, so rolling back after a rejected
  draft is free (just do not advance the position).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- config

@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    rope_scaling: Optional[dict] = None
    bos_token_id: Optional[int] = None
    eos_token_ids: tuple = field(default_factory=tuple)

    @classmethod
    def from_hf_dir(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        eos = cfg.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        gen_path = path / "generation_config.json"
        if gen_path.exists():
            gen = json.loads(gen_path.read_text())
            g_eos = gen.get("eos_token_id")
            g_eos = g_eos if isinstance(g_eos, list) else ([g_eos] if g_eos is not None else [])
            eos_ids = tuple(dict.fromkeys(list(eos_ids) + g_eos))
        n_heads = cfg["num_attention_heads"]
        # transformers>=5 stores RoPE settings in `rope_parameters`; older configs use
        # `rope_theta` + `rope_scaling`.  Accept both.
        rope_params = cfg.get("rope_parameters") or {}
        rope_theta = cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0))
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is None and rope_params.get("rope_type", "default") != "default":
            rope_scaling = rope_params
        return cls(
            vocab_size=cfg["vocab_size"],
            dim=cfg["hidden_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // n_heads,
            intermediate_size=cfg["intermediate_size"],
            norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            rope_scaling=rope_scaling,
            bos_token_id=cfg.get("bos_token_id"),
            eos_token_ids=eos_ids,
        )

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache per token over all layers (bf16/fp16)."""
        return self.n_layers * 2 * self.n_kv_heads * self.head_dim * 2

    @property
    def layer_param_count(self) -> int:
        attn = self.dim * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim + self.n_heads * self.head_dim * self.dim
        mlp = 3 * self.dim * self.intermediate_size
        return attn + mlp + 2 * self.dim

    @property
    def param_count(self) -> int:
        emb = self.vocab_size * self.dim
        head = 0 if self.tie_word_embeddings else emb
        return emb + head + self.n_layers * self.layer_param_count + self.dim


# --------------------------------------------------------------------------- rope

def _llama3_inv_freq(inv_freq: Tensor, scaling: dict) -> Tensor:
    """Same as transformers' `_compute_llama3_parameters`."""
    factor = scaling["factor"]
    low = scaling["low_freq_factor"]
    high = scaling["high_freq_factor"]
    old_len = scaling["original_max_position_embeddings"]
    low_wavelen = old_len / low
    high_wavelen = old_len / high
    wavelen = 2 * math.pi / inv_freq
    out = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_len / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * out / factor + smooth * out
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, smoothed, out)


def build_rope_cache(cfg: ModelConfig, max_seq: int, device) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim))
    scaling = cfg.rope_scaling or {}
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if rope_type == "llama3":
        inv_freq = _llama3_inv_freq(inv_freq, scaling)
    elif rope_type not in (None, "default"):
        raise NotImplementedError(f"rope_scaling type {rope_type!r} is not supported")
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # HF rotate-half layout
    return emb.cos().to(device), emb.sin().to(device)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x [b, T, h, D]; cos/sin [b, T, 1, D] already cast to x.dtype
    return x * cos + _rotate_half(x) * sin


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


# --------------------------------------------------------------------------- weights

@dataclass
class LayerWeights:
    wqkv: Tensor
    wo: Tensor
    w13: Tensor  # [gate; up]
    w2: Tensor
    attn_norm: Tensor
    ffn_norm: Tensor


class KVCache:
    """Static contiguous KV cache, one K and one V tensor per layer."""

    def __init__(self, cfg: ModelConfig, max_batch: int, max_seq: int, dtype, device):
        self.max_batch = max_batch
        self.max_seq = max_seq
        shape = (max_batch, cfg.n_kv_heads, max_seq, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]

    @staticmethod
    def bytes_needed(cfg: ModelConfig, max_batch: int, max_seq: int, dtype_bytes: int = 2) -> int:
        return cfg.n_layers * 2 * max_batch * cfg.n_kv_heads * max_seq * cfg.head_dim * dtype_bytes


@dataclass
class AttnCtx:
    """Per-call attention metadata shared by every layer of one segment."""

    pos: Tensor            # [b, T] int64 absolute positions of the query tokens
    b0: int                # first cache row of this call
    L: int                 # number of cache positions to read (>= max(pos)+1, padded)
    cos: Tensor            # [b, T, 1, D]
    sin: Tensor
    bidx: Tensor           # [b, T] cache rows
    mask: Optional[Tensor]  # [b, 1, G*T, L] bool, or None for causal prefill from 0
    causal_prefill: bool
    rows: Optional[Tensor] = None  # [b] cache rows of a non-contiguous request subset (S1 deep segment)

    @property
    def batch(self) -> int:
        return self.pos.shape[0]

    @property
    def tokens(self) -> int:
        return self.pos.shape[0] * self.pos.shape[1]

    @property
    def kv_read(self) -> int:
        """KV positions read per layer by this call (feature of the cost model)."""
        return self.pos.shape[0] * self.L


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


class LayerSkipLlama:
    """Inference-only Llama with layer-range forward.  Not an nn.Module on purpose:
    no autograd, no hooks, plain tensors."""

    def __init__(self, cfg: ModelConfig, embed: Tensor, layers: list[LayerWeights], norm: Tensor,
                 lm_head: Tensor, device, dtype, max_seq: int):
        self.cfg = cfg
        self.embed_w = embed
        self.layers = layers
        self.norm_w = norm
        self.lm_head_w = lm_head
        self.device = torch.device(device)
        self.dtype = dtype
        self.group = cfg.n_heads // cfg.n_kv_heads
        self.q_size = cfg.n_heads * cfg.head_dim
        self.kv_size = cfg.n_kv_heads * cfg.head_dim
        self.rope_cos, self.rope_sin = build_rope_cache(cfg, max_seq, self.device)
        self.cache: Optional[KVCache] = None

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        max_seq: Optional[int] = None) -> "LayerSkipLlama":
        from safetensors import safe_open

        path = Path(path)
        cfg = ModelConfig.from_hf_dir(path)
        index = path / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        raw: dict[str, Tensor] = {}
        for fname in files:
            with safe_open(str(path / fname), framework="pt", device="cpu") as f:
                for key in f.keys():
                    raw[key] = f.get_tensor(key)

        def take(key: str) -> Tensor:
            return raw.pop(key).to(device=device, dtype=dtype)

        layers = []
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            wqkv = torch.cat([raw.pop(p + "self_attn.q_proj.weight"), raw.pop(p + "self_attn.k_proj.weight"),
                              raw.pop(p + "self_attn.v_proj.weight")], dim=0).to(device=device, dtype=dtype)
            w13 = torch.cat([raw.pop(p + "mlp.gate_proj.weight"), raw.pop(p + "mlp.up_proj.weight")],
                            dim=0).to(device=device, dtype=dtype)
            layers.append(LayerWeights(
                wqkv=wqkv,
                wo=take(p + "self_attn.o_proj.weight"),
                w13=w13,
                w2=take(p + "mlp.down_proj.weight"),
                attn_norm=take(p + "input_layernorm.weight"),
                ffn_norm=take(p + "post_attention_layernorm.weight"),
            ))
        embed = take("model.embed_tokens.weight")
        norm = take("model.norm.weight")
        lm_head = take("lm_head.weight") if "lm_head.weight" in raw else embed
        max_seq = max_seq or cfg.max_position_embeddings
        return cls(cfg, embed, layers, norm, lm_head, device, dtype, max_seq)

    @property
    def n_layers(self) -> int:
        return self.cfg.n_layers

    # ---------------------------------------------------------------- cache
    def allocate_cache(self, max_batch: int, max_seq: int) -> KVCache:
        max_seq = _round_up(max_seq, 64)
        if max_seq > self.rope_cos.shape[0]:
            self.rope_cos, self.rope_sin = build_rope_cache(self.cfg, max_seq, self.device)
        self.cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.cache = KVCache(self.cfg, max_batch, max_seq, self.dtype, self.device)
        return self.cache

    def free_cache(self) -> None:
        self.cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- building blocks
    def make_ctx(self, pos: Tensor, b0: int = 0, max_pos: Optional[int] = None,
                 causal_prefill: bool = False, rows: Optional[Tensor] = None) -> AttnCtx:
        """`max_pos` is the largest position in `pos` as a host int (avoids a GPU sync).
        `rows` [b] (int64): the call's requests live in these cache rows instead of
        b0..b0+b-1 (request subsets, e.g. the verifying slots of a mixed step); the
        attention then gathers their KV (a copy of b x L positions per layer)."""
        b, T = pos.shape
        if max_pos is None:
            max_pos = int(pos.max())
        L = min(_round_up(max_pos + 1, 64), self.cache.max_seq)
        cos = self.rope_cos[pos].to(self.dtype).unsqueeze(2)
        sin = self.rope_sin[pos].to(self.dtype).unsqueeze(2)
        if rows is not None:
            bidx = rows.view(b, 1).expand(b, T)
        else:
            bidx = (torch.arange(b, device=pos.device) + b0).unsqueeze(1).expand(b, T)
        mask = None
        if not causal_prefill:
            keys = torch.arange(L, device=pos.device)
            m = keys.view(1, 1, L) <= pos.unsqueeze(-1)                     # [b, T, L]
            G = self.group
            mask = m.unsqueeze(1).unsqueeze(1).expand(b, 1, G, T, L).reshape(b, 1, G * T, L)
        return AttnCtx(pos, b0, L, cos, sin, bidx, mask, causal_prefill, rows)

    def embed(self, tokens: Tensor) -> Tensor:
        return F.embedding(tokens, self.embed_w)

    def head(self, h: Tensor) -> Tensor:
        """Final norm + LM head (shared by every exit, as in LayerSkip)."""
        return F.linear(rms_norm(h, self.norm_w, self.cfg.norm_eps), self.lm_head_w)

    def _attention(self, i: int, x: Tensor, ctx: AttnCtx) -> Tensor:
        lw = self.layers[i]
        cfg = self.cfg
        b, T, _ = x.shape
        qkv = F.linear(x, lw.wqkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = apply_rope(q.view(b, T, cfg.n_heads, cfg.head_dim), ctx.cos, ctx.sin)
        k = apply_rope(k.view(b, T, cfg.n_kv_heads, cfg.head_dim), ctx.cos, ctx.sin)
        v = v.view(b, T, cfg.n_kv_heads, cfg.head_dim)

        ck, cv = self.cache.k[i], self.cache.v[i]
        ck[ctx.bidx, :, ctx.pos] = k
        cv[ctx.bidx, :, ctx.pos] = v

        if ctx.causal_prefill:
            # fresh request, no past: plain causal attention over the new tokens
            kk = k.transpose(1, 2).repeat_interleave(self.group, dim=1)
            vv = v.transpose(1, 2).repeat_interleave(self.group, dim=1)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, is_causal=True)
            o = o.transpose(1, 2).reshape(b, T, self.q_size)
        else:
            if ctx.rows is not None:
                K = ck[ctx.rows, :, :ctx.L]
                V = cv[ctx.rows, :, :ctx.L]
            else:
                K = ck[ctx.b0:ctx.b0 + b, :, :ctx.L]
                V = cv[ctx.b0:ctx.b0 + b, :, :ctx.L]
            G = self.group
            # GQA without copying KV: fold the G query heads of a KV head into the row axis
            qg = q.view(b, T, cfg.n_kv_heads, G, cfg.head_dim).permute(0, 2, 3, 1, 4).reshape(
                b, cfg.n_kv_heads, G * T, cfg.head_dim)
            o = F.scaled_dot_product_attention(qg, K, V, attn_mask=ctx.mask)
            o = o.view(b, cfg.n_kv_heads, G, T, cfg.head_dim).permute(0, 3, 1, 2, 4).reshape(b, T, self.q_size)
        return F.linear(o, lw.wo)

    def layer(self, i: int, h: Tensor, ctx: AttnCtx) -> Tensor:
        lw = self.layers[i]
        eps = self.cfg.norm_eps
        h = h + self._attention(i, rms_norm(h, lw.attn_norm, eps), ctx)
        gu = F.linear(rms_norm(h, lw.ffn_norm, eps), lw.w13)
        g, u = gu.chunk(2, dim=-1)
        return h + F.linear(F.silu(g) * u, lw.w2)

    def forward_layers(self, h: Tensor, start: int, end: int, ctx: AttnCtx,
                       capture: Optional[list] = None, layer_hook=None) -> Tensor:
        """Run layers [start, end).  If `capture` is a list, the output of every layer
        is appended to it (capture[j] = hidden state *after* layer start+j, i.e. the
        input of exit layer start+j+1).  `layer_hook(i, when)` is called with
        when in {"pre", "post"} (used for per-layer timing)."""
        for i in range(start, end):
            if layer_hook is not None:
                layer_hook(i, "pre")
            h = self.layer(i, h, ctx)
            if layer_hook is not None:
                layer_hook(i, "post")
            if capture is not None:
                capture.append(h)
        return h

    # ---------------------------------------------------------------- convenience
    @torch.inference_mode()
    def prefill(self, tokens: list[int], slot: int, capture_exits: bool = False):
        """Prefill one request into cache row `slot`.  Returns (last-position logits [V],
        list of last-position hidden states after every layer if capture_exits)."""
        t = torch.tensor([tokens], dtype=torch.long, device=self.device)
        pos = torch.arange(len(tokens), device=self.device).unsqueeze(0)
        ctx = self.make_ctx(pos, b0=slot, max_pos=len(tokens) - 1, causal_prefill=True)
        cap = [] if capture_exits else None
        h = self.forward_layers(self.embed(t), 0, self.n_layers, ctx, capture=cap)
        logits = self.head(h[:, -1])[0]
        hidden = [c[0, -1] for c in cap] if cap is not None else None
        return logits, hidden
