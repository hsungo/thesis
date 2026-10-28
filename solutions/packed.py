"""Packed (ragged) forward over an arbitrary set of request token blocks.

Every request block r contributes T_r consecutive tokens at positions p_r .. p_r+T_r-1 of KV
cache row `row_r`.  The blocks are concatenated into one token axis (qo_indptr = cumsum(T_r));
all GEMMs (QKV, O, MLP, LM head) run on the concatenated tokens, so nothing is padded there.
Attention follows one rule for every token - a query at position p of row r attends to the
cached positions <= p of row r (after this call's K/V have been written) - which covers decode
tokens (T=1), verification blocks (T=d+1), and prefill chunks (which see their earlier chunks).

Attention is computed per *bucket* of blocks that share the same T (``bucketing="exact"``,
no padding at all) - one SDPA call per distinct T - or with all blocks of a segment padded to
the longest one (``bucketing="pad"``, fewer kernel launches).  A bucket whose rows form a
contiguous range reads the cache with a slice; otherwise the rows' KV is gathered (index_select
copy, see observation B1 for its cost).  No GPU->CPU synchronisation happens here: all metadata
is built on the CPU and uploaded through pinned memory with non_blocking copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from common.model import apply_rope, rms_norm


def upload(arr, device: torch.device, dtype=torch.long) -> torch.Tensor:
    """CPU numpy/list -> device tensor without a stream synchronisation (pinned + non_blocking)."""
    t = torch.as_tensor(np.asarray(arr), dtype=dtype)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass
class Block:
    row: int          # KV cache row (slot)
    pos0: int         # position of the first token
    T: int            # number of tokens
    tag: object = None  # caller's handle (e.g. ("verify", slot) or ("chunk", slot))


@dataclass
class Bucket:
    start: int        # token range [start, end) in the packed axis
    end: int
    b: int            # blocks in the bucket
    T: int            # tokens per block (padded length when bucketing="pad")
    L: int            # cache positions read (padded to 64)
    rows: Optional[torch.Tensor]   # None -> contiguous rows b0 .. b0+b-1
    b0: int
    mask: torch.Tensor             # [b, 1, G*T, L] bool
    pad_index: Optional[torch.Tensor] = None   # bucketing="pad": packed index of every padded slot (-1 = pad)


@dataclass
class PackedBatch:
    blocks: list            # blocks in packed order
    offsets: np.ndarray     # [n_blocks + 1] qo_indptr (host)
    pos: torch.Tensor       # [N] positions
    rows: torch.Tensor      # [N] cache rows
    cos: torch.Tensor       # [N, 1, D]
    sin: torch.Tensor
    buckets: list = field(default_factory=list)
    n_tokens: int = 0
    kv_read: int = 0        # sum over blocks of attended positions (per layer)

    def block_slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))


def build_packed(model, blocks: list, bucketing: str = "exact") -> PackedBatch:
    """Order the blocks bucket-major (by T, then row) and build all attention metadata."""
    dev = model.device
    G = model.group
    max_seq = model.cache.max_seq
    if bucketing == "exact":
        groups = {}
        for b in blocks:
            groups.setdefault(b.T, []).append(b)
        ordered_groups = [sorted(groups[T], key=lambda x: (x.row, x.pos0)) for T in sorted(groups)]
    elif bucketing == "pad":
        ordered_groups = [sorted(blocks, key=lambda x: (x.row, x.pos0))] if blocks else []
    else:
        raise ValueError(bucketing)
    ordered = [b for g in ordered_groups for b in g]
    Ts = np.array([b.T for b in ordered], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(Ts)]).astype(np.int64)
    N = int(offsets[-1])
    pos_h = np.empty(N, dtype=np.int64)
    rows_h = np.empty(N, dtype=np.int64)
    for i, b in enumerate(ordered):
        pos_h[offsets[i]:offsets[i + 1]] = np.arange(b.pos0, b.pos0 + b.T)
        rows_h[offsets[i]:offsets[i + 1]] = b.row
    pos = upload(pos_h, dev)
    rows = upload(rows_h, dev)
    cos = model.rope_cos[pos].to(model.dtype).unsqueeze(1)
    sin = model.rope_sin[pos].to(model.dtype).unsqueeze(1)
    pb = PackedBatch(ordered, offsets, pos, rows, cos, sin, n_tokens=N)
    i0 = 0
    for g in ordered_groups:
        nb = len(g)
        start, end = int(offsets[i0]), int(offsets[i0 + nb])
        Tmax = max(b.T for b in g)
        last = np.array([b.pos0 + b.T - 1 for b in g], dtype=np.int64)
        L = min(_round_up(int(last.max()) + 1, 64), max_seq)
        rws = [b.row for b in g]
        contiguous = rws == list(range(rws[0], rws[0] + nb))
        # query positions per block, padded slots repeat the block's last position (never all-masked)
        qpos = np.empty((nb, Tmax), dtype=np.int64)
        pad_idx = None
        if bucketing == "pad" and any(b.T != Tmax for b in g):
            pad_idx = np.full((nb, Tmax), -1, dtype=np.int64)
        for j, b in enumerate(g):
            qpos[j, :b.T] = np.arange(b.pos0, b.pos0 + b.T)
            qpos[j, b.T:] = b.pos0 + b.T - 1
            if pad_idx is not None:
                pad_idx[j, :b.T] = np.arange(offsets[i0 + j], offsets[i0 + j + 1])
        qp = upload(qpos, dev)
        keys = torch.arange(L, device=dev)
        m = keys.view(1, 1, L) <= qp.unsqueeze(-1)                         # [b, T, L]
        mask = m.unsqueeze(1).unsqueeze(1).expand(nb, 1, G, Tmax, L).reshape(nb, 1, G * Tmax, L)
        pb.buckets.append(Bucket(start, end, nb, Tmax, L, None if contiguous else upload(rws, dev),
                                 rws[0], mask, upload(pad_idx, dev) if pad_idx is not None else None))
        pb.kv_read += sum(b.pos0 + b.T for b in g)
        i0 += nb
    return pb


def _attend(model, i: int, q: torch.Tensor, pb: PackedBatch) -> torch.Tensor:
    """q [N, Hq, D] (RoPE applied) -> attention output [N, Hq*D]."""
    cfg = model.cfg
    G, Hkv, D = model.group, cfg.n_kv_heads, cfg.head_dim
    ck, cv = model.cache.k[i], model.cache.v[i]
    out = torch.empty(q.shape[0], model.q_size, dtype=q.dtype, device=q.device)
    for bk in pb.buckets:
        if bk.rows is None:
            K, V = ck[bk.b0:bk.b0 + bk.b, :, :bk.L], cv[bk.b0:bk.b0 + bk.b, :, :bk.L]
        else:
            K, V = ck[bk.rows, :, :bk.L], cv[bk.rows, :, :bk.L]
        if bk.pad_index is None:
            qb = q[bk.start:bk.end].view(bk.b, bk.T, Hkv, G, D)
        else:   # padded bucket: gather queries into [b, T] (pad slots copy any token, outputs discarded)
            idx = bk.pad_index.clamp(min=0)
            qb = q[idx.view(-1)].view(bk.b, bk.T, Hkv, G, D)
        qg = qb.permute(0, 2, 3, 1, 4).reshape(bk.b, Hkv, G * bk.T, D)
        o = F.scaled_dot_product_attention(qg, K, V, attn_mask=bk.mask)
        o = o.view(bk.b, Hkv, G, bk.T, D).permute(0, 3, 1, 2, 4).reshape(bk.b * bk.T, model.q_size)
        if bk.pad_index is None:
            out[bk.start:bk.end] = o
        else:
            valid = bk.pad_index.view(-1) >= 0
            out[bk.pad_index.view(-1)[valid]] = o[valid]
    return out


def packed_layer(model, i: int, x: torch.Tensor, pb: PackedBatch) -> torch.Tensor:
    """One decoder layer on packed tokens x [N, dim]; writes this call's K/V into the cache."""
    cfg = model.cfg
    lw = model.layers[i]
    eps = cfg.norm_eps
    N = x.shape[0]
    qkv = F.linear(rms_norm(x, lw.attn_norm, eps), lw.wqkv)
    q, k, v = qkv.split([model.q_size, model.kv_size, model.kv_size], dim=-1)
    q = apply_rope(q.view(N, cfg.n_heads, cfg.head_dim), pb.cos, pb.sin)
    k = apply_rope(k.view(N, cfg.n_kv_heads, cfg.head_dim), pb.cos, pb.sin)
    v = v.view(N, cfg.n_kv_heads, cfg.head_dim)
    model.cache.k[i][pb.rows, :, pb.pos] = k
    model.cache.v[i][pb.rows, :, pb.pos] = v
    x = x + F.linear(_attend(model, i, q, pb), lw.wo)
    gu = F.linear(rms_norm(x, lw.ffn_norm, eps), lw.w13)
    g, u = gu.chunk(2, dim=-1)
    return x + F.linear(F.silu(g) * u, lw.w2)


def packed_layers(model, x: torch.Tensor, pb: PackedBatch, start: int, end: int,
                  capture: Optional[dict] = None, capture_layers: tuple = ()) -> torch.Tensor:
    """Run layers [start, end).  capture[k] = hidden state entering exit layer k (= output of
    layer k-1) for every k in capture_layers with start < k <= end."""
    for i in range(start, end):
        x = packed_layer(model, i, x, pb)
        if capture is not None and (i + 1) in capture_layers:
            capture[i + 1] = x
    return x


def qo_indptr(lengths) -> np.ndarray:
    """cumsum of per-request token counts with a leading 0 (FlashInfer's qo_indptr)."""
    return np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))]).astype(np.int64)
