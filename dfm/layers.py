"""Building blocks: mask-aware attention and local mixing layers.

Everything here is padding-mask aware by construction - the encoder-side
mask bug of the b-tagging model (review M1: unmasked ISAB) cannot recur,
because there is no unmasked attention block to reach for.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedMAB(nn.Module):
    """Multihead attention block with key masking.

    Port of the b-tagging ``MABMasked`` with two deliberate changes:
    per-head attention scaling ``1/sqrt(d_head)`` (the original used
    ``1/sqrt(dim_V)``), and a pre-norm residual layout. New package,
    no checkpoint compatibility to preserve.
    """

    def __init__(self, dim_q: int, dim_kv: int, dim_out: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        if dim_out % num_heads:
            raise ValueError(f"dim_out {dim_out} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.d_head = dim_out // num_heads
        self.wq = nn.Linear(dim_q, dim_out)
        self.wk = nn.Linear(dim_kv, dim_out)
        self.wv = nn.Linear(dim_kv, dim_out)
        self.wo = nn.Linear(dim_out, dim_out)
        self.resid_proj = nn.Linear(dim_q, dim_out) if dim_q != dim_out else nn.Identity()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.norm_ff = nn.LayerNorm(dim_out)
        self.ff = nn.Sequential(nn.Linear(dim_out, 2 * dim_out), nn.GELU(),
                                nn.Linear(2 * dim_out, dim_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Optional[Tensor] = None,
                return_attn: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """q [B, M, dq], kv [B, N, dkv], kv_mask [B, N] True=valid."""
        b, m, _ = q.shape
        n = kv.shape[1]
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        qh = self.wq(qn).view(b, m, self.num_heads, self.d_head).transpose(1, 2)
        kh = self.wk(kvn).view(b, n, self.num_heads, self.d_head).transpose(1, 2)
        vh = self.wv(kvn).view(b, n, self.num_heads, self.d_head).transpose(1, 2)
        scores = qh @ kh.transpose(-1, -2) / math.sqrt(self.d_head)  # [B,H,M,N]
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # a fully-masked key row would give NaNs; only possible for events with
        # zero valid tokens, which the adapters exclude - guard anyway
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ vh).transpose(1, 2).reshape(b, m, -1)
        h = self.resid_proj(q) + self.dropout(self.wo(out))
        h = h + self.dropout(self.ff(self.norm_ff(h)))
        return (h, attn.mean(dim=1)) if return_attn else (h, None)


class MaskedISAB(nn.Module):
    """Induced set attention block (Set Transformer), fully mask-aware.

    ``H = MAB(I, X, mask)`` then ``MAB(X, H)``: the inducing summary sees only
    valid tokens, so padding never contaminates real-token embeddings - the
    fix for review M1, structurally.
    """

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int,
                 dropout: float = 0.0):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing)
        self.mab_summarize = MaskedMAB(dim_out, dim_in, dim_out, num_heads, dropout)
        self.mab_broadcast = MaskedMAB(dim_in, dim_out, dim_out, num_heads, dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h, _ = self.mab_summarize(self.inducing.expand(x.shape[0], -1, -1), x, mask)
        out, _ = self.mab_broadcast(x, h, None)  # all induced tokens are valid
        return out


def knn_indices(coords: Tensor, k: int, mask: Optional[Tensor] = None) -> Tensor:
    """Mask-aware k-NN in coordinate space.

    ``coords [B, N, C]``, ``mask [B, N]``. Guards small events (review M4):
    the effective k is ``min(k, N-1)`` and invalid points get +inf distance so
    they are only selected when an event has fewer valid points than k_eff+1
    (their contribution is later zeroed by the conv's mask).
    """
    b, n, _ = coords.shape
    k_eff = max(1, min(k, n - 1))
    d2 = torch.cdist(coords, coords)
    if mask is not None:
        invalid = ~mask
        d2 = d2.masked_fill(invalid[:, None, :], float("inf"))
    eye = torch.eye(n, dtype=torch.bool, device=coords.device)
    d2 = d2.masked_fill(eye[None], float("inf"))
    # replace inf with a large finite value so topk never returns NaN grads
    d2 = torch.nan_to_num(d2, posinf=1e9)
    return d2.topk(k_eff, largest=False).indices  # [B, N, k_eff]


class MaskedEdgeConv(nn.Module):
    """DGCNN-style EdgeConv over k-NN graphs, mask-aware (track local mixing)."""

    def __init__(self, dim_in: int, dim_out: int, k: int = 10):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(2 * dim_in, 2 * dim_out), nn.GELU(),
                                 nn.Linear(2 * dim_out, dim_out), nn.GELU())

    def forward(self, x: Tensor, coords: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, n, d = x.shape
        idx = knn_indices(coords, self.k, mask)                    # [B,N,k]
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)           # [B,N,k,D]
        neigh = torch.gather(x.unsqueeze(1).expand(-1, n, -1, -1), 2, gather)
        center = x.unsqueeze(2).expand(-1, -1, idx.shape[2], -1)
        msgs = self.mlp(torch.cat([neigh - center, center], dim=-1))  # [B,N,k,D']
        if mask is not None:
            # a masked mean: invalid neighbor slots contribute exactly nothing,
            # and normalization is by the VALID-neighbor count, not k
            valid_neigh = torch.gather(mask.unsqueeze(1).expand(-1, n, -1), 2, idx)
            w = valid_neigh.unsqueeze(-1).float()
            h = (msgs * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
            h = h * mask.unsqueeze(-1)
        else:
            h = msgs.mean(dim=2)
        return h


class NeighborConv(nn.Module):
    """SAGE-style conv over a fixed edge list (calo-cell local mixing).

    ``h_i' = lin_l(mean_{j in N(i)} h_j) + lin_r(h_i)`` - the same semantics
    AND parameter roles as PyG ``SAGEConv`` (lin_l acts on the aggregated
    neighbors and carries the bias; lin_r acts on the root, no bias), so
    weights from the existing foundation-model checkpoints can be ported
    later. Pure torch, no PyG dependency. Edges are mirrored internally:
    message passing is always bidirectional (the July-review bug cannot
    recur here).
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.lin_l = nn.Linear(dim_in, dim_out)              # neighbors (bias)
        self.lin_r = nn.Linear(dim_in, dim_out, bias=False)  # root/self

    def forward(self, x: Tensor, pairs: Tensor) -> Tensor:
        """x [N, D] one event's valid tokens; pairs [E, 2] (any orientation)."""
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must be [E, 2], got {tuple(pairs.shape)}")
        n = x.shape[0]
        src = torch.cat([pairs[:, 0], pairs[:, 1]])
        dst = torch.cat([pairs[:, 1], pairs[:, 0]])
        agg = x.new_zeros((n, x.shape[1]))
        agg.index_add_(0, dst, x[src])
        deg = torch.bincount(dst, minlength=n).clamp(min=1).unsqueeze(-1)
        return self.lin_l(agg / deg) + self.lin_r(x)


class NeighborConvBlock(nn.Module):
    """Residual (LayerNorm -> NeighborConv -> GELU) over per-event edge lists.

    Batched: all events' valid tokens are flattened once and the conv runs on
    a single block-diagonal edge index (per-event edges shifted by token
    offsets), so cost does not scale with batch size in Python.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = NeighborConv(dim, dim)

    def forward(self, x: Tensor, mask: Tensor, edges: List[Tensor]) -> Tensor:
        b, n_max, d = x.shape
        counts = mask.sum(dim=1)                      # valid tokens per event
        flat = x[mask]                                # [N_total, D]
        if flat.shape[0] == 0:
            return x
        offsets = torch.cumsum(counts, dim=0) - counts
        shifted = [e + offsets[i] for i, e in enumerate(edges) if e.numel()]
        if not shifted:
            return x
        big = torch.cat(shifted, dim=0)               # [E_total, 2]
        h = F.gelu(self.conv(self.norm(flat), big))
        out = x.clone()
        out[mask] = flat + h
        return out
