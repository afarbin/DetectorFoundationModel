"""Downstream heads that attach to SharedEventEncoder token embeddings."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dfm.layers import MaskedMAB


class EdgeClassifierHead(nn.Module):
    """Cells -> topocluster edge prediction.

    Two readouts (July review 1.2):
      - ``"concat"``: ``[h_src, h_dst]`` - order-dependent, compatible with the
        existing 5-class labels (classes 2/3 encode edge orientation).
      - ``"symmetric"``: ``[h_src + h_dst, |h_src - h_dst|]`` - recommended
        with binary P(same-cluster) targets (``num_classes=1``).
    """

    def __init__(self, dim: int, num_classes: int = 5, readout: str = "concat"):
        super().__init__()
        if readout not in ("concat", "symmetric"):
            raise ValueError(f"unknown readout '{readout}'")
        self.readout = readout
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                 nn.Linear(dim, num_classes))

    def forward(self, cell_tokens: Tensor, cell_mask: Tensor,
                pairs: List[Tensor]) -> Tensor:
        """cell_tokens [B, N, d]; pairs: per-event [E_i, 2] valid-token indices.

        Returns concatenated logits ``[sum_i E_i, num_classes]`` aligned with
        the per-event pair lists in order - the same contract as the existing
        trainer's ``(E_total, C)`` output.
        """
        outs = []
        for i, p in enumerate(pairs):
            h = cell_tokens[i]
            hs, hd = h[p[:, 0]], h[p[:, 1]]
            if self.readout == "concat":
                edge = torch.cat([hs, hd], dim=-1)
            else:
                edge = torch.cat([hs + hd, (hs - hd).abs()], dim=-1)
            outs.append(self.mlp(edge))
        return torch.cat(outs, dim=0)


class JetQueryDecoder(nn.Module):
    """Jet queries cross-attend to event tokens; per-jet output.

    Port of the b-tagging ``JetQueriesDecoder`` onto the shared backbone,
    with the review's upgrades:
      - query self-attention between decoder layers (M9): jets negotiate
        shared tracks/energy - required for HH->4b-style ambiguity and for
        Stage-2 event-level calibration;
      - cross-attention over ANY token set (tracks, cells, or the combined
        tokens - Part 4 step 4);
      - output modes: ``"binary"`` (b-tag logit), ``"multiclass"`` (b/c/light,
        fixes D1's charm-into-light collapse), ``"gauss"`` ((mu, log_var) for
        heteroscedastic jet calibration, Part 5).
    """

    def __init__(self, dim: int = 128, jet_in_dim: int = 4, depth: int = 2,
                 num_heads: int = 4, mode: str = "binary", num_classes: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        if mode not in ("binary", "multiclass", "gauss"):
            raise ValueError(f"unknown mode '{mode}'")
        self.mode = mode
        self.jet_embed = nn.Sequential(nn.Linear(jet_in_dim, dim), nn.GELU(),
                                       nn.Linear(dim, dim))
        self.cross_layers = nn.ModuleList(
            [MaskedMAB(dim, dim, dim, num_heads, dropout) for _ in range(depth)])
        self.self_layers = nn.ModuleList(
            [MaskedMAB(dim, dim, dim, num_heads, dropout) for _ in range(depth)])
        out_dim = {"binary": 1, "multiclass": num_classes, "gauss": 2}[self.mode]
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, out_dim))

    def forward(self, jets: Tensor, jet_mask: Tensor, tokens: Tensor,
                token_mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """jets [B, M, jet_in_dim] = (eta, phi, log pT, log m); jet_mask [B, M];
        tokens [B, N, d]; token_mask [B, N].

        Returns ``(out, attn)``: out is [B, M, 1] logits ("binary"),
        [B, M, C] logits ("multiclass") or [B, M, 2] = (mu, log_var)
        ("gauss"); attn [B, M, N] is the last cross-attention map
        (which tokens each jet's decision used).
        """
        z = self.jet_embed(jets)
        attn = None
        for i, (cross, self_att) in enumerate(zip(self.cross_layers, self.self_layers)):
            last = i == len(self.cross_layers) - 1
            z, attn = cross(z, tokens, token_mask, return_attn=last)
            # jets negotiate among themselves; padded jets masked as keys
            z, _ = self_att(z, z, jet_mask)
        return self.head(z), attn


class ReconstructionHead(nn.Module):
    """Per-token feature reconstruction for masked pretraining."""

    def __init__(self, dim: int, num_features: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, num_features))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.mlp(tokens)


def gauss_nll(mu_logvar: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Heteroscedastic Gaussian NLL over valid jets (Part 5 calibration loss)."""
    mu, log_var = mu_logvar[..., 0], mu_logvar[..., 1].clamp(-10, 10)
    nll = 0.5 * (log_var + (target - mu) ** 2 / log_var.exp())
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def masked_bce(logits: Tensor, targets: Tensor, mask: Tensor,
               pos_weight: Optional[Tensor] = None) -> Tensor:
    """BCE over valid jets; targets in {0,1} with -1 = padding (excluded)."""
    valid = mask & (targets >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1)[valid], targets[valid].float(), pos_weight=pos_weight)


def masked_ce(logits: Tensor, targets: Tensor, mask: Tensor,
              class_weights: Optional[Tensor] = None) -> Tensor:
    """Cross-entropy over valid jets; -1-padded targets excluded."""
    valid = mask & (targets >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid],
                                       weight=class_weights)
