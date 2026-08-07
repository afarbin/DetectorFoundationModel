"""The shared event encoder: one backbone for every detector modality.

Review Part 4, step 1, made literal:

    tokens = detector elements with (type embedding, coordinates, features);
    local mixing (fixed calo-neighbor edges / track k-NN EdgeConv) +
    global mixing (ISAB), all mask-aware.

Each modality declares how it is projected and locally mixed via
:class:`ModalityConfig`; the global ISAB stack is shared, and can attend
within one modality or across all of them (Part 4, step 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from dfm.tokens import TokenBatch, CombinedTokens, combine_token_batches
from dfm.layers import MaskedISAB, MaskedEdgeConv, NeighborConvBlock


@dataclass
class ModalityConfig:
    """How one modality enters the shared backbone.

    Attributes:
        name: modality key, must match ``TokenBatch.modality``.
        num_features: base input feature width F (WITHOUT the mask indicator).
        local: local-mixing flavor: ``"edges"`` (fixed graph, NeighborConv),
            ``"knn"`` (EdgeConv over coords), or ``"none"``.
        local_depth: number of local-mixing layers.
        knn_k: k for the ``"knn"`` flavor.
        mask_indicator: when True the input projection is built for F+1
            channels and the encoder accepts BOTH widths: F+1 (masked batches
            from ``MaskedTokenPretraining.mask``) and F (clean batches - a
            zero indicator column is appended internally). This makes the
            pretrained and finetuned encoders byte-compatible, so pretraining
            weights transfer verbatim (the point of a foundation model).
    """

    name: str
    num_features: int
    local: str = "none"
    local_depth: int = 2
    knn_k: int = 10
    mask_indicator: bool = False


class SharedEventEncoder(nn.Module):
    """Typed-token encoder shared by all modalities and all heads.

    forward(batches) -> dict of per-modality embeddings [B, N_mod, d] plus
    the combined tokens (for cross-modality heads) and a per-event global
    token (masked mean over all valid tokens).
    """

    def __init__(self, modalities: Sequence[ModalityConfig], dim: int = 128,
                 global_depth: int = 2, num_heads: int = 4, num_inds: int = 32,
                 dropout: float = 0.0, cross_modality: bool = True):
        super().__init__()
        self.dim = dim
        self.cross_modality = cross_modality
        self.configs: Dict[str, ModalityConfig] = {m.name: m for m in modalities}

        self.input_proj = nn.ModuleDict()
        self.type_embed = nn.ParameterDict()
        self.local_mixers = nn.ModuleDict()
        for m in modalities:
            width = m.num_features + (1 if m.mask_indicator else 0)
            self.input_proj[m.name] = nn.Sequential(
                nn.Linear(width, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
            self.type_embed[m.name] = nn.Parameter(torch.zeros(dim))
            if m.local == "edges":
                self.local_mixers[m.name] = nn.ModuleList(
                    [NeighborConvBlock(dim) for _ in range(m.local_depth)])
            elif m.local == "knn":
                self.local_mixers[m.name] = nn.ModuleList(
                    [MaskedEdgeConv(dim, dim, k=m.knn_k) for _ in range(m.local_depth)])
            elif m.local != "none":
                raise ValueError(f"unknown local mixing '{m.local}'")

        self.global_stack = nn.ModuleList(
            [MaskedISAB(dim, dim, num_heads, num_inds, dropout) for _ in range(global_depth)])
        self.out_norm = nn.LayerNorm(dim)

    def _embed_modality(self, tokens: TokenBatch) -> Tensor:
        cfg = self.configs[tokens.modality]
        x = tokens.features
        if cfg.mask_indicator:
            if x.shape[-1] == cfg.num_features:      # clean batch: indicator = 0
                x = torch.cat([x, x.new_zeros(x.shape[:-1] + (1,))], dim=-1)
            elif x.shape[-1] != cfg.num_features + 1:
                raise ValueError(
                    f"modality '{cfg.name}': expected {cfg.num_features} or "
                    f"{cfg.num_features + 1} channels, got {x.shape[-1]}")
        elif x.shape[-1] != cfg.num_features:
            raise ValueError(f"modality '{cfg.name}': expected "
                             f"{cfg.num_features} channels, got {x.shape[-1]}")
        h = self.input_proj[tokens.modality](x)
        h = h + self.type_embed[tokens.modality]
        h = h * tokens.mask.unsqueeze(-1)  # keep padding exactly zero
        if cfg.local == "edges":
            if tokens.edges is None:
                raise ValueError(f"modality '{cfg.name}' configured for edge mixing "
                                 f"but TokenBatch.edges is None")
            for block in self.local_mixers[cfg.name]:
                h = block(h, tokens.mask, tokens.edges)
        elif cfg.local == "knn":
            if tokens.coords is None:
                raise ValueError(f"modality '{cfg.name}' configured for knn mixing "
                                 f"but TokenBatch.coords is None")
            for conv in self.local_mixers[cfg.name]:
                h = h + conv(h, tokens.coords, tokens.mask)
        return h

    def forward(self, batches: Sequence[TokenBatch]) -> Dict[str, object]:
        """Encode one or more modalities of the same events.

        Returns dict with:
            ``"<modality>"``: [B, N_mod, d] token embeddings;
            ``"combined"``: :class:`CombinedTokens` over all modalities;
            ``"global"``: [B, d] masked mean over all valid tokens.
        """
        projected = []
        for tokens in batches:
            h = self._embed_modality(tokens)
            projected.append(TokenBatch(features=h, mask=tokens.mask,
                                        modality=tokens.modality))

        if self.cross_modality or len(projected) == 1:
            combined = combine_token_batches(projected)
            h, mask = combined.features, combined.mask
            for isab in self.global_stack:
                h = isab(h, mask)
            h = self.out_norm(h) * mask.unsqueeze(-1)
            combined = CombinedTokens(features=h, mask=mask, slices=combined.slices)
        else:
            per_mod = []
            for t in projected:
                h = t.features
                for isab in self.global_stack:
                    h = isab(h, t.mask)
                per_mod.append(TokenBatch(features=self.out_norm(h) * t.mask.unsqueeze(-1),
                                          mask=t.mask, modality=t.modality))
            combined = combine_token_batches(per_mod)

        denom = combined.mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_token = (combined.features * combined.mask.unsqueeze(-1)).sum(dim=1) / denom

        out: Dict[str, object] = {"combined": combined, "global": global_token}
        for t in projected:
            out[t.modality] = combined.modality(t.modality)
        return out
