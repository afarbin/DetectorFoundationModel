"""Modality-agnostic masked-token pretraining.

One masking module for every modality (review Part 4, step 2): random
whole-token, feature-column, coordinate-region, and group masking (group =
topocluster for cells, vertex for tracks). Fixes the implementation issues
of the calo-only version by design:

- P1: group ids ride inside ``TokenBatch.group_ids`` - group masking either
  works or raises; it can never silently fall back to random.
- P2: the loss is computed with dense ``[B, N]`` masks - batch-size safe.
- P3: masked tokens are replaced by a learnable mask token and a binary
  ``is_masked`` indicator channel is appended, so the model knows what to
  reconstruct and finetuning inputs are not distributionally distorted.
- P6: reconstruction targets default to a caller-chosen *dynamic* feature
  subset, standardized per feature over the batch's valid tokens.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dfm.tokens import TokenBatch


class MaskedTokenPretraining(nn.Module):
    """Apply a masking strategy to a TokenBatch and score reconstructions.

    Usage::

        masker = MaskedTokenPretraining(num_features=7, target_features=[0])
        masked_batch, mask_map = masker.mask(tokens, strategy="group")
        emb = encoder([masked_batch])[tokens.modality]
        pred = recon_head(emb)               # [B, N, num_features]
        loss = masker.loss(pred, tokens, mask_map)

    Note ``masked_batch.features`` has ``num_features + 1`` channels (the
    indicator); size the encoder's ModalityConfig accordingly.
    """

    def __init__(self, num_features: int, mask_ratio: float = 0.15,
                 target_features: Optional[Sequence[int]] = None,
                 region_scale: float = 0.3, seed: Optional[int] = None):
        super().__init__()
        self.num_features = num_features
        self.mask_ratio = mask_ratio
        self.target_features = list(range(num_features)) if target_features is None \
            else list(target_features)
        self.region_scale = region_scale
        self.mask_token = nn.Parameter(torch.zeros(num_features))
        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ mask
    def _rand(self, *shape, device=None) -> Tensor:
        # draw on CPU with the (seedable) generator, then move: reproducible
        # regardless of where the tokens live
        return torch.rand(*shape, generator=self.generator).to(device)

    def _select(self, tokens: TokenBatch, strategy: str,
                feature_columns: Optional[Sequence[int]]) -> Tensor:
        b, n = tokens.mask.shape
        valid = tokens.mask
        dev = valid.device
        if strategy in ("random", "feature"):
            sel = (self._rand(b, n, device=dev) < self.mask_ratio) & valid
        elif strategy == "region":
            if tokens.coords is None:
                raise ValueError("region masking requires TokenBatch.coords")
            sel = torch.zeros_like(valid)
            for i in range(b):
                idx = valid[i].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                seed_tok = idx[int(self._rand(1).item() * idx.numel()) % idx.numel()]
                c = tokens.coords[i]
                span = (c[idx].max(dim=0).values - c[idx].min(dim=0).values)
                radius = self.region_scale * span.norm() / 2.0
                d = (c - c[seed_tok]).norm(dim=-1)
                sel[i] = (d <= radius) & valid[i]
        elif strategy == "group":
            if tokens.group_ids is None:
                raise ValueError(
                    "group masking requires TokenBatch.group_ids (cluster ids for "
                    "cells, vertex ids for tracks) - refusing to fall back to "
                    "random masking silently (review P1)")
            sel = torch.zeros_like(valid)
            for i in range(b):
                gids = tokens.group_ids[i]
                groups = torch.unique(gids[(gids >= 0) & valid[i]])
                if groups.numel() == 0:
                    continue
                n_pick = max(1, int(round(self.mask_ratio * groups.numel())))
                perm = torch.randperm(groups.numel(),
                                      generator=self.generator).to(dev)
                chosen = groups[perm[:n_pick]]
                sel[i] = torch.isin(gids, chosen) & valid[i]
        else:
            raise ValueError(f"unknown strategy '{strategy}'")
        return sel

    def mask(self, tokens: TokenBatch, strategy: str = "random",
             feature_columns: Optional[Sequence[int]] = None
             ) -> Tuple[TokenBatch, Tensor]:
        """Return (masked TokenBatch with indicator channel, mask_map [B, N])."""
        sel = self._select(tokens, strategy, feature_columns)
        x = tokens.features.clone()
        mask_token = self.mask_token.to(x.device)
        if strategy == "feature":
            cols = list(feature_columns) if feature_columns is not None \
                else self.target_features
            if not set(cols) <= set(self.target_features):
                raise ValueError(
                    f"feature_columns {cols} must be a subset of "
                    f"target_features {self.target_features}: the loss scores "
                    f"target_features, so masking other columns would go "
                    f"unscored while unmasked columns are penalized")
            for c in cols:
                x[..., c] = torch.where(sel, mask_token[c].expand_as(x[..., c]),
                                        x[..., c])
        else:
            x = torch.where(sel.unsqueeze(-1), mask_token.expand_as(x), x)
        indicator = sel.float().unsqueeze(-1)
        x = torch.cat([x, indicator], dim=-1)
        out = TokenBatch(features=x, mask=tokens.mask, modality=tokens.modality,
                         coords=tokens.coords, edges=tokens.edges,
                         group_ids=tokens.group_ids, extras=tokens.extras)
        return out, sel

    # ------------------------------------------------------------------ loss
    def loss(self, predictions: Tensor, clean_tokens: TokenBatch,
             mask_map: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Standardized MSE on masked positions, target features only.

        predictions [B, N, num_features] (indicator channel NOT predicted),
        clean_tokens: the un-masked TokenBatch, mask_map [B, N] bool.
        Returns (scalar loss, per-feature detached breakdown).
        """
        target = clean_tokens.features
        sel = mask_map & clean_tokens.mask
        if sel.sum() == 0:
            return predictions.sum() * 0.0, {}
        valid = clean_tokens.mask
        total, n_scored, breakdown = None, 0, {}
        for c in self.target_features:
            col = target[..., c]
            sd = col[valid].std()
            if sd < 1e-3:
                # (near-)constant column in this batch - standardizing would
                # blow the loss up by 1/sd^2 (e.g. snr_gt2 under an SNR>2
                # subset is identically 1). Skip it, and say so.
                breakdown[f"f{c}"] = float("nan")
                continue
            err = ((predictions[..., c] - col) / sd) ** 2
            term = err[sel].mean()
            total = term if total is None else total + term
            n_scored += 1
            breakdown[f"f{c}"] = float(term.detach())
        if total is None:
            return predictions.sum() * 0.0, breakdown
        return total / n_scored, breakdown
