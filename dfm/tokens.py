"""Token containers for detector elements.

A *token* is one detector element (calorimeter cell, track, ...) with a
feature vector. Tokens of one modality for a batch of events live in a
dense padded tensor plus a validity mask - the representation both existing
models already use on the track side, adopted here for every modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class TokenBatch:
    """A padded batch of same-modality tokens.

    Attributes:
        features: float tensor ``[B, N, F]`` (zero-padded).
        mask: bool tensor ``[B, N]``, True = valid token.
        modality: name, e.g. ``"cell"`` or ``"track"``.
        coords: optional ``[B, N, C]`` coordinates for k-NN local mixing
            (e.g. (eta, phi)); defaults to None.
        edges: optional per-event edge lists for fixed-graph local mixing:
            a list of ``[E_i, 2]`` long tensors indexing the *valid* tokens
            of event i (0..n_i-1). Stored undirected-single (src < dst) or
            directed; the conv mirrors them.
        group_ids: optional ``[B, N]`` long tensor of group labels for group
            masking (cluster index for cells, vertex index for tracks);
            values < 0 = no group. Explicitly threaded so cluster masking
            can never silently degrade to random (review P1).
        extras: free-form per-modality side data (labels, raw arrays, ...).
    """

    features: Tensor
    mask: Tensor
    modality: str
    coords: Optional[Tensor] = None
    edges: Optional[List[Tensor]] = None
    group_ids: Optional[Tensor] = None
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.features.shape[1]

    @property
    def num_features(self) -> int:
        return self.features.shape[2]

    def to(self, device: torch.device, non_blocking: bool = True) -> "TokenBatch":
        return TokenBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
            modality=self.modality,
            coords=None if self.coords is None else self.coords.to(device, non_blocking=non_blocking),
            edges=None if self.edges is None else [e.to(device, non_blocking=non_blocking) for e in self.edges],
            group_ids=None if self.group_ids is None else self.group_ids.to(device, non_blocking=non_blocking),
            extras=self.extras,
        )


def pad_token_list(per_event: Sequence[Tensor], modality: str,
                   coords: Optional[Sequence[Tensor]] = None,
                   edges: Optional[Sequence[Tensor]] = None,
                   group_ids: Optional[Sequence[Tensor]] = None) -> TokenBatch:
    """Pad a list of per-event ``[n_i, F]`` feature tensors into a TokenBatch."""
    b = len(per_event)
    n_max = max(int(t.shape[0]) for t in per_event)
    f = int(per_event[0].shape[1])
    features = per_event[0].new_zeros((b, n_max, f))
    mask = torch.zeros((b, n_max), dtype=torch.bool)
    coord_out = None
    if coords is not None:
        c = int(coords[0].shape[1])
        coord_out = coords[0].new_zeros((b, n_max, c))
    gid_out = None
    if group_ids is not None:
        gid_out = torch.full((b, n_max), -1, dtype=torch.long)
    for i, t in enumerate(per_event):
        n = int(t.shape[0])
        features[i, :n] = t
        mask[i, :n] = True
        if coord_out is not None:
            coord_out[i, :n] = coords[i]
        if gid_out is not None:
            gid_out[i, :n] = group_ids[i].long()
    return TokenBatch(features=features, mask=mask, modality=modality,
                      coords=coord_out,
                      edges=None if edges is None else list(edges),
                      group_ids=gid_out)


def combine_token_batches(batches: Sequence[TokenBatch]) -> "CombinedTokens":
    """Concatenate several modalities along the token axis for global mixing.

    Returns a :class:`CombinedTokens` that remembers each modality's slice so
    per-modality embeddings can be recovered after shared attention.
    """
    b = batches[0].batch_size
    for t in batches:
        if t.batch_size != b:
            raise ValueError("all modalities must share the batch dimension")
    d = batches[0].features.shape[-1]
    for t in batches:
        if t.features.shape[-1] != d:
            raise ValueError("combine after projection to the shared width")
    features = torch.cat([t.features for t in batches], dim=1)
    mask = torch.cat([t.mask for t in batches], dim=1)
    slices, start = {}, 0
    for t in batches:
        slices[t.modality] = (start, start + t.num_tokens)
        start += t.num_tokens
    return CombinedTokens(features=features, mask=mask, slices=slices)


@dataclass
class CombinedTokens:
    """Multi-modality token tensor with per-modality slice bookkeeping."""

    features: Tensor
    mask: Tensor
    slices: Dict[str, tuple]

    def modality(self, name: str) -> Tensor:
        lo, hi = self.slices[name]
        return self.features[:, lo:hi]

    def modality_mask(self, name: str) -> Tensor:
        lo, hi = self.slices[name]
        return self.mask[:, lo:hi]


def subset_subgraph(pairs: Tensor, keep: Tensor) -> tuple:
    """Restrict an edge list to a kept-node subset and reindex.

    Args:
        pairs: ``[E, 2]`` long tensor of node-index pairs over the full set.
        keep: bool tensor ``[num_nodes]``; True = node kept.

    Returns:
        ``(sub_pairs, new_index)``: ``sub_pairs [E_kept, 2]`` reindexed to the
        compact kept-node numbering, and ``new_index [num_nodes]`` mapping old
        index -> compact index (-1 for dropped nodes).
    """
    new_index = torch.full((keep.shape[0],), -1, dtype=torch.long)
    new_index[keep] = torch.arange(int(keep.sum()), dtype=torch.long)
    both = keep[pairs[:, 0]] & keep[pairs[:, 1]]
    sub = pairs[both]
    return torch.stack([new_index[sub[:, 0]], new_index[sub[:, 1]]], dim=1), new_index
