"""Event-level models: MET regression (T3) and query-based jet finding (T4)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dfm.encoder import SharedEventEncoder, ModalityConfig
from dfm.layers import MaskedMAB
from dfm.data import TRACK_FEATURES

N_TRACK = len(TRACK_FEATURES)
N_CELL = 7


def make_encoder(use_cells: bool, cell_encoding: str = "graph",
                 dim: int = 128) -> SharedEventEncoder:
    mods = [ModalityConfig("track", N_TRACK, local="knn", local_depth=2)]
    if use_cells:
        mods.append(ModalityConfig(
            "cell", N_CELL, local="edges" if cell_encoding == "graph" else "none",
            local_depth=2, mask_indicator=True))
    return SharedEventEncoder(mods, dim=dim, global_depth=2)


class METModel(nn.Module):
    """Event tokens -> (MET_x, MET_y) in units of 100 GeV."""

    def __init__(self, use_cells: bool, dim: int = 128):
        super().__init__()
        self.use_cells = use_cells
        self.encoder = make_encoder(use_cells, dim=dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Linear(dim, 2))

    def forward(self, batch: Dict) -> Tensor:
        toks = [batch["tracks"]]
        if self.use_cells:
            toks.append(batch["cells"])
        return self.head(self.encoder(toks)["global"])


class JetFinder(nn.Module):
    """Learned queries + existence head + kinematics: constituents in, jets out.

    Per query: [exist_logit, log_pt, eta, sin_phi, cos_phi].
    """

    def __init__(self, use_cells: bool, num_queries: int = 12, dim: int = 128,
                 depth: int = 3):
        super().__init__()
        self.use_cells = use_cells
        self.encoder = make_encoder(use_cells, dim=dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.cross = nn.ModuleList([MaskedMAB(dim, dim, dim, 4) for _ in range(depth)])
        self.selfa = nn.ModuleList([MaskedMAB(dim, dim, dim, 4) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Linear(dim, 5))

    def forward(self, batch: Dict) -> Tensor:
        toks = [batch["tracks"]]
        if self.use_cells:
            toks.append(batch["cells"])
        out = self.encoder(toks)
        comb = out["combined"]
        z = self.queries.expand(comb.features.shape[0], -1, -1)
        for cross, selfa in zip(self.cross, self.selfa):
            z, _ = cross(z, comb.features, comb.mask)
            z, _ = selfa(z, z, None)
        return self.head(z)          # [B, K, 5]


def hungarian_targets(pred: Tensor, truth_pt: Tensor, truth_eta: Tensor,
                      truth_phi: Tensor, truth_mask: Tensor):
    """Per-event Hungarian assignment of queries to truth jets.

    pred [B, K, 5]; truth arrays [B, M] with mask. Returns (match_q, match_t)
    lists of index tensors per event.
    """
    from scipy.optimize import linear_sum_assignment
    B, K, _ = pred.shape
    out = []
    p_exist = torch.sigmoid(pred[..., 0]).detach().cpu()
    p_lpt = pred[..., 1].detach().cpu()
    p_eta = pred[..., 2].detach().cpu()
    p_phi = torch.atan2(pred[..., 3], pred[..., 4]).detach().cpu()
    for b in range(B):
        m = truth_mask[b].cpu().numpy()
        n_t = int(m.sum())
        if n_t == 0:
            out.append((torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long)))
            continue
        t_lpt = torch.log(truth_pt[b, :n_t].cpu().clamp(min=1e-3))
        t_eta = truth_eta[b, :n_t].cpu()
        t_phi = truth_phi[b, :n_t].cpu()
        dphi = torch.atan2(torch.sin(p_phi[b][:, None] - t_phi[None, :]),
                           torch.cos(p_phi[b][:, None] - t_phi[None, :]))
        cost = ((p_lpt[b][:, None] - t_lpt[None, :]).abs()
                + (p_eta[b][:, None] - t_eta[None, :]).abs()
                + dphi.abs() - 2.0 * p_exist[b][:, None])
        qi, ti = linear_sum_assignment(cost.numpy())
        out.append((torch.as_tensor(qi, dtype=torch.long),
                    torch.as_tensor(ti, dtype=torch.long)))
    return out


def jetfinder_loss(pred: Tensor, truth_pt: Tensor, truth_eta: Tensor,
                   truth_phi: Tensor, truth_mask: Tensor,
                   w_exist: float = 1.0, w_kin: float = 2.0) -> Tensor:
    """DETR-style set loss: existence BCE + matched kinematic L1."""
    matches = hungarian_targets(pred, truth_pt, truth_eta, truth_phi, truth_mask)
    B, K, _ = pred.shape
    exist_target = torch.zeros(B, K, device=pred.device)
    kin_terms = []
    for b, (qi, ti) in enumerate(matches):
        if len(qi) == 0:
            continue
        qi_d = qi.to(pred.device)
        ti_d = ti.to(pred.device)
        exist_target[b, qi_d] = 1.0
        t_lpt = torch.log(truth_pt[b, ti_d].clamp(min=1e-3))
        t_eta = truth_eta[b, ti_d]
        t_phi = truth_phi[b, ti_d]
        pq = pred[b, qi_d]
        dphi = torch.atan2(pq[:, 3], pq[:, 4]) - t_phi
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        kin_terms.append((pq[:, 1] - t_lpt).abs() + (pq[:, 2] - t_eta).abs()
                         + dphi.abs())
    exist_loss = nn.functional.binary_cross_entropy_with_logits(
        pred[..., 0], exist_target)
    kin_loss = torch.cat(kin_terms).mean() if kin_terms else pred.sum() * 0.0
    return w_exist * exist_loss + w_kin * kin_loss
