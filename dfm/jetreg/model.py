"""Config-driven per-jet regression model on the dfm shared backbone.

Config string grammar: subset of {T, J, C} plus cell encoding and options,
e.g. "J", "T", "C-graph", "C-set", "TC-graph", "TJC-set", "C-graph+mu".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from dfm.encoder import SharedEventEncoder, ModalityConfig
from dfm.data import TRACK_FEATURES
from dfm.tokens import TokenBatch

N_CELL_FEAT = 7
N_TRACK_FEAT = len(TRACK_FEATURES)
N_JET_FEAT = 6


@dataclass
class JetRegConfig:
    cells: str = "off"        # "graph" | "set" | "off"
    tracks: bool = False
    jet: bool = False
    mu: bool = False
    flavor_cond: bool = False
    panoptic: bool = False
    dim: int = 128
    local_depth: int = 2
    global_depth: int = 2
    num_heads: int = 4
    num_inds: int = 32
    dropout: float = 0.1

    @classmethod
    def parse(cls, name: str) -> "JetRegConfig":
        """e.g. 'TC-graph+mu' -> tracks + cells(graph) + mu conditioning."""
        base = name.split("+")
        opts = base[1:]
        core = base[0]
        cells = "off"
        if "-" in core:
            core, enc = core.split("-", 1)
            if "C" not in core:
                raise ValueError(f"'{name}': encoding given but no C input")
            if enc not in ("graph", "set"):
                raise ValueError(f"unknown cell encoding '{enc}'")
            cells = enc
        elif "C" in core:
            raise ValueError(f"'{name}': C requires an encoding (-graph|-set)")
        bad = set(core) - set("TJC")
        if bad:
            raise ValueError(f"unknown inputs {bad} in '{name}'")
        return cls(cells=cells, tracks="T" in core, jet="J" in core,
                   mu="mu" in opts, flavor_cond="fc" in opts,
                   panoptic="pan" in opts)


class JetRegModel(nn.Module):
    def __init__(self, cfg: JetRegConfig):
        super().__init__()
        self.cfg = cfg
        mods = []
        if cfg.cells != "off":
            mods.append(ModalityConfig(
                "cell", N_CELL_FEAT,
                local="edges" if cfg.cells == "graph" else "none",
                local_depth=cfg.local_depth, mask_indicator=True))
        if cfg.tracks:
            mods.append(ModalityConfig("track", N_TRACK_FEAT, local="knn",
                                       local_depth=cfg.local_depth))
        self.encoder = SharedEventEncoder(
            mods, dim=cfg.dim, global_depth=cfg.global_depth,
            num_heads=cfg.num_heads, num_inds=cfg.num_inds,
            dropout=cfg.dropout) if mods else None

        head_in = (cfg.dim if mods else 0)
        if cfg.jet:
            self.jet_embed = nn.Sequential(
                nn.Linear(N_JET_FEAT, cfg.dim), nn.GELU(),
                nn.Linear(cfg.dim, cfg.dim))
            head_in += cfg.dim
        if cfg.mu:
            head_in += 1
        if cfg.flavor_cond:
            head_in += 3
        if head_in == 0:
            raise ValueError("config enables no inputs at all")
        out_dim = 5 if cfg.panoptic else 2   # [p_b,p_c,p_l logits, mu, logvar]
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, cfg.dim), nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, out_dim))

    def forward(self, batch: Dict) -> torch.Tensor:
        """batch: {'cells': TokenBatch?, 'tracks': TokenBatch?,
        'jet': [B,6]?, 'mu': [B,1]?}. Returns [B, 2] = (mu, log_var)."""
        parts = []
        if self.encoder is not None:
            toks = []
            if self.cfg.cells != "off":
                toks.append(batch["cells"])
            if self.cfg.tracks:
                toks.append(batch["tracks"])
            parts.append(self.encoder(toks)["global"])
        if self.cfg.jet:
            parts.append(self.jet_embed(batch["jet"]))
        if self.cfg.mu:
            parts.append(batch["mu"])
        if self.cfg.flavor_cond:
            parts.append(batch["flav"])
        return self.head(torch.cat(parts, dim=-1))
