#!/usr/bin/env python3
"""
DETR-like Event-Level B-Tagging Model with Set Transformer Backbone

Architecture:
  1. EventEncoderSetTx/EventEncoderPET: Encodes tracks into contextual embeddings + global pooled token
     - EventEncoderSetTx: Pure Set Transformer (ISAB layers)
     - EventEncoderPET: Point-Edge Transformer (EdgeConv + ISAB layers)
  2. JetQueriesDecoder: Jet queries (from kinematics) cross-attend to track embeddings
  3. ClassificationHeads: Per-jet binary classification (b-jet or light jet)

Handles variable-length tracks and jets with mask support throughout.

Expected .npz input (from make_event_level_btagging.py):
  events: [n_events] array of dicts with:
    - 'tracks': [n_tracks, n_features]
      Feature order: [Track_pt, Track_eta, Track_phi, Track_charge, ...]
      NOTE: For PET encoder, indices 1-2 (eta, phi) are used for EdgeConv k-NN graph construction
    - 'jets_pt', 'jets_eta', 'jets_phi', 'jets_m': [n_jets]
    - 'jets_label': [n_jets] in {0, 1}
    - 'track_mask': [n_tracks] bool
    - 'jet_mask': [n_jets] bool
    - 'weight': float (b-jet/light-jet ratio)
"""

import os
import math
import argparse
import numpy as np
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


def default_mask_from_data(x: torch.Tensor) -> torch.Tensor:
    """Infer mask (True = valid row) by checking if any feature is nonzero."""
    return (x.abs().sum(dim=-1) > 0)


def count_trainable_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_binary_from_logits(logits: torch.Tensor,
                                targets: torch.Tensor,
                                mask: Optional[torch.Tensor],
                                threshold: float = 0.5) -> torch.Tensor:
    """
    Thresholded accuracy for binary logits.
    logits:  [B,M,1]
    targets: [B,M] in {0,1,-1}
    mask:    [B,M] True=valid jet
    """
    with torch.no_grad():
        if mask is not None:
            targets = targets.masked_fill(~mask, -1)
        valid = (targets >= 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        probs = torch.sigmoid(logits.squeeze(-1))
        pred = (probs >= threshold).long()
        return (pred[valid] == targets[valid]).float().mean()


def roc_auc_from_logits(logits: torch.Tensor,
                        targets: torch.Tensor,
                        mask: Optional[torch.Tensor]) -> float:
    """
    ROC AUC score for binary classification.
    logits:  [B,M,1]
    targets: [B,M] in {0,1,-1}
    mask:    [B,M] True=valid jet
    """
    with torch.no_grad():
        valid = (targets >= 0)
        if mask is not None:
            valid = valid & mask
        if valid.sum() == 0:
            return 0.0
        
        probs = torch.sigmoid(logits.squeeze(-1))
        probs_valid = probs[valid].cpu().numpy()
        targets_valid = targets[valid].cpu().numpy()
        
        try:
            auc = roc_auc_score(targets_valid, probs_valid)
            return float(auc)
        except Exception as e:
            print(f"[warning] Could not compute ROC AUC: {e}")
            return 0.0


class MaskedBCEWithLogits(nn.Module):
    """Masked BCEWithLogits for per-jet binary classification."""
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = "mean", use_focal: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.pos_weight = None if pos_weight is None else torch.tensor([float(pos_weight)], dtype=torch.float32)
        assert reduction in ("mean", "sum")
        self.reduction = reduction
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.crit = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

    def set_pos_weight(self, pw: Optional[float], device):
        self.pos_weight = None if pw is None else torch.tensor([float(pw)], dtype=torch.float32, device=device)
        self.crit = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

    def forward(self, logits, targets, mask: Optional[torch.Tensor]):
        """
        logits:  [B,M,1]
        targets: [B,M] in {0,1,-1}
        mask:    [B,M] True=valid jet
        """
        valid = (targets >= 0)
        if mask is not None:
            valid = valid & mask
        if valid.sum() == 0:
            return logits.sum() * 0.0

        l = logits.squeeze(-1)[valid]
        t = targets[valid].float()
        loss_per = self.crit(l, t)
        
        # Apply focal loss if enabled
        if self.use_focal:
            p_t = torch.sigmoid(l)
            p_t = torch.where(t == 1, p_t, 1 - p_t)
            focal_weight = (1 - p_t) ** self.focal_gamma
            loss_per = self.focal_alpha * focal_weight * loss_per
        
        if self.reduction == "sum":
            loss = loss_per.sum()
        else:
            loss = loss_per.mean()
        return loss


# =========================
# Set Transformer Blocks (mask-aware)
# =========================
class MAB(nn.Module):
    """Multihead Attention Block (standard, no masks)."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MABMasked(nn.Module):
    """
    Mask-aware Multihead Attention Block.
    Queries attend to keys/values with optional key-padding mask.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, return_attn=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.return_attn = return_attn
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, K_mask: Optional[torch.Tensor] = None):
        """
        Q: [B, MQ, dQ], K: [B, NK, dK], K_mask: [B, NK] (True=valid)
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        d = self.dim_V // self.num_heads
        Qh = torch.cat(Q.split(d, dim=2), dim=0)  # [B*h, MQ, d]
        Kh = torch.cat(K.split(d, dim=2), dim=0)  # [B*h, NK, d]
        Vh = torch.cat(V.split(d, dim=2), dim=0)  # [B*h, NK, d]

        S = Qh @ Kh.transpose(1, 2) / math.sqrt(self.dim_V)  # [B*h, MQ, NK]

        if K_mask is not None:
            if K_mask.size(1) != K.size(1):
                if K_mask.size(1) > K.size(1):
                    K_mask = K_mask[:, :K.size(1)]
                else:
                    pad_len = K.size(1) - K_mask.size(1)
                    pad = torch.ones(K_mask.size(0), pad_len, dtype=K_mask.dtype, device=K_mask.device)
                    K_mask = torch.cat([K_mask, pad], dim=1)
            Kmask_h = K_mask.repeat_interleave(self.num_heads, dim=0)  # [B*h, NK]
            S = S.masked_fill(~Kmask_h.unsqueeze(1), -1e9)

        A = torch.softmax(S, dim=-1)  # [B*h, MQ, NK]
        Oh = Qh + A @ Vh  # residual
        O = torch.cat(Oh.split(Q.size(0), dim=0), dim=2)  # [B, MQ, dim_V]
        O = self.ln0(O) if hasattr(self, 'ln0') else O
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O) if hasattr(self, 'ln1') else O

        if self.return_attn:
            A_all = torch.cat(A.split(Q.size(0), dim=0), dim=2)  # [B, MQ, NK*h]
            A_mean = A_all.view(Q.size(0), A.size(1), self.num_heads, -1).mean(dim=2)  # [B, MQ, NK]
            return O, A_mean
        return O, None


class SAB(nn.Module):
    """Self-Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Self-Attention Block (scalable, O(N*m))."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


# =========================
# EdgeConv Layer (DGCNN/PET-style)
# =========================
def knn_indices(x, k, mask=None):
    """
    Find k-nearest neighbors for each point.
    
    Args:
        x: [B, N, D] coordinates (e.g., eta, phi)
        k: number of neighbors
        mask: [B, N] boolean mask (True = valid)
    
    Returns:
        indices: [B, N, k] indices of k nearest neighbors
    """
    B, N, D = x.shape
    
    # Compute pairwise distances: [B, N, N]
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1)
    
    # Apply mask: set distance to invalid points to large value
    if mask is not None:
        # Mask out invalid points (set their distance to infinity)
        mask_mat = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
        pairwise_distance = pairwise_distance.masked_fill(~mask_mat, 1e9)
    
    # Get k+1 nearest (includes self), then remove self
    idx = pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1]  # [B, N, k+1]
    idx = idx[:, :, 1:]  # [B, N, k] (remove self)
    
    return idx


def get_edge_features(x, idx):
    """
    Gather features of k-nearest neighbors.
    
    Args:
        x: [B, N, D] features
        idx: [B, N, k] neighbor indices
    
    Returns:
        edge_features: [B, N, k, D] neighbor features
    """
    B, N, D = x.shape
    _, _, k = idx.shape
    
    # Expand indices for gathering
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    
    # Gather neighbors
    x_neighbors = x.view(B * N, -1)[idx, :]
    x_neighbors = x_neighbors.view(B, N, k, D)
    
    return x_neighbors


class EdgeConv(nn.Module):
    """
    EdgeConv layer from DGCNN.
    Learns features from local neighborhoods defined by k-NN in coordinate space.
    
    Architecture similar to PET's get_neighbors function:
    - Find k nearest neighbors in coordinate space (eta, phi)
    - Compute edge features: concat(neighbor_features - center_features, center_features)
    - Apply MLP to edge features
    - Aggregate (mean pooling over neighbors)
    """
    def __init__(self, in_dim, out_dim, k=10):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU()
        )
    
    def forward(self, x, coords, mask=None):
        """
        Args:
            x: [B, N, in_dim] input features
            coords: [B, N, 2] coordinates for k-NN (e.g., eta, phi)
            mask: [B, N] boolean mask (True = valid)
        
        Returns:
            out: [B, N, out_dim] output features
        """
        # Find k-nearest neighbors in coordinate space
        idx = knn_indices(coords, self.k, mask)  # [B, N, k]
        
        # Get neighbor features
        x_neighbors = get_edge_features(x, idx)  # [B, N, k, in_dim]
        
        # Compute edge features: [neighbor - center, center]
        x_center = x.unsqueeze(2).expand_as(x_neighbors)  # [B, N, k, in_dim]
        edge_features = torch.cat([x_neighbors - x_center, x_center], dim=-1)  # [B, N, k, 2*in_dim]
        
        # Apply MLP to edge features
        edge_features = self.mlp(edge_features)  # [B, N, k, out_dim]
        
        # Aggregate: mean pooling over neighbors
        out = edge_features.mean(dim=2)  # [B, N, out_dim]
        
        # Apply mask
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        
        return out


# =========================
# Event Encoder (Track processor)
# =========================
class EventEncoderSetTx(nn.Module):
    """
    Processes variable-length track sets into contextual embeddings + global token.
    Uses ISAB stacks for O(N*m) scalability.
    
    Output: Hc [B, 1+N, d_hidden], Hc_mask [B, 1+N]
      - First token: global (masked mean pooling)
      - Remaining tokens: contextual per-track embeddings
    """
    def __init__(self, dim_input, dim_hidden=128, num_heads=4, num_inds=32, depth=2, ln=True):
        super().__init__()
        enc = []
        din = dim_input
        for _ in range(depth):
            enc.append(ISAB(din, dim_hidden, num_heads, num_inds, ln=ln))
            din = dim_hidden
        self.enc = nn.Sequential(*enc)

    def forward(self, X, X_mask):
        """
        X: [B, N, Dp], X_mask: [B, N] True=valid
        Returns: Hc [B, 1+N, d_hidden], Hc_mask [B, 1+N]
        """
        H = self.enc(X)  # [B, N, dim_hidden]
        denom = X_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        g = (H * X_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B, 1, d]
        Hc = torch.cat([g, H], dim=1)  # [B, 1+N, d]
        Hc_mask = torch.cat([torch.ones_like(X_mask[:, :1], dtype=torch.bool), X_mask], dim=1)
        return Hc, Hc_mask


class EventEncoderPET(nn.Module):
    """
    Point-Edge Transformer (PET) style encoder combining EdgeConv with Set Transformer.
    
    Architecture inspired by Point-Edge Transformer:
    1. Initial feature encoding: MLP to project input features to hidden dimension
    2. Local processing with EdgeConv layers: Captures geometric structure via k-NN graphs
       - Constructs k-NN graph using spatial coordinates (eta, phi) from indices 1-2
       - For each point, aggregates information from k nearest neighbors
       - Edge features: concat(neighbor_feat - center_feat, center_feat)
       - Multiple layers allow hierarchical local feature learning
       - First layer uses (eta, phi), subsequent layers use learned feature space
    3. Global processing with ISAB layers: Captures long-range dependencies
       - Induced Self-Attention Blocks for O(N*m) complexity
       - Learns global context across all tracks
    4. Residual connections: Combines local geometric and global contextual information
    
    Key differences from pure Set Transformer:
    - EdgeConv captures local geometric structure (inductive bias for spatial data)
    - Better for tasks where nearby points in space are semantically related
    - Hybrid approach: local EdgeConv + global attention
    
    Output: Hc [B, 1+N, d_hidden], Hc_mask [B, 1+N]
      - First token: global (masked mean pooling over all tracks)
      - Remaining tokens: contextual per-track embeddings
    
    Usage:
        # Enable PET encoder with command line:
        python script.py --use_pet --num_local 2 --k 10
        
        # This will use 2 EdgeConv layers with k=10 nearest neighbors
        # followed by standard ISAB layers for global context
        
        # Track features expected order: [Track_pt, Track_eta, Track_phi, ...]
        # EdgeConv uses Track_eta (index 1) and Track_phi (index 2) for k-NN
    """
    def __init__(self, 
                 dim_input, 
                 dim_hidden=128, 
                 num_heads=4, 
                 num_inds=32, 
                 depth=2, 
                 num_local=2,
                 k=10,
                 ln=True,
                 use_local=True):
        """
        Args:
            dim_input: Input feature dimension
            dim_hidden: Hidden dimension for all layers
            num_heads: Number of attention heads for ISAB
            num_inds: Number of inducing points for ISAB
            depth: Number of ISAB layers (global attention)
            num_local: Number of EdgeConv layers (local processing)
            k: Number of neighbors for EdgeConv
            ln: Use LayerNorm
            use_local: Whether to use EdgeConv layers
        """
        super().__init__()
        self.use_local = use_local
        self.dim_hidden = dim_hidden
        
        # Initial feature encoding
        self.input_encoder = nn.Sequential(
            nn.Linear(dim_input, dim_hidden * 2),
            nn.GELU(),
            nn.Linear(dim_hidden * 2, dim_hidden)
        )
        
        # Local EdgeConv layers (process geometric neighborhoods)
        if use_local:
            self.edge_convs = nn.ModuleList([
                EdgeConv(dim_hidden, dim_hidden, k=k)
                for _ in range(num_local)
            ])
        
        # Global ISAB layers (process long-range dependencies)
        isab_layers = []
        for _ in range(depth):
            isab_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.isab_layers = nn.Sequential(*isab_layers)
    
    def forward(self, X, X_mask):
        """
        Args:
            X: [B, N, Dp] input track features
                Feature order: [Track_pt, Track_eta, Track_phi, Track_charge, ...]
            X_mask: [B, N] boolean mask (True = valid)
        
        Returns:
            Hc: [B, 1+N, d_hidden] contextual embeddings with global token
            Hc_mask: [B, 1+N] mask
        
        Note: Uses indices 1-2 (eta, phi) as spatial coordinates for k-NN graph construction
        """
        # Extract coordinates for EdgeConv (indices 1-2 are eta, phi)
        coords = X[:, :, 1:3]  # [B, N, 2] - (eta, phi)
        
        # Initial encoding
        encoded = self.input_encoder(X)  # [B, N, dim_hidden]
        
        # Apply mask
        encoded = encoded * X_mask.unsqueeze(-1)
        
        # Store input for skip connection
        skip_connection = encoded
        
        # Local processing with EdgeConv
        if self.use_local:
            local_features = encoded
            for i, edge_conv in enumerate(self.edge_convs):
                local_features = edge_conv(local_features, coords, X_mask)
                # For subsequent layers, use learned feature space for k-NN
                # (first layer uses eta-phi, later layers use learned features)
                if i < len(self.edge_convs) - 1:
                    coords = local_features[:, :, :2]
            
            # Add local features to encoded
            encoded = encoded + local_features
        
        # Global processing with ISAB
        encoded = self.isab_layers(encoded)  # [B, N, dim_hidden]
        
        # Add skip connection
        encoded = encoded + skip_connection
        
        # Apply final mask
        encoded = encoded * X_mask.unsqueeze(-1)
        
        # Create global token via masked mean pooling
        denom = X_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        g = (encoded * X_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B, 1, d]
        
        # Concatenate global token with track embeddings
        Hc = torch.cat([g, encoded], dim=1)  # [B, 1+N, d]
        Hc_mask = torch.cat([
            torch.ones_like(X_mask[:, :1], dtype=torch.bool), 
            X_mask
        ], dim=1)
        
        return Hc, Hc_mask


# =========================
# Jet Decoder (queries + cross-attention)
# =========================
class JetQueriesDecoder(nn.Module):
    """
    DETR-style decoder:
      1. Embed jet kinematics into queries
      2. Cross-attend to track embeddings (multi-layer)
      3. Per-jet classification head
    """
    def __init__(self, jet_in_dim=4, d_model=128, num_heads=4, depth=2, ln=True, return_attn=True):
        super().__init__()
        self.jet_embed = nn.Sequential(
            nn.Linear(jet_in_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.decoder_layers = nn.ModuleList([
            MABMasked(d_model, d_model, d_model, num_heads, ln=ln, return_attn=(return_attn and (i == depth - 1)))
            for i in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)  # single logit for binary classification
        )
        self.return_attn = return_attn

    def forward(self, J, J_mask, Hc, Hc_mask):
        """
        J: [B, M, 4] jet kinematics (eta, phi, pt, m)
        J_mask: [B, M] True=valid jet
        Hc: [B, 1+N, d_model] track embeddings (1 global + N contextual)
        Hc_mask: [B, 1+N] True=valid
        
        Returns:
          logits: [B, M, 1] per-jet binary classification logits
          attn: [B, M, N] attention weights to tracks (None if not requested)
        """
        Z = self.jet_embed(J)  # [B, M, d_model]
        A_last = None

        for layer in self.decoder_layers:
            Z, A = layer(Z, Hc, K_mask=Hc_mask)  # jet queries cross-attend to track embeddings
            if A is not None:
                A_last = A

        logits = self.head(Z)  # [B, M, 1]
        
        # Extract attention to actual tracks (skip global token at position 0)
        attn = A_last[..., 1:] if (self.return_attn and A_last is not None) else None  # [B, M, N]
        return logits, attn


# =========================
# Full DETR-like model
# =========================
class DETREventBTag_SetTransformer(nn.Module):
    """
    DETR-like event-level b-tagging model.
    
    Inputs:
      - X: [B, N, Dp] track features (variable Dp)
      - X_mask: [B, N] track validity mask
      - J: [B, M, 4] jet kinematics (eta, phi, pt, m)
      - J_mask: [B, M] jet validity mask
    
    Outputs:
      - logits: [B, M, 1] per-jet binary classification logits
      - attn: [B, M, N] attention weights from jets to tracks
    """
    def __init__(self,
                 dim_input_tracks,
                 jet_in_dim=4,
                 dim_hidden=128,
                 num_heads=4,
                 num_inds=32,
                 enc_depth=2,
                 dec_depth=2,
                 ln=True,
                 return_attn=True,
                 use_pet=False,
                 num_local=2,
                 k=10):
        """
        Args:
            dim_input_tracks: Input track feature dimension
            jet_in_dim: Jet kinematics dimension (default 4: eta, phi, pt, m)
            dim_hidden: Hidden dimension
            num_heads: Number of attention heads
            num_inds: Number of inducing points for ISAB
            enc_depth: Number of encoder layers (ISAB/global attention)
            dec_depth: Number of decoder layers
            ln: Use LayerNorm
            return_attn: Return attention weights
            use_pet: Use Point-Edge Transformer encoder (EdgeConv + ISAB)
            num_local: Number of EdgeConv layers (only used if use_pet=True)
            k: Number of neighbors for EdgeConv (only used if use_pet=True)
        """
        super().__init__()
        
        # Choose encoder based on use_pet flag
        if use_pet:
            self.encoder = EventEncoderPET(
                dim_input=dim_input_tracks,
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                num_inds=num_inds,
                depth=enc_depth,
                num_local=num_local,
                k=k,
                ln=ln,
                use_local=True
            )
        else:
            self.encoder = EventEncoderSetTx(
                dim_input=dim_input_tracks,
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                num_inds=num_inds,
                depth=enc_depth,
                ln=ln
            )
        
        self.decoder = JetQueriesDecoder(
            jet_in_dim=jet_in_dim,
            d_model=dim_hidden,
            num_heads=num_heads,
            depth=dec_depth,
            ln=ln,
            return_attn=return_attn
        )

    def forward(self, X, X_mask, J, J_mask):
        """
        X: [B, N, Dp], X_mask: [B, N]
        J: [B, M, 4], J_mask: [B, M]
        """
        Hc, Hc_mask = self.encoder(X, X_mask)
        logits, attn = self.decoder(J, J_mask, Hc, Hc_mask)
        return logits, attn


# =========================
# Dataset for event-level data
# =========================
class EventLevelDataset(Dataset):
    """
    Loads from .npz saved by make_event_level_btagging.py
    
    Expected keys:
      events: [n_events] array of dicts with:
        - 'tracks': [n_tracks, n_features]
        - 'jets_pt', 'jets_eta', 'jets_phi', 'jets_m': [n_jets]
        - 'jets_label': [n_jets]
        - 'track_mask', 'jet_mask': [n_tracks/n_jets]
      track_features: feature names (optional)
    """
    def __init__(self, path: str, compute_stats: bool = True):
        super().__init__()
        data = np.load(path, allow_pickle=True)
        self.events = data['events']
        self.n_events = len(self.events)
        
        # Feature indices that should be log-transformed
        # Track_pt is at index 0
        self.log_transform_indices = [0]  # Track_pt
        
        # Compute normalization statistics if requested
        self.track_mean = None
        self.track_std = None
        if compute_stats:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std using batch-wise Welford's algorithm (fast + memory-efficient)."""
        n_features = None
        n_total_tracks = 0
        mean = None
        M2 = None  # For Welford's algorithm
        
        # Stream through all events one at a time with progress bar
        for event_dict in tqdm(self.events, desc="Computing statistics"):
            event_data = event_dict.item() if hasattr(event_dict, 'item') else event_dict
            tracks = event_data['tracks']  # [n_tracks, n_features]
            
            if tracks.shape[0] == 0:
                continue
            
            # Initialize on first event with tracks
            if n_features is None:
                n_features = tracks.shape[1]
                mean = np.zeros(n_features, dtype=np.float64)
                M2 = np.zeros(n_features, dtype=np.float64)
            
            # Apply log transforms to this batch (vectorized)
            tracks_processed = tracks.astype(np.float64)  # Use float64 for numerical stability
            for idx in self.log_transform_indices:
                tracks_processed[:, idx] = np.log(np.maximum(tracks[:, idx], 1e-6))
            
            # Batch-wise Welford's algorithm (vectorized)
            n_batch = tracks_processed.shape[0]
            if n_total_tracks == 0:
                # First batch
                mean = np.mean(tracks_processed, axis=0)
                M2 = np.sum((tracks_processed - mean) ** 2, axis=0)
                n_total_tracks = n_batch
            else:
                # Subsequent batches - use parallel Welford's algorithm
                old_mean = mean.copy()
                delta = tracks_processed - old_mean
                mean = old_mean + np.sum(delta, axis=0) / (n_total_tracks + n_batch)
                delta2 = tracks_processed - mean
                M2 += np.sum(delta * delta2, axis=0)
                n_total_tracks += n_batch
        
        # Convert to variance then std
        variance = M2 / max(n_total_tracks - 1, 1)
        std = np.sqrt(variance)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        
        self.track_mean = mean.reshape(1, -1).astype(np.float32)
        self.track_std = std.reshape(1, -1).astype(np.float32)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        event_dict = self.events[idx].item() if hasattr(self.events[idx], 'item') else self.events[idx]
        
        X = torch.from_numpy(event_dict['tracks']).float()  # [n_tracks, n_features]
        
        # Apply log transforms
        X_processed = X.clone()
        for feat_idx in self.log_transform_indices:
            X_processed[:, feat_idx] = torch.log(torch.clamp(X[:, feat_idx], min=1e-6))
        
        # Apply normalization if statistics are available
        if self.track_mean is not None:
            X_processed = (X_processed.numpy() - self.track_mean) / self.track_std
            X_processed = torch.from_numpy(X_processed).float()
        
        X_mask = torch.from_numpy(event_dict['track_mask']).bool()  # [n_tracks]
        
        # Stack jet kinematics: [eta, phi, pt, m]
        jets_eta = torch.from_numpy(event_dict['jets_eta']).float()
        jets_phi = torch.from_numpy(event_dict['jets_phi']).float()
        jets_pt = torch.from_numpy(event_dict['jets_pt']).float()
        jets_m = torch.from_numpy(event_dict['jets_m']).float()
        
        # Log transform jet kinematics
        jets_pt_log = torch.log(torch.clamp(jets_pt, min=1e-6))
        jets_m_log = torch.log(torch.clamp(jets_m, min=1e-6))
        
        J = torch.stack([jets_eta, jets_phi, jets_pt_log, jets_m_log], dim=1)  # [n_jets, 4]
        
        J_mask = torch.from_numpy(event_dict['jet_mask']).bool()  # [n_jets]
        Y = torch.from_numpy(event_dict['jets_label']).long()  # [n_jets]
        weight = float(event_dict['weight'])
        
        return {
            'X': X_processed,
            'X_mask': X_mask,
            'J': J,
            'J_mask': J_mask,
            'Y': Y,
            'weight': weight,
        }


def collate_fn_variable_length(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-length tracks/jets.
    Pads to max length in batch.
    """
    # Determine max lengths in batch
    max_tracks = max(len(item['X']) for item in batch)
    max_jets = max(len(item['J']) for item in batch)
    
    batch_X = []
    batch_X_mask = []
    batch_J = []
    batch_J_mask = []
    batch_Y = []
    batch_weight = []
    
    for item in batch:
        n_tracks = len(item['X'])
        n_jets = len(item['J'])
        
        # Pad tracks
        X_padded = torch.zeros(max_tracks, item['X'].shape[1])
        X_padded[:n_tracks] = item['X']
        batch_X.append(X_padded)
        
        X_mask_padded = torch.zeros(max_tracks, dtype=torch.bool)
        X_mask_padded[:n_tracks] = item['X_mask']
        batch_X_mask.append(X_mask_padded)
        
        # Pad jets
        J_padded = torch.zeros(max_jets, 4)
        J_padded[:n_jets] = item['J']
        batch_J.append(J_padded)
        
        J_mask_padded = torch.zeros(max_jets, dtype=torch.bool)
        J_mask_padded[:n_jets] = item['J_mask']
        batch_J_mask.append(J_mask_padded)
        
        # Pad labels (use -1 for padding)
        Y_padded = torch.full((max_jets,), -1, dtype=torch.long)
        Y_padded[:n_jets] = item['Y']
        batch_Y.append(Y_padded)
        
        batch_weight.append(item['weight'])
    
    return {
        'X': torch.stack(batch_X),
        'X_mask': torch.stack(batch_X_mask),
        'J': torch.stack(batch_J),
        'J_mask': torch.stack(batch_J_mask),
        'Y': torch.stack(batch_Y),
        'weight': torch.tensor(batch_weight),
    }


# =========================
# Training / Evaluation
# =========================
def train_one_epoch(model, loader, optimizer, criterion, acc_threshold=0.5, device='cuda', scheduler=None):
    model.train()
    total_loss, total_acc, total_auc, n = 0.0, 0.0, 0.0, 0
    for batch in tqdm(loader, desc="Training"):
        X = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        J = batch['J'].to(device)
        J_mask = batch['J_mask'].to(device)
        Y = batch['Y'].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(X, X_mask, J, J_mask)  # [B, M, 1]
        loss = criterion(logits, Y, J_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        acc = accuracy_binary_from_logits(logits, Y, J_mask, threshold=acc_threshold)
        auc = roc_auc_from_logits(logits, Y, J_mask)
        total_loss += float(loss.detach().cpu())
        total_acc += float(acc.detach().cpu())
        total_auc += auc
        n += 1
    return total_loss / n, total_acc / n, total_auc / n


@torch.no_grad()
def evaluate(model, loader, criterion, acc_threshold=0.5, device='cuda'):
    model.eval()
    # Unwrap DataParallel model if needed to avoid gather issues with variable-length tensors
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    total_loss, total_acc, total_auc, n = 0.0, 0.0, 0.0, 0
    for batch in tqdm(loader, desc="Evaluating"):
        X = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        J = batch['J'].to(device)
        J_mask = batch['J_mask'].to(device)
        Y = batch['Y'].to(device)

        logits, _ = actual_model(X, X_mask, J, J_mask)
        loss = criterion(logits, Y, J_mask)
        acc = accuracy_binary_from_logits(logits, Y, J_mask, threshold=acc_threshold)
        auc = roc_auc_from_logits(logits, Y, J_mask)
        total_loss += float(loss.detach().cpu())
        total_acc += float(acc.detach().cpu())
        total_auc += auc
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_acc / n, total_auc / n


@torch.no_grad()
def collect_test_predictions(model, loader, device='cuda', track_mean=None, track_std=None, log_transform_indices=None):
    """
    Collect all test set predictions and features.
    
    Optionally reverse normalization and log transforms before saving.
    
    Returns dict with:
      - X: [total_tracks, n_features] all tracks (un-normalized and un-log-transformed if stats provided)
      - J: [total_jets, 4] all jet kinematics (un-log-transformed if stats provided)
      - Y: [total_jets] true labels
      - probs: [total_jets] predicted probabilities
      - X_mask: [total_tracks] track validity
      - J_mask: [total_jets] jet validity
      - event_indices: [total_jets] which event each jet belongs to
      - n_events: number of events
    """
    model.eval()
    # Unwrap DataParallel model if needed to avoid gather issues with variable-length tensors
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    all_X = []
    all_J = []
    all_Y = []
    all_probs = []
    all_X_mask = []
    all_J_mask = []
    all_event_idx = []
    
    track_offset = 0
    event_idx = 0
    
    for batch in tqdm(loader, desc="Collecting predictions"):
        X = batch['X'].to(device)
        X_mask = batch['X_mask'].to(device)
        J = batch['J'].to(device)
        J_mask = batch['J_mask'].to(device)
        Y = batch['Y'].to(device)
        
        batch_size = X.shape[0]
        
        # Run model
        logits, _ = actual_model(X, X_mask, J, J_mask)
        probs = torch.sigmoid(logits.squeeze(-1))
        
        # Process each sample in batch
        for i in range(batch_size):
            n_tracks = (X_mask[i].sum()).item()
            n_jets = (J_mask[i].sum()).item()
            
            # Collect tracks for this event (only valid ones)
            X_i = X[i, :n_tracks].cpu().numpy()
            X_mask_i = X_mask[i, :n_tracks].cpu().numpy()
            
            # Collect jets for this event (only valid ones)
            J_i = J[i, :n_jets].cpu().numpy()
            all_Y.append(Y[i, :n_jets].cpu().numpy())
            all_probs.append(probs[i, :n_jets].cpu().numpy())
            all_J_mask.append(J_mask[i, :n_jets].cpu().numpy())
            
            # Reverse normalization and log transforms if stats are provided
            if track_mean is not None and track_std is not None:
                # Un-normalize X
                X_i = X_i * track_std + track_mean
                
                # Un-log-transform features (track_pt at index 0)
                if log_transform_indices:
                    for idx in log_transform_indices:
                        X_i[:, idx] = np.exp(X_i[:, idx])
            
            # Reverse log transforms for J kinematics (pt at index 2, m at index 3)
            if J_i.shape[0] > 0:
                J_i = J_i.copy()
                J_i[:, 2] = np.exp(J_i[:, 2])  # un-log jet_pt
                J_i[:, 3] = np.exp(J_i[:, 3])  # un-log jet_m
            
            all_X.append(X_i)
            all_J.append(J_i)
            all_X_mask.append(X_mask_i)
            
            # Track which event each jet belongs to
            all_event_idx.append(np.full(n_jets, event_idx))
            
            event_idx += 1
    
    # Concatenate all events
    result = {
        'X': np.vstack(all_X) if all_X else np.zeros((0, X.shape[-1])),
        'J': np.vstack(all_J) if all_J else np.zeros((0, 4)),
        'Y': np.concatenate(all_Y) if all_Y else np.array([], dtype=np.int32),
        'probs': np.concatenate(all_probs) if all_probs else np.array([]),
        'X_mask': np.concatenate(all_X_mask) if all_X_mask else np.array([], dtype=bool),
        'J_mask': np.concatenate(all_J_mask) if all_J_mask else np.array([], dtype=bool),
        'event_indices': np.concatenate(all_event_idx) if all_event_idx else np.array([], dtype=np.int32),
        'n_events': event_idx,
    }
    
    print(f"[info] Collected {event_idx} events, {result['J'].shape[0]} jets, {result['X'].shape[0]} tracks")
    if track_mean is not None:
        print(f"[info] Un-normalized and un-log-transformed features before saving")
    
    return result


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="DETR-like Event-Level B-Tagging with Set Transformer Backbone"
    )
    parser.add_argument('--data_npz', type=str, required=True, help='Path to single .npz file with all events')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction for training (default 0.7)')
    parser.add_argument('--val_split', type=float, default=0.10, help='Fraction for validation (default 0.15)')
    parser.add_argument('--test_split', type=float, default=0.10, help='Fraction for testing (default 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--dim_hidden', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--num_inds', type=int, default=48)
    parser.add_argument('--enc_depth', type=int, default=3)
    parser.add_argument('--dec_depth', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pos_weight', type=float, default=None, help='Positive weight for BCE')
    parser.add_argument('--acc_threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for checkpoints')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of standard BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--use_warm_restarts', action='store_true', help='Use cosine annealing with warm restarts (SGDR)')
    parser.add_argument('--restart_interval', type=int, default=10, help='Number of epochs per restart cycle')
    parser.add_argument('--restart_multiplier', type=int, default=2, help='Multiplier for restart interval (T_mult)')
    
    # PET (Point-Edge Transformer) specific arguments
    parser.add_argument('--use_pet', action='store_true', help='Use Point-Edge Transformer encoder (EdgeConv + ISAB)')
    parser.add_argument('--num_local', type=int, default=2, help='Number of EdgeConv layers for local processing (PET only)')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors for EdgeConv k-NN (PET only)')

    args = parser.parse_args()

    # Validate splits
    assert args.train_split + args.val_split + args.test_split == 1.0, \
        "Train, val, and test splits must sum to 1.0"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("[info] Loading dataset")
    full_ds = EventLevelDataset(args.data_npz)
    n_total = len(full_ds)
    print(f"[info] Total events: {n_total}")

    np.random.seed(args.seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"[info] Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    test_ds = Subset(full_ds, test_indices)

    print("[info] Creating data loaders")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_variable_length
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_variable_length
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_variable_length
    )

    sample_event = full_ds[0]
    dim_input_tracks = sample_event['X'].shape[1]
    print(f"[info] Track feature dimension: {dim_input_tracks}")

    print("[info] Instantiating model")
    if args.use_pet:
        print(f"[info] Using Point-Edge Transformer (PET) encoder with {args.num_local} EdgeConv layers (k={args.k})")
    model = DETREventBTag_SetTransformer(
        dim_input_tracks=dim_input_tracks,
        jet_in_dim=4,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_inds=args.num_inds,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        ln=True,
        return_attn=True,
        use_pet=args.use_pet,
        num_local=args.num_local,
        k=args.k
    ).to(args.device)

    # Auto-detect and enable DataParallel for multiple GPUs
    num_gpus = torch.cuda.device_count() if args.device == 'cuda' else 0
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[info] Using DataParallel with {num_gpus} GPUs")
    
    use_dataparallel = isinstance(model, nn.DataParallel)

    print(f"[info] Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = MaskedBCEWithLogits(pos_weight=args.pos_weight, use_focal=args.use_focal_loss, 
                                     focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma)
    criterion.set_pos_weight(args.pos_weight, device=args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Create learning rate scheduler
    if args.use_warm_restarts:
        # Cosine Annealing with Warm Restarts (SGDR)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_interval * len(train_loader),
            T_mult=int(args.restart_multiplier),
            eta_min=0.0,
            last_epoch=-1
        )
        
        def lr_lambda_warmup(current_step: int):
            warmup_steps = len(train_loader) * args.warmup_epochs
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return 1.0
        
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[len(train_loader) * args.warmup_epochs]
        )
        
        print(f"[info] LR scheduler: linear warmup {args.warmup_epochs} epochs, then cosine annealing with warm restarts")
        print(f"[info]   - Restart interval: {args.restart_interval} epochs")
        print(f"[info]   - Restart multiplier (T_mult): {args.restart_multiplier}")
    else:
        # Original scheduler: linear warmup + cosine annealing
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[info] LR scheduler: linear warmup {args.warmup_epochs} epochs then cosine annealing")
    
    if args.use_focal_loss:
        print(f"[info] Using focal loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            acc_threshold=args.acc_threshold, device=args.device, scheduler=scheduler
        )
        
        va_loss, va_acc, va_auc = evaluate(
            model, val_loader, criterion,
            acc_threshold=args.acc_threshold, device=args.device
        )
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(
                model.module.state_dict() if use_dataparallel else model.state_dict(),
                f"{args.output_dir}/best_detr_settx_event_btag.pt"
            )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} | lr {current_lr:.2e}")

    # Load best model for test evaluation
    print(f"\n[info] Loading best model from epoch {best_epoch}...")
    best_checkpoint_path = f"{args.output_dir}/best_detr_settx_event_btag.pt"
    if use_dataparallel:
        model.module.load_state_dict(torch.load(best_checkpoint_path, map_location=args.device))
    else:
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=args.device))
    print(f"[info] Best model loaded successfully (val acc: {best_val_acc:.4f})")

    # Evaluate on test set
    print("\n[info] Evaluating on test set with best model...")
    test_loss, test_acc, test_auc = evaluate(
        model, test_loader, criterion,
        acc_threshold=args.acc_threshold, device=args.device
    )
    print(f"[info] Test loss {test_loss:.4f} acc {test_acc:.4f} auc {test_auc:.4f}")
    
    # Collect test predictions and features
    print("\n[info] Collecting test set predictions...")
    test_predictions = collect_test_predictions(
        model, test_loader, device=args.device,
        track_mean=full_ds.track_mean, track_std=full_ds.track_std,
        log_transform_indices=full_ds.log_transform_indices
    )
    
    # Save test predictions
    test_pred_path = f"{args.output_dir}/test_predictions.npz"
    np.savez(
        test_pred_path,
        X=test_predictions['X'],
        J=test_predictions['J'],
        Y=test_predictions['Y'],
        probs=test_predictions['probs'],
        X_mask=test_predictions['X_mask'],
        J_mask=test_predictions['J_mask'],
        event_indices=test_predictions['event_indices'],
        n_events=test_predictions['n_events'],
    )
    print(f"[info] Saved test predictions to {test_pred_path}")

    torch.save(model.module.state_dict() if use_dataparallel else model.state_dict(), f"{args.output_dir}/last_detr_settx_event_btag.pt")
    print(f"\n[info] ===== SUMMARY =====")
    print(f"[info] Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[info] Test set results (using best model):")
    print(f"[info]   - Test accuracy: {test_acc:.4f}")
    print(f"[info]   - Test ROC AUC: {test_auc:.4f}")
    print(f"[info] Saved checkpoints to {args.output_dir}/")


if __name__ == "__main__":
    main()
