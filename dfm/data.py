"""Adapters from the two existing dataset formats to TokenBatches.

- :class:`CaloGraphAdapter`: Mohammad Ali's processed cell datasets
  (``cells_*.npy`` + ``pairs_*.npy`` + ``events_*.h5``) -> cell tokens with
  the baseline-7 feature schema of ``train_gnn_models.py``
  (snr_scaled, snr>4, snr>2, snr>0, eta, sin phi, cos phi), an SNR-based
  cell subset (review Part 5, S3), the subset's neighbor subgraph, cluster
  group ids, and 5-class / binary edge labels.

- :class:`TrackJetAdapter`: 5D ntuples -> track tokens + jet queries +
  labels, adapted to the branches actually present in the server ntuples
  (the original ``make_event_level_btagging.py`` schema expects ``btagIp_*``
  branches that do not exist there; impact-parameter significances are
  computed from the covariance branches instead). Labels come from
  ``HadronConeExclTruthLabelID`` (fixes review D3) with charm kept as its
  own class (fixes D1): 0 = light, 1 = c, 2 = b.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dfm.tokens import TokenBatch, pad_token_list, subset_subgraph

CALO_FEATURES = ["snr_scaled", "snr_gt4", "snr_gt2", "snr_gt0",
                 "eta", "sin_phi", "cos_phi"]

TRACK_BRANCHES = [
    "Track_pt", "Track_eta", "Track_phi", "Track_charge",
    "Track_d0", "Track_z0", "Track_qOverP",
    "Track_var_d0", "Track_var_z0", "Track_var_qOverP",
    "Track_nPixelHits", "Track_nStripHits",
    "Track_nPixelHoles", "Track_nStripHoles",
    "Track_chi2", "Track_ndof",
]
TRACK_DERIVED = ["d0_sig", "z0_sig"]  # d0/sqrt(var_d0), z0/sqrt(var_z0)
TRACK_FEATURES = TRACK_BRANCHES + TRACK_DERIVED  # column order of the tokens


def _single(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


class CaloGraphAdapter:
    """Cell tokens from a processed foundation-model dataset directory."""

    def __init__(self, dataset_dir: str, snr_min: float = 2.0,
                 e_abs_max_mev: Optional[float] = 100e3):
        """``e_abs_max_mev``: drop cells with |E| above this (default 100 GeV) -
        the agreed cell-level cut against extreme forward-noise deposits
        (FCAL pileup noise reaches tens of GeV/cell). None disables."""
        import h5py  # local import: optional dependency

        self.dataset_dir = dataset_dir
        self.snr_min = snr_min
        self.e_abs_max_mev = e_abs_max_mev
        self.cells = np.load(_single(os.path.join(dataset_dir, "cells_*.npy")))
        self.pairs = np.load(_single(os.path.join(dataset_dir, "pairs_*.npy")))
        self.h5_files = sorted(glob.glob(os.path.join(dataset_dir, "events_*.h5")))
        if not self.h5_files:
            raise FileNotFoundError(f"no events_*.h5 under {dataset_dir}")
        self._h5 = h5py.File(self.h5_files[0], "r")
        self.num_events = int(self._h5["cell/snr_computed"].shape[0])
        self.eta = self.cells["eta_event0"].astype(np.float32)
        self.phi = self.cells["phi_event0"].astype(np.float32)
        self.pairs_t = torch.from_numpy(self.pairs.astype(np.int64))

    def event_tokens(self, indices: Sequence[int]) -> TokenBatch:
        """Cell tokens for the given event rows of the first h5 file.

        Cells with ``|SNR| > snr_min`` are kept (S3 subset); the neighbor
        subgraph is restricted and reindexed per event. Extras carry the
        per-event edge lists' 5-class and binary labels.
        """
        feats, gids, edge_lists, labels5, labels_bin = [], [], [], [], []
        for i in indices:
            snr = self._h5["cell/snr_computed"][i].astype(np.float32)
            clus = self._h5["cell/cell_cluster_index"][i].astype(np.int64)
            keep_np = np.abs(snr) > self.snr_min
            if self.e_abs_max_mev is not None:
                e_raw = self._h5["cell/energy_raw"][i].astype(np.float32)
                keep_np &= np.abs(e_raw) < self.e_abs_max_mev
            keep = torch.from_numpy(keep_np)
            sub_pairs, _ = subset_subgraph(self.pairs_t, keep)
            s = snr[keep_np]
            x = np.stack([
                np.sign(s) * np.log1p(np.abs(s)),
                (np.abs(s) > 4).astype(np.float32),
                (np.abs(s) > 2).astype(np.float32),
                (np.abs(s) > 0).astype(np.float32),
                self.eta[keep_np],
                np.sin(self.phi[keep_np]),
                np.cos(self.phi[keep_np]),
            ], axis=1).astype(np.float32)
            feats.append(torch.from_numpy(x))
            # cluster ids: >=1 means in-cluster; use -1 for lone cells so group
            # masking only ever masks real clusters
            g = clus[keep_np]
            gids.append(torch.from_numpy(np.where(g >= 1, g, -1)))
            edge_lists.append(sub_pairs)
            c0 = g[sub_pairs[:, 0].numpy()]
            c1 = g[sub_pairs[:, 1].numpy()]
            lab = np.zeros(len(sub_pairs), dtype=np.int64)
            in0, in1 = c0 >= 1, c1 >= 1
            lab[in0 & ~in1] = 2
            lab[~in0 & in1] = 3
            lab[in0 & in1 & (c0 == c1)] = 1
            lab[in0 & in1 & (c0 != c1)] = 4
            labels5.append(torch.from_numpy(lab))
            labels_bin.append(torch.from_numpy((lab == 1).astype(np.float32)))
        batch = pad_token_list(feats, "cell", edges=edge_lists, group_ids=gids)
        batch.extras["edge_labels_5class"] = labels5
        batch.extras["edge_labels_binary"] = labels_bin
        return batch


def _greedy_dr_match(reco_eta, reco_phi, truth_eta, truth_phi, dr_max=0.2):
    """One-to-one greedy dR matching; returns bool keep-mask over reco jets."""
    keep = np.zeros(len(reco_eta), dtype=bool)
    if len(truth_eta) == 0:
        return keep
    deta = reco_eta[:, None] - truth_eta[None, :]
    dphi = np.arctan2(np.sin(reco_phi[:, None] - truth_phi[None, :]),
                      np.cos(reco_phi[:, None] - truth_phi[None, :]))
    dr = np.hypot(deta, dphi)
    used = np.zeros(len(truth_eta), dtype=bool)
    order = np.argsort(dr.min(axis=1))
    for r in order:
        t = int(np.argmin(np.where(used, np.inf, dr[r])))
        if not used[t] and dr[r, t] < dr_max:
            keep[r] = True
            used[t] = True
    return keep


class TrackJetAdapter:
    """Track tokens + jet queries + labels from a 5D ntuple."""

    def __init__(self, root_file: str, tree: str = "ntuple",
                 pt_min: float = 1.0, max_tracks: int = 200,
                 jet_eta_max: float = 2.5, max_jets: int = 10,
                 jet_collection: str = "AntiKt4EMTopoJets",
                 vtx_filter: Optional[Tuple[int, ...]] = (0, -1)):
        """``vtx_filter``: keep tracks whose ``Track_recoVtx_idx`` is in this
        set (default = PV-or-unassociated, the b-tagging pileup cut). Pass
        ``None`` to keep all vertices - REQUIRED for vertex ("group") masking
        pretraining: under the default filter at most one vertex group exists,
        so group ids are not emitted at all (group masking would raise, per
        P1, instead of silently masking the entire primary vertex)."""
        import uproot  # local import

        self.tree = uproot.open(root_file)[tree]
        self.pt_min = pt_min
        self.max_tracks = max_tracks
        self.jet_eta_max = jet_eta_max
        self.max_jets = max_jets
        self.jc = jet_collection
        self.vtx_filter = vtx_filter
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None

    def _selection(self, pt: np.ndarray, vtx: np.ndarray) -> np.ndarray:
        """Ordered indices of kept tracks (pt-descending, capped) - the ONE
        place selection logic lives, so tokens and group ids cannot drift."""
        sel = pt > self.pt_min
        if self.vtx_filter is not None:
            sel &= np.isin(vtx, self.vtx_filter)
        idx = np.nonzero(sel)[0]
        # stable sort: deterministic tie-breaking between equal-pt tracks
        order = idx[np.argsort(-pt[idx], kind="stable")][: self.max_tracks]
        return order

    def _read(self, branches: List[str], indices: Sequence[int]):
        import awkward as ak
        lo, hi = min(indices), max(indices) + 1
        arr = self.tree.arrays(branches, entry_start=lo, entry_stop=hi)
        return arr, lo

    def compute_normalization(self, indices: Sequence[int]) -> None:
        """Mean/std of track features over the given (training) events only -
        avoids the full-dataset normalization leakage of the original (M6).
        MUST be called (with training indices) before ``event_tokens``."""
        batch, _ = self._track_matrices(indices)
        allx = np.concatenate(batch, axis=0)
        allx[:, 0] = np.log(np.maximum(allx[:, 0], 1e-6))
        self.norm_mean = allx.mean(axis=0, keepdims=True).astype(np.float32)
        self.norm_std = np.maximum(allx.std(axis=0, keepdims=True), 1e-6).astype(np.float32)

    def save_normalization(self, path: str) -> None:
        np.savez(path, mean=self.norm_mean, std=self.norm_std)

    def load_normalization(self, path: str) -> None:
        data = np.load(path)
        self.norm_mean, self.norm_std = data["mean"], data["std"]

    def _track_matrices(self, indices: Sequence[int]):
        """Returns (per-event feature matrices, per-event kept vtx indices)."""
        import awkward as ak
        arr, lo = self._read(TRACK_BRANCHES + ["Track_recoVtx_idx"], indices)
        out, vtx_out = [], []
        for i in indices:
            e = arr[i - lo]
            cols = {b: np.asarray(ak.to_numpy(e[b]), dtype=np.float32)
                    for b in TRACK_BRANCHES}
            vtx = np.asarray(ak.to_numpy(e["Track_recoVtx_idx"]), dtype=np.int64)
            order = self._selection(cols["Track_pt"], vtx)
            x = np.stack([cols[b][order] for b in TRACK_BRANCHES], axis=1)
            d0_sig = x[:, 4] / np.sqrt(np.maximum(x[:, 7], 1e-12))
            z0_sig = x[:, 5] / np.sqrt(np.maximum(x[:, 8], 1e-12))
            x = np.concatenate([x, d0_sig[:, None], z0_sig[:, None]], axis=1)
            out.append(x.astype(np.float32))
            vtx_out.append(vtx[order])
        return out, vtx_out

    def event_tokens(self, indices: Sequence[int]) -> Tuple[TokenBatch, Dict]:
        """Returns (track TokenBatch, jets dict).

        jets dict: 'queries' [B, M, 4] = (eta, phi, log pT, log m),
        'mask' [B, M], 'labels3' [B, M] in {0=light, 1=c, 2=b, -1=pad or
        excluded}, 'labels_binary' [B, M] in {0, 1, -1}. Jets whose
        ``HadronConeExclTruthLabelID`` is outside {0, 4, 5} (e.g. 15 = tau,
        which carries real lifetime signatures) get label -1 and are excluded
        by the masked losses rather than contaminating the light class.
        """
        import awkward as ak
        tracks, vtx_kept = self._track_matrices(indices)
        if self.norm_mean is None:
            raise ValueError(
                "call compute_normalization(train_indices) (or "
                "load_normalization) before event_tokens - computing "
                "statistics implicitly from arbitrary batches is exactly the "
                "normalization leakage flagged in review M6")
        normed = []
        for x in tracks:
            x = x.copy()
            x[:, 0] = np.log(np.maximum(x[:, 0], 1e-6))
            normed.append(torch.from_numpy((x - self.norm_mean) / self.norm_std))

        jb = [f"{self.jc}_{s}" for s in
              ("pt", "eta", "phi", "m", "HadronConeExclTruthLabelID")]
        arr, lo = self._read(jb + ["TruthHSJet_eta", "TruthHSJet_phi"], indices)
        queries, jmasks, labels3 = [], [], []
        m_max = 1
        per_event = []
        for i in indices:
            e = arr[i - lo]
            pt, eta, phi, m, lab = (np.asarray(ak.to_numpy(e[b]), dtype=np.float64)
                                    for b in jb)
            keep = np.abs(eta) < self.jet_eta_max
            keep &= _greedy_dr_match(eta, phi,
                                     np.asarray(ak.to_numpy(e["TruthHSJet_eta"])),
                                     np.asarray(ak.to_numpy(e["TruthHSJet_phi"])))
            pt, eta, phi, m, lab = (v[keep][: self.max_jets] for v in (pt, eta, phi, m, lab))
            q = np.stack([eta, phi, np.log(np.maximum(pt, 1e-6)),
                          np.log(np.maximum(m, 1e-6))], axis=1).astype(np.float32)
            al = np.abs(lab)
            y3 = np.select([al == 5, al == 4, al == 0], [2, 1, 0], default=-1)
            per_event.append((q, y3.astype(np.int64)))
            m_max = max(m_max, len(q))

        b = len(indices)
        qs = torch.zeros((b, m_max, 4))
        jm = torch.zeros((b, m_max), dtype=torch.bool)
        y3 = torch.full((b, m_max), -1, dtype=torch.long)
        for i, (q, y) in enumerate(per_event):
            n = len(q)
            if n:
                qs[i, :n] = torch.from_numpy(q)
                jm[i, :n] = True
                y3[i, :n] = torch.from_numpy(y)

        # group ids for vertex ("group") masking - only meaningful when the
        # vertex filter is off; under the default {0,-1} filter at most one
        # group exists, so emit none (group masking then raises, per P1,
        # instead of silently masking the whole primary vertex)
        gids = None
        if self.vtx_filter is None:
            gids = [torch.from_numpy(np.where(v >= 0, v, -1)) for v in vtx_kept]

        # k-NN coordinates: (eta, sin phi, cos phi) - chord distance in the
        # sin/cos plane wraps phi correctly at +/-pi (raw phi would put
        # physically adjacent tracks ~2*pi apart)
        raw_coords = [torch.from_numpy(np.stack(
            [x[:, 1], np.sin(x[:, 2]), np.cos(x[:, 2])], axis=1)) for x in tracks]
        batch = pad_token_list(normed, "track", coords=raw_coords,
                               group_ids=gids)
        jets = {"queries": qs, "mask": jm, "labels3": y3,
                "labels_binary": torch.where(y3 < 0, y3, (y3 == 2).long())}
        return batch, jets
