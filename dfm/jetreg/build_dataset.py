"""Build the per-jet pT-response regression dataset from 5D ntuples.

One sample = one matched light jet passing the quality cuts of
JET_ENERGY_STUDY.md section 2.3 (isolation stored as flags, not cut - D1).

Per jet: target y = ln(pt_true/pt_reco), jet features, ghost-associated
tracks (18 features), cone cells (baseline-7 features) with their
calo-geometry neighbor subgraph (via cell matching), isolation flags, meta.

Run on the server:
    python -m dfm.jetreg.build_dataset \
        --files /storage/mxg1065/ttbar_100GB/user.bbullard.50733453._0000{11,12,13,14}.ntuple.root \
        --calo-ntuple /storage/mxg1065/input_data/ttbar_100events.root \
        --calo-processed /storage/mxg1065/processed_data/ttbar_1000 \
        --out /storage/mxg1065/jetreg/data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from cell_matching import CaloCellGeometry, CellMatcher, CellIDMap
from dfm.data import TRACK_BRANCHES

JC = "AntiKt4EMTopoJets"

CELL_BRANCHES = ["Cell_ID", "Cell_x", "Cell_y", "Cell_z", "Cell_eta", "Cell_phi",
                 "Cell_significance", "Cell_sampling"]
JET_BRANCHES = [f"{JC}_{s}" for s in
                ("pt", "eta", "phi", "m", "nConstituents", "response",
                 "HadronConeExclTruthLabelID", "ghostTrack_idx",
                 "matchedTruth_pt", "matchedTruth_eta", "matchedTruth_phi",
                 "matchedTruth_m")]
EVENT_BRANCHES = ["averageInteractionsPerCrossing", "eventNumber"]


def dphi(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def build_file(path, matcher, id_map, inv_orig_to_compact, pairs_compact,
               n_compact, args, out_path):
    import awkward as ak
    import uproot
    import torch
    from dfm.tokens import subset_subgraph

    tree = uproot.open(path)["ntuple"]
    n_events = tree.num_entries if args.max_events is None \
        else min(tree.num_entries, args.max_events)
    samples = {k: [] for k in
               ("y", "jet", "mu", "tracks", "cells", "cell_edges", "n_unmapped",
                "iso_reco", "iso_truth", "pt_true", "pt_reco", "eta_reco", "phi_reco",
                "response", "event", "n_cells", "n_edges", "n_iso_cells",
                "n_tracks")}
    cuts = dict(jets_total=0, matched=0, light=0, pt_true=0, dr=0, eta=0,
                dup_truth=0, kept=0)
    t0 = time.perf_counter()

    for lo in range(0, n_events, args.chunk):
        hi = min(lo + args.chunk, n_events)
        arr = tree.arrays(CELL_BRANCHES + JET_BRANCHES + EVENT_BRANCHES
                          + TRACK_BRANCHES, entry_start=lo, entry_stop=hi)
        for i in range(hi - lo):
            e = arr[i]
            # ---- cells: match once per event -> compact calo indices
            cid = np.asarray(ak.to_numpy(e["Cell_ID"]), dtype=np.int64)
            hids = id_map.lookup(cid)
            unknown = hids < 0
            if unknown.any():
                res = matcher.match_xyz(
                    np.asarray(ak.to_numpy(e["Cell_x"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_y"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_z"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_sampling"]))[unknown])
                # geometry is hash-ordered: MatchResult.index == branch position
                full = np.full(len(cid), -1, dtype=np.int64)
                full[unknown] = np.where(res.matched, res.index, -1)
                id_map.update(cid[unknown], res)
                hids = np.where(unknown, full, hids)
            compact = np.where(hids >= 0, inv_orig_to_compact[np.clip(hids, 0, None)], -1)
            ceta = np.asarray(ak.to_numpy(e["Cell_eta"]), dtype=np.float64)
            cphi = np.asarray(ak.to_numpy(e["Cell_phi"]), dtype=np.float64)
            snr = np.asarray(ak.to_numpy(e["Cell_significance"]), dtype=np.float32)

            # restrict the 1.25M-pair calo graph to THIS event's cells once;
            # per-jet subgraphs then operate on a few hundred pairs
            keep_full = torch.zeros(n_compact, dtype=torch.bool)
            keep_full[torch.from_numpy(compact[compact >= 0])] = True
            ev_pairs, ev_newidx = subset_subgraph(pairs_compact, keep_full)
            loc = np.full(len(compact), -1, dtype=np.int64)
            m_ok = compact >= 0
            loc[m_ok] = ev_newidx.numpy()[compact[m_ok]]
            n_ev_local = int(keep_full.sum())

            # ---- tracks (event-level matrices; per-jet gather below)
            tcols = {b: np.asarray(ak.to_numpy(e[b]), dtype=np.float32)
                     for b in TRACK_BRANCHES}
            tx = np.stack([tcols[b] for b in TRACK_BRANCHES], axis=1) \
                if len(tcols["Track_pt"]) else np.zeros((0, len(TRACK_BRANCHES)), np.float32)
            d0s = tx[:, 4] / np.sqrt(np.maximum(tx[:, 7], 1e-12))
            z0s = tx[:, 5] / np.sqrt(np.maximum(tx[:, 8], 1e-12))
            tx = np.concatenate([tx, d0s[:, None], z0s[:, None]], axis=1)

            # ---- jets
            jpt = np.asarray(ak.to_numpy(e[f"{JC}_pt"]), dtype=np.float64)
            jeta = np.asarray(ak.to_numpy(e[f"{JC}_eta"]), dtype=np.float64)
            jphi = np.asarray(ak.to_numpy(e[f"{JC}_phi"]), dtype=np.float64)
            jm = np.asarray(ak.to_numpy(e[f"{JC}_m"]), dtype=np.float64)
            jnc = np.asarray(ak.to_numpy(e[f"{JC}_nConstituents"]), dtype=np.float64)
            lab = np.asarray(ak.to_numpy(e[f"{JC}_HadronConeExclTruthLabelID"]))
            tpt = np.asarray(ak.to_numpy(e[f"{JC}_matchedTruth_pt"]), dtype=np.float64)
            teta = np.asarray(ak.to_numpy(e[f"{JC}_matchedTruth_eta"]), dtype=np.float64)
            tphi = np.asarray(ak.to_numpy(e[f"{JC}_matchedTruth_phi"]), dtype=np.float64)
            mu_val = float(e["averageInteractionsPerCrossing"])
            evnum = int(e["eventNumber"])

            nj = len(jpt)
            cuts["jets_total"] += nj
            matched = tpt > 0
            cuts["matched"] += int(matched.sum())
            sel = matched & (lab == 0)
            cuts["light"] += int(sel.sum())
            sel &= tpt > args.pt_true_min
            cuts["pt_true"] += int(sel.sum())
            dr_t = np.hypot(jeta - teta, dphi(jphi, tphi))
            sel &= dr_t < args.dr_max
            cuts["dr"] += int(sel.sum())
            sel &= np.abs(jeta) < args.eta_max
            cuts["eta"] += int(sel.sum())
            # duplicate-truth guard among matched jets
            if matched.sum() > 1:
                key = np.round(np.stack([tpt, teta, tphi]), 5).T
                _, first_idx, counts = np.unique(key[matched], axis=0,
                                                 return_index=True, return_counts=True)
                dup_keys = key[matched][first_idx[counts > 1]]
                if len(dup_keys):
                    is_dup = (key[:, None, :] == dup_keys[None, :, :]).all(-1).any(-1)
                    sel &= ~(is_dup & matched)
            cuts["dup_truth"] += int(sel.sum())

            for j in np.nonzero(sel)[0]:
                # isolation flags (D1)
                others = np.arange(nj) != j
                iso_reco = not np.any((jpt[others] > args.iso_reco_pt) &
                                      (np.hypot(jeta[others] - jeta[j],
                                                dphi(jphi[others], jphi[j])) < args.iso_reco_dr))
                om = others & matched
                iso_truth = not np.any(np.hypot(teta[om] - teta[j],
                                                dphi(tphi[om], tphi[j])) < args.iso_truth_dr)

                # cone cells, ordered by event-local (== compact) index so
                # tokens align with subset_subgraph numbering
                cone = (np.hypot(ceta - jeta[j], dphi(cphi, jphi[j])) < args.cone_dr)
                cc = np.nonzero(cone)[0]
                mapped = cc[loc[cc] >= 0]
                unmapped = cc[loc[cc] < 0]
                order = mapped[np.argsort(loc[mapped])]
                keep = torch.zeros(n_ev_local, dtype=torch.bool)
                keep[loc[order]] = True
                sub_pairs, _ = subset_subgraph(ev_pairs, keep)
                rows = np.concatenate([order, unmapped])
                s = snr[rows]
                feats = np.stack([np.sign(s) * np.log1p(np.abs(s)),
                                  (np.abs(s) > 4), (np.abs(s) > 2), (np.abs(s) > 0),
                                  ceta[rows], np.sin(cphi[rows]), np.cos(cphi[rows])],
                                 axis=1).astype(np.float32)

                ghost = np.asarray(ak.to_list(e[f"{JC}_ghostTrack_idx"][j]), dtype=np.int64)
                jtr = tx[ghost] if len(ghost) else np.zeros((0, tx.shape[1]), np.float32)

                deg = np.bincount(sub_pairs.reshape(-1).numpy(),
                                  minlength=len(rows)) if len(sub_pairs) else \
                    np.zeros(len(rows))
                samples["y"].append(np.float32(np.log(tpt[j] / jpt[j])))
                samples["jet"].append(np.array(
                    [jeta[j], np.sin(jphi[j]), np.cos(jphi[j]),
                     np.log(max(jpt[j], 1e-6)), np.log(max(jm[j], 1e-6)),
                     np.log1p(jnc[j])], dtype=np.float32))
                samples["mu"].append(np.float32(mu_val))
                samples["tracks"].append(jtr.astype(np.float32))
                samples["cells"].append(feats)
                samples["cell_edges"].append(sub_pairs.numpy().astype(np.int32))
                samples["n_unmapped"].append(len(unmapped))
                samples["iso_reco"].append(iso_reco)
                samples["iso_truth"].append(iso_truth)
                samples["pt_true"].append(np.float32(tpt[j]))
                samples["pt_reco"].append(np.float32(jpt[j]))
                samples["eta_reco"].append(np.float32(jeta[j]))
                samples["phi_reco"].append(np.float32(jphi[j]))
                samples["response"].append(np.float32(jpt[j] / tpt[j]))
                samples["event"].append(evnum)
                samples["n_cells"].append(len(rows))
                samples["n_edges"].append(len(sub_pairs))
                samples["n_iso_cells"].append(int((deg == 0).sum()))
                samples["n_tracks"].append(len(ghost))
                cuts["kept"] += 1
        print(f"  {os.path.basename(path)}: events {hi}/{n_events}, "
              f"jets kept {cuts['kept']}  [{time.perf_counter()-t0:.0f}s]",
              flush=True)

    obj = {k: np.array(v, dtype=object) if k in
           ("tracks", "cells", "cell_edges") else np.array(v)
           for k, v in samples.items()}
    np.savez_compressed(out_path, **obj)
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--calo-ntuple", required=True)
    ap.add_argument("--calo-processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pt-true-min", type=float, default=20.0)
    ap.add_argument("--dr-max", type=float, default=0.3)
    ap.add_argument("--eta-max", type=float, default=2.5)
    ap.add_argument("--iso-reco-pt", type=float, default=15.0)
    ap.add_argument("--iso-reco-dr", type=float, default=0.8)
    ap.add_argument("--iso-truth-dr", type=float, default=1.0)
    ap.add_argument("--cone-dr", type=float, default=0.4)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    import uproot
    import torch

    os.makedirs(args.out, exist_ok=True)
    print("setting up calo geometry + matcher ...", flush=True)
    geo = CaloCellGeometry.from_tree(uproot.open(args.calo_ntuple)["analysis"])
    assert np.array_equal(geo.hash_id, np.arange(geo.n)), \
        "calo geometry not hash-ordered; MatchResult.index != branch position"
    matcher = CellMatcher(geo)
    id_map = CellIDMap()
    cells_npy = np.load(glob.glob(os.path.join(args.calo_processed, "cells_*.npy"))[0])
    pairs_npy = np.load(glob.glob(os.path.join(args.calo_processed, "pairs_*.npy"))[0])
    inv = np.full(geo.n, -1, dtype=np.int64)
    inv[cells_npy["orig_idx"]] = cells_npy["global_idx"]
    n_compact = len(cells_npy)
    pairs_t = torch.from_numpy(pairs_npy.astype(np.int64))

    manifest = {"args": vars(args), "files": []}
    for f in args.files:
        stem = os.path.basename(f).replace(".ntuple.root", "")
        out_path = os.path.join(args.out, f"jets_{stem}.npz")
        print(f"processing {f} -> {out_path}", flush=True)
        cuts = build_file(f, matcher, id_map, inv, pairs_t, n_compact, args, out_path)
        manifest["files"].append({"input": f, "output": os.path.basename(out_path),
                                  "cuts": cuts})
        print(f"  cutflow: {cuts}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
