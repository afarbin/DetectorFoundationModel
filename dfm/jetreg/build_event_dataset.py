"""Event-level dataset for T3 (MET) and T4 (jet finding).

One sample = one event: selected tracks (18 features, PV/unassociated,
pt > 0.5 GeV, pt-ordered cap 500), all matched cells (7 features + event
neighbor subgraph via cell matching), reco jets, truth HS jets, and MET:
truth MET from the TruthPart neutrino record, plus precomputed classical
baselines (negative vector sums of tracks / cells / reco jets).

    python -m dfm.jetreg.build_event_dataset --files ... \
        --calo-ntuple ... --calo-processed ... --out .../event_data
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
CELL_BR = ["Cell_ID", "Cell_x", "Cell_y", "Cell_z", "Cell_eta", "Cell_phi",
           "Cell_significance", "Cell_e", "Cell_sampling"]
JET_BR = [f"{JC}_{s}" for s in ("pt", "eta", "phi", "m",
                                "HadronConeExclTruthLabelID", "matchedTruth_pt")]
TRUTH_BR = ["TruthHSJet_pt", "TruthHSJet_eta", "TruthHSJet_phi", "TruthHSJet_m",
            "TruthPart_pt", "TruthPart_eta", "TruthPart_phi", "TruthPart_pdgId"]
EV_BR = ["averageInteractionsPerCrossing", "eventNumber"]


def build_file(path, matcher, id_map, inv, pairs_compact, n_compact, args, out):
    import awkward as ak
    import uproot
    import torch
    from dfm.tokens import subset_subgraph

    tree = uproot.open(path)["ntuple"]
    n_events = tree.num_entries if args.max_events is None \
        else min(tree.num_entries, args.max_events)
    S = {k: [] for k in ("tracks", "cells", "cell_edges", "jets", "jet_flavor",
                         "truth_jets", "met_true", "met_tracks", "met_cells",
                         "met_jets", "mu", "event", "n_nu")}
    t0 = time.perf_counter()
    for lo in range(0, n_events, args.chunk):
        hi = min(lo + args.chunk, n_events)
        arr = tree.arrays(CELL_BR + JET_BR + TRUTH_BR + EV_BR + TRACK_BRANCHES
                          + ["Track_recoVtx_idx"], entry_start=lo, entry_stop=hi)
        for i in range(hi - lo):
            e = arr[i]
            # ---- tracks (PV/unassociated, pt > 0.5, cap 500)
            tc = {b: np.asarray(ak.to_numpy(e[b]), dtype=np.float32)
                  for b in TRACK_BRANCHES}
            vtx = np.asarray(ak.to_numpy(e["Track_recoVtx_idx"]), dtype=np.int64)
            sel = (tc["Track_pt"] > 0.5) & np.isin(vtx, (0, -1))
            idxs = np.nonzero(sel)[0]
            order = idxs[np.argsort(-tc["Track_pt"][idxs], kind="stable")][:500]
            tx = np.stack([tc[b][order] for b in TRACK_BRANCHES], axis=1)
            d0s = tx[:, 4] / np.sqrt(np.maximum(tx[:, 7], 1e-12))
            z0s = tx[:, 5] / np.sqrt(np.maximum(tx[:, 8], 1e-12))
            tx = np.concatenate([tx, d0s[:, None], z0s[:, None]], axis=1)
            S["tracks"].append(tx.astype(np.float32))
            tpx = tx[:, 0] * np.cos(tx[:, 2])
            tpy = tx[:, 0] * np.sin(tx[:, 2])
            S["met_tracks"].append(np.array([-tpx.sum(), -tpy.sum()], np.float32))

            # ---- cells: match -> compact -> event subgraph; ET vector sum
            cid = np.asarray(ak.to_numpy(e["Cell_ID"]), dtype=np.int64)
            hids = id_map.lookup(cid)
            unk = hids < 0
            if unk.any():
                res = matcher.match_xyz(
                    np.asarray(ak.to_numpy(e["Cell_x"]))[unk],
                    np.asarray(ak.to_numpy(e["Cell_y"]))[unk],
                    np.asarray(ak.to_numpy(e["Cell_z"]))[unk],
                    np.asarray(ak.to_numpy(e["Cell_sampling"]))[unk])
                full = np.full(len(cid), -1, dtype=np.int64)
                full[unk] = np.where(res.matched, res.index, -1)
                id_map.update(cid[unk], res)
                hids = np.where(unk, full, hids)
            compact = np.where(hids >= 0, inv[np.clip(hids, 0, None)], -1)
            ceta = np.asarray(ak.to_numpy(e["Cell_eta"]), dtype=np.float64)
            cphi = np.asarray(ak.to_numpy(e["Cell_phi"]), dtype=np.float64)
            snr = np.asarray(ak.to_numpy(e["Cell_significance"]), dtype=np.float32)
            ce = np.asarray(ak.to_numpy(e["Cell_e"]), dtype=np.float64)
            ok = compact >= 0
            keep = torch.zeros(n_compact, dtype=torch.bool)
            keep[torch.from_numpy(compact[ok])] = True
            sub_pairs, newidx = subset_subgraph(pairs_compact, keep)
            loc = np.full(len(compact), -1, dtype=np.int64)
            loc[ok] = newidx.numpy()[compact[ok]]
            # mapped cells ordered by event-local index (aligns with subgraph
            # numbering), unmapped cells appended edgeless
            rows = np.concatenate([np.nonzero(loc >= 0)[0][np.argsort(loc[loc >= 0])],
                                   np.nonzero(loc < 0)[0]])
            s = snr[rows]
            x = np.stack([np.sign(s) * np.log1p(np.abs(s)),
                          (np.abs(s) > 4), (np.abs(s) > 2), (np.abs(s) > 0),
                          ceta[rows], np.sin(cphi[rows]), np.cos(cphi[rows])],
                         axis=1).astype(np.float32)
            S["cells"].append(x)
            S["cell_edges"].append(sub_pairs.numpy().astype(np.int32))
            cet = ce / np.cosh(ceta)
            S["met_cells"].append(np.array([-(cet * np.cos(cphi)).sum(),
                                            -(cet * np.sin(cphi)).sum()], np.float32))

            # ---- reco jets + jet-sum MET (matched jets pt>20)
            jpt = np.asarray(ak.to_numpy(e[f"{JC}_pt"]), dtype=np.float64)
            jeta = np.asarray(ak.to_numpy(e[f"{JC}_eta"]), dtype=np.float64)
            jphi = np.asarray(ak.to_numpy(e[f"{JC}_phi"]), dtype=np.float64)
            jm = np.asarray(ak.to_numpy(e[f"{JC}_m"]), dtype=np.float64)
            lab = np.asarray(ak.to_numpy(e[f"{JC}_HadronConeExclTruthLabelID"]))
            mtpt = np.asarray(ak.to_numpy(e[f"{JC}_matchedTruth_pt"]), dtype=np.float64)
            jk = np.stack([jpt, jeta, jphi, jm], axis=1).astype(np.float32)
            S["jets"].append(jk)
            S["jet_flavor"].append(np.select(
                [np.abs(lab) == 5, np.abs(lab) == 4, lab == 0], [2, 1, 0],
                default=-1).astype(np.int8))
            good = (mtpt > 0) & (jpt > 20)
            S["met_jets"].append(np.array(
                [-(jpt[good] * np.cos(jphi[good])).sum(),
                 -(jpt[good] * np.sin(jphi[good])).sum()], np.float32))

            # ---- truth jets and truth MET (neutrinos)
            thp = np.asarray(ak.to_numpy(e["TruthHSJet_pt"]), dtype=np.float32)
            the = np.asarray(ak.to_numpy(e["TruthHSJet_eta"]), dtype=np.float32)
            thf = np.asarray(ak.to_numpy(e["TruthHSJet_phi"]), dtype=np.float32)
            thm = np.asarray(ak.to_numpy(e["TruthHSJet_m"]), dtype=np.float32)
            S["truth_jets"].append(np.stack([thp, the, thf, thm], axis=1))
            pid = np.asarray(ak.to_numpy(e["TruthPart_pdgId"]))
            ppt = np.asarray(ak.to_numpy(e["TruthPart_pt"]), dtype=np.float64)
            pphi = np.asarray(ak.to_numpy(e["TruthPart_phi"]), dtype=np.float64)
            nu = np.isin(np.abs(pid), (12, 14, 16))
            S["met_true"].append(np.array([(ppt[nu] * np.cos(pphi[nu])).sum(),
                                           (ppt[nu] * np.sin(pphi[nu])).sum()],
                                          np.float32))
            S["n_nu"].append(int(nu.sum()))
            S["mu"].append(np.float32(e["averageInteractionsPerCrossing"]))
            S["event"].append(int(e["eventNumber"]))
        print(f"  {os.path.basename(path)}: {hi}/{n_events} "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    def obj_array(v):   # 1-D object array even when shapes are uniform
        a = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            a[i] = x
        return a
    obj = {k: (obj_array(v)
               if k in ("tracks", "cells", "cell_edges", "jets", "jet_flavor",
                        "truth_jets") else np.array(v))
           for k, v in S.items()}
    np.savez_compressed(out, **obj)
    return n_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--calo-ntuple", required=True)
    ap.add_argument("--calo-processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    import uproot
    import torch

    os.makedirs(args.out, exist_ok=True)
    geo = CaloCellGeometry.from_tree(uproot.open(args.calo_ntuple)["analysis"])
    assert np.array_equal(geo.hash_id, np.arange(geo.n))
    matcher = CellMatcher(geo)
    id_map = CellIDMap()
    cells_npy = np.load(glob.glob(os.path.join(args.calo_processed, "cells_*.npy"))[0])
    pairs_npy = np.load(glob.glob(os.path.join(args.calo_processed, "pairs_*.npy"))[0])
    inv = np.full(geo.n, -1, dtype=np.int64)
    inv[cells_npy["orig_idx"]] = cells_npy["global_idx"]
    pairs_t = torch.from_numpy(pairs_npy.astype(np.int64))

    manifest = {"files": []}
    for f in args.files:
        stem = os.path.basename(f).replace(".ntuple.root", "")
        out_path = os.path.join(args.out, f"events_{stem}.npz")
        print(f"processing {f}", flush=True)
        n = build_file(f, matcher, id_map, inv, pairs_t, len(cells_npy), args,
                       out_path)
        manifest["files"].append({"input": f, "output": os.path.basename(out_path),
                                  "n_events": n})
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
