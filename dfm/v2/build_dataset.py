"""Dataset v2 per-jet builder (Phase 1; spec = notes/cards/dataset-v2.md).

Differences from dfm/jetreg/build_dataset.py (v1):
  - schema_v2: RAW columns stored (feature engineering moves to load time);
    all verified-filled branches from notes/provenance/branch-inventory.md,
    known-empty branches excluded.
  - split id per jet: md5(str(eventNumber)) % 10 -> 0-6 train / 7 val /
    8-9 test. Event-keyed: later files (SLAC transfers) auto-assign.
  - taus kept with their extended label (excluded at training, not build).
  - tracks = ghost association only (btagTrack_idx verified EMPTY in the
    files, like GN2 — see branch-inventory.md).

Event-level shards (pileup truth jets, full TruthPart, TruthVtx) are a
separate follow-up builder — this file covers the per-jet program.

Run on the server:
    /test/afarbin/venvs/CaloGraphNet/bin/python build_dataset.py \
        --files /storage/mxg1065/ttbar_100GB/*.ntuple.root \
        --calo-ntuple /storage/mxg1065/input_data/ttbar_100events.root \
        --calo-processed /storage/mxg1065/processed_data/ttbar_1000 \
        --out /storage/afarbin/jetreg/data_v2 \
        --id-map /storage/afarbin/jetreg/data_v2/cell_id_map.npz
(run as a file, not -m; needs the repo root plus cell_matching.py on path)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from cell_matching import CaloCellGeometry, CellMatcher, CellIDMap  # noqa: E402

JC = "AntiKt4EMTopoJets"
SCHEMA_VERSION = "schema_v2.0"

# ---- raw column schema (order = storage order; names go in the manifest) ----
CELL_COLS = ["e", "eta", "phi", "significance", "time", "quality",
             "provenance", "badcell", "electronicNoise", "totalNoise",
             "sampling", "layer", "isEM_Barrel", "isEM_EndCap", "isTile",
             "isHEC", "isFCAL"]
CELL_BRANCHES = [f"Cell_{c}" for c in CELL_COLS] + \
    ["Cell_ID", "Cell_x", "Cell_y", "Cell_z"]

TRACK_COLS = ["pt", "eta", "phi", "charge", "d0", "z0", "qOverP",
              "var_d0", "var_z0", "var_qOverP",
              "nInmostPixelHits", "nPixelHits", "nStripHits",
              "nPixelHoles", "nStripHoles", "chi2", "ndof",
              "time", "timeRes", "hasValidTime",
              "recoVtx_idx", "recoVtx_weight", "truthProb"] + \
    [f"{l}_{q}" for l in ("PreSamplerB", "PreSamplerE", "EMB1", "EMB2",
                          "EMB3", "EME1", "EME2", "EME3")
     for q in ("eta", "phi")]
TRACK_BRANCHES = [f"Track_{c}" for c in TRACK_COLS]

JET_SCALARS = ["pt", "eta", "phi", "m", "nConstituents", "width", "isHS",
               "HadronConeExclTruthLabelID",
               "HadronConeExclExtendedTruthLabelID",
               "matchedTruth_pt", "matchedTruth_eta", "matchedTruth_phi",
               "matchedTruth_m", "matchedTruth_dR", "matchedTruth_GhostFrac"]
JET_BRANCHES = [f"{JC}_{s}" for s in JET_SCALARS] + [f"{JC}_ghostTrack_idx"]
EVENT_BRANCHES = ["averageInteractionsPerCrossing", "eventNumber"]


def split_of(evnum: int) -> int:
    """PROTOCOL S2: md5(str(eventNumber)) % 10; 0-6 train, 7 val, 8-9 test."""
    return int(hashlib.md5(str(int(evnum)).encode()).hexdigest(), 16) % 10


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
               ("split", "y", "mu", "event", "flavor", "label_ext",
                "jet_scalars", "tracks", "cells", "cell_edges",
                "n_unmapped", "iso_reco", "iso_truth",
                "pt_true", "pt_reco", "eta_reco", "phi_reco", "response")}
    cuts = dict(jets_total=0, matched=0, kept_flavor=0, pt_true=0, dr=0,
                eta=0, dup_truth=0, kept=0)
    t0 = time.perf_counter()

    for lo in range(0, n_events, args.chunk):
        hi = min(lo + args.chunk, n_events)
        arr = tree.arrays(CELL_BRANCHES + JET_BRANCHES + EVENT_BRANCHES
                          + TRACK_BRANCHES, entry_start=lo, entry_stop=hi)
        for i in range(hi - lo):
            e = arr[i]
            # ---- cells: raw column matrix + compact calo-graph indices
            cid = np.asarray(ak.to_numpy(e["Cell_ID"]), dtype=np.int64)
            hids = id_map.lookup(cid)
            unknown = hids < 0
            if unknown.any():
                res = matcher.match_xyz(
                    np.asarray(ak.to_numpy(e["Cell_x"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_y"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_z"]))[unknown],
                    np.asarray(ak.to_numpy(e["Cell_sampling"]))[unknown])
                full = np.full(len(cid), -1, dtype=np.int64)
                full[unknown] = np.where(res.matched, res.index, -1)
                id_map.update(cid[unknown], res)
                hids = np.where(unknown, full, hids)
            compact = np.where(hids >= 0,
                               inv_orig_to_compact[np.clip(hids, 0, None)], -1)
            cell_mat = np.stack(
                [np.asarray(ak.to_numpy(e[f"Cell_{c}"]), dtype=np.float32)
                 for c in CELL_COLS], axis=1)
            ceta, cphi = cell_mat[:, 1].astype(np.float64), \
                cell_mat[:, 2].astype(np.float64)

            keep_full = torch.zeros(n_compact, dtype=torch.bool)
            keep_full[torch.from_numpy(compact[compact >= 0])] = True
            ev_pairs, ev_newidx = subset_subgraph(pairs_compact, keep_full)
            loc = np.full(len(compact), -1, dtype=np.int64)
            m_ok = compact >= 0
            loc[m_ok] = ev_newidx.numpy()[compact[m_ok]]
            n_ev_local = int(keep_full.sum())

            # ---- tracks: raw column matrix (sentinels preserved)
            tx = np.stack(
                [np.asarray(ak.to_numpy(e[b]), dtype=np.float32)
                 for b in TRACK_BRANCHES], axis=1) \
                if len(e["Track_pt"]) else \
                np.zeros((0, len(TRACK_BRANCHES)), np.float32)

            # ---- jets
            j = {s: np.asarray(ak.to_numpy(e[f"{JC}_{s}"]))
                 for s in JET_SCALARS}
            jpt, jeta, jphi = (j["pt"].astype(np.float64),
                               j["eta"].astype(np.float64),
                               j["phi"].astype(np.float64))
            tpt, teta, tphi = (j["matchedTruth_pt"].astype(np.float64),
                               j["matchedTruth_eta"].astype(np.float64),
                               j["matchedTruth_phi"].astype(np.float64))
            lab = j["HadronConeExclTruthLabelID"]
            mu_val = float(e["averageInteractionsPerCrossing"])
            evnum = int(e["eventNumber"])
            spl = split_of(evnum)

            nj = len(jpt)
            cuts["jets_total"] += nj
            matched = tpt > 0
            cuts["matched"] += int(matched.sum())
            al = np.abs(lab)
            # taus (15) kept with their label; only unknown labels dropped
            sel = matched & np.isin(al, (0, 4, 5, 15))
            cuts["kept_flavor"] += int(sel.sum())
            sel &= tpt > args.pt_true_min
            cuts["pt_true"] += int(sel.sum())
            dr_t = np.hypot(jeta - teta, dphi(jphi, tphi))
            sel &= dr_t < args.dr_max
            cuts["dr"] += int(sel.sum())
            sel &= np.abs(jeta) < args.eta_max
            cuts["eta"] += int(sel.sum())
            if matched.sum() > 1:
                key = np.round(np.stack([tpt, teta, tphi]), 5).T
                _, first_idx, counts = np.unique(
                    key[matched], axis=0, return_index=True,
                    return_counts=True)
                dup_keys = key[matched][first_idx[counts > 1]]
                if len(dup_keys):
                    is_dup = (key[:, None, :] ==
                              dup_keys[None, :, :]).all(-1).any(-1)
                    sel &= ~(is_dup & matched)
            cuts["dup_truth"] += int(sel.sum())

            for jj in np.nonzero(sel)[0]:
                others = np.arange(nj) != jj
                iso_reco = not np.any(
                    (jpt[others] > args.iso_reco_pt) &
                    (np.hypot(jeta[others] - jeta[jj],
                              dphi(jphi[others], jphi[jj]))
                     < args.iso_reco_dr))
                om = others & matched
                iso_truth = not np.any(
                    np.hypot(teta[om] - teta[jj],
                             dphi(tphi[om], tphi[jj])) < args.iso_truth_dr)

                cone = (np.hypot(ceta - jeta[jj], dphi(cphi, jphi[jj]))
                        < args.cone_dr)
                cc = np.nonzero(cone)[0]
                mapped = cc[loc[cc] >= 0]
                unmapped = cc[loc[cc] < 0]
                order = mapped[np.argsort(loc[mapped])]
                keep = torch.zeros(n_ev_local, dtype=torch.bool)
                keep[loc[order]] = True
                sub_pairs, _ = subset_subgraph(ev_pairs, keep)
                rows = np.concatenate([order, unmapped])

                ghost = np.asarray(ak.to_list(e[f"{JC}_ghostTrack_idx"][jj]),
                                   dtype=np.int64)
                empty = np.zeros((0, tx.shape[1]), np.float32)

                samples["split"].append(np.uint8(spl))
                samples["y"].append(np.float32(np.log(tpt[jj] / jpt[jj])))
                samples["mu"].append(np.float32(mu_val))
                samples["event"].append(evnum)
                samples["flavor"].append(
                    2 if al[jj] == 5 else (1 if al[jj] == 4 else
                                           (3 if al[jj] == 15 else 0)))
                samples["label_ext"].append(
                    np.int16(j["HadronConeExclExtendedTruthLabelID"][jj]))
                samples["jet_scalars"].append(np.array(
                    [j[s][jj] for s in JET_SCALARS], dtype=np.float32))
                samples["tracks"].append(tx[ghost] if len(ghost) else empty)
                samples["cells"].append(cell_mat[rows])
                samples["cell_edges"].append(
                    sub_pairs.numpy().astype(np.int32))
                samples["n_unmapped"].append(len(unmapped))
                samples["iso_reco"].append(iso_reco)
                samples["iso_truth"].append(iso_truth)
                samples["pt_true"].append(np.float32(tpt[jj]))
                samples["pt_reco"].append(np.float32(jpt[jj]))
                samples["eta_reco"].append(np.float32(jeta[jj]))
                samples["phi_reco"].append(np.float32(jphi[jj]))
                samples["response"].append(np.float32(jpt[jj] / tpt[jj]))
                cuts["kept"] += 1
        print(f"  {os.path.basename(path)}: events {hi}/{n_events}, "
              f"jets kept {cuts['kept']}  [{time.perf_counter()-t0:.0f}s]",
              flush=True)

    obj = {k: np.array(v, dtype=object) if k in
           ("tracks", "cells", "cell_edges")
           else np.array(v) for k, v in samples.items()}
    np.savez_compressed(out_path, **obj)
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--calo-ntuple", required=True)
    ap.add_argument("--calo-processed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-map", default=None,
                    help="persistent CellIDMap npz (loaded if exists, saved)")
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
        "calo geometry not hash-ordered"
    matcher = CellMatcher(geo)
    id_map = CellIDMap()
    if args.id_map and os.path.exists(CellIDMap._norm_path(args.id_map)):
        id_map = CellIDMap.load(args.id_map)
        print(f"warm CellIDMap: {id_map.n} entries", flush=True)
    cells_npy = np.load(glob.glob(
        os.path.join(args.calo_processed, "cells_*.npy"))[0])
    pairs_npy = np.load(glob.glob(
        os.path.join(args.calo_processed, "pairs_*.npy"))[0])
    inv = np.full(geo.n, -1, dtype=np.int64)
    inv[cells_npy["orig_idx"]] = cells_npy["global_idx"]
    n_compact = len(cells_npy)
    pairs_t = torch.from_numpy(pairs_npy.astype(np.int64))

    manifest = {"schema": SCHEMA_VERSION, "args": vars(args),
                "split_rule": "md5(str(eventNumber))%10: 0-6 train, 7 val, "
                              "8-9 test",
                "cell_cols": CELL_COLS, "track_cols": TRACK_COLS,
                "jet_scalars": JET_SCALARS, "files": []}
    for f in args.files:
        stem = os.path.basename(f).replace(".ntuple.root", "")
        out_path = os.path.join(args.out, f"jets_{stem}.npz")
        print(f"processing {f} -> {out_path}", flush=True)
        cuts = build_file(f, matcher, id_map, inv, pairs_t, n_compact,
                          args, out_path)
        manifest["files"].append(
            {"input": f, "output": os.path.basename(out_path), "cuts": cuts})
        print(f"  cutflow: {cuts}", flush=True)
        if args.id_map:
            id_map.save(args.id_map)
    # unique name so parallel invocations over disjoint file sets don't
    # clobber each other; merge_manifests() combines them at gate G1
    stem0 = os.path.basename(args.files[0]).replace(".ntuple.root", "")
    with open(os.path.join(args.out, f"manifest_{stem0}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("done.")


def merge_manifests(out_dir):
    """Combine per-invocation manifests into manifest.json (call at G1)."""
    parts = sorted(glob.glob(os.path.join(out_dir, "manifest_*.json")))
    merged = None
    for p in parts:
        with open(p) as fh:
            m = json.load(fh)
        if merged is None:
            merged = {k: v for k, v in m.items() if k != "files"}
            merged["files"] = []
        merged["files"] += m["files"]
    if merged:
        merged["files"].sort(key=lambda f: f["input"])
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(merged, fh, indent=2)
    return merged


if __name__ == "__main__":
    main()
