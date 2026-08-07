import math
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

import numpy as np
import awkward as ak
import uproot


# -----------------------------
# Feature and branch selection
# -----------------------------
TRACK_FEATURES: List[str] = [
    "Track_pt",
    "Track_eta",
    "Track_phi",
    "Track_charge",
    "Track_d0",
    "Track_z0",
    "Track_qOverP",
    "Track_var_d0",
    "Track_var_z0",
    "Track_var_qOverP",
    "Track_numberOfSCTHoles",
    "Track_numberOfPixelHoles",
    "Track_numberOfPixelHits",
    "Track_numberOfSCTSharedHits",
    "Track_leptonID",
    "Track_btagIp_d0",
    "Track_btagIp_d0Uncertainty",
    "Track_btagIp_z0SinTheta",
    "Track_btagIp_z0SinThetaUncertainty",
    "Track_nearestVtx_idx",
    "Track_nearestVtx_sig",
    "Track_nearestVtx_d0",
    "Track_nearestVtx_d0Uncertainty",
    "Track_nearestVtx_z0SinTheta",
    "Track_nearestVtx_z0SinThetaUncertainty",
]

TRACK_RECO_VTX_IDX = "Track_recoVtx_idx"

# Reference jets (EMTopo seeds)
EMTOPO_PREFIX = "AntiKt4EMTopoJets_"
EMTOPO_FEATURES = [EMTOPO_PREFIX + x for x in ["eta", "phi", "pt", "m"]]
EMTOPO_LABEL = EMTOPO_PREFIX + "PartonTruthLabelID"
TRUTH_HS_ETAPHI = ["TruthHSJet_eta", "TruthHSJet_phi"]


# -----------------------------
# Utilities
# -----------------------------
def delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def pairwise_dr2(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    dphi = delta_phi(phi1[:, None], phi2[None, :])
    deta = eta1[:, None] - eta2[None, :]
    return deta * deta + dphi * dphi


def greedy_match(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray, dr_max: float) -> dict:
    if eta1.size == 0 or eta2.size == 0:
        return {}
    d2 = pairwise_dr2(eta1, phi1, eta2, phi2)
    used2 = np.zeros(eta2.shape[0], dtype=bool)
    matches = {}
    while True:
        d2_masked = d2.copy()
        if used2.any():
            d2_masked[:, used2] = np.inf
        used1 = np.array(list(matches.keys()), dtype=int) if matches else np.array([], dtype=int)
        if used1.size > 0:
            d2_masked[used1, :] = np.inf
        i, j = np.unravel_index(np.argmin(d2_masked), d2_masked.shape)
        if not np.isfinite(d2_masked[i, j]):
            break
        if d2_masked[i, j] < dr_max * dr_max:
            matches[i] = j
            used2[j] = True
        else:
            break
    return matches


def build_expressions() -> List[str]:
    return list(TRACK_FEATURES) + [TRACK_RECO_VTX_IDX] + TRUTH_HS_ETAPHI + EMTOPO_FEATURES + [EMTOPO_LABEL]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Build event-level b-tagging dataset with tracks, jets, and labels")
    p.add_argument("--input_dir", required=True, help="Directory containing input ROOT files")
    p.add_argument("--output", required=True, help="Output .npz path")
    p.add_argument("--tree", help="Tree path inside ROOT file (e.g., 'ntuple;1')", default="ntuple;1")
    p.add_argument("--dr_hs", type=float, default=0.2, help="ΔR threshold for EMTopo→TruthHS matching")
    p.add_argument("--ptmin_track", type=float, default=1.0, help="Minimum track pT to include")
    p.add_argument("--eta_max_track", type=float, default=2.0, help="Track |eta| selection")
    p.add_argument("--eta_max_jet", type=float, default=1.6, help="EMTopo jet |eta| selection")
    p.add_argument("--step_size", default="200 MB", help="uproot iterate step size")
    p.add_argument("--max_tracks", type=int, default=200, help="Maximum tracks per event")
    p.add_argument("--max_jets", type=int, default=10, help="Maximum jets per event")
    p.add_argument("--max_events", type=int, default=5000000, help="Maximum number of events to process")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    assert in_dir.exists() and in_dir.is_dir(), f"Input directory not found: {in_dir}"

    root_files = sorted([p for p in in_dir.rglob("*.root") if p.is_file()])
    assert len(root_files) > 0, f"No ROOT files found under {in_dir}"

    expressions = build_expressions()

    events_data = []

    total_events = 0
    events_processed = 0

    for fpath in tqdm(root_files, desc="Files"):
        if events_processed >= args.max_events:
            break
        treepath = f"{fpath}:{args.tree}"
        for arrays in tqdm(uproot.iterate([treepath], expressions=expressions, step_size=args.step_size, library="ak"), desc=f"Chunks ({fpath.name})", leave=False):
            if events_processed >= args.max_events:
                break
            n_evt = len(arrays[TRACK_FEATURES[0]])

            truth_eta_all = arrays[TRUTH_HS_ETAPHI[0]]
            truth_phi_all = arrays[TRUTH_HS_ETAPHI[1]]
            emt_eta_all = arrays[EMTOPO_FEATURES[0]]
            emt_phi_all = arrays[EMTOPO_FEATURES[1]]
            emt_pt_all = arrays[EMTOPO_FEATURES[2]]
            emt_m_all = arrays[EMTOPO_FEATURES[3]]
            emt_lab_all = arrays[EMTOPO_LABEL]

            event_pbar = tqdm(range(n_evt), desc="Events", leave=False)
            for ievt in event_pbar:
                if events_processed >= args.max_events:
                    break
                total_events += 1

                # Gather per-track features as [nTracks, D]
                cols = []
                for name in TRACK_FEATURES:
                    arr = arrays[name][ievt]
                    cols.append(ak.to_numpy(arr))
                X_full = np.vstack(cols).T.astype(np.float32) if len(cols) else np.zeros((0, 0), dtype=np.float32)
                if X_full.shape[0] == 0:
                    continue

                n_tracks_raw = len(X_full)
                tr_vidx = ak.to_numpy(arrays[TRACK_RECO_VTX_IDX][ievt]).astype(np.int32)
                pt = X_full[:, 0]
                eta = X_full[:, 1]
                phi = X_full[:, 2]

                # Track selection: pT > 1 GeV, |d_0| < 2 mm, and recoVtx_idx in {0, -1} (remove pileup)
                keep_tr = (pt > args.ptmin_track) & ((tr_vidx == 0) | (tr_vidx == -1))
                if keep_tr.sum() == 0:
                    continue
                X_evt = X_full[keep_tr]
                n_tracks_after_cuts = len(X_evt)
                eta_evt = eta[keep_tr]
                phi_evt = phi[keep_tr]

                # Sort tracks by pT (descending) and keep first max_tracks
                sort_idx = np.argsort(-X_evt[:, 0])
                if len(sort_idx) > args.max_tracks:
                    sort_idx = sort_idx[:args.max_tracks]
                X_evt = X_evt[sort_idx]
                n_tracks_selected = len(X_evt)
                eta_evt = eta_evt[sort_idx]
                phi_evt = phi_evt[sort_idx]

                # Select EMTopo jets within |eta|<1.6
                emt_eta_evt = ak.to_numpy(emt_eta_all[ievt]).astype(np.float32)
                emt_phi_evt = ak.to_numpy(emt_phi_all[ievt]).astype(np.float32)
                emt_pt_evt = ak.to_numpy(emt_pt_all[ievt]).astype(np.float32)
                emt_m_evt = ak.to_numpy(emt_m_all[ievt]).astype(np.float32)
                emt_lab_evt = ak.to_numpy(emt_lab_all[ievt]).astype(np.int32)

                mask_emt = np.abs(emt_eta_evt) < args.eta_max_jet
                emt_eta_sel = emt_eta_evt[mask_emt]
                emt_phi_sel = emt_phi_evt[mask_emt]
                emt_pt_sel = emt_pt_evt[mask_emt]
                emt_m_sel = emt_m_evt[mask_emt]
                emt_lab_sel = emt_lab_evt[mask_emt]

                if emt_eta_sel.size == 0:
                    continue

                # Match EMTopo (selected) -> TruthHS one-to-one match
                truth_eta_evt = ak.to_numpy(truth_eta_all[ievt]).astype(np.float32)
                truth_phi_evt = ak.to_numpy(truth_phi_all[ievt]).astype(np.float32)
                hs_match = greedy_match(emt_eta_sel, emt_phi_sel, truth_eta_evt, truth_phi_evt, dr_max=args.dr_hs)
                keep_indices = sorted(list(hs_match.keys()))
                if len(keep_indices) == 0:
                    continue

                # Collect matched jets (limit to max_jets)
                matched_jets = []
                n_b_jets = 0
                n_light_jets = 0

                for i_keep in keep_indices:
                    if i_keep < 0 or i_keep >= emt_lab_sel.shape[0]:
                        continue

                    lbl_val = int(emt_lab_sel[i_keep])
                    if lbl_val == -1:
                        continue

                    # Determine label: b-jet if |label| == 5, otherwise light jet
                    is_b = 1 if abs(lbl_val) == 5 else 0

                    matched_jets.append({
                        "eta": float(emt_eta_sel[i_keep]),
                        "phi": float(emt_phi_sel[i_keep]),
                        "pt": float(emt_pt_sel[i_keep]),
                        "m": float(emt_m_sel[i_keep]),
                        "label": is_b,
                    })

                    if is_b:
                        n_b_jets += 1
                    else:
                        n_light_jets += 1

                if len(matched_jets) == 0:
                    continue

                # Limit to max_jets
                matched_jets = matched_jets[:args.max_jets]

                # Calculate weight: ratio of b-jets to light jets
                # Avoid division by zero
                if n_light_jets > 0:
                    weight = n_b_jets / n_light_jets
                else:
                    weight = 1.0 if n_b_jets > 0 else 0.0

                # Store event data
                event_data = {
                    "n_tracks_raw": n_tracks_raw,
                    "n_tracks_after_cuts": n_tracks_after_cuts,
                    "n_tracks_selected": n_tracks_selected,
                    "n_jets": len(matched_jets),
                    "tracks": X_evt.astype(np.float32),  # [n_tracks, n_features]
                    "jets_pt": np.array([j["pt"] for j in matched_jets], dtype=np.float32),
                    "jets_eta": np.array([j["eta"] for j in matched_jets], dtype=np.float32),
                    "jets_phi": np.array([j["phi"] for j in matched_jets], dtype=np.float32),
                    "jets_m": np.array([j["m"] for j in matched_jets], dtype=np.float32),
                    "jets_label": np.array([j["label"] for j in matched_jets], dtype=np.int32),
                    "track_mask": np.ones(len(X_evt), dtype=np.bool_),
                    "jet_mask": np.ones(len(matched_jets), dtype=np.bool_),
                    "weight": float(weight),
                    "n_b_jets": n_b_jets,
                    "n_light_jets": n_light_jets,
                }

                events_data.append(event_data)
                events_processed += 1

                event_pbar.set_postfix(processed=events_processed, total=total_events, limit=args.max_events)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(events_data) > 0:
        np.savez(
            out_path,
            events=np.array(events_data, dtype=object),
            track_features=np.array(TRACK_FEATURES, dtype=object),
            n_events=len(events_data),
        )
        print(f"Saved {len(events_data)} events to {out_path}")
        print(f"Total events processed: {total_events}")
        print(f"Events with matched jets: {events_processed}")
    else:
        print("No events were saved!")


if __name__ == "__main__":
    main()
