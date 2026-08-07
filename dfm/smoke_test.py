"""End-to-end smoke test of the merged backbone on real data.

Exercises, with backward passes and optimizer steps:
  1. masked pretraining on calo cells (group/cluster masking, S3 subset)
  2. edge classification on calo cells (5-class, existing label scheme)
  3. b-tagging on tracks (jet-query decoder, 3-class b/c/light)
  4. cross-modality forward: jet queries attending to cells + tracks jointly

NOTE: the cell events (Calo ntuple ttbar) and track events (5D ntuple ttbar)
are different event samples - fine for a mechanics test; genuine cross-
modality *training* needs both modalities dumped for the same events (see
README, "Data prerequisites").

Run on the UTA server:
    python -m dfm.smoke_test \
        --calo-dir /storage/mxg1065/processed_data/ttbar_1000 \
        --fived-file /storage/mxg1065/ttbar_100GB/user.bbullard.50733453._000011.ntuple.root
"""

from __future__ import annotations

import argparse
import time

import torch

from dfm.data import CaloGraphAdapter, TrackJetAdapter, CALO_FEATURES, TRACK_FEATURES
from dfm.encoder import SharedEventEncoder, ModalityConfig
from dfm.heads import (EdgeClassifierHead, JetQueryDecoder, ReconstructionHead,
                       masked_ce)
from dfm.masking import MaskedTokenPretraining


def check(name, loss):
    assert torch.isfinite(loss), f"{name}: non-finite loss {loss}"
    print(f"  {name}: loss = {float(loss):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calo-dir", required=True)
    ap.add_argument("--fived-file", required=True)
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--snr-min", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    t0 = time.perf_counter()
    calo = CaloGraphAdapter(args.calo_dir, snr_min=args.snr_min)
    cells = calo.event_tokens(range(args.events))
    print(f"cells: {cells.features.shape} valid/event "
          f"{cells.mask.sum(1).tolist()} edges/event "
          f"{[len(e) for e in cells.edges]}  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    tracks_ad = TrackJetAdapter(args.fived_file)
    tracks_ad.compute_normalization(range(args.events))  # "train" stats, explicit (M6)
    tracks, jets = tracks_ad.event_tokens(range(args.events))
    n_jets = int(jets["mask"].sum())
    print(f"tracks: {tracks.features.shape} valid/event {tracks.mask.sum(1).tolist()}; "
          f"jets: {n_jets} total, labels3 counts "
          f"{[int((jets['labels3'] == c).sum()) for c in (0, 1, 2)]} (l/c/b)"
          f"  [{time.perf_counter()-t0:.1f}s]")

    n_cell_feat = len(CALO_FEATURES)
    n_track_feat = len(TRACK_FEATURES)

    cell_cfg = ModalityConfig("cell", n_cell_feat, local="edges", local_depth=2,
                              mask_indicator=True)
    track_cfg = ModalityConfig("track", n_track_feat, local="knn", local_depth=2)

    # --- 1. masked pretraining on cells (group masking; indicator channel)
    print("\n[1] masked cluster pretraining on cells")
    masker = MaskedTokenPretraining(num_features=n_cell_feat,
                                    target_features=[0],  # dynamic: snr_scaled (P6)
                                    seed=17)
    enc_pre = SharedEventEncoder([cell_cfg], dim=args.dim).to(device)
    recon = ReconstructionHead(args.dim, n_cell_feat).to(device)
    opt = torch.optim.AdamW(list(enc_pre.parameters()) + list(recon.parameters())
                            + list(masker.parameters()), lr=1e-3)
    masked_cells, mask_map = masker.mask(cells.to(device), strategy="group")
    out = enc_pre([masked_cells])
    pred = recon(out["cell"])
    loss, breakdown = masker.loss(pred, cells.to(device), mask_map)
    print(f"  masked {int(mask_map.sum())} cells across "
          f"{cells.batch_size} events; per-feature {breakdown}")
    opt.zero_grad(); loss.backward(); opt.step()
    check("pretrain-group-mask", loss)

    # P1 guard: vertex masking on default-selection tracks must raise
    try:
        masker_t = MaskedTokenPretraining(num_features=n_track_feat, seed=3)
        masker_t.mask(tracks, strategy="group")
        raise AssertionError("group masking without group ids did NOT raise")
    except ValueError as e:
        print(f"  P1 guard OK: group masking on filtered tracks raises "
              f"({str(e)[:60]}...)")

    # --- 2. edge classification on cells (5-class, concat readout for compat)
    print("\n[2] edge classification on cells (+ pretrain weight transfer)")
    enc = SharedEventEncoder([cell_cfg, track_cfg], dim=args.dim).to(device)
    # the mask_indicator design makes pretrain/finetune encoders byte-
    # compatible: port the pretrained cell projection + global stack verbatim
    transferred = enc.load_state_dict(enc_pre.state_dict(), strict=False)
    n_ported = len(enc_pre.state_dict())
    print(f"  transferred {n_ported - len(transferred.unexpected_keys)}"
          f"/{len(enc.state_dict())} tensors from the pretrained encoder "
          f"(missing in source: {len(transferred.missing_keys)} track-side)")
    edge_head = EdgeClassifierHead(args.dim, num_classes=5, readout="concat").to(device)
    btag_head = JetQueryDecoder(args.dim, mode="multiclass", num_classes=3).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(edge_head.parameters())
                            + list(btag_head.parameters()), lr=1e-3)

    out = enc([cells.to(device)])
    logits = edge_head(out["cell"], cells.mask.to(device),
                       [e.to(device) for e in cells.edges])
    y5 = torch.cat(cells.extras["edge_labels_5class"]).to(device)
    loss = torch.nn.functional.cross_entropy(logits, y5)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  {logits.shape[0]} edges, class counts "
          f"{[int((y5 == c).sum()) for c in range(5)]}")
    check("edge-classify", loss)

    # --- 3. b-tagging on tracks alone
    print("\n[3] b-tagging (tracks only, 3-class b/c/light)")
    out = enc([tracks.to(device)])
    jl, attn = btag_head(jets["queries"].to(device), jets["mask"].to(device),
                         out["track"], tracks.mask.to(device))
    loss = masked_ce(jl, jets["labels3"].to(device), jets["mask"].to(device))
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  logits {tuple(jl.shape)}, attention {tuple(attn.shape)}")
    check("btag-tracks", loss)

    # --- 4. cross-modality: jets attend to cells AND tracks
    print("\n[4] cross-modality (cells + tracks -> jet queries)")
    out = enc([cells.to(device), tracks.to(device)])
    comb = out["combined"]
    jl, attn = btag_head(jets["queries"].to(device), jets["mask"].to(device),
                         comb.features, comb.mask)
    loss = masked_ce(jl, jets["labels3"].to(device), jets["mask"].to(device))
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  combined tokens {tuple(comb.features.shape)} "
          f"(cell slice {comb.slices['cell']}, track slice {comb.slices['track']})")
    check("btag-cross-modality", loss)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
