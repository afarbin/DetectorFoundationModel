"""Masked-cell pretraining on the calo dataset (Tier 4 / D5: ttbar_1000).

Cluster ("group") masking on the S3 cell subset, reconstructing snr_scaled,
with the mask-indicator channel so the encoder transfers verbatim into the
jet-regression models. Writes: loss history JSON, best encoder checkpoint,
and validation reconstruction arrays for the QA plots.

    python -m dfm.jetreg.pretrain_calo --calo-dir .../processed_data/ttbar_1000 \
        --out /storage/afarbin/jetreg/pretrain --encoding graph
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from dfm.data import CaloGraphAdapter
from dfm.encoder import SharedEventEncoder, ModalityConfig
from dfm.heads import ReconstructionHead
from dfm.masking import MaskedTokenPretraining
from dfm.tokens import TokenBatch, pad_token_list

N_FEAT = 7


def preload(adapter, indices):
    """Per-event (features, edges, group_ids) tensors, loaded once."""
    out = []
    for i in indices:
        tb = adapter.event_tokens([i])
        n = int(tb.mask[0].sum())
        out.append((tb.features[0, :n].clone(), tb.edges[0],
                    tb.group_ids[0, :n].clone()))
    return out


def make_batch(events, idxs):
    feats = [events[i][0] for i in idxs]
    edges = [events[i][1] for i in idxs]
    gids = [events[i][2] for i in idxs]
    return pad_token_list(feats, "cell", edges=edges, group_ids=gids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calo-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoding", choices=["graph", "set"], default="graph")
    ap.add_argument("--strategy", default="group")
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--train-events", type=int, default=640)
    ap.add_argument("--val-events", type=int, default=160)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"pretrain_{args.encoding}_{args.strategy}_r{args.mask_ratio}"
    os.makedirs(args.out, exist_ok=True)

    adapter = CaloGraphAdapter(args.calo_dir)   # S3 subset + cell energy cut
    t0 = time.perf_counter()
    train_ev = preload(adapter, range(args.train_events))
    val_ev = preload(adapter, range(args.train_events,
                                    args.train_events + args.val_events))
    sizes = [len(e[0]) for e in train_ev]
    print(f"[{tag}] preloaded {len(train_ev)}+{len(val_ev)} events, "
          f"cells/event median {int(np.median(sizes))} "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    cfg = ModalityConfig("cell", N_FEAT,
                         local="edges" if args.encoding == "graph" else "none",
                         local_depth=2, mask_indicator=True)
    enc = SharedEventEncoder([cfg], dim=128).to(dev)
    recon = ReconstructionHead(128, N_FEAT).to(dev)
    masker = MaskedTokenPretraining(N_FEAT, args.mask_ratio,
                                    target_features=[0], seed=args.seed)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(recon.parameters())
                            + list(masker.parameters()), lr=args.lr)

    def run(events, train_mode, masker_):
        enc.train(train_mode); recon.train(train_mode)
        order = torch.randperm(len(events)).tolist() if train_mode \
            else list(range(len(events)))
        tot = n = 0.0
        with torch.set_grad_enabled(train_mode):
            for lo in range(0, len(order), args.batch):
                idxs = order[lo:lo + args.batch]
                clean = make_batch(events, idxs).to(dev)
                masked, mmap = masker_.mask(clean, strategy=args.strategy)
                pred = recon(enc([masked])["cell"])
                loss, _ = masker_.loss(pred, clean, mmap)
                if train_mode:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(enc.parameters()) + list(recon.parameters()), 1.0)
                    opt.step()
                tot += float(loss) * len(idxs); n += len(idxs)
        return tot / n

    history, best, bad = [], np.inf, 0
    ckpt = os.path.join(args.out, f"{tag}.pt")
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr = run(train_ev, True, masker)
        vl = run(val_ev, False, MaskedTokenPretraining(
            N_FEAT, args.mask_ratio, target_features=[0], seed=1234))
        history.append({"epoch": ep, "train": tr, "val": vl,
                        "seconds": time.perf_counter() - t0})
        star = vl < best - 1e-4
        if star:
            best, bad = vl, 0
            torch.save({"encoder": enc.state_dict(), "recon": recon.state_dict(),
                        "cfg": vars(args)}, ckpt)
        else:
            bad += 1
        print(f"[{tag}] ep {ep}: train {tr:.4f} val {vl:.4f}"
              f"{' *' if star else ''} [{history[-1]['seconds']:.0f}s]", flush=True)
        if bad >= args.patience:
            break
    with open(os.path.join(args.out, f"{tag}_history.json"), "w") as fh:
        json.dump(history, fh)

    # ---- validation reconstruction arrays for QA plots
    sd = torch.load(ckpt, weights_only=False)
    enc.load_state_dict(sd["encoder"]); recon.load_state_dict(sd["recon"])
    enc.eval(); recon.eval()
    rows = {"true": [], "pred": [], "strategy": [], "n_gids": []}
    with torch.no_grad():
        for strat in ("group", "random"):
            mk = MaskedTokenPretraining(N_FEAT, args.mask_ratio,
                                        target_features=[0], seed=77)
            for lo in range(0, len(val_ev), args.batch):
                idxs = list(range(lo, min(lo + args.batch, len(val_ev))))
                clean = make_batch(val_ev, idxs).to(dev)
                masked, mmap = mk.mask(clean, strategy=strat)
                pred = recon(enc([masked])["cell"])
                sel = mmap & clean.mask
                rows["true"].append(clean.features[..., 0][sel].cpu().numpy())
                rows["pred"].append(pred[..., 0][sel].cpu().numpy())
                rows["strategy"].append(np.full(int(sel.sum()),
                                                0 if strat == "group" else 1,
                                                dtype=np.int8))
    np.savez_compressed(
        os.path.join(args.out, f"{tag}_valrecon.npz"),
        true=np.concatenate(rows["true"]), pred=np.concatenate(rows["pred"]),
        strategy=np.concatenate(rows["strategy"]))
    print(f"[{tag}] done, best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
