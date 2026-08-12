"""Train event-level tasks: --task met (T3) or --task jetfind (T4).

    python -m dfm.jetreg.train_event --task met --inputs T \
        --data-dir .../event_data --out .../results_t3 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dfm.tokens import TokenBatch, pad_token_list
from dfm.jetreg.event_models import METModel, JetFinder, jetfinder_loss, \
    hungarian_targets

MET_SCALE = 100.0  # GeV


class EventShards(Dataset):
    def __init__(self, paths):
        self.data = []
        keys = None
        for p in paths:
            z = np.load(p, allow_pickle=True)
            keys = [k for k in z.files]
            self.data.append({k: z[k] for k in keys})
        self.arrays = {k: np.concatenate([d[k] for d in self.data])
                       for k in keys}
        self.n = len(self.arrays["met_true"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.arrays[k][i] for k in self.arrays}


def make_collate(use_cells, tmean, tstd):
    def collate(items):
        out = {}
        feats, coords = [], []
        for it in items:
            x = it["tracks"].astype(np.float32).copy()
            raw = x[:, 1:3].copy()
            x[:, 0] = np.log(np.maximum(x[:, 0], 1e-6))
            x = (x - tmean) / tstd
            feats.append(torch.from_numpy(x))
            coords.append(torch.from_numpy(np.stack(
                [raw[:, 0], np.sin(raw[:, 1]), np.cos(raw[:, 1])], axis=1)))
        out["tracks"] = pad_token_list(feats, "track", coords=coords)
        if use_cells:
            cf, ce = [], []
            for it in items:
                cf.append(torch.from_numpy(it["cells"].astype(np.float32)))
                ce.append(torch.from_numpy(
                    it["cell_edges"].astype(np.int64)).reshape(-1, 2))
            out["cells"] = pad_token_list(cf, "cell", edges=ce)
        out["met_true"] = torch.tensor(
            np.stack([it["met_true"] for it in items]), dtype=torch.float32)
        for k in ("met_tracks", "met_cells", "met_jets"):
            out[k] = torch.tensor(np.stack([it[k] for it in items]),
                                  dtype=torch.float32)
        # truth jets padded (pt, eta, phi), pt > 20, cap 10
        tj = [it["truth_jets"] for it in items]
        mmax = max(1, max(min((t[:, 0] > 20).sum(), 10) for t in tj))
        B = len(items)
        tpt = torch.zeros(B, mmax); teta = torch.zeros(B, mmax)
        tphi = torch.zeros(B, mmax); tmask = torch.zeros(B, mmax, dtype=torch.bool)
        for b, t in enumerate(tj):
            sel = t[t[:, 0] > 20][:10]
            n = len(sel)
            if n:
                tpt[b, :n] = torch.from_numpy(sel[:, 0])
                teta[b, :n] = torch.from_numpy(sel[:, 1])
                tphi[b, :n] = torch.from_numpy(sel[:, 2])
                tmask[b, :n] = True
        out.update(tj_pt=tpt, tj_eta=teta, tj_phi=tphi, tj_mask=tmask)
        return out
    return collate


def to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, (torch.Tensor, TokenBatch)) else v)
            for k, v in batch.items()}


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["met", "jetfind"], required=True)
    ap.add_argument("--inputs", choices=["T", "TC"], default="T")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cells = args.inputs == "TC"
    tag = f"{args.task}_{args.inputs}_seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.data_dir, "manifest.json")) as fh:
        man = json.load(fh)
    shards = [os.path.join(args.data_dir, f["output"]) for f in man["files"]]
    train = EventShards(shards[:-2])
    val = EventShards(shards[-2:-1])
    test = EventShards(shards[-1:])
    print(f"[{tag}] events: train {len(train)} val {len(val)} test {len(test)}",
          flush=True)

    ttr = np.concatenate([t for t in train.arrays["tracks"][:20000]])
    ttr = ttr.astype(np.float64)
    ttr[:, 0] = np.log(np.maximum(ttr[:, 0], 1e-6))
    tmean = ttr.mean(0, keepdims=True).astype(np.float32)
    tstd = np.maximum(ttr.std(0, keepdims=True), 1e-6).astype(np.float32)

    collate = make_collate(use_cells, tmean, tstd)
    dl_tr = DataLoader(train, batch_size=args.batch, shuffle=True,
                       collate_fn=collate, num_workers=4, persistent_workers=True)
    dl_va = DataLoader(val, batch_size=args.batch, collate_fn=collate)
    dl_te = DataLoader(test, batch_size=args.batch, collate_fn=collate)

    model = (METModel(use_cells) if args.task == "met"
             else JetFinder(use_cells)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

    def loss_of(batch):
        if args.task == "met":
            pred = model(batch)
            return F.huber_loss(pred, batch["met_true"] / MET_SCALE, delta=0.5)
        pred = model(batch)
        return jetfinder_loss(pred, batch["tj_pt"], batch["tj_eta"],
                              batch["tj_phi"], batch["tj_mask"])

    def run(dl, train_mode):
        model.train(train_mode)
        tot = n = 0.0
        with torch.set_grad_enabled(train_mode):
            for batch in dl:
                batch = to_device(batch, dev)
                loss = loss_of(batch)
                if train_mode:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += float(loss) * len(batch["met_true"])
                n += len(batch["met_true"])
        return tot / n

    best, bad = np.inf, 0
    ckpt = os.path.join(args.out, f"{tag}.pt")
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        tr = run(dl_tr, True)
        vl = run(dl_va, False)
        sched.step()
        star = vl < best - 1e-5
        if star:
            best, bad = vl, 0
            torch.save({"model": model.state_dict()}, ckpt)
        else:
            bad += 1
        print(f"[{tag}] ep {ep}: train {tr:.4f} val {vl:.4f}"
              f"{' *' if star else ''}", flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    model.eval()
    metrics = {"config": tag, "best_val": float(best), "seed": args.seed,
               "minutes": (time.perf_counter() - t0) / 60}

    if args.task == "met":
        preds, rows = [], {k: [] for k in ("met_true", "met_tracks",
                                           "met_cells", "met_jets")}
        with torch.no_grad():
            for batch in dl_te:
                b = to_device(batch, dev)
                preds.append((model(b) * MET_SCALE).cpu().numpy())
                for k in rows:
                    rows[k].append(batch[k].numpy())
        pred = np.concatenate(preds)
        arr = {k: np.concatenate(v) for k, v in rows.items()}
        np.savez_compressed(os.path.join(args.out, f"{tag}_pred.npz"),
                            pred=pred, **arr)
        def resol(est):
            d = est - arr["met_true"]
            return float(np.mean(np.percentile(np.abs(d), 68, axis=0)))
        metrics.update(
            resol_model=resol(pred), resol_tracks=resol(arr["met_tracks"]),
            resol_cells=resol(arr["met_cells"]), resol_jets=resol(arr["met_jets"]))
    else:
        effs, fakes, nres = [], [], []
        with torch.no_grad():
            for batch in dl_te:
                b = to_device(batch, dev)
                pred = model(b)
                matches = hungarian_targets(pred, b["tj_pt"], b["tj_eta"],
                                            b["tj_phi"], b["tj_mask"])
                pe = torch.sigmoid(pred[..., 0]).cpu()
                for ev, (qi, ti) in enumerate(matches):
                    n_t = int(batch["tj_mask"][ev].sum())
                    found = (pe[ev][qi] > 0.5).sum().item() if len(qi) else 0
                    effs.append(found / max(n_t, 1))
                    n_pred = (pe[ev] > 0.5).sum().item()
                    fakes.append(max(n_pred - found, 0) / max(n_pred, 1))
                    if len(qi):
                        t_lpt = torch.log(batch["tj_pt"][ev][ti].clamp(min=1e-3))
                        nres.append((pred[ev, qi, 1].cpu() - t_lpt).abs().mean().item())
        metrics.update(efficiency=float(np.mean(effs)),
                       fake_rate=float(np.mean(fakes)),
                       log_pt_mae=float(np.mean(nres)))

    with open(os.path.join(args.out, f"{tag}_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[{tag}] done: {json.dumps({k: v for k, v in metrics.items() if k != 'config'}, default=float)}",
          flush=True)


if __name__ == "__main__":
    main()
