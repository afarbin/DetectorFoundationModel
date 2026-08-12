"""Train one jet-regression config; write metrics JSON + test predictions.

    python -m dfm.jetreg.train --data-dir .../jetreg/data --out .../jetreg/results \
        --config TC-graph --seed 0
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
from dfm.jetreg.model import JetRegModel, JetRegConfig

PT_FLAT_RANGE = (20.0, 250.0)
N_FLAT_BINS = 30


class JetShards(Dataset):
    """Per-jet samples from one or more builder shards."""

    def __init__(self, paths, iso_only: bool):
        self.data = []
        keys = None
        for p in paths:
            z = np.load(p, allow_pickle=True)
            keys = list(z.keys())
            n = len(z["y"])
            iso = z["iso_reco"] & z["iso_truth"]
            keep = np.nonzero(iso)[0] if iso_only else np.arange(n)
            self.data.append({k: z[k][keep] for k in keys})
        self.arrays = {k: np.concatenate([d[k] for d in self.data])
                       for k in keys}
        self.n = len(self.arrays["y"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.arrays[k][i] for k in self.arrays}


def make_collate(cfg: JetRegConfig, norm):
    tmean, tstd, jmean, jstd = norm

    def collate(items):
        out = {"y": torch.tensor(np.array([it["y"] for it in items]),
                                 dtype=torch.float32)}
        out["w"] = torch.tensor(np.array([it["w"] for it in items]),
                                dtype=torch.float32)
        if cfg.cells != "off":
            feats, edges = [], []
            for it in items:
                feats.append(torch.from_numpy(it["cells"].astype(np.float32)))
                edges.append(torch.from_numpy(
                    it["cell_edges"].astype(np.int64)).reshape(-1, 2))
            out["cells"] = pad_token_list(
                feats, "cell", edges=edges if cfg.cells == "graph" else None)
            if cfg.cells != "graph":
                out["cells"].edges = None
        if cfg.tracks:
            feats, coords = [], []
            for it in items:
                x = it["tracks"].astype(np.float32).copy()
                if len(x) == 0:
                    x = np.zeros((0, tmean.shape[1]), np.float32)
                raw_etaphi = x[:, 1:3].copy()
                x[:, 0] = np.log(np.maximum(x[:, 0], 1e-6))
                x = (x - tmean) / tstd
                feats.append(torch.from_numpy(x))
                coords.append(torch.from_numpy(np.stack(
                    [raw_etaphi[:, 0], np.sin(raw_etaphi[:, 1]),
                     np.cos(raw_etaphi[:, 1])], axis=1)))
            out["tracks"] = pad_token_list(feats, "track", coords=coords)
        if cfg.jet:
            j = np.stack([it["jet"] for it in items]).astype(np.float32)
            out["jet"] = torch.from_numpy((j - jmean) / jstd)
        if cfg.mu:
            out["mu"] = torch.tensor(
                np.array([[it["mu"] / 50.0] for it in items]), dtype=torch.float32)
        if cfg.flavor_cond:
            f = np.array([it["flavor"] for it in items])
            out["flav"] = torch.eye(3)[torch.from_numpy(f).long()]
        return out

    return collate


def to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, (torch.Tensor, TokenBatch)) else v)
            for k, v in batch.items()}


def flat_pt_weights(pt_true_train, pt_true):
    lo, hi = PT_FLAT_RANGE
    edges = np.geomspace(lo, hi, N_FLAT_BINS + 1)
    hist, _ = np.histogram(np.clip(pt_true_train, lo, hi - 1e-6), bins=edges)
    hist = np.maximum(hist, 1)
    w = 1.0 / hist[np.clip(np.digitize(np.clip(pt_true, lo, hi - 1e-6), edges) - 1,
                           0, N_FLAT_BINS - 1)]
    w = np.minimum(w, np.percentile(w, 99))
    return (w / w.mean()).astype(np.float32)


def main():
    # many concurrent jobs x persistent workers exhaust the default
    # file_descriptor sharing strategy ("received 0 items of ancdata")
    torch.multiprocessing.set_sharing_strategy("file_system")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--loss", choices=["nll", "mae"], default="nll",
                    help="mae = median-targeting cross-check (sigma untrained)")
    ap.add_argument("--pretrained", default=None,
                    help="pretrain_calo checkpoint: init the encoder from it")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="linear probe: train only the head")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="label-efficiency: subsample the train jets")
    ap.add_argument("--flavor-select", choices=["all", "b", "c", "light"],
                    default="all", help="restrict all splits to one flavor")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = JetRegConfig.parse(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_name = args.config + ("-mae" if args.loss == "mae" else "") \
        + ("-pre" if args.pretrained else "") \
        + ("-probe" if args.freeze_encoder else "") \
        + (f"-f{args.train_frac:g}" if args.train_frac < 1 else "") \
        + ({"b": "-b", "c": "-c", "light": "-l"}.get(args.flavor_select, ""))
    tag = f"{cfg_name}_seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.data_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    shards = [os.path.join(args.data_dir, f["output"]) for f in manifest["files"]]
    assert len(shards) >= 4, "need at least 4 shards"
    if len(shards) >= 8:   # full-data mode: last 2 test, 2 before that val
        tr_s, va_s, te_s = shards[:-4], shards[-4:-2], shards[-2:]
    else:                  # original 4-file layout
        tr_s, va_s, te_s = shards[:2], shards[2:3], shards[3:4]
    train = JetShards(tr_s, iso_only=True)
    val = JetShards(va_s, iso_only=True)
    test = JetShards(te_s, iso_only=False)   # both populations (D1)
    if args.flavor_select != "all":
        want = {"light": 0, "c": 1, "b": 2}[args.flavor_select]
        for ds in (train, val, test):
            keep = np.nonzero(ds.arrays["flavor"] == want)[0]
            ds.arrays = {k: v[keep] for k, v in ds.arrays.items()}
            ds.n = len(keep)
    if args.train_frac < 1.0:
        rng = np.random.default_rng(1000 + args.seed)
        keep = rng.choice(train.n, max(1, int(args.train_frac * train.n)),
                          replace=False)
        train.arrays = {k: v[keep] for k, v in train.arrays.items()}
        train.n = len(keep)
    print(f"[{tag}] jets: train {len(train)} (iso), val {len(val)} (iso), "
          f"test {len(test)} (all)", flush=True)

    # train-only normalization + weights (M6)
    ttr = np.concatenate([t for t in train.arrays["tracks"]]) \
        if cfg.tracks else np.zeros((1, 18))
    ttr = ttr.astype(np.float64)
    ttr[:, 0] = np.log(np.maximum(ttr[:, 0], 1e-6))
    tmean = ttr.mean(0, keepdims=True).astype(np.float32)
    tstd = np.maximum(ttr.std(0, keepdims=True), 1e-6).astype(np.float32)
    jtr = np.stack(list(train.arrays["jet"])).astype(np.float64)
    jmean = jtr.mean(0, keepdims=True).astype(np.float32)
    jstd = np.maximum(jtr.std(0, keepdims=True), 1e-6).astype(np.float32)
    for ds in (train, val, test):
        ds.arrays["w"] = flat_pt_weights(train.arrays["pt_true"],
                                         ds.arrays["pt_true"])

    collate = make_collate(cfg, (tmean, tstd, jmean, jstd))
    dl_train = DataLoader(train, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=4,
                          persistent_workers=True)
    dl_val = DataLoader(val, batch_size=512, shuffle=False, collate_fn=collate)
    dl_test = DataLoader(test, batch_size=512, shuffle=False, collate_fn=collate)

    model = JetRegModel(cfg).to(dev)
    if args.pretrained:
        sd = torch.load(args.pretrained, weights_only=False, map_location=dev)
        missing, unexpected = model.encoder.load_state_dict(sd["encoder"],
                                                            strict=False)
        n_all = len(model.encoder.state_dict())
        print(f"[{tag}] pretrained init: {n_all - len(missing)}/{n_all} encoder "
              f"tensors loaded from {os.path.basename(args.pretrained)}", flush=True)
    if args.freeze_encoder:
        if model.encoder is None:
            raise ValueError("--freeze-encoder needs a token modality")
        for prm in model.encoder.parameters():
            prm.requires_grad = False
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-5)

    def loss_fn(out, y, w, epoch):
        mu, logv = out[:, 0], out[:, 1].clamp(-8, 8)
        if epoch < args.warmup_epochs:
            per = F.huber_loss(mu, y, delta=0.3, reduction="none")
        elif args.loss == "mae":
            per = (y - mu).abs()   # median-targeting; sigma gets no gradient
        else:
            per = 0.5 * (logv + (y - mu) ** 2 / logv.exp())
        return (per * w).sum() / w.sum()

    def run_epoch(dl, epoch, train_mode):
        model.train(train_mode)
        tot = n = 0.0
        with torch.set_grad_enabled(train_mode):
            for batch in dl:
                batch = to_device(batch, dev)
                out = model(batch)
                loss = loss_fn(out, batch["y"], batch["w"], epoch)
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += float(loss) * len(batch["y"])
                n += len(batch["y"])
        return tot / n

    best, best_ep, bad = np.inf, -1, 0
    ckpt = os.path.join(args.out, f"{tag}.pt")
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        tr = run_epoch(dl_train, ep, True)
        vl = run_epoch(dl_val, ep, False)
        if ep >= args.warmup_epochs:
            sched.step()
        improved = vl < best - 1e-5 and ep >= args.warmup_epochs
        if improved:
            best, best_ep, bad = vl, ep, 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "norm": [tmean, tstd, jmean, jstd]}, ckpt)
        elif ep >= args.warmup_epochs:
            bad += 1
        print(f"[{tag}] ep {ep}: train {tr:.4f} val {vl:.4f}"
              f"{' *' if improved else ''}", flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    model.eval()
    mus, sigs = [], []
    with torch.no_grad():
        for batch in dl_test:
            batch = to_device(batch, dev)
            out = model(batch)
            mus.append(out[:, 0].cpu().numpy())
            sigs.append(np.exp(0.5 * out[:, 1].clamp(-8, 8).cpu().numpy()))
    mu = np.concatenate(mus)
    sig = np.concatenate(sigs)
    pred_path = os.path.join(args.out, f"{tag}_pred.npz")
    np.savez_compressed(
        pred_path, y=test.arrays["y"], mu=mu, sigma=sig,
        pt_true=test.arrays["pt_true"], pt_reco=test.arrays["pt_reco"],
        eta=test.arrays["eta_reco"], phi=test.arrays["phi_reco"],
        response=test.arrays["response"],
        iso=(test.arrays["iso_reco"] & test.arrays["iso_truth"]),
        flavor=test.arrays.get("flavor", np.zeros(len(mu), np.int8)))

    from dfm.jetreg.evaluate import prediction_metrics
    metrics = prediction_metrics(pred_path)
    metrics.update(config=cfg_name, seed=args.seed, params=n_par,
                   best_val=float(best), best_epoch=best_ep,
                   train_minutes=(time.perf_counter() - t0) / 60,
                   n_train=len(train), n_val=len(val), n_test=len(test))
    with open(os.path.join(args.out, f"{tag}_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[{tag}] done: {json.dumps({k: metrics[k] for k in ('nll', 'jes_closure_rms', 'jer_mid')}, default=float)}",
          flush=True)


if __name__ == "__main__":
    main()
