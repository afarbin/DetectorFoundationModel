"""Offline T4 evaluation: per-truth-jet efficiency arrays + residuals."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/storage/afarbin/jetreg/code")
from torch.utils.data import DataLoader
from dfm.jetreg.train_event import EventShards, make_collate, to_device
from dfm.jetreg.event_models import JetFinder, hungarian_targets

DATA = "/storage/afarbin/jetreg/event_data"
OUT = "/storage/afarbin/jetreg/results_t4"
man = json.load(open(f"{DATA}/manifest.json"))
shards = [os.path.join(DATA, f["output"]) for f in man["files"]]
BASE_NEED = ["tracks", "truth_jets", "met_true", "met_tracks", "met_cells",
             "met_jets"]
dev = torch.device("cuda")

for tag, use_cells in [("T", False), ("TC", True)]:
    need = BASE_NEED + (["cells", "cell_edges"] if use_cells else [])
    train = EventShards(shards[:-2], need)
    test = EventShards(shards[-1:], need)
    ttr = np.concatenate([t for t in train.arrays["tracks"][:20000]]).astype(np.float64)
    ttr[:, 0] = np.log(np.maximum(ttr[:, 0], 1e-6))
    tmean = ttr.mean(0, keepdims=True).astype(np.float32)
    tstd = np.maximum(ttr.std(0, keepdims=True), 1e-6).astype(np.float32)
    del train
    collate = make_collate(use_cells, tmean, tstd)
    dl = DataLoader(test, batch_size=24, collate_fn=collate)
    model = JetFinder(use_cells).to(dev)
    ck = torch.load(f"{OUT}/jetfind_{tag}_seed0.pt", weights_only=False,
                    map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    tpt, found, dlpt, n_true, n_pred = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            b = to_device(batch, dev)
            pred = model(b)
            matches = hungarian_targets(pred, b["tj_pt"], b["tj_eta"],
                                        b["tj_phi"], b["tj_mask"])
            pe = torch.sigmoid(pred[..., 0]).cpu()
            for ev, (qi, ti) in enumerate(matches):
                n_t = int(batch["tj_mask"][ev].sum())
                n_true.append(n_t)
                n_pred.append(int((pe[ev] > 0.5).sum()))
                if n_t and len(qi):
                    tp = batch["tj_pt"][ev][ti].numpy()
                    fnd = (pe[ev][qi] > 0.5).numpy()
                    lp = pred[ev, qi, 1].cpu().numpy()
                    tpt.append(tp); found.append(fnd)
                    dlpt.append(lp - np.log(np.maximum(tp, 1e-3)))
    np.savez(f"{OUT}/eval_jetfind_{tag}_seed0.npz",
             tpt=np.concatenate(tpt), found=np.concatenate(found),
             dlpt=np.concatenate(dlpt), n_true=np.array(n_true),
             n_pred=np.array(n_pred))
    print(tag, "done:", len(np.concatenate(tpt)), "truth jets", flush=True)
