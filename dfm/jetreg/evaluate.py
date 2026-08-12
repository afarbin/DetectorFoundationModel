"""Metrics, Tier-0 baselines/QA, and cross-run aggregation.

    python -m dfm.jetreg.evaluate tier0 --data-dir .../jetreg/data
    python -m dfm.jetreg.evaluate aggregate --results .../jetreg/results
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PT_EDGES = np.array([20, 30, 45, 65, 90, 125, 175, 250, 400], dtype=float)
# |eta| bins isolate the barrel, the 1.37-1.52 crack, and the endcap
ETA_EDGES = np.array([0.0, 0.6, 1.0, 1.37, 1.52, 2.0, 2.5])
PHI_EDGES = np.linspace(-np.pi, np.pi, 9)
BULK = slice(0, 7)  # 20-250 GeV bins are "bulk"; last bin is extrapolation


def robust_stats(resp, var, edges=PT_EDGES):
    """Binned median response and Gaussian-equivalent width vs any variable."""
    med, jer, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (var >= lo) & (var < hi)
        if m.sum() < 20:
            med.append(np.nan); jer.append(np.nan); n.append(int(m.sum()))
            continue
        r = resp[m]
        q25, q50, q75 = np.percentile(r, [25, 50, 75])
        med.append(q50)
        jer.append((q75 - q25) / 1.349 / q50)
        n.append(int(m.sum()))
    return np.array(med), np.array(jer), np.array(n)


def _population(y, mu, sig, resp, pt_true, eta=None, phi=None):
    r_corr = resp * np.exp(mu)
    med_c, jer_c, nbin = robust_stats(r_corr, pt_true)
    med_0, jer_0, _ = robust_stats(resp, pt_true)
    pull = (y - mu) / np.maximum(sig, 1e-6)
    nll = float(np.mean(np.log(np.maximum(sig, 1e-6))
                        + 0.5 * ((y - mu) / np.maximum(sig, 1e-6)) ** 2))
    out = {
        "nll": nll,
        "mae": float(np.mean(np.abs(y - mu))),
        "pull_std": float(np.std(pull)),
        "jes_closure_rms": float(np.sqrt(np.nanmean((med_c[BULK] - 1.0) ** 2))),
        "jer_mid": float(jer_c[3]) if np.isfinite(jer_c[3]) else None,
        "jer_mid_uncorr": float(jer_0[3]) if np.isfinite(jer_0[3]) else None,
        "med_corr": med_c.tolist(), "jer_corr": jer_c.tolist(),
        "med_uncorr": med_0.tolist(), "jer_uncorr": jer_0.tolist(),
        "n_bin": nbin.tolist(),
    }
    # eta / phi profiles (JES and JER uniformity; phi should be flat in MC)
    if eta is not None:
        for name, var, edges in (("eta", np.abs(eta), ETA_EDGES),
                                 ("phi", phi, PHI_EDGES)):
            if var is None:
                continue
            mc, jc, nb = robust_stats(r_corr, var, edges)
            m0, j0, _ = robust_stats(resp, var, edges)
            out.update({f"med_corr_{name}": mc.tolist(),
                        f"jer_corr_{name}": jc.tolist(),
                        f"med_uncorr_{name}": m0.tolist(),
                        f"jer_uncorr_{name}": j0.tolist(),
                        f"n_bin_{name}": nb.tolist()})
    return out


def prediction_metrics(pred_path):
    z = np.load(pred_path)
    y, mu, sig = z["y"], z["mu"], z["sigma"]
    resp, pt, iso = z["response"], z["pt_true"], z["iso"].astype(bool)
    eta, phi = z["eta"], z["phi"] if "phi" in z.files else None
    def pop(m):
        return _population(y[m], mu[m], sig[m], resp[m], pt[m], eta[m],
                           None if phi is None else phi[m])
    every = np.ones(len(y), bool)
    out = {"all": pop(every), "iso": pop(iso),
           "noniso": pop(~iso) if (~iso).sum() > 50 else None}
    if "flavor" in z.files:
        fl = z["flavor"]
        for name, code in (("light", 0), ("c", 1), ("b", 2)):
            m = iso & (fl == code)
            out[f"flav_{name}"] = pop(m) if m.sum() > 200 else None
    # headline scalars = isolated population (the trained-on population)
    for k in ("nll", "mae", "pull_std", "jes_closure_rms", "jer_mid"):
        out[k] = out["iso"][k]
    return out


def tier0(data_dir):
    """Dataset QA + no-training baselines from the test shard."""
    with open(os.path.join(data_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    report = {"cutflows": [f["cuts"] for f in manifest["files"]]}
    test = np.load(os.path.join(data_dir, manifest["files"][3]["output"]),
                   allow_pickle=True)
    resp, pt = test["response"], test["pt_true"]
    iso = test["iso_reco"] & test["iso_truth"]
    med, jer, nbin = robust_stats(resp, pt)
    train = [np.load(os.path.join(data_dir, manifest["files"][i]["output"]),
                     allow_pickle=True) for i in (0, 1)]
    resp_tr = np.concatenate([t["response"] for t in train])
    # B1: one global median correction, derived on train, applied to test
    corr = 1.0 / np.median(resp_tr)
    med_b1, jer_b1, _ = robust_stats(resp * corr, pt)
    report.update({
        "test_jets": int(len(resp)),
        "iso_fraction": float(iso.mean()),
        "response_median": float(np.median(resp)),
        "response_mean": float(np.mean(resp)),
        "B0_med_per_bin": med.tolist(), "B0_jer_per_bin": jer.tolist(),
        "B1_global_corr": float(corr),
        "B1_med_per_bin": med_b1.tolist(), "B1_jer_per_bin": jer_b1.tolist(),
        "n_per_bin": nbin.tolist(),
        "cells_per_jet_median": float(np.median(test["n_cells"])),
        "edges_per_jet_median": float(np.median(test["n_edges"])),
        "isolated_cell_fraction": float(np.sum(test["n_iso_cells"])
                                        / max(np.sum(test["n_cells"]), 1)),
        "unmapped_cell_total": int(np.sum(test["n_unmapped"])),
        "tracks_per_jet_median": float(np.median(test["n_tracks"])),
    })
    gate = 0.9 < report["response_median"] < 1.15
    report["gate_response_sane"] = bool(gate)
    out = os.path.join(data_dir, "tier0_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nGATE {'PASSED' if gate else 'FAILED'}: test response median "
          f"{report['response_median']:.3f} (sane window 0.90-1.15)")
    return report


def aggregate(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        with open(p) as fh:
            m = json.load(fh)
        rows.append(m)
    by_cfg = {}
    for m in rows:
        by_cfg.setdefault(m["config"], []).append(m)
    lines = [f"{'config':<14}{'seeds':>6}{'NLL':>9}{'MAE':>9}"
             f"{'closRMS':>9}{'JER@65':>8}{'JERunc':>8}{'pull':>7}{'min/run':>8}"]
    for cfg, ms in sorted(by_cfg.items()):
        def agg(k):
            v = [m[k] for m in ms if m.get(k) is not None]
            return (np.mean(v), np.std(v)) if v else (np.nan, 0)
        nll, nll_s = agg("nll")
        mae, _ = agg("mae")
        clos, _ = agg("jes_closure_rms")
        jer, jer_s = agg("jer_mid")
        jeru = ms[0]["iso"]["jer_mid_uncorr"]
        pull, _ = agg("pull_std")
        tmin, _ = agg("train_minutes")
        lines.append(f"{cfg:<14}{len(ms):>6}{nll:>9.4f}{mae:>9.4f}"
                     f"{clos:>9.4f}{jer:>8.4f}{jeru:>8.4f}{pull:>7.3f}{tmin:>8.1f}")
    table = "\n".join(lines)
    with open(os.path.join(results_dir, "summary.txt"), "w") as fh:
        fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tier0", "aggregate"])
    ap.add_argument("--data-dir")
    ap.add_argument("--results")
    a = ap.parse_args()
    if a.mode == "tier0":
        tier0(a.data_dir)
    else:
        aggregate(a.results)
