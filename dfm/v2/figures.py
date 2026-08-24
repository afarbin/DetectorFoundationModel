"""DFM v2 figure factory (protocol S6).

Every figure: sample annotation on the canvas, self-contained caption written
as a sidecar `<name>.caption.txt`, both themes readable (single style, high
contrast). Pure matplotlib.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE = ("mc21 " + r"$\sqrt{s}=14$ TeV " + r"$t\bar{t}$" +
          " SingleLep (PhPy8EG A14 hdamp258p75), AntiKt4EMTopo, "
          r"$|\eta|<2.5$, $p_T^{true}>20$ GeV")

PT_CENTERS = None  # filled from edges at call time

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.family": "DejaVu Sans",
})


def _annotate(ax, extra=""):
    txt = SAMPLE + (f"  |  {extra}" if extra else "")
    ax.text(0.02, 1.02, txt, transform=ax.transAxes, fontsize=6.6,
            color="0.35", va="bottom")


def _save(fig, out_dir, name, caption):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out_dir, name + ".caption.txt"), "w") as fh:
        fh.write(caption.strip() + "\n")
    return path


def coverage_diagram(cov_by_model, out_dir, name, population="isolated jets"):
    """cov_by_model: {label: coverage() output}."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    noms = [c["nominal"] for c in next(iter(cov_by_model.values()))]
    x = np.arange(len(noms))
    for i, (lab, cov) in enumerate(cov_by_model.items()):
        emp = [c["empirical"] for c in cov]
        err = [c["err"] for c in cov]
        ax.errorbar(x + 0.08 * i, emp, yerr=err, fmt="o", ms=4, capsize=2,
                    label=lab)
    ax.plot(x, noms, "k--", lw=1, label="perfect calibration")
    ax.set_xticks(x, [f"{n:.2%}" for n in noms])
    ax.set_xlabel("nominal central interval")
    ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=7.5, frameon=False)
    _annotate(ax, population)
    return _save(fig, out_dir, name, f"""
Empirical coverage of the predicted per-jet Gaussian intervals vs nominal,
{population}. {SAMPLE}. Perfect calibration lies on the dashed line; points
below it indicate over-confident (too narrow) predicted uncertainties.
Binomial errors; test split.""")


def pull_tail_plot(tails_by_model, out_dir, name, population="isolated jets"):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ks = [1, 2, 3]
    for i, (lab, t) in enumerate(tails_by_model.items()):
        obs = [t[f"p_gt{k}"]["observed"] for k in ks]
        err = [t[f"p_gt{k}"]["err"] for k in ks]
        ax.errorbar(np.array(ks) + 0.06 * i, obs, yerr=err, fmt="s", ms=4,
                    capsize=2, label=lab)
    ax.plot(ks, [t[f"p_gt{k}"]["normal"] for k in ks], "k--", lw=1,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_yscale("log")
    ax.set_xticks(ks, [r"$|z|>1$", r"$|z|>2$", r"$|z|>3$"])
    ax.set_ylabel("tail fraction")
    ax.legend(fontsize=7.5, frameon=False)
    _annotate(ax, population)
    return _save(fig, out_dir, name, f"""
Pull tail fractions P(|z|>k), z=(y-mu)/sigma, {population}, against the
normal expectation (dashed). {SAMPLE}. Excess at k=2,3 measures non-Gaussian
residual tails the single-Gaussian head cannot represent (H8 motivation).
Binomial errors; test split.""")


def sigma_closure_plot(rows_by_model, edges, out_dir, name,
                       population="isolated jets"):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(4.8, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.06})
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for lab, rows in rows_by_model.items():
        pred = [r["pred"] for r in rows]
        obs = [r["obs"] for r in rows]
        ratio = [r["ratio"] for r in rows]
        ax.plot(ctr, pred, "o--", ms=3.5, lw=1, label=f"{lab} predicted")
        ax.plot(ctr, obs, "s-", ms=3.5, lw=1, label=f"{lab} observed")
        axr.plot(ctr, ratio, "o-", ms=3.5, lw=1)
    ax.set_xscale("log"); ax.set_ylabel(r"$\sigma(y)$")
    ax.legend(fontsize=7, frameon=False)
    axr.axhline(1.0, color="k", ls="--", lw=1)
    axr.set_ylabel("obs / pred"); axr.set_xlabel(r"$p_T^{true}$ [GeV]")
    axr.set_ylim(0.7, 1.3)
    _annotate(ax, population)
    return _save(fig, out_dir, name, f"""
Predicted (median per-jet sigma) vs observed (IQR/1.349 of y-mu) resolution
per truth-pT bin, with ratio, {population}. {SAMPLE}. Ratio = 1 everywhere
means the predicted uncertainties close against realized residuals; the ratio
panel is the input to the S4 rescaling protocol. Test split.""")


def jer_vs_pt(core_by_model, edges, out_dir, name, nsc_by_model=None,
              population="isolated jets"):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for lab, core in core_by_model.items():
        res = np.array(core["res"], float) * 100
        err = np.array(core["err_res"], float) * 100
        ax.errorbar(ctr, res, yerr=err, fmt="o-", ms=3.5, lw=1, capsize=2,
                    label=lab)
        if nsc_by_model and nsc_by_model.get(lab):
            f = nsc_by_model[lab]
            ax.plot(f["pt"], np.array(f["fitted"]) * 100, ":", lw=1.2,
                    color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T^{true}$ [GeV]")
    ax.set_ylabel("jet energy resolution [%] (Gaussian core)")
    ax.legend(fontsize=7.5, frameon=False)
    _annotate(ax, population)
    cap = f"""
Gaussian-core JER vs truth pT, {population}, with N/S/C fits (dotted) where
shown. {SAMPLE}. Iterative mu+-2sigma core fit per bin (ATLAS convention);
errors ~ res/sqrt(2n). Test split."""
    if nsc_by_model:
        for lab, f in nsc_by_model.items():
            if f:
                cap += (f"\n{lab}: N={f['N']:.2f}+-{f['err_N']:.2f} GeV, "
                        f"S={f['S']:.3f}+-{f['err_S']:.3f} sqrt(GeV), "
                        f"C={f['C']:.4f}+-{f['err_C']:.4f}, "
                        f"chi2/ndf={f['chi2']:.1f}/{f['ndf']}.")
    return _save(fig, out_dir, name, cap)
