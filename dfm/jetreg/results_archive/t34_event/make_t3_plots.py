"""T3 MET figures for the workshop deck backup, matching the note figure style."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white"})

D = "/private/tmp/claude-503/-Users-afarbin-CaloGNN-CaloGraphNet/641347a1-f97c-4347-b083-6ec9af739b1e/scratchpad/t34"
z = np.load(f"{D}/met_TC_seed0_pred.npz")
pred, true = z["pred"], z["met_true"]
base = {k: z[k] for k in ("met_tracks", "met_cells", "met_jets")}

def atlas_label(ax, x=0.04, y=0.95, dx=0.155):
    ax.text(x, y, "ATLAS", transform=ax.transAxes, fontsize=13,
            fontweight="bold", fontstyle="italic", va="top")
    ax.text(x + dx, y, "Simulation Internal", transform=ax.transAxes,
            fontsize=12, va="top")
    ax.text(x, y - 0.075, r"$\sqrt{s}$ = 14 TeV", transform=ax.transAxes,
            fontsize=10, va="top")

def sig68(d):
    return np.percentile(np.abs(d), 68)

# ---- Fig A: per-component residual overlay --------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
bins = np.linspace(-150, 150, 75)
entries = [
    ("model (tracks+cells)", np.concatenate([(pred - true)[:, 0], (pred - true)[:, 1]]), "#c1272d"),
    ("cell sum", np.concatenate([(base["met_cells"] - true)[:, 0], (base["met_cells"] - true)[:, 1]]), "#1f77b4"),
    ("jet sum", np.concatenate([(base["met_jets"] - true)[:, 0], (base["met_jets"] - true)[:, 1]]), "#2ca02c"),
    ("track sum", np.concatenate([(base["met_tracks"] - true)[:, 0], (base["met_tracks"] - true)[:, 1]]), "#7f7f7f"),
]
for lab, d, c in entries:
    h, e = np.histogram(d, bins=bins, density=True)
    ax.stairs(h, e, color=c, lw=1.4,
              label=f"{lab}  ($\\sigma_{{68}}$ = {sig68(d):.1f} GeV)")
ax.set_xlabel(r"$E_{x,y}^{\mathrm{miss}}$ (estimate $-$ truth)  [GeV]")
ax.set_ylabel("Density")
ax.set_xlim(-150, 150); ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
atlas_label(ax)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
fig.tight_layout(); fig.savefig(f"{D}/T3A_residuals.png", dpi=150); plt.close(fig)

# ---- Fig B: resolution vs true |MET| --------------------------------------
mag_t = np.hypot(true[:, 0], true[:, 1])
edges = np.array([0, 20, 40, 60, 80, 100, 130, 170, 250])
ctr = 0.5 * (edges[1:] + edges[:-1])
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for lab, est, c, m in [("model (tracks+cells)", pred, "#c1272d", "o"),
                       ("cell sum", base["met_cells"], "#1f77b4", "s"),
                       ("jet sum", base["met_jets"], "#2ca02c", "^")]:
    d = est - true
    r, rerr = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_ = (mag_t >= lo) & (mag_t < hi)
        dd = np.concatenate([d[m_, 0], d[m_, 1]])
        r.append(sig68(dd))
        # bootstrap-free stat err: sigma68 err ~ sigma/sqrt(2n) scaling proxy
        rerr.append(sig68(dd) / np.sqrt(2 * max(m_.sum(), 1)))
    ax.errorbar(ctr, r, yerr=rerr, color=c, marker=m, ms=4.5, lw=1.3,
                capsize=2, label=lab)
ax.set_xlabel(r"true $|E_{\mathrm{T}}^{\mathrm{miss}}|$  [GeV]")
ax.set_ylabel(r"$\sigma_{68}(E_{x,y}^{\mathrm{miss}})$  [GeV]")
ax.set_ylim(0, 78)
atlas_label(ax)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{D}/T3B_resolution_vs_met.png", dpi=150); plt.close(fig)

# ---- Fig C: |MET| correlation: model vs collapsed T-only -------------------
zT = np.load(f"{D}/met_T_seed0_pred.npz")
mag_pT = np.hypot(zT["pred"][:, 0], zT["pred"][:, 1])
mag_p = np.hypot(pred[:, 0], pred[:, 1])
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
for ax_, mp, ttl in [(axes[0], mag_p, "tracks + cells"),
                     (axes[1], mag_pT, "tracks only (collapsed)")]:
    ax_.hist2d(mag_t, mp, bins=[60, 60], range=[[0, 250], [0, 250]],
               cmap="Blues", cmin=1)
    ax_.plot([0, 250], [0, 250], "k--", lw=0.8)
    ax_.set_xlabel(r"true $|E_{\mathrm{T}}^{\mathrm{miss}}|$  [GeV]")
    ax_.set_title(ttl, fontsize=11)
axes[0].set_ylabel(r"predicted $|E_{\mathrm{T}}^{\mathrm{miss}}|$  [GeV]")
atlas_label(axes[0], dx=0.24)
fig.tight_layout(); fig.savefig(f"{D}/T3C_correlation.png", dpi=150); plt.close(fig)
print("done")
