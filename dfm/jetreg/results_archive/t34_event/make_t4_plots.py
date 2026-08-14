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

def atlas_label(ax, x=0.04, y=0.95, dx=0.155):
    ax.text(x, y, "ATLAS", transform=ax.transAxes, fontsize=13,
            fontweight="bold", fontstyle="italic", va="top")
    ax.text(x + dx, y, "Simulation Internal", transform=ax.transAxes,
            fontsize=12, va="top")
    ax.text(x, y - 0.075, r"$\sqrt{s}$ = 14 TeV", transform=ax.transAxes,
            fontsize=10, va="top")

zT = np.load(f"{D}/eval_jetfind_T_seed0.npz")
zC = np.load(f"{D}/eval_jetfind_TC_seed0.npz")

# ---- A: efficiency vs truth pT --------------------------------------------
edges = np.array([20, 30, 45, 65, 90, 125, 175, 250, 400])
ctr = 0.5 * (edges[1:] + edges[:-1])
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for z, lab, c, m in [(zC, "tracks + cells", "#c1272d", "o"),
                     (zT, "tracks only", "#1f77b4", "s")]:
    tpt, fnd = z["tpt"], z["found"].astype(float)
    eff, err = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tpt >= lo) & (tpt < hi)
        n = max(sel.sum(), 1)
        e = fnd[sel].mean() if sel.any() else 0.0
        eff.append(e); err.append(np.sqrt(e * (1 - e) / n))
    ax.errorbar(ctr, eff, yerr=err, color=c, marker=m, ms=4.5, lw=1.3,
                capsize=2, label=lab)
ax.set_xscale("log")
ax.set_xticks([20, 30, 50, 100, 200, 400])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r"truth jet $p_{\mathrm{T}}$  [GeV]")
ax.set_ylabel("Finding efficiency")
ax.set_ylim(0, 1.12)
ax.axhline(1.0, color="gray", lw=0.7, ls=":")
atlas_label(ax)
ax.legend(frameon=False, fontsize=10, loc="lower right",
          title="best seed, existence threshold 0.5", title_fontsize=9)
fig.tight_layout(); fig.savefig(f"{D}/T4A_efficiency.png", dpi=150); plt.close(fig)

# ---- B: matched log-pT residual -------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
bins = np.linspace(-1.5, 1.5, 75)
for z, lab, c in [(zC, "tracks + cells", "#c1272d"),
                  (zT, "tracks only", "#1f77b4")]:
    d = z["dlpt"][z["found"].astype(bool)]
    s68 = np.percentile(np.abs(d - np.median(d)), 68)
    h, e = np.histogram(d, bins=bins, density=True)
    ax.stairs(h, e, color=c, lw=1.4,
              label=f"{lab}  ($\\sigma_{{68}}$ = {s68:.2f})")
ax.set_xlabel(r"$\ln(p_{\mathrm{T}}^{\mathrm{pred}} / p_{\mathrm{T}}^{\mathrm{true}})$  (found jets)")
ax.set_ylabel("Density")
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
atlas_label(ax)
ax.legend(frameon=False, fontsize=9.5, loc="upper right",
          bbox_to_anchor=(1.0, 0.86))
fig.tight_layout(); fig.savefig(f"{D}/T4B_residuals.png", dpi=150); plt.close(fig)
print("done")
