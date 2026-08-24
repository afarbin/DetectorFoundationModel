"""DFM v2 harmonized metrics (Phase 0, protocol S3/S4).

One module, both groups' conventions:
  - legacy v1 estimators (robust IQR JER, closure) — kept formula-identical to
    dfm/jetreg/evaluate.py so the G0 shakedown can assert exact reproduction
    of the archived v1 metrics;
  - ATLAS-convention iterative Gaussian-core JER and the N/S/C decomposition
    (the SLAC study's conventions, slides 10/15);
  - uncertainty-validation diagnostics (S4): coverage, pull tails, per-bin
    reduced chi2, predicted-vs-observed sigma closure, and the
    fit-on-one-split/freeze-on-the-other sigma rescaling.

Pure numpy + (optional) scipy; importable without torch. Run scripts as
files, not `python -m` (the dfm package __init__ imports torch).
"""

from __future__ import annotations

import json

import numpy as np

try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except ImportError:          # pragma: no cover - server venv has scipy
    HAVE_SCIPY = False

# ---- v1 conventions, unchanged (reproduction contract) ----------------------
PT_EDGES = np.array([20, 30, 45, 65, 90, 125, 175, 250, 400], dtype=float)
ETA_EDGES = np.array([0.0, 0.6, 1.0, 1.37, 1.52, 2.0, 2.5])
BULK = slice(0, 7)

NORMAL_TAILS = {1: 0.31731, 2: 0.04550, 3: 0.00270}   # P(|z|>k), N(0,1)
# std of a unit normal truncated at +-2 sigma (used by the moment fallback)
_TRUNC2_STD = 0.87962


def robust_stats(resp, var, edges=PT_EDGES):
    """v1 estimator: binned median response and IQR/1.349/median width."""
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


# ---- ATLAS-convention Gaussian core -----------------------------------------

def gaussian_core(x, max_iter=50, window=2.0, tol=1e-5):
    """Iterative Gaussian core fit (SLAC slide 10 convention).

    Fit a Gaussian, refit inside mu +- window*sigma, iterate to convergence.
    Uses a binned least-squares fit when scipy is present; otherwise truncated
    moments with the +-2sigma normal correction. Returns dict with mu, sigma,
    res = sigma/mu, err_res (~ sigma/mu / sqrt(2n_window)), n, n_window,
    converged.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return dict(mu=np.nan, sigma=np.nan, res=np.nan, err_res=np.nan,
                    n=len(x), n_window=0, converged=False)
    q25, mu, q75 = np.percentile(x, [25, 50, 75])
    sig = max((q75 - q25) / 1.349, 1e-9)
    converged = False
    n_win = 0
    for _ in range(max_iter):
        m = (x > mu - window * sig) & (x < mu + window * sig)
        n_win = int(m.sum())
        if n_win < 30:
            break
        w = x[m]
        if HAVE_SCIPY and n_win >= 200:
            nbins = max(20, min(60, n_win // 50))
            cnt, edges = np.histogram(w, bins=nbins)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            keep = cnt > 0
            try:
                p0 = (cnt.max(), np.mean(w), np.std(w))
                popt, _ = curve_fit(
                    lambda t, a, m0, s0: a * np.exp(-0.5 * ((t - m0) / s0) ** 2),
                    ctr[keep], cnt[keep], p0=p0,
                    sigma=np.sqrt(cnt[keep]), absolute_sigma=True, maxfev=5000)
                mu_new, sig_new = popt[1], abs(popt[2])
            except RuntimeError:
                mu_new, sig_new = np.mean(w), np.std(w) / _TRUNC2_STD
        else:
            mu_new, sig_new = np.mean(w), np.std(w) / _TRUNC2_STD
        if abs(mu_new - mu) < tol * sig and abs(sig_new - sig) < tol * sig:
            mu, sig, converged = mu_new, sig_new, True
            break
        mu, sig = mu_new, max(sig_new, 1e-9)
    res = sig / mu if mu else np.nan
    err = res / np.sqrt(max(2 * (n_win - 1), 1)) if n_win > 1 else np.nan
    return dict(mu=float(mu), sigma=float(sig), res=float(res),
                err_res=float(err), n=len(x), n_window=n_win,
                converged=bool(converged))


def core_stats(resp, var, edges=PT_EDGES):
    """Binned Gaussian-core response and resolution vs `var`."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (var >= lo) & (var < hi)
        out.append(gaussian_core(resp[m]) if m.sum() >= 50 else
                   dict(mu=np.nan, sigma=np.nan, res=np.nan, err_res=np.nan,
                        n=int(m.sum()), n_window=0, converged=False))
    return out


def nsc_fit(edges, res, err):
    """Fit sigma(pt)/pt = sqrt((N/pt)^2 + S^2/pt + C^2) over the finite bins.

    `res`/`err` per bin of `edges`; pt taken at the bin geometric mean.
    Returns N, S, C with parameter errors and chi2/ndf (nan-safe).
    """
    if not HAVE_SCIPY:
        return None
    pt = np.sqrt(edges[:-1] * edges[1:])
    res, err = np.asarray(res, float), np.asarray(err, float)
    ok = np.isfinite(res) & np.isfinite(err) & (err > 0)
    if ok.sum() < 4:
        return None

    def f(p, N, S, C):
        return np.sqrt((N / p) ** 2 + S ** 2 / p + C ** 2)

    try:
        popt, pcov = curve_fit(f, pt[ok], res[ok], p0=(5.0, 1.0, 0.05),
                               sigma=err[ok], absolute_sigma=True,
                               bounds=(0, np.inf), maxfev=10000)
    except RuntimeError:
        return None
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((res[ok] - f(pt[ok], *popt)) / err[ok]) ** 2))
    ndf = int(ok.sum() - 3)
    return dict(N=float(popt[0]), S=float(popt[1]), C=float(popt[2]),
                err_N=float(perr[0]), err_S=float(perr[1]), err_C=float(perr[2]),
                chi2=chi2, ndf=ndf, pt=pt[ok].tolist(),
                fitted=f(pt[ok], *popt).tolist())


# ---- S4: uncertainty validation ---------------------------------------------

def pulls(y, mu, sigma):
    return (y - mu) / np.maximum(sigma, 1e-6)


def coverage(z, levels=(0.6827, 0.9545, 0.9973)):
    """Empirical coverage of |z| <= z_nominal for each nominal level."""
    from math import erf, sqrt
    out = []
    n = len(z)
    for lv in levels:
        # z threshold for a central `lv` interval of N(0,1)
        zt = _z_for_level(lv)
        emp = float(np.mean(np.abs(z) <= zt)) if n else np.nan
        err = float(np.sqrt(max(emp * (1 - emp), 1e-12) / n)) if n else np.nan
        out.append(dict(nominal=lv, z=zt, empirical=emp, err=err))
    return out


def _z_for_level(lv):
    """Inverse of erf for the central-interval threshold (bisection, no scipy)."""
    from math import erf, sqrt
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if erf(mid / sqrt(2.0)) < lv:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pull_tails(z):
    """P(|z|>k), k=1,2,3, with binomial errors, vs normal expectation."""
    n = len(z)
    out = {}
    for k, expect in NORMAL_TAILS.items():
        frac = float(np.mean(np.abs(z) > k)) if n else np.nan
        err = float(np.sqrt(max(frac * (1 - frac), 1e-12) / n)) if n else np.nan
        out[f"p_gt{k}"] = dict(observed=frac, err=err, normal=expect)
    return out


def binned_pull_chi2(z, var, edges=PT_EDGES):
    """Per-bin pull std, reduced chi2 (= mean z^2), and s_b = sqrt(chi2_red)."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (var >= lo) & (var < hi)
        n = int(m.sum())
        if n < 50:
            rows.append(dict(lo=lo, hi=hi, n=n, pull_std=np.nan,
                             chi2_red=np.nan, s=np.nan))
            continue
        zz = z[m]
        c2 = float(np.mean(zz ** 2))
        rows.append(dict(lo=lo, hi=hi, n=n, pull_std=float(np.std(zz)),
                         chi2_red=c2, s=float(np.sqrt(c2))))
    return rows


def sigma_closure(y, mu, sigma, var, edges=PT_EDGES):
    """Predicted vs observed resolution per bin of `var`.

    predicted = median per-jet sigma; observed = IQR/1.349 of the residual
    (y - mu). Ratio ~ 1 everywhere is closure.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (var >= lo) & (var < hi)
        n = int(m.sum())
        if n < 50:
            rows.append(dict(lo=lo, hi=hi, n=n, pred=np.nan, obs=np.nan,
                             ratio=np.nan))
            continue
        pred = float(np.median(sigma[m]))
        q25, q75 = np.percentile(y[m] - mu[m], [25, 75])
        obs = float((q75 - q25) / 1.349)
        rows.append(dict(lo=lo, hi=hi, n=n, pred=pred, obs=obs,
                         ratio=obs / pred if pred > 0 else np.nan))
    return rows


def rescaling(z_fit, var_fit, z_apply, var_apply, edges=None):
    """PHYS-018-style sigma rescaling: s_b = sqrt(mean z^2) per bin of the
    *predicted*-pT-like variable, fitted on one split, applied frozen to the
    other. Returns the factors and the post-rescaling tail/coverage of the
    application split."""
    if edges is None:
        edges = PT_EDGES
    fit_rows = binned_pull_chi2(z_fit, var_fit, edges)
    s = np.array([r["s"] for r in fit_rows])
    idx = np.clip(np.digitize(var_apply, edges) - 1, 0, len(edges) - 2)
    s_apply = np.where(np.isfinite(s[idx]), s[idx], 1.0)
    z_res = z_apply / s_apply
    return dict(edges=list(edges), s=[None if not np.isfinite(v) else float(v)
                                      for v in s],
                tails_after=pull_tails(z_res),
                coverage_after=coverage(z_res),
                pull_std_after=float(np.std(z_res)))


# ---- the harmonized per-population report -----------------------------------

def population_report(y, mu, sig, resp, pt_true, eta=None):
    """Everything S3/S4 asks of one prediction population."""
    r_corr = resp * np.exp(mu)
    z = pulls(y, mu, sig)

    # v1-compatible block (reproduction contract with archived metrics)
    med_c, jer_c, nbin = robust_stats(r_corr, pt_true)
    med_0, jer_0, _ = robust_stats(resp, pt_true)
    legacy = dict(
        nll=float(np.mean(np.log(np.maximum(sig, 1e-6))
                          + 0.5 * z ** 2)),
        mae=float(np.mean(np.abs(y - mu))),
        pull_std=float(np.std(z)),
        jes_closure_rms=float(np.sqrt(np.nanmean((med_c[BULK] - 1.0) ** 2))),
        jer_mid=float(jer_c[3]) if np.isfinite(jer_c[3]) else None,
        jer_mid_uncorr=float(jer_0[3]) if np.isfinite(jer_0[3]) else None,
    )

    # ATLAS-convention block
    core_c = core_stats(r_corr, pt_true)
    core_0 = core_stats(resp, pt_true)
    core_incl = gaussian_core(r_corr)
    core_incl_unc = gaussian_core(resp)
    nsc_c = nsc_fit(PT_EDGES, [c["res"] for c in core_c],
                    [c["err_res"] for c in core_c])
    nsc_0 = nsc_fit(PT_EDGES, [c["res"] for c in core_0],
                    [c["err_res"] for c in core_0])

    # S4 block
    s4 = dict(
        coverage=coverage(z),
        tails=pull_tails(z),
        pull_chi2_bins=binned_pull_chi2(z, pt_true),
        sigma_closure=sigma_closure(y, mu, sig, pt_true),
    )
    # retro rescaling: fit on even-index half, apply to odd (v2 trainings will
    # use validation predictions instead — this is the archived-v1 proxy)
    half = np.arange(len(z)) % 2 == 0
    if half.sum() > 500:
        # binned in *predicted* pT (deployable), per the PHYS-018 protocol:
        # pt_pred = pt_reco * exp(mu) = resp * pt_true * exp(mu)
        pt_pred = resp * pt_true * np.exp(mu)
        s4["rescaling_halfsplit"] = rescaling(z[half], pt_pred[half],
                                              z[~half], pt_pred[~half])

    return dict(
        legacy=legacy,
        core=dict(
            inclusive=core_incl, inclusive_uncorr=core_incl_unc,
            med=[c["mu"] for c in core_c], res=[c["res"] for c in core_c],
            err_res=[c["err_res"] for c in core_c],
            res_uncorr=[c["res"] for c in core_0],
            n_bin=nbin.tolist(),
        ),
        nsc=nsc_c, nsc_uncorr=nsc_0,
        s4=s4,
    )


def evaluate_predictions(pred_path):
    """Harmonized report for one *_pred.npz (v1 schema)."""
    zf = np.load(pred_path)
    y, mu, sig = (zf[k].astype(np.float64) for k in ("y", "mu", "sigma"))
    resp, pt = zf["response"].astype(np.float64), zf["pt_true"].astype(np.float64)
    iso = zf["iso"].astype(bool)
    out = {}
    pops = {"all": np.ones(len(y), bool), "iso": iso}
    if (~iso).sum() > 50:
        pops["noniso"] = ~iso
    if "flavor" in zf.files:
        fl = zf["flavor"]
        for name, code in (("light", 0), ("c", 1), ("b", 2)):
            m = iso & (fl == code)
            if m.sum() > 200:
                pops[f"flav_{name}"] = m
    for name, m in pops.items():
        out[name] = population_report(y[m], mu[m], sig[m], resp[m], pt[m])
    return out


def check_reproduction(report, metrics_json_path, rtol=1e-5):
    """Assert the harmonized module reproduces the archived v1 numbers."""
    with open(metrics_json_path) as fh:
        ref = json.load(fh)
    fails = []
    for pop in ("all", "iso"):
        if pop not in report or not isinstance(ref.get(pop), dict):
            continue
        new = report[pop]["legacy"]
        for k in ("nll", "mae", "pull_std", "jes_closure_rms", "jer_mid",
                  "jer_mid_uncorr"):
            a, b = new.get(k), ref[pop].get(k)
            if a is None or b is None:
                continue
            if not np.isclose(a, b, rtol=rtol, atol=1e-9):
                fails.append((pop, k, a, b))
    return fails
