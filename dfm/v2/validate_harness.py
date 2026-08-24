"""G0 shakedown: the harmonized harness must reproduce the archived v1
numbers before it is trusted with v2 (DFM-V2-PROGRAM Phase 0).

For every *_pred.npz in the given results dirs:
  1. recompute the v1 legacy metrics and assert equality with the archived
     *_metrics.json (reproduction contract);
  2. compute the harmonized report (Gaussian core, N/S/C, coverage, tails,
     sigma closure, rescaling) and archive it as *_g0.json;
  3. for headline configs, produce the S4 retro-diagnostic figures.

Run as a file (not -m), with numpy/scipy/matplotlib:
    python validate_harness.py --results DIR [DIR ...] --out OUTDIR \
        --headline TJC-graph TJC-graph+fc TJC-graph+pan
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402


def _js(o):
    if isinstance(o, dict):
        return {k: _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline", nargs="*", default=[])
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = dict(checked=0, reproduced=0, failures=[], figures=[])
    headline_reports = {}

    for rdir in args.results:
        for pred in sorted(glob.glob(os.path.join(rdir, "*_pred.npz"))):
            stem = os.path.basename(pred).replace("_pred.npz", "")
            mpath = os.path.join(rdir, stem + "_metrics.json")
            rep = M.evaluate_predictions(pred)
            summary["checked"] += 1
            if os.path.exists(mpath):
                fails = M.check_reproduction(rep, mpath, rtol=args.rtol)
                if fails:
                    summary["failures"].append(
                        {"run": stem,
                         "diffs": [dict(pop=p, key=k, new=a, ref=b)
                                   for p, k, a, b in fails]})
                else:
                    summary["reproduced"] += 1
            with open(os.path.join(args.out, stem + "_g0.json"), "w") as fh:
                json.dump(_js(rep), fh)
            cfg, _, seed = stem.rpartition("_seed")
            if cfg in args.headline and seed == "0":
                headline_reports[cfg] = rep
            print(f"[g0] {stem}: ok", flush=True)

    if headline_reports:
        import figures as F
        pop = "iso"
        cov = {c: r[pop]["s4"]["coverage"] for c, r in headline_reports.items()}
        tails = {c: r[pop]["s4"]["tails"] for c, r in headline_reports.items()}
        clos = {c: r[pop]["s4"]["sigma_closure"]
                for c, r in headline_reports.items()}
        core = {c: r[pop]["core"] for c, r in headline_reports.items()}
        nsc = {c: r[pop]["nsc"] for c, r in headline_reports.items()}
        fig_dir = os.path.join(args.out, "figures")
        summary["figures"] += [
            F.coverage_diagram(cov, fig_dir, "G0_coverage"),
            F.pull_tail_plot(tails, fig_dir, "G0_pull_tails"),
            F.sigma_closure_plot(clos, M.PT_EDGES, fig_dir, "G0_sigma_closure"),
            F.jer_vs_pt(core, M.PT_EDGES_REPORT, fig_dir, "G0_jer_core",
                        nsc_by_model=nsc),
        ]

    with open(os.path.join(args.out, "g0_report.json"), "w") as fh:
        json.dump(_js(summary), fh, indent=2)
    print(f"\nG0: {summary['reproduced']}/{summary['checked']} runs "
          f"reproduced archived metrics; {len(summary['failures'])} failures")
    for f in summary["failures"]:
        print("  FAIL", f["run"], f["diffs"][:3])
    sys.exit(0 if not summary["failures"] else 1)


if __name__ == "__main__":
    main()
