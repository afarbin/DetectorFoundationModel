# Ariel's comments on the SLAC slides → v2 program actions

*2026-08-25. Source: Ariel Schwartzman's email review of Umar's transformer
regression slides. Each comment is mapped to a concrete v2 action and owner.
Items marked (done) were implemented the day this file was committed.*

## Framing and physics stance (slides 3, 18)

**Comment.** Don't frame the work as "jet calibration"; frame it as
addressing the key sources of hadronic shower fluctuations that limit JER,
via a *global* correction that compensates all effects simultaneously —
reducing particle-flow's confusion term rather than competing constituent by
constituent. Position it as the modern implementation/extension of the
Global Sequential Calibration (introduced by members of this team), and as
complementary to p-flow (dual-readout-inspired precision scale vs calibrated
constituents for substructure). Correct only to *particle* (truth-jet)
level; out-of-cone/parton-level effects are process-dependent color-flow
physics, NOT calibration.

**Action.** Adopted as the program's narrative frame: notes' introductions
and the workshop deck rewrite under this stance (P7/F). The target
definition already is particle-level (`ln(pt_truth/pt_reco)` vs matched
truth jets) — the notes must *say* the parton/particle distinction
explicitly. (Program §0 updated — done.)

## Figure of merit: MPV, not median (slides 3, 12)

**Action (done).** The harmonized harness's Gaussian-core fit μ *is* the
MPV; it is now exposed as `mpv` alongside the robust median, and note
templates quote MPV for response. Median stays as the robust cross-check.

## HL-LHC conventions (slides 4, 14)

**Comments.** Sample is HL-LHC (confirmed independently by our provenance
work: Run-4, ATLAS-P2-RUN4-03-01-00). Forward tracks (|η|>2.5) have min pT
1 GeV; jets beyond 2.5 need HGTD; HL-LHC calibration starts at 30 GeV —
plots should start at 30, and low-pT (<150 GeV) deserves a zoomed/log-x
view.

**Actions.**
- (done) Headline binning now starts at 30 GeV (`PT_EDGES_REPORT`); the
  20–30 bin is demoted to a diagnostic (it is also our floor-clipped bin —
  two reasons to drop it from headlines). v1-compatible edges stay untouched
  inside the reproduction contract.
- Log-x / <150 GeV zoom added to the figure-factory conventions.
- Forward jets (|η|>2.5): v1/v2 datasets cut |η|<2.5, so this is a **scope
  decision for Amir/team** — extending needs a rebuild plus HGTD-aware
  track handling (our files: track time valid for 39.6% of tracks, forward-
  concentrated, i.e. exactly the HGTD population). Flagged, not actioned.

## Feature philosophy: physics ladder, not kitchen sink (slide 5)

**Comment.** Baseline should be jet (pT, η, φ) only; add feature groups only
if JER improves — separating a physics-motivated investigation from a
kitchen-sink regression. Track hits probably unnecessary; jet-relative
angles suffice; a jets+tracks-only setting mirrors GSC.

**Action.** P3 gains an explicit **feature-ablation ladder** as a
first-class deliverable: J(pt,η,φ) → +GSC-like engineered (track fraction,
layer fractions, n_trk) → +track relatives → +cells → +hits → +timing, each
step's ΔJER with CIs. Our raw-column dataset schema is unaffected (storage ≠
model inputs); the *configs* implement the ladder. (Program P3 updated —
done.)

## Reporting matrix and example fits (slides 10, 12, 13, 14)

**Action.** Figure factory to add: response-distribution grids (3×3 pT×η),
example core-fit panels (we already archive per-bin fits — G0_jer_core
shows them), response vs pT in η bins / vs η in pT bins, and vs N_PV,
n_HS-tracks, n_PU-tracks per jet (all available: `recoVtx_idx` grouping,
`Track_isTruthHS`). Same-axes discipline for cross-config comparison.
Queued for the G1 datasheet + P2 baseline figures.

## N/S/C caution (slide 15)

**Action (done).** `nsc_fit` now also returns the parameter correlation
matrix from the fit covariance; note templates must present N/S/C as
*descriptive*, flagged very-preliminary, with the correlations shown.
The planned "interplay of N,S,C separately for tracks and cells" systematic
study is P3 material.

## "Ideal" vs "default"; failure-mode event displays (slide 16)

**Actions.**
- Terminology fixed in our comparison doc: the reference is the ATLAS
  *default* calibrated jet.
- **Failure-mode program adopted into P3**: select jets with very
  large/small corrections, produce event displays (cone cells + tracks +
  truth overlay — gap-plan E2 machinery) and feature distributions for
  these populations. This is also exactly Ariel's slide-17 mechanism story
  (threshold effects at the jet boundary: a second pion reaching an
  already-seeded cluster keeps all its energy; alone it fails 4/2 — most
  pronounced at the jet edge, worst for wide gluon jets).

## Truth-jet constituent definition (slide 17) — affects H8

**Comment.** Whether the b-jet "neutrino loss" story is right depends on
whether particle jets include the neutrino (and muon) — Ariel believes
neutrinos are NOT in the particle jets.

**MEASURED (2026-08-25, 2k events, jets pT>30 |η|<2.5).** For jets with a
neutrino (pT>3) in the cone: all-stable cone sum / TruthHSJet_pt median
**0.966**; removing neutrinos → **0.731**. Conclusion: **the truth jets
include neutrinos** — Ariel's expectation is contradicted by this
production; the target contains the invisible energy and the
semileptonic-b "the network predicts the neutrino loss" mechanism stands.
Caveat: TruthPart appears soft-thresholded (overall cone-sum median 0.76),
but the ν-subset differential is robust to that. To be restated in the
datasheet with the full-statistics numbers, and worth reporting back to
Ariel/Umar.

## Electron-matched jets (slide 3)

**MEASURED (2026-08-25, 2k events).** **5.33%** of matched reco jets carry
a TruthPart electron (pT>10) within ΔR<0.2 — the SingleLep lepton faking a
jet. Non-negligible: v1 results (and plausibly the SLAC study) include this
population in "light". v2 action: per-jet electron-overlap **sidecar flag**
(post-pass keyed on event + jet η/φ — no rebuild needed), vetoed in
trainings and reported as a v1→v2 explained difference. Flag the same issue
to Umar.

## Response > 1 concern (slide 17)

**Action.** Matches our measured `med_uncorr` = 1.197 in the 20–30 GeV bin
falling to ~1.0 by 175 GeV: the >1 average IS low-pT dominated. Addressed
structurally by the 30 GeV reporting floor + per-bin (never inclusive-only)
reporting; the numerical-inversion baseline (P2) is the classical fix.

## Quark/gluon vs heavy flavor bookkeeping (slide 3)

**Action.** Our notes state per-flavor populations explicitly (per-bin
counts in the datasheet); no silent mixing. Gluon labels remain unavailable
in our files (TruthPart has no partons — on the Umar ask list).
