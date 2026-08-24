# Comparison: Umar's transformer jet-pT regression vs the dfm v1 campaign

*2026-08-24. Source: "Jet Hadronic Energy Reconstruction with Cells and Tracks"
(Farbin, Ghaznavi, Qureshi, Schwartzman — 19 slides, `Transformer_jet_regression (1).pdf`),
an independent study on the full 5D/SuperNtuple dataset at SLAC, compared against
our archived `jetreg-campaign-v1` results (Note 01, `results_archive/`).*

## Verdict

Two independently written pipelines, on the same underlying sample, reach the
**same physics conclusions with numerically consistent relative gains**. This is
the strongest cross-validation either study has, and it partially discharges the
"validate v1" goal of the v2 restart — neither result rests on one codebase's
quirks anymore.

## Setup comparison

| | Umar (SLAC) | Ours (dfm v1) |
|---|---|---|
| Sample | mc21 14 TeV ttbar SingleLep (PhPy8EG_A14_ttbar_hdamp258p75) SuperNtuples | Same production, 17-file subset (~1.2M jets kept) |
| Target | y = log(pT_truth/pT_reco), pT_pred = pT_reco·e^f | Identical |
| Reference | ATLAS default calibrated jet pT, same jets | Same ntuple jet pT, "uncorrected" row |
| Track assignment | Cone ΔR<0.4, pT>0.5 GeV, HS-vertex/unassigned, cap 64 | Ghost association, same vertex filter |
| Cell assignment | Cone ΔR<0.4, \|sig\|>2, cap 128 highest-E, **set only** | Cone ΔR<0.4, all cells, **calo-neighbor graph** (and set ablation) |
| Features | 26 track / 25 cell (incl. **time, quality**, noise split) / 10–15 jet | 20 track / 7 cell / 6 jet (+mu) |
| Architecture | (1) 4-layer encoder + jet-context token, zero-init head; (2) GN3-style two-stream, 3+3+3 fusion | SharedEventEncoder: typed tokens, per-modality local mixing + shared ISAB; JetQueryDecoder |
| Loss | Huber(δ=0.1) — point estimate | Heteroscedastic NLL (μ, σ) |
| JER estimator | Iterative Gaussian core fit (ATLAS convention), inclusive + vs pT; N/S/C fit | IQR/1.349/median in truth-pT bins |
| Seeds | Not stated (appears single) | 3 per config |

## Results, made comparable

Absolute JERs cannot be compared directly (Gaussian-core vs IQR estimator;
inclusive vs binned; his population includes gluon/tau splits, wider pT range to
750 GeV). **Relative improvement over the same-jets default is dimensionless and
comparable**:

| Quantity | Umar | Ours (light-only campaign, JER@65) |
|---|---|---|
| Default/uncorrected response median | 1.056 | ~1.03–1.09 by bin (1.044 @65) — consistent |
| Tracks-only relative JER gain | **34.0%** (22.8→15.1%) | **34.5%** (0.2298→0.1504, config T) |
| Tracks+cells relative gain | **42.7%** (22.8→13.1%) | **43.3%** (0.2298→0.1304, TC-graph) |
| + jet features (our TJ / TJC) | (jet inputs already included) | 45.4% (TJ), 50.0% (TJC-graph) |

pT-dependence pattern — his slide 14 vs our per-bin `jer_corr` arrays:

- **Him**: tracks-only gain peaks near 45 GeV and decays to 14–22% above
  100 GeV; tracks+cells holds a flat 34–42% out to 750 GeV.
- **Us**: T gains 35% at 65–90 GeV but only 20–23% in the 175–400 GeV bins;
  TC-graph holds 41–43% there. **Same shape, same crossover.**

Physics conclusions in common:

1. Track information alone gives a large improvement — but both studies agree it
   is a *correction on top of the calorimeter jet* (the target parameterization
   makes this explicit in both), and it saturates at high pT.
2. Cells specifically buy the high-pT reach: his N/S/C decomposition (noise
   11.0→2.3–4.0 GeV either way; S 1.17→0.88 and C 0.043→0.028 **only with
   cells**) is the cleanest statement of *why* — a decomposition we did not do.
3. Flavor-universal improvement with the same semileptonic-b story: his b-jets
   18.9→12.9% with the network "predicting the neutrino loss from the displaced
   track signature" — the same mechanism behind our Note-02 conditioning result
   and the b-response shoulder.
4. Default calibration overshoots (his median 1.056, 10–20% high in the spectra
   ratio at 100–280 GeV; our med_uncorr 1.03–1.20 at low pT) — worth a joint
   reconciliation, since the default was derived for a different sample/config.

## What each study has that the other lacks

**His, that we should adopt in v2:**
- ATLAS-convention **Gaussian core fit** JER and the **N/S/C decomposition**
  with fit quality — instantly legible to ATLAS reviewers (gap-plan G6/G8
  territory), plus spectra-ratio and correction-vs-ideal sanity plots.
- Richer cell features: sampling layer, **time, quality**, noise decomposition —
  note this means cell timing exists in the SuperNtuples, softening our
  "timing tasks blocked" assessment.
- Jet-relative constituent features (Δη, Δφ, ΔR, log pT-fractions) and the
  train-only robust scaler.
- Zero-initialized head (starts at "no correction") and the jet-context token.
- Wider pT coverage (to 750 GeV) and gluon/tau splits — his gluon result
  (25.9→18.6%, the biggest gain) is a slice we never made because our labels
  merge gluons into "light".
- **Sample provenance on slide 3** — this answers most of our blocking D1
  request for the 5D production (mc21, 14 TeV, PhPy8EG_A14_ttbar_hdamp258p75);
  still needed: pileup profile, simulation chain, exact derivation/cuts.

**Ours, that his lacks:**
- **Uncertainty quantification**: heteroscedastic σ with pull/closure checks
  (and the planned coverage/tails/MoG program). His Huber loss yields no
  per-jet uncertainty — this is our clearest differentiator and the PHYS-018
  alignment.
- Seed spread (his numbers appear single-seed), isolation-split evaluation,
  duplicate-truth guards, closure discipline (his medians sit at 0.985/0.991
  with no closure step).
- The **ladder** (C, T, TJ, TC, TJC × graph/set × pretraining × label
  fraction): his study is two points on it. Notably his set-only cells working
  well is consistent with our graph≈set finding (TJC-graph 0.1150 vs TJC-set
  0.1162).
- Multi-task program (flavor conditioning, panoptic, MET, jet finding) and the
  foundation-model claims (pretraining, label efficiency).

## Discrepancies to reconcile (none look alarming)

- His tracks+cells relative gain (42.7%) vs our TJC-graph (50.0%): plausibly
  explained by estimator (Gaussian core vs IQR), population (his includes
  gluons and taus; inclusive weighting emphasizes low pT), his 64/128
  constituent caps, and jets: cone vs ghost association. Needs a common-jets,
  common-estimator evaluation to close.
- His default-calibration overshoot (10–20%) is larger than our med_uncorr
  (3–4% in the same bins) — likely the population/weighting again; verify we
  are reading the same jet-pT branch.

## Proposed joint actions (feeds the v2 program)

1. **Harmonized evaluation**: one shared metrics module — Gaussian-core JER +
   N/S/C (his) + robust IQR, coverage, pulls (ours) — run on both models'
   predictions for the *same* jet set and splits. This is the definitive
   apples-to-apples and a natural first joint deliverable.
2. Adopt his cell feature set (time, quality, layer) and jet-relative features
   in the v2 dataset build; test whether they close the TC vs TJC gap.
3. Put a (μ, σ) head with our validation suite on his two-stream model — tests
   whether our uncertainty machinery transfers to an independent architecture.
4. Fold his slide-3 provenance into the D1 dossier; ask for the remaining
   items (pileup profile, sim chain, derivation cuts).
5. Add gluon/tau splits to our evaluation (needs the parton-label branch his
   SuperNtuples carry — check our 17 files for it).
