# DFM project briefing (for proposal writing)

*Prepared 2026-08-25 from the CaloGraphNet workspace (branch `dfm-v2`).
Everything below is committed evidence in the repo unless marked otherwise.*

## What exists

A working prototype of the proposal's Phase-I **Detector Foundation Model**:
one shared geometry-aware encoder over detector constituents (typed tokens:
calorimeter cells with their neighbor graph, tracks) with swappable heads —
implemented, code-reviewed, and exercised end-to-end on real ATLAS
simulation (`dfm/` package, ~3k lines + experiment harness).

**Data**: the samples are HL-LHC **Run-4 upgrade-detector** simulation
(mc21_14TeV ttbar single-lepton, full Geant4, geometry
ATLAS-P2-RUN4-03-01-00, ITk tracker) — provenance fully pinned via the PanDA
production record. We hold ~340k events (1.5M jets) at UTA = 0.6% of a
44.7M-event production at SLAC; a larger transfer is being arranged. A
50k-event **VBF H→invisible** sample (the proposal's own benchmark process)
exists from the same dumper. The ntuples carry cell timing/quality, partial
HGTD-era track timing, truth vertex time, and in/out-of-time pileup truth
jets — so timing-aware reconstruction and pileup-jet discrimination are
supported by data already in hand.

## Demonstrated results (v1 campaign, independently cross-validated)

All from 3-seed campaigns with archived per-seed metrics; robust estimators:

1. **Jet energy**: a global learned correction on the shared backbone
   improves jet energy resolution over the same-jets baseline by **35%
   (tracks only) → 50% (tracks+jets+cells, graph-encoded)**. An independent
   SLAC implementation (Umar Qureshi + Ariel Schwartzman, different
   pipeline, same production) reproduces the relative gains to within a
   percent (34.0%/42.7% by their estimator) — two codebases, one
   conclusion, including the pT-dependence pattern (cells specifically buy
   the high-pT reach: stochastic and constant terms).
2. **One backbone, many heads**: a single flavor-conditioned model beats
   dedicated per-flavor calibrations; a "panoptic" head does b-tagging and
   calibration jointly with no regression in either task.
3. **Event-level**: learned MET from tracks+cells reaches 27.0±0.1 GeV
   resolution vs 43.2 GeV best classical sum (−37%); a DETR-style learned
   jet finder reaches 0.99 efficiency (vs 0.95 tracks-only) at 0.11 fake
   rate — constituents in, jet set out, no anti-kT seeding.
4. **Pretraining**: masked-token pretraining gains are marginal at full
   labels; the foundation-model claim is being staked properly in the
   low-label / transfer regime (pre-registered margins).
5. Physics framing (adopted from Ariel Schwartzman's review): this is a
   **global correction compensating hadronic-shower-fluctuation sources**
   — the modern extension of GSC, complementary to particle-flow (avoids
   its per-constituent confusion term). Not "another ML calibration".

## Rigor posture (differentiator worth citing)

A restart-from-scratch **validation program** (DFM-V2) is underway: v1
results are held as pre-registered hypotheses (H1–H8) with success margins
committed before training; protocol S1–S8 covers frozen event-hash splits
with test blinding, 10-seed headlines with CIs, uncertainty *validation*
(coverage, pull tails, σ-closure — v1's tails are measured at 3× normal,
motivating the mixture-density flagship study), baseline-anchored tables,
and full reproducibility (every table/figure regenerates from archived
predictions; environment pinned). Documentation targets ATLAS PUB-note
standard, benchmarked against three real ATLAS notes. Gates G0–G7 with
named reviewers; G0 (metrics harness reproduces all 60 archived v1 runs
exactly) has passed.

## Team and infrastructure

- Amir Farbin (UTA, PI), Umar Sohail Qureshi (SLAC study, 5D ntuples,
  with Ariel Schwartzman/SLAC), Mohammadali Ghaznavi (Calo ntuples,
  foundation-model line, SLAC data access).
- UTA 4-GPU server (campaigns of ~40 trainings run routinely; v2 needs
  ~2–3 GPU-weeks), CERN lxplus/HTCondor access, direct SLAC→UTA transfer
  path scripted. Multiple coordinated Claude sessions share these
  resources under negotiated splits.

## Near-term roadmap (14-week program, in flight)

Dataset v2 (built, in QA) → classical baselines + harmonized cross-group
evaluation with SLAC → per-jet ladder + probabilistic heads (MoG flagship)
→ pretraining/label-efficiency/cross-process transfer (ttbar→HH→bbττ) →
multi-task + event-level (incl. the ΔR-sliced "event context gain" plot,
the proposal's central thesis) → foundation-model scorecard → notes + ONNX
deployment artifact with timing.

## Honest gaps (don't overclaim in the proposal)

Cells available only as a thinned subset (~75% energy retention) until a
combined cells+tracks ntuplizer production; single physics process in hand
today (VBF H→inv incoming); no reco-MET soft term; gluon labels and GN2
tagger baseline not in current files; multi-vertex truth limited to the HS
vertex. Grid-cert renewal pending for AMI/rucio specifics.

## Pointers

Repo branch `dfm-v2` (tag `dfm-v1` = archived first campaign). Key docs:
`notes/DFM-V2-PROGRAM.md`, `PROTOCOL.md`, `notes/HYPOTHESES.md`,
`notes/provenance/D1-dossier.md`, `reviews/umar_transformer_regression_comparison.md`,
`reviews/ariel_comments_actions.md`. Program page (shareable):
https://claude.ai/code/artifact/842d12cf-4a11-44b3-9f6f-80d935c3207b
