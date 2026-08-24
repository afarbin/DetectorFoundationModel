# DFM v2 Hypothesis Register

*Phase 0 deliverable. Pre-registered 2026-08-24, before any v2 training.
Margins below are the success criteria the studies will be judged against;
changing them after the corresponding study starts requires a logged
deviation (PROTOCOL S1). Margins marked (proposed) await Amir's sign-off at
Gate G0.*

Conventions: "confirmed" = the v2 measurement (10 seeds, mean ± sd) agrees
with the v1 value within the joint 95% CI, or exceeds it in the claimed
direction. JER = Gaussian-core unless stated; relative gains are vs the
same-jets uncorrected baseline of the same population and estimator.

| # | Claim (v1 evidence) | Pre-registered v2 criterion | Phase |
|---|---|---|---|
| H1 | Modality ordering TJC > TJ > TC > T > C; graph ≈ set for cells | Ordering reproduced with non-overlapping 95% CIs between adjacent rungs (TJC vs TJ, TJ vs TC, TC vs T, T vs C); "graph ≈ set" = CIs overlap (proposed) | P3 |
| H2 | Relative JER gains: tracks ≈ 35%, tracks+cells ≈ 43% (v1: 34.5/43.3; SLAC: 34.0/42.7) | v2 gains within ±3 percentage points of the v1 values, per estimator; harmonized common-jets evaluation agrees with SLAC within ±2 pp (proposed) | P2–P3 |
| H3 | Flavor-conditioned single model ≥ dedicated per-flavor models | Conditioned model's per-flavor JER ≤ dedicated model's, per flavor, within 95% CI; any flavor where dedicated wins by >1σ refutes (proposed) | P5 |
| H4 | Panoptic head: no regression in either task | Rejection curves at 70/77/85 WPs statistically indistinguishable from dedicated tagger (95% CI overlap at every pT bin); calibration NLL within 0.02 of dedicated (proposed) | P5 |
| H5 | Learned MET (tracks+cells) beats best classical sum by ≥30% | v2: ≥30% resolution improvement vs the Phase-2 classical baselines, 10 seeds (v1: 37%) (proposed) | P5 |
| H6 | DETR finder: eff ≥ 0.97 (TC), fake ≤ 0.15, and TC > T | With the stability program: seed sd of efficiency < 0.02, else the claim is "unstable" regardless of mean (proposed) | P5 |
| H7 | Pretraining: marginal at 100% labels; positive at 1% | At 1% labels: pretrained beats scratch by ≥5% relative JER with 95% CI excluding zero across 10 seeds; at 100%: report, no claim. Transfer (ttbar→HH bbττ probe): pretrained probe beats random-init probe by 95% CI (proposed) | P4 |
| H8 | b-response bimodality: a mixture component isolates semileptonic b's | MoG (best K) reduces b-jet P(\|z\|>2) toward normal by ≥half the single-Gaussian excess AND one component's weight correlates with the soft-muon/displaced-track signature (proposed). **Caveat (Ariel, 2026-08-25)**: the mechanism reading depends on whether truth jets include the ν/μ — measured empirically in the G1 QA (TruthHSJet_pt vs TruthPart cone sums ±ν/μ) before H8 is interpreted | P3 |

## Known-in-advance diagnostics (from the G0 retro-analysis of v1)

Recorded here so v2 improvements are claims, not surprises: v1 TJC-graph
(seed 0, iso) shows coverage 67.4% at 68.27% nominal, P(|z|>3) ≈ 0.8% vs
0.27% normal (the H8 target), σ-closure ratio 0.93–0.99 mid-range with
first-bin 0.77, and inclusive Gaussian-core resolution 13.5% (vs SLAC
tracks+cells 13.1% — consistent once the estimator is common).
