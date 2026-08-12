# Closing the Gap to ATLAS PUB-Note Standards

*Plan, 2026-08-13. Exemplars analyzed: ATL-PHYS-PUB-2026-018 (probabilistic jet
calibration, 104 pp — directly our topic), ATL-DAQ-PUB-2026-005 (integrated GNN
+ timing, 35 pp), ATL-DAQ-PUB-2026-004 (compact feasibility, 12 pp).*

## 1. Gap analysis

Where our three notes already meet the bar: robust estimators (median,
68%-interval widths), seed spreads on headlines, train-only fitting of
weights/normalizations, isolation-split evaluation, explicit caveat sections,
tiered/gated study design, per-flavor breakdowns.

Where we fall short, ordered by severity:

| # | Gap | Exemplar standard |
|---|---|---|
| G1 | **No predictive-calibration validation.** We quote pull std only. | PHYS-018 makes uncertainty *validation* a results pillar: coverage diagrams, per-bin reduced-chi2, tail fractions P(\|z\|>k) for k=1,2,3 vs normal expectations, predicted-vs-observed resolution closure with ratio panels, and a fit-on-validation/apply-frozen-to-test sigma-rescaling protocol binned in *predicted* pT (deployability argued). |
| G2 | **Single-Gaussian head only.** Our b-jet shoulder finding stops at "mixture-density is the upgrade path". | PHYS-018's whole subject: Gaussian vs Mixture-of-Gaussians vs Generalized-Normal, with full likelihoods, four named stabilization penalties (each tied to a cited failure mode, functional forms in an appendix), variance decomposition (within/between-component/epistemic), and mixture-component anatomy mapped onto the physical JER terms. |
| G3 | **Sample provenance undocumented.** We do not know (or state) generator, tune, PDF, pileup profile, simulation chain of either ntuple production. | Exemplars pin every sample: generator+version+PDF+tune with citations, simulation chain, per-split jet counts down to the last bin's population, resampling procedures with rationale. **This gap needs input from the sample producers (Umar/SLAC for 5D; Mohammad Ali for Calo).** |
| G4 | 3 seeds; no HPO; fixed hyperparameters undocumented as choices. | 10-seed mean±std as the headline convention; staged Bayesian HPO documented as methodology with fANOVA importance and admitted weaknesses; K-scans and 1D ablations with plateau statements. |
| G5 | No epistemic uncertainty. | MC-dropout (25 passes) epistemic term in quadrature, explicitly flagged as a convenience vs deep ensembles. |
| G6 | Fair-comparison discipline partial. | Every figure comparing to the nominal chain re-states, in the caption, that target/estimator/sample differ; aggregation-weighting sensitivity stated ("jet-weighted would favour X"). |
| G7 | No deployment/timing story. | DAQ notes evaluate the exported ONNX artifact (not the checkpoint), embed feature ordering in metadata, and report two-tier timing (chain vs standalone) with hardware caveats and utilization numbers; mechanism metrics bridge cause and effect. |
| G8 | Citations minimal; no detector section; captions not self-contained. | 32 references incl. generic ML concepts; detector boilerplate + coordinate footnote; captions restate sample, operating point, uncertainty treatment. |
| G9 | No appendix corpus. | ~75 of 104 pages are appendices: per-flavor decompositions, robustness audits of the note's own calibration procedure, diagnostic upper bounds explicitly labelled non-deployable. |
| G10 | Operating-point discipline (for tagging/finding). | Pre-declared physics requirement fixes the WP, threshold value printed, per-process breakdowns with the caveat that mixtures shift rates; binomial errors with the reasoning restated in captions. |

## 2. The plan

### Phase A — Metrology upgrade (retrofits all existing results; ~1 week, no external blockers)
- **A1 Calibration-diagnostics harness** in `evaluate.py`: coverage diagrams,
  per-bin chi2_red, pull tails k=1,2,3, predicted-vs-observed sigma closure
  with ratio panels; the s_b = sqrt(chi2_red) rescaling fit on validation in
  predicted-pT bins, frozen for test; tail-sensitivity audit (refit with |z|
  clipped). Re-run over the archived prediction files — no retraining needed.
- **A2 Ten-seed convention** for headline configs (TJC-graph, +fc, +pan):
  7 additional seeds x 3 configs ~ 21 runs, one night.
- **A3 Numerical inversion** for the 20–30 GeV closure (fit on validation in
  reco-pT bins, deployable), retiring our known low-pT non-closure.
- **A4 Figure/caption factory**: sample annotation on every canvas
  (process, mu), self-contained captions, threshold lines with values,
  binomial errors on all efficiencies.

### Phase B — Probabilistic-heads study → DFM-NOTE-2026-04 (~2 weeks; the flagship)
Mirrors PHYS-018 on our backbone and data; directly resolves our b-shoulder:
- **B1** Implement MoG (K scan 2–6, entropy/variance-floor/ordering/KL
  penalties with documented forms) and Generalized-Normal heads; hybrid loss.
- **B2** Full three-head comparison per flavor: resolution, bias, coverage,
  pulls, tails; mixture-component anatomy vs pT (weights, means, widths,
  entropy) interpreted against noise/constant JER terms; does a MoG component
  isolate the semileptonic b population?
- **B3** Epistemic term via MC dropout; variance decomposition figure.
- Compute: 3 heads x 10 seeds on the all-flavor dataset ≈ 30 runs ≈ 2 days on
  4 GPUs, plus K-scan and ablations.

### Phase C — Rigor infrastructure (~1 week, parallel with B)
- **C1** Staged Bayesian HPO (Optuna) on lr/batch → dropout/reweighting →
  head penalties; fANOVA importances; document plateaus honestly.
- **C2** Ablations as first-class documented results (existing ad-hoc ones
  formalized: graph-vs-set, mu-conditioning, MAE, input matrix).
- **C3** Multi-task non-regression protocol (we have the T2 result; adopt the
  "statistically indistinguishable rejection curves" presentation).

### Phase D — Samples & provenance (external dependencies; start immediately)
- **D1 (BLOCKING for publication-grade dataset sections)**: obtain from the
  producers: generator/version/PDF/tune, pileup profile, simulation and
  reconstruction chain, and the jet/track/cell definitions of both ntuple
  productions. Owners: Umar + SLAC contacts (5D), Mohammad Ali (Calo).
- **D2** Detector section + coordinate footnote + upstream-reconstruction
  subsection (EMTopo jets, ghost association, cell significance) so notes are
  self-contained.
- **D3** Sample-dependence: no second track-level sample exists on the server
  — a different-process 5D production is a **long-lead request** (flag now).
  Interim: mu-binned dependence studies within ttbar (data in hand), and
  train/test splits across mu.
- **D4** Pileup-dependence appendix: all key metrics vs mu.

### Phase E — Deployment discipline (~3 days; feeds any online/trigger story)
- **E1** ONNX export with feature-ordering metadata; evaluate the exported
  artifact; two-tier timing (per-jet and per-event; standalone vs with data
  prep), utilization/memory reported, hardware caveats stated.
- **E2** Input-display figure (DAQ-004 Fig. 3 convention): one event's cone
  cells + tracks + jet, truth overlaid.

### Phase F — Documentation rebuild (~1 week after A–C results exist)
- **F1** Note template upgrade: detector section, dataset tables (counts per
  split per flavor per bin), hyperparameter tables, loss equations,
  bibliography (~30 refs: MDN/Bishop, Set Transformer, DETR, ATLAS ML-JES
  notes incl. ATL-PHYS-PUB-2026-018 itself, topoclustering, fANOVA,
  MC dropout, ...).
- **F2** Appendix factories: auto-generate per-flavor/per-bin distribution
  appendices from prediction files (target: main body ~20 pp + appendices).
- **F3** Results sections restructured around explicit questions; scope
  sentences repeated abstract/results/conclusions (DAQ discipline).
- **F4** Reissue Notes 01–03 as v2 under the standard; T3 (MET) and T4 (jet
  finding) notes born under it, with G10 operating-point discipline.

### Suggested order and gates
1. **Week 1**: A1–A4 + C2 + D1 request sent + E1. Gate: diagnostics harness
   reviewed on existing predictions.
2. **Weeks 2–3**: B1–B3 + C1 (+ D4). Gate: three-head comparison reviewed →
   Note 04 drafting.
3. **Week 4**: F1–F4 reissues; T3/T4 notes to the same standard.
4. **Ongoing**: D3 sample request; in-situ-style validation ideas (Z+jets
   equivalent) noted as future work exactly as PHYS-018 does.

### What we deliberately do NOT copy
Collaboration authorship/branding ("The ATLAS Collaboration", official note
numbers) — ours remain clearly-labelled internal drafts; and the full
30M-jet scale (we have 1.2M — quoted honestly, with per-bin counts printed).
