# DFM v2 Protocol

*Phase 0 deliverable (DFM-V2-PROGRAM §1). Binding for every v2 study. Version
1.0, 2026-08-24. Changes to this document require a note in the study's
deviation log and re-review at the next gate.*

## S1 — Pre-registration

Every study starts with a card in `notes/cards/<study>.md`, committed before
the first training of that study, containing: the question (one sentence);
dataset id + split hash; metrics (from `dfm/v2/metrics.py` — no ad-hoc
estimators); baselines that will appear in every table; success criteria with
numbers; the planned figure list. Deviations discovered during the study are
appended to the card under "Deviations", never silently absorbed.

## S2 — Splits and blinding

Assignment by `hash(eventNumber) % 10`: 0–6 train, 7 validation, 8–9 test,
frozen at Gate G1 and recorded in the dataset manifest. Every metrics file
carries the split hash. The test split is evaluated once per note version;
all model selection, HPO, rescaling fits, and iteration happen on validation.
A test evaluation is logged (who, when, which models) in the study card.

## S3 — Statistics

- Headline configs: 10 seeds; scans and ablations: 3 seeds. Report mean ± sd
  across seeds; bootstrap CIs (1000 resamples) for derived quantities.
- Resolutions reported both ways, labeled: robust (IQR/1.349/median) and
  Gaussian core (iterative μ±2σ fit, ATLAS convention), with N/S/C fits and
  χ²/ndf where a pT dependence is shown.
- Efficiencies and tail fractions carry binomial errors.
- Comparisons across samples/targets/estimators restate the difference in the
  caption (fair-comparison discipline). Never report a best seed.

## S4 — Uncertainty validation

Any model emitting a per-jet uncertainty reports, on validation during the
study and on test in the note: coverage at 68.27/95.45/99.73%; pull tails
P(|z|>1,2,3) vs normal; per-bin reduced χ²; predicted-vs-observed σ closure
with ratio panel; and the σ rescaling (s_b = √χ²_red in predicted-pT bins)
fitted on validation, frozen, applied to test — with post-rescaling coverage
restated. Implemented once in `dfm/v2/metrics.py`.

## S5 — Baseline anchoring

Every results table opens with: uncorrected; global-median correction;
numerical inversion; GSC-style MLP (once Phase 2 defines them — frozen
thereafter). Every modality claim states the target parameterization and what
information it already provides (a correction-to-reco-jet model is never
described as "X-only measurement").

## S6 — Figures

All figures through `dfm/v2/figures.py`: sample annotation on the canvas,
self-contained sidecar captions (sample, population, estimator, split,
uncertainty treatment), thresholds drawn with printed values. A note's every
figure regenerates from archived prediction files by one script checked in
next to the note.

## S7 — Documents and gates

The unit of record is a numbered DFM note (PUB-note template). Each program
gate (G0–G7) is a short review by a named person who did not run the study;
the review outcome (pass / conditional / fail + actions) is recorded in the
note's directory. Notes are internal drafts and say so.

## S8 — Reproducibility

- Environment: `env/server-CaloGraphNet.lock` (pip freeze) + python version;
  updated only with a commit that says why.
- Every training archives: config, seed, split hash, per-seed metrics JSON,
  prediction npz. Summaries and figures regenerate from the repo + archive
  alone (no numbers exist only in a terminal).
- Deployment claims are made on the exported ONNX artifact, not the
  checkpoint (Phase 7).

## Server etiquette (operational)

GPUs 1–2 are DFM's; 0 and 3 belong to the LSHash session until its runs
complete (coordination of 2026-08-24; renegotiate via cross-session message).
Write only under `/storage/afarbin/jetreg/`; `/storage/afarbin/lshash/` is
off-limits; nothing durable in home directories (wiped 2026-08-13).
