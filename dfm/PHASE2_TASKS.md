# Phase 2 — Multi-Task Program on the Shared Backbone

*Follow-up to the completed jet-pT campaign (tag `jetreg-campaign-v1`).
Data facts below verified on the server 2026-08-12. Status: plan for review.*

The four tasks, in proposed order of execution (each reuses the dfm backbone,
the 17-file per-jet/event data machinery, and the pretrained encoders):

---

## T1. Flavor-dependent calibration (b and c jets)

**Goal**: separate JES/JER for b, c, light — the flavor dependence is real
physics (semileptonic b decays lose neutrino energy; different shower
composition) and the known pain point of the standard chain.

**Data: ready now.** The builder already reads `HadronConeExclTruthLabelID`;
it only *keeps* light jets (a `--flavors` flag away). Verified statistics
(matched, pt_true > 20): **~556k b, ~150k c, ~850k light** over 17 files —
ttbar is b-rich, so this is the easiest high-value extension.

**Approach**: rebuild dataset with all flavors + stored label; then two
variants: (a) one model, flavor-conditioned input (one-hot; and a
flavor-blind control to measure what conditioning buys); (b) independent
per-flavor trainings. Report response/JER per flavor before/after; the
b-response *distribution* check for the semileptonic bimodality (a
mixture-density head is the upgrade if a single Gaussian splits the
difference — the review's Part-5 risk note).

**Effort**: builder flag + rebuild (~2 h) + ~15 trainings. Days, not weeks.

## T2. Panoptic head: joint b-tagging + energy determination

**Goal**: one decoder pass per jet emitting flavor probabilities AND
calibrated (mu, sigma) — the "panoptic" step of the proposal; tests whether
tagging and calibration share representation (they should: IP significance
and shower shape inform both).

**Data: same rebuild as T1** (all flavors + labels + tracks with the
IP-significance features already in the schema).

**Approach**: `JetQueryDecoder` already has both output modes — add a
`mode="panoptic"` emitting `[p_b, p_c, p_light, mu, log_var]` with combined
loss `CE + lambda * NLL` (lambda swept). Baselines: the separate single-task
models (does joint training help or hurt each task?), and Umar's tagger
metrics (light/c rejection at fixed b-efficiency vs pT — the FTAG-standard
plots, finally on the same footing as calibration).

**Effort**: small head + loss change; the interesting work is the evaluation
harness (rejection curves). ~1 week including studies.

## T3. Missing ET

**Data: no MET branches (confirmed)** — but **truth MET is derivable**:
`TruthPart` includes neutrinos (verified: 2.4 nu/event, plus ~2.9 e/mu) so
`MET_true = |vector-sum pT(nu)|`. That makes MET *estimation* a trainable
task even without a reco-MET branch to compare against:

- **Target**: truth MET (x, y components; magnitude+phi derived).
- **Inputs**: event-level token sets — all tracks + all (thinned) cells +
  optionally all jets. This is our first genuinely *event-level* model —
  a stepping stone to Stage-2 event-level calibration (same encoder scale).
- **Baselines**: negative vector sum of tracks; of cells; of calibrated jets
  (with and without our corrections). The thinned-cell subset caps the
  cell-based estimate (75% energy retention, biased) - measuring that cap is
  itself the "what did thinning discard" question at event level.
- **Caveat to state up front**: true reco MET needs soft-term/full-cell
  information we don't have; results are "MET from available detector
  subsets", compared honestly against the track-only baseline.

**Effort**: event-level dataset builder (new but simple — no per-jet
association) + event-level model config. ~1 week.

## T4. Jet finding (set prediction)

**Goal**: the genuine DETR step — learned queries + existence head +
Hungarian matching to truth jets: constituents in, jet set out (4-vectors,
optionally flavor: full panoptic reconstruction). Replaces the reco-jet
seeding assumption everywhere above.

**Data: ready** (truth HS jets with 4-vectors as targets; tracks + cells as
inputs; no new branches needed).

**Approach**: extend `JetQueryDecoder` with learned (not kinematics-built)
queries + per-query existence logit; Hungarian matching on (dR, pT) cost;
losses: existence BCE + matched-pair regression (+ flavor CE for panoptic).
Start Stage-1-style: truth-jet-count <= 10, dense events excluded; metrics:
jet-finding efficiency/fake rate vs pT, then 4-vector residuals of matched
jets vs the classical anti-kT chain.

**Effort**: the largest item (matching machinery + new evaluation). ~2-3
weeks. Do last; T1-T3 results inform its heads.

---

## Shared infrastructure (first PR of Phase 2)

One dataset rebuild serving all four: per-jet shards with **all flavors +
labels** (T1/T2) plus a parallel **event-level shard** (all tracks, all
cells+edges, all jets, truth jets, truth MET) for T3/T4 and Stage-2. Single
pass over the 17 files (~2 h with the warm Cell-ID map, which is saved).

## Decision points for review

- **D-P1**: task order (proposed: T1 -> T2 -> T3 -> T4)?
- **D-P2**: flavor calibration as conditioning, separate models, or both
  (proposed: both — the comparison is itself informative)?
- **D-P3**: for T2, working points for the rejection curves (70/77/85%)?
- **D-P4**: event-level shards also feed Stage-2 calibration (non-isolated
  jets) — fold that into Phase 2 or keep separate?
