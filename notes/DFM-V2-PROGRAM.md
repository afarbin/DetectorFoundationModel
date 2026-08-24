# DFM v2 — Build, Demonstrate, Validate

*Program document, 2026-08-24. Branch `dfm-v2`; v1 archived at tag `dfm-v1`.
Owners: Amir Farbin (program), Umar Sohail Qureshi (5D data, SLAC study),
Mohammadali Ghaznavi (Calo data, foundation model).*

## 0. Purpose

v1 (the jetreg campaign + Phase-2 T1–T4, Notes 01–03) demonstrated feasibility
fast, at the cost of protocol: baselines were added late, uncertainty was
quoted but not validated, three seeds carried headlines, and the target
parameterization quietly baked the calorimeter measurement into every
"single-modality" result. v2 rebuilds the **results** under a strict protocol —
not the library. The `dfm` package survived two reviews and stays; datasets,
splits, trainings, evaluations, and documents are redone from scratch.

Every v1 headline becomes a **pre-registered hypothesis** (§4) that v2 either
confirms with confidence intervals or root-causes. One hypothesis class is
already independently corroborated: Umar's SLAC study, written against the
same production with a different pipeline, reproduces our relative JER gains
to within a percent (`reviews/umar_transformer_regression_comparison.md`).
v2 turns that "consistent" into "confirmed on common jets with a common
estimator."

**What the paper prescribes** (Genesis proposal, Phase I): a Detector
Foundation Model — a shared geometry-aware encoder over detector constituents
(point-cloud/graph tokens), pretrained with self-supervised objectives (masked
reconstruction / graph completion), finetuned with parameter-efficient heads
for panoptic reconstruction tasks, benchmarked against classical chains. The
program below builds exactly that, and §5 states honestly which proposal
claims our current data can and cannot support.

---

## 1. The protocol (applies to every step)

- **S1 — Pre-registration.** Each study opens with a one-page card committed
  *before* training: question, dataset + split ids, metrics, baselines,
  success criteria, planned figures. Deviations are recorded, not silently
  absorbed.
- **S2 — Frozen splits and blinding.** Train/val/test assigned by hash of
  event number, frozen in Phase 1, identical for every study. The test split
  is evaluated once per note version; all iteration happens on validation.
- **S3 — Statistics.** 10 seeds for headline configs (3 for scans), mean ± sd
  plus bootstrap CIs on derived quantities; robust (IQR) *and*
  ATLAS-convention Gaussian-core JER reported side by side; N/S/C fits;
  binomial errors on all efficiencies; fair-comparison caveats in captions.
- **S4 — Uncertainty is validated, not just quoted.** Coverage diagrams,
  pull tails P(|z|>1,2,3), per-bin reduced χ², predicted-vs-observed σ
  closure with ratio panels, and the fit-on-validation/frozen-on-test σ
  rescaling — a results pillar for every probabilistic head (per
  ATL-PHYS-PUB-2026-018).
- **S5 — Baseline-anchored tables.** Every results table begins with the
  uncorrected row and the classical baselines of Phase 2. Any claim of the
  form "modality X achieves Y" states what information the target
  parameterization already provides (v1's tracks-only lesson).
- **S6 — Standard figure set.** Response spectra with ratio panels, JER vs
  pT/η/μ, per-bin fit panels, coverage/pulls — produced by one figure factory
  with sample annotations and self-contained captions; every figure
  regenerable from archived predictions by one script.
- **S7 — Documents are the unit of record.** Each phase gate produces or
  updates a numbered DFM note (PUB-note template: detector section, dataset
  tables per split/flavor/bin, hyperparameter tables, ~30 refs, appendix
  factories). Gates are reviewed by a named person who did not run the study.
- **S8 — Reproducibility.** Pinned environment, dataset manifests with
  cutflows, archived per-seed metrics JSONs and prediction files, summaries
  regenerable from the repo alone (enforced since the T1/T2 archive fix),
  ONNX export evaluated as the deliverable artifact in Phase 7.

---

## 2. Program at a glance

| Phase | Weeks | Delivers | Gate |
|---|---|---|---|
| P0 Protocol & provenance | 0–1 | Harmonized metrics harness; provenance dossier; hypothesis register | G0: harness validated on v1 archives |
| P1 Dataset v2 | 1–2 | Per-jet + event shards, datasheet, frozen splits | G1: datasheet review; QA figures |
| P2 Baselines & harmonized eval | 2–3 | Classical anchors; joint eval with SLAC study | G2: cross-group memo |
| P3 Per-jet calibration | 3–6 | Ladder under protocol; probabilistic heads | G3: v1 reproduction report; Notes 01v2 + 04 |
| P4 Pretraining claims | 6–8 | Label-efficiency; cross-process transfer | G4: pretraining verdict with CIs |
| P5 Multi-task & event-level | 8–11 | Flavor, panoptic, MET, finder; ΔR context gain | G5: per-task note sections |
| P6 Foundation synthesis | 11–12 | One-backbone scorecard; probes; robustness | G6: scorecard vs pre-declared criteria |
| P7 Documentation & release | 12–14 | Notes v2 series; ONNX + timing; deck | G7: external-style review |

---

## 3. The phases

### Phase 0 — Protocol and provenance (weeks 0–1)

**Build.** (a) The **harmonized metrics module**: Gaussian-core JER + N/S/C
decomposition (Umar's conventions) merged with our robust estimators,
coverage/pull/χ²/σ-closure diagnostics and the rescaling protocol — one
module both groups run. (b) The figure/caption factory (S6). (c) The
hypothesis register (§4) committed. (d) Environment pinned (server venv +
lockfile).

**Assess.** Shake the harness down on the archived v1 predictions — it must
reproduce every v1 table number and Umar's slide numbers from his prediction
files before it is trusted with v2.

**Visualize.** First coverage/pull-tail plots for v1 models — retroactive
diagnostics that also calibrate expectations for §4's H8.

**Document.** `PROTOCOL.md` (S1–S8 in full); provenance dossier: Umar's
slide 3 gives the 5D production (mc21, 14 TeV, PhPy8EG_A14_ttbar_hdamp258p75
SingleLep); request the remainder (pileup profile, simulation chain,
derivation and object definitions) from Umar/SLAC, and the Calo-ntuple
equivalents from Mohammadali. **This request is the one external blocker for
publication-grade dataset sections — send day 1.**

**Gate G0.** Harness reproduces v1 + SLAC numbers; protocol reviewed.

### Phase 1 — Dataset v2 (weeks 1–2)

**Build.** One pass over the 17 SuperNtuple files producing per-jet and
event-level shards with: all flavors + hadron labels, **parton labels if the
branch exists** (for the gluon split Umar showed), Umar's richer cell
features (sampling layer, **time, quality**, noise split), jet-relative
constituent features, isolation flags, truth MET from neutrinos, μ.
Event-number-hash splits frozen (S2). Unit tests on the builder; cell
matching validated against the Calo geometry as in v1.

**Assess.** Cutflows vs v1's (differences explained); truth-match purity and
duplicate-truth audit; thinning bias re-quantified (energy retention vs
η/layer — the known ~75% cap on cell claims).

**Visualize.** Datasheet figures: occupancies, response spectra by
flavor/pT/η/μ, cell-retention maps, per-split population tables down to the
last bin.

**Document.** `DATASHEET.md` — the dataset section every note will import.

**Gate G1.** Datasheet review; splits frozen and hashed into every later
metrics file.

### Phase 2 — Classical baselines and the harmonized evaluation (weeks 2–3)

**Build.** The anchors every later table carries (S5): uncorrected response;
global median correction; **numerical inversion** in reco-pT bins (the
deployable classical calibration — retires v1's low-pT non-closure); a
GSC-style MLP on engineered features (track fraction, layer fractions,
n_trk); for later phases, track-sum/cell-sum/jet-sum MET estimators and
anti-kT truth-matching efficiency curves.

**Assess.** The first joint deliverable with SLAC: run the harmonized module
on **both groups' v1 predictions over the same jet selection** — closing the
two open discrepancies (his 42.7% vs our 50% tracks+cells gain; the
default-calibration overshoot 10–20% vs our 3–4%).

**Visualize.** Baseline JER/closure vs pT/η; spectra-ratio plot (his slide 11
convention) adopted as a standard sanity figure.

**Document.** Baselines note section + cross-group memo.

**Gate G2.** Discrepancies closed or explained; baselines frozen.

### Phase 3 — Per-jet calibration, re-validated (weeks 3–6)

**Build.** The modality ladder re-run under protocol: C, T, TJ, TC, TJC ×
{graph, set} — 10 seeds on headline configs. Two target parameterizations:
(a) correction to the calo jet (deployable, v1's choice), and (b)
**constituents-only absolute pT** — the honest measurement of standalone
information content that v1 lacked and that the "tracks alone" question
demanded. Probabilistic heads as first-class study: single Gaussian vs
mixture-of-Gaussians (K scan 2–6, documented stabilization penalties) vs
generalized normal; staged Optuna HPO with fANOVA importances; ablations
(μ-conditioning, feature groups — does Umar's cell time/quality close the
TC↔TJC gap?).

**Assess.** Tests **H1, H2, H8** (§4). Full S4 uncertainty validation per
flavor. Epistemic term via MC dropout.

**Visualize.** Ladder tables (baseline rows first), JER vs pT/η with
Gaussian-core fit panels, N/S/C decomposition per config, coverage diagrams,
mixture-component anatomy vs pT (does one component isolate semileptonic
b's?), gluon split.

**Document.** Note 01-v2 (calibration) and Note 04 (probabilistic heads — the
flagship, mirroring PHYS-018 on our backbone).

**Gate G3.** Reproduction report: each v1 headline confirmed within CI or
root-caused. Test split touched once, at the end.

### Phase 4 — Pretraining and the foundation claims (weeks 6–8)

**Build.** Masked-token pretraining on unlabeled events (cell cluster-masking
and track vertex-masking); finetune + linear-probe evaluations; label
efficiency at 1% / 10% / 100% with **pre-registered success margins** (v1
verdict at full labels was "marginal" — the foundation-model claim lives in
the low-label regime and must be stated there); **cross-process transfer of
the calo encoder: pretrain on ttbar Calo ntuples, probe on the HH→bbττ Calo
samples** — the one transfer test our data already permits.

**Assess.** Tests **H7**. Frozen-vs-finetuned deltas with CIs; probe suite on
representations.

**Visualize.** Label-efficiency curves with bands; transfer-gap plots;
probe-accuracy tables.

**Document.** Note 05 (pretraining), whatever the verdict — a validated null
is a publishable protocol result.

**Gate G4.** Verdict with CIs against the pre-registered margins.

### Phase 5 — Multi-task and event-level demonstrations (weeks 8–11)

**Build.** Under the v2 protocol: flavor-conditioned vs dedicated calibration
(**H3**); panoptic head with FTAG working points 70/77/85, threshold values
printed, binomial errors (**H4**); event-level MET vs the Phase-2 classical
sums, framed honestly as "MET from available detector subsets" (**H5**); the
DETR jet finder with a dedicated seed-variance program — 10 seeds,
Hungarian-matching stability diagnostics, cost-function ablation (**H6**).
**New:** the event-level vs per-jet study sliced by ΔR to the nearest jet —
the proposal's central "context gain" thesis plot, which v1 never produced.
Quantify how much event-level gain is ttbar kinematics (top/W constraints) by
μ-binned and topology-binned splits, and say so.

**Assess/Visualize.** Rejection-vs-pT curves at fixed WPs; MET resolution vs
truth MET and vs Σ-subset baselines; finder efficiency/fake/residuals vs the
anti-kT chain; the ΔR-sliced gain plot.

**Document.** Notes 02-v2, 03-v2, 06 (MET + finding, born under G10
operating-point discipline).

**Gate G5.** Per-task sections reviewed; non-regression protocol
("statistically indistinguishable rejection curves") applied.

### Phase 6 — The foundation-model synthesis (weeks 11–12)

**Build.** The claim the proposal actually makes: **one backbone, many
heads**. Joint multi-task training vs the dedicated single-task models of
P3–P5; parameter-efficient adaptation (frozen shared encoder + small heads);
representation probes (attention maps on physically interpretable events,
embedding structure by flavor/pT/μ); robustness slices (μ dependence, η
regions incl. the 1.37–1.52 crack); data- and model-scaling curves as compute
allows.

**Assess.** A **pre-declared scorecard**: (i) multi-task non-regression on
every task, (ii) parameter-efficient heads reach ≥X% of full-finetune
performance, (iii) pretraining verdict from P4, (iv) transfer verdict from
P4. The scorecard criteria are fixed at G0 — the "is it a foundation model"
question gets answered against numbers chosen before the runs.

**Document.** Note 07 — the synthesis note; the honest map from our evidence
to the proposal's Phase-I claims.

**Gate G6.** Scorecard reviewed.

### Phase 7 — Documentation and release (weeks 12–14)

Notes v2 series finalized to the PUB-note standard (gap-closure plan Phase F:
detector section + coordinate footnote, provenance-complete dataset tables,
loss equations, bibliography, auto-generated appendices); ONNX export with
feature-ordering metadata, evaluated as the artifact, two-tier timing with
hardware caveats; reproducibility package (one command regenerates every
table and figure from archived predictions); workshop deck v2.

**Gate G7.** An external-style review pass (Umar and Mohammadali review notes
they didn't write) — then the series is tagged `dfm-v2` and circulated.

---

## 4. Hypothesis register (from v1, to confirm or refute)

| # | Hypothesis (v1 evidence) | Tested in |
|---|---|---|
| H1 | Modality ordering TJC > TJ > TC > T > C; graph ≈ set for cells (0.1150 vs 0.1162) | P3 |
| H2 | Relative JER gains vs same-jets baseline: tracks ≈ 35%, tracks+cells ≈ 43% — already corroborated independently by the SLAC study (34.0% / 42.7%); cells specifically buy high-pT (S, C terms) | P2–P3 |
| H3 | One flavor-conditioned model ≥ dedicated per-flavor models (JER 0.1234 vs 0.1384 b / 0.1536 c) | P5 |
| H4 | Panoptic head matches dedicated tagger and near-optimal calibration jointly (NLL −1.406 vs −1.422) | P5 |
| H5 | Learned MET from tracks+cells: 27.0 ± 0.1 GeV vs 43.2 best classical (−37%); tracks-only collapses | P5 |
| H6 | DETR finder: eff 0.99 (TC) vs 0.95 (T), fake 0.11 — with large seed variance to be tamed | P5 |
| H7 | Pretraining ≈ marginal at full labels; small positive at 1% labels — the claim lives in label efficiency | P4 |
| H8 | b-response is bimodal (semileptonic); a mixture component should isolate it — mechanism corroborated by SLAC flavor result | P3 |

---

## 5. What the current data supports

**In hand:** 17 SuperNtuple files (mc21 14 TeV ttbar SingleLep), ~1.5M
matched jets (~556k b / 150k c / 850k light) with tracks, thinned cells
(|SNR|>2, ~75% energy retention), truth jets, truth neutrinos, μ — and, per
Umar's study, cell **time/quality** branches to be confirmed in our copies.
Calo ntuples: full-granularity 187,652-cell events with topocluster truth —
ttbar (~1k processed; more raw on disk) and **HH→bbττ (3k + 10k events)**.
`cell_matching.py` bridges the formats exactly.

*Branch verification done 2026-08-24
(`notes/provenance/branch-inventory.md`): the 17 files are far richer than
v1 used — cell time/quality/noise-split (~80% filled), track timing
(HGTD-era, 39.6% of tracks), track→calo-layer extrapolations, truth
HS-vertex time, in/out-of-time pileup truth jets, extended flavor labels.
Empty stubs: GN2 tagger scores (all zero), track→truth-vertex links.*

**Achievable now**
- The whole per-jet program: modality ladder, probabilistic heads,
  flavor splits (gluon via TruthPart parton matching in the builder), full
  uncertainty validation, HPO, label efficiency.
- Event-level: MET-from-subsets, DETR jet finding, the ΔR context-gain study.
- **Timing-aware features** (cell time/quality, partial track time, truth
  vertex time) — moved up from "blocked".
- **Pileup-jet discrimination** (HS vs in-time vs out-of-time PU truth jets)
  — a new candidate task, fully supported by the data.
- Foundation claims: masked pretraining, linear probes, label-efficiency
  curves, parameter-efficient heads, multi-task non-regression.
- **Cross-process transfer of the calo encoder** (ttbar → HH→bbττ, Calo side).
- μ-dependence robustness; harmonized cross-group evaluation with SLAC.

**Achievable with reprocessing of data already on disk**
- Larger full-granularity Calo processing (raw events exist beyond the ~1k).

**Blocked pending external input** (flag now, in writing)
- Same-event cells+tracks at full granularity → needs a combined ntuplizer
  production (the long-lead request; until then, cell claims carry the
  thinned-subset caveat).
- Sample-dependence of the calibration → a VBF H→inv SuperHJD test slice
  (50k events) exists at SLAC and its 5M-event AOD is rerunnable
  (`notes/provenance/D1-dossier.md`); transfer requested via
  `SLAC-transfer-request.md`. A dijet/Z+jets production remains a request.
  Interim: μ- and topology-binned splits within ttbar.
- True reco-MET comparison → needs soft-term/full-cell MET branches.
- Provenance: **largely resolved via the BigPanDA task record** (HL-LHC
  Run-4 samples — mc21_14TeV, full G4 sim, geometry ATLAS-P2-RUN4-03-01-00,
  SuperHJD dumper in Athena 25.0.62); remaining: AMI decode of
  e8481/s4446/r16176 (pileup profile, generator versions — needs the renewed
  grid cert) and the SuperHJD source.
- GN2 production-tagger baseline → needs a rerun with the scores filled
  (branches exist but are all zero).
- Multi-vertex inference → only the HS truth vertex is stored and
  track→truth-vertex links are empty; reco-vertex *grouping* (per-track
  indices) is available, full multi-vertex truth is not.
- Proposal benchmarks still out of scope: HH→4b (5D side); VBF H→inv is
  *partially* unblocked (50k-event test slice + rerunnable AOD — see above).

## 6. Compute and effort

Server: 4 GPUs (v1's 39-run campaign ≈ 2 days). v2 totals ≈ 200–300 trainings
(P3 ≈ 150 incl. 10-seed headline configs and the head/K scans; P4 ≈ 60;
P5 ≈ 80) → roughly 2–3 GPU-weeks spread over the 14 calendar weeks, well
within capacity with the existing queue/watchdog ops. People: phases are
sequential in their gates but internally parallel; the SLAC joint evaluation
(P2) and the Calo transfer study (P4) are natural Umar / Mohammadali
ownership.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Provenance never fully arrives | Notes ship with "sample as configured" caveat; slide-3 facts already cover generator/tune |
| Thinned-cell ceiling caps cell claims | Quantified in datasheet; combined-ntuplizer request filed as long-lead |
| Event-level models exploit ttbar kinematics | Stated explicitly; μ/topology-binned checks; transfer sample flagged as the real test |
| DETR seed variance persists | Dedicated stability program in P5; report spread, not best seed |
| Test-set exhaustion | S2 blinding: one test evaluation per note version |
| Bandwidth (two students, part-time) | Gates are cheap reviews; heavy compute is queued, not attended |
