# Jet Energy Regression Study — Design & Plan

*Target: per-jet log response `y = ln(E_true/E_reco)` for light jets from the 5D
ntuples, comparing input modalities (tracks / jet features / cells), cell
encodings (neighbor-graph vs set), and calo-dataset pretraining. Status:
**plan for review — no training has been run.** Branch tagged `dfm-merge-v0`
before any study modifications. This revision incorporates a two-lens
(physics + implementation) internal review of the plan.*

---

## 1. Goal and headline comparisons

Regress the jet energy response and measure jet energy scale (JES) and
resolution (JER) against the calibrated jets already in the ntuple. The
comparisons, in decreasing priority:

| # | Comparison | Question it answers |
|---|---|---|
| C1 | Inputs: {T, J, C, TJ, TC, JC, TJC} | Which detector information carries the calibration — and **how much of the information the calibration chain distilled into J survives in the sparse cell subset C?** |
| C2 | Cell encoding: neighbor-**graph** (calo geometry via cell matching) vs **set** (pure ISAB, no local mixing) | Is detector topology an inductive bias worth carrying, at these (sparse) cell multiplicities? |
| C3 | Pretraining: masked calo pretraining vs from scratch | Does the foundation-model pretext transfer across datasets and improve calibration, especially at low label counts? |
| C4 | Everything vs the **ntuple's own calibrated jets** | Is any of this better than what reconstruction already did? |

T = ghost-associated tracks, J = reco-jet features, C = cells in a ΔR < 0.4
cone around the jet axis.

**What the C tokens actually contain** (measured on the calo dataset,
emulating the 5D selection on cells in jets with pT > 20 GeV, clean events):
the E > 0.5 GeV ∧ 2σ cut retains **74.6% of the jets' clustered energy**
(energy-weighted; per-jet median 62%, 10th–90th percentile 33–85%, rising
with jet energy). Per layer, in jets: Tile A/BC 81–86%, EME2 82%,
EMB2 61%, **EMB1 32%, EMB3 34%**. So the 5D subset carries the *majority*
of jet energy — what's missing is the 4-2-0 growth/perimeter soft component
(plus all negative-energy cells, dropped one-sidedly) — and earlier
count-based language ("EMB1/EMB3 essentially absent") overstated the
deficit: cell *counts* in those layers are ~0.2–0.3% retained, but energy
is ~a third.

**The C1 prior, calibrated by those numbers**: J still carries information C
lacks — the event-level pileup conditioning (ρ, NPV, µ) injected by the
calibration chain, and the ~25% (jet-dependent, *fluctuating*) soft
component. The average missing fraction is learnable (a scale correction);
its jet-by-jet fluctuation is a resolution cost C cannot recover. So a
JC−C gap is still expected, but modest — and its size is exactly the
measurement: it quantifies the calibration information in the sub-threshold
component plus pileup conditioning. To separate the two we add **µ
(`averageInteractionsPerCrossing`) as an optional conditioning input**: if
JC−C shrinks substantially when µ is added to C, the gap was mostly pileup
conditioning, not lost constituents. The original hypothesis — "jet
features are learned from the constituents" — corresponds to the full-cell
limit (S2/S4), which 5D data approaches but does not reach.

## 2. Data pipeline

### 2.1 Verified input facts (checked on the server, 2026-08-08)

- `AntiKt4EMTopoJets_matchedTruth_{pt,eta,phi,m}` **exist and are filled**
  (48.3% of jets matched; unmatched sentinel −9.999). The review's
  "long-lead item" (truth jet 4-vectors) is **not** blocking.
  `AntiKt4EMTopoJets_response` ≡ `pt/matchedTruth_pt` exactly (verified).
- The `matchedTruth_dR` *branch* is unfilled (9999) — but ΔR is directly
  **recomputable** from the stored truth η/φ vs reco η/φ, and is used as a
  match-quality cut (§2.3).
- Response of ntuple jets before cuts: median 1.15, mean 1.66 — a severe
  bad-match / pileup-match tail. **Quality cuts are load-bearing** (§2.3),
  and the B0 response distribution after cuts is a mandatory sanity gate:
  if the median does not come down to ~1, the matching itself must be
  understood before any training.
- Labels: 80.9% light, 13.3% b, 4.4% c, 1.5% τ per jet (ttbar). ~3.9 matched
  light jets/event ⇒ ~1.3M light-jet samples over all 17 files.
- Track association: `ghostTrack_idx` (median 18 tracks/jet, 1.9% empty);
  `btagTrack_idx` is empty for 100% of EMTopo jets — dead, do not use.
- Cells in a ΔR < 0.4 cone: median 20, p90 64, max ~124 — the 5D subset is
  sparse in *counts* but retains **~75% of jet clustered energy**
  (energy-weighted; see §1). Token counts are tiny.
- **Data-quality alert (calo files)**: 40/100 events of `ttbar_100events`
  and ~1/3 of `ttbar_1000` contain unphysical cell energies (single cells
  to −127 GeV; cluster sums of ±tens of TeV). `hh_bbtt` is far cleaner
  (~2%). Every consumer of calo energies (pretraining included) must filter
  events on sane cluster-energy sums; the retention numbers above use clean
  events only. Root cause unknown — flagged for separate investigation
  (also affects existing foundation-model trainings).
- Jet pT range ~15–370 GeV; all energies GeV. `nConstituents` and
  `averageInteractionsPerCrossing` branches exist (read successfully).

### 2.2 Per-jet dataset builder (`dfm/jetreg/build_dataset.py`, new)

One sample = one matched **light** jet:

| Field | Content | Source |
|---|---|---|
| `y` | `ln(E_true/E_reco)`; E from (pt, η, m): `E² = (pt·coshη)² + m²` | matchedTruth vs reco branches |
| `jet` (J) | `[η, sinφ, cosφ, log pt, log m, log(1+nConstituents)]` | reco jet branches |
| `mu` | `averageInteractionsPerCrossing` (optional conditioning, §1) | event branch |
| `tracks` (T) | ghost-associated tracks × 18 features (`dfm.data.TRACK_FEATURES`) | `ghostTrack_idx` → `Track_*` |
| `cells` (C) | cone cells × baseline-7 feature *formulas* (`snr_scaled, snr>4, snr>2, snr>0, η, sinφ, cosφ`), snr ≡ `Cell_significance` | `Cell_*` in ΔR < 0.4 |
| `cell_edges` | neighbor-graph edges among the cone cells | cell matching (below) |
| `w` | flattened-spectrum training weight, **capped** (§4) | computed at build time |
| meta | event idx, file, pt_true, η, E_reco, E-response, per-jet edge/degree stats | bookkeeping / metrics |

Builder guards (from plan review): all four `matchedTruth` components are
validated jointly (reject sentinels, require m ≥ 0, sane η; rejection counts
in the manifest), and the E_true distribution is spot-checked against
`pt_true·cosh η_true` before first training.

**The graph bridge (new capability unlocked by `cell_matching.py`):** cone
cells are matched to the calo geometry with the exact (x,y,z)+sampling
matcher; we use `MatchResult.index` (the position in the hash-ordered branch)
directly — no reliance on hashID conventions — then map to the processed
dataset's compact numbering via `cells_*.npy` (`orig_idx`), and restrict
`pairs_*.npy` to the cone subset with `dfm.tokens.subset_subgraph`.
Cone-cell feature rows are ordered by compact index (the same permutation
`subset_subgraph` uses), so edges and tokens cannot misalign, and a
one-shard assert checks edge endpoints are geometric neighbors. Cells
missing from the compact set (the processed pipeline drops noiseSigma==0
cells) are **kept as edgeless tokens** and counted in the manifest. Edges
are computed at build time and stored — training-time cost is zero.

**C2 interpretability guard**: the builder records per-jet edge counts,
degree distributions, and the isolated-cell fraction. If the thinned cones
turn out to have near-zero median degree, `local="edges"` ≈ `local="none"`
numerically and a null C2 would be "topology was absent from the data", not
"topology is useless" — the manifest stats let us say which. (Fallback for a
real C2 answer: run the same A/B on calo-format cells, where the graph is
dense.)

Output: sharded `.npz` (one per input file) + `manifest.json`, written to
`/storage/mxg1065/jetreg/` (6.5 TB free).

### 2.3 Selection & target hygiene

- Jet: `HadronConeExclTruthLabelID == 0`, `matchedTruth_pt > 20 GeV`,
  **recomputed ΔR(reco, truth) < 0.3** (kills geometrically bad matches the
  pt cut alone cannot), `|η_reco| < 2.5`.
- Duplicate-truth guard: two reco jets carrying identical matchedTruth
  4-vectors (both matched to the same truth jet) are both dropped.
- Isolation, **both sides** (builder flags, default on): no other reco jet
  with pt > 15 GeV within ΔR < 0.8 of the jet, and no *other* jet's truth
  match within ΔR < 1.0 of this jet's truth axis (truth-side overlap makes
  E_true itself ambiguous — a label bias, not an input effect).
- Splits **by file** (train = files 11+12, val = 13, test = 14; ~150k train
  jets before isolation): normalization, reweighting, and all model
  selection touch train files only.

### 2.4 Pretraining path (C3)

Masked pretraining uses `dfm.masking.MaskedTokenPretraining` as smoke-tested
(S3 subset, cluster masking, reconstruct `snr_scaled`, mask indicator on),
with the pretrained cell branch transferring verbatim into the regression
model (`mask_indicator` width compatibility; 91/104 tensors ported in the
smoke test). Set-encoding variant: same pretext with `local="none"`.

**Pretraining corpus — changed after plan review.** The obvious choice
(`processed_data/ttbar_1000`) has an unverifiable event overlap with the 5D
ttbar files: EventNumber is unfilled in the Calo ntuples (a known ntuplizer
gap), so disjointness cannot be demonstrated, which would taint exactly the
most contamination-sensitive results (label-efficiency curves). Primary
corpus is therefore **`hh_bbtt_10k_events`** (10k events, 10 shards —
a *different physics process*, so event disjointness holds by construction,
and 10× more pretraining events). `ttbar_1000` becomes a secondary variant
(same-process pretraining) reported with the overlap caveat. The corruption
finding (§2.1) reinforces this choice — hh_bbtt is ~2% corrupted vs ~1/3 of
`ttbar_1000` — and adds a mandatory event-quality filter (sane cluster-energy
sum) to `pretrain_calo.py` regardless of corpus.
Implementation note: `CaloGraphAdapter` currently opens only the first
`events_*.h5` shard — `pretrain_calo.py` extends it to iterate all shards.

Honesty guards: pretraining never touches 5D events; the **linear probe**
(frozen encoder) is reported alongside full finetuning; and `evaluate.py`
includes a cheap domain-shift check (snr_scaled histograms, calo S3 vs 5D
cones — the feature *formulas* match, the distributions need not: signed
MeV-derived SNR vs positive-only GeV-derived significance).

## 3. Architecture (all options are one config away in `dfm`)

One model class (`dfm/jetreg/model.py`), assembled from existing dfm parts:

```
cells  (C) ─ ModalityConfig("cell", 7, local="edges"|"none", mask_indicator=True) ─┐
tracks (T) ─ ModalityConfig("track", 18, local="knn")                              ├─ SharedEventEncoder
                                                                                   │  (ISAB global mixing)
                       masked-mean global token  ◄────────────────────────────────┘
                              │
jet features (J) ─ MLP embed ─┤ (concat when J enabled; µ optionally appended)
                              ▼
                     head MLP → (μ, log σ²)      ← heteroscedastic Gaussian
```

- **Input combinations** = which TokenBatches are fed + whether the J branch
  concatenates. Absent modalities are simply not constructed.
- **C2 graph vs set** = `local="edges"` vs `local="none"`, everything else
  identical, isolating the topology prior.
- **Model scale**: dim 128, 2 local + 2 global layers, ~1M parameters for
  the multimodal configs — deliberately small; this study compares
  *information content*, not capacity.
- σ output doubles as a per-jet resolution estimate and the correct loss
  weighting across jet qualities.
- Implementation prerequisite (from plan review): `NeighborConvBlock`
  currently loops per event in Python; before the matrix runs, the loop is
  flattened to one block-diagonal edge index per batch (pure torch, offset
  arithmetic) so graph configs train at the same rate as set configs.

## 4. Training protocol

- Loss: Huber on `y` for warmup epochs → heteroscedastic Gaussian NLL.
  Note the estimator convention: μ estimates the conditional *mean* of the
  log response, while JES closure uses the binned *median* of the ratio —
  for skewed distributions these differ. We quantify the gap on the
  baseline and cross-check with an MAE-trained (median-targeting) variant;
  if closure-vs-reco misses the target, the fallback is standard numerical
  inversion (correction as a function of reco pT from the truth-binned
  response).
- Spectrum flattening: flat-in-log-pT_true weights over **20–250 GeV only**
  (above 250 GeV the ttbar tail is too sparse — reported as extrapolation),
  weights capped at their 99th percentile; per-bin effective sample size
  `(Σw)²/Σw²` recorded in the manifest so bin populations are known.
- AdamW, cosine schedule + warmup, batch 256, early stop on val NLL;
  3 seeds per configuration, metrics as mean ± spread.
- Normalization: the track adapter *enforces* explicit train-only stats
  (M6); cell features are bounded by construction and used unnormalized
  (as in the calo pipeline); jet features standardized with train-file
  stats stored by the builder.

## 5. Metrics & baselines

Per (pT_true, η) bin, on the test file:

- **JES / closure**: median response `E_corr/E_true` after correction, vs
  pT_true *and* vs pT_reco. Acceptance target: |non-closure| ≲ 1–2% in the
  bulk (20–250 GeV); edge bins reported but not gated (resolution migration
  at the 15–25 GeV spectrum edge is expected even with flat weights).
- **JER**: robust width **IQR/1.349** (Gaussian-equivalent σ) of the
  response per bin — chosen over IQR/2 so any later NSC-style fit
  (`σ/E = N/E ⊕ S/√E ⊕ C`, stretch goal) is in conventional units.
- **Baselines**: (B0) the ntuple's calibrated jets — recomputed as an
  **E-response** with exactly the target's 4-vector construction and cuts
  (the `response` branch is a pT-response; kept only as a cross-check);
  (B1) global median correction; (B2) J-only model (a learned pT/η
  recalibration — the floor any C/T model must beat to claim constituent
  information matters).
- Secondary: pull `(y−μ)/σ` ≈ unit Gaussian (σ calibration).

## 6. Experiment matrix

| Tier | Runs | Configs |
|---|---|---|
| 0 | — | B0/B1 baselines + dataset QA (response after cuts, edge/degree stats, E_true checks) |
| 1 | 4×3 seeds | `J`, `T`, `C-graph`, `C-set` |
| 2 | 5×3 | `TC-graph`, `JC-graph`, `TJ`, `TC-set`, `C-graph+µ` |
| 3 | 2×3 | `TJC-graph`, `TJC-set` |
| 4 | 4×3 | `C-graph`+pre, `TC-graph`+pre, linear probe, label-efficiency (1/10/100% on `TC-graph`±pre) |

≈ 48 trainings. With the batched NeighborConv (§3), an A30 trains one config
in ~10–20 min (~150k jets × ~50 epochs); the matrix is an afternoon on one
GPU or **~2 h across the 4 idle A30s** (one config per GPU). Dataset build:
one pass over 4 of 17 ntuple files with cell matching — tens of minutes,
once.

## 7. Implementation plan (all new code under `dfm/jetreg/`)

| File | Role |
|---|---|
| `build_dataset.py` | 5D files → per-jet shards (§2.2–2.3), incl. cell matching, subgraph extraction, QA stats |
| `pretrain_calo.py` | masked pretraining on hh_bbtt (primary) / ttbar_1000 (variant), all shards → transferable encoder checkpoints (graph + set) |
| `model.py` | config-driven assembly (inputs, encoding, µ conditioning, pretrained init, freeze) |
| `train.py` | one run: config in → checkpoint + per-jet test predictions + metrics JSON |
| `evaluate.py` | closure/resolution/pull plots, domain-shift check, comparison tables across runs |
| `run_matrix.py` | tiered driver (sequential or one-config-per-GPU) |

Plus one `dfm/` core change: batched block-diagonal `NeighborConvBlock` (§3).

Server workflow: code on branch `dfm-merge` (tag `dfm-merge-v0` = pre-study
state), synced to `/tmp/pycharm_project_9ac966d7`, runs in the `CaloGraphNet`
venv (torch 2.13.0+cu130, CUDA verified on the A30s), artifacts under
`/storage/mxg1065/jetreg/{data,ckpt,results}`. Long runs via `nohup` + logs;
results re-entrant (per-run JSON; `evaluate.py` aggregates whatever exists).

## 8. Risks & honest caveats

1. **The 5D cell subset truncates the soft component**: ~75% of jet
   clustered energy retained on average (§1), but with large jet-to-jet
   spread (33–85%, 10th–90th pct) — the fluctuating part is an
   irreducible resolution cost for cells-only models, so C results are a
   *lower bound* on full-cell performance. The per-layer energy losses
   (EMB1/EMB3 ~⅔ lost, Tile ~15% lost) are now measured, not inferred from
   counts.
2. **Truth-match purity** rests on the recomputed ΔR + pt_true cuts; gate:
   B0 response must tighten to a sane core after cuts, else stop and
   investigate matching before training.
3. **Pretraining domain shift** (calo MeV full-detector vs 5D GeV sparse
   cones): dimensionless snr features are the main defense; the linear
   probe and the histogram domain-shift check are the diagnostics.
4. **Light jets only** defers flavor dependence (b/c response differs); the
   builder keeps labels so extension is a flag, not a rewrite.
5. `Cell_significance` is positive-only in 5D, so snr>0 is degenerate and
   snr>2 nearly so; kept for schema compatibility with the pretrained
   encoder (the masking loss's constant-column guard already handles them).
6. **Spectrum shortcut**: an event-level or J-rich model can exploit the
   ttbar spectrum prior; flat weights + closure-vs-reco are the controls,
   and any transfer claim to other samples needs a different-process test
   set (future: hh_bbtt jets, once cells+tracks exist for the same events).

## 9. What happens next (after your review)

Decision points where your input changes the work:

- **D1**: cuts — pt_true > 20 GeV, ΔR < 0.3, isolation 0.8/1.0 (reco/truth)?
- **D2**: regress E-response (plan) or pT-response?
- **D3**: matrix scope — all tiers, or Tier 0+1 first with a checkpoint
  review before Tier 2+?
- **D4**: files — 4 of 17 (plan), or more?
- **D5**: pretraining corpus — hh_bbtt primary (plan, disjointness by
  construction) vs ttbar_1000 (same-process, unverifiable overlap)?

On approval: implement `dfm/jetreg/` + the batched NeighborConv, build the
dataset, run Tier 0, gate on the §8.2 sanity check, then Tier 1 and report.
