# CaloGraphNet Additions — Event-Level b-Tagging Pipeline & Cell-Masking Pretraining

*Review of `make_event_level_btagging.py`, `detr_settx_event_btagging.py`, and the masked-pretraining extension to `train_gnn_models.py` — August 2026.*
*Part 5 (added after abstract submission) lays out the hadronic jet calibration project as the first stepping stone on this architecture.*

## TL;DR

The repo now contains three coupled capabilities: **(1)** an ntuple→NumPy pipeline + attention-based event-level b-tagger (tracks in, per-jet b/light out), **(2)** a masked-cell pretraining framework bolted onto the calorimeter GNN, and **(3)** the original edge-classification topoclustering. Together these are, structurally, a miniature version of the DFM program in the proposal: a shared encoder over detector constituents, a self-supervised pretext task, and supervised downstream heads. That's the good news.

The specifics need work before results are trustworthy:

- **b-tagger**: the Set Transformer *encoder ignores the track padding mask* — padded zero-tracks participate in attention. Model selection uses accuracy instead of AUC on an imbalanced task. The advertised d0 and track-|η| cuts are not actually applied, and charm jets are silently labeled "light."
- **Cell masking**: the design menu (random / feature / region / cluster masking) is exactly right, but the implementation currently (a) never receives event IDs, so cluster masking silently degrades to random; (b) crashes in all-features mode; (c) computes a valid loss only at `batch_size=1`; (d) gives the model no way to know which cells were masked (no mask flag/token); and (e) pretrains on the downstream test events. Selection of the "best" pretrained model is on *training* loss.
- The **directed message-passing bug** from the July review (edges only flow low-index → high-index; `to_undirected` never applied) is still present and now also degrades the pretraining task, since masked-cell reconstruction depends entirely on messages from neighbors.

None of these are conceptual dead-ends — they're all fixable in days — and the section at the end lays out how the three pieces snap into the foundation-model architecture from the proposal (shared token encoder + pretext masking + per-task decoders, with the student's query decoder being directly reusable as the cells→clusters set-prediction head).

---

## Part 1 — Data processing: `make_event_level_btagging.py`

### What it does

Converts flat ROOT ntuples into a single `.npz` of per-event records for event-level b-tagging:

1. **Reads** 25 per-track branches (kinematics, charge, IP significances `btagIp_*`, hit content, lepton ID, nearest/reco vertex association), plus AntiKt4 EMTopo jet kinematics with `PartonTruthLabelID`, plus truth hard-scatter jet η/φ.
2. **Track selection**: `pT > 1 GeV` and `recoVtx_idx ∈ {0, −1}` (PV-associated or unassociated — a pileup rejection cut). Tracks are sorted by pT descending and truncated at 200.
3. **Jet selection**: EMTopo jets with |η| < 1.6, then **greedy one-to-one ΔR matching** (ΔR < 0.2) to truth hard-scatter jets — this removes pileup jets. Jets with `PartonTruthLabelID = −1` are dropped; the label is binarized as **b if |ID| = 5, else light**. Up to 10 jets/event.
4. **Output**: an object-array `.npz` where each event is a dict: `tracks [n_trk, 25]`, jet kinematics arrays, binary `jets_label`, all-True validity masks, and a per-event `weight = n_b/n_light`.

The overall design — event as a *set of tracks* plus a *set of jet seeds*, labels at the jet level — is the right shape for the event-level tagging question the proposal poses, and greedy HS-matching for pileup-jet removal is sensible.

### Issues

| # | Severity | Issue |
|---|---|---|
| D1 | **High** | **Charm is labeled light.** `is_b = (|ID|==5)`; `|ID|==4` falls into the light class. c-jets have real secondary vertices and are the dominant background at working points that matter — mixing them into "light" both contaminates training and makes the light-rejection numbers unphysical. Keep c as a third class (or at minimum a separate evaluation category). |
| D2 | **High** | **Advertised cuts not applied.** The comment claims `|d0| < 2 mm`; no d0 cut exists in the code. `--eta_max_track` (default 2.0) is parsed and never used — no track |η| cut is applied. Either apply them or fix the comment/args; as-is the dataset card is wrong. |
| D3 | Med | **Parton labeling.** `PartonTruthLabelID` is the deprecated labeling in ATLAS FTAG; hadron-based cone labeling (`HadronConeExclTruthLabelID`) is the standard and avoids known parton-matching ambiguities. If the ntuple has it, switch. |
| D4 | Med | **Per-event weight is computed and then never used** — the training script reads `weight` into the batch and ignores it in the loss. Also `n_b/n_light` as an event weight is an odd quantity (it up-weights b-rich events *and* is undefined-ish for all-b events → 1.0). Class imbalance is better handled with `pos_weight` at the jet level (which the trainer already supports). Delete or justify. |
| D5 | Med | **Jets truncated to `max_jets` after counting** — `n_b_jets`/`n_light_jets` (and the weight) are computed on the full matched list, then the list is truncated to 10; the stored labels and the stored counts can disagree. |
| D6 | Low | **Performance**: the inner loop converts each of 25 awkward branches to NumPy *per event*. This is 10–100× slower than vectorizing per chunk (`ak.to_numpy` on the whole chunk, then slice). Fine at 10⁵ events; painful at the 5×10⁶ cap. |
| D7 | Low | Object-dict `.npz` requires `allow_pickle=True` and loads fully into RAM. At ~20 kB/event, 5M events ≈ 100 GB. Move to padded arrays or HDF5/parquet with lazy loading before scaling. |

---

## Part 2 — Model: `detr_settx_event_btagging.py`

### Architecture

```
tracks [B, N≤200, 25] ──► Encoder ──► Hc [B, 1+N, d]   (global token + per-track embeddings)
                                        │
jet kinematics [B, M≤10, 4] ─► MLP ─► queries ──► L× masked cross-attention ──► head ──► per-jet logit
      (η, φ, log pT, log m)
```

- **Encoder, option 1 (`EventEncoderSetTx`)**: a stack of ISAB blocks (Set Transformer; inducing-point attention, O(N·m)) followed by masked mean pooling to form a global token.
- **Encoder, option 2 (`EventEncoderPET`, `--use_pet`)**: input MLP → *k*-NN EdgeConv layers (DGCNN-style; first layer's graph built in (η,φ), later layers in learned feature space) → ISAB stack → skip connection. This adds the local geometric inductive bias — the same DGCNN/ParticleNet-style local-then-global pattern named in the proposal's DFM section.
- **Decoder (`JetQueriesDecoder`)**: jet kinematics are embedded into queries that cross-attend (mask-aware `MABMasked`) to the track embeddings over `dec_depth` layers; a small MLP head yields one logit per jet. Last-layer attention maps are returned — a nice interpretability hook (which tracks each jet decision used).

**A naming caveat worth making to the student**: this is not DETR in the meaningful sense — queries are *constructed from known reco jets*, not learned object queries, so there is no set prediction, no Hungarian matching, no existence head. It is a **conditional cross-attention tagger** (per-jet attention pooling over the whole event's tracks). That's a perfectly good design — arguably better than DETR for this task — but calling it DETR sets up the wrong expectations. The true DETR machinery becomes relevant for the cells→clusters task (Part 4).

### Training & evaluation process

- Split: random 80/10/10 by event, seeded. Loss: masked BCE-with-logits, optional `pos_weight` and optional focal modulation. AdamW + linear warmup → cosine (or SGDR warm restarts), gradient clipping at 1.0, multi-GPU via DataParallel.
- Feature preprocessing: `log(pT)` for tracks and jets; per-feature standardization with a streaming Welford pass.
- Per epoch: train/val loss, thresholded accuracy, ROC AUC. Best checkpoint by **validation accuracy**; final test pass with the best model; test predictions (un-normalized, un-logged) exported to `.npz` for downstream analysis. That export is good practice.

### Issues

| # | Severity | Issue |
|---|---|---|
| M1 | **High** | **Encoder attention ignores the padding mask.** `ISAB`/`MAB` are the *unmasked* variants; only the decoder uses `MABMasked`. Padded zero-tracks act as keys and queries inside the encoder — valid-track embeddings are contaminated by attention to padding, and the contamination varies with batch composition (max-length padding is per-batch), so the same event gets different embeddings in different batches. The masked pooling at the end doesn't undo this. Fix: thread `X_mask` into the ISAB blocks (mask keys in both MABs; inducing points make this easy) — `MABMasked` already exists, it just isn't used there. |
| M2 | **High** | **Model selection on accuracy.** With b-jets a small minority, val accuracy is nearly saturated by the majority class and is threshold-dependent; checkpoints should be selected on val AUC (already computed) or on light-rejection at fixed b-efficiency. |
| M3 | Med | **AUC is averaged per batch.** Both `train_one_epoch` and `evaluate` mean per-batch AUCs — a biased estimator (batches with few positives are noisy or skipped as 0.0, dragging the average). Accumulate all scores, compute AUC once per epoch. |
| M4 | Med | **k-NN crash on small events**: `topk(k+1)` with `k=10` fails whenever the batch's padded track dimension < 11. Any batch composed of low-multiplicity events will crash the PET path. Guard with `k_eff = min(k, N-1)`. |
| M5 | Med | **Focal loss is not focal**: `focal_alpha` multiplies *every* term (should be α for positives, 1−α for negatives); as written it's just a 0.25× global loss scale interacting confusingly with `pos_weight`. |
| M6 | Low | Normalization statistics are computed on the full dataset before splitting (test leakage of means/σ — mild but avoidable: compute on train indices). |
| M7 | Low | φ enters both tracks and jet queries raw (±π discontinuity). Use sin/cos — and note the standardization of φ makes the discontinuity worse. Same point as the calo review. |
| M8 | Low | PET's "learned-space k-NN" takes the *first two channels* of the feature tensor as coordinates — arbitrary slice, not the DGCNN full feature-space distance. Works, but should be a deliberate choice. |
| M9 | Low | No jet-query self-attention: jets are classified independently given tracks. For HH→4b-style physics (shared/ambiguous tracks between nearby jets), an inter-jet self-attention layer is exactly the "inter-jet correlation" capability the proposal claims — cheap to add (M ≤ 10). |

### Missing physics evaluation

AUC and accuracy are ML metrics; the b-tagging deliverable is **light-jet (and c-jet) rejection at fixed b-efficiency working points (70/77/85%), reported vs jet pT and η**, plus a calibration check of the output scores. The exported `test_predictions.npz` already contains everything needed — this is one plotting script away, and it's the plot that makes the result legible to ATLAS people. The most important experiment for the proposal narrative, though, is a **baseline control**: the same head fed only tracks in a ΔR < 0.4 cone around each jet (i.e., an object-level tagger). Event-level attention beating the cone baseline is precisely the "AI advantage from global event-level inference" claim of the proposal — right now the code can't show it because the baseline doesn't exist.

---

## Part 3 — The cell-masking pretraining task (evaluation)

### What was added

`train_gnn_models.py` gained a `--pretrain` mode implementing masked-autoencoder-style self-supervised pretraining on calorimeter cell graphs:

- **`CalorimeterMasking`** with four strategies: `random` (BERT-style whole-cell masking at ratio 0.15), `feature` (mask chosen feature columns), `geometry` (mask a contiguous η–φ window around a random seed cell — MAE-style block masking), `cluster` (mask all cells of randomly chosen truth topoclusters).
- **`GraphFoundationModel`** refactored into a shared `encode()` (embedding + message-passing stack) with two heads: the original edge classifier, or a **`FeatureReconstructionHead`** (3-layer MLP) for pretraining. Masked-position-only **`MaskedReconstructionLoss`** (MSE/L1, with a categorical branch).
- A **pretrain → transfer → finetune** flow: pretrain for `--pretrain-epochs`, save best, copy encoder weights (excluding heads) into a fresh classification model, finetune for `--finetune-epochs`. Inference mode detects pretrained checkpoints and loads encoder-only.

### Assessment of the concept: right idea, and well-chosen strategy menu

This is the correct pretext family for this data, and the four-way masking menu maps cleanly onto physics: *random* tests local interpolation, *geometry* forces genuine shower-shape completion (can't interpolate from immediate neighbors when a whole region is gone), *cluster* is the most physics-aligned (reconstruct a whole shower from its surroundings — closest to what a clustering-capable representation needs). This is squarely the "masked reconstruction / graph completion" stage-1 objective in the proposal's DFM training strategy, and it's the piece that makes CaloGraphNet a foundation-model testbed rather than a task-specific model. The refactor into `encode()` + swappable heads is exactly the right structural move.

### Implementation problems (ordered by impact)

| # | Severity | Issue |
|---|---|---|
| P1 | **High** | **Cluster masking never gets event IDs.** `pretrain_epoch` unpacks `event_ids` from the batch, but `collate_data` returns `None` in that slot (unchanged from before). So `event_id` is always `None` → `event_id or 0` → every event either falls back to random masking (baseline mode, where `cluster_info` is empty) or uses **event 0's cluster map for all events** (all-features mode). The most physics-relevant masking strategy is silently not functioning. Fix: have `_load_chunk` include the event index in each sample and pass it through collate. |
| P2 | **High** | **Loss is only correct at `batch_size=1`, and crashes otherwise.** `zip([predictions], targets_list, mask_list)` pairs the *concatenated* predictions with only the first graph's targets/mask — at `batch_size>1` the boolean index shape mismatch raises. Also inside the loss, `predictions[mask]` with a 2-D mask flattens to 1-D, after which the per-feature loop degenerates: for all-continuous features every "per-feature" term is the same MSE over all flattened values (loss = F × global MSE — harmless but not what's written), and **any categorical feature (subcalo/sampling one-hots in all-features mode) hits a 1-D indexing error → pretraining crashes in all-features mode**. The `categorical_heads` ModuleDict is constructed and never used — dead code. |
| P3 | **High** | **No mask indicator.** Masked cells are set to 0.0 in all features — but (SNR=0, η=0, φ=0) is not distinguishable from a real quiet cell near the origin, and the model is never told which cells to reconstruct. Standard practice (BERT/MAE): append a binary `is_masked` input feature or add a learnable mask-token embedding (the comment says "can be made learnable" — it's a scalar constant). Without this, the model must treat *every* cell as possibly corrupted, which both weakens the pretext and distorts the finetuning input distribution (finetuning never sees zeros, pretraining sees 15% zeros — a train/finetune mismatch a mask flag would localize). |
| P4 | **High** | **Pretraining set includes the downstream test events** (`train_ratio=1.0` — "use all data"). SSL orthodoxy tolerates unlabeled-pretraining overlap, but here the finetune test events are bit-identical inputs seen during pretraining, so any "pretraining helps" claim is confounded. Pretrain on the train split (or better, on *different files* — there is no shortage of unlabeled events; that's the whole point of SSL). |
| P5 | Med | **Best pretrained checkpoint chosen on training loss** with no validation reconstruction loss — can't detect overfitting/memorization (a real risk given static geometry, see P6). Hold out a val split for reconstruction loss. |
| P6 | Med | **Loss mass goes to static geometry.** Whole-cell masking asks the model to reconstruct SNR *and* η *and* φ. η/φ are event-independent constants recoverable exactly from neighbors → the easy terms dominate the (unnormalized!) loss while the physics term — SNR — is heavy-tailed and unstandardized. Reconstruct **dynamic features only** (SNR/energy), or per-feature-normalize and weight the loss. Related: in all-features mode the leakage features `in_cluster`/`cluster_id_norm` are still present from before — during *cluster-masked* pretraining, reconstructing `in_cluster` from context is uncomfortably close to training the downstream label as a pretext target. Remove them (also flagged in the July review). |
| P7 | Med | **Directed edges hurt pretraining even more than classification.** Reconstruction of a masked cell is only possible via incoming messages; with the still-unfixed low→high-index-only edge orientation, cells receive messages from an arbitrary ~half of their neighbors (and the lowest-index cells from almost none). Fix `to_undirected` first — it's still the single highest-leverage line of code in the repo. |
| P8 | Low | Geometry masking is an O(N) Python loop over 187k cells per event per epoch (slow; vectorize with NumPy), uses a fixed 0.3×radius η–φ box rather than the actual neighbor graph (fine as v1), and `rng = np.random.RandomState()` is unseeded (irreproducible masks). Per-epoch reported `mask_ratio` divides by the *last batch's* mask only. |
| P9 | Low | The pretrain loop repeats the no-autocast `GradScaler` pattern (FP16 is again a no-op), and the finetune path constructs `criterion` twice. |

### What a convincing masking study looks like

Right now nothing measures whether pretraining *helps*. The standard protocol, all cheap once P1–P4 are fixed:

1. **Controls**: from-scratch vs pretrained+finetuned, matched total epochs and LR schedule; report downstream FSS *and* the clustering metrics from the July review.
2. **Linear probe**: freeze the pretrained encoder, train only the edge head. The probe-vs-scratch gap is the cleanest measure of representation quality (and the canonical SSL diagnostic).
3. **Label-efficiency curves**: finetune on 1%, 10%, 100% of labeled events. "Pretraining helps most when labels are few" is *the* foundation-model claim, and it's the claim the Genesis proposal needs to demonstrate at prototype scale.
4. **Strategy ablation**: random vs geometry vs cluster masking at a couple of ratios (0.15 / 0.4 — MAE showed high ratios help when the pretext is too easy, which P6 suggests it currently is).

---

## Part 4 — Foundation-model integration

The proposal's Phase-I architecture (research §3) is: a **Detector Foundation Model** over geometry-aware point-cloud/graph inputs, pretrained with self-supervised objectives (masked reconstruction, graph completion, contrastive/JEPA), then fine-tuned with parameter-efficient heads for multi-vertex inference, panoptic jet tagging, and timing-aware reconstruction — benchmarked on VBF H→inv and HH→4b. The three codebases now in the repo are prototype fragments of exactly this, and the mapping is worth making explicit to the students:

| Repo component | DFM role |
|---|---|
| Cell graphs + GNN encoder (`train_gnn_models.py`) | Calorimeter branch of the DFM input representation (fixed detector-geometry edges) |
| `CalorimeterMasking` + reconstruction head | Stage-1 SSL objective (masked reconstruction / graph completion) |
| Edge-classification / (future) cluster head | Clustering decoder head |
| Track sets + ISAB/PET encoder (`detr_settx…`) | Tracker branch of the DFM (dynamic k-NN + attention — the DGCNN→transformer pattern named in the proposal) |
| `JetQueriesDecoder` | Prototype of a *panoptic decoder head*; directly relevant to the HH→4b benchmark |

Concrete unification path, in order:

1. **One backbone, many heads — make it literal.** The b-tagging script and the calo script currently have parallel, incompatible model classes. Define a shared encoder interface: tokens = detector elements with (type embedding, coordinates, features); local mixing (fixed calo-neighbor edges / track k-NN EdgeConv) + global mixing (ISAB or windowed attention). The calo `encode()` refactor already points this direction; port the b-tagger to the same interface. This is the proposal's "relation-specific inductive biases → transformer backbone" in miniature.
2. **The masking task becomes modality-agnostic.** `CalorimeterMasking` generalizes verbatim to tracks (mask whole tracks / kinematic features / ΔR regions / all tracks of a truth vertex — "vertex masking" is the tracker analogue of cluster masking). Pretrain each modality's encoder with the same recipe, then jointly. A single `MaskedPretraining` module shared by both scripts removes the current duplication risk.
3. **The student's decoder is the cells→clusters head.** Replace jet-kinematics queries with *learned* queries + an existence head + Hungarian matching to truth topoclusters (mask over cells per query = soft cluster membership) and you have the query-based set-prediction option from the July review — genuine DETR this time, and a literal graph(cells)→graph(clusters) mapping with fractional membership for free. The cross-attention, masking, and attention-map plumbing already written for b-tagging is ~70% of that implementation.
4. **Cross-modality is where the event-level claim lives.** Once cells and tracks share a backbone, the b-tagging decoder can attend to *calorimeter* tokens too, and the clustering head can see tracks — the "unified treatment across sub-detectors" of the proposal. The ΔR-cone baseline (Part 2) then upgrades to the central Phase-I experiment: object-based vs event-level inference on the same data.
5. **Adopt the FM evaluation protocol now** (linear probe, label-efficiency, frozen-vs-finetuned) across *both* downstream tasks. These curves — not single-task metrics — are the evidence the Genesis narrative needs, and building the harness at prototype scale is cheap.

### Priority fix list (across everything)

1. `to_undirected` in the calo message passing (July P-item, still open — now blocks pretraining quality too).
2. Mask-aware ISAB in the b-tagger encoder (M1); select on val AUC (M2).
3. Masking plumbing: event IDs through collate (P1), batch-robust loss with correct per-feature handling (P2), mask-indicator input (P3), pretrain/test separation (P4).
4. Labels: charm class (D1), apply-or-delete the d0/η cuts (D2).
5. Reconstruct dynamic features only, normalized (P6); add val reconstruction loss (P5).
6. Add the two decisive experiments: ΔR-cone baseline for b-tagging; linear-probe + label-efficiency for masking.

---

## Part 5 — First stepping stone: hadronic jet energy calibration

### The submitted abstract

> We are investigating hadronic jet calibration using calorimeter cells, or selected subsets of cells, together with charged-particle tracks in simulated t t̄ events. The approach constructs a shared representation of the detector event from low-level detector information while preserving the calorimeter topology and the associations between detector elements and reconstructed jets. We begin by reconstructing the energies of individual jets using the cells and tracks associated with each jet, and then extend the framework to simultaneously reconstruct the energies of all jets within an event. This event-level approach is intended to exploit both local shower development and the broader detector context while maintaining a unified representation of the calorimeter. We will evaluate the potential of this framework for hadronic jet calibration and provide a foundation for future AI-based calorimeter reconstruction techniques.

### Why this is the right first task for the architecture

Calibration is the ideal proving ground for the Part-4 unified backbone, for four reasons:

1. **It is a regression with an unambiguous truth target and a strong classical baseline** (the ATLAS JES chain: origin correction → MC-based calibration → global sequential calibration → in-situ). Unlike the topoclustering task, there is no label-formulation ambiguity — the deliverable is response and resolution vs (pT, η), directly comparable to published performance.
2. **It exercises every component already in the repo.** Cells + calo topology from `build_graph_dataset.py`; tracks + jet association + truth matching from `make_event_level_btagging.py`; the encoder/decoder from `detr_settx_event_btagging.py` with the classification head swapped for regression; the masking framework as the pretraining stage. Nothing new has to be invented — only unified.
3. **The per-jet → event-level progression is built into the task**, and the marginal value of event context is *measurable*: overlapping showers, out-of-cone leakage, and pileup are exactly the effects a per-jet model cannot see. In t t̄ (busy, ≥4 jets typical, b-enriched) the per-jet vs all-jets comparison is the cleanest small-scale demonstration of the proposal's central claim — global event-level inference beating object-based methods.
4. **Physics stakes are real and legible**: GSC-style track/cell-fraction corrections, punch-through, and flavor dependence (b-jets with semileptonic decays — neutrino energy, muon punch-through) are known pain points of the standard chain, and t t̄ supplies b-jets in bulk. A flavor-aware calibration head also couples naturally to the b-tagging decoder already built (Part 2), pointing at a genuinely panoptic head later.

### Task definition (get this right first — it determines everything downstream)

- **Target.** Per reco jet, regress the **log response** `y = ln(E_true / E_reco)` — equivalently a multiplicative correction — rather than E_true directly. It is dimensionless, roughly Gaussian, and stable across three decades of energy. `E_reco` = constituent-scale (EM-scale) energy of the associated cells, so the model learns the full calibration, not a residual on top of the ATLAS chain (do a second variant on top of calibrated jets later to measure *marginal* gain).
- **Truth matching.** Truth HS jets (ΔR < 0.3, one-to-one greedy match — the code exists), with an isolation requirement in Stage 1 (no second truth jet within ΔR < 1.0) so per-jet training isn't contaminated by overlap; *drop* the isolation cut in Stage 2, where overlap is precisely the signal for event context. The ntuple must add truth jet **pT/E** (currently only η/φ are dumped) — flag this to the student first, it's the long-lead item.
- **Loss.** Start with Huber on `y`; move quickly to **heteroscedastic Gaussian NLL** (predict μ and σ per jet). The σ output is a per-jet resolution estimate — valuable physics output in itself, and the correct weighting when jets of very different quality share a batch.
- **Spectrum bias.** A regressor trained on a steeply falling spectrum pulls low-pT jets up and high-pT jets down (regression to the prior). Reweight training jets to a flat `pT_true` spectrum (or sample uniformly in log pT), and always report **closure**: median response vs `pT_true` *and* vs `pT_reco` after correction. This is the standard failure mode of ML calibrations; building the closure plot into the metrics harness on day one prevents a month of confusion.

### Inputs and the "selected subsets of cells"

Token budget is the practical constraint (all cells = ~187k/event). The abstract's "selected subsets" should be an explicit, ablatable axis:

| Level | Cell subset | ~Tokens/event | Use |
|---|---|---|---|
| S0 | Topoclusters as tokens (no cells) | 10² | Fast baseline; ≈ what ATLAS calibrates today |
| S1 | Cells in topoclusters associated to the jet | 10²–10³/jet | **Stage 1 default** |
| S2 | All cells in topoclusters, event-wide | 10³–10⁴ | **Stage 2 default** |
| S3 | All cells with \|SNR\| > 2 (pre-clustering) | 10⁴ | Tests whether clustering itself discards calibration-relevant information — a headline question for the proposal |
| S4 | All cells | 1.9×10⁵ | Only after S3, with windowed/inducing attention |

Cell features: energy (log-scaled), η, sin φ, cos φ, layer/subcalo one-hots, cell size (`CELL_SIZES`), sampling depth — all already produced by `build_graph_dataset.py`. Track features: the 25 already dumped (IP variables matter for the flavor-dependent response). Associations: cells→cluster→jet (constituent linking — needs the cluster→jet index added to the dump), tracks→jet by ghost association if available, else ΔR < 0.4 + PV requirement. **Preserving calo topology** = keep the fixed neighbor graph within the cell subset for the local-mixing layers (and fix `to_undirected` before any of this — Part 3, P7).

### Stages

**Stage 0 — infrastructure (blocks everything).** Extend the ntuple dump + `make_event_level_*` builder: truth jet 4-vectors, cluster→jet association, per-jet cell lists, per-event cell subsets. Build the metrics harness *first*: response histograms per (pT_true, η) bin; median response + closure; resolution as **IQR/2 ÷ median** (robust — never fit naive Gaussians to response tails); flavor-split (b / c / light — reuse the labels, with the charm fix from D1). Baselines to beat, in order: raw EM-scale sum; S0 cluster-sum + simple MLP on engineered GSC-like features (track fraction, layer fractions, n_trk); the ATLAS-calibrated jet pT if it's in the ntuple.

**Stage 1 — per-jet regression (cells + tracks of one jet).** One jet = one sample: tokens are its S1 cells + associated tracks, type-embedded into the shared encoder (EdgeConv over calo-neighbor/k-NN edges → ISAB), masked mean pool → (μ, σ) head. This is a DeepSets/PET-scale model — fast to train, easy to interrogate. Deliverables: response/resolution vs pT and η against the Stage-0 baselines; ablations = cells-only vs tracks-only vs both (the "does tracking information help and where" plot ≈ learned GSC), and S1 vs S0 (do cells beat clusters?).

**Stage 2 — event-level, all jets simultaneously.** The Part-2/Part-4 machinery, literally: encode *all* S2/S3 cells + all tracks once; jet queries built from reco-jet kinematics cross-attend to the full event (the student's `JetQueriesDecoder` with the head swapped to (μ, σ) and — new — **self-attention among the jet queries**, so jets negotiate shared energy). Key experiments:
   - Event-level vs Stage-1 per-jet on the *same* jets — overall, then sliced by ΔR to nearest jet (the gain should concentrate at small ΔR — that plot *is* the abstract's thesis).
   - Energy-sum consistency: does Σ corrected jets + soft term behave better? (Free look at a MET-adjacent observable.)
   - S3 vs S2: is there calibration signal in below-threshold / out-of-cluster cells?

**Stage 3 — pretraining hook.** Once Part-3 fixes land, pretrain the cell/track encoders with the masking task on unlabeled events (t t̄ + whatever else exists), finetune the calibration heads; report linear-probe and label-efficiency curves (1% / 10% / 100%). Calibration is a *better* downstream probe of the masking pretext than edge classification — the target is continuous and the encoder is shared verbatim.

**Stage 4 — talk/write-up.** The result package for the abstract: (i) per-jet model vs classical baselines, (ii) event-level vs per-jet with the ΔR-sliced gain, (iii) cell-subset ablation (S0→S3), (iv) flavor dependence, (v) if ready, pretraining label-efficiency. That is a complete, honest "first stepping stone" story even if only (i)–(iii) land in time.

### Suggested milestone order (each is a PR-sized unit)

| # | Milestone | Depends on |
|---|---|---|
| 1 | Ntuple: truth jet 4-vectors + cluster→jet index | — |
| 2 | Metrics harness (closure, IQR resolution, flavor split) with EM-sum + S0-MLP baselines | 1 |
| 3 | `to_undirected` fix + shared token-encoder interface (Part 4, step 1) | — |
| 4 | Stage-1 per-jet regressor + ablations | 1–3 |
| 5 | Stage-2 event-level decoder (query self-attention, (μ,σ) head) | 4 |
| 6 | Masking fixes (P1–P6) → Stage-3 pretraining study | 3, 4 |
| 7 | Multi-task probe: calibration + b-tagging heads on one backbone | 5 |

### Risks to watch

- **Pileup handling**: if the dump has no truth pileup split, the track PV cut is the only handle; state clearly what µ scenario the sample is. Event-level gains can otherwise be pileup-modeling artifacts.
- **b-jets with neutrinos**: response is intrinsically bimodal (semileptonic decays) — a single Gaussian μ/σ will split the difference. The heteroscedastic head hides this; check b-jet response *distributions*, not just medians. (A small mixture-density head is the upgrade if it bites.)
- **Don't let Stage 2 memorize the spectrum**: the event-level model sees all jets and can exploit event-level pT balance in t t̄ (W mass, top mass constraints). That's *physics context, and legitimate* — but say so explicitly, and quantify it (e.g., evaluate on a dijet or Z+jets sample later to test transfer). Sample-dependence of the calibration is the referee question.
