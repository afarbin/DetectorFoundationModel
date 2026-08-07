# dfm — Detector Foundation Model (merged backbone)

The unification of the two model lines in this repository, implementing
"Part 4: Foundation-model integration" of
[`reviews/btagging_masking_review.md`](../reviews/btagging_masking_review.md):
**one backbone, many heads**.

| Component | Origin | What changed in the merge |
|---|---|---|
| `SharedEventEncoder` | new | Typed tokens (cells, tracks, ...) with per-modality input projections + type embeddings; local mixing per modality; shared mask-aware ISAB global mixing, optionally cross-modality |
| `NeighborConv` (cell local mixing) | `foundation_model/train_gnn_models.py` `encode()` | SAGE-style over the fixed calo-neighbor graph, pure torch (PyG-compatible `lin_l`/`lin_r` naming for later checkpoint transfer); edges mirrored internally — the July-review directed-edge bug cannot recur |
| `MaskedEdgeConv` + `MaskedISAB` (track path) | `btagging/detr_settx_event_btagging.py` | All attention is padding-mask aware (**fixes M1**); k-NN guards small events (**M4**); per-head attention scaling |
| `JetQueryDecoder` | `btagging` `JetQueriesDecoder` | Cross-attends to *any* token set (tracks, cells, or both — Part 4 step 4); jet-query self-attention (**M9**); modes: binary b-tag, 3-class b/c/light (**D1**), heteroscedastic (μ, log σ²) for jet calibration (Part 5) |
| `EdgeClassifierHead` | `foundation_model` edge head | 5-class `concat` readout (label-compatible) or symmetric readout + binary P(same-cluster) (July review §1.2) |
| `MaskedTokenPretraining` | `foundation_model` `CalorimeterMasking` | Modality-agnostic (cluster masking ⇔ vertex masking); group ids threaded explicitly (**P1**), batch-safe standardized loss (**P2**), learnable mask token + `is_masked` indicator channel (**P3**), dynamic-features-only targets (**P6**) |
| `CaloGraphAdapter` | reads Mohammad Ali's processed datasets | Baseline-7 feature schema reproduced exactly; S3 cell subset (\|SNR\| > 2) with reindexed neighbor subgraph |
| `TrackJetAdapter` | reads 5D ntuples directly | Adapted to branches present in the server ntuples (btagIp_* absent → IP significances computed from covariances); labels from `HadronConeExclTruthLabelID` (**D3**), charm kept as its own class (**D1**); normalization computed on caller-chosen (train) events (**M6**) |

## Quick start

```python
from dfm import SharedEventEncoder, ModalityConfig, JetQueryDecoder
from dfm.data import CaloGraphAdapter, TrackJetAdapter

calo = CaloGraphAdapter("/storage/mxg1065/processed_data/ttbar_1000")
cells = calo.event_tokens(range(8))                    # S3 subset + subgraph
tracks_ad = TrackJetAdapter(".../user.bbullard...ntuple.root")
tracks, jets = tracks_ad.event_tokens(range(8))

enc = SharedEventEncoder([
    ModalityConfig("cell", 7, local="edges", local_depth=2),
    ModalityConfig("track", 18, local="knn", local_depth=2),
], dim=128, global_depth=2)

out = enc([cells, tracks])          # cross-modality attention
head = JetQueryDecoder(128, mode="multiclass")   # b/c/light
logits, attn = head(jets["queries"], jets["mask"],
                    out["combined"].features, out["combined"].mask)
```

Smoke test (server): `python -m dfm.smoke_test --calo-dir ... --fived-file ...`

## Design notes (post-review)

- **Pretrain/finetune weight compatibility**: modalities declare
  `mask_indicator=True`; the encoder then accepts both clean (F-channel) and
  masked (F+1-channel) batches with one set of weights — pretrained encoders
  load verbatim into finetuning (demonstrated in the smoke test).
- **S3 subsetting changes the edge task**: restricting cells to \|SNR\| > 2
  collapses label classes 2/3 (in-cluster↔lone edges lose their lone
  endpoint; observed counts ~[6.7k, 16.9k, 4, 2, 1.8k]). Subset-trained
  5-class heads are *not* comparable to the full-graph task — use the
  binary/symmetric readout on subsets, or full-graph data for the 5-class
  reproduction.
- **Jet labels**: `HadronConeExclTruthLabelID` ∈ {0, 4, 5} map to
  light/c/b; anything else (15 = tau, ...) gets label −1 and is excluded by
  the masked losses rather than contaminating "light".
- **Vertex masking needs `vtx_filter=None`**: the default b-tagging track
  selection keeps only PV/unassociated tracks (one vertex group), so the
  adapter emits no group ids and group masking raises (per P1) instead of
  silently masking the whole primary vertex.
- **Normalization is explicit**: `TrackJetAdapter.event_tokens` raises until
  `compute_normalization(train_indices)` (or `load_normalization`) is called
  — no silent stats from whatever batch arrives first (M6).

## Data prerequisites & known gaps

- **Cross-modality training needs both modalities for the same events.**
  Today the calo cell datasets (Calo ntuples) and the track/jet data
  (5D ntuples) come from *different* event samples; the smoke test mixes
  them only to exercise the mechanics. The unblock is either an ntuplizer
  run that dumps cells *and* tracks/jets for the same events, or using the
  5D ntuples' own (thresholded, biased — see
  `analysis/Cell-Selection-Comparison.ipynb`) cell subset for both.
- **Calibration head (Part 5)** needs truth-jet 4-vectors in the ntuple
  (currently only η/φ are dumped) — the review's "long-lead item".
- The DETR-style cells→clusters set-prediction head (learned queries +
  Hungarian matching, Part 4 step 3) is future work; `JetQueryDecoder`
  contains the cross-attention plumbing it will reuse.
