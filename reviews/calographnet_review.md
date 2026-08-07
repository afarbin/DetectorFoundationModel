# CaloGraphNet — Code & Approach Review, and a Path to Graph→Graph (Cells→Clusters)

*Review of MoGhaznavi/CaloGraphNet (build_graph_dataset.py, train_gnn_models.py, analyze_results.py, README) — July 2026*

## TL;DR

The pipeline is solid engineering, and the README's self-diagnosis is pointing in the right direction — but three things are capping performance before you ever get to the formulation question:

1. **A real bug: message passing is one-directional.** Edges are canonicalized to `(low_idx → high_idx)` and never symmetrized, so every cell only receives messages from its lower-index neighbors. This alone plausibly costs a large fraction of your FSS gap.
2. **The 5-class edge scheme fights the physics.** Classes 2/3 encode an arbitrary index ordering, and class 4 requires non-local information (cluster splitting) that a 6–8-hop GNN cannot see. For clustering you only need P(same cluster).
3. **You're measuring a proxy, not the objective.** Edge F1 ≠ clustering quality. Nobody ever runs connected components on the predictions and compares reconstructed clusters to the target topoclusters.

For the reformulation you want — cells graph in, clusters out — the field has converged on two families that fit exactly: **object condensation** (GravNet/CMS-HGCAL lineage) and **hypergraph incidence prediction** (HGPflow lineage, which natively handles the fractional/overlapping-cluster issue the README worries about). Details and a suggested roadmap below.

---

## Part 1 — Why the current formulation caps out

### 1.1 Topoclustering is seeded region-growing → it's a *reachability* problem

ATLAS 4-2-0 topoclustering ([ATLAS, arXiv:1603.02934](https://arxiv.org/abs/1603.02934)) is: seed at |ζ|>4, grow through |ζ|>2, absorb perimeter at |ζ|>0, then split at local maxima. Whether two *adjacent* cells land in the same cluster is mostly **not** a local question — it depends on whether both are reachable from a common seed through a >2σ path, which can be tens of hops away. A fixed-depth message-passing network is structurally bad at long-range reachability: with 6–8 layers, a cell 15 growth-steps from its seed literally cannot know the seed exists. This is consistent with your observations that (a) deeper SAGE won, (b) architectures all plateau together, and (c) class 1 vs class 4 confusion dominates — the 1-vs-4 distinction is set by the *splitting* step (local maxima elsewhere in the cluster), the most non-local part of the algorithm.

**Cheap, decisive diagnostic:** you have everything needed to re-implement 4-2-0 in ~50 lines of NumPy on the stored `snr_computed` + `pairs` arrays. Run it and compare to `cell_cluster_index`:
- If plain 4-2-0 (no splitting) reproduces the labels up to splitting, your labels are clean and the failure is purely architectural/formulational → the fixes below should move you a lot.
- If it doesn't, your labels contain things the input features can't determine (e.g., splitting decisions, out-of-time pileup in the noise, cells outside your `CELL_SIZES` map) and *no* model on these inputs can reach FSS 5 — you'd finally know your ceiling.
This one script is the highest information-per-hour experiment available to you.

### 1.2 The 5-class scheme wastes capacity and encodes an artifact

- **Classes 2 vs 3 are the same physics.** Edges are canonicalized `src = min, dst = max` in `build_graph_dataset.py`, so "source-in-cluster" vs "dest-in-cluster" distinguishes only which endpoint happens to have the lower global index. The model must burn capacity learning an indexing artifact, and the edge readout `concat[h_src, h_dst]` is order-dependent for the same reason.
- **For clustering, the target is binary.** Downstream you'd run connected components on "same cluster" edges; classes 0/2/3/4 are all just "cut." Recommended: predict **P(same cluster)** per edge (optionally + a per-*node* auxiliary head "is this cell in any cluster," which recovers everything classes 2/3 were encoding, without the artifact).
- **CC is asymmetrically fragile.** One false-positive "same-cluster" edge merges two clusters; one false negative usually does nothing (other paths reconnect). Argmax over 5 classes is the wrong operating point — you want a tuned threshold on P(same), selected on a *clustering* metric, not edge F1. This is also why edge-level FSS can improve while clustering gets worse.

### 1.3 Missing end-to-end evaluation

`analyze_results.py` has rich edge-level diagnostics, but the actual deliverable — do reconstructed clusters match CaloTopoClusters? — is never computed. Add per-event: union-find on predicted same-cluster edges → then

- pair-counting metrics restricted to in-cluster cells (Rand/ARI),
- **energy-weighted** efficiency & purity per matched cluster (match by shared energy fraction),
- cluster multiplicity and cluster-energy residuals vs truth.

Energy weighting matters: getting a 3-cell 200 MeV cluster wrong ≠ getting the core of a 50 GeV cluster wrong, and edge F1 treats them identically.

### 1.4 Label leakage in all-features mode

`in_cluster` and `cluster_id_norm` are derived from `cell_cluster_index` — **the label** (edge labels are a deterministic function of exactly this array). Their presence in the 42-feature set makes those runs invalid as science (and the fact that the model *still* did worse with them suggests optimization/normalization problems in that mode, e.g. unscaled heterogeneous features). Remove both.

---

## Part 2 — Concrete code issues

Ordered by expected impact.

### Bugs

1. **Directed message passing** (`train_gnn_models.py`, `MultiClassBatchGenerator`): `pairs_t = self.neighbor_pairs.T` is passed straight to GCN/GAT/SAGE/TransformerConv. PyG convs do **not** symmetrize `edge_index`; messages flow only low-index → high-index. The `is_bi_directional=True` kwarg is accepted and **never used** (grep confirms: 3 hits, all parameter passing). Fix:
   ```python
   from torch_geometric.utils import to_undirected
   self.pairs_mp = to_undirected(self.neighbor_pairs.T.contiguous())  # for message passing
   # keep the canonical (E,2) list separately for prediction/labels
   ```
   Keep prediction edges canonical so labels still align. I would re-run the SAGE-8/focal baseline with only this change before touching anything else — it's a clean A/B and I'd expect a real jump.

2. **Mixed precision is a no-op** — `GradScaler` is created and `scaler.scale(loss).backward()` is called, but there is no `torch.autocast` context anywhere, so the forward/backward run in FP32. Harmless numerically, but you're paying scaler overhead for nothing and the "FP16" log line is wrong. Either wrap the forward+loss in `with torch.autocast('cuda', dtype=torch.float16):` or drop the scaler.

3. **`analyze_results.py` cannot run**: line 1 of the file is a literal markdown fence (```` ```python ````) — SyntaxError on import. Presumably a copy-paste artifact; worth fixing in the repo since it means results plots weren't produced by *this* file as committed.

4. **Class weights computed on train+test**: `create_loss_function(args, labels, ...)` receives the full label array before splitting. Mild prior leakage; compute on train indices only.

### Modeling/feature issues

5. **φ periodicity**: raw φ as a feature has a discontinuity at ±π; two adjacent cells across the seam look maximally far apart. Use `sin φ, cos φ` (and wrapped Δφ if used as an edge feature).
6. **No edge features at all.** GCN/SAGE can't consume them, but TransformerConv and GATv2 accept `edge_dim`. Even three edge features — wrapped Δη, Δφ, same-sampling-layer flag — give the network the local geometry that 4-2-0 implicitly uses. Cross-layer neighbors (your `neighbor` branch includes them) behave very differently from in-layer neighbors; currently the model can't tell them apart.
7. **SNR scaling**: `snr_computed` is fed raw; it's heavy-tailed (seeds can be ≫100). Consider `sign(ζ)·log(1+|ζ|)`, and — since 4-2-0 is literally a function of threshold crossings — add the three indicator features `[|ζ|>4, |ζ|>2, |ζ|>0]`. You're handing the model the algorithm's own primitives; it's the cheapest possible inductive bias.
8. **Loss is dominated by trivial edges** (~92% noise–noise, both cells |ζ|<2). Focal loss mitigates but a blunter tool works better: drop or heavily downweight edges where both endpoints have |ζ|<1.5 *in the loss only* (keep them in message passing). Faster epochs, and capacity focuses on the boundary/splitting regions where you actually fail.
9. **No LR schedule, dropout=0.0 default, fixed Adam 1e-3** — with the other fixes in place, add cosine decay + warmup before drawing architecture conclusions.
10. **Scaling note**: `features_dict` holds every event in RAM and each event is the full ~180k-cell graph. Fine now; will not survive a 10× dataset. The chunked generator is already halfway to an HDF5-backed lazy loader.

### Minor

- `get_cell_volume` = deta·dphi·1000 is not a volume (no radial extent, no η-dependence of cell size for Tile); fine as a feature, but don't let anyone use it downstream as geometry.
- Every-epoch full checkpoints accumulate; keep last-k.
- `compute_class_weights` result is discarded on the focal path (hardcoded alphas take over) — confusing but not wrong.
- 70/30 split is on *sorted event index* with multi-file datasets concatenated in filename order — fine if files are one homogeneous sample; revisit if you mix physics samples across files.

---

## Part 3 — The graph→graph reformulation (cells → clusters)

This is the right instinct, and there is an established menu. Ordered by (my judgment of) fit-to-your-problem × implementation cost:

### A. Object condensation — recommended first move
[Kieseler, arXiv:2002.03605](https://arxiv.org/abs/2002.03605); built on GravNet layers ([Qasim et al., arXiv:1902.07987](https://arxiv.org/abs/1902.07987)); this is the production approach for CMS HGCAL clustering, i.e. exactly this problem at higher granularity.

- Each cell predicts a **condensation strength β** and coordinates **x** in a learned 2–3D clustering space. The loss attracts cells to the highest-β cell of their own (truth) cluster and repels them from others; a β-loss makes one representative per cluster emerge.
- Inference: pick high-β condensation points, assign cells by distance in the learned space. **Variable, unbounded number of clusters for free** — no fixed-K, no Hungarian matching.
- Your existing backbone is reusable: same node encoder + message passing, swap the edge head for (β, x) heads and the OC loss (~200 lines). You can even keep the geometric neighbor graph initially, then replace conv layers with GravNet (dynamic kNN in feature space) — which *is* the "learn the graph" idea from your README, in its proven form.
- Cluster properties (energy, position) come from β-weighted aggregation over assigned cells → the output really is a graph of cluster-nodes with regressed attributes, which is where your foundation-model program wants to go (calibrated clusters, not just memberships).

### B. Hypergraph incidence prediction — the right home for fractional membership
[Di Bello et al., arXiv:2212.01328](https://arxiv.org/abs/2212.01328) (set transformer + hypergraph prediction for particles-in-jets) and the collider-event extension **HGPflow** [arXiv:2410.23236](https://arxiv.org/abs/2410.23236).

- Predict an **incidence matrix** cells × clusters with *fractional* weights, trained with an energy-fraction target. This is literally the README's "Direction 3+4," already built and published.
- Key physics point: ATLAS cluster splitting already shares cells between (up to two) clusters with geometric weights — so your current `cell_cluster_index` scalar is a **lossy projection of the truth**. If Mo can extend the ROOT dump to write per-cell *cluster weights* (they exist in the CaloCluster/CellLink objects), you get proper soft targets and the class-1-vs-4 ambiguity largely dissolves.
- Costlier to implement than OC (iterative refinement, matching), but the natural endpoint if overlapping showers are the physics you care about.

### C. Query-based set prediction (MaskFormer/DETR-style)
K learnable cluster queries cross-attend over cell embeddings; each query emits an existence probability + a soft mask over cells (+ regressed cluster properties); Hungarian matching to truth clusters. Elegant, and the query–query self-attention layer literally *is* a learned cluster-graph. Downsides at your scale: attention over ~180k cells needs windowing/sampling, and fixed-K + matching adds machinery. I'd hold this in reserve unless A/B disappoint.

### D. Cheap two-stage hybrid (lowest risk, uses what you have)
Binary edge model (fixed, from Part 2) → threshold + connected components → build a **cluster graph** (nodes = candidate clusters, features = aggregated cell stats, edges = adjacency between candidates) → a second small GNN refines: merge/split decisions or property regression. This is literally graph(cells)→graph(clusters) as two supervised stages, debuggable independently, and stage 2 is where splitting (the non-local part) gets the global view it needs.

### Training strategy for any of the above

- **Curriculum**: first train against *unsplit* 4-2-0 connected-components labels (deterministically recomputable from your own SNR arrays — same script as the diagnostic in §1.1); then fine-tune on the real split clusters. Separates the learnable-locally part from the global part.
- **Evaluation** (all approaches): the end-to-end clustering metrics from §1.3, energy-weighted, reported vs. cluster energy and η — never edge F1 as the headline number.

---

## Part 4 — Suggested order of operations

| Step | What | Cost | What you learn |
|---|---|---|---|
| 1 | NumPy 4-2-0 on stored SNR + pairs; compare to labels | hours | Your achievable ceiling; label cleanliness |
| 2 | `to_undirected` fix; rerun SAGE-8 focal A/B | hours | How much the bug cost |
| 3 | Binary P(same) + node "in-cluster" aux head; symmetric readout; sin/cos φ; SNR transform + threshold indicators; loss mask on dead edges | days | Best fair edge-baseline |
| 4 | CC + energy-weighted purity/efficiency/ARI eval; threshold scan | days | The number that actually matters, for every model incl. steps 2–3 |
| 5 | Object condensation head on the same backbone | 1–2 weeks | The graph→graph result; variable #clusters |
| 6 | GravNet layers replacing fixed-graph convs | days on top of 5 | "Learned graph" hypothesis, tested |
| 7 | Export per-cell cluster *weights* from ROOT; soft-target OC or HGPflow-style incidence | longer | Fractional membership done right |

Steps 1–4 are worth doing even though the formulation is changing: they give you an honest baseline and the evaluation harness that every later model will be judged by.

---

## References

- ATLAS topological cell clustering — [arXiv:1603.02934](https://arxiv.org/abs/1603.02934)
- Object condensation — Kieseler, [arXiv:2002.03605](https://arxiv.org/abs/2002.03605)
- GravNet — Qasim, Kieseler, Iiyama, Pierini, [arXiv:1902.07987](https://arxiv.org/abs/1902.07987)
- Hypergraph particle reconstruction — Di Bello et al., [arXiv:2212.01328](https://arxiv.org/abs/2212.01328)
- HGPflow (collider events) — [arXiv:2410.23236](https://arxiv.org/abs/2410.23236)
- MLPF (per-node particle flow, useful contrast) — Pata et al., [arXiv:2101.08578](https://arxiv.org/abs/2101.08578)
