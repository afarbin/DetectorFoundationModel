# CaloGraphNet

> **Graph Neural Networks for Calorimeter Cell Clustering in Particle Physics**
> 
> Processes ATLAS ROOT files into graph datasets and trains GNNs (GCN, GAT, Graph Transformer, SAGE) for edge classification with optional masked pretraining.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Table of Contents

### Core Documentation
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Pipeline Architecture](#pipeline-architecture)

### Data Processing
- [File 1: build_graph_dataset.py](#file-1-build_graph_datasetpy)
  - [What it does](#what-it-does)
  - [The 4 Output Files](#the-4-output-files)
  - [Geometry Handling](#geometry-handling)
  - [Usage](#usage-build_graph_datasetpy)

### Model Training
- [File 2: train_gnn_models.py](#file-2-train_gnn_modelspy)
  - [What it does](#what-it-does-1)
  - [Feature Sets](#feature-sets)
  - [Model Architectures](#model-architectures)
  - [Loss Functions](#loss-functions)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Output Files](#output-files)
  - [Usage](#usage-train_gnn_modelspy)
  - [Masked Pretraining (NEW!)](#masked-pretraining-new)
    - [What is Masked Pretraining?](#what-is-masked-pretraining)
    - [Masking Strategies](#masking-strategies)
    - [How Pretraining Works](#how-pretraining-works)
    - [Pretraining Usage Examples](#pretraining-usage-examples)

### Analysis & Visualization
- [File 3: analyze_results.py](#file-3-analyze_resultspy)
  - [What it does](#what-it-does-2)
  - [Visualization Outputs](#visualization-outputs)
  - [Understanding the Metrics](#understanding-the-metrics)
  - [Pretraining Analysis](#pretraining-analysis-new)
  - [Complete Analysis Workflow](#complete-analysis-workflow)
  - [Interpreting Results Quick Guide](#interpreting-results-quick-guide)

### Getting Started
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Complete Example](#complete-example-from-root-to-analysis-report)
  - [Pretraining Example](#pretraining-example)
  - [Expected Outputs](#expected-outputs)
  - [Analyze Results with Python](#analyze-results-with-python)

### Reference
- [Configuration Reference](#configuration-reference)
  - [build_graph_dataset.py Arguments](#build_graph_datasetpy-full-arguments)
  - [train_gnn_models.py Arguments](#train_gnn_modelspy-full-arguments)

### Lessons Learned & Future Directions
- [Current Limitations & Lessons Learned](#current-limitations--lessons-learned)
  - [The Challenge](#the-challenge-why-performance-isnt-where-we-want-it)
  - [Our Best Results So Far](#our-best-results-so-far)
  - [Lesson 1: Accuracy is Misleading](#lesson-1-accuracy-is-dangerously-misleading-for-imbalanced-data)
  - [Lesson 2: Focal Loss Helps](#lesson-2-focal-loss-helps-but-doesnt-solve-the-core-issue)
  - [Lesson 3: More Features Can Hurt](#lesson-3-more-features-dont-always-help-surprising-finding)
  - [Lesson 4: Architecture Differences Are Small](#lesson-4-architecture-differences-are-smaller-than-expected)
  - [The Core Issue: Graph Construction](#lesson-5-the-core-issue-may-be-graph-construction-itself)
- [The Graph Learning Hypothesis](#the-graph-learning-hypothesis)
- [Future Directions](#future-directions)
- [Practical Recommendations](#practical-recommendations-for-users-of-this-code)
- [What 2.75 FSS Actually Means](#what-275-fss-actually-means)
- [Summary of Experiments](#summary-table-of-what-weve-tried)
- [Open Questions](#open-questions-for-the-community)
- [Conclusion](#conclusion)

---

## Overview

CaloGraphNet is a complete pipeline for graph-based machine learning on calorimeter cell data from particle physics detectors (e.g., ATLAS). It:

1. **Converts ROOT files** → Graph-structured datasets (HDF5 + NumPy)
2. **Trains Graph Neural Networks** for edge classification (with optional masked pretraining)
3. **Supports multiple architectures**: GCN, GAT, Graph Transformer, GraphSAGE
4. **Supports self-supervised pretraining**: BERT/MAE-style masked reconstruction

---

## Problem Statement

In a calorimeter, particles deposit energy across multiple cells. The goal is to group cells belonging to the same physical particle (clustering). We formulate this as an **edge classification problem** on a graph where:

- **Nodes** = Calorimeter cells
- **Edges** = Geometric neighbor connections between cells

For each edge (cell pair), the model predicts one of **5 classes**:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | Noise-Noise | Both cells are noise (no particle cluster) |
| 1 | Same Cluster | Both cells belong to the same particle cluster |
| 2 | Source Only | Source cell in cluster, destination is noise |
| 3 | Destination Only | Destination in cluster, source is noise |
| 4 | Different Clusters | Cells belong to different particle clusters |

**Why edge classification?** Once we predict which cell pairs belong to the same cluster, we can use graph clustering algorithms (e.g., connected components) to reconstruct full particle clusters.

---

## Pipeline Architecture

```
ROOT File (produced from RDO format)
       ↓
╔══════════════════════════════════════════════════════════════════╗
║              build_graph_dataset.py                              ║
║  - Reads cell-level branches (energy, noise, eta, phi, etc.)     ║
║  - Builds graph connectivity from neighbor information           ║
║  - Defines geometry via CELL_SIZES mapping                       ║
║  - Saves 4 output files                                          ║
╚══════════════════════════════════════════════════════════════════╝
       ↓
  HDF5 + NumPy Dataset (4 files)
       ↓
╔══════════════════════════════════════════════════════════════════╗
║              train_gnn_models.py                                 ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  OPTIONAL: Masked Pretraining (--pretrain)              │    ║
║  │  - Mask random cells/features/clusters/regions          │    ║
║  │  - Train encoder to reconstruct masked values           │    ║
║  │  - Learn general calorimeter representations            │    ║
║  │  - Transfer encoder to downstream task                  │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                          ↓                                       ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  Supervised Finetuning (or Training from Scratch)       │    ║
║  │  - Edge classification with 5 classes                   │    ║
║  │  - Weighted/focal loss for class imbalance              │    ║
║  │  - Comprehensive metrics (F1 Sum Score)                 │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
       ↓
  Trained Model + Predictions + Metrics
```

---

## File 1: build_graph_dataset.py

### What it does

This script takes a ROOT file containing calorimeter cell data and converts it into a graph-structured dataset suitable for GNN training. It processes the data in chunks to handle large files efficiently.

**Key operations:**
1. Reads cell-level branches from the ROOT file (energy, noise, eta, phi, cluster indices, neighbor connectivity)
2. Filters out invalid cells (e.g., where noise = 0)
3. Builds a graph by creating undirected edges between neighboring cells (using the `neighbor` branch)
4. Computes static cell metadata (subcalorimeter type, sampling layer, cell dimensions, volume)
5. Computes per-cell noise statistics (mean, std, category) across all events
6. Fits an energy scaler (robust or standard normalization) for consistent feature scaling
7. Processes each event to extract energy/SNR values and cluster assignments
8. Assigns edge labels based on cluster membership (5-class scheme)
9. Saves everything to 4 structured output files

### The 4 Output Files

All outputs are saved in: `output_dir/{dataset_name}/`

#### 1. `pairs_{dataset_name}.npy`
- **Format**: NumPy array, shape `(num_edges, 2)`, dtype `int32`
- **What it contains**: The graph connectivity - each row is `[source_cell_idx, target_cell_idx]`
- **How it's built**: From the `neighbor` branch in the ROOT file, which lists neighboring cells for each cell. The script:
  - Maps original cell indices to global indices (after filtering invalid cells)
  - Creates undirected edges (ensures src < dst)
  - Removes duplicate edges
  - Orders edges by source cell for efficient lookup
- **Size**: In the case of the ATLAS calorimeter, around 1.2 million pairs

#### 2. `cells_{dataset_name}.npy`
- **Format**: Structured NumPy array, shape `(num_cells,)` with named fields
- **What it contains**: Static metadata for each cell (does NOT change per event):

| Field | Type | Description |
|-------|------|-------------|
| `global_idx` | int32 | Index in this dataset (0 to num_cells-1) |
| `orig_idx` | int32 | Original index from ROOT file |
| `eta_event0` | float64 | Eta coordinate from first event (used as static geometry) |
| `phi_event0` | float64 | Phi coordinate from first event |
| `broken_event0` | bool | Whether cell had noise=0 in event 0 |
| `num_neighbors` | int32 | Number of neighboring cells (degree) |
| `subcalo` | int32 | Sub-calorimeter ID (0=EMB, 1=EMEC, 2=HEC, 3=Tile) |
| `sampling` | int32 | Sampling layer index |
| `deta` | float32 | Cell size in eta (radians) - from CELL_SIZES mapping |
| `dphi` | float32 | Cell size in phi (radians) - from CELL_SIZES mapping |
| `volume` | float32 | Cell volume (deta × dphi × R, where R=1000mm) |
| `noise_mean` | float32 | Mean noise across all events for this cell |
| `noise_std` | float32 | Standard deviation of noise across all events |
| `noise_count` | int32 | Number of events this cell had valid noise |
| `noise_category` | int8 | Categorization: 0=low noise, 1=medium, 2=high, -1=invalid |

- **Why it's useful**: Provides static features that don't change per event, saving disk space and allowing reuse across multiple training runs.

#### 3. `events_{dataset_name}.h5`
- **Format**: HDF5 file with multiple datasets, shape `(num_events, num_cells)` for each dataset
- **What it contains**: Per-event, per-cell features (dynamic data):

| Dataset | Description |
|---------|-------------|
| `cell/energy_raw` | Raw energy deposits (if available) |
| `cell/noise_raw` | Raw noise values (if available) |
| `cell/snr_computed` | Signal-to-noise = energy / noise (computed) |
| `cell/snr_raw` | Direct SNR from ROOT (if available) |
| `cell/energy_normalized` | Normalized energy: `(energy - center) / scale` |
| `cell/cell_cluster_index` | Cluster ID for each cell (0=noise, ≥1=valid cluster) |

- **HDF5 attributes** (metadata stored in file):
  - `dataset_name`, `input_file`, `num_events`, `num_cells`, `num_pairs`

#### 4. `labels_{dataset_name}.npy`
- **Format**: NumPy array, shape `(num_events, num_edges)`, dtype `int8`
- **What it contains**: Ground truth labels for each edge in each event
- **Label assignment logic** (critical for correct training):

```python
if c0 == 0 and c1 == 0:      # Both noise
    label = 0
elif c0 == c1 and c0 >= 1:   # Same valid cluster
    label = 1
elif c0 >= 1 and c1 == 0:    # Source in cluster, destination noise
    label = 2
elif c0 == 0 and c1 >= 1:    # Source noise, destination in cluster
    label = 3
elif c0 != c1 and c0 >= 1 and c1 >= 1:  # Different clusters
    label = 4
```

### Geometry Handling

**Where geometry is defined** (lines ~169-185):

```python
CELL_SIZES = {
    '0_0': (0.025, 0.1),     # (Δη, Δφ) for subcalo 0, sampling 0
    '0_1': (0.0031, 0.0245), # Electromagnetic barrel, fine sampling
    '0_2': (0.025, 0.0245),
    '0_3': (0.05, 0.0245),
    '0_4': (0.025, 0.1),
    '1_8': (0.1, 0.09817481), # EMEC (endcap)
    '2_21': (0.1, 0.1),       # HEC (hadronic)
    '3_12': (0.1, 0.09817481), # Tile (hadronic barrel)
    # ... etc.
}
```

This mapping connects `(subCalo, sampling)` pairs to physical cell dimensions in η-φ space. The mapping was derived from the ATLAS detector geometry.

**Geometry features computed per cell:**
- `eta`, `phi`: Cell center coordinates from ROOT branches
- `deta`, `dphi`: Cell dimensions from CELL_SIZES mapping
- `volume`: `deta × dphi × 1000 mm` (approximate)
- `subcalo`, `sampling`: One-hot encoded for model input

### Usage: build_graph_dataset.py

```bash
python build_graph_dataset.py \
    --input /path/to/file.root \
    --output-dir /storage/processed_data \
    --dataset-name my_dataset \
    --normalize-energy \
    --energy-normalization robust \
    --chunk-size 50 
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | Required | Input ROOT file path |
| `--output-dir`, `-o` | `/storage/mxg1065/processed_data` | Base output directory |
| `--dataset-name`, `-n` | Auto from filename | Custom dataset name |
| `--normalize-energy` | True | Apply energy normalization |
| `--energy-normalization` | `robust` | `robust` (median/IQR) or `standard` |
| `--skip-snr-scaling` | False | Don't scale SNR features |
| `--chunk-size` | 50 | Events per chunk (memory control) |
| `--debug` | True | Enable debug output |

---

## File 2: train_gnn_models.py

### What it does

This script loads the dataset created by `build_graph_dataset.py`, trains Graph Neural Networks for edge classification, and saves comprehensive results. It now supports **masked pretraining** followed by supervised finetuning.

**Key operations:**
1. Loads features with user selection (baseline 3-feature or all 42+ features)
2. Splits events into train/test sets (default 70/30)
3. Using custom generator to read the data and construct the batches
4. Optionally runs **masked pretraining** to learn general representations
5. Builds a GNN model (GCN, GAT, Graph Transformer, or GraphSAGE)
6. Sets up loss function (standard, weighted, or focal loss for class imbalance)
7. Trains with mixed precision (FP16) for faster GPU training
8. Evaluates using comprehensive metrics including F1 Sum Score (class imbalance-aware)
9. Saves best model checkpoint and prediction results (Parquet format)

### Feature Sets

#### Baseline Mode (`--baseline`) - 3 features
| Feature | Source | Description |
|---------|--------|-------------|
| SNR | `events.h5` | Signal-to-noise ratio (energy/noise) |
| η (eta) | `events.h5` | Pseudo-rapidity coordinate |
| φ (phi) | `events.h5` | Azimuthal angle coordinate |

**Use baseline when:** You want a fast, minimal model for quick experiments or as a reference baseline.

#### All Features Mode (`--all-features`) - 42+ features
| Category | Features | Source |
|----------|----------|--------|
| Energy/SNR | SNR (computed), raw energy, raw noise | `events.h5` |
| Geometry | η, φ, Δη, Δφ, volume | `events.h5` + `cells.npy` |
| Noise Stats | noise_mean, noise_std, noise_count, noise_category | `cells.npy` |
| Topology | num_neighbors | `cells.npy` |
| Detector Info | subcalo one-hot (4 classes), sampling one-hot (3+ classes) | `cells.npy` |
| Cluster Info | in_cluster flag, cluster_id_norm | `events.h5` |

**Use all features when:** You want the best possible performance and have sufficient GPU memory.

### Model Architectures

All models follow the same unified design with a shared encoder:

```
Input: Node features (N_cells × F)
       ↓
Node Embedding (Linear: F → hidden_dim)
       ↓
L GNN Layers (configurable)
  ├── Message passing (GCNConv / GATConv / TransformerConv / SAGEConv)
  ├── BatchNorm / LayerNorm
  ├── ReLU activation
  └── Residual connection
       ↓
┌──────────────────────────────────────────────────────┐
│  Two heads, depending on mode:                       │
│                                                      │
│  Pretraining:                                        │
│    Reconstruction Head → Reconstructed Features      │
│    Loss: MSE/L1 on masked positions only             │
│                                                      │
│  Finetuning:                                         │
│    Edge Representation: [h_src || h_dst]             │
│    Classifier (Linear: 2*hidden_dim → 5)             │
│    Loss: CrossEntropy / Focal                        │
└──────────────────────────────────────────────────────┘
```

| Model | `--model` | Key Parameter | Best For |
|-------|-----------|---------------|----------|
| GCN | `gcn` | Symmetric normalization | Baseline, fast training |
| GAT | `gat` | Multi-head attention | Graphs with varying importance |
| Graph Transformer | `transformer` | Self-attention on edges | Long-range dependencies |
| GraphSAGE | `sage` | Neighborhood aggregation | Large graphs, inductive learning |

### Loss Functions

#### Focal Loss (Recommended for extreme class imbalance)
```python
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- **Class-specific alphas** (hardcoded for physics dataset):
  ```python
  RECOMMENDED_ALPHAS = [0.10, 0.60, 0.70, 0.70, 1.00]  # Classes 0,1,2,3,4
  ```
- **γ (gamma)** = 2.0 (controls focus on hard examples)
- Use with: `--weighted-loss --weight-strategy focal`

#### Weighted CrossEntropy
- Weighted by inverse class frequency
- Use with: `--weighted-loss --weight-strategy inverse`

#### Standard CrossEntropy
- No class weighting (default)

#### Masked Reconstruction Loss (for pretraining)
- **MSE** or **L1** for continuous features (SNR, η, φ, energy, etc.)
- **CrossEntropy** for categorical features (subcalo, sampling, noise_category)
- Computed **only on masked positions** to prevent trivial copying

### Evaluation Metrics

The script computes comprehensive metrics. **Best model selection uses F1 Sum Score** (accounts for both false positives and false negatives).

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **F1 Sum Score (FSS)** | Σ(class F1) | 1.0 = random, 5.0 = perfect |
| **Recall Sum Score (RSS)** | Σ(class recall) | 1.0 = random, 5.0 = perfect |
| **Macro F1** | Mean(class F1) | Each class equally important |
| **Weighted F1** | Weighted by class frequency | Favors frequent classes |
| **Per-class metrics** | Precision, recall, F1 | Debug individual class performance |

**Why F1 Sum Score?** Traditional accuracy is misleading with class imbalance (e.g., if 90% of edges are class 0, a model predicting all class 0 gets 90% accuracy but is useless). FSS penalizes models that ignore rare but important classes.

### Output Files

Training produces these files in `--save-dir` (default: `/storage/mxg1065/foundation_experiments/`):

#### 1. Model Checkpoints
- `best_{exp_name}.pt` - Best model based on F1 Sum Score
- `pretrained_{exp_name}.pt` - Best pretrained encoder (if `--pretrain` used)
- `{exp_name}_epoch{N}.pt` - Checkpoint at each epoch (for resuming)

#### 2. Metrics File
- `{exp_name}_metrics.pkl` - Pickle file containing:
  - Training history (loss, accuracy per epoch)
  - Test metrics per epoch
  - Best epoch and best metrics
  - Model arguments (for reproducibility)
- `pretrain_metrics_{exp_name}.pkl` - Pretraining loss history (if `--pretrain` used)

#### 3. Predictions (Parquet format)
- `results_{exp_name}.parquet` - Per-edge predictions with schema:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int32 | Event index |
| `edge_id` | int32 | Edge index within event |
| `source_id` | int32 | Source cell global index |
| `target_id` | int32 | Target cell global index |
| `true_label` | int8 | Ground truth (0-4) |
| `pred_label` | int8 | Model prediction (0-4) |
| `confidence` | float32 | Softmax probability of predicted class |
| `score_class_0` | float32 | Score for class 0 |
| `score_class_1` | float32 | Score for class 1 |
| `score_class_2` | float32 | Score for class 2 |
| `score_class_3` | float32 | Score for class 3 |
| `score_class_4` | float32 | Score for class 4 |
| `model_name` | string | Model identifier |
| `source_cluster` | int32 | Cluster ID of source cell (if available) |
| `target_cluster` | int32 | Cluster ID of target cell (if available) |
| `same_cluster` | bool | Whether cells share same cluster (if available) |

### Usage: train_gnn_models.py

```bash
# Train baseline GCN
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /storage/processed_data/my_dataset \
    --gpu 0 \
    --epochs 30

# Train Graph Transformer with all features and focal loss
python train_gnn_models.py \
    --model transformer \
    --all-features \
    --hidden-dim 256 \
    --layers 8 \
    --heads 4 \
    --weighted-loss \
    --weight-strategy focal \
    --focal-gamma 2.0 \
    --gpu 1

# Train all architectures sequentially (for comparison)
python train_gnn_models.py \
    --model all \
    --baseline \
    --epochs 20 \
    --gpu 0

# Run inference only (use best checkpoint, no training)
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --inference-only \
    --data-dir /storage/processed_data/my_dataset \
    --gpu 0
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model`, `-m` | `gcn` | Model type: `gcn`, `gat`, `transformer`, `sage`, `all` |
| `--baseline` | True | Use 3 features only |
| `--all-features` | False | Use all 42+ features |
| `--hidden-dim` | 128 | Hidden dimension size |
| `--layers`, `-l` | 6 | Number of GNN layers |
| `--heads` | 2 | Attention heads (GAT/Transformer) |
| `--dropout` | 0.0 | Dropout rate |
| `--norm` | `batch` | Normalization: `batch`, `layer`, `none` |
| `--epochs`, `-e` | 30 | Training/finetuning epochs |
| `--batch-size`, `-b` | 1 | Batch size (events per batch) |
| `--lr` | 1e-3 | Learning rate |
| `--weighted-loss` | False | Use weighted loss for imbalance |
| `--weight-strategy` | `inverse` | `inverse`, `focal`, `logarithmic` |
| `--focal-gamma` | 2.0 | Focal loss focusing parameter |
| `--gpu`, `-g` | 0 | GPU ID (-1 for CPU) |
| `--mixed-precision` | True | Enable FP16 training |
| `--inference-only` | False | Skip training, only run inference |
| `--resume` | True | Resume from checkpoint if exists |
| `--patience` | 10 | Early stopping patience |

---

### Masked Pretraining (NEW!)

#### What is Masked Pretraining?

Masked pretraining is a self-supervised learning technique inspired by BERT and Masked Autoencoders (MAE). Instead of training directly on labeled edge classification, the model first learns general representations of calorimeter data by:

1. **Masking** (hiding) parts of the input
2. **Reconstructing** the hidden parts from neighboring context
3. **Transferring** the learned encoder to downstream tasks

This allows the model to learn meaningful patterns in calorimeter cell data **without using any edge labels**. The encoder learns that nearby cells have correlated energies, that SNR varies systematically across the detector, and that cluster structure imposes constraints on cell features.

#### Masking Strategies

The script supports four masking strategies, each designed to teach the model different aspects of calorimeter physics:

| Strategy | `--mask-type` | What Gets Masked | Physics Motivation |
|----------|---------------|------------------|-------------------|
| **Random Cell** | `random` | ~15% of cells (all features) | Learn from neighboring cells (like BERT) |
| **Feature** | `feature` | Specific features (e.g., SNR) across cells | Learn relationships between features |
| **Geometry** | `geometry` | Cells in a contiguous spatial region | Force long-range spatial reasoning |
| **Cluster** | `cluster` | All cells in selected topo-clusters | Learn cluster-level structure |

**Detailed explanation of each strategy:**

##### 1. Random Cell Masking (`--mask-type random`)
- Randomly selects `--mask-ratio` fraction of cells
- Replaces ALL features of selected cells with zeros (or a learnable mask token)
- Model must reconstruct SNR, η, φ from neighboring cells
- **Best starting point** - simplest to implement and validate
- **Physics analogy**: Like inferring a cell's energy from its neighbors

##### 2. Feature Masking (`--mask-type feature`)
- Masks specific features (e.g., only SNR) across a subset of cells
- Use `--mask-features snr eta phi` to specify which features
- Model learns cross-feature relationships
- **Use when:** You want to learn which features are redundant/predictable
- **Physics analogy**: Like predicting SNR from η, φ, and neighbor information

##### 3. Geometry Masking (`--mask-type geometry`)
- Selects a random seed cell, masks nearby cells in η-φ space
- Window size controlled by `--geometry-radius`
- Forces model to use distant cells for reconstruction
- **Use when:** You want to improve long-range edge prediction
- **Physics analogy**: Like reconstructing a localized energy deposit from surrounding detector regions

##### 4. Cluster Masking (`--mask-type cluster`)
- Masks all cells belonging to selected topo-clusters
- Requires `cell_cluster_index` in event data (from `events.h5`)
- Model must reconstruct entire cluster properties from outside the cluster
- **Use when:** You want pretraining aligned with the clustering objective
- **Physics motivation:** Most similar to actual reconstruction task - identifying clusters from surrounding noise

#### How Pretraining Works

**Architecture during pretraining:**
```
Input: Node features (with some masked to 0)
       ↓
Graph Encoder (GCN/GAT/Transformer/SAGE)
  - Message passing over unmasked + masked cells
  - Masked cells still receive messages from neighbors
       ↓
Reconstruction Head (MLP: hidden → hidden/2 → input_dim)
       ↓
Output: Reconstructed features (all cells)
       ↓
Loss: Computed ONLY on masked positions
  - Continuous features: MSE or L1 loss
  - Categorical features: CrossEntropy loss
```

**After pretraining, the workflow is:**

```
Step 1: Pretrain encoder on masked reconstruction
        (no edge labels needed for this step!)
        → Saves: pretrained_{exp_name}.pt
        
Step 2: Transfer encoder weights to classification model
        (reconstruction head is discarded)
        (classification head is randomly initialized)
        
Step 3: Finetune on labeled edge classification data
        (use --finetune-epochs instead of --epochs)
        → Saves: best_{exp_name}.pt
```

**Key implementation details:**

- **Loss computed only on masked positions**: Prevents model from simply copying input. Forces learning from neighborhood context.
- **Feature type detection**: Automatically identifies continuous vs categorical features
  - Continuous (SNR, η, φ, energy): MSE/L1 loss
  - Categorical (subcalo, sampling, noise_category): CrossEntropy loss
- **Encoder transfer**: Only GNN layer weights and embedding are transferred
  - Reconstruction head is discarded after pretraining
  - Classification head (edge classifier) is randomly initialized
- **Pretraining uses all data** (train_ratio=1.0) since no labels are needed

#### Pretraining Usage Examples

```bash
# Basic pretraining: random cell masking, then finetune
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --pretrain \
    --mask-type random \
    --mask-ratio 0.15 \
    --pretrain-epochs 100 \
    --finetune-epochs 30 \
    --gpu 0

# Feature masking: learn cross-feature relationships
python train_gnn_models.py \
    --model transformer \
    --all-features \
    --pretrain \
    --mask-type feature \
    --mask-features snr energy_raw noise_raw \
    --mask-ratio 0.20 \
    --pretrain-epochs 150 \
    --finetune-epochs 50 \
    --gpu 1

# Cluster masking: pretrain on cluster structure
python train_gnn_models.py \
    --model sage \
    --baseline \
    --pretrain \
    --mask-type cluster \
    --mask-ratio 0.30 \
    --pretrain-epochs 200 \
    --finetune-epochs 30 \
    --gpu 0

# Geometry masking with all features
python train_gnn_models.py \
    --model gat \
    --all-features \
    --pretrain \
    --mask-type geometry \
    --geometry-radius 3 \
    --mask-ratio 0.15 \
    --pretrain-epochs 200 \
    --finetune-epochs 30 \
    --gpu 1

# Pretrain only (no finetuning) - save encoder for later use
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --pretrain \
    --mask-type random \
    --mask-ratio 0.15 \
    --pretrain-epochs 500 \
    --finetune-epochs 0 \
    --gpu 0

# Compare all masking strategies with same architecture
for mask_type in random feature geometry cluster; do
    python train_gnn_models.py \
        --model gcn --baseline --pretrain \
        --mask-type $mask_type \
        --pretrain-epochs 100 --finetune-epochs 30 \
        --exp-name "pretrain_${mask_type}" \
        --gpu 0
done
```

**Monitoring pretraining progress:**
- Reconstruction loss should decrease steadily over epochs
- Mask ratio stays constant (logged for verification)
- Best pretrained model saved based on lowest reconstruction loss
- Log messages show: `[Pretrain Epoch X/Y] Loss: Z.ZZZZZZ | Mask ratio: 0.XXX`

**Pretraining output files:**

| File | Description |
|------|-------------|
| `pretrained_{exp_name}.pt` | Best pretrained encoder (before finetuning) |
| `pretrain_metrics_{exp_name}.pkl` | Pretraining loss history per epoch |
| `{exp_name}_metrics.pkl` | Finetuning metrics (standard format) |
| `best_{exp_name}.pt` | Best finetuned model |

---

## File 3: analyze_results.py

### What it does

Generates comprehensive analysis and visualization for trained models. This script takes the model checkpoint files (`.pkl`) and prediction files (`.parquet`) from `train_gnn_models.py` and produces publication-ready figures and reports. It now supports analysis of pretrained models.

**Visualization outputs:**

| Plot Type | Description | What it shows |
|-----------|-------------|----------------|
| **ROC Curves** | Receiver Operating Characteristic curves | AUC per class, trade-off between TPR and FPR |
| **Precision-Recall Curves** | PR curves with F1 iso-lines | AP score, best F1 point, class imbalance effects |
| **Confusion Matrix** | Normalized confusion matrix | Per-class misclassification patterns |
| **Per-Class Bar Chart** | Recall, Precision, F1 per class | Compare performance across 5 classes |
| **Radar Chart** | Multi-metric comparison | Top models compared across 5 metrics |
| **FSS vs RSS Scatter** | F1 Sum vs Recall Sum | Identify models with good precision |
| **Architecture Box Plot** | Performance by architecture | Compare GCN vs GAT vs Transformer vs SAGE |
| **Pretraining Curves** 🆕 | Reconstruction loss + mask ratio | Monitor pretraining convergence |
| **Pretrain vs Scratch** 🆕 | Box plot comparison | Impact of pretraining on FSS |

**Report outputs:**

- **`master_report.html`** - Interactive HTML report with all results (including pretraining)
- **`comprehensive_metrics.csv`** - All metrics in CSV format
- **`comprehensive_table.html`** - Styled table of top 10 models

### Key Features

- **Memory-efficient**: Processes parquet files in chunks (handles millions of edges)
- **Primary metric**: F1 Sum Score (FSS) = Σ(F1 per class) - balances precision AND recall
  - Perfect score = 5.0
  - Random baseline = 1.0
- **Model ranking**: Automatically ranks models and identifies best performer by FSS
- **Architecture comparison**: Compares GCN vs GAT vs Transformer vs SAGE
- **Pretraining support** 🆕: Detects pretrained models, shows pretraining curves, compares vs scratch
- **Fallback handling**: If FSS not found, uses RSS with warning

### Output Directory Structure

After running, you'll get this organized output:

```
analysis_output/
├── figures/
│   ├── roc_curves/              # ROC + F1 tradeoff per model
│   │   └── {model_name}_roc_f1.png
│   ├── pr_curves/               # Precision-Recall curves
│   │   └── {model_name}_pr_curves.png
│   ├── confusion_matrices/      # Confusion matrices + per-class bars
│   │   └── {model_name}_confusion_enhanced.png
│   ├── comparison_plots/        # Cross-model comparisons
│   │   ├── comprehensive_metrics.png
│   │   └── pretrain_vs_scratch.png      🆕
│   ├── metrics_radar/           # Radar charts of top models
│   │   └── metrics_radar.png
│   ├── loss_curves/             # Training/validation loss curves
│   │   ├── {model_name}_loss_curves.png
│   │   └── loss_comparison.png
│   └── pretrain_metrics/        🆕 Pretraining analysis
│       └── pretraining_curves.png
├── tables/
│   └── comprehensive_table.html  # Styled performance table
├── reports/
│   └── master_report.html        # Complete analysis report
└── data/
    └── comprehensive_metrics.csv  # All metrics in CSV format
```

### Understanding the Metrics

| Metric | Abbrev | Range | What it penalizes | Best for |
|--------|--------|-------|-------------------|----------|
| **F1 Sum Score** | FSS | 1.0-5.0 | False Positives + False Negatives | **Primary metric** |
| Recall Sum Score | RSS | 1.0-5.0 | False Negatives only | Legacy comparison |
| Macro F1 | mF1 | 0-1 | Rare class mistakes | Balanced evaluation |
| Weighted F1 | wF1 | 0-1 | Frequent class mistakes | Production deployment |
| Mean Average Precision | mAP | 0-1 | Rank-order mistakes | Threshold tuning |

**Why FSS is the primary metric:**
- A model that predicts all edges as Class 0 (majority class) gets:
  - High Accuracy (~90%) ❌ misleading
  - High RSS (~4.5) ❌ misses the problem
  - Low FSS (~1.2) ✅ reveals the issue

### Usage

**Basic usage:**
```bash
python analyze_results.py \
    --models-dir /path/to/models \
    --parquet-dir /path/to/parquet_files \
    --output-dir ./analysis_output
```

**With pretraining analysis:**
```bash
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./analysis_output \
    --include-pretrain-metrics
```

**Advanced usage with custom limits:**
```bash
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./comprehensive_analysis \
    --max-rows-roc 1000000 \
    --max-rows-confusion 2000000 \
    --top-n 10 \
    --batch-size 50000 \
    --include-pretrain-metrics
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--models-dir` | Required | Directory containing `*_metrics.pkl` files |
| `--parquet-dir` | Required | Directory containing `results_*.parquet` files |
| `--output-dir` | `./analysis_output` | Output directory for all results |
| `--max-rows-roc` | 500000 | Max rows for ROC/PR computation |
| `--max-rows-confusion` | 1000000 | Max rows for confusion matrix |
| `--top-n` | 5 | Number of top models in radar chart |
| `--batch-size` | 100000 | Batch size for parquet reading |
| `--include-pretrain-metrics` 🆕 | False | Include pretraining analysis |

### Pretraining Analysis (NEW!)

When models are trained with `--pretrain`, the analysis script can generate additional plots:

**Pretraining Loss Curves:**
- Shows reconstruction loss decreasing over pretraining epochs
- Log scale for better visibility of convergence
- Mask ratio is also plotted (should remain constant)

**Pretrain vs Scratch Comparison:**
- Box plot comparing F1 Sum Score of pretrained vs scratch models
- Per-class F1 comparison between best pretrained and best scratch model
- Useful for quantifying the benefit of pretraining

**HTML Report Updates:**
- Pretrained models are flagged with 🔧 badge
- Pretraining strategy summary (mask types, ratios used)
- Dedicated pretraining analysis section

**Usage:**
```bash
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./analysis_output \
    --include-pretrain-metrics
```

### What Each Plot Tells You

#### 1. ROC + F1 Tradeoff Plot
- **Left panel**: ROC curves per class with AUC scores
  - AUC > 0.9 = excellent discrimination
  - Diagonal line = random guessing (AUC=0.5)
- **Right panel**: F1 vs Recall scatter plot
  - Bubble size = Precision (larger = better precision)

#### 2. Precision-Recall Curves
- **Each class subplot**: PR curve with F1 iso-lines
  - Dashed lines = constant F1 values (0.2, 0.4, 0.6, 0.8)
  - Dot = point with best F1 score
- **Summary plot**: All classes together for comparison

#### 3. Confusion Matrix + Per-Class Bars
- **Left**: Normalized confusion matrix (diagonal = correct)
- **Right**: Per-class Recall, Precision, F1 bar chart

#### 4. Comprehensive Comparison Plots
- **Top-left**: FSS vs RSS bar chart
- **Top-right**: FSS vs RSS scatter
- **Bottom-left**: Per-class F1 scores (top 7 models)
- **Bottom-right**: Architecture comparison box plot

#### 5. Radar Chart
- Shows top N models across 5 metrics
- Larger area = better overall performance

#### 6. Pretraining Curves 🆕
- **Left**: Reconstruction loss per epoch (log scale)
- **Right**: Mask ratio during pretraining

#### 7. Pretrain vs Scratch 🆕
- **Left**: Box plot comparing FSS distributions
- **Right**: Per-class F1 of best pretrained vs best scratch

### Complete Analysis Workflow

```bash
# Step 1: Train models
python train_gnn_models.py --model gcn --baseline --gpu 0
python train_gnn_models.py --model gat --baseline --gpu 0
python train_gnn_models.py --model gcn --baseline --pretrain --mask-type cluster --gpu 0

# Step 2: Run inference on best checkpoints
python train_gnn_models.py --model gcn --baseline --inference-only

# Step 3: Analyze all results (including pretraining)
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./final_analysis \
    --include-pretrain-metrics

# Step 4: Open report
open ./final_analysis/reports/master_report.html
```

### Interpreting Results: Quick Guide

| If you see... | This means... | Action |
|---------------|----------------|--------|
| FSS > 4.0 | Excellent model | Deploy or use for physics analysis |
| FSS < 2.5 | Poor performance | Try pretraining or different architecture |
| FSS much lower than RSS | Precision problem | Use focal loss or class weights |
| Pretrained FSS > Scratch FSS 🆕 | Pretraining helps | Increase pretrain epochs or try cluster masking |
| Pretrain loss plateaus early 🆕 | Model capacity saturated | Increase hidden_dim or layers |
| Class 0 F1 < 0.5 | Noise identification issues | Check noise statistics |
| Class 4 F1 < 0.3 | Cluster separation failing | Try cluster masking pretraining |

### Troubleshooting

**"No parquet files found"**
- Ensure you ran inference with `--inference-only` or training completed fully

**"Memory error"**
- Reduce `--max-rows-roc` and `--max-rows-confusion`
- Reduce `--batch-size`

**"Missing metrics in pickle"**
- The script falls back to RSS if FSS not found

**"Pretraining metrics not found"** 🆕
- Use `--include-pretrain-metrics` flag
- Ensure `pretrain_metrics_*.pkl` files exist in models directory

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CaloGraphNet.git
cd CaloGraphNet

# Install dependencies
pip install torch torch-geometric numpy awkward uproot h5py pandas pyarrow scikit-learn psutil

# Optional: For progress bars and enhanced analysis
pip install tqdm seaborn matplotlib
```

### Complete Example: From ROOT to Analysis Report

```bash
# Step 1: Build dataset from ROOT file
python build_graph_dataset.py \
    --input /data/run12345.root \
    --output-dir /my_data \
    --dataset-name physics_run \
    --normalize-energy

# Output will be in: /my_data/physics_run/
# - pairs_physics_run.npy (edges)
# - cells_physics_run.npy (cell metadata)
# - events_physics_run.h5 (features)
# - labels_physics_run.npy (ground truth)

# Step 2: Train models
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /my_data/physics_run \
    --gpu 0 \
    --epochs 30

# Step 3: Run inference on best model (saves parquet predictions)
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --inference-only \
    --data-dir /my_data/physics_run \
    --gpu 0

# Step 4: Generate comprehensive analysis report
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./analysis_results

# Step 5: Open the interactive report
open ./analysis_results/reports/master_report.html
```

### Pretraining Example

```bash
# Step 1: Build dataset (same as above)
python build_graph_dataset.py \
    --input /data/run12345.root \
    --output-dir /my_data \
    --dataset-name physics_run

# Step 2: Pretrain with cluster masking, then finetune
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /my_data/physics_run \
    --pretrain \
    --mask-type cluster \
    --mask-ratio 0.25 \
    --pretrain-epochs 200 \
    --finetune-epochs 30 \
    --gpu 0

# Step 3: Also train a baseline without pretraining for comparison
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /my_data/physics_run \
    --epochs 30 \
    --gpu 0

# Step 4: Analyze with pretraining metrics
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./pretrain_analysis \
    --include-pretrain-metrics

# Step 5: Open report to compare pretrained vs scratch
open ./pretrain_analysis/reports/master_report.html
```

### Expected Outputs

After running the full pipeline, you'll have:

```
/storage/foundation_experiments/
├── best_gcn_baseline_h128_l6.pt              # Best model checkpoint
├── pretrained_gcn_baseline_h128_l6.pt        # Best pretrained encoder (if pretrained)
├── gcn_baseline_h128_l6_metrics.pkl          # Training metrics
├── pretrain_metrics_gcn_baseline_h128_l6.pkl # Pretraining metrics (if pretrained)
├── results_gcn_baseline_h128_l6.parquet      # Predictions

./analysis_results/
├── figures/                                   # All plots
│   ├── roc_curves/                           # ROC per model
│   ├── pr_curves/                            # PR per model
│   ├── confusion_matrices/                   # Confusion matrices
│   ├── comparison_plots/                     # Cross-model comparisons
│   ├── metrics_radar/                        # Radar charts
│   ├── loss_curves/                          # Loss curves
│   └── pretrain_metrics/                     # Pretraining analysis 🆕
├── tables/
│   └── comprehensive_table.html              # Sortable results table
├── reports/
│   └── master_report.html                    # Complete analysis report
└── data/
    └── comprehensive_metrics.csv             # All metrics in CSV
```

### Analyze Results with Python

```python
import pandas as pd

# Load predictions
df = pd.read_parquet("results_gcn_baseline_h128_l6.parquet")

# Compute F1 Sum Score
def f1_sum_score(df):
    f1_scores = []
    for c in range(5):
        tp = ((df['true_label'] == c) & (df['pred_label'] == c)).sum()
        fp = ((df['true_label'] != c) & (df['pred_label'] == c)).sum()
        fn = ((df['true_label'] == c) & (df['pred_label'] != c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return sum(f1_scores)

print(f"F1 Sum Score: {f1_sum_score(df):.2f} (range: 1.0=random, 5.0=perfect)")

# Load comprehensive metrics
metrics_df = pd.read_csv("analysis_results/data/comprehensive_metrics.csv")
print(f"Best model: {metrics_df.iloc[0]['name']} (FSS={metrics_df.iloc[0]['f1_sum_score']:.2f})")
```

---

## Configuration Reference

### build_graph_dataset.py Full Arguments

```
usage: build_graph_dataset.py [-h] --input INPUT [--output-dir OUTPUT_DIR]
                              [--dataset-name DATASET_NAME] [--debug]
                              [--skip-snr-scaling] [--normalize-energy]
                              [--energy-normalization {robust,standard}]
                              [--workers WORKERS] [--batch-size BATCH_SIZE]
                              [--chunk-size CHUNK_SIZE]
                              [--h5-compression H5_COMPRESSION]
                              [--h5-chunk-rows H5_CHUNK_ROWS]

Required:
  --input, -i INPUT       Input ROOT file path

Optional:
  --output-dir, -o DIR    Base output directory
  --dataset-name, -n NAME Custom dataset name
  --debug                 Enable debug output (default: True)
  --skip-snr-scaling      Don't scale SNR features
  --normalize-energy      Apply energy normalization (default: True)
  --energy-normalization  'robust' (median/IQR) or 'standard'
  --batch-size N          Events per batch (default: 20)
  --chunk-size N          Events per chunk (default: 50)
  --h5-compression        'gzip', 'lzf', or None (default: 'gzip')
```

### train_gnn_models.py Full Arguments

```
usage: train_gnn_models.py [-h] [--model {gcn,gat,transformer,sage,all}]
                           [--hidden-dim HIDDEN_DIM] [--layers LAYERS]
                           [--heads HEADS] [--dropout DROPOUT]
                           [--layer-weights] [--softmax-weights]
                           [--norm {batch,layer,none}] [--baseline]
                           [--all-features] [--epochs EPOCHS]
                           [--batch-size BATCH_SIZE] [--lr LR]
                           [--weight-decay WEIGHT_DECAY] [--patience PATIENCE]
                           [--weighted-loss]
                           [--weight-strategy {inverse,focal,logarithmic,manual}]
                           [--focal-alpha FOCAL_ALPHA]
                           [--focal-gamma FOCAL_GAMMA] [--train-ratio TRAIN_RATIO]
                           [--gpu GPU] [--mixed-precision] [--no-mixed-precision]
                           [--save-dir SAVE_DIR] [--exp-name EXP_NAME]
                           [--data-dir DATA_DIR] [--resume] [--debug]
                           [--inference-only]

Model Architecture:
  --model {gcn,gat,transformer,sage,all}
                        Backbone architecture (default: gcn)
  --hidden-dim HIDDEN_DIM
                        Hidden dimension (default: 128)
  --layers, -l LAYERS   Number of GNN layers (default: 6)
  --heads HEADS         Attention heads for GAT/Transformer (default: 2)
  --dropout DROPOUT     Dropout rate (default: 0.0)
  --norm {batch,layer,none}
                        Normalization type (default: batch)

Feature Selection:
  --baseline            Use 3 baseline features (default: True)
  --all-features        Use all 42+ features

Training:
  --epochs, -e EPOCHS   Number of finetuning epochs (default: 30)
  --batch-size, -b BATCH_SIZE
                        Batch size in events (default: 1)
  --lr LR               Learning rate (default: 1e-3)
  --patience PATIENCE   Early stopping patience (default: 10)

Loss Functions:
  --weighted-loss       Use weighted loss for class imbalance
  --weight-strategy {inverse,focal,logarithmic,manual}
                        Class weighting strategy (default: inverse)
  --focal-gamma GAMMA   Focal loss gamma (default: 2.0)

Pretraining (NEW):
  --pretrain            Enable masked pretraining before finetuning
  --mask-type {random,feature,geometry,cluster}
                        Masking strategy (default: random)
  --mask-ratio RATIO    Fraction of cells/features to mask (default: 0.15)
  --mask-features FEAT [FEAT ...]
                        Specific features to mask (for feature masking)
  --geometry-radius R   Radius for geometry masking window (default: 2)
  --pretrain-epochs N   Number of pretraining epochs (default: 100)
  --finetune-epochs N   Number of finetuning epochs after pretraining (default: 30)
  --continuous-loss {mse,l1}
                        Loss for continuous features in pretraining (default: mse)

Hardware:
  --gpu, -g GPU         GPU ID (-1 for CPU) (default: 0)
  --mixed-precision     Enable mixed precision (default: True)
  --no-mixed-precision  Disable mixed precision

Experiment:
  --save-dir SAVE_DIR   Directory to save models
  --exp-name EXP_NAME   Experiment name (auto-generated if not provided)
  --data-dir DATA_DIR   Data directory
  --inference-only      Skip training, only run inference
  --resume              Resume from checkpoint (default: True)
  --debug               Debug mode (only runs a few events)
```

---

## Current Limitations & Lessons Learned

### The Challenge: Why Performance Isn't Where We Want It

Despite experimenting with multiple architectures, feature sets, loss functions, and now **masked pretraining**, the best supervised model achieves an **F1 Sum Score of ~2.75** (where 5.0 would be perfect). This section documents what we've learned and where we believe the fundamental issues lie.

### Our Best Results So Far

From trained models, here are the top performers:

| Rank | Model | F1 Sum Score | Accuracy | Macro F1 | Key Features |
|------|-------|--------------|----------|----------|--------------|
| 1 | SAGE (8 layers, focal loss) | **2.75** | 95.1% | 0.55 | Baseline features |
| 2 | Transformer (8 heads, focal loss) | 2.63 | 94.0% | 0.53 | Baseline features |
| 3 | SAGE (all features, inverse loss) | 2.31 | 86.4% | 0.46 | All 42+ features |
| 4 | SAGE (inverse loss) | 2.25 | 84.9% | 0.45 | Baseline |
| 5 | Transformer (inverse loss) | 2.25 | 85.3% | 0.45 | Baseline |

**Note:** Pretraining results are under active investigation and will be added as they become available.

### Lesson 1: Accuracy is Dangerously Misleading for Imbalanced Data

Class 0 (Lone-Lone) dominates with ~92% of all edges. A trivial model predicting all Class 0 achieves 92% accuracy but FSS ~1.2. **Always use FSS, not accuracy.**

### Lesson 2: Focal Loss Helps, But Doesn't Solve the Core Issue

Focal loss improves FSS by ~0.5 over inverse weighting, but the ceiling remains around 2.75.

### Lesson 3: More Features Don't Always Help

All-features mode (42+ features) performed WORSE than baseline (3 features) in several cases.

### Lesson 4: Architecture Differences Are Smaller Than Expected

The gap between best (SAGE) and worst (GCN) is only ~0.5 FSS.

### Lesson 5: The Core Issue May Be Graph Construction Itself

Cells can receive energy from multiple overlapping particle showers, but our hard-label formulation assumes single-cluster membership.

### The Graph Learning Hypothesis

We believe the right approach is to **learn the graph structure itself** rather than classify edges on a fixed graph. Masked pretraining is a step in this direction - learning general representations that may transfer better to the clustering task.

### Future Directions

#### Direction 1: Masked Pretraining (NOW AVAILABLE!) 🆕
We have implemented BERT/MAE-style masked pretraining. Key questions to investigate:
- Does pretraining improve FSS over training from scratch?
- Which masking strategy (random, feature, geometry, cluster) works best?
- How does pretraining affect rare class performance (Classes 1 and 4)?

#### Direction 2: Soft Edge Prediction
Instead of 5 hard classes, predict continuous energy-sharing probabilities.

#### Direction 3: Graph Learning
Learn the adjacency matrix directly rather than classifying fixed edges.

#### Direction 4: Hypergraph Representations
Allow cells to belong to multiple clusters simultaneously.

### Practical Recommendations

1. **Don't trust accuracy** - Use FSS
2. **Try pretraining** - Especially cluster masking for physics alignment
3. **Use focal loss** - Consistently outperforms inverse weighting
4. **Start with baseline features** - All-features may hurt
5. **SAGE seems best** - For supervised training
6. **Compare pretrained vs scratch** - Use `--include-pretrain-metrics`

### Summary Table of What We've Tried

| Approach | What We Did | Best Result | Lesson |
|----------|-------------|-------------|--------|
| **Metrics** | Accuracy → FSS | 2.75 FSS max | Accuracy is useless |
| **Features** | Baseline → All (42+) | Worse performance | More isn't better |
| **Loss** | CrossEntropy → Focal | +0.5 FSS | Focal helps with imbalance |
| **Architecture** | GCN → SAGE → Transformer | SAGE best (2.75) | Differences are small |
| **Pretraining** 🆕 | Masked reconstruction → Finetune | *Under investigation* | May learn better representations |
| **Graph** | Fixed geometric neighbors | Stuck at 2.75 FSS | **Root cause** |

### Open Questions

1. Does masked pretraining improve FSS for rare classes?
2. Which masking strategy best aligns with physics reconstruction?
3. Can soft edge prediction overcome the hard-label limitation?
4. Are there public datasets with fractional energy contributions?

### Conclusion

CaloGraphNet now provides:
- ✅ Graph dataset building from ROOT files
- ✅ Multiple GNN architectures (GCN, GAT, Transformer, SAGE)
- ✅ **Masked pretraining** with 4 strategies 🆕
- ✅ Comprehensive analysis tools (ROC, PR, FSS, HTML reports)
- ✅ **Pretraining analysis** and comparison 🆕

The best supervised model achieves FSS 2.75. We believe pretraining and soft labeling are promising directions for improvement.

---

*We hope this honest assessment and the new pretraining capabilities help advance calorimeter clustering research.*
