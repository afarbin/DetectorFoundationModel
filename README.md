# CaloGraphNet workspace

Foundation models for detector reconstruction — models, ntuple makers, and analysis
notebooks for the ATLAS calorimeter ML effort.

## Layout

- **`foundation_model/`** — Mohammad Ali's CaloGraphNet foundation model: graph
  dataset builder, GNN training (GCN/GAT/Graph Transformer/SAGE, masked
  pretraining), and results analysis. See its own `README.md` for details.
- **`btagging/`** — Umar's event-level b-tagging model: DETR-style Set-Transformer
  (`detr_settx_event_btagging.py`) and the ntuple → `.npz` dataset maker
  (`make_event_level_btagging.py`).
- **`genesis_ntuplizers/`** — Athena ntuple makers (`atlascalontuplemaker`,
  `calotiming_ntuplizer`) producing the Calo ntuples.
- **`analysis/`** — analysis notebooks: `5D-Ntuple-FirstLook` (first look at the two
  ntuple formats), `Cell-Matching` (matching 5D-ntuple cells to Calo-ntuple cells),
  `Cell-Selection-Comparison` (the cell-selection bias of the 5D format).
- **`cell_matching.py`** — library for matching cells between the two ntuple formats
  (exact (x,y,z)+sampling KD-tree match, persistent `Cell_ID -> hashID` map);
  demonstrated in `analysis/Cell-Matching.ipynb`.
- **`reviews/`** — code and approach reviews (`calographnet_review.md`,
  `btagging_masking_review.md`).

## Data formats

Two ntuple types (details and measured selections in `analysis/`):

- **5D ntuples** (Umar): per-event *subset* of calorimeter cells
  (`E > 0.5 GeV` and `E > 2 sigma`, positive only), energies in GeV, no hash ID.
- **Calo ntuples** (`genesis_ntuplizers`): all 187,652 cells every event in fixed
  hash order, energies in MeV, with full topocluster -> cell association.
