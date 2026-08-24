# Sample Provenance Dossier (D1)

*Phase 0 deliverable, 2026-08-24. Status: PARTIAL — the blocking external
request below should go out day 1. This file is the single place provenance
facts live; the notes' dataset sections import from here.*

## What we know

### 5D / SuperNtuple production (Umar, SLAC)
Source: SLAC study slides ("Jet Hadronic Energy Reconstruction with Cells and
Tracks", 2026-08-24, slide 3) and our own inspection of the 17 files.

- Sample: `mc21_14TeV` ttbar single-lepton, **PhPy8EG_A14_ttbar_hdamp258p75**
  (Powheg + Pythia8, A14 tune, hdamp = 1.5 m_top), √s = 14 TeV.
- Files: `user.bbullard.*.ntuple.root`, 17 files, ~20k events each, tree
  `ntuple`; energies in GeV.
- Cells: per-event subset, |E/σ_noise| > 2 (and E > 0.5 GeV per our
  measurement in `analysis/Cell-Selection-Comparison.ipynb`; ~75% energy
  retention); per the SLAC study the format also carries cell time and
  quality — **verify in our copies (P1)**.
- Jets: AntiKt4EMTopoJets with `matchedTruth_*` and
  `HadronConeExclTruthLabelID`.

### Calo ntuple production (Mohammadali)
- Produced by `genesis_ntuplizers/atlascalontuplemaker`; all 187,652 cells
  per event in fixed hash order, MeV, full topocluster→cell association;
  tree `analysis`. Samples on server: ttbar and HH→bbττ (3k / 10k events).
- Generator/simulation details: **unknown — part of the request.**

## RESOLVED via BigPanDA task record (2026-08-24, kerberos query authorized by Amir)

Task **50733453** (owner Brendon Bullard, run 2026-06-05→09, status done):

- **Input**: `mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.AOD.e8481_s4446_r16176`
  (17 tid sub-datasets, **44,679,000 events**, 446,790 AOD files). DSID 601229 =
  Powheg+Pythia8 ttbar, A14 tune, hdamp = 1.5 m_top, single-lepton filter.
- **AMI tags**: `e8481` (evgen) / `s4446` (**full Geant4 simulation**) /
  `r16176` (digi+reco incl. pileup). Exact generator versions, PDF and μ
  profile: one AMI lookup on these tags once the grid cert is renewed (or ask
  production).
- **This is an HL-LHC / Run-4 upgrade production**: the dumper ran with
  `SuperHJD.run=RUN4`, geometry tag **ATLAS-P2-RUN4-03-01-00** (Phase-II
  detector, ITk tracker), √s = 14 TeV — our studies are on the *upgrade*
  detector, which every note's detector section must now say.
- **Dumper**: package `SuperHJD`, Athena **25.0.62** (`x86_64-el9-gcc14-opt`,
  AlmaLinux9 container), exact command:
  `python -m SuperHJD.run --filesInput=%IN SuperHJD.run=RUN4
  SuperHJD.getVerticesTracks=True SuperHJD.getCellsInfo=True
  SuperHJD.getTruthParticles=True SuperHJD.geometryTag=ATLAS-P2-RUN4-03-01-00`
  → ask Umar for the SuperHJD repo/tag to close the object-definition items.
- **Output container**: `user.bbullard.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.ntuple.e8481_s4446_r16176.20260604_ntuple.root`
  — ~2,930 files. **UTA's 17 files ≈ 0.6% of the production** (~340k of
  44.7M events; full set ≈ 17 TB). Locating replicas (`rucio
  list-dataset-replicas`) needs the renewed cert; the obvious ask to Umar is
  which site hosts it and whether UTA can pull a larger slice — a 10×
  dataset would transform the P6 scaling study and per-bin statistics.
- **A second process already exists (partially)**: bbullard's task list shows
  parallel SuperHJD productions of **mc21_14TeV.600026
  PhH7EG_NNPDF3_AZNLO_VBFH125_ZZ4nu_MET75** (VBF H(125)→ZZ→4ν, MET>75
  filter — the proposal's VBF H→inv benchmark), tags `e8481_s4290_r15700`
  (note: different s/r tags than the ttbar — not the same pileup/sim
  config; a fair-comparison caveat if used together). Status: the full
  attempts broke/aborted; the surviving task **50789452** (2026-06-08)
  processed a **50k-event test slice → 10 ntuple files** in container
  `user.bbullard.mc21_14TeV.600026...20260608_ntuple.root`. The input AOD
  dataset (4.99M events, tid40102068) exists for a full rerun. Asks to
  Umar: where are the 10 files, and can the full VBF production be rerun?
  This converts our "second-process sample" from long-lead to concrete, and
  gives the MET study a genuine invisible-Higgs signal.
- A *third* process (dijet / Z+jets) remains a request: the same SuperHJD
  command over the corresponding mc21_14TeV AOD dataset.

## CERN trail (checked 2026-08-24, negative)

Searched from lxplus with kerberos (no grid cert available — the `~/.globus`
cert expired 2008 and belongs to another user; renewal via ca.cern.ch is on
Amir): the 17 files are **not** on CERN EOS (deterministic rucio hash paths
probed on `atlasscratchdisk` and `atlasdatadisk`); bbullard's rucio areas
there are empty (deleted-dataset husks); a full scratchdisk walk found zero
files; CERNBox unreadable; no provenance metadata embedded in the ntuples.
Conclusion: replicas live at SLAC (and UTA) only. The one untapped
kerberos-reachable source is the PanDA task record — task id **50733453**
from the filenames; `https://bigpanda.cern.ch/task/50733453/` (CERN SSO)
lists the input DAOD dataset name, whose AMI tags encode generator versions,
sim chain, and pileup — most of the request below in one page.

## What we still need (the request)

| Item | 5D (Umar/SLAC) | Calo (Mohammadali) |
|---|---|---|
| Generator + version + PDF + tune (citations) | partially known (above) — need versions/PDF | needed |
| Pileup profile (μ distribution used) | needed | needed |
| Simulation chain (full sim / fast sim, geometry tag) | needed | needed |
| Reconstruction release + derivation, object definitions/cuts in the dumper | needed | needed |
| Cross-section / filter efficiency (for any rate statement) | needed | needed |
| Is a second-process production feasible (dijet or Z+jets)? | **long-lead ask** | HH→bbττ exists; ttbar details needed |

## Draft request — Umar (+ SLAC contacts)

> Subject: provenance details for the 5D/SuperNtuple ttbar production
>
> Umar — for the v2 notes we need the dataset section to be
> publication-grade, which means pinning the sample beyond what's on your
> slide 3. Could you (or Ariel/whoever ran the production) confirm: exact
> generator versions and PDF set behind PhPy8EG_A14_ttbar_hdamp258p75 as
> configured; the pileup profile (μ distribution) overlaid; simulation chain
> (full/fast sim, geometry/conditions tags); reco release and the
> derivation/dumper cuts defining the tracks, cells (incl. the exact cell
> selection), and jets in the ntuples; and cross-section × filter efficiency.
> Also two forward-looking asks: (1) do your SuperNtuples' cell time/quality
> branches exist in the 17 files we hold at UTA, or only in a newer
> production? (2) how feasible is a second-process production (dijet or
> Z+jets) with the same dumper — this is our long-lead item for
> sample-dependence studies. And for the harmonized evaluation we agreed on:
> could you share your test-set prediction files (jet identifiers +
> predicted correction per model)?

## Draft request — Mohammadali

> Subject: provenance details for the Calo ntuple productions
>
> For the v2 dataset sections: which generator/tune/PDF and simulation chain
> produced the ttbar and HH→bbττ samples behind the Calo ntuples, which
> geometry/conditions tags, and what event selection (if any) the ntuplizer
> applied? If any of this traces to standard ATLAS datasets, the dataset
> names are enough — we can cite from there.
