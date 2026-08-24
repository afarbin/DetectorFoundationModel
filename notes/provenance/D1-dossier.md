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
