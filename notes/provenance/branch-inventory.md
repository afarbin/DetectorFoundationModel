# 5D SuperNtuple branch inventory and fill status

*Phase 1 verification, 2026-08-24. File `user.bbullard.50733453._000011`,
182 branches; fill status measured on 200 events. This is the input record
for `DATASHEET.md`; feature choices in the v2 builder cite this file.*

## Verified capabilities (branch exists AND is filled)

| Capability | Branches | Fill status / notes |
|---|---|---|
| Cell timing + quality | `Cell_time`, `Cell_quality`, `Cell_provenance`, `Cell_badcell` | ~80% nonzero (zeros consistent with out-of-cluster cells); enables timing-aware features **now** |
| Noise decomposition | `Cell_electronicNoise`, `Cell_totalNoise` | filled; v1 used only `significance` |
| Detector region flags | `Cell_isEM(_Barrel/_EndCap)`, `Cell_isTile/isHEC/isFCAL`, `Cell_detName`, `Cell_layer` | filled |
| Track timing (HGTD-era) | `Track_time`, `Track_timeRes`, `Track_hasValidTime` | **39.6% of tracks have valid time**; −999 sentinel otherwise |
| Track→calo extrapolation | `Track_{PreSamplerB/E,EMB1-3,EME1-3}_{eta,phi}` | filled where extrapolation exists (−999 sentinel) — enables track-cell association without ΔR cones |
| Track hit content | `Track_nInmostPixelHits`, `nPixelHits/Holes`, `nStripHits/Holes` ("Strip" = ITk, consistent with Run-4) | filled |
| Full track covariance | `Track_cov_*` (10), `Track_var_*` (5) | filled; v1 derived IP significances from these |
| Reco-vertex grouping | `Track_recoVtx_idx` (up to ~130 vtx/event), `Track_recoVtx_weight` | filled; **no RecoVtx_* collection branches** (positions not dumped) — grouping only |
| Truth HS vertex + time | `TruthVtx_{x,y,z,time,isHS,track_idx}` | **one vertex/event (HS only)**, time filled |
| Truth particles | `TruthPart_{pt,eta,phi,m,pdgId,charge,prodVtx_*}` | filled; neutrino-sum truth MET confirmed; **parton (gluon) labels derivable** by jet–parton matching on pdgId — no dedicated branch |
| Pileup truth jets | `TruthITPUJet_*` (~20/evt), `TruthOOTPUJet_*` (~41/evt), `TruthHSJet_*` | filled → **HS-vs-pileup jet discrimination is a live task** (with `AntiKt4EMTopoJets_isHS`, `truthHSJet_idx`, `truthITPUJet_idx`, `truthOOTPUJet_idx`) |
| Extended flavor labels | `HadronConeExclExtendedTruthLabelID` | filled (values incl. 44/55 double-c/b, 15 tau) |
| Jet substructure inputs | `constituentPt/Eta/Phi`, `width`, `btagTrack_idx`, `matchedTruth_dR/GhostFrac` | filled |

## Present but EMPTY (do not use; flag to Umar)

- **`AntiKt4EMTopoJets_GN2_{pb,pc,pu,Db}` — all zero.** No production-tagger
  baseline in this dump. Ask: can a rerun fill GN2 (or was it absent from
  the Run-4 AODs)?
- **`Track_truthPart_idx` / `Track_truthProb`** — truthPart_idx present
  (unverified fill), but **`Track_truthVtx_idx` is −1 everywhere**: no
  track-to-truth-vertex association; multi-vertex *truth* supervision is
  limited to the single HS vertex.

## Still absent from the format

- MET / soft-term branches (unchanged conclusion; truth MET from neutrinos).
- Reco vertex collection (positions/times) — only per-track indices.
- Dedicated parton-flavor label branch (derive via TruthPart matching).

## Program impact (folded into DFM-V2-PROGRAM §5)

1. "Timing-aware reconstruction" moves **BLOCKED → available now (partial)**:
   cell time/quality + 40%-coverage track time + truth HS vertex time.
2. **New candidate task**: pileup-jet discrimination (HS vs ITPU vs OOTPU) —
   directly the proposal's pileup narrative; data fully in hand.
3. Gluon split: **derivable in the v2 builder** (TruthPart pdgId matching),
   no reprocessing needed.
4. GN2 comparison: **not possible with current files** — stays on the Umar
   ask list.
5. The v2 feature schema should adopt: cell time/quality/noise-split/region
   flags; track hit counts + timing + layer extrapolations; jet width +
   constituents; extended labels.
