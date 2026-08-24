# Card: Dataset v2 build (Phase 1)

*Pre-registration per PROTOCOL S1. Committed before the builder runs.
2026-08-24. Facts below measured on the server today (see
`notes/provenance/branch-inventory.md`).*

## Question

Produce the frozen per-jet and event-level datasets for every v2 study, with
a feature schema that exposes what the ntuples actually contain, splits that
survive dataset growth, and a datasheet meeting gap G3/G8 standards.

## Inputs

`/storage/mxg1065/ttbar_100GB/*.ntuple.root` — 17 files, 340,000 events
(measured; all eventNumbers unique, range 5,000,001–36,767,400). HL-LHC
Run-4 production (D1 dossier). Pending arrivals extend, not replace: SLAC
P1/P2 transfers (`superhjd/vbf_hinv`, `superhjd/ttbar`) join under the same
rules.

## Split policy (S2, frozen at G1)

`md5(str(eventNumber)) % 10` → 0–6 train, 7 validation, 8–9 test.
- Measured: even eventNumber%10 is uniform to 4 decimals; md5 chosen for
  robustness under future file additions.
- Event-keyed, not file-keyed (v1 split by file): new files auto-assign;
  both jets of one event never straddle a split; the same split applies to
  per-jet and event-level shards.
- The VBF sample uses the same rule (its own event numbers) and is
  test-only until a full production exists.

## Schema changes vs v1 (justified by the branch inventory)

**Store raw columns, engineer features at load time** (v1 baked 7 cell
features in). Versioned as `schema_v2`.

- Cells: v1 columns + `time`, `quality`, `provenance`, `badcell`,
  `electronicNoise`, `totalNoise`, region flags, `layer`. (~2× cell bytes.)
- Tracks: v1 columns + hit counts (5), `time`/`timeRes`/`hasValidTime`,
  calo-layer extrapolations (16, −999 sentinel), `recoVtx_idx/weight`,
  `truthProb`/`truthPart_idx`.
- Jets: v1 + `width`, `isHS`, `HadronConeExclExtendedTruthLabelID`,
  `truthHSJet_idx`/`truthITPUJet_idx`/`truthOOTPUJet_idx`,
  `matchedTruth_dR`/`GhostFrac`, `btagTrack_idx`. **No gluon label**
  (TruthPart has no partons — measured).
- Event shards: all tracks, thinned cells + edges, jets, TruthHSJet,
  TruthITPUJet/TruthOOTPUJet (new — pileup-jet task), full TruthPart
  (~145/event stable record; truth MET = ν vector sum), TruthVtx (HS, with
  time), μ, eventNumber, split id.
- Known-empty branches are NOT stored: GN2 scores, `Track_truthVtx_idx`.

## Cuts (unchanged from v1 unless noted)

Matched truth (pt>0 GeV), pt_true > 20, ΔR(reco,truth) < 0.3, |η| < 2.5,
duplicate-truth guard, isolation stored as flags (not cut). Deviation from
v1: keep tau-labeled jets in the shards with their extended label (excluded
at training time, not build time) so tagging studies can use them.

## Success criteria (gate G1)

1. Cutflow matches v1's within the differences explained by the tau change.
2. Split fractions 70/10/20 ± 0.3% in jets and events.
3. Cell-ID matching: unmapped-cell rate ≤ v1's; warm CellIDMap reused.
4. Datasheet (`DATASHEET.md`) with per-split/per-flavor/per-bin counts and
   the QA figure set (occupancy, thinning bias vs η/layer, response spectra
   by flavor/pT/η/μ, truth-match purity, μ profile).
   Added after Ariel's review (2026-08-25):
   - **Truth-jet constituent test**: TruthHSJet_pt vs ΔR-cone vector sum of
     TruthPart with and without ν/μ — decides whether particle jets include
     the neutrino (gates the H8 interpretation).
   - **Electron-overlap check**: rate of kept jets with a TruthPart
     electron (pT > 10 GeV) within ΔR < 0.2; veto if non-negligible.
   - Reporting: MPV alongside median; headline bins start at 30 GeV
     (20–30 diagnostic-only).
5. Rebuild is deterministic: same inputs → byte-identical manifests.

## Planned figures

Datasheet QA set above, produced by `dfm/v2/figures.py` factory additions.

## Deviations

(append here)
