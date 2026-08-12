# DFM Internal Notes

ATLAS-internal-style documentation of the Detector Foundation Model studies,
in logical order — each note builds on and references the previous:

| Note | Study | Reads first |
|---|---|---|
| [DFM-NOTE-2026-01](DFM-NOTE-2026-01-jet-pt-calibration/note.html) | Machine-learned jet pT calibration: input modalities, graph vs set encoding, masked-cell pretraining (Tiers 0–4, 620k light jets) | — |
| [DFM-NOTE-2026-02](DFM-NOTE-2026-02-flavor-calibration/note.html) | Flavor-dependent calibration: blind vs conditioned vs dedicated (1.2M all-flavor jets) | Note 01 |
| [DFM-NOTE-2026-03](DFM-NOTE-2026-03-panoptic-tagging-calibration/note.html) | Panoptic tagging + calibration in one head | Notes 01–02 |

Each directory contains `note.html` (the document, self-contained with embedded
figures) and `supporting/` (the interactive analysis reports produced during
the study). These are internal drafts of the UTA DFM group, not official ATLAS
documents.
