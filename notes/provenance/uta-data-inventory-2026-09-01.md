# UTA data inventory after the SLAC transfer (2026-09-01)

## What arrived

`/storage/mxg1065/slac_ttbar_root_files/` — **85 files, 231 GB, 824,500
events** of the ttbar SuperHJD production (task 50733453), all verified:
every file opens, all carry the full 182-branch schema, **zero duplicate
eventNumbers** across the set, and the v2 split rule stays uniform on the
new events (md5%10 within ±0.4%).

- File numbers: 000011–000020, 000023, 000025, 000028–000035, 000037–000101
  (with gaps 000021/22/24/26/27/36 and 000094–96/99/100).
- The **original 17 files are included** in this directory, and the old
  `/storage/mxg1065/ttbar_100GB/` directory **no longer exists** — the
  transfer consolidated everything here. Any script/doc referencing
  `ttbar_100GB` must switch to `slac_ttbar_root_files`.
- Entry counts are variable: 26 files at 20k events, 56 files at 4.7k,
  plus 4.3k/18k/19k singletons — consistent with the production's variable
  output sizes, not corruption. Net statistics: **2.4×** the original
  340k events (not the ~10× hoped for from "~150 files", because the new
  range is dominated by small files).

## Against the request (SLAC-transfer-request.md)

| Item | Status |
|---|---|
| P1 — VBF H→inv (10 files) | **NOT transferred** — still the top ask |
| P2 — ~150 ttbar files (~1 TB) | Partial: +68 files / +484k events (2.4× total) |
| P4 — container manifests, SuperHJD source | Not received |
| Destination convention (`superhjd/ttbar`) | Not followed (`slac_ttbar_root_files/`); fine, recorded here |

## Consequences

1. **Dataset v2 (built from the original 17) is unaffected** — shards live
   in `/storage/afarbin/jetreg/dataset_v2/`, and the new files share no
   eventNumbers with the old, so the frozen splits extend cleanly.
2. Per the program decision: the 17-file dataset stays frozen for the P3
   re-runs; the +68 files become **dataset v2.1** for the P6 scaling study
   (builder invocation ready — same command, new file list, same id-map).
3. Path updates needed where `ttbar_100GB` is referenced (build docs,
   memory); done in this commit where in-repo.
