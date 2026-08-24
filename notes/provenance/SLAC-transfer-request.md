# SLAC → UTA data transfer request (for Mohammadali)

*2026-08-24. Context: BigPanDA task records (see `D1-dossier.md`) show the
full SuperHJD productions are far larger than what UTA holds. Mohammadali has
SLAC S3DF access and has previously copied files by staging through the jump
node. The script `dfm/v2/ops/slac_pull.sh` removes the staging step: run it
**on cn-1e1901**, and rsync streams directly from SLAC through the jump host
to `/storage` (verified: cn-1e1901 → s3dflogin.slac.stanford.edu:22 is open;
Duo/MFA happens once per session via connection multiplexing).*

*Space check (2026-08-24): /storage has 6.5 TB free. The full request below
is ≈ 1.0–1.3 TB, leaving > 5 TB headroom.*

## What we ask for, in priority order

### P1 — the VBF H→invisible test slice (small, do first)
All **10 files** of
`user.bbullard.mc21_14TeV.600026.PhH7EG_NNPDF3_AZNLO_VBFH125_ZZ4nu_MET75.ntuple.e8481_s4290_r15700.20260608_ntuple.root`
(50k events total; expect roughly 15–60 GB).
→ destination `/storage/mxg1065/superhjd/vbf_hinv/`
This is the proposal's VBF H→inv benchmark process — it unlocks
sample-transfer studies and gives the MET work a genuine signal sample.

### P2 — ttbar extension, ~150 additional files (~0.9–1.1 TB)
From `user.bbullard.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.ntuple.e8481_s4446_r16176.20260604_ntuple.root`
(~2,930 files exist; UTA holds 17 ≈ 0.6%). Any ~150 files **not already at
UTA** — preferably contiguous file numbers for bookkeeping. Already held
(rsync skips already-complete files automatically if you pull into the
existing dir, but we prefer the fresh dir below; for reference):

```
_000011 _000012 _000013 _000014 _000015 _000016 _000017 _000018 _000019
_000020 _000023 _000025 _000028 _000029 _000030 _000031 _000032
```

→ destination `/storage/mxg1065/superhjd/ttbar/`
This is ~10× our current statistics (≈3M more events) — it feeds the v2
per-bin populations and the P6 data-scaling study.

### P3 — rolling ttbar top-up (only after P1+P2 verified)
Continue the ttbar container in ~0.5 TB chunks as /storage space allows
(soft ceiling for now: 2.5 TB total for `superhjd/` — we'll revisit with
Amir before going beyond).

### P4 — provenance items (tiny, one-time)
- The **SuperHJD package**: repo URL + tag/commit used for the June
  productions (or a tarball of the source if it's private to SLAC).
- Before transferring, an `ls -la` **manifest of both containers** as stored
  at SLAC (the script's `list` command produces this) — so we can verify
  counts and sizes against the PanDA record.
- If lying around: the prun log tarballs for tasks 50733453 / 50789452.

## Two questions for you (Mohammadali)

1. What are the actual SLAC filesystem paths of the two containers (e.g.
   under `/sdf/data/...` or group space)? The script takes them as
   arguments.
2. Is the data visible on the jump/login node's filesystem, or only from an
   interior interactive node? The script supports both (set `INNER` for the
   two-hop case).

## How to run (summary; full usage in the script header)

```bash
# on cn-1e1901 (UTA), as mxg1065:
./slac_pull.sh auth                       # one Duo prompt, session cached 12h
./slac_pull.sh list  <slac_path>          # manifest -> saved next to dest
./slac_pull.sh pull  <slac_path> vbf_hinv # stream directly, resumable
./slac_pull.sh pull  <slac_path> ttbar --max-files 150
```
