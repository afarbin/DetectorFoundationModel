# Memory diagnostic: `train_gnn_models.py` at ~186 GB RSS

*For Mohammad Ali (and whichever AI assistant looks at this) — written 2026-08-11
after observing the running job on cn-1e1901:*

```
python3 train_gnn_models.py --model sage --layers 8 ... --batch-size 1 \
    --data-dir /storage/mxg1065/hh_bbtt_3000_events --pretrain \
    --mask-type random --mask-ratio 0.15
RSS: ~186 GB for 3,000 events
```

## Why it's this large (~60 MB *per event* held permanently)

This is the scaling issue flagged in both code reviews (July review item 10;
August review "P" items). The pipeline keeps **everything for all events in
RAM for the whole run**, in several copies:

1. **`load_features_with_selection` reads six full HDF5 datasets up front**
   (`energy_raw, noise_raw, snr_computed, snr_raw, energy_normalized,
   cell_cluster_index`), each `(3000, 187642)` — ~13.5 GB in float32 before
   anything else happens.
2. **`features_dict` holds a dense per-event feature matrix for every event**
   (187,642 cells × features). Watch for silent **float64 promotion**: the
   static `eta_event0/phi_event0` columns from `cells_*.npy` are `f8`, and one
   `np.stack` with them promotes the whole matrix — doubling memory.
3. **`unscaled_data_dict` duplicates the features** a second time.
4. **Per-event edge-label tensors** are built as `long` `(E,1)` and pinned:
   1.26M pairs × 8 B ≈ 10 MB/event ⇒ ~30 GB, held for all events at once
   (labels on disk are int8 — 8× smaller — and only one batch is ever needed
   at a time).
5. `cluster_info_dict` and assorted per-event dicts add overhead on top.

Sum: ~(2× features + 6 h5 arrays + pinned labels + bookkeeping) ≈ what you
see. It grows *linearly with events* — 10k events would not fit on the node.

## The fix (same direction the review suggested)

The chunked generator is already halfway there — make it actually lazy:

- Keep the **h5 files open and read per chunk** inside `_load_chunk` instead
  of materializing all events in `load_features_with_selection`. Nothing
  outside the current chunk (2,000 events default — make it ~100) needs to
  exist in RAM.
- **Delete the raw h5 arrays** after computing the feature columns you keep;
  never store `unscaled_data_dict` unless `--all-features` diagnostics truly
  need it (and then only for the current chunk).
- Cast features **explicitly to `float32`** at build time (kill the f8
  promotion from `eta_event0`).
- Keep labels **int8 until batch assembly**; convert + (optionally) pin only
  the current batch.
- With `num_workers > 0`, set `torch.multiprocessing.set_sharing_strategy
  ("file_system")` (we hit fd exhaustion on this node otherwise).

Expected footprint after: **a few GB** independent of event count.

## Two node-etiquette flags (both bit us this week)

- torch/numpy default their OpenMP pools to **all 56 cores per process**; set
  `OMP_NUM_THREADS` (we use 6) or the machine becomes unresponsive when jobs
  stack — this is very likely what required the recent admin reboot.
- The node has 251 GB total; at 186 GB resident, one more sizeable job OOMs
  the box. There is now a watchdog (`/storage/afarbin/jetreg/watchdog.sh`)
  alerting on low memory.

## Alternative worth considering

`dfm/data.py::CaloGraphAdapter` (branch `dfm-merge`) does this pretraining's
data handling lazily over the same processed datasets (S3 subset, ~20k cells/
event instead of 187k, multi-shard, cell-energy cut) — the dfm pretraining of
the same masked-cell pretext runs in ~2 GB. The masking-framework fixes from
the August review (P1-P6) are also implemented there, so
`dfm/jetreg/pretrain_calo.py` may already do what this job is trying to do.
