#!/usr/bin/env python3
"""
MULTI-FILE DATASET BUILDER with STRUCTURED CLUSTER INFORMATION
- Processes multiple ROOT files into independent event/label pairs
- Shared static geometry (pairs, cells) computed once
- Per-file timing and overall progress tracking
- Vectorized operations for maximum performance
"""

import os
import sys
import time
import json
import gc
import math
import psutil
import argparse
import pickle
import re
from functools import partial
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import h5py

# ------------------ PARSE COMMAND LINE ARGUMENTS ------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Process multiple ROOT files into graph ML dataset')
    parser.add_argument('--input-dir', '-d', type=str, required=True,
                        help='Directory containing ROOT files')
    parser.add_argument('--input-pattern', type=str, default='*.root',
                        help='File pattern to match (default: *.root)')
    parser.add_argument('--output-dir', '-o', type=str, default='./processed_data',
                        help='Base output directory (default: ./processed_data)')
    parser.add_argument('--dataset-name', '-n', type=str, default=None,
                        help='Custom dataset name (default: derived from first input file)')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Enable debug output')
    parser.add_argument('--normalize-energy', action='store_true', default=True)
    parser.add_argument('--energy-normalization', type=str, default='robust',
                        choices=['robust', 'standard'])
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Events to process at once (default: 100)')
    parser.add_argument('--h5-compression', type=str, default='gzip')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    return parser.parse_args()

# ------------------ HELPER FUNCTIONS ------------------
def get_dataset_name(file_path, custom_name=None):
    """Generate dataset name from first input filename"""
    if custom_name:
        return custom_name
    
    file_path = Path(file_path)
    base_name = file_path.stem
    
    # Remove common patterns and trailing numbers
    base_name = base_name.replace('.outputs', '')
    base_name = base_name.replace('xAODAnalysis_', '')
    
    # Remove trailing _number_number pattern
    base_name = re.sub(r'_\d+_\d+$', '', base_name)
    
    return base_name

def log(msg, debug_mode=True):
    """Conditional logging with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def format_time(seconds):
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ------------------ CELL SIZE AND VOLUME MAPPING ------------------
CELL_SIZES = {
    '0_0': (0.025, 0.1), '0_1': (0.0031, 0.0245), '0_2': (0.025, 0.0245), '0_3': (0.05, 0.0245),
    '0_4': (0.025, 0.1), '0_5': (0.0031, 0.1), '0_6': (0.1, 0.1), '0_7': (0.1, 0.1),
    '1_8': (0.1, 0.09817481), '1_9': (0.1, 0.09817481), '1_10': (0.1, 0.09817481), '1_11': (0.1, 0.09817481),
    '2_21': (0.1, 0.1), '2_22': (0.1, 0.1), '2_23': (0.1, 0.1),
    '3_12': (0.1, 0.09817481), '3_13': (0.1, 0.09817481), '3_14': (0.2, 0.09817481), '3_15': (0.2, 0.09817481),
    '3_16': (0.2, 0.09817481), '3_17': (0.25, 0.09817481), '3_18': (0.1, 0.09817481), '3_19': (0.1, 0.09817481),
    '3_20': (0.2, 0.09817481),
}

SUB_CALO_CATEGORIES = [0, 1, 2, 3]
SUB_CALO_TO_INDEX = {val: idx for idx, val in enumerate(SUB_CALO_CATEGORIES)}
NUM_SUB_CALO = len(SUB_CALO_CATEGORIES)

# This is an estimated volume
def get_cell_volume(subCalo, sampling, eta, phi):
    key = f"{subCalo}_{sampling}"
    if key in CELL_SIZES:
        deta, dphi = CELL_SIZES[key]
        r = 1000.0
        volume = deta * dphi * r
        return volume
    return 1.0

# ------------------ FILE DISCOVERY ------------------
def discover_input_files(args):
    """Discover ROOT files from input directory"""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.input_dir}")
    
    files = sorted(input_dir.glob(args.input_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{args.input_pattern}' in {args.input_dir}")
    
    # Apply max-files limit if specified
    if args.max_files and args.max_files < len(files):
        files = files[:args.max_files]
        log(f"Limited to first {args.max_files} files (--max-files)")
    
    log(f"Found {len(files)} input files:")
    for f in files:
        log(f"  - {f.name}")
    
    return files

# ------------------ TREE OPENING ------------------
def open_tree(file_path):
    """Open ROOT file and find appropriate tree"""
    f = uproot.open(file_path)
    possible_trees = ["analysis;1", "analysis", "tree", "CollectionTree"]
    
    for n in possible_trees:
        if n in f:
            return f, f[n], n
    
    # If none found, try first available tree
    tree_name = list(f.keys())[0]
    return f, f[tree_name], tree_name

# ------------------ DATA EXTRACTION HELPER ------------------
def extract_array_safe(ak_array, global_to_orig, num_cells_global, dtype=np.float32, default_value=0):
    """Safely extract numpy array from awkward array, handling variable lengths"""
    try:
        # Try direct numpy conversion first (works for fixed-size arrays)
        return ak.to_numpy(ak_array)[:, global_to_orig]
    except (ValueError, TypeError):
        # Handle variable-length arrays
        py_list = ak.to_list(ak_array)
        result = np.full((len(py_list), num_cells_global), default_value, dtype=dtype)
        
        for i, row in enumerate(py_list):
            r = np.asarray(row).ravel()
            if len(r) >= len(global_to_orig):
                result[i, :] = r[global_to_orig]
            else:
                for j, orig_idx in enumerate(global_to_orig):
                    if orig_idx < len(r):
                        result[i, j] = r[orig_idx]
        return result

# ------------------ STATIC GEOMETRY BUILDER ------------------
def build_static_geometry(reference_file_path, args, key_branch_names):
    """Build static geometry from reference file (called once)"""
    log("\n" + "="*70)
    log("PHASE 1: BUILDING STATIC GEOMETRY")
    log("="*70)
    t_start = time.time()
    
    # Open reference file
    f, tree, tree_name = open_tree(reference_file_path)
    log(f"Using reference file: {reference_file_path.name}")
    log(f"Found tree '{tree_name}' with {tree.num_entries} events")
    
    # Read event 0 for structural reference
    log("Reading event 0 for geometry structure...")
    entry0 = tree.arrays(list(key_branch_names.values()), entry_start=0, entry_stop=1, library="ak")
    
    def get_event0_field(key):
        br = key_branch_names.get(key)
        if br is None:
            return None
        arr0 = entry0[br][0]
        if arr0 is None:
            return np.array([], dtype=np.float32)
        return ak.to_numpy(arr0)
    
    energy_name = key_branch_names.get("cell_energy")
    noise_name = key_branch_names.get("cell_noiseSigma")
    snr_name = key_branch_names.get("cell_SNR")
    
    if energy_name and noise_name:
        use_energy_noise = True
        log(f"  ✓ Found energy branch: {energy_name}")
        log(f"  ✓ Found noise branch: {noise_name}")
    elif snr_name:
        use_energy_noise = False
        log(f"  ✓ Found SNR branch: {snr_name}")
    else:
        use_energy_noise = False
        log(f"  ⚠️  No energy/noise/SNR branches found")
    
    # Build valid mask
    cell_noise_ev0 = None
    if noise_name:
        cell_noise_ev0 = get_event0_field("cell_noiseSigma")
        valid_mask0 = (cell_noise_ev0 != 0) if cell_noise_ev0 is not None else np.ones(1, dtype=bool)
    elif snr_name:
        cell_snr_ev0 = get_event0_field("cell_SNR")
        valid_mask0 = (cell_snr_ev0 != 0)
    else:
        valid_mask0 = np.ones(1, dtype=bool)
    
    num_cells_orig = valid_mask0.size if valid_mask0.size > 0 else 0
    global_valid_indices = np.where(valid_mask0)[0] if valid_mask0.size > 0 else np.array([], dtype=np.int32)
    num_cells_global = len(global_valid_indices)
    global_to_orig = global_valid_indices.astype(np.int32)
    orig_to_global = {int(orig): int(i) for i, orig in enumerate(global_to_orig)}
    log(f"  ✓ {num_cells_global} valid cells (out of {num_cells_orig} original)")
    
    # Build connectivity
    log("Building cell connectivity...")
    neighbor_name = key_branch_names.get("neighbor")
    if neighbor_name is None:
        log("⚠️  neighbor branch not found")
        pairs_array = np.zeros((0, 2), dtype=np.int32)
    else:
        neighbor0_ak = tree.arrays([neighbor_name], entry_start=0, entry_stop=1, library="ak")[neighbor_name][0]
        neighbor0 = ak.to_list(neighbor0_ak)
        pairs = []
        for orig_src, nbrs in enumerate(neighbor0):
            if orig_src not in orig_to_global:
                continue
            src = orig_to_global[orig_src]
            if nbrs is None:
                continue
            for raw in nbrs:
                try:
                    r = int(raw)
                except Exception:
                    continue
                if r in orig_to_global and src != orig_to_global[r]:
                    pairs.append((src, orig_to_global[r]))
        
        # Deduplicate
        canonical_pairs = []
        for src, dst in pairs:
            if src != dst:
                a, b = min(src, dst), max(src, dst)
                canonical_pairs.append((a, b))
        unique_pairs = list(set(canonical_pairs))
        unique_pairs.sort()
        
        # Order by source
        pairs_by_source = [[] for _ in range(num_cells_global)]
        for a, b in unique_pairs:
            pairs_by_source[a].append(b)
        
        ordered_pairs = []
        for src in range(num_cells_global):
            if pairs_by_source[src]:
                pairs_by_source[src] = sorted(set(pairs_by_source[src]))
                for dst in pairs_by_source[src]:
                    ordered_pairs.append([src, dst])
        
        pairs_array = np.array(ordered_pairs, dtype=np.int32)
    
    log(f"  ✓ {pairs_array.shape[0]} edges")
    
    # Compute static cell features
    log("Computing static cell features...")
    subcalo_name = key_branch_names.get("cell_subCalo")
    sampling_name = key_branch_names.get("cell_sampling")
    eta_name = key_branch_names.get("cell_eta")
    phi_name = key_branch_names.get("cell_phi")
    
    cell_eta_ev0 = get_event0_field("cell_eta") if eta_name else np.full(num_cells_orig, np.nan)
    cell_phi_ev0 = get_event0_field("cell_phi") if phi_name else np.full(num_cells_orig, np.nan)
    
    if subcalo_name and sampling_name:
        static_data = tree.arrays([subcalo_name, sampling_name], entry_start=0, entry_stop=1, library="ak")
        subcalo_ev0 = ak.to_numpy(static_data[subcalo_name][0])
        sampling_ev0 = ak.to_numpy(static_data[sampling_name][0])
        subcalo_global = subcalo_ev0[global_to_orig]
        sampling_global = sampling_ev0[global_to_orig]
        
        # Compute volume and size
        cell_volumes = np.zeros(num_cells_global, dtype=np.float32)
        cell_deta = np.zeros(num_cells_global, dtype=np.float32)
        cell_dphi = np.zeros(num_cells_global, dtype=np.float32)
        
        for i in range(num_cells_global):
            cell_volumes[i] = get_cell_volume(subcalo_global[i], sampling_global[i],
                                              cell_eta_ev0[global_to_orig][i] if cell_eta_ev0 is not None else 0,
                                              cell_phi_ev0[global_to_orig][i] if cell_phi_ev0 is not None else 0)
            key = f"{subcalo_global[i]}_{sampling_global[i]}"
            if key in CELL_SIZES:
                cell_deta[i], cell_dphi[i] = CELL_SIZES[key]
    else:
        log("  ⚠️  Missing subCalo or sampling branches - using defaults")
        subcalo_global = np.zeros(num_cells_global, dtype=np.int32)
        sampling_global = np.zeros(num_cells_global, dtype=np.int32)
        cell_volumes = np.ones(num_cells_global, dtype=np.float32)
        cell_deta = np.ones(num_cells_global, dtype=np.float32)
        cell_dphi = np.ones(num_cells_global, dtype=np.float32)
    
    # Compute noise statistics
    log("Computing per-cell noise statistics...")
    if use_energy_noise and noise_name:
        noise_means = np.zeros(num_cells_global, dtype=np.float32)
        noise_stds = np.zeros(num_cells_global, dtype=np.float32)
        noise_counts = np.zeros(num_cells_global, dtype=np.int32)
        
        num_events = tree.num_entries
        for chunk_start in range(0, min(num_events, 500), args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, num_events)
            arrs = tree.arrays([noise_name], entry_start=chunk_start, entry_stop=chunk_end, library="ak")
            noise_chunk_np = extract_array_safe(arrs[noise_name], global_to_orig, num_cells_global)
            
            # Update noise statistics per cell
            for cell_idx in range(num_cells_global):
                cell_noise = noise_chunk_np[:, cell_idx]
                valid = (cell_noise != 0) & ~np.isnan(cell_noise)
                if valid.any():
                    valid_noise = cell_noise[valid]
                    old_count = noise_counts[cell_idx]
                    new_count = old_count + len(valid_noise)
                    if old_count == 0:
                        noise_means[cell_idx] = np.mean(valid_noise)
                        noise_stds[cell_idx] = np.std(valid_noise)
                    else:
                        old_mean = noise_means[cell_idx]
                        new_mean = old_mean + np.sum(valid_noise - old_mean) / new_count
                        old_ssd = noise_stds[cell_idx]**2 * old_count
                        new_ssd = old_ssd + np.sum((valid_noise - old_mean) * (valid_noise - new_mean))
                        noise_means[cell_idx] = new_mean
                        noise_stds[cell_idx] = np.sqrt(new_ssd / new_count)
                    noise_counts[cell_idx] = new_count
            
            del arrs, noise_chunk_np
            gc.collect()
        
        # Categorize noise levels
        noise_categories = np.full(num_cells_global, -1, dtype=np.int8)
        valid_noise_cells = noise_counts > 0
        if valid_noise_cells.any():
            noise_percentiles = np.percentile(noise_means[valid_noise_cells], [33, 66])
            noise_categories[valid_noise_cells] = np.digitize(
                noise_means[valid_noise_cells], noise_percentiles
            ).astype(np.int8)
    else:
        noise_means = np.zeros(num_cells_global, dtype=np.float32)
        noise_stds = np.zeros(num_cells_global, dtype=np.float32)
        noise_counts = np.zeros(num_cells_global, dtype=np.int32)
        noise_categories = np.full(num_cells_global, -1, dtype=np.int8)
    
    log("  ✓ Noise statistics computed")
    
    # Create cells structured array
    cells_struct = np.zeros(num_cells_global, dtype=[
        ('global_idx', 'i4'), ('orig_idx', 'i4'),
        ('eta_event0', 'f8'), ('phi_event0', 'f8'),
        ('broken_event0', 'b1'), ('num_neighbors', 'i4'),
        ('subcalo', 'i4'), ('sampling', 'i4'),
        ('deta', 'f4'), ('dphi', 'f4'), ('volume', 'f4'),
        ('noise_mean', 'f4'), ('noise_std', 'f4'), ('noise_count', 'i4'), ('noise_category', 'i1')
    ])
    
    cells_struct['global_idx'] = np.arange(num_cells_global, dtype=np.int32)
    cells_struct['orig_idx'] = global_to_orig
    cells_struct['eta_event0'] = cell_eta_ev0[global_to_orig] if cell_eta_ev0 is not None else np.zeros(num_cells_global)
    cells_struct['phi_event0'] = cell_phi_ev0[global_to_orig] if cell_phi_ev0 is not None else np.zeros(num_cells_global)
    cells_struct['broken_event0'] = (cell_noise_ev0[global_to_orig] == 0) if noise_name and cell_noise_ev0 is not None else np.zeros(num_cells_global, dtype=bool)
    
    num_pairs = pairs_array.shape[0]
    if num_pairs > 0:
        cells_struct['num_neighbors'] = np.bincount(pairs_array[:, 0], minlength=num_cells_global)
    else:
        cells_struct['num_neighbors'] = np.zeros(num_cells_global)
    cells_struct['subcalo'] = subcalo_global
    cells_struct['sampling'] = sampling_global
    cells_struct['deta'] = cell_deta
    cells_struct['dphi'] = cell_dphi
    cells_struct['volume'] = cell_volumes
    cells_struct['noise_mean'] = noise_means
    cells_struct['noise_std'] = noise_stds
    cells_struct['noise_count'] = noise_counts
    cells_struct['noise_category'] = noise_categories
    
    elapsed = time.time() - t_start
    log(f"\n✅ Static geometry built in {format_time(elapsed)}")
    log(f"   Cells: {num_cells_global:,} | Edges: {num_pairs:,}")
    
    f.close()
    return pairs_array, cells_struct, global_to_orig, num_cells_global, use_energy_noise, key_branch_names

# ------------------ ENERGY SCALER FITTER ------------------
def fit_energy_scaler_all_files(input_files, args, key_branch_names, global_to_orig, num_cells_global):
    """Fit energy scaler across all files"""
    log("\n" + "="*70)
    log("PHASE 2: FITTING ENERGY SCALER ACROSS ALL FILES")
    log("="*70)
    t_start = time.time()
    
    if not args.normalize_energy:
        log("Energy normalization disabled - using identity scaling")
        return 0.0, 1.0
    
    energy_name = key_branch_names.get("cell_energy")
    if not energy_name:
        log("No energy branch found - using identity scaling")
        return 0.0, 1.0
    
    sample_energies = []
    total_vals = 0
    
    for file_idx, file_path in enumerate(input_files):
        log(f"  Sampling file {file_idx+1}/{len(input_files)}: {file_path.name}")
        f, tree, _ = open_tree(file_path)
        num_events = tree.num_entries
        
        # Sample up to 200 events per file
        sample_events = min(num_events, 200)
        sample_indices = np.linspace(0, num_events-1, sample_events, dtype=int)
        
        for batch_start in range(0, sample_events, args.chunk_size):
            batch_end = min(batch_start + args.chunk_size, sample_events)
            batch_indices = sample_indices[batch_start:batch_end]
            
            # Process batch
            arrs = tree.arrays([energy_name], entry_start=int(batch_indices[0]), 
                             entry_stop=int(batch_indices[-1]) + 1, library="ak")
            energy_np = extract_array_safe(arrs[energy_name], global_to_orig, num_cells_global)
            
            valid_energy = energy_np[energy_np > 0]
            if valid_energy.size > 0:
                if valid_energy.size > 5000:
                    idx = np.random.choice(valid_energy.size, size=5000, replace=False)
                    sample_energies.append(valid_energy[idx])
                    total_vals += 5000
                else:
                    sample_energies.append(valid_energy)
                    total_vals += valid_energy.size
        
        f.close()
        gc.collect()
    
    if total_vals > 0:
        all_energies = np.concatenate(sample_energies)
        if args.energy_normalization == "robust":
            med = np.median(all_energies)
            q1 = np.percentile(all_energies, 25)
            q3 = np.percentile(all_energies, 75)
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
            energy_center = med
            energy_scale = iqr
        else:
            energy_center = np.mean(all_energies)
            energy_scale = np.std(all_energies)
        
        log(f"  ✅ Energy scaler fitted: center={energy_center:.6g}, scale={energy_scale:.6g}")
        log(f"     Method: {args.energy_normalization}")
        log(f"     Samples used: {total_vals:,} values from {len(input_files)} files")
        
        del all_energies, sample_energies
    else:
        energy_center, energy_scale = 0.0, 1.0
        log("  ⚠️  No valid energies found - using identity scaling")
    
    elapsed = time.time() - t_start
    log(f"  ⏱️  Scaler fitting completed in {format_time(elapsed)}")
    
    return energy_center, energy_scale

# ------------------ SINGLE FILE PROCESSOR ------------------
def process_single_file(input_file, output_dir, file_output_name,
                       pairs_array, cells_struct, global_to_orig, 
                       num_cells_global, energy_center, energy_scale,
                       use_energy_noise, key_branch_names, args,
                       file_idx, total_files):
    """Process events from a single file into its own events.h5 + labels.npy"""
    log("\n" + "="*70)
    log(f"PROCESSING FILE {file_idx+1}/{total_files}: {input_file.name}")
    log("="*70)
    t_start = time.time()
    
    # Open file
    f, tree, tree_name = open_tree(input_file)
    num_events = tree.num_entries
    num_pairs = pairs_array.shape[0]
    
    log(f"  Events: {num_events:,} | Cells: {num_cells_global:,} | Edges: {num_pairs:,}")
    
    # Define output paths
    events_file = output_dir / f"events_{file_output_name}.h5"
    labels_file = output_dir / f"labels_{file_output_name}.npy"
    
    energy_name = key_branch_names.get("cell_energy")
    noise_name = key_branch_names.get("cell_noiseSigma")
    snr_name = key_branch_names.get("cell_SNR")
    cluster_name = key_branch_names.get("cell_cluster_index")
    
    # Create labels memmap
    if num_pairs > 0:
        labels_mmap_path = output_dir / f"labels_memmap_{file_output_name}.npy"
        labels_mmap = np.memmap(labels_mmap_path, dtype=np.int8, mode='w+', shape=(num_events, num_pairs))
    else:
        labels_mmap_path = output_dir / f"labels_memmap_{file_output_name}.npy"
        labels_mmap = np.memmap(labels_mmap_path, dtype=np.int8, mode='w+', shape=(num_events, 0))
    
    # Create HDF5 file
    with h5py.File(events_file, "w") as h5f:
        # Add metadata attributes
        h5f.attrs['source_file'] = input_file.name
        h5f.attrs['file_output_name'] = file_output_name
        h5f.attrs['num_events'] = num_events
        h5f.attrs['num_cells'] = num_cells_global
        h5f.attrs['num_pairs'] = num_pairs
        h5f.attrs['energy_center'] = energy_center
        h5f.attrs['energy_scale'] = energy_scale
        
        # Create datasets
        if use_energy_noise:
            energy_raw_dset = h5f.create_dataset("cell/energy_raw", shape=(num_events, num_cells_global), 
                                                dtype=np.float32, compression=args.h5_compression)
            noise_dset = h5f.create_dataset("cell/noise_raw", shape=(num_events, num_cells_global), 
                                           dtype=np.float32, compression=args.h5_compression)
            snr_computed_dset = h5f.create_dataset("cell/snr_computed", shape=(num_events, num_cells_global), 
                                                  dtype=np.float32, compression=args.h5_compression)
            if snr_name:
                snr_raw_dset = h5f.create_dataset("cell/snr_raw", shape=(num_events, num_cells_global), 
                                                 dtype=np.float32, compression=args.h5_compression)
            if args.normalize_energy:
                energy_norm_dset = h5f.create_dataset("cell/energy_normalized", shape=(num_events, num_cells_global), 
                                                     dtype=np.float32, compression=args.h5_compression)
        else:
            if snr_name:
                snr_raw_dset = h5f.create_dataset("cell/snr_raw", shape=(num_events, num_cells_global), 
                                                 dtype=np.float32, compression=args.h5_compression)
        
        cluster_dset = h5f.create_dataset("cell/cell_cluster_index", shape=(num_events, num_cells_global), 
                                         dtype=np.int32, compression=args.h5_compression)
        
        # Pre-compute source/destination indices for all edges (used in label generation)
        if num_pairs > 0:
            edge_src = pairs_array[:, 0]  # Source cell indices for all edges
            edge_dst = pairs_array[:, 1]  # Destination cell indices for all edges
        
        # Process events in chunks
        for chunk_start in range(0, num_events, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, num_events)
            chunk_size_actual = chunk_end - chunk_start
            
            if chunk_start % (args.chunk_size * 5) == 0:
                progress = chunk_start / num_events * 100
                log(f"    Progress: {chunk_start:,}/{num_events:,} events ({progress:.1f}%)")
            
            # Read data
            to_read = []
            if use_energy_noise:
                if energy_name: to_read.append(energy_name)
                if noise_name: to_read.append(noise_name)
                if snr_name: to_read.append(snr_name)
            else:
                if snr_name: to_read.append(snr_name)
            if cluster_name:
                to_read.append(cluster_name)
            
            arrs = tree.arrays(to_read, entry_start=chunk_start, entry_stop=chunk_end, library="ak")
            
            # Extract arrays using safe helper
            snr_np = None
            if snr_name and snr_name in arrs.fields:
                snr_np = extract_array_safe(arrs[snr_name], global_to_orig, num_cells_global)
            
            if use_energy_noise:
                energy_np = extract_array_safe(arrs[energy_name], global_to_orig, num_cells_global)
                noise_np = extract_array_safe(arrs[noise_name], global_to_orig, num_cells_global, default_value=1.0)
                
                energy_raw_dset[chunk_start:chunk_end] = energy_np
                noise_dset[chunk_start:chunk_end] = noise_np
                
                # Vectorized SNR computation (fixed warning)
                snr_computed = np.zeros_like(energy_np)
                np.divide(energy_np, noise_np, out=snr_computed, where=(noise_np != 0))
                snr_computed[~np.isfinite(snr_computed)] = 0
                snr_computed_dset[chunk_start:chunk_end] = snr_computed
                
                if snr_name and snr_np is not None:
                    snr_raw_dset[chunk_start:chunk_end] = snr_np
                
                if args.normalize_energy:
                    energy_norm = (energy_np - energy_center) / energy_scale
                    energy_norm_dset[chunk_start:chunk_end] = energy_norm
            
            elif snr_np is not None:
                snr_raw_dset[chunk_start:chunk_end] = snr_np
            
            # Extract clusters
            if cluster_name:
                cluster_np = extract_array_safe(arrs[cluster_name], global_to_orig, num_cells_global, dtype=np.int32)
            else:
                cluster_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.int32)
            
            cluster_dset[chunk_start:chunk_end] = cluster_np
            
            # VECTORIZED LABEL GENERATION
            if num_pairs > 0:
                # Get cluster assignments for source and destination cells of all edges
                c0_all = cluster_np[:, edge_src]  # shape: (chunk_size, num_pairs)
                c1_all = cluster_np[:, edge_dst]  # shape: (chunk_size, num_pairs)
                
                # Boolean masks for different conditions
                c0_in_cluster = (c0_all >= 1)
                c1_in_cluster = (c1_all >= 1)
                c0_lone = (c0_all == 0)
                c1_lone = (c1_all == 0)
                same_cluster = (c0_all == c1_all) & c0_in_cluster & c1_in_cluster
                diff_cluster = (c0_all != c1_all) & c0_in_cluster & c1_in_cluster
                
                # Initialize labels to 0 (both lone)
                labels_chunk = np.zeros((chunk_size_actual, num_pairs), dtype=np.int8)
                
                # Apply conditions (mutually exclusive, order doesn't matter)
                labels_chunk[same_cluster] = 1
                labels_chunk[c0_in_cluster & c1_lone] = 2
                labels_chunk[c0_lone & c1_in_cluster] = 3
                labels_chunk[diff_cluster] = 4
                
                labels_mmap[chunk_start:chunk_end] = labels_chunk
            
            # Cleanup
            del arrs
            gc.collect()
    
    # Finalize labels
    labels_mmap.flush()
    labels_mmap._mmap.close()
    
    if os.path.exists(labels_mmap_path):
        if num_pairs > 0:
            final_labels = np.memmap(labels_mmap_path, dtype=np.int8, mode='r', shape=(num_events, num_pairs))
            np.save(labels_file, final_labels)
            del final_labels
        else:
            np.save(labels_file, np.zeros((num_events, 0), dtype=np.int8))
        os.remove(labels_mmap_path)
    
    f.close()
    
    # File statistics
    elapsed = time.time() - t_start
    events_size = events_file.stat().st_size / 1024**3 if events_file.exists() else 0
    labels_size = labels_file.stat().st_size / 1024**3 if labels_file.exists() else 0
    
    log(f"  ✅ File processed in {format_time(elapsed)}")
    log(f"     Output: events={events_size:.2f} GB, labels={labels_size:.2f} GB")
    log(f"     Speed: {num_events/elapsed:.1f} events/second")
    
    return num_events, elapsed

# ------------------ MAIN FUNCTION ------------------
def main():
    overall_start = time.time()
    
    args = parse_args()
    
    log("\n" + "="*70)
    log("MULTI-FILE DATASET BUILDER (VECTORIZED)")
    log("="*70)
    log(f"Input directory: {args.input_dir}")
    log(f"Input pattern:   {args.input_pattern}")
    log(f"Output directory: {args.output_dir}")
    log(f"Chunk size:      {args.chunk_size}")
    log(f"Max files:       {args.max_files if args.max_files else 'All'}")
    log(f"[MEMORY] Initial: {get_memory_usage():.2f} GB")
    
    # Discover input files    
    input_files = discover_input_files(args)
    
    # Setup output directory
    dataset_name = get_dataset_name(input_files[0], args.dataset_name)
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"\nDataset name: {dataset_name}")
    log(f"Output directory: {output_dir}")
    
    # Define key branch names
    key_branch_names = {
        "cell_energy": "cell_e",
        "cell_SNR": "cell_SNR",
        "cell_noiseSigma": "cell_noiseSigma",
        "cell_eta": "cell_eta",
        "cell_phi": "cell_phi",
        "cell_cluster_index": "cell_cluster_index",
        "neighbor": "neighbor",
        "cell_subCalo": "cell_subCalo",
        "cell_sampling": "cell_sampling"
    }
    
    # PHASE 1: Build static geometry from first file
    pairs_array, cells_struct, global_to_orig, num_cells_global, use_energy_noise, _ = \
        build_static_geometry(input_files[0], args, key_branch_names)
    
    # Save static files
    np.save(output_dir / f"pairs_{dataset_name}.npy", pairs_array)
    np.save(output_dir / f"cells_{dataset_name}.npy", cells_struct)
    log(f"\n✅ Saved static files:")
    log(f"   pairs_{dataset_name}.npy ({pairs_array.nbytes/1024**2:.1f} MB)")
    log(f"   cells_{dataset_name}.npy ({cells_struct.nbytes/1024**2:.1f} MB)")
    
    gc.collect()
    
    # PHASE 2: Fit energy scaler across all files
    energy_center, energy_scale = fit_energy_scaler_all_files(
        input_files, args, key_branch_names, global_to_orig, num_cells_global
    )
    
    # Save scaler
    scaler_info = {'center': energy_center, 'scale': energy_scale, 'method': args.energy_normalization}
    with open(output_dir / f"scaler_{dataset_name}.pkl", 'wb') as f:
        pickle.dump(scaler_info, f)
    log(f"✅ Saved scaler_{dataset_name}.pkl")
    
    # PHASE 3: Process each file
    log("\n" + "="*70)
    log(f"PHASE 3: PROCESSING {len(input_files)} FILES")
    log("="*70)
    
    file_timings = []
    total_events = 0
    
    for file_idx, input_file in enumerate(input_files):
        # Generate output name from input filename
        file_output_name = input_file.stem
        
        n_events, file_time = process_single_file(
            input_file, output_dir, file_output_name,
            pairs_array, cells_struct, global_to_orig,
            num_cells_global, energy_center, energy_scale,
            use_energy_noise, key_branch_names, args,
            file_idx, len(input_files)
        )
        
        total_events += n_events
        file_timings.append({
            'file': input_file.name,
            'output_name': file_output_name,
            'events': n_events,
            'time_seconds': file_time,
            'time_formatted': format_time(file_time),
            'events_per_second': n_events / file_time if file_time > 0 else 0
        })
        
        gc.collect()
        log(f"    [MEMORY] After file {file_idx+1}: {get_memory_usage():.2f} GB")
    
    # Save metadata
    overall_time = time.time() - overall_start
    metadata = {
        'dataset_name': dataset_name,
        'input_directory': str(args.input_dir),
        'input_pattern': args.input_pattern,
        'input_files': [str(f) for f in input_files],
        'output_dir': str(output_dir),
        'total_events': total_events,
        'num_files': len(input_files),
        'num_cells': int(num_cells_global),
        'num_pairs': int(pairs_array.shape[0]),
        'energy_center': float(energy_center),
        'energy_scale': float(energy_scale),
        'normalization_method': args.energy_normalization,
        'processing_time_seconds': overall_time,
        'processing_time_formatted': format_time(overall_time),
        'file_timings': file_timings,
        'outputs': {
            'pairs': f"pairs_{dataset_name}.npy",
            'cells': f"cells_{dataset_name}.npy",
            'scaler': f"scaler_{dataset_name}.pkl",
            'event_files': [f"events_{f.stem}.h5" for f in input_files],
            'label_files': [f"labels_{f.stem}.npy" for f in input_files],
            'metadata': f"metadata_{dataset_name}.json"
        }
    }
    
    with open(output_dir / f"metadata_{dataset_name}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    log("\n" + "="*70)
    log("🎉 PROCESSING COMPLETE!")
    log("="*70)
    log(f"📁 Dataset: {dataset_name}")
    log(f"📂 Output directory: {output_dir}")
    log(f"")
    log(f"📊 Overall Statistics:")
    log(f"   Files processed:  {len(input_files)}")
    log(f"   Total events:     {total_events:,}")
    log(f"   Cells per event:  {num_cells_global:,}")
    log(f"   Edges per event:  {pairs_array.shape[0]:,}")
    log(f"   Total time:       {format_time(overall_time)}")
    log(f"   Avg speed:        {total_events/overall_time:.1f} events/s")
    log(f"")
    log(f"⏱️  Per-file timings:")
    for ft in file_timings:
        log(f"   {ft['file']}: {ft['time_formatted']} ({ft['events']:,} events, {ft['events_per_second']:.1f} evt/s)")
    log(f"")
    log(f"📄 Output files in {output_dir}:")
    log(f"   Static:  pairs_{dataset_name}.npy, cells_{dataset_name}.npy")
    log(f"   Scaler:  scaler_{dataset_name}.pkl")
    log(f"   Events:  {len(input_files)} events_*.h5 files")
    log(f"   Labels:  {len(input_files)} labels_*.npy files")
    log(f"   Meta:    metadata_{dataset_name}.json")
    log("="*70)
    log(f"[MEMORY] Final: {get_memory_usage():.2f} GB")

if __name__ == "__main__":
    main()  
