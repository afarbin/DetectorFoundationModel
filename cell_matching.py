"""Match calorimeter cells between 5D ntuples (SLAC) and Calo ntuples.

The Calo ntuples (TTree ``analysis``) store the *full* list of calorimeter
cells per event, including ``cell_hashID``.  The 5D ntuples (TTree ``ntuple``)
store a per-event *subset* of cells and carry no hash ID, so cells cannot be
identified directly.  Both ntuples, however, store the cell-centre position
(x, y, z), (eta, phi) and the sampling-layer index, all of which come from the
same detector description.  A 5D cell can therefore be identified by finding
the Calo-ntuple cell at the same position within the same sampling layer.

Typical use::

    import uproot
    from cell_matching import CaloCellGeometry, CellMatcher, CellIDMap

    calo = uproot.open("ttbar_100events.root")["analysis"]
    fived = uproot.open("5d_ntuple.root")["ntuple"]

    geo = CaloCellGeometry.from_tree(calo, entry=0)
    matcher = CellMatcher(geo)

    id_map = CellIDMap()
    for entry in range(5):
        cell_ids, result = match_5d_event(fived, matcher, entry)
        print(result.summary())
        id_map.update(cell_ids, result)

    id_map.save("cellid_to_hashid.npz")

    # later / elsewhere: O(1) lookup, no geometry needed
    id_map = CellIDMap.load("cellid_to_hashid.npz")
    hash_ids = id_map.lookup(cell_ids)

Distances are in mm (ATLAS convention).  The default tolerance of 1 mm is far
below the smallest cell pitch (~4 mm for EMB1 strips) but far above float32
rounding of the stored coordinates (<~1e-3 mm), so matches are unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:  # scipy is optional; a pure-numpy fallback is used if it is missing
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None


# ATLAS CaloSampling::CaloSample enumeration
CALO_SAMPLING_NAMES = {
    0: "PreSamplerB", 1: "EMB1", 2: "EMB2", 3: "EMB3",
    4: "PreSamplerE", 5: "EME1", 6: "EME2", 7: "EME3",
    8: "HEC0", 9: "HEC1", 10: "HEC2", 11: "HEC3",
    12: "TileBar0", 13: "TileBar1", 14: "TileBar2",
    15: "TileGap1", 16: "TileGap2", 17: "TileGap3",
    18: "TileExt0", 19: "TileExt1", 20: "TileExt2",
    21: "FCAL0", 22: "FCAL1", 23: "FCAL2",
}


def sampling_name(s: int) -> str:
    return CALO_SAMPLING_NAMES.get(int(s), f"Unknown({int(s)})")


def _to_numpy(array_like) -> np.ndarray:
    """Convert an awkward/numpy array (one event's flat list) to numpy."""
    try:
        import awkward as ak
        if isinstance(array_like, (ak.Array, ak.Record)):
            return ak.to_numpy(array_like)
    except Exception:
        pass
    return np.asarray(array_like)


class _BruteForceNN:
    """Chunked brute-force nearest neighbour, used when scipy is absent.

    Distances use the expansion |q-p|^2 = |q|^2 + |p|^2 - 2 q.p so peak memory
    is one (chunk, n_points) block instead of a (chunk, n_points, 3) broadcast.
    The cancellation error of the expansion (~1e-4 mm at ATLAS coordinate
    scales) is far below any matching tolerance.
    """

    def __init__(self, points: np.ndarray):
        self._points = np.ascontiguousarray(points, dtype=np.float64)
        self._p2 = (self._points ** 2).sum(axis=1)

    def query(self, q: np.ndarray, chunk: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        dist = np.empty(len(q), dtype=np.float64)
        idx = np.empty(len(q), dtype=np.int64)
        for lo in range(0, len(q), chunk):
            block = q[lo:lo + chunk]
            d2 = ((block ** 2).sum(axis=1)[:, None] + self._p2[None, :]
                  - 2.0 * block @ self._points.T)
            j = np.argmin(d2, axis=1)
            idx[lo:lo + chunk] = j
            dist[lo:lo + chunk] = np.sqrt(
                np.maximum(d2[np.arange(len(block)), j], 0.0))
        return dist, idx


def _make_nn(points: np.ndarray):
    if _KDTree is not None:
        return _KDTree(points)
    return _BruteForceNN(points)


@dataclass
class CaloCellGeometry:
    """The full cell list of the calorimeter, read from a Calo ntuple event."""

    hash_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    sampling: np.ndarray
    sub_calo: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.hash_id)

    @classmethod
    def from_tree(cls, tree, entry: int = 0, prefix: str = "cell_") -> "CaloCellGeometry":
        """Read the cell list from one entry of a Calo ntuple TTree."""
        names = {
            "hash_id": prefix + "hashID",
            "x": prefix + "x", "y": prefix + "y", "z": prefix + "z",
            "eta": prefix + "eta", "phi": prefix + "phi",
            "sampling": prefix + "sampling",
        }
        optional = {"sub_calo": prefix + "subCalo"}
        available = set(tree.keys())
        to_read = [b for b in names.values() if b in available]
        missing = [b for b in names.values() if b not in available]
        if missing:
            raise KeyError(f"Calo ntuple is missing branches: {missing}")
        for attr, b in optional.items():
            if b in available:
                to_read.append(b)
        arrays = tree.arrays(to_read, entry_start=entry, entry_stop=entry + 1)
        kwargs = {}
        for attr, b in {**names, **optional}.items():
            if b not in to_read:
                continue
            v = _to_numpy(arrays[b][0])
            if attr in ("hash_id", "sampling", "sub_calo"):
                v = v.astype(np.int64)
            else:
                v = v.astype(np.float64)
            kwargs[attr] = v
        return cls(**kwargs)

    @staticmethod
    def is_static_across_events(tree, entries: Iterable[int] = (0, 1, 2),
                                prefix: str = "cell_") -> bool:
        """True if the cell list (hashID sequence) is identical across entries."""
        entries = [e for e in entries if e < tree.num_entries]
        if len(entries) < 2:
            return True
        branch = tree[prefix + "hashID"]
        ref = _to_numpy(branch.array(entry_start=entries[0],
                                     entry_stop=entries[0] + 1)[0])
        for e in entries[1:]:
            v = _to_numpy(branch.array(entry_start=e, entry_stop=e + 1)[0])
            if len(v) != len(ref) or not np.array_equal(v, ref):
                return False
        return True


@dataclass
class MatchResult:
    """Result of matching one set of query cells against the calo geometry.

    ``index`` is the index into the ``CaloCellGeometry`` arrays (-1 when
    unmatched), ``hash_id`` the matched cell's hash ID (-1 when unmatched),
    ``distance`` the match distance (mm for xyz matching, eta-phi metric
    otherwise; inf when no candidate existed).
    """

    index: np.ndarray
    hash_id: np.ndarray
    distance: np.ndarray
    matched: np.ndarray
    sampling: np.ndarray
    tolerance: float
    metric: str = "xyz"

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def n_matched(self) -> int:
        return int(self.matched.sum())

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n if self.n else float("nan")

    def duplicate_calo_indices(self) -> np.ndarray:
        """Calo cell indices claimed by more than one query cell (should be empty)."""
        idx = self.index[self.matched]
        uniq, counts = np.unique(idx, return_counts=True)
        return uniq[counts > 1]

    def per_sampling(self):
        """Per-sampling stats as a list of dicts."""
        rows = []
        for s in np.unique(self.sampling):
            sel = self.sampling == s
            m = self.matched & sel
            d = self.distance[m]
            rows.append({
                "sampling": int(s),
                "name": sampling_name(s),
                "n_query": int(sel.sum()),
                "n_matched": int(m.sum()),
                "max_dist": float(d.max()) if len(d) else float("nan"),
                "median_dist": float(np.median(d)) if len(d) else float("nan"),
            })
        return rows

    def summary(self) -> str:
        unit = "mm" if self.metric == "xyz" else ""
        lines = [
            f"matched {self.n_matched}/{self.n} cells "
            f"({100.0 * self.match_rate:.2f}%)  "
            f"[metric={self.metric}, tolerance={self.tolerance:g}{unit}]",
            f"{'sampling':<14}{'n_query':>8}{'n_match':>8}{'median_d':>12}{'max_d':>12}",
        ]
        for r in self.per_sampling():
            lines.append(
                f"{r['name']:<14}{r['n_query']:>8}{r['n_matched']:>8}"
                f"{r['median_dist']:>12.4g}{r['max_dist']:>12.4g}"
            )
        dup = self.duplicate_calo_indices()
        if len(dup):
            lines.append(f"WARNING: {len(dup)} calo cells matched by >1 query cell")
        return "\n".join(lines)


class CellMatcher:
    """Nearest-neighbour matcher from cell positions to the full calo cell list.

    One KD-tree is built per sampling layer, so a query cell can only ever
    match a calo cell in the same layer.  Primary matching uses (x, y, z) in
    mm; ``match_eta_phi`` is a fallback for inputs without positions (it uses
    the metric sqrt(deta^2 + (2 sin(dphi/2))^2), which handles phi wrap).
    """

    def __init__(self, geometry: CaloCellGeometry, tolerance_mm: float = 1.0):
        self.geometry = geometry
        self.tolerance_mm = float(tolerance_mm)
        self._xyz_trees: Dict[int, Tuple[np.ndarray, object]] = {}
        self._etaphi_trees: Dict[int, Tuple[np.ndarray, object]] = {}
        for s in np.unique(geometry.sampling):
            sel = np.where(geometry.sampling == s)[0]
            pts = np.column_stack([geometry.x[sel], geometry.y[sel], geometry.z[sel]])
            self._xyz_trees[int(s)] = (sel, _make_nn(pts))

    @property
    def samplings(self):
        """Sampling layers present in the calo geometry."""
        return sorted(self._xyz_trees)

    def _etaphi_tree(self, s: int):
        if s not in self._etaphi_trees:
            geo = self.geometry
            sel = np.where(geo.sampling == s)[0]
            pts = np.column_stack([geo.eta[sel], np.cos(geo.phi[sel]), np.sin(geo.phi[sel])])
            self._etaphi_trees[s] = (sel, _make_nn(pts))
        return self._etaphi_trees[s]

    def match_xyz(self, x, y, z, sampling,
                  tolerance_mm: Optional[float] = None) -> MatchResult:
        """Match query cells by position.  All arguments are flat arrays."""
        tol = self.tolerance_mm if tolerance_mm is None else float(tolerance_mm)
        x = _to_numpy(x).astype(np.float64)
        y = _to_numpy(y).astype(np.float64)
        z = _to_numpy(z).astype(np.float64)
        sampling = _to_numpy(sampling).astype(np.int64)
        n = len(x)
        index = np.full(n, -1, dtype=np.int64)
        distance = np.full(n, np.inf, dtype=np.float64)
        for s in np.unique(sampling):
            q = np.where(sampling == s)[0]
            if int(s) not in self._xyz_trees:
                continue  # no calo cells in this sampling -> stays unmatched
            sel, tree = self._xyz_trees[int(s)]
            d, j = tree.query(np.column_stack([x[q], y[q], z[q]]))
            index[q] = sel[np.atleast_1d(j)]
            distance[q] = np.atleast_1d(d)
        matched = distance <= tol
        hash_id = np.where(matched, self.geometry.hash_id[np.clip(index, 0, None)], -1)
        index = np.where(matched, index, -1)
        return MatchResult(index=index, hash_id=hash_id, distance=distance,
                           matched=matched, sampling=sampling,
                           tolerance=tol, metric="xyz")

    def match_eta_phi(self, eta, phi, sampling,
                      tolerance: float = 0.0015) -> MatchResult:
        """Fallback matching in (eta, phi).  Tolerance is in the eta-phi metric
        (default 0.0015, half the finest strip granularity deta=0.003)."""
        eta = _to_numpy(eta).astype(np.float64)
        phi = _to_numpy(phi).astype(np.float64)
        sampling = _to_numpy(sampling).astype(np.int64)
        n = len(eta)
        index = np.full(n, -1, dtype=np.int64)
        distance = np.full(n, np.inf, dtype=np.float64)
        for s in np.unique(sampling):
            q = np.where(sampling == s)[0]
            if int(s) not in self._xyz_trees:
                continue
            sel, tree = self._etaphi_tree(int(s))
            d, j = tree.query(np.column_stack([eta[q], np.cos(phi[q]), np.sin(phi[q])]))
            index[q] = sel[np.atleast_1d(j)]
            distance[q] = np.atleast_1d(d)
        matched = distance <= tolerance
        hash_id = np.where(matched, self.geometry.hash_id[np.clip(index, 0, None)], -1)
        index = np.where(matched, index, -1)
        return MatchResult(index=index, hash_id=hash_id, distance=distance,
                           matched=matched, sampling=sampling,
                           tolerance=tolerance, metric="etaphi")


def match_5d_event(tree5d, matcher: CellMatcher, entry: int,
                   prefix: str = "Cell_") -> Tuple[np.ndarray, MatchResult]:
    """Match all cells of one 5D-ntuple event.

    Returns ``(cell_ids, MatchResult)`` where ``cell_ids`` is the event's
    ``Cell_ID`` array (int64), aligned with the MatchResult arrays.
    """
    branches = [prefix + b for b in ("ID", "x", "y", "z", "sampling")]
    arrays = tree5d.arrays(branches, entry_start=entry, entry_stop=entry + 1)
    cell_ids = _to_numpy(arrays[prefix + "ID"][0]).astype(np.int64)
    result = matcher.match_xyz(
        arrays[prefix + "x"][0], arrays[prefix + "y"][0],
        arrays[prefix + "z"][0], arrays[prefix + "sampling"][0],
    )
    return cell_ids, result


@dataclass
class CellIDMap:
    """Persistent Cell_ID -> hashID mapping, accumulated over matched events.

    Once built, lookups are O(log n) and need no geometry at all.  Two kinds
    of inconsistency are recorded as the map is filled (both should stay empty
    if matching is correct) and survive ``save``/``load``:

    - ``conflicts``: a Cell_ID re-matched to a *different* hashID.  The
      Cell_ID is removed from the map (``lookup`` returns -1 for it) rather
      than silently keeping either candidate.
    - ``duplicate_targets``: a new Cell_ID matched to a hashID some other
      Cell_ID already owns (non-injective).  Both entries are kept, but the
      collision is recorded; ``is_injective()`` is the summary check.
    """

    _map: Dict[int, int] = field(default_factory=dict)
    conflicts: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    duplicate_targets: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _hash_owner: Dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _sorted_cache: Optional[Tuple[np.ndarray, np.ndarray]] = field(
        default=None, repr=False, compare=False)

    @property
    def n(self) -> int:
        return len(self._map)

    @property
    def clean(self) -> bool:
        """True if no conflicts or duplicate-target collisions were recorded."""
        return not self.conflicts and not self.duplicate_targets

    def update(self, cell_ids: np.ndarray, result: MatchResult) -> int:
        """Add matched cells from one event; returns number of new entries."""
        new = 0
        self._sorted_cache = None
        ids = np.asarray(cell_ids, dtype=np.int64)[result.matched]
        hids = result.hash_id[result.matched]
        for cid, hid in zip(ids.tolist(), hids.tolist()):
            if cid in self.conflicts:
                continue  # known-bad ID stays unmapped
            prev = self._map.get(cid)
            if prev is None:
                owner = self._hash_owner.get(hid)
                if owner is not None:
                    self.duplicate_targets[cid] = (owner, hid)
                self._map[cid] = hid
                self._hash_owner.setdefault(hid, cid)
                new += 1
            elif prev != hid:
                self.conflicts[cid] = (prev, hid)
                del self._map[cid]
        return new

    def items_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """The mapping as ``(cell_ids, hash_ids)`` arrays, sorted by cell_id."""
        if self._sorted_cache is None:
            keys = np.fromiter(self._map.keys(), dtype=np.int64, count=len(self._map))
            vals = np.fromiter(self._map.values(), dtype=np.int64, count=len(self._map))
            order = np.argsort(keys)
            self._sorted_cache = (keys[order], vals[order])
        return self._sorted_cache

    def is_injective(self) -> bool:
        """True if no two Cell_IDs map to the same hashID."""
        _, vals = self.items_arrays()
        return len(np.unique(vals)) == len(vals)

    def lookup(self, cell_ids) -> np.ndarray:
        """Vectorised Cell_ID -> hashID lookup; -1 for unknown IDs."""
        ids = np.asarray(_to_numpy(cell_ids), dtype=np.int64)
        keys, vals = self.items_arrays()
        if not len(keys):
            return np.full(len(ids), -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(keys, ids), 0, len(keys) - 1)
        return np.where(keys[pos] == ids, vals[pos], -1)

    @staticmethod
    def _norm_path(path: str) -> str:
        # np.savez appends ".npz" itself; normalise so save/load agree
        path = str(path)
        return path if path.endswith(".npz") else path + ".npz"

    def save(self, path: str) -> None:
        if not self.clean:
            import warnings
            warnings.warn(
                f"saving CellIDMap with {len(self.conflicts)} conflicts and "
                f"{len(self.duplicate_targets)} duplicate targets - they are "
                f"persisted and will be restored by load()")
        keys, vals = self.items_arrays()
        as_arr = lambda d: np.array(  # dict -> (n, 3) rows of [key, v0, v1]
            [[k, v[0], v[1]] for k, v in d.items()], dtype=np.int64).reshape(-1, 3)
        np.savez_compressed(self._norm_path(path), cell_id=keys, hash_id=vals,
                            conflicts=as_arr(self.conflicts),
                            duplicate_targets=as_arr(self.duplicate_targets))

    @classmethod
    def load(cls, path: str) -> "CellIDMap":
        obj = cls()
        with np.load(cls._norm_path(path)) as data:
            obj._map = dict(zip(data["cell_id"].tolist(), data["hash_id"].tolist()))
            for name, target in (("conflicts", obj.conflicts),
                                 ("duplicate_targets", obj.duplicate_targets)):
                if name in data.files:
                    for k, v0, v1 in data[name].tolist():
                        target[k] = (v0, v1)
        for cid, hid in obj._map.items():
            obj._hash_owner.setdefault(hid, cid)
        return obj
