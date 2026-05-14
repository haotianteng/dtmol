"""Build a synthetic crystal/unit-cell dataset in UnifiedRecord LMDB format.

Output:
    /data/dtMol_Project/datasets/unit_cell_synthesized/{train,valid,test}.lmdb

Each record encodes a supercell of a known crystal lattice with realistic lattice
constants, small thermal noise on positions, and is structurally one component
(no protein / ligand split). All records have ``pes_tier='C'`` since no DFT
energies/forces are available.

Element substitutions vs. the "canonical" textbook crystals
-----------------------------------------------------------
The diffusion model only sees atomic numbers it has embeddings for, namely the
26 entries in ``dtmol/data/atom_mapping.json``. Several common metals (Cu, Mg,
Zn-as-metal, Ti, Ni) are absent. We substitute lattice templates with supported
elements; the result is a *toy* crystal whose atom indices are valid even if the
"real" element wouldn't crystallise that way. The diffusion target is purely
geometric (recognisable unit-cell motifs) so this is fine.

Substitutions:
- FCC / BCC pure metal: Al (13, real FCC, a=4.05 A), Au (79, real FCC, a=4.08 A),
  Fe (26, real BCC, a=2.87 A), Ca (20, real FCC, a=5.58 A).
- Diamond cubic: C-diamond (6, a=3.567 A), Si (14, a=5.43 A), Sn (50, alpha-Sn,
  a=6.49 A).
- Graphite (hexagonal layers): pure C (a=2.46 A, c=6.71 A).
- Zinc blende: SiC (Si-C, a=4.36 A), BN-cubic (B-N, a=3.62 A), AlP (Al-P, a=5.46 A).
- Wurtzite (hexagonal): BN-wurtzite (B-N, a=2.55 A, c=4.20 A), AlN-toy (Al-N).
- Rocksalt: NaCl (Na-Cl, a=5.64 A), LiF (Li-F, a=4.03 A), KBr (K-Br, a=6.60 A).
- HCP: toy Ca-HCP (a=3.95 A, c=6.45 A) and toy Zn (Zn=30, a=2.66 A, c=4.95 A).
- Simple cubic: toy "Po-substitute" using Se (a=2.97 A) — pure variety filler.

Run
---
    python3 scripts/build_unit_cell_dataset.py [--records-per-type 800]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtmol.data.converters.base import BaseConverter, UnifiedRecord  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_unit_cell_dataset")

OUT_DIR = Path("/data/dtMol_Project/datasets/unit_cell_synthesized")
ATOM_MAPPING_PATH = REPO_ROOT / "dtmol" / "data" / "atom_mapping.json"

DATASET_SOURCE = "unit_cell_synth"
ATOM_COUNT_MIN = 20
ATOM_COUNT_MAX = 300
NEIGHBOR_CUTOFF = 5.0
THERMAL_SIGMA = 0.04
LATTICE_JITTER = 0.05  # +/-5 % around target
SEED = 20260502

# ---------- crystal motif builders ----------------------------------------


def _supercell_from_basis(
    frac_basis: np.ndarray,
    elements: List[int],
    lattice_vecs: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate a fractional-coordinate basis on an n_x x n_y x n_z grid.

    Args:
        frac_basis: (B, 3) fractional coordinates inside the unit cell, range [0, 1).
        elements: list of length B with atomic numbers Z for each basis atom.
        lattice_vecs: (3, 3) lattice basis vectors in Angstrom (rows are a, b, c).
        nx, ny, nz: integer supercell tiling along each lattice direction.

    Returns:
        atom_types (M,) int64 array, positions (M, 3) float64 in Angstrom.
        M = B * nx * ny * nz.
    """
    assert frac_basis.shape[0] == len(elements)
    cells = []
    types = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i, j, k], dtype=np.float64)
                cells.append(frac_basis + shift)
                types.extend(elements)
    frac_all = np.concatenate(cells, axis=0)
    pos = frac_all @ lattice_vecs
    atom_types = np.array(types, dtype=np.int64)
    return atom_types, pos.astype(np.float64)


def _cubic_lattice(a: float) -> np.ndarray:
    return np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=np.float64)


def _hex_lattice(a: float, c: float) -> np.ndarray:
    # Hexagonal lattice with conventional orientation: a1 along x, a2 in xy plane.
    return np.array([
        [a, 0.0, 0.0],
        [-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ], dtype=np.float64)


# Each builder returns (frac_basis, elements, lattice_vecs_func, atoms_per_cell)
# ``lattice_vecs_func(a_or_pair)`` returns the lattice matrix.

_FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
])
_BCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])
_DIAMOND_BASIS = np.concatenate([
    _FCC_BASIS,
    _FCC_BASIS + np.array([0.25, 0.25, 0.25]),
], axis=0)
_ZB_BASIS_A = _FCC_BASIS                                  # cation sites
_ZB_BASIS_B = _FCC_BASIS + np.array([0.25, 0.25, 0.25])    # anion sites
_ROCKSALT_BASIS_A = _FCC_BASIS                              # cation
_ROCKSALT_BASIS_B = _FCC_BASIS + np.array([0.5, 0.5, 0.5])  # anion
_SC_BASIS = np.array([[0.0, 0.0, 0.0]])

# Hexagonal motifs (c-axis is the third lattice vector).
# Graphite: 4 atoms per conventional hexagonal cell (AB stacking).
_GRAPHITE_BASIS = np.array([
    [0.0,       0.0,       0.00],
    [1.0 / 3.0, 2.0 / 3.0, 0.00],
    [0.0,       0.0,       0.50],
    [2.0 / 3.0, 1.0 / 3.0, 0.50],
])
# HCP: 2 atoms per primitive hexagonal cell.
_HCP_BASIS = np.array([
    [0.0,       0.0,       0.00],
    [1.0 / 3.0, 2.0 / 3.0, 0.50],
])
# Wurtzite: 4 atoms per cell, two species alternating.
_WURTZITE_BASIS_A = np.array([
    [0.0,       0.0,       0.000],
    [1.0 / 3.0, 2.0 / 3.0, 0.500],
])
_WURTZITE_BASIS_B = np.array([
    [0.0,       0.0,       0.375],
    [1.0 / 3.0, 2.0 / 3.0, 0.875],
])


# Crystal type registry: each entry generates one record per call.
# tuple = (name, family, build_fn, supported_elements_or_pair, lattice_target)
def build_fcc(z: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=4, target_atoms=120,
                                          a_min=ATOM_COUNT_MIN, a_max=ATOM_COUNT_MAX)
    types, pos = _supercell_from_basis(_FCC_BASIS, [z] * 4, _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_bcc(z: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=2, target_atoms=120)
    types, pos = _supercell_from_basis(_BCC_BASIS, [z] * 2, _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_diamond(z: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=8, target_atoms=120)
    types, pos = _supercell_from_basis(_DIAMOND_BASIS, [z] * 8, _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_zincblende(z_a: int, z_b: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=8, target_atoms=120)
    basis = np.concatenate([_ZB_BASIS_A, _ZB_BASIS_B], axis=0)
    elements = [z_a] * 4 + [z_b] * 4
    types, pos = _supercell_from_basis(basis, elements, _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_rocksalt(z_a: int, z_b: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=8, target_atoms=120)
    basis = np.concatenate([_ROCKSALT_BASIS_A, _ROCKSALT_BASIS_B], axis=0)
    elements = [z_a] * 4 + [z_b] * 4
    types, pos = _supercell_from_basis(basis, elements, _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_simple_cubic(z: int, a: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx = ny = nz = _pick_cubic_supercell(atoms_per_cell=1, target_atoms=120)
    types, pos = _supercell_from_basis(_SC_BASIS, [z], _cubic_lattice(a), nx, ny, nz)
    return types, pos, (nx, ny, nz), _cubic_lattice(a) * np.array([nx, ny, nz])[:, None]


def build_graphite(z: int, a: float, c: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    # Graphite has 4 atoms per (hex) cell. Pick supercell that keeps in atom band.
    target_atoms = 140
    nx, ny, nz = _pick_hex_supercell(atoms_per_cell=4, target_atoms=target_atoms, c_pref=2)
    types, pos = _supercell_from_basis(_GRAPHITE_BASIS, [z] * 4, _hex_lattice(a, c), nx, ny, nz)
    return types, pos, (nx, ny, nz), _hex_lattice(a, c) * np.array([nx, ny, nz])[:, None]


def build_hcp(z: int, a: float, c: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx, ny, nz = _pick_hex_supercell(atoms_per_cell=2, target_atoms=140, c_pref=3)
    types, pos = _supercell_from_basis(_HCP_BASIS, [z] * 2, _hex_lattice(a, c), nx, ny, nz)
    return types, pos, (nx, ny, nz), _hex_lattice(a, c) * np.array([nx, ny, nz])[:, None]


def build_wurtzite(z_a: int, z_b: int, a: float, c: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    nx, ny, nz = _pick_hex_supercell(atoms_per_cell=4, target_atoms=140, c_pref=2)
    basis = np.concatenate([_WURTZITE_BASIS_A, _WURTZITE_BASIS_B], axis=0)
    elements = [z_a] * 2 + [z_b] * 2
    types, pos = _supercell_from_basis(basis, elements, _hex_lattice(a, c), nx, ny, nz)
    return types, pos, (nx, ny, nz), _hex_lattice(a, c) * np.array([nx, ny, nz])[:, None]


_CUBIC_SUPERCELL_RNG: List[np.random.Generator] = []


def _set_supercell_rng(rng: np.random.Generator) -> None:
    """Install a global RNG used by the supercell pickers so each call samples
    a random valid (n_x, n_y, n_z) instead of always returning the largest."""
    if _CUBIC_SUPERCELL_RNG:
        _CUBIC_SUPERCELL_RNG[0] = rng
    else:
        _CUBIC_SUPERCELL_RNG.append(rng)


def _rng() -> np.random.Generator:
    return _CUBIC_SUPERCELL_RNG[0]


def _pick_cubic_supercell(atoms_per_cell: int, target_atoms: int = 120,
                           a_min: int = ATOM_COUNT_MIN, a_max: int = ATOM_COUNT_MAX) -> int:
    """Pick a random cubic n with n^3 * atoms_per_cell in [a_min, a_max]."""
    candidates = [n for n in range(1, 12)
                  if a_min <= n * n * n * atoms_per_cell <= a_max]
    if not candidates:
        # Fallback: pick the smallest n meeting a_min, even if above a_max.
        for n in range(1, 12):
            if n * n * n * atoms_per_cell >= a_min:
                return n
        return 1
    return int(_rng().choice(candidates))


def _pick_hex_supercell(atoms_per_cell: int, target_atoms: int = 140, c_pref: int = 2) -> Tuple[int, int, int]:
    """Pick a random valid (n_x, n_y, n_z) for a hexagonal cell.

    We constrain n_x == n_y for visual symmetry but allow n_z to vary
    independently, since c-axis tiling is structurally meaningful (number
    of stacked layers in graphite, etc.)."""
    candidates: List[Tuple[int, int, int]] = []
    for n_lat in range(1, 8):
        for n_c in range(1, 8):
            total = n_lat * n_lat * n_c * atoms_per_cell
            if ATOM_COUNT_MIN <= total <= ATOM_COUNT_MAX:
                candidates.append((n_lat, n_lat, n_c))
    if not candidates:
        return (1, 1, 1)
    idx = int(_rng().integers(len(candidates)))
    return candidates[idx]


# ---------- crystal type definitions --------------------------------------

# Each spec: (system_id_prefix, builder, args (Z or (Z_a, Z_b)), lattice target(s))
# lattice target is `a` for cubic types and `(a, c)` for hexagonal types.

CRYSTAL_SPECS: List[dict] = [
    # FCC pure metals (real FCC)
    {"name": "fcc_Al",   "family": "fcc",        "fn": build_fcc,        "z": 13,            "a": 4.05},
    {"name": "fcc_Au",   "family": "fcc",        "fn": build_fcc,        "z": 79,            "a": 4.08},
    {"name": "fcc_Ca",   "family": "fcc",        "fn": build_fcc,        "z": 20,            "a": 5.58},
    # BCC pure metals
    {"name": "bcc_Fe",   "family": "bcc",        "fn": build_bcc,        "z": 26,            "a": 2.87},
    {"name": "bcc_Cr",   "family": "bcc",        "fn": build_bcc,        "z": 24,            "a": 2.88},
    {"name": "bcc_K",    "family": "bcc",        "fn": build_bcc,        "z": 19,            "a": 5.33},
    # Diamond cubic
    {"name": "diamond_C",  "family": "diamond",  "fn": build_diamond,    "z": 6,             "a": 3.567},
    {"name": "diamond_Si", "family": "diamond",  "fn": build_diamond,    "z": 14,            "a": 5.43},
    {"name": "diamond_Sn", "family": "diamond",  "fn": build_diamond,    "z": 50,            "a": 6.49},
    # Zinc blende
    {"name": "zb_SiC",  "family": "zincblende",  "fn": build_zincblende, "z_a": 14, "z_b": 6,  "a": 4.36},
    {"name": "zb_BN",   "family": "zincblende",  "fn": build_zincblende, "z_a": 5,  "z_b": 7,  "a": 3.62},
    {"name": "zb_AlP",  "family": "zincblende",  "fn": build_zincblende, "z_a": 13, "z_b": 15, "a": 5.46},
    # Rocksalt
    {"name": "rs_NaCl", "family": "rocksalt",    "fn": build_rocksalt,   "z_a": 11, "z_b": 17, "a": 5.64},
    {"name": "rs_LiF",  "family": "rocksalt",    "fn": build_rocksalt,   "z_a": 3,  "z_b": 9,  "a": 4.03},
    {"name": "rs_KBr",  "family": "rocksalt",    "fn": build_rocksalt,   "z_a": 19, "z_b": 35, "a": 6.60},
    # Simple cubic (rare in nature; toy filler)
    {"name": "sc_Se",   "family": "simple_cubic", "fn": build_simple_cubic, "z": 34,         "a": 2.97},
    # Graphite
    {"name": "graphite_C", "family": "graphite", "fn": build_graphite,   "z": 6,             "a": 2.46, "c": 6.71},
    # HCP
    {"name": "hcp_Ca",  "family": "hcp",         "fn": build_hcp,        "z": 20,            "a": 3.95, "c": 6.45},
    {"name": "hcp_Zn",  "family": "hcp",         "fn": build_hcp,        "z": 30,            "a": 2.66, "c": 4.95},
    # Wurtzite
    {"name": "wz_BN",   "family": "wurtzite",    "fn": build_wurtzite,   "z_a": 5, "z_b": 7,  "a": 2.55, "c": 4.20},
    {"name": "wz_AlN",  "family": "wurtzite",    "fn": build_wurtzite,   "z_a": 13, "z_b": 7, "a": 3.11, "c": 4.98},
]


def _generate_record(spec: dict, rng: np.random.Generator, supported_zs: set, idx: int) -> UnifiedRecord:
    family = spec["family"]
    fn = spec["fn"]
    name = spec["name"]

    # Sample lattice constants near target (+/- 5%).
    a_target = spec["a"]
    a = float(a_target * (1.0 + (rng.random() * 2.0 - 1.0) * LATTICE_JITTER))

    if "c" in spec:
        c_target = spec["c"]
        c = float(c_target * (1.0 + (rng.random() * 2.0 - 1.0) * LATTICE_JITTER))

    # Build geometry.
    if family in {"fcc", "bcc", "diamond", "simple_cubic"}:
        atom_types, pos, supercell, _ = fn(spec["z"], a)
    elif family in {"zincblende", "rocksalt"}:
        atom_types, pos, supercell, _ = fn(spec["z_a"], spec["z_b"], a)
    elif family in {"graphite", "hcp"}:
        atom_types, pos, supercell, _ = fn(spec["z"], a, c)
    elif family == "wurtzite":
        atom_types, pos, supercell, _ = fn(spec["z_a"], spec["z_b"], a, c)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Sanity checks: all elements supported, atom count in band.
    bad_zs = set(int(z) for z in atom_types) - supported_zs
    assert not bad_zs, f"{name}: unsupported Zs {bad_zs}"
    assert ATOM_COUNT_MIN <= len(atom_types) <= ATOM_COUNT_MAX, (
        f"{name}: produced {len(atom_types)} atoms (allowed {ATOM_COUNT_MIN}-{ATOM_COUNT_MAX})"
    )

    # Add Gaussian thermal noise.
    noise = rng.normal(loc=0.0, scale=THERMAL_SIGMA, size=pos.shape)
    pos = pos + noise

    # Center positions roughly at origin to match other datasets' convention.
    pos = pos - pos.mean(axis=0, keepdims=True)

    pos = pos.astype(np.float64)
    neighbor_list = BaseConverter.compute_neighbor_list(pos, cutoff=NEIGHBOR_CUTOFF)

    sx, sy, sz = supercell
    system_id = f"{name}_{sx}x{sy}x{sz}_seed{idx:06d}"

    record: UnifiedRecord = {
        "atom_types": atom_types,
        "positions": pos,
        "num_atoms": int(len(atom_types)),
        "dataset_source": DATASET_SOURCE,
        "system_id": system_id,
        "pes_tier": "C",
        "forces": None,
        "noise_target": None,
        "noise_level": None,
        "energy": None,
        "binding_affinity": None,
        "relative_energy": None,
        "trajectory_id": None,
        "timestep": None,
        "positions_prev": None,
        "positions_next": None,
        "component_mask": None,
        "pocket_mask": None,
        "partial_charges": None,
        "dipole": None,
        "homo": None,
        "lumo": None,
        "neighbor_list": neighbor_list,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-per-type", type=int, default=300,
                        help="How many random variants to draw for each crystal type "
                             "(default 300 — total = 300 * num types).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Load supported Zs.
    with open(ATOM_MAPPING_PATH) as f:
        mapping = json.load(f)
    supported_zs = {int(k) for k in mapping.keys() if k.isdigit()}
    logger.info("Supported atomic numbers: %s", sorted(supported_zs))

    # Validate every spec uses supported elements.
    for spec in CRYSTAL_SPECS:
        zs = []
        if "z" in spec: zs.append(spec["z"])
        if "z_a" in spec: zs.append(spec["z_a"])
        if "z_b" in spec: zs.append(spec["z_b"])
        bad = [z for z in zs if z not in supported_zs]
        if bad:
            raise ValueError(f"Crystal {spec['name']} uses unsupported Zs {bad}")

    rng = np.random.default_rng(args.seed)
    _set_supercell_rng(rng)

    # Generate records.
    all_records: List[UnifiedRecord] = []
    type_counts: Counter = Counter()
    for spec in CRYSTAL_SPECS:
        for k in range(args.records_per_type):
            rec = _generate_record(spec, rng, supported_zs, idx=k)
            all_records.append(rec)
            type_counts[spec["name"]] += 1

    logger.info("Generated %d total records across %d crystal types",
                len(all_records), len(CRYSTAL_SPECS))

    # 80/10/10 random split.
    perm = rng.permutation(len(all_records))
    n = len(all_records)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    splits = {
        "train": perm[:n_train],
        "valid": perm[n_train:n_train + n_valid],
        "test":  perm[n_train + n_valid:],
    }

    # Write three LMDBs.
    for split_name, idxs in splits.items():
        recs = [all_records[i] for i in idxs]
        out_path = str(out_dir / f"{split_name}.lmdb")
        # Match BaseConverter.write_lmdb expectation: it creates dir if absent.
        # Remove any pre-existing data to avoid stale entries.
        p = Path(out_path)
        if p.exists():
            import shutil
            shutil.rmtree(p)
        BaseConverter.write_lmdb(recs, out_path)
        logger.info("Wrote %d %s records to %s", len(recs), split_name, out_path)

    # ---------- summary -----------------------------------------------------
    print()
    print("=" * 70)
    print("Synthetic unit-cell dataset summary")
    print("=" * 70)

    # Per-split per-type breakdown.
    split_type_counts: Dict[str, Counter] = defaultdict(Counter)
    for split_name, idxs in splits.items():
        for i in idxs:
            sid = all_records[i]["system_id"]
            ctype = sid.rsplit("_", 2)[0]  # e.g. fcc_Al
            split_type_counts[split_name][ctype] += 1

    type_names = sorted(type_counts.keys())
    header = f"{'crystal_type':<18} {'train':>7} {'valid':>7} {'test':>7} {'total':>7}"
    print(header)
    print("-" * len(header))
    for tn in type_names:
        tr = split_type_counts["train"][tn]
        va = split_type_counts["valid"][tn]
        te = split_type_counts["test"][tn]
        print(f"{tn:<18} {tr:>7} {va:>7} {te:>7} {tr + va + te:>7}")
    print("-" * len(header))
    print(f"{'TOTAL':<18} "
          f"{sum(split_type_counts['train'].values()):>7} "
          f"{sum(split_type_counts['valid'].values()):>7} "
          f"{sum(split_type_counts['test'].values()):>7} "
          f"{len(all_records):>7}")

    # Atom-count stats.
    counts = np.array([r["num_atoms"] for r in all_records])
    print(f"\nAtom-count stats: min={counts.min()} median={int(np.median(counts))} "
          f"mean={counts.mean():.1f} max={counts.max()}")

    # Element histogram.
    elem_hist: Counter = Counter()
    for r in all_records:
        for z in r["atom_types"]:
            elem_hist[int(z)] += 1
    inv_map = {int(k): v for k, v in mapping.items() if k.isdigit()}
    print("\nElement histogram (Z: symbol = atom count):")
    for z, cnt in sorted(elem_hist.items()):
        print(f"  {z:>3}: {inv_map.get(z, '?'):>4} = {cnt}")
    print()


if __name__ == "__main__":
    main()
