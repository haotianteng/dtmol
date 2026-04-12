"""MISATO converter: converts MISATO QM HDF5 to unified LMDB format.

MISATO QM HDF5 structure (per PDB group):
  - atom_properties/atom_names: (N,) strings of atomic numbers (e.g. b'7' = nitrogen)
  - atom_properties/atom_properties_values: (N, 28) float32
      columns 0-2: x, y, z coordinates in Angstrom
      columns 3+: hybridisation, group, various charge/polarisation properties
  - atom_properties/bonds: (B, 3) bond info
  - mol_properties/: scalar molecular properties (Electron_Affinity, Ionization_Potential, etc.)

Each group is a protein-ligand complex identified by a 4-char PDB ID.
The QM data contains ligand atoms only; protein atoms are fetched from RCSB PDB
structures and combined with the QM ligand to produce records with both
protein (component_mask=0) and ligand (component_mask=1).
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import lmdb
import numpy as np
import requests
from numpy.typing import NDArray

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Standard amino acid 3-letter codes (protein residues)
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Common modified residues treated as standard
    "MSE", "SEC",
}

# Water residue names to exclude
_WATER_NAMES = {"HOH", "WAT", "TIP3", "TIP", "TP3", "SOL"}

# Element symbol -> atomic number
_ELEMENT_TO_Z: Dict[str, int] = {
    "H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
    "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Se": 34, "Br": 35, "I": 53,
}


# Skip PDB files larger than 20MB — extremely large assemblies
_MAX_PDB_FILE_SIZE = 20 * 1024 * 1024


def _fetch_pdb_file(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """Download a PDB structure from RCSB and cache it locally."""
    pdb_id_lower = pdb_id.strip().lower()
    cached = cache_dir / f"{pdb_id_lower}.pdb"
    if cached.exists() and cached.stat().st_size > 0:
        if cached.stat().st_size > _MAX_PDB_FILE_SIZE:
            return None  # Too large, skip
        return cached

    url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            cached.write_text(resp.text)
            if cached.stat().st_size > _MAX_PDB_FILE_SIZE:
                return None  # Too large
            return cached
        else:
            logger.debug("RCSB returned %d for %s", resp.status_code, pdb_id)
            return None
    except requests.RequestException as e:
        logger.debug("Failed to fetch PDB %s: %s", pdb_id, e)
        return None


def _extract_pocket_atoms(
    pdb_path: Path,
    ligand_center: NDArray[np.float64],
    pocket_cutoff: float = 10.0,
) -> Optional[Tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Extract protein pocket atoms near the ligand from a PDB file.

    Uses fast manual PDB parsing (no BioPython) with vectorized distance
    filtering. Only includes heavy atoms from standard amino acid ATOM records
    within pocket_cutoff of the ligand center.

    Returns (atom_types, positions) arrays or None if parsing fails.
    """
    all_z: List[int] = []
    all_coords: List[Tuple[float, float, float]] = []

    try:
        with open(pdb_path) as fh:
            for line in fh:
                # Only standard residue ATOM records (not HETATM)
                if not line.startswith("ATOM  "):
                    if line.startswith("ENDMDL"):
                        break  # Only first model
                    continue
                resname = line[17:20].strip()
                if resname not in _STANDARD_AA:
                    continue
                # Element from columns 76-78, fallback to atom name
                element = line[76:78].strip() if len(line) > 77 else ""
                if not element:
                    element = line[12:16].strip()[0]
                if element in ("H", "D"):
                    continue
                z = _ELEMENT_TO_Z.get(element, 0)
                if z == 0:
                    z = _ELEMENT_TO_Z.get(element.capitalize(), 0)
                if z == 0:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z_coord = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                all_z.append(z)
                all_coords.append((x, y, z_coord))
    except Exception as e:
        logger.debug("PDB parse failed for %s: %s", pdb_path, e)
        return None

    if not all_z:
        return None

    # Vectorized distance filter
    coords_arr = np.array(all_coords, dtype=np.float64)  # (N, 3)
    dists = np.linalg.norm(coords_arr - ligand_center, axis=1)
    mask = dists < pocket_cutoff

    if not mask.any():
        return None

    z_arr = np.array(all_z, dtype=np.int64)
    return z_arr[mask], coords_arr[mask]


def _load_dedup_list(dedup_path: str) -> Set[str]:
    """Load a list of PDB IDs to skip (for deduplication with PDBBind)."""
    pdb_ids: Set[str] = set()
    with open(dedup_path) as f:
        for line in f:
            pdb_id = line.strip().lower()
            if pdb_id:
                pdb_ids.add(pdb_id)
    logger.info("Loaded %d PDB IDs for deduplication from %s", len(pdb_ids), dedup_path)
    return pdb_ids


def _load_binding_affinities(index_path: str) -> Dict[str, float]:
    """Load binding affinities from a PDBBind-style index file."""
    affinities: Dict[str, float] = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pdb_id = parts[0].lower()
                    affinity = float(parts[1])
                    affinities[pdb_id] = affinity
                except (ValueError, IndexError):
                    continue
    logger.info("Loaded %d binding affinities from %s", len(affinities), index_path)
    return affinities


# Indices into atom_properties_values columns (from atom_properties_names)
_COL_X, _COL_Y, _COL_Z = 0, 1, 2
_COL_GFN2_CHARGE = 5  # 'gfn2_charge'


class MISATOConverter(BaseConverter):
    """Converter for MISATO QM HDF5 to unified format.

    Each HDF5 group is a PDB ID containing QM-computed properties for the
    ligand extracted from that protein-ligand complex. Protein atoms are
    fetched from RCSB PDB structures and combined with QM ligand data to
    produce records with both protein (component_mask=0) and ligand
    (component_mask=1).
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
        dedup_list: Optional[str] = None,
        no_dedup: bool = False,
        affinity_index: Optional[str] = None,
        pdb_cache_dir: Optional[str] = None,
    ) -> None:
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # PDB cache directory: default to sibling of input file
        if pdb_cache_dir:
            pdb_cache = Path(pdb_cache_dir)
        else:
            pdb_cache = input_p.parent / "pdb_cache"
        pdb_cache.mkdir(parents=True, exist_ok=True)
        logger.info("PDB structure cache: %s", pdb_cache)

        # Load dedup PDB IDs if provided
        dedup_ids: Set[str] = set()
        if dedup_list and not no_dedup:
            dedup_ids = _load_dedup_list(dedup_list)

        # Load binding affinities if provided
        affinities: Dict[str, float] = {}
        if affinity_index:
            affinities = _load_binding_affinities(affinity_index)

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open three LMDB environments for streaming writes
        map_size = 1 << 40  # 1 TB
        envs: Dict[str, lmdb.Environment] = {}
        counters: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        for split in counters:
            split_path = str(output_dir / f"{split}.lmdb")
            if Path(split_path).exists():
                shutil.rmtree(split_path)
            Path(split_path).mkdir(parents=True, exist_ok=True)
            envs[split] = lmdb.open(split_path, map_size=map_size)

        rng = np.random.RandomState(42)
        skipped_dedup = 0
        skipped_error = 0
        skipped_no_protein = 0

        with h5py.File(input_p, "r") as f:
            pdb_groups = sorted(f.keys())
            n_pdbs = len(pdb_groups)
            logger.info("Found %d PDB groups in %s", n_pdbs, input_path)

            # Filter out dedup IDs
            valid_groups: List[str] = []
            for pdb_id in pdb_groups:
                if pdb_id.strip().lower() in dedup_ids:
                    skipped_dedup += 1
                    continue
                valid_groups.append(pdb_id)

            # Assign PDB-level splits: 80/10/10
            n_valid_pdbs = len(valid_groups)
            perm = rng.permutation(n_valid_pdbs)
            n_train = int(0.8 * n_valid_pdbs)
            n_valid = int(0.1 * n_valid_pdbs)

            pdb_split: Dict[int, str] = {}
            for pi in perm[:n_train]:
                pdb_split[pi] = "train"
            for pi in perm[n_train:n_train + n_valid]:
                pdb_split[pi] = "valid"
            for pi in perm[n_train + n_valid:]:
                pdb_split[pi] = "test"

            # Track RCSB request pacing
            last_fetch_time = 0.0

            for pdb_idx, pdb_id in enumerate(valid_groups):
                pdb_group = f[pdb_id]

                # Validate required fields
                if "atom_properties" not in pdb_group:
                    logger.warning("Skipping %s: no atom_properties", pdb_id)
                    skipped_error += 1
                    continue

                atom_props = pdb_group["atom_properties"]
                if "atom_names" not in atom_props or "atom_properties_values" not in atom_props:
                    logger.warning("Skipping %s: missing atom_names or atom_properties_values", pdb_id)
                    skipped_error += 1
                    continue

                # --- Ligand atoms from QM HDF5 ---
                atom_names_raw = atom_props["atom_names"][()]  # (N,) bytes
                try:
                    lig_atom_types = np.array(
                        [int(name.decode() if isinstance(name, bytes) else str(name))
                         for name in atom_names_raw],
                        dtype=np.int64,
                    )
                except (ValueError, UnicodeDecodeError) as e:
                    logger.warning("Skipping %s: cannot parse atom_names: %s", pdb_id, e)
                    skipped_error += 1
                    continue

                n_lig = len(lig_atom_types)
                if n_lig == 0:
                    logger.warning("Skipping %s: no ligand atoms", pdb_id)
                    skipped_error += 1
                    continue

                prop_values = atom_props["atom_properties_values"][()]  # (N, 28)
                lig_positions = np.array(
                    prop_values[:, _COL_X:_COL_Z + 1], dtype=np.float64
                )  # (N, 3)

                # GFN2 partial charges (ligand only)
                lig_charges: Optional[NDArray[np.float64]] = None
                if prop_values.shape[1] > _COL_GFN2_CHARGE:
                    lig_charges = np.array(
                        prop_values[:, _COL_GFN2_CHARGE], dtype=np.float64
                    )

                # --- Protein pocket atoms from RCSB PDB ---
                # Rate-limit RCSB requests (~5/sec)
                pdb_file = pdb_cache / f"{pdb_id.strip().lower()}.pdb"
                if not (pdb_file.exists() and pdb_file.stat().st_size > 0):
                    now = time.monotonic()
                    elapsed = now - last_fetch_time
                    if elapsed < 0.2:
                        time.sleep(0.2 - elapsed)
                    last_fetch_time = time.monotonic()

                pdb_path = _fetch_pdb_file(pdb_id, pdb_cache)
                protein_result = None
                if pdb_path is not None:
                    lig_center = lig_positions.mean(axis=0)
                    protein_result = _extract_pocket_atoms(
                        pdb_path, lig_center, pocket_cutoff=10.0
                    )

                if protein_result is None:
                    logger.debug("Skipping %s: no protein structure available", pdb_id)
                    skipped_no_protein += 1
                    continue

                prot_atom_types, prot_positions = protein_result
                n_prot = len(prot_atom_types)

                # --- Combine protein + ligand ---
                atom_types = np.concatenate([prot_atom_types, lig_atom_types])
                positions = np.concatenate([prot_positions, lig_positions])
                num_atoms = n_prot + n_lig

                # component_mask: 0=protein, 1=ligand
                component_mask = np.concatenate([
                    np.zeros(n_prot, dtype=np.int64),
                    np.ones(n_lig, dtype=np.int64),
                ])

                # Partial charges: only available for ligand; pad protein with NaN
                partial_charges: Optional[NDArray[np.float64]] = None
                if lig_charges is not None:
                    partial_charges = np.concatenate([
                        np.full(n_prot, np.nan, dtype=np.float64),
                        lig_charges,
                    ])

                # Binding affinity from external index
                pdb_id_lower = pdb_id.strip().lower()
                binding_affinity: Optional[float] = affinities.get(pdb_id_lower)

                # Molecular properties (from QM)
                energy: Optional[float] = None
                homo: Optional[float] = None
                lumo: Optional[float] = None
                if "mol_properties" in pdb_group:
                    mol_props = pdb_group["mol_properties"]
                    if "Ionization_Potential" in mol_props:
                        homo = -float(mol_props["Ionization_Potential"][()])
                    if "Electron_Affinity" in mol_props:
                        lumo = -float(mol_props["Electron_Affinity"][()])

                neighbor_list = self.compute_neighbor_list(positions, cutoff=5.0)

                record: UnifiedRecord = {
                    "atom_types": atom_types,
                    "positions": positions,
                    "num_atoms": num_atoms,
                    "dataset_source": "misato",
                    "system_id": pdb_id,
                    "pes_tier": "B",
                    "forces": None,
                    "noise_target": None,
                    "noise_level": None,
                    "energy": energy,
                    "binding_affinity": binding_affinity,
                    "relative_energy": None,
                    "trajectory_id": pdb_id,
                    "timestep": None,
                    # QM data is single-snapshot per complex (not MD trajectory),
                    # so there are no trajectory frames and thus no non-boundary
                    # frames requiring positions_prev/positions_next.
                    "positions_prev": None,
                    "positions_next": None,
                    "component_mask": component_mask,
                    "pocket_mask": None,
                    "partial_charges": partial_charges,
                    "dipole": None,
                    "homo": homo,
                    "lumo": lumo,
                    "neighbor_list": neighbor_list,
                }

                split = pdb_split[pdb_idx]
                txn = envs[split].begin(write=True)
                txn.put(str(counters[split]).encode(), pickle.dumps(record))
                counters[split] += 1
                txn.commit()

                if (pdb_idx + 1) % 2000 == 0:
                    logger.info(
                        "Processed %d/%d complexes (train=%d, valid=%d, test=%d, "
                        "skipped_no_protein=%d)",
                        pdb_idx + 1, n_valid_pdbs,
                        counters["train"], counters["valid"], counters["test"],
                        skipped_no_protein,
                    )

        for env in envs.values():
            env.close()

        total = sum(counters.values())
        logger.info(
            "Wrote %d total records — train: %d, valid: %d, test: %d "
            "(skipped %d dedup, %d no_protein, %d errors)",
            total, counters["train"], counters["valid"], counters["test"],
            skipped_dedup, skipped_no_protein, skipped_error,
        )


def register() -> None:
    """Register MISATO converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("misato", MISATOConverter)


register()
