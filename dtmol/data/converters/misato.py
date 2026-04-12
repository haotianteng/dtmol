"""MISATO converter: converts MISATO QM HDF5 to unified LMDB format.

MISATO QM HDF5 structure (per PDB group):
  - atom_properties/atom_names: (N,) strings of atomic numbers (e.g. b'7' = nitrogen)
  - atom_properties/atom_properties_values: (N, 28) float32
      columns 0-2: x, y, z coordinates in Angstrom
      columns 3+: hybridisation, group, various charge/polarisation properties
  - atom_properties/bonds: (B, 3) bond info
  - mol_properties/: scalar molecular properties (Electron_Affinity, Ionization_Potential, etc.)

Each group is a protein-ligand complex identified by a 4-char PDB ID.
The QM data contains ligand atoms only (no protein, no water, single snapshot).
"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py
import lmdb
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)


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
    ligand extracted from that protein-ligand complex. Creates one
    UnifiedRecord per complex.
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
        dedup_list: Optional[str] = None,
        no_dedup: bool = False,
        affinity_index: Optional[str] = None,
    ) -> None:
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

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

                # atom_names are atomic numbers stored as byte strings
                atom_names_raw = atom_props["atom_names"][()]  # (N,) bytes
                try:
                    atom_types = np.array(
                        [int(name.decode() if isinstance(name, bytes) else str(name))
                         for name in atom_names_raw],
                        dtype=np.int64,
                    )
                except (ValueError, UnicodeDecodeError) as e:
                    logger.warning("Skipping %s: cannot parse atom_names: %s", pdb_id, e)
                    skipped_error += 1
                    continue

                num_atoms = len(atom_types)
                if num_atoms == 0:
                    logger.warning("Skipping %s: no atoms", pdb_id)
                    skipped_error += 1
                    continue

                # Coordinates from first 3 columns of atom_properties_values
                prop_values = atom_props["atom_properties_values"][()]  # (N, 28)
                positions = np.array(
                    prop_values[:, _COL_X:_COL_Z + 1], dtype=np.float64
                )  # (N, 3)

                # GFN2 partial charges
                partial_charges: Optional[np.ndarray] = None
                if prop_values.shape[1] > _COL_GFN2_CHARGE:
                    partial_charges = np.array(
                        prop_values[:, _COL_GFN2_CHARGE], dtype=np.float64
                    )

                # Component mask: all ligand (1) since QM data is ligand-only
                component_mask = np.ones(num_atoms, dtype=np.int64)

                # Binding affinity from external index
                pdb_id_lower = pdb_id.strip().lower()
                binding_affinity: Optional[float] = affinities.get(pdb_id_lower)

                # Molecular properties
                energy: Optional[float] = None
                homo: Optional[float] = None
                lumo: Optional[float] = None
                if "mol_properties" in pdb_group:
                    mol_props = pdb_group["mol_properties"]
                    # Ionization potential and electron affinity can serve as
                    # HOMO/LUMO proxies via Koopman's theorem
                    if "Ionization_Potential" in mol_props:
                        # IP ~ -HOMO (in eV)
                        homo = -float(mol_props["Ionization_Potential"][()])
                    if "Electron_Affinity" in mol_props:
                        # EA ~ -LUMO (in eV)
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
                        "Processed %d/%d complexes (train=%d, valid=%d, test=%d)",
                        pdb_idx + 1, n_valid_pdbs,
                        counters["train"], counters["valid"], counters["test"],
                    )

        for env in envs.values():
            env.close()

        total = sum(counters.values())
        logger.info(
            "Wrote %d total records — train: %d, valid: %d, test: %d "
            "(skipped %d dedup, %d errors)",
            total, counters["train"], counters["valid"], counters["test"],
            skipped_dedup, skipped_error,
        )


def register() -> None:
    """Register MISATO converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("misato", MISATOConverter)


register()
