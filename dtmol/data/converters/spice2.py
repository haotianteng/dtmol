"""SPICE v2 converter: converts SPICE v2 HDF5 to unified LMDB format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Unit conversions
# SPICE v2 energies are in kJ/mol, forces (gradients) in kJ/mol/nm
KJ_MOL_TO_EV = 1.0 / 96.485  # 1 eV = 96.485 kJ/mol
KJ_MOL_NM_TO_EV_ANGSTROM = KJ_MOL_TO_EV / 10.0  # 1 nm = 10 A


def _detect_dimer_components(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    """Try to detect dimer components via connectivity (distance-based).

    Returns component_mask (0 for first component, 1 for second) if the
    system has exactly two disconnected components, otherwise None.
    """
    n = len(atomic_numbers)
    if n < 2:
        return None

    # Build adjacency via covalent-like cutoff (1.8 A)
    from scipy.spatial import KDTree

    tree = KDTree(positions)
    pairs = tree.query_pairs(r=1.8, output_type="ndarray")
    if len(pairs) == 0:
        return None

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    roots = {find(i) for i in range(n)}
    if len(roots) != 2:
        return None

    roots_list = sorted(roots)
    mask = np.array(
        [0 if find(i) == roots_list[0] else 1 for i in range(n)],
        dtype=np.int64,
    )
    return mask


class SPICE2Converter(BaseConverter):
    """Converter for SPICE v2 HDF5 dataset to unified format."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert SPICE v2 HDF5 to unified LMDB format.

        Each HDF5 group represents a molecule/dimer with multiple conformations.
        Creates one UnifiedRecord per conformation.

        Splits by molecule (not conformation) to avoid data leakage.

        Args:
            input_path: Path to the SPICE v2 HDF5 file.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: 'random' splits by molecule (default).
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        mol_records: Dict[str, List[UnifiedRecord]] = {}

        with h5py.File(input_p, "r") as f:
            mol_groups = list(f.keys())
            logger.info("Found %d molecule groups in %s", len(mol_groups), input_path)

            for mol_name in mol_groups:
                mol_group = f[mol_name]

                # Atomic numbers: (N,)
                if "atomic_numbers" not in mol_group:
                    logger.warning("Skipping %s: no atomic_numbers", mol_name)
                    continue
                atomic_numbers = mol_group["atomic_numbers"][()]
                atom_types = np.array(atomic_numbers, dtype=np.int64)
                num_atoms = len(atom_types)

                # Conformations: (M, N, 3) in Angstrom (SPICE v2 stores in Angstrom)
                if "conformations" not in mol_group:
                    logger.warning("Skipping %s: no conformations", mol_name)
                    continue
                conformations = mol_group["conformations"][()]  # (M, N, 3)

                # Energies: (M,) in kJ/mol
                if "dft_total_energy" not in mol_group:
                    logger.warning("Skipping %s: no dft_total_energy", mol_name)
                    continue
                energies_kj = mol_group["dft_total_energy"][()]

                # Gradients: (M, N, 3) in kJ/mol/nm — forces = -gradient
                if "dft_total_gradient" not in mol_group:
                    logger.warning("Skipping %s: no dft_total_gradient", mol_name)
                    continue
                gradients = mol_group["dft_total_gradient"][()]

                # Optional: MBIS partial charges (M, N)
                mbis_charges: Optional[np.ndarray] = None
                if "mbis_charges" in mol_group:
                    mbis_charges = mol_group["mbis_charges"][()]

                n_conformations = conformations.shape[0]
                records: List[UnifiedRecord] = []

                # Detect dimer component mask from first conformation
                component_mask = _detect_dimer_components(
                    atom_types, conformations[0]
                )

                for conf_idx in range(n_conformations):
                    pos = np.array(conformations[conf_idx], dtype=np.float64)
                    energy_ev = float(energies_kj[conf_idx]) * KJ_MOL_TO_EV

                    # Forces = -gradient, convert kJ/mol/nm -> eV/A
                    forces_ev_a = (
                        -np.array(gradients[conf_idx], dtype=np.float64)
                        * KJ_MOL_NM_TO_EV_ANGSTROM
                    )

                    # Partial charges for this conformation
                    partial_charges: Optional[np.ndarray] = None
                    if mbis_charges is not None:
                        partial_charges = np.array(
                            mbis_charges[conf_idx], dtype=np.float64
                        )

                    neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

                    record: UnifiedRecord = {
                        "atom_types": atom_types.copy(),
                        "positions": pos,
                        "num_atoms": num_atoms,
                        "dataset_source": "spice2",
                        "system_id": f"{mol_name}_conf{conf_idx}",
                        "pes_tier": "A",
                        "forces": forces_ev_a,
                        "noise_target": None,
                        "noise_level": None,
                        "energy": energy_ev,
                        "binding_affinity": None,
                        "relative_energy": None,
                        "trajectory_id": None,
                        "timestep": None,
                        "positions_prev": None,
                        "positions_next": None,
                        "component_mask": component_mask.copy() if component_mask is not None else None,
                        "pocket_mask": None,
                        "partial_charges": partial_charges,
                        "dipole": None,
                        "homo": None,
                        "lumo": None,
                        "neighbor_list": neighbor_list,
                    }
                    records.append(record)

                mol_records[mol_name] = records

        # Split by molecule to avoid leakage
        mol_names = list(mol_records.keys())
        rng = np.random.RandomState(42)
        mol_indices = rng.permutation(len(mol_names))
        n_mols = len(mol_names)
        n_train = int(0.8 * n_mols)
        n_valid = int(0.1 * n_mols)

        splits = {
            "train": mol_indices[:n_train],
            "valid": mol_indices[n_train : n_train + n_valid],
            "test": mol_indices[n_train + n_valid :],
        }

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_records = sum(len(recs) for recs in mol_records.values())
        logger.info(
            "Converted %d total conformations from %d molecules",
            total_records,
            n_mols,
        )

        for split_name, split_mol_indices in splits.items():
            split_records: List[UnifiedRecord] = []
            for mi in split_mol_indices:
                split_records.extend(mol_records[mol_names[mi]])
            out_path = str(output_dir / f"{split_name}.lmdb")
            self.write_lmdb(split_records, out_path)
            logger.info(
                "Wrote %d %s records (%d molecules) to %s",
                len(split_records),
                split_name,
                len(split_mol_indices),
                out_path,
            )


def register() -> None:
    """Register SPICE v2 converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("spice2", SPICE2Converter)


register()
