"""ANI-2x converter: converts ANI-2x HDF5 to unified LMDB format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Unit conversions
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177
# Hartree/Bohr -> eV/Angstrom
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

# Element symbol -> atomic number
_SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17,
}


class ANI2xConverter(BaseConverter):
    """Converter for ANI-2x HDF5 dataset to unified format."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert ANI-2x HDF5 to unified LMDB format.

        Each HDF5 group represents a molecule with multiple conformations.
        Creates one UnifiedRecord per conformation.

        Splits by molecule (not conformation) to avoid data leakage.

        Args:
            input_path: Path to the ANI-2x HDF5 file.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: 'random' splits by molecule (default).
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # First pass: collect records grouped by molecule
        mol_records: Dict[str, List[UnifiedRecord]] = {}

        with h5py.File(input_p, "r") as f:
            mol_groups = list(f.keys())
            logger.info("Found %d molecule groups in %s", len(mol_groups), input_path)

            for mol_name in mol_groups:
                mol_group = f[mol_name]
                species = mol_group["species"][()]  # (N,) element symbols
                coordinates = mol_group["coordinates"][()]  # (M, N, 3)
                energies = mol_group["energies"][()]  # (M,)
                forces = mol_group["forces"][()]  # (M, N, 3)

                # Convert species bytes to strings if needed, then to atomic numbers
                if isinstance(species[0], bytes):
                    species_str = [s.decode("utf-8") for s in species]
                else:
                    species_str = [str(s) for s in species]

                atom_types = np.array(
                    [_SYMBOL_TO_Z.get(s, 0) for s in species_str], dtype=np.int64
                )
                num_atoms = len(atom_types)

                records: List[UnifiedRecord] = []
                n_conformations = coordinates.shape[0]

                for conf_idx in range(n_conformations):
                    pos = np.array(coordinates[conf_idx], dtype=np.float64)  # (N, 3) in Angstrom
                    energy_ev = float(energies[conf_idx]) * HARTREE_TO_EV
                    # ANI-2x forces are in Hartree/Bohr -> convert to eV/Angstrom
                    force_ev_a = np.array(forces[conf_idx], dtype=np.float64) * HARTREE_BOHR_TO_EV_ANGSTROM

                    neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

                    record: UnifiedRecord = {
                        "atom_types": atom_types.copy(),
                        "positions": pos,
                        "num_atoms": num_atoms,
                        "dataset_source": "ani2x",
                        "system_id": f"{mol_name}_conf{conf_idx}",
                        "pes_tier": "A",
                        "forces": force_ev_a,
                        "noise_target": None,
                        "noise_level": None,
                        "energy": energy_ev,
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
    """Register ANI-2x converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("ani2x", ANI2xConverter)


register()
