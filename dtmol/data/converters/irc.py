"""IRC/Transition1x converter: converts reaction path data to unified LMDB format."""

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
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class IRCConverter(BaseConverter):
    """Converter for Transition1x HDF5 dataset to unified format.

    Transition1x contains structures along IRC (intrinsic reaction coordinate)
    paths. Each reaction has multiple structures from reactant through
    transition state to product, with DFT energies and forces.
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert Transition1x HDF5 to unified LMDB format.

        Each HDF5 group represents a reaction with multiple structures along
        the IRC path. Creates one UnifiedRecord per structure.

        Splits by reaction (not by structure) to avoid data leakage.

        Args:
            input_path: Path to the Transition1x HDF5 file.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: 'random' splits by reaction (default).
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Collect records grouped by reaction
        reaction_records: Dict[str, List[UnifiedRecord]] = {}

        with h5py.File(input_p, "r") as f:
            # Transition1x structure: top-level groups are reactions
            # Each reaction group has: atomic_numbers, positions, energies, forces
            reaction_keys = self._collect_reaction_keys(f)
            logger.info(
                "Found %d reaction groups in %s", len(reaction_keys), input_path
            )

            for rxn_key in reaction_keys:
                rxn_group = f[rxn_key]
                records = self._convert_reaction(rxn_key, rxn_group)
                if records:
                    reaction_records[rxn_key] = records

        # Split by reaction to avoid leakage
        rxn_names = list(reaction_records.keys())
        rng = np.random.RandomState(42)
        rxn_indices = rng.permutation(len(rxn_names))
        n_rxn = len(rxn_names)
        n_train = int(0.8 * n_rxn)
        n_valid = int(0.1 * n_rxn)

        splits = {
            "train": rxn_indices[:n_train],
            "valid": rxn_indices[n_train : n_train + n_valid],
            "test": rxn_indices[n_train + n_valid :],
        }

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_records = sum(len(recs) for recs in reaction_records.values())
        logger.info(
            "Converted %d total structures from %d reactions",
            total_records,
            n_rxn,
        )

        for split_name, split_rxn_indices in splits.items():
            split_records: List[UnifiedRecord] = []
            for ri in split_rxn_indices:
                split_records.extend(reaction_records[rxn_names[ri]])
            out_path = str(output_dir / f"{split_name}.lmdb")
            self.write_lmdb(split_records, out_path)
            logger.info(
                "Wrote %d %s records (%d reactions) to %s",
                len(split_records),
                split_name,
                len(split_rxn_indices),
                out_path,
            )

    def _collect_reaction_keys(self, f: h5py.File) -> List[str]:
        """Recursively collect all leaf group keys that contain reaction data.

        Transition1x HDF5 may have nested groups (e.g. data/<rxn_type>/<rxn_id>).
        We look for groups containing 'atomic_numbers' and 'positions' datasets.
        """
        keys: List[str] = []

        def _visit(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                if "atomic_numbers" in obj and "positions" in obj:
                    keys.append(name)

        f.visititems(_visit)  # type: ignore[arg-type]
        return keys

    def _convert_reaction(
        self,
        rxn_key: str,
        rxn_group: h5py.Group,
    ) -> List[UnifiedRecord]:
        """Convert a single reaction group into a list of UnifiedRecords."""
        atomic_numbers = rxn_group["atomic_numbers"][()]  # (N,) or (M, N)
        positions = rxn_group["positions"][()]  # (M, N, 3)
        energies = rxn_group["energies"][()]  # (M,)

        # Forces: may be stored as 'forces' or 'gradients' (negate if gradients)
        if "forces" in rxn_group:
            raw_forces = rxn_group["forces"][()]  # (M, N, 3)
            negate_forces = False
        elif "gradients" in rxn_group:
            raw_forces = rxn_group["gradients"][()]  # (M, N, 3)
            negate_forces = True
        else:
            raw_forces = None
            negate_forces = False

        # Handle atomic_numbers: may be (N,) shared across structures or (M, N)
        if atomic_numbers.ndim == 1:
            atom_types = np.array(atomic_numbers, dtype=np.int64)
        else:
            # (M, N) — take first row, assume all structures share same species
            atom_types = np.array(atomic_numbers[0], dtype=np.int64)

        # Decode bytes if needed
        if atom_types.dtype.kind in ("S", "U", "O"):
            # Stored as element symbols rather than atomic numbers
            from dtmol.data.converters.ani2x import _SYMBOL_TO_Z

            decoded = []
            for s in atom_types:
                sym = s.decode("utf-8") if isinstance(s, bytes) else str(s)
                decoded.append(_SYMBOL_TO_Z.get(sym, 0))
            atom_types = np.array(decoded, dtype=np.int64)

        num_atoms = len(atom_types)
        n_structures = positions.shape[0]

        # Compute relative_energy = energy - min(energies) for the reaction path
        energies_ev = np.array(energies, dtype=np.float64) * HARTREE_TO_EV
        min_energy = float(np.min(energies_ev))

        records: List[UnifiedRecord] = []
        for struct_idx in range(n_structures):
            pos = np.array(positions[struct_idx], dtype=np.float64)  # (N, 3)
            energy_ev = float(energies_ev[struct_idx])
            rel_energy = energy_ev - min_energy

            # Convert forces
            forces_ev_a = None
            if raw_forces is not None:
                f_raw = np.array(raw_forces[struct_idx], dtype=np.float64)
                if negate_forces:
                    f_raw = -f_raw
                forces_ev_a = f_raw * HARTREE_BOHR_TO_EV_ANGSTROM

            neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

            record: UnifiedRecord = {
                "atom_types": atom_types.copy(),
                "positions": pos,
                "num_atoms": num_atoms,
                "dataset_source": "irc",
                "system_id": f"{rxn_key}_struct{struct_idx}",
                "pes_tier": "A",
                "forces": forces_ev_a,
                "noise_target": None,
                "noise_level": None,
                "energy": energy_ev,
                "binding_affinity": None,
                "relative_energy": rel_energy,
                "trajectory_id": rxn_key,
                "timestep": struct_idx,
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

        return records


def register() -> None:
    """Register IRC converter with the CLI."""
    from dtmol.data.convert import register_converter

    register_converter("irc", IRCConverter)


register()
