"""IRC/Transition1x converter: converts reaction path data to unified LMDB format."""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import lmdb
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Unit conversions
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

# HDF5 dataset key names in Transition1x
_ENERGY_KEY = "wB97x_6-31G(d).energy"
_FORCES_KEY = "wB97x_6-31G(d).forces"

# Sub-groups to skip (they duplicate data from the IRC path)
_SKIP_SUBGROUPS = {"reactant", "product", "transition_state"}


class IRCConverter(BaseConverter):
    """Converter for Transition1x HDF5 dataset to unified format.

    Transition1x contains structures along IRC (intrinsic reaction coordinate)
    paths. Each reaction has multiple structures from reactant through
    transition state to product, with DFT energies and forces.

    The HDF5 file has pre-defined train/val/test splits at the top level.
    Each split contains formula groups, each with reaction sub-groups.
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert Transition1x HDF5 to unified LMDB format.

        Uses the pre-defined train/val/test splits from the HDF5 file.
        Each reaction group produces multiple UnifiedRecords (one per IRC
        structure).

        Args:
            input_path: Path to the Transition1x HDF5 file.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: Ignored — uses HDF5-provided splits.
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open three LMDB environments for streaming writes
        map_size = 1 << 40  # 1 TB
        envs: Dict[str, lmdb.Environment] = {}
        counters: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        # HDF5 split name -> LMDB split name
        split_map = {"train": "train", "val": "valid", "test": "test"}

        for split in counters:
            split_path = str(output_dir / f"{split}.lmdb")
            if Path(split_path).exists():
                shutil.rmtree(split_path)
            Path(split_path).mkdir(parents=True, exist_ok=True)
            envs[split] = lmdb.open(split_path, map_size=map_size)

        with h5py.File(input_p, "r") as f:
            for h5_split, lmdb_split in split_map.items():
                if h5_split not in f:
                    logger.warning("Split '%s' not found in HDF5, skipping", h5_split)
                    continue

                split_group = f[h5_split]
                rxn_keys = self._collect_reaction_keys(split_group)
                logger.info(
                    "Split '%s': found %d reactions", h5_split, len(rxn_keys)
                )

                txn = envs[lmdb_split].begin(write=True)
                commit_interval = 5000  # commit every N records

                for rxn_key in rxn_keys:
                    rxn_group = split_group[rxn_key]
                    records = self._convert_reaction(rxn_key, rxn_group)

                    for record in records:
                        key = str(counters[lmdb_split]).encode()
                        txn.put(key, pickle.dumps(record))
                        counters[lmdb_split] += 1

                        if counters[lmdb_split] % commit_interval == 0:
                            txn.commit()
                            txn = envs[lmdb_split].begin(write=True)

                txn.commit()

        for env in envs.values():
            env.close()

        total = sum(counters.values())
        logger.info(
            "Transition1x conversion complete: %d total records "
            "(train=%d, valid=%d, test=%d)",
            total,
            counters["train"],
            counters["valid"],
            counters["test"],
        )

    def _collect_reaction_keys(self, group: h5py.Group) -> List[str]:
        """Collect all reaction group paths within a split group.

        Structure: <formula>/<rxn_id>/ — iterates two levels manually
        since visititems may not work on all HDF5 files.
        """
        keys: List[str] = []
        for formula_name in group:
            formula_group = group[formula_name]
            if not isinstance(formula_group, h5py.Group):
                continue
            for rxn_name in formula_group:
                if rxn_name in _SKIP_SUBGROUPS:
                    continue
                rxn_group = formula_group[rxn_name]
                if isinstance(rxn_group, h5py.Group) and "atomic_numbers" in rxn_group:
                    keys.append(f"{formula_name}/{rxn_name}")
        return keys

    def _convert_reaction(
        self,
        rxn_key: str,
        rxn_group: h5py.Group,
    ) -> List[UnifiedRecord]:
        """Convert a single reaction group into a list of UnifiedRecords."""
        atomic_numbers = rxn_group["atomic_numbers"][()]  # (N,)
        positions = rxn_group["positions"][()]  # (M, N, 3)

        # Energy in Hartree -> eV
        energies = rxn_group[_ENERGY_KEY][()]  # (M,)
        energies_ev = np.array(energies, dtype=np.float64) * HARTREE_TO_EV

        # Forces in Hartree/Bohr -> eV/Angstrom (these are forces, not gradients)
        raw_forces = None
        if _FORCES_KEY in rxn_group:
            raw_forces = rxn_group[_FORCES_KEY][()]  # (M, N, 3)

        # Handle atomic_numbers: (N,) shared across structures or (M, N)
        if atomic_numbers.ndim == 1:
            atom_types = np.array(atomic_numbers, dtype=np.int64)
        else:
            atom_types = np.array(atomic_numbers[0], dtype=np.int64)

        num_atoms = len(atom_types)
        n_structures = positions.shape[0]

        # Relative energy: 0 for lowest-energy structure in this reaction
        min_energy = float(np.min(energies_ev))

        records: List[UnifiedRecord] = []
        for struct_idx in range(n_structures):
            pos = np.array(positions[struct_idx], dtype=np.float64)  # (N, 3)
            energy_ev = float(energies_ev[struct_idx])
            rel_energy = energy_ev - min_energy

            forces_ev_a = None
            if raw_forces is not None:
                forces_ev_a = (
                    np.array(raw_forces[struct_idx], dtype=np.float64)
                    * HARTREE_BOHR_TO_EV_ANGSTROM
                )

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
