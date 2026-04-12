"""ANI-2x converter: converts ANI-2x HDF5 to unified LMDB format."""

from __future__ import annotations

import logging
import pickle
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
# Hartree/Bohr -> eV/Angstrom
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class ANI2xConverter(BaseConverter):
    """Converter for ANI-2x HDF5 dataset to unified format.

    The ANI-2x HDF5 has groups keyed by atom count (e.g. '002', '012').
    Each group has:
      - species: (M, N) int64 atomic numbers
      - coordinates: (M, N, 3) float32 positions in Angstrom
      - energies: (M,) float64 in Hartree
      - forces: (M, N, 3) float64 in Hartree/Bohr

    Each conformation is a distinct molecule or distinct conformation.
    We split conformations within each group 80/10/10 to keep the split
    balanced across atom counts.
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open three LMDB environments for streaming writes
        map_size = 1 << 40  # 1 TB
        envs: Dict[str, lmdb.Environment] = {}
        counters: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        for split in counters:
            split_path = str(output_dir / f"{split}.lmdb")
            Path(split_path).mkdir(parents=True, exist_ok=True)
            envs[split] = lmdb.open(split_path, map_size=map_size)

        rng = np.random.RandomState(42)

        with h5py.File(input_p, "r") as f:
            group_keys = sorted(f.keys())
            logger.info("Found %d atom-count groups in %s", len(group_keys), input_path)

            for gk in group_keys:
                g = f[gk]
                species_all = g["species"][()]      # (M, N) int64
                coords_all = g["coordinates"][()]   # (M, N, 3)
                energies_all = g["energies"][()]     # (M,)
                forces_all = g["forces"][()]         # (M, N, 3)

                n_conf = coords_all.shape[0]
                num_atoms = coords_all.shape[1]

                # Random permutation for splitting within this group
                perm = rng.permutation(n_conf)
                n_train = int(0.8 * n_conf)
                n_valid = int(0.1 * n_conf)
                split_assignments = np.empty(n_conf, dtype="U5")
                split_assignments[perm[:n_train]] = "train"
                split_assignments[perm[n_train:n_train + n_valid]] = "valid"
                split_assignments[perm[n_train + n_valid:]] = "test"

                # Start a transaction per split for this group
                txns = {s: envs[s].begin(write=True) for s in envs}

                for ci in range(n_conf):
                    atom_types = np.array(species_all[ci], dtype=np.int64)
                    pos = np.array(coords_all[ci], dtype=np.float64)
                    energy_ev = float(energies_all[ci]) * HARTREE_TO_EV
                    force_ev_a = np.array(forces_all[ci], dtype=np.float64) * HARTREE_BOHR_TO_EV_ANGSTROM

                    neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

                    record: UnifiedRecord = {
                        "atom_types": atom_types,
                        "positions": pos,
                        "num_atoms": num_atoms,
                        "dataset_source": "ani2x",
                        "system_id": f"g{gk}_c{ci}",
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

                    split = split_assignments[ci]
                    key = str(counters[split]).encode()
                    txns[split].put(key, pickle.dumps(record))
                    counters[split] += 1

                # Commit all transactions for this group
                for txn in txns.values():
                    txn.commit()

                logger.info(
                    "Group %s: %d conformations (%d atoms each) processed",
                    gk, n_conf, num_atoms,
                )

        for env in envs.values():
            env.close()

        total = sum(counters.values())
        logger.info(
            "Wrote %d total records — train: %d, valid: %d, test: %d",
            total, counters["train"], counters["valid"], counters["test"],
        )


def register() -> None:
    """Register ANI-2x converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("ani2x", ANI2xConverter)


register()
