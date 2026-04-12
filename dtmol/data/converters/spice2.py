"""SPICE v2 converter: converts SPICE v2 HDF5 to unified LMDB format.

SPICE v1.1.4 units (per OpenMM SPICE documentation):
  - conformations: Angstrom
  - dft_total_energy: Hartree
  - dft_total_gradient: Hartree/Angstrom
  - mbis_charges: shape (M, N, 1), elementary charge units
"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import lmdb
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Unit conversions: SPICE stores energy in Hartree, gradients in Hartree/Angstrom
HARTREE_TO_EV = 27.2114


def _detect_dimer_components(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    """Detect dimer components via Union-Find on 1.8A covalent cutoff.

    Returns component_mask (0 for first component, 1 for second) if the
    system has exactly two disconnected components, otherwise None.
    """
    n = len(atomic_numbers)
    if n < 2:
        return None

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
    """Converter for SPICE v2 HDF5 dataset to unified format.

    SPICE v1.1.4 HDF5 groups represent molecules/dimers with multiple conformations.
    Each group has:
      - atomic_numbers: (N,) int16
      - conformations: (M, N, 3) float32 in Angstrom
      - dft_total_energy: (M,) float64 in Hartree
      - dft_total_gradient: (M, N, 3) float32 in Hartree/Angstrom
      - mbis_charges: (M, N, 1) float32 (optional)

    Splits are by molecule (not conformation) to avoid data leakage.
    Uses streaming writes to handle the large dataset (~1M records).
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
            if Path(split_path).exists():
                shutil.rmtree(split_path)
            Path(split_path).mkdir(parents=True, exist_ok=True)
            envs[split] = lmdb.open(split_path, map_size=map_size)

        rng = np.random.RandomState(42)

        with h5py.File(input_p, "r") as f:
            mol_groups = sorted(f.keys())
            n_mols = len(mol_groups)
            logger.info("Found %d molecule groups in %s", n_mols, input_path)

            # Assign molecule-level splits: 80/10/10
            mol_perm = rng.permutation(n_mols)
            n_train = int(0.8 * n_mols)
            n_valid = int(0.1 * n_mols)

            mol_split: Dict[int, str] = {}
            for mi in mol_perm[:n_train]:
                mol_split[mi] = "train"
            for mi in mol_perm[n_train:n_train + n_valid]:
                mol_split[mi] = "valid"
            for mi in mol_perm[n_train + n_valid:]:
                mol_split[mi] = "test"

            logger.info(
                "Molecule splits: %d train, %d valid, %d test",
                n_train, n_valid, n_mols - n_train - n_valid,
            )

            for mol_idx, mol_name in enumerate(mol_groups):
                mol_group = f[mol_name]

                if "atomic_numbers" not in mol_group:
                    logger.warning("Skipping %s: no atomic_numbers", mol_name)
                    continue
                if "conformations" not in mol_group:
                    logger.warning("Skipping %s: no conformations", mol_name)
                    continue
                if "dft_total_energy" not in mol_group:
                    logger.warning("Skipping %s: no dft_total_energy", mol_name)
                    continue
                if "dft_total_gradient" not in mol_group:
                    logger.warning("Skipping %s: no dft_total_gradient", mol_name)
                    continue

                atom_types = np.array(mol_group["atomic_numbers"][()], dtype=np.int64)
                num_atoms = len(atom_types)
                conformations = mol_group["conformations"][()]  # (M, N, 3) Angstrom
                energies_ha = mol_group["dft_total_energy"][()]  # (M,) Hartree
                gradients = mol_group["dft_total_gradient"][()]  # (M, N, 3) Ha/A

                # Optional: MBIS partial charges (M, N, 1) -> squeeze to (M, N)
                mbis_raw: Optional[np.ndarray] = None
                if "mbis_charges" in mol_group:
                    mbis_raw = mol_group["mbis_charges"][()]  # (M, N, 1)

                n_conf = conformations.shape[0]

                # Detect dimer components from first conformation
                component_mask = _detect_dimer_components(
                    atom_types, conformations[0]
                )

                split = mol_split[mol_idx]
                txn = envs[split].begin(write=True)

                for ci in range(n_conf):
                    pos = np.array(conformations[ci], dtype=np.float64)
                    energy_ev = float(energies_ha[ci]) * HARTREE_TO_EV

                    # Forces = -gradient; gradient is Ha/A -> forces in eV/A
                    forces_ev_a = (
                        -np.array(gradients[ci], dtype=np.float64) * HARTREE_TO_EV
                    )

                    # Partial charges: squeeze (N, 1) -> (N,)
                    partial_charges: Optional[np.ndarray] = None
                    if mbis_raw is not None:
                        partial_charges = np.array(
                            mbis_raw[ci].squeeze(-1), dtype=np.float64
                        )

                    neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

                    record: UnifiedRecord = {
                        "atom_types": atom_types.copy(),
                        "positions": pos,
                        "num_atoms": num_atoms,
                        "dataset_source": "spice2",
                        "system_id": f"{mol_name}_conf{ci}",
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

                    key = str(counters[split]).encode()
                    txn.put(key, pickle.dumps(record))
                    counters[split] += 1

                txn.commit()

                if (mol_idx + 1) % 1000 == 0:
                    logger.info(
                        "Processed %d/%d molecules (train=%d, valid=%d, test=%d)",
                        mol_idx + 1, n_mols,
                        counters["train"], counters["valid"], counters["test"],
                    )

        for env in envs.values():
            env.close()

        total = sum(counters.values())
        logger.info(
            "Wrote %d total records — train: %d, valid: %d, test: %d",
            total, counters["train"], counters["valid"], counters["test"],
        )


def register() -> None:
    """Register SPICE v2 converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("spice2", SPICE2Converter)


register()
