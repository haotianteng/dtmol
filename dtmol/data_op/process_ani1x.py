"""Process ANI-1x dataset into unified LMDB format for dtmol training.

ANI-1x contains ~5.5M conformations of ~63k molecules (H, C, N, O) with
DFT energies and forces at wB97x/6-31G(d) level.

Download: https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041
Format: HDF5 (.h5)
Labels: Energies (Hartree) and forces (Hartree/Bohr)

Usage:
    python process_ani1x.py --h5_path /data/ani1x/ani1x-release.h5 --output_dir /data/ani1x/lmdb
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import logging

from dtmol.data_op import HARTREE_TO_EV, BOHR_TO_ANGSTROM, ATOMIC_NUMBERS, write_lmdb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Force conversion: Hartree/Bohr -> eV/Angstrom
FORCE_CONVERSION = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def iter_ani1x_h5(h5_path, energy_key="wb97x_dz.energy", forces_key="wb97x_dz.forces"):
    """Iterate over molecule groups in ANI-1x HDF5 file.

    Yields:
        dict with keys: species (atomic numbers), coordinates, energies, forces
        Each molecule may have multiple conformations.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        for mol_key in f.keys():
            mol = f[mol_key]
            species = mol["species"][()]  # (n_atoms,) int array
            coords = mol["coordinates"][()]  # (n_conf, n_atoms, 3)

            energies = None
            if energy_key in mol:
                energies = mol[energy_key][()]  # (n_conf,)

            forces = None
            if forces_key in mol:
                forces = mol[forces_key][()]  # (n_conf, n_atoms, 3)

            yield {
                "mol_key": mol_key,
                "species": species,
                "coordinates": coords,
                "energies": energies,
                "forces": forces,
            }


def process_ani1x(h5_path, output_dir, seed=42, train_ratio=0.9, val_ratio=0.05,
                  energy_key="wb97x_dz.energy", forces_key="wb97x_dz.forces"):
    """Convert ANI-1x HDF5 to unified LMDB format.

    Split is done by molecule (not conformation) to avoid data leakage.

    Args:
        h5_path: str, path to ani1x-release.h5.
        output_dir: str, directory to write train.lmdb, valid.lmdb, test.lmdb.
        seed: int, random seed for molecule-level split.
        train_ratio: float, fraction of molecules for training.
        val_ratio: float, fraction of molecules for validation.
        energy_key: str, HDF5 key for energy values.
        forces_key: str, HDF5 key for force values.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    os.makedirs(output_dir, exist_ok=True)

    # First pass: collect molecule keys for splitting
    logger.info(f"Scanning molecules in {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        mol_keys = list(f.keys())
    n_mol = len(mol_keys)
    logger.info(f"Found {n_mol} molecules.")

    # Split by molecule
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_mol)
    n_train = int(n_mol * train_ratio)
    n_val = int(n_mol * val_ratio)
    mol_split = {}
    for i in perm[:n_train]:
        mol_split[mol_keys[i]] = "train"
    for i in perm[n_train:n_train + n_val]:
        mol_split[mol_keys[i]] = "valid"
    for i in perm[n_train + n_val:]:
        mol_split[mol_keys[i]] = "test"

    # Second pass: process and write records per split
    split_records = {"train": [], "valid": [], "test": []}
    for mol_data in tqdm(iter_ani1x_h5(h5_path, energy_key, forces_key),
                         total=n_mol, desc="Processing ANI-1x"):
        split_name = mol_split[mol_data["mol_key"]]
        species = mol_data["species"]
        atoms = [ATOMIC_NUMBERS.get(int(z), "X") for z in species]
        coords_all = mol_data["coordinates"]  # (n_conf, n_atoms, 3)
        energies = mol_data["energies"]
        forces_all = mol_data["forces"]

        n_conf = coords_all.shape[0]
        for j in range(n_conf):
            # Skip conformations with missing energy
            if energies is None or not np.isfinite(energies[j]):
                continue

            coordinates = coords_all[j].astype(np.float32)
            energy = float(energies[j]) * HARTREE_TO_EV

            forces = None
            if forces_all is not None and np.all(np.isfinite(forces_all[j])):
                forces = (forces_all[j] * FORCE_CONVERSION).astype(np.float32)

            record = {
                "task_type": "energy_force",
                "dataset": "ani1x",
                "atoms": atoms,
                "coordinates": coordinates,
                "energy": energy,
                "forces": forces,
                "properties": None,
            }
            split_records[split_name].append(record)

    for split_name, records in split_records.items():
        db_path = os.path.join(output_dir, f"{split_name}.lmdb")
        count = write_lmdb(records, db_path)
        logger.info(f"Wrote {count} records to {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ANI-1x dataset to LMDB")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to ani1x-release.h5")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LMDB files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--energy_key", type=str, default="wb97x_dz.energy")
    parser.add_argument("--forces_key", type=str, default="wb97x_dz.forces")
    args = parser.parse_args()
    process_ani1x(args.h5_path, args.output_dir, args.seed,
                  energy_key=args.energy_key, forces_key=args.forces_key)
