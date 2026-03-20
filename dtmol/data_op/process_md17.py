"""Process MD17 / rMD17 dataset into unified LMDB format for dtmol training.

MD17 contains ab-initio MD trajectories for 10 small organic molecules with
DFT-computed energies and atomic forces.

rMD17 is the revised version with tighter SCF convergence.

Download:
    MD17: http://www.sgdml.org/#datasets
    rMD17: https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

Format: NumPy .npz files
Labels: Energies (kcal/mol) and forces (kcal/mol/Angstrom), converted to eV

Usage:
    # Single molecule
    python process_md17.py --npz_path /data/md17/aspirin.npz --output_dir /data/md17/aspirin_lmdb --molecule_name aspirin

    # Revised MD17
    python process_md17.py --npz_path /data/rmd17/rmd17_aspirin.npz --output_dir /data/rmd17/aspirin_lmdb --molecule_name aspirin --revised
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import logging

from dtmol.data_op import KCAL_MOL_TO_EV, ATOMIC_NUMBERS, write_lmdb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def process_md17(npz_path, output_dir, molecule_name=None, seed=42,
                 n_train=1000, n_val=1000, revised=False):
    """Convert MD17/rMD17 npz to unified LMDB format.

    Standard benchmark uses 1000 train / 1000 test. We add a 1000 validation set.

    Args:
        npz_path: str, path to .npz file for one molecule.
        output_dir: str, directory to write train.lmdb, valid.lmdb, test.lmdb.
        molecule_name: str, name of the molecule (e.g., "aspirin"). Auto-detected if None.
        seed: int, random seed for split.
        n_train: int, number of training conformations.
        n_val: int, number of validation conformations.
        revised: bool, if True, expects rMD17 format.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading {'rMD17' if revised else 'MD17'} from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    if molecule_name is None:
        molecule_name = os.path.basename(npz_path).replace(".npz", "").replace("rmd17_", "").replace("md17_", "")

    # Extract arrays - handle both MD17 and rMD17 key conventions
    if "nuclear_charges" in data:
        atomic_numbers = data["nuclear_charges"].astype(int)
    elif "z" in data:
        atomic_numbers = data["z"].astype(int)
    else:
        raise KeyError(f"Cannot find atomic numbers. Available keys: {list(data.keys())}")

    atoms = [ATOMIC_NUMBERS.get(int(z), "X") for z in atomic_numbers]

    if "coords" in data:
        all_coords = data["coords"]  # (n_conf, n_atoms, 3)
    elif "R" in data:
        all_coords = data["R"]
    else:
        raise KeyError(f"Cannot find coordinates. Available keys: {list(data.keys())}")

    if "energies" in data:
        all_energies = data["energies"]  # (n_conf,)
    elif "E" in data:
        all_energies = data["E"].flatten()
    else:
        raise KeyError(f"Cannot find energies. Available keys: {list(data.keys())}")

    if "forces" in data:
        all_forces = data["forces"]  # (n_conf, n_atoms, 3)
    elif "F" in data:
        all_forces = data["F"]
    else:
        all_forces = None
        logger.warning("No forces found in npz file.")

    n_conf = all_coords.shape[0]
    logger.info(f"Molecule: {molecule_name}, {n_conf} conformations, {len(atoms)} atoms")

    # Split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_conf)
    splits = {
        "train": perm[:n_train],
        "valid": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:n_train + n_val + n_train],  # same size as train for standard benchmark
    }
    # If not enough conformations for the standard split, use all remaining for test
    if n_train + n_val + n_train > n_conf:
        splits["test"] = perm[n_train + n_val:]

    for split_name, indices in splits.items():
        records = []
        for idx in tqdm(indices, desc=f"{molecule_name}/{split_name}"):
            coordinates = all_coords[idx].astype(np.float32)
            energy = float(all_energies[idx]) * KCAL_MOL_TO_EV

            forces = None
            if all_forces is not None:
                forces = (all_forces[idx] * KCAL_MOL_TO_EV).astype(np.float32)

            record = {
                "task_type": "energy_force",
                "dataset": "rmd17" if revised else "md17",
                "atoms": atoms,
                "coordinates": coordinates,
                "energy": energy,
                "forces": forces,
                "properties": {"molecule_name": molecule_name},
            }
            records.append(record)

        db_path = os.path.join(output_dir, f"{split_name}.lmdb")
        count = write_lmdb(records, db_path)
        logger.info(f"Wrote {count} records to {db_path}")


def process_md17_directory(md17_dir, output_dir, seed=42, revised=False):
    """Process all .npz files in a directory, creating per-molecule LMDB subdirectories.

    Args:
        md17_dir: str, directory containing .npz files.
        output_dir: str, base output directory.
        seed: int, random seed.
        revised: bool, if True, expects rMD17 format.
    """
    npz_files = sorted([f for f in os.listdir(md17_dir) if f.endswith(".npz")])
    logger.info(f"Found {len(npz_files)} .npz files in {md17_dir}")
    for npz_file in npz_files:
        mol_name = npz_file.replace(".npz", "").replace("rmd17_", "").replace("md17_", "")
        mol_output = os.path.join(output_dir, mol_name)
        process_md17(
            os.path.join(md17_dir, npz_file),
            mol_output,
            molecule_name=mol_name,
            seed=seed,
            revised=revised,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MD17/rMD17 dataset to LMDB")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Path to a single .npz file")
    parser.add_argument("--npz_dir", type=str, default=None,
                        help="Directory containing multiple .npz files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LMDB files")
    parser.add_argument("--molecule_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--revised", action="store_true",
                        help="Use rMD17 format")
    args = parser.parse_args()

    if args.npz_dir:
        process_md17_directory(args.npz_dir, args.output_dir, args.seed, args.revised)
    elif args.npz_path:
        process_md17(args.npz_path, args.output_dir, args.molecule_name,
                     args.seed, args.n_train, args.n_val, args.revised)
    else:
        parser.error("Either --npz_path or --npz_dir must be specified.")
