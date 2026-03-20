"""Process QM9 dataset into unified LMDB format for dtmol training.

QM9 contains ~134k small organic molecules (up to 9 heavy atoms: C, H, O, N, F)
with DFT-computed properties including energies, HOMO/LUMO, dipole moment, etc.

Download: Automatically handled by torch_geometric.datasets.QM9
Labels: 19 molecular properties (energy in Hartree, converted to eV)
Forces: Not available in QM9 (stored as None)

Usage:
    python process_qm9.py --raw_dir /data/qm9/raw --output_dir /data/qm9/lmdb
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import logging

from dtmol.data_op import HARTREE_TO_EV, ATOMIC_NUMBERS, write_lmdb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# QM9 property names (indices into data.y)
QM9_PROPERTIES = [
    "mu",       # 0: Dipole moment (D)
    "alpha",    # 1: Isotropic polarizability (a0^3)
    "homo",     # 2: HOMO energy (eV) - already in eV in PyG
    "lumo",     # 3: LUMO energy (eV)
    "gap",      # 4: HOMO-LUMO gap (eV)
    "r2",       # 5: Electronic spatial extent (a0^2)
    "zpve",     # 6: Zero-point vibrational energy (eV)
    "u0",       # 7: Internal energy at 0K (eV)
    "u298",     # 8: Internal energy at 298.15K (eV)
    "h298",     # 9: Enthalpy at 298.15K (eV)
    "g298",     # 10: Free energy at 298.15K (eV)
    "cv",       # 11: Heat capacity at 298.15K (cal/mol/K)
    "u0_atom",  # 12: Atomization energy at 0K (eV)
    "u298_atom",# 13: Atomization energy at 298.15K (eV)
    "h298_atom",# 14: Atomization enthalpy at 298.15K (eV)
    "g298_atom",# 15: Atomization free energy at 298.15K (eV)
    "A",        # 16: Rotational constant A (GHz)
    "B",        # 17: Rotational constant B (GHz)
    "C",        # 18: Rotational constant C (GHz)
]


def process_qm9(raw_dir, output_dir, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Convert QM9 to unified LMDB format.

    Args:
        raw_dir: str, directory for torch_geometric to download/cache QM9.
        output_dir: str, directory to write train.lmdb, valid.lmdb, test.lmdb.
        seed: int, random seed for train/val/test split.
        train_ratio: float, fraction of data for training.
        val_ratio: float, fraction of data for validation.
    """
    try:
        from torch_geometric.datasets import QM9
    except ImportError:
        raise ImportError("torch_geometric is required: pip install torch_geometric")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading QM9 from {raw_dir}...")
    dataset = QM9(root=raw_dir)
    n = len(dataset)
    logger.info(f"Loaded {n} molecules from QM9.")

    # Split by random permutation
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": perm[:n_train],
        "valid": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }

    for split_name, indices in splits.items():
        logger.info(f"Processing {split_name} split: {len(indices)} molecules")
        records = []
        for idx in tqdm(indices, desc=split_name):
            data = dataset[int(idx)]
            atomic_numbers = data.z.numpy().astype(int)
            atoms = [ATOMIC_NUMBERS.get(z, "X") for z in atomic_numbers]
            coordinates = data.pos.numpy().astype(np.float32)

            # Extract properties - PyG QM9 stores them in data.y as a (1, 19) tensor
            # Note: PyG QM9 already converts energies to eV internally
            props = {}
            y = data.y.squeeze(0).numpy()
            for i, name in enumerate(QM9_PROPERTIES):
                if i < len(y):
                    props[name] = float(y[i])

            # Use internal energy at 0K as the primary energy target
            energy = props.get("u0", None)

            record = {
                "task_type": "energy_force",
                "dataset": "qm9",
                "atoms": atoms,
                "coordinates": coordinates,
                "energy": energy,
                "forces": None,  # QM9 does not provide forces
                "properties": props,
            }
            records.append(record)

        db_path = os.path.join(output_dir, f"{split_name}.lmdb")
        count = write_lmdb(records, db_path)
        logger.info(f"Wrote {count} records to {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process QM9 dataset to LMDB")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory for torch_geometric to download/cache QM9")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LMDB files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_qm9(args.raw_dir, args.output_dir, args.seed)
