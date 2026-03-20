"""Process CrossDocked2020 dataset into unified LMDB format for dtmol training.

CrossDocked2020 contains ~22.5M docked poses of ligands cross-docked into
similar binding pockets across the PDB.

Download: http://bits.csb.pitt.edu/files/crossdock2020/
Format: SDF (ligands) + PDB (proteins) + types files (index)

Usage:
    python process_crossdocked.py \
        --crossdocked_dir /data/CrossDocked2020/ \
        --types_file /data/CrossDocked2020/split_by_name/it2_tt_0_train0.types \
        --output_dir /data/crossdocked/lmdb \
        --rmsd_threshold 1.0
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import logging

from dtmol.data_op import write_lmdb
from dtmol.data_op.process_pdbbind import extract_pocket, sdf_to_dataframe

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_types_file(types_file):
    """Parse CrossDocked2020 types file.

    Each line has format: <label> <rmsd> <receptor_path> <ligand_path>

    Returns:
        list of dict with keys: label, rmsd, receptor, ligand
    """
    entries = []
    with open(types_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            entries.append({
                "label": int(parts[0]),
                "rmsd": float(parts[1]),
                "receptor": parts[2],
                "ligand": parts[3],
            })
    return entries


def process_crossdocked(crossdocked_dir, types_file, output_dir,
                        rmsd_threshold=1.0, pocket_radius=10.0):
    """Convert CrossDocked2020 to unified docking LMDB format.

    Args:
        crossdocked_dir: str, root directory of CrossDocked2020 data.
        types_file: str, path to a .types file listing receptor-ligand pairs.
        output_dir: str, directory to write output .lmdb file.
        rmsd_threshold: float, only include poses with RMSD below this value.
        pocket_radius: float, distance threshold for pocket extraction in Angstroms.
    """
    os.makedirs(output_dir, exist_ok=True)
    entries = parse_types_file(types_file)
    logger.info(f"Parsed {len(entries)} entries from {types_file}")

    if rmsd_threshold is not None:
        entries = [e for e in entries if e["rmsd"] <= rmsd_threshold]
        logger.info(f"{len(entries)} entries remain after RMSD <= {rmsd_threshold} filter")

    records = []
    failed = 0
    for entry in tqdm(entries, desc="Processing CrossDocked2020"):
        receptor_path = os.path.join(crossdocked_dir, entry["receptor"])
        ligand_path = os.path.join(crossdocked_dir, entry["ligand"])

        if not os.path.isfile(receptor_path) or not os.path.isfile(ligand_path):
            failed += 1
            continue

        try:
            ligand_df, smiles = sdf_to_dataframe(ligand_path)
            if ligand_df is None or len(ligand_df) == 0:
                failed += 1
                continue

            ligand_coords = ligand_df[['x_coord', 'y_coord', 'z_coord']].values
            pocket_atoms_df = extract_pocket(receptor_path, ligand_coords, distance=pocket_radius)

            if pocket_atoms_df is None or len(pocket_atoms_df) == 0:
                failed += 1
                continue

            # Derive a pocket identifier from the receptor filename
            pocket_id = os.path.basename(entry["receptor"]).replace(".pdb", "")

            record = {
                "task_type": "docking",
                "dataset": "crossdocked",
                "atoms": list(ligand_df['atom_name']),
                "coordinates": ligand_coords.astype(np.float32),
                "smi": smiles,
                "pocket_atoms": list(pocket_atoms_df['atom_name']),
                "pocket_coordinates": pocket_atoms_df[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32),
                "residue": list(pocket_atoms_df['residue_chain']) if 'residue_chain' in pocket_atoms_df.columns else [],
                "config": {"pocket_radius": pocket_radius},
                "pdb_id": pocket_id,
            }
            records.append(record)
        except Exception as e:
            logger.debug(f"Failed to process {entry['ligand']}: {e}")
            failed += 1
            continue

    logger.info(f"Processed {len(records)} records, {failed} failures")

    # Write to LMDB - use the types file basename to determine split name
    types_basename = os.path.basename(types_file)
    if "train" in types_basename:
        split_name = "train"
    elif "test" in types_basename:
        split_name = "test"
    else:
        split_name = "data"

    db_path = os.path.join(output_dir, f"{split_name}.lmdb")
    count = write_lmdb(records, db_path)
    logger.info(f"Wrote {count} records to {db_path}")


def process_crossdocked_splits(crossdocked_dir, types_dir, output_dir,
                               rmsd_threshold=1.0, pocket_radius=10.0):
    """Process multiple types files (train/test splits) from CrossDocked2020.

    Args:
        crossdocked_dir: str, root directory of CrossDocked2020 data.
        types_dir: str, directory containing .types files (e.g., split_by_name/).
        output_dir: str, output directory for LMDB files.
        rmsd_threshold: float, RMSD filter threshold.
        pocket_radius: float, pocket extraction radius.
    """
    types_files = sorted([f for f in os.listdir(types_dir) if f.endswith(".types")])
    logger.info(f"Found {len(types_files)} .types files in {types_dir}")
    for types_file in types_files:
        logger.info(f"Processing {types_file}...")
        process_crossdocked(
            crossdocked_dir,
            os.path.join(types_dir, types_file),
            output_dir,
            rmsd_threshold=rmsd_threshold,
            pocket_radius=pocket_radius,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CrossDocked2020 to LMDB")
    parser.add_argument("--crossdocked_dir", type=str, required=True,
                        help="Root directory of CrossDocked2020")
    parser.add_argument("--types_file", type=str, default=None,
                        help="Path to a single .types file")
    parser.add_argument("--types_dir", type=str, default=None,
                        help="Directory of .types files for multiple splits")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LMDB files")
    parser.add_argument("--rmsd_threshold", type=float, default=1.0,
                        help="Max RMSD to include (default: 1.0 Angstrom)")
    parser.add_argument("--pocket_radius", type=float, default=10.0,
                        help="Pocket extraction radius in Angstroms")
    args = parser.parse_args()

    if args.types_dir:
        process_crossdocked_splits(args.crossdocked_dir, args.types_dir,
                                   args.output_dir, args.rmsd_threshold, args.pocket_radius)
    elif args.types_file:
        process_crossdocked(args.crossdocked_dir, args.types_file,
                            args.output_dir, args.rmsd_threshold, args.pocket_radius)
    else:
        parser.error("Either --types_file or --types_dir must be specified.")
