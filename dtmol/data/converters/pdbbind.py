"""PDBBind converter: converts existing UniMol PDBBind LMDB to unified format."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import numpy as np
from numpy.typing import NDArray

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Symbol -> atomic number lookup (covers elements in UniMol dicts)
_SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Cr": 24, "Fe": 26,
    "Zn": 30, "As": 33, "Se": 34, "Br": 35, "Sn": 50, "I": 53, "Gd": 64,
    "Au": 79, "Hg": 80,
}


def _symbol_to_atomic_number(symbol: str) -> int:
    """Convert an element symbol to atomic number.

    Handles single-character symbols from pocket_atoms (e.g. 'C', 'N')
    and multi-character symbols from ligand atoms.
    """
    # Try exact match first
    if symbol in _SYMBOL_TO_Z:
        return _SYMBOL_TO_Z[symbol]
    # Try capitalizing (e.g. 'c' -> 'C')
    cap = symbol.capitalize()
    if cap in _SYMBOL_TO_Z:
        return _SYMBOL_TO_Z[cap]
    # Try first character only (pocket atoms sometimes stored as full atom name)
    first = symbol[0].upper()
    if first in _SYMBOL_TO_Z:
        return _SYMBOL_TO_Z[first]
    logger.warning("Unknown element symbol %r, mapping to Z=0", symbol)
    return 0


def _read_source_lmdb(lmdb_path: str) -> List[Dict[str, Any]]:
    """Read all records from a source PDBBind LMDB file.

    Automatically detects whether the LMDB is a directory (subdir=True)
    or a single file (subdir=False).
    """
    path = Path(lmdb_path)
    # If path is a directory containing data.mdb, it's subdir=True style
    is_subdir = path.is_dir() and (path / "data.mdb").exists()
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=is_subdir)
    records: List[Dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            record = pickle.loads(value)
            records.append(record)
    env.close()
    return records


def _convert_record(raw: Dict[str, Any]) -> UnifiedRecord:
    """Convert a single raw PDBBind record to UnifiedRecord."""
    # Extract atom symbols
    ligand_atoms: List[str] = raw["atoms"]
    pocket_atoms: List[str] = raw["pocket_atoms"]

    # Extract coordinates
    ligand_coords = np.asarray(raw["coordinates"], dtype=np.float64)
    # coordinates may be a list of arrays (multiple conformations) — take first
    if ligand_coords.ndim == 3:
        ligand_coords = ligand_coords[0]
    pocket_coords = np.asarray(raw["pocket_coordinates"], dtype=np.float64)
    if pocket_coords.ndim == 3:
        pocket_coords = pocket_coords[0]

    # Pocket atoms are sometimes full atom names; use first character
    pocket_z = np.array(
        [_symbol_to_atomic_number(a[0]) for a in pocket_atoms], dtype=np.int64
    )
    ligand_z = np.array(
        [_symbol_to_atomic_number(a) for a in ligand_atoms], dtype=np.int64
    )

    # Concatenate: protein first, then ligand
    atom_types = np.concatenate([pocket_z, ligand_z])
    positions = np.concatenate([pocket_coords, ligand_coords], axis=0)
    num_atoms = len(atom_types)

    # Component mask: 0=protein, 1=ligand
    component_mask = np.concatenate([
        np.zeros(len(pocket_z), dtype=np.int64),
        np.ones(len(ligand_z), dtype=np.int64),
    ])

    # Neighbor list
    neighbor_list = BaseConverter.compute_neighbor_list(positions, cutoff=5.0)

    # System ID from pdb_id or pocket field
    system_id = raw.get("pdb_id", raw.get("pocket", "unknown"))

    record: UnifiedRecord = {
        "atom_types": atom_types,
        "positions": positions,
        "num_atoms": num_atoms,
        "dataset_source": "pdbbind",
        "system_id": str(system_id),
        "pes_tier": "C",
        "forces": None,
        "noise_target": None,
        "noise_level": None,
        "energy": None,
        "binding_affinity": None,
        "relative_energy": None,
        "trajectory_id": None,
        "timestep": None,
        "positions_prev": None,
        "positions_next": None,
        "component_mask": component_mask,
        "pocket_mask": None,
        "partial_charges": None,
        "dipole": None,
        "homo": None,
        "lumo": None,
        "neighbor_list": neighbor_list,
    }
    return record


class PDBBindConverter(BaseConverter):
    """Converter for PDBBind UniMol LMDB files to unified format.

    Supports two input formats:
    1. Directory with split LMDBs: train.lmdb, valid.lmdb, test.lmdb (preserves splits)
    2. Single LMDB directory (e.g. pdbbind.lmdb/) — auto-splits 80/10/10
    """

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert PDBBind LMDB files to unified format.

        Args:
            input_path: Either a directory containing train.lmdb/valid.lmdb/test.lmdb,
                or a single LMDB directory (containing data.mdb).
            output_path: Directory where output unified LMDB files are written.
            split_strategy: Split strategy (used for auto-splitting single LMDB).
        """
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Detect input format
        has_split_files = any(
            (input_dir / f"{s}.lmdb").exists() for s in ("train", "valid", "test")
        )
        is_single_lmdb = (input_dir / "data.mdb").exists()

        if has_split_files:
            self._convert_split_lmdbs(input_dir, output_dir)
        elif is_single_lmdb:
            self._convert_single_lmdb(input_dir, output_dir)
        else:
            raise FileNotFoundError(
                f"No split LMDBs (train.lmdb etc.) or single LMDB (data.mdb) "
                f"found at {input_path}"
            )

    def _convert_split_lmdbs(self, input_dir: Path, output_dir: Path) -> None:
        """Convert pre-split LMDB files preserving train/valid/test splits."""
        for split in ("train", "valid", "test"):
            src_path = input_dir / f"{split}.lmdb"
            if not src_path.exists():
                logger.warning("Split file %s not found, skipping.", src_path)
                continue

            logger.info("Converting %s ...", src_path)
            raw_records = _read_source_lmdb(str(src_path))
            unified_records = self._convert_records(raw_records)

            out_path = str(output_dir / f"{split}.lmdb")
            self.write_lmdb(unified_records, out_path)
            logger.info("Wrote %d records to %s", len(unified_records), out_path)

    def _convert_single_lmdb(self, input_dir: Path, output_dir: Path) -> None:
        """Convert a single LMDB and auto-split into train/valid/test (80/10/10)."""
        logger.info("Converting single LMDB at %s (auto-splitting 80/10/10) ...", input_dir)
        raw_records = _read_source_lmdb(str(input_dir))
        unified_records = self._convert_records(raw_records)

        # Deterministic shuffle for reproducible splits
        rng = np.random.RandomState(42)
        indices = np.arange(len(unified_records))
        rng.shuffle(indices)

        n = len(unified_records)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)

        split_map = {
            "train": indices[:n_train],
            "valid": indices[n_train:n_train + n_valid],
            "test": indices[n_train + n_valid:],
        }

        for split_name, split_indices in split_map.items():
            split_records = [unified_records[i] for i in split_indices]
            out_path = str(output_dir / f"{split_name}.lmdb")
            self.write_lmdb(split_records, out_path)
            logger.info("Wrote %d records to %s", len(split_records), out_path)

    @staticmethod
    def _convert_records(raw_records: List[Dict[str, Any]]) -> List[UnifiedRecord]:
        """Convert a list of raw PDBBind records to unified format."""
        unified_records: List[UnifiedRecord] = []
        for raw in raw_records:
            try:
                unified = _convert_record(raw)
                unified_records.append(unified)
            except Exception:
                sid = raw.get("pdb_id", raw.get("pocket", "?"))
                logger.warning("Skipping record %s due to error", sid, exc_info=True)
        return unified_records


def register() -> None:
    """Register PDBBind converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("pdbbind", PDBBindConverter)


register()
