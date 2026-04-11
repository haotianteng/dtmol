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
    """Read all records from a source PDBBind LMDB file."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
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
    """Converter for PDBBind UniMol LMDB files to unified format."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert PDBBind LMDB files preserving train/valid/test splits.

        Args:
            input_path: Directory containing train.lmdb, valid.lmdb, test.lmdb.
            output_path: Directory where output unified LMDB files are written.
        """
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "valid", "test"):
            src_path = input_dir / f"{split}.lmdb"
            if not src_path.exists():
                logger.warning("Split file %s not found, skipping.", src_path)
                continue

            logger.info("Converting %s ...", src_path)
            raw_records = _read_source_lmdb(str(src_path))
            unified_records: List[UnifiedRecord] = []

            for raw in raw_records:
                try:
                    unified = _convert_record(raw)
                    unified_records.append(unified)
                except Exception:
                    sid = raw.get("pdb_id", raw.get("pocket", "?"))
                    logger.warning("Skipping record %s due to error", sid, exc_info=True)

            out_path = str(output_dir / f"{split}.lmdb")
            self.write_lmdb(unified_records, out_path)
            logger.info("Wrote %d records to %s", len(unified_records), out_path)


def register() -> None:
    """Register PDBBind converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("pdbbind", PDBBindConverter)


register()
