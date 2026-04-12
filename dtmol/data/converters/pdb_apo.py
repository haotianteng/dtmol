"""PDB apo structure converter: converts PDB/mmCIF protein structures to unified LMDB format."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Element symbol -> atomic number for common protein heavy atoms
_ELEMENT_TO_Z = {
    "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "SE": 34,
    "FE": 26, "ZN": 30, "MG": 12, "CA": 20, "MN": 25, "CU": 29,
    "CO": 27, "NI": 28, "NA": 11, "K": 19, "CL": 17, "F": 9,
    "BR": 35, "I": 53,
}


def _atom_element(atom: object) -> Optional[str]:
    """Extract the element symbol from a BioPython Atom object.

    Returns the upper-cased element string, or None if not determinable.
    """
    # BioPython Atom has an .element attribute
    elem: str = getattr(atom, "element", "").strip().upper()
    if elem:
        return elem
    # Fallback: derive from atom name (first non-digit character(s))
    name: str = getattr(atom, "name", "").strip()
    if not name:
        return None
    # Strip digits from the start
    alpha = "".join(c for c in name if c.isalpha())
    if alpha:
        return alpha[0:2].upper() if len(alpha) >= 2 and alpha[0:2].upper() in _ELEMENT_TO_Z else alpha[0].upper()
    return None


def _parse_pdb_id(file_path: Path) -> str:
    """Extract PDB ID from filename (e.g., '1a07.pdb' -> '1a07')."""
    return file_path.stem.lower()


def _convert_structure(file_path: Path) -> Optional[UnifiedRecord]:
    """Convert a single PDB/mmCIF file to a UnifiedRecord.

    Takes the first model only for multi-model files.
    Excludes hydrogen atoms.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    suffix = file_path.suffix.lower()
    pdb_id = _parse_pdb_id(file_path)

    try:
        if suffix in (".cif", ".mmcif"):
            cif_parser = MMCIFParser(QUIET=True)
            structure = cif_parser.get_structure(pdb_id, str(file_path))
        else:
            pdb_parser = PDBParser(QUIET=True)
            structure = pdb_parser.get_structure(pdb_id, str(file_path))
    except Exception:
        logger.warning("Failed to parse %s", file_path, exc_info=True)
        return None

    # Take first model only
    models = list(structure.get_models())
    if not models:
        logger.warning("No models found in %s, skipping.", file_path)
        return None
    model = models[0]

    coords_list: List[NDArray[np.floating]] = []
    z_list: List[int] = []

    for atom in model.get_atoms():
        elem = _atom_element(atom)
        if elem is None:
            continue
        # Skip hydrogens
        if elem == "H" or elem == "D":
            continue
        z = _ELEMENT_TO_Z.get(elem)
        if z is None:
            logger.debug("Unknown element %r in %s, skipping atom.", elem, file_path)
            continue
        z_list.append(z)
        coords_list.append(atom.get_vector().get_array())

    if len(z_list) == 0:
        logger.warning("No heavy atoms found in %s, skipping.", file_path)
        return None

    atom_types = np.array(z_list, dtype=np.int64)
    positions = np.array(coords_list, dtype=np.float64)
    num_atoms = len(atom_types)

    # All protein, no ligand
    component_mask = np.zeros(num_atoms, dtype=np.int64)

    neighbor_list = BaseConverter.compute_neighbor_list(positions, cutoff=5.0)

    record: UnifiedRecord = {
        "atom_types": atom_types,
        "positions": positions,
        "num_atoms": num_atoms,
        "dataset_source": "pdb_apo",
        "system_id": pdb_id,
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


class PDBApoConverter(BaseConverter):
    """Converter for apo PDB/mmCIF structures (protein only, no ligand)."""

    _VALID_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert a directory of PDB/mmCIF files to unified LMDB.

        Args:
            input_path: Directory containing PDB/mmCIF files.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: Split strategy ('random' supported). Default 80/10/10.
        """
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gather all structure files
        files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self._VALID_EXTENSIONS
        )
        if not files:
            logger.error("No PDB/mmCIF files found in %s", input_dir)
            return

        logger.info("Found %d structure files in %s", len(files), input_dir)

        # Convert all structures
        records: List[UnifiedRecord] = []
        for f in files:
            record = _convert_structure(f)
            if record is not None:
                records.append(record)

        if not records:
            logger.error("No valid records produced.")
            return

        logger.info("Converted %d / %d structures.", len(records), len(files))

        # Split 80/10/10
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(records))
        n = len(records)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)

        splits = {
            "train": indices[:n_train],
            "valid": indices[n_train : n_train + n_valid],
            "test": indices[n_train + n_valid :],
        }

        for split_name, split_idx in splits.items():
            split_records = [records[i] for i in split_idx]
            if split_records:
                out_path = str(output_dir / f"{split_name}.lmdb")
                self.write_lmdb(split_records, out_path)


def register() -> None:
    """Register PDB apo converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("pdb_apo", PDBApoConverter)


register()
