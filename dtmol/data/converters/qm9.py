"""QM9 converter: converts QM9 SDF + property CSV to unified LMDB format."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Hartree to eV conversion
HARTREE_TO_EV = 27.2114


def _read_property_csv(csv_path: str) -> Dict[str, Dict[str, float]]:
    """Read QM9 property CSV and return a dict keyed by molecule index.

    Supports both standard QM9 column names (idx, U0, homo, lumo, mu) and
    alternate names (mol_id, u0) via case-insensitive matching.

    Returns:
        dict mapping mol index (str) to property dict.
    """
    # Map from canonical name -> list of accepted CSV column names (lowercase)
    _COLUMN_ALIASES: Dict[str, List[str]] = {
        "idx": ["idx", "mol_id"],
        "mu": ["mu"],
        "homo": ["homo"],
        "lumo": ["lumo"],
        "U0": ["u0"],
    }

    properties: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]

        # Resolve column indices from header names
        col_indices: Dict[str, int] = {}
        for canonical, aliases in _COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in header:
                    col_indices[canonical] = header.index(alias)
                    break
            if canonical not in col_indices:
                logger.warning("Column '%s' not found in CSV header", canonical)

        idx_col = col_indices.get("idx")
        if idx_col is None:
            logger.error("No index column (idx/mol_id) found in CSV")
            return properties

        for row in reader:
            if len(row) <= max(col_indices.values()):
                continue
            idx = row[idx_col].strip()
            try:
                props: Dict[str, float] = {}
                for key in ("mu", "homo", "lumo", "U0"):
                    if key in col_indices:
                        props[key] = float(row[col_indices[key]])
                properties[idx] = props
            except (ValueError, IndexError):
                logger.warning("Skipping malformed CSV row for idx=%s", idx)
    return properties


def _parse_qm9_xyz(xyz_path: str) -> Optional[Dict[str, object]]:
    """Parse a QM9 .xyz file (extended XYZ format).

    QM9 XYZ format:
        Line 1: number of atoms
        Line 2: properties (gdb tag, index, A, B, C, mu, alpha, homo, lumo, gap,
                 r2, zpve, U0, U, H, G, Cv)
        Lines 3..N+2: atom_symbol x y z mulliken_charge
        Line N+3: SMILES (two variants)
        Line N+4: InChI

    Returns:
        dict with 'atoms' (list[str]), 'coords' (Nx3 array), 'properties' dict,
        'index' (int), or None on failure.
    """
    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        # Parse property line
        prop_line = lines[1].strip().split("\t")
        # prop_line[0] is 'gdb <index>', rest are properties
        idx_str = prop_line[0].split()[1] if len(prop_line[0].split()) > 1 else "0"
        idx = int(idx_str)

        props: Dict[str, float] = {}
        if len(prop_line) >= 16:
            props = {
                "mu": float(prop_line[4]),
                "homo": float(prop_line[7]),
                "lumo": float(prop_line[8]),
                "U0": float(prop_line[11]),
            }

        atoms: List[str] = []
        coords: List[List[float]] = []
        for i in range(2, 2 + num_atoms):
            parts = lines[i].strip().replace("*^", "e").split()
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return {
            "atoms": atoms,
            "coords": np.array(coords, dtype=np.float64),
            "properties": props,
            "index": idx,
        }
    except Exception:
        logger.warning("Failed to parse %s", xyz_path, exc_info=True)
        return None


# Element symbol -> atomic number
_SYMBOL_TO_Z: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
}


class QM9Converter(BaseConverter):
    """Converter for QM9 dataset (SDF or XYZ + property CSV) to unified format."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert QM9 data to unified LMDB format.

        Supports two input formats:
        1. Directory of .xyz files (QM9 extended XYZ format)
        2. SDF file + property CSV

        For XYZ directory: reads all .xyz files, properties embedded in files.
        For SDF: reads molecules via RDKit, properties from separate CSV.

        Args:
            input_path: Path to directory of .xyz files, or path to .sdf file.
                        If SDF, expects a properties CSV at <input_path>.csv or
                        <parent>/gdb9.sdf.csv.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: 'random' for 80/10/10 split (default).
        """
        input_p = Path(input_path)

        if input_p.is_dir():
            xyz_files = sorted(input_p.glob("*.xyz"))
            if xyz_files:
                records = self._convert_from_xyz_dir(input_p)
            else:
                # Fall back to SDF file in the directory
                sdf_files = sorted(input_p.glob("*.sdf"))
                if sdf_files:
                    logger.info("No .xyz files found, using SDF: %s", sdf_files[0])
                    records = self._convert_from_sdf(sdf_files[0])
                else:
                    raise ValueError(
                        f"Directory {input_path} contains neither .xyz nor .sdf files"
                    )
        elif input_p.suffix == ".sdf":
            records = self._convert_from_sdf(input_p)
        else:
            raise ValueError(
                f"Input must be a directory of .xyz files or a .sdf file, got: {input_path}"
            )

        if not records:
            logger.error("No records converted from %s", input_path)
            return

        logger.info("Converted %d total records from QM9", len(records))

        # Split 80/10/10
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(records))
        n = len(records)
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)

        splits = {
            "train": indices[:n_train],
            "valid": indices[n_train : n_train + n_valid],
            "test": indices[n_train + n_valid :],
        }

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_indices in splits.items():
            split_records = [records[i] for i in split_indices]
            out_path = str(output_dir / f"{split_name}.lmdb")
            self.write_lmdb(split_records, out_path)
            logger.info("Wrote %d %s records to %s", len(split_records), split_name, out_path)

    def _convert_from_xyz_dir(self, xyz_dir: Path) -> List[UnifiedRecord]:
        """Convert from a directory of QM9 .xyz files."""
        xyz_files = sorted(xyz_dir.glob("*.xyz"))
        logger.info("Found %d .xyz files in %s", len(xyz_files), xyz_dir)

        records: List[UnifiedRecord] = []
        for xyz_file in xyz_files:
            parsed = _parse_qm9_xyz(str(xyz_file))
            if parsed is None:
                continue

            atoms: List[str] = parsed["atoms"]  # type: ignore[assignment]
            coords: NDArray[np.floating] = parsed["coords"]  # type: ignore[assignment]
            props: Dict[str, float] = parsed["properties"]  # type: ignore[assignment]
            mol_idx: int = parsed["index"]  # type: ignore[assignment]

            atom_types = np.array(
                [_SYMBOL_TO_Z.get(a, 0) for a in atoms], dtype=np.int64
            )
            num_atoms = len(atom_types)
            neighbor_list = self.compute_neighbor_list(coords, cutoff=5.0)

            # Unit conversions: Hartree -> eV
            homo = props.get("homo")
            lumo = props.get("lumo")
            u0 = props.get("U0")
            dipole_mu = props.get("mu")

            record: UnifiedRecord = {
                "atom_types": atom_types,
                "positions": coords,
                "num_atoms": num_atoms,
                "dataset_source": "qm9",
                "system_id": f"gdb_{mol_idx}",
                "pes_tier": "C",
                "forces": None,
                "noise_target": None,
                "noise_level": None,
                "energy": u0 * HARTREE_TO_EV if u0 is not None else None,
                "binding_affinity": None,
                "relative_energy": None,
                "trajectory_id": None,
                "timestep": None,
                "positions_prev": None,
                "positions_next": None,
                "component_mask": None,
                "pocket_mask": None,
                "partial_charges": None,
                "dipole": np.array([dipole_mu, 0.0, 0.0], dtype=np.float64) if dipole_mu is not None else None,
                "homo": homo * HARTREE_TO_EV if homo is not None else None,
                "lumo": lumo * HARTREE_TO_EV if lumo is not None else None,
                "neighbor_list": neighbor_list,
            }
            records.append(record)

        return records

    def _convert_from_sdf(self, sdf_path: Path) -> List[UnifiedRecord]:
        """Convert from QM9 SDF file with property CSV."""
        from rdkit import Chem  # type: ignore[import-untyped]

        # Look for property CSV
        csv_path = sdf_path.with_suffix(".csv")
        if not csv_path.exists():
            csv_path = sdf_path.parent / "gdb9.sdf.csv"

        properties: Dict[str, Dict[str, float]] = {}
        if csv_path.exists():
            logger.info("Reading properties from %s", csv_path)
            properties = _read_property_csv(str(csv_path))
        else:
            logger.warning("No property CSV found, QM properties will be None")

        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

        records: List[UnifiedRecord] = []
        for mol_idx, mol in enumerate(suppl):
            if mol is None:
                logger.warning("Skipping invalid molecule at index %d", mol_idx)
                continue

            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()

            # Extract coordinates
            coords = np.zeros((num_atoms, 3), dtype=np.float64)
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                coords[i] = [pos.x, pos.y, pos.z]

            # Extract atomic numbers
            atom_types = np.array(
                [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)],
                dtype=np.int64,
            )

            neighbor_list = self.compute_neighbor_list(coords, cutoff=5.0)

            # Get properties: try both numeric key and gdb_N format
            mol_key = str(mol_idx + 1)
            props = properties.get(mol_key, {})
            if not props:
                props = properties.get(f"gdb_{mol_idx + 1}", {})

            homo = props.get("homo")
            lumo = props.get("lumo")
            u0 = props.get("U0")
            dipole_mu = props.get("mu")

            record: UnifiedRecord = {
                "atom_types": atom_types,
                "positions": coords,
                "num_atoms": num_atoms,
                "dataset_source": "qm9",
                "system_id": f"gdb_{mol_idx + 1}",
                "pes_tier": "C",
                "forces": None,
                "noise_target": None,
                "noise_level": None,
                "energy": u0 * HARTREE_TO_EV if u0 is not None else None,
                "binding_affinity": None,
                "relative_energy": None,
                "trajectory_id": None,
                "timestep": None,
                "positions_prev": None,
                "positions_next": None,
                "component_mask": None,
                "pocket_mask": None,
                "partial_charges": None,
                "dipole": np.array([dipole_mu, 0.0, 0.0], dtype=np.float64) if dipole_mu is not None else None,
                "homo": homo * HARTREE_TO_EV if homo is not None else None,
                "lumo": lumo * HARTREE_TO_EV if lumo is not None else None,
                "neighbor_list": neighbor_list,
            }
            records.append(record)

        return records


def register() -> None:
    """Register QM9 converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("qm9", QM9Converter)


register()
