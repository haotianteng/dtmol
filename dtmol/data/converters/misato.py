"""MISATO converter: converts MISATO MD trajectory HDF5 to unified LMDB format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np

from dtmol.data.converters.base import BaseConverter, UnifiedRecord

logger = logging.getLogger(__name__)

# Atom type index -> element symbol mapping for MISATO
# MISATO uses integer atom type indices; the mapping depends on the specific
# encoding. Common protein+ligand atom types: H, C, N, O, S, P, F, Cl, Br, I
# We use periodic table atomic numbers directly when available.
# MISATO atoms_type stores atomic numbers directly.


def _load_dedup_list(dedup_path: str) -> Set[str]:
    """Load a list of PDB IDs to skip (for deduplication with PDBBind).

    The file should contain one PDB ID per line (case-insensitive).
    """
    pdb_ids: Set[str] = set()
    with open(dedup_path) as f:
        for line in f:
            pdb_id = line.strip().lower()
            if pdb_id:
                pdb_ids.add(pdb_id)
    logger.info("Loaded %d PDB IDs for deduplication from %s", len(pdb_ids), dedup_path)
    return pdb_ids


def _load_binding_affinities(index_path: str) -> Dict[str, float]:
    """Load binding affinities from a PDBBind-style index file.

    Expected format: lines with PDB ID and -logKd/Ki value, e.g.:
        1a07  3.00  ...
    Skips comment lines starting with '#'.
    """
    affinities: Dict[str, float] = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pdb_id = parts[0].lower()
                    affinity = float(parts[1])
                    affinities[pdb_id] = affinity
                except (ValueError, IndexError):
                    continue
    logger.info("Loaded %d binding affinities from %s", len(affinities), index_path)
    return affinities


def _identify_components(
    atoms_residue: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build component_mask from atoms_residue metadata.

    Protein residues are standard amino acid residue names.
    Ligand atoms have residue names like 'LIG', 'UNL', 'UNK', or non-standard names.
    Water atoms ('HOH', 'WAT', 'TIP3') are excluded (returns indices to keep).

    Returns:
        component_mask: (N,) array with 0=protein, 1=ligand for non-water atoms,
        or None if classification fails.
    """
    # Standard amino acid 3-letter codes
    PROTEIN_RESIDUES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # Common non-standard/modified
        "HIE", "HID", "HIP", "CYX", "ASH", "GLH",
    }
    WATER_RESIDUES = {"HOH", "WAT", "TIP3", "TIP", "SOL"}

    n = len(atoms_residue)
    mask = np.zeros(n, dtype=np.int64)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        res = atoms_residue[i]
        if isinstance(res, bytes):
            res = res.decode("utf-8")
        res = str(res).strip().upper()

        if res in WATER_RESIDUES:
            keep[i] = False
        elif res in PROTEIN_RESIDUES:
            mask[i] = 0  # protein
        else:
            mask[i] = 1  # ligand

    return mask[keep], keep


class MISATOConverter(BaseConverter):
    """Converter for MISATO MD trajectory HDF5 to unified format."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
        dedup_list: Optional[str] = None,
        no_dedup: bool = False,
        affinity_index: Optional[str] = None,
        dt: float = 0.08,
    ) -> None:
        """Convert MISATO HDF5 to unified LMDB format.

        Each HDF5 group represents a protein-ligand complex with trajectory frames.
        Creates one UnifiedRecord per frame.

        Args:
            input_path: Path to the MISATO HDF5 file.
            output_path: Directory where output unified LMDB files are written.
            split_strategy: 'random' splits by PDB ID (default).
            dedup_list: Path to file with PDB IDs to skip (for PDBBind dedup).
            no_dedup: If True, disable deduplication even if dedup_list given.
            affinity_index: Path to PDBBind index file for binding affinities.
            dt: Trajectory timestep in nanoseconds (default 0.08 ns = 80 ps for
                MISATO 100 frames over 8 ns).
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load dedup PDB IDs if provided
        dedup_ids: Set[str] = set()
        if dedup_list and not no_dedup:
            dedup_ids = _load_dedup_list(dedup_list)

        # Load binding affinities if provided
        affinities: Dict[str, float] = {}
        if affinity_index:
            affinities = _load_binding_affinities(affinity_index)

        # Collect records grouped by PDB ID (for splitting)
        pdb_records: Dict[str, List[UnifiedRecord]] = {}

        with h5py.File(input_p, "r") as f:
            pdb_groups = list(f.keys())
            logger.info("Found %d PDB groups in %s", len(pdb_groups), input_path)

            for pdb_id in pdb_groups:
                pdb_id_lower = pdb_id.strip().lower()

                # Deduplication check
                if pdb_id_lower in dedup_ids:
                    logger.debug("Skipping %s (in dedup list)", pdb_id)
                    continue

                pdb_group = f[pdb_id]

                # Trajectory coordinates: (T, N, 3) in Angstrom
                if "trajectory_coordinates" not in pdb_group:
                    logger.warning("Skipping %s: no trajectory_coordinates", pdb_id)
                    continue
                traj_coords = pdb_group["trajectory_coordinates"][()]  # (T, N, 3)

                # Atom types: (N,) atomic numbers
                if "atoms_type" not in pdb_group:
                    logger.warning("Skipping %s: no atoms_type", pdb_id)
                    continue
                atoms_type_raw = pdb_group["atoms_type"][()]

                # Atom residues for component classification
                if "atoms_residue" not in pdb_group:
                    logger.warning("Skipping %s: no atoms_residue", pdb_id)
                    continue
                atoms_residue = pdb_group["atoms_residue"][()]

                # Identify protein/ligand components and strip water
                result = _identify_components(atoms_residue)
                if result is None:
                    logger.warning("Skipping %s: could not classify components", pdb_id)
                    continue
                component_mask, keep_mask = result

                # Apply water stripping
                atom_types = np.array(atoms_type_raw[keep_mask], dtype=np.int64)
                num_atoms = len(atom_types)

                if num_atoms == 0:
                    logger.warning("Skipping %s: no atoms after water stripping", pdb_id)
                    continue

                # Binding affinity
                binding_affinity: Optional[float] = affinities.get(pdb_id_lower)

                n_frames = traj_coords.shape[0]
                records: List[UnifiedRecord] = []

                for frame_idx in range(n_frames):
                    # Strip water from coordinates
                    pos = np.array(traj_coords[frame_idx][keep_mask], dtype=np.float64)

                    # Previous and next frame positions (for finite-difference forces)
                    positions_prev: Optional[np.ndarray] = None
                    positions_next: Optional[np.ndarray] = None
                    forces: Optional[np.ndarray] = None

                    if frame_idx > 0:
                        positions_prev = np.array(
                            traj_coords[frame_idx - 1][keep_mask], dtype=np.float64
                        )

                    if frame_idx < n_frames - 1:
                        positions_next = np.array(
                            traj_coords[frame_idx + 1][keep_mask], dtype=np.float64
                        )
                        # Finite-difference force approximation: F ~ (x_{t+1} - x_t) / dt
                        # dt in ns, positions in Angstrom -> forces in A/ns
                        # This is a displacement-based proxy, not a true force in eV/A
                        forces = (
                            np.array(traj_coords[frame_idx + 1][keep_mask], dtype=np.float64)
                            - pos
                        ) / dt

                    neighbor_list = self.compute_neighbor_list(pos, cutoff=5.0)

                    record: UnifiedRecord = {
                        "atom_types": atom_types.copy(),
                        "positions": pos,
                        "num_atoms": num_atoms,
                        "dataset_source": "misato",
                        "system_id": f"{pdb_id}_frame{frame_idx}",
                        "pes_tier": "B",
                        "forces": forces,
                        "noise_target": None,
                        "noise_level": None,
                        "energy": None,
                        "binding_affinity": binding_affinity,
                        "relative_energy": None,
                        "trajectory_id": pdb_id,
                        "timestep": frame_idx,
                        "positions_prev": positions_prev,
                        "positions_next": positions_next,
                        "component_mask": component_mask.copy(),
                        "pocket_mask": None,
                        "partial_charges": None,
                        "dipole": None,
                        "homo": None,
                        "lumo": None,
                        "neighbor_list": neighbor_list,
                    }
                    records.append(record)

                pdb_records[pdb_id] = records

        # Split by PDB ID to avoid leakage
        pdb_ids = list(pdb_records.keys())
        rng = np.random.RandomState(42)
        pdb_indices = rng.permutation(len(pdb_ids))
        n_pdbs = len(pdb_ids)
        n_train = int(0.8 * n_pdbs)
        n_valid = int(0.1 * n_pdbs)

        splits = {
            "train": pdb_indices[:n_train],
            "valid": pdb_indices[n_train : n_train + n_valid],
            "test": pdb_indices[n_train + n_valid :],
        }

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_records = sum(len(recs) for recs in pdb_records.values())
        logger.info(
            "Converted %d total frames from %d complexes (skipped %d for dedup)",
            total_records,
            n_pdbs,
            len(dedup_ids),
        )

        for split_name, split_pdb_indices in splits.items():
            split_records: List[UnifiedRecord] = []
            for pi in split_pdb_indices:
                split_records.extend(pdb_records[pdb_ids[pi]])
            out_path = str(output_dir / f"{split_name}.lmdb")
            self.write_lmdb(split_records, out_path)
            logger.info(
                "Wrote %d %s records (%d complexes) to %s",
                len(split_records),
                split_name,
                len(split_pdb_indices),
                out_path,
            )


def register() -> None:
    """Register MISATO converter with the CLI."""
    from dtmol.data.convert import register_converter
    register_converter("misato", MISATOConverter)


register()
