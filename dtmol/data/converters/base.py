"""Base converter and unified record schema for multi-dataset integration."""

from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import lmdb
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class UnifiedRecord(TypedDict, total=False):
    """Unified data record shared across all dataset converters.

    Required fields are marked with `Required` in comments. All optional fields
    default to None when absent.
    """

    # --- required fields ---
    atom_types: NDArray[np.int_]  # (N,) atomic numbers
    positions: NDArray[np.floating]  # (N, 3) Cartesian coordinates in Angstrom
    num_atoms: int
    dataset_source: str
    system_id: str
    pes_tier: Literal["A", "B", "C"]

    # --- optional physics fields ---
    forces: Optional[NDArray[np.floating]]  # (N, 3) eV/A
    noise_target: Optional[NDArray[np.floating]]
    noise_level: Optional[float]
    energy: Optional[float]  # eV
    binding_affinity: Optional[float]
    relative_energy: Optional[float]  # eV

    # --- optional trajectory fields ---
    trajectory_id: Optional[str]
    timestep: Optional[int]
    positions_prev: Optional[NDArray[np.floating]]  # (N, 3)
    positions_next: Optional[NDArray[np.floating]]  # (N, 3)

    # --- optional structural fields ---
    component_mask: Optional[NDArray[np.int_]]  # (N,) 0=protein, 1=ligand
    pocket_mask: Optional[NDArray[np.bool_]]  # (N,)
    partial_charges: Optional[NDArray[np.floating]]  # (N,)
    dipole: Optional[NDArray[np.floating]]  # (3,)
    homo: Optional[float]  # eV
    lumo: Optional[float]  # eV
    neighbor_list: NDArray[np.int_]  # (K, 2) pairs within cutoff


_REQUIRED_FIELDS = {
    "atom_types",
    "positions",
    "num_atoms",
    "dataset_source",
    "system_id",
    "pes_tier",
    "neighbor_list",
}


class BaseConverter(abc.ABC):
    """Abstract base class for dataset converters."""

    @abc.abstractmethod
    def convert(
        self,
        input_path: str,
        output_path: str,
        split_strategy: str = "random",
    ) -> None:
        """Convert a dataset from its native format to unified LMDB.

        Args:
            input_path: Path to the source dataset.
            output_path: Directory where output LMDB files are written.
            split_strategy: How to split data (e.g. 'random', 'scaffold').
        """
        ...

    @staticmethod
    def compute_neighbor_list(
        positions: NDArray[np.floating],
        cutoff: float = 5.0,
    ) -> NDArray[np.int_]:
        """Find all atom pairs within *cutoff* Angstrom.

        Args:
            positions: (N, 3) array of atom coordinates.
            cutoff: Distance cutoff in Angstrom.

        Returns:
            (K, 2) int array of index pairs (i < j).
        """
        if len(positions) == 0:
            return np.empty((0, 2), dtype=np.int64)
        tree = KDTree(positions)
        pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return np.empty((0, 2), dtype=np.int64)
        return pairs.astype(np.int64)

    @staticmethod
    def validate_record(record: UnifiedRecord) -> bool:
        """Validate that *record* has all required fields with correct types/shapes.

        Raises:
            ValueError: If validation fails.

        Returns:
            True if the record is valid.
        """
        raw = dict(record)  # plain dict to allow variable-key access
        for field in _REQUIRED_FIELDS:
            if field not in raw or raw[field] is None:
                raise ValueError(f"Missing required field: {field}")

        atom_types = record["atom_types"]
        positions = record["positions"]
        num_atoms = record["num_atoms"]

        if not isinstance(atom_types, np.ndarray) or atom_types.ndim != 1:
            raise ValueError(
                f"atom_types must be a 1-D array, got shape {getattr(atom_types, 'shape', type(atom_types))}"
            )
        if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3) array, got shape {getattr(positions, 'shape', type(positions))}"
            )
        if atom_types.shape[0] != num_atoms:
            raise ValueError(
                f"atom_types length {atom_types.shape[0]} != num_atoms {num_atoms}"
            )
        if positions.shape[0] != num_atoms:
            raise ValueError(
                f"positions rows {positions.shape[0]} != num_atoms {num_atoms}"
            )
        if record["pes_tier"] not in ("A", "B", "C"):
            raise ValueError(f"pes_tier must be 'A', 'B', or 'C', got {record['pes_tier']!r}")

        neighbor_list = record["neighbor_list"]
        if not isinstance(neighbor_list, np.ndarray) or neighbor_list.ndim != 2 or neighbor_list.shape[1] != 2:
            raise ValueError(
                f"neighbor_list must be (K, 2) array, got shape {getattr(neighbor_list, 'shape', type(neighbor_list))}"
            )

        # Validate optional array shapes
        forces = record.get("forces")
        if forces is not None:
            if not isinstance(forces, np.ndarray) or forces.shape != (num_atoms, 3):
                raise ValueError(f"forces must be ({num_atoms}, 3), got {getattr(forces, 'shape', type(forces))}")

        component_mask = record.get("component_mask")
        if component_mask is not None:
            if not isinstance(component_mask, np.ndarray) or component_mask.shape != (num_atoms,):
                raise ValueError(f"component_mask must be ({num_atoms},)")

        dipole = record.get("dipole")
        if dipole is not None:
            if not isinstance(dipole, np.ndarray) or dipole.shape != (3,):
                raise ValueError(f"dipole must be (3,), got {getattr(dipole, 'shape', type(dipole))}")

        return True

    @staticmethod
    def write_lmdb(
        records: Sequence[UnifiedRecord],
        output_path: str,
        map_size: int = 1 << 40,
    ) -> None:
        """Write validated records to an LMDB file.

        Records are stored as pickle under sequential integer keys (b'0', b'1', ...).
        Logs stats on completion: number of records, atom count min/mean/max,
        and which optional fields are present.

        Args:
            records: Sequence of UnifiedRecord dicts.
            output_path: Path to output LMDB file.
            map_size: Maximum LMDB map size in bytes (default 1 TB).
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        env = lmdb.open(output_path, map_size=map_size)
        atom_counts: List[int] = []
        available_fields: set[str] = set()

        with env.begin(write=True) as txn:
            for idx, record in enumerate(records):
                BaseConverter.validate_record(record)
                txn.put(str(idx).encode(), pickle.dumps(record))
                atom_counts.append(record["num_atoms"])
                available_fields.update(
                    k for k, v in record.items() if v is not None
                )

        env.close()

        if atom_counts:
            arr = np.array(atom_counts)
            logger.info(
                "Wrote %d records to %s — atoms min=%d mean=%.1f max=%d — fields: %s",
                len(atom_counts),
                output_path,
                int(arr.min()),
                float(arr.mean()),
                int(arr.max()),
                ", ".join(sorted(available_fields)),
            )
        else:
            logger.warning("Wrote 0 records to %s", output_path)
