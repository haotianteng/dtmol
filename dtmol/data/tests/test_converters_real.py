"""Converter validation tests on real converted LMDBs.

Opens each dataset's train.lmdb and validates record-level correctness:
schema compliance via BaseConverter.validate_record(), unit ranges,
and field relationships.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import numpy as np
import pytest

from dtmol.data.converters.base import BaseConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_records(lmdb_path: str | Path, n: int = 10) -> List[Dict[str, Any]]:
    """Read first *n* records from an LMDB."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    records: List[Dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (_, v) in enumerate(cursor.iternext()):
            if i >= n:
                break
            records.append(pickle.loads(v))
    env.close()
    return records


def _lmdb_train_path(data_root: Path, dataset_name: str) -> Path:
    return data_root / dataset_name / "unified" / "train.lmdb"


# ---------------------------------------------------------------------------
# Dataset parametrization
# ---------------------------------------------------------------------------

DATASETS = ["QM9", "ANI-2x", "SPICE2", "Transition1x", "PDBBind", "MISATO", "PDB_apo"]


@pytest.fixture(params=DATASETS)
def dataset_records(request: pytest.FixtureRequest, data_root: Path):
    """Load first 10 records from the parametrized dataset, skip if missing."""
    name = request.param
    lmdb_path = _lmdb_train_path(data_root, name)
    if not lmdb_path.exists():
        pytest.skip(f"LMDB not found: {lmdb_path}")
    if lmdb_path.is_dir() and not (lmdb_path / "data.mdb").exists():
        pytest.skip(f"LMDB directory has no data.mdb: {lmdb_path}")
    records = _read_records(lmdb_path)
    if not records:
        pytest.skip(f"No records in {lmdb_path}")
    return name, records


# ---------------------------------------------------------------------------
# Shared: validate_record on every dataset
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """BaseConverter.validate_record() passes for all datasets."""

    def test_validate_record(self, dataset_records):
        name, records = dataset_records
        for i, rec in enumerate(records):
            try:
                BaseConverter.validate_record(rec)
            except ValueError as exc:
                pytest.fail(f"{name} record {i} failed validation: {exc}")


# ---------------------------------------------------------------------------
# Per-dataset specific checks
# ---------------------------------------------------------------------------


class TestQM9:
    """QM9-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "QM9")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("QM9 unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "qm9"

    def test_pes_tier(self):
        for rec in self.records:
            assert rec["pes_tier"] == "C"

    def test_energy_range(self):
        """QM9 total energies (U0 in eV) are large negative values."""
        for rec in self.records:
            e = rec["energy"]
            assert e is not None, "QM9 energy must not be None"
            assert np.isfinite(e), f"QM9 energy not finite: {e}"
            assert -50000 < e < 0, f"QM9 energy out of range: {e}"

    def test_homo_lt_lumo(self):
        for rec in self.records:
            homo = rec.get("homo")
            lumo = rec.get("lumo")
            if homo is not None and lumo is not None:
                assert homo < lumo, f"homo ({homo}) >= lumo ({lumo})"

    def test_forces_none(self):
        """QM9 is single-point; no forces available."""
        for rec in self.records:
            assert rec.get("forces") is None

    def test_num_atoms(self):
        for rec in self.records:
            assert 3 <= rec["num_atoms"] <= 29


class TestANI2x:
    """ANI-2x-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "ANI-2x")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("ANI-2x unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "ani2x"

    def test_forces_present(self):
        for rec in self.records:
            f = rec.get("forces")
            assert f is not None, "ANI-2x must have forces"
            assert f.shape == (rec["num_atoms"], 3)

    def test_force_magnitude(self):
        """Per-atom force magnitude should be < 50 eV/Angstrom."""
        for rec in self.records:
            f = rec["forces"]
            mag = np.linalg.norm(f, axis=1)
            assert np.all(mag < 50), f"Force magnitude too large: max={mag.max():.2f}"

    def test_energy_negative(self):
        for rec in self.records:
            e = rec["energy"]
            assert e is not None and e < 0, f"ANI-2x energy must be negative, got {e}"


class TestSPICE2:
    """SPICE2-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "SPICE2")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("SPICE2 unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "spice2"

    def test_forces_present(self):
        for rec in self.records:
            f = rec.get("forces")
            assert f is not None, "SPICE2 must have forces"
            assert f.shape == (rec["num_atoms"], 3)

    def test_energy_finite(self):
        for rec in self.records:
            e = rec["energy"]
            assert e is not None and np.isfinite(e), f"SPICE2 energy not finite: {e}"

    def test_partial_charges(self):
        """SPICE2 records should have partial charges when available."""
        has_charges = any(rec.get("partial_charges") is not None for rec in self.records)
        assert has_charges, "Expected at least some SPICE2 records with partial_charges"


class TestIRC:
    """Transition1x (IRC) specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "Transition1x")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("Transition1x unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "irc"

    def test_relative_energy(self):
        for rec in self.records:
            re = rec.get("relative_energy")
            assert re is not None, "IRC must have relative_energy"
            assert re >= 0, f"IRC relative_energy should be >= 0, got {re}"

    def test_timestep(self):
        for rec in self.records:
            ts = rec.get("timestep")
            assert ts is not None, "IRC must have timestep"
            assert ts >= 0, f"IRC timestep should be >= 0, got {ts}"

    def test_forces_present(self):
        for rec in self.records:
            f = rec.get("forces")
            assert f is not None, "IRC must have forces"
            assert f.shape == (rec["num_atoms"], 3)


class TestPDBBind:
    """PDBBind-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "PDBBind")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("PDBBind unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "pdbbind"

    def test_component_mask_both(self):
        """PDBBind should have both protein (0) and ligand (1) components."""
        for rec in self.records:
            cm = rec.get("component_mask")
            assert cm is not None, "PDBBind must have component_mask"
            unique = set(cm.tolist())
            assert 0 in unique and 1 in unique, f"Expected {{0, 1}} in component_mask, got {unique}"

    def test_num_atoms(self):
        for rec in self.records:
            assert rec["num_atoms"] > 10, f"PDBBind num_atoms too small: {rec['num_atoms']}"


class TestMISATO:
    """MISATO-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "MISATO")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("MISATO unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "misato"

    def test_trajectory_id_4char(self):
        """MISATO trajectory_id should be a 4-char PDB ID."""
        for rec in self.records:
            tid = rec.get("trajectory_id")
            assert tid is not None, "MISATO must have trajectory_id"
            assert isinstance(tid, str) and len(tid) == 4, (
                f"MISATO trajectory_id should be 4-char PDB ID, got {tid!r}"
            )

    def test_component_mask_present(self):
        for rec in self.records:
            cm = rec.get("component_mask")
            assert cm is not None, "MISATO must have component_mask"

    def test_num_atoms(self):
        for rec in self.records:
            assert rec["num_atoms"] > 50, f"MISATO num_atoms too small: {rec['num_atoms']}"


class TestPDBApo:
    """PDB apo-specific field checks."""

    @pytest.fixture(autouse=True)
    def _load(self, data_root: Path):
        path = _lmdb_train_path(data_root, "PDB_apo")
        if not path.exists() or (path.is_dir() and not (path / "data.mdb").exists()):
            pytest.skip("PDB_apo unified LMDB not found")
        self.records = _read_records(path)

    def test_dataset_source(self):
        for rec in self.records:
            assert rec["dataset_source"] == "pdb_apo"

    def test_component_mask_all_zero(self):
        """PDB apo is protein-only: all component_mask entries should be 0."""
        for rec in self.records:
            cm = rec.get("component_mask")
            assert cm is not None, "PDB_apo must have component_mask"
            assert np.all(cm == 0), f"PDB_apo component_mask should be all 0, got unique {set(cm.tolist())}"

    def test_num_atoms(self):
        for rec in self.records:
            assert rec["num_atoms"] > 20, f"PDB_apo num_atoms too small: {rec['num_atoms']}"
