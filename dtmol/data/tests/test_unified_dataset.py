"""UnifiedDataset integration tests on all 7 real converted LMDBs.

Opens each dataset's train.lmdb via UnifiedDataset, validates output dict
structure from __getitem__, checks single_molecule_mask correctness, and
verifies batch collation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from dtmol.data.unified_dataset import UnifiedDataset, UnifiedDatasetConfig


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

# (dir_name, dataset_source, expected_single_molecule)
# single_molecule: True for molecule-only, False for protein-ligand,
#   True for protein-only (PDB apo — treated as single molecule internally)
DATASETS = [
    ("QM9", "qm9", True),
    ("ANI-2x", "ani2x", True),
    ("Transition1x", "irc", True),
    ("SPICE2", "spice2", True),
    ("PDBBind", "pdbbind", False),
    ("MISATO", "misato", False),
    ("PDB_apo", "pdb_apo", True),
]

DATASET_IDS = [d[0] for d in DATASETS]


def _lmdb_train_path(data_root: Path, name: str) -> Path:
    return data_root / name / "unified" / "train.lmdb"


# ---------------------------------------------------------------------------
# Required net_input keys
# ---------------------------------------------------------------------------

REQUIRED_NET_INPUT_KEYS = {
    "mol_tokens",
    "mol_edge_type",
    "mol_src_coord",
    "mol_src_distance",
    "mol_src_displacement",
    "mol_holo_coord",
    "mol_holo_distance",
    "mol_holo_displacement",
    "pocket_tokens",
    "pocket_edge_type",
    "pocket_distance",
    "pocket_displacement",
    "pocket_src_coord",
    "pocket_holo_coord",
    "cross_distance",
    "cross_displacement",
    "cross_edge_type",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=DATASETS, ids=DATASET_IDS)
def dataset_item(
    request: pytest.FixtureRequest,
    data_root: Path,
    ligand_dict,  # type: ignore[no-untyped-def]
    protein_dict,  # type: ignore[no-untyped-def]
    unified_config: UnifiedDatasetConfig,
):
    """Open UnifiedDataset for each parametrized source and return (item, meta).

    Skips if the LMDB or pretrain dicts are missing.
    """
    dir_name, source, expected_single = request.param
    lmdb_path = _lmdb_train_path(data_root, dir_name)

    if not lmdb_path.exists():
        pytest.skip(f"LMDB not found: {lmdb_path}")
    if lmdb_path.is_dir() and not (lmdb_path / "data.mdb").exists():
        pytest.skip(f"LMDB directory has no data.mdb: {lmdb_path}")

    ds = UnifiedDataset(
        lmdb_path=str(lmdb_path),
        ligand_dict=ligand_dict,
        protein_dict=protein_dict,
        config=unified_config,
    )
    assert len(ds) > 0, f"Dataset {dir_name} is empty"

    item = ds[0]
    return item, {
        "dir_name": dir_name,
        "source": source,
        "expected_single": expected_single,
        "ds": ds,
    }


# ---------------------------------------------------------------------------
# Tests: output dict structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Validate the dict returned by __getitem__."""

    def test_top_level_keys(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        assert "net_input" in item
        assert "pes_tier" in item
        assert "holo_center_coordinates" in item
        assert "single_molecule_mask" in item

    def test_net_input_keys(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        net_input = item["net_input"]
        missing = REQUIRED_NET_INPUT_KEYS - set(net_input.keys())
        assert not missing, f"Missing net_input keys: {missing}"

    def test_mol_tokens_shape(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        mol_tokens = item["net_input"]["mol_tokens"]
        assert mol_tokens.ndim == 1
        # At least 3 tokens: [BOS, one atom, EOS]
        assert mol_tokens.shape[0] >= 3

    def test_mol_distance_shape(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        n = item["net_input"]["mol_tokens"].shape[0]
        assert item["net_input"]["mol_src_distance"].shape == (n, n)
        assert item["net_input"]["mol_holo_distance"].shape == (n, n)
        assert item["net_input"]["mol_edge_type"].shape == (n, n)

    def test_mol_displacement_shape(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        n = item["net_input"]["mol_tokens"].shape[0]
        assert item["net_input"]["mol_src_displacement"].shape == (n, n, 3)
        assert item["net_input"]["mol_holo_displacement"].shape == (n, n, 3)

    def test_mol_coord_shape(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        n = item["net_input"]["mol_tokens"].shape[0]
        assert item["net_input"]["mol_src_coord"].shape == (n, 3)
        assert item["net_input"]["mol_holo_coord"].shape == (n, 3)

    def test_pocket_shapes_consistent(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        p = item["net_input"]["pocket_tokens"].shape[0]
        assert item["net_input"]["pocket_distance"].shape == (p, p)
        assert item["net_input"]["pocket_edge_type"].shape == (p, p)
        assert item["net_input"]["pocket_displacement"].shape == (p, p, 3)
        assert item["net_input"]["pocket_src_coord"].shape == (p, 3)
        assert item["net_input"]["pocket_holo_coord"].shape == (p, 3)

    def test_cross_shapes_consistent(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        n = item["net_input"]["mol_tokens"].shape[0]
        p = item["net_input"]["pocket_tokens"].shape[0]
        assert item["net_input"]["cross_distance"].shape == (n, p)
        assert item["net_input"]["cross_displacement"].shape == (n, p, 3)
        assert item["net_input"]["cross_edge_type"].shape == (n, p)

    def test_pes_tier_valid(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        assert item["pes_tier"] in ("A", "B", "C", "none")

    def test_holo_center_coordinates(self, dataset_item: tuple) -> None:
        item, _ = dataset_item
        c = item["holo_center_coordinates"]
        assert c.shape == (3,)
        assert torch.isfinite(c).all()

    def test_tensors_are_finite_where_expected(self, dataset_item: tuple) -> None:
        """Distances and edge types should be finite. Coords have inf for BOS/EOS."""
        item, _ = dataset_item
        ni = item["net_input"]
        for key in ["mol_src_distance", "mol_holo_distance", "pocket_distance",
                     "cross_distance"]:
            assert torch.isfinite(ni[key]).all(), f"{key} has non-finite values"
        for key in ["mol_edge_type", "pocket_edge_type", "cross_edge_type"]:
            assert torch.isfinite(ni[key].float()).all(), f"{key} has non-finite values"


# ---------------------------------------------------------------------------
# Tests: single_molecule_mask correctness
# ---------------------------------------------------------------------------


class TestSingleMoleculeMask:
    """Verify single_molecule_mask matches expected dataset type."""

    def test_single_molecule_mask_value(self, dataset_item: tuple) -> None:
        item, meta = dataset_item
        expected = meta["expected_single"]
        actual = item["single_molecule_mask"]
        assert actual == expected, (
            f"{meta['dir_name']}: expected single_molecule_mask={expected}, got {actual}"
        )

    def test_protein_ligand_has_real_pocket(self, dataset_item: tuple) -> None:
        """Protein-ligand datasets should have pocket tokens beyond dummy [PAD]."""
        item, meta = dataset_item
        if meta["expected_single"]:
            pytest.skip("Single-molecule dataset")
        p = item["net_input"]["pocket_tokens"].shape[0]
        # Real protein: at least [BOS, one atom, EOS] = 3 tokens
        assert p >= 3, f"{meta['dir_name']}: expected real pocket, got {p} tokens"

    def test_single_mol_has_dummy_pocket(self, dataset_item: tuple) -> None:
        """Single-molecule datasets should have a 1-token dummy pocket."""
        item, meta = dataset_item
        if not meta["expected_single"]:
            pytest.skip("Protein-ligand dataset")
        p = item["net_input"]["pocket_tokens"].shape[0]
        assert p == 1, f"{meta['dir_name']}: expected dummy pocket (1 token), got {p}"


# ---------------------------------------------------------------------------
# Tests: batch collation
# ---------------------------------------------------------------------------


class TestCollation:
    """Verify collate_fn works without error for each dataset."""

    def test_collate_two_samples(self, dataset_item: tuple) -> None:
        """collate_fn([ds[0], ds[1]]) should produce valid padded batch."""
        _, meta = dataset_item
        ds = meta["ds"]
        if len(ds) < 2:
            pytest.skip(f"{meta['dir_name']} has fewer than 2 samples")

        batch = UnifiedDataset.collate_fn([ds[0], ds[1]])

        # Batch should have net_input
        assert "net_input" in batch
        ni = batch["net_input"]

        # All required keys present
        missing = REQUIRED_NET_INPUT_KEYS - set(ni.keys())
        assert not missing, f"Missing keys in collated batch: {missing}"

        # Batch dimension should be 2
        assert ni["mol_tokens"].shape[0] == 2
        assert ni["pocket_tokens"].shape[0] == 2

        # single_molecule_mask should be a bool tensor of shape (2,)
        assert batch["single_molecule_mask"].shape == (2,)
        assert batch["single_molecule_mask"].dtype == torch.bool

        # holo_center_coordinates should be (2, 3)
        assert batch["holo_center_coordinates"].shape == (2, 3)

    def test_collated_shapes_consistent(self, dataset_item: tuple) -> None:
        """Verify 2D/3D tensor shapes are internally consistent after collation."""
        _, meta = dataset_item
        ds = meta["ds"]
        if len(ds) < 2:
            pytest.skip(f"{meta['dir_name']} has fewer than 2 samples")

        batch = UnifiedDataset.collate_fn([ds[0], ds[1]])
        ni = batch["net_input"]
        B, N = ni["mol_tokens"].shape
        _, P = ni["pocket_tokens"].shape

        assert ni["mol_src_distance"].shape == (B, N, N)
        assert ni["mol_holo_distance"].shape == (B, N, N)
        assert ni["mol_edge_type"].shape == (B, N, N)
        assert ni["mol_src_coord"].shape == (B, N, 3)
        assert ni["mol_holo_coord"].shape == (B, N, 3)

        assert ni["pocket_distance"].shape == (B, P, P)
        assert ni["pocket_edge_type"].shape == (B, P, P)
        assert ni["pocket_src_coord"].shape == (B, P, 3)
        assert ni["pocket_holo_coord"].shape == (B, P, 3)

        assert ni["cross_distance"].shape == (B, N, P)
        assert ni["cross_edge_type"].shape == (B, N, P)

    def test_collated_distances_finite(self, dataset_item: tuple) -> None:
        """Padded distances should be finite (padded with 0, not inf)."""
        _, meta = dataset_item
        ds = meta["ds"]
        if len(ds) < 2:
            pytest.skip(f"{meta['dir_name']} has fewer than 2 samples")

        batch = UnifiedDataset.collate_fn([ds[0], ds[1]])
        ni = batch["net_input"]
        for key in ["mol_src_distance", "mol_holo_distance", "pocket_distance",
                     "cross_distance"]:
            assert torch.isfinite(ni[key]).all(), f"Collated {key} has non-finite values"
