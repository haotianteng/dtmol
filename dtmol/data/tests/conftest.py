"""Shared pytest fixtures for dtmol data tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper: skip when an LMDB does not exist
# ---------------------------------------------------------------------------


def skip_if_no_lmdb(path: str | os.PathLike[str]) -> None:
    """Call ``pytest.skip()`` with a clear message if *path* is not an LMDB."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"LMDB not found: {p}")
    # An LMDB directory contains data.mdb; a flat file is also acceptable
    if p.is_dir() and not (p / "data.mdb").exists():
        pytest.skip(f"LMDB directory exists but has no data.mdb: {p}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Root directory for raw / converted datasets.

    Configurable via the ``DTMOL_DATA_ROOT`` environment variable.
    """
    return Path(os.environ.get("DTMOL_DATA_ROOT", "/data/dtMol_Project/datasets"))


@pytest.fixture(scope="session")
def ligand_dict():
    """Loaded molecule dictionary; skips if the pretrain dict file is missing."""
    dict_path = Path(__file__).resolve().parents[2] / "pretrain_models" / "unimol_molecule_dict.txt"
    if not dict_path.exists():
        pytest.skip(f"Molecule dict not found: {dict_path}")

    from dtmol.utils.dictionary import Dictionary

    return Dictionary.load(str(dict_path))


@pytest.fixture(scope="session")
def protein_dict():
    """Loaded protein dictionary; skips if the pretrain dict file is missing."""
    dict_path = Path(__file__).resolve().parents[2] / "pretrain_models" / "unimol_protein_dict.txt"
    if not dict_path.exists():
        pytest.skip(f"Protein dict not found: {dict_path}")

    from dtmol.utils.dictionary import Dictionary

    return Dictionary.load(str(dict_path))


@pytest.fixture(scope="session")
def diffusion_samplers():
    """Molecule and protein ChainSamplers with T=10 for fast tests."""
    from dtmol.diffusion import (
        ChainSampler,
        GaussianSampler,
        LogLinearScheduler,
        RotationSampler,
        TranslationSampler,
    )

    T = 10
    ll_sch_tr = LogLinearScheduler(T, sigma_min=0.1, sigma_max=19.0)
    ll_sch_rot = LogLinearScheduler(T, sigma_min=0.1, sigma_max=1.65)
    ll_sch_pert = LogLinearScheduler(T, sigma_min=0.04, sigma_max=1.5)
    ll_sch_pert2 = LogLinearScheduler(T, sigma_min=0.04, sigma_max=1.5)

    rot_sampler = RotationSampler(schedular=ll_sch_rot, sde_format="ve")
    tr_sampler = TranslationSampler(schedular=ll_sch_tr, sde_format="ve")
    g_sampler = GaussianSampler(schedular=ll_sch_pert, sde_format="ve")
    g_sampler2 = GaussianSampler(schedular=ll_sch_pert2, sde_format="ve")

    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)

    return {"molecule": molecule_sampler, "protein": protein_sampler}


@pytest.fixture(scope="session")
def unified_config():
    """Small UnifiedDatasetConfig for test speed."""
    from dtmol.data.unified_dataset import UnifiedDatasetConfig

    return UnifiedDatasetConfig(max_seq_len=128, max_pocket_atoms=64)
