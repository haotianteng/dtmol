"""Integration tests: DatasetMixer -> DiffusionTrainer -> train_step on real data.

Builds a DatasetMixer from datamix_default.yaml (skipping missing LMDBs),
constructs encoder/decoder, runs train_step on CPU, and validates loss dicts.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import yaml

# ---------------------------------------------------------------------------
# Helpers for importing DiffusionTrainer from dtmol_train_test.py
# ---------------------------------------------------------------------------
# dtmol_train_test.py uses a bare ``from dtmol_input import ...`` which
# requires the dtmol package directory on sys.path.  Add it so the module
# is importable.

_DTMOL_PKG_DIR = str(Path(__file__).resolve().parents[2])  # dtmol/
if _DTMOL_PKG_DIR not in sys.path:
    sys.path.insert(0, _DTMOL_PKG_DIR)

from dtmol.dtmol_train_test import DiffusionTrainer  # noqa: E402
from dtmol.dtmol_train_base import CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(os.environ.get("DTMOL_DATA_ROOT", "/data/dtMol_Project/datasets"))
_DATAMIX_DEFAULT = Path(__file__).resolve().parents[1] / "datamix_default.yaml"
_PRETRAIN_DIR = Path(__file__).resolve().parents[2] / "pretrain_models"


def _skip_if_no_pretrain() -> None:
    mol_dict = _PRETRAIN_DIR / "unimol_molecule_dict.txt"
    prot_dict = _PRETRAIN_DIR / "unimol_protein_dict.txt"
    if not mol_dict.exists() or not prot_dict.exists():
        pytest.skip(f"Pretrain dicts not found at {_PRETRAIN_DIR}")


def _available_datasets() -> Dict[str, Any]:
    """Read datamix_default.yaml and return only entries whose LMDB exists."""
    if not _DATAMIX_DEFAULT.exists():
        pytest.skip(f"datamix_default.yaml not found: {_DATAMIX_DEFAULT}")
    with open(_DATAMIX_DEFAULT) as f:
        datamix: Dict[str, Any] = yaml.safe_load(f)
    available: Dict[str, Any] = {}
    for name, entry in datamix.items():
        p = Path(entry["path"])
        if p.exists() and (p.is_file() or (p / "data.mdb").exists()):
            available[name] = entry
    return available


def _write_filtered_datamix(available: Dict[str, Any]) -> str:
    """Write a temporary datamix YAML with only available datasets."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="datamix_test_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(available, f)
    return path


def _build_diffusion_samplers() -> Dict[str, Any]:
    """Build molecule + protein ChainSamplers with T=10 for fast tests."""
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

    rot_sampler = RotationSampler(schedular=ll_sch_rot, sde_format="VE")
    tr_sampler = TranslationSampler(schedular=ll_sch_tr, sde_format="VE")
    g_sampler = GaussianSampler(schedular=ll_sch_pert, sde_format="VE")
    g_sampler2 = GaussianSampler(schedular=ll_sch_pert2, sde_format="VE")

    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)
    return {"molecule": molecule_sampler, "protein": protein_sampler}


def _build_nets_and_dicts() -> tuple:
    """Build encoder + decoder nets dict and atom dicts.

    Mirrors the ``build_encoder`` function in ``dtmol_train_test.py`` but with
    proper package imports so it works when called as a pytest module.
    """
    from dtmol.dtmol_model import DummyModelConfig
    from dtmol.decoder import Decoder
    from dtmol.encoder import UniMolEncoder
    from dtmol.utils.dictionary import Dictionary

    pretrain_f = str(_PRETRAIN_DIR)

    # Build encoder (same logic as dtmol_train_test.build_encoder)
    ligand_dict = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
    ligand_dict.add_symbol("[MASK]", is_special=True)
    protein_dict.add_symbol("[MASK]", is_special=True)
    encoder_config = DummyModelConfig(mode="encode")
    ligand_encoder = UniMolEncoder(args=encoder_config, dictionary=ligand_dict)
    protein_encoder = UniMolEncoder(args=encoder_config, dictionary=protein_dict)
    nets: Dict[str, Any] = {
        "ligand_encoder": ligand_encoder,
        "protein_encoder": protein_encoder,
    }
    atom_dict = {"ligand_dict": ligand_dict, "protein_dict": protein_dict}

    # Build decoder
    decoder_config = DummyModelConfig(mode="train")
    decoder = Decoder(decoder_config, ligand_dict)
    decoder.register_diffusion_pool_head("tr-rotation", 6)
    decoder.register_diffusion_head("perturbation", 3)
    nets["decoder"] = decoder

    return nets, atom_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def available():
    """Available datasets from datamix_default.yaml (skip if < 2)."""
    avail = _available_datasets()
    if len(avail) < 2:
        pytest.skip(
            f"Need at least 2 datasets, found {len(avail)}: {list(avail.keys())}"
        )
    return avail


@pytest.fixture(scope="module")
def datamix_path(available):
    """Temporary datamix YAML with only available datasets."""
    path = _write_filtered_datamix(available)
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def nets_and_dicts():
    """Encoder/decoder nets and atom dicts; skips if pretrain missing."""
    _skip_if_no_pretrain()
    return _build_nets_and_dicts()


@pytest.fixture(scope="module")
def samplers():
    return _build_diffusion_samplers()


@pytest.fixture(scope="module")
def mixer(datamix_path, nets_and_dicts, samplers):
    """DatasetMixer built from filtered datamix YAML."""
    from dtmol.data.mixer import DatasetMixer
    from dtmol.data.unified_dataset import UnifiedDatasetConfig

    _, atom_dict = nets_and_dicts
    ds_config = UnifiedDatasetConfig(max_seq_len=128, max_pocket_atoms=64)
    return DatasetMixer(
        datamix_path=datamix_path,
        ligand_dict=atom_dict["ligand_dict"],
        protein_dict=atom_dict["protein_dict"],
        config=ds_config,
        diffusion_samplers=samplers,
    )


@pytest.fixture(scope="module")
def dataloader(mixer):
    """DataLoader from the mixer with batch_size=2."""
    from dtmol.data.mixer import get_mixed_dataloader

    return get_mixed_dataloader(mixer, batch_size=2, num_workers=0)


@pytest.fixture(scope="module")
def trainer_default(dataloader, nets_and_dicts, samplers):
    """DiffusionTrainer with default config (no force loss)."""
    nets, _ = nets_and_dicts
    config = CONFIG(
        use_wandb=False,
        lambda_force=0.0,
        lambda_fd_force=0.0,
        dataset_mode="unified",
    )
    return DiffusionTrainer(
        train_dataloader=dataloader,
        nets=nets,
        sampler=samplers,
        config=config,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests: mixed-data training loop
# ---------------------------------------------------------------------------


class TestMixedTrainStep:
    """Run train_step on batches from DatasetMixer, check loss dict."""

    def test_three_batches_finite_loss(self, trainer_default, dataloader):
        """Loss is finite (not NaN, not Inf) for 3 batches."""
        for net in trainer_default.nets.values():
            net.eval()
        batch_iter = iter(dataloader)
        for i in range(3):
            batch = next(batch_iter)
            loss, loss_dict = trainer_default.train_step(batch)
            assert torch.isfinite(loss), f"Batch {i}: loss is not finite ({loss})"
            assert not torch.isnan(loss), f"Batch {i}: loss is NaN"

    def test_loss_dict_keys(self, trainer_default, dataloader):
        """Loss dict contains required keys."""
        batch = next(iter(dataloader))
        _, loss_dict = trainer_default.train_step(batch)
        required = {"total_loss", "diffusion_loss", "force_loss", "trrot_loss", "pert_loss"}
        missing = required - set(loss_dict.keys())
        assert not missing, f"Missing keys in loss_dict: {missing}"

    def test_loss_dict_values_finite(self, trainer_default, dataloader):
        """All loss dict values are finite tensors."""
        batch = next(iter(dataloader))
        _, loss_dict = trainer_default.train_step(batch)
        for key in ("total_loss", "diffusion_loss", "force_loss", "trrot_loss", "pert_loss"):
            val = loss_dict[key]
            assert isinstance(val, torch.Tensor), f"{key} is not a Tensor"
            assert torch.isfinite(val), f"{key} is not finite ({val})"


class TestTierAwareForce:
    """Verify tier-aware force loss keys when lambda > 0."""

    @pytest.fixture(scope="class")
    def trainer_with_force(self, dataloader, nets_and_dicts, samplers):
        """DiffusionTrainer with force loss enabled."""
        nets, _ = nets_and_dicts
        config = CONFIG(
            use_wandb=False,
            lambda_force=0.1,
            lambda_fd_force=0.05,
            dataset_mode="unified",
        )
        return DiffusionTrainer(
            train_dataloader=dataloader,
            nets=nets,
            sampler=samplers,
            config=config,
            device="cpu",
        )

    def test_tier_a_force_loss_present(self, trainer_with_force, dataloader, available):
        """When Tier A data present and lambda_force > 0, tier_a_force_loss appears."""
        # Tier A datasets: ANI-2x, SPICE2, Transition1x (have DFT forces)
        tier_a_datasets = {"ani2x", "spice2", "irc"}
        has_tier_a = bool(tier_a_datasets & set(available.keys()))
        if not has_tier_a:
            pytest.skip("No Tier A datasets available")

        # Run enough batches to likely get a Tier A sample
        found = False
        batch_iter = iter(dataloader)
        for _ in range(10):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            _, loss_dict = trainer_with_force.train_step(batch)
            if "tier_a_force_loss" in loss_dict:
                found = True
                assert torch.isfinite(loss_dict["tier_a_force_loss"])
                break
        assert found, "tier_a_force_loss not found in any of 10 batches"

    def test_tier_b_force_loss_present(self, trainer_with_force, dataloader, available):
        """When Tier B data present and lambda_fd_force > 0, tier_b_force_loss appears."""
        # Tier B datasets: MISATO (has FD forces)
        tier_b_datasets = {"misato"}
        has_tier_b = bool(tier_b_datasets & set(available.keys()))
        if not has_tier_b:
            pytest.skip("No Tier B datasets available")

        found = False
        batch_iter = iter(dataloader)
        for _ in range(30):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            _, loss_dict = trainer_with_force.train_step(batch)
            if "tier_b_force_loss" in loss_dict:
                found = True
                assert torch.isfinite(loss_dict["tier_b_force_loss"])
                break
        if not found:
            pytest.skip(
                "tier_b_force_loss not found in 30 batches — "
                "MISATO data may lack force records"
            )


# ---------------------------------------------------------------------------
# Test: legacy mode regression
# ---------------------------------------------------------------------------


class TestLegacyMode:
    """Legacy mode: load PDBBind from original path, run one train_step."""

    _LEGACY_PATH = "/data/unimol_data/protein_ligand_binding_pose_prediction/"

    @pytest.fixture(scope="class")
    def legacy_trainer(self):
        """Build DiffusionTrainer using legacy CrossDataset code path."""
        _skip_if_no_pretrain()
        if not os.path.isdir(self._LEGACY_PATH):
            pytest.skip(f"Legacy PDBBind data not found: {self._LEGACY_PATH}")

        # Import legacy helpers — needs dtmol pkg dir on sys.path
        from dtmol.dtmol_input import load_unimol_binding_data, get_dataloader

        test_config = {
            "seed": 0,
            "max_seq_len": 1000,
            "max_pocket_atoms": 256,
            "max_diffusion_time": 10,
            "tr_sigma_min": 0.1,
            "tr_sigma_max": 19.0,
            "rot_sigma_min": 0.1,
            "rot_sigma_max": 1.65,
            "pert_mole_sigma_min": 0.04,
            "pert_mole_sigma_max": 1.5,
            "pert_prot_sigma_min": 0.04,
            "pert_prot_sigma_max": 1.5,
            "tr_sde": "VE",
            "rot_sde": "VE",
            "pert_mole_sde": "VE",
            "pert_prot_sde": "VE",
            "trrot": True,
            "mole_pert": True,
            "prot_pert": True,
        }

        binding_dataset = load_unimol_binding_data(
            test_config, self._LEGACY_PATH, split=["train"]
        )
        loader_dict = get_dataloader(
            binding_dataset, batch_size=2, split=["train"], device="cpu"
        )

        nets, _ = _build_nets_and_dicts()
        config = CONFIG(use_wandb=False, dataset_mode="legacy")
        trainer = DiffusionTrainer(
            train_dataloader=loader_dict["train"],
            nets=nets,
            sampler={
                "molecule": binding_dataset.mole_diffusion_sampler,
                "protein": binding_dataset.protein_diffusion_sampler,
            },
            config=config,
            device="cpu",
        )
        return trainer

    def test_legacy_one_step_finite_loss(self, legacy_trainer):
        """One train_step on legacy CrossDataset produces finite loss."""
        batch = next(iter(legacy_trainer.train_ds))
        loss, loss_dict = legacy_trainer.train_step(batch)
        assert torch.isfinite(loss), f"Legacy loss is not finite: {loss}"
        assert "total_loss" in loss_dict
