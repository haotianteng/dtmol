"""DatasetMixer: weighted sampling across multiple UnifiedDataset instances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from dtmol.data.unified_dataset import UnifiedDataset, UnifiedDatasetConfig
from dtmol.utils.dictionary import Dictionary

logger = logging.getLogger(__name__)


class DatasetMixer(Dataset):  # type: ignore[type-arg]
    """Mixes multiple UnifiedDataset instances with configurable sampling weights.

    Each call to ``__getitem__`` draws from a dataset chosen by weighted random
    sampling based on the configured weights in ``datamix.yaml``.

    datamix.yaml format::

        pdbbind:
          path: /data/unified/pdbbind/train.lmdb
          weight: 0.5
        ani2x:
          path: /data/unified/ani2x/train.lmdb
          weight: 0.3
          frame_stride: 1
        qm9:
          path: /data/unified/qm9/train.lmdb
          weight: 0.2
    """

    def __init__(
        self,
        datamix_path: str,
        ligand_dict: Dictionary,
        protein_dict: Dictionary,
        config: Optional[UnifiedDatasetConfig] = None,
        diffusion_samplers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._config = config or UnifiedDatasetConfig()

        with open(datamix_path, "r") as f:
            datamix: Dict[str, Any] = yaml.safe_load(f)

        self._datasets: List[UnifiedDataset] = []
        self._names: List[str] = []
        weights_raw: List[float] = []

        for name, entry in datamix.items():
            ds_path = entry["path"]
            weight = float(entry.get("weight", 1.0))

            # Per-dataset config overrides (e.g. frame_stride for MISATO)
            ds_config = UnifiedDatasetConfig(
                max_seq_len=self._config.max_seq_len,
                max_pocket_atoms=self._config.max_pocket_atoms,
                seed=self._config.seed,
                frame_stride=int(entry.get("frame_stride", self._config.frame_stride)),
            )

            ds = UnifiedDataset(
                lmdb_path=ds_path,
                ligand_dict=ligand_dict,
                protein_dict=protein_dict,
                config=ds_config,
                diffusion_samplers=diffusion_samplers,
            )

            self._datasets.append(ds)
            self._names.append(name)
            weights_raw.append(weight)

            logger.info(
                "DatasetMixer: loaded '%s' with %d records, weight=%.3f",
                name, len(ds), weight,
            )

        assert len(self._datasets) > 0, "datamix.yaml must contain at least one dataset"

        # Normalize weights to probabilities
        total = sum(weights_raw)
        self._weights = np.array([w / total for w in weights_raw], dtype=np.float64)

        # Pre-compute cumulative lengths for total __len__
        self._lengths = [len(ds) for ds in self._datasets]
        self._cum_lengths = np.cumsum([0] + self._lengths)

        # RNG for weighted sampling
        self._rng = np.random.RandomState(self._config.seed)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1])

    def set_epoch(self, epoch: int) -> None:
        """Shuffle sampling order for DistributedSampler compatibility."""
        self._rng = np.random.RandomState(self._config.seed + epoch)
        for ds in self._datasets:
            ds.set_epoch(epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Choose a dataset by weighted random sampling
        ds_idx = int(self._rng.choice(len(self._datasets), p=self._weights))
        ds = self._datasets[ds_idx]
        # Map the global index to a local index within the chosen dataset
        local_idx = idx % len(ds)
        return ds[local_idx]

    @staticmethod
    def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Delegate to UnifiedDataset.collate_fn (shared across all datasets)."""
        return UnifiedDataset.collate_fn(samples)


def get_mixed_dataloader(
    mixer: DatasetMixer,
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from a DatasetMixer.

    Parameters
    ----------
    mixer : DatasetMixer
        The mixed dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data-loading workers.
    distributed : bool
        If True, wrap with DistributedSampler.
    """
    sampler = None
    shuffle = True
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(mixer)
        shuffle = False

    return DataLoader(
        mixer,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=DatasetMixer.collate_fn,
        pin_memory=True,
        sampler=sampler,
    )
