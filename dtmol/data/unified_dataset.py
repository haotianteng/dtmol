"""UnifiedDataset: reads unified LMDB records and produces CrossDataset-compatible batches."""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lmdb
import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset

from dtmol.utils.dictionary import Dictionary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedDatasetConfig:
    max_seq_len: int = 512
    max_pocket_atoms: int = 256
    seed: int = 42
    frame_stride: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_atom_mapping() -> Dict[int, str]:
    """Load atomic-number -> UniMol symbol mapping from atom_mapping.json."""
    mapping_path = Path(__file__).parent / "atom_mapping.json"
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    # JSON keys are strings; convert to int
    return {int(k): v for k, v in raw.items() if k != "_comment"}


def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix (N, N) from (N, 3) coordinates."""
    return distance_matrix(coords, coords).astype(np.float32)


def _compute_cross_distance(mol_coords: np.ndarray, pkt_coords: np.ndarray) -> np.ndarray:
    """Cross distance matrix (M, P) between molecule and pocket coordinates."""
    return distance_matrix(mol_coords, pkt_coords).astype(np.float32)


def _compute_displacement(coords: np.ndarray) -> np.ndarray:
    """Displacement tensor (N, N, 3): disp[i,j] = coords[i] - coords[j]."""
    return (coords[:, None, :] - coords[None, :, :]).astype(np.float32)


def _compute_cross_displacement(mol_coords: np.ndarray, pkt_coords: np.ndarray) -> np.ndarray:
    """Cross displacement (M, P, 3): disp[i,j] = mol[i] - pkt[j]."""
    return (mol_coords[:, None, :] - pkt_coords[None, :, :]).astype(np.float32)


def _edge_type_matrix(tokens: np.ndarray, num_types: int) -> np.ndarray:
    """Edge type matrix (N, N): tokens[i]*num_types + tokens[j]."""
    return (tokens[:, None] * num_types + tokens[None, :]).astype(np.int64)


def _cross_edge_type(mol_tokens: np.ndarray, pkt_tokens: np.ndarray, num_types: int) -> np.ndarray:
    """Cross edge type (M, P): mol_tokens[i]*num_types + pkt_tokens[j]."""
    return (mol_tokens[:, None] * num_types + pkt_tokens[None, :]).astype(np.int64)


def _prepend_append_1d(arr: np.ndarray, pre_val: float, app_val: float) -> np.ndarray:
    """Prepend and append scalar values to a 1-D array."""
    return np.concatenate([[pre_val], arr, [app_val]])


def _prepend_append_coord(coords: np.ndarray, pad_val: float) -> np.ndarray:
    """Prepend and append padding rows to (N, 3) coordinate array."""
    pad_row = np.full((1, 3), pad_val, dtype=coords.dtype)
    return np.concatenate([pad_row, coords, pad_row], axis=0)


def _prepend_append_2d(mat: np.ndarray, pad_val: float) -> np.ndarray:
    """Add one row/col of pad_val on each side of a 2-D matrix."""
    h, w = mat.shape
    new = np.full((h + 2, w + 2), pad_val, dtype=mat.dtype)
    new[1:-1, 1:-1] = mat
    return new


def _prepend_append_3d(tensor: np.ndarray, pad_val: float) -> np.ndarray:
    """Add one row/col of pad_val on each side of a (N, N, 3) tensor."""
    h, w, d = tensor.shape
    new = np.full((h + 2, w + 2, d), pad_val, dtype=tensor.dtype)
    new[1:-1, 1:-1, :] = tensor
    return new


def _prepend_append_cross_2d(mat: np.ndarray, pad_val: float) -> np.ndarray:
    """Add BOS/EOS padding to cross-distance (M, P) -> (M+2, P+2)."""
    h, w = mat.shape
    new = np.full((h + 2, w + 2), pad_val, dtype=mat.dtype)
    new[1:-1, 1:-1] = mat
    return new


def _prepend_append_cross_3d(tensor: np.ndarray, pad_val: float) -> np.ndarray:
    """Add BOS/EOS padding to cross-displacement (M, P, 3) -> (M+2, P+2, 3)."""
    h, w, d = tensor.shape
    new = np.full((h + 2, w + 2, d), pad_val, dtype=tensor.dtype)
    new[1:-1, 1:-1, :] = tensor
    return new


# ---------------------------------------------------------------------------
# Collation helpers (match CrossDataset padding conventions)
# ---------------------------------------------------------------------------

def _pad_1d(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of 1-D tensors to common length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out


def _pad_coord(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of (N, 3) tensors to common length."""
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len, 3), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.size(0), :] = t
    return out


def _pad_2d(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of (N, N) tensors to common size."""
    max_size = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_size, max_size), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        n = t.size(0)
        out[i, :n, :n] = t
    return out


def _pad_3d(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of (N, N, 3) tensors to common size."""
    max_size = max(t.size(0) for t in tensors)
    d = tensors[0].size(-1)
    out = torch.full((len(tensors), max_size, max_size, d), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        n = t.size(0)
        out[i, :n, :n, :] = t
    return out


def _pad_cross_2d(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of (M, P) cross-distance tensors."""
    max_h = max(t.size(0) for t in tensors)
    max_w = max(t.size(1) for t in tensors)
    out = torch.full((len(tensors), max_h, max_w), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.size(0), :t.size(1)] = t
    return out


def _pad_cross_3d(tensors: List[torch.Tensor], pad_val: float) -> torch.Tensor:
    """Pad list of (M, P, 3) cross-displacement tensors."""
    max_h = max(t.size(0) for t in tensors)
    max_w = max(t.size(1) for t in tensors)
    d = tensors[0].size(-1)
    out = torch.full((len(tensors), max_h, max_w, d), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.size(0), :t.size(1), :] = t
    return out


# ---------------------------------------------------------------------------
# UnifiedDataset
# ---------------------------------------------------------------------------

class UnifiedDataset(Dataset):  # type: ignore[type-arg]
    """Reads unified LMDB and produces batches matching CrossDataset format.

    Each sample is a dict with 'net_input' containing mol_* and pocket_* keys
    plus 'pes_tier'.
    """

    def __init__(
        self,
        lmdb_path: str,
        ligand_dict: Dictionary,
        protein_dict: Dictionary,
        config: Optional[UnifiedDatasetConfig] = None,
        diffusion_samplers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.ligand_dict = ligand_dict
        self.protein_dict = protein_dict
        self.config = config or UnifiedDatasetConfig()
        self.diffusion_samplers = diffusion_samplers
        self._epoch = 1

        self._atom_mapping = _load_atom_mapping()

        # Open LMDB env lazily — just read keys now
        assert os.path.exists(lmdb_path), f"LMDB not found: {lmdb_path}"
        self._lmdb_subdir = os.path.isdir(lmdb_path)
        env = lmdb.open(lmdb_path, subdir=self._lmdb_subdir, readonly=True,
                        lock=False, readahead=False, meminit=False,
                        max_readers=256)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        env.close()
        self._env: Optional[lmdb.Environment] = None

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, subdir=self._lmdb_subdir, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256,
            )
        return self._env

    def __len__(self) -> int:
        return len(self._keys)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible diffusion sampling."""
        self._epoch = epoch

    def _read_record(self, idx: int) -> Dict[str, Any]:
        env = self._get_env()
        data = env.begin().get(self._keys[idx])
        assert data is not None
        return pickle.loads(data)

    def _z_to_ligand_tokens(self, atom_types: np.ndarray) -> np.ndarray:
        """Map atomic numbers to ligand dictionary indices."""
        symbols = [self._atom_mapping.get(int(z), "[UNK]") for z in atom_types]
        return np.array([self.ligand_dict.index(s) for s in symbols], dtype=np.int64)

    def _z_to_protein_tokens(self, atom_types: np.ndarray) -> np.ndarray:
        """Map atomic numbers to protein dictionary indices."""
        symbols = [self._atom_mapping.get(int(z), "[UNK]") for z in atom_types]
        return np.array([self.protein_dict.index(s) for s in symbols], dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._read_record(idx)

        # MISATO frame_stride: skip records where timestep doesn't match stride
        if (self.config.frame_stride > 1
                and record.get('dataset_source') == 'misato'):
            attempts = 0
            while ((record.get('timestep') or 0) % self.config.frame_stride != 0
                   and attempts < len(self)):
                idx = (idx + 1) % len(self)
                record = self._read_record(idx)
                attempts += 1

        atom_types: np.ndarray = record["atom_types"]
        positions: np.ndarray = np.array(record["positions"], dtype=np.float64)
        component_mask: Optional[np.ndarray] = record.get("component_mask")
        pes_tier: str = record["pes_tier"]

        # ----- Split into ligand / protein by component_mask -----
        if component_mask is not None and np.any(component_mask == 0):
            prot_idx = np.where(component_mask == 0)[0]
            lig_idx = np.where(component_mask == 1)[0]

            # Handle case where there are no ligand atoms (protein-only)
            if len(lig_idx) == 0:
                lig_idx = prot_idx
                prot_idx = np.array([], dtype=np.int64)

            lig_types = atom_types[lig_idx]
            lig_pos = positions[lig_idx]
            prot_types = atom_types[prot_idx] if len(prot_idx) > 0 else np.array([], dtype=atom_types.dtype)
            prot_pos = positions[prot_idx] if len(prot_idx) > 0 else np.zeros((0, 3), dtype=positions.dtype)
            has_protein = len(prot_idx) > 0
        else:
            # Single molecule — all atoms are "ligand"
            lig_types = atom_types
            lig_pos = positions
            prot_types = np.array([], dtype=atom_types.dtype)
            prot_pos = np.zeros((0, 3), dtype=positions.dtype)
            has_protein = False

        # Track whether this sample has no real protein (for cross-attention masking)
        single_molecule = not has_protein

        # ----- Truncate -----
        max_mol = self.config.max_seq_len
        max_pkt = self.config.max_pocket_atoms
        if len(lig_types) > max_mol:
            lig_types = lig_types[:max_mol]
            lig_pos = lig_pos[:max_mol]
        if has_protein and len(prot_types) > max_pkt:
            prot_types = prot_types[:max_pkt]
            prot_pos = prot_pos[:max_pkt]

        # ----- Normalize coordinates by protein centroid (or molecule centroid) -----
        if has_protein:
            centroid = prot_pos.mean(axis=0)
        else:
            centroid = lig_pos.mean(axis=0)
        lig_pos = (lig_pos - centroid).astype(np.float32)
        if has_protein:
            prot_pos = (prot_pos - centroid).astype(np.float32)

        # ----- Tokenize -----
        mol_tokens = self._z_to_ligand_tokens(lig_types)
        num_mol_dict = len(self.ligand_dict)
        num_pkt_dict = len(self.protein_dict)

        if has_protein:
            pkt_tokens = self._z_to_protein_tokens(prot_types)
        else:
            pkt_tokens = np.array([], dtype=np.int64)

        # ----- Compute matrices BEFORE BOS/EOS -----
        # Molecule
        mol_distance = _compute_distance_matrix(lig_pos)
        mol_displacement = _compute_displacement(lig_pos)

        # Protein
        if has_protein:
            pkt_distance = _compute_distance_matrix(prot_pos)
            pkt_displacement = _compute_displacement(prot_pos)
            cross_dist = _compute_cross_distance(lig_pos, prot_pos)
            cross_disp = _compute_cross_displacement(lig_pos, prot_pos)

        # ----- Add BOS / EOS -----
        # Tokens: prepend bos, append eos
        mol_tokens_padded = np.concatenate(
            [[self.ligand_dict.bos], mol_tokens, [self.ligand_dict.eos]]
        ).astype(np.int64)
        mol_edge = _edge_type_matrix(mol_tokens_padded, num_mol_dict)

        if has_protein:
            pkt_tokens_padded = np.concatenate(
                [[self.protein_dict.bos], pkt_tokens, [self.protein_dict.eos]]
            ).astype(np.int64)
            pkt_edge = _edge_type_matrix(pkt_tokens_padded, num_pkt_dict)
            cross_edge = _cross_edge_type(mol_tokens_padded, pkt_tokens_padded, num_mol_dict)
        else:
            # Dummy protein: single padding token so dual-encoder architecture
            # receives valid input shapes. The decoder must zero out cross-attention
            # weights when single_molecule_mask is True (see US-014).
            pkt_tokens_padded = np.array([1], dtype=np.int64)  # pad token
            pkt_edge = np.zeros((1, 1), dtype=np.int64)
            cross_edge = np.zeros((len(mol_tokens_padded), 1), dtype=np.int64)

        # Coordinates: prepend/append np.inf
        mol_coord = _prepend_append_coord(lig_pos, np.inf)
        mol_dist_padded = _prepend_append_2d(mol_distance, 0.0)
        mol_disp_padded = _prepend_append_3d(mol_displacement, 0.0)

        if has_protein:
            pkt_coord = _prepend_append_coord(prot_pos, np.inf)
            pkt_dist_padded = _prepend_append_2d(pkt_distance, 0.0)
            pkt_disp_padded = _prepend_append_3d(pkt_displacement, 0.0)
            cross_dist_padded = _prepend_append_cross_2d(cross_dist, 0.0)
            cross_disp_padded = _prepend_append_cross_3d(cross_disp, 0.0)
        else:
            # Dummy protein: zeros for coordinates and all distance/displacement
            pkt_coord = np.zeros((1, 3), dtype=np.float32)
            pkt_dist_padded = np.zeros((1, 1), dtype=np.float32)
            pkt_disp_padded = np.zeros((1, 1, 3), dtype=np.float32)
            cross_dist_padded = np.zeros((len(mol_tokens_padded), 1), dtype=np.float32)
            cross_disp_padded = np.zeros((len(mol_tokens_padded), 1, 3), dtype=np.float32)

        # ----- Build result dict -----
        result: Dict[str, Any] = {
            "net_input": {
                "mol_tokens": torch.from_numpy(mol_tokens_padded),
                "mol_edge_type": torch.from_numpy(mol_edge),
                "mol_src_coord": torch.from_numpy(mol_coord),
                "mol_src_distance": torch.from_numpy(mol_dist_padded),
                "mol_src_displacement": torch.from_numpy(mol_disp_padded),
                # holo == src for unified data (no separate apo/holo distinction)
                "mol_holo_coord": torch.from_numpy(mol_coord.copy()),
                "mol_holo_distance": torch.from_numpy(mol_dist_padded.copy()),
                "mol_holo_displacement": torch.from_numpy(mol_disp_padded.copy()),
                "pocket_tokens": torch.from_numpy(pkt_tokens_padded),
                "pocket_edge_type": torch.from_numpy(pkt_edge),
                "pocket_distance": torch.from_numpy(pkt_dist_padded),
                "pocket_displacement": torch.from_numpy(pkt_disp_padded),
                "pocket_src_coord": torch.from_numpy(pkt_coord.copy()) if has_protein else torch.from_numpy(pkt_coord),
                "pocket_holo_coord": torch.from_numpy(pkt_coord),
                "cross_distance": torch.from_numpy(cross_dist_padded),
                "cross_displacement": torch.from_numpy(cross_disp_padded),
                "cross_edge_type": torch.from_numpy(cross_edge),
            },
            "pes_tier": pes_tier,
            "holo_center_coordinates": torch.from_numpy(centroid.astype(np.float32)),
            # True when this sample has no real protein — the decoder must zero out
            # cross-attention weights (bias = -inf) for these samples so that
            # protein-to-ligand and ligand-to-protein attention produces zero weights.
            "single_molecule_mask": single_molecule,
        }

        # --- Diffusion sampling ---
        if self.diffusion_samplers is not None:
            mol_sampler = self.diffusion_samplers.get('molecule')
            prot_sampler = self.diffusion_samplers.get('protein')

            # Reproducible seeding (matches DiffusionDataset pattern)
            seed_val = int(hash((self.config.seed, self._epoch, idx)) % 1_000_000)
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)

            try:
                diffused: Dict[str, Any] = {}
                diffused_mol_np = lig_pos.copy()
                diffused_prot_np = prot_pos.copy() if has_protein else prot_pos

                # --- Molecule diffusion ---
                if mol_sampler is not None:
                    mol_input = lig_pos.astype(np.float64)[None, ...]  # (1, M, 3)
                    x_t, score, norm, ts = mol_sampler(mol_input)
                    diffused_mol_np = x_t[0].float().numpy()

                    score_0 = np.asarray(score[0], dtype=np.float32)
                    norm_0 = np.asarray(norm[0], dtype=np.float32)

                    # Chain score: first 2 = rot + tr (system-wise),
                    # rest = perturbation (per-atom)
                    trrot_score = score_0[:2]       # (2, 3)
                    perturb_score = score_0[2:]     # (M, 3)
                    trrot_norm = norm_0[:2]          # (2,)
                    perturb_norm = norm_0[2:]        # (M,)

                    perturb_score_pad = _prepend_append_coord(perturb_score, 0.0)
                    perturb_norm_pad = np.concatenate(
                        [[0.0], perturb_norm, [0.0]]
                    ).astype(np.float32)

                    d_mol_dist = _compute_distance_matrix(diffused_mol_np)
                    d_mol_disp = _compute_displacement(diffused_mol_np)

                    diffused['mol_holo_coord'] = torch.from_numpy(
                        _prepend_append_coord(diffused_mol_np, np.inf))
                    diffused['mol_holo_distance'] = torch.from_numpy(
                        _prepend_append_2d(d_mol_dist, 0.0))
                    diffused['mol_holo_displacement'] = torch.from_numpy(
                        _prepend_append_3d(d_mol_disp, 0.0))
                    diffused['mol_diffuse_time'] = torch.tensor(
                        int(ts[0]) if hasattr(ts, '__getitem__') else int(ts))
                    diffused['mol_diffuse_trrot_score'] = torch.from_numpy(
                        trrot_score)
                    diffused['mol_diffuse_perturb_score'] = torch.from_numpy(
                        perturb_score_pad)
                    diffused['mol_diffuse_trrot_norm'] = torch.from_numpy(
                        trrot_norm)
                    diffused['mol_diffuse_perturb_norm'] = torch.from_numpy(
                        perturb_norm_pad)

                # --- Protein diffusion ---
                if prot_sampler is not None and has_protein:
                    prot_input = prot_pos.astype(np.float64)[None, ...]
                    x_t_p, score_p, norm_p, ts_p = prot_sampler(prot_input)
                    diffused_prot_np = x_t_p[0].float().numpy()

                    sc_p = np.asarray(score_p[0], dtype=np.float32)
                    nm_p = np.asarray(norm_p[0], dtype=np.float32)

                    pkt_sc_pad = _prepend_append_coord(sc_p, 0.0)
                    pkt_nm_pad = np.concatenate(
                        [[0.0], nm_p, [0.0]]
                    ).astype(np.float32)

                    d_pkt_dist = _compute_distance_matrix(diffused_prot_np)
                    d_pkt_disp = _compute_displacement(diffused_prot_np)

                    diffused['pocket_holo_coord'] = torch.from_numpy(
                        _prepend_append_coord(diffused_prot_np, np.inf))
                    diffused['pocket_distance'] = torch.from_numpy(
                        _prepend_append_2d(d_pkt_dist, 0.0))
                    diffused['pocket_displacement'] = torch.from_numpy(
                        _prepend_append_3d(d_pkt_disp, 0.0))
                    diffused['pocket_diffuse_time'] = torch.tensor(
                        int(ts_p[0]) if hasattr(ts_p, '__getitem__')
                        else int(ts_p))
                    diffused['pocket_diffuse_score'] = torch.from_numpy(
                        pkt_sc_pad)
                    diffused['pocket_diffuse_norm'] = torch.from_numpy(
                        pkt_nm_pad)
                elif not has_protein:
                    # Dummy protein diffused entries (zeros)
                    diffused['pocket_holo_coord'] = torch.zeros(1, 3)
                    diffused['pocket_distance'] = torch.zeros(1, 1)
                    diffused['pocket_displacement'] = torch.zeros(1, 1, 3)
                    diffused['pocket_diffuse_time'] = torch.tensor(0)
                    diffused['pocket_diffuse_score'] = torch.zeros(1, 3)
                    diffused['pocket_diffuse_norm'] = torch.zeros(1)

                # --- Cross distance/displacement (between diffused coords) ---
                if has_protein:
                    d_cross = _compute_cross_distance(
                        diffused_mol_np, diffused_prot_np)
                    d_cross_disp = _compute_cross_displacement(
                        diffused_mol_np, diffused_prot_np)
                    cross_edge_d = _cross_edge_type(
                        mol_tokens_padded, pkt_tokens_padded, num_mol_dict)
                    diffused['cross_distance'] = torch.from_numpy(
                        _prepend_append_cross_2d(d_cross, 0.0))
                    diffused['cross_displacement'] = torch.from_numpy(
                        _prepend_append_cross_3d(d_cross_disp, 0.0))
                    diffused['cross_edge_type'] = torch.from_numpy(
                        cross_edge_d)
                else:
                    m_len = len(mol_tokens_padded)
                    diffused['cross_distance'] = torch.zeros(m_len, 1)
                    diffused['cross_displacement'] = torch.zeros(
                        m_len, 1, 3)
                    diffused['cross_edge_type'] = torch.zeros(
                        m_len, 1, dtype=torch.int64)

                result['diffused'] = diffused
            finally:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)

        # --- Force targets ---
        forces_raw = record.get('forces')
        if pes_tier in ('A', 'B') and forces_raw is not None:
            result['force_tier'] = pes_tier
            forces_all = np.array(forces_raw, dtype=np.float32)

            if component_mask is not None and np.any(component_mask == 0):
                f_prot_idx = np.where(component_mask == 0)[0]
                f_lig_idx = np.where(component_mask == 1)[0]
                if len(f_lig_idx) == 0:
                    f_lig_idx = f_prot_idx
                    f_prot_idx = np.array([], dtype=np.int64)
                mol_forces = forces_all[f_lig_idx][:max_mol]
                pkt_forces = (forces_all[f_prot_idx][:max_pkt]
                              if len(f_prot_idx) > 0
                              else np.zeros((0, 3), dtype=np.float32))
            else:
                mol_forces = forces_all[:max_mol]
                pkt_forces = np.zeros((0, 3), dtype=np.float32)

            mol_f_padded = _prepend_append_coord(mol_forces, 0.0)
            pkt_f_padded = (_prepend_append_coord(pkt_forces, 0.0)
                           if len(pkt_forces) > 0
                           else np.zeros((1, 3), dtype=np.float32))

            if 'diffused' in result:
                result['diffused']['mol_real_forces'] = torch.from_numpy(
                    mol_f_padded)
                result['diffused']['pocket_real_forces'] = torch.from_numpy(
                    pkt_f_padded)
        else:
            result['force_tier'] = 'none'
            if 'diffused' in result:
                # Zero placeholders so mixed-tier batches have uniform keys
                m_len = len(mol_tokens_padded)
                p_len = len(pkt_tokens_padded)
                result['diffused']['mol_real_forces'] = torch.zeros(
                    m_len, 3)
                result['diffused']['pocket_real_forces'] = torch.zeros(
                    p_len, 3)

        return result

    # -------------------------------------------------------------------
    # Collate
    # -------------------------------------------------------------------

    @staticmethod
    def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of sample dicts into a padded batch."""
        if len(samples) == 0:
            return {}

        net_inputs = [s["net_input"] for s in samples]

        batch_net: Dict[str, torch.Tensor] = {}

        # 1-D token sequences — pad with pad token (1)
        pad_token = 1  # [PAD] index in Dictionary
        batch_net["mol_tokens"] = _pad_1d(
            [ni["mol_tokens"] for ni in net_inputs], pad_token
        )
        batch_net["pocket_tokens"] = _pad_1d(
            [ni["pocket_tokens"] for ni in net_inputs], pad_token
        )

        # Coordinates — pad with np.inf
        for key in ["mol_src_coord", "mol_holo_coord", "pocket_src_coord", "pocket_holo_coord"]:
            batch_net[key] = _pad_coord(
                [ni[key] for ni in net_inputs], float("inf")
            )

        # 2-D distance/edge matrices — pad with 0
        for key in ["mol_edge_type", "mol_src_distance", "mol_holo_distance",
                     "pocket_edge_type", "pocket_distance"]:
            batch_net[key] = _pad_2d(
                [ni[key] for ni in net_inputs], 0
            )

        # 3-D displacement tensors — pad with 0
        for key in ["mol_src_displacement", "mol_holo_displacement", "pocket_displacement"]:
            batch_net[key] = _pad_3d(
                [ni[key] for ni in net_inputs], 0
            )

        # Cross 2-D (mol x pocket) — pad with 0
        for key in ["cross_distance", "cross_edge_type"]:
            batch_net[key] = _pad_cross_2d(
                [ni[key] for ni in net_inputs], 0
            )

        # Cross 3-D — pad with 0
        batch_net["cross_displacement"] = _pad_cross_3d(
            [ni["cross_displacement"] for ni in net_inputs], 0
        )

        batch: Dict[str, Any] = {
            "net_input": batch_net,
            "pes_tier": [s["pes_tier"] for s in samples],
            "holo_center_coordinates": torch.stack(
                [s["holo_center_coordinates"] for s in samples]
            ),
            # Per-sample boolean mask: True when sample has no real protein.
            # The decoder must zero out cross-attention weights (set bias to
            # -inf before softmax) for these samples so cross-attention
            # produces zero weights. See US-014.
            "single_molecule_mask": torch.tensor(
                [s["single_molecule_mask"] for s in samples], dtype=torch.bool
            ),
        }

        # --- Collate diffused dict ---
        if 'diffused' in samples[0]:
            dd = [s['diffused'] for s in samples]
            bd: Dict[str, torch.Tensor] = {}

            # Coordinates padded with inf
            for k in ('mol_holo_coord', 'pocket_holo_coord'):
                if k in dd[0]:
                    bd[k] = _pad_coord([d[k] for d in dd], float('inf'))

            # (N, 3) tensors padded with 0: scores, forces
            for k in ('mol_diffuse_trrot_score', 'mol_diffuse_perturb_score',
                       'pocket_diffuse_score',
                       'mol_real_forces', 'pocket_real_forces'):
                if k in dd[0]:
                    bd[k] = _pad_coord([d[k] for d in dd], 0.0)

            # (N, N) distance matrices
            for k in ('mol_holo_distance', 'pocket_distance'):
                if k in dd[0]:
                    bd[k] = _pad_2d([d[k] for d in dd], 0.0)

            # (N, N, 3) displacement tensors
            for k in ('mol_holo_displacement', 'pocket_displacement'):
                if k in dd[0]:
                    bd[k] = _pad_3d([d[k] for d in dd], 0.0)

            # Cross (M, P) matrices
            for k in ('cross_distance', 'cross_edge_type'):
                if k in dd[0]:
                    bd[k] = _pad_cross_2d([d[k] for d in dd], 0.0)

            # Cross (M, P, 3) displacement
            if 'cross_displacement' in dd[0]:
                bd['cross_displacement'] = _pad_cross_3d(
                    [d['cross_displacement'] for d in dd], 0.0)

            # 1-D norm arrays
            for k in ('mol_diffuse_trrot_norm', 'mol_diffuse_perturb_norm',
                       'pocket_diffuse_norm'):
                if k in dd[0]:
                    bd[k] = _pad_1d([d[k] for d in dd], 0.0)

            # Scalar time steps
            for k in ('mol_diffuse_time', 'pocket_diffuse_time'):
                if k in dd[0]:
                    bd[k] = torch.stack([d[k] for d in dd])

            # Combine trrot + perturb into a single mol_diffuse_score / norm
            # so DiffusionTrainer.diffusion_loss can slice [:, :2, :] for trrot.
            if 'mol_diffuse_trrot_score' in bd and 'mol_diffuse_perturb_score' in bd:
                bd['mol_diffuse_score'] = torch.cat(
                    [bd['mol_diffuse_trrot_score'], bd['mol_diffuse_perturb_score']],
                    dim=1)
            if 'mol_diffuse_trrot_norm' in bd and 'mol_diffuse_perturb_norm' in bd:
                bd['mol_diffuse_norm'] = torch.cat(
                    [bd['mol_diffuse_trrot_norm'], bd['mol_diffuse_perturb_norm']],
                    dim=1)

            batch['diffused'] = bd

        # Force tier (list of strings for per-sample dispatch)
        if 'force_tier' in samples[0]:
            batch['force_tier'] = [s['force_tier'] for s in samples]

        return batch
