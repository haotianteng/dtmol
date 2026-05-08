"""NaCl-style 2D checkerboard grid: train an overfit denoiser, eyeball-check.

Builds a W x H 2D rocksalt slab at z=0 with Na/Cl alternating like:

    Na Cl Na Cl Na Cl
    Cl Na Cl Na Cl Na
    Na Cl Na Cl Na Cl

Default 6x3 = 18 atoms. Lattice spacing 1 A. Train an overfit-one model
with small per-atom noise (sigma_pert_max=0.15), then plot the standard
truth/noisy/denoised triptych.

Usage:
    python scripts/eyeball_nacl_grid.py --width 6 --height 3 \\
        --steps 1500 --device cuda \\
        --out-dir ralph/test_results/eyeball_nacl_6x3
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import lmdb
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dtmol"))

from dtmol.dtmol_train_test import DiffusionTrainer
from dtmol.dtmol_train_base import CONFIG
from dtmol.dtmol_model import DummyModelConfig
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.utils.dictionary import Dictionary
from dtmol.data.unified_dataset import UnifiedDataset, UnifiedDatasetConfig
from dtmol.diffusion import (
    RotationSampler, GaussianSampler, TranslationSampler,
    ChainSampler, LogLinearScheduler,
)


def per_atom_rmsd(pred, truth, mask):
    return float(np.sqrt(((pred[mask] - truth[mask]) ** 2).sum(-1).mean()))


def kabsch_align(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray):
    """Find the rigid motion (R, t) that minimises RMSD(pred -> truth) and
    return (pred_aligned, R, t, rmsd_aligned).

    Allows a global translation + rotation. Useful when a learned denoiser
    has the right *internal* structure but is shifted/rotated from the
    reference (analogous to protein docking with receptor in a different
    pose than the reference).

    Standard Kabsch / orthogonal Procrustes:
      1. Centre both clouds on their centroids.
      2. H = P^T @ T  (covariance), SVD  H = U S V^T
      3. R = V @ diag(1, 1, sign(det(V U^T))) @ U^T  (proper rotation)
      4. t = centroid(truth) - R @ centroid(pred)
      5. pred_aligned = (pred - centroid(pred)) @ R^T + centroid(truth)
    """
    P = pred[mask].astype(np.float64)
    T = truth[mask].astype(np.float64)
    if P.shape[0] < 3:
        return pred.copy(), np.eye(3), np.zeros(3), per_atom_rmsd(pred, truth, mask)
    cP = P.mean(axis=0)
    cT = T.mean(axis=0)
    Pc = P - cP
    Tc = T - cT
    H = Pc.T @ Tc                  # 3x3
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T              # rotation (3x3)
    t = cT - R @ cP
    P_aligned = (P @ R.T) + (cT - cP @ R.T)  # = R @ (P - cP)^T transpose'd + cT
    out = pred.copy()
    out[mask] = P_aligned
    rmsd_a = float(np.sqrt(((P_aligned - T) ** 2).sum(-1).mean()))
    return out, R, t, rmsd_a


def build_grid_record(width: int, height: int, spacing: float = 1.0) -> dict:
    """W x H 2D NaCl checkerboard at z=0 centred at origin.

    atom_type at (i, j) = Na if (i+j) is even else Cl.
    """
    pos = []
    types = []
    for j in range(height):
        for i in range(width):
            x = (i - (width  - 1) / 2.0) * spacing
            y = (j - (height - 1) / 2.0) * spacing
            pos.append([x, y, 0.0])
            types.append(11 if (i + j) % 2 == 0 else 17)  # 11=Na, 17=Cl
    pos = np.array(pos, dtype=np.float64)
    types = np.array(types, dtype=np.int64)
    return {
        "atom_types": types,
        "positions": pos,
        "num_atoms": int(len(types)),
        "dataset_source": "unit_cell_synth",
        "system_id": f"nacl_grid_{width}x{height}_seed000000",
        "pes_tier": "C",
        "forces": np.zeros_like(pos),
        "noise_target": np.zeros_like(pos),
        "noise_level": 0.0,
        "energy": 0.0,
    }


def write_lmdb(path: str, record: dict) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, subdir=True, map_size=10 * 1024 * 1024,
                    readonly=False, lock=False)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(record))
    env.close()
    print(f"[lmdb] wrote 1 record to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ralph/test_results/eyeball_nacl_grid")
    ap.add_argument("--width", type=int, default=6, help="Grid width (atoms along x)")
    ap.add_argument("--height", type=int, default=3, help="Grid height (atoms along y)")
    ap.add_argument("--spacing", type=float, default=1.0, help="Lattice spacing (A)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-t", type=int, default=4500)
    ap.add_argument("--sigma-rot-max", type=float, default=0.11)
    ap.add_argument("--sigma-tr-max", type=float, default=0.11)
    ap.add_argument("--sigma-pert-max", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    record = build_grid_record(args.width, args.height, args.spacing)
    print(f"[grid] {args.width}x{args.height} atoms, "
          f"{int((record['atom_types']==11).sum())} Na, "
          f"{int((record['atom_types']==17).sum())} Cl, "
          f"spacing={args.spacing}A", flush=True)
    lmdb_path = os.path.join(args.out_dir, "grid_lmdb")
    write_lmdb(lmdb_path, record)

    pretrain_f = os.path.join(ROOT, "dtmol/pretrain_models")
    ligand_dict = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
    ligand_dict.add_symbol("[MASK]", is_special=True)
    protein_dict.add_symbol("[MASK]", is_special=True)

    enc_cfg = DummyModelConfig(mode="encode")
    le = UniMolEncoder(args=enc_cfg, dictionary=ligand_dict)
    pe = UniMolEncoder(args=enc_cfg, dictionary=protein_dict)
    dec_cfg = DummyModelConfig(mode="train")
    dec = Decoder(dec_cfg, ligand_dict)
    dec.register_diffusion_pool_head("tr-rotation", 6)
    dec.register_diffusion_head("perturbation", 3)
    nets = {"ligand_encoder": le, "protein_encoder": pe, "decoder": dec}

    T = 1000
    print(f"[samplers] sigma_max: rot={args.sigma_rot_max} tr={args.sigma_tr_max} "
          f"pert={args.sigma_pert_max}", flush=True)
    rot = RotationSampler(schedular=LogLinearScheduler(T, 0.1, max(args.sigma_rot_max, 0.11)),
                          sde_format="VE")
    tr = TranslationSampler(schedular=LogLinearScheduler(T, 0.1, max(args.sigma_tr_max, 0.11)),
                             sde_format="VE")
    g = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, args.sigma_pert_max),
                         sde_format="VE")
    g2 = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, args.sigma_pert_max),
                          sde_format="VE")
    mole_sampler = ChainSampler(rot).compose(tr).compose(g)
    prot_sampler = ChainSampler(g2)
    prot_sampler.conjugate(mole_sampler)

    cached_t = int(args.fixed_t)
    print(f"[overfit] fixed_t={cached_t}", flush=True)
    def _fixed_sample_time(size):
        return np.full(size if isinstance(size, int) else size, cached_t, dtype=int)
    for s in mole_sampler.samplers:
        s.sample_time = _fixed_sample_time

    ds_cfg = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    ds = UnifiedDataset(
        lmdb_path=lmdb_path,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_cfg,
        diffusion_samplers={"molecule": mole_sampler, "protein": prot_sampler},
    )

    config = CONFIG(lambda_force=0.0, lambda_fd_force=0.0, force_loss_fn="mse",
                    dataset_mode="unified", use_wandb=False)

    def loader():
        while True:
            item = ds[0]
            yield UnifiedDataset.collate_fn([item])

    class _SingleLoader:
        def __init__(self, gen, length=10):
            self._gen = gen; self._length = length
        def __iter__(self): return self._gen
        def __len__(self): return self._length

    train_iter = loader()
    eval_iter = loader()
    trainer = DiffusionTrainer(
        train_dataloader=_SingleLoader(train_iter),
        eval_dataloader=_SingleLoader(eval_iter),
        nets=nets,
        sampler={"molecule": mole_sampler, "protein": prot_sampler},
        config=config, device=args.device,
    )
    trainer.load_unimol_pretrain(pretrain_f)
    for net in trainer.nets.values():
        net.to(args.device)
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    optimizer = torch.optim.Adam(trainer.nets["decoder"].parameters(), lr=args.lr)

    def to_device(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(args.device)
            elif isinstance(v, dict):
                out[k] = to_device(v)
            else:
                out[k] = v
        return out

    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w")
    print(f"[train] {args.steps} steps lr={args.lr}", flush=True)
    t0 = time.time()
    for i in range(args.steps):
        batch = to_device(next(train_iter))
        loss, _ = trainer.train_step(batch)
        if torch.isnan(loss):
            print(f"step {i}: NaN, skipping", flush=True); continue
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if i % 100 == 0 or i == args.steps - 1:
            line = f"step {i:5d}  loss={loss.item():.4f}"
            print(line, flush=True); log_f.write(line + "\n"); log_f.flush()
    log_f.close()
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)

    ckpt_path = os.path.join(args.out_dir, "ckpt-final.pt")
    torch.save({k: net.state_dict() for k, net in trainer.nets.items()}, ckpt_path)
    print(f"[ckpt] {ckpt_path}", flush=True)

    for net in trainer.nets.values():
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    def run_forward(b_in):
        mol = {"src_tokens": b_in['net_input']['mol_tokens'],
               "src_distance": b_in['net_input']['mol_holo_distance'],
               "src_coord": b_in['net_input']['mol_holo_coord'],
               "src_edge_type": b_in['net_input']['mol_edge_type']}
        pkt = {"src_tokens": b_in['net_input']['pocket_tokens'],
               "src_distance": b_in['net_input']['pocket_distance'],
               "src_coord": b_in['net_input']['pocket_holo_coord'],
               "src_edge_type": b_in['net_input']['pocket_edge_type']}
        cd = b_in['net_input']['cross_distance']
        ce = b_in['net_input']['cross_edge_type']
        mt = b_in['net_input']['mol_diffuse_time'].view(-1)
        (m_emb, m_attn, m_pad) = trainer.nets['ligand_encoder'](**mol, features_only=True)
        (p_emb, p_attn, p_pad) = trainer.nets['protein_encoder'](**pkt, features_only=True)
        out, _ = trainer.nets['decoder'](
            embd_molecule=m_emb, embd_protein=p_emb,
            coor_molecule=mol['src_coord'], coor_protein=pkt['src_coord'],
            timesteps=mt,
            padding_molecule=m_pad, padding_protein=p_pad,
            attn_mole=m_attn, attn_protein=p_attn,
            cross_distance=cd, cross_edges=ce,
            diffusion_heads=["tr-rotation", "perturbation"],
            single_molecule_mask=b_in.get('single_molecule_mask'),
        )
        return out

    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    item = ds[0]
    batch = to_device(UnifiedDataset.collate_fn([item]))
    diff = batch['diffused']
    norms = diff['mol_diffuse_perturb_norm'].view(-1)
    sigma_t = float(norms[norms > 0][0].item())
    print(f"[eval] sigma_t (perturb) at t={cached_t} = {sigma_t:.3f}", flush=True)

    mol_clean = batch['net_input']['mol_holo_coord']
    mol_noisy = diff['mol_holo_coord']
    finite = torch.isfinite(mol_clean)
    n_mol = mol_clean.size(1)
    truth = mol_clean.cpu().numpy()[0]
    noisy_np = mol_noisy.cpu().numpy()[0]
    mask_np = finite.cpu().numpy()[0, :, 0]

    batch['net_input']['mol_holo_coord'] = mol_noisy
    batch['net_input']['mol_holo_distance'] = diff['mol_holo_distance']
    batch['net_input']['mol_diffuse_time'] = diff['mol_diffuse_time'].view(1, 1)
    batch['net_input']['pocket_diffuse_time'] = diff['pocket_diffuse_time'].view(1, 1)
    with torch.no_grad():
        out = run_forward(batch)
    eps_pred = out['perturbation'][:, :n_mol, :]
    eps_true = diff['mol_diffuse_perturb_score']
    valid = (diff['mol_diffuse_perturb_norm'] > 0)

    x0_est = mol_noisy.clone()
    x0_est[finite] = mol_noisy[finite] - sigma_t * eps_pred[finite]
    x0_np = x0_est.cpu().numpy()[0]

    rmsd_n = per_atom_rmsd(noisy_np, truth, mask_np)
    rmsd_d = per_atom_rmsd(x0_np, truth, mask_np)
    red = 1.0 - rmsd_d / max(rmsd_n, 1e-9)
    # Kabsch-aligned RMSDs: allow a global rigid motion before measuring.
    # If the model has learned the local structure but is offset by a
    # uniform translation/rotation (the rot+tr noise the perturbation head
    # leaves untouched), Kabsch RMSD reveals the *internal* match quality.
    noisy_aligned, _, _, rmsd_n_a = kabsch_align(noisy_np, truth, mask_np)
    denoised_aligned, _, _, rmsd_d_a = kabsch_align(x0_np, truth, mask_np)
    red_a = 1.0 - rmsd_d_a / max(rmsd_n_a, 1e-9)
    ep = eps_pred[valid].reshape(-1).float()
    et = eps_true[valid].reshape(-1).float()
    cosv = float((ep * et).sum().item() / (ep.norm() * et.norm() + 1e-9).item())
    print(f"[eval] noisy_RMSD={rmsd_n:.3f}A  denoised={rmsd_d:.3f}A  "
          f"reduction={red:+.1%}  cos={cosv:+.3f}", flush=True)
    print(f"[eval] (Kabsch-aligned) noisy={rmsd_n_a:.3f}A  denoised={rmsd_d_a:.3f}A  "
          f"reduction={red_a:+.1%}", flush=True)

    report = {
        "structure": f"nacl_grid_{args.width}x{args.height} (Z=11/Z=17 checkerboard)",
        "n_atoms": int(record["num_atoms"]),
        "ckpt": ckpt_path,
        "sigma_t": sigma_t, "fixed_t": cached_t,
        "sigma_pert_max": args.sigma_pert_max,
        "rmsd_noisy": rmsd_n, "rmsd_denoised": rmsd_d, "rmsd_reduction": red,
        "rmsd_noisy_kabsch": rmsd_n_a, "rmsd_denoised_kabsch": rmsd_d_a,
        "rmsd_reduction_kabsch": red_a,
        "cos": cosv,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    real_idx = np.where(mask_np)[0]
    truth_real = truth[real_idx]
    noisy_real = noisy_np[real_idx]
    denoised_real = x0_np[real_idx]
    noisy_aligned_real = noisy_aligned[real_idx]
    denoised_aligned_real = denoised_aligned[real_idx]
    real_types = record["atom_types"]
    is_na = (real_types == 11)

    def draw(ax, coords, is_na_local, title, draw_ghost=True):
        ax.scatter(coords[is_na_local, 0], coords[is_na_local, 1], s=120, c="C0",
                   marker="o", edgecolor="k", linewidth=1.0, label="Na (Z=11)")
        ax.scatter(coords[~is_na_local, 0], coords[~is_na_local, 1], s=120, c="C3",
                   marker="s", edgecolor="k", linewidth=1.0, label="Cl (Z=17)")
        if draw_ghost:
            # Match marker shape: circle for Na, SQUARE for Cl.
            ax.scatter(truth_real[is_na_local, 0], truth_real[is_na_local, 1],
                       s=120, c="none", marker="o",
                       edgecolor="gray", linestyle="--", linewidth=0.8)
            ax.scatter(truth_real[~is_na_local, 0], truth_real[~is_na_local, 1],
                       s=120, c="none", marker="s",
                       edgecolor="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-pad_x, pad_x); ax.set_ylim(-pad_y, pad_y)
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")

    pad_x = max(args.width / 2 + 0.5, 1.5)
    pad_y = max(args.height / 2 + 0.5, 1.5)

    panels = [
        ("clean truth", truth_real, False),
        (f"noisy raw\nRMSD={rmsd_n:.3f}A", noisy_real, True),
        (f"noisy Kabsch-aligned\nRMSD={rmsd_n_a:.3f}A", noisy_aligned_real, True),
        (f"denoised raw\nRMSD={rmsd_d:.3f}A (-{red*100:.1f}%)", denoised_real, True),
        (f"denoised Kabsch-aligned\nRMSD={rmsd_d_a:.3f}A (-{red_a*100:.1f}%)", denoised_aligned_real, True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (title, coords, gh) in zip(axes, panels):
        draw(ax, coords, is_na, title, draw_ghost=gh)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"NaCl checkerboard {args.width}×{args.height} — overfit denoise (sigma_pert={args.sigma_pert_max})",
                 fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "nacl_denoise.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)

    # ---------- Iterated Tweedie trajectory: show intermediate denoising ----------
    # Each step: model at fixed trained_t, compute eps_pred, Tweedie x0_est,
    # then re-noise x0_est with FRESH eps at the SAME sigma_train (the model's
    # in-distribution noise level). The sequence x0_est_1, x0_est_2, ... gives
    # multiple "denoising attempts" — the spread shows how reliable the
    # denoiser is on this structure.
    n_iter = 8
    iterates = []
    x_t_iter = mol_noisy.clone()
    for k in range(n_iter):
        batch['net_input']['mol_holo_coord'] = x_t_iter
        batch['net_input']['mol_diffuse_time'] = torch.tensor(
            [[cached_t]], device=args.device, dtype=torch.long)
        batch['net_input']['pocket_diffuse_time'] = torch.tensor(
            [[cached_t]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            out_k = run_forward(batch)
        eps_k = out_k['perturbation'][:, :n_mol, :]
        x0_k = x_t_iter.clone()
        x0_k[finite] = x_t_iter[finite] - sigma_t * eps_k[finite]
        x0_k_np = x0_k.cpu().numpy()[0]
        rmsd_k = per_atom_rmsd(x0_k_np, truth, mask_np)
        x0_k_aligned, _, _, rmsd_k_a = kabsch_align(x0_k_np, truth, mask_np)
        iterates.append({
            "step": k, "x0_est": x0_k_np[real_idx],
            "x0_est_aligned": x0_k_aligned[real_idx],
            "rmsd": rmsd_k, "rmsd_kabsch": rmsd_k_a,
        })
        # Re-noise at SAME sigma_train so model stays in-distribution
        torch.manual_seed(args.seed + 100 + k)
        fresh = torch.randn_like(mol_clean) * sigma_t
        x_t_iter = x0_k.clone()
        x_t_iter[finite] = x0_k[finite] + fresh[finite]

    fig2, axes2 = plt.subplots(2, n_iter, figsize=(3.5 * n_iter, 7))
    for k, it in enumerate(iterates):
        # Top row: raw x0_est
        ax = axes2[0, k]
        draw(ax, it["x0_est"], is_na,
             f"iter {k}\nRMSD={it['rmsd']:.3f}A", draw_ghost=True)
        if k == 0:
            ax.set_ylabel("raw x0_est", fontsize=11)
        # Bottom row: Kabsch-aligned x0_est
        ax = axes2[1, k]
        draw(ax, it["x0_est_aligned"], is_na,
             f"iter {k} (aligned)\nRMSD={it['rmsd_kabsch']:.3f}A", draw_ghost=True)
        if k == 0:
            ax.set_ylabel("Kabsch-aligned", fontsize=11)
    fig2.suptitle(f"Iterated Tweedie attempts ({n_iter} fresh-eps trials at sigma_train) — "
                  f"NaCl {args.width}×{args.height}", fontsize=13)
    plt.tight_layout()
    out_png2 = os.path.join(args.out_dir, "nacl_iterated.png")
    fig2.savefig(out_png2, dpi=140)
    plt.close(fig2)
    print(f"[plot] wrote {out_png2}", flush=True)
    report["iterated_rmsd"] = [it["rmsd"] for it in iterates]
    report["iterated_rmsd_kabsch"] = [it["rmsd_kabsch"] for it in iterates]
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] artefacts in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
