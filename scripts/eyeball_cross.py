"""Build a 2D cross of Na+Cl, train an overfit denoiser on it, eyeball-check.

Structure (all atoms at z=0 so we can scatter-plot in xy):
    Na atoms (vertical bar):       (0, -2), (0, -1), (0, 0), (0, 1), (0, 2)
    Cl atoms (horizontal bar):     (-2, 0), (-1, 0),         (1, 0), (2, 0)
    -> 9 atoms forming a '+' shape with two element types.

Lattice spacing 1 unit (1 Å). Noise is small (sigma_pert_max=0.15) so the
cross shape is clearly recognisable in the 'noisy' panel and we can see
whether the denoised panel snaps atoms back toward their lattice sites.

Usage:
    python scripts/eyeball_cross.py --steps 1500 --device cuda \\
        --out-dir ralph/test_results/eyeball_cross
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


def build_cross_record() -> dict:
    """Return a UnifiedRecord-shaped dict for our cross structure."""
    # Na (Z=11) vertical bar (5 atoms), Cl (Z=17) horizontal bar (4 atoms).
    pos = np.array([
        [0.0, -2.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0],
        [0.0,  1.0, 0.0], [0.0,  2.0, 0.0],
        [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [ 1.0, 0.0, 0.0], [ 2.0, 0.0, 0.0],
    ], dtype=np.float64)
    types = np.array([11, 11, 11, 11, 11, 17, 17, 17, 17], dtype=np.int64)
    return {
        "atom_types": types,
        "positions": pos,
        "num_atoms": int(len(types)),
        "dataset_source": "unit_cell_synth",
        "system_id": "cross_NaCl_2d_seed000000",
        "pes_tier": "C",
        "forces": np.zeros_like(pos),
        "noise_target": np.zeros_like(pos),
        "noise_level": 0.0,
        "energy": 0.0,
    }


def write_cross_lmdb(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, subdir=True, map_size=10 * 1024 * 1024,
                    readonly=False, lock=False)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(build_cross_record()))
    env.close()
    print(f"[lmdb] wrote 1 cross record to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ralph/test_results/eyeball_cross")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-t", type=int, default=4500)
    # Small sigmas -> small noise the eyeball can resolve easily.
    ap.add_argument("--sigma-rot-max", type=float, default=0.11)
    ap.add_argument("--sigma-tr-max", type=float, default=0.11)
    ap.add_argument("--sigma-pert-max", type=float, default=0.15,
                    help="Small per-atom noise: at fixed_t=4500 this gives "
                         "noisy_RMSD ~ sigma * sqrt(3) ~ 0.15 A.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------- Build the LMDB ----------
    lmdb_path = os.path.join(args.out_dir, "cross_lmdb")
    write_cross_lmdb(lmdb_path)

    # ---------- Nets ----------
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

    # Patch sample_time to always return fixed_t (overfit at one t)
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

    # ---------- Trainer (single-loader) ----------
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

    # ---------- Train (overfit) ----------
    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w")
    print(f"[train] {args.steps} steps lr={args.lr}", flush=True)
    t0 = time.time()
    for i in range(args.steps):
        batch = to_device(next(train_iter))
        loss, _ = trainer.train_step(batch)
        if torch.isnan(loss):
            print(f"step {i}: NaN, skipping", flush=True)
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0 or i == args.steps - 1:
            line = f"step {i:5d}  loss={loss.item():.4f}"
            print(line, flush=True)
            log_f.write(line + "\n"); log_f.flush()
    log_f.close()
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)

    # ---------- Save checkpoint ----------
    ckpt_path = os.path.join(args.out_dir, "ckpt-final.pt")
    torch.save({k: net.state_dict() for k, net in trainer.nets.items()}, ckpt_path)
    print(f"[ckpt] {ckpt_path}", flush=True)

    # ---------- Eval (KEEP train mode for BatchNorm batch stats) ----------
    for net in trainer.nets.values():
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    def run_forward(batch_in):
        mol = {"src_tokens": batch_in['net_input']['mol_tokens'],
               "src_distance": batch_in['net_input']['mol_holo_distance'],
               "src_coord": batch_in['net_input']['mol_holo_coord'],
               "src_edge_type": batch_in['net_input']['mol_edge_type']}
        pkt = {"src_tokens": batch_in['net_input']['pocket_tokens'],
               "src_distance": batch_in['net_input']['pocket_distance'],
               "src_coord": batch_in['net_input']['pocket_holo_coord'],
               "src_edge_type": batch_in['net_input']['pocket_edge_type']}
        cd = batch_in['net_input']['cross_distance']
        ce = batch_in['net_input']['cross_edge_type']
        mole_t = batch_in['net_input']['mol_diffuse_time'].view(-1)
        (m_emb, m_attn, m_pad) = trainer.nets['ligand_encoder'](**mol, features_only=True)
        (p_emb, p_attn, p_pad) = trainer.nets['protein_encoder'](**pkt, features_only=True)
        out, _ = trainer.nets['decoder'](
            embd_molecule=m_emb, embd_protein=p_emb,
            coor_molecule=mol['src_coord'], coor_protein=pkt['src_coord'],
            timesteps=mole_t,
            padding_molecule=m_pad, padding_protein=p_pad,
            attn_mole=m_attn, attn_protein=p_attn,
            cross_distance=cd, cross_edges=ce,
            diffusion_heads=["tr-rotation", "perturbation"],
            single_molecule_mask=batch_in.get('single_molecule_mask'),
        )
        return out

    # Generate one diffused eval batch via the same sampler
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

    # Run model
    batch['net_input']['mol_holo_coord'] = mol_noisy
    batch['net_input']['mol_holo_distance'] = diff['mol_holo_distance']
    batch['net_input']['mol_diffuse_time'] = diff['mol_diffuse_time'].view(1, 1)
    batch['net_input']['pocket_diffuse_time'] = diff['pocket_diffuse_time'].view(1, 1)
    with torch.no_grad():
        out = run_forward(batch)
    eps_pred = out['perturbation'][:, :n_mol, :]
    eps_true = diff['mol_diffuse_perturb_score']
    valid = (diff['mol_diffuse_perturb_norm'] > 0)

    # Tweedie
    x0_est = mol_noisy.clone()
    x0_est[finite] = mol_noisy[finite] - sigma_t * eps_pred[finite]
    x0_np = x0_est.cpu().numpy()[0]

    rmsd_n = per_atom_rmsd(noisy_np, truth, mask_np)
    rmsd_d = per_atom_rmsd(x0_np, truth, mask_np)
    red = 1.0 - rmsd_d / max(rmsd_n, 1e-9)
    ep = eps_pred[valid].reshape(-1).float()
    et = eps_true[valid].reshape(-1).float()
    cosv = float((ep * et).sum().item() / (ep.norm() * et.norm() + 1e-9).item())
    print(f"[eval] noisy_RMSD={rmsd_n:.3f}A  denoised={rmsd_d:.3f}A  "
          f"reduction={red:+.1%}  cos={cosv:+.3f}", flush=True)

    report = {
        "structure": "cross_NaCl_2d (5 Na vertical + 4 Cl horizontal at z=0)",
        "ckpt": ckpt_path,
        "sigma_t": sigma_t, "fixed_t": cached_t,
        "sigma_pert_max": args.sigma_pert_max,
        "rmsd_noisy": rmsd_n, "rmsd_denoised": rmsd_d, "rmsd_reduction": red,
        "cos": cosv,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------- 2D visualization (z=0 by construction) ----------
    # Get atom types so we can colour Na/Cl differently.
    types = np.array(item.get("net_input", {}).get("mol_tokens",
                     batch['net_input']['mol_tokens'].cpu().numpy()[0]))
    # mol_tokens has BOS/EOS — we need atom types to colour real atoms only.
    rec = build_cross_record()
    real_types = rec["atom_types"]  # length 9
    # The collated mol_holo_coord has [BOS, atoms..., EOS] → 11 entries; mask
    # selects real atoms only (the 9 finite ones).
    real_idx_in_collate = np.where(mask_np)[0]
    truth_real = truth[real_idx_in_collate]
    noisy_real = noisy_np[real_idx_in_collate]
    denoised_real = x0_np[real_idx_in_collate]
    is_na = (real_types == 11)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f"clean truth", f"noisy (sigma={sigma_t:.3f}, RMSD={rmsd_n:.3f}A)",
              f"denoised (RMSD={rmsd_d:.3f}A, -{red*100:.1f}%)"]
    for ax, coords, title in zip(axes, [truth_real, noisy_real, denoised_real], titles):
        ax.scatter(coords[is_na, 0], coords[is_na, 1], s=200, c="C0",
                   marker="o", edgecolor="k", linewidth=1.5, label="Na (Z=11)")
        ax.scatter(coords[~is_na, 0], coords[~is_na, 1], s=200, c="C3",
                   marker="s", edgecolor="k", linewidth=1.5, label="Cl (Z=17)")
        # overlay clean truth as light grey for reference (except on the truth panel)
        if title != titles[0]:
            ax.scatter(truth_real[is_na, 0], truth_real[is_na, 1], s=200, c="none",
                       edgecolor="gray", linestyle="--", linewidth=1)
            ax.scatter(truth_real[~is_na, 0], truth_real[~is_na, 1], s=200, c="none",
                       edgecolor="gray", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Crossed grid (Na+Cl) — overfit denoise eyeball test", fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "cross_denoise.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)
    print(f"[done] artefacts in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
