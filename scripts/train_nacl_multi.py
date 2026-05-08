"""Train on >100 NaCl checkerboard grids of different sizes, eval on held-out 11×11.

Builds 192 records (64 base grids × 3 thermal-noise copies), trains with
the full t range (no fixed_t), and evaluates with a multi-step DDIM reverse
trajectory on a held-out 11×11 grid that was never seen during training.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python scripts/train_nacl_multi.py --steps 5000 --device cuda \\
        --out-dir ralph/test_results/nacl_multi
"""
from __future__ import annotations
import argparse
import json
import logging
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

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
from eyeball_nacl_grid import kabsch_align, per_atom_rmsd


# ---------------------------------------------------------------------------
# 1. Build LMDB with >100 NaCl checkerboard records of various sizes
# ---------------------------------------------------------------------------

def build_grid_record(width: int, height: int, spacing: float = 1.0,
                      thermal_sigma: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    pos = []
    types = []
    for j in range(height):
        for i in range(width):
            x = (i - (width  - 1) / 2.0) * spacing
            y = (j - (height - 1) / 2.0) * spacing
            pos.append([x, y, 0.0])
            types.append(11 if (i + j) % 2 == 0 else 17)
    pos = np.array(pos, dtype=np.float64)
    if thermal_sigma > 0:
        pos += rng.normal(0, thermal_sigma, pos.shape)
    types = np.array(types, dtype=np.int64)
    return {
        "atom_types": types,
        "positions": pos,
        "num_atoms": int(len(types)),
        "dataset_source": "unit_cell_synth",
        "system_id": f"nacl_{width}x{height}_s{seed:04d}",
        "pes_tier": "C",
        "forces": np.zeros_like(pos),
        "noise_target": np.zeros_like(pos),
        "noise_level": 0.0,
        "energy": 0.0,
    }


def write_multi_lmdb(path: str, thermal_sigma: float = 0.02) -> int:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, subdir=True, map_size=100 * 1024 * 1024,
                    readonly=False, lock=False)
    idx = 0
    with env.begin(write=True) as txn:
        for w in range(3, 11):          # width 3..10
            for h in range(3, 11):      # height 3..10
                for copy in range(3):   # 3 thermal noise copies
                    rec = build_grid_record(w, h, spacing=1.0,
                                           thermal_sigma=thermal_sigma,
                                           seed=w * 1000 + h * 100 + copy)
                    txn.put(str(idx).encode(), pickle.dumps(rec))
                    idx += 1
    env.close()
    return idx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_device(d, device):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out


def draw_panel(ax, coords, is_na, truth_real, title, pad_x, pad_y,
               draw_ghost=True, marker_size=60):
    ax.scatter(coords[is_na, 0], coords[is_na, 1], s=marker_size, c="C0",
               marker="o", edgecolor="k", linewidth=0.8, label="Na")
    ax.scatter(coords[~is_na, 0], coords[~is_na, 1], s=marker_size, c="C3",
               marker="s", edgecolor="k", linewidth=0.8, label="Cl")
    if draw_ghost:
        ax.scatter(truth_real[is_na, 0], truth_real[is_na, 1],
                   s=marker_size, c="none", marker="o",
                   edgecolor="gray", linestyle="--", linewidth=0.6)
        ax.scatter(truth_real[~is_na, 0], truth_real[~is_na, 1],
                   s=marker_size, c="none", marker="s",
                   edgecolor="gray", linestyle="--", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-pad_x, pad_x); ax.set_ylim(-pad_y, pad_y)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-width", type=int, default=11)
    ap.add_argument("--eval-height", type=int, default=11)
    ap.add_argument("--traj-steps", type=int, default=50)
    ap.add_argument("--sigma-pert-max", type=float, default=0.3)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1,
                    help="Accumulate gradients over N forward passes before "
                         "optimizer.step(). Effective batch = 1 × N.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.WARNING)

    # ---- 1. build LMDB ----
    lmdb_path = os.path.join(args.out_dir, "train.lmdb")
    n_rec = write_multi_lmdb(lmdb_path)
    print(f"[data] wrote {n_rec} NaCl grid records to {lmdb_path}", flush=True)

    # ---- nets ----
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

    # ---- samplers (full t range, NO fixed_t) ----
    T = 1000
    sigma_rot_max = 0.11
    sigma_tr_max = 0.11
    sigma_pert_max = args.sigma_pert_max
    print(f"[samplers] sigma_max: rot={sigma_rot_max} tr={sigma_tr_max} "
          f"pert={sigma_pert_max}", flush=True)
    rot = RotationSampler(
        schedular=LogLinearScheduler(T, 0.1, max(sigma_rot_max, 0.11)),
        sde_format="VE")
    tr = TranslationSampler(
        schedular=LogLinearScheduler(T, 0.1, max(sigma_tr_max, 0.11)),
        sde_format="VE")
    g = GaussianSampler(
        schedular=LogLinearScheduler(T, 0.04, sigma_pert_max),
        sde_format="VE")
    g2 = GaussianSampler(
        schedular=LogLinearScheduler(T, 0.04, sigma_pert_max),
        sde_format="VE")
    mole_sampler = ChainSampler(rot).compose(tr).compose(g)
    prot_sampler = ChainSampler(g2)
    prot_sampler.conjugate(mole_sampler)

    ds_cfg = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    ds = UnifiedDataset(
        lmdb_path=lmdb_path,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_cfg,
        diffusion_samplers={"molecule": mole_sampler, "protein": prot_sampler},
    )
    print(f"[data] dataset length = {len(ds)}", flush=True)

    # ---- trainer ----
    config = CONFIG(lambda_force=0.0, lambda_fd_force=0.0, force_loss_fn="mse",
                    dataset_mode="unified", use_wandb=False)

    def loader(dataset):
        while True:
            for idx in np.random.permutation(len(dataset)):
                item = dataset[int(idx)]
                yield UnifiedDataset.collate_fn([item])

    class _LoopLoader:
        def __init__(self, gen, length=10):
            self._gen = gen; self._length = length
        def __iter__(self): return self._gen
        def __len__(self): return self._length

    train_gen = loader(ds)
    trainer = DiffusionTrainer(
        train_dataloader=_LoopLoader(train_gen),
        eval_dataloader=_LoopLoader(loader(ds)),
        nets=nets,
        sampler={"molecule": mole_sampler, "protein": prot_sampler},
        config=config, device=args.device,
    )
    trainer.load_unimol_pretrain(pretrain_f)
    for net in trainer.nets.values():
        net.to(args.device)
        net.train()
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    optimizer = torch.optim.Adam(trainer.nets["decoder"].parameters(), lr=args.lr)
    accum_steps = args.gradient_accumulation_steps

    # ---- 2. train ----
    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w")
    print(f"[train] {args.steps} steps, lr={args.lr}, "
          f"grad_accum={accum_steps} (effective_bsz={accum_steps})",
          flush=True)
    t0 = time.time()
    optimizer.zero_grad()
    for i in range(args.steps):
        batch = to_device(next(train_gen), args.device)
        loss, _ = trainer.train_step(batch)
        if torch.isnan(loss):
            continue
        scaled_loss = loss / accum_steps
        scaled_loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if i % 100 == 0 or i == args.steps - 1:
            line = f"step {i:5d}  loss={loss.item():.4f}"
            print(line, flush=True); log_f.write(line + "\n"); log_f.flush()
    # Flush any remaining accumulated gradients
    if args.steps % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    log_f.close()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)", flush=True)

    ckpt_path = os.path.join(args.out_dir, "ckpt-final.pt")
    torch.save({k: net.state_dict() for k, net in trainer.nets.items()}, ckpt_path)
    print(f"[ckpt] {ckpt_path}", flush=True)

    # ---- 3. eval: multi-step DDIM on held-out 11×11 ----
    # KEEP train mode (BatchNorm uses batch stats, matching training regime).
    for net in trainer.nets.values():
        net.train()
        for m in net.modules():
            if isinstance(m, nn.Dropout):
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
        (m_emb, m_attn, m_pad) = nets['ligand_encoder'](**mol, features_only=True)
        (p_emb, p_attn, p_pad) = nets['protein_encoder'](**pkt, features_only=True)
        out, _ = nets['decoder'](
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

    # Build held-out record
    eval_rec = build_grid_record(args.eval_width, args.eval_height,
                                 spacing=1.0, thermal_sigma=0.0, seed=99999)
    eval_lmdb_path = os.path.join(args.out_dir, "eval.lmdb")
    if os.path.isdir(eval_lmdb_path):
        shutil.rmtree(eval_lmdb_path)
    os.makedirs(eval_lmdb_path, exist_ok=True)
    env = lmdb.open(eval_lmdb_path, subdir=True, map_size=10 * 1024 * 1024,
                    readonly=False, lock=False)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(eval_rec))
    env.close()

    eval_ds = UnifiedDataset(
        lmdb_path=eval_lmdb_path,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_cfg,
        diffusion_samplers=None,  # we add our own noise for the trajectory
    )
    print(f"[eval] held-out {args.eval_width}×{args.eval_height} grid "
          f"({eval_rec['num_atoms']} atoms)", flush=True)

    # Get clean batch
    torch.manual_seed(args.seed + 42)
    np.random.seed(args.seed + 42)
    item_eval = eval_ds[0]
    batch_eval = to_device(UnifiedDataset.collate_fn([item_eval]), args.device)
    mol_clean = batch_eval['net_input']['mol_holo_coord'].clone()
    finite = torch.isfinite(mol_clean)
    truth = mol_clean.cpu().numpy()[0]
    mask_np = finite.cpu().numpy()[0, :, 0]
    n_mol = mol_clean.size(1)

    # ---- Multi-step DDIM reverse trajectory ----
    chain_T = g.T  # after compose, sub-samplers share the same T
    start_t = chain_T - 1
    sigma_start = float(g.noise[start_t])
    print(f"[eval] chain T={chain_T}, start_t={start_t}, sigma_start={sigma_start:.3f}",
          flush=True)

    torch.manual_seed(args.seed + 42)
    eps_init = torch.randn_like(mol_clean) * sigma_start
    x_t = mol_clean.clone()
    x_t[finite] = mol_clean[finite] + eps_init[finite]

    ts = np.linspace(start_t, 0, args.traj_steps).astype(int)
    traj_rows = []
    for i, t in enumerate(ts):
        sigma_t = float(g.noise[int(t)])
        batch_eval['net_input']['mol_holo_coord'] = x_t
        batch_eval['net_input']['mol_diffuse_time'] = torch.tensor(
            [[int(t)]], device=args.device, dtype=torch.long)
        batch_eval['net_input']['pocket_diffuse_time'] = torch.tensor(
            [[int(t)]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            out = run_forward(batch_eval)
        eps_pred = out['perturbation'][:, :n_mol, :]

        x0_est = x_t.clone()
        x0_est[finite] = x_t[finite] - sigma_t * eps_pred[finite]

        rmsd_xt = per_atom_rmsd(x_t.cpu().numpy()[0], truth, mask_np)
        rmsd_x0 = per_atom_rmsd(x0_est.cpu().numpy()[0], truth, mask_np)
        _, _, _, rmsd_xt_k = kabsch_align(x_t.cpu().numpy()[0], truth, mask_np)
        _, _, _, rmsd_x0_k = kabsch_align(x0_est.cpu().numpy()[0], truth, mask_np)

        traj_rows.append({
            "step": i, "t": int(t), "sigma_t": sigma_t,
            "rmsd_x_t": rmsd_xt, "rmsd_x0_est": rmsd_x0,
            "rmsd_x_t_kabsch": rmsd_xt_k, "rmsd_x0_est_kabsch": rmsd_x0_k,
            "x_t_np": x_t.cpu().numpy()[0].copy(),
            "x0_est_np": x0_est.cpu().numpy()[0].copy(),
        })

        if i % 10 == 0 or i == len(ts) - 1:
            print(f"  step {i:>3d}  t={int(t):>5d}  sigma={sigma_t:.3f}  "
                  f"RMSD x_t={rmsd_xt:.3f}  x0_est={rmsd_x0:.3f}  "
                  f"(Kabsch: {rmsd_xt_k:.3f} / {rmsd_x0_k:.3f})", flush=True)

        # DDIM Euler step
        if i + 1 < len(ts):
            t_next = int(ts[i + 1])
            sigma_next = float(g.noise[t_next])
            x_t = x_t.clone()
            x_t[finite] = x_t[finite] - (sigma_t - sigma_next) * eps_pred[finite]

    print(f"[traj] RMSD x_t  : start={traj_rows[0]['rmsd_x_t']:.3f} -> "
          f"end={traj_rows[-1]['rmsd_x_t']:.3f}", flush=True)
    print(f"[traj] RMSD Kabsch: start={traj_rows[0]['rmsd_x_t_kabsch']:.3f} -> "
          f"end={traj_rows[-1]['rmsd_x_t_kabsch']:.3f}", flush=True)
    print(f"[traj] best x0 RMSD:       {min(r['rmsd_x0_est'] for r in traj_rows):.3f}", flush=True)
    print(f"[traj] best x0 RMSD Kabsch: {min(r['rmsd_x0_est_kabsch'] for r in traj_rows):.3f}", flush=True)

    # ---- also single-step Tweedie via chain sampler (for cos measurement) ----
    eval_ds_diff = UnifiedDataset(
        lmdb_path=eval_lmdb_path,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_cfg,
        diffusion_samplers={"molecule": mole_sampler, "protein": prot_sampler},
    )
    torch.manual_seed(args.seed + 99)
    np.random.seed(args.seed + 99)
    item_diff = eval_ds_diff[0]
    batch_diff = to_device(UnifiedDataset.collate_fn([item_diff]), args.device)
    diff = batch_diff['diffused']
    norms = diff['mol_diffuse_perturb_norm'].view(-1)
    sigma_tw = float(norms[norms > 0][0].item())
    mol_noisy_tw = diff['mol_holo_coord']
    batch_diff['net_input']['mol_holo_coord'] = mol_noisy_tw
    batch_diff['net_input']['mol_holo_distance'] = diff['mol_holo_distance']
    batch_diff['net_input']['mol_diffuse_time'] = diff['mol_diffuse_time'].view(1, 1)
    batch_diff['net_input']['pocket_diffuse_time'] = diff['pocket_diffuse_time'].view(1, 1)
    with torch.no_grad():
        out_tw = run_forward(batch_diff)
    eps_pred_tw = out_tw['perturbation'][:, :n_mol, :]
    eps_true_tw = diff['mol_diffuse_perturb_score']
    valid = (diff['mol_diffuse_perturb_norm'] > 0)
    ep = eps_pred_tw[valid].reshape(-1).float()
    et = eps_true_tw[valid].reshape(-1).float()
    cosv = float((ep * et).sum().item() / (ep.norm() * et.norm() + 1e-9).item())

    x0_tw = mol_noisy_tw.clone()
    x0_tw[finite] = mol_noisy_tw[finite] - sigma_tw * eps_pred_tw[finite]
    noisy_tw_np = mol_noisy_tw.cpu().numpy()[0]
    x0_tw_np = x0_tw.cpu().numpy()[0]
    rmsd_n_tw = per_atom_rmsd(noisy_tw_np, truth, mask_np)
    rmsd_d_tw = per_atom_rmsd(x0_tw_np, truth, mask_np)
    _, _, _, rmsd_d_tw_k = kabsch_align(x0_tw_np, truth, mask_np)
    print(f"[tweedie] cos={cosv:+.3f}  noisy={rmsd_n_tw:.3f}A  "
          f"denoised={rmsd_d_tw:.3f}A  (Kabsch={rmsd_d_tw_k:.3f}A)", flush=True)

    # ---- 4. plots ----
    real_idx = np.where(mask_np)[0]
    truth_real = truth[real_idx]
    real_types = eval_rec["atom_types"]
    is_na = (real_types == 11)

    noisy_start_np = traj_rows[0]["x_t_np"]
    denoised_end_np = traj_rows[-1]["x_t_np"]
    noisy_aligned, _, _, rmsd_n_a = kabsch_align(noisy_start_np, truth, mask_np)
    denoised_aligned, _, _, rmsd_d_a = kabsch_align(denoised_end_np, truth, mask_np)

    rmsd_n_raw = traj_rows[0]["rmsd_x_t"]
    rmsd_d_raw = traj_rows[-1]["rmsd_x_t"]
    red_raw = 1.0 - rmsd_d_raw / max(rmsd_n_raw, 1e-9)
    red_k = 1.0 - rmsd_d_a / max(rmsd_n_a, 1e-9) if rmsd_n_a > 0 else 0

    pad_x = max(args.eval_width / 2 + 0.5, 1.5)
    pad_y = max(args.eval_height / 2 + 0.5, 1.5)

    # 5-panel triptych
    panels = [
        ("clean truth", truth_real, False),
        (f"noisy raw\nRMSD={rmsd_n_raw:.3f}A", noisy_start_np[real_idx], True),
        (f"noisy Kabsch\nRMSD={rmsd_n_a:.3f}A", noisy_aligned[real_idx], True),
        (f"denoised raw ({args.traj_steps}-step DDIM)\n"
         f"RMSD={rmsd_d_raw:.3f}A ({red_raw:+.1%})", denoised_end_np[real_idx], True),
        (f"denoised Kabsch\n"
         f"RMSD={rmsd_d_a:.3f}A ({red_k:+.1%})", denoised_aligned[real_idx], True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (title, coords, gh) in zip(axes, panels):
        draw_panel(ax, coords, is_na, truth_real, title, pad_x, pad_y,
                   draw_ghost=gh, marker_size=40)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"NaCl multi-size train -> eval {args.eval_width}×{args.eval_height} "
                 f"({args.traj_steps}-step DDIM, sigma_pert={sigma_pert_max})", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "triptych.png"), dpi=140)
    plt.close(fig)

    # Multi-step trajectory panel: steps 0, 10, 20, 30, 40, 49
    show_steps = [0, 10, 20, 30, 40, min(args.traj_steps - 1, 49)]
    show_steps = sorted(set(s for s in show_steps if s < len(traj_rows)))
    n_show = len(show_steps)
    fig2, axes2 = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes2 = [axes2]
    for ax, si in zip(axes2, show_steps):
        r = traj_rows[si]
        xt_aligned, _, _, rmsd_k = kabsch_align(r["x_t_np"], truth, mask_np)
        draw_panel(ax, xt_aligned[real_idx], is_na, truth_real,
                   f"step {si}  t={r['t']}\nsigma={r['sigma_t']:.3f}\n"
                   f"RMSD={r['rmsd_x_t']:.3f} (K={rmsd_k:.3f})",
                   pad_x, pad_y, draw_ghost=True, marker_size=30)
    fig2.suptitle(f"DDIM reverse trajectory (Kabsch-aligned) — "
                  f"eval {args.eval_width}×{args.eval_height}", fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "trajectory.png"), dpi=140)
    plt.close(fig2)

    # RMSD curve
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    steps_arr = [r["step"] for r in traj_rows]
    ax3.plot(steps_arr, [r["rmsd_x_t"] for r in traj_rows], "b-o",
             markersize=2, label="RMSD x_t (raw)")
    ax3.plot(steps_arr, [r["rmsd_x_t_kabsch"] for r in traj_rows], "b--s",
             markersize=2, label="RMSD x_t (Kabsch)")
    ax3.plot(steps_arr, [r["rmsd_x0_est"] for r in traj_rows], "r-o",
             markersize=2, label="RMSD x0_est (raw)")
    ax3.plot(steps_arr, [r["rmsd_x0_est_kabsch"] for r in traj_rows], "r--s",
             markersize=2, label="RMSD x0_est (Kabsch)")
    ax3.axhline(1.0, color="gray", linestyle=":", label="target RMSD<1")
    ax3.set_xlabel("DDIM step"); ax3.set_ylabel("RMSD (A)")
    ax3.set_title(f"DDIM reverse trajectory RMSD — {args.eval_width}×{args.eval_height}")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.out_dir, "rmsd_curve.png"), dpi=140)
    plt.close(fig3)

    # Report JSON
    report = {
        "n_train_records": n_rec,
        "train_steps": args.steps, "lr": args.lr,
        "sigma_pert_max": sigma_pert_max,
        "eval_grid": f"{args.eval_width}x{args.eval_height}",
        "n_atoms_eval": eval_rec["num_atoms"],
        "traj_steps": args.traj_steps,
        "traj_start_sigma": sigma_start,
        "traj_rmsd_start_raw": traj_rows[0]["rmsd_x_t"],
        "traj_rmsd_end_raw": traj_rows[-1]["rmsd_x_t"],
        "traj_rmsd_start_kabsch": traj_rows[0]["rmsd_x_t_kabsch"],
        "traj_rmsd_end_kabsch": traj_rows[-1]["rmsd_x_t_kabsch"],
        "traj_best_x0_rmsd": min(r["rmsd_x0_est"] for r in traj_rows),
        "traj_best_x0_rmsd_kabsch": min(r["rmsd_x0_est_kabsch"] for r in traj_rows),
        "tweedie_cos": cosv,
        "tweedie_rmsd_noisy": rmsd_n_tw,
        "tweedie_rmsd_denoised": rmsd_d_tw,
        "tweedie_rmsd_denoised_kabsch": rmsd_d_tw_k,
        "trajectory": [{k: v for k, v in r.items()
                         if k not in ("x_t_np", "x0_est_np")} for r in traj_rows],
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] artefacts in {args.out_dir}", flush=True)

    for png in ["triptych.png", "trajectory.png", "rmsd_curve.png"]:
        print(f"  {os.path.join(args.out_dir, png)}", flush=True)


if __name__ == "__main__":
    main()
