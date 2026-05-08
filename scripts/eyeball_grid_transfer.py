"""Layout-transfer eyeball test: load a checkpoint trained on grid A, eval on grid B.

If the denoiser learned LOCAL atom-pair rules (e.g. "Na-Cl distance should be
1 A, Na-Na should be sqrt(2) A"), then a model trained on a 6x3 grid should
also denoise a 9x9 grid of the same checkerboard pattern. If instead it
just memorised the 6x3 layout, the 9x9 eval will collapse.

Usage:
    python scripts/eyeball_grid_transfer.py \\
        --ckpt-dir ralph/test_results/eyeball_nacl_6x3_s03 \\
        --eval-width 9 --eval-height 9 \\
        --out-dir ralph/test_results/eyeball_transfer_6x3_to_9x9
"""
from __future__ import annotations
import argparse
import glob
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

from dtmol.dtmol_model import DummyModelConfig
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.utils.dictionary import Dictionary
from dtmol.data.unified_dataset import UnifiedDataset, UnifiedDatasetConfig
from dtmol.diffusion import (
    RotationSampler, GaussianSampler, TranslationSampler,
    ChainSampler, LogLinearScheduler,
)

# Re-use the helpers from eyeball_nacl_grid.py.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eyeball_nacl_grid import (
    build_grid_record, write_lmdb,
    per_atom_rmsd, kabsch_align,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="Path to checkpoint dir (uses latest ckpt-*.pt with size > 1KB)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-width", type=int, default=9)
    ap.add_argument("--eval-height", type=int, default=9)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--fixed-t", type=int, default=4500)
    ap.add_argument("--sigma-rot-max", type=float, default=0.11)
    ap.add_argument("--sigma-tr-max", type=float, default=0.11)
    ap.add_argument("--sigma-pert-max", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Find checkpoint
    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt-*.pt")),
                   key=os.path.getmtime, reverse=True)
    ckpt_path = next((p for p in paths if os.path.getsize(p) > 1024), None)
    if ckpt_path is None:
        sys.exit(f"No ckpt-*.pt in {args.ckpt_dir}")
    print(f"[eval] ckpt = {ckpt_path}", flush=True)

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
    state = torch.load(ckpt_path, map_location=args.device)
    for k, v in state.items():
        if k in nets:
            nets[k].load_state_dict(v, strict=False)
    for net in nets.values():
        net.to(args.device)
        net.train()
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    T = 1000
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
    def _fixed_sample_time(size):
        return np.full(size if isinstance(size, int) else size, cached_t, dtype=int)
    for s in mole_sampler.samplers:
        s.sample_time = _fixed_sample_time

    # Build the EVAL grid (different size than training)
    record = build_grid_record(args.eval_width, args.eval_height, args.spacing)
    print(f"[grid] eval {args.eval_width}x{args.eval_height} = {record['num_atoms']} atoms",
          flush=True)
    lmdb_path = os.path.join(args.out_dir, "grid_lmdb")
    write_lmdb(lmdb_path, record)

    ds_cfg = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    ds = UnifiedDataset(
        lmdb_path=lmdb_path,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_cfg,
        diffusion_samplers={"molecule": mole_sampler, "protein": prot_sampler},
    )

    def to_dev(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(args.device)
            elif isinstance(v, dict):
                out[k] = to_dev(v)
            else:
                out[k] = v
        return out

    def run_forward(b):
        mol = {"src_tokens": b['net_input']['mol_tokens'],
               "src_distance": b['net_input']['mol_holo_distance'],
               "src_coord": b['net_input']['mol_holo_coord'],
               "src_edge_type": b['net_input']['mol_edge_type']}
        pkt = {"src_tokens": b['net_input']['pocket_tokens'],
               "src_distance": b['net_input']['pocket_distance'],
               "src_coord": b['net_input']['pocket_holo_coord'],
               "src_edge_type": b['net_input']['pocket_edge_type']}
        cd = b['net_input']['cross_distance']
        ce = b['net_input']['cross_edge_type']
        mt = b['net_input']['mol_diffuse_time'].view(-1)
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
            single_molecule_mask=b.get('single_molecule_mask'),
        )
        return out

    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    item = ds[0]
    batch = to_dev(UnifiedDataset.collate_fn([item]))
    diff = batch['diffused']
    norms = diff['mol_diffuse_perturb_norm'].view(-1)
    sigma_t = float(norms[norms > 0][0].item())
    print(f"[eval] sigma_t at t={cached_t} = {sigma_t:.3f}", flush=True)

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
        "transfer": True,
        "ckpt": ckpt_path,
        "eval_grid": f"{args.eval_width}x{args.eval_height}",
        "n_atoms_eval": int(record["num_atoms"]),
        "fixed_t": cached_t, "sigma_t": sigma_t,
        "sigma_pert_max": args.sigma_pert_max,
        "rmsd_noisy": rmsd_n, "rmsd_denoised": rmsd_d, "rmsd_reduction": red,
        "rmsd_noisy_kabsch": rmsd_n_a, "rmsd_denoised_kabsch": rmsd_d_a,
        "rmsd_reduction_kabsch": red_a,
        "cos": cosv,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # plot
    real_idx = np.where(mask_np)[0]
    truth_real = truth[real_idx]
    noisy_real = noisy_np[real_idx]
    denoised_real = x0_np[real_idx]
    noisy_aligned_real = noisy_aligned[real_idx]
    denoised_aligned_real = denoised_aligned[real_idx]
    real_types = record["atom_types"]
    is_na = (real_types == 11)

    pad_x = max(args.eval_width / 2 + 0.5, 1.5)
    pad_y = max(args.eval_height / 2 + 0.5, 1.5)

    def draw(ax, coords, title, draw_ghost=True):
        ax.scatter(coords[is_na, 0], coords[is_na, 1], s=80, c="C0",
                   marker="o", edgecolor="k", linewidth=0.8, label="Na (Z=11)")
        ax.scatter(coords[~is_na, 0], coords[~is_na, 1], s=80, c="C3",
                   marker="s", edgecolor="k", linewidth=0.8, label="Cl (Z=17)")
        if draw_ghost:
            ax.scatter(truth_real[is_na, 0], truth_real[is_na, 1],
                       s=80, c="none", marker="o",
                       edgecolor="gray", linestyle="--", linewidth=0.6)
            ax.scatter(truth_real[~is_na, 0], truth_real[~is_na, 1],
                       s=80, c="none", marker="s",
                       edgecolor="gray", linestyle="--", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-pad_x, pad_x); ax.set_ylim(-pad_y, pad_y)
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

    panels = [
        ("clean truth", truth_real, False),
        (f"noisy raw\nRMSD={rmsd_n:.3f}A", noisy_real, True),
        (f"noisy Kabsch-aligned\nRMSD={rmsd_n_a:.3f}A", noisy_aligned_real, True),
        (f"denoised raw\nRMSD={rmsd_d:.3f}A (-{red*100:.1f}%)", denoised_real, True),
        (f"denoised Kabsch-aligned\nRMSD={rmsd_d_a:.3f}A (-{red_a*100:.1f}%)", denoised_aligned_real, True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (title, coords, gh) in zip(axes, panels):
        draw(ax, coords, title, draw_ghost=gh)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Layout transfer: ckpt {os.path.basename(args.ckpt_dir)} -> "
                 f"eval {args.eval_width}×{args.eval_height} (sigma_pert={args.sigma_pert_max})",
                 fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "transfer_denoise.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)

    # Iterated trajectory at sigma_train
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
        torch.manual_seed(args.seed + 100 + k)
        fresh = torch.randn_like(mol_clean) * sigma_t
        x_t_iter = x0_k.clone()
        x_t_iter[finite] = x0_k[finite] + fresh[finite]

    fig2, axes2 = plt.subplots(2, n_iter, figsize=(3.5 * n_iter, 7))
    for k, it in enumerate(iterates):
        ax = axes2[0, k]
        draw(ax, it["x0_est"], f"iter {k}\nRMSD={it['rmsd']:.3f}A", draw_ghost=True)
        if k == 0:
            ax.set_ylabel("raw x0_est", fontsize=11)
        ax = axes2[1, k]
        draw(ax, it["x0_est_aligned"],
             f"iter {k} (aligned)\nRMSD={it['rmsd_kabsch']:.3f}A", draw_ghost=True)
        if k == 0:
            ax.set_ylabel("Kabsch-aligned", fontsize=11)
    fig2.suptitle(f"Iterated Tweedie ({n_iter} fresh-eps trials) — transfer "
                  f"{args.eval_width}×{args.eval_height}", fontsize=13)
    plt.tight_layout()
    out_png2 = os.path.join(args.out_dir, "transfer_iterated.png")
    fig2.savefig(out_png2, dpi=140)
    plt.close(fig2)
    print(f"[plot] wrote {out_png2}", flush=True)
    report["iterated_rmsd"] = [it["rmsd"] for it in iterates]
    report["iterated_rmsd_kabsch"] = [it["rmsd_kabsch"] for it in iterates]
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
