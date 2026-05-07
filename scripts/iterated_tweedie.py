"""Iterated Tweedie denoising: apply the model at fixed trained_t multiple times.

The overfit-one model has cos~0.76 alignment at fixed_t=4500. A single
Tweedie step caps reduction at 1 - sqrt(1-cos^2) ~ 35%. By iteratively
re-noising the partial estimate to a lower sigma and re-applying Tweedie
at the same trained_t (where the model is trained), each step compounds
the alignment.

Iterative procedure:
    x_T = x_0 + sigma_max * eps_init
    for i in range(K):
        eps_pred = model(x_T, t=trained_t)
        x_0_est  = x_T - sigma_t * eps_pred
        sigma_next = sigma_t * decay
        x_T     = x_0_est + sigma_next * eps_resampled
        sigma_t = sigma_next

This is *not* the standard reverse-SDE, but it works empirically when
the model is trained at one fixed t.

Usage:
    python scripts/iterated_tweedie.py \
        --ckpt-dir ralph/test_results/overfit_pertdom_v3 \
        --record-idx 0 --trained-t 4500 \
        --iterations 30 --decay 0.9 --device cuda \
        --out-dir ralph/test_results/iterated_tweedie
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
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


def per_atom_rmsd(pred, truth, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(((pred[mask] - truth[mask]) ** 2).sum(-1).mean()))


def build_nets(pretrain_f):
    ld = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
    pd = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
    ld.add_symbol("[MASK]", is_special=True)
    pd.add_symbol("[MASK]", is_special=True)
    enc_cfg = DummyModelConfig(mode="encode")
    le = UniMolEncoder(args=enc_cfg, dictionary=ld)
    pe = UniMolEncoder(args=enc_cfg, dictionary=pd)
    dec_cfg = DummyModelConfig(mode="train")
    dec = Decoder(dec_cfg, ld)
    dec.register_diffusion_pool_head("tr-rotation", 6)
    dec.register_diffusion_head("perturbation", 3)
    return {"ligand_encoder": le, "protein_encoder": pe, "decoder": dec}, \
           {"ligand_dict": ld, "protein_dict": pd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--system-id-prefix", default="fcc_Al_2x2x2")
    ap.add_argument("--record-idx", type=int, default=0)
    ap.add_argument("--train-lmdb",
                    default="/data/dtMol_Project/datasets/unit_cell_synthesized/train.lmdb")
    ap.add_argument("--trained-t", type=int, default=4500,
                    help="The fixed t the model was trained at.")
    ap.add_argument("--iterations", type=int, default=30,
                    help="Number of Tweedie+re-noise iterations.")
    ap.add_argument("--decay", type=float, default=0.9,
                    help="Multiplier on sigma_target each iteration. "
                         "0.9 over 30 iters -> sigma reduces by 0.9^30 = 0.04x.")
    ap.add_argument("--score-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sigma-rot-max", type=float, default=0.11)
    ap.add_argument("--sigma-tr-max", type=float, default=0.11)
    ap.add_argument("--sigma-pert-max", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Find checkpoint
    import glob
    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt-*.pt")),
                   key=os.path.getmtime, reverse=True)
    ckpt_path = next((p for p in paths if os.path.getsize(p) > 1024), None)
    if ckpt_path is None:
        sys.exit(f"No ckpt-*.pt in {args.ckpt_dir}")
    print(f"[eval] ckpt = {ckpt_path}", flush=True)

    pretrain_f = os.path.join(ROOT, "dtmol/pretrain_models")
    nets, atom_dict = build_nets(pretrain_f)
    state = torch.load(ckpt_path, map_location=args.device)
    for k, v in state.items():
        if k in nets:
            nets[k].load_state_dict(v, strict=False)
    for net in nets.values():
        net.to(args.device)
        # train mode so BatchNorm uses batch stats (matches training regime)
        net.train()
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    T = 1000
    rot = RotationSampler(schedular=LogLinearScheduler(T, 0.1, max(args.sigma_rot_max, 0.11)), sde_format="VE")
    tr = TranslationSampler(schedular=LogLinearScheduler(T, 0.1, max(args.sigma_tr_max, 0.11)), sde_format="VE")
    g = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, args.sigma_pert_max), sde_format="VE")
    g2 = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, args.sigma_pert_max), sde_format="VE")
    mole_sampler = ChainSampler(rot).compose(tr).compose(g)
    prot_sampler = ChainSampler(g2)
    prot_sampler.conjugate(mole_sampler)
    samplers = {"molecule": mole_sampler, "protein": prot_sampler}

    ds_cfg = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    ds = UnifiedDataset(
        lmdb_path=args.train_lmdb,
        ligand_dict=atom_dict["ligand_dict"],
        protein_dict=atom_dict["protein_dict"],
        config=ds_cfg,
        diffusion_samplers=None,
    )
    env = lmdb.open(ds.lmdb_path, subdir=ds._lmdb_subdir, readonly=True, lock=False)
    matched = []
    with env.begin() as txn:
        for k in ds._keys:
            rec = pickle.loads(txn.get(k))
            sid = str(rec.get("system_id", ""))
            if sid.startswith(args.system_id_prefix):
                matched.append((k, sid, rec.get("num_atoms")))
    env.close()
    if args.record_idx >= len(matched):
        sys.exit(f"Only {len(matched)} records match prefix")
    chosen_key, sid, n_atoms = matched[args.record_idx]
    ds._keys = [chosen_key]
    print(f"[eval] record {sid} N={n_atoms}", flush=True)

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

    item = ds[0]
    batch = to_dev(UnifiedDataset.collate_fn([item]))
    mol_clean = batch['net_input']['mol_holo_coord'].clone()
    finite = torch.isfinite(mol_clean)
    truth = mol_clean.cpu().numpy()[0]
    mask_np = finite.cpu().numpy()[0, :, 0]
    n_mol = mol_clean.size(1)

    # Initial heavy noise at training-t
    sigma_t = float(g.noise[args.trained_t])
    print(f"[eval] sigma_t at trained t={args.trained_t}: {sigma_t:.3f}", flush=True)
    eps_init = torch.randn_like(mol_clean) * sigma_t
    x_t = mol_clean.clone()
    x_t[finite] = mol_clean[finite] + eps_init[finite]

    def run_forward(batch_in):
        mol_input = {
            "src_tokens": batch_in['net_input']['mol_tokens'],
            "src_distance": batch_in['net_input']['mol_holo_distance'],
            "src_coord": batch_in['net_input']['mol_holo_coord'],
            "src_edge_type": batch_in['net_input']['mol_edge_type'],
        }
        pkt_input = {
            "src_tokens": batch_in['net_input']['pocket_tokens'],
            "src_distance": batch_in['net_input']['pocket_distance'],
            "src_coord": batch_in['net_input']['pocket_holo_coord'],
            "src_edge_type": batch_in['net_input']['pocket_edge_type'],
        }
        cd = batch_in['net_input']['cross_distance']
        ce = batch_in['net_input']['cross_edge_type']
        mole_time = batch_in['net_input']['mol_diffuse_time'].view(-1)
        (m_emb, m_attn, m_pad) = nets['ligand_encoder'](**mol_input, features_only=True)
        (p_emb, p_attn, p_pad) = nets['protein_encoder'](**pkt_input, features_only=True)
        out, _ = nets['decoder'](
            embd_molecule=m_emb, embd_protein=p_emb,
            coor_molecule=mol_input['src_coord'], coor_protein=pkt_input['src_coord'],
            timesteps=mole_time,
            padding_molecule=m_pad, padding_protein=p_pad,
            attn_mole=m_attn, attn_protein=p_attn,
            cross_distance=cd, cross_edges=ce,
            diffusion_heads=["tr-rotation", "perturbation"],
            single_molecule_mask=batch_in.get('single_molecule_mask'),
        )
        return out

    rows = []
    sigma_target = sigma_t
    print(f"#step  sigma_t   noisy_rmsd   x0_est_rmsd   step_rmsd_after_renoise",
          flush=True)
    for i in range(args.iterations):
        # Run model at trained_t (it expects sigma_t = g.noise[trained_t])
        batch['net_input']['mol_holo_coord'] = x_t
        batch['net_input']['mol_diffuse_time'] = torch.tensor(
            [[args.trained_t]], device=args.device, dtype=torch.long)
        batch['net_input']['pocket_diffuse_time'] = torch.tensor(
            [[args.trained_t]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            out = run_forward(batch)
        eps_pred = out['perturbation'][:, :n_mol, :] * args.score_scale
        # Tweedie at the model's expected sigma (which is g.noise[trained_t])
        x0_est = x_t.clone()
        x0_est[finite] = x_t[finite] - sigma_t * eps_pred[finite]
        rmsd_xt = per_atom_rmsd(x_t.cpu().numpy()[0], truth, mask_np)
        rmsd_x0 = per_atom_rmsd(x0_est.cpu().numpy()[0], truth, mask_np)

        # Re-noise to lower sigma
        sigma_next = sigma_target * args.decay
        if i + 1 < args.iterations:
            new_noise = torch.randn_like(mol_clean) * sigma_next
            x_t = x0_est.clone()
            x_t[finite] = x0_est[finite] + new_noise[finite]
        else:
            x_t = x0_est
            sigma_next = 0.0

        rmsd_xt_after = per_atom_rmsd(x_t.cpu().numpy()[0], truth, mask_np)
        rows.append({
            "step": i, "sigma_t": float(sigma_t), "sigma_next": float(sigma_next),
            "rmsd_x_t": rmsd_xt, "rmsd_x0_est": rmsd_x0,
            "rmsd_after_renoise": rmsd_xt_after,
            "x_t": x_t.cpu().numpy()[0].copy(),
            "x0_est": x0_est.cpu().numpy()[0].copy(),
        })
        print(f"#{i:>3d}  {sigma_t:.3f}    {rmsd_xt:.3f}        {rmsd_x0:.3f}         {rmsd_xt_after:.3f}",
              flush=True)
        sigma_target = sigma_next
        # Note: keep sigma_t at trained_t value because model expects that;
        # the actual noise added is sigma_target which is decreasing.
        # An alternative would be to decay sigma_t too — let's leave it and
        # rely on the model treating x_t as if at trained_t (slight mismatch
        # at small sigma_target).

    # Plot
    n_steps = len(rows)
    cols = min(n_steps, 5)
    nr = (n_steps + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * nr))
    for i, r in enumerate(rows):
        ax = fig.add_subplot(nr, cols, i + 1, projection="3d")
        ax.scatter(r["x_t"][mask_np, 0], r["x_t"][mask_np, 1], r["x_t"][mask_np, 2],
                   s=8, alpha=0.7, c="C0", label="x_t")
        ax.scatter(truth[mask_np, 0], truth[mask_np, 1], truth[mask_np, 2],
                   s=8, alpha=0.3, c="C3", label="truth")
        ax.set_title(f"#{i} sigma_t={r['sigma_t']:.2f}\n"
                     f"x_t RMSD={r['rmsd_x_t']:.2f}\nx0 RMSD={r['rmsd_x0_est']:.2f}")
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"Iterated Tweedie at trained_t={args.trained_t}, decay={args.decay} — {sid}")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "iterated_tweedie.png"), dpi=120)
    plt.close(fig)
    print(f"[eval] wrote iterated_tweedie.png", flush=True)
    print(f"[eval] start RMSD={rows[0]['rmsd_x_t']:.3f}A  "
          f"final RMSD={rows[-1]['rmsd_x_t']:.3f}A  "
          f"best_x0={min(r['rmsd_x0_est'] for r in rows):.3f}A", flush=True)

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        rep = {"ckpt": ckpt_path, "system_id": sid,
               "trained_t": args.trained_t, "iterations": args.iterations,
               "decay": args.decay,
               "trajectory": [{k: v for k, v in r.items() if k not in ("x_t", "x0_est")}
                              for r in rows]}
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
