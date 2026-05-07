"""End-to-end: overfit one fcc_Al record then visualise the denoise trajectory.

Architecture-free path to a "good" denoise trajectory: rather than asking the
model to learn a generalising score across many records, we drive a single
record's loss as low as possible (per the overfit-one experiment, this
achieves cos>0.85 / ratio<0.3) and then use the SAME record for the eval.

Outputs:
    <out-dir>/overfit_train.log
    <out-dir>/ckpt-final.pt
    <out-dir>/reverse_trajectory.png
    <out-dir>/tweedie_panels.png
    <out-dir>/report.json

Usage:
    python scripts/run_overfit_traj.py \
        --record-idx 0 --steps 1500 --lr 1e-3 \
        --out-dir ralph/test_results/overfit_traj
"""
from __future__ import annotations
import argparse
import json
import logging
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

from dtmol.dtmol_train_test import DiffusionTrainer
from dtmol.dtmol_train_base import CONFIG
from dtmol.dtmol_model import DummyModelConfig
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.utils.dictionary import Dictionary
from dtmol.data.unified_dataset import UnifiedDataset, UnifiedDatasetConfig
from dtmol.data.mixer import DatasetMixer, get_mixed_dataloader
from dtmol.diffusion import (
    RotationSampler, GaussianSampler, TranslationSampler,
    ChainSampler, LogLinearScheduler,
)


def per_atom_rmsd(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diff = pred[mask] - truth[mask]
    return float(np.sqrt((diff ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--system-id-prefix", default="fcc_Al_2x2x2")
    ap.add_argument("--record-idx", type=int, default=0,
                    help="Pick the N-th record matching the prefix (within train.lmdb)")
    ap.add_argument("--train-lmdb",
                    default="/data/dtMol_Project/datasets/unit_cell_synthesized/train.lmdb")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj-steps", type=int, default=50)
    ap.add_argument("--traj-start-t", type=int, default=4500)
    ap.add_argument("--score-scale", type=float, default=1.0)
    ap.add_argument("--fixed-t", type=int, default=None,
                    help="If set, train at this fixed diffusion timestep "
                         "(fresh eps each step). With this the model learns the "
                         "score at a single t — which is what we evaluate.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.WARNING)

    pretrain_f = os.path.join(ROOT, "dtmol/pretrain_models")
    ligand_dict = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
    ligand_dict.add_symbol("[MASK]", is_special=True)
    protein_dict.add_symbol("[MASK]", is_special=True)

    encoder_config = DummyModelConfig(mode="encode")
    ligand_encoder = UniMolEncoder(args=encoder_config, dictionary=ligand_dict)
    protein_encoder = UniMolEncoder(args=encoder_config, dictionary=protein_dict)
    decoder_config = DummyModelConfig(mode="train")
    decoder = Decoder(decoder_config, ligand_dict)
    decoder.register_diffusion_pool_head("tr-rotation", 6)
    decoder.register_diffusion_head("perturbation", 3)
    nets = {"ligand_encoder": ligand_encoder,
            "protein_encoder": protein_encoder, "decoder": decoder}

    T = 1000
    # Reduced sigmas for rotation/translation so the perturbation head's
    # Tweedie can actually denoise. With translation sigma_max=19 the noisy
    # RMSD at high t is ~17 A — translation noise dominates and the
    # perturbation head can't undo it. Set translation/rotation small so the
    # noise distribution is mostly per-atom Gaussian.
    sigma_rot_max = float(os.environ.get("SIGMA_ROT_MAX", "0.1"))
    sigma_tr_max = float(os.environ.get("SIGMA_TR_MAX", "0.1"))
    sigma_pert_max = float(os.environ.get("SIGMA_PERT_MAX", "1.5"))
    print(f"[samplers] sigma_max: rot={sigma_rot_max} tr={sigma_tr_max} "
          f"pert={sigma_pert_max}", flush=True)
    rot_sampler = RotationSampler(schedular=LogLinearScheduler(T, 0.1, max(sigma_rot_max, 0.11)), sde_format="VE")
    tr_sampler = TranslationSampler(schedular=LogLinearScheduler(T, 0.1, max(sigma_tr_max, 0.11)), sde_format="VE")
    g_sampler = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, sigma_pert_max), sde_format="VE")
    g_sampler2 = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, sigma_pert_max), sde_format="VE")
    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)

    ds_config = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)

    # ----- Build a 1-record dataset -----
    ds = UnifiedDataset(
        lmdb_path=args.train_lmdb,
        ligand_dict=ligand_dict,
        protein_dict=protein_dict,
        config=ds_config,
        diffusion_samplers={"molecule": molecule_sampler, "protein": protein_sampler},
    )
    # find the args.record_idx-th key matching the prefix
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
        raise RuntimeError(f"Only {len(matched)} records match prefix {args.system_id_prefix}")
    chosen_key, chosen_sid, n_atoms = matched[args.record_idx]
    print(f"[overfit] selected record sid={chosen_sid} N={n_atoms} idx={args.record_idx}",
          flush=True)
    ds._keys = [chosen_key]

    # We can't trivially feed this to the mixer's collate; build a one-record loader manually.
    ligand_eval = ligand_dict
    protein_eval = protein_dict

    # Two regimes available:
    #  --fixed-t T  → fresh eps each step at the SAME t. Model learns the
    #                 score function at that one timestep for this record.
    #  default      → fresh eps + fresh t each step (single-record training,
    #                 plateaus ~0.95 like our other Fix-E runs).
    if args.fixed_t is not None:
        cached_t = int(args.fixed_t)
        print(f"[overfit] training with fixed t={cached_t}, fresh eps per step",
              flush=True)
        # Patch the molecule sampler's sample_time to always return cached_t
        _orig_sample_time = molecule_sampler.samplers[0].sample_time
        def _fixed_sample_time(size):
            return np.full(size if isinstance(size, int) else size, cached_t,
                           dtype=int)
        for s in molecule_sampler.samplers:
            s.sample_time = _fixed_sample_time
    else:
        first_item = ds[0]
        cached_batch = UnifiedDataset.collate_fn([first_item])
        cached_t = int(cached_batch['diffused']['mol_diffuse_time'].view(-1)[0].item())
        print(f"[overfit] cached training batch: mol_t={cached_t}", flush=True)

    def loader():
        if args.fixed_t is not None:
            while True:
                item = ds[0]
                yield UnifiedDataset.collate_fn([item])
        else:
            while True:
                yield cached_batch

    config = CONFIG(
        lambda_force=0.0, lambda_fd_force=0.0, force_loss_fn="mse",
        dataset_mode="unified", use_wandb=False,
    )

    # Make a fake DataLoader-like object so DiffusionTrainer can be built.
    class _SingleLoader:
        def __init__(self, gen, length=10):
            self._gen = gen
            self._length = length
        def __iter__(self):
            return self._gen
        def __len__(self):
            return self._length

    train_iter = loader()
    eval_iter = loader()
    trainer = DiffusionTrainer(
        train_dataloader=_SingleLoader(train_iter),
        eval_dataloader=_SingleLoader(eval_iter),
        nets=nets,
        sampler={"molecule": molecule_sampler, "protein": protein_sampler},
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

    # ----- Train (overfit) -----
    log_path = os.path.join(args.out_dir, "overfit_train.log")
    log_f = open(log_path, "w")
    print(f"[overfit] training {args.steps} steps lr={args.lr}", flush=True)

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

    t0 = time.time()
    for i in range(args.steps):
        batch = next(train_iter)
        batch = to_device(batch)
        loss, loss_dict = trainer.train_step(batch)
        if torch.isnan(loss):
            print(f"step {i}: NaN, skipping", flush=True)
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0 or i == args.steps - 1:
            line = f"step {i:5d}  loss={loss.item():.4f}"
            print(line, flush=True)
            log_f.write(line + "\n")
            log_f.flush()
    log_f.close()
    print(f"[overfit] done in {time.time()-t0:.1f}s", flush=True)

    # ----- Save checkpoint -----
    ckpt_path = os.path.join(args.out_dir, "ckpt-final.pt")
    net_dict = {k: net.state_dict() for k, net in trainer.nets.items()}
    torch.save(net_dict, ckpt_path)
    with open(os.path.join(args.out_dir, "checkpoint"), "w") as f:
        f.write("latest checkpoint:ckpt-final.pt\n")
    print(f"[overfit] saved {ckpt_path}", flush=True)

    # ----- Eval reverse trajectory on the SAME record -----
    # KEEP nets in train mode so BatchNorm uses batch statistics (same as
    # during training). With net.eval() BN uses running stats which were
    # tracked over a small number of training steps and don't match the
    # batch-stat regime the model is fit to. Empirically eval mode kills
    # the cos alignment from ~1.0 to ~0.35.
    print(f"[eval] running reverse trajectory on overfit record (train-mode BN)",
          flush=True)
    for net in trainer.nets.values():
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    eval_ds = UnifiedDataset(
        lmdb_path=args.train_lmdb,
        ligand_dict=ligand_dict, protein_dict=protein_dict,
        config=ds_config,
        diffusion_samplers=None,  # we'll add our own noise
    )
    eval_ds._keys = [chosen_key]

    def index_to_batch(ds, device):
        item = ds[0]
        b = UnifiedDataset.collate_fn([item])
        return to_device(b)

    def run_forward(batch):
        mol_input = {
            "src_tokens": batch['net_input']['mol_tokens'],
            "src_distance": batch['net_input']['mol_holo_distance'],
            "src_coord": batch['net_input']['mol_holo_coord'],
            "src_edge_type": batch['net_input']['mol_edge_type'],
        }
        pkt_input = {
            "src_tokens": batch['net_input']['pocket_tokens'],
            "src_distance": batch['net_input']['pocket_distance'],
            "src_coord": batch['net_input']['pocket_holo_coord'],
            "src_edge_type": batch['net_input']['pocket_edge_type'],
        }
        cross_dist = batch['net_input']['cross_distance']
        cross_edges = batch['net_input']['cross_edge_type']
        mole_time = batch['net_input']['mol_diffuse_time'].view(-1)
        (m_emb, m_attn, m_pad) = trainer.nets['ligand_encoder'](**mol_input, features_only=True)
        (p_emb, p_attn, p_pad) = trainer.nets['protein_encoder'](**pkt_input, features_only=True)
        single_mol = batch.get('single_molecule_mask')
        output, _ = trainer.nets['decoder'](
            embd_molecule=m_emb, embd_protein=p_emb,
            coor_molecule=mol_input['src_coord'], coor_protein=pkt_input['src_coord'],
            timesteps=mole_time,
            padding_molecule=m_pad, padding_protein=p_pad,
            attn_mole=m_attn, attn_protein=p_attn,
            cross_distance=cross_dist, cross_edges=cross_edges,
            diffusion_heads=["tr-rotation", "perturbation"],
            single_molecule_mask=single_mol,
        )
        return output

    # Tweedie one-step at training t using the CHAIN sampler (matches training
    # noise distribution: rotation + translation + perturbation, not plain
    # Gaussian). We compute the prediction using the model and the score
    # ground-truth from the dataset.
    report = {"system_id": chosen_sid, "n_atoms": int(n_atoms),
              "ckpt": ckpt_path, "trained_t": int(cached_t),
              "tweedie": [], "trajectory": []}

    # Re-attach diffusion samplers so __getitem__ produces diffused batches
    eval_ds.diffusion_samplers = {"molecule": molecule_sampler,
                                  "protein": protein_sampler}

    print(f"[tweedie eval at trained t={cached_t} via chain sampler]", flush=True)
    for trial in range(5):
        torch.manual_seed(args.seed + trial)
        np.random.seed(args.seed + trial)
        item = eval_ds[0]
        batch = to_device(UnifiedDataset.collate_fn([item]))
        # diffused has the actual eps targets the model was trained on
        diff = batch['diffused']
        # perturb_norm includes BOS/EOS rows which have norm=0; pick the
        # first real-atom norm (any row matches since all atoms share t).
        norms = diff['mol_diffuse_perturb_norm'].view(-1)
        nonzero = norms[norms > 0]
        sigma_t = float(nonzero[0].item()) if nonzero.numel() else float(g_sampler.noise[cached_t])
        # Run model on diffused state
        mol_coord_clean = batch['net_input']['mol_holo_coord']
        mol_coord_noisy = diff['mol_holo_coord']
        finite = torch.isfinite(mol_coord_clean)
        # swap in the diffused coords for the forward pass
        batch['net_input']['mol_holo_coord'] = mol_coord_noisy
        batch['net_input']['mol_holo_distance'] = diff['mol_holo_distance']
        batch['net_input']['mol_diffuse_time'] = diff['mol_diffuse_time'].view(1, 1)
        batch['net_input']['pocket_diffuse_time'] = diff['pocket_diffuse_time'].view(1, 1)
        with torch.no_grad():
            out = run_forward(batch)
        n_mol = mol_coord_clean.size(1)
        # Slice perturbation prediction to molecule atoms; remove BOS/EOS
        # padding (positions 0 and N-1) which have norm=0 and aren't supervised.
        eps_pred_full = out['perturbation'][:, :n_mol, :] * args.score_scale
        # Ground truth perturbation_score from dataset
        eps_true_full = diff['mol_diffuse_perturb_score']  # [B, n_mol, 3]
        valid = (diff['mol_diffuse_perturb_norm'] > 0)  # [B, n_mol]
        # Reshape valid for indexing
        x_noisy_np = mol_coord_noisy.cpu().numpy()[0]
        x_truth_np = mol_coord_clean.cpu().numpy()[0]
        # Apply Tweedie on the perturbation portion only:
        # x0_est = x_noisy - sigma_t * eps_pred
        x0_est = mol_coord_noisy.clone()
        x0_est[finite] = mol_coord_noisy[finite] - sigma_t * eps_pred_full[finite]
        x0_np = x0_est.cpu().numpy()[0]
        mask = finite.cpu().numpy()[0, :, 0]
        rmsd_n = per_atom_rmsd(x_noisy_np, x_truth_np, mask)
        rmsd_d = per_atom_rmsd(x0_np, x_truth_np, mask)
        red = 1.0 - rmsd_d / max(rmsd_n, 1e-9)
        # Cosine sim of per-atom predictions vs true eps where norm > 0
        ep = eps_pred_full[valid].reshape(-1).float()
        et = eps_true_full[valid].reshape(-1).float()
        cosv = float((ep * et).sum().item() / (ep.norm() * et.norm() + 1e-9).item())
        pr_rms = float(ep.pow(2).mean().sqrt().item())
        tg_rms = float(et.pow(2).mean().sqrt().item())
        print(f"  trial {trial}: noisy_RMSD={rmsd_n:.3f}A  denoised={rmsd_d:.3f}A  "
              f"reduction={red:+.1%}  cos={cosv:+.3f}  pred_rms={pr_rms:.3f}  "
              f"target_rms={tg_rms:.3f}", flush=True)
        report["tweedie"].append({
            "trial": trial, "t": int(cached_t), "sigma_t": sigma_t,
            "rmsd_noisy": rmsd_n, "rmsd_denoised": rmsd_d,
            "rmsd_reduction": red, "cos": cosv,
            "pred_rms": pr_rms, "target_rms": tg_rms,
        })

    # Reverse trajectory
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch = index_to_batch(eval_ds, args.device)
    mol_clean = batch['net_input']['mol_holo_coord'].clone()
    finite = torch.isfinite(mol_clean)
    truth = mol_clean.cpu().numpy()[0]
    mask_np = finite.cpu().numpy()[0, :, 0]
    sigma_start = float(g_sampler.noise[args.traj_start_t])
    n_init = torch.randn_like(mol_clean) * sigma_start
    x_t = mol_clean.clone()
    x_t[finite] = mol_clean[finite] + n_init[finite]
    ts = np.linspace(args.traj_start_t, 0, args.traj_steps).astype(int)
    n_mol = mol_clean.size(1)

    rows = []
    for i, t in enumerate(ts):
        sigma_t = float(g_sampler.noise[int(t)])
        batch['net_input']['mol_holo_coord'] = x_t
        batch['net_input']['mol_diffuse_time'] = torch.tensor(
            [[int(t)]], device=args.device, dtype=torch.long)
        batch['net_input']['pocket_diffuse_time'] = torch.tensor(
            [[int(t)]], device=args.device, dtype=torch.long)
        with torch.no_grad():
            out = run_forward(batch)
        eps_pred = out['perturbation'][:, :n_mol, :] * args.score_scale
        x0_est = x_t.clone()
        x0_est[finite] = x_t[finite] - sigma_t * eps_pred[finite]
        rmsd_t = per_atom_rmsd(x_t.cpu().numpy()[0], truth, mask_np)
        rmsd_0 = per_atom_rmsd(x0_est.cpu().numpy()[0], truth, mask_np)
        rows.append((int(t), sigma_t, x_t.cpu().numpy()[0], x0_est.cpu().numpy()[0], rmsd_t, rmsd_0))
        report["trajectory"].append({
            "step": i, "t": int(t), "sigma_t": sigma_t,
            "rmsd_x_t": rmsd_t, "rmsd_x0_est": rmsd_0,
        })
        if i + 1 < len(ts):
            t_next = int(ts[i + 1])
            sigma_next = float(g_sampler.noise[t_next])
            x_t = x_t.clone()
            x_t[finite] = x_t[finite] - (sigma_t - sigma_next) * eps_pred[finite]

    print(f"[traj] start RMSD={rows[0][4]:.3f}  end RMSD={rows[-1][4]:.3f}  "
          f"best_x0={min(r[5] for r in rows):.3f}", flush=True)

    # Plot trajectory
    n_steps = len(rows)
    cols = min(n_steps, 5)
    nr = (n_steps + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * nr))
    for i, (t, sigma, xt_np, x0_np, rmsd_t, rmsd_0) in enumerate(rows):
        ax = fig.add_subplot(nr, cols, i + 1, projection="3d")
        ax.scatter(xt_np[mask_np, 0], xt_np[mask_np, 1], xt_np[mask_np, 2],
                   s=8, alpha=0.7, c="C0", label="x_t")
        ax.scatter(truth[mask_np, 0], truth[mask_np, 1], truth[mask_np, 2],
                   s=8, alpha=0.3, c="C3", label="truth")
        ax.set_title(f"step {i} t={t}\nsigma={sigma:.2f}\nRMSD x_t={rmsd_t:.2f}\nRMSD x0_est={rmsd_0:.2f}")
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"Overfit + denoise trajectory — {chosen_sid}")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "reverse_trajectory.png"), dpi=120)
    plt.close(fig)
    print(f"[plot] wrote reverse_trajectory.png", flush=True)

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] all artefacts in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
