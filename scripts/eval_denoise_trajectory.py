"""Generate a denoising trajectory + visualization on a trained dtmol checkpoint.

Companion to scripts/eval_unit_cell_denoise.py — that one is hard-coded to
diamond_C / graphite_C; this one takes --system-id-prefix so we can evaluate
against whatever crystal type the checkpoint was trained on (e.g. fcc_Al).

Output:
    1) Tweedie one-step denoising at several t values (PNG with rows per t).
    2) Full reverse trajectory of ~20 steps (PNG with one row per step).
    3) JSON report with per-atom RMSD reduction at each t.

Usage:
    python scripts/eval_denoise_trajectory.py \
        --ckpt-dir dtmol/models/bindingpose_<DATE>_fixE_fcc_Al \
        --system-id-prefix fcc_Al_2x2x2 \
        --out-dir ralph/test_results/fixE_fcc_Al
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
from typing import Optional

import lmdb
import numpy as np
import torch

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


def latest_ckpt(ckpt_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt-*.pt")),
                   key=os.path.getmtime, reverse=True)
    for p in paths:
        if os.path.getsize(p) > 1024:
            return p
    return None


def build_nets(pretrain_f: str):
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
    return {
        "ligand_encoder": ligand_encoder,
        "protein_encoder": protein_encoder,
        "decoder": decoder,
    }, {"ligand_dict": ligand_dict, "protein_dict": protein_dict}


def build_samplers(T: int = 1000):
    rot_sampler = RotationSampler(schedular=LogLinearScheduler(T, 0.1, 1.65), sde_format="VE")
    tr_sampler = TranslationSampler(schedular=LogLinearScheduler(T, 0.1, 19.0), sde_format="VE")
    g_sampler = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, 1.5), sde_format="VE")
    g_sampler2 = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, 1.5), sde_format="VE")
    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)
    return {"molecule": molecule_sampler, "protein": protein_sampler}


def find_test_idx(ds: UnifiedDataset, prefix: str) -> Optional[int]:
    env = lmdb.open(ds.lmdb_path, subdir=ds._lmdb_subdir, readonly=True, lock=False)
    found = None
    with env.begin() as txn:
        for idx, key in enumerate(ds._keys):
            rec = pickle.loads(txn.get(key))
            sid = str(rec.get("system_id", ""))
            if sid.startswith(prefix):
                found = (idx, sid, rec.get("num_atoms"))
                break
    env.close()
    return found


def index_to_batch(idx: int, ds: UnifiedDataset, device: str) -> dict:
    item = ds[idx]
    batch = UnifiedDataset.collate_fn([item])

    def to_dev(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, dict):
                out[k] = to_dev(v)
            else:
                out[k] = v
        return out
    return to_dev(batch)


def run_forward(nets: dict, batch: dict) -> tuple[dict, torch.Tensor]:
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

    (m_emb, m_attn, m_pad) = nets['ligand_encoder'](**mol_input, features_only=True)
    (p_emb, p_attn, p_pad) = nets['protein_encoder'](**pkt_input, features_only=True)
    single_mol = batch.get('single_molecule_mask')
    output, full_pad = nets['decoder'](
        embd_molecule=m_emb, embd_protein=p_emb,
        coor_molecule=mol_input['src_coord'], coor_protein=pkt_input['src_coord'],
        timesteps=mole_time,
        padding_molecule=m_pad, padding_protein=p_pad,
        attn_mole=m_attn, attn_protein=p_attn,
        cross_distance=cross_dist, cross_edges=cross_edges,
        diffusion_heads=["tr-rotation", "perturbation"],
        single_molecule_mask=single_mol,
    )
    return output, full_pad


def per_atom_rmsd(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diff = pred[mask] - truth[mask]
    return float(np.sqrt((diff ** 2).sum(-1).mean()))


def tweedie_step(nets, samplers, idx, ds, device, t_target: int, seed: int = 0):
    """One-step Tweedie denoise at specific t. Returns (x0_est, x_noisy, x_truth, mask)."""
    batch = index_to_batch(idx, ds, device)
    g_sampler = samplers['molecule'].samplers[-1]
    sigma_t = float(g_sampler.noise[t_target])
    np.random.seed(seed)
    torch.manual_seed(seed)

    mol_coord = batch['net_input']['mol_holo_coord'].clone()
    finite = torch.isfinite(mol_coord)
    noise = torch.randn_like(mol_coord) * sigma_t
    noisy = mol_coord.clone()
    noisy[finite] = mol_coord[finite] + noise[finite]
    batch['net_input']['mol_holo_coord'] = noisy
    batch['net_input']['mol_diffuse_time'] = torch.tensor(
        [[t_target]], device=device, dtype=torch.long)
    batch['net_input']['pocket_diffuse_time'] = torch.tensor(
        [[t_target]], device=device, dtype=torch.long)

    with torch.no_grad():
        output, _ = run_forward(nets, batch)
    n_mol = mol_coord.size(1)
    score_mol = output['perturbation'][:, :n_mol, :]
    x0_est = noisy.clone()
    # The model predicts eps (raw noise) per sampler convention. Tweedie step:
    # x0_est = x_t - sigma_t * eps_pred  (since x_t = x_0 + sigma_t * eps)
    x0_est[finite] = noisy[finite] - sigma_t * score_mol[finite]

    truth = mol_coord.cpu().numpy()[0]
    noisy_np = noisy.cpu().numpy()[0]
    pred = x0_est.cpu().numpy()[0]
    mask = finite.cpu().numpy()[0, :, 0]
    return pred, noisy_np, truth, mask, sigma_t


def reverse_trajectory(nets, samplers, idx, ds, device, n_steps: int = 20,
                        seed: int = 0, start_t: Optional[int] = None,
                        scale: float = 1.0, mode: str = "ddim"):
    """Reverse-diffuse from start_t (default T-1) down to t=0 using the
    probability-flow ODE / DDIM-style deterministic step:

        x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * eps_pred

    This is the correct VE discretisation when the model predicts eps. The
    earlier "Tweedie x0 + fresh randn" step was wrong — it re-injected
    sigma_{t-1}-scale noise that the model couldn't undo on later steps.

    `mode`:
      - "ddim"     : pure deterministic Euler step (default).
      - "tweedie"  : at every step jump to x0_est directly (one-shot Tweedie
                     applied repeatedly). Useful as a sanity check.

    `scale` multiplies eps_pred at sample time to compensate for an
    undersized score head (set 1.0 for plain inference).

    Returns: (traj_list, truth, atom_mask_np). traj_list = list of
    (t, sigma_t, x_displayed[N,3], mask[N]). For ddim mode x_displayed is
    the running x_t (i.e. what the trajectory looks like at each step). The
    Tweedie x0_est at each step is reported in `report.json` separately.
    """
    g_sampler = samplers['molecule'].samplers[-1]
    T = g_sampler.T
    start_t = T - 1 if start_t is None else int(start_t)

    batch = index_to_batch(idx, ds, device)
    mol_clean = batch['net_input']['mol_holo_coord'].clone()
    finite = torch.isfinite(mol_clean)

    np.random.seed(seed)
    torch.manual_seed(seed)
    # Initialise x_t = x_0 + sigma_{start_t} * eps
    sigma_start = float(g_sampler.noise[start_t])
    noise = torch.randn_like(mol_clean) * sigma_start
    x_t = mol_clean.clone()
    x_t[finite] = mol_clean[finite] + noise[finite]

    # Use a log-spaced schedule from sigma_start down to sigma_min so we spend
    # more steps at lower sigmas (where structure forms). np.linspace on t
    # over a log-linear schedule is roughly geometric in sigma already.
    ts = np.linspace(start_t, 0, n_steps).astype(int)
    truth = mol_clean.cpu().numpy()[0]
    mask_np = finite.cpu().numpy()[0, :, 0]
    traj = []
    n_mol = mol_clean.size(1)

    for i, t in enumerate(ts):
        sigma_t = float(g_sampler.noise[int(t)])
        batch['net_input']['mol_holo_coord'] = x_t
        batch['net_input']['mol_diffuse_time'] = torch.tensor(
            [[int(t)]], device=device, dtype=torch.long)
        batch['net_input']['pocket_diffuse_time'] = torch.tensor(
            [[int(t)]], device=device, dtype=torch.long)
        with torch.no_grad():
            output, _ = run_forward(nets, batch)
        eps_pred = output['perturbation'][:, :n_mol, :] * scale

        # Tweedie x0 estimate (for monitoring)
        x0_est = x_t.clone()
        x0_est[finite] = x_t[finite] - sigma_t * eps_pred[finite]
        traj.append({
            "t": int(t), "sigma_t": sigma_t,
            "x_t": x_t.cpu().numpy()[0].copy(),
            "x0_est": x0_est.cpu().numpy()[0].copy(),
            "mask": mask_np,
        })

        if i + 1 >= len(ts):
            break

        if mode == "tweedie":
            # Jump to x0 estimate, then forward-diffuse to next sigma using
            # SAME eps direction (DDIM-equivalent reformulation).
            t_next = int(ts[i + 1])
            sigma_next = float(g_sampler.noise[t_next])
            x_t = x0_est.clone()
            x_t[finite] = x0_est[finite] + sigma_next * eps_pred[finite]
        else:  # "ddim" — Euler step on probability flow ODE
            t_next = int(ts[i + 1])
            sigma_next = float(g_sampler.noise[t_next])
            x_t = x_t.clone()
            x_t[finite] = x_t[finite] - (sigma_t - sigma_next) * eps_pred[finite]
    return traj, truth, mask_np


def plot_panels(rows: list, out_path: str, title: str):
    """rows = [(label, coords[N,3], mask[N])]. Plots all in one row."""
    n = len(rows)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (label, coords, mask) in enumerate(rows):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        c = coords[mask]
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=12, alpha=0.7)
        ax.set_title(label)
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir containing ckpt-*.pt")
    ap.add_argument("--system-id-prefix", required=True,
                    help="Prefix to match in test.lmdb (e.g. fcc_Al_2x2x2)")
    ap.add_argument("--test-lmdb", default="/data/dtMol_Project/datasets/unit_cell_synthesized/test.lmdb")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--t-targets", type=int, nargs="+", default=[100, 300, 500, 800])
    ap.add_argument("--traj-steps", type=int, default=20)
    ap.add_argument("--traj-start-t", type=int, default=None,
                    help="Start the reverse trajectory at this t (default T-1).")
    ap.add_argument("--score-scale", type=float, default=1.0,
                    help="Multiply eps_pred by this at sample time to "
                         "compensate for an undersized score head.")
    ap.add_argument("--mode", choices=["ddim", "tweedie"], default="ddim",
                    help="Reverse-step rule: deterministic ODE Euler (ddim) "
                         "or repeated Tweedie jump (tweedie).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[eval] start = {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    ckpt_path = latest_ckpt(args.ckpt_dir)
    if ckpt_path is None:
        print(f"[eval] ERROR: no ckpt-*.pt in {args.ckpt_dir}", flush=True)
        sys.exit(2)
    print(f"[eval] checkpoint = {ckpt_path}", flush=True)

    pretrain_f = os.path.join(ROOT, "dtmol/pretrain_models")
    nets, atom_dict = build_nets(pretrain_f)
    state = torch.load(ckpt_path, map_location=args.device)
    for k, v in state.items():
        if k in nets:
            nets[k].load_state_dict(v, strict=False)
    for net in nets.values():
        net.to(args.device)
        net.eval()
    print(f"[eval] loaded {list(state.keys())}", flush=True)

    samplers = build_samplers(T=1000)
    ds_config = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    ds = UnifiedDataset(
        lmdb_path=args.test_lmdb,
        ligand_dict=atom_dict["ligand_dict"],
        protein_dict=atom_dict["protein_dict"],
        config=ds_config,
        diffusion_samplers=None,
    )

    found = find_test_idx(ds, args.system_id_prefix)
    if found is None:
        print(f"[eval] ERROR: no test record matching {args.system_id_prefix}", flush=True)
        sys.exit(3)
    idx, sid, n_atoms = found
    print(f"[eval] test record idx={idx} sid={sid} N={n_atoms}", flush=True)

    # ---------- Tweedie one-step at multiple t ----------
    report = {
        "ckpt": ckpt_path, "system_id": sid, "n_atoms": int(n_atoms),
        "tweedie": [],
    }
    rows_by_t = {}
    for t in args.t_targets:
        x0_est, x_noisy, x_truth, mask, sigma = tweedie_step(
            nets, samplers, idx, ds, args.device, t_target=t)
        rmsd_noisy = per_atom_rmsd(x_noisy, x_truth, mask)
        rmsd_denoise = per_atom_rmsd(x0_est, x_truth, mask)
        red = 1.0 - rmsd_denoise / max(rmsd_noisy, 1e-9)
        print(f"[tweedie t={t:>4d} sigma={sigma:.3f}] noisy={rmsd_noisy:.3f}A  "
              f"denoised={rmsd_denoise:.3f}A  reduction={red:+.1%}", flush=True)
        report["tweedie"].append({
            "t": int(t), "sigma_t": sigma,
            "rmsd_noisy": rmsd_noisy, "rmsd_denoised": rmsd_denoise,
            "rmsd_reduction": red,
        })
        rows_by_t[t] = (x_truth, x_noisy, x0_est, mask, sigma)

    # ---------- plot Tweedie panels (one row per t, columns: truth/noisy/denoised) ----------
    n_t = len(args.t_targets)
    fig = plt.figure(figsize=(15, 5 * n_t))
    for r, t in enumerate(args.t_targets):
        truth, noisy, denoised, mask, sigma = rows_by_t[t]
        for c, (label, coords) in enumerate([
            ("ground truth", truth),
            (f"noisy (sigma={sigma:.2f})", noisy),
            ("denoised", denoised),
        ]):
            ax = fig.add_subplot(n_t, 3, r * 3 + c + 1, projection="3d")
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.7)
            ax.set_title(f"t={t}  {label}")
            ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"Tweedie one-step denoise — {sid}")
    plt.tight_layout()
    tw_path = os.path.join(args.out_dir, "tweedie_panels.png")
    fig.savefig(tw_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {tw_path}", flush=True)

    # ---------- reverse trajectory ----------
    traj, truth, mask = reverse_trajectory(
        nets, samplers, idx, ds, args.device,
        n_steps=args.traj_steps, seed=0,
        start_t=args.traj_start_t, scale=args.score_scale, mode=args.mode)
    report["trajectory"] = []
    rmsd_x_t = []
    rmsd_x0 = []
    n_steps = len(traj)
    cols = min(n_steps, 5)
    rows = (n_steps + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i, step in enumerate(traj):
        t = step["t"]; sigma = step["sigma_t"]
        x_t = step["x_t"]; x0 = step["x0_est"]; m = step["mask"]
        rmsd_t = per_atom_rmsd(x_t, truth, m)
        rmsd_0 = per_atom_rmsd(x0, truth, m)
        rmsd_x_t.append(rmsd_t); rmsd_x0.append(rmsd_0)
        report["trajectory"].append({
            "step": i, "t": int(t), "sigma_t": sigma,
            "rmsd_x_t": rmsd_t, "rmsd_x0_est": rmsd_0,
        })
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pts_t = x_t[m]
        ax.scatter(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], s=8, alpha=0.7,
                   c="C0", label="x_t")
        ptst = truth[m]
        ax.scatter(ptst[:, 0], ptst[:, 1], ptst[:, 2], s=8, alpha=0.3,
                   c="C3", label="truth")
        ax.set_title(
            f"step {i} t={t}\nsigma={sigma:.2f}  RMSD x_t={rmsd_t:.2f}\n"
            f"RMSD x0_est={rmsd_0:.2f}")
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"Reverse diffusion trajectory — {sid} ({args.mode}, scale={args.score_scale})")
    plt.tight_layout()
    traj_path = os.path.join(args.out_dir, "reverse_trajectory.png")
    fig.savefig(traj_path, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote {traj_path}", flush=True)
    print(f"[eval] reverse trajectory RMSD: start={rmsd_x_t[0]:.3f}A  "
          f"end={rmsd_x_t[-1]:.3f}A  best_x0={min(rmsd_x0):.3f}A", flush=True)

    # ---------- save report ----------
    rep_path = os.path.join(args.out_dir, "report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] wrote {rep_path}", flush=True)
    print(f"[eval] done in {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
