"""Diagnose why diffusion training loss plateaus.

For each of the first N batches, runs trainer.train_step() and reports per
head (tr-rotation, perturbation):

    target_rms        sqrt(mean(score**2))         — magnitude of what we are
                                                     trying to predict
    pred_rms          sqrt(mean(output**2))        — magnitude of what the
                                                     network actually produced
    baseline_loss     mean(score**2)               — MSE if pred were 0
    actual_loss       mean((output-score)**2)      — what we are minimising
    ratio             actual_loss / baseline_loss  — ~1.0 means "head is dead"
                                                     (predicting near-zero)

Plus the magnitudes of the two head inputs:

    decoder_rep_rms   scalar branch into x_gate(linear(decoder_rep))
    node_rep_rms      vector branch (SE3 stack output)

Hypothesis under test: plateau across (with-encoder, no-encoder, from-scratch)
points at the heads producing near-zero predictions because the SE3 branch
collapses or the scalar gate is closed. If ratio ~ 1.0 for many timesteps,
the head is dead and weight updates can't easily climb out of it.

Usage:
    python scripts/debug_plateau.py --max-batches 30
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import logging
import math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dtmol"))

from dtmol.dtmol_train_test import DiffusionTrainer
from dtmol.dtmol_train_base import CONFIG
from dtmol.dtmol_model import DummyModelConfig
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.utils.dictionary import Dictionary
from dtmol.data.mixer import DatasetMixer, get_mixed_dataloader
from dtmol.data.unified_dataset import UnifiedDatasetConfig
from dtmol.diffusion import (
    RotationSampler, GaussianSampler, TranslationSampler,
    ChainSampler, LogLinearScheduler,
)


def rms(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.float().pow(2).mean().sqrt().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datamix", type=str,
                    default=os.path.join(ROOT, "dtmol/data/datamix_unit_cell.yaml"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="Skip loading pretrained encoder weights")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.WARNING)

    DEVICE = args.device
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
    rot_sampler = RotationSampler(schedular=LogLinearScheduler(T, 0.1, 1.65), sde_format="VE")
    tr_sampler = TranslationSampler(schedular=LogLinearScheduler(T, 0.1, 19.0), sde_format="VE")
    g_sampler = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, 1.5), sde_format="VE")
    g_sampler2 = GaussianSampler(schedular=LogLinearScheduler(T, 0.04, 1.5), sde_format="VE")
    molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
    protein_sampler = ChainSampler(g_sampler2)
    protein_sampler.conjugate(molecule_sampler)

    ds_config = UnifiedDatasetConfig(max_seq_len=1000, max_pocket_atoms=256, seed=0)
    mixer = DatasetMixer(
        datamix_path=args.datamix, ligand_dict=ligand_dict,
        protein_dict=protein_dict, config=ds_config,
        diffusion_samplers={"molecule": molecule_sampler, "protein": protein_sampler},
    )
    train_loader = get_mixed_dataloader(mixer, batch_size=args.batch_size, num_workers=0)

    config = CONFIG(
        lambda_force=0.0, lambda_fd_force=0.0, force_loss_fn="mse",
        dataset_mode="unified", use_wandb=False,
    )
    trainer = DiffusionTrainer(
        train_dataloader=train_loader, eval_dataloader=train_loader,
        nets=nets, sampler={"molecule": molecule_sampler, "protein": protein_sampler},
        config=config, device=DEVICE,
    )
    if not args.no_pretrain:
        trainer.load_unimol_pretrain(pretrain_f)
    for net in trainer.nets.values():
        net.to(DEVICE)
        net.train()

    if os.environ.get("NO_DROPOUT", "0") == "1":
        for net in trainer.nets.values():
            for m in net.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
        print("[mode] all nn.Dropout p=0 (BN still in train mode)", flush=True)

    optimizer = torch.optim.Adam(trainer.nets["decoder"].parameters(), lr=args.lr)

    # ----- Hooks: capture decoder_rep, node_rep, and head inputs/outputs -----
    captured = {}

    decoder_module = trainer.nets["decoder"]
    inner_decoder = decoder_module.decoder  # TransformerDecoderWithPair

    def cap_outer_decoder(_mod, _inp, out):
        # forward returns 6-tuple when diffusion_heads=None,
        # but when called with heads we get (scores_dict, padding_mask).
        # We hook the inner TransformerDecoderWithPair instead — see below.
        pass

    def cap_inner_decoder(_mod, _inp, out):
        # out = (decoder_rep, decoder_pair_rep, delta_decoder_pair_rep,
        #        x_norm, delta_decoder_pair_rep_norm, node_rep)
        if isinstance(out, tuple) and len(out) >= 6:
            captured["decoder_rep"] = out[0].detach()
            captured["node_rep"] = out[5].detach()

    inner_decoder.register_forward_hook(cap_inner_decoder)

    head_trrot = decoder_module.diffusion_heads["tr-rotation"]
    head_pert = decoder_module.diffusion_heads["perturbation"]

    def cap_head(name):
        def _cap(_mod, _inp, out):
            captured[f"out_{name}"] = out.detach()
        return _cap

    head_trrot.register_forward_hook(cap_head("trrot"))
    head_pert.register_forward_hook(cap_head("pert"))

    # Optional fix-under-test: zero-init the geometric projection in each head
    # so pred = 0 at init (loss = baseline), letting gradient align direction
    # without the optimizer first killing the y_branch.
    if os.environ.get("ZERO_INIT_HEADY", "0") == "1":
        with torch.no_grad():
            head_trrot.out_proj2.weight.zero_()
            head_pert.linear3.weight.zero_()
        print("[fix] zero-init head_trrot.out_proj2 and head_pert.linear3", flush=True)

    def cos_sim(a, b):
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        af = a.float().reshape(-1)
        bf = b.float().reshape(-1)
        n = af.norm() * bf.norm()
        if n.item() == 0:
            return float("nan")
        return float((af * bf).sum().item() / n.item())

    print(
        f"#{'i':>3} {'mol_t':>5} {'sgl':>3} | "
        f"{'tg_tr':>8} {'pr_tr':>8} {'cos_tr':>6} {'r_tr':>5} | "
        f"{'tg_p':>8} {'pr_p':>8} {'cos_p':>6} {'r_p':>5} | "
        f"{'dec':>7} {'node':>7} {'loss':>8}",
        flush=True,
    )

    history: dict[str, list[float]] = {
        "ratio_tr": [], "ratio_p": [], "cos_tr": [], "cos_p": [],
        "pred_tr": [], "pred_p": [], "loss": [],
    }

    t0 = time.time()
    for i, batch in enumerate(train_loader):
        if i >= args.max_batches:
            break

        def to_device(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(DEVICE)
                elif isinstance(v, dict):
                    out[k] = to_device(v)
                else:
                    out[k] = v
            return out
        batch = to_device(batch)

        captured.clear()
        try:
            loss, loss_dict = trainer.train_step(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise

        diff = batch["diffused"]
        # Targets
        if "mol_diffuse_trrot_score" in diff:
            tg_trrot = diff["mol_diffuse_trrot_score"][:, :2, :].float()
            tg_pert_mol = diff["mol_diffuse_perturb_score"].float()
        else:
            mol_score = diff["mol_diffuse_score"].float()
            tg_trrot = mol_score[:, :2, :]
            tg_pert_mol = mol_score[:, 2:, :]
        tg_pert_pkt = diff["pocket_diffuse_score"].float()
        tg_pert = torch.cat([tg_pert_mol, tg_pert_pkt], dim=1)

        # Predictions (from hooks)
        pr_trrot = captured.get("out_trrot")
        if pr_trrot is not None:
            pr_trrot = pr_trrot.float().view(-1, 2, 3)
        pr_pert = captured.get("out_pert")
        if pr_pert is not None:
            pr_pert = pr_pert.float()

        # Trrot: only consider non-zero score rows (norm > 0)
        if "mol_diffuse_trrot_norm" in diff:
            tr_norm = diff["mol_diffuse_trrot_norm"][:, :2].float()  # [B, 2]
            tr_mask = tr_norm > 0  # [B, 2]
        else:
            tr_norm = diff["mol_diffuse_norm"][:, :2].float()
            tr_mask = tr_norm > 0
        tg_trrot_active = tg_trrot[tr_mask]
        if pr_trrot is not None:
            pr_trrot_active = pr_trrot[tr_mask]
        else:
            pr_trrot_active = torch.zeros_like(tg_trrot_active)

        # Pert: drop padding using padding_mask from full sequence
        # (we don't have it here without re-computing; approximate by norm > 0)
        if "mol_diffuse_perturb_norm" in diff:
            pert_norm_mol = diff["mol_diffuse_perturb_norm"].float()
        else:
            pert_norm_mol = diff["mol_diffuse_norm"][:, 2:].float()
        pert_norm_pkt = diff["pocket_diffuse_norm"].float()
        pert_norm = torch.cat([pert_norm_mol, pert_norm_pkt], dim=1)
        pert_mask = (pert_norm > 0).unsqueeze(-1)
        tg_pert_active = tg_pert[pert_mask.expand_as(tg_pert)].view(-1, 3)
        if pr_pert is not None:
            pr_pert_active = pr_pert[pert_mask.expand_as(pr_pert)].view(-1, 3)
        else:
            pr_pert_active = torch.zeros_like(tg_pert_active)

        baseline_trrot = float(tg_trrot_active.float().pow(2).mean().item()) if tg_trrot_active.numel() else float("nan")
        actual_trrot = float((pr_trrot_active - tg_trrot_active).float().pow(2).mean().item()) if tg_trrot_active.numel() else float("nan")
        baseline_pert = float(tg_pert_active.float().pow(2).mean().item()) if tg_pert_active.numel() else float("nan")
        actual_pert = float((pr_pert_active - tg_pert_active).float().pow(2).mean().item()) if tg_pert_active.numel() else float("nan")

        mol_t = batch["diffused"]["mol_diffuse_time"].view(-1)[0].item()
        pkt_t = batch["diffused"]["pocket_diffuse_time"].view(-1)[0].item()
        sgl = bool(batch.get("single_molecule_mask", torch.tensor([False]))[0].item()) if "single_molecule_mask" in batch else False

        dec_rep_rms = rms(captured.get("decoder_rep", torch.tensor([])))
        node_rep_rms = rms(captured.get("node_rep", torch.tensor([])))

        ratio_tr = actual_trrot / baseline_trrot if baseline_trrot > 0 else float("nan")
        ratio_p = actual_pert / baseline_pert if baseline_pert > 0 else float("nan")

        cos_tr = cos_sim(pr_trrot_active, tg_trrot_active)
        cos_p = cos_sim(pr_pert_active, tg_pert_active)

        history["ratio_tr"].append(ratio_tr if ratio_tr == ratio_tr else float("nan"))
        history["ratio_p"].append(ratio_p if ratio_p == ratio_p else float("nan"))
        history["cos_tr"].append(cos_tr if cos_tr == cos_tr else float("nan"))
        history["cos_p"].append(cos_p if cos_p == cos_p else float("nan"))
        history["pred_tr"].append(rms(pr_trrot_active))
        history["pred_p"].append(rms(pr_pert_active))
        history["loss"].append(loss.item())

        print(
            f"#{i:>3d} {int(mol_t):>5d} {int(sgl):>3d} | "
            f"{rms(tg_trrot_active):>8.2e} {rms(pr_trrot_active):>8.2e} "
            f"{cos_tr:>+6.3f} {ratio_tr:>5.2f} | "
            f"{rms(tg_pert_active):>8.2e} {rms(pr_pert_active):>8.2e} "
            f"{cos_p:>+6.3f} {ratio_p:>5.2f} | "
            f"{dec_rep_rms:>7.1e} {node_rep_rms:>7.1e} {loss.item():>8.3e}",
            flush=True,
        )

        if torch.isfinite(loss):
            optimizer.zero_grad()
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

        del loss, loss_dict
        if i % 10 == 0 and DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    # ----- Aggregate over last K steps -----
    K = max(1, min(50, len(history["ratio_tr"])))
    def _tail_mean(name: str) -> float:
        vals = [v for v in history[name][-K:] if v == v]  # drop NaN
        return float(sum(vals) / len(vals)) if vals else float("nan")
    def _tail_mean_abs(name: str) -> float:
        vals = [abs(v) for v in history[name][-K:] if v == v]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    print(f"\nDone in {time.time()-t0:.1f}s, {len(history['loss'])} steps")
    print(f"\n=== AGGREGATE over last {K} steps ===")
    print(f"  ratio_tr (mean):   {_tail_mean('ratio_tr'):.3f}    (lower=better, baseline=1.0)")
    print(f"  ratio_p  (mean):   {_tail_mean('ratio_p'):.3f}    (lower=better, baseline=1.0)")
    print(f"  |cos_tr| (mean):   {_tail_mean_abs('cos_tr'):.3f}    (higher=better)")
    print(f"  |cos_p|  (mean):   {_tail_mean_abs('cos_p'):.3f}    (higher=better)")
    print(f"  pred_tr_rms (mean):{_tail_mean('pred_tr'):.3e}")
    print(f"  pred_p_rms  (mean):{_tail_mean('pred_p'):.3e}")
    print(f"  loss     (mean):   {_tail_mean('loss'):.3f}")
    print("\nLegend: ratio < 0.5 -> learning. cos|>| > 0.3 -> direction-aligned.")


if __name__ == "__main__":
    main()
