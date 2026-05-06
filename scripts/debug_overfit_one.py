"""Try to overfit a SINGLE record from the unit_cell datamix.

If the model can drive ratio_p << 1 and |cos_p| -> 1 on one fixed batch,
the architecture is OK and the multi-record plateau is sample-efficiency
or signal-to-noise. If it can't even memorise one sample, the architecture
is fundamentally broken (gradient doesn't flow, equivariance break, etc.).

Usage:
    python scripts/debug_overfit_one.py --steps 500 --lr 1e-3
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn

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


def rms(t):
    if t.numel() == 0:
        return float("nan")
    f = t[torch.isfinite(t)]
    if f.numel() == 0:
        return float("nan")
    return float(f.float().pow(2).mean().sqrt().item())


def cos_sim(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    n = (af.norm() * bf.norm()).item()
    if n == 0:
        return float("nan")
    return float((af * bf).sum().item() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datamix", type=str,
                    default=os.path.join(ROOT, "dtmol/data/datamix_unit_cell.yaml"))
    ap.add_argument("--device", type=str, default="cuda")
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
    train_loader = get_mixed_dataloader(mixer, batch_size=1, num_workers=0)

    config = CONFIG(
        lambda_force=0.0, lambda_fd_force=0.0, force_loss_fn="mse",
        dataset_mode="unified", use_wandb=False,
    )
    trainer = DiffusionTrainer(
        train_dataloader=train_loader, eval_dataloader=train_loader,
        nets=nets, sampler={"molecule": molecule_sampler, "protein": protein_sampler},
        config=config, device=DEVICE,
    )
    trainer.load_unimol_pretrain(pretrain_f)
    no_dropout = os.environ.get("NO_DROPOUT", "0") == "1"
    for net in trainer.nets.values():
        net.to(DEVICE)
        net.train()
        if no_dropout:
            for m in net.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
    if no_dropout:
        print("[mode] all nn.Dropout p=0 (BN still in train mode)", flush=True)

    if os.environ.get("ZERO_INIT_HEADY", "0") == "1":
        with torch.no_grad():
            trainer.nets["decoder"].diffusion_heads["tr-rotation"].out_proj2.weight.zero_()
            trainer.nets["decoder"].diffusion_heads["perturbation"].linear3.weight.zero_()
        print("[fix] zero-init head y-branch", flush=True)

    # ----- Get ONE batch and freeze it -----
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

    fixed_batch = next(iter(train_loader))
    fixed_batch = to_device(fixed_batch)
    print(f"Frozen batch: mol_seq={fixed_batch['net_input']['mol_tokens'].shape[1]} "
          f"sgl={fixed_batch['single_molecule_mask'].tolist()} "
          f"mol_t={fixed_batch['diffused']['mol_diffuse_time'].view(-1).tolist()}",
          flush=True)

    optimizer = torch.optim.Adam(trainer.nets["decoder"].parameters(), lr=args.lr)

    captured = {}
    inner_decoder = trainer.nets["decoder"].decoder

    def cap(_m, _i, out):
        if isinstance(out, tuple) and len(out) >= 6:
            captured["decoder_rep"] = out[0].detach()
            captured["node_rep"] = out[5].detach()
    inner_decoder.register_forward_hook(cap)

    head_trrot = trainer.nets["decoder"].diffusion_heads["tr-rotation"]
    head_pert = trainer.nets["decoder"].diffusion_heads["perturbation"]
    head_trrot.register_forward_hook(lambda _m, _i, o: captured.update(out_trrot=o.detach()))
    head_pert.register_forward_hook(lambda _m, _i, o: captured.update(out_pert=o.detach()))

    print(
        f"#{'i':>5} | {'tg_tr':>8} {'pr_tr':>8} {'cos_tr':>6} {'r_tr':>5} | "
        f"{'tg_p':>8} {'pr_p':>8} {'cos_p':>6} {'r_p':>5} | "
        f"{'dec':>7} {'node':>7} {'loss':>8}",
        flush=True,
    )

    diff = fixed_batch["diffused"]
    if "mol_diffuse_trrot_score" in diff:
        tg_trrot_full = diff["mol_diffuse_trrot_score"][:, :2, :].float()
        tg_pert_mol = diff["mol_diffuse_perturb_score"].float()
        tr_norm = diff["mol_diffuse_trrot_norm"][:, :2].float()
        pert_norm_mol = diff["mol_diffuse_perturb_norm"].float()
    else:
        mol_score = diff["mol_diffuse_score"].float()
        tg_trrot_full = mol_score[:, :2, :]
        tg_pert_mol = mol_score[:, 2:, :]
        tr_norm = diff["mol_diffuse_norm"][:, :2].float()
        pert_norm_mol = diff["mol_diffuse_norm"][:, 2:].float()
    tg_pert_pkt = diff["pocket_diffuse_score"].float()
    tg_pert_full = torch.cat([tg_pert_mol, tg_pert_pkt], dim=1)
    pert_norm_pkt = diff["pocket_diffuse_norm"].float()
    pert_norm = torch.cat([pert_norm_mol, pert_norm_pkt], dim=1)

    tr_mask = tr_norm > 0
    tg_trrot_active = tg_trrot_full[tr_mask]
    pert_mask = (pert_norm > 0).unsqueeze(-1)
    tg_pert_active = tg_pert_full[pert_mask.expand_as(tg_pert_full)].view(-1, 3)

    t0 = time.time()
    log_steps = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000, args.steps - 1]))
    log_set = set(s for s in log_steps if 0 <= s < args.steps)

    for i in range(args.steps):
        captured.clear()
        loss, loss_dict = trainer.train_step(fixed_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i in log_set or i == args.steps - 1:
            pr_trrot = captured["out_trrot"].float().view(-1, 2, 3)
            pr_pert = captured["out_pert"].float()
            pr_trrot_active = pr_trrot[tr_mask]
            pr_pert_active = pr_pert[pert_mask.expand_as(pr_pert)].view(-1, 3)
            bs_tr = tg_trrot_active.pow(2).mean().item() if tg_trrot_active.numel() else float("nan")
            ac_tr = (pr_trrot_active - tg_trrot_active).pow(2).mean().item() if tg_trrot_active.numel() else float("nan")
            bs_p = tg_pert_active.pow(2).mean().item() if tg_pert_active.numel() else float("nan")
            ac_p = (pr_pert_active - tg_pert_active).pow(2).mean().item() if tg_pert_active.numel() else float("nan")
            r_tr = ac_tr / bs_tr if bs_tr > 0 else float("nan")
            r_p = ac_p / bs_p if bs_p > 0 else float("nan")
            print(
                f"#{i:>5d} | "
                f"{rms(tg_trrot_active):>8.2e} {rms(pr_trrot_active):>8.2e} "
                f"{cos_sim(pr_trrot_active, tg_trrot_active):>+6.3f} {r_tr:>5.2f} | "
                f"{rms(tg_pert_active):>8.2e} {rms(pr_pert_active):>8.2e} "
                f"{cos_sim(pr_pert_active, tg_pert_active):>+6.3f} {r_p:>5.2f} | "
                f"{rms(captured.get('decoder_rep', torch.tensor([]))):>7.1e} "
                f"{rms(captured.get('node_rep', torch.tensor([]))):>7.1e} "
                f"{loss.item():>8.3e}",
                flush=True,
            )

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("\nIf ratio -> 0 and |cos| -> 1, architecture is OK; multi-record plateau is sample efficiency.")
    print("If ratio stays ~ 1 even on one batch, architecture is broken.")


if __name__ == "__main__":
    main()
