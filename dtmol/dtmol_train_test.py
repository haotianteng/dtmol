import torch
import toml
import os
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict,Union
from dtmol_input import load_unimol_binding_data,get_dataloader
from dtmol.dtmol_train_base import Trainer,CONFIG
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.dtmol_model import DummyModelConfig
from dtmol.diffusion import RotationSampler, GaussianSampler
from torch.utils.data import DataLoader

class DiffusionTrainer(Trainer):
    def __init__(self, train_dataloader: DataLoader,
                 nets: Dict[str, Union[UniMolEncoder, Decoder]],
                 sampler: Union[RotationSampler, GaussianSampler],
                 config: Union[dict],
                 device: str = None,
                 eval_dataloader: DataLoader = None):
        super().__init__(train_dataloader, nets, config, device, eval_dataloader)
        self.sampler = sampler
        self.lambda_force: float = config.TRAIN.get('lambda_force', 0.0)
        self.lambda_fd_force: float = config.TRAIN.get('lambda_fd_force', 0.0)
        self.force_loss_fn_name: str = config.TRAIN.get('force_loss_fn', 'mse')

    def _get_mol_input(self, batch):
        return {"src_tokens": batch['net_input']['mol_tokens'],
                "src_distance": batch['net_input']['mol_holo_distance'],
                "src_coord": batch['net_input']['mol_holo_coord'],
                "src_edge_type": batch['net_input']['mol_edge_type']}

    def _get_pocket_input(self, batch):
        return {"src_tokens": batch['net_input']['pocket_tokens'],
                "src_distance": batch['net_input']['pocket_distance'],
                "src_coord": batch['net_input']['pocket_holo_coord'],
                "src_edge_type": batch['net_input']['pocket_edge_type']}

    def _get_pocket_diffused(self, batch, change_atom=False):
        atom_key = "diffused" if change_atom else "net_input"
        return {"src_tokens": batch[atom_key]['pocket_tokens'],
                "src_distance": batch['diffused']['pocket_distance'],
                "src_coord": batch['diffused']['pocket_holo_coord'],
                "src_edge_type": batch[atom_key]['pocket_edge_type']}

    def _get_mole_diffused(self, batch, change_atom=False):
        atom_key = "diffused" if change_atom else "net_input"
        return {"src_tokens": batch[atom_key]['mol_tokens'],
                "src_distance": batch['diffused']['mol_holo_distance'],
                "src_coord": batch['diffused']['mol_holo_coord'],
                "src_edge_type": batch[atom_key]['mol_edge_type']}

    def load_unimol_pretrain(self, pretrain_folder):
        ligand_model_dict = torch.load(f"{pretrain_folder}/unimol_molecule_pretrain.pt")
        protein_model_dict = torch.load(f"{pretrain_folder}/unimol_protein_pretrain.pt")
        self.nets['ligand_encoder'].load_state_dict(ligand_model_dict["model"], strict=False)
        self.nets['protein_encoder'].load_state_dict(protein_model_dict["model"], strict=False)

    def train(self, epoches: int, optimizer, save_every_n_steps: int = 100,
              valid_every_n_steps: int = 100, save_folder: str = None,
              param_norm_every_n_steps: int = 100):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            for i_step, batch in enumerate(self.train_ds):
                loss, loss_dict = self.train_step(batch)
                if torch.isnan(loss):
                    self._alert("NaN loss detected, skip this training step.",level = "warning")
                    continue
                optimizer.zero_grad()
                loss.backward()
                # Snapshot parameter and gradient norms BEFORE optimizer.step
                # so the wandb panel reflects the state that produced the
                # current gradients (early warning for weight blow-up).
                pn_metrics = None
                if (self.use_wandb and i_step % param_norm_every_n_steps == 0):
                    pn_metrics = self._param_norm_metrics(include_grads=True)
                optimizer.step()
                if i_step % save_every_n_steps == 0:
                    self.save()
                if i_step % valid_every_n_steps == 0:
                    with torch.no_grad():
                        valid_batch = next(iter(self.eval_ds))
                        valid_loss, valid_loss_dict = self.valid_step(valid_batch)
                        msg = f"Epoch {epoch_i}: Step {i_step}, train loss {loss:.4f}, valid loss {valid_loss:.4f}"
                        self.logger.info(msg)
                        if self.use_wandb:
                            log_dict: Dict[str, object] = {
                                "epoch": epoch_i,
                                "global_step": self.global_step,
                                "train_loss": loss,
                                "train_diffusion_loss": loss_dict['diffusion_loss'],
                                "train_force_loss": loss_dict['force_loss'],
                                "valid_loss": valid_loss,
                                "valid_diffusion_loss": valid_loss_dict['diffusion_loss'],
                                "valid_force_loss": valid_loss_dict['force_loss'],
                            }
                            if pn_metrics is not None:
                                log_dict.update(pn_metrics)
                            wandb.log(log_dict)
                elif pn_metrics is not None:
                    # Param-norm cadence may be tighter than valid cadence;
                    # log on its own when there's no valid step this iter.
                    pn_metrics["epoch"] = epoch_i
                    pn_metrics["global_step"] = self.global_step
                    wandb.log(pn_metrics)

    def _call_decoder(self, batch, mole_input, pocket_input, mole_embd, pocket_embd,
                      mole_padding, pocket_padding, mole_attn, pocket_attn):
        """Call decoder with proper args matching Decoder.forward signature."""
        mole_time = batch['diffused']['mol_diffuse_time'].view(-1)
        pocket_time = batch['diffused']['pocket_diffuse_time'].view(-1)
        cross_dist = batch['diffused']['cross_distance']
        cross_edges = batch['diffused']['cross_edge_type']
        single_molecule_mask = batch.get('single_molecule_mask', None)
        output, padding_mask = self.nets['decoder'](
            embd_molecule=mole_embd,
            embd_protein=pocket_embd,
            coor_molecule=mole_input['src_coord'],
            coor_protein=pocket_input['src_coord'],
            timesteps=mole_time,
            padding_molecule=mole_padding,
            padding_protein=pocket_padding,
            attn_mole=mole_attn,
            attn_protein=pocket_attn,
            cross_distance=cross_dist,
            cross_edges=cross_edges,
            diffusion_heads=["tr-rotation", "perturbation"],
            single_molecule_mask=single_molecule_mask,
        )
        return output, padding_mask

    def train_step(self, batch):
        mole_input = self._get_mole_diffused(batch)
        pocket_input = self._get_pocket_diffused(batch)
        (mole_embd, mole_attn, mole_padding) = self.nets['ligand_encoder'](**mole_input, features_only=True)
        (pocket_embd, pocket_attn, pocket_padding) = self.nets['protein_encoder'](**pocket_input, features_only=True)

        output, padding_mask = self._call_decoder(
            batch, mole_input, pocket_input, mole_embd, pocket_embd,
            mole_padding, pocket_padding, mole_attn, pocket_attn)
        loss_dict = self.diffusion_loss(output, padding_mask.clone(), batch)
        loss = loss_dict['total_loss']
        return loss, loss_dict

    def valid_step(self, batch):
        with torch.no_grad():
            mole_input = self._get_mole_diffused(batch)
            pocket_input = self._get_pocket_diffused(batch)
            (mole_embd, mole_attn, mole_padding) = self.nets['ligand_encoder'](**mole_input, features_only=True)
            (pocket_embd, pocket_attn, pocket_padding) = self.nets['protein_encoder'](**pocket_input, features_only=True)

            output, padding_mask = self._call_decoder(
                batch, mole_input, pocket_input, mole_embd, pocket_embd,
                mole_padding, pocket_padding, mole_attn, pocket_attn)
            loss_dict = self.diffusion_loss(output, padding_mask.clone(), batch)
            loss = loss_dict['total_loss']

        return loss, loss_dict

    def _compute_force_loss(self, pred: torch.Tensor, target: torch.Tensor,
                            reduction: str = 'none') -> torch.Tensor:
        if self.force_loss_fn_name == 'smooth_l1':
            return F.smooth_l1_loss(pred, target, reduction=reduction)
        return F.mse_loss(pred, target, reduction=reduction)

    def diffusion_loss(self, output, padding_mask, batch, atom_diffusion=False):
        diffused_dict = batch['diffused']
        # Use separate trrot/perturb keys if available (UnifiedDataset format),
        # otherwise fall back to combined mol_diffuse_score (CrossDataset format).
        if 'mol_diffuse_trrot_score' in diffused_dict:
            # trrot has exactly 2 real rows (rot, tr); legacy CrossDataset may
            # pad to mol_length, so always slice to [:, :2].
            mol_trrot_score = diffused_dict['mol_diffuse_trrot_score'][:, :2, :].to(torch.float32)
            mol_trrot_norm = diffused_dict['mol_diffuse_trrot_norm'][:, :2].to(torch.float32)
            mol_perturb_score = diffused_dict['mol_diffuse_perturb_score'].to(torch.float32)
            mol_perturb_norm = diffused_dict['mol_diffuse_perturb_norm'].to(torch.float32)
        else:
            mol_score = diffused_dict['mol_diffuse_score'].to(torch.float32)
            mol_trrot_score = mol_score[:, :2, :]
            mol_trrot_norm = diffused_dict['mol_diffuse_norm'].to(torch.float32)[:, :2]
            mol_perturb_score = mol_score[:, 2:, :]
            mol_perturb_norm = diffused_dict['mol_diffuse_norm'].to(torch.float32)[:, 2:]
        pocket_score = diffused_dict['pocket_diffuse_score'].to(torch.float32)
        pocket_norm = diffused_dict['pocket_diffuse_norm'].to(torch.float32)
        perturbation_score = torch.cat([mol_perturb_score, pocket_score], axis=1)
        perturbation_norm = torch.cat([mol_perturb_norm, pocket_norm], axis=1)
        tr_rot = output['tr-rotation'].view(-1, 2, 3)  # [B,6] -> [B,2,3]
        pert = output['perturbation']
        trrot_loss = self.nets['decoder'].diffusion_heads['tr-rotation'].loss(tr_rot, mol_trrot_score, mol_trrot_norm)
        pert_loss = self.nets['decoder'].diffusion_heads['perturbation'].loss(pert, perturbation_score, perturbation_norm, padding_mask)
        diffusion_loss = trrot_loss + pert_loss

        loss_dict: Dict[str, torch.Tensor] = {
            'trrot_loss': trrot_loss,
            'pert_loss': pert_loss,
            'diffusion_loss': diffusion_loss,
        }

        # --- Tier-aware force loss ---
        force_tiers = batch.get('force_tier')
        if (self.lambda_force > 0.0 or self.lambda_fd_force > 0.0) and force_tiers is not None:
            bsz = pert.size(0)
            mol_len = mol_perturb_score.size(1)

            # Split perturbation prediction into mol and pocket parts
            pred_mol = pert[:, :mol_len, :]   # [B, M, 3]
            pred_pkt = pert[:, mol_len:, :]   # [B, P, 3]

            # Get target forces from diffused dict
            mol_forces = diffused_dict['mol_real_forces'].to(torch.float32)  # [B, M, 3]
            pkt_forces = diffused_dict['pocket_real_forces'].to(torch.float32)  # [B, P, 3]

            # Compute force normalization: scale forces to match perturbation score magnitudes
            # Use perturbation norm as the reference scale
            mol_pert_norm = perturbation_norm[:, :mol_len]   # [B, M]
            pkt_pert_norm = perturbation_norm[:, mol_len:]   # [B, P]

            # Normalize forces by perturbation norm so they are on the same scale as scores
            mol_force_norm = mol_pert_norm.unsqueeze(-1).clamp(min=1e-6)  # [B, M, 1]
            pkt_force_norm = pkt_pert_norm.unsqueeze(-1).clamp(min=1e-6)  # [B, P, 1]
            mol_forces_scaled = mol_forces / mol_force_norm
            pkt_forces_scaled = pkt_forces / pkt_force_norm

            # Build per-sample masks for Tier A and Tier B
            tier_a_mask = torch.zeros(bsz, dtype=torch.bool, device=pert.device)
            tier_b_mask = torch.zeros(bsz, dtype=torch.bool, device=pert.device)
            for i, tier in enumerate(force_tiers):
                if tier == 'A':
                    tier_a_mask[i] = True
                elif tier == 'B':
                    tier_b_mask[i] = True

            force_loss = torch.tensor(0.0, device=pert.device)

            # Tier A: real DFT forces
            if self.lambda_force > 0.0 and tier_a_mask.any():
                # Molecule force loss for Tier A samples
                mol_fl = self._compute_force_loss(
                    pred_mol[tier_a_mask], mol_forces_scaled[tier_a_mask], reduction='none')
                # Mask out padding (BOS/EOS have zero forces, padding_mask[:, :2] already True)
                mol_pad = padding_mask[tier_a_mask, :mol_len]
                mol_fl = mol_fl * (~mol_pad.unsqueeze(-1))
                mol_fl = mol_fl[~mol_pad].mean() if (~mol_pad).any() else torch.tensor(0.0, device=pert.device)

                # Pocket force loss for Tier A samples
                pkt_pad = torch.zeros(tier_a_mask.sum(), pred_pkt.size(1),
                                      dtype=torch.bool, device=pert.device)
                # Mask out padding positions in pocket (where pocket_norm == 0)
                pkt_pad = pkt_pad | (pkt_pert_norm[tier_a_mask] == 0)
                pkt_fl = self._compute_force_loss(
                    pred_pkt[tier_a_mask], pkt_forces_scaled[tier_a_mask], reduction='none')
                pkt_fl = pkt_fl * (~pkt_pad.unsqueeze(-1))
                pkt_fl = pkt_fl[~pkt_pad].mean() if (~pkt_pad).any() else torch.tensor(0.0, device=pert.device)

                tier_a_force_loss = self.lambda_force * (mol_fl + pkt_fl)
                force_loss = force_loss + tier_a_force_loss
                loss_dict['tier_a_force_loss'] = tier_a_force_loss

            # Tier B: finite-difference forces
            if self.lambda_fd_force > 0.0 and tier_b_mask.any():
                mol_fl = self._compute_force_loss(
                    pred_mol[tier_b_mask], mol_forces_scaled[tier_b_mask], reduction='none')
                mol_pad = padding_mask[tier_b_mask, :mol_len]
                mol_fl = mol_fl * (~mol_pad.unsqueeze(-1))
                mol_fl = mol_fl[~mol_pad].mean() if (~mol_pad).any() else torch.tensor(0.0, device=pert.device)

                pkt_pad = torch.zeros(tier_b_mask.sum(), pred_pkt.size(1),
                                      dtype=torch.bool, device=pert.device)
                pkt_pad = pkt_pad | (pkt_pert_norm[tier_b_mask] == 0)
                pkt_fl = self._compute_force_loss(
                    pred_pkt[tier_b_mask], pkt_forces_scaled[tier_b_mask], reduction='none')
                pkt_fl = pkt_fl * (~pkt_pad.unsqueeze(-1))
                pkt_fl = pkt_fl[~pkt_pad].mean() if (~pkt_pad).any() else torch.tensor(0.0, device=pert.device)

                tier_b_force_loss = self.lambda_fd_force * (mol_fl + pkt_fl)
                force_loss = force_loss + tier_b_force_loss
                loss_dict['tier_b_force_loss'] = tier_b_force_loss

            loss_dict['force_loss'] = force_loss
        else:
            force_loss = torch.tensor(0.0, device=pert.device)
            loss_dict['force_loss'] = force_loss

        loss_dict['total_loss'] = diffusion_loss + force_loss

        if atom_diffusion:
            raise NotImplementedError("Atom diffusion is not implemented yet.")
        return loss_dict

def build_encoder(pretrain_f):
    ligand_dict = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
    ligand_dict.add_symbol("[MASK]", is_special=True)
    protein_dict.add_symbol("[MASK]", is_special=True)
    encoder_config = DummyModelConfig(mode="encode")
    ligand_encoder = UniMolEncoder(args = encoder_config, dictionary=ligand_dict)
    protein_encoder = UniMolEncoder(args = encoder_config, dictionary=protein_dict)
    return {"ligand_encoder": ligand_encoder, "protein_encoder": protein_encoder}, {"ligand_dict": ligand_dict, "protein_dict": protein_dict}

if __name__ == "__main__":
    import argparse
    import logging
    import time
    from dtmol.utils.dictionary import Dictionary
    from dtmol.utils.datasets import CrossDataset

    # --- CLI argument parsing ---
    parser = argparse.ArgumentParser(description="dtmol training script")
    parser.add_argument("--dataset-mode", type=str, default="legacy",
                        choices=["legacy", "unified"],
                        help="Dataset mode: 'legacy' uses CrossDataset, "
                             "'unified' uses UnifiedDataset + DatasetMixer (default: legacy)")
    parser.add_argument("--datamix", type=str, default=None,
                        help="Path to datamix.yaml for unified dataset mode")
    parser.add_argument("--datamix-val", type=str, default=None,
                        help="Path to validation datamix YAML for unified mode. "
                             "If not provided, eval_loader = train_loader.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size (default: 5)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--lambda-force", type=float, default=0.0,
                        help="Weight for Tier A force loss (default: 0.0)")
    parser.add_argument("--lambda-fd-force", type=float, default=0.0,
                        help="Weight for Tier B finite-difference force loss (default: 0.0)")
    parser.add_argument("--force-loss-fn", type=str, default="mse",
                        choices=["mse", "smooth_l1"],
                        help="Force loss function (default: mse)")
    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Enable wandb logging")
    args = parser.parse_args()

    package_path = "/home/haotiant/Projects/CMU/dtmol/"
    date = time.strftime("%Y%m%d")
    model_folder = os.path.join(package_path, f"dtmol/models/bindingpose_{date}")
    DEVICE = args.device
    os.makedirs(model_folder, exist_ok=True)

    ##% Load the pretrained encoder
    pretrain_f = os.path.join(package_path, "dtmol/pretrain_models")
    nets, atom_dict = build_encoder(pretrain_f)

    ##% Load the decoder
    print("Loading the decoder")
    decoder_config = DummyModelConfig(mode="train")
    decoder = Decoder(decoder_config, atom_dict['ligand_dict'])
    decoder.register_diffusion_pool_head("tr-rotation", 6)
    decoder.register_diffusion_head("perturbation", 3)
    nets.update({"decoder": decoder})

    ##% Build config
    config = CONFIG(
        lambda_force=args.lambda_force,
        lambda_fd_force=args.lambda_fd_force,
        force_loss_fn=args.force_loss_fn,
        dataset_mode=args.dataset_mode,
        use_wandb=args.use_wandb,
    )

    if args.dataset_mode == "unified":
        # --- Unified dataset mode: UnifiedDataset + DatasetMixer ---
        from dtmol.data.mixer import DatasetMixer, get_mixed_dataloader
        from dtmol.data.unified_dataset import UnifiedDatasetConfig
        from dtmol.diffusion import (
            RotationSampler, GaussianSampler, TranslationSampler,
            ChainSampler, DummySampler, LogLinearScheduler, CosineScheduler,
        )

        if args.datamix is None:
            datamix_path = os.path.join(
                os.path.dirname(__file__), "data", "datamix_default.yaml"
            )
            print(f"No --datamix specified, using default: {datamix_path}")
        else:
            datamix_path = args.datamix

        # Build diffusion samplers (same as legacy mode)
        T = 1000
        ll_sch_tr = LogLinearScheduler(T, sigma_min=0.1, sigma_max=19.0)
        ll_sch_rot = LogLinearScheduler(T, sigma_min=0.1, sigma_max=1.65)
        ll_sch_pert = LogLinearScheduler(T, sigma_min=0.04, sigma_max=1.5)
        ll_sch_pert2 = LogLinearScheduler(T, sigma_min=0.04, sigma_max=1.5)
        rot_sampler = RotationSampler(schedular=ll_sch_rot, sde_format="VE")
        tr_sampler = TranslationSampler(schedular=ll_sch_tr, sde_format="VE")
        g_sampler = GaussianSampler(schedular=ll_sch_pert, sde_format="VE")
        g_sampler2 = GaussianSampler(schedular=ll_sch_pert2, sde_format="VE")
        molecule_sampler = ChainSampler(rot_sampler).compose(tr_sampler).compose(g_sampler)
        protein_sampler = ChainSampler(g_sampler2)
        protein_sampler.conjugate(molecule_sampler)

        ds_config = UnifiedDatasetConfig(
            max_seq_len=1000,
            max_pocket_atoms=256,
            seed=0,
        )

        mixer = DatasetMixer(
            datamix_path=datamix_path,
            ligand_dict=atom_dict['ligand_dict'],
            protein_dict=atom_dict['protein_dict'],
            config=ds_config,
            diffusion_samplers={"molecule": molecule_sampler, "protein": protein_sampler},
        )

        train_loader = get_mixed_dataloader(
            mixer, batch_size=args.batch_size, num_workers=0,
        )

        # Build validation loader
        if args.datamix_val is not None:
            val_mixer = DatasetMixer(
                datamix_path=args.datamix_val,
                ligand_dict=atom_dict['ligand_dict'],
                protein_dict=atom_dict['protein_dict'],
                config=ds_config,
                diffusion_samplers={"molecule": molecule_sampler, "protein": protein_sampler},
            )
            eval_loader = get_mixed_dataloader(
                val_mixer, batch_size=args.batch_size, num_workers=0,
            )
        else:
            logging.warning(
                "No --datamix-val provided; eval_loader = train_loader. "
                "Consider providing a separate validation datamix YAML."
            )
            eval_loader = train_loader

        trainer = DiffusionTrainer(
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            nets=nets,
            sampler={"molecule": molecule_sampler, "protein": protein_sampler},
            config=config,
            device=DEVICE,
        )
    else:
        # --- Legacy dataset mode: CrossDataset ---
        biding_ds_path = "/data/unimol_data/protein_ligand_binding_pose_prediction/"
        test_config = {
            "seed": 0,
            "max_seq_len": 1000,
            "max_pocket_atoms": 256,
        }
        binding_dataset = load_unimol_binding_data(test_config, biding_ds_path)
        loader_dict = get_dataloader(binding_dataset, batch_size=args.batch_size, device=DEVICE)

        trainer = DiffusionTrainer(
            train_dataloader=loader_dict['train'],
            eval_dataloader=loader_dict['valid'],
            nets=nets,
            sampler={"molecule": binding_dataset.mole_diffusion_sampler,
                     "protein": binding_dataset.protein_diffusion_sampler},
            config=config,
            device=DEVICE,
        )

    trainer.load_unimol_pretrain(pretrain_f)
    optimizer = torch.optim.Adam(trainer.nets['decoder'].parameters(), lr=args.lr)
    trainer.train(epoches=args.epochs, optimizer=optimizer, save_folder=model_folder)

    # Example: multi-dataset training with force loss
    # python dtmol_train_test.py --dataset-mode unified \
    #     --datamix /path/to/datamix.yaml \
    #     --lambda-force 0.1 --lambda-fd-force 0.05 \
    #     --force-loss-fn mse --batch-size 8 --epochs 50 \
    #     --device cuda --use-wandb