import torch
import toml
import os
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import argparse
import time
import dtmol
from dtmol.dtmol_model import ScoreNetwork
from dtmol.utils.dictionary import Dictionary
from dtmol.utils.datasets import CrossDataset
from typing import Dict,Union
from dtmol_input import load_unimol_binding_data,get_dataloader,check_score_correlation
from dtmol.dtmol_train_base import Trainer,CONFIG
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.dtmol_model import DummyModelConfig
from dtmol.diffusion import RotationSampler, GaussianSampler
from dtmol.utils.sampling import reverse_sampling, rmsd
from dtmol.utils.arguments import parse_args, print_args
from dtmol.utils.train_monitor import plot_grad_flow
from dtmol.dtmol_init import PRETRAIN_FOLDER as pretrain_f
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

class DiffusionTrainer(Trainer):
    def __init__(self, train_dataloader: DataLoader,
                 nets: Dict[str, Union[UniMolEncoder, Decoder]],
                 sampler: Union[RotationSampler, GaussianSampler],
                 config: Union[dict],
                 device: Union[str,int] = None,
                 eval_dataloader: DataLoader = None,
                 distributed: bool = False):
        super().__init__(train_dataloader, nets, config, device, eval_dataloader, distributed)
        self.sampler = sampler

    def train(self, epoches: int, optimizer, save_every_n_steps: int = 100,
              valid_every_n_steps: int = 100, eval_every_n_epoches: int = 5 ,save_folder: str = None,
              schedular = None):
        self.save_folder = save_folder
        self._save_config()
        loss = 0.0
        for epoch_i in range(epoches):
            if self.distributed:
                self.train_ds.dataloader.sampler.set_epoch(epoch_i)
                self.eval_ds.dataloader.sampler.set_epoch(epoch_i)

            ### Evaluation
            if (epoch_i+1) % eval_every_n_epoches == 0:
                if self._on_main_rank():
                    msg = f"Epoch {epoch_i}: Evaluating the model"
                    self.logger.info(msg)
                mole_rmsds,prot_rmsds = [],[]
                for eval_i,eval_batch in enumerate(self.eval_ds):
                    mole_rmsd,mole_rmsd_baseline,prot_rmsd = self.eval_step(eval_batch)
                    mole_rmsd_mean, prot_rmsd_mean, mole_rmsd_baseline_mean = mole_rmsd.mean().item(),prot_rmsd.mean().item(),mole_rmsd_baseline.mean().item()
                    mole_rmsd_std, prot_rmsd_std, mole_rmsd_baseline_std = mole_rmsd.std().item(),prot_rmsd.std().item(), mole_rmsd_baseline.std().item()
                    mole_rmsds += mole_rmsd.tolist()
                    prot_rmsds += prot_rmsd.tolist()
                    msg = f"Eval {eval_i}/{len(self.eval_ds)}: Mole RMSD {mole_rmsd_mean:.2f} +- {mole_rmsd_std:.2f}, Original Mole RMSD {mole_rmsd_baseline_mean:.2f} +- {mole_rmsd_baseline_std:.2f}, Prot RMSD {prot_rmsd_mean:.2f} +- {prot_rmsd_std:.2f}"
                    self.logger.info(msg)
                mole_rmsd = np.mean(mole_rmsds)
                prot_rmsd = np.mean(prot_rmsds)
                if self._on_main_rank():
                    msg = f"Epoch {epoch_i}: mean mole RMSD {mole_rmsd:.4f}, mean prot RMSD {prot_rmsd:.4f}"
                    self.logger.info(msg)
                    if self.use_wandb:
                        wandb.log({"mean mole_rmsd": mole_rmsd,
                                   "mean prot_rmsd": prot_rmsd,
                                   "mole_rmsd": wandb.Histogram(np.array(mole_rmsds)),
                                   "prot_rmsd": wandb.Histogram(np.array(prot_rmsds)),
                                   "epoch": epoch_i,
                                   "global_step": self.global_step})
                self.epoch_save(epoch_i)

            ### Training
            if self.config.TRAIN['warmup'] is not None and epoch_i >= self.config.TRAIN['warmup']:
                if self.distributed:
                    for param in self.nets.module['ligand_encoder'].parameters():
                        param.requires_grad = True
                    for param in self.nets.module['protein_encoder'].parameters():
                        param.requires_grad = True
                else:
                    for param in self.nets['ligand_encoder'].parameters():
                        param.requires_grad = True
                    for param in self.nets['protein_encoder'].parameters():
                        param.requires_grad = True
            pbar = tqdm(enumerate(self.train_ds),total = len(self.train_ds),desc = f"Epoch {epoch_i}, loss {loss:.4f}")
            for i_step, batch in pbar:
                if self.config.DATASET['tr_sde'] == 'VE' and self.config.TRAIN['mode'] == "debug":
                    corrs,ts,check = check_score_correlation(batch, 
                                                             corr_threshold = 0.8, 
                                                             min_diffusion_time = 0.1 * self.config.DATASET['max_diffusion_time'])
                    if (not check) and self._on_main_rank():
                        self.logger.warning(f"Translation score and mean coordinates displacement has low correlation {min(corrs)} at batch {i_step}")
                loss = self.train_step(batch)
                pbar.set_description(f"Epoch {epoch_i}, loss {loss:.4f}")
                if torch.isnan(loss):
                    self._alert("NaN loss detected, skip this training step.",level = "warning")
                    continue
                optimizer.zero_grad()
                loss.backward()
                if self.config.TRAIN['grad_norm']:
                    nn.utils.clip_grad_norm_(self.nets.parameters(), self.config.TRAIN['grad_norm'])
                if self.config.TRAIN['mode'] == "debug":
                    plot_grad_flow(self.nets.named_parameters(),use_wandb = self.use_wandb)
                optimizer.step()
                if (i_step+1) % save_every_n_steps == 0:
                    self.save()
                if (i_step+1) % valid_every_n_steps == 0:
                    with torch.no_grad():
                        trrot_losses, pert_losses = [], []
                        for valid_i,valid_batch in enumerate(self.eval_ds):
                            trrot_loss,pert_loss = self.valid_step(valid_batch)
                            if trrot_loss is not None:
                                trrot_losses.append(trrot_loss.item())
                            if pert_loss is not None:
                                pert_losses.append(pert_loss.item())
                            if valid_i > self.config.TRAIN['valid_first_n']:
                                break
                        trrot_loss = np.mean(trrot_losses) if self.config.TRAIN['trrot_loss'] else 0.0
                        pert_loss = np.mean(pert_losses) if self.config.TRAIN['perturbation_loss'] else 0.0
                        if self._on_main_rank():
                            msg = f"Epoch {epoch_i}: Step {i_step}, train loss {loss:.4f}, valid trrot_loss {trrot_loss:.4f}, perturbation loss {pert_loss:.4f}"
                            self.logger.info(msg)
                            if self.use_wandb:
                                wandb.log({"epoch":epoch_i,
                                        "train_loss": loss, 
                                        "global_step": self.global_step})
                self.global_step += 1
            if schedular is not None:
                schedular.step()

    def loss(self, output, padding_mask, batch,norm_weighted = False, validation = False):
        if validation:
            reduction = "none"
        else:
            reduction = "mean"
        if self.distributed:
            losses = self.nets.module.diffusion_loss(output, 
                                                     padding_mask.clone(), 
                                                     batch['diffused'], 
                                                     norm_weighted=norm_weighted,
                                                     trrot_diffusion = self.config.TRAIN['trrot_loss'],
                                                     perturbation_diffusion = self.config.TRAIN['perturbation_loss'],
                                                     reduction = reduction)
        else:
            losses = self.nets.diffusion_loss(output, 
                                              padding_mask.clone(), 
                                              batch['diffused'],
                                              norm_weighted=norm_weighted,
                                              trrot_diffusion = self.config.TRAIN['trrot_loss'],
                                              perturbation_diffusion = self.config.TRAIN['perturbation_loss'],
                                              reduction = reduction)
        return losses

    def rmsd(self, coord, label, mole_padding, prot_padding):
        rmsd_mole, rmsd_prot = rmsd(coord, label, mole_padding, prot_padding)
        return rmsd_mole, rmsd_prot
    
    def train_step(self, batch):
        output, padding_mask = self.nets(batch)
        losses = self.loss(output, padding_mask, batch, norm_weighted=self.config.TRAIN['norm_weighted'])
        loss = sum([val for key,val in losses.items()])
        return loss

    def valid_step(self, batch):
        with torch.no_grad():
            output, padding_mask = self.nets(batch)
            losses = self.loss(output, padding_mask, batch,norm_weighted=False, validation = True)
            trrot_loss = losses['trrot_loss'] if self.config.TRAIN['trrot_loss'] else None #shape [batch_size, 2, 3]
            pert_loss = losses['perturbation_loss'] if self.config.TRAIN['perturbation_loss'] else None #shape [batch_size, N, 3]
            print(pert_loss.shape)
            pert_loss = pert_loss.mean() if pert_loss is not None else None
            rotation_loss = trrot_loss[:,0,:].mean() if trrot_loss is not None else None
            translation_loss = trrot_loss[:,1,:].mean() if trrot_loss is not None else None
            if self.use_wandb and self._on_main_rank():
                wandb.log({"valid_rotation_loss": rotation_loss,
                           "valid_translation_loss": translation_loss, 
                           "perturbation loss":pert_loss, "global_step": self.global_step})
        return trrot_loss, pert_loss

    def eval_step(self, batch):
        with torch.no_grad():
            if self.distributed:
                coord,mole_padding,prot_padding = self.nets.module.eval_once(batch, self.sampler, 
                                                                            T = self.config.TRAIN['max_reverse_diffusion_time'],
                                                                             stochastic=self.config.TRAIN['stochastic_reverse_sampling'])
            else:
                coord,mole_padding,prot_padding = self.nets.eval_once(batch, self.sampler, 
                                                                      T = self.config.TRAIN['max_reverse_diffusion_time'],
                                                                      stochastic=self.config.TRAIN['stochastic_reverse_sampling'])
            label = self.get_label_coord(batch)
            mole_rmsd,prot_rmsd = self.rmsd(coord, label, mole_padding, prot_padding)
            #calculate the original rmsd
            orig_coord = torch.cat([batch['net_input']['mol_src_coord'],batch['net_input']['pocket_src_coord']],dim=1)
            mole_rmsd_ori,prot_rmsd_ori = self.rmsd(orig_coord,label,mole_padding, prot_padding)
        return mole_rmsd, mole_rmsd_ori, prot_rmsd
    
    def get_label_coord(self, batch):
        label = torch.cat([batch['net_input']['mol_holo_coord'],batch['net_input']['pocket_holo_coord']],dim=1)
        return label

    def record_config(self,config):
        if self.use_wandb:
            wandb.config.update(config)

def worker(idx,world_size,args):
    distributed = world_size > 1
    train_config=  args['train']
    dataset_config = args['dataset']
    if args['train']['mode'] == "debug":
        args['train']['eval_every_n_epoches'] = 10 
    if distributed:
        dist.init_process_group(backend="nccl", rank=idx, world_size=world_size)
    package_path = dtmol.__path__[0]
    date = time.strftime("%Y%m%d")
    model_name = args['model_name']
    if args['train']['retrain'] is not None:
        model_folder = args['train']['retrain']
        if idx == 0:
            print(f"Retrain the model from {model_folder}")
    else:
        model_folder = os.path.join(package_path, f"models/{model_name}_{date}")
        if idx == 0:
            print(f"Train the model from scratch, save to {model_folder}")
    ds_path = args['data_f']

    #create the model folder
    config = CONFIG()
    config.TRAIN.update(train_config)
    config.TRAIN['model_folder'] = model_folder
    os.makedirs(model_folder, exist_ok=True)
    
    ##% Buildt the model
    dropout = args['model']['dropout']
    independent_se3_attention = args['model']['independent_se3_attention']
    update_distance_matrix = args['model']['update_distance_matrix']
    MODEL_NO_PRETRAIN = {'pretrain_folder': pretrain_f,
                'load_pretrain': False,
                'encoder': {'dropout':dropout,
                            'encoder_layers':1,
                            'emb_dropout':dropout,
                            'attention_dropout':dropout,
                            'activation_dropout':dropout,
                            'pooler_dropout':dropout,
                },
                'decoder': {'layers':8,
                            'embed_dim':512,
                            'ffn_embed_dim':1024,
                            'attention_heads':64,
                            'independent_se3_attention':independent_se3_attention,
                            'update_distance_matrix': update_distance_matrix}
                            }
    MODEL_S = {'pretrain_folder': pretrain_f,
                'load_pretrain': True,
                'encoder': {'dropout':dropout,
                            'emb_dropout':dropout,
                            'attention_dropout':dropout,
                            'activation_dropout':dropout,
                            'pooler_dropout':dropout,
                },
                'decoder': {'layers':8,
                            'embed_dim':512,
                            'ffn_embed_dim':1024,
                            'attention_heads':64,
                            'independent_se3_attention':independent_se3_attention,
                            'update_distance_matrix': update_distance_matrix}
                            }

    MODEL_L = {'pretrain_folder': pretrain_f,
                'load_pretrain': True,
                'encoder': {'dropout':dropout,
                            'emb_dropout':dropout,
                            'attention_dropout':dropout,
                            'activation_dropout':dropout,
                            'pooler_dropout':dropout,
                },
                'decoder': {'layers':16,
                            'embed_dim':512,
                            'ffn_embed_dim':2048,
                            'attention_heads':64,
                            'independent_se3_attention':independent_se3_attention,
                            'update_distance_matrix': update_distance_matrix}
                            }

    MODEL_XL = {'pretrain_folder': pretrain_f,
                'load_pretrain': True,
                'encoder': {'dropout':dropout,
                            'emb_dropout':dropout,
                            'attention_dropout':dropout,
                            'activation_dropout':dropout,
                            'pooler_dropout':dropout,
                },
                'decoder': {'layers':24,
                            'embed_dim':512,
                            'ffn_embed_dim':3072,
                            'attention_heads':64,
                            'independent_se3_attention':independent_se3_attention,
                            'update_distance_matrix': update_distance_matrix}
                            }
    if model_name.endswith("large"):
        config.MODEL = MODEL_L
    elif model_name.endswith("xl"):
        config.MODEL = MODEL_XL
    elif "no_pretrain" in model_name:
        print("Training model from scratch, pretrain model is disabled.")
        config.MODEL = MODEL_NO_PRETRAIN
        args['train']['fine_tune_pretrain'] = True
        args['train']['warmup'] = None
    else:
        config.MODEL= MODEL_S
    config.MODEL['max_diffusion_time'] = dataset_config['max_diffusion_time']
    net = ScoreNetwork(config.MODEL)
    if args['train']['fine_tune_pretrain'] and args['train']['warmup'] is None:
        for param in net['ligand_encoder'].parameters():
            param.requires_grad = True
        for param in net['protein_encoder'].parameters():
            param.requires_grad = True
    else:
        for param in net['ligand_encoder'].parameters():
            param.requires_grad = False
        for param in net['protein_encoder'].parameters():
            param.requires_grad = False
    net.to(idx)
    if distributed:
        net = DDP(net,device_ids=[idx],find_unused_parameters=True)
    config.DATASET = dataset_config
    binding_dataset = load_unimol_binding_data(config.DATASET,ds_path)
    loader_dict = get_dataloader(binding_dataset,
                                 batch_size = args['batch_size'],
                                 device = idx,
                                 distributed= distributed)
    
    # for batch in loader_dict['train']:
    #     check_score_correlation(batch)
    #     raise

    ##% Build the trainer
    if args['train']['mode'] == "debug":
        train_loader = loader_dict['test'] #Use a small test set to see if the model convergence
    else:
        train_loader = loader_dict['train']
    trainer = DiffusionTrainer(train_dataloader=train_loader,
                               eval_dataloader=loader_dict['valid'],
                               nets=net,
                               sampler = {"molecule": binding_dataset.mole_diffusion_sampler,
                                          "protein": binding_dataset.protein_diffusion_sampler},
                               config = config,
                               device = idx,
                               distributed = distributed)
    if args['train']['retrain']:
        trainer.load(model_folder)
    optimizer = optim.Adam(net.parameters(),lr = train_config['learning_rate'])
    warmup_scheduler = optim.lr_scheduler.ConstantLR(optimizer,
                                                           factor = config.TRAIN['start_lr_factor'],
                                                           total_iters=config.TRAIN['lr_warmup'])
    if config.TRAIN['lr_scheduler'] == "LinearLR":
        schedular = optim.lr_scheduler.LinearLR(optimizer, 
                                                      start_factor = config.TRAIN['start_lr_factor'],
                                                      total_iters = config.TRAIN['epoches'],
                                                      last_epoch=-1)
    elif config.TRAIN['lr_scheduler'] == "CosineAnnealingLR":
        schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max = config.TRAIN['epoches'],
                                                               eta_min = config.TRAIN['learning_rate'] * config.TRAIN['start_lr_factor'],
                                                               last_epoch=-1)
    elif config.TRAIN['lr_scheduler'] == "CosineAnnealingWarmRestarts":
        schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0 = 10,
                                                                         T_mult = 2,
                                                                         eta_min = config.TRAIN['learning_rate'] * config.TRAIN['start_lr_factor'],
                                                                         last_epoch=-1)
    else:
        raise ValueError(f"Unkown lr scheduler {config.TRAIN['lr_scheduler']}")
    schedular = optim.lr_scheduler.SequentialLR(optimizer,[warmup_scheduler,schedular],
                                                milestones=[config.TRAIN['lr_warmup']])
    trainer.train(epoches=train_config['epoches'],
                  optimizer=optimizer,
                  schedular=schedular,
                  valid_every_n_steps=train_config['report_every'],
                  eval_every_n_epoches=train_config['eval_every_n_epoches'],
                  save_folder=model_folder)

def main(args):
    world_size = args['world_size']
    if world_size > 1:
        mp.spawn(worker,
                 args=(world_size,args),
                 nprocs=world_size,
                 join=True)
    else:
        worker(0,world_size,args)

if __name__ == "__main__":  
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    args = parse_args()
    #print the args in a nice format
    print("#"*20+"Arguments"+"#"*20)
    print_args(args)
    print("#"*49)
    main(args)