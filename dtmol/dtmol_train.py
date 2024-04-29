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
from dtmol_input import load_unimol_binding_data,get_dataloader
from dtmol.dtmol_train_base import Trainer,CONFIG
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder
from dtmol.dtmol_model import DummyModelConfig
from dtmol.diffusion import RotationSampler, GaussianSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
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
              valid_every_n_steps: int = 100, save_folder: str = None):
        self.save_folder = save_folder
        self._save_config()
        for epoch_i in range(epoches):
            if self.distributed:
                self.train_ds.dataloader.sampler.set_epoch(epoch_i)
                self.eval_ds.dataloader.sampler.set_epoch(epoch_i)
            for i_step, batch in enumerate(self.train_ds):
                loss = self.train_step(batch)
                if torch.isnan(loss):
                    self._alert("NaN loss detected, skip this training step.",level = "warning")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i_step % save_every_n_steps == 0:
                    self.save()
                if i_step % valid_every_n_steps == 0:
                    with torch.no_grad():
                        trrot_losses, pert_losses = [], []
                        for valid_i,valid_batch in enumerate(self.eval_ds):
                            trrot_loss,pert_loss = self.valid_step(valid_batch)
                            trrot_losses.append(trrot_loss.item())
                            pert_losses.append(pert_loss.item())
                            if valid_i > self.config.TRAIN['valid_first_n']:
                                break
                        trrot_loss = np.mean(trrot_losses)
                        pert_loss = np.mean(pert_losses)
                        if self._on_main_rank():
                            msg = f"Epoch {epoch_i}: Step {i_step}, train loss {loss:.4f}, valid trrot_loss {trrot_loss:.4f}, perturbation loss {pert_loss:.4f}"
                            self.logger.info(msg)
                            if self.use_wandb:
                                wandb.log({"epoch":epoch_i,
                                        "train_loss": loss, 
                                        "global_step": self.global_step})
                self.global_step += 1

    def loss(self, output, padding_mask, batch,norm_weighted = False):
        if self.distributed:
            losses = self.nets.module.diffusion_loss(output, 
                                                     padding_mask.clone(), 
                                                     batch['diffused'], 
                                                     norm_weighted=norm_weighted)
        else:
            losses = self.nets.diffusion_loss(output, 
                                              padding_mask.clone(), 
                                              batch['diffused'],
                                              norm_weighted=norm_weighted)
        return losses

    def train_step(self, batch):
        output, padding_mask = self.nets(batch)
        loss = sum(self.loss(output, padding_mask, batch, norm_weighted=self.config.TRAIN['norm_weighted']))
        return loss

    def valid_step(self, batch):
        with torch.no_grad():
            output, padding_mask = self.nets(batch)
            trrot_loss, pert_loss = self.loss(output, padding_mask, batch,norm_weighted=False)
            if self.use_wandb and self._on_main_rank():
                wandb.log({"valid_trrot_loss": trrot_loss,"perturbation loss":pert_loss, "global_step": self.global_step})
        return trrot_loss, pert_loss

    def eval_step(self, batch):
        with torch.no_grad():
            output, padding_mask = self.nets(batch)
            loss = sum(self.loss(output, padding_mask, batch))
        return loss

    def record_config(self,config):
        if self.use_wandb:
            wandb.config.update(config)

def worker(idx,world_size,args):
    distributed = world_size > 1
    train_config=  args['train']
    if distributed:
        dist.init_process_group(backend="nccl", rank=idx, world_size=world_size)
    package_path = dtmol.__path__[0]
    date = time.strftime("%Y%m%d")
    model_folder = os.path.join(package_path, f"models/bindingpose_{date}")
    model_name = args['model_name']
    model_folder = os.path.join(package_path, f"models/{model_name}_{date}")
    ds_path = args['data_f']

    #create the model folder
    config = CONFIG()
    config.TRAIN.update(train_config)
    config.TRAIN['model_folder'] = model_folder
    os.makedirs(model_folder, exist_ok=True)
    
    ##% Buildt the model
    pretrain_f = os.path.join(package_path, "models/pretrain")
    config.MODEL={'pretrain_folder': pretrain_f,
                  'load_pretrain': True,
                  'decoder': {'layers':7,
                              'embed_dim':512,
                              'ffn_embed_dim':2048,
                              'attention_heads':64}
                              }
    net = ScoreNetwork(config.MODEL)
    if args['train']['fine_tune_pretrain']:
        for param in net['ligand_encoder'].parameters():
            param.requires_grad = True
        for param in net['protein_encoder'].parameters():
            param.requires_grad = True
    net.to(idx)
    if distributed:
        net = DDP(net,device_ids=[idx],find_unused_parameters=True)

    dataset_config = {
        "seed": 0,
        "max_seq_len": 768,
        "max_pocket_atoms": 256,
    }
    config.DATASET = dataset_config
    binding_dataset = load_unimol_binding_data(dataset_config,ds_path)
    loader_dict = get_dataloader(binding_dataset,
                                 batch_size = args['batch_size'],
                                 device = idx,
                                 distributed= distributed)

    ##% Build the trainer
    trainer = DiffusionTrainer(train_dataloader=loader_dict['train'],
                               eval_dataloader=loader_dict['valid'],
                               nets=net,
                               sampler = {"molecule": binding_dataset.mole_diffusion_sampler,
                                          "protein": binding_dataset.protein_diffusion_sampler},
                               config = config,
                               device = idx,
                               distributed = distributed)
    optimizer = torch.optim.Adam(net.parameters(),lr = train_config['learning_rate'])
    trainer.train(epoches=train_config['epoches'],
                  optimizer=optimizer,
                  valid_every_n_steps=train_config['report_every'],
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
    args = {
        'world_size': 2,
        'batch_size': 8,
        'model_name': "bindingpose",
        'data_f': "/data/unimol_data/protein_ligand_binding_pose_prediction/",
        'train':{
            'learning_rate':1e-4,
            'epoches': 100,
            'report_every': 10,
            'valid_first_n': 10,
            'fine_tune_pretrain': False,
            'norm_weighted': False, # if the perturbation loss is weighted by normalization factor
            'use_wandb': True,
        }
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--data_f",type=str,default = None)
    parser.add_argument("--world_size",type=int,default=None)
    parser.add_argument("--batch_size",type=int,default=None)
    parser.add_argument("--model_name",type=str,default=None)
    parser.add_argument("--fine_tune_pretrain",action='store_true',dest = 'fine_tune_pretrain')
    parser.add_argument("--no_wandb",action='store_false',dest='use_wandb')
    cmd_args = vars(parser.parse_args(sys.argv[1:]))
    print(cmd_args)
    #update the args with the parsed args if parser is not None
    for key in cmd_args:
        if cmd_args[key] is not None:
            if key in args:
                args[key] = cmd_args[key]
            elif key in args['train']:
                args['train'][key] = cmd_args[key]
            else:
                print('Warning: key {key} is not found in the args, will create a new key in the base-level of args.')
                args[key] = cmd_args[key]
    print(args)
    main(args)
