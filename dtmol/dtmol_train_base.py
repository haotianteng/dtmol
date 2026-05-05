"""
Created on Thu Mar 11 16:07:12 2021

@author: Haotian Teng
"""
import os
import toml
import torch
import wandb
import logging
import itertools
import numpy as np
from typing import Union,Dict
from torch import nn
from torch.utils.data.dataloader import DataLoader
from dtmol.encoder import UniMolEncoder
from dtmol.decoder import Decoder

class CONFIG(object):
    def __init__(self,
                 project = "dtmol",
                 group = "docking",
                 experiment = "1-docking",
                 keep_record = 5,
                 device = "cuda",
                 grad_norm = None,
                 use_wandb = True,
                 lambda_force: float = 0.0,
                 lambda_fd_force: float = 0.0,
                 force_loss_fn: str = 'mse',
                 dataset_mode: str = 'legacy'):
        self.TRAIN = {"project":project,
                      "group":group,
                      "experiment":experiment,
                      "keep_record":keep_record,
                      "device":device,
                      "grad_norm":grad_norm,
                      "use_wandb":use_wandb,
                      "lambda_force":lambda_force,
                      "lambda_fd_force":lambda_fd_force,
                      "force_loss_fn":force_loss_fn,
                      "dataset_mode":dataset_mode}

class Trainer(object):
    def __init__(self,
                 train_dataloader:DataLoader,
                 nets:Union[Dict,nn.Module, nn.ModuleDict],
                 config:Union[dict],
                 device:str = None,
                 eval_dataloader:DataLoader = None,
                 distributed:bool = False):
        """

        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader.
        nets : Union[Dict, nn.ModuleDict]
            A CRNN or REVCNN network instance.
        device: str
            The device used to train the model, can be 'cpu' or 'cuda'.
            Default is None, use cuda device if it's available.
        config: dict
            A dictionary contains training configurations. Need to contain
            at least these parameters: keep_record, device and grad_norm.
        eval_dataloader : DataLoader, optional
            Evaluation dataloader, if None training dataloader will be used.
            The default is None.

        Returns
        -------
        None.

        """
        self.train_ds = train_dataloader
        self.device = self._get_device(device)
        self.config = config
        if eval_dataloader is None:
            self.eval_ds = self.train_ds
        else:
            self.eval_ds = eval_dataloader
        self.nets = nets
        self.global_step = 0
        self.save_list = []
        self.keep_record = config.TRAIN['keep_record']
        self.grad_norm = config.TRAIN['grad_norm']
        self.use_wandb = config.TRAIN.get('use_wandb',True)
        self.distributed = distributed
        if self.use_wandb and self._on_main_rank():
            wandb.init(project = config.TRAIN['project'],
                       group = config.TRAIN['group'],
                       name = config.TRAIN['experiment'])
        self._set_logger(getattr(config.TRAIN,'log_file',None))
        self.losses = []
        self.errors = []
    
    @property
    def _name(self):
        return getattr(self.config,'project','dtmol')

    def _alert(self,msg,level = 'info'):
        if level == 'info':
            self.logger.info(msg)
        elif level == 'error':
            self.logger.error(msg)
        elif level == 'warning':
            self.logger.warning(msg)

    def _record(self,dict):
        if self.use_wandb:
            self.logger.log(dict)
        self.logger.info(dict)

    @staticmethod
    def _group_key(net_name: str, param_name: str) -> str:
        """Bucket a parameter into a coarse group for wandb dashboards.

        Examples (decoder.* paths):
            "decoder.layers.7.self_attn.in_proj.weight" -> "decoder.layers"
            "decoder.se3_equiv_layers.3.conv1.tp.weight" -> "decoder.se3_equiv_layers"
            "decoder.se3_equiv_layers.3.batch_norm.weight" -> "decoder.se3_batchnorm"
            "decoder.diffusion_heads.tr-rotation.linear1.weight" -> "decoder.diffusion_heads"
            "decoder.gbf.means.weight" -> "decoder.other"
        """
        parts = param_name.split(".")
        if net_name != "decoder":
            return net_name
        if len(parts) >= 1 and parts[0] == "layers":
            return "decoder.layers"
        if len(parts) >= 2 and parts[0] == "se3_equiv_layers":
            if len(parts) >= 3 and parts[2] == "batch_norm":
                return "decoder.se3_batchnorm"
            return "decoder.se3_equiv_layers"
        if len(parts) >= 1 and parts[0] == "diffusion_heads":
            return "decoder.diffusion_heads"
        if len(parts) >= 1 and parts[0] == "decoder":
            # nested decoder.decoder.* (TransformerDecoderWithPair internals)
            return Trainer._group_key("decoder", ".".join(parts[1:]))
        return "decoder.other"

    def _param_norm_metrics(self, include_grads: bool = True) -> Dict[str, float]:
        """Collect parameter (and optionally gradient) L2 norms grouped by
        coarse module bucket. Returned dict is suitable for ``wandb.log``.

        Per group reported:
            param_norm/<group>           sqrt(sum w**2)  over all params in group
            param_absmax/<group>         max |w|
            grad_norm/<group>            sqrt(sum g**2)  (if include_grads)
            grad_absmax/<group>          max |g|         (if include_grads)
        Plus aggregates ``param_norm/total`` and ``grad_norm/total``.
        Non-finite values are reported as 'nan' so they show up in dashboards
        instead of silently breaking the log call.
        """
        if self.distributed:
            nets_dict = self.nets.module
        else:
            nets_dict = self.nets
        if isinstance(nets_dict, dict):
            nets_iter = list(nets_dict.items())
        else:
            nets_iter = [("model", nets_dict)]

        param_sq: Dict[str, float] = {}
        param_max: Dict[str, float] = {}
        grad_sq: Dict[str, float] = {}
        grad_max: Dict[str, float] = {}
        with torch.no_grad():
            for net_name, net in nets_iter:
                for pname, p in net.named_parameters():
                    if not p.requires_grad:
                        continue
                    g = self._group_key(net_name, pname)
                    val = p.detach()
                    sq = float(val.pow(2).sum().item())
                    mx = float(val.abs().max().item()) if val.numel() else 0.0
                    param_sq[g] = param_sq.get(g, 0.0) + sq
                    param_max[g] = max(param_max.get(g, 0.0), mx)
                    if include_grads and p.grad is not None:
                        gv = p.grad.detach()
                        gsq = float(gv.pow(2).sum().item())
                        gmx = float(gv.abs().max().item()) if gv.numel() else 0.0
                        grad_sq[g] = grad_sq.get(g, 0.0) + gsq
                        grad_max[g] = max(grad_max.get(g, 0.0), gmx)

        out: Dict[str, float] = {}
        total_p_sq = 0.0
        total_g_sq = 0.0
        for g, sq in param_sq.items():
            out[f"param_norm/{g}"] = float(sq ** 0.5)
            total_p_sq += sq
        for g, mx in param_max.items():
            out[f"param_absmax/{g}"] = mx
        for g, sq in grad_sq.items():
            out[f"grad_norm/{g}"] = float(sq ** 0.5)
            total_g_sq += sq
        for g, mx in grad_max.items():
            out[f"grad_absmax/{g}"] = mx
        out["param_norm/total"] = float(total_p_sq ** 0.5)
        out["param_absmax/total"] = max(param_max.values()) if param_max else 0.0
        if include_grads:
            out["grad_norm/total"] = float(total_g_sq ** 0.5)
            out["grad_absmax/total"] = max(grad_max.values()) if grad_max else 0.0
        return out

    def _on_main_rank(self):
        # torch.device("cuda") and torch.device("cuda:0") compare unequal
        # because the former has index=None. Match by type + index instead
        # so users passing --device cuda (without an index) still trigger
        # save() / wandb / config dump from the main rank.
        dev = torch.device(self.device)
        if dev.type == "cpu":
            return True
        if dev.type == "cuda" and (dev.index is None or dev.index == 0):
            return True
        return False

    def _set_logger(self,log_file = None):
        # Create a logger that log both to a log fiel and console
        logger = logging.getLogger(self._name)
        logger.setLevel(logging.DEBUG)

        # Create a file handler
        if log_file:
            file_handler = logging.FileHandler('logfile.log')
            file_handler.setLevel(logging.DEBUG)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Set the formatter for both handlers
        if log_file:
            file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        if log_file:
            logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger

    def reload_data(self,train_dataloader, eval_dataloader = None):
        self.train_ds = train_dataloader
        if eval_dataloader is None:
            self.eval_ds = train_dataloader
        else:
            self.eval_ds = eval_dataloader
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _update_records(self):
        record_file = os.path.join(self.save_folder,'records.toml')
        with open(record_file,'w+') as f:
            toml.dump(self.records,f)

    def epoch_save(self,epoch):
        if self._on_main_rank():
            current_ckpt = 'ckpt-'+str(self.global_step)+ f"-epoch{epoch}" +'.pt'
            model_file = os.path.join(self.save_folder,current_ckpt)
            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
            if os.path.isfile(model_file):
                os.remove(model_file)
            if self.distributed:
                net_dict = {key:net.state_dict() for key,net in self.nets.module.items()}
            else:
                net_dict = {key:net.state_dict() for key,net in self.nets.items()}
            torch.save(net_dict,model_file) 

    def save(self):
        if self._on_main_rank():
            ckpt_file = os.path.join(self.save_folder,'checkpoint')
            current_ckpt = 'ckpt-'+str(self.global_step)+'.pt'
            model_file = os.path.join(self.save_folder,current_ckpt)
            self.save_list.append(current_ckpt)
            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
            if len(self.save_list) > self.keep_record:
                os.remove(os.path.join(self.save_folder,self.save_list[0]))
                self.save_list = self.save_list[1:]
            if os.path.isfile(model_file):
                os.remove(model_file)
            with open(ckpt_file,'w+') as f:
                f.write("latest checkpoint:" + current_ckpt + '\n')
                for path in self.save_list:
                    f.write("checkpoint file:" + path + '\n')
                    f.write('\n')
            if self.distributed:
                net_dict = {key:net.state_dict() for key,net in self.nets.module.items()}
            else:
                net_dict = {key:net.state_dict() for key,net in self.nets.items()}
            torch.save(net_dict,model_file)
    
    def _save_config(self):
        if self._on_main_rank():
            config_file = os.path.join(self.save_folder,'config.toml')
            config_modules = [x for x in self.config.__dir__() if not x .startswith('_')][::-1]
            config_dict = {x:getattr(self.config,x) for x in config_modules}
            with open(config_file,'w+') as f:
                toml.dump(config_dict,f)
            if self.use_wandb:
                wandb.config.update(config_dict)

    def load(self,save_folder,update_global_step = True):
        self.save_folder = save_folder
        ckpt_file = os.path.join(save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            if update_global_step:
                if latest_ckpt.endswith('.pt'):
                    self.global_step = int(latest_ckpt[:-3].split('-')[1])
                else:
                    self.global_step = int(latest_ckpt.split('-')[1])  
        ckpt = torch.load(os.path.join(save_folder,latest_ckpt),map_location=self.device)
        nets = self.nets if not self.distributed else self.nets.module
        for key,net in ckpt.items():
            if key in nets.keys():
                try:
                    nets[key].load_state_dict(net,strict = True)
                except RuntimeError:
                    print(f"Exact loading {key} failed, try load loosely.")
                    nets[key].load_state_dict(net,strict = False)
                nets[key].to(self.device)
            else:
                msg = "%s net is defined in the checkpoint but is not imported because it's not defined in the model."%(key)
                self._alert(msg, level = 'warning')

def load_config(config_file):
    class CONFIG(object):
        pass
    with open(config_file,'r') as f:
        config_dict = toml.load(f)
    config = CONFIG()
    for k,v in config_dict.items():
        setattr(config,k,v)
    return config