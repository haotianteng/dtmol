import torch
from torch import nn
from dtmol.decoder import Decoder
from dtmol.encoder import UniMolEncoder
from dtmol.utils.dictionary import Dictionary

class DummyModelConfig(object):
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

class ScoreNetwork(nn.ModuleDict):
    def __init__(self,config):
        super().__init__()
        self.config = config
        decoder_config = {} if "decoder" not in config else config["decoder"]
        dicts = self.build_encoder(config['pretrain_folder'])
        if self.config['load_pretrain']:
            self.load_unimol_pretrain(config['pretrain_folder'])
        decoder_config = DummyModelConfig(mode = "train",**decoder_config)
        decoder = Decoder(decoder_config, dicts['ligand_dict'])
        decoder.register_diffusion_pool_head("tr-rotation", 6)
        decoder.register_diffusion_head("perturbation", 3)
        self['decoder'] = decoder  

    def build_encoder(self, pretrain_f):
        ligand_dict = Dictionary.load(f"{pretrain_f}/unimol_molecule_dict.txt")
        protein_dict = Dictionary.load(f"{pretrain_f}/unimol_protein_dict.txt")
        ligand_dict.add_symbol("[MASK]", is_special=True)
        protein_dict.add_symbol("[MASK]", is_special=True)
        encoder_config = {} if "encoder" not in self.config else self.config["encoder"]
        encoder_config = DummyModelConfig(mode = "encode",**encoder_config)
        self['ligand_encoder'] = UniMolEncoder(args = encoder_config, dictionary=ligand_dict)
        self['protein_encoder'] = UniMolEncoder(args = encoder_config, dictionary=protein_dict)
        for param in self['ligand_encoder'].parameters():
            param.requires_grad = False
        for param in self['protein_encoder'].parameters():
            param.requires_grad = False
        return {"ligand_dict": ligand_dict, "protein_dict": protein_dict}


    def load_unimol_pretrain(self, pretrain_folder):
        ligand_model_dict = torch.load(f"{pretrain_folder}/unimol_molecule_pretrain.pt")
        protein_model_dict = torch.load(f"{pretrain_folder}/unimol_protein_pretrain.pt")
        self['ligand_encoder'].load_state_dict(ligand_model_dict["model"], strict=False)
        self['protein_encoder'].load_state_dict(protein_model_dict["model"], strict=False)

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
    
    def forward(self,batch):
        mole_input = self._get_mole_diffused(batch)
        pocket_input = self._get_pocket_diffused(batch)
        (mole_embd, mole_attn, mole_padding) = self['ligand_encoder'](**mole_input, features_only=True)
        (pocket_embd, pocket_attn, pocket_padding) = self['protein_encoder'](**pocket_input, features_only=True)
        mole_time = batch['diffused']['mol_diffuse_time']
        pocket_time = batch['diffused']['pocket_diffuse_time']
        assert torch.equal(mole_time, pocket_time), "Molecule and pocket diffusion time should be the same."
        output, padding_mask = self['decoder'](embd_molecule = mole_embd, 
                                               embd_protein = pocket_embd,
                                               timesteps = mole_time.squeeze(1), 
                                               padding_molecule = mole_padding, 
                                               padding_protein = pocket_padding,
                                               attn_mole = mole_attn, 
                                               attn_protein = pocket_attn, 
                                               cross_distance = batch['net_input']['cross_distance'],
                                               cross_edges = batch['net_input']['cross_edge_type'],
                                               diffusion_heads=["tr-rotation", "perturbation"])
        
        ##% debugging code for NaN loss
        decoder_inpt = (mole_embd, 
                        pocket_embd, 
                        mole_time.squeeze(1), 
                        mole_padding, 
                        pocket_padding, 
                        mole_attn, 
                        pocket_attn, 
                        batch['net_input']['cross_distance'], 
                        batch['net_input']['cross_edge_type'])
        for input in decoder_inpt:
            if torch.isnan(input).any():
                print("NaN detected in decoder input")
                print("input:", input)
                raise

        if torch.isnan(output['tr-rotation']).any() or torch.isnan(output['perturbation']).any():
            print("NaN detected in tr-rotation output")
            print("output['tr-rotation']:", output['tr-rotation'])
            raise
        ###

        return output, padding_mask

    def diffusion_loss(self, output, padding_mask, diffused_dict, atom_diffusion=False):
        losses = []
        mol_trrot_score = diffused_dict['mol_diffuse_trrot_score'][:,:2,:].to(torch.float32)
        mol_score = diffused_dict['mol_diffuse_perturb_score'].to(torch.float32)
        mol_norm = diffused_dict['mol_diffuse_perturb_norm'].to(torch.float32)
        mol_trrot_norm = diffused_dict['mol_diffuse_trrot_norm'][:,:2].to(torch.float32)
        pocket_score = diffused_dict['pocket_diffuse_score'].to(torch.float32)
        pocket_norm = diffused_dict['pocket_diffuse_norm'].to(torch.float32)
        perturbation_score = torch.cat([mol_score, pocket_score], axis=1)
        perturbation_norm = torch.cat([mol_norm, pocket_norm], axis=1)
        tr_rot = output['tr-rotation'].view(-1, 2, 3)  # [B,6] -> [B,2,3]
        pert = output['perturbation']
        trrot_loss = self['decoder'].diffusion_heads['tr-rotation'].loss(tr_rot, mol_trrot_score, mol_trrot_norm)

        losses.append(trrot_loss)
        pert_loss = self['decoder'].diffusion_heads['perturbation'].loss(pert, perturbation_score, perturbation_norm, padding_mask)
        losses.append(pert_loss)
        if atom_diffusion:
            raise NotImplementedError("Atom diffusion is not implemented yet.")
        return losses

if __name__ == "__main__":
    import os
    import time
    package_path = "/home/haotiant/Projects/CMU/dtmol/"
    date = time.strftime("%Y%m%d")
    model_folder = os.path.join(package_path, f"dtmol/models/bindingpose_{date}")
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"    
    ##% Buildt the model
    pretrain_f = os.path.join(package_path, "dtmol/models/pretrain")
    config = {"pretrain_folder": pretrain_f, "load_pretrain": True}
    net = ScoreNetwork(config)

    #testing pytorch DDP
    # import torch.distributed as dist
    # from torch.nn.parallel import DistributedDataParallel as DDP
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29500"
    # dist.init_process_group(backend="nccl", rank=0, world_size=2)
    # net = DDP(net,device_ids=[0],find_unused_parameters=True)
    # print("Calculating the size of the model")

    
    def count_parameters(module, all_params = True,dtype = torch.float32):
        dtype_size = {torch.float32: 4, torch.float16: 2}
        total_parameters = 0
        total_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad or all_params)
        print(f"Number of parameters in net {module._get_name()}: {total_parameters}")
        total_size = total_parameters * dtype_size[dtype] / 1024 / 1024 
        print(f"Size of the model: {total_size:.2f} MB")

    for key,module in net.items():
        count_parameters(module)
