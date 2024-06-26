import torch
import copy
from torch import nn
from dtmol.decoder import Decoder
from dtmol.encoder import UniMolEncoder
from dtmol.utils.sampling import reverse_sampling
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
        decoder.register_diffusion_pool_head("translation", 3, parity = -1)
        decoder.register_diffusion_pool_head("rotation", 3, parity = 1) #rotation vector is pseudovector
        decoder.register_diffusion_head("perturbation", 3, parity = -1) 
        self.pert_weight = 1. if "perturbation_weight" not in config else config["perturbation_weight"]
        self.rotation_weight = 1. if "rotation_weight" not in config else config["rotation_weight"]
        self.translation_weight = 1. if "translation_weight" not in config else config["translation_weight"]
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
    
    def _get_pocket_eval(self, batch):
        return {"src_tokens": batch['net_input']['pocket_tokens'],
                "src_distance": batch['net_input']['pocket_distance'],
                "src_coord": batch['net_input']['pocket_holo_coord'],
                "src_edge_type": batch['net_input']['pocket_edge_type']}
    
    def _get_mole_eval(self, batch):
        return {"src_tokens": batch['net_input']['mol_tokens'],
                "src_distance": batch['net_input']['mol_src_distance'],
                "src_coord": batch['net_input']['mol_src_coord'],
                "src_edge_type": batch['net_input']['mol_edge_type']}

    def get_diffusion_time(self,orig_T, current_T, t, scheduler = None):
        if scheduler is None:
            return int(t/current_T*orig_T)
        else:
            return scheduler.get_time(t, current_T, orig_T)

    def eval_once(self,batch,rev_sampler,T = 20, stochastic = False ,record_intermediate = False):
        #copy the batch
        batch = copy.deepcopy(batch)
        n_batch = batch['net_input']['mol_src_coord'].size(0)
        mole_sampler = rev_sampler['molecule']
        prot_sampler = rev_sampler['protein']
        orig_T = mole_sampler.T
        mole_sampler.set_T(T)
        prot_sampler.set_T(T)
        mol_coord = batch['net_input']['mol_src_coord']
        pocket_coord = batch['net_input']['pocket_holo_coord']
        n_mole = mol_coord.size(1)
        coord = torch.cat([mol_coord,pocket_coord],dim=1)
        if record_intermediate:
            ensembel = []
        ### debugging code ###
        # orig_coord = coord.clone()
        ######

        for i in range(T-1,-1,-1):
            t = self.get_diffusion_time(orig_T, T, i)
            batch['net_input']['mol_diffuse_time'] = torch.tensor([t]*n_batch,device=mol_coord.device).unsqueeze(1)
            batch['net_input']['pocket_diffuse_time'] = torch.tensor([t]*n_batch,device=pocket_coord.device).unsqueeze(1)
            score_dict, mole_padding, prot_padding = self.forward(batch, training=False)
            score = torch.cat([score_dict['rotation'].view(-1,1,3),score_dict['translation'].view(-1,1,3),score_dict['perturbation']],dim=1)
            coord[torch.isinf(coord)] = torch.nan #Fill the padding inf coordinates with nan to use nanmean.
            coord,distance = reverse_sampling(coord, 
                                              score, 
                                              mole_sampler=mole_sampler,
                                              prot_sampler=prot_sampler, 
                                              mole_padding=mole_padding, 
                                              prot_padding=prot_padding,
                                              t=i,
                                              stochastic = stochastic,)
            coord[torch.isnan(coord)] = torch.inf #reverse back the nan to inf
            batch['net_input']['mol_src_coord'] = coord[:,:n_mole,:]
            batch['net_input']['src_coord'] = coord[:,n_mole:,:] #change this to pocket_holo_coord to enable flexible pocket chain
            batch['net_input']['mol_src_distance'] = distance[:,:n_mole,:n_mole]
            batch['net_input']['pocket_distance'] = distance[:,n_mole:,n_mole:]
            batch['net_input']['cross_distance'] = distance[:,:n_mole,n_mole:]
            ### Debugging code 
            # import time
            # current_time = time.strftime("%Y%m%d_%H%M%S")
            # mole_diff = torch.norm(mol_coord.cpu()-batch['net_input']['mol_src_coord'].cpu())
            # prot_diff = torch.norm(pocket_coord.cpu()-batch['net_input']['src_coord'].cpu())
            # print(f"Time: {current_time}, Timestep: {i}, Molecule diff: {mole_diff}, Protein diff: {prot_diff}")
            ###
            if record_intermediate:
                ensembel.append(coord)
            else:
                ensembel = [coord]
        mole_sampler.set_T(orig_T)
        prot_sampler.set_T(orig_T)
        return ensembel, mole_padding, prot_padding

    def forward(self,batch,training = True):
        if training:
            mole_input = self._get_mole_diffused(batch)
            pocket_input = self._get_pocket_diffused(batch)
        else:
            mole_input = self._get_mole_eval(batch)
            pocket_input = self._get_pocket_eval(batch)
        (mole_embd, mole_attn, mole_padding) = self['ligand_encoder'](**mole_input, features_only=True)
        (pocket_embd, pocket_attn, pocket_padding) = self['protein_encoder'](**pocket_input, features_only=True)
        if training:
            mole_time = batch['diffused']['mol_diffuse_time']
            pocket_time = batch['diffused']['pocket_diffuse_time']
        else:
            mole_time = batch['net_input']['mol_diffuse_time']
            pocket_time = batch['net_input']['pocket_diffuse_time']
        assert torch.equal(mole_time, pocket_time), "Molecule and pocket diffusion time should be the same."
        if training:
            cross_dist, cross_edges = batch['diffused']['cross_distance'], batch['diffused']['cross_edge_type']
        else:
            cross_dist, cross_edges = batch['net_input']['cross_distance'], batch['net_input']['cross_edge_type']
        output, padding_mask = self['decoder'](embd_molecule = mole_embd, 
                                               embd_protein = pocket_embd,
                                               coor_molecule = mole_input['src_coord'],
                                               coor_protein = pocket_input['src_coord'],
                                               timesteps = mole_time.squeeze(1), 
                                               padding_molecule = mole_padding, 
                                               padding_protein = pocket_padding,
                                               attn_mole = mole_attn, 
                                               attn_protein = pocket_attn, 
                                               cross_distance = cross_dist,
                                               cross_edges = cross_edges,
                                               diffusion_heads=["rotation","translation", "perturbation"])
        
        # ##% debugging code for NaN loss
        # decoder_inpt = {"mole_embd":mole_embd, 
        #                 "pocket_embd":pocket_embd,
        #                 "mole_time":mole_time,
        #                 "mole_attn":mole_attn,
        #                 "pocket_attn":pocket_attn,
        #                 "cross_distance":cross_dist,
        #                 "cross_edges":cross_edges}
        
        # for key, input in decoder_inpt.items():
        #     if (input is None) or (torch.isnan(input).any()):
        #         print("NaN detected in decoder input")
        #         print(f"{key} input:", input)
        #         raise

        # if torch.isnan(output['tr-rotation']).any() or torch.isnan(output['perturbation']).any():
        #     print("NaN detected in tr-rotation output")
        #     print("output['tr-rotation']:", output['tr-rotation'])
        #     raise
        # ###
        if training:
            return output, padding_mask
        else:
            return output, mole_padding, pocket_padding

    def diffusion_loss(self, output, padding_mask, diffused_dict, 
                       perturbation_diffusion = True, 
                       trrot_diffusion = True, 
                       atom_diffusion=False, 
                       norm_weighted=False,
                       reduction = "mean"):
        if not(perturbation_diffusion) and not(trrot_diffusion) and not(atom_diffusion):
            raise ValueError("No diffusion loss has been enabled.")
        losses = {}
        mol_trrot_score = diffused_dict['mol_diffuse_trrot_score'][:,:2,:].to(torch.float32)
        mol_score = diffused_dict['mol_diffuse_perturb_score'].to(torch.float32)
        mol_norm = diffused_dict['mol_diffuse_perturb_norm'].to(torch.float32) 
        mol_trrot_norm = diffused_dict['mol_diffuse_trrot_norm'][:,:2].to(torch.float32)
        pocket_score = diffused_dict['pocket_diffuse_score'].to(torch.float32)
        pocket_norm = diffused_dict['pocket_diffuse_norm'].to(torch.float32)
        perturbation_score = torch.cat([mol_score, pocket_score], axis=1)
        perturbation_norm = torch.cat([mol_norm, pocket_norm], axis=1)
        rotation = output['rotation'].view(-1, 1, 3)  # [B,NX3] -> [B,N,3]
        translation = output['translation'].view(-1, 1, 3)  # [B,NX3] -> [B,N,3]
        pert = output['perturbation']
        if trrot_diffusion:
            rotation_loss = self['decoder'].diffusion_heads['rotation'].loss(rotation, 
                                                                          mol_trrot_score[:,0,:].unsqueeze(1), 
                                                                          norm = mol_trrot_norm[:,0].unsqueeze(1),
                                                                          norm_weighted = True,
                                                                          reduction = reduction)
            translation_loss = self['decoder'].diffusion_heads['translation'].loss(translation, 
                                                                                  mol_trrot_score[:,1,:].unsqueeze(1),
                                                                                  norm = mol_trrot_norm[:,1].unsqueeze(1),
                                                                                  norm_weighted = True,
                                                                                  reduction = reduction)
            losses['rotation_loss'] = self.rotation_weight*rotation_loss
            losses['translation_loss'] = self.translation_weight*translation_loss
        if perturbation_diffusion:
            padding_mask[:,0] = True # The first token <s> is reserved for classification task
            pert_loss = self['decoder'].diffusion_heads['perturbation'].loss(pert, 
                                                                            perturbation_score, 
                                                                            norm = perturbation_norm, 
                                                                            padding_mask = padding_mask, 
                                                                            norm_weighted = norm_weighted,
                                                                            reduction = reduction)
            losses["perturbation_loss"] = self.pert_weight*pert_loss
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
    pretrain_f = os.path.join(package_path, "dtmol/pretrain_models")
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
