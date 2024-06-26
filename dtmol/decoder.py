from typing import Dict
from torch import nn
import torch
from dtmol.block import TransformerDecoderWithPair, DiffusionHead,DiffusionPoolHead, NonLinearHead, GaussianAttentionLayer, get_distance_matrix
import logging
logger = logging.getLogger(__name__)

def base_architecture(args):
    args.num_layers = getattr(args, "layers", 14)
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.divisor = getattr(args, "divisor", 16) # for e3nn layers (l = 3 spherical harmonics give 16 terms), attention heads should be divisible by this number
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.n_gaussian_basis = getattr(args, "n_gaussian_basis", 128)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.head_dropout = getattr(args, "head_dropout", 0.1)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.head_activate_fn = getattr(args, "head_activate_fn", "gelu")
    args.post_ln = getattr(args, "post_ln", False)
    args.rot_loss = getattr(args, "rotation_loss", -1.0)
    args.tr_loss = getattr(args, "translation_loss", -1.0)
    args.g_loss = getattr(args, "g_noise_loss", -1.0)
    args.max_diffusion_time = getattr(args, "max_diffusion_time", 5000)
    args.independent_se3_attention = getattr(args, "independent_se3_attention", True)
    args.update_distance_matrix = getattr(args, "update_distance_matrix", False)
    args.use_cross_product_update = getattr(args, "use_cross_product_update", False)

class Decoder(nn.Module):
    def __init__(self, config, dictionary) -> None:
        base_architecture(config)
        self.config = config
        n_edge_type = len(dictionary) * len(dictionary)
        super().__init__()
        self.padding_idx = dictionary.pad
        self.decoder = TransformerDecoderWithPair(
            encoder_layers=config.num_layers,
            embed_dim=config.embed_dim,
            ffn_embed_dim=config.ffn_embed_dim,
            attention_heads=config.attention_heads,
            divisor=config.divisor,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn,
            max_seq_len=config.max_seq_len,
            post_ln=config.post_ln,
            max_time = config.max_diffusion_time,
            independent_SE3_attention = config.independent_se3_attention,
            update_distance_matrix=config.update_distance_matrix,
            use_cross_product_update=config.use_cross_product_update,
        )
        self.gbf_proj = NonLinearHead(
            input_dim=config.n_gaussian_basis,
            out_dim=config.attention_heads,
            activation_fn=config.activation_fn
        )
        self.gbf = GaussianAttentionLayer(config.n_gaussian_basis, n_edge_type)
        self.diffusion_heads = nn.ModuleDict()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, 
                embd_molecule, 
                embd_protein,
                coor_molecule,
                coor_protein,
                timesteps,
                padding_molecule,
                padding_protein,
                attn_mole, 
                attn_protein, 
                cross_distance,
                cross_edges,
                diffusion_heads = None,
                ):
        """Decoder forwarding function that takes the concatenate embedding input 
        from a protein encoder and a molecule encoder, and take cross distnace matrix
        from the molecule and protein coordinates as the input to attention matrix
        Inpurt Args;
            embd_molecule: (batch, n_molecule, embd_dim) the embedding of the molecule from the molecule encoder
            embd_protein: (batch, n_protein, embd_dim) the embedding of the protein from the protein encoder.
            coor_molecule: (batch, n_molecule, 3) the coordinate matrix of the molecule.
            coor_protein: (batch, n_protein, 3) the coordinate matrix of the protein.
            time_steps: (batch) the time steps for the diffusion.
            padding_molecule: (batch, n_molecule) the padding mask for the molecule.
            padding_protein: (batch, n_protein) the padding mask for the protein.
            attn_mole: (batch, n_molecule, n_molecule, attention_heads) the attention (pair) matrix from the molecular model.
            attn_protein: (batch, n_protein, n_protein, attention_heads) the attention (pair) matrix from the protein model.
            cross_distance: (batch, n_molecule, n_protein) the distance matrix including the molecule and the protein.
            cross_edges: (batch, n_molecule, n_protein) the edge matrix including the molecule and the protein.
            diffusion_heads: name of diffusion heads to run.
        """
        full_embd = torch.cat([embd_molecule, embd_protein], dim=1)
        full_coor = torch.cat([coor_molecule, coor_protein], dim=1)
        n_molecule = embd_molecule.size(1)
        n_protein = embd_protein.size(1)
        with torch.no_grad():
            # calculate the mean but ignore the inf values
            inf_mask = torch.isinf(coor_molecule)
            coor_molecule[inf_mask] = torch.nan
            full_coor[:,0,:] = torch.nanmean(coor_molecule,dim=1) 
            # Set the first coordinate to the center of the ligand 
            # (which will be used tp calculate the system score later), as intuitively the translation is the linear acceleration of the center of mass of the ligand
            coor_molecule[torch.isnan(coor_molecule)] = torch.inf
            # Update the cross distance matrix
            mask = torch.isinf(cross_distance)
            full_distance = get_distance_matrix(full_coor, coor_protein)
            cross_distance = full_distance[:, :n_molecule, :]

        if padding_molecule is None:
            padding_molecule = torch.zeros(embd_molecule.size(0), embd_molecule.size(1),dtype = torch.bool).to(embd_molecule.device)
        if padding_protein is None:
            padding_protein = torch.zeros(embd_protein.size(0), embd_protein.size(1),dtype = torch.bool).to(embd_protein.device)
        full_padding = torch.cat([padding_molecule, padding_protein], dim=1)
        bsz = embd_molecule.size(0)
        n_full = cross_distance.size(1) + cross_distance.size(2)
        assert n_full == n_molecule + n_protein
        def get_cross_attn(attn_mole, attn_protein, cross_dist, cross_et):
            n_node = cross_dist.size(-1)
            gbf_feature = self.gbf(cross_dist, cross_et)
            gbf_result = self.gbf_proj(gbf_feature)
            cross_attn_bias = gbf_result
            cross_attn_bias = cross_attn_bias.permute(0, 3, 1, 2).contiguous() # [bsz, head, n_molecule, n_protein]
            graph_attn_bias = torch.zeros(bsz,self.config.attention_heads,n_full,n_full).to(cross_attn_bias.device)
            graph_attn_bias[:, :, :n_molecule, :n_molecule] = attn_mole.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias[:, :, n_molecule:, n_molecule:] = attn_protein.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias[:, :, :n_molecule, n_molecule:] = cross_attn_bias.clone()
            graph_attn_bias[:, :, n_molecule:, :n_molecule] = cross_attn_bias.permute(0, 1, 3, 2).contiguous().clone()
            graph_attn_bias = graph_attn_bias.view(-1, n_full, n_full) # [bsz*head, n_node, n_node]

            
            return graph_attn_bias
        full_attn = get_cross_attn(attn_mole, attn_protein, cross_distance, cross_edges)
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            x_norm,
            delta_decoder_pair_rep_norm,
            node_rep,
        ) = self.decoder(full_embd, full_coor, timesteps, padding_mask=full_padding, attn_mask=full_attn)
        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0
        if diffusion_heads is None:
            return decoder_rep, decoder_pair_rep, delta_decoder_pair_rep, x_norm, delta_decoder_pair_rep_norm,node_rep
        else:
            scores = {}
            for head in diffusion_heads:
                if head not in self.diffusion_heads:
                    raise ValueError(f"Head {head} not registered")
                if self.diffusion_heads[head].parity is not None:
                    parity_mask = self.decoder.parity_out == self.diffusion_heads[head].parity
                    node_rep_parity = node_rep[:,parity_mask]
                else:
                    node_rep_parity = node_rep
                scores[head] = self.diffusion_heads[head](decoder_rep,node_rep_parity)
            return scores, full_padding
        
    def register_diffusion_head(
        self, name, out_dim=None, hidden_dim=None,parity = None,
    ):
        """Register a classification head."""
        if name in self.diffusion_heads:
            prev_out_dim = self.diffusion_heads[name].out_proj.out_features
            prev_inner_dim = self.diffusion_heads[name].dense.out_features
            if out_dim != prev_out_dim or hidden_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with output dimesnion {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, out_dim, prev_out_dim, hidden_dim, prev_inner_dim
                    )
                )
        if parity is None:
            se3_dim = len(self.decoder.parity_out)
        else:
            #If partiy is specified, only output with same parity is used.
            se3_dim = sum(self.decoder.parity_out == parity)
        self.diffusion_heads[name] = DiffusionHead(
            input_dim=self.config.embed_dim,
            input_dim2 = se3_dim, #dimension of output node features of SE3_layer
            hidden_dim=hidden_dim or self.config.embed_dim,
            out_dim=out_dim,
            activation_fn=self.config.head_activate_fn,
            parity = parity,
        )

    def register_diffusion_pool_head(
        self, name, out_dim=None, hidden_dim=None,pool_dropout = 0.1, parity = None
    ):
        """Register a classification head."""
        if name in self.diffusion_heads:
            prev_out_dim = self.diffusion_heads[name].out_proj.out_features
            prev_inner_dim = self.diffusion_heads[name].dense.out_features
            if out_dim != prev_out_dim or hidden_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with output dimesnion {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, out_dim, prev_out_dim, hidden_dim, prev_inner_dim
                    )
                )
        if parity is None:
            se3_dim = len(self.decoder.parity_out)
        else:
            #If partiy is specified, only output with same parity is used.
            se3_dim = sum(self.decoder.parity_out == parity)
        self.diffusion_heads[name] = DiffusionPoolHead(
            input_dim=self.config.embed_dim,
            input_dim2 = se3_dim, #dimension of output node features of SE3_layer
            hidden_dim=hidden_dim or self.config.embed_dim,
            out_dim=out_dim,
            activation_fn=self.config.head_activate_fn,
            dropout = pool_dropout,
            parity = parity,
        )

if __name__ == "__main__":
    from dtmol.encoder import UniMolEncoder
    from dtmol.utils.dictionary import Dictionary
    from dtmol.utils.datasets import CrossDataset
    base_path = "/home/haotiant/Projects/CMU/dtmol/dtmol/"
    logger.info("Loading the dataset")
    ligand_dict = Dictionary.load(f"{base_path}/models/pretrain/unimol_molecule_dict.txt")
    protein_dict = Dictionary.load(f"{base_path}/models/pretrain/unimol_protein_dict.txt")
    ligand_dict.add_symbol("[MASK]", is_special=True)
    protein_dict.add_symbol("[MASK]", is_special=True)
    biding_ds_path = "/data/unimol_data/protein_ligand_binding_pose_prediction/"
    test_config = {
        "seed": 0,
        "max_seq_len": 500,
        "max_pocket_atoms": 500,
    }
    pocket_dataset = CrossDataset(test_config,ligand_dict,protein_dict)
    pocket_dataset.load_lmdb(biding_ds_path,"train")
    
    logger.info("Loading the pre-trained model")
    class TestArgs:
        def __init__(self):
            self.mode = "encode"
    test_encoder_args = TestArgs()
    ligand_encoder = UniMolEncoder(args = test_encoder_args, dictionary=ligand_dict)
    ligand_model_dict = torch.load(f"{base_path}/models/pretrain/unimol_molecule_pretrain.pt")
    ligand_encoder.load_state_dict(ligand_model_dict["model"],strict=False)
    protein_encoder = UniMolEncoder(args = test_encoder_args, dictionary=protein_dict)
    protein_model_dict = torch.load(f"{base_path}/models/pretrain/unimol_protein_pretrain.pt")
    protein_encoder.load_state_dict(protein_model_dict["model"],strict=False)

    logger.info("Loading the decoder")
    test_decoder_args = TestArgs()
    decoder = Decoder(test_decoder_args, ligand_dict)

    dataset = pocket_dataset
    def get_mole_input(batch):
        return {"src_tokens": batch['net_input']['mol_tokens'],
                "src_distance": batch['net_input']['mol_holo_distance'],
                "src_coord": batch['net_input']['mol_holo_coord'],
                "src_edge_type": batch['net_input']['mol_edge_type'],
        }
    def get_pocket_input(batch):
        return {"src_tokens": batch['net_input']['pocket_tokens'],
                "src_distance": batch['net_input']['pocket_distance'],
                "src_coord": batch['net_input']['pocket_holo_coord'],
                "src_edge_type": batch['net_input']['pocket_edge_type'],
        }
    from dtmol.dtmol_input import get_dataloader
    loader_dict = get_dataloader(dataset,
                                 batch_size = 5,
                                 device = 'cpu',
                                 split = ['train'],
                                 distributed= False)
    batch = next(iter(loader_dict['train']))
    trian_loader = loader_dict['train']
    mole_input = get_mole_input(batch)
    pocket_input = get_pocket_input(batch)
    (mole_embd, 
     mole_attn,
     mole_padding
    ) = ligand_encoder(**mole_input,features_only = True)
    (pocket_embd, 
     pocket_attn,
     pocket_padding
    ) = protein_encoder(**pocket_input,features_only = True)
    decoder.register_diffusion_pool_head("tr-rotation", 6)
    decoder.register_diffusion_head("perturbation", 3)
    t = torch.tensor([0.0])
    output,padding_mask = decoder(mole_embd, 
            pocket_embd, 
            t,
            mole_padding,
            pocket_padding,
            mole_attn, 
            pocket_attn, 
            batch['net_input']['cross_distance'],
            batch['net_input']['cross_edge_type'],
            diffusion_heads = ["tr-rotation","perturbation"],
    )
