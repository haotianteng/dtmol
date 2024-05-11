import torch
import logging
from torch import nn
from dtmol.utils.dictionary import Dictionary
from dtmol.block import GaussianLayer, TransformerEncoderWithPair, MaskLMHead, NonLinearHead, DistanceHead, ClassificationHead
logger = logging.getLogger(__name__)


def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

class UniMolEncoder(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
            post_ln=args.post_ln,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargsclass 
    ):  
        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        # if not padding_mask.any():
        #     padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() # [bsz, head, n_node, n_node]
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node) # [bsz*head, n_node, n_node]
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        # encoder_distance = None
        # encoder_coord = None

        # if not features_only:
        #     if self.args.masked_token_loss > 0:
        #         logits = self.lm_head(encoder_rep, encoder_masked_tokens)
        #     if self.args.masked_coord_loss > 0:
        #         coords_emb = src_coord
        #         if padding_mask is not None:
        #             atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
        #                 -1, 1, 1, 1
        #             )
        #         else:
        #             atom_num = src_coord.shape[1] - 1
        #         delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
        #         attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
        #         coord_update = delta_pos / atom_num * attn_probs
        #         coord_update = torch.sum(coord_update, dim=2)
        #         encoder_coord = coords_emb + coord_update
        #     if self.args.masked_dist_loss > 0:
        #         encoder_distance = self.dist_head(encoder_pair_rep)

        # if classification_head_name is not None:
        #     logits = self.classification_heads[classification_head_name](encoder_rep)
        # if self.args.mode == 'infer':
        #     return encoder_rep, encoder_pair_rep
        # elif self.args.mode == 'encode':
        #     return encoder_rep, encoder_pair_rep, padding_mask
        # else:
        #     return (
        #         logits,
        #         encoder_distance,
        #         encoder_coord,
        #         x_norm,
        #         delta_encoder_pair_rep_norm,
        #     )         
        return encoder_rep, encoder_pair_rep, padding_mask

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

if __name__ == "__main__":
    class TestArgs:
        def __init__(self):
            self.mode = "infer"
    test_args = TestArgs()
    #%% Load molecular model
    dict = Dictionary.load("/home/haotiant/dtmol/models/pretrain/unimol_molecule_dict.txt")
    mask_idx = dict.add_symbol("[MASK]", is_special=True) #Add mask token according to line 83 in tasks/unimol_conf_gen.py
    molecular_model = UniMolEncoder(args = test_args, dictionary=dict)
    model_dict = torch.load("/home/haotiant/dtmol/models/pretrain/unimol_molecule_pretrain.pt")
    molecular_model.load_state_dict(model_dict["model"],strict=False)

    #%% Load protein pocket model
    protein_dict = Dictionary.load("/home/haotiant/dtmol/models/pretrain/unimol_protein_dict.txt")
    mask_idx = protein_dict.add_symbol("[MASK]", is_special=True) #Add mask token according to line 83 in tasks/unimol_conf_gen.py
    protein_model = UniMolEncoder(args = test_args, dictionary=protein_dict)
    model_dict = torch.load("/home/haotiant/dtmol/models/pretrain/unimol_protein_pretrain.pt")
    protein_model.load_state_dict(model_dict["model"],strict=False)

    from dtmol.utils.datasets import MoleculeDataset, ProteinDataset
    #%% Load molecule data
    lmdb_path = "/data/unimol_data/conformation_generation/drugs/"
    test_config = {
        "seed": 0,
        "beta": 0.1, #beta for conformation importance sampling
        "smooth": 0.1, #smoothing for conformation importance sampling
        "topN": 10, 
        "max_seq_len": 100,
    }
    molecule_dataset = MoleculeDataset(dict,test_config)
    molecule_dataset.load_lmdb(lmdb_path,"train")

    #%% Test molecule model
    DEVICE = "cuda:0"
    batch = molecule_dataset.datasets["train"][0]
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            batch[k] = v.to(DEVICE)
    src_tokens = batch["net_input.src_tokens"].unsqueeze(0)
    src_coord = batch["net_input.src_coord"].unsqueeze(0)
    src_edge_type = batch["net_input.src_edge_type"].unsqueeze(0)
    src_distance = batch["net_input.src_distance"].unsqueeze(0)
    molecular_model.to(DEVICE)
    embd_out = molecular_model(src_tokens,src_distance,src_coord,src_edge_type)

    #%% Load protein pocket data
    protein_path = "/data/unimol_data/pockets/"
    test_config = {
        "seed": 0,
        "beta": 0.1,
        "smooth": 0.1,
        "topN": 10,
        "max_seq_len": 500,
        "dict_type": "dict_coarse.txt", #coarse or fine atom type used in the dictionary file
        "max_atoms": 256, #maximum number of atoms in a pocket
        "noise_type": "normal", #noise type in coordinate noise, can be "trunc_normal", "uniform", "normal", "none"
        "noise": 1.,#coordinate noise for masked atoms
        "mask_prob":0.15, #probability of replacing a token with mask
        "leave_unmasked_prob":0.1, #probability that a masked token is unmasked
        "random_token_prob":0.05, #probability of replacing a mask token with a random token
    }
    split = "train"
    pocket_dataset = ProteinDataset(protein_dict,test_config)
    pocket_dataset.load_lmdb(protein_path,'train')
    pocket_dataset.load_lmdb(protein_path,'valid')

    #%% Test protein model
    print(f"Total number of pockets: {len(pocket_dataset.datasets[split])}")
    batch = pocket_dataset.datasets[split][0]
    src_tokens = batch["net_input.src_tokens"].unsqueeze(0)
    src_coord = batch["net_input.src_coord"].unsqueeze(0)
    src_edge_type = batch["net_input.src_edge_type"].unsqueeze(0)
    src_distance = batch["net_input.src_distance"].unsqueeze(0)
    protein_model.eval()
    embd_out = protein_model(src_tokens,src_distance,src_coord,src_edge_type,features_only=True)
    print(embd_out[0].shape,embd_out[1].shape) #shape would be [batch_size,seq_len,embed_dim(512)] [batch_size,seq_len,seq_len,contact_embd_dim(64)]
    print(embd_out[0])
    #%% Test binding dataset
    test_config = {
    "seed": 0,
    "beta": 0.1,
    "smooth": 0.1,
    "topN": 10,
    "max_seq_len": 500,
    "dict_type": "coarse", #coarse or fine atom type used in the dictionary file
    "max_atoms": 512, #maximum number of atoms in a pocket
    "noise_type": "normal", #noise type in coordinate noise, can be "trunc_normal", "uniform", "normal", "none"
    "noise": 1.,#coordinate noise for masked atoms
    "mask_prob":0.15, #probability of replacing a token with mask
    "leave_unmasked_prob":0.1, #probability that a masked token is unmasked
    "random_token_prob":0.05, #probability of replacing a mask token with a random token
    }
    binding_dataset_f = "/data/unimol_data/protein_ligand_binding_pose_prediction/"
    binding_dataset = ProteinDataset(protein_dict,test_config)
    binding_dataset.load_lmdb(binding_dataset_f,"train")
    # binding_dataset.datasets["train"][0]
