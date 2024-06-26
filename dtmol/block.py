import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dtmol.utils.time_embedding import get_timestep_embedding_func
from dtmol.utils.layer_norm import LayerNorm
from dtmol.utils.base import get_activation_fn
from dtmol.utils.attention import SelfMultiheadAttention
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def displacement(x:torch.tensor, y:torch.tensor = None, unit_vector:bool = True):
    """Get the displacement matrix (relative position) of given x and y
    Args:
        x: (batch, n, 3) the coordinate matrix of the first node.
        y: (batch, m, 3) the coordinate matrix of the second node.
    Returns:
        displacement: (batch, n, m, 3) the displacement matrix.
    """
    if y is None:
        y = x
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    displacement = x - y
    displacement[torch.isinf(displacement)] = 0 # fill -inf caused by (padding coordinates - normal coordinates) with 0
    displacement[torch.isnan(displacement)] = 0 # fill nan caused by (padding coordinates - padding coordinates) with 0        
    if unit_vector:
        displacement = displacement / (torch.norm(displacement, dim=-1, keepdim=True) + 1e-5)
    return displacement

def get_distance_matrix(x:torch.tensor, y:torch.tensor = None):
    """Get the distance matrix of given x and y
    Args:
        x: (batch, n, 3) the coordinate matrix of the first node.
        y: (batch, m, 3) the coordinate matrix of the second node.
    Returns:
        distance: (batch, n, m) the distance matrix.
    """
    if y is None:
        y = x
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    distance = x - y
    mask = torch.isnan(distance) | torch.isinf(distance)
    distance = distance.masked_fill(mask, 0)
    distance = torch.norm(distance, dim=-1)
    return distance

class Mlp(nn.Module):
    """ A simple 2 layer perceptron with GELU activation and dropout.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop,drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GaussianLayer(nn.Module):
    """This will calculate the gaussian kernal for the given coordinates.
    o_ijk = Gaussian(d_ij * mu(e_ij) + b(e_ij), mean_k, std_k) where e_ij is the edge type
    for distance d_ij between atom i and j.
    """
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        X here is the distance matrix batches with shape [B,N,N], where B
        is the batch size, N is the number of atoms in the molecule. edge_type
        is the edge type between any two atoms and have the same shape as X.
        Output:
        The output is a tensor with shape [B,N,N,K].
        """
        mul = self.mul(edge_type).type_as(x) # [B,N,N,1]
        bias = self.bias(edge_type).type_as(x) # [B,N,N,1]
        x = mul * x.unsqueeze(-1) + bias # [B,N,N,1]
        x = x.expand(-1, -1, -1, self.K) # [B,N,N,K]
        mean = self.means.weight.float().view(-1) # [K]
        std = self.stds.weight.float().view(-1).abs() + 1e-5 # [K]
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class GaussianAttentionLayer(nn.Module):
    """This will calculate the gaussian kernal for the given coordinates.
    o_ijk = Gaussian(d_ij * mu(e_ij) + b(e_ij), mean_k, std_k) where e_ij is the edge type
    for distance d_ij between atom i and j.
    """
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, K)
        self.scale = nn.Embedding(edge_types, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.scale.weight, 1)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        X here is the distance matrix batches with shape [B,N,N], where B
            is the batch size, N is the number of atoms in the molecule. 
        edge_type is a integer tensor the edge type between any two atoms and have shape [B,N,N].
        Output:
        The output is a tensor with shape [B,N,N,K]. Where K is the number of gaussian basis
            used.
        """
        mul = self.mul(edge_type).type_as(x) # [B,N,N,K]
        scale = self.scale(edge_type).type_as(x) # [B,N,N,K]
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.K) # [B,N,N,K]
        mean = self.means.weight.float().view(-1) # [K]
        std = self.stds.weight.float().view(-1).abs() + 1e-5 # [K]
        return scale * gaussian(mul*x.float(), mean, std).type_as(self.means.weight)


class TransformerLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)()
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

class DiTLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        independent_SE3_attention: bool = True,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.indp_attn = independent_SE3_attention
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        if self.indp_attn:
            self.disp_attn = SelfMultiheadAttention(
                self.embed_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )

        # DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=ffn_embed_dim, act_layer=self.activation_fn, drop=activation_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        Args:
            x: torch.Tensor, the input embedding tensor with shape [B, N, D].
            t: torch.Tensor, the timestep embedding tensor with shape [B, N, D].
            c: torch.Tensor, the coordinates tensor with shape [B, N, 3].
            attn_bias: torch.Tensor, the attention bias tensor with shape [B, N, N].
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
            if self.indp_attn:
                _,attn_weights_disp,attn_prob_disp = self.disp_attn(query=x,
                                                       key_padding_mask=padding_mask,
                                                       attn_bias=attn_bias,
                                                       return_attn=True)
            else:
                attn_weights_disp = attn_weights
                attn_prob_disp = attn_probs
        x = x + gate_msa.unsqueeze(1) * x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    
        if not return_attn:
            return x
        else:
            
            return x, attn_weights, attn_probs, attn_weights_disp, attn_prob_disp


class DiTFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class GlobalConv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out,heads,inter_hidden = None) -> None:
        super().__init__()

        tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        if inter_hidden is None:
            inter_hidden = heads*2
        self.fc = FullyConnectedNet([heads, inter_hidden, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_attr, atten, num_nodes_norm) -> torch.Tensor:
        weight = self.fc(atten)
        node_features = node_features.unsqueeze(1).expand(-1, edge_attr.size(1), -1, -1)
        edge_features = self.tp(node_features, edge_attr, weight)
        node_features = torch.sum(edge_features, dim=1)/num_nodes_norm
        return node_features

class SE3ELayer(nn.Module):
    """
    The SE(3)-equivariant layer.
        Args:
            heads: int, the number of attention heads.
            use_cross_product_update: bool, whether to use cross product to update the coordinate.
            update_distance_matrix: bool, whether to update the distance matrix in each layer when calculate attention matrix.
            max_l: int, default is 3, the maximum l value for the spherical harmonics.
    """
    def __init__(self,heads:int,
                 use_cross_product_update:bool,
                 update_distance_matrix:bool,
                 max_l:int = 3,
                 irreps_in = None,
                 irreps_out = None):
        super().__init__()
        self.attention_heads = heads
        self.use_cross_product_update = use_cross_product_update
        n_disp_heads = 2 * heads if use_cross_product_update else heads
        self.norm_attn = nn.LayerNorm(n_disp_heads, elementwise_affine=False, eps=1e-6)
        self.norm_disp = nn.LayerNorm(heads, elementwise_affine=False, eps=1e-6)
        self.disp_proj = nn.Linear(heads, heads)
        self.attn_proj = nn.Linear(heads, n_disp_heads)
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()
        self.update_distance_matrix = update_distance_matrix

        #e3nn layer
        assert heads % 16 == 0, "The number of heads must be divisible by 16."
        multiplier = heads // 16 
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_l) #irreductible representation of spherical harmonics
        if irreps_in is None:
            irreps_in = self.irreps_sh
        self.gate = Gate(
            f"{multiplier*2}x0e + {multiplier*2}x0o",
            [torch.relu, torch.abs],  # scalar
            f"{multiplier}x0e + {multiplier}x0o + {multiplier}x0e + {multiplier}x0o", 
            [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            f"{multiplier*2}x1o + {multiplier*2}x1e",  
        )
        self.irrpes_inter = self.gate.irreps_in
        if irreps_out is None:
            irreps_out = o3.Irreps(f"{multiplier}x1o + {multiplier}x1e")
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.conv1 = GlobalConv(self.irreps_in, self.irreps_sh, self.irrpes_inter, heads)
        self.conv2 = GlobalConv(self.gate.irreps_out, self.irreps_sh, self.irreps_out, heads)

        if update_distance_matrix:
            self.dist_update_proj = nn.Linear(heads, heads, bias=False)

    def forward(self, attn_disp, attn_mask, coordinates, node_features = None, edge_features = None):
        """
        Args:
            attn_disp: torch.Tensor, the attention displacement tensor with shape [B, H, N, N].
            attn_mask: torch.Tensor, the attention mask tensor with shape [B*H, N, N].
            coordinates: torch.Tensor, the coordinates tensor with shape [B, N, D].
            node_features: torch.Tensor, the node features tensor with shape [B, N, Z].
            edge_features: torch.Tensor, the edge features tensor with shape [B, N, N, Z].

        """
        bsz,seq_len,d = coordinates.size()
        if self.update_distance_matrix:
            old_distance_matrix = get_distance_matrix(coordinates.view(bsz*self.attention_heads, seq_len, d))
        normalizer = torch.sqrt(torch.sum(~torch.isinf(attn_disp[:,0,:,:]),axis = -1)) # number of non-inf values in each row [B, N]
        normalizer = normalizer.unsqueeze(-1) # [B, 1, N]
        # fill -inf with 0
        attn_disp = attn_disp.permute(0,2,3,1).contiguous() # [B, N, N, H]
        inf_mask = torch.isinf(attn_disp)
        attn_disp = attn_disp.masked_fill(inf_mask, 0)
        #project attn_mask with self.atten_proj
        
        
        # SE(3)-equivariant branch
        if edge_features is None:
            displacement_tensor = displacement(coordinates) # [bsz, seq_len, seq_len, d]
            edge_features = o3.spherical_harmonics(l=self.irreps_sh, x=displacement_tensor, normalize=True, normalization="component") # [bsz, seq_len, seq_len, (self.max_l+1)**2]
        if node_features is None:
            node_features = torch.sum(edge_features, dim=1)/normalizer
        node_features = self.conv1(node_features, edge_features, attn_disp, normalizer)
        node_features = self.gate(node_features)
        node_features = self.conv2(node_features, edge_features, attn_disp, normalizer)
        if self.update_distance_matrix:
            #TODO update coordinates according to node features
            raise NotImplementedError("Coordinates update hasn't been implmeneted yet.")
            # Check the response of https://github.com/e3nn/e3nn/discussions/439
            # coordinates = coordinates + displacement_tensor  # [bsz, head, seq_len, d]

            # update the attn_mask
            distance_matrix = get_distance_matrix(coordinates.view(bsz*self.attention_heads, seq_len, d)) # [bsz*head, seq_len, seq_len]
            delta_distance_matrix = distance_matrix - old_distance_matrix
            old_distance_matrix = distance_matrix
            delta_distance_matrix = delta_distance_matrix.view(bsz,-1,seq_len,seq_len).permute(0,2,3,1).contiguous() # [bsz, seq_len, seq_len, head]
            delta_distance_matrix = self.dist_update_proj(delta_distance_matrix)
            delta_distance_matrix = delta_distance_matrix.permute(0,3,1,2).contiguous() # [bsz, head, seq_len, seq_len]
            attn_mask = attn_mask +  delta_distance_matrix.view(-1,seq_len,seq_len)# d exp{(-Ax+b)^2} = -2(Ax+b) exp{(-Ax+b)^2} dx
        return attn_disp,attn_mask, coordinates, node_features, edge_features
        

class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()
        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        ) # [bsz, seq_len, seq_len, head]
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm

class TransformerDecoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 16,
        divisor: int = 16,
        independent_SE3_attention: bool = True,
        update_distance_matrix: bool = True,
        use_cross_product_update: bool = True,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        time_embedding_type = "sinusoidal",
        max_time: int = 5000,
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.t_embedder = get_timestep_embedding_func(time_embedding_type, embed_dim, embedding_scale = max_time**1.5)
        self.attention_heads = attention_heads
        self.use_cross_product_update = use_cross_product_update
        self.update_distance_matrix = update_distance_matrix
        multiplier = attention_heads // divisor
        num_odd_vec,num_even_vec = multiplier,multiplier
        irreps_out = [f"{num_odd_vec}x1o + {num_even_vec}x1e"]*encoder_layers
        irreps_out = [o3.Irreps(x) for x in irreps_out]
        irreps_in = [None] + irreps_out[:-1]
        #generate parity according to last irreps_out 
        #1o is odd vector with parity = -1 and 1e is even vector with parity = 1 (pseudo vector)
        self.parity_out = np.array([-1]*num_odd_vec + [1]*num_even_vec)
        self.se3_equiv_layers = nn.ModuleList(
            [
            SE3ELayer(attention_heads, 
                      use_cross_product_update=use_cross_product_update,
                      update_distance_matrix=update_distance_matrix,
                      irreps_in = irreps_in[i],
                      irreps_out = irreps_out[i],
                      )
            for i in range(encoder_layers)
            ]
        )
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer = DiTFinalLayer(embed_dim, embed_dim)
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None
        self.layers = nn.ModuleList(
            [
                DiTLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    independent_SE3_attention=independent_SE3_attention,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initliaze the Dit Blocks
        ## Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        ## Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        ## Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        emb: torch.Tensor,
        coordinates: torch.Tensor,
        t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        d = coordinates.size(-1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        t = self.t_embedder(t)
        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask
        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        # coordinates = coordinates.repeat(self.attention_heads, 1, 1).view(self.attention_heads, bsz, seq_len, d).transpose(0,1).contiguous() # [bsz, head, seq_len, d]
        for i in range(len(self.layers)):
            x, attn_mask,attn_prob,attn_disp,attn_disp_prob = self.layers[i](
                x,t, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
            # print("Attention mask shape: ", attn_mask.shape) # [bsz*head, seq_len, seq_len]
            # print("Attention prob shape: ", attn_prob.shape) # [bsz*head, seq_len, seq_len]
            # print("x shape: ", x.shape) # [bsz, seq_len, embed_dim]
            attn_disp = attn_disp.view(bsz, self.attention_heads, seq_len, seq_len).contiguous() # [bsz, head, seq_len, seq_len]
            attn_disp_prob = attn_disp_prob.view(bsz, self.attention_heads, seq_len, seq_len).contiguous() # [bsz, head, seq_len, seq_len]
            if i == 0:
                attn_disp,attn_mask, coordinates, node_features, edge_features = self.se3_equiv_layers[i](attn_disp_prob, attn_mask, coordinates)
            else:
                attn_disp,attn_mask, coordinates, node_features, edge_features = self.se3_equiv_layers[i](attn_disp_prob, attn_mask, coordinates, node_features, edge_features)
            
        node_features = node_features.reshape(bsz, seq_len, -1, 3).permute(0,2,1,3).contiguous() # [bsz, head, seq_len, 3]

        x = self.final_layer(x, t) # [bsz, seq_len, embed_dim]

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()
        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        ) # [bsz, seq_len, seq_len, head]
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm, node_features


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)()
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DiffusionHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        input_dim2,
        activation_fn,
        hidden_dim=None,
        coord_dim=3,
        parity = None,
    ):
        super().__init__()
        hidden_dim = input_dim if not hidden_dim else hidden_dim
        assert out_dim % coord_dim == 0, "Output dimension must be divisible by coord_dim"
        self.out_dim = out_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.x_gate = nn.Sigmoid()
        self.linear3 = nn.Linear(input_dim2, out_dim//coord_dim, bias = False)
        self.activation_fn = get_activation_fn(activation_fn)()
        self.layer_norm = LayerNorm(hidden_dim)
        self.mse_loss = nn.MSELoss(reduction="none")
        self.parity = parity

    def forward(self, x, y):
        """
        x: the output of the embedding with shape [B, N, D]
        y: the displacement tensor with shape [B, H, N, 3]
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.linear2(x) # [B, N, O]
        y = y.transpose(1, 2).transpose(2, 3) # [B, N, 3, H]
        y = self.linear3(y) # [B, N, 3, O/3]
        bsz, n, _, _ = y.size()
        y = y.reshape(bsz,n,self.out_dim) # [B, N, O]
        return self.x_gate(x)*y

    def loss(self, output, score, norm, 
             padding_mask = None, 
             norm_weighted = False,
             reduction = "mean"):
        loss = self.mse_loss(output, score)

        # ### Debugging code ###
        # print("Debugging output in DiffusionHead loss function")
        # print(f"output {output[0][:3]}")
        # print(f"score {score[0][:3]}") 
        # print(f"padding_mask {padding_mask[0][:3]}")
        # ######################

        norm = norm.unsqueeze(-1)
        mask = norm > 0
        norm[~mask] = 1
        if norm_weighted:
            loss = loss / norm
        if padding_mask is not None:
            mask = mask * (~padding_mask.unsqueeze(-1))
            loss = loss * (~padding_mask.unsqueeze(-1))
        loss = loss[mask.squeeze(-1)]
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class DiffusionPoolHead(nn.Module):
    """Head for system-level diffusion noise."""

    def __init__(
        self,
        input_dim,
        input_dim2,
        out_dim,
        activation_fn,
        hidden_dim=None,
        dropout = 0.1,
        coord_dim = 3,
        parity = None,
    ):
        super().__init__()
        hidden_dim = input_dim if not hidden_dim else hidden_dim
        assert out_dim % coord_dim == 0, "Output dimension must be divisible by coord_dim"
        self.out_dim = out_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.x_gate = nn.SiLU()
        self.out_proj2 = nn.Linear(input_dim2, out_dim//coord_dim, bias = False)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = get_activation_fn(activation_fn)()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.parity = parity

    def forward(self, x ,y):
        """
        x: the output of the embedding with shape [B, N, D]
        y: the displacement tensor with shape [B, H, N, 3]
        """
        bsz, n, d = x.size()
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        y = y.permute(0,2,3,1) # [B, N, 3, H]
        y = self.out_proj2(y) # [B, N, 3, O/3]
        y = y.permute(0,1,3,2) # [B, N, O/3, 3]
        y = y.reshape(bsz,n,self.out_dim) # [B, N, O]
        out = self.x_gate(x)*y # [B, N, O]
        return out.mean(dim=1) # [B, O]

    def loss(self, output, score, norm, norm_weighted = False,reduction = "mean"):
        loss = self.mse_loss(output, score)
        if norm_weighted:
            loss = loss / norm.unsqueeze(-1)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class DiffusionClassificationHead(nn.Module):
    """Head for system-level diffusion noise."""

    def __init__(
        self,
        input_dim,
        input_dim2,
        out_dim,
        activation_fn,
        hidden_dim=None,
        dropout = 0.1,
        coord_dim = 3,
        parity = None,
    ):
        super().__init__()
        hidden_dim = input_dim if not hidden_dim else hidden_dim
        assert out_dim % coord_dim == 0, "Output dimension must be divisible by coord_dim"
        self.out_dim = out_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.x_gate = nn.Sigmoid()
        self.out_proj2 = nn.Linear(input_dim2, out_dim//coord_dim, bias = False)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = get_activation_fn(activation_fn)()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.parity = parity

    def forward(self, x ,y):
        """
        x: the output of the embedding with shape [B, N, D]
        y: the displacement tensor with shape [B, H, N, 3]
        """
        x = x[:, 0, :]  # take <s> token (equiv. to [CLS])
        y = y[:,:,0,:] # [B, H, 3]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        y = y.transpose(1, 2) # [B, 3, H]
        y = self.out_proj2(y) # [B, 3, O/3]
        y = y.permute(0,2,1) # [B, O/3, 3]
        y = y.reshape(-1,self.out_dim) # [B, O]
        return self.x_gate(x)*y

    def loss(self, output, score, norm, norm_weighted = False,reduction = "mean"):
        loss = self.mse_loss(output, score)
        if norm_weighted:
            loss = loss / norm.unsqueeze(-1)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = get_activation_fn(activation_fn)()

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x
