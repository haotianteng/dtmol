from typing import Dict

from pytorch import nn
from dtmol.block import TransformerEncoderWithPair

class Decoder(nn.module):
    def __init__(self, config:Dict) -> None:
        n_layers = config["decoder_layers"]
        embed_dim = config["decoder_embed_dim"]
        ffn_embed_dim = config["decoder_ffn_embed_dim"] 
        attention_heads = config["attention_heads"]
        dropout = config["dropout"]

        super().__init__()
        self.decoder = TransformerEncoderWithPair(
            encoder_layers=n_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            attention_heads=attention_heads,
            dropout=dropout,
        )

    def forward(self, 
                embd_molecule, 
                embd_protein, 
                attn_mole, 
                attn_protein, 
                cross_distance,
                cross_edges,
                ):
        """Decoder forwarding function that takes the concatenate embedding input 
        from a protein encoder and a molecule encoder, and take cross distnace matrix
        from the molecule and protein coordinates as the input to attention matrix
        Inpurt Args;
            embd_molecule: the embedding of the molecule from the molecule encoder
            embd_protein: the embedding of the protein from the protein encoder


        """
