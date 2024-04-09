import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period = 10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb
    
def get_timestep_embedding_func(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = TimestepEmbedder(embedding_dim, max_period=embedding_scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    T = 10000
    n_embd = 128
    sin_func = get_timestep_embedding_func('sinusoidal', n_embd, T)
    gf_func = get_timestep_embedding_func('fourier', n_embd, T)

    x = torch.arange(100)
    emb_sin = sin_func.timestep_embedding(x, n_embd, T)
    emb_gf = gf_func(x)

    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(emb_sin.T)
    axs[0].set_title("Sinusoidal Embedding")
    axs[1].imshow(emb_gf.T)
    axs[1].set_title("Gaussian Fourier Embedding")
    for ax in axs:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Embedding Dimension")
    plt.show()

    
