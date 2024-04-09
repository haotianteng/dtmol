import torch
import numbers
import importlib
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
try:
    import unicore_fused_layernorm
    import unicore_fused_layernorm_backward_gamma_beta
    HAS_FUSED_LAYER_NORM = 2
except:
    print("unicore_fused_layernorm is not installed corrected")
    try:
        from apex.normalization import FusedLayerNormAffineFunction, FusedLayerNormFunction
        HAS_FUSED_LAYER_NORM = 1
    except:
        print("apex fused layer norm is not installed correctly, apex will be disabled.")
        HAS_FUSED_LAYER_NORM = 0

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
    HAS_FUSED_LAYER_NORM = 0

class FusedLayerNormFastFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output, mean, invvar = unicore_fused_layernorm.forward(
        input, ctx.normalized_shape, weight, bias, ctx.eps)
    ctx.save_for_backward(input, weight, bias, mean, invvar)
    return output
  @staticmethod
  def backward(ctx, grad_output):
    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input = unicore_fused_layernorm.backward(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)
    grad_weight, grad_bias = unicore_fused_layernorm_backward_gamma_beta.backward(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)
    return grad_input, grad_weight, grad_bias, None, None

FUSED_LAYER_NORM_SUPPORT_DIM = set([64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048, 2560, 5120])

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
        def torch_layer_norm(input):
            return F.layer_norm(
                input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)
        def fused_layer_norm_unicore(input):
            assert elementwise_affine
            if input.is_cuda:
                return FusedLayerNormFastFunction.apply(
                    input, self.weight.type(input.dtype), self.bias.type(input.dtype), self.normalized_shape, self.eps)
            else:
                return F.layer_norm(
                    input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)
        def fused_layer_norm_apex(input):
            if elementwise_affine:
                return FusedLayerNormAffineFunction.apply(
                    input, self.weight, self.bias, self.normalized_shape,self.eps)
            else:
                return FusedLayerNormFunction.apply(
                    input, self.normalized_shape, self.eps)
        if HAS_FUSED_LAYER_NORM==2:
            self.func = fused_layer_norm_unicore
        elif HAS_FUSED_LAYER_NORM==1:
            self.func = fused_layer_norm_apex
        else:
            self.func = torch_layer_norm

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the timestep embeddings.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time_step_embd: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(time_step_embd)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x

if __name__ == "__main__":
    # NLP Example
    from dtmol.utils.time_embedding import get_timestep_embedding
    time = torch.arange(0, 100)
    embd_func = get_timestep_embedding('sinusoidal', 128, 5000)
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = nn.LayerNorm(embedding_dim)
    # Activate module
    layer_norm(embedding)
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = nn.LayerNorm([C, H, W])
    output = layer_norm(input)