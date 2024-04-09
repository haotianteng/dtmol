import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

ACT_FN = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": torch.tanh,
    "linear": torch.nn.Identity,
}

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    try:
        return ACT_FN[activation]
    except KeyError:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def is_debugging():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

if __name__ == "__main__":
    print(get_activation_fn("relu"))
    print(get_activation_fn("gelu"))
    print(get_activation_fn("tanh"))
    print(get_activation_fn("linear"))