import numpy as np
import torch


def layer_init(
    layer: torch.nn.Module,
    std: float = np.sqrt(2),
    bias: float = 0.0,
):
    torch.nn.init.orthogonal_(tensor=layer.weight, gain=std)
    torch.nn.init.constant_(tensor=layer.bias, val=bias)
    return layer
