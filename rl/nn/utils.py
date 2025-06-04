import numpy as np
import torch


def layer_init(
    layer: torch.nn.Module,
    weight_gain: float = np.sqrt(2),
    bias_const: float = 0.0,
):
    torch.nn.init.orthogonal_(tensor=layer.weight, gain=weight_gain)
    torch.nn.init.constant_(tensor=layer.bias, val=bias_const)
    return layer
