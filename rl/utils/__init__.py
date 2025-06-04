import torch

from rl.utils.types import Device

DEVICE: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
