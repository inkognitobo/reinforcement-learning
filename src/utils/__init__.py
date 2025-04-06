import torch

from src.utils.types import Device

DEVICE: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
