from typing import TypeAlias

import torch
import gymnasium as gym


Env: TypeAlias = gym.Env
Device: TypeAlias = torch.DeviceObjType | str | None
