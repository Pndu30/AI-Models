import torch
from torch import nn
from torch.nn import functional as F
from PFN import PFN
from FMG import FMG
from CN import HarDNetCN

class FMPN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
