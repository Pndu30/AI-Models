import torch
from torch import nn
from torch.nn import functional as F
    
class PFN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
