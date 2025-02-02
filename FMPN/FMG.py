import torch
from torch import nn
from torch.nn import functional as F
from helper import Conv

class DoubleConv(nn.Module):
    def __init__(self, inch, outch, midch=None, *args, **kwargs):
        super().__init__()
        if midch is not None:
            self.layers = nn.Sequential(
                Conv(in_channel=inch, out_channel=midch, kernel=3, padding=1, bias=False),
                Conv(in_channel=midch, out_channel=outch, kernel=3, padding=1, bias=False),
            )
        else:
            self.layers = nn.Sequential(
                Conv(in_channel=inch, out_channel=outch, kernel=3, padding=1, bias=False),
                Conv(in_channel=outch, out_channel=outch, kernel=3, padding=1, bias=False),
            )
        
    def forward(self, x):
        return self.layers(x)
    

class Down(nn.Module):
    def __init__(self, inch, outch, *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channel=inch, out_channel=outch)
        )

    def forward(self, x):
        return self.layers(x)
    
class Up(nn.Module):
    def __init__(self, inch, outch, bilinear=True, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        if bilinear:
            self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.layers.append(DoubleConv(in_channel=inch, out_channel=outch, midch=inch // 2))
        else:
            self.layers.append(nn.ConvTranspose2d(inch, inch // 2, kernel_size=2, stride=2))
            self.layers.append(DoubleConv(in_channel=inch, out_channel=outch, midch=inch // 2))

    def forward(self, x1, x2):
        # x1 = self.layers[0](x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        # return self.layers[1](x)
    
        x1 = self.layers[0](x1)
        x = torch.cat([x2, x1], dim=1)
        return self.layers[1](x)

class UnetFMG(nn.Module):
    def __init__(self, inch, *args, **kwargs):
        super().__init__()
        

    def forward(self, x):
        return x
