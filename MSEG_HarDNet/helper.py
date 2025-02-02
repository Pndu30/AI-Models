import torch
from torch import nn
from torch.nn import functional as F


class DWConvTransition(nn.Sequential):
    def __init__(self, in_channels, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.add_module(
            "dwconv",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
        )
        self.add_module("norm", nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return super().forward(x)


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        act="relu6",
        kernel=3,
        stride=1,
        bias=False,
        padding=0,
        dilation=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
                dilation=dilation,
            ),
        )

        self.add_module(name="bn", module=nn.BatchNorm2d(num_features=out_channel))
        if act == "relu":
            self.add_module(name="act", module=nn.ReLU())
        elif act == "leaky":
            self.add_module(name="act", module=nn.LeakyReLU())
        elif act == "relu6":
            self.add_module(name="act", module=nn.ReLU6(True))
        elif act == "tanh":
            self.add_module(name="act", module=nn.Tanh())
        else:
            print("Unknown activation function")

    def forward(self, x):
        return super().forward(x)


class CombConv(nn.Sequential):
    def __init__(
        self, in_channel, out_channel, act="relu6", kernel=1, stride=1, padding=0
    ):
        super().__init__()

        self.add_module(
            "conv",
            Conv(in_channel, out_channel, act=act, kernel=kernel, padding=padding),
        )
        self.add_module(
            "dwconv",
            DWConvTransition(out_channel, stride=stride),
        )

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        k,
        m,
        n_layers,
        act="relu",
        padding=0,
        dwconv=True,
        keepbase=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.links = []
        layers = []
        self.keepbase = keepbase
        self.out_channels = 0

        for i in range(n_layers):
            in_ch, out_ch, links = self.get_links(i + 1, in_channels, k, m)
            self.links.append(links)
            if dwconv:
                layers.append(CombConv(in_ch, out_ch, act=act, padding=padding))
            else:
                layers.append(Conv(in_ch, out_ch, act=act, padding=padding))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += out_ch

        self.layers = nn.ModuleList(layers)

    def get_links(self, layer, base_ch, k, m):
        if layer == 0:
            return 0, base_ch, []

        out_ch = k
        links = []

        for i in range(10):  # At most 2^10 layers check
            check = 2**i
            if layer % check == 0:
                link = layer - check
                links.append(link)
                if i > 0:
                    out_ch *= m

        out_ch = int(int(out_ch + 1) / 2) * 2
        in_ch = 0

        for j in links:
            _, ch, _ = self.get_links(j, base_ch, k, m)
            in_ch += ch
        return in_ch, out_ch, links

    def get_out_ch(self):
        return self.out_channels

    def forward(self, x):
        layers = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers.append(out)

        t = len(layers)
        out = []
        for i in range(t):
            if (self.keepbase and i == 0) or (i == t - 1) or (i % 2 == 1):
                out.append(layers[i])
        out = torch.cat(out, 1)
        return out
