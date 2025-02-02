import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from config_dic import config_files

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



class HarDNet(nn.Module):  # pragma: no cover
    def __init__(self, arch="68", act="relu", *args, **kwargs):
        super().__init__()

        config_path = os.path.join(
            os.getcwd(), "HarDNet", "configs", config_files[arch]
        )
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.arch = arch
        self.model_type = "2D"
        second_kernel = 3
        max_pool = True
        init_ch = 1
        first_ch = config.get("first_ch")[0]
        ch_list = config.get("ch_list")[0]
        gr = config.get("gr")[0]
        m = config.get("grmul")
        n_layers = config.get("n_layers")[0]
        downSamp = config.get("downSamp")[0]
        drop_rate = config.get("drop_rate")
        depthwise = config.get("depthwise")

        if depthwise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blocks = len(n_layers)
        self.layers = nn.ModuleList([])
        # self.start = nn.ModuleList([])

        # self.start.append(Conv(init_ch, first_ch[0], kernel=3, stride=2, bias=False))
        # self.start.append(Conv(first_ch[0], first_ch[1], kernel=second_kernel))
        self.layers.append(Conv(init_ch, first_ch[0], kernel=3, stride=2, bias=False))
        self.layers.append(
            Conv(first_ch[0], first_ch[1], kernel=second_kernel, padding=1)
        )

        if max_pool:
            # self.start.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            # self.start.append(DWConvTransition(first_ch[1], first_ch[1], stride=2))
            self.layers.append(DWConvTransition(first_ch[1], first_ch[1], stride=2))

        ch = first_ch[1]
        for i in range(blocks):
            block = HarDBlock(
                ch, gr[i], m, n_layers[i], padding=1, act=act, dwconv=depthwise
            )
            ch = block.get_out_ch()
            self.layers.append(block)

            if (i == (blocks - 1)) and (arch == "85"):
                self.layers.append(nn.Dropout(drop_rate))

            self.layers.append(Conv(ch, ch_list[i], act=act, kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.layers.append(DWConvTransition(ch, stride=2))

        ch = ch_list[blocks - 1]
        self.layers.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000),
            )
        )

    def forward(self, x):
        for i in range(len(self.start)):
            layer = self.start[i]
            x = layer(x)
        return x

    def get_layers(self):
        print(self.layers)
