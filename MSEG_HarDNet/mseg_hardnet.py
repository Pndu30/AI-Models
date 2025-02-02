import os
import yaml
import torch.nn as nn
from MSEG_HarDNet.helper import Conv, HarDBlock, DWConvTransition
from config_dic import config_files
import torch.nn.functional as F
import torch


class HarDNet(nn.Module):
    def __init__(self, arch="68", act="relu", *args, **kwargs):
        super().__init__()

        config_path = os.path.join(
            os.getcwd(), "src", "models", "configs", config_files[arch]
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
        out_branch = []

        # for i in range(len(self.start)):
        #     layer = self.start[i]
        #     x = layer(x)

        # for i in range(len(self.layers)):
        #     layer = self.layers[i]
        #     x = layer(x)
        #     if isinstance(layer, Conv):
        #         out_branch.append(x)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            if self.arch == "39DS":
                if i == 4 or i == 7 or i == 10 or i == 13:
                    out_branch.append(x)
            elif self.arch == "68":
                if i == 4 or i == 9 or i == 12 or i == 15:
                    out_branch.append(x)
            elif self.arch == "85":
                if i == 4 or i == 9 or i == 14 or i == 18:
                    out_branch.append(x)

        # out = x

        return out_branch

    def get_layers(self):
        print(self.layers)


class RFB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            Conv(in_ch, out_ch, kernel=1),
        )
        self.branch1 = nn.Sequential(
            Conv(in_ch, out_ch, kernel=1),
            Conv(out_ch, out_ch, kernel=(1, 3), padding=(0, 1)),
            Conv(out_ch, out_ch, kernel=(3, 1), padding=(1, 0)),
            Conv(out_ch, out_ch, kernel=3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            Conv(in_ch, out_ch, kernel=1),
            Conv(out_ch, out_ch, kernel=(1, 5), padding=(0, 2)),
            Conv(out_ch, out_ch, kernel=(5, 1), padding=(2, 0)),
            Conv(out_ch, out_ch, kernel=3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            Conv(in_ch, out_ch, kernel=1),
            Conv(out_ch, out_ch, kernel=(1, 7), padding=(0, 3)),
            Conv(out_ch, out_ch, kernel=(7, 1), padding=(3, 0)),
            Conv(out_ch, out_ch, kernel=3, padding=7, dilation=7),
        )
        self.conv_cat = Conv(4 * out_ch, out_ch, kernel=3, padding=1)
        self.conv_res = Conv(in_ch, out_ch, kernel=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Aggr(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = Conv(ch, ch, kernel=3, padding=1)
        self.conv_upsample2 = Conv(ch, ch, kernel=3, padding=1)
        self.conv_upsample3 = Conv(ch, ch, kernel=3, padding=1)
        self.conv_upsample4 = Conv(ch, ch, kernel=3, padding=1)
        self.conv_upsample5 = Conv(2 * ch, 2 * ch, kernel=3, padding=1)

        self.conv_concat2 = Conv(2 * ch, 2 * ch, kernel=3, padding=1)
        self.conv_concat3 = Conv(3 * ch, 3 * ch, kernel=3, padding=1)
        self.conv4 = Conv(3 * ch, 3 * ch, kernel=3, padding=1)
        self.conv5 = nn.Conv2d(3 * ch, 1, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        # print(x1.shape, x2.shape)
        temp = self.conv_upsample1(self.upsample(x1))
        # print(temp.shape)
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class HarDMSEG(nn.Module):
    # res2net based encoder decoder
    def __init__(self, act="relu", arch="68", channel=32):
        super().__init__()
        config_path = os.path.join(os.getcwd(), "src", "models", "configs", config_files[arch])
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        rfb1 = config.get("rfb")[0][0]
        rfb2 = config.get("rfb")[0][1]
        rfb3 = config.get("rfb")[0][2]
        # print(arch, rfb1, rfb2, rfb3)

        self.relu = nn.ReLU(True)
        self.rfb2_1 = RFB(rfb1, channel)
        self.rfb3_1 = RFB(rfb2, channel)
        self.rfb4_1 = RFB(rfb3, channel)
        self.agg1 = Aggr(ch=32)
        self.hardnet = HarDNet(arch=arch, act=act)
        self.model_type = "2D"

    def get_model_type(self):
        return self.model_type

    def forward(self, x):
        # print("input",x.size())

        hardnetout = self.hardnet(x)

        x1 = hardnetout[0]
        x2 = hardnetout[1]
        x3 = hardnetout[2]
        x4 = hardnetout[3]

        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)

        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=8, mode="bilinear"
        )  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5  # , lateral_map_4, lateral_map_3, lateral_map_2


# if __name__ == '__main__':
#     import torch
#     awa = HarDMSEG(arch='68')
#     # awa.get_layers()
#     test = torch.rand(1, 1, 224, 224)
#     out = awa(test)
#     print(out.shape)
