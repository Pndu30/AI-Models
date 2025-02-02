import os
import yaml
import torch.nn as nn
from helper import Conv, HarDBlock, DWConvTransition, GlobalAveragePool
from config_dic import config_files
import torch.nn.functional as F
import torch

class HarDNet(nn.Module):
    def __init__(self, arch="68", act="relu", *args, **kwargs):
        super().__init__()

        config_path = os.path.join(
            os.getcwd(), "models", "configs", config_files[arch]
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
        self.layers.append(Conv(init_ch, first_ch[0], kernel=3, stride=2, bias=False))
        self.layers.append(
            Conv(first_ch[0], first_ch[1], kernel=second_kernel, padding=1)
        )

        if max_pool:
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
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

        return out_branch

    def get_layers(self):
        print(self.layers)

class MSCAM(nn.Module):
    def __init__(self, inch, act='relu', *args, **kwargs):
        super().__init__()
        self.top = nn.Sequential(
                                Conv(in_channel=inch, out_channel=inch, kernel=1, act=act),
                                nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=1)
                                )
        self.bot = nn.Sequential(
                                GlobalAveragePool(),
                                nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=inch, out_channels=inch, kernel_size=1)
        )

        self.comb = nn.Sigmoid()

    def forward(self, x):
        top_x = x.clone().detach()
        bot_x = x.clone().detach()

        top_x = self.top(top_x)
        bot_x = self.bot(bot_x)
        print(f'add {top_x.shape} {bot_x.shape}')
        temp = top_x + bot_x

        temp_out = self.comb(temp)
        out = x * temp_out
        print(f'mscam output shape {out.shape}')
        return out

class ASFF(nn.Module):
    def __init__(self, inch, level, arch='68'):
        super().__init__()
        config_path = os.path.join(
            os.getcwd(), "models", "configs", config_files[arch]
        )
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.dim = config.get("rfb")[0]
        self.inter_dim0 = self.dim[self.level]

        if level==0:
            self.stride_level_1 = Conv(inch, self.inter_dim, kernel=3, stride=2)
            self.stride_level_2 = Conv(inch, self.inter_dim, kernel=3, stride=2)
            self.expand = Conv(self.inter_dim, 1024, kernel=3, stride=1)
        elif level==1:
            self.compress_level_0 = Conv(inch, self.inter_dim, kernel=1, stride=1)
            self.stride_level_2 = Conv(inch, self.inter_dim, kernel=3, stride=2)
            self.expand = Conv(self.inter_dim, 512, kernel=3, stride=1)
        elif level==2:
            self.compress_level_0 = Conv(inch, self.inter_dim, kernel=1, stride=1)
            self.expand = Conv(self.inter_dim, 256, kernel=3, stride=1)

        compress_c = 8

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        return out

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
    
        temp = self.conv_upsample1(self.upsample(x1))
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

class Hardnet_CPS(nn.Module):
    def __init__(self, arch='68', act='relu',*args, **kwargs):
        super().__init__()
        config_path = os.path.join(
            os.getcwd(), "models", "configs", config_files[arch]
        )
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        ext0 = config.get("rfb")[0][0]
        ext1 = config.get("rfb")[0][1]
        ext2 = config.get("rfb")[0][2]

        self.hardnet = HarDNet(arch=arch, act=act)
        self.mscam0 = MSCAM(inch=ext0, act=act)
        self.mscam1 = MSCAM(inch=ext1, act=act)
        self.mscam2 = MSCAM(inch=ext2, act=act)
        self.asff1 = ASFF(inch=ext0, level=0, arch=arch)
        self.asff1 = ASFF(inch=ext1, level=1, arch=arch)
        self.asff1 = ASFF(inch=ext2, level=2, arch=arch)


    def forward(self, x):
        hardnet_out = self.hardnet(x)
        print(f'xs {[i.shape for i in hardnet_out]}')
        x0 = self.mscam0(hardnet_out[1])
        x1 = self.mscam1(hardnet_out[2])
        x2 = self.mscam2(hardnet_out[3])
        return x
    

if __name__ == '__main__':
    import torch
    awa = Hardnet_CPS(arch='68')
    # awa.get_layers()
    test = torch.rand(1, 1, 224, 224)
    out = awa(test)
    print(out.shape)