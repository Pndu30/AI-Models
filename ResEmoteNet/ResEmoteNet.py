import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inch, outch, kernel=3, stride=1, padding=1, bias=False, relu=True, maxpool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=kernel, padding=padding, bias=bias, stride=stride))
        self.layers.append(nn.BatchNorm2d(outch))
        
        if relu:
            self.layers.append(nn.ReLU())
        if maxpool:
            self.layers.append(nn.MaxPool2d(kernel_size=2))
    
    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self, ch_list: list, n_blocks=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([])
        if len(ch_list) < n_blocks:
            raise("n_blocks can't be more than ch_list")
        i = 0
        for _ in range(n_blocks):
            block = Conv(inch=ch_list[i], outch=ch_list[i+1], relu=True, maxpool=True, kernel=3, padding=1)
            self.blocks.append(block)
            i += 1
    
    def forward(self, x):
        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            x = layer(x)
        return x

class SEBBlock(nn.Module):
    def __init__(self, inch, reduction=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(inch, inch // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(inch // reduction, inch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, inch, outch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv(inch=inch, outch=outch, maxpool=False, relu=True, stride=2, padding=1, kernel=3)
        self.conv2 = Conv(inch=outch, outch=outch, maxpool=False, relu=False, stride=1, padding=1, kernel=3)
        self.shortcut = Conv(inch=inch, outch=outch, maxpool=False, relu=False, stride=2, padding=0, kernel=1)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResEmoteNet(nn.Module):
    def __init__(self, inch, outch, softmax=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if inch == 3:
            ch_list = [3, 64, 128, 256]
        else:
            ch_list = [1, 64, 128, 256]
        self.softmax = softmax
        self.cnn = CNN(ch_list=ch_list, n_blocks=3)
        self.seb = SEBBlock(inch=256)

        self.res = nn.Sequential(
            ResBlock(inch=256, outch=512),
            ResBlock(inch=512, outch=1024),
            ResBlock(inch=1024, outch=2048),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, outch)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.seb(x)
        x = self.res(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        if self.softmax:
            x = F.softmax(x)
        return x
    
    
if __name__ == '__main__':
    x_temp = torch.rand(1, 3, 192, 192)
    model = ResEmoteNet(inch=3, outch=7)
    y_temp = model(x_temp)
    print(x_temp.shape)
    print(y_temp.shape)
