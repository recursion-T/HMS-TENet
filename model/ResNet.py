import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,conv=nn.Conv2d):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义残差网络
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks,conv=nn.Conv2d):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.in_channels=out_channels
        self.layer1 = self.make_layer(ResidualBlock, out_channels, num_blocks[0], 1,conv)
        self.layer2 = self.make_layer(ResidualBlock, out_channels * 2, num_blocks[1], 2,conv)
        self.layer3 = self.make_layer(ResidualBlock, out_channels*4, num_blocks[2], 2,conv)
        self.layer4 = self.make_layer(ResidualBlock, out_channels*8, num_blocks[3], 2,conv)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def make_layer(self, block, out_channels, num_blocks, stride,conv):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride,conv=conv))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out

