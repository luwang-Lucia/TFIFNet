import torch
import torch.nn.functional as F
from torch import nn
from models.MultiScaleAttMod import MSAM
from models.CSModel import CSModel


class FPN(nn.Module):
    def __init__(
            self, in_channels_list, out_channels, top_blocks=None  # 传入参数
    ):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        self.top_blocks = LastLevelP6P7(in_channels=256, out_channels=256)


    def _process_branch(self, x, inner_blocks, layer_blocks):
        c3, c4, c5 = x
        p5 = inner_blocks[2](c5)
        p4 = inner_blocks[1](c4) + F.interpolate(p5, scale_factor=2)
        p3 = inner_blocks[0](c3) + F.interpolate(p4, scale_factor=2)
        return p3, p4, p5

    def forward(self, x):
        p3, p4, p5 = self._process_branch(x, self.inner_blocks, self.layer_blocks)
        p6, p7 = self.top_blocks(p5)
        return tuple([p3, p4, p5, p6, p7])


class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
