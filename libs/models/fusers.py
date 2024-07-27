import numpy as np
import math
import torch
import torch.nn.functional as F
from enum import Enum
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module



class ContentLayerDeepFast(nn.Module):
    def __init__(self, len=4, inp=1024, out=512):
        super().__init__()
        self.body = nn.ModuleList([])
        self.conv = nn.Conv2d(inp, out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        for i in range(len - 1):
            self.body.append(FeatureEncoderBlock(out, out))

    def forward(self, x):
        x = self.conv(x)
        for i, block in enumerate(self.body):
            x = x + block(x)
        return x


class FeatureEncoderBlock(nn.Module):
    def __init__(self, inp=1024, out=512):
        super().__init__()
        self.body = nn.Sequential(
                nn.BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Conv2d(inp, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.PReLU(num_parameters=out),
                nn.Conv2d(out, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    def forward(self, x):
        return self.body(x)