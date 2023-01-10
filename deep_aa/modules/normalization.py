import torch
from torch import nn


class NormalizationWrapper2d(nn.Module):
    def __init__(self, channels, normalization):
        super(NormalizationWrapper2d, self).__init__()
        assert normalization in [None, 'bn', 'gn', 'in']
        if normalization is None:
            self.module = nn.Identity()
        elif normalization == 'bn':
            self.module = nn.BatchNorm2d(channels)
        elif normalization == 'in':
            self.module = nn.InstanceNorm2d(channels)
        elif normalization == 'gn':
            group_size = [group_size for group_size in [8,4,2,1] if channels % group_size == 0][0]
            self.module = nn.GroupNorm(channels//group_size, channels)

    def forward(self, x):
        return self.module(x)
