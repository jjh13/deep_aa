import torch
import torch.nn as nn

import unittest
from deep_aa.modules import SafeConv2d, SafeConv2dTranspose


class TestSafeConv(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]

        # if torch.backends.mps.is_available():
        #     self.devices += [torch.device('mps:0')]

        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]


    def test2d_stride2(self):
        sc = SafeConv2d(channels_in=128, channels_out=64, stride=2, kernel_size=3, padding=1)
        print(sc(torch.rand(2,128, 64,64)).shape)

    def test2d_stride2_di2(self):
        sc = SafeConv2d(channels_in=128, channels_out=64, stride=2, dilation=2, kernel_size=3, padding=1)
        print(sc(torch.rand(2,128, 64,64)).shape)

        nc = nn.Conv2d(128, 64, stride=2, dilation=2, kernel_size=3, padding=1)
        print(nc(torch.rand(2,128, 64,64)).shape)

