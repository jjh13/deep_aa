import torch
import torch.nn as nn

import unittest
from deep_aa.modules.bpunet import BPU_Recurse, BPUNet


class TestBPUNet(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]

        # if torch.backends.mps.is_available():
        #     self.devices += [torch.device('mps:0')]

        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]


    def test_bpunet(self):

        bpu = BPU_Recurse(64)
        print(bpu(torch.rand(1,64,256,256)).shape)

    def test_bpunet_full(self):

        bpu = BPUNet(3, 3, 64)
        print(bpu(torch.rand(1,3,256,256)).shape)