import torch
import torch.nn as nn

import unittest
from deep_aa.modules.shuffle import WaveShuffle1d, WaveUnShuffle1d
from deep_aa.modules.shuffle import WaveShuffle2d, WaveUnShuffle2d
from deep_aa.modules.shuffle import WaveShuffle3d, WaveUnShuffle3d




class TestWaveletUpDown(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device('cpu')]

        if torch.backends.mps.is_available():
            self.devices += [torch.device('mps:0')]

        if torch.cuda.is_available():
            self.devices += [torch.device('cuda:0')]


    def test_1d(self):
        for device in self.devices:
            ran = torch.rand(1, 1, 8, device=device)
            ran = torch.nn.functional.pad(ran, pad=(10,10))

            wus = WaveUnShuffle1d(1).to(device)
            wsh = WaveShuffle1d(2).to(device)

            x = wsh(wus(ran))

            print(nn.functional.mse_loss(x, ran))
            print(x.shape)

    def test_2d(self):
        for device in self.devices:
            ran = torch.rand(1, 1, 8, 8, device=device)
            ran = torch.nn.functional.pad(ran, pad=(10,10,10,10))

            wus = WaveUnShuffle2d(1, family='db1').to(device)
            wsh = WaveShuffle2d(4, family='db1').to(device)

            x = wsh(wus(ran))

            print(nn.functional.mse_loss(x, ran))
            print(x.shape)

    def test_3d(self):

        ran = torch.rand(1, 1, 8, 8, 8)
        ran = torch.nn.functional.pad(ran, pad=(10,10, 10,10, 10,10))

        wus = WaveUnShuffle3d(1)
        wsh = WaveShuffle3d(8)

        x = wsh(wus(ran))

        print(nn.functional.mse_loss(x, ran))
        print(x.shape)