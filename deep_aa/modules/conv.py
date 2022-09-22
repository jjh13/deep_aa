import math
import torch
import torch.nn as nn
from deep_aa.modules.wavelet import DyadicWavelet


class SafeConv(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 kernel_size,
                 dilation=1,
                 stride=1):

        if not isinstance(dilation, int):
            raise ValueError("dilation parameter must be an integer!")

        if not isinstance(stride, int):
            raise ValueError("stride must be an integer!")
        super().__init__()

        self.pre_conv = int(math.floor(math.log2(dilation)))
        self.stride_convs = int(math.floor(math.log2(stride)))


class SafeConv1d(nn.Module):
    """
    A ``safe'' version of Conv1d.

    This version avoid aliasing
    """
    pass


class SafeConv2d(SafeConv):
    def __init__(self,
                 # Options for Conv2D
                 channels_in,
                 channels_out,
                 kernel_size,
                 dilation=1,
                 stride=1,
                 padding=0,
                 padding_mode='zero',
                 bias=True,
                 groups=1,

                 # Post conv AA activation
                 preact=None,

                 # AA options
                 family='db2',
                 aa_padding_mode='zero',
                 dual=False,
                 ):
        super().__init__(channels_in=channels_in,
                         channels_out=channels_out,
                         kernel_size=kernel_size,
                         dilation=dilation,
                         stride=stride)

        self.convs = []

        # AA before the conv for dilation
        for _ in range(self.pre_conv):
            self.convs += [
                DyadicWavelet(channels_in,
                              family,
                              dim=2,
                              step='dec' if not dual else 'rec',
                              filters=['LL'],
                              no_resample=True,
                              padding=aa_padding_mode)
            ]

        # Do the conv itself
        self.main_conv = nn.Conv2d(
            channels_in,
            channels_out,
            padding=padding,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            bias=bias,
            groups=groups
        )
        self.convs += [self.main_conv]

        # Apply an optional activation function
        if preact is not None:
            self.convs += [
                preact
            ]

        # Now do the safe down sampling operation
        for _ in range(self.stride_convs):
            self.convs += [
                DyadicWavelet(channels_out,
                              family,
                              dim=2,
                              step='dec' if not dual else 'rec',
                              filters=['LL'],
                              no_resample=False,
                              padding=aa_padding_mode)
            ]

        self.total_conv = nn.Sequential(
            *self.convs
        )

    def forward(self, x):
        return self.total_conv(x)


class SafeConv3d(nn.Module):
    pass


class SafeConv1dTranspose(nn.Module):
    pass



class SafeConv2dTranspose(SafeConv):
    def __init__(self,
                 # Options for Conv2D
                 channels_in,
                 channels_out,
                 kernel_size,
                 dilation=1,
                 stride=1,
                 padding=0,
                 padding_mode='zero',
                 bias=True,
                 groups=1,

                 # Post conv AA activation
                 preact=None,

                 # AA options
                 family='db2',
                 aa_padding_mode='zero',
                 dual=False,
                 ):
        super().__init__(channels_in=channels_in,
                         channels_out=channels_out,
                         kernel_size=kernel_size,
                         dilation=dilation,
                         stride=stride)

        self.convs = []

        # AA before the conv for dilation
        for _ in range(self.stride_convs):
            self.convs += [
                DyadicWavelet(channels_in,
                              family,
                              dim=2,
                              step='dec' if not dual else 'rec',
                              filters=['LL'],
                              no_resample=False,
                              padding=aa_padding_mode)
            ]

        # Now do the safe down sampling operation
        for _ in range(self.pre_conv):
            self.convs += [
                DyadicWavelet(channels_out,
                              family,
                              dim=2,
                              step='dec' if not dual else 'rec',
                              filters=['LL'],
                              no_resample=True,
                              padding=aa_padding_mode)
            ]

        # Do the conv itself
        self.main_conv = nn.ConvTranspose2d(
            channels_in,
            channels_out,
            padding=padding,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            bias=bias,
            groups=groups
        )
        self.convs += [self.main_conv]

        # Apply an optional activation function
        if preact is not None:
            self.convs += [
                preact
            ]

        self.total_conv = nn.Sequential(
            *self.convs
        )

    def forward(self, x):
        return self.total_conv(x)

class SafeConv3dTranspose(nn.Module):
    pass



