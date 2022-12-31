import torch
import torch.nn as nn
from deep_aa.modules.conv import SafeConv2d
from deep_aa.modules.shuffle import WaveShuffle2d, WaveUnShuffle2d


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


class ConvNormRelu(nn.Module):
    def __init__(self, channels_in, channels_out, normalization):
        super(ConvNormRelu, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            NormalizationWrapper2d(channels_out, normalization),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class ResConv(nn.Module):
    def __init__(self, channels_in, channels_mid, channels_out, normalization):
        super(ResConv, self).__init__()

        self.encode = ConvNormRelu(channels_in, channels_out, normalization)
        self.sqz = ConvNormRelu(channels_out, channels_mid, normalization)
        self.decode = ConvNormRelu(channels_mid, channels_out, normalization)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(self.sqz(z)) + z


class BPUNet(nn.Module):
    def __init__(self,
                 channels_in=3,
                 channels_out=3,
                 first_layer_channels=64,
                 max_depth=3,
                 max_channels=512,
                 wavelet_family='db2',
                 final_act=None,
                 normalization='bn'):

        no_wave = wavelet_family is None
        super(BPUNet, self).__init__()
        self.inner_module = BPU_Recurse(first_layer_channels,
                                        max_channels,
                                        depth=0,
                                        max_depth=max_depth,
                                        wavelet_family=wavelet_family,
                                        normalization=normalization)

        self.network = nn.Sequential(
            # Preamble
            nn.Conv2d(channels_in, first_layer_channels, kernel_size=3, padding=1),
            NormalizationWrapper2d(first_layer_channels, normalization),
            nn.LeakyReLU(inplace=True),

            # Do the recursive part
            self.inner_module,

            # Do the postamble
            nn.Conv2d(first_layer_channels, first_layer_channels, kernel_size=3, padding=1),
            NormalizationWrapper2d(first_layer_channels, normalization),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(first_layer_channels, channels_out, kernel_size=1, padding=0),
            final_act if final_act is not None else nn.Sigmoid()
        )

    def innermost_module(self):
        return self.inner_module.innermost_module()

    def forward(self, x):
        return self.network(x)


class BPU_Recurse(nn.Module):
    def __init__(self, channels_in, max_channels=512, depth=0, max_depth=3, wavelet_family='db2', normalization=None):
        super().__init__()

        channels_down = min(channels_in * 2, max_channels)

        if wavelet_family is None:
            self.down = nn.Sequential(
                ResConv(channels_in, channels_down//2, channels_down, normalization=normalization),
                ResConv(channels_down, channels_down//2, channels_down, normalization=normalization),
                nn.Conv2d(channels_down,
                          channels_down*4,
                          stride=2,
                          kernel_size=(3,3),
                          padding=(1,1)
                )
            )
        else:
            self.down = nn.Sequential(
                ResConv(channels_in, channels_down//2, channels_down, normalization=normalization),
                ResConv(channels_down, channels_down//2, channels_down, normalization=normalization),
                WaveUnShuffle2d(channels_down, family=wavelet_family)
            )

        if depth >= max_depth:
            self.middle = ResConv(channels_down, channels_down//2, channels_down, normalization=normalization)
        else:
            self.middle = BPU_Recurse(channels_down,
                                      max_channels=max_channels,
                                      depth=depth+1,
                                      max_depth=max_depth,
                                      wavelet_family=wavelet_family,
                                      normalization=normalization)

        if wavelet_family is None:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(channels_down*4,
                                   channels_down,
                                   stride=2,
                                   kernel_size=(4, 4),
                                   padding=(1, 1)),
                ResConv(channels_down, channels_down//2, channels_in, normalization=normalization),
                ResConv(channels_in, channels_in//2, channels_in, normalization=normalization),
            )
        else:
            self.up = nn.Sequential(
                WaveShuffle2d(channels_down * 4, family=wavelet_family),
                ResConv(channels_down, channels_down//2, channels_in, normalization=normalization),
                ResConv(channels_in, channels_in//2, channels_in, normalization=normalization),
            )

    def innermost_module(self):
        if isinstance(self.middle, BPU_Recurse):
            return self.middle.innermost_module()
        return self.middle

    def forward(self, x):
        ll, hl, lh, hh = self.down(x).chunk(4, dim=-3)
        llp = self.middle(ll)
        return self.up(torch.cat([llp, hl, lh, hh], dim=-3))


