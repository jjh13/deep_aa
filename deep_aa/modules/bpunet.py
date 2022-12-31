import torch
import torch.nn as nn
from deep_aa.modules.conv import SafeConv2d
from deep_aa.modules.shuffle import WaveShuffle2d, WaveUnShuffle2d


class ConvNormRelu(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ConvNormRelu, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.GroupNorm(8,channels_out),
            # nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class ResConv(nn.Module):
    def __init__(self, channels_in, channels_mid, channels_out):
        super(ResConv, self).__init__()

        self.encode = ConvNormRelu(channels_in, channels_out)
        self.sqz = ConvNormRelu(channels_out, channels_mid)
        self.decode = ConvNormRelu(channels_mid, channels_out)

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
                 no_wave=False,
                 final_act=None):
        super(BPUNet, self).__init__()

        self.network = nn.Sequential(
            # Preamble
            nn.Conv2d(channels_in, first_layer_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(first_layer_channels),
            nn.GroupNorm(8, first_layer_channels),

            nn.LeakyReLU(inplace=True),

            # Do the recursive part
            BPU_Recurse(first_layer_channels, max_channels, depth=0, max_depth=max_depth, no_wave=no_wave),

            # Do the postamble
            nn.Conv2d(first_layer_channels, first_layer_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(first_layer_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(first_layer_channels, channels_out, kernel_size=1, padding=0),
            final_act if final_act is not None else nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)


class BPU_Recurse(nn.Module):
    def __init__(self,
                 channels_in,
                 max_channels=512,
                 depth=0,
                 max_depth=3,
                 no_wave=False):
        super().__init__()

        channels_down = min(channels_in * 2, max_channels)

        if no_wave:
            self.down = nn.Sequential(
                ResConv(channels_in, channels_down//2, channels_down),
                ResConv(channels_down, channels_down//2, channels_down),
                nn.Conv2d(channels_down,
                          channels_down*4,
                          stride=2,
                          kernel_size=(3,3),
                          padding=(1,1)
                )
            )
        else:
            self.down = nn.Sequential(
                ResConv(channels_in, channels_down//2, channels_down),
                ResConv(channels_down, channels_down//2, channels_down),
                WaveUnShuffle2d(channels_down)
            )

        if depth >= max_depth:
            self.middle = ResConv(channels_down, channels_down//2, channels_down)
        else:
            self.middle = BPU_Recurse(channels_down,
                                      max_channels=max_channels,
                                      depth=depth+1,
                                      max_depth=max_depth,
                                      no_wave=no_wave)

        if no_wave:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(channels_down*4,
                          channels_down,
                          stride=2,
                          kernel_size=(4,4),
                          padding=(1,1)
                ),
                ResConv(channels_down, channels_down//2, channels_in),
                ResConv(channels_in, channels_in//2, channels_in),
            )
        else:
            self.up = nn.Sequential(
                WaveShuffle2d(channels_down * 4),
                ResConv(channels_down, channels_down//2, channels_in),
                ResConv(channels_in, channels_in//2, channels_in),
            )

    def forward(self, x):
        ll, hl, lh, hh = self.down(x).chunk(4, dim=-3)
        llp = self.middle(ll)
        return self.up(torch.cat([llp, hl, lh, hh], dim=-3))


