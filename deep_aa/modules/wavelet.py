import torch.nn as nn
import torch.nn.functional as F
from deep_aa.functional.wavelet import get_wavelet_filter_tensor, channel_shuffle1d, channel_shuffle2d, channel_shuffle3d


class DyadicWavelet(nn.Module):
    def __init__(self,
                 channels_in,
                 family,
                 dim=2,
                 step='dec',
                 filters='full',
                 no_resample=False,
                 flip=True,
                 padding='zero'):

        super().__init__()

        # Validate inputs
        if not (1 <= dim <= 3):
            raise ValueError(f"Convolutions of dimension {dim} not supported")

        if step not in ['dec', 'rec']:
            raise ValueError(f"Parameter 'step' must be one of 'dec' or 'rec'")

        # If we're resampling the convolution, then we need to use a transposed
        # conv2d, so we need to flip the filters (this is somewhat of an irritating
        # technicality)
        self.resample = not no_resample
        if self.resample and step == 'rec':
            flip = not flip

        # 'filters' will be validated by get_wavelet_filter_tensor()
        filter_tensor = get_wavelet_filter_tensor(family,
                                                  dim=dim,
                                                  step=step,
                                                  filters=filters,
                                                  flip=flip)
        self.register_buffer('firbank', filter_tensor)

        p_all = [[p//2, p//2 - (p+1) % 2] for p in filter_tensor.shape[-dim:]]

        # If we're flipping the filter (xor) doing the reconstruction
        # stage, we change the shape of the padding (in the even case of padding)
        if flip != (step == 'rec'):
            for idx in range(dim):
                p_all[idx][0], p_all[idx][1] = p_all[idx][1], p_all[idx][0]

        p_all = sum(p_all, [])
        self.padding = tuple(p_all)

        self.step = step
        self.dim = dim
        self.bands_out = filter_tensor.shape[0]
        self.channels_in = channels_in
        self.channels_out = channels_in * self.bands_out if step == 'dec' else channels_in // self.bands_out

    def pre_pad(self, x):
        padding = [0, self.padding[0]+self.padding[1], 0, self.padding[2]+self.padding[3]]
        return F.pad(x, padding, mode='reflect')


    def forward(self, x, no_pad=False):

        local_stride = 2 if self.resample else 1

        if self.step == 'dec':
            if not no_pad:
                x = F.pad(x, self.padding, mode='constant')
            if self.dim == 1:
                local_filter = self.firbank[:, None, ...].repeat(self.channels_in, 1, 1)
                x = F.conv1d(x, local_filter, groups=self.channels_in, stride=local_stride)
                x = channel_shuffle1d(x, self.channels_in)
            elif self.dim == 2:
                local_filter = self.firbank[:, None, ...].repeat(self.channels_in, 1, 1, 1)
                x = F.conv2d(x, local_filter, groups=self.channels_in, stride=local_stride)
                x = channel_shuffle2d(x, self.channels_in)
            elif self.dim == 3:
                local_filter = self.firbank[:, None, ...].repeat(self.channels_in, 1, 1, 1, 1)
                x = F.conv3d(x, local_filter, groups=self.channels_in, stride=local_stride)
                x = channel_shuffle3d(x, self.channels_in)

        else:
            groups = self.channels_in // self.bands_out

            if self.resample:

                if self.dim == 1:
                    local_filter = self.firbank[:, None, ...].repeat(groups, 1, 1)
                    x = channel_shuffle1d(x, self.channels_in // self.channels_out)
                    padding = [(p-2)//2 for p in local_filter.shape[-self.dim:]] if not no_pad else 0
                    x = F.conv_transpose1d(x,
                                           local_filter,
                                           groups=groups,
                                           padding=padding,
                                           stride=local_stride)
                elif self.dim == 2:
                    local_filter = self.firbank[:, None, ...].repeat(groups, 1, 1, 1)
                    x = channel_shuffle2d(x, self.channels_in // self.channels_out)
                    padding = [(p-2)//2 if not no_pad else p for p in local_filter.shape[-self.dim:]]
                    x = F.conv_transpose2d(x,
                                           local_filter,
                                           groups=groups,
                                           padding=padding,
                                           stride=local_stride)
                elif self.dim == 3:
                    local_filter = self.firbank[:, None, ...].repeat(groups, 1, 1, 1, 1)
                    x = channel_shuffle3d(x, self.channels_in // self.channels_out)
                    padding = [(p-2)//2 for p in local_filter.shape[-self.dim:]] if not no_pad else 0
                    x = F.conv_transpose3d(x,
                                           local_filter,
                                           groups=groups,
                                           padding=padding,
                                           stride=local_stride)

            else:
                local_filter = self.firbank[None, :, ...].repeat(groups, 1, 1, 1)
                if not no_pad:
                    x = F.pad(x, self.padding, mode='constant')

                if self.dim == 1:
                    x = F.conv1d(x, local_filter, groups=groups)
                elif self.dim == 2:
                    x = F.conv2d(x, local_filter, groups=groups)
                elif self.dim == 3:
                    x = F.conv3d(x, local_filter, groups=groups)
        return x

