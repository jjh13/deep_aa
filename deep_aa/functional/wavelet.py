import pywt
import torch
import itertools


def channel_shuffle1d(x: torch.Tensor, groups: int) -> torch.tensor:
    batchsize, num_channels, t = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, t)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, t)

    return x


def channel_shuffle2d(x: torch.Tensor, groups: int) -> torch.tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def channel_shuffle3d(x: torch.Tensor, groups: int) -> torch.tensor:
    batchsize, num_channels, height, width, depth = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width, depth)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width, depth)

    return x

def get_wavelet_filter_tensor(family, dim=2, step='dec', filters='full', flip=True, transpose=False):
    """
    This function takes a FIR wavelet family from PyWavelets and translates it into
    a PyTorch tensor. The 1D FIR filter is extended to {dim} dimensions via tensor product.

    :param family: The wavelet family to consider. See pywt.families() and the PyWavelet documentation.
    :param dim: The dimension over which to operate.
    :param step: 'dec' for decomposition (analysis or downsampling), 'rec' for reconstruction (synthesis
                  or upsampling)
    :param filters: Either 'full' for the full wavelet decomposition, or a list of strings indicating
                    which bands to extract.
    """
    if step not in ['dec', 'rec']:
        raise ValueError(f"Keyword argument 'step' must be either 'dec' or 'rec'")
    if filters == 'full':
        filters = [''.join(_) for _ in itertools.product(*[['L', 'H']] * dim)]

    if not isinstance(filters, list) or any([type(_) != str for _ in filters]):
        raise ValueError("Filters must be a list strings, i.e. HH, LL, HL etc... matching the dimension")

    for f in filters:
        for _ in f:
            if _ not in 'HL' or len(f) != dim:
                raise ValueError(
                    f"The filter '{f}' is invalid. It must have exactly {dim} characters, and contain only 'H' or 'L'")

    filters = sorted(filters, reverse=True)

    # Get the filter
    wavelet_filter = pywt.Wavelet(family)
    fdict = {}

    if transpose:
        step = 'dec' if step == 'rec' else 'rec'

    if step == 'dec':
        fdict['H'] = wavelet_filter.dec_hi
        fdict['L'] = wavelet_filter.dec_lo
    else:
        fdict['H'] = wavelet_filter.rec_hi
        fdict['L'] = wavelet_filter.rec_lo

    # Tensor product expand the filters
    phi = []
    for f in filters:
        q = fdict[f[0]]
        q = torch.tensor(q)
        if flip:
            q = q.flip(-1)
        q = q.reshape(*[len(q) if _ == 0 else 1 for _ in range(dim)])

        for fidx in range(1, dim):
            qp = fdict[f[fidx]]
            qp = torch.tensor(qp)
            if flip:
                qp = qp.flip(-1)
            qp = qp.reshape(*[len(qp) if _ == fidx else 1 for _ in range(dim)])
            q = q * qp

        phi += [q]
    return torch.stack(phi, dim=0)


get_wavelet_filter_tensor('db2', 2, step='dec', filters='full', flip=True)