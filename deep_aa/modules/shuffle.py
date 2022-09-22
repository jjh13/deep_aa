from deep_aa.modules.wavelet import DyadicWavelet


class WaveShuffle1d(DyadicWavelet):
    """
    WaveShuffle1d: An anti-aliased version of PixelShuffle.

    This performs a 2-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self,  channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=1,
            step='rec',
            filters='full',
            padding=padding
        )


class WaveShuffle2d(DyadicWavelet):
    """
    WaveShuffle2d: An anti-aliased version of PixelShuffle.

    This performs a 4-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self, channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=2,
            step='rec',
            filters='full',
            padding=padding
        )


class WaveShuffle3d(DyadicWavelet):
    """
    WaveShuffle3d: An anti-aliased version of VoxelShuffle.

    This performs a 4-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self,  channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=3,
            step='rec',
            filters='full',
            padding=padding
        )


class WaveUnShuffle1d(DyadicWavelet):
    """
    WaveShuffle1d: An anti-aliased version of PixelShuffle.

    This performs a 2-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self,  channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=1,
            step='dec',
            filters='full',
            padding=padding
        )


class WaveUnShuffle2d(DyadicWavelet):
    """
    WaveShuffle2d: An anti-aliased version of PixelShuffle.

    This performs a 4-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self, channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=2,
            step='dec',
            filters='full',
            padding=padding
        )


class WaveUnShuffle3d(DyadicWavelet):
    """
    WaveShuffle3d: An anti-aliased version of VoxelShuffle.

    This performs a 4-to-1 reduction in channels, treating half the channels as
    'high frequency', and the other half as low frequency.

    Technically, this the synthesis portion of an MRA, however, this operation
    bears a lot of resemblance to the PixelShuffle.
    """

    def __init__(self,  channels_in, family='db2', padding='zero'):
        super().__init__(
            channels_in,
            family,
            dim=3,
            step='dec',
            filters='full',
            padding=padding
        )
