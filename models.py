"""
Models used in the task.
"""

from torch import nn


class fMRIEncoder(nn.Module):
    """TODO: Implement the fMRI encoder.
    We will have more than one encoder for fMRI data. The name of this place holder does not suggest we only have one.
    We STARTS with an MLP encoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class ImageEncoder(nn.Module):
    """TODO: Implement the Image encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class CLIP(nn.Module):
    """TODO: Implement the CLIP classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass
