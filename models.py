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


class ScanMLP(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.input_dims, self.hidden_dims, self.output_dims = (
            input_dims,
            hidden_dims,
            output_dims,
        )

        self.network = nn.Sequential(
            nn.Linear(input_dims[0], hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            # nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.network(x)
