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


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, *hidden_dims):
        super().__init__()
        self.input_dim, self.hidden_dims, self.output_dim = (
            input_dim,
            hidden_dims,
            output_dim,
        )

        dims = (input_dim, *hidden_dims, output_dim)

        activation = nn.ReLU()

        linear_layers = []

        for layer_input_dim, layer_output_dim in zip(dims, dims[1:]):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            linear_layers.append(activation)

        self.model = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.model(x)
