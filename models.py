"""
Models used in the task.
"""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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
    def __init__(self, input_features, output_features, *hidden_dims):
        """
        A simple multi-layer perceptron.

        :param input_features: The number of input features
        :type input_features: int
        :param output_features: The number of output features
        :type output_features: int
        :param hidden_dims: The number of neurons for each hidden layer
        :type hidden_dims: int
        """
        super().__init__()
        self.input_features, self.hidden_dims, self.output_features = (
            input_features,
            hidden_dims,
            output_features,
        )

        dims = (input_features, *hidden_dims, output_features)

        activation = nn.ReLU()

        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(dims, dims[1:]):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            linear_layers.append(activation)

        self.model = nn.Sequential(*linear_layers[:-1])

    def forward(self, x):
        return self.model(x)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional=True,
        num_layers=1,
    ):
        """
        A flexible Long Short-Term Memory (LSTM) model that can deal with packed sequences.

        For input as a tensor, do model(x); for input as a packed sequence,
        do model(x, packing) where packing is the lengths of the sequences.

        :param input_size: The number of expected features in the input x
        :type input_size: int
        :param hidden_size: The number of features in the hidden state h
        :type hidden_size: int
        :param bidirectional: If True, becomes a bidirectional LSTM
        :type bidirectional: bool
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        """
        super().__init__()

        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x, packing=None):
        if packing is not None:
            x = pack_padded_sequence(x, packing, batch_first=True, enforce_sorted=False)

        return self.model(x)
