import torch.nn as nn


class MLP(nn.Module):
    """ Multi Layer Perceptron module, inheriting from torch.nn. """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        """
        Initialize an instance of MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim: list of int, dimensions of the hidden layers.
            output_dim: int, dimension of the output.
            dropout: float, dropout of the network.
        """

        super(MLP, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.ReLU()

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass on the batch of data x.

        Args:
            x: torch.tensor, input batch of data.

        Returns:
            torch.tensor, output batch of data.
        """

        x = self.dropout(x)
        x = self.input_layer(x)
        x = self.input_activation(x)

        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.output_activation(x)

        return x
