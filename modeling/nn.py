import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        """
        Initialize an instance of MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim1: int, first hidden dimension.
            hidden_dim2: int, second hidden dimension.
            dropout: float, dropout of the network.
        """

        super(MLP, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.input_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden_activation = nn.ReLU()

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

        x = self.hidden_layer(x)
        x = self.hidden_activation(x)

        x = self.output_layer(x)
        x = self.output_activation(x)

        return x


class RegressionMLP(MLP):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        """
        Initialize an instance of the Regression MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim1: int, first hidden dimension.
            hidden_dim2: int, second hidden dimension.
            dropout: float, dropout of the network.
        """

        super(RegressionMLP, self).__init__(input_dim, hidden_dim1, hidden_dim2, dropout)

        self.output_layer = nn.Linear(hidden_dim2, 1)
        self.output_activation = nn.Sigmoid()


class ClassificationMLP(MLP):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        """
        Initialize an instance of the Regression MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim1: int, first hidden dimension.
            hidden_dim2: int, second hidden dimension.
            dropout: float, dropout of the network.
        """

        super(ClassificationMLP, self).__init__(input_dim, hidden_dim1, hidden_dim2, dropout)

        self.output_layer = nn.Linear(hidden_dim2, 2)
        self.output_activation = nn.Softmax(dim=1)
