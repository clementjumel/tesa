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

        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2) if hidden_dim2 is not None else None
        self.hidden_activation = nn.ReLU() if hidden_dim2 is not None else None

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

        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.hidden_activation(x)

        x = self.output_layer(x)
        x = self.output_activation(x)

        return x


class RegressionMLP(MLP):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2=None, dropout=0):
        """
        Initialize an instance of the Regression MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim1: int, first hidden dimension.
            hidden_dim2: int, second hidden dimension.
            dropout: float, dropout of the network.
        """

        super(RegressionMLP, self).__init__(input_dim, hidden_dim1, hidden_dim2, dropout)

        output_dim = hidden_dim2 if hidden_dim2 is not None else hidden_dim1

        self.output_layer = nn.Linear(output_dim, 1)
        self.output_activation = nn.Sigmoid()


class ClassificationMLP(MLP):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2=None, dropout=0):
        """
        Initialize an instance of the Regression MLP.

        Args:
            input_dim: int, dimension of the input.
            hidden_dim1: int, first hidden dimension.
            hidden_dim2: int, second hidden dimension.
            dropout: float, dropout of the network.
        """

        super(ClassificationMLP, self).__init__(input_dim, hidden_dim1, hidden_dim2, dropout)

        output_dim = hidden_dim2 if hidden_dim2 is not None else hidden_dim1

        self.output_layer = nn.Linear(output_dim, 2)
        self.output_activation = nn.Softmax(dim=1)


class Bilinear(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim, dropout):
        """
        Initialize an instance of Bilinear one layer model.

        Args:
            input_dim1: int, dimension of the first input.
            input_dim2: int, dimension of the second input.`
            output_dim: int, dimension of the output.
            dropout: float, dropout of the network.
        """

        super(Bilinear, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.layer = nn.Bilinear(in1_features=input_dim1, in2_features=input_dim2, out_features=output_dim)

    def forward(self, x1_x2):
        """
        Forward pass on the batch of data x.

        Args:
            x1_x2: tuple, pair of inputs tensor batch of data.

        Returns:
            torch.tensor, output batch of data.
        """

        x1, x2 = x1_x2

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x = self.layer(x1, x2)
        x = self.activation(x)

        return x


class RegressionBilinear(Bilinear):

    def __init__(self, input_dim1, input_dim2, dropout):
        """
        Initialize an instance of Bilinear one layer model.

        Args:
            input_dim1: int, dimension of the first input.
            input_dim2: int, dimension of the second input.`
            dropout: float, dropout of the network.
        """

        super(RegressionBilinear, self).__init__(input_dim1, input_dim2, 1, dropout)

        self.activation = nn.Sigmoid()


class ClassificationBilinear(Bilinear):

    def __init__(self, input_dim1, input_dim2, dropout):
        """
        Initialize an instance of Bilinear one layer model.

        Args:
            input_dim1: int, dimension of the first input.
            input_dim2: int, dimension of the second input.`
            dropout: float, dropout of the network.
        """

        super(ClassificationBilinear, self).__init__(input_dim1, input_dim2, 2, dropout)

        self.activation = nn.Softmax(dim=1)

