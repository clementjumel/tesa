import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.layer(x))

